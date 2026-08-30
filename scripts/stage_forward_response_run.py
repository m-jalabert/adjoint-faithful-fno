"""Execution step 9 of docs/Adjoint_faithful_response_training_plan.md.

Generates every shared nominal and signed production **train**/**validation**
response branch (section 25 step 9: "short cases first and then the frozen
60-day long subsets. Do not generate or expose blind response data."). This
module never reads the blind manifest and has no code path that could -- it
only ever loads the public inventory
(``outputs/af_fno/response/forward_response_v1/forward_response_inventory_v1.jsonl``),
filtered to ``role in {"train", "validation"}``, which structurally excludes
both pilot (already run in step 7/8) and blind (a separate, evaluator-only
manifest this module does not import or open).

Reuses, unchanged, the trusted building blocks the amplitude pilot (step
7/8) already proved at 154 real MITgcm branches: ``build_amplitude_pilot``'s
kernel construction / RMS-standardized direction vector / pickup-edit
construction / source resolution, ``af_pilot_segment``'s restart-safe
segment driver (now accepting a ``scratch_subdir`` so production runs land
in their own scratch tree, not the pilot's), and ``af_s0_twin``'s trusted
byte-level pickup editor. The only new logic here is: reading a *production*
inventory row's schema instead of the pilot's own 24-row geometry, applying
the single frozen amplitude per family (no alpha sweep -- Gate D2 already
froze U=0.10, V=0.10, Theta=0.005, SSH=0.05), 60-day (not 90-day) long
duration, and sharing one nominal branch across every direction at the same
(role, regime, anchor_day) -- reusing the amplitude pilot's own existing
90-day nominal runs at the 6 (regime, anchor_day) pairs that overlap pilot's
own anchors (day 720/3600), so those need no new MITgcm run at all.

Subcommands:

``list-work``
    Enumerate every signed and nominal branch step 9 must run (excluding the
    6 pilot-overlap nominal groups, which are already done). Read-only, no
    MITgcm compute -- used by ``submit_forward_response_run.py`` and for
    auditing before submitting anything.

``run-signed --regime --day --family --slot``
    Build the additively-edited pickup for one signed production branch (a
    single row of the public inventory, addressed by
    (regime, anchor_day, family, direction_slot) rather than its own
    ``direction_id`` -- that string embeds a JSON fragment with commas,
    which ``sbatch --export`` silently truncates on, verified directly
    against a real canary submission) and run it for its declared horizon
    (10 or 60 days, from the row's own ``long`` field).

``run-nominal --role --regime --day``
    Run the unperturbed shared nominal branch for one (role, regime,
    anchor_day) group, at that group's required duration (60 days if any
    direction sharing the anchor is long, else 10). Raises if the anchor is
    one of the 6 pilot-overlap anchors -- those are never run here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "archive" / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from bire_repro import af_pilot_segment as segment  # noqa: E402
from bire_repro import af_response_pickup_bank as pickup_bank  # noqa: E402
from bire_repro.af_s0_twin import (  # noqa: E402
    PickupEdit,
    pickup_record_index,
    write_declared_pickup_edits,
)
import build_amplitude_pilot as pilot  # noqa: E402
from build_forward_response_inventory import (  # noqa: E402
    ContractError,
    InventoryError,
    load_json_strict,
    sha256_file,
)


DEFAULT_PUBLIC_INVENTORY = (
    PROJECT_ROOT
    / "outputs"
    / "af_fno"
    / "response"
    / "forward_response_v1"
    / "forward_response_inventory_v1.jsonl"
)
DEFAULT_FINAL_AMPLITUDES = (
    PROJECT_ROOT
    / "outputs"
    / "af_fno"
    / "response"
    / "forward_response_v1"
    / "amplitude_pilot_final_selection_v1.json"
)
DEFAULT_PILOT_GEOMETRY = pilot.DEFAULT_GEOMETRY_OUTPUT
DEFAULT_SCRATCH_ROOT = pilot.DEFAULT_SCRATCH_ROOT
DEFAULT_EXECUTABLE = pilot.DEFAULT_EXECUTABLE
DEFAULT_REPORT_ROOT = (
    PROJECT_ROOT / "outputs" / "af_fno" / "response" / "forward_response_v1" / "production_runs"
)
SCRATCH_SUBDIR = "mitgcm_forward_response_v1"
PRODUCTION_ROLES = ("train", "validation")
LONG_DURATION_DAYS = 60
SHORT_DURATION_DAYS = 10


class ProductionRunError(RuntimeError):
    """Raised when a production response branch cannot legitimately proceed."""


#: Reviewed, documented exceptions to the SSH_PEAK_METERS_MAX cap check in
#: run_signed, keyed by (regime, anchor_day, family, direction_slot).
#: Follows the same convention as GATE_D2_EXCEPTIONS in
#: analyze_amplitude_pilot_controls.py (the S1/day720/V tight-CG case):
#: reviewed and accepted as within tolerance, not silently dropped or
#: treated as a defect. Of the 18 SSH directions found to exceed the 1cm
#: cap at alpha_SSH=0.05 during real step-9 execution, these 3 are the ones
#: at <=5% overshoot, cleanly separated by a real gap from the next-closest
#: (10.2%) -- see the roadmap's 2026-08-26 amendment for the full
#: distribution and the decision to split by severity. The other 15 are
#: covered by config/forward_response_amplitude_pilot_ssh_v2.json instead,
#: not added here.
SSH_PEAK_CAP_EXCEPTIONS: dict[tuple[str, int, str, int], dict[str, str]] = {
    ("S1", 360, "SSH", 15): {
        "reason": "peak 0.010109 m, 1.1% over the 0.01 m cap at alpha=0.05 -- the smallest overshoot of any violation found.",
        "decision": "accepted_as_documented_exception_not_a_defect",
    },
    ("S2", 1440, "SSH", 14): {
        "reason": "peak 0.010454 m, 4.5% over the 0.01 m cap at alpha=0.05.",
        "decision": "accepted_as_documented_exception_not_a_defect",
    },
    ("S0", 720, "SSH", 14): {
        "reason": "peak 0.010454 m, 4.5% over the 0.01 m cap at alpha=0.05 (same magnitude as S2/day1440/slot14 above; independent locations).",
        "decision": "accepted_as_documented_exception_not_a_defect",
    },
}

#: Per-direction alpha override, keyed the same way as
#: SSH_PEAK_CAP_EXCEPTIONS. The 12 train-role SSH directions that exceed the
#: cap at the frozen family-wide alpha_SSH=0.05 by more than the accepted
#: exception threshold use config/forward_response_amplitude_pilot_ssh_v2.json's
#: chosen alpha (0.03, the largest of {0.03, 0.025, 0.02} that passed every
#: Q_lin/Q_SNR/adjacent-alpha/P32 criterion at all 6 frozen pilot locations,
#: both point and smooth) instead of the family default -- every other SSH
#: direction (204 already-succeeded + the 3 SSH_PEAK_CAP_EXCEPTIONS above)
#: is untouched. The 3 validation-role directions that also exceed the cap
#: are deliberately NOT here yet: per Gate D3, changing their amplitude
#: requires swapping them for freshly-allocated validation centres first
#: (see the roadmap's 2026-08-26 amendment) -- a separate step.
SSH_ALPHA_OVERRIDES: dict[tuple[str, int, str, int], float] = {
    key: 0.03
    for key in (
        ("S2", 360, "SSH", 15),
        ("S2", 0, "SSH", 14),
        ("S0", 720, "SSH", 15),
        ("S0", 2160, "SSH", 15),
        ("S0", 1800, "SSH", 14),
        ("S0", 3240, "SSH", 14),
        ("S0", 1440, "SSH", 14),
        ("S1", 4320, "SSH", 14),
        ("S1", 5040, "SSH", 15),
        ("S1", 0, "SSH", 15),
        ("S1", 5760, "SSH", 14),
        ("S1", 1800, "SSH", 14),
    )
}


# ---------------------------------------------------------------------------
# Inventory loading -- public rows only, train/validation only.


def load_production_rows(
    inventory_path: Path = DEFAULT_PUBLIC_INVENTORY,
    roles: Sequence[str] = PRODUCTION_ROLES,
) -> list[dict[str, Any]]:
    """Every row of ``inventory_path`` whose role is in ``roles``.

    Both arguments default to step 9's scope -- the public manifest (0444)
    and ``PRODUCTION_ROLES``, which excludes pilot rows that step 7/8 already
    ran. **This module still never names the blind manifest**; the defaults
    cannot reach it.

    The parameters exist for execution step 16, whose evaluator-only sibling
    ``stage_blind_forward_response_run.py`` supplies the blind manifest and
    the ``blind_test`` role explicitly, after section 17's precondition
    (every checkpoint, lambda, development report and ordinary forward report
    frozen and hashed) is satisfied. Passing them is the caller's assertion
    that it is allowed to; defaulting them keeps every step-9 call site
    structurally unable to read blind data, exactly as before.
    """

    if not inventory_path.is_file():
        raise ProductionRunError(f"public inventory is not materialized yet: {inventory_path}")
    rows = []
    with inventory_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["role"] in tuple(roles):
                rows.append(row)
    if not rows:
        raise ProductionRunError(f"no {'/'.join(roles)} rows found in {inventory_path}")
    return rows


def load_final_amplitudes(path: Path = DEFAULT_FINAL_AMPLITUDES) -> dict[str, float]:
    payload = load_json_strict(path)
    if payload.get("final_selection") != "PASS -- all four amplitudes frozen":
        raise ProductionRunError(f"amplitude selection is not frozen PASS: {path}")
    amplitudes = payload["selected_amplitudes"]
    missing = {"U", "V", "Theta", "SSH"} - set(amplitudes)
    if missing:
        raise ProductionRunError(f"frozen amplitude selection is missing families: {missing}")
    return {family: float(value) for family, value in amplitudes.items()}


def _pilot_overlap_anchors(geometry_path: Path = DEFAULT_PILOT_GEOMETRY) -> set[tuple[str, int]]:
    """The (regime, anchor_day) pairs the amplitude pilot already ran to 90 days.

    Read from the frozen pilot geometry file rather than hardcoded, so this
    stays correct if the pilot's own anchor set is ever regenerated.
    """

    geometry = load_json_strict(geometry_path)
    return {(row["regime"], int(row["anchor_day"])) for row in geometry["directions"]}


def _resolve_source_for_anchor(
    regime: str, anchor_day: int, roots: Mapping[str, Sequence[Path]], scratch_root: Path
) -> tuple[Path, Path, str, str]:
    """Route to the right pickup store for this anchor's source pickup.

    Train anchors sit on the regular annual checkpoint cadence and resolve
    from the canonical trajectory-v3 chain (``pilot._resolve_source``).
    Validation anchors (6010/6050/6080) are deliberately off that cadence
    (section 7.3: "distinct from all nominal selection starts") and were
    never in the canonical chain at all -- step 6 built a dedicated
    day-5,760-to-6,080 bridge specifically so they have an available
    pickup, verified against trajectory-v3's P32 projection when that
    bridge was built. Verified directly: a real submission's first
    validation-role branches all failed with "no complete pickup in
    canonical trajectory-v3 chain roots" before this routing existed --
    the data was never missing, only this function was.
    """

    if pickup_bank.SOURCE_DAY < int(anchor_day) <= pickup_bank.END_DAY:
        iteration = pickup_bank.day_to_iteration(anchor_day)
        root = (
            scratch_root
            / pickup_bank.ROOT_NAME
            / regime
            / f"bridge_{pickup_bank.SOURCE_DAY}_{pickup_bank.END_DAY}"
        )
        meta_path = root / f"pickup.{iteration:010d}.meta"
        data_path = root / f"pickup.{iteration:010d}.data"
        if not (meta_path.is_file() and data_path.is_file()):
            raise ProductionRunError(
                f"validation bridge pickup missing for {regime}/day{anchor_day}: {meta_path}"
            )
        return meta_path, data_path, sha256_file(meta_path), sha256_file(data_path)
    return pilot._resolve_source(regime, int(anchor_day), roots)


def _find_row_by_coordinates(
    rows: Sequence[Mapping[str, Any]], regime: str, anchor_day: int, family: str, slot: int
) -> Mapping[str, Any]:
    """Look up a row by (regime, anchor_day, family, direction_slot).

    The primary lookup key for CLI/Slurm use: ``direction_id`` is a single
    string that embeds a JSON fragment (``{"levels":[...],"weights_hex":...}``),
    and Slurm's ``sbatch --export`` splits its argument on commas -- a
    comma anywhere inside an exported value silently truncates it (verified
    directly: a real canary submission passing ``AF_DIRECTION_ID`` this way
    failed with a 0-row lookup, the direction_id cut off at the first
    embedded comma). ``direction_slot`` is already verified unique within
    (regime, anchor_day, family) at production scale (max 6 directions per
    group), so this 4-tuple is an equally exact, comma-free identifier.
    """

    matches = [
        row
        for row in rows
        if row["regime"] == regime
        and int(row["anchor_day"]) == int(anchor_day)
        and row["family"] == family
        and int(row["direction_slot"]) == int(slot)
    ]
    if len(matches) != 1:
        raise ProductionRunError(
            f"expected exactly one row for {regime}/day{anchor_day}/{family}/slot{slot}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _as_pilot_style_direction(row: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt one production inventory row to the shape build_amplitude_pilot's
    ``direction_vector``/``pickup_edits_for`` expect (flat j/i, ``levels`` not
    ``levels_one_based``). Same physical quantities, different JSON schema --
    the production inventory (``inventory_row`` in
    ``build_forward_response_inventory.py``) and the pilot's own geometry
    file (``materialize_geometry`` in ``build_amplitude_pilot.py``) were
    frozen independently and were never meant to share one schema.
    """

    return {
        "j": row["centre"]["j"],
        "i": row["centre"]["i"],
        "kernel": row["kernel"],
        "family": row["family"],
        "levels": row["levels_one_based"],
        "long": row["long"],
    }


def _row_vertical_weights(row: Mapping[str, Any]) -> tuple[float, ...]:
    """Decode a row's frozen per-level weights (index-aligned with ``levels_one_based``).

    Read from the inventory rather than recomputed from
    ``build_forward_response_inventory._unit_weights``: the inventory row is
    the single frozen source of truth for this direction's exact vertical
    weighting, already unit-L2-normalized by construction.
    """

    return tuple(float.fromhex(value) for value in row["vertical_weights_float64_hex"])


def direction_vector_by_level(
    row: Mapping[str, Any], wet_mask: np.ndarray, sigma_46_62_62: np.ndarray
) -> dict[int, np.ndarray]:
    """Generalizes ``build_amplitude_pilot.direction_vector`` to a weighted multi-level combination.

    Section 8.6 ("held-out vertical combinations"): training directions are
    always single-level, but validation (and blind, out of step 9's scope)
    directions include genuine 2- and 3-level combinations -- verified
    directly against a real submission: 27 of 888 production directions
    (9 each of U/V/Theta, all validation-role) have ``len(levels) > 1``, and
    ``build_amplitude_pilot``'s single-level-only ``direction_vector`` raises
    ``ValueError: too many values to unpack`` on every one of them.

    The horizontal kernel pattern does not vary by level (section 8.1); what
    varies is each level's own channel normalizer sigma and its frozen
    ``_unit_weights`` coefficient (read from the row, see
    ``_row_vertical_weights``). The whole multi-level support -- every
    level's cells together -- is jointly RMS-normalized to unit RMS
    (section 8.1's rule applied across the direction's full nonzero
    support, not per level), which is exactly what ``_unit_weights``'
    unit-L2-normalization is for: it makes this joint normalization reduce
    *exactly* to ``build_amplitude_pilot.direction_vector``'s own formula
    when there is only one level (weight 1.0) -- verified directly,
    byte-for-byte, in
    ``test_direction_vector_by_level_matches_pilot_for_a_single_level``.
    """

    direction = _as_pilot_style_direction(row)
    weights = _row_vertical_weights(row)
    if len(weights) != len(direction["levels"]):
        raise ProductionRunError(f"level/weight count mismatch for {row['direction_id']}")
    native = pilot._native_kernel(direction)
    centred = pilot._centred_projection(direction["family"], native)
    support = (centred != 0.0) & wet_mask
    if not support.any():
        raise ProductionRunError(f"direction has empty wet centred support: {row['direction_id']}")

    standardized_by_level: dict[int, np.ndarray] = {}
    for level, weight in zip(direction["levels"], weights):
        sigma = sigma_46_62_62[pilot.channel_index(direction["family"], [level])]
        standardized_by_level[level] = weight * centred[support] / sigma[support]

    stacked = np.concatenate(list(standardized_by_level.values()))
    rms = math.sqrt(float(np.mean(stacked**2)))
    if not math.isfinite(rms) or rms <= 0.0:
        raise ProductionRunError(f"direction has a non-finite RMS scale: {row['direction_id']}")

    return {level: (native * weight) / rms for level, weight in zip(direction["levels"], weights)}


def pickup_edits_for_by_level(
    row: Mapping[str, Any], v_q_by_level: Mapping[int, np.ndarray], alpha: float, sign: int
) -> tuple[str, list[PickupEdit], float]:
    """Multi-level generalization of ``build_amplitude_pilot.pickup_edits_for``.

    Builds one edit per nonzero cell *per level*, each against that level's
    own pickup record -- a single-level direction's ``v_q_by_level`` has
    exactly one key, so this reduces to the same edit list
    ``pickup_edits_for`` would build (verified in the same equivalence test
    as ``direction_vector_by_level``).
    """

    field = pilot.FIELD_BY_FAMILY[row["family"]]
    edits: list[PickupEdit] = []
    peak = 0.0
    for level, v_q in v_q_by_level.items():
        record = pickup_record_index(field, level)
        for j, i in np.argwhere(v_q != 0.0):
            value = float(sign) * float(alpha) * float(v_q[j, i])
            edits.append(PickupEdit(record=record, j=int(j), i=int(i), value=value))
            peak = max(peak, abs(value))
    if not edits:
        raise ProductionRunError(f"direction produced no nonzero edits: {row['direction_id']}")
    return field, edits, peak


# ---------------------------------------------------------------------------
# Shared nominal branch grouping (one branch serves every direction at that
# (role, regime, anchor_day)).


def nominal_groups(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int], int]:
    """(role, regime, anchor_day) -> required nominal duration in days.

    The longest horizon any direction sharing that anchor declares -- matching
    section 12.4's rule ("training/validation long branches, including their
    nominal controls, as six validated 10-day segments"), since a shared
    nominal branch must reach at least as far as the furthest signed branch it
    is the control for.

    For train and validation this is exactly the previous constant behaviour:
    their rows declare horizons of 10 and 60 only, so anchors with a long
    direction return LONG_DURATION_DAYS and the rest SHORT_DURATION_DAYS. It
    is written from the rows rather than from those constants because the
    step-16 blind manifest declares 90-day long horizons (section 17 evaluates
    days 20..90), where returning 60 would leave the day-90 nominal reference
    missing and make S_resp^90 uncomputable.
    """

    groups: dict[tuple[str, str, int], int] = {}
    for row in rows:
        key = (row["role"], row["regime"], int(row["anchor_day"]))
        # `horizon_days` is what every materialized manifest declares; the
        # `long` flag is the equivalent for rows that carry only it.
        if "horizon_days" in row:
            horizon = int(row["horizon_days"])
        else:
            horizon = LONG_DURATION_DAYS if bool(row["long"]) else SHORT_DURATION_DAYS
        if horizon % SHORT_DURATION_DAYS != 0:
            raise ProductionRunError(
                f"{row['direction_id']} declares a {horizon}-day horizon, not a whole number of "
                f"{SHORT_DURATION_DAYS}-day segments"
            )
        groups[key] = max(groups.get(key, 0), horizon)
    return groups


# ---------------------------------------------------------------------------
# Run one signed branch.


def run_signed(
    regime: str,
    anchor_day: int,
    family: str,
    slot: int,
    *,
    project_root: Path = PROJECT_ROOT,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    executable: Path = DEFAULT_EXECUTABLE,
    inventory_path: Path = DEFAULT_PUBLIC_INVENTORY,
    dataset_contract_path: Path = pilot.DEFAULT_DATASET_CONTRACT,
    pilot_contract_path: Path = pilot.DEFAULT_PILOT_CONTRACT,
    final_amplitudes_path: Path = DEFAULT_FINAL_AMPLITUDES,
    report_root: Path = DEFAULT_REPORT_ROOT,
    roles: Sequence[str] = PRODUCTION_ROLES,
) -> dict[str, Any]:
    rows = load_production_rows(inventory_path, roles)
    row = _find_row_by_coordinates(rows, regime, anchor_day, family, slot)
    direction_id = row["direction_id"]
    amplitudes = load_final_amplitudes(final_amplitudes_path)
    override_key = (row["regime"], int(row["anchor_day"]), row["family"], row["direction_slot"])
    alpha = SSH_ALPHA_OVERRIDES.get(override_key, amplitudes[row["family"]])
    is_multi_level = len(row["levels_one_based"]) > 1

    contract, roots, grid = pilot._load_sources(dataset_contract_path)
    pilot_contract = load_json_strict(pilot_contract_path)
    sigma = pilot._load_normalizer(pilot_contract)
    if is_multi_level:
        # Section 8.6's held-out vertical combinations: 27 of 888 production
        # directions (validation-role only) have more than one level.
        # build_amplitude_pilot's direction_vector/pickup_edits_for are
        # single-level only (pilot itself never has a multi-level
        # direction) -- verified directly: every one of these 27 raised
        # ValueError there. direction_vector_by_level/pickup_edits_for_by_level
        # are the exact generalization (equivalence to the single-level
        # formula proven in
        # test_direction_vector_by_level_matches_pilot_for_a_single_level),
        # kept as a separate path rather than switched in for every
        # direction so the other 861 already-proven-working directions are
        # untouched.
        v_q_by_level = direction_vector_by_level(row, grid.wet, sigma)
    else:
        direction = _as_pilot_style_direction(row)
        v_q = pilot.direction_vector(direction, grid.wet, sigma)

    results: dict[str, Any] = {}
    for sign in (1, -1):
        if is_multi_level:
            field, edits, peak = pickup_edits_for_by_level(row, v_q_by_level, alpha, sign)
        else:
            field, edits, peak = pilot.pickup_edits_for(direction, v_q, alpha, sign)
        if (
            row["family"] == "SSH"
            and peak > pilot.SSH_PEAK_METERS_MAX
            and override_key not in SSH_PEAK_CAP_EXCEPTIONS
        ):
            raise ProductionRunError(
                f"SSH direction {direction_id} at the frozen alpha {alpha} exceeds the "
                f"{pilot.SSH_PEAK_METERS_MAX} m cap (peak {peak:.6f} m); this is a recorded "
                "failure of an already-frozen amplitude, not something to silently clip -- stop"
            )

        source_meta, source_data, _meta_sha, _data_sha = _resolve_source_for_anchor(
            row["regime"], int(row["anchor_day"]), roots, scratch_root
        )
        sign_token = "plus" if sign == 1 else "minus"
        run_label = (
            f"{row['regime']}_d{int(row['anchor_day']):04d}_{row['family']}"
            f"_q{row['direction_slot']}_a{pilot._alpha_token(alpha)}_{sign_token}"
        )
        staging_dir = scratch_root / SCRATCH_SUBDIR / "_edited_pickups" / run_label
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_meta = staging_dir / source_meta.name
        staged_data = staging_dir / source_data.name
        if not (staged_meta.is_file() and staged_data.is_file()):
            write_declared_pickup_edits(
                source_meta,
                staging_dir,
                expected_iteration=segment.day_to_iteration(int(row["anchor_day"])),
                declared_fields=(field,),
                edits=edits,
                operation="add",
            )

        # Data-driven for the same reason nominal_groups is: train/validation
        # rows declare 10- and 60-day horizons, so this is unchanged for them,
        # but the step-16 blind manifest declares 90-day long horizons and a
        # signed branch must reach as far as the lead set it will be scored at.
        duration_days = (
            int(row["horizon_days"])
            if "horizon_days" in row
            else (LONG_DURATION_DAYS if row["long"] else SHORT_DURATION_DAYS)
        )
        if duration_days % SHORT_DURATION_DAYS != 0:
            raise ProductionRunError(
                f"{row['direction_id']} declares a {duration_days}-day horizon, not a whole "
                f"number of {SHORT_DURATION_DAYS}-day segments"
            )
        manifest = segment.prepare_segment(
            project_root,
            scratch_root,
            executable,
            run_label,
            source_meta.parent,
            staged_meta,
            staged_data,
            sha256_file(staged_meta),
            sha256_file(staged_data),
            int(row["anchor_day"]),
            duration_days,
            scratch_subdir=SCRATCH_SUBDIR,
        )
        result = segment.run_segment(manifest)
        report = {
            "kind": "signed",
            "direction_id": direction_id,
            "run_label": run_label,
            "regime": row["regime"],
            "anchor_day": row["anchor_day"],
            "family": row["family"],
            "direction_slot": row["direction_slot"],
            "role": row["role"],
            "alpha": alpha,
            "sign": sign,
            "long": bool(row["long"]),
            "duration_days": duration_days,
            "ssh_peak_m": peak if row["family"] == "SSH" else None,
            "ssh_peak_cap_exception": SSH_PEAK_CAP_EXCEPTIONS.get(override_key),
            "manifest": manifest,
            "result": {key: value for key, value in result.items() if key != "archived_pickups"},
            "archived_pickup_count": len(result["archived_pickups"]),
        }
        report_root.mkdir(parents=True, exist_ok=True)
        (report_root / f"{run_label}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        results[sign_token] = report
    return results


# ---------------------------------------------------------------------------
# Run one shared nominal branch.


def run_nominal(
    role: str,
    regime: str,
    anchor_day: int,
    *,
    project_root: Path = PROJECT_ROOT,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    executable: Path = DEFAULT_EXECUTABLE,
    inventory_path: Path = DEFAULT_PUBLIC_INVENTORY,
    dataset_contract_path: Path = pilot.DEFAULT_DATASET_CONTRACT,
    pilot_geometry_path: Path = DEFAULT_PILOT_GEOMETRY,
    report_root: Path = DEFAULT_REPORT_ROOT,
    roles: Sequence[str] = PRODUCTION_ROLES,
) -> dict[str, Any]:
    if role not in tuple(roles):
        raise ProductionRunError(f"role must be one of {tuple(roles)}")
    if (regime, int(anchor_day)) in _pilot_overlap_anchors(pilot_geometry_path):
        raise ProductionRunError(
            f"{regime}/day{anchor_day} is a pilot-overlap anchor -- reuse the amplitude "
            "pilot's existing 90-day nominal run (its day-60 checkpoint), do not rerun"
        )
    rows = load_production_rows(inventory_path, roles)
    groups = nominal_groups(rows)
    key = (role, regime, int(anchor_day))
    if key not in groups:
        raise ProductionRunError(f"no production directions share the anchor {key}")
    duration_days = groups[key]

    contract, roots, _grid = pilot._load_sources(dataset_contract_path)
    source_meta, source_data, meta_sha256, data_sha256 = _resolve_source_for_anchor(
        regime, int(anchor_day), roots, scratch_root
    )
    run_label = f"{regime}_d{int(anchor_day):04d}_{role}_nominal"
    manifest = segment.prepare_segment(
        project_root,
        scratch_root,
        executable,
        run_label,
        source_meta.parent,
        source_meta,
        source_data,
        meta_sha256,
        data_sha256,
        int(anchor_day),
        duration_days,
        scratch_subdir=SCRATCH_SUBDIR,
    )
    result = segment.run_segment(manifest)
    report = {
        "kind": "nominal",
        "run_label": run_label,
        "role": role,
        "regime": regime,
        "anchor_day": anchor_day,
        "duration_days": duration_days,
        "manifest": manifest,
        "result": {key: value for key, value in result.items() if key != "archived_pickups"},
        "archived_pickup_count": len(result["archived_pickups"]),
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / f"{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


# ---------------------------------------------------------------------------
# Work enumeration for the submit driver.


def list_work(
    inventory_path: Path = DEFAULT_PUBLIC_INVENTORY,
    pilot_geometry_path: Path = DEFAULT_PILOT_GEOMETRY,
) -> dict[str, Any]:
    rows = load_production_rows(inventory_path)
    overlap = _pilot_overlap_anchors(pilot_geometry_path)
    groups = nominal_groups(rows)

    signed = [
        {
            "direction_id": row["direction_id"],
            "regime": row["regime"],
            "anchor_day": row["anchor_day"],
            "family": row["family"],
            "role": row["role"],
            "direction_slot": row["direction_slot"],
            "long": bool(row["long"]),
        }
        for row in rows
    ]
    nominal = [
        {"role": role, "regime": regime, "anchor_day": anchor_day, "duration_days": duration_days}
        for (role, regime, anchor_day), duration_days in sorted(groups.items())
        if (regime, anchor_day) not in overlap
    ]
    reused = [
        {"role": role, "regime": regime, "anchor_day": anchor_day, "duration_days": duration_days}
        for (role, regime, anchor_day), duration_days in sorted(groups.items())
        if (regime, anchor_day) in overlap
    ]
    return {
        "signed_branches": len(signed) * 2,
        "signed_directions": signed,
        "new_nominal_branches": len(nominal),
        "new_nominal_groups": nominal,
        "reused_pilot_nominal_groups": reused,
        "short_signed_directions": sum(1 for item in signed if not item["long"]),
        "long_signed_directions": sum(1 for item in signed if item["long"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("list-work")

    signed_parser = subparsers.add_parser("run-signed")
    signed_parser.add_argument("--regime", choices=("S0", "S1", "S2"), required=True)
    signed_parser.add_argument("--day", type=int, required=True)
    signed_parser.add_argument("--family", choices=("U", "V", "Theta", "SSH"), required=True)
    signed_parser.add_argument("--slot", type=int, required=True)
    signed_parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    signed_parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    signed_parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)

    nominal_parser = subparsers.add_parser("run-nominal")
    nominal_parser.add_argument("--role", choices=PRODUCTION_ROLES, required=True)
    nominal_parser.add_argument("--regime", choices=("S0", "S1", "S2"), required=True)
    nominal_parser.add_argument("--day", type=int, required=True)
    nominal_parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    nominal_parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    nominal_parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)

    arguments = parser.parse_args(argv)
    try:
        if arguments.mode == "list-work":
            result = list_work()
        elif arguments.mode == "run-signed":
            result = run_signed(
                arguments.regime,
                arguments.day,
                arguments.family,
                arguments.slot,
                project_root=arguments.project_root.resolve(),
                scratch_root=arguments.scratch_root.resolve(),
                executable=arguments.executable.resolve(),
            )
        else:
            result = run_nominal(
                arguments.role,
                arguments.regime,
                arguments.day,
                project_root=arguments.project_root.resolve(),
                scratch_root=arguments.scratch_root.resolve(),
                executable=arguments.executable.resolve(),
            )
    except (InventoryError, ContractError, segment.PilotSegmentError, ProductionRunError) as error:
        print(f"FORWARD RESPONSE RUN: FAIL -- {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
