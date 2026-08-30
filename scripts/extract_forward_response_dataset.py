"""Execution step 10 of docs/Adjoint_faithful_response_training_plan.md.

Extracts and freezes the curated development response store (plan section 13)
from the already-completed step 6/7/8/9 MITgcm output: pilot + train +
validation roles by default. The blind manifest is never reached by any
default: ``inventory_path`` and ``report_root`` both default to the
development paths, and execution step 16 supplies the evaluator-only blind
manifest and report root explicitly, after plan section 17's precondition is
satisfied. Historically this module opened no blind manifest
(``forward_response_blind_v1/forward_response_blind_inventory_v1.jsonl``) and
has no code path that could -- every inventory read here goes through
``load_public_rows``, which reads only the public
(pilot+train+validation) manifest and is never given the blind path.

Produces, under the frozen paths in
``config/forward_response_schema_v1.json``:

- one zarr store, ``forward_response_v1.zarr`` (three top-level groups:
  ``pilot``, ``train``, ``validation``), each holding that role's
  ``anchors/*``, ``short/*``, ``long/*`` arrays (plan section 13's
  ``array_contract``) plus, for ``pilot`` only, the ``controls/*`` arrays
  (section 10.3's duplicate/tight-CG confirmatory branches);
- per-role ``anchor_table.jsonl`` / ``direction_table.jsonl``
  (``pilot`` additionally gets ``solver_control_table.jsonl``), written
  once via the same O_EXCL convention ``build_forward_response_inventory``
  already uses for the geometry manifest;
- ``response_scales_v1.json``: the frozen section-14.2 response normalization
  scales ``d_{h,g,k}``, floored per section 10.3's combined numerical floor,
  computed here because both need real extracted response arrays and the
  pilot's control branches, neither of which existed before this step.

Two real gaps versus the schema this step also fixes, in place, because they
are only visible once the actual completed output is read (see the module
docstring for ``build_forward_response_inventory.pickup_to_trajectory_p64``
and the 2026-08-27 amendment in the plan document for the full account):

1. ``config/forward_response_schema_v1.json``'s ``alpha`` enum only listed
   the original three pilot candidates (0.025/0.05/0.1) -- missing the two
   later-frozen amplitudes, Theta's 0.005 and SSH's 0.03 override. Fixed by
   widening the enum; nothing else in the schema referenced the old list.
2. The schema's own Gate-D3 checklist asserted the pilot has exactly 36 long
   ``(base_direction, alpha)`` rows. One of those 36 -- (S0, day 3600, SSH,
   alpha=0.10) -- never ran a real branch at all: both signs hit the
   section-10.1 SSH peak cap before any MITgcm run and were correctly
   recorded as ``status: "failed_ssh_peak_cap"`` with no manifest/response.
   Real count is 35 long + 36 short = 71, not 72. Fixed in the schema text;
   this module omits that one row rather than fabricating it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numcodecs
import numpy as np
import zarr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "archive" / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from bire_repro import af_response_pickup_bank as pickup_bank  # noqa: E402
import build_amplitude_pilot as pilot  # noqa: E402
from build_forward_response_inventory import (  # noqa: E402
    REGIMES,
    ContractError,
    InventoryError,
    SourceCopy,
    SourceError,
    SourceResolution,
    _reject_symlinked_output_path,
    _support_counts,
    _verified_chain_roots,
    _write_jsonl_exclusive,
    canonical_json,
    load_json_strict,
    pickup_to_trajectory_p32,
    pickup_to_trajectory_p64,
    resolve_annual_pickup,
    sha256_file,
)
import stage_forward_response_run as stage  # noqa: E402
import analyze_amplitude_pilot as pilot_analysis  # noqa: E402


DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "config" / "forward_response_schema_v1.json"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "af_fno" / "response" / "forward_response_v1"
)
DEFAULT_DATASET_PATH = Path(
    "/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/forward_response_v1.zarr"
)
DEFAULT_PUBLIC_INVENTORY = stage.DEFAULT_PUBLIC_INVENTORY
DEFAULT_PILOT_REPORT_ROOT = pilot.DEFAULT_REPORT_ROOT
DEFAULT_PRODUCTION_REPORT_ROOT = stage.DEFAULT_REPORT_ROOT
DEFAULT_FINAL_AMPLITUDES = stage.DEFAULT_FINAL_AMPLITUDES
DEFAULT_SCRATCH_ROOT = stage.DEFAULT_SCRATCH_ROOT

ROLES = ("pilot", "train", "validation")
GROUP_SLICES = pilot_analysis.GROUP_SLICES  # {"U": 0:15, "V": 15:30, "Theta": 30:45, "SSH": 45:46}
GROUPS = ("U", "V", "Theta", "SSH")
NATIVE_GRID_MAP = {"W": "W_face", "S": "S_face", "C": "tracer"}
LONG_LEADS_PRODUCTION = tuple(range(10, 61, 10))  # train/validation long: 10..60
LONG_LEADS_PILOT = tuple(range(10, 91, 10))  # pilot long: 10..90
CHANNELS, NY, NX = 46, 62, 62

#: The one pilot (base_direction, alpha) pair that never ran a real branch --
#: both signs hit the section-10.1 SSH peak cap before any MITgcm run (see
#: module docstring). Real, verified, not fabricated: this module omits it.
PILOT_CAP_FAILURES = {("S0", 3600, "SSH", 0.1)}


class ExtractionError(RuntimeError):
    """Raised when the completed MITgcm output cannot be legitimately extracted."""


# ---------------------------------------------------------------------------
# Small shared helpers.


def _array_sha256(value: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C")).hexdigest()


def _gb_norm_by_group(z: np.ndarray, wet: np.ndarray) -> dict[str, float]:
    """Per-output-group RMS over wet cells -- the un-pooled term inside the
    frozen group-balanced norm (plan section 10.2's ``||.||_GB``)."""

    return {
        group: float(np.sqrt(np.mean(z[sl][:, wet] ** 2))) for group, sl in GROUP_SLICES.items()
    }


def _split_forcing_static_hashes(configuration_sha256: Mapping[str, str]) -> tuple[dict, dict]:
    """Section-13 anchor columns ``forcing_hashes``/``static_hashes``, split from
    one run's ``manifest.configuration_sha256``. Forcing = the two
    regime-dependent physical inputs (wind stress, SST relaxation target,
    both listed as production-map inputs in section 2.1); static = bathymetry
    and the MITgcm control/diagnostics files, which do not vary by regime."""

    forcing_names = {"windx_cosy.bin", "SST_relax.bin"}
    forcing = {k: v for k, v in configuration_sha256.items() if k in forcing_names}
    static = {k: v for k, v in configuration_sha256.items() if k not in forcing_names}
    if not forcing or not static:
        raise ExtractionError(f"unexpected configuration_sha256 keys: {sorted(configuration_sha256)}")
    return forcing, static


def _checkpoint_paths(report: Mapping[str, Any], lead: int) -> tuple[Path, Path]:
    manifest = report["manifest"]
    run_dir = Path(manifest["run_dir"])
    entry = next(
        c for c in manifest["archived_checkpoints"] if c["day"] - manifest["start_day"] == lead
    )
    stem = f"pickup.{entry['iteration']:010d}"
    return run_dir / f"{stem}.meta", run_dir / f"{stem}.data"


def _resolve_nominal_initial_path(nominal_report: Mapping[str, Any]) -> Path:
    return Path(pilot_analysis._resolve_nominal_initial(nominal_report))


# ---------------------------------------------------------------------------
# Inventory and report loading.


def load_public_rows(inventory_path: Path = DEFAULT_PUBLIC_INVENTORY) -> list[dict[str, Any]]:
    """Every row of the public (pilot+train+validation) inventory. Never reads
    the blind manifest -- no argument or default here names one."""

    if not inventory_path.is_file():
        raise ExtractionError(f"public inventory is not materialized yet: {inventory_path}")
    rows = []
    with inventory_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ExtractionError(f"no rows found in {inventory_path}")
    return rows


def _pilot_signed_report(regime: str, day: int, family: str, alpha: float, sign_token: str) -> dict:
    path = (
        DEFAULT_PILOT_REPORT_ROOT
        / f"{regime}_d{day:04d}_{family}_a{pilot._alpha_token(alpha)}_{sign_token}.json"
    )
    return load_json_strict(path)


def _pilot_nominal_report(regime: str, day: int, *, duplicate: bool = False, tight: bool = False) -> dict:
    suffix = "_dup" if duplicate else "_tight" if tight else ""
    path = DEFAULT_PILOT_REPORT_ROOT / f"{regime}_d{day:04d}_nominal{suffix}.json"
    return load_json_strict(path)


def _pilot_control_signed_report(
    regime: str, day: int, family: str, alpha: float, sign_token: str, *, tight: bool
) -> dict:
    suffix = "_tight" if tight else "_duplicate"
    path = (
        DEFAULT_PILOT_REPORT_ROOT
        / f"{regime}_d{day:04d}_{family}_a{pilot._alpha_token(alpha)}_{sign_token}{suffix}.json"
    )
    return load_json_strict(path)


def _production_signed_report(
    regime: str, day: int, family: str, slot: int, alpha: float, sign_token: str,
    report_root: Path = DEFAULT_PRODUCTION_REPORT_ROOT,
) -> dict:
    run_label = f"{regime}_d{day:04d}_{family}_q{slot}_a{pilot._alpha_token(alpha)}_{sign_token}"
    return load_json_strict(report_root / f"{run_label}.json")


def _production_nominal_report(
    role: str, regime: str, day: int, report_root: Path = DEFAULT_PRODUCTION_REPORT_ROOT
) -> dict:
    path = report_root / f"{regime}_d{day:04d}_{role}_nominal.json"
    return load_json_strict(path)


def resolve_nominal_report(
    role: str,
    regime: str,
    day: int,
    pilot_overlap: set[tuple[str, int]],
    report_root: Path = DEFAULT_PRODUCTION_REPORT_ROOT,
) -> dict[str, Any]:
    if role == "pilot" or (role == "train" and (regime, day) in pilot_overlap):
        return _pilot_nominal_report(regime, day)
    return _production_nominal_report(role, regime, day, report_root)


def resolve_anchor_provenance(
    regime: str, day: int, roots: Mapping[str, Sequence[Path]], scratch_root: Path
) -> SourceResolution:
    """Full-provenance version of ``stage_forward_response_run._resolve_source_for_anchor``:
    same bridge-vs-canonical-chain routing, but keeps ``duplicate_sources`` and
    ``canonical_choice_reason`` for the anchor table instead of collapsing to a
    4-tuple."""

    if pickup_bank.SOURCE_DAY < int(day) <= pickup_bank.END_DAY:
        iteration = pickup_bank.day_to_iteration(day)
        root = (
            scratch_root
            / pickup_bank.ROOT_NAME
            / regime
            / f"bridge_{pickup_bank.SOURCE_DAY}_{pickup_bank.END_DAY}"
        )
        meta_path = root / f"pickup.{iteration:010d}.meta"
        data_path = root / f"pickup.{iteration:010d}.data"
        if not (meta_path.is_file() and data_path.is_file()):
            raise ExtractionError(f"validation bridge pickup missing for {regime}/day{day}: {meta_path}")
        copy = SourceCopy(
            segment=str(root),
            meta_path=str(meta_path),
            data_path=str(data_path),
            meta_sha256=sha256_file(meta_path),
            data_sha256=sha256_file(data_path),
        )
        return SourceResolution(
            regime,
            day,
            iteration,
            copy,
            (copy,),
            "validation anchor resolved from the day-5760-6080 response pickup-bank bridge "
            "chain (step 6), off the canonical annual pickup cadence",
        )
    return resolve_annual_pickup(regime, int(day), roots)


# ---------------------------------------------------------------------------
# Direction-level geometry: alpha, edits, physical/standardized magnitudes.


def resolve_signed_alpha(row: Mapping[str, Any], final_amplitudes: Mapping[str, float]) -> float:
    key = (row["regime"], int(row["anchor_day"]), row["family"], int(row["direction_slot"]))
    return stage.SSH_ALPHA_OVERRIDES.get(key, final_amplitudes[row["family"]])


def _is_multi_level(row: Mapping[str, Any]) -> bool:
    return len(row["levels_one_based"]) > 1


def _direction_vectors(row: Mapping[str, Any], wet: np.ndarray, sigma: np.ndarray) -> dict[int, np.ndarray]:
    """One native-grid, alpha=1 sign=+1 direction vector per level, keyed the
    same way regardless of level count. Mirrors ``stage_forward_response_run.run_signed``'s
    own branch exactly: SSH (zero levels) and every single-level U/V/Theta
    direction go through ``build_amplitude_pilot.direction_vector`` (which
    ``channel_index``/``pickup_edits_for`` already handle via their own
    ``level = ... if levels else 1`` fallback); only genuine multi-level
    (section 8.6) directions use ``direction_vector_by_level``. Calling the
    by-level path unconditionally crashes on SSH: its empty ``levels`` zips to
    an empty ``standardized_by_level`` and ``np.concatenate`` has nothing to
    join -- verified directly against a real validation-role SSH row."""

    if _is_multi_level(row):
        return stage.direction_vector_by_level(row, wet, sigma)
    direction = stage._as_pilot_style_direction(row)
    v_q = pilot.direction_vector(direction, wet, sigma)
    level = int(row["levels_one_based"][0]) if row["levels_one_based"] else 1
    return {level: v_q}


def _edits_for(
    row: Mapping[str, Any], v_q_by_level: Mapping[int, np.ndarray], alpha: float, sign: int
):
    if _is_multi_level(row):
        return stage.pickup_edits_for_by_level(row, v_q_by_level, alpha, sign)
    direction = stage._as_pilot_style_direction(row)
    (v_q,) = v_q_by_level.values()
    return pilot.pickup_edits_for(direction, v_q, alpha, sign)


def _sparse_edit_json(edit) -> dict[str, Any]:
    return {
        "record_index0": edit.record,
        "j_index0": edit.j,
        "i_index0": edit.i,
        "value": edit.value,
    }


def _p32_realized_and_antisymmetry(
    row: Mapping[str, Any],
    wet: np.ndarray,
    sigma: np.ndarray,
    initial_nominal: np.ndarray,
    initial_plus: np.ndarray,
    initial_minus: np.ndarray,
) -> tuple[float, float, float]:
    """Generalizes ``analyze_amplitude_pilot._p32_checks`` to a joint
    multi-level support (section 8.6): every level's own channel, standardized
    by that channel's sigma, pooled into one RMS/antisymmetry pair across the
    direction's whole support -- exactly what ``direction_vector_by_level``'s
    own joint-RMS normalization measures. Reduces to ``_p32_checks`` byte-for-byte
    when there is one level."""

    direction = stage._as_pilot_style_direction(row)
    native_kernel = pilot._native_kernel(direction)
    centred = pilot._centred_projection(row["family"], native_kernel)
    support = (centred != 0.0) & wet

    plus_values, minus_values = [], []
    for level in row["levels_one_based"] or [1]:
        channel = pilot.channel_index(row["family"], [level])
        plus_values.append((initial_plus[channel][support] - initial_nominal[channel][support]) / sigma[channel][support])
        minus_values.append((initial_minus[channel][support] - initial_nominal[channel][support]) / sigma[channel][support])
    delta_plus = np.concatenate(plus_values)
    delta_minus = np.concatenate(minus_values)

    magnitude_plus = float(np.sqrt(np.mean(delta_plus**2)))
    magnitude_minus = float(np.sqrt(np.mean(delta_minus**2)))
    norm_plus = float(np.sqrt(np.sum(delta_plus**2)))
    norm_minus = float(np.sqrt(np.sum(delta_minus**2)))
    antisymmetry = float(np.sqrt(np.sum((delta_plus + delta_minus) ** 2))) / (
        0.5 * (norm_plus + norm_minus)
    )
    return magnitude_minus, magnitude_plus, antisymmetry


# ---------------------------------------------------------------------------
# Zarr array writing.

_COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)


def _create(group, name: str, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: str):
    return group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype, compressor=_COMPRESSOR)


def _create_index(group, name: str, shape: tuple[int, ...], dtype: str):
    return group.create_dataset(name, shape=shape, chunks=shape, dtype=dtype)


# ---------------------------------------------------------------------------
# Per-role extraction.


@dataclass
class RoleContext:
    role: str
    rows: list[dict[str, Any]]
    grid_wet: np.ndarray
    sigma: np.ndarray
    roots: Mapping[str, Sequence[Path]]
    scratch_root: Path
    final_amplitudes: Mapping[str, float]
    pilot_overlap: set[tuple[str, int]]
    #: Where this role's MITgcm run reports live. Defaults to the development
    #: root; execution step 16 supplies the evaluator-only blind root, which
    #: is a different directory so the blind reports never mingle with the
    #: development ones (plan section 17's sealed evaluator-only path).
    report_root: Path = DEFAULT_PRODUCTION_REPORT_ROOT


def _anchor_iteration(day: int) -> int:
    return 2_592_000 + 72 * int(day)


def _long_leads_for_role(role: str) -> tuple[int, ...]:
    return LONG_LEADS_PILOT if role in ("pilot", "blind_test") else LONG_LEADS_PRODUCTION


def _base_directions_by_anchor(role: str, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], list]:
    grouped: dict[tuple[str, int], list] = {}
    for row in rows:
        grouped.setdefault((row["regime"], int(row["anchor_day"])), []).append(row)
    return grouped


def extract_role(ctx: RoleContext) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    role = ctx.role
    grouped = _base_directions_by_anchor(role, ctx.rows)
    anchors_sorted = sorted(grouped, key=lambda key: (REGIMES.index(key[0]), key[1]))
    long_leads = _long_leads_for_role(role)

    anchor_table: list[dict[str, Any]] = []
    anchor_row_index: dict[tuple[str, int], int] = {}
    anchor_group: dict[tuple[str, int], str] = {}  # "short" | "long"
    state_p32_list, state_p64_list = [], []
    nominal_short_states, nominal_short_anchor_rows = [], []
    nominal_long_states, nominal_long_anchor_rows = [], []

    for anchor_row, (regime, day) in enumerate(anchors_sorted):
        base_rows = grouped[(regime, day)]
        has_long = any(bool(row["long"]) for row in base_rows)
        leads = long_leads if has_long else (10,)
        anchor_group[(regime, day)] = "long" if has_long else "short"
        anchor_row_index[(regime, day)] = anchor_row

        provenance = resolve_anchor_provenance(regime, day, ctx.roots, ctx.scratch_root)
        state_p32_list.append(pickup_to_trajectory_p32(provenance.canonical.meta_path, ctx.grid_wet))
        state_p64_list.append(pickup_to_trajectory_p64(provenance.canonical.meta_path, ctx.grid_wet))

        nominal_report = resolve_nominal_report(role, regime, day, ctx.pilot_overlap, ctx.report_root)
        endpoint_hashes = []
        states = []
        for lead in leads:
            meta_path, data_path = _checkpoint_paths(nominal_report, lead)
            states.append(pickup_to_trajectory_p64(meta_path, ctx.grid_wet))
            endpoint_hashes.append(sha256_file(data_path))
        if has_long:
            nominal_long_states.append(np.stack(states, axis=0))
            nominal_long_anchor_rows.append(anchor_row)
        else:
            nominal_short_states.append(np.stack(states, axis=0))
            nominal_short_anchor_rows.append(anchor_row)

        forcing_hashes, static_hashes = _split_forcing_static_hashes(
            nominal_report["manifest"]["configuration_sha256"]
        )
        duplicate_sources = [
            {
                "meta": {"path": copy.meta_path, "sha256": copy.meta_sha256},
                "data": {"path": copy.data_path, "sha256": copy.data_sha256},
            }
            for copy in provenance.candidates
            if copy is not provenance.canonical
        ]
        anchor_table.append(
            {
                "anchor_id": f"response-v1:anchor:{role}:{regime}:d{day:04d}",
                "role": role,
                "regime": regime,
                "day": day,
                "iteration": _anchor_iteration(day),
                "horizon_days": max(leads),
                "canonical_source_segment": provenance.canonical.segment,
                "canonical_source_meta": {
                    "path": provenance.canonical.meta_path,
                    "sha256": provenance.canonical.meta_sha256,
                },
                "canonical_source_data": {
                    "path": provenance.canonical.data_path,
                    "sha256": provenance.canonical.data_sha256,
                },
                "duplicate_sources": duplicate_sources,
                "canonical_choice_reason": provenance.canonical_choice_reason,
                "forcing_hashes": forcing_hashes,
                "static_hashes": static_hashes,
                "nominal_endpoint_hashes": endpoint_hashes,
            }
        )

    direction_table: list[dict[str, Any]] = []
    short_rows: list[dict[str, Any]] = []  # transient, becomes short/* arrays
    long_rows: list[dict[str, Any]] = []

    if role == "pilot":
        row_alpha_pairs = _pilot_row_alpha_pairs(ctx.rows)
    else:
        row_alpha_pairs = [(row, resolve_signed_alpha(row, ctx.final_amplitudes)) for row in ctx.rows]

    for row, alpha in row_alpha_pairs:
        regime, day, family = row["regime"], int(row["anchor_day"]), row["family"]
        anchor_key = (regime, day)
        anchor_row = anchor_row_index[anchor_key]
        is_long = bool(row["long"])
        leads = long_leads if is_long else (10,)
        levels = row["levels_one_based"]

        if role == "pilot":
            signed_reports = {
                sign_token: _pilot_signed_report(regime, day, family, alpha, sign_token)
                for sign_token in ("plus", "minus")
            }
        else:
            signed_reports = {
                sign_token: _production_signed_report(
                    regime, day, family, int(row["direction_slot"]), alpha, sign_token, ctx.report_root
                )
                for sign_token in ("plus", "minus")
            }

        nominal_report = resolve_nominal_report(role, regime, day, ctx.pilot_overlap, ctx.report_root)
        nominal_initial_path = _resolve_nominal_initial_path(nominal_report)
        initial_nominal = pickup_to_trajectory_p32(nominal_initial_path, ctx.grid_wet)
        initial_plus = pickup_to_trajectory_p32(
            Path(signed_reports["plus"]["manifest"]["pickup_meta_path"]), ctx.grid_wet
        )
        initial_minus = pickup_to_trajectory_p32(
            Path(signed_reports["minus"]["manifest"]["pickup_meta_path"]), ctx.grid_wet
        )
        input_state_p32 = np.stack([initial_minus, initial_plus], axis=0).astype(np.float32)

        v_q_by_level = _direction_vectors(row, ctx.grid_wet, ctx.sigma)
        _field, edits_plus, peak = _edits_for(row, v_q_by_level, alpha, 1)
        _field, edits_minus, _peak_minus = _edits_for(row, v_q_by_level, alpha, -1)
        edits_by_level_values = np.concatenate(
            [v_q_by_level[level][v_q_by_level[level] != 0.0] for level in levels or [1]]
        )
        physical_support = alpha * edits_by_level_values
        physical_peak = float(np.max(np.abs(physical_support)))
        physical_rms = float(np.sqrt(np.mean(physical_support**2)))
        physical_l2 = float(np.sqrt(np.sum(physical_support**2)))
        if abs(physical_peak - peak) > 1e-9 * max(1.0, peak):
            raise ExtractionError(f"recomputed peak disagrees with pickup_edits_for for {row['direction_id']}")

        magnitude_minus, magnitude_plus, antisymmetry = _p32_realized_and_antisymmetry(
            row, ctx.grid_wet, ctx.sigma, initial_nominal, initial_plus, initial_minus
        )

        response_by_lead = []
        response_hashes = []
        for sign_token, sign in (("minus", -1), ("plus", 1)):
            per_lead = []
            for lead in leads:
                meta_path, _data_path = _checkpoint_paths(signed_reports[sign_token], lead)
                perturbed = pickup_to_trajectory_p64(meta_path, ctx.grid_wet)
                nominal_meta_path, _nom_data = _checkpoint_paths(nominal_report, lead)
                nominal_state = pickup_to_trajectory_p64(nominal_meta_path, ctx.grid_wet)
                delta = perturbed - nominal_state
                per_lead.append(delta)
                response_hashes.append(_array_sha256(delta, "<f8"))
            response_by_lead.append(np.stack(per_lead, axis=0))
        response_p64 = np.stack(response_by_lead, axis=0).astype(np.float64)  # (2, L_or_1, 46, 62, 62)

        native_count, centred_count = _support_counts(
            _RowLike(family=family, kernel=row["kernel"], levels=levels)
        )
        kernel = row["kernel"]
        sigma_cells = 1.0 if kernel == "gaussian_5x5_sigma1" else None
        radius_cells = 2 if kernel == "gaussian_5x5_sigma1" else 0

        entry = {
            "direction_id": row["direction_id"]
            if role != "pilot"
            else f"{row['direction_id']}:a{pilot._alpha_token(alpha)}",
            "role": role,
            "array_group": "long" if is_long else "short",
            "array_row": len(long_rows) if is_long else len(short_rows),
            "anchor_row": anchor_row,
            "input_family": family,
            "native_field": pilot.FIELD_BY_FAMILY[family],
            "native_grid": NATIVE_GRID_MAP[{"U": "W", "V": "S", "Theta": "C", "SSH": "C"}[family]],
            "one_based_levels": list(levels),
            "vertical_weights": [float.fromhex(value) for value in row["vertical_weights_float64_hex"]],
            "j_index0": row["centre"]["j"],
            "i_index0": row["centre"]["i"],
            "longitude_deg": row["centre"]["lon"],
            "latitude_deg": row["centre"]["lat"],
            "region": row["region"],
            "kernel": kernel,
            "sigma_cells": sigma_cells,
            "radius_cells": radius_cells,
            "native_support_count": native_count,
            "centred_support_count": centred_count,
            "alpha": alpha,
            "unit_direction_standardized_rms": 1.0,
            "physical_peak": physical_peak,
            "physical_rms": physical_rms,
            "physical_l2": physical_l2,
            "p32_realized_standardized_rms": [magnitude_minus, magnitude_plus],
            "p32_antisymmetry_relative_error": antisymmetry,
            "long": is_long,
            "sparse_edits_minus": [_sparse_edit_json(e) for e in edits_minus],
            "sparse_edits_plus": [_sparse_edit_json(e) for e in edits_plus],
            "input_hashes": [
                signed_reports["minus"]["manifest"]["pickup_sha256"]["data"],
                signed_reports["plus"]["manifest"]["pickup_sha256"]["data"],
            ],
            "response_hashes": response_hashes,
        }
        direction_table.append(entry)
        payload = {"input_state_p32": input_state_p32, "response_p64": response_p64}
        (long_rows if is_long else short_rows).append(payload)

    role_shape = {
        "A": len(anchor_table),
        "A_short": len(nominal_short_anchor_rows),
        "A_long": len(nominal_long_anchor_rows),
        "Q_short": len(short_rows),
        "Q_long": len(long_rows),
        "L": len(long_leads),
    }
    arrays = {
        "anchors/state_p32": np.stack(state_p32_list, axis=0).astype(np.float32),
        "anchors/state_p64": np.stack(state_p64_list, axis=0).astype(np.float64),
        "anchors/nominal_short": (
            np.stack(nominal_short_states, axis=0).astype(np.float64)
            if nominal_short_states
            else np.zeros((0, 1, CHANNELS, NY, NX), dtype=np.float64)
        ),
        "anchors/nominal_short_anchor_row": np.asarray(nominal_short_anchor_rows, dtype=np.int32),
        "anchors/nominal_long": (
            np.stack(nominal_long_states, axis=0).astype(np.float64)
            if nominal_long_states
            else np.zeros((0, role_shape["L"], CHANNELS, NY, NX), dtype=np.float64)
        ),
        "anchors/nominal_long_anchor_row": np.asarray(nominal_long_anchor_rows, dtype=np.int32),
        "short/anchor_row": np.asarray(
            [entry["anchor_row"] for entry in direction_table if entry["array_group"] == "short"],
            dtype=np.int32,
        ),
        "short/direction_row": np.asarray(
            [i for i, entry in enumerate(direction_table) if entry["array_group"] == "short"],
            dtype=np.int32,
        ),
        "short/input_state_p32": (
            np.stack([p["input_state_p32"] for p in short_rows], axis=0)
            if short_rows
            else np.zeros((0, 2, CHANNELS, NY, NX), dtype=np.float32)
        ),
        "short/response_p64": (
            np.stack([p["response_p64"] for p in short_rows], axis=0)
            if short_rows
            else np.zeros((0, 2, 1, CHANNELS, NY, NX), dtype=np.float64)
        ),
        "long/anchor_row": np.asarray(
            [entry["anchor_row"] for entry in direction_table if entry["array_group"] == "long"],
            dtype=np.int32,
        ),
        "long/direction_row": np.asarray(
            [i for i, entry in enumerate(direction_table) if entry["array_group"] == "long"],
            dtype=np.int32,
        ),
        "long/input_state_p32": (
            np.stack([p["input_state_p32"] for p in long_rows], axis=0)
            if long_rows
            else np.zeros((0, 2, CHANNELS, NY, NX), dtype=np.float32)
        ),
        "long/response_p64": (
            np.stack([p["response_p64"] for p in long_rows], axis=0)
            if long_rows
            else np.zeros((0, 2, role_shape["L"], CHANNELS, NY, NX), dtype=np.float64)
        ),
        "lead_days": np.asarray(long_leads, dtype=np.int16),
    }
    payload = {
        "anchor_table": anchor_table,
        "direction_table": direction_table,
        "role_shape": role_shape,
    }
    return payload, arrays


class _RowLike:
    def __init__(self, family: str, kernel: str, levels: Sequence[int]):
        self.family = family
        self.kernel = kernel
        self.levels = tuple(levels)


def _pilot_row_alpha_pairs(pilot_rows: Sequence[Mapping[str, Any]]) -> list[tuple[dict, float]]:
    pilot_contract = load_json_strict(pilot.DEFAULT_PILOT_CONTRACT)
    candidate_alphas = pilot_contract["directions"]["candidate_alphas"]
    pairs = []
    for row in pilot_rows:
        for alpha in candidate_alphas:
            key = (row["regime"], int(row["anchor_day"]), row["family"], float(alpha))
            if key in PILOT_CAP_FAILURES:
                continue
            pairs.append((row, float(alpha)))
    return pairs


# ---------------------------------------------------------------------------
# Pilot-only solver controls (section 10.3).


def extract_pilot_controls(ctx: RoleContext) -> tuple[list[dict], dict[str, np.ndarray]]:
    long_base_rows = [row for row in ctx.rows if row["long"]]
    if len(long_base_rows) != 12:
        raise ExtractionError(f"expected 12 long pilot base directions, found {len(long_base_rows)}")
    long_base_rows = sorted(
        long_base_rows, key=lambda row: (REGIMES.index(row["regime"]), GROUPS.index(row["family"]))
    )
    anchors = sorted(
        {(row["regime"], int(row["anchor_day"])) for row in ctx.rows}, key=lambda a: (REGIMES.index(a[0]), a[1])
    )
    anchor_index = {anchor: i for i, anchor in enumerate(anchors)}
    if len(anchors) != 6:
        raise ExtractionError(f"expected 6 pilot anchors, found {len(anchors)}")

    perturbed = np.zeros((12, 2, 2, 9, CHANNELS, NY, NX), dtype=np.float64)
    duplicate_nominal = np.zeros((6, 9, CHANNELS, NY, NX), dtype=np.float64)
    tight_nominal = np.zeros((6, 9, CHANNELS, NY, NX), dtype=np.float64)

    solver_control_table: list[dict[str, Any]] = []
    final_amplitudes = ctx.final_amplitudes
    theta_v2 = load_json_strict(
        DEFAULT_OUTPUT_ROOT / "amplitude_pilot_theta_v2_selection.json"
    )["provisional_alpha_theta"]
    provisional = {
        "U": final_amplitudes["U"],
        "V": final_amplitudes["V"],
        "SSH": final_amplitudes["SSH"] if final_amplitudes["SSH"] != 0.03 else 0.05,
        "Theta": theta_v2,
    }

    for anchor_pos, anchor in enumerate(anchors):
        regime, day = anchor
        for tight, array_name, array in (
            (False, "controls/duplicate_nominal_p64", duplicate_nominal),
            (True, "controls/tight_nominal_p64", tight_nominal),
        ):
            report = _pilot_nominal_report(regime, day, duplicate=not tight, tight=tight)
            endpoint_hashes = []
            for lead_index, lead in enumerate(LONG_LEADS_PILOT):
                meta_path, data_path = _checkpoint_paths(report, lead)
                array[anchor_pos, lead_index] = pickup_to_trajectory_p64(meta_path, ctx.grid_wet)
                endpoint_hashes.append(sha256_file(data_path))
            solver_control_table.append(
                {
                    "control_id": f"response-v1:control:pilot:{regime}:d{day:04d}:"
                    f"{'tight_cg_nominal' if tight else 'nominal_duplicate'}",
                    "base_direction_id": None,
                    "anchor_id": f"response-v1:anchor:pilot:{regime}:d{day:04d}",
                    "condition": "tight_cg_nominal" if tight else "nominal_duplicate",
                    "array_name": array_name,
                    "array_indices": [anchor_pos],
                    "sign": None,
                    "matched_nominal_row": anchor_pos,
                    "cg2d_target_residual": 1e-10 if tight else 1e-07,
                    "run_manifest_sha256": hashlib.sha256(
                        canonical_json(report["manifest"]).encode("utf-8")
                    ).hexdigest(),
                    "endpoint_hashes": endpoint_hashes,
                }
            )

    for base_index, row in enumerate(long_base_rows):
        regime, day, family = row["regime"], int(row["anchor_day"]), row["family"]
        alpha = provisional[family]
        anchor_pos = anchor_index[(regime, day)]
        for tight, condition, condition_axis in (
            (False, "perturbed_duplicate", 0),
            (True, "tight_cg_perturbed", 1),
        ):
            for sign_index, sign_token in enumerate(("minus", "plus")):
                report = _pilot_control_signed_report(regime, day, family, alpha, sign_token, tight=tight)
                endpoint_hashes = []
                nominal_report = _pilot_nominal_report(regime, day, duplicate=not tight, tight=tight)
                for lead_index, lead in enumerate(LONG_LEADS_PILOT):
                    meta_path, data_path = _checkpoint_paths(report, lead)
                    perturbed_state = pickup_to_trajectory_p64(meta_path, ctx.grid_wet)
                    nominal_meta, _nom_data = _checkpoint_paths(nominal_report, lead)
                    nominal_state = pickup_to_trajectory_p64(nominal_meta, ctx.grid_wet)
                    perturbed[base_index, condition_axis, sign_index, lead_index] = (
                        perturbed_state - nominal_state
                    )
                    endpoint_hashes.append(sha256_file(data_path))
                solver_control_table.append(
                    {
                        "control_id": f"response-v1:control:pilot:{regime}:d{day:04d}:{family}:"
                        f"{condition}:{sign_token}",
                        "base_direction_id": row["direction_id"],
                        "anchor_id": f"response-v1:anchor:pilot:{regime}:d{day:04d}",
                        "condition": condition,
                        "array_name": "controls/perturbed_response_p64",
                        "array_indices": [base_index, condition_axis, sign_index],
                        "sign": -1 if sign_token == "minus" else 1,
                        "matched_nominal_row": anchor_pos,
                        "cg2d_target_residual": 1e-10 if tight else 1e-07,
                        "run_manifest_sha256": hashlib.sha256(
                            canonical_json(report["manifest"]).encode("utf-8")
                        ).hexdigest(),
                        "endpoint_hashes": endpoint_hashes,
                    }
                )

    if len(solver_control_table) != 60:
        raise ExtractionError(f"expected 60 solver control rows, got {len(solver_control_table)}")
    arrays = {
        "controls/perturbed_response_p64": perturbed,
        "controls/duplicate_nominal_p64": duplicate_nominal,
        "controls/tight_nominal_p64": tight_nominal,
    }
    return solver_control_table, arrays


# ---------------------------------------------------------------------------
# Orchestration.


def _load_context(
    role: str, rows: Sequence[Mapping[str, Any]], report_root: Path = DEFAULT_PRODUCTION_REPORT_ROOT
) -> RoleContext:
    contract, roots, grid = pilot._load_sources(pilot.DEFAULT_DATASET_CONTRACT)
    pilot_contract = load_json_strict(pilot.DEFAULT_PILOT_CONTRACT)
    sigma = pilot._load_normalizer(pilot_contract)
    final_amplitudes = stage.load_final_amplitudes(DEFAULT_FINAL_AMPLITUDES)
    pilot_overlap = stage._pilot_overlap_anchors(stage.DEFAULT_PILOT_GEOMETRY)
    role_rows = [row for row in rows if row["role"] == role]
    if not role_rows:
        raise ExtractionError(f"no {role} rows in the inventory")
    return RoleContext(
        role, role_rows, grid.wet, sigma, roots, DEFAULT_SCRATCH_ROOT,
        final_amplitudes, pilot_overlap, report_root,
    )


def write_role(
    zarr_group,
    role: str,
    payload: dict[str, Any],
    arrays: Mapping[str, np.ndarray],
    output_root: Path,
    solver_control_table: list[dict] | None = None,
    control_arrays: Mapping[str, np.ndarray] | None = None,
) -> dict[str, str]:
    role_group = zarr_group.require_group(role)
    for name, array in arrays.items():
        if "/" in name:
            sub_group_name, leaf = name.split("/", 1)
            sub_group = role_group.require_group(sub_group_name)
        else:
            sub_group, leaf = role_group, name
        chunks = _chunk_shape(name, array.shape)
        dtype = "<i2" if array.dtype == np.int16 else "<i4" if array.dtype == np.int32 else (
            "<f4" if array.dtype == np.float32 else "<f8"
        )
        dataset = _create(sub_group, leaf, array.shape, chunks, dtype)
        if array.size:
            dataset[...] = array
    if control_arrays:
        for name, array in control_arrays.items():
            sub_group_name, leaf = name.split("/", 1)
            sub_group = role_group.require_group(sub_group_name)
            chunks = _chunk_shape(name, array.shape)
            dataset = _create(sub_group, leaf, array.shape, chunks, "<f8")
            dataset[...] = array

    hashes = {}
    anchor_path = output_root / f"{role}_anchor_table.jsonl"
    direction_path = output_root / f"{role}_direction_table.jsonl"
    _reject_symlinked_output_path(anchor_path)
    _reject_symlinked_output_path(direction_path)
    hashes["anchor_table_sha256"] = _write_jsonl_exclusive(
        anchor_path, payload["anchor_table"], final_mode=0o444
    )
    hashes["direction_table_sha256"] = _write_jsonl_exclusive(
        direction_path, payload["direction_table"], final_mode=0o444
    )
    if solver_control_table is not None:
        control_path = output_root / f"{role}_solver_control_table.jsonl"
        _reject_symlinked_output_path(control_path)
        hashes["solver_control_table_sha256"] = _write_jsonl_exclusive(
            control_path, solver_control_table, final_mode=0o444
        )
    return hashes


def _chunk_shape(name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Section 13: response/control chunks are one (row, sign, lead, 46,62,62)
    field; input/state chunks one (row, sign, 46,62,62) or (row, 46,62,62)
    field -- i.e. every axis but the trailing (46,62,62) block chunks at 1."""

    if shape == ():
        return shape
    return tuple(1 for _ in shape[:-3]) + shape[-3:] if len(shape) >= 3 else tuple(1 for _ in shape)


def extract(
    roles: Sequence[str] = ROLES,
    *,
    inventory_path: Path = DEFAULT_PUBLIC_INVENTORY,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    report_root: Path = DEFAULT_PRODUCTION_REPORT_ROOT,
) -> dict[str, Any]:
    started = time.monotonic()
    _reject_symlinked_output_path(dataset_path)
    temporary = dataset_path.with_name(dataset_path.name + ".tmp")
    if dataset_path.exists() or temporary.exists():
        raise ExtractionError(f"refusing to overwrite an existing dataset: {dataset_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    rows = load_public_rows(inventory_path)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(temporary))
    root_group = zarr.group(store=store, overwrite=False)
    root_group.attrs["store_kind"] = "development"
    root_group.attrs["roles"] = list(roles)

    table_hashes: dict[str, Any] = {}
    role_shapes: dict[str, Any] = {}
    for role in roles:
        ctx = _load_context(role, rows, report_root)
        payload, arrays = extract_role(ctx)
        solver_control_table = None
        control_arrays = None
        if role == "pilot":
            solver_control_table, control_arrays = extract_pilot_controls(ctx)
        table_hashes[role] = write_role(
            root_group, role, payload, arrays, output_root, solver_control_table, control_arrays
        )
        role_shapes[role] = payload["role_shape"]

    zarr.consolidate_metadata(store)
    os.replace(temporary, dataset_path)

    manifest = {
        "version": "forward_response_dataset_v1_extraction",
        "dataset": str(dataset_path),
        "roles": list(roles),
        "role_shapes": role_shapes,
        "table_hashes": table_hashes,
        "metadata_sha256": sha256_file(dataset_path / ".zmetadata"),
        "elapsed_seconds": time.monotonic() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    manifest_path = output_root / "forward_response_dataset_v1_manifest.json"
    _reject_symlinked_output_path(manifest_path)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(manifest_path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles", nargs="+", choices=ROLES, default=list(ROLES))
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    manifest = extract(args.roles, dataset_path=args.dataset_path, output_root=args.output_root)
    print(json.dumps({"role_shapes": manifest["role_shapes"], "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
