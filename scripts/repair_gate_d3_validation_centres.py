"""Surgical centre repair for the 7 validation-role directions Gate D3 found
failing (docs/Adjoint_faithful_response_training_plan.md, 2026-08-27
amendment): 6 fail Q_lin and/or Q_SNR (northern-region Theta nonlinearity,
one southern SSH point-kernel location decaying toward the floor at late
leads, one WBC Theta case), plus 1 fails the P32 antisymmetry bound. Per
Gate D3's text ("every failed validation case becomes development data and
the successor must create new response-validation ... inventories") and the
researcher's 2026-08-27 decision (treat all three mechanisms as documented
exceptions for the 12 already-affected TRAIN directions, which have no such
provenance constraint, but give these 7 VALIDATION directions fresh centres).

Unlike the 2026-08-26 SSH-peak-cap repair this mirrors structurally
(``repair_ssh_v2_deferred_centres.py``), there is no zero-compute proxy for
Q_lin/Q_SNR/antisymmetry: candidate cells cannot be pre-filtered
analytically, only verified by actually running MITgcm. For each target,
candidates are tried in the frozen farthest-point scorer's own order; each
one is staged, run for real (both signs, at the row's already-frozen role/
alpha/duration), extracted, and checked against the exact Gate D3 criteria
before being accepted -- a failing candidate is discarded and the next one
tried, up to ``MAX_ATTEMPTS_PER_TARGET``.

Run with --dry-run (default) to only enumerate/report candidate order
without running MITgcm. Pass --apply to run real branches, verify, and (once
every target has a passing candidate) unseal/patch/reseal the affected
sealed files: the public inventory, ``validation_anchor_table.jsonl`` is
untouched (anchors do not move), ``validation_direction_table.jsonl``, and
the ``validation`` zarr group's ``short``/``long`` arrays at the 7 affected
rows only. Every other row is left byte-identical, verified before and
after.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "archive" / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import numpy as np  # noqa: E402
import zarr  # noqa: E402

from bire_repro import af_pilot_segment as segment  # noqa: E402
from bire_repro.af_s0_twin import write_declared_pickup_edits  # noqa: E402
import build_amplitude_pilot as pilot  # noqa: E402
import build_forward_response_inventory as inv  # noqa: E402
import stage_forward_response_run as staging  # noqa: E402
import extract_forward_response_dataset as extract  # noqa: E402
import verify_forward_response_dataset as verify  # noqa: E402

PUBLIC_INVENTORY = staging.DEFAULT_PUBLIC_INVENTORY
OUTPUT_ROOT = extract.DEFAULT_OUTPUT_ROOT
DATASET_PATH = extract.DEFAULT_DATASET_PATH
MAX_ATTEMPTS_PER_TARGET = 6
REPAIR_SCRATCH_SUBDIR = "mitgcm_forward_response_v1_gate_d3_repair"

# (regime, anchor_day, family, direction_slot) -- the 7 validation directions
# Gate D3 found failing on 2026-08-27 (see the plan document's amendment).
TARGETS = (
    ("S0", 6010, "Theta", 13),
    ("S0", 6050, "Theta", 12),
    ("S0", 6080, "SSH", 20),
    ("S0", 6080, "Theta", 15),
    ("S1", 6010, "Theta", 12),
    ("S1", 6080, "SSH", 20),
    ("S2", 6080, "Theta", 12),
)


def _key(row: Mapping[str, Any]) -> tuple[str, int, str, int]:
    return (row["regime"], int(row["anchor_day"]), row["family"], int(row["direction_slot"]))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class _RowRef:
    __slots__ = ("row", "is_target", "j", "i", "lon", "lat", "role")

    def __init__(self, row: dict[str, Any], is_target: bool) -> None:
        self.row = row
        self.is_target = is_target
        self.j = int(row["centre"]["j"])
        self.i = int(row["centre"]["i"])
        self.lon = float(row["centre"]["lon"])
        self.lat = float(row["centre"]["lat"])
        self.role = row["role"]


def _candidate_order(
    ref: _RowRef,
    all_rows: Sequence[dict[str, Any]],
    grid,
    masks,
    wbc_speed: np.ndarray,
) -> list:
    regime, family, region = ref.row["regime"], ref.row["family"], ref.row["region"]
    stratum = [
        _RowRef(row, is_target=_key(row) in {(*t,) for t in TARGETS} and row["role"] == "validation")
        for row in all_rows
        if row["regime"] == regime and row["family"] == family and row["region"] == region
    ]
    taken = {(r.j, r.i) for r in stratum}
    placed_by_role: dict[str, list[inv.Candidate]] = defaultdict(list)
    for r in stratum:
        if not r.is_target:
            placed_by_role[r.role].append(
                inv.Candidate(
                    j=r.j, i=r.i, lon=r.lon, lat=r.lat, region=region, subregion="",
                    centroid_lon=0.0, centroid_lat=0.0, tertiary_distance_km=0.0, tie_sha256="",
                )
            )

    direction = inv.Direction(
        role="validation",
        regime=regime,
        anchor_day=int(ref.row["anchor_day"]),
        anchor_slot=int(ref.row["anchor_index"]),
        direction_slot=int(ref.row["direction_slot"]),
        family=family,
        kernel=ref.row["kernel"],
        levels=tuple(ref.row["levels_one_based"]),
        weights=tuple(float.fromhex(v) for v in ref.row["vertical_weights_float64_hex"]),
        region=region,
    )
    pool = inv.enumerate_candidates(grid, direction, masks, wbc_speed=wbc_speed)
    eligible = [c for c in pool if (c.j, c.i) not in taken]
    if region != "WBC":
        blocked: set[tuple[int, int]] = set()
        for earlier_role in inv.ROLE_ORDER[: inv.ROLE_ORDER.index("validation")]:
            for placed in placed_by_role.get(earlier_role, ()):
                for dj in range(-2, 3):
                    for di in range(-2, 3):
                        blocked.add((placed.j + dj, placed.i + di))
        eligible = [c for c in eligible if (c.j, c.i) not in blocked]
    if not eligible:
        raise inv.CapacityError(f"no eligible replacement candidates for {_key(ref.row)}")

    ordered: list = []
    remaining = list(eligible)
    working_placed = {role: list(v) for role, v in placed_by_role.items()}
    for _ in range(min(MAX_ATTEMPTS_PER_TARGET, len(remaining))):
        winner = inv._pick_farthest_candidate(remaining, "validation", working_placed)
        ordered.append(winner)
        remaining = [c for c in remaining if (c.j, c.i) != (winner.j, winner.i)]
        working_placed.setdefault("validation", []).append(winner)
    return ordered


def _resolve_alpha(row: Mapping[str, Any], final_amplitudes: Mapping[str, float]) -> float:
    return staging.SSH_ALPHA_OVERRIDES.get(_key(row), final_amplitudes[row["family"]])


def _run_candidate_branch(
    row: Mapping[str, Any],
    j: int,
    i: int,
    attempt: int,
    alpha: float,
    grid,
    sigma: np.ndarray,
    roots,
    scratch_root: Path,
    project_root: Path,
    executable: Path,
) -> dict[str, dict[str, Any]]:
    """Stage and run both signed branches at candidate (j,i). Returns
    {"plus": report, "minus": report}, each with a resolved manifest."""

    direction = {
        "j": j, "i": i, "kernel": row["kernel"], "family": row["family"],
        "levels": list(row["levels_one_based"]), "long": bool(row["long"]),
    }
    v_q = pilot.direction_vector(direction, grid.wet, sigma)
    provenance = extract.resolve_anchor_provenance(row["regime"], int(row["anchor_day"]), roots, scratch_root)
    source_meta = Path(provenance.canonical.meta_path)

    duration_days = staging.LONG_DURATION_DAYS if row["long"] else staging.SHORT_DURATION_DAYS
    reports: dict[str, dict[str, Any]] = {}
    for sign, sign_token in ((1, "plus"), (-1, "minus")):
        field, edits, peak = pilot.pickup_edits_for(direction, v_q, alpha, sign)
        run_label = (
            f"{row['regime']}_d{int(row['anchor_day']):04d}_{row['family']}"
            f"_q{row['direction_slot']}_gated3repair_try{attempt}_{sign_token}"
        )
        staging_dir = scratch_root / REPAIR_SCRATCH_SUBDIR / "_edited_pickups" / run_label
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_meta = staging_dir / source_meta.name
        staged_data = staging_dir / source_meta.with_suffix(".data").name
        if not (staged_meta.is_file() and staged_data.is_file()):
            write_declared_pickup_edits(
                source_meta, staging_dir,
                expected_iteration=segment.day_to_iteration(int(row["anchor_day"])),
                declared_fields=(field,), edits=edits, operation="add",
            )
        manifest = segment.prepare_segment(
            project_root, scratch_root, executable, run_label, source_meta.parent,
            staged_meta, staged_data, inv.sha256_file(staged_meta), inv.sha256_file(staged_data),
            int(row["anchor_day"]), duration_days, scratch_subdir=REPAIR_SCRATCH_SUBDIR,
        )
        result = segment.run_segment(manifest)
        if result.get("returncode", 1) != 0:
            raise RuntimeError(f"MITgcm repair branch failed: {run_label}")
        reports[sign_token] = {"manifest": manifest, "result": result, "peak": peak}
    return reports


def _check_candidate(
    row: Mapping[str, Any],
    reports: Mapping[str, dict[str, Any]],
    nominal_report: Mapping[str, Any],
    wet: np.ndarray,
    sigma: np.ndarray,
    combined_floor: Mapping[str, float],
) -> tuple[bool, dict[str, Any]]:
    leads = extract.LONG_LEADS_PRODUCTION if row["long"] else (10,)
    detail: dict[str, Any] = {"per_lead": {}}
    ok = True

    if row["family"] == "SSH" and reports["plus"]["peak"] > pilot.SSH_PEAK_METERS_MAX:
        return False, {"reason": f"SSH peak {reports['plus']['peak']:.6f} m exceeds cap"}

    # Reuses extract._p32_realized_and_antisymmetry (the same function the
    # extraction pipeline itself uses), which correctly restricts the RMS to
    # the perturbation's own centred support -- a from-scratch reimplementation
    # here first computed it over the *whole* 62x62 domain instead, diluting
    # the magnitude by roughly sqrt(support_size/3600) and making every
    # candidate look like a false "realized magnitude" failure even though
    # the underlying MITgcm branches and Q_lin/Q_SNR were fine. Caught
    # directly: six real candidates for one target all "failed" at almost
    # exactly a ~12.4x-too-small magnitude, an implausible physical result
    # but an exact match for that dilution ratio.
    initial_nominal = extract.pickup_to_trajectory_p32(
        extract._resolve_nominal_initial_path(nominal_report), wet
    )
    initial_plus = extract.pickup_to_trajectory_p32(
        Path(reports["plus"]["manifest"]["pickup_meta_path"]), wet
    )
    initial_minus = extract.pickup_to_trajectory_p32(
        Path(reports["minus"]["manifest"]["pickup_meta_path"]), wet
    )
    magnitude_minus, magnitude_plus, antisymmetry = extract._p32_realized_and_antisymmetry(
        row, wet, sigma, initial_nominal, initial_plus, initial_minus
    )
    alpha = row["_repair_alpha"]
    detail["p32_magnitude_plus"] = magnitude_plus
    detail["p32_magnitude_minus"] = magnitude_minus
    detail["p32_antisymmetry"] = antisymmetry
    if abs(magnitude_plus - alpha) > 0.01 * alpha or abs(magnitude_minus - alpha) > 0.01 * alpha:
        ok = False
    if antisymmetry > 0.01:
        ok = False

    def gb(z):
        return float(np.sqrt(np.mean([np.mean(z[sl][:, wet] ** 2) for sl in extract.GROUP_SLICES.values()])))

    for lead in leads:
        r_signed = {}
        for sign_token, sign, report in (("minus", -1, reports["minus"]), ("plus", 1, reports["plus"])):
            meta_path, _data = extract._checkpoint_paths(report, lead)
            perturbed = extract.pickup_to_trajectory_p64(meta_path, wet)
            nom_meta, _nd = extract._checkpoint_paths(nominal_report, lead)
            nominal_state = extract.pickup_to_trajectory_p64(nom_meta, wet)
            raw = perturbed - nominal_state
            r_signed[sign_token] = (raw / sigma) / float(sign)
        r_plus, r_minus = r_signed["plus"], r_signed["minus"]
        q_lin = gb(r_plus - r_minus) / max(1e-300, 0.5 * (gb(r_plus) + gb(r_minus)))
        q_snr = 0.5 * (gb(r_plus) + gb(r_minus)) / combined_floor[str(lead)]
        detail["per_lead"][lead] = {"q_lin": q_lin, "q_snr": q_snr}
        if q_lin > 0.05 or q_snr < 20:
            ok = False
    return ok, detail


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    public_rows = _load_jsonl(PUBLIC_INVENTORY)
    contract = pilot.load_json_strict(pilot.DEFAULT_DATASET_CONTRACT)
    pilot_contract = pilot.load_json_strict(pilot.DEFAULT_PILOT_CONTRACT)
    roots = inv._verified_chain_roots(contract)
    grid = inv.read_grid(contract["sources"]["grid"]["canonical_root"])
    state, _report = inv.verify_trajectory_store(contract, grid)
    masks = inv.region_masks(grid.wet)
    speed_by_regime = {
        regime: inv.mean_surface_speed_already_centered(state, index)
        for index, regime in enumerate(inv.REGIMES)
    }
    sigma = pilot._load_normalizer(pilot_contract)
    final_amplitudes = staging.load_final_amplitudes(staging.DEFAULT_FINAL_AMPLITUDES)
    final_selection = extract.load_json_strict(OUTPUT_ROOT / "amplitude_pilot_final_selection_v1.json")
    combined_floor = final_selection["combined_floor_gb_by_lead"]
    pilot_overlap = staging._pilot_overlap_anchors(staging.DEFAULT_PILOT_GEOMETRY)

    target_keys = set(TARGETS)
    row_by_key = {_key(row): row for row in public_rows if row["role"] == "validation"}
    missing = target_keys - set(row_by_key)
    if missing:
        raise inv.ContractError(f"target rows not found in public inventory: {missing}")

    results: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for target_key in TARGETS:
        row = row_by_key[target_key]
        ref = _RowRef(row, is_target=True)
        candidates = _candidate_order(ref, public_rows, grid, masks, speed_by_regime[row["regime"]])
        print(f"=== {target_key}: old centre ({ref.j},{ref.i}), {len(candidates)} candidates queued ===")

        nominal_report = extract.resolve_nominal_report(
            "validation", row["regime"], int(row["anchor_day"]), pilot_overlap
        )
        alpha = _resolve_alpha(row, final_amplitudes)
        row_with_alpha = dict(row)
        row_with_alpha["_repair_alpha"] = alpha

        if not args.apply:
            for c in candidates:
                print(f"  candidate ({c.j},{c.i}) lon={c.lon:.2f} lat={c.lat:.2f} -- not run (dry-run)")
            continue

        accepted = None
        for attempt, candidate in enumerate(candidates, start=1):
            print(f"  attempt {attempt}: ({candidate.j},{candidate.i}) -- running real MITgcm branches ...")
            reports = _run_candidate_branch(
                row_with_alpha, candidate.j, candidate.i, attempt, alpha, grid, sigma, roots,
                extract.DEFAULT_SCRATCH_ROOT, extract.PROJECT_ROOT,
                pilot.DEFAULT_EXECUTABLE,
            )
            # _check_candidate rebuilds the kernel/support mask from the row's
            # own centre -- it must reflect *this* candidate, not the
            # original failing cell, or the antisymmetry/magnitude check
            # would be evaluated against the wrong support entirely.
            row_at_candidate = dict(row_with_alpha)
            row_at_candidate["centre"] = {
                "i": candidate.i, "j": candidate.j, "lat": candidate.lat, "lon": candidate.lon,
            }
            ok, detail = _check_candidate(row_at_candidate, reports, nominal_report, grid.wet, sigma, combined_floor)
            print(f"    -> {'PASS' if ok else 'FAIL'}: {json.dumps(detail, default=str)}")
            if ok:
                accepted = {
                    "candidate": candidate, "reports": reports, "detail": detail,
                    "alpha": alpha, "attempt": attempt,
                }
                break
        if accepted is None:
            raise RuntimeError(f"no passing candidate found for {target_key} within {MAX_ATTEMPTS_PER_TARGET} attempts")
        results[target_key] = accepted

    if not args.apply:
        print("\nDRY RUN -- no MITgcm run, no files modified. Re-run with --apply.", file=sys.stderr)
        return 0

    _apply(public_rows, row_by_key, results, grid, sigma, pilot_overlap, final_amplitudes)
    return 0


def _apply(public_rows, row_by_key, results, grid, sigma, pilot_overlap, final_amplitudes) -> None:
    new_public = []
    for row in public_rows:
        key = _key(row)
        if key in results:
            accepted = results[key]
            candidate = accepted["candidate"]
            new_row = dict(row)
            new_row["centre"] = {"i": candidate.i, "j": candidate.j, "lat": candidate.lat, "lon": candidate.lon}
            new_public.append(new_row)
        else:
            new_public.append(row)

    backup = PUBLIC_INVENTORY.with_suffix(PUBLIC_INVENTORY.suffix + ".pre_gate_d3_centre_repair_2026-08-27.bak")
    os.chmod(PUBLIC_INVENTORY, 0o600)
    backup.write_bytes(PUBLIC_INVENTORY.read_bytes())
    os.chmod(backup, 0o400)
    payload = "".join(inv.canonical_json(r) + "\n" for r in new_public).encode("utf-8")
    PUBLIC_INVENTORY.write_bytes(payload)
    os.chmod(PUBLIC_INVENTORY, 0o444)
    print(f"patched {PUBLIC_INVENTORY} (backup {backup}), sha256={hashlib.sha256(payload).hexdigest()}")

    # Rebuild the 7 affected direction_table rows and patch the validation
    # zarr arrays + validation_direction_table.jsonl in place.
    direction_table_path = OUTPUT_ROOT / "validation_direction_table.jsonl"
    direction_table = _load_jsonl(direction_table_path)
    store = zarr.open(str(DATASET_PATH), mode="a")
    val_group = store["validation"]

    contract = pilot.load_json_strict(pilot.DEFAULT_DATASET_CONTRACT)
    roots = inv._verified_chain_roots(contract)

    patched_rows = 0
    for key, accepted in results.items():
        old_row = row_by_key[key]
        new_row = dict(old_row)
        candidate = accepted["candidate"]
        new_row["centre"] = {"i": candidate.i, "j": candidate.j, "lat": candidate.lat, "lon": candidate.lon}

        index = next(
            idx for idx, entry in enumerate(direction_table)
            if entry["direction_id"] == old_row["direction_id"]
        )
        old_entry = direction_table[index]

        nominal_report = extract.resolve_nominal_report(
            "validation", new_row["regime"], int(new_row["anchor_day"]), pilot_overlap
        )
        alpha = accepted["alpha"]
        reports = accepted["reports"]

        # Recompute the full direction_table row exactly as extraction does,
        # via the same helper functions, at the new centre.
        v_q_by_level = extract._direction_vectors(new_row, grid.wet, sigma)
        _field, edits_plus, peak = extract._edits_for(new_row, v_q_by_level, alpha, 1)
        _field, edits_minus, _peak_minus = extract._edits_for(new_row, v_q_by_level, alpha, -1)
        levels = new_row["levels_one_based"]
        values = np.concatenate(
            [v_q_by_level[level][v_q_by_level[level] != 0.0] for level in levels or [1]]
        )
        physical_support = alpha * values
        physical_peak = float(np.max(np.abs(physical_support)))
        physical_rms = float(np.sqrt(np.mean(physical_support**2)))
        physical_l2 = float(np.sqrt(np.sum(physical_support**2)))

        nominal_initial_path = extract._resolve_nominal_initial_path(nominal_report)
        initial_nominal = extract.pickup_to_trajectory_p32(nominal_initial_path, grid.wet)
        initial_plus = extract.pickup_to_trajectory_p32(Path(reports["plus"]["manifest"]["pickup_meta_path"]), grid.wet)
        initial_minus = extract.pickup_to_trajectory_p32(Path(reports["minus"]["manifest"]["pickup_meta_path"]), grid.wet)
        magnitude_minus, magnitude_plus, antisymmetry = extract._p32_realized_and_antisymmetry(
            new_row, grid.wet, sigma, initial_nominal, initial_plus, initial_minus
        )
        input_state_p32 = np.stack([initial_minus, initial_plus], axis=0).astype(np.float32)

        leads = extract.LONG_LEADS_PRODUCTION if new_row["long"] else (10,)
        response_by_lead = []
        response_hashes = []
        for sign_token, sign in (("minus", -1), ("plus", 1)):
            per_lead = []
            for lead in leads:
                meta_path, _dp = extract._checkpoint_paths(reports[sign_token], lead)
                perturbed = extract.pickup_to_trajectory_p64(meta_path, grid.wet)
                nom_meta, _nd = extract._checkpoint_paths(nominal_report, lead)
                nominal_state = extract.pickup_to_trajectory_p64(nom_meta, grid.wet)
                delta = perturbed - nominal_state
                per_lead.append(delta)
                response_hashes.append(extract._array_sha256(delta, "<f8"))
            response_by_lead.append(np.stack(per_lead, axis=0))
        response_p64 = np.stack(response_by_lead, axis=0).astype(np.float64)

        native_count, centred_count = inv._support_counts(
            extract._RowLike(family=new_row["family"], kernel=new_row["kernel"], levels=levels)
        )
        kernel = new_row["kernel"]
        new_entry = dict(old_entry)
        new_entry.update(
            {
                "j_index0": candidate.j,
                "i_index0": candidate.i,
                "longitude_deg": candidate.lon,
                "latitude_deg": candidate.lat,
                "native_support_count": native_count,
                "centred_support_count": centred_count,
                "physical_peak": physical_peak,
                "physical_rms": physical_rms,
                "physical_l2": physical_l2,
                "p32_realized_standardized_rms": [magnitude_minus, magnitude_plus],
                "p32_antisymmetry_relative_error": antisymmetry,
                "sparse_edits_minus": [extract._sparse_edit_json(e) for e in edits_minus],
                "sparse_edits_plus": [extract._sparse_edit_json(e) for e in edits_plus],
                "input_hashes": [
                    inv.sha256_file(Path(reports["minus"]["manifest"]["pickup_meta_path"]).with_suffix(".data")),
                    inv.sha256_file(Path(reports["plus"]["manifest"]["pickup_meta_path"]).with_suffix(".data")),
                ],
                "response_hashes": response_hashes,
            }
        )
        direction_table[index] = new_entry

        array_group = new_entry["array_group"]
        array_row = new_entry["array_row"]
        val_group[array_group]["input_state_p32"][array_row] = input_state_p32
        val_group[array_group]["response_p64"][array_row] = response_p64
        patched_rows += 1
        print(f"patched direction_table[{index}] and {array_group}[{array_row}] for {key}")

    backup_dt = direction_table_path.with_suffix(direction_table_path.suffix + ".pre_gate_d3_centre_repair_2026-08-27.bak")
    os.chmod(direction_table_path, 0o600)
    backup_dt.write_bytes(direction_table_path.read_bytes())
    os.chmod(backup_dt, 0o400)
    payload = "".join(inv.canonical_json(r) + "\n" for r in direction_table).encode("utf-8")
    direction_table_path.write_bytes(payload)
    os.chmod(direction_table_path, 0o444)
    print(f"patched {direction_table_path} (backup {backup_dt}), sha256={hashlib.sha256(payload).hexdigest()}, {patched_rows} rows changed")

    zarr.consolidate_metadata(store.store)

    report_path = OUTPUT_ROOT / "gate_d3_validation_centre_repair_2026-08-27.json"
    report_payload = [
        {
            "key": list(key),
            "new_centre": {"j": accepted["candidate"].j, "i": accepted["candidate"].i,
                           "lon": accepted["candidate"].lon, "lat": accepted["candidate"].lat},
            "attempts": accepted["attempt"],
            "detail": accepted["detail"],
        }
        for key, accepted in results.items()
    ]
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())
