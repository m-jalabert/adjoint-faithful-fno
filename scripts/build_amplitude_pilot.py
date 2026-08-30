"""Execution step 7 of docs/Adjoint_faithful_response_training_plan.md.

The forward-only amplitude pilot (plan section 7.2/10): six anchors (S0/S1/S2
x days 720/3600), one U/V/Theta/SSH direction each (24 base directions), each
run at candidate alphas {0.025, 0.05, 0.10} and both signs (144 signed
branches), plus 6 paired 90-day nominal branches and their 6 duplicates.
Twelve of the 24 base directions (one per input group and regime, per the
frozen ``long_assignment`` table) run to 90 days; the rest stop at 10 days.

This step never reads adjoint or blind data, so -- like step 6 -- it needs
none of the OS-identity machinery retired in the roadmap amendments.

Subcommands:

``materialize-geometry``
    Solve the 24 pilot centres once (the same deterministic joint solve
    ``build_forward_response_inventory`` uses for the full inventory,
    restricted here to the pilot rows) and freeze them to a JSON file. Must
    run before any of the below. No MITgcm compute, no numeric response.

``directions``
    Print the frozen per-direction kernel/scale/edit summary. Read-only.

``run-signed --regime --day --family --alpha --sign``
    Build the additively-edited pickup for one signed pilot branch and run
    it for its declared horizon (10 or 90 days).

``run-nominal --regime --day [--duplicate]``
    Run the unperturbed 90-day paired nominal branch (or its duplicate, used
    to measure the deterministic/numerical floor).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "archive" / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from bire_repro import af_pilot_segment as segment  # noqa: E402
from bire_repro.af_s0_twin import PickupEdit, pickup_record_index, write_declared_pickup_edits  # noqa: E402
from build_forward_response_inventory import (  # noqa: E402
    DEFAULT_DATASET_CONTRACT,
    DEFAULT_PILOT_CONTRACT,
    REGIMES,
    ContractError,
    InventoryError,
    SourceError,
    _prepare_inventory_context,
    enumerate_candidates,
    load_json_strict,
    mean_surface_speed_already_centered,
    resolve_annual_pickup,
    sha256_file,
)


DEFAULT_GEOMETRY_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "af_fno"
    / "response"
    / "forward_response_v1"
    / "amplitude_pilot_geometry_v1.json"
)
DEFAULT_SCRATCH_ROOT = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno")
DEFAULT_EXECUTABLE = PROJECT_ROOT / "build" / "af_s0" / "mitgcmuv"
DEFAULT_REPORT_ROOT = (
    PROJECT_ROOT / "outputs" / "af_fno" / "response" / "forward_response_v1" / "amplitude_pilot"
)

FIELD_BY_FAMILY = {"U": "Uvel", "V": "Vvel", "Theta": "Theta", "SSH": "EtaN"}
CHANNEL_BASE = {"U": 0, "V": 15, "Theta": 30}
SSH_PEAK_METERS_MAX = 0.01


class PilotOrchestratorError(RuntimeError):
    """Raised when the amplitude pilot cannot legitimately proceed."""


# ---------------------------------------------------------------------------
# Geometry: the 24 pilot centres, solved once and frozen to a file.


def _select_pilot_centre(candidates: list, region: str):
    """Section-9.3 tie-break rule (iii) plus the SHA tie, specialized to pilot.

    The full frozen algorithm lexicographically maximizes (i) minimum
    cross-role separation, then (ii) minimum within-role separation, before
    (iii) this tie-break. For pilot specifically, both (i) and (ii) are
    provably vacuous: pilot is first in ``ROLE_ORDER`` so it has no earlier
    role to separate from (see the caller's docstring), and within each
    (regime, family) group pilot has exactly two rows whose regions are
    always different by construction of the frozen anchor/region table
    (section 7.2) -- so their candidate pools are disjoint masks with no
    possible overlap for any within-role objective to arbitrate. The
    lexicographic max therefore reduces exactly to (iii) for pilot: this is
    not an approximation, it is what the full algorithm would compute here.
    ``enumerate_candidates`` already sorts its output by ``tie_sha256``
    ascending, so picking with a ``(objective, tie_sha256)`` key reproduces
    the frozen SHA tie-break exactly.
    """

    if region == "WBC":
        return min(candidates, key=lambda c: (-c.wbc_speed, c.tie_sha256))
    return min(candidates, key=lambda c: (c.tertiary_distance_km, c.tie_sha256))


def _solve_pilot_centres(
    dataset_contract_path: Path = DEFAULT_DATASET_CONTRACT,
    pilot_contract_path: Path = DEFAULT_PILOT_CONTRACT,
):
    """Solve the 24 pilot centres. See :func:`_select_pilot_centre` for why
    this reduces to a per-row selection instead of the full joint MIP that
    ``allocate_centres_lexicographically`` runs for the complete inventory
    (which needs cross-role rows to be meaningful -- called with pilot rows
    alone it hits an empty-region-pair edge case -- and which timed out past
    180s on the full 1128-row problem in any case).
    """

    contract, _pilot, roots, grid, state, rows, masks, _context = _prepare_inventory_context(
        dataset_contract_path, pilot_contract_path
    )
    pilot_rows_in = [row for row in rows if row.role == "pilot"]
    if len(pilot_rows_in) != 24:
        raise ContractError(f"expected 24 pilot rows before solving, got {len(pilot_rows_in)}")
    speed_by_regime = {
        regime: mean_surface_speed_already_centered(state, index)
        for index, regime in enumerate(REGIMES)
    }
    pilot_rows = []
    for row in pilot_rows_in:
        candidates = enumerate_candidates(grid, row, masks, wbc_speed=speed_by_regime[row.regime])
        if not candidates:
            raise ContractError(f"no eligible candidates for {row.slot_id}")
        best = _select_pilot_centre(candidates, row.region)
        pilot_rows.append(replace(row, j=best.j, i=best.i, lon=best.lon, lat=best.lat))
    if len({(row.j, row.i, row.regime, row.family) for row in pilot_rows}) != 24:
        raise ContractError("pilot centre solve produced a duplicate (regime, family) centre")
    return pilot_rows, contract, roots, grid


def materialize_geometry(
    output: Path = DEFAULT_GEOMETRY_OUTPUT,
    dataset_contract_path: Path = DEFAULT_DATASET_CONTRACT,
    pilot_contract_path: Path = DEFAULT_PILOT_CONTRACT,
) -> dict[str, Any]:
    pilot_rows, _contract, _roots, grid = _solve_pilot_centres(
        dataset_contract_path, pilot_contract_path
    )
    rows_json = [
        {
            "regime": row.regime,
            "anchor_day": row.anchor_day,
            "family": row.family,
            "kernel": row.kernel,
            "region": row.region,
            "levels": list(row.levels),
            "long": row.long,
            "j": row.j,
            "i": row.i,
            "lon": row.lon,
            "lat": row.lat,
        }
        for row in pilot_rows
    ]
    payload = {
        "version": "amplitude_pilot_geometry_v1",
        "grid_wet_tracer_cells": int(grid.wet.sum()),
        "directions": rows_json,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _load_geometry(path: Path = DEFAULT_GEOMETRY_OUTPUT) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PilotOrchestratorError(f"pilot geometry is not materialized yet: {path}")
    return load_json_strict(path)["directions"]


def _find_direction(
    rows: list[dict[str, Any]], regime: str, day: int, family: str
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["regime"] == regime and row["anchor_day"] == day and row["family"] == family
    ]
    if len(matches) != 1:
        raise PilotOrchestratorError(
            f"expected exactly one direction for {regime}/{day}/{family}, found {len(matches)}"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Kernel construction and RMS-standardized direction vectors (section 8.1/10.1).


def _gaussian_weights() -> np.ndarray:
    offsets = range(-2, 3)
    raw = np.array([[math.exp(-(a * a + b * b) / 2.0) for b in offsets] for a in offsets])
    return raw / math.sqrt(float(np.sum(raw**2)))


_GAUSSIAN_5X5 = _gaussian_weights()


def _native_kernel(
    direction: Mapping[str, Any], grid_shape: tuple[int, int] = (62, 62)
) -> np.ndarray:
    j0, i0 = int(direction["j"]), int(direction["i"])
    array = np.zeros(grid_shape, dtype=np.float64)
    if direction["kernel"] == "point":
        array[j0, i0] = 1.0
        return array
    if not (2 <= j0 <= grid_shape[0] - 3 and 2 <= i0 <= grid_shape[1] - 3):
        raise ContractError(f"pilot centre too close to the grid edge: {direction}")
    array[j0 - 2 : j0 + 3, i0 - 2 : i0 + 3] = _GAUSSIAN_5X5
    return array


def _centred_projection(family: str, native: np.ndarray) -> np.ndarray:
    if family == "U":
        return 0.5 * (native + np.roll(native, -1, axis=1))
    if family == "V":
        return 0.5 * (native + np.roll(native, -1, axis=0))
    return native


def channel_index(family: str, levels: Sequence[int]) -> int:
    if family == "SSH":
        return 45
    (level,) = levels
    return CHANNEL_BASE[family] + (int(level) - 1)


def direction_vector(
    direction: Mapping[str, Any], wet_mask: np.ndarray, sigma_46_62_62: np.ndarray
) -> np.ndarray:
    """The alpha=1, sign=+1 native-grid edit satisfying the section-10.1 unit-RMS rule."""

    native = _native_kernel(direction)
    centred = _centred_projection(direction["family"], native)
    sigma = sigma_46_62_62[channel_index(direction["family"], direction["levels"])]
    support = (centred != 0.0) & wet_mask
    if not support.any():
        raise ContractError(f"direction has empty wet centred support: {direction}")
    standardized = centred[support] / sigma[support]
    rms = math.sqrt(float(np.mean(standardized**2)))
    if not math.isfinite(rms) or rms <= 0.0:
        raise ContractError(f"direction has a non-finite RMS scale: {direction}")
    return native / rms


def pickup_edits_for(
    direction: Mapping[str, Any], v_q: np.ndarray, alpha: float, sign: int
) -> tuple[str, list[PickupEdit], float]:
    field = FIELD_BY_FAMILY[direction["family"]]
    level = int(direction["levels"][0]) if direction["levels"] else 1
    record = pickup_record_index(field, level)
    edits: list[PickupEdit] = []
    peak = 0.0
    for j, i in np.argwhere(v_q != 0.0):
        value = float(sign) * float(alpha) * float(v_q[j, i])
        edits.append(PickupEdit(record=record, j=int(j), i=int(i), value=value))
        peak = max(peak, abs(value))
    if not edits:
        raise ContractError(f"direction produced no nonzero edits: {direction}")
    return field, edits, peak


# ---------------------------------------------------------------------------
# Sources and normalizer.


def _load_sources(dataset_contract_path: Path):
    contract = load_json_strict(dataset_contract_path)
    from build_forward_response_inventory import (
        _verified_chain_roots,
        read_grid,
        verify_trajectory_store,
    )

    roots = _verified_chain_roots(contract)
    grid = read_grid(contract["sources"]["grid"]["canonical_root"])
    _state, _report = verify_trajectory_store(contract, grid)
    return contract, roots, grid


def _load_normalizer(pilot_contract: Mapping[str, Any]) -> np.ndarray:
    normalization = pilot_contract["normalization"]
    path = Path(str(normalization["parent_artifact"]))
    expected = str(normalization["sha256"])
    if sha256_file(path) != expected:
        raise SourceError(f"parent normalizer hash mismatch: {path}")
    return np.load(path)["pointwise_scale"]


def _resolve_source(regime: str, day: int, roots) -> tuple[Path, Path, str, str]:
    resolution = resolve_annual_pickup(regime, day, roots)
    return (
        Path(resolution.canonical.meta_path),
        Path(resolution.canonical.data_path),
        resolution.canonical.meta_sha256,
        resolution.canonical.data_sha256,
    )


# ---------------------------------------------------------------------------
# Run one signed or nominal branch.


def _alpha_token(alpha: float) -> str:
    return f"{alpha:.3f}".replace("0.", "p").replace(".", "p")


def run_signed(
    regime: str,
    day: int,
    family: str,
    alpha: float,
    sign: int,
    *,
    project_root: Path = PROJECT_ROOT,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    executable: Path = DEFAULT_EXECUTABLE,
    dataset_contract_path: Path = DEFAULT_DATASET_CONTRACT,
    pilot_contract_path: Path = DEFAULT_PILOT_CONTRACT,
    geometry_path: Path = DEFAULT_GEOMETRY_OUTPUT,
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> dict[str, Any]:
    if sign not in (-1, 1):
        raise PilotOrchestratorError("sign must be -1 or 1")
    direction = _find_direction(_load_geometry(geometry_path), regime, day, family)
    contract, roots, grid = _load_sources(dataset_contract_path)
    pilot_contract = load_json_strict(pilot_contract_path)
    sigma = _load_normalizer(pilot_contract)
    v_q = direction_vector(direction, grid.wet, sigma)
    field, edits, peak = pickup_edits_for(direction, v_q, alpha, sign)
    if family == "SSH" and peak > SSH_PEAK_METERS_MAX:
        raise PilotOrchestratorError(
            f"SSH direction {regime}/{day} alpha {alpha} exceeds the {SSH_PEAK_METERS_MAX} m cap "
            f"(peak {peak:.6f} m); this alpha is a recorded failure, not silently clipped"
        )

    source_meta, source_data, source_meta_sha256, source_data_sha256 = _resolve_source(
        regime, day, roots
    )
    sign_token = "plus" if sign == 1 else "minus"
    run_label = f"{regime}_d{day:04d}_{family}_a{_alpha_token(alpha)}_{sign_token}"
    staging_dir = scratch_root / "mitgcm_amplitude_pilot_v1" / "_edited_pickups" / run_label
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_meta = staging_dir / source_meta.name
    staged_data = staging_dir / source_data.name
    if not (staged_meta.is_file() and staged_data.is_file()):
        write_declared_pickup_edits(
            source_meta,
            staging_dir,
            expected_iteration=segment.day_to_iteration(day),
            declared_fields=(field,),
            edits=edits,
            operation="add",
        )

    duration_days = 90 if direction["long"] else 10
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
        day,
        duration_days,
    )
    result = segment.run_segment(manifest)
    report = {
        "kind": "signed",
        "run_label": run_label,
        "regime": regime,
        "day": day,
        "family": family,
        "alpha": alpha,
        "sign": sign,
        "long": bool(direction["long"]),
        "duration_days": duration_days,
        "ssh_peak_m": peak if family == "SSH" else None,
        "manifest": manifest,
        "result": {key: value for key, value in result.items() if key != "archived_pickups"},
        "archived_pickup_count": len(result["archived_pickups"]),
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / f"{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def run_nominal(
    regime: str,
    day: int,
    *,
    duplicate: bool = False,
    cg2d_target_residual: float | None = None,
    project_root: Path = PROJECT_ROOT,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    executable: Path = DEFAULT_EXECUTABLE,
    dataset_contract_path: Path = DEFAULT_DATASET_CONTRACT,
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> dict[str, Any]:
    contract, roots, _grid = _load_sources(dataset_contract_path)
    source_meta, source_data, source_meta_sha256, source_data_sha256 = _resolve_source(
        regime, day, roots
    )
    suffix = "_tight" if cg2d_target_residual is not None else ("_dup" if duplicate else "")
    run_label = f"{regime}_d{day:04d}_nominal{suffix}"
    manifest = segment.prepare_segment(
        project_root,
        scratch_root,
        executable,
        run_label,
        source_meta.parent,
        source_meta,
        source_data,
        source_meta_sha256,
        source_data_sha256,
        day,
        90,
        cg2d_target_residual=cg2d_target_residual,
    )
    result = segment.run_segment(manifest)
    report = {
        "kind": "nominal",
        "run_label": run_label,
        "regime": regime,
        "day": day,
        "duplicate": duplicate,
        "cg2d_target_residual": cg2d_target_residual,
        "duration_days": 90,
        "manifest": manifest,
        "result": {key: value for key, value in result.items() if key != "archived_pickups"},
        "archived_pickup_count": len(result["archived_pickups"]),
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / f"{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def run_control(
    regime: str,
    day: int,
    family: str,
    alpha: float,
    sign: int,
    condition: str,
    *,
    project_root: Path = PROJECT_ROOT,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    executable: Path = DEFAULT_EXECUTABLE,
    dataset_contract_path: Path = DEFAULT_DATASET_CONTRACT,
    geometry_path: Path = DEFAULT_GEOMETRY_OUTPUT,
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> dict[str, Any]:
    """Section 10.3: an independent rerun of an already-selected signed branch.

    ``condition="duplicate"`` reruns at the production 1.E-7 solver tolerance
    (measures run-to-run/deterministic noise); ``condition="tight"`` reruns
    at 1.E-10 (measures solver-tolerance noise). Both reuse the *existing*
    staged edited pickup from the original ``run-signed`` call at this exact
    (regime, day, family, alpha, sign) -- the physical edit does not change,
    only the MITgcm solver tolerance and the run directory, so no new pickup
    editing is needed or performed.
    """

    if condition not in ("duplicate", "tight"):
        raise PilotOrchestratorError("condition must be 'duplicate' or 'tight'")
    if sign not in (-1, 1):
        raise PilotOrchestratorError("sign must be -1 or 1")
    direction = _find_direction(_load_geometry(geometry_path), regime, day, family)
    if not direction["long"]:
        raise PilotOrchestratorError(
            f"{regime}/{day}/{family} is not one of the 12 preassigned long directions"
        )
    contract, roots, _grid = _load_sources(dataset_contract_path)
    source_meta, _source_data, _m, _d = _resolve_source(regime, day, roots)

    sign_token = "plus" if sign == 1 else "minus"
    original_label = f"{regime}_d{day:04d}_{family}_a{_alpha_token(alpha)}_{sign_token}"
    staging_dir = scratch_root / "mitgcm_amplitude_pilot_v1" / "_edited_pickups" / original_label
    staged_meta = staging_dir / source_meta.name
    staged_data = staged_meta.with_suffix(".data")
    if not (staged_meta.is_file() and staged_data.is_file()):
        raise PilotOrchestratorError(
            f"expected an already-staged edited pickup from run-signed at {staging_dir}; "
            "run the provisional-alpha signed branch first"
        )

    run_label = f"{original_label}_{condition}"
    cg2d = 1.0e-10 if condition == "tight" else None
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
        day,
        90,
        cg2d_target_residual=cg2d,
    )
    result = segment.run_segment(manifest)
    report = {
        "kind": "control",
        "condition": condition,
        "run_label": run_label,
        "regime": regime,
        "day": day,
        "family": family,
        "alpha": alpha,
        "sign": sign,
        "cg2d_target_residual": cg2d,
        "duration_days": 90,
        "manifest": manifest,
        "result": {key: value for key, value in result.items() if key != "archived_pickups"},
        "archived_pickup_count": len(result["archived_pickups"]),
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / f"{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    geometry_parser = subparsers.add_parser("materialize-geometry")
    geometry_parser.add_argument("--output", type=Path, default=DEFAULT_GEOMETRY_OUTPUT)

    directions_parser = subparsers.add_parser("directions")
    directions_parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY_OUTPUT)

    signed_parser = subparsers.add_parser("run-signed")
    signed_parser.add_argument("--regime", choices=REGIMES, required=True)
    signed_parser.add_argument("--day", type=int, required=True)
    signed_parser.add_argument("--family", choices=("U", "V", "Theta", "SSH"), required=True)
    signed_parser.add_argument("--alpha", type=float, required=True)
    signed_parser.add_argument("--sign", type=int, required=True, choices=(-1, 1))
    signed_parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    signed_parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    signed_parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)

    nominal_parser = subparsers.add_parser("run-nominal")
    nominal_parser.add_argument("--regime", choices=REGIMES, required=True)
    nominal_parser.add_argument("--day", type=int, required=True)
    nominal_parser.add_argument("--duplicate", action="store_true")
    nominal_parser.add_argument(
        "--tight", action="store_true", help="section 10.3: cg2dTargetResidual=1.E-10"
    )
    nominal_parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    nominal_parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    nominal_parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)

    control_parser = subparsers.add_parser("run-control")
    control_parser.add_argument("--regime", choices=REGIMES, required=True)
    control_parser.add_argument("--day", type=int, required=True)
    control_parser.add_argument("--family", choices=("U", "V", "Theta", "SSH"), required=True)
    control_parser.add_argument("--alpha", type=float, required=True)
    control_parser.add_argument("--sign", type=int, required=True, choices=(-1, 1))
    control_parser.add_argument("--condition", choices=("duplicate", "tight"), required=True)
    control_parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    control_parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    control_parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)

    arguments = parser.parse_args(argv)
    try:
        if arguments.mode == "materialize-geometry":
            result = materialize_geometry(arguments.output)
        elif arguments.mode == "directions":
            result = {"directions": _load_geometry(arguments.geometry)}
        elif arguments.mode == "run-signed":
            result = run_signed(
                arguments.regime,
                arguments.day,
                arguments.family,
                arguments.alpha,
                arguments.sign,
                project_root=arguments.project_root.resolve(),
                scratch_root=arguments.scratch_root.resolve(),
                executable=arguments.executable.resolve(),
            )
        elif arguments.mode == "run-nominal":
            result = run_nominal(
                arguments.regime,
                arguments.day,
                duplicate=arguments.duplicate,
                cg2d_target_residual=1.0e-10 if arguments.tight else None,
                project_root=arguments.project_root.resolve(),
                scratch_root=arguments.scratch_root.resolve(),
                executable=arguments.executable.resolve(),
            )
        else:
            result = run_control(
                arguments.regime,
                arguments.day,
                arguments.family,
                arguments.alpha,
                arguments.sign,
                arguments.condition,
                project_root=arguments.project_root.resolve(),
                scratch_root=arguments.scratch_root.resolve(),
                executable=arguments.executable.resolve(),
            )
    except (InventoryError, segment.PilotSegmentError, PilotOrchestratorError) as error:
        print(f"AMPLITUDE PILOT: FAIL -- {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
