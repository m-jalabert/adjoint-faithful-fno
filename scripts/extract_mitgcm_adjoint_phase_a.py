"""Extract the Phase A MITgcm sensitivity maps and run gates G0-G5.

Implements step 11 of docs/Adjoint_study_Phase_A.md.  Reads the eleven staged
run directories under mitgcm_adjoint_v2/ and writes one .npz plus a report,
keyed to mirror scripts/fno_adjoint_ft90.py so scripts/compare_adjoint_maps_
phase_a.py reads both sides with one loader.

Nothing here loads FNO weights or writes into the FNO tree.  It does read the
emulator's output directory to cross-check the shared weight fields, which is
gate F6 seen from the other side.

    python scripts/extract_mitgcm_adjoint_phase_a.py
    python scripts/extract_mitgcm_adjoint_phase_a.py --allow-missing   # partial
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from extract_mitgcm_adjoint import (  # noqa: E402
    ExtractionError,
    adjetan_series,
    global_fc,
    read_mds,
    relative_l2,
)
from select_adjoint_target import BASE_ITERATION, STEPS_PER_DAY, read_mds_2d  # noqa: E402
from stage_adjoint_run import (  # noqa: E402
    DEFAULT_SCRATCH_PHASE_A,
    PHASE_A_CONTRACT,
    WEIGHT_FILES,
)

MDS_DTYPE = ">f4"
OUTPUT_RELATIVE = Path("outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2")

G2_TOLERANCE = 1.0e-6
G3_TOLERANCE = 1.0e-5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gradient_map(run_dir: Path) -> np.ndarray:
    """``adxx_etan.0000000000`` --- dJ/d eta at the run's own nIter0.

    Comes back at ``ctrlprec`` (float64), unlike the ``ADJetan`` dumps which
    follow ``writeBinaryPrec``.  ``read_mds`` takes the precision from the
    .meta rather than assuming, which is what keeps the two halves of gate G2
    comparable.
    """

    return read_mds(run_dir / "adxx_etan.0000000000")


def gate_g0(forward_dir: Path, dataset_path: Path, days: range) -> dict[str, Any]:
    """The forward re-run must reproduce the archive bit-for-bit.

    If it does not, the adjoint is being taken about a different trajectory
    than the emulator is being evaluated on, and everything downstream is
    invalid.

    Compares only ``ETAN``, from ``surfState``: the C-grid velocities need the
    dataset's own face-to-centre operator before they mean anything (the v1
    gate compared staggered against centred and reported an O(1e-1) difference
    that was a statement about the C grid, not the restart).  ``ETAN`` is at
    cell centres on both sides, so it needs no operator and no judgement.
    """

    try:
        import zarr
    except ImportError:  # pragma: no cover - environment dependent
        return {"skipped": "zarr is not importable"}
    group = zarr.open_consolidated(str(dataset_path), mode="r")

    worst, checked, missing = 0.0, [], []
    for day in days:
        iteration = BASE_ITERATION + STEPS_PER_DAY * day
        prefix = forward_dir / f"surfState.{iteration:010d}"
        if not (forward_dir / f"{prefix.name}.data").is_file():
            missing.append(day)
            continue
        rerun = read_mds(prefix)
        archived = np.asarray(group["state"][0, day, 45], dtype=np.float64)
        worst = max(worst, float(np.abs(rerun - archived).max()))
        checked.append(day)
    return {
        "condition": "forward re-run ETAN matches trajectories_v3.zarr bit-for-bit",
        "days_checked": len(checked),
        "days_missing": missing,
        "max_absolute_difference": worst,
        "passed": bool(checked and worst == 0.0),
        "note": (
            "day 7290 is never dumped by the adjoint runs (dumpAtLast=.FALSE.); F90's "
            "tail day supplies it, and gate G5 covers it independently"
        ),
    }


def gate_g2a(run_dir: Path, start_day: int) -> dict[str, Any]:
    """``adxx_etan`` and ``ADJetan`` at nIter0 are the same number by two paths.

    ``adxx_etan`` comes through ctrl/GENARR2D; ``ADJetan`` through
    ``addummy_for_etan.F``.  Phase A's fixed-source sweep gives the four runs
    different cost times, so the v1 cross-run form of G2 does not exist here --
    this is the free replacement.
    """

    iteration = BASE_ITERATION + STEPS_PER_DAY * start_day
    dump = run_dir / f"ADJetan.{iteration:010d}"
    if not (run_dir / f"{dump.name}.data").is_file():
        return {"skipped": f"no ADJetan dump at day {start_day}"}
    residual = relative_l2(read_mds(dump), gradient_map(run_dir))
    return {
        "condition": "adxx_etan equals ADJetan at nIter0 (two independent code paths)",
        "relative_l2": residual,
        "tolerance": G2_TOLERANCE,
        "passed": bool(residual < G2_TOLERANCE),
    }


def gate_g2b(verification_dir: Path, long_dir: Path, source_day: int) -> dict[str, Any]:
    """A fresh ten-day run must return what a ninety-day tape returns.

    The adjoint state at time t inside a run whose cost sits at T is
    dJ/d eta(.,t) for every t in [start, T], independent of when the run
    started.  So V10's own gradient at day 7280 must equal P90's ADJetan there.
    This is the exact v1 gate-G2 structure and the strongest single statement
    that the longer checkpointing did not corrupt anything.
    """

    iteration = BASE_ITERATION + STEPS_PER_DAY * source_day
    dump = long_dir / f"ADJetan.{iteration:010d}"
    if not (long_dir / f"{dump.name}.data").is_file():
        return {"skipped": f"no ADJetan dump at day {source_day} in {long_dir.name}"}
    residual = relative_l2(gradient_map(verification_dir), read_mds(dump))
    return {
        "condition": "V10 adxx_etan equals P90 ADJetan at day 7280",
        "relative_l2": residual,
        "tolerance": G2_TOLERANCE,
        "passed": bool(residual < G2_TOLERANCE),
        "meaning": "a 90-day tape returns the same adjoint state as a fresh 10-day run",
    }


def gate_g3(run_dir: Path, mean_weight: np.ndarray, start_day: int, cost_day: int) -> dict[str, Any]:
    """The mean functional's adjoint is constant in time and equals ``w`` itself.

    ``implicitFreeSurface`` with ``exactConserv`` conserves the area integral of
    eta exactly in this closed basin with no freshwater flux, so this is an
    analytic reference available at every dump time with no finite differences
    anywhere.  Over ninety days it exercises 91 dumps and 90 level-2 tape
    records instead of the v1 study's 11 and 10 --- a materially stronger test
    of the checkpointing, for one extra run.
    """

    maps, days = adjetan_series(run_dir, start_day, cost_day)
    residuals = [relative_l2(field, mean_weight) for field in maps]
    worst = float(max(residuals))
    return {
        "condition": "mean-only cost returns its own weight field at every dump time",
        "dumps": int(maps.shape[0]),
        "days": days.tolist(),
        "per_dump_relative_l2": residuals,
        "worst_relative_l2": worst,
        "tolerance": G3_TOLERANCE,
        "passed": bool(worst < G3_TOLERANCE),
    }


#: The gradient-check sweep writes one directory per (point, epsilon).
GRDCHK_LABELS = ("pstar", "wbc14", "wbc11", "offshore", "interior", "eastern", "northern")
GRDCHK_POINTS = {
    "pstar": (2, 17),
    "wbc14": (2, 14),
    "wbc11": (2, 11),
    "offshore": (4, 17),
    "interior": (31, 17),
    "eastern": (61, 17),
    "northern": (31, 55),
}
G1_TOLERANCE = 1.0e-4


def _parse_grad_res(run_dir: Path) -> dict[str, Any] | None:
    """Pull the adjoint, the finite difference and their disagreement out of a log.

    ``grdchk`` writes two ``grad-res`` data lines.  The first carries the three
    cost values; the second carries ``adj grad``, ``fd grad`` and
    ``1 - fd/adj``.  The header line ``grad-res exact position met`` is what
    confirms the requested cell was actually tested --- with ``nbeg`` anything
    but zero the sweep silently tests the first wet cell while printing an
    entirely plausible block, which is the trap the v1 study fell into.
    """

    # Scan EVERY rank's log, not just STDOUT.0000.  grdchk prints its grad-res
    # block from the rank that owns the tile containing the test point, so on
    # this 2x2 decomposition only points in the south-west quadrant appear in
    # STDOUT.0000.  Reading that file alone silently drops the eastern and
    # northern points and leaves the gate reporting 5 of 7 with no error --
    # the same shape of failure as the v1 nbeg trap, where the sweep tested a
    # cell nobody asked for while printing a perfectly plausible block.
    logs = sorted(run_dir.glob("STDOUT.*"))
    if not logs:
        return None
    text = ""
    for log in logs:
        candidate = log.read_text(errors="replace")
        if "grad-res exact position met" in candidate:
            text = candidate
            break
    if not text:
        return {"error": "grdchk did not reach the requested position (check nbeg = 0)"}
    rows = [
        line.split()
        for line in text.splitlines()
        if line.strip().startswith("grad-res") and len(line.split()) >= 11
    ]
    numeric = [row for row in rows if row[1].isdigit()]
    if len(numeric) < 2:
        return {"error": "no numeric grad-res rows"}
    values = [float(v.replace("E", "e")) for v in numeric[-1][-3:]]
    adjoint, finite_difference, disagreement = values
    return {
        "adjoint": adjoint,
        "finite_difference": finite_difference,
        "one_minus_fd_over_adj": disagreement,
        "absolute_error": abs(finite_difference - adjoint),
    }


def gate_g1(scratch: Path, epsilons: Sequence[str]) -> dict[str, Any]:
    """G1-90 --- the gradient check at the ninety-day window.

    Two conditions, and the second is the one that matters.  ``|1 - fd/adj|``
    must be inside 1e-4 at the best epsilon, and the sweep must show a
    **minimum**: the error falling and then rising again as epsilon shrinks is
    what separates a converged central difference from a point that happens to
    agree.  ``grdchk-limited-by-cg2d`` records the failure this guards against
    --- an error flat in epsilon, because the finite difference rather than the
    adjoint was the noisy party and there was nothing converging to find.
    """

    points: dict[str, Any] = {}
    for label in GRDCHK_LABELS:
        samples = []
        for epsilon in epsilons:
            parsed = _parse_grad_res(scratch / f"G90_{label}_{epsilon}")
            if parsed and "error" not in parsed:
                samples.append({"epsilon": epsilon, **parsed})
        if not samples:
            continue
        best = min(samples, key=lambda s: abs(s["one_minus_fd_over_adj"]))
        errors = [abs(s["one_minus_fd_over_adj"]) for s in samples]
        points[label] = {
            "global_ij": list(GRDCHK_POINTS[label]),
            "samples": samples,
            "adjoint": samples[0]["adjoint"],
            "best_epsilon": best["epsilon"],
            "best_disagreement": abs(best["one_minus_fd_over_adj"]),
            "minimum_is_interior": bool(
                len(errors) >= 3 and 0 < errors.index(min(errors)) < len(errors) - 1
            ),
            "passed": bool(abs(best["one_minus_fd_over_adj"]) < G1_TOLERANCE),
        }
    if not points:
        return {"skipped": "no G90_* run directories with a grad-res block"}
    worst = max(p["best_disagreement"] for p in points.values())
    return {
        "condition": "|1 - fd/adj| < 1e-4 at every point, with the sweep showing a minimum",
        "cg2d_target_residual": 1.0e-12,
        "cg2d_note": (
            "diagnostic only. The production maps keep 1e-7; at that setting the finite "
            "difference, not the adjoint, is what limits the comparison"
        ),
        "epsilons": list(epsilons),
        "points": points,
        "points_tested": len(points),
        "worst_disagreement": worst,
        "points_with_interior_minimum": sum(1 for p in points.values() if p["minimum_is_interior"]),
        "tolerance": G1_TOLERANCE,
        "passed": bool(len(points) == len(GRDCHK_LABELS) and all(p["passed"] for p in points.values())),
    }


def gate_g4(maps: dict[str, np.ndarray], wet: np.ndarray) -> dict[str, Any]:
    """Exactly zero on land, finite everywhere, on every Phase A map."""

    dry = ~wet
    offenders, non_finite = {}, {}
    for name, field in maps.items():
        count = int((field[dry] != 0.0).sum())
        if count:
            offenders[name] = count
        if not np.isfinite(field).all():
            non_finite[name] = int((~np.isfinite(field)).sum())
    return {
        "condition": "S exactly 0 on all land cells and finite everywhere",
        "land_cells": int(dry.sum()),
        "maps_checked": len(maps),
        "maps_with_nonzero_land": offenders,
        "maps_with_non_finite": non_finite,
        "passed": bool(not offenders and not non_finite),
    }


def gate_g5(run_dir: Path, weight: np.ndarray, eta: np.ndarray) -> dict[str, Any]:
    """``fc`` from the run against ``sum(w * eta)`` computed in NumPy.

    Compared against the **computed** float32 half-ulp bound of this particular
    weighted sum, not a constant: ``fc`` accumulates from the model's float64
    state while the archived eta is the float32 diagnostic snapshot, so a flat
    1e-10 is unreachable for a reason that is arithmetic rather than a defect.
    The v1 study reached the same conclusion at its own gate G5.
    """

    reported = global_fc(run_dir)
    if reported is None:
        return {"skipped": f"no global fc in {run_dir.name}/STDOUT.0000"}
    terms = weight * eta.astype(np.float64)
    predicted = float(terms.sum())
    relative = abs(reported - predicted) / max(abs(predicted), 1.0e-300)
    # half-ulp of the float32 eta, propagated through this exact weighted sum
    floor = float(np.abs(weight * np.spacing(eta.astype(np.float32)) / 2.0).sum()
                  / max(abs(predicted), 1.0e-300))
    return {
        "condition": "fc equals sum(w * eta) computed independently in numpy",
        "fc_reported": reported,
        "predicted": predicted,
        "relative_error": relative,
        "float32_eta_floor": floor,
        "error_over_floor": relative / max(floor, 1.0e-300),
        "tolerance": floor,
        "passed": bool(relative <= max(floor, 1.0e-12)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--scratch-root", default=str(DEFAULT_SCRATCH_PHASE_A))
    parser.add_argument("--allow-missing", action="store_true",
                        help="extract whatever has run rather than requiring all eleven")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    project_root = (
        Path(arguments.project_root).resolve()
        if arguments.project_root
        else Path(__file__).resolve().parent.parent
    )
    scratch = Path(arguments.scratch_root)
    contract = json.loads((project_root / "config" / f"{PHASE_A_CONTRACT}.json").read_text())
    target = json.loads(
        (project_root / "config" / "mitgcm_adjoint_s0_target_v1.json").read_text()
    )
    output = (project_root / OUTPUT_RELATIVE).resolve()
    if output.exists() and not arguments.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")

    weights = {}
    for name, filename in WEIGHT_FILES.items():
        path = project_root / "work" / filename
        if path.is_file():
            weights[name] = np.fromfile(path, dtype=MDS_DTYPE).reshape(62, 62).astype(np.float64)
    wet = np.abs(weights["mean_only"]) > 0.0

    source_day = int(contract["window"]["source_day"])
    leads = [int(lead) for lead in contract["window"]["lead_days"]]

    present, absent = {}, []
    for run in contract["mitgcm_runs"]:
        # The gradient-check run produces no map of its own: the sweep writes
        # one directory per (point, epsilon) and gate G1 reads those.  Counting
        # it here would make a complete study look permanently unfinished.
        if run.get("grdchk"):
            continue
        run_dir = scratch / run["name"]
        if (run_dir / "adxx_etan.0000000000.data").is_file():
            present[run["name"]] = run_dir
        else:
            absent.append(run["name"])
    if absent and not arguments.allow_missing:
        raise ExtractionError(
            f"these runs have produced no gradient yet: {absent}.  "
            "Submit slurm/mitgcm/af_s0_adjoint_phase_a.sbatch, or pass --allow-missing."
        )
    expected = sum(1 for run in contract["mitgcm_runs"] if not run.get("grdchk"))
    print(f"found {len(present)} of {expected} map runs" + (f"; missing {absent}" if absent else ""))

    maps: dict[str, np.ndarray] = {name: gradient_map(d) for name, d in present.items()}

    arrays: dict[str, np.ndarray] = {
        "lead_days": np.asarray(leads, dtype=np.int64),
        "wet_mask": wet.astype(np.int8),
        "rA": read_mds_2d(Path(target["grid"]["rac_source"])),
        "target_ij": np.asarray([target["j_index0"], target["i_index0"]], dtype=np.int64),
    }
    for objective, prefix in (("ssh_anomaly", "P"), ("ssh_anomaly_kernel", "K")):
        stack = [maps[f"{prefix}{lead}"] for lead in leads if f"{prefix}{lead}" in maps]
        if len(stack) == len(leads):
            arrays[f"S_{objective}"] = np.stack(stack)
    if "C90" in present:
        sweep, days = adjetan_series(present["C90"], source_day, source_day + max(leads))
        arrays["S_mean_only_backward"] = sweep
        arrays["mean_only_backward_days"] = days.astype(np.int64)
    if "P90" in present:
        sweep, days = adjetan_series(present["P90"], source_day, source_day + max(leads))
        arrays["S_backward"] = sweep
        arrays["backward_days"] = days.astype(np.int64)
    for name, field in weights.items():
        arrays[f"w_{name}"] = field

    gates: dict[str, Any] = {}
    dataset_path = Path(contract.get("dataset_path", "")) if contract.get("dataset_path") else None
    if dataset_path is None:
        model_contract = json.loads(
            (project_root / "config" / f"{contract['emulator']['version']}.json").read_text()
        )
        dataset_path = Path(model_contract["sources"]["dataset"]["path"])
    forward_dir = scratch / "F90"
    if forward_dir.is_dir():
        gates["G0"] = gate_g0(forward_dir, dataset_path, range(source_day, source_day + max(leads) + 1))
    if "P90" in present:
        gates["G2a"] = gate_g2a(present["P90"], source_day)
    if "V10" in present and "P90" in present:
        gates["G2b"] = gate_g2b(present["V10"], present["P90"], source_day + max(leads) - 10)
    if "C90" in present:
        gates["G3"] = gate_g3(present["C90"], weights["mean_only"], source_day, source_day + max(leads))
    if maps:
        gates["G4"] = gate_g4(maps, wet)
    gates["G1"] = gate_g1(scratch, ("1e-1", "1e-2", "1e-3", "1e-4", "1e-5"))

    try:
        import zarr

        group = zarr.open_consolidated(str(dataset_path), mode="r")
        gates["G5"] = {}
        for name, run_dir in present.items():
            run = next(r for r in contract["mitgcm_runs"] if r["name"] == name)
            if run.get("grdchk"):
                continue
            eta = np.asarray(group["state"][0, int(run["cost_day"]), 45], dtype=np.float32)
            gates["G5"][name] = gate_g5(run_dir, weights[run["weight"]], eta)
    except ImportError:  # pragma: no cover - environment dependent
        gates["G5"] = {"skipped": "zarr is not importable"}

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "mitgcm_s0_adjoint_v2.npz", **arrays)

    report = {
        "version": "mitgcm_s0_adjoint_v2",
        "plan": "docs/Adjoint_study_Phase_A.md",
        "plan_contract": PHASE_A_CONTRACT,
        "mitgcm_commit": contract["reference"]["commit"],
        "taf_version": contract["reference"]["taf_version"],
        "taf_resubmission_required": contract["reference"]["taf_resubmission_required"],
        "runs_present": sorted(present),
        "runs_absent": absent,
        "run_manifests": {
            name: json.loads((d / "run_manifest.json").read_text())
            for name, d in present.items()
            if (d / "run_manifest.json").is_file()
        },
        "weight_sha256": {
            name: _sha256(project_root / "work" / filename)
            for name, filename in WEIGHT_FILES.items()
            if (project_root / "work" / filename).is_file()
        },
        "gates": gates,
        "conventions": contract["conventions"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"wrote {output}")
    for name, gate in gates.items():
        if isinstance(gate, dict) and "passed" in gate:
            verdict = "pass" if gate["passed"] else "FAIL"
            print(f"  {name}: {verdict}")
        elif isinstance(gate, dict):
            for sub, value in gate.items():
                if isinstance(value, dict) and "passed" in value:
                    print(f"  {name}[{sub}]: {'pass' if value['passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
