"""Restart-safe S1/S2 wind-branch trajectories for the AF--FNO project."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .af_s0 import (
    DELTA_T_SECONDS,
    MITGCM_COMMIT,
    MODEL_YEAR_DAYS,
    MPI_RANKS,
    STEPS_PER_YEAR,
    _atomic_json,
    _git_revision,
    _link_parent_pickups,
    _sha256,
    render_data,
)


S0_BRANCH_YEAR = 100
ADJUST_YEARS = 5
PRODUCTION_YEARS = 10
EXPERIMENTS = {
    "S1": {"tau0_n_m2": 0.075, "wind_scale": 0.75},
    "S2": {"tau0_n_m2": 0.125, "wind_scale": 1.25},
}


def scale_wind(source: Path, destination: Path, scale: float) -> None:
    """Scale the official big-endian float32 wind field without changing its layout."""
    values = np.fromfile(source, dtype=">f4")
    if values.size != 62 * 62 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid tutorial wind field: {source}")
    (values.astype(np.float64) * scale).astype(">f4").tofile(destination)


def _s0_branch_result(scratch_root: Path) -> dict[str, Any]:
    path = (
        scratch_root
        / "mitgcm/S0/spinup/years_090_100/segment_result.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"validated S0 year-100 branch is missing: {path}")
    result = json.loads(path.read_text())
    if int(result["end_year"]) != S0_BRANCH_YEAR:
        raise RuntimeError("S0 branch result does not end at year 100")
    if int(result["end_iteration"]) != S0_BRANCH_YEAR * STEPS_PER_YEAR:
        raise RuntimeError("S0 branch result has the wrong model iteration")
    result["_result_path"] = str(path)
    return result


def _branch_parent(scratch_root: Path, experiment: str, local_start_year: int) -> dict[str, Any]:
    if local_start_year == 0:
        return _s0_branch_result(scratch_root)
    candidates: list[dict[str, Any]] = []
    for path in sorted((scratch_root / "mitgcm" / experiment).glob("*/years_*/segment_result.json")):
        result = json.loads(path.read_text())
        if int(result["end_year"]) == local_start_year:
            result["_result_path"] = str(path)
            candidates.append(result)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one {experiment} parent ending at local year {local_start_year}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _validate_request(experiment: str, phase: str, start_year: int, years: int) -> None:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {sorted(EXPERIMENTS)}")
    expected = {
        "adjust": (0, ADJUST_YEARS),
        "production": (ADJUST_YEARS, PRODUCTION_YEARS),
    }
    if phase not in expected:
        raise ValueError("phase must be 'adjust' or 'production'")
    if (start_year, years) != expected[phase]:
        raise ValueError(
            f"{experiment} {phase} must use local start/year count {expected[phase]}, "
            f"received {(start_year, years)}"
        )


def prepare_segment(
    project_root: Path,
    scratch_root: Path,
    executable: Path,
    experiment: str,
    phase: str,
    start_year: int,
    years: int,
) -> dict[str, Any]:
    """Prepare an immutable S1/S2 branch segment from the S0 year-100 checkpoint."""
    _validate_request(experiment, phase, start_year, years)
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"AF--FNO executable is missing: {executable}")
    run_dir = (
        scratch_root
        / "mitgcm"
        / experiment
        / phase
        / f"years_{start_year:03d}_{start_year + years:03d}"
    )
    spec = EXPERIMENTS[experiment]
    identity = {
        "experiment": experiment,
        "phase": phase,
        "start_year": start_year,
        "years": years,
        "tau0_n_m2": spec["tau0_n_m2"],
        "executable_sha256": _sha256(executable),
    }
    manifest_path = run_dir / "segment_manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if any(prior.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"{experiment} segment identity changed: {manifest_path}")
        return prior

    run_dir.mkdir(parents=True, exist_ok=False)
    absolute_start_year = S0_BRANCH_YEAR + start_year
    start_iteration = absolute_start_year * STEPS_PER_YEAR
    end_iteration = (absolute_start_year + years) * STEPS_PER_YEAR
    rendered = render_data(start_iteration, years * STEPS_PER_YEAR).replace(
        "AF--FNO S0", f"AF--FNO {experiment}", 1
    )
    (run_dir / "data").write_text(rendered)
    input_root = project_root / "af_fno/mitgcm/input"
    for name in ("data.pkg", "eedata"):
        shutil.copy2(input_root / name, run_dir / name)
    diagnostics_name = "adjust" if phase == "adjust" else "production"
    shutil.copy2(input_root / f"data.diagnostics.{diagnostics_name}", run_dir / "data.diagnostics")

    tutorial_input = project_root / "external/MITgcm/verification/tutorial_baroclinic_gyre/input"
    for name in ("bathy.bin", "SST_relax.bin"):
        shutil.copy2(tutorial_input / name, run_dir / name)
    scale_wind(tutorial_input / "windx_cosy.bin", run_dir / "windx_cosy.bin", spec["wind_scale"])
    (run_dir / "mitgcmuv").symlink_to(executable)

    parent = _branch_parent(scratch_root, experiment, start_year)
    if int(parent["end_iteration"]) != start_iteration:
        raise RuntimeError(f"{experiment} parent iteration does not match requested start")
    parent_pickups = _link_parent_pickups(parent, run_dir, start_iteration)

    revision = _git_revision(project_root / "external/MITgcm")
    if revision != MITGCM_COMMIT:
        raise RuntimeError(f"expected MITgcm {MITGCM_COMMIT}, found {revision}")
    plan = project_root / "docs/AF_FNO_Project_Plan.tex"
    manifest = {
        **identity,
        "local_end_year": start_year + years,
        "absolute_start_year": absolute_start_year,
        "absolute_end_year": absolute_start_year + years,
        "start_iteration": start_iteration,
        "end_iteration": end_iteration,
        "n_time_steps": years * STEPS_PER_YEAR,
        "delta_t_seconds": DELTA_T_SECONDS,
        "run_dir": str(run_dir),
        "parent_result": str(parent["_result_path"]),
        "parent_pickups": parent_pickups,
        "mitgcm_commit": revision,
        "project_plan": {"path": str(plan), "sha256": _sha256(plan)},
        "configuration_sha256": {
            name: _sha256(run_dir / name)
            for name in ("data", "data.pkg", "eedata", "data.diagnostics")
        },
        "forcing_sha256": {
            name: _sha256(run_dir / name)
            for name in ("bathy.bin", "windx_cosy.bin", "SST_relax.bin")
        },
        "created_by": {
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def run_segment(manifest: Mapping[str, Any], launcher: Sequence[str] | None = None) -> dict[str, Any]:
    """Run one prepared branch and verify its pickup and complete diagnostic inventory."""
    run_dir = Path(str(manifest["run_dir"]))
    result_path = run_dir / "segment_result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    partial = sorted(
        path
        for pattern in (
            "run.log",
            "STDOUT.*",
            "STDERR.*",
            "dynAdjust.*",
            "dynState.*",
            "surfState.*",
        )
        for path in run_dir.glob(pattern)
    )
    if partial:
        raise RuntimeError(f"refusing to overwrite incomplete branch output: {partial[:8]}")
    if launcher is None:
        launcher = ["srun", "--mpi=pmix", "-n", str(MPI_RANKS)]
    command = [*launcher, str(run_dir / "mitgcmuv")]
    started = time.monotonic()
    with (run_dir / "run.log").open("w") as stdout:
        completed = subprocess.run(
            command,
            cwd=run_dir,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    if completed.returncode:
        raise RuntimeError(f"MITgcm exited with {completed.returncode}; inspect {run_dir / 'run.log'}")

    end_iteration = int(manifest["end_iteration"])
    pickup_meta = run_dir / f"pickup.{end_iteration:010d}.meta"
    pickup_data = pickup_meta.with_suffix(".data")
    if not pickup_meta.is_file() or not pickup_data.is_file():
        raise RuntimeError(f"missing branch end pickup at iteration {end_iteration}")
    years = int(manifest["years"])
    if manifest["phase"] == "adjust":
        diagnostics = {"dynAdjust": len(list(run_dir.glob("dynAdjust.*.meta")))}
        expected = {"dynAdjust": years * 12}
    else:
        diagnostics = {
            "dynState": len(list(run_dir.glob("dynState.*.meta"))),
            "surfState": len(list(run_dir.glob("surfState.*.meta"))),
        }
        expected = {"dynState": years * MODEL_YEAR_DAYS, "surfState": years * MODEL_YEAR_DAYS}
    if diagnostics != expected:
        raise RuntimeError(f"branch diagnostic count mismatch: expected {expected}, found {diagnostics}")

    result = {
        "experiment": manifest["experiment"],
        "phase": manifest["phase"],
        "start_year": manifest["start_year"],
        "end_year": int(manifest["start_year"]) + years,
        "absolute_start_year": manifest["absolute_start_year"],
        "absolute_end_year": manifest["absolute_end_year"],
        "start_iteration": manifest["start_iteration"],
        "end_iteration": end_iteration,
        "tau0_n_m2": manifest["tau0_n_m2"],
        "run_dir": str(run_dir),
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "diagnostics": diagnostics,
        "pickup_meta": str(pickup_meta),
        "pickup_data": str(pickup_data),
        "pickup_sha256": {"meta": _sha256(pickup_meta), "data": _sha256(pickup_data)},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(result_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--phase", choices=("adjust", "production"), required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--years", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = prepare_segment(
        args.project_root.resolve(),
        args.scratch_root.resolve(),
        args.executable.resolve(),
        args.experiment,
        args.phase,
        args.start_year,
        args.years,
    )
    result = run_segment(manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
