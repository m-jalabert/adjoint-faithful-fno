"""Independently equilibrated wind-regime trajectories for AF--FNO.

The stored S1 and S2 records are *branches*: both restart from the validated S0
year-100 state and receive only a five-year adjustment before ten years of daily
production.  Their slow fields therefore still carry the S0 equilibrium, and the
three regimes are not independent samples of their own forcing.

This module equilibrates each regime from the original MITgcm tutorial initial
condition under its own wind for the full 100 years, then produces 25 years of
daily output::

    regime   tau0 (N m-2)   wind scale   spin-up      production
    S0       0.100          1.00         0--100 y     100--125 y
    S1       0.075          0.75         0--100 y     100--125 y
    S2       0.125          1.25         0--100 y     100--125 y

Every regime uses the same bathymetry, the same SST relaxation field, the same
executable, the same 1,200-second time step, and the same 360-day year; only the
zonal wind amplitude differs, applied from year zero.

S0 is a special case and is deliberately *not* re-run.  The existing S0 campaign
is already an independent 100-year equilibration at tau0 = 0.100 from the
tutorial initial condition, and its production record now spans years 100--126
across three chained campaigns --- 26 years, more than the 25 required.  Re-running
it would consume two CPU-hours to reproduce bytes that already exist and are
hash-locked by several frozen contracts.  :func:`s0_production_inventory` checks
that the existing chain is contiguous and long enough instead.

Segments are ten model-years each, restart-safe, and chained through MITgcm
pickups exactly as :mod:`af_s0` does: a completed segment writes
``segment_result.json`` and re-running it is a no-op, while a partially written
directory is refused rather than overwritten.  Measured cost is about 700 seconds
per ten model-years on four MPI ranks, so one regime is roughly 2.6 hours of
wall clock across 13 segments.
"""

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
from .af_wind_trajectories import scale_wind

VERSION = "af_independent_wind_trajectories_v1"

SPINUP_YEARS = 100
PRODUCTION_YEARS = 25
SEGMENT_YEARS = 10
ROOT_NAME = "mitgcm_independent_v1"

REGIMES = {
    "S0": {"tau0_n_m2": 0.100, "wind_scale": 1.00},
    "S1": {"tau0_n_m2": 0.075, "wind_scale": 0.75},
    "S2": {"tau0_n_m2": 0.125, "wind_scale": 1.25},
}

#: S0 already has an independent 100-year equilibration and 26 production years,
#: so it is served from the existing chain rather than recomputed.
S0_EXISTING_CHAIN = (
    ("mitgcm/S0/production/years_100_110", 100, 110),
    ("mitgcm_v2/S0/production/years_110_120", 110, 120),
    ("mitgcm_long_truth_v1/S0/production/years_120_126", 120, 126),
)


class IndependentTrajectoryError(RuntimeError):
    """Raised when an independent wind trajectory violates its contract."""


def segment_plan(phase: str) -> tuple[tuple[int, int], ...]:
    """Ten-year segments covering one phase, with a short final production leg."""

    if phase == "spinup":
        return tuple(
            (start, SEGMENT_YEARS) for start in range(0, SPINUP_YEARS, SEGMENT_YEARS)
        )
    if phase == "production":
        segments = []
        start = SPINUP_YEARS
        remaining = PRODUCTION_YEARS
        while remaining > 0:
            years = min(SEGMENT_YEARS, remaining)
            segments.append((start, years))
            start += years
            remaining -= years
        return tuple(segments)
    raise ValueError("phase must be 'spinup' or 'production'")


def _validate_request(regime: str, phase: str, start_year: int, years: int) -> None:
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {sorted(REGIMES)}")
    if (start_year, years) not in segment_plan(phase):
        raise ValueError(
            f"{phase} segment ({start_year}, {years}) is not in the declared plan "
            f"{segment_plan(phase)}"
        )


def _find_parent(regime_root: Path, start_year: int) -> dict[str, Any]:
    """The completed segment whose end year is this segment's start year."""

    matches = []
    for result_path in regime_root.rglob("segment_result.json"):
        result = json.loads(result_path.read_text())
        if int(result["end_year"]) == start_year:
            result["_result_path"] = str(result_path)
            matches.append(result)
    if len(matches) != 1:
        raise IndependentTrajectoryError(
            f"expected exactly one parent ending at year {start_year}, found {len(matches)}"
        )
    return matches[0]


def prepare_segment(
    project_root: Path,
    scratch_root: Path,
    executable: Path,
    regime: str,
    phase: str,
    start_year: int,
    years: int,
) -> dict[str, Any]:
    """Create one immutable independently equilibrated segment and its manifest."""

    _validate_request(regime, phase, start_year, years)
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"AF--FNO executable is missing: {executable}")
    regime_root = scratch_root / ROOT_NAME / regime
    run_dir = regime_root / phase / f"years_{start_year:03d}_{start_year + years:03d}"
    manifest_path = run_dir / "segment_manifest.json"
    specification = REGIMES[regime]
    expected_identity = {
        "version": VERSION,
        "regime": regime,
        "phase": phase,
        "start_year": start_year,
        "years": years,
        "wind_scale": specification["wind_scale"],
        "executable_sha256": _sha256(executable),
    }
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if any(prior.get(key) != value for key, value in expected_identity.items()):
            raise IndependentTrajectoryError(
                f"independent segment identity changed: {manifest_path}"
            )
        return prior

    run_dir.mkdir(parents=True, exist_ok=False)
    start_iteration = start_year * STEPS_PER_YEAR
    end_iteration = (start_year + years) * STEPS_PER_YEAR
    (run_dir / "data").write_text(render_data(start_iteration, years * STEPS_PER_YEAR))
    input_root = project_root / "af_fno" / "mitgcm" / "input"
    for name in ("data.pkg", "eedata"):
        shutil.copy2(input_root / name, run_dir / name)
    shutil.copy2(input_root / f"data.diagnostics.{phase}", run_dir / "data.diagnostics")

    tutorial_input = (
        project_root
        / "external"
        / "MITgcm"
        / "verification"
        / "tutorial_baroclinic_gyre"
        / "input"
    )
    for name in ("bathy.bin", "SST_relax.bin"):
        shutil.copy2(tutorial_input / name, run_dir / name)
    # The only physical difference between regimes: the zonal wind amplitude,
    # applied from year zero rather than after a branch.
    scale_wind(
        tutorial_input / "windx_cosy.bin",
        run_dir / "windx_cosy.bin",
        float(specification["wind_scale"]),
    )
    (run_dir / "mitgcmuv").symlink_to(executable)

    parent_pickups: list[str] = []
    parent_result: str | None = None
    if start_year:
        parent = _find_parent(regime_root, start_year)
        if int(parent["end_iteration"]) != start_iteration:
            raise IndependentTrajectoryError(
                "parent iteration does not match the requested start year"
            )
        parent_pickups = _link_parent_pickups(parent, run_dir, start_iteration)
        parent_result = str(parent["_result_path"])

    revision = _git_revision(project_root / "external" / "MITgcm")
    if revision != MITGCM_COMMIT:
        raise IndependentTrajectoryError(
            f"expected MITgcm {MITGCM_COMMIT}, found {revision}"
        )
    manifest = {
        **expected_identity,
        "tau0_n_m2": specification["tau0_n_m2"],
        "equilibration": "independent_from_the_tutorial_initial_condition_under_this_regimes_own_wind",
        "start_iteration": start_iteration,
        "end_iteration": end_iteration,
        "n_time_steps": years * STEPS_PER_YEAR,
        "delta_t_seconds": DELTA_T_SECONDS,
        "run_dir": str(run_dir),
        "parent_result": parent_result,
        "parent_pickups": parent_pickups,
        "mitgcm_commit": revision,
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


def run_segment(
    manifest: Mapping[str, Any],
    launcher: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run one prepared segment and verify its pickup and diagnostic inventory."""

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
            "dynSpin.*",
            "dynState.*",
            "surfState.*",
        )
        for path in run_dir.glob(pattern)
    )
    if partial:
        raise IndependentTrajectoryError(
            f"refusing to overwrite incomplete output: {partial[:8]}"
        )
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
        raise IndependentTrajectoryError(
            f"MITgcm exited with {completed.returncode}; inspect {run_dir / 'run.log'}"
        )

    end_iteration = int(manifest["end_iteration"])
    pickup_meta = run_dir / f"pickup.{end_iteration:010d}.meta"
    pickup_data = pickup_meta.with_suffix(".data")
    if not pickup_meta.is_file() or not pickup_data.is_file():
        raise IndependentTrajectoryError(
            f"missing end pickup at iteration {end_iteration}"
        )
    years = int(manifest["years"])
    if manifest["phase"] == "spinup":
        diagnostics = {"dynSpin": len(list(run_dir.glob("dynSpin.*.meta")))}
        expected = {"dynSpin": years * 12}
    else:
        diagnostics = {
            "dynState": len(list(run_dir.glob("dynState.*.meta"))),
            "surfState": len(list(run_dir.glob("surfState.*.meta"))),
        }
        expected = {
            "dynState": years * MODEL_YEAR_DAYS,
            "surfState": years * MODEL_YEAR_DAYS,
        }
    if diagnostics != expected:
        raise IndependentTrajectoryError(
            f"diagnostic count mismatch: expected {expected}, found {diagnostics}"
        )

    result = {
        "version": VERSION,
        "regime": manifest["regime"],
        "tau0_n_m2": manifest["tau0_n_m2"],
        "phase": manifest["phase"],
        "start_year": manifest["start_year"],
        "end_year": int(manifest["start_year"]) + years,
        "start_iteration": manifest["start_iteration"],
        "end_iteration": end_iteration,
        "run_dir": str(run_dir),
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "diagnostics": diagnostics,
        "pickup_meta": str(pickup_meta),
        "pickup_data": str(pickup_data),
        "pickup_sha256": {
            "meta": _sha256(pickup_meta),
            "data": _sha256(pickup_data),
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(result_path, result)
    return result


def s0_production_inventory(scratch_root: Path) -> dict[str, Any]:
    """Verify the existing S0 chain covers 25 contiguous production years.

    S0 is not re-run: its campaign is already an independent 100-year
    equilibration at tau0 = 0.100 from the tutorial initial condition.
    """

    segments = []
    cursor = SPINUP_YEARS
    for relative, start_year, end_year in S0_EXISTING_CHAIN:
        directory = scratch_root / relative
        if start_year != cursor:
            raise IndependentTrajectoryError(
                f"S0 production chain is not contiguous at year {start_year}"
            )
        result_path = directory / "segment_result.json"
        if not result_path.is_file():
            raise IndependentTrajectoryError(f"missing S0 segment result: {result_path}")
        days = (end_year - start_year) * MODEL_YEAR_DAYS
        segments.append(
            {
                "run_dir": str(directory),
                "start_year": start_year,
                "end_year": end_year,
                "days": days,
            }
        )
        cursor = end_year
    total = (cursor - SPINUP_YEARS) * MODEL_YEAR_DAYS
    if total < PRODUCTION_YEARS * MODEL_YEAR_DAYS:
        raise IndependentTrajectoryError(
            f"S0 has {total} production days, fewer than the required "
            f"{PRODUCTION_YEARS * MODEL_YEAR_DAYS}"
        )
    return {
        "regime": "S0",
        "tau0_n_m2": REGIMES["S0"]["tau0_n_m2"],
        "recomputed": False,
        "reason": (
            "the existing S0 campaign is already an independent 100-year "
            "equilibration from the tutorial initial condition and its production "
            "chain exceeds 25 years"
        ),
        "segments": segments,
        "available_production_days": total,
        "required_production_days": PRODUCTION_YEARS * MODEL_YEAR_DAYS,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--regime", choices=tuple(REGIMES), required=True)
    parser.add_argument("--phase", choices=("spinup", "production"), required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--years", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = prepare_segment(
        args.project_root.resolve(),
        args.scratch_root.resolve(),
        args.executable,
        args.regime,
        args.phase,
        args.start_year,
        args.years,
    )
    result = run_segment(manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
