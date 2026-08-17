"""Turbulent 0.25-degree MITgcm ground-truth trajectories for AF--FNO.

Three wind regimes, each independently equilibrated from the tutorial initial
condition under its own wind for 100 years and then run for 25 further years of
daily output::

    regime    tau0 (N m-2)   wind scale   spin-up     production
    S0_turb   0.100          1.00         0--100 y    100--125 y
    S1_turb   0.075          0.75         0--100 y    100--125 y
    S2_turb   0.125          1.25         0--100 y    100--125 y

S1 and S2 are *not* branched from the S0 equilibrium: changing tau0 changes the
statistically steady circulation, so each regime carries its own wind from year
zero.  Everything else -- bathymetry, SST restoring, grid, vertical
discretisation, viscosity, diffusivity, numerics, output cadence and
initialisation protocol -- is identical across the three.

Relative to the validated 1-degree S0/S1/S2 campaign the only changes are the
resolution and the dissipation that goes with it::

    1 degree      62 x 62     viscAh 5000   diffKhT 1000   deltaT 1200 s
    0.25 degree   248 x 248   viscAh  500   diffKhT  500   deltaT  300 s

with viscAr = 1e-2 and diffKrT = 1e-5 unchanged.  Those are the values locked in
``config/bire_a0_reference.toml`` for the turbulent configuration.

One numerical setting is added that the 1-degree deck does not have:
``useSRCGSolver=.TRUE.``.  The elliptic surface-pressure solve dominates the
0.25-degree cost -- 92 per cent of the wall clock in a 64-rank benchmark --
because it performs a global reduction per iteration and needs roughly four
times as many iterations as the coarse grid.  The single-reduction conjugate
gradient solver halves the reductions and reproduced the stock solver's monitor
statistics to all printed digits over the benchmark, so it changes the solver's
communication pattern rather than the solution.

This module is deliberately standalone: the historical ``bire_repro`` package it
descends from is no longer installed, so it depends on nothing but the standard
library.  Segments are ten model-years each, restart-safe, and chained through
MITgcm pickups: a completed segment writes ``segment_result.json`` and re-running
it is a no-op, while a partially written directory is refused rather than
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

VERSION = "af_turb_trajectories_v1"
ROOT_NAME = "mitgcm_turb_v1"

MODEL_YEAR_DAYS = 360
DELTA_T_SECONDS = 300
STEPS_PER_DAY = 86400 // DELTA_T_SECONDS
STEPS_PER_YEAR = MODEL_YEAR_DAYS * STEPS_PER_DAY
SPINUP_YEARS = 100
PRODUCTION_YEARS = 25
SEGMENT_YEARS = 10

GRID_CELLS = 248
GRID_SPACING_DEG = 0.25
MPI_RANKS = 64
MITGCM_COMMIT = "f03a2f5e214bc57b8393f6201a6a1266dd1f53d6"

REGIMES = {
    "S0_turb": {"tau0_n_m2": 0.100, "wind_scale": 1.00},
    "S1_turb": {"tau0_n_m2": 0.075, "wind_scale": 0.75},
    "S2_turb": {"tau0_n_m2": 0.125, "wind_scale": 1.25},
}


class TurbulentTrajectoryError(RuntimeError):
    """Raised when a turbulent trajectory violates its contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_revision(source: Path) -> str:
    """Read a checkout revision even when compute-node modules omit ``git``."""

    if shutil.which("git"):
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    head = (source / ".git" / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    reference = head.removeprefix("ref: ")
    loose = source / ".git" / reference
    if loose.is_file():
        return loose.read_text().strip()
    packed = source / ".git" / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line and not line.startswith(("#", "^")):
                revision, name = line.split(" ", 1)
                if name == reference:
                    return revision
    raise TurbulentTrajectoryError(f"cannot resolve {reference} under {source / '.git'}")


def render_data(n_iter0: int, n_time_steps: int) -> str:
    """Render the turbulent 0.25-degree physics for one finite segment."""

    if n_iter0 < 0 or n_time_steps <= 0:
        raise ValueError("n_iter0 must be nonnegative and n_time_steps positive")
    return f"""# AF--FNO turbulent 0.25-degree baroclinic gyre
 &PARM01
 viscAh=500.,
 viscAr=1.E-2,
 no_slip_sides=.TRUE.,
 no_slip_bottom=.FALSE.,
 diffKhT=500.,
 diffKrT=1.E-5,
 ivdc_kappa=1.,
 implicitDiffusion=.TRUE.,
 eosType='LINEAR',
 tRef=30.,27.,24.,21.,18.,15.,13.,11.,9.,7.,6.,5.,4.,3.,2.,
 tAlpha=2.E-4,
 sBeta=0.,
 rhoNil=999.8,
 gravity=9.81,
 rigidLid=.FALSE.,
 implicitFreeSurface=.TRUE.,
 exactConserv=.TRUE.,
 saltStepping=.FALSE.,
 useSingleCpuIO=.TRUE.,
 &
 &PARM02
 cg2dTargetResidual=1.E-7,
 cg2dMaxIters=1000,
 useSRCGSolver=.TRUE.,
 &
 &PARM03
 nIter0={n_iter0},
 nTimeSteps={n_time_steps},
 deltaT=300.,
 pChkptFreq=31104000.,
 chkptFreq=0.,
 dumpFreq=0.,
 monitorFreq=2592000.,
 monitorSelect=2,
 tauThetaClimRelax=2592000.,
 &
 &PARM04
 usingSphericalPolarGrid=.TRUE.,
 delX={GRID_CELLS}*{GRID_SPACING_DEG},
 delY={GRID_CELLS}*{GRID_SPACING_DEG},
 xgOrigin=-1.,
 ygOrigin=14.,
 delR=50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,
 &
 &PARM05
 bathyFile='bathy.bin',
 zonalWindFile='windx_cosy.bin',
 thetaClimFile='SST_relax.bin',
 &
"""


def segment_plan(phase: str) -> tuple[tuple[int, int], ...]:
    """Ten-year segments covering one phase, with a short final production leg."""

    if phase == "spinup":
        return tuple((start, SEGMENT_YEARS) for start in range(0, SPINUP_YEARS, SEGMENT_YEARS))
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
        raise TurbulentTrajectoryError(
            f"expected exactly one parent ending at year {start_year}, found {len(matches)}"
        )
    return matches[0]


def _link_parent_pickups(parent: Mapping[str, Any], run_dir: Path, iteration: int) -> list[str]:
    parent_dir = Path(str(parent["run_dir"])).resolve()
    linked: list[str] = []
    for metadata in sorted(parent_dir.glob(f"pickup*.{iteration:010d}.meta")):
        data = metadata.with_suffix(".data")
        if not data.is_file():
            raise TurbulentTrajectoryError(f"incomplete parent pickup: {data}")
        for source in (metadata, data):
            destination = run_dir / source.name
            destination.symlink_to(source.resolve())
            linked.append(str(destination))
    if not linked:
        raise TurbulentTrajectoryError(f"no parent pickup at iteration {iteration} in {parent_dir}")
    return linked


def _scale_wind(source: Path, destination: Path, scale: float) -> None:
    """Scale the big-endian float32 wind field without changing its layout.

    Done on the raw bytes so the module keeps its standard-library-only footing:
    each record is a four-byte big-endian float.
    """

    import struct

    payload = source.read_bytes()
    expected = GRID_CELLS * GRID_CELLS * 4
    if len(payload) != expected:
        raise TurbulentTrajectoryError(
            f"invalid turbulent wind field {source}: {len(payload)} bytes, expected {expected}"
        )
    values = struct.unpack(f">{GRID_CELLS * GRID_CELLS}f", payload)
    destination.write_bytes(
        struct.pack(f">{GRID_CELLS * GRID_CELLS}f", *(value * scale for value in values))
    )


def prepare_segment(
    project_root: Path,
    scratch_root: Path,
    executable: Path,
    regime: str,
    phase: str,
    start_year: int,
    years: int,
) -> dict[str, Any]:
    """Create one immutable turbulent segment and its provenance manifest."""

    _validate_request(regime, phase, start_year, years)
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"turbulent MITgcm executable is missing: {executable}")
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
            raise TurbulentTrajectoryError(f"turbulent segment identity changed: {manifest_path}")
        return prior

    run_dir.mkdir(parents=True, exist_ok=False)
    start_iteration = start_year * STEPS_PER_YEAR
    end_iteration = (start_year + years) * STEPS_PER_YEAR
    (run_dir / "data").write_text(render_data(start_iteration, years * STEPS_PER_YEAR))
    input_root = project_root / "af_fno" / "mitgcm" / "input_turb"
    for name in ("data.pkg", "eedata"):
        shutil.copy2(input_root / name, run_dir / name)
    shutil.copy2(input_root / f"data.diagnostics.{phase}", run_dir / "data.diagnostics")
    for name in ("bathy.bin", "SST_relax.bin"):
        shutil.copy2(input_root / name, run_dir / name)
    # The only physical difference between regimes: the zonal wind amplitude,
    # applied from year zero rather than after a branch.
    _scale_wind(
        input_root / "windx_cosy.bin",
        run_dir / "windx_cosy.bin",
        float(specification["wind_scale"]),
    )
    (run_dir / "mitgcmuv").symlink_to(executable)

    parent_pickups: list[str] = []
    parent_result: str | None = None
    if start_year:
        parent = _find_parent(regime_root, start_year)
        if int(parent["end_iteration"]) != start_iteration:
            raise TurbulentTrajectoryError(
                "parent iteration does not match the requested start year"
            )
        parent_pickups = _link_parent_pickups(parent, run_dir, start_iteration)
        parent_result = str(parent["_result_path"])

    revision = _git_revision(project_root / "external" / "MITgcm")
    if revision != MITGCM_COMMIT:
        raise TurbulentTrajectoryError(f"expected MITgcm {MITGCM_COMMIT}, found {revision}")
    manifest = {
        **expected_identity,
        "tau0_n_m2": specification["tau0_n_m2"],
        "equilibration": "independent_from_the_tutorial_initial_condition_under_this_regimes_own_wind",
        "resolution": {
            "nx": GRID_CELLS,
            "ny": GRID_CELLS,
            "nr": 15,
            "spacing_deg": GRID_SPACING_DEG,
        },
        "dissipation": {
            "viscAh_m2_s": 500.0,
            "diffKhT_m2_s": 500.0,
            "viscAr_m2_s": 1e-2,
            "diffKrT_m2_s": 1e-5,
        },
        "start_iteration": start_iteration,
        "end_iteration": end_iteration,
        "n_time_steps": years * STEPS_PER_YEAR,
        "delta_t_seconds": DELTA_T_SECONDS,
        "mpi_ranks": MPI_RANKS,
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
        raise TurbulentTrajectoryError(f"refusing to overwrite incomplete output: {partial[:8]}")
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
        raise TurbulentTrajectoryError(
            f"MITgcm exited with {completed.returncode}; inspect {run_dir / 'run.log'}"
        )

    end_iteration = int(manifest["end_iteration"])
    pickup_meta = run_dir / f"pickup.{end_iteration:010d}.meta"
    pickup_data = pickup_meta.with_suffix(".data")
    if not pickup_meta.is_file() or not pickup_data.is_file():
        raise TurbulentTrajectoryError(f"missing end pickup at iteration {end_iteration}")
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
        raise TurbulentTrajectoryError(
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
        "pickup_sha256": {"meta": _sha256(pickup_meta), "data": _sha256(pickup_data)},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(result_path, result)
    return result


def write_build_manifest(project_root: Path, executable: Path) -> dict[str, Any]:
    """Record the turbulent executable and the code directory that produced it."""

    code = project_root / "af_fno" / "mitgcm" / "code_turb"
    executable = executable.resolve()
    result = {
        "version": VERSION,
        "experiment": "AF--FNO turbulent 0.25-degree S0/S1/S2",
        "mitgcm_commit": _git_revision(project_root / "external" / "MITgcm"),
        "executable": str(executable),
        "executable_sha256": _sha256(executable),
        "mpi_ranks": MPI_RANKS,
        "grid": {"nx": GRID_CELLS, "ny": GRID_CELLS, "nr": 15, "spacing_deg": GRID_SPACING_DEG},
        "delta_t_seconds": DELTA_T_SECONDS,
        "configuration_sha256": {
            path.name: _sha256(path) for path in sorted(code.iterdir()) if path.is_file()
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(executable.parent / "af_turb_build_manifest.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    segment = commands.add_parser("run-segment")
    segment.add_argument("--project-root", type=Path, required=True)
    segment.add_argument("--scratch-root", type=Path, required=True)
    segment.add_argument("--executable", type=Path, required=True)
    segment.add_argument("--regime", choices=tuple(REGIMES), required=True)
    segment.add_argument("--phase", choices=("spinup", "production"), required=True)
    segment.add_argument("--start-year", type=int, required=True)
    segment.add_argument("--years", type=int, required=True)

    build = commands.add_parser("build-manifest")
    build.add_argument("--project-root", type=Path, required=True)
    build.add_argument("--executable", type=Path, required=True)

    plan = commands.add_parser("plan")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run-segment":
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
    elif args.command == "build-manifest":
        result = write_build_manifest(args.project_root.resolve(), args.executable.resolve())
    else:
        result = {
            "version": VERSION,
            "regimes": REGIMES,
            "spinup": segment_plan("spinup"),
            "production": segment_plan("production"),
            "steps_per_year": STEPS_PER_YEAR,
            "production_days_per_regime": PRODUCTION_YEARS * MODEL_YEAR_DAYS,
        }
    print(json.dumps(result, indent=2, sort_keys=True, default=list))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
