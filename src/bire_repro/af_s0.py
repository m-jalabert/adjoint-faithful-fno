"""Restart-safe MITgcm S0 workflow for the AF--FNO project plan."""

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


MODEL_YEAR_DAYS = 360
DELTA_T_SECONDS = 1200
STEPS_PER_DAY = 86400 // DELTA_T_SECONDS
STEPS_PER_YEAR = MODEL_YEAR_DAYS * STEPS_PER_DAY
S0_SPINUP_YEARS = 100
S0_PRODUCTION_YEARS = 10
MPI_RANKS = 4
MITGCM_COMMIT = "f03a2f5e214bc57b8393f6201a6a1266dd1f53d6"


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
    raise RuntimeError(f"Cannot resolve {reference} under {source / '.git'}")


def render_data(n_iter0: int, n_time_steps: int) -> str:
    """Render the official tutorial physics for one finite S0 segment."""
    if n_iter0 < 0 or n_time_steps <= 0:
        raise ValueError("n_iter0 must be nonnegative and n_time_steps positive")
    return f"""# AF--FNO S0: official 1-degree baroclinic-gyre tutorial
 &PARM01
 viscAh=5000.,
 viscAr=1.E-2,
 no_slip_sides=.TRUE.,
 no_slip_bottom=.FALSE.,
 diffKhT=1000.,
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
 &
 &PARM03
 nIter0={n_iter0},
 nTimeSteps={n_time_steps},
 deltaT=1200.,
 pChkptFreq=31104000.,
 chkptFreq=0.,
 dumpFreq=0.,
 monitorFreq=2592000.,
 monitorSelect=2,
 tauThetaClimRelax=2592000.,
 &
 &PARM04
 usingSphericalPolarGrid=.TRUE.,
 delX=62*1.,
 delY=62*1.,
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


def _find_parent(root: Path, start_year: int) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/years_*/segment_result.json")):
        value = json.loads(path.read_text())
        if int(value["end_year"]) == start_year:
            value["_result_path"] = str(path)
            candidates.append(value)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one completed S0 parent ending at year {start_year}, found {len(candidates)}"
        )
    return candidates[0]


def _link_parent_pickups(parent: Mapping[str, Any], run_dir: Path, iteration: int) -> list[str]:
    parent_dir = Path(str(parent["run_dir"])).resolve()
    linked: list[str] = []
    for metadata in sorted(parent_dir.glob(f"pickup*.{iteration:010d}.meta")):
        data = metadata.with_suffix(".data")
        if not data.is_file():
            raise RuntimeError(f"Incomplete parent pickup: {data}")
        for source in (metadata, data):
            destination = run_dir / source.name
            destination.symlink_to(source.resolve())
            linked.append(str(destination))
    if not linked:
        raise RuntimeError(f"No parent pickup at iteration {iteration} in {parent_dir}")
    return linked


def prepare_segment(
    project_root: Path,
    scratch_root: Path,
    executable: Path,
    phase: str,
    start_year: int,
    years: int,
) -> dict[str, Any]:
    """Create one immutable S0 segment and its provenance manifest."""
    if phase not in {"spinup", "production"}:
        raise ValueError("phase must be 'spinup' or 'production'")
    if start_year < 0 or years <= 0:
        raise ValueError("start_year must be nonnegative and years positive")
    if phase == "spinup" and start_year + years > S0_SPINUP_YEARS:
        raise ValueError("spin-up segments cannot pass year 100")
    if phase == "production" and (
        start_year < S0_SPINUP_YEARS
        or start_year + years > S0_SPINUP_YEARS + S0_PRODUCTION_YEARS
    ):
        raise ValueError("production must cover years 100 through 110")

    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"AF--FNO executable is missing: {executable}")
    run_dir = scratch_root / "mitgcm" / "S0" / phase / f"years_{start_year:03d}_{start_year + years:03d}"
    manifest_path = run_dir / "segment_manifest.json"
    expected_identity = {
        "phase": phase,
        "start_year": start_year,
        "years": years,
        "executable_sha256": _sha256(executable),
    }
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if any(prior.get(key) != value for key, value in expected_identity.items()):
            raise RuntimeError(f"S0 segment identity changed: {manifest_path}")
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
    for name in ("bathy.bin", "windx_cosy.bin", "SST_relax.bin"):
        shutil.copy2(tutorial_input / name, run_dir / name)
    (run_dir / "mitgcmuv").symlink_to(executable)

    parent_pickups: list[str] = []
    parent_result: str | None = None
    if start_year:
        parent = _find_parent(scratch_root / "mitgcm" / "S0", start_year)
        if int(parent["end_iteration"]) != start_iteration:
            raise RuntimeError("Parent S0 iteration does not match requested start year")
        parent_pickups = _link_parent_pickups(parent, run_dir, start_iteration)
        parent_result = str(parent["_result_path"])

    source = project_root / "external" / "MITgcm"
    revision = _git_revision(source)
    if revision != MITGCM_COMMIT:
        raise RuntimeError(f"Expected MITgcm {MITGCM_COMMIT}, found {revision}")
    plan = project_root / "docs" / "AF_FNO_Project_Plan.tex"
    manifest = {
        **expected_identity,
        "experiment": "S0",
        "tau0_n_m2": 0.1,
        "start_iteration": start_iteration,
        "end_iteration": end_iteration,
        "n_time_steps": years * STEPS_PER_YEAR,
        "delta_t_seconds": DELTA_T_SECONDS,
        "run_dir": str(run_dir),
        "parent_result": parent_result,
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
        "created_by": {"host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def run_segment(manifest: Mapping[str, Any], launcher: Sequence[str] | None = None) -> dict[str, Any]:
    """Run and validate one prepared S0 segment."""
    run_dir = Path(str(manifest["run_dir"]))
    result_path = run_dir / "segment_result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    partial = sorted(
        path
        for pattern in ("run.log", "STDOUT.*", "STDERR.*", "dynSpin.*", "dynState.*", "surfState.*")
        for path in run_dir.glob(pattern)
    )
    if partial:
        raise RuntimeError(f"Refusing to overwrite incomplete S0 output: {partial[:8]}")
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
        raise RuntimeError(f"Missing S0 end pickup at iteration {end_iteration}")
    years = int(manifest["years"])
    if manifest["phase"] == "spinup":
        diagnostics = {"dynSpin": len(list(run_dir.glob("dynSpin.*.meta")))}
        expected = {"dynSpin": years * 12}
    else:
        diagnostics = {
            "dynState": len(list(run_dir.glob("dynState.*.meta"))),
            "surfState": len(list(run_dir.glob("surfState.*.meta"))),
        }
        expected = {"dynState": years * MODEL_YEAR_DAYS, "surfState": years * MODEL_YEAR_DAYS}
    if diagnostics != expected:
        raise RuntimeError(f"S0 diagnostic count mismatch: expected {expected}, found {diagnostics}")

    result = {
        "experiment": "S0",
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
    source = project_root / "external" / "MITgcm"
    revision = _git_revision(source)
    code = project_root / "af_fno" / "mitgcm" / "code"
    result = {
        "experiment": "AF--FNO S0",
        "mitgcm_commit": revision,
        "executable": str(executable.resolve()),
        "executable_sha256": _sha256(executable.resolve()),
        "mpi_ranks": MPI_RANKS,
        "configuration_sha256": {
            path.name: _sha256(path) for path in sorted(code.iterdir()) if path.is_file()
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(executable.resolve().parent / "af_s0_build_manifest.json", result)
    return result


def simulation_inventory() -> dict[str, Any]:
    """Return the active programme after the evidence-authorized v2 expansion."""
    original_trajectories = 5
    trajectory_extensions = 3
    trajectories = original_trajectories + trajectory_extensions
    response_forwards = 224
    adjoints = 12
    control_updates = 3
    original_trajectory_years = 110 + 15 + 15 + 1 + 1
    trajectory_years = original_trajectory_years + 3 * 10
    response_days = 224 * 10
    adjoint_days = 3 * 2 * (10 + 30)
    control_update_days = 3 * 30
    return {
        "scientific_integrations": {
            "trajectory_forward": trajectories,
            "original_trajectory_forward": original_trajectories,
            "trajectory_v2_extensions": trajectory_extensions,
            "perturbed_response_forward": response_forwards,
            "blind_true_adjoint": adjoints,
            "forward_control_update_checks": control_updates,
            "minimum_total": trajectories + response_forwards + adjoints + control_updates,
        },
        "adjoint_ground_truth_integrations": adjoints,
        "optional_90_day_adjoint_extension": 6,
        "model_time": {
            "trajectory_years": trajectory_years,
            "response_days": response_days,
            "adjoint_days": adjoint_days,
            "control_update_days_upper_bound": control_update_days,
            "total_model_years_upper_bound": trajectory_years
            + (response_days + adjoint_days + control_update_days) / MODEL_YEAR_DAYS,
        },
        "note": (
            "The original listed durations sum to 142 model-years; the three "
            "evidence-authorized ten-year extensions raise the active total to 172."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    segment = commands.add_parser("run-segment")
    segment.add_argument("--project-root", type=Path, required=True)
    segment.add_argument("--scratch-root", type=Path, required=True)
    segment.add_argument("--executable", type=Path, required=True)
    segment.add_argument("--phase", choices=("spinup", "production"), required=True)
    segment.add_argument("--start-year", type=int, required=True)
    segment.add_argument("--years", type=int, required=True)
    build = commands.add_parser("build-manifest")
    build.add_argument("--project-root", type=Path, required=True)
    build.add_argument("--executable", type=Path, required=True)
    commands.add_parser("inventory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run-segment":
        manifest = prepare_segment(
            args.project_root.resolve(),
            args.scratch_root.resolve(),
            args.executable.resolve(),
            args.phase,
            args.start_year,
            args.years,
        )
        result = run_segment(manifest)
    elif args.command == "build-manifest":
        result = write_build_manifest(args.project_root.resolve(), args.executable.resolve())
    else:
        result = simulation_inventory()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
