"""Restart-safe MITgcm twin of the S0 control trajectory.

The twin is a pure MITgcm experiment.  It consumes the immutable year-100 S0
pickup, multiplies only the two velocity records by ``1 + epsilon``, and
integrates the identical physics forward for 25 model years.  Nothing about the
FNO -- code, checkpoints, normalisation, datasets -- is read or written here.

    x'_100 = x_100 + delta x,   delta x = epsilon * (Uvel, Vvel),   epsilon = 1e-6

Segments are immutable: a completed segment is never re-run, and a segment
directory that already exists without a manifest is reported rather than
silently reused.  The low-level machinery (physics rendering, provenance,
parent discovery, pickup linking) is imported unchanged from :mod:`af_s0`; only
the segment plan and the perturbation are new.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
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
    _find_parent,
    _git_revision,
    _link_parent_pickups,
    _sha256,
    render_data,
)
from .mds import parse_mds_meta


EXPERIMENT = "S0_twin"
CONTROL_REGIME = "S0"
EPSILON = 1.0e-6
PERTURBED_FIELDS = ("Uvel", "Vvel")
PERTURBATION = "multiplicative"
PERTURBATION_FORMULA = "field_twin = (1 + epsilon) * field_control"

TWIN_ROOT_NAME = "mitgcm_twin_v1"
TWIN_LABEL = "S0_eps1e-6"
TWIN_PHASE = "production"
TWIN_START_YEAR = 100
TWIN_END_YEAR = 125
#: The only segments this experiment is allowed to create, in restart order.
TWIN_SEGMENTS = (
    (100, 10),
    (110, 10),
    (120, 5),
)
TWIN_START_ITERATION = TWIN_START_YEAR * STEPS_PER_YEAR  # 2_592_000


@dataclass(frozen=True)
class TwinSpec:
    """Identity of one twin experiment: all that differs between amplitudes.

    The physics, the segment plan, the pickup record layout and every
    verification step are shared, so a twin is described completely by its
    name, its perturbation amplitude, and where its segments live on scratch.
    Running both amplitudes through one code path is the point: the epsilon
    comparison is only clean if nothing else about the two runs can differ.
    """

    experiment: str
    epsilon: float
    root_name: str
    label: str

    @property
    def scale(self) -> float:
        """The multiplicative factor applied to the perturbed velocity records."""

        return 1.0 + self.epsilon


#: The original twin.  Every function below defaults to it, so existing callers
#: and the manifests already written to scratch are unaffected.
DEFAULT_SPEC = TwinSpec(
    experiment=EXPERIMENT,
    epsilon=EPSILON,
    root_name=TWIN_ROOT_NAME,
    label=TWIN_LABEL,
)

#: Record layout of the S0 pickup, asserted against the control ``.meta``.
PICKUP_FIELD_LAYOUT = (
    ("Uvel", 15),
    ("Vvel", 15),
    ("Theta", 15),
    ("Salt", 15),
    ("GuNm1", 15),
    ("GvNm1", 15),
    ("GtNm1", 15),
    ("EtaN", 1),
    ("dEtaHdt", 1),
    ("EtaH", 1),
)
PICKUP_GRID = (62, 62)
PICKUP_DTYPE = np.dtype(">f8")
#: Plausible potential-temperature range, used to catch a byte-order mistake.
THETA_RANGE_C = (-5.0, 45.0)


class TwinExperimentError(RuntimeError):
    """Raised when the twin experiment design or its inputs are violated."""


def segment_plan(spec: TwinSpec = DEFAULT_SPEC) -> dict[str, Any]:
    """Return the immutable twin segment plan and its iteration boundaries."""

    segments = []
    for start_year, years in TWIN_SEGMENTS:
        segments.append(
            {
                "start_year": start_year,
                "end_year": start_year + years,
                "years": years,
                "start_iteration": start_year * STEPS_PER_YEAR,
                "end_iteration": (start_year + years) * STEPS_PER_YEAR,
                "perturbed_start": start_year == TWIN_START_YEAR,
                "expected_daily_records": years * MODEL_YEAR_DAYS,
            }
        )
    covered = sum(years for _, years in TWIN_SEGMENTS)
    if segments[0]["start_year"] != TWIN_START_YEAR or segments[-1]["end_year"] != TWIN_END_YEAR:
        raise TwinExperimentError("twin segment plan does not span years 100 through 125")
    if any(
        later["start_year"] != earlier["end_year"]
        for earlier, later in zip(segments, segments[1:])
    ):
        raise TwinExperimentError("twin segment plan is not contiguous")
    return {
        "experiment": spec.experiment,
        "control_regime": CONTROL_REGIME,
        "start_year": TWIN_START_YEAR,
        "end_year": TWIN_END_YEAR,
        "years": covered,
        "epsilon": spec.epsilon,
        "perturbed_fields": list(PERTURBED_FIELDS),
        "perturbation": PERTURBATION,
        "formula": PERTURBATION_FORMULA,
        "delta_t_seconds": DELTA_T_SECONDS,
        "expected_daily_records": covered * MODEL_YEAR_DAYS,
        "segments": segments,
    }


def _pickup_record_slices(meta_path: Path) -> tuple[Any, dict[str, slice]]:
    """Validate the control pickup metadata and map each field to its records."""

    meta = parse_mds_meta(meta_path)
    if meta.timestep != TWIN_START_ITERATION:
        raise TwinExperimentError(
            f"{meta_path} is at iteration {meta.timestep}, expected {TWIN_START_ITERATION}"
        )
    if meta.dtype != PICKUP_DTYPE:
        raise TwinExperimentError(f"{meta_path} is {meta.dtype}, expected {PICKUP_DTYPE}")
    if meta.dimensions != PICKUP_GRID:
        raise TwinExperimentError(f"{meta_path} has dimensions {meta.dimensions}, expected {PICKUP_GRID}")
    names = tuple(name.strip() for name in meta.fields)
    expected_names = tuple(name for name, _ in PICKUP_FIELD_LAYOUT)
    if names != expected_names:
        raise TwinExperimentError(
            f"{meta_path} fldList is {names}, expected {expected_names}"
        )
    expected_records = sum(count for _, count in PICKUP_FIELD_LAYOUT)
    if meta.nrecords != expected_records:
        raise TwinExperimentError(
            f"{meta_path} has {meta.nrecords} records, expected {expected_records}"
        )
    slices: dict[str, slice] = {}
    offset = 0
    for name, count in PICKUP_FIELD_LAYOUT:
        slices[name] = slice(offset, offset + count)
        offset += count
    missing = [name for name in PERTURBED_FIELDS if name not in slices]
    if missing:
        raise TwinExperimentError(f"{meta_path} is missing perturbation targets {missing}")
    return meta, slices


def write_perturbed_pickup(
    source_meta: Path, run_dir: Path, spec: TwinSpec = DEFAULT_SPEC
) -> dict[str, Any]:
    """Copy the year-100 pickup into ``run_dir`` and scale only ``Uvel``/``Vvel``.

    The ``.meta`` file is copied byte for byte so MITgcm's pickup-format check
    sees the control metadata unchanged; only the ``.data`` records of the
    perturbed fields differ.
    """

    source_meta = Path(source_meta).resolve()
    source_data = source_meta.with_suffix(".data")
    if not source_meta.is_file() or not source_data.is_file():
        raise TwinExperimentError(f"incomplete control pickup beside {source_meta}")
    meta, slices = _pickup_record_slices(source_meta)

    destination_meta = run_dir / source_meta.name
    destination_data = run_dir / source_data.name
    shutil.copy2(source_meta, destination_meta)

    count = meta.nrecords * PICKUP_GRID[0] * PICKUP_GRID[1]
    control = np.fromfile(source_data, dtype=PICKUP_DTYPE, count=count)
    if control.size != count:
        raise TwinExperimentError(
            f"{source_data} holds {control.size} values, expected {count}"
        )
    if not np.isfinite(control).all():
        raise TwinExperimentError(f"{source_data} decoded to non-finite values as {PICKUP_DTYPE}")
    control = control.reshape(meta.nrecords, *PICKUP_GRID)
    theta = control[slices["Theta"]]
    if not THETA_RANGE_C[0] <= float(theta.min()) or not float(theta.max()) <= THETA_RANGE_C[1]:
        raise TwinExperimentError(
            f"{source_data} Theta range {float(theta.min())}..{float(theta.max())} is implausible"
        )

    scale = spec.scale
    twin = control.copy()
    statistics: dict[str, Any] = {}
    for name in PERTURBED_FIELDS:
        block = control[slices[name]]
        twin[slices[name]] = block * scale
        delta = twin[slices[name]].astype(np.float64) - block.astype(np.float64)
        statistics[name] = {
            "records": [slices[name].start, slices[name].stop - 1],
            "max_abs_control": float(np.abs(block).max()),
            "max_abs_delta": float(np.abs(delta).max()),
            "rms_delta": float(np.sqrt(np.mean(np.square(delta)))),
            "l2_delta": float(np.sqrt(np.sum(np.square(delta)))),
            "zero_cells": int((block == 0.0).sum()),
        }
    twin.tofile(destination_data)

    written = np.fromfile(destination_data, dtype=PICKUP_DTYPE, count=count).reshape(
        meta.nrecords, *PICKUP_GRID
    )
    untouched = np.ones(meta.nrecords, dtype=bool)
    for name in PERTURBED_FIELDS:
        untouched[slices[name]] = False
    if not np.array_equal(written[untouched], control[untouched]):
        raise TwinExperimentError(f"{destination_data} altered records outside {PERTURBED_FIELDS}")
    for name in PERTURBED_FIELDS:
        before = control[slices[name]]
        after = written[slices[name]]
        nonzero = before != 0.0
        if not np.array_equal(after[~nonzero], before[~nonzero]):
            raise TwinExperimentError(f"{destination_data} moved a zero {name} cell")
        deviation = float(np.abs(after[nonzero] / before[nonzero] - scale).max())
        if deviation > 1.0e-12:
            raise TwinExperimentError(
                f"{destination_data} {name} scale deviates from {scale} by {deviation}"
            )
        statistics[name]["max_scale_deviation"] = deviation

    return {
        "applied": True,
        "epsilon": spec.epsilon,
        "mode": PERTURBATION,
        "formula": PERTURBATION_FORMULA,
        "fields": list(PERTURBED_FIELDS),
        "iteration": TWIN_START_ITERATION,
        "dataprec": str(PICKUP_DTYPE),
        "meta_copied_verbatim": True,
        "source_pickup_meta": str(source_meta),
        "source_pickup_data": str(source_data),
        "twin_pickup_meta": str(destination_meta),
        "twin_pickup_data": str(destination_data),
        "source_pickup_sha256": {"meta": _sha256(source_meta), "data": _sha256(source_data)},
        "twin_pickup_sha256": {
            "meta": _sha256(destination_meta),
            "data": _sha256(destination_data),
        },
        "field_statistics": statistics,
    }


def prepare_segment(
    project_root: Path,
    scratch_root: Path,
    executable: Path,
    start_year: int,
    years: int,
    spec: TwinSpec = DEFAULT_SPEC,
) -> dict[str, Any]:
    """Create one immutable twin segment and its provenance manifest."""

    if (start_year, years) not in TWIN_SEGMENTS:
        raise TwinExperimentError(
            f"segment ({start_year}, {years}) is outside the twin plan {TWIN_SEGMENTS}"
        )
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"AF--FNO executable is missing: {executable}")

    experiment_root = scratch_root / spec.root_name / spec.label
    run_dir = experiment_root / TWIN_PHASE / f"years_{start_year:03d}_{start_year + years:03d}"
    manifest_path = run_dir / "segment_manifest.json"
    expected_identity = {
        "experiment": spec.experiment,
        "phase": TWIN_PHASE,
        "start_year": start_year,
        "years": years,
        "epsilon": spec.epsilon,
        "executable_sha256": _sha256(executable),
    }
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if any(prior.get(key) != value for key, value in expected_identity.items()):
            raise TwinExperimentError(f"twin segment identity changed: {manifest_path}")
        return prior
    if run_dir.exists():
        raise TwinExperimentError(
            f"{run_dir} exists without a manifest; inspect and remove it before retrying"
        )

    run_dir.mkdir(parents=True, exist_ok=False)
    start_iteration = start_year * STEPS_PER_YEAR
    end_iteration = (start_year + years) * STEPS_PER_YEAR
    (run_dir / "data").write_text(render_data(start_iteration, years * STEPS_PER_YEAR))
    input_root = project_root / "af_fno" / "mitgcm" / "input"
    for name in ("data.pkg", "eedata"):
        shutil.copy2(input_root / name, run_dir / name)
    shutil.copy2(input_root / f"data.diagnostics.{TWIN_PHASE}", run_dir / "data.diagnostics")

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

    if start_year == TWIN_START_YEAR:
        control_parent = _find_parent(scratch_root / "mitgcm" / CONTROL_REGIME, TWIN_START_YEAR)
        if int(control_parent["end_iteration"]) != start_iteration:
            raise TwinExperimentError("control S0 iteration does not match the twin start year")
        if control_parent.get("experiment") != CONTROL_REGIME:
            raise TwinExperimentError("twin start pickup does not come from the S0 control")
        perturbation = write_perturbed_pickup(
            Path(str(control_parent["pickup_meta"])), run_dir, spec=spec
        )
        parent_result = str(control_parent["_result_path"])
        parent_pickups = [
            perturbation["twin_pickup_meta"],
            perturbation["twin_pickup_data"],
        ]
    else:
        parent = _find_parent(experiment_root, start_year)
        if int(parent["end_iteration"]) != start_iteration:
            raise TwinExperimentError("parent twin iteration does not match the requested start year")
        if parent.get("experiment") != spec.experiment:
            raise TwinExperimentError("twin restart pickup does not come from the twin trajectory")
        parent_pickups = _link_parent_pickups(parent, run_dir, start_iteration)
        parent_result = str(parent["_result_path"])
        perturbation = {
            "applied": False,
            "epsilon": spec.epsilon,
            "mode": PERTURBATION,
            "formula": PERTURBATION_FORMULA,
            "fields": list(PERTURBED_FIELDS),
            "note": "unmodified restart from the preceding twin segment",
        }

    source = project_root / "external" / "MITgcm"
    revision = _git_revision(source)
    if revision != MITGCM_COMMIT:
        raise TwinExperimentError(f"Expected MITgcm {MITGCM_COMMIT}, found {revision}")
    forcing_sha256 = {
        name: _sha256(run_dir / name)
        for name in ("bathy.bin", "windx_cosy.bin", "SST_relax.bin")
    }
    source_pickup_sha256 = perturbation.get("source_pickup_sha256")
    twin_pickup_sha256 = perturbation.get("twin_pickup_sha256")
    manifest = {
        **expected_identity,
        "control_regime": CONTROL_REGIME,
        "twin_start_year": TWIN_START_YEAR,
        "twin_end_year": TWIN_END_YEAR,
        "end_year": start_year + years,
        "tau0_n_m2": 0.1,
        "start_iteration": start_iteration,
        "end_iteration": end_iteration,
        "n_time_steps": years * STEPS_PER_YEAR,
        "delta_t_seconds": DELTA_T_SECONDS,
        "perturbed_fields": list(PERTURBED_FIELDS),
        "perturbation": PERTURBATION,
        "formula": PERTURBATION_FORMULA,
        "source_pickup_sha256": source_pickup_sha256,
        "twin_pickup_sha256": twin_pickup_sha256,
        "wind_sha256": forcing_sha256["windx_cosy.bin"],
        "perturbation_detail": perturbation,
        "run_dir": str(run_dir),
        "parent_result": parent_result,
        "parent_pickups": parent_pickups,
        "mitgcm_commit": revision,
        "configuration_sha256": {
            name: _sha256(run_dir / name)
            for name in ("data", "data.pkg", "eedata", "data.diagnostics")
        },
        "forcing_sha256": forcing_sha256,
        "created_by": {
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def run_segment(manifest: Mapping[str, Any], launcher: Sequence[str] | None = None) -> dict[str, Any]:
    """Run and validate one prepared twin segment."""

    run_dir = Path(str(manifest["run_dir"]))
    result_path = run_dir / "segment_result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    partial = sorted(
        path
        for pattern in ("run.log", "STDOUT.*", "STDERR.*", "dynState.*", "surfState.*")
        for path in run_dir.glob(pattern)
    )
    if partial:
        raise TwinExperimentError(f"Refusing to overwrite incomplete twin output: {partial[:8]}")
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
        raise TwinExperimentError(
            f"MITgcm exited with {completed.returncode}; inspect {run_dir / 'run.log'}"
        )

    end_iteration = int(manifest["end_iteration"])
    pickup_meta = run_dir / f"pickup.{end_iteration:010d}.meta"
    pickup_data = pickup_meta.with_suffix(".data")
    if not pickup_meta.is_file() or not pickup_data.is_file():
        raise TwinExperimentError(f"Missing twin end pickup at iteration {end_iteration}")
    years = int(manifest["years"])
    diagnostics = {
        "dynState": len(list(run_dir.glob("dynState.*.meta"))),
        "surfState": len(list(run_dir.glob("surfState.*.meta"))),
    }
    expected = {"dynState": years * MODEL_YEAR_DAYS, "surfState": years * MODEL_YEAR_DAYS}
    if diagnostics != expected:
        raise TwinExperimentError(
            f"twin diagnostic count mismatch: expected {expected}, found {diagnostics}"
        )

    result = {
        # Taken from the manifest, not the module constant, so a segment always
        # reports the experiment it was actually prepared as.
        "experiment": manifest["experiment"],
        "control_regime": CONTROL_REGIME,
        "phase": manifest["phase"],
        "epsilon": manifest["epsilon"],
        "perturbed_fields": list(PERTURBED_FIELDS),
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-segment", "run-segment"):
        segment = commands.add_parser(name)
        segment.add_argument("--project-root", type=Path, required=True)
        segment.add_argument("--scratch-root", type=Path, required=True)
        segment.add_argument("--executable", type=Path, required=True)
        segment.add_argument("--start-year", type=int, required=True)
        segment.add_argument("--years", type=int, required=True)
    commands.add_parser("plan")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        result: Any = segment_plan()
    else:
        manifest = prepare_segment(
            args.project_root.resolve(),
            args.scratch_root.resolve(),
            args.executable.resolve(),
            args.start_year,
            args.years,
        )
        result = manifest if args.command == "prepare-segment" else run_segment(manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
