"""Generic restart-safe MITgcm segment driver for the amplitude pilot.

Unlike :mod:`af_response_pickup_bank` (a single fixed 320-day unperturbed
bridge), the amplitude pilot (plan section 7.2/10) needs many short
independent segments that share one shape -- start from a source pickup
(unperturbed for the nominal/duplicate branches, additively edited for the
144 signed perturbed branches), run for an arbitrary number of days, and
archive a complete pickup every 10 days -- but differ in start day, duration,
and regime. This module factors that shared shape out; the caller (
``scripts/build_amplitude_pilot.py``) is responsible for producing the
correct source pickup (via
:func:`bire_repro.af_s0_twin.write_declared_pickup_edits` for perturbed
branches) and for the forcing/config directory to copy from.
"""

from __future__ import annotations

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
    MPI_RANKS,
    STEPS_PER_DAY,
    STEPS_PER_YEAR,
    _atomic_json,
    _git_revision,
    _sha256,
    render_data,
)

VERSION = "af_pilot_segment_v1"

#: Same year-100 anchor as af_response_pickup_bank.BASE_ITERATION.
BASE_ITERATION = 100 * STEPS_PER_YEAR

FORCING_FILES = (
    "data.pkg",
    "eedata",
    "data.diagnostics",
    "bathy.bin",
    "windx_cosy.bin",
    "SST_relax.bin",
)


class PilotSegmentError(RuntimeError):
    """Raised when a pilot segment is misconfigured."""


def day_to_iteration(day: int) -> int:
    return BASE_ITERATION + STEPS_PER_DAY * int(day)


def _pchkpt_freq_override(rendered: str, checkpoint_interval_days: int) -> str:
    needle = "pChkptFreq=31104000.,"
    if rendered.count(needle) != 1:
        raise PilotSegmentError(f"expected exactly one {needle!r} in the rendered physics")
    seconds = checkpoint_interval_days * STEPS_PER_DAY * DELTA_T_SECONDS
    return rendered.replace(needle, f"pChkptFreq={seconds}.,", 1)


def _cg2d_target_residual_override(rendered: str, target_residual: float) -> str:
    """section 10.3: rerun a control branch at a tighter cg2d solver tolerance."""

    needle = "cg2dTargetResidual=1.E-7,"
    if rendered.count(needle) != 1:
        raise PilotSegmentError(f"expected exactly one {needle!r} in the rendered physics")
    return rendered.replace(needle, f"cg2dTargetResidual={target_residual:.0E},", 1)


def render_segment_data(
    start_day: int,
    duration_days: int,
    checkpoint_interval_days: int = 10,
    cg2d_target_residual: float | None = None,
) -> str:
    if duration_days <= 0 or duration_days % checkpoint_interval_days:
        raise PilotSegmentError(
            "duration_days must be a positive multiple of checkpoint_interval_days"
        )
    start_iteration = day_to_iteration(start_day)
    n_time_steps = duration_days * STEPS_PER_DAY
    rendered = _pchkpt_freq_override(
        render_data(start_iteration, n_time_steps), checkpoint_interval_days
    )
    if cg2d_target_residual is not None:
        rendered = _cg2d_target_residual_override(rendered, cg2d_target_residual)
    return rendered


def archived_checkpoints(
    start_day: int, duration_days: int, checkpoint_interval_days: int = 10
) -> tuple[tuple[int, int], ...]:
    segments = duration_days // checkpoint_interval_days
    return tuple(
        (
            start_day + checkpoint_interval_days * segment,
            day_to_iteration(start_day + checkpoint_interval_days * segment),
        )
        for segment in range(1, segments + 1)
    )


def prepare_segment(
    project_root: Path,
    scratch_root: Path,
    executable: Path,
    run_label: str,
    forcing_source_dir: Path,
    pickup_meta_path: Path,
    pickup_data_path: Path,
    pickup_meta_sha256: str,
    pickup_data_sha256: str,
    start_day: int,
    duration_days: int,
    checkpoint_interval_days: int = 10,
    cg2d_target_residual: float | None = None,
    scratch_subdir: str = "mitgcm_amplitude_pilot_v1",
) -> dict[str, Any]:
    """Create one immutable segment run directory identified by ``run_label``.

    ``pickup_meta_path``/``pickup_data_path`` is the exact pickup to place at
    the start of this run -- the original annual pickup for a nominal branch,
    or an already-edited perturbed pickup for a signed branch. Its filename
    must already encode the correct starting iteration. ``cg2d_target_residual``
    overrides the production ``1.E-7`` solver tolerance for section-10.3's
    tight-CG controls; leave it ``None`` for every ordinary branch.
    ``scratch_subdir`` namespaces run directories under ``scratch_root``;
    the default preserves the original amplitude-pilot-only behaviour byte
    for byte, so every existing caller is unaffected. Step 9's production
    response runs pass ``"mitgcm_forward_response_v1"`` to keep the two
    study phases in clearly separate scratch trees.
    """

    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"AF--FNO executable is missing: {executable}")
    pickup_meta_path = Path(pickup_meta_path).resolve()
    pickup_data_path = Path(pickup_data_path).resolve()
    if _sha256(pickup_meta_path) != pickup_meta_sha256:
        raise PilotSegmentError(f"pickup meta hash mismatch: {pickup_meta_path}")
    if _sha256(pickup_data_path) != pickup_data_sha256:
        raise PilotSegmentError(f"pickup data hash mismatch: {pickup_data_path}")

    start_iteration = day_to_iteration(start_day)
    expected_name = f"pickup.{start_iteration:010d}.meta"
    if pickup_meta_path.name != expected_name:
        raise PilotSegmentError(
            f"pickup {pickup_meta_path.name} does not match day {start_day} iteration {start_iteration}"
        )
    end_day = start_day + duration_days
    n_time_steps = duration_days * STEPS_PER_DAY

    run_dir = scratch_root / scratch_subdir / run_label
    manifest_path = run_dir / "segment_manifest.json"
    expected_identity = {
        "version": VERSION,
        "run_label": run_label,
        "start_day": start_day,
        "end_day": end_day,
        "checkpoint_interval_days": checkpoint_interval_days,
        "cg2d_target_residual": cg2d_target_residual,
        "executable_sha256": _sha256(executable),
        "pickup_sha256": {"meta": pickup_meta_sha256, "data": pickup_data_sha256},
    }
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if any(prior.get(key) != value for key, value in expected_identity.items()):
            raise PilotSegmentError(f"pilot segment identity changed: {manifest_path}")
        return prior
    if run_dir.exists():
        raise PilotSegmentError(
            f"{run_dir} exists without a manifest; inspect and remove it before retrying"
        )

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "data").write_text(
        render_segment_data(
            start_day, duration_days, checkpoint_interval_days, cg2d_target_residual
        )
    )
    for name in FORCING_FILES:
        shutil.copy2(Path(forcing_source_dir) / name, run_dir / name)
    (run_dir / pickup_meta_path.name).symlink_to(pickup_meta_path)
    (run_dir / pickup_data_path.name).symlink_to(pickup_data_path)
    (run_dir / "mitgcmuv").symlink_to(executable)

    revision = _git_revision(project_root / "external" / "MITgcm")
    if revision != MITGCM_COMMIT:
        raise PilotSegmentError(f"Expected MITgcm {MITGCM_COMMIT}, found {revision}")

    manifest = {
        **expected_identity,
        "start_iteration": start_iteration,
        "end_iteration": day_to_iteration(end_day),
        "n_time_steps": n_time_steps,
        "archived_checkpoints": [
            {"day": day, "iteration": iteration}
            for day, iteration in archived_checkpoints(
                start_day, duration_days, checkpoint_interval_days
            )
        ],
        "run_dir": str(run_dir),
        "pickup_meta_path": str(pickup_meta_path),
        "pickup_data_path": str(pickup_data_path),
        "mitgcm_commit": revision,
        "configuration_sha256": {
            name: _sha256(run_dir / name) for name in ("data", *FORCING_FILES)
        },
        "created_by": {
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def run_segment(
    manifest: Mapping[str, Any], launcher: Sequence[str] | None = None
) -> dict[str, Any]:
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
        raise PilotSegmentError(f"refusing to overwrite incomplete pilot output: {partial[:8]}")
    if launcher is None:
        launcher = ["srun", "--mpi=pmix", "-n", str(MPI_RANKS)]
    command = [*launcher, str(run_dir / "mitgcmuv")]
    started = time.monotonic()
    with (run_dir / "run.log").open("w") as stdout:
        completed = subprocess.run(
            command, cwd=run_dir, stdout=stdout, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.monotonic() - started
    if completed.returncode:
        raise PilotSegmentError(
            f"MITgcm exited with {completed.returncode}; inspect {run_dir / 'run.log'}"
        )

    total_days = int(manifest["end_day"]) - int(manifest["start_day"])
    diagnostics = {
        "dynState": len(list(run_dir.glob("dynState.*.meta"))),
        "surfState": len(list(run_dir.glob("surfState.*.meta"))),
    }
    expected_diagnostics = {"dynState": total_days, "surfState": total_days}
    if diagnostics != expected_diagnostics:
        raise PilotSegmentError(
            f"pilot diagnostic count mismatch: expected {expected_diagnostics}, found {diagnostics}"
        )

    archived: list[dict[str, Any]] = []
    for entry in manifest["archived_checkpoints"]:
        day, iteration = int(entry["day"]), int(entry["iteration"])
        pickup_meta = run_dir / f"pickup.{iteration:010d}.meta"
        pickup_data = pickup_meta.with_suffix(".data")
        if not pickup_meta.is_file() or not pickup_data.is_file():
            raise PilotSegmentError(
                f"missing archived pickup for day {day} at iteration {iteration}"
            )
        archived.append(
            {
                "day": day,
                "iteration": iteration,
                "meta_path": str(pickup_meta),
                "data_path": str(pickup_data),
                "meta_sha256": _sha256(pickup_meta),
                "data_sha256": _sha256(pickup_data),
            }
        )

    result = {
        "version": manifest["version"],
        "run_label": manifest["run_label"],
        "start_day": manifest["start_day"],
        "end_day": manifest["end_day"],
        "run_dir": str(run_dir),
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "diagnostics": diagnostics,
        "archived_pickups": archived,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(result_path, result)
    return result
