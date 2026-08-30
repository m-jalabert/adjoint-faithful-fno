"""Restart-safe MITgcm validation pickup-bank bridge for the response study.

Section 7.1 of ``docs/Adjoint_faithful_response_training_plan.md`` requires an
UNPERTURBED continuation of each regime (S0, S1, S2) from the existing,
already-archived day-5,760 pickup through day 6,080, archiving a complete
pickup every 10 model-days.  Days 6,010, 6,050, and 6,080 become the
response-validation source pickups; the caller is responsible for verifying
their P32 projections against ``trajectories_v3.zarr`` (Gate D0) using
``scripts/build_forward_response_inventory.py::pickup_to_trajectory_p32`` --
that check needs the trajectory-v3 grid/state and therefore lives in the
``scripts/`` orchestrator, not here.

This module is a pure MITgcm driver, structured exactly like its siblings
:mod:`af_s0` and :mod:`af_independent_wind_trajectories`: ``prepare_segment``
stages one immutable, restart-safe run directory; ``run_segment`` runs it and
verifies the resulting pickup/diagnostic inventory.  It intentionally does
*not* import anything from ``scripts/`` -- the caller resolves and hashes the
day-5,760 source pickup (against the hash-pinned trajectory-v3 source
manifest) and passes the concrete paths and expected hashes in.

Unlike the twin/independent-wind chains, this is a single 320-day segment per
regime, not a multi-segment chain: the source pickup is a mid-segment annual
checkpoint of the existing trajectory-v3 chain, not the output of an earlier
call into this module.
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
    MPI_RANKS,
    STEPS_PER_DAY,
    STEPS_PER_YEAR,
    _atomic_json,
    _git_revision,
    _sha256,
    render_data,
)

VERSION = "af_response_pickup_bank_v1"

REGIMES = ("S0", "S1", "S2")
ROOT_NAME = "mitgcm_response_pickup_bank_v1"

#: The day-0 iteration used throughout the response study (year 100 in the
#: af_s0/af_independent_wind_trajectories numbering).  Kept derived, not a
#: bare literal, so it stays tied to the same year-100 anchor the trajectory
#: chains use; it must equal 2,592,000, the ``BASE_ITERATION`` constant in
#: ``scripts/build_forward_response_inventory.py``.
BASE_ITERATION = 100 * STEPS_PER_YEAR

SOURCE_DAY = 5760
END_DAY = 6080
SEGMENT_DAYS = 10
TOTAL_DAYS = END_DAY - SOURCE_DAY
N_TIME_STEPS = TOTAL_DAYS * STEPS_PER_DAY
STEPS_PER_SEGMENT = SEGMENT_DAYS * STEPS_PER_DAY
PCHKPT_FREQ_SECONDS = STEPS_PER_SEGMENT * DELTA_T_SECONDS
N_SEGMENTS = TOTAL_DAYS // SEGMENT_DAYS
RETAINED_ANCHOR_DAYS = (6010, 6050, 6080)

if TOTAL_DAYS % SEGMENT_DAYS:
    raise RuntimeError("TOTAL_DAYS must be an exact multiple of SEGMENT_DAYS")
if any(
    day <= SOURCE_DAY or day > END_DAY or (day - SOURCE_DAY) % SEGMENT_DAYS
    for day in RETAINED_ANCHOR_DAYS
):
    raise RuntimeError("RETAINED_ANCHOR_DAYS must fall on an archived segment boundary")


class PickupBankError(RuntimeError):
    """Raised when the validation pickup-bank bridge chain is misconfigured."""


def day_to_iteration(day: int) -> int:
    return BASE_ITERATION + STEPS_PER_DAY * int(day)


#: (day, iteration) for every archived checkpoint, source exclusive.
ARCHIVED_CHECKPOINTS: tuple[tuple[int, int], ...] = tuple(
    (SOURCE_DAY + SEGMENT_DAYS * segment, day_to_iteration(SOURCE_DAY + SEGMENT_DAYS * segment))
    for segment in range(1, N_SEGMENTS + 1)
)


def _pchkpt_freq_override(rendered: str) -> str:
    """Replace the annual ``pChkptFreq`` in :func:`render_data` with 10 days."""

    needle = "pChkptFreq=31104000.,"
    if rendered.count(needle) != 1:
        raise PickupBankError(f"expected exactly one {needle!r} in the rendered physics")
    replacement = f"pChkptFreq={PCHKPT_FREQ_SECONDS}.,"
    return rendered.replace(needle, replacement, 1)


def render_bridge_data() -> str:
    """The full 320-day unperturbed bridge segment, checkpointed every 10 days."""

    start_iteration = day_to_iteration(SOURCE_DAY)
    return _pchkpt_freq_override(render_data(start_iteration, N_TIME_STEPS))


def prepare_segment(
    project_root: Path,
    scratch_root: Path,
    executable: Path,
    regime: str,
    source_meta_path: Path,
    source_data_path: Path,
    source_meta_sha256: str,
    source_data_sha256: str,
) -> dict[str, Any]:
    """Create the one immutable 320-day bridge segment for ``regime``.

    ``source_meta_path``/``source_data_path`` must be the already-resolved,
    already-hash-verified day-5,760 pickup for this regime (see
    ``scripts/build_forward_response_inventory.py::resolve_annual_pickup``);
    this function re-checks the declared hashes itself as a second,
    independent integrity gate before linking anything into the run.
    """

    if regime not in REGIMES:
        raise PickupBankError(f"regime must be one of {REGIMES}, got {regime!r}")
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"AF--FNO executable is missing: {executable}")
    source_meta_path = Path(source_meta_path).resolve()
    source_data_path = Path(source_data_path).resolve()
    if _sha256(source_meta_path) != source_meta_sha256:
        raise PickupBankError(f"source pickup meta hash mismatch: {source_meta_path}")
    if _sha256(source_data_path) != source_data_sha256:
        raise PickupBankError(f"source pickup data hash mismatch: {source_data_path}")

    start_iteration = day_to_iteration(SOURCE_DAY)
    end_iteration = day_to_iteration(END_DAY)
    expected_name = f"pickup.{start_iteration:010d}.meta"
    if source_meta_path.name != expected_name:
        raise PickupBankError(
            f"source pickup {source_meta_path.name} does not match day {SOURCE_DAY} "
            f"iteration {start_iteration}"
        )

    run_dir = scratch_root / ROOT_NAME / regime / f"bridge_{SOURCE_DAY}_{END_DAY}"
    manifest_path = run_dir / "segment_manifest.json"
    expected_identity = {
        "version": VERSION,
        "regime": regime,
        "source_day": SOURCE_DAY,
        "end_day": END_DAY,
        "segment_days": SEGMENT_DAYS,
        "executable_sha256": _sha256(executable),
        "source_pickup_sha256": {"meta": source_meta_sha256, "data": source_data_sha256},
    }
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if any(prior.get(key) != value for key, value in expected_identity.items()):
            raise PickupBankError(f"bridge segment identity changed: {manifest_path}")
        return prior
    if run_dir.exists():
        raise PickupBankError(
            f"{run_dir} exists without a manifest; inspect and remove it before retrying"
        )

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "data").write_text(render_bridge_data())
    source_run_dir = source_meta_path.parent
    for name in (
        "data.pkg",
        "eedata",
        "data.diagnostics",
        "bathy.bin",
        "windx_cosy.bin",
        "SST_relax.bin",
    ):
        shutil.copy2(source_run_dir / name, run_dir / name)
    (run_dir / source_meta_path.name).symlink_to(source_meta_path)
    (run_dir / source_data_path.name).symlink_to(source_data_path)
    (run_dir / "mitgcmuv").symlink_to(executable)

    revision = _git_revision(project_root / "external" / "MITgcm")
    if revision != MITGCM_COMMIT:
        raise PickupBankError(f"Expected MITgcm {MITGCM_COMMIT}, found {revision}")

    manifest = {
        **expected_identity,
        "start_iteration": start_iteration,
        "end_iteration": end_iteration,
        "n_time_steps": N_TIME_STEPS,
        "pchkpt_freq_seconds": PCHKPT_FREQ_SECONDS,
        "delta_t_seconds": DELTA_T_SECONDS,
        "archived_checkpoints": [
            {"day": day, "iteration": iteration} for day, iteration in ARCHIVED_CHECKPOINTS
        ],
        "retained_anchor_days": list(RETAINED_ANCHOR_DAYS),
        "run_dir": str(run_dir),
        "source_meta_path": str(source_meta_path),
        "source_data_path": str(source_data_path),
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
    manifest: Mapping[str, Any], launcher: Sequence[str] | None = None
) -> dict[str, Any]:
    """Run one prepared bridge segment and archive/hash every 10-day pickup."""

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
        raise PickupBankError(f"refusing to overwrite incomplete bridge output: {partial[:8]}")
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
        raise PickupBankError(
            f"MITgcm exited with {completed.returncode}; inspect {run_dir / 'run.log'}"
        )

    total_days = int(manifest["end_day"]) - int(manifest["source_day"])
    diagnostics = {
        "dynState": len(list(run_dir.glob("dynState.*.meta"))),
        "surfState": len(list(run_dir.glob("surfState.*.meta"))),
    }
    expected_diagnostics = {"dynState": total_days, "surfState": total_days}
    if diagnostics != expected_diagnostics:
        raise PickupBankError(
            f"bridge diagnostic count mismatch: expected {expected_diagnostics}, found {diagnostics}"
        )

    archived_pickups: list[dict[str, Any]] = []
    for entry in manifest["archived_checkpoints"]:
        day, iteration = int(entry["day"]), int(entry["iteration"])
        pickup_meta = run_dir / f"pickup.{iteration:010d}.meta"
        pickup_data = pickup_meta.with_suffix(".data")
        if not pickup_meta.is_file() or not pickup_data.is_file():
            raise PickupBankError(f"missing archived pickup for day {day} at iteration {iteration}")
        archived_pickups.append(
            {
                "day": day,
                "iteration": iteration,
                "meta_path": str(pickup_meta),
                "data_path": str(pickup_data),
                "meta_sha256": _sha256(pickup_meta),
                "data_sha256": _sha256(pickup_data),
            }
        )

    retained_days = set(manifest["retained_anchor_days"])
    retained = [entry for entry in archived_pickups if entry["day"] in retained_days]
    if {entry["day"] for entry in retained} != retained_days:
        raise PickupBankError("not every retained anchor day has an archived pickup")

    result = {
        "version": manifest["version"],
        "regime": manifest["regime"],
        "source_day": manifest["source_day"],
        "end_day": manifest["end_day"],
        "start_iteration": manifest["start_iteration"],
        "end_iteration": manifest["end_iteration"],
        "run_dir": str(run_dir),
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "diagnostics": diagnostics,
        "archived_pickups": archived_pickups,
        "retained_pickups": retained,
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
        segment.add_argument("--regime", choices=REGIMES, required=True)
        segment.add_argument("--source-meta", type=Path, required=True)
        segment.add_argument("--source-data", type=Path, required=True)
        segment.add_argument("--source-meta-sha256", required=True)
        segment.add_argument("--source-data-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = prepare_segment(
        args.project_root.resolve(),
        args.scratch_root.resolve(),
        args.executable.resolve(),
        args.regime,
        args.source_meta.resolve(),
        args.source_data.resolve(),
        args.source_meta_sha256,
        args.source_data_sha256,
    )
    result = manifest if args.command == "prepare-segment" else run_segment(manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
