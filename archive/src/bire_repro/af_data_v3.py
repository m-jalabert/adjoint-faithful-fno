"""Immutable 25-year-per-regime trajectory-v3 store on independent equilibria.

Trajectory-v2 pools three regimes whose S1 and S2 members are *branches*: both
restart from the validated S0 year-100 state and receive only a five-year
adjustment, so their slow fields still carry the S0 equilibrium.  Trajectory-v3
replaces them with regimes equilibrated independently from the original MITgcm
tutorial initial condition under their own wind for the full 100 years, then run
25 years of daily production::

    regime   tau0 (N m-2)   equilibration                production days
    S0       0.100          existing independent 100 y   9,000 (years 100--125)
    S1       0.075          new independent 100 y        9,000 (years 100--125)
    S2       0.125          new independent 100 y        9,000 (years 100--125)

S0 is assembled from its three existing production campaigns rather than re-run:
that campaign already *is* an independent 100-year equilibration at
tau0 = 0.100, and its chain spans years 100--126, so the first 9,000 days are
taken and the remaining 360 are left out of the store.

Each regime's ``windx_cosy.bin`` is read from its own run directory, so the
``wind_stress_x`` static channel carries that regime's actual scaled amplitude
rather than the control field.

The split is strictly chronological with the project's 90-day buffers::

    code  split        indices      days
    1     train        0--5039      5040
    0     buffer       5040--5129     90
    2     validation   5130--6389   1260
    0     buffer       6390--6479     90
    3     test         6480--8999   2520

Training and validation blocks are pooled across the three regimes: one FNO is
trained on all three training blocks and selected on all three validation blocks.
The 2,520-day test block admits a complete 2,000-day rollout from any start in
its first 520 days, 6480--6999.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numcodecs
import numpy as np
import zarr

from .af_data import (
    MODEL_DAY_STEPS,
    STATE_CHANNELS,
    STATIC_FEATURES,
    DatasetSpec,
    TrajectoryInventory,
    _iteration,
    _read_field_pair,
    geometry_and_forcing,
)

DATASET_VERSION = "trajectories_v3"
EXPERIMENTS = ("S0", "S1", "S2")
PRODUCTION_DAYS = 9000
STATE_CHANNEL_COUNT = len(STATE_CHANNELS)

TRAIN_RANGE = (0, 5040)
VALIDATION_RANGE = (5130, 6390)
TEST_RANGE = (6480, 9000)
BUFFER_RANGES = ((5040, 5130), (6390, 6480))
HORIZON_DAYS = 10
TRAIN_CODE, VALIDATION_CODE, TEST_CODE, BUFFER_CODE = 1, 2, 3, 0

#: Longest rollout the test block supports, and the start window that admits it.
MAXIMUM_TEST_ROLLOUT_DAYS = 2000
TEST_START_WINDOW = (TEST_RANGE[0], TEST_RANGE[1] - MAXIMUM_TEST_ROLLOUT_DAYS)

#: Ordered production chains.  ``take`` trims the final S0 campaign, which runs
#: 360 days past the 25-year mark.
CHAINS: dict[str, tuple[tuple[str, int], ...]] = {
    "S0": (
        ("mitgcm/S0/production/years_100_110", 3600),
        ("mitgcm_v2/S0/production/years_110_120", 3600),
        ("mitgcm_long_truth_v1/S0/production/years_120_126", 1800),
    ),
    "S1": (
        ("mitgcm_independent_v1/S1/production/years_100_110", 3600),
        ("mitgcm_independent_v1/S1/production/years_110_120", 3600),
        ("mitgcm_independent_v1/S1/production/years_120_125", 1800),
    ),
    "S2": (
        ("mitgcm_independent_v1/S2/production/years_100_110", 3600),
        ("mitgcm_independent_v1/S2/production/years_110_120", 3600),
        ("mitgcm_independent_v1/S2/production/years_120_125", 1800),
    ),
}


class TrajectoryV3Error(RuntimeError):
    """Raised when the trajectory-v3 build violates its contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def split_codes() -> tuple[np.ndarray, np.ndarray]:
    """Chronological snapshot and pair codes for the 9,000-day record."""

    snapshots = np.zeros(PRODUCTION_DAYS, dtype=np.uint8)
    pairs = np.zeros(PRODUCTION_DAYS, dtype=np.uint8)
    for (start, stop), code in (
        (TRAIN_RANGE, TRAIN_CODE),
        (VALIDATION_RANGE, VALIDATION_CODE),
        (TEST_RANGE, TEST_CODE),
    ):
        snapshots[start:stop] = code
        pair_stop = stop - HORIZON_DAYS
        if pair_stop > start:
            pairs[start:pair_stop] = code
    return snapshots, pairs


def verify_split() -> dict[str, Any]:
    """Assert the split is ordered, buffered, and admits the declared rollouts."""

    snapshots, pairs = split_codes()
    train = np.flatnonzero(snapshots == TRAIN_CODE)
    validation = np.flatnonzero(snapshots == VALIDATION_CODE)
    test = np.flatnonzero(snapshots == TEST_CODE)
    if not (train.max() < validation.min() and validation.max() < test.min()):
        raise TrajectoryV3Error("trajectory-v3 splits are not strictly ordered")
    for start, stop in BUFFER_RANGES:
        if np.any(snapshots[start:stop] != BUFFER_CODE):
            raise TrajectoryV3Error("a declared buffer is not empty")
    if (
        validation.min() - train.max() - 1 != 90
        or test.min() - validation.max() - 1 != 90
    ):
        raise TrajectoryV3Error("buffers are not the project's 90 days")
    if int(np.flatnonzero(pairs == TRAIN_CODE).max()) + HORIZON_DAYS >= VALIDATION_RANGE[0]:
        raise TrajectoryV3Error("a training pair reaches into validation")
    if int(np.flatnonzero(pairs == VALIDATION_CODE).max()) + HORIZON_DAYS >= TEST_RANGE[0]:
        raise TrajectoryV3Error("a validation pair reaches into test")
    latest_start = TEST_START_WINDOW[1] - 1
    if latest_start + MAXIMUM_TEST_ROLLOUT_DAYS > int(test.max()):
        raise TrajectoryV3Error("the declared test start window overruns the test block")
    return {
        "train": [int(train.min()), int(train.max())],
        "validation": [int(validation.min()), int(validation.max())],
        "test": [int(test.min()), int(test.max())],
        "train_days": int(train.size),
        "validation_days": int(validation.size),
        "test_days": int(test.size),
        "buffer_days": int(np.count_nonzero(snapshots == BUFFER_CODE)),
        "test_start_window": list(TEST_START_WINDOW),
        "maximum_test_rollout_days": MAXIMUM_TEST_ROLLOUT_DAYS,
        "pooled": "training and validation blocks are pooled across S0, S1 and S2",
    }


@dataclass(frozen=True)
class ChainInventory:
    """One regime's ordered 9,000-day production chain."""

    experiment: str
    dyn_meta: tuple[Path, ...]
    surf_meta: tuple[Path, ...]
    iterations: tuple[int, ...]
    segments: tuple[dict[str, Any], ...]
    production_root: Path

    def as_trajectory_inventory(self) -> TrajectoryInventory:
        """Adapter so the unchanged geometry reader can be reused."""

        return TrajectoryInventory(
            self.experiment,
            self.dyn_meta,
            self.surf_meta,
            self.iterations,
            self.production_root,
        )


def inventory_chain(scratch_root: Path, experiment: str) -> ChainInventory:
    """Assemble and verify one regime's contiguous daily production chain."""

    dyn: list[Path] = []
    segments: list[dict[str, Any]] = []
    for relative, take in CHAINS[experiment]:
        directory = (Path(scratch_root) / relative).resolve()
        result_path = directory / "segment_result.json"
        if not result_path.is_file():
            raise TrajectoryV3Error(f"missing segment result: {result_path}")
        result = json.loads(result_path.read_text())
        found = sorted(
            directory.glob("dynState.*.meta"),
            key=lambda path: _iteration(path, "dynState"),
        )
        if len(found) < take:
            raise TrajectoryV3Error(
                f"{experiment} segment {relative} has {len(found)} records, needs {take}"
            )
        dyn.extend(found[:take])
        segments.append(
            {
                "run_dir": str(directory),
                "taken": int(take),
                "available": len(found),
                "start_year": result.get("start_year"),
                "end_year": result.get("end_year"),
            }
        )
    if len(dyn) != PRODUCTION_DAYS:
        raise TrajectoryV3Error(
            f"{experiment} chain has {len(dyn)} records; expected {PRODUCTION_DAYS}"
        )
    iterations = tuple(_iteration(path, "dynState") for path in dyn)
    gaps = [
        (before, after)
        for before, after in zip(iterations, iterations[1:])
        if after - before != MODEL_DAY_STEPS
    ]
    if gaps:
        raise TrajectoryV3Error(
            f"{experiment} chain is not contiguous daily output at {gaps[:3]}"
        )
    surf = tuple(
        path.with_name(f"surfState.{iteration:010d}.meta")
        for path, iteration in zip(dyn, iterations)
    )
    missing = [
        str(path)
        for path in surf
        if not path.is_file() or not path.with_suffix(".data").is_file()
    ]
    if missing:
        raise TrajectoryV3Error(f"{experiment} is missing surfState pairs: {missing[:3]}")
    return ChainInventory(
        experiment,
        tuple(dyn),
        surf,
        iterations,
        tuple(segments),
        Path(segments[0]["run_dir"]),
    )


def build_dataset_v3(
    scratch_root: str | Path,
    output_path: str | Path,
    *,
    chunk_days: int = 8,
) -> dict[str, Any]:
    """Build the immutable 25-year-per-regime trajectory-v3 Zarr store."""

    started = time.monotonic()
    scratch = Path(scratch_root).resolve()
    output = Path(output_path).resolve()
    temporary = output.with_name(output.name + ".tmp")
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or temporary.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite trajectory-v3 dataset: {output}")
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")

    split_summary = verify_split()
    snapshots, pairs = split_codes()
    spec = DatasetSpec()
    inventories = [inventory_chain(scratch, experiment) for experiment in EXPERIMENTS]
    geometries = [
        geometry_and_forcing(inventory.as_trajectory_inventory(), spec)
        for inventory in inventories
    ]
    reference_mask = geometries[0]["wet_mask"]
    if any(
        not np.array_equal(geometry["wet_mask"], reference_mask)
        for geometry in geometries[1:]
    ):
        raise TrajectoryV3Error("regimes disagree on the wet mask")
    for name in ("longitude_deg", "latitude_deg", "distance_to_wall_normalized"):
        if any(
            not np.allclose(geometry[name], geometries[0][name])
            for geometry in geometries[1:]
        ):
            raise TrajectoryV3Error(f"regimes disagree on static field {name}")
    # The wind field is the one quantity that must differ between regimes.
    winds = [float(np.abs(geometry["wind_stress_x"]).max()) for geometry in geometries]
    if len(set(round(value, 6) for value in winds)) != 3:
        raise TrajectoryV3Error(
            f"the three regimes must carry distinct wind amplitudes, found {winds}"
        )

    temporary.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(temporary))
    group = zarr.group(store=store, overwrite=False)
    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    state = group.create_dataset(
        "state",
        shape=(len(EXPERIMENTS), PRODUCTION_DAYS, STATE_CHANNEL_COUNT, spec.ny, spec.nx),
        chunks=(1, chunk_days, STATE_CHANNEL_COUNT, spec.ny, spec.nx),
        dtype="f4",
        compressor=compressor,
    )
    static = group.create_dataset(
        "static_features",
        shape=(len(EXPERIMENTS), len(STATIC_FEATURES), spec.ny, spec.nx),
        chunks=(1, len(STATIC_FEATURES), spec.ny, spec.nx),
        dtype="f4",
        compressor=compressor,
    )
    iterations_array = group.create_dataset(
        "iteration",
        shape=(len(EXPERIMENTS), PRODUCTION_DAYS),
        chunks=(1, PRODUCTION_DAYS),
        dtype="i8",
    )
    group.create_dataset(
        "longitude_deg", data=geometries[0]["longitude_deg"], compressor=compressor
    )
    group.create_dataset(
        "latitude_deg", data=geometries[0]["latitude_deg"], compressor=compressor
    )
    group.create_dataset(
        "wet_mask", data=reference_mask.astype("u1"), compressor=compressor
    )
    group.create_dataset("snapshot_split", data=snapshots)
    group.create_dataset("pair_split", data=pairs)

    # Global per-channel wet-cell normalizers, accumulated over the v3 *training*
    # block only and pooled across regimes, matching the v1/v2 convention.  These
    # are not the pointwise anomaly statistics the model trains on; several
    # downstream readers require them to be present in the store.
    sums = np.zeros(STATE_CHANNEL_COUNT, dtype=np.float64)
    squares = np.zeros_like(sums)
    training_values = 0

    per_experiment: list[dict[str, Any]] = []
    for index, (inventory, geometry) in enumerate(zip(inventories, geometries)):
        static[index] = np.stack(
            [
                geometry[name]
                if name != "wet_mask"
                else geometry[name].astype(np.float32)
                for name in STATIC_FEATURES
            ]
        )
        iterations_array[index] = np.asarray(inventory.iterations, dtype=np.int64)
        finite = True
        for start in range(0, PRODUCTION_DAYS, chunk_days):
            stop = min(start + chunk_days, PRODUCTION_DAYS)
            block = np.empty(
                (stop - start, STATE_CHANNEL_COUNT, spec.ny, spec.nx), dtype=np.float32
            )
            for offset, record in enumerate(range(start, stop)):
                snapshot = _read_field_pair(
                    inventory.dyn_meta[record], inventory.surf_meta[record], spec
                )
                snapshot[:, ~reference_mask] = 0.0
                if not np.all(np.isfinite(snapshot)):
                    finite = False
                if TRAIN_RANGE[0] <= record < TRAIN_RANGE[1]:
                    wet_values = snapshot[:, reference_mask].astype(np.float64, copy=False)
                    sums += wet_values.sum(axis=1)
                    squares += np.square(wet_values).sum(axis=1)
                    training_values += wet_values.shape[1]
                block[offset] = snapshot
            state[index, start:stop] = block
        if not finite:
            raise TrajectoryV3Error(f"{inventory.experiment} contains non-finite state")
        per_experiment.append(
            {
                "experiment": inventory.experiment,
                "records": PRODUCTION_DAYS,
                "first_iteration": int(inventory.iterations[0]),
                "last_iteration": int(inventory.iterations[-1]),
                "segments": list(inventory.segments),
                "wind_stress_max_abs": float(
                    np.abs(geometry["wind_stress_x"]).max()
                ),
            }
        )

    if training_values == 0:
        raise TrajectoryV3Error("the v3 training block produced no wet state values")
    means = sums / training_values
    variances = np.maximum(squares / training_values - np.square(means), 0.0)
    scales = np.maximum(np.sqrt(variances), 1.0e-6)
    group.create_dataset("state_mean", data=means.astype(np.float32))
    group.create_dataset("state_scale", data=scales.astype(np.float32))

    group.attrs["version"] = DATASET_VERSION
    group.attrs["experiments"] = list(EXPERIMENTS)
    group.attrs["state_channels"] = list(STATE_CHANNELS)
    group.attrs["static_features"] = list(STATIC_FEATURES)
    group.attrs["horizon_days"] = HORIZON_DAYS
    group.attrs["production_days"] = PRODUCTION_DAYS
    group.attrs["split"] = split_summary
    group.attrs["equilibration"] = (
        "every regime equilibrated independently for 100 years from the tutorial "
        "initial condition under its own wind"
    )
    zarr.consolidate_metadata(store)
    os.replace(temporary, output)

    manifest = {
        "version": DATASET_VERSION,
        "dataset": str(output),
        "metadata_sha256": _sha256(output / ".zmetadata"),
        "split": split_summary,
        "state_mean_sha256": _array_sha256(means.astype(np.float32)),
        "state_scale_sha256": _array_sha256(scales.astype(np.float32)),
        "training_wet_values": int(training_values),
        "snapshot_codes_sha256": _array_sha256(snapshots),
        "pair_codes_sha256": _array_sha256(pairs),
        "experiments": per_experiment,
        "chunk_days": chunk_days,
        "elapsed_seconds": time.monotonic() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def validate_dataset_v3(dataset_path: str | Path) -> dict[str, Any]:
    """Independently re-verify the finished store without rebuilding it."""

    path = Path(dataset_path).resolve()
    group = zarr.open_consolidated(str(path), mode="r")
    state = group["state"]
    if tuple(state.shape) != (
        len(EXPERIMENTS),
        PRODUCTION_DAYS,
        STATE_CHANNEL_COUNT,
        62,
        62,
    ):
        raise TrajectoryV3Error(f"unexpected state shape {state.shape}")
    if tuple(group.attrs["experiments"]) != EXPERIMENTS:
        raise TrajectoryV3Error("experiment order changed")
    snapshots, pairs = split_codes()
    if not np.array_equal(np.asarray(group["snapshot_split"][:]), snapshots):
        raise TrajectoryV3Error("stored snapshot split does not match the declaration")
    if not np.array_equal(np.asarray(group["pair_split"][:]), pairs):
        raise TrajectoryV3Error("stored pair split does not match the declaration")
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    for name in ("state_mean", "state_scale"):
        if name not in set(group.array_keys()):
            raise TrajectoryV3Error(f"the store is missing {name}")
        if np.asarray(group[name]).shape != (STATE_CHANNEL_COUNT,):
            raise TrajectoryV3Error(f"{name} has the wrong shape")
    if np.any(np.asarray(group["state_scale"]) <= 0.0):
        raise TrajectoryV3Error("state_scale must be strictly positive")
    winds = [
        float(np.abs(np.asarray(group["static_features"][index, 0])).max())
        for index in range(len(EXPERIMENTS))
    ]
    if len(set(round(value, 6) for value in winds)) != 3:
        raise TrajectoryV3Error(f"regimes do not carry distinct winds: {winds}")
    checked = 0
    land_clean = True
    for index in range(len(EXPERIMENTS)):
        for record in (0, PRODUCTION_DAYS // 2, PRODUCTION_DAYS - 1):
            block = np.asarray(state[index, record], dtype=np.float32)
            if not np.all(np.isfinite(block)):
                raise TrajectoryV3Error(f"non-finite state at {index}, {record}")
            if np.any(block[:, ~wet] != 0.0):
                land_clean = False
            checked += 1
    iterations = np.asarray(group["iteration"][:])
    for index in range(len(EXPERIMENTS)):
        steps = np.diff(iterations[index])
        if not np.all(steps == MODEL_DAY_STEPS):
            raise TrajectoryV3Error(f"{EXPERIMENTS[index]} iterations are not daily")
    return {
        "status": "valid",
        "version": DATASET_VERSION,
        "dataset": str(path),
        "metadata_sha256": _sha256(path / ".zmetadata"),
        "shape": list(state.shape),
        "split": verify_split(),
        "wind_stress_max_abs": winds,
        "sampled_records": checked,
        "land_zero": land_clean,
        "wet_cells": int(wet.sum()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--scratch-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--chunk-days", type=int, default=8)
    check = commands.add_parser("validate")
    check.add_argument("--dataset", type=Path, required=True)
    commands.add_parser("split")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build_dataset_v3(
            args.scratch_root, args.output, chunk_days=args.chunk_days
        )
    elif args.command == "validate":
        result = validate_dataset_v3(args.dataset)
    else:
        result = verify_split()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
