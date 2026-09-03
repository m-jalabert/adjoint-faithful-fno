"""Build the 0.25-degree turbulent trajectory store for the AF--FNO emulator.

The high-resolution counterpart of ``trajectories_v3.zarr``.  Three regimes,
each independently equilibrated for 100 years from the tutorial initial
condition under its own wind and then run 25 further years of daily output::

    regime     tau0 (N m-2)   grid       records
    S0_turb    0.100          248 x 248  9,000 (years 100--125)
    S1_turb    0.075          248 x 248  9,000
    S2_turb    0.125          248 x 248  9,000

The day count per regime is *identical* to the 1-degree store, so the Bire
protocol split, the 17,820 training sequences and every lead in the evaluation
suite carry over unchanged.  Only the grid changes.

Two settings differ from the 1-degree builder and both are forced by the
resolution:

``MODEL_DAY_STEPS``
    86400/300 = 288 rather than 86400/1200 = 72.  The chain contiguity check
    compares consecutive ``dynState`` iteration numbers against it, so the wrong
    value rejects every turbulent chain.

``chunk_days = 1``
    Each training sample reads seven *non-adjacent* day slices (t, t+10 ... t+60),
    so an eight-day chunk decompresses roughly eight times what it uses.  At
    1 degree that was invisible because the whole 13 GB store lives in page
    cache; a 207 GB store does not, and the amplification would put ~5 GB of
    decompression on every optimizer step instead of ~630 MB.

This module is deliberately standalone --- the ``bire_repro`` package it
descends from is no longer installed --- so it depends on nothing but the
standard library, NumPy, SciPy, Zarr and numcodecs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numcodecs
import numpy as np
import zarr
from scipy.ndimage import distance_transform_edt

VERSION = "trajectories_turb_v1"
ROOT_NAME = "mitgcm_turb_v1"

EXPERIMENTS = ("S0_turb", "S1_turb", "S2_turb")
TAU0_N_M2 = {"S0_turb": 0.100, "S1_turb": 0.075, "S2_turb": 0.125}

#: 0.25 degree, 15 levels.  deltaT = 300 s, so a model day is 288 steps.
NY = NX = 248
NR = 15
MODEL_DAY_STEPS = 288
DEGREES_PER_CELL = 0.25

STATE_CHANNELS = tuple(
    [f"U_{level:02d}" for level in range(1, NR + 1)]
    + [f"V_{level:02d}" for level in range(1, NR + 1)]
    + [f"Theta_{level:02d}" for level in range(1, NR + 1)]
    + ["Eta"]
)
STATE_CHANNEL_COUNT = len(STATE_CHANNELS)

STATIC_FEATURES = (
    "wind_stress_x",
    "longitude_normalized",
    "latitude_normalized",
    "wet_mask",
    "distance_to_wall_normalized",
)

PRODUCTION_DAYS = 9000
HORIZON_DAYS = 10

#: The store's own chronological split with the project's 90-day buffers.  Held
#: identical to the 1-degree store so the protocol split that reads it is
#: unchanged; the Bire protocol split is applied in memory downstream.
TRAIN_RANGE = (0, 5040)
VALIDATION_RANGE = (5130, 6390)
TEST_RANGE = (6480, 9000)
BUFFER_RANGES = ((5040, 5130), (6390, 6480))
TRAIN_CODE, VALIDATION_CODE, TEST_CODE, BUFFER_CODE = 1, 2, 3, 0

MAXIMUM_TEST_ROLLOUT_DAYS = 2000
TEST_START_WINDOW = (TEST_RANGE[0], TEST_RANGE[1] - MAXIMUM_TEST_ROLLOUT_DAYS)

#: Ordered production chains: 3,600 + 3,600 + 1,800 = 9,000 daily records.
CHAINS: dict[str, tuple[tuple[str, int], ...]] = {
    experiment: (
        (f"{experiment}/production/years_100_110", 3600),
        (f"{experiment}/production/years_110_120", 3600),
        (f"{experiment}/production/years_120_125", 1800),
    )
    for experiment in EXPERIMENTS
}


class TurbDatasetError(RuntimeError):
    """Raised when the turbulent dataset build violates its contract."""


# --------------------------------------------------------------------------
# MITgcm MDS readers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MDSMeta:
    dimensions: tuple[int, ...]
    nrecords: int
    dtype: np.dtype
    fields: tuple[str, ...]
    timestep: int | None


def parse_mds_meta(path: str | Path) -> MDSMeta:
    metadata_path = Path(path)
    text = metadata_path.read_text()
    dim_match = re.search(r"dimList\s*=\s*\[(.*?)\];", text, re.DOTALL)
    if not dim_match:
        raise ValueError(f"missing dimList in {metadata_path}")
    dim_values = [int(value) for value in re.findall(r"[-+]?\d+", dim_match.group(1))]
    if not dim_values or len(dim_values) % 3:
        raise ValueError(f"invalid dimList in {metadata_path}")
    dimensions = tuple(dim_values[index] for index in range(0, len(dim_values), 3))
    record_match = re.search(r"nrecords\s*=\s*\[\s*(\d+)\s*\]", text)
    nrecords = int(record_match.group(1)) if record_match else 1
    precision_match = re.search(r"dataprec\s*=\s*\[\s*'([^']+)'", text)
    precision = precision_match.group(1).strip().lower() if precision_match else "float32"
    precision_map = {
        "float32": np.dtype(">f4"),
        "real*4": np.dtype(">f4"),
        "float64": np.dtype(">f8"),
        "real*8": np.dtype(">f8"),
    }
    if precision not in precision_map:
        raise ValueError(f"unsupported MDS precision {precision!r}")
    fields_match = re.search(r"fldList\s*=\s*\{(.*?)\};", text, re.DOTALL)
    fields = (
        tuple(value.strip() for value in re.findall(r"'([^']+)'", fields_match.group(1)))
        if fields_match
        else ()
    )
    timestep_match = re.search(r"timeStepNumber\s*=\s*\[\s*(\d+)\s*\]", text)
    timestep = int(timestep_match.group(1)) if timestep_match else None
    return MDSMeta(dimensions, nrecords, precision_map[precision], fields, timestep)


def read_mds(path: str | Path) -> tuple[MDSMeta, np.ndarray]:
    metadata_path = Path(path)
    if metadata_path.suffix == ".data":
        metadata_path = metadata_path.with_suffix(".meta")
    meta = parse_mds_meta(metadata_path)
    data_path = metadata_path.with_suffix(".data")
    count = meta.nrecords * math.prod(meta.dimensions)
    values = np.fromfile(data_path, dtype=meta.dtype, count=count)
    if values.size != count:
        raise ValueError(
            f"MDS size mismatch for {data_path}: expected {count}, got {values.size}"
        )
    return meta, values.reshape((meta.nrecords, *reversed(meta.dimensions)))


def mds_fields(meta: MDSMeta, values: np.ndarray) -> dict[str, np.ndarray]:
    if not meta.fields:
        raise ValueError("MDS metadata contains no fldList")
    if meta.nrecords % len(meta.fields):
        raise ValueError("MDS record count is not divisible by fldList length")
    records_per_field = meta.nrecords // len(meta.fields)
    fields: dict[str, np.ndarray] = {}
    for index, name in enumerate(meta.fields):
        value = values[index * records_per_field : (index + 1) * records_per_field]
        fields[name] = value[0] if records_per_field == 1 else value
    return fields


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _iteration(path: Path, prefix: str) -> int:
    name = path.name
    if not name.startswith(f"{prefix}.") or not name.endswith(".meta"):
        raise ValueError(f"unexpected {prefix} file name: {name}")
    return int(name[len(prefix) + 1 : -len(".meta")])


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------


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
        raise TurbDatasetError("splits are not strictly ordered")
    for start, stop in BUFFER_RANGES:
        if np.any(snapshots[start:stop] != BUFFER_CODE):
            raise TurbDatasetError("a declared buffer is not empty")
    if (
        validation.min() - train.max() - 1 != 90
        or test.min() - validation.max() - 1 != 90
    ):
        raise TurbDatasetError("buffers are not the project's 90 days")
    if int(np.flatnonzero(pairs == TRAIN_CODE).max()) + HORIZON_DAYS >= VALIDATION_RANGE[0]:
        raise TurbDatasetError("a training pair reaches into validation")
    if int(np.flatnonzero(pairs == VALIDATION_CODE).max()) + HORIZON_DAYS >= TEST_RANGE[0]:
        raise TurbDatasetError("a validation pair reaches into test")
    if TEST_START_WINDOW[1] - 1 + MAXIMUM_TEST_ROLLOUT_DAYS > int(test.max()):
        raise TurbDatasetError("the declared test start window overruns the test block")
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
        "pooled": "training and validation blocks are pooled across the three regimes",
    }


# --------------------------------------------------------------------------
# inventory and geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainInventory:
    experiment: str
    dyn_meta: tuple[Path, ...]
    surf_meta: tuple[Path, ...]
    iterations: tuple[int, ...]
    segments: tuple[dict[str, Any], ...]
    production_root: Path


def inventory_chain(scratch_root: Path, experiment: str) -> ChainInventory:
    """Assemble and verify one regime's contiguous daily production chain."""

    dyn: list[Path] = []
    segments: list[dict[str, Any]] = []
    for relative, take in CHAINS[experiment]:
        directory = (Path(scratch_root) / relative).resolve()
        result_path = directory / "segment_result.json"
        if not result_path.is_file():
            raise TurbDatasetError(f"missing segment result: {result_path}")
        result = json.loads(result_path.read_text())
        if int(result.get("returncode", 1)) != 0:
            raise TurbDatasetError(f"segment did not complete cleanly: {result_path}")
        found = sorted(
            directory.glob("dynState.*.meta"),
            key=lambda path: _iteration(path, "dynState"),
        )
        if len(found) < take:
            raise TurbDatasetError(
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
                "tau0_n_m2": result.get("tau0_n_m2"),
            }
        )
    if len(dyn) != PRODUCTION_DAYS:
        raise TurbDatasetError(
            f"{experiment} chain has {len(dyn)} records; expected {PRODUCTION_DAYS}"
        )
    iterations = tuple(_iteration(path, "dynState") for path in dyn)
    gaps = [
        (before, after)
        for before, after in zip(iterations, iterations[1:])
        if after - before != MODEL_DAY_STEPS
    ]
    if gaps:
        raise TurbDatasetError(
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
        raise TurbDatasetError(f"{experiment} is missing surfState pairs: {missing[:3]}")
    return ChainInventory(
        experiment, tuple(dyn), surf, iterations, tuple(segments),
        Path(segments[0]["run_dir"]),
    )


def _scalar_mds(path: Path) -> np.ndarray:
    _, values = read_mds(path)
    value = np.asarray(values)
    if value.shape != (1, NY, NX):
        raise TurbDatasetError(f"unexpected static MDS shape at {path}: {value.shape}")
    return value[0].astype(np.float32, copy=False)


def geometry_and_forcing(inventory: ChainInventory) -> dict[str, np.ndarray]:
    """Read the static grid and forcing products from one production segment."""

    root = inventory.production_root
    depth = _scalar_mds(root / "Depth.meta")
    longitude = _scalar_mds(root / "XC.meta")
    latitude = _scalar_mds(root / "YC.meta")
    wet = depth > 0.0
    if not np.any(wet) or np.all(wet):
        raise TurbDatasetError(f"invalid wet mask in {root / 'Depth.meta'}")
    wall_distance = distance_transform_edt(wet).astype(np.float32)
    wall_distance /= float(wall_distance[wet].max())
    longitude_normalized = (
        2.0 * (longitude - longitude.min()) / (longitude.max() - longitude.min()) - 1.0
    )
    latitude_normalized = (
        2.0 * (latitude - latitude.min()) / (latitude.max() - latitude.min()) - 1.0
    )
    wind_file = root / "windx_cosy.bin"
    wind_raw = np.fromfile(wind_file, dtype=">f4")
    if wind_raw.size != NY * NX:
        raise TurbDatasetError(f"unexpected wind forcing size at {wind_file}")
    # MITgcm writes the x index fastest; a C-order (y, x) reshape reproduces it.
    wind = wind_raw.reshape((NY, NX)).astype(np.float32, copy=False)
    return {
        "longitude_deg": longitude,
        "latitude_deg": latitude,
        "wet_mask": wet,
        "wind_stress_x": wind,
        "longitude_normalized": longitude_normalized.astype(np.float32),
        "latitude_normalized": latitude_normalized.astype(np.float32),
        "distance_to_wall_normalized": wall_distance,
    }


def read_field_pair(dyn_meta: Path, surf_meta: Path) -> np.ndarray:
    """Return one 46-channel tracer-grid snapshot from a dyn/surf MDS pair."""

    dyn_info, dyn_values = read_mds(dyn_meta)
    fields = mds_fields(dyn_info, dyn_values)
    if set(fields) != {"UVEL", "VVEL", "THETA"}:
        raise TurbDatasetError(f"unexpected dynState fields at {dyn_meta}")
    expected_3d = (NR, NY, NX)
    if any(np.asarray(fields[name]).shape != expected_3d for name in fields):
        raise TurbDatasetError(f"unexpected dynState shape at {dyn_meta}")
    surf_info, surf_values = read_mds(surf_meta)
    surface = mds_fields(surf_info, surf_values)
    if set(surface) != {"ETAN"} or np.asarray(surface["ETAN"]).shape != (NY, NX):
        raise TurbDatasetError(f"unexpected surfState layout at {surf_meta}")
    # MITgcm U/V live on west/south faces; this is the same fixed centering
    # operator the 1-degree store uses.
    u = np.asarray(fields["UVEL"], dtype=np.float32)
    v = np.asarray(fields["VVEL"], dtype=np.float32)
    theta = np.asarray(fields["THETA"], dtype=np.float32)
    u_center = 0.5 * (u + np.roll(u, -1, axis=-1))
    v_center = 0.5 * (v + np.roll(v, -1, axis=-2))
    eta = np.asarray(surface["ETAN"], dtype=np.float32)[None]
    return np.concatenate((u_center, v_center, theta, eta), axis=0)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def build(
    scratch_root: str | Path,
    output_path: str | Path,
    *,
    chunk_days: int = 1,
) -> dict[str, Any]:
    """Build the immutable turbulent trajectory store."""

    started = time.monotonic()
    scratch = Path(scratch_root).resolve()
    output = Path(output_path).resolve()
    temporary = output.with_name(output.name + ".tmp")
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or temporary.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite turbulent dataset: {output}")
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")

    split_summary = verify_split()
    snapshots, pairs = split_codes()
    inventories = [inventory_chain(scratch, experiment) for experiment in EXPERIMENTS]
    geometries = [geometry_and_forcing(inventory) for inventory in inventories]

    reference_mask = geometries[0]["wet_mask"]
    if any(
        not np.array_equal(geometry["wet_mask"], reference_mask)
        for geometry in geometries[1:]
    ):
        raise TurbDatasetError("regimes disagree on the wet mask")
    for name in ("longitude_deg", "latitude_deg", "distance_to_wall_normalized"):
        if any(
            not np.allclose(geometry[name], geometries[0][name])
            for geometry in geometries[1:]
        ):
            raise TurbDatasetError(f"regimes disagree on static field {name}")
    winds = [float(np.abs(geometry["wind_stress_x"]).max()) for geometry in geometries]
    if len(set(round(value, 6) for value in winds)) != 3:
        raise TurbDatasetError(
            f"the three regimes must carry distinct wind amplitudes, found {winds}"
        )

    temporary.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(temporary))
    group = zarr.group(store=store, overwrite=False)
    compressor = numcodecs.Blosc(
        cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
    )
    state = group.create_dataset(
        "state",
        shape=(len(EXPERIMENTS), PRODUCTION_DAYS, STATE_CHANNEL_COUNT, NY, NX),
        chunks=(1, chunk_days, STATE_CHANNEL_COUNT, NY, NX),
        dtype="f4",
        compressor=compressor,
    )
    static = group.create_dataset(
        "static_features",
        shape=(len(EXPERIMENTS), len(STATIC_FEATURES), NY, NX),
        chunks=(1, len(STATIC_FEATURES), NY, NX),
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

    # Global per-channel wet-cell normalizers over the store's own training
    # block, pooled across regimes.  These are not the pointwise statistics the
    # model trains on -- those are recomputed by the training entry point over
    # the protocol's training days -- but downstream readers require them.
    sums = np.zeros(STATE_CHANNEL_COUNT, dtype=np.float64)
    squares = np.zeros_like(sums)
    training_values = 0
    write_block = max(chunk_days, 8)

    per_experiment: list[dict[str, Any]] = []
    for index, (inventory, geometry) in enumerate(zip(inventories, geometries)):
        static[index] = np.stack(
            [
                geometry[name].astype(np.float32)
                for name in STATIC_FEATURES
            ]
        )
        iterations_array[index] = np.asarray(inventory.iterations, dtype=np.int64)
        finite = True
        for start in range(0, PRODUCTION_DAYS, write_block):
            stop = min(start + write_block, PRODUCTION_DAYS)
            block = np.empty(
                (stop - start, STATE_CHANNEL_COUNT, NY, NX), dtype=np.float32
            )
            for offset, record in enumerate(range(start, stop)):
                snapshot = read_field_pair(
                    inventory.dyn_meta[record], inventory.surf_meta[record]
                )
                snapshot[:, ~reference_mask] = 0.0
                if not np.all(np.isfinite(snapshot)):
                    finite = False
                if TRAIN_RANGE[0] <= record < TRAIN_RANGE[1]:
                    wet_values = snapshot[:, reference_mask].astype(
                        np.float64, copy=False
                    )
                    sums += wet_values.sum(axis=1)
                    squares += np.square(wet_values).sum(axis=1)
                    training_values += wet_values.shape[1]
                block[offset] = snapshot
            state[index, start:stop] = block
        if not finite:
            raise TurbDatasetError(f"{inventory.experiment} contains non-finite state")
        per_experiment.append(
            {
                "experiment": inventory.experiment,
                "records": PRODUCTION_DAYS,
                "first_iteration": int(inventory.iterations[0]),
                "last_iteration": int(inventory.iterations[-1]),
                "segments": list(inventory.segments),
                "tau0_n_m2": TAU0_N_M2[inventory.experiment],
                "wind_stress_max_abs": float(np.abs(geometry["wind_stress_x"]).max()),
            }
        )
        print(
            f"[{time.strftime('%H:%M:%S')}] {inventory.experiment} written "
            f"({time.monotonic() - started:.0f} s elapsed)",
            flush=True,
        )

    if training_values == 0:
        raise TurbDatasetError("the training block produced no wet state values")
    means = sums / training_values
    variances = np.maximum(squares / training_values - np.square(means), 0.0)
    scales = np.maximum(np.sqrt(variances), 1.0e-6)
    group.create_dataset("state_mean", data=means.astype(np.float32))
    group.create_dataset("state_scale", data=scales.astype(np.float32))

    group.attrs["version"] = VERSION
    group.attrs["experiments"] = list(EXPERIMENTS)
    group.attrs["state_channels"] = list(STATE_CHANNELS)
    group.attrs["static_features"] = list(STATIC_FEATURES)
    group.attrs["horizon_days"] = HORIZON_DAYS
    group.attrs["production_days"] = PRODUCTION_DAYS
    group.attrs["grid_shape"] = [NY, NX]
    group.attrs["degrees_per_cell"] = DEGREES_PER_CELL
    group.attrs["model_day_steps"] = MODEL_DAY_STEPS
    group.attrs["split"] = split_summary
    group.attrs["equilibration"] = (
        "every regime equilibrated independently for 100 years from the tutorial "
        "initial condition under its own wind"
    )
    zarr.consolidate_metadata(store)
    os.replace(temporary, output)

    manifest = {
        "version": VERSION,
        "dataset": str(output),
        "grid_shape": [NY, NX],
        "degrees_per_cell": DEGREES_PER_CELL,
        "model_day_steps": MODEL_DAY_STEPS,
        "metadata_sha256": _sha256(output / ".zmetadata"),
        "split": split_summary,
        "state_mean_sha256": _array_sha256(means.astype(np.float32)),
        "state_scale_sha256": _array_sha256(scales.astype(np.float32)),
        "training_wet_values": int(training_values),
        "snapshot_codes_sha256": _array_sha256(snapshots),
        "pair_codes_sha256": _array_sha256(pairs),
        "wet_cells": int(reference_mask.sum()),
        "experiments": per_experiment,
        "chunk_days": chunk_days,
        "elapsed_seconds": time.monotonic() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def validate(dataset_path: str | Path) -> dict[str, Any]:
    """Independently re-verify the finished store without rebuilding it."""

    path = Path(dataset_path).resolve()
    group = zarr.open_consolidated(zarr.DirectoryStore(str(path)), mode="r")
    if group.attrs.get("version") != VERSION:
        raise TurbDatasetError("store version attribute is wrong")
    if tuple(group["state"].shape) != (
        len(EXPERIMENTS), PRODUCTION_DAYS, STATE_CHANNEL_COUNT, NY, NX
    ):
        raise TurbDatasetError(f"unexpected state shape {group['state'].shape}")
    if tuple(group["static_features"].shape) != (
        len(EXPERIMENTS), len(STATIC_FEATURES), NY, NX
    ):
        raise TurbDatasetError("unexpected static_features shape")
    if tuple(group.attrs.get("static_features", ())) != STATIC_FEATURES:
        raise TurbDatasetError("static feature order changed")
    iterations = np.asarray(group["iteration"][:])
    if np.any(np.diff(iterations, axis=1) != MODEL_DAY_STEPS):
        raise TurbDatasetError("stored iterations are not contiguous daily output")
    snapshots, pairs = split_codes()
    if not np.array_equal(np.asarray(group["snapshot_split"][:]), snapshots):
        raise TurbDatasetError("stored snapshot codes do not match the declared split")
    if not np.array_equal(np.asarray(group["pair_split"][:]), pairs):
        raise TurbDatasetError("stored pair codes do not match the declared split")
    wet = np.asarray(group["wet_mask"][:]).astype(bool)
    # Spot-check a handful of days per regime rather than the whole 285 GB.
    rng = np.random.default_rng(0)
    checked = []
    for experiment in range(len(EXPERIMENTS)):
        for day in rng.integers(0, PRODUCTION_DAYS, 8):
            block = np.asarray(group["state"][experiment, int(day)])
            if not np.all(np.isfinite(block)):
                raise TurbDatasetError(f"non-finite state at {experiment}, {day}")
            if np.any(block[:, ~wet] != 0.0):
                raise TurbDatasetError(f"land cells are not zero at {experiment}, {day}")
            checked.append([int(experiment), int(day)])
    return {
        "version": VERSION,
        "dataset": str(path),
        "state_shape": list(group["state"].shape),
        "wet_cells": int(wet.sum()),
        "spot_checked": checked,
        "metadata_sha256": _sha256(path / ".zmetadata"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--scratch-root", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--chunk-days", type=int, default=1)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--dataset", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build(args.scratch_root, args.output, chunk_days=args.chunk_days)
    else:
        result = validate(args.dataset)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
