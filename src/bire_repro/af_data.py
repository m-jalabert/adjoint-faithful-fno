"""Build the shared AF--FNO trajectory dataset from S0--S2 MITgcm MDS output.

The conversion is deliberately independent of the retired 0.25-degree Bire
workflow.  It reads only the active 1-degree tutorial productions, places U/V
on the tracer grid with declared linear operators, stores a 46-channel Markov
state, and seals the chronological split and training-only normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import zarr
from numcodecs import Blosc
from scipy.ndimage import distance_transform_edt

from .mds import mds_fields, read_mds


EXPERIMENTS = ("S0", "S1", "S2")
WIND_STRESS_N_M2 = {"S0": 0.100, "S1": 0.075, "S2": 0.125}
MODEL_DAY_STEPS = 72


def state_channels(nr: int) -> tuple[str, ...]:
    """Names for x=[U_1:Nr,V_1:Nr,Theta_1:Nr,eta]."""

    return tuple(
        [f"U_{level:02d}" for level in range(1, nr + 1)]
        + [f"V_{level:02d}" for level in range(1, nr + 1)]
        + [f"Theta_{level:02d}" for level in range(1, nr + 1)]
        + ["Eta"]
    )


def state_units(nr: int) -> list[str]:
    return ["m s-1"] * (2 * nr) + ["degC"] * nr + ["m"]


STATE_CHANNELS = state_channels(15)
STATIC_FEATURES = (
    "wind_stress_x",
    "longitude_normalized",
    "latitude_normalized",
    "wet_mask",
    "distance_to_wall_normalized",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Fixed AF--FNO dataset dimensions and chronological split contract."""

    expected_records: int = 3600
    nr: int = 15
    ny: int = 62
    nx: int = 62
    horizon_days: int = 10
    train_stop: int = 2520
    validation_start: int = 2610
    validation_stop: int = 2880
    inference_start: int = 2970

    def __post_init__(self) -> None:
        if min(self.expected_records, self.nr, self.ny, self.nx, self.horizon_days) <= 0:
            raise ValueError("dataset dimensions and horizon must be positive")
        if not (0 < self.train_stop < self.validation_start < self.validation_stop < self.inference_start):
            raise ValueError("chronological split boundaries must be strictly ordered")
        if self.inference_start + self.horizon_days >= self.expected_records:
            raise ValueError("inference split must contain at least one complete pair")


@dataclass(frozen=True)
class TrajectoryInventory:
    """Ordered raw diagnostic pairs and exact iteration evidence for one regime."""

    experiment: str
    dyn_meta: tuple[Path, ...]
    surf_meta: tuple[Path, ...]
    iterations: tuple[int, ...]
    production_root: Path

    def summary(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "records": len(self.dyn_meta),
            "first_iteration": self.iterations[0],
            "last_iteration": self.iterations[-1],
            "production_root": str(self.production_root),
            "first_dyn_meta": str(self.dyn_meta[0]),
            "last_dyn_meta": str(self.dyn_meta[-1]),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _iteration(path: Path, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}\.(\d+)\.meta", path.name)
    if not match:
        raise ValueError(f"unexpected {prefix} metadata name: {path.name}")
    return int(match.group(1))


def _production_root(scratch_root: Path, experiment: str) -> Path:
    candidates = sorted((scratch_root / "mitgcm" / experiment).glob("production/years_*"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one production directory for {experiment}, found {candidates}"
        )
    return candidates[0]


def inventory_trajectory(
    scratch_root: str | Path,
    experiment: str,
    spec: DatasetSpec = DatasetSpec(),
) -> TrajectoryInventory:
    """Verify one completed production trajectory before reading any state data."""

    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}; choose from {EXPERIMENTS}")
    production_root = _production_root(Path(scratch_root), experiment)
    dyn_meta = tuple(sorted(production_root.glob("dynState.*.meta"), key=lambda path: _iteration(path, "dynState")))
    if len(dyn_meta) != spec.expected_records:
        raise ValueError(
            f"{experiment} has {len(dyn_meta)} dynState records; expected {spec.expected_records}"
        )
    iterations = tuple(_iteration(path, "dynState") for path in dyn_meta)
    if any(next_value - value != MODEL_DAY_STEPS for value, next_value in zip(iterations, iterations[1:])):
        raise ValueError(f"{experiment} dynState iterations are not contiguous daily output")

    surf_meta = tuple(path.with_name(f"surfState.{iteration:010d}.meta") for path, iteration in zip(dyn_meta, iterations))
    missing = [str(path) for path in surf_meta if not path.is_file() or not path.with_suffix(".data").is_file()]
    if missing:
        raise FileNotFoundError(f"{experiment} is missing paired surfState files: {missing[:3]}")
    missing_dyn_data = [str(path.with_suffix(".data")) for path in dyn_meta if not path.with_suffix(".data").is_file()]
    if missing_dyn_data:
        raise FileNotFoundError(f"{experiment} is missing dynState data: {missing_dyn_data[:3]}")
    return TrajectoryInventory(experiment, dyn_meta, surf_meta, iterations, production_root)


def _read_field_pair(dyn_meta: Path, surf_meta: Path, spec: DatasetSpec) -> np.ndarray:
    dyn_info, dyn_values = read_mds(dyn_meta)
    fields = mds_fields(dyn_info, dyn_values)
    if set(fields) != {"UVEL", "VVEL", "THETA"}:
        raise ValueError(f"unexpected dynState fields at {dyn_meta}: {tuple(fields)}")
    expected_3d = (spec.nr, spec.ny, spec.nx)
    if any(np.asarray(fields[name]).shape != expected_3d for name in fields):
        raise ValueError(f"unexpected dynState shape at {dyn_meta}")
    surf_info, surf_values = read_mds(surf_meta)
    surface = mds_fields(surf_info, surf_values)
    if set(surface) != {"ETAN"} or np.asarray(surface["ETAN"]).shape != (spec.ny, spec.nx):
        raise ValueError(f"unexpected surfState layout at {surf_meta}")

    # MITgcm U/V live on west/south faces.  The forward centering is a fixed,
    # documented operator; adjoint comparisons must use its exact transpose.
    u = np.asarray(fields["UVEL"], dtype=np.float32)
    v = np.asarray(fields["VVEL"], dtype=np.float32)
    theta = np.asarray(fields["THETA"], dtype=np.float32)
    u_center = 0.5 * (u + np.roll(u, -1, axis=-1))
    v_center = 0.5 * (v + np.roll(v, -1, axis=-2))
    return np.concatenate((u_center, v_center, theta, np.asarray(surface["ETAN"], dtype=np.float32)[None]), axis=0)


def _scalar_mds(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    _, values = read_mds(path)
    value = np.asarray(values)
    if value.shape != (1, *expected_shape):
        raise ValueError(f"unexpected static MDS shape at {path}: {value.shape}")
    return value[0].astype(np.float32, copy=False)


def geometry_and_forcing(inventory: TrajectoryInventory, spec: DatasetSpec) -> dict[str, np.ndarray]:
    """Read exact static grid/forcing products from one immutable production segment."""

    root = inventory.production_root
    expected_shape = (spec.ny, spec.nx)
    depth = _scalar_mds(root / "Depth.meta", expected_shape)
    longitude = _scalar_mds(root / "XC.meta", expected_shape)
    latitude = _scalar_mds(root / "YC.meta", expected_shape)
    wet = depth > 0.0
    if not np.any(wet) or np.all(wet):
        raise ValueError(f"invalid wet mask in {root / 'Depth.meta'}")
    wall_distance = distance_transform_edt(wet).astype(np.float32)
    wall_distance /= float(wall_distance[wet].max())
    longitude_normalized = 2.0 * (longitude - longitude.min()) / (longitude.max() - longitude.min()) - 1.0
    latitude_normalized = 2.0 * (latitude - latitude.min()) / (latitude.max() - latitude.min()) - 1.0

    wind_file = root / "windx_cosy.bin"
    wind_raw = np.fromfile(wind_file, dtype=">f4")
    if wind_raw.size != spec.ny * spec.nx:
        raise ValueError(f"unexpected wind forcing size at {wind_file}")
    # MITgcm writes the x index fastest; C-order (y, x) reshape reproduces that layout.
    wind = wind_raw.reshape(expected_shape).astype(np.float32, copy=False)
    return {
        "longitude_deg": longitude,
        "latitude_deg": latitude,
        "wet_mask": wet,
        "wind_stress_x": wind,
        "longitude_normalized": longitude_normalized.astype(np.float32),
        "latitude_normalized": latitude_normalized.astype(np.float32),
        "distance_to_wall_normalized": wall_distance,
    }


def split_manifest(spec: DatasetSpec) -> dict[str, Any]:
    """Return sealed snapshot/pair ranges with 90-day buffers at split boundaries."""

    ranges = {
        "train": (0, spec.train_stop),
        "validation": (spec.validation_start, spec.validation_stop),
        "inference": (spec.inference_start, spec.expected_records),
    }
    snapshots = np.zeros(spec.expected_records, dtype=np.uint8)
    pairs = np.zeros(spec.expected_records, dtype=np.uint8)
    labels = {"train": 1, "validation": 2, "inference": 3}
    payload: dict[str, Any] = {
        "horizon_days": spec.horizon_days,
        "excluded_ranges": [
            [spec.train_stop, spec.validation_start],
            [spec.validation_stop, spec.inference_start],
        ],
        "splits": {},
    }
    for name, (start, stop) in ranges.items():
        snapshots[start:stop] = labels[name]
        pair_stop = stop - spec.horizon_days
        if pair_stop > start:
            pairs[start:pair_stop] = labels[name]
        payload["splits"][name] = {
            "snapshot_start": start,
            "snapshot_stop": stop,
            "pair_start": start,
            "pair_stop": max(start, pair_stop),
            "pair_count_per_trajectory": max(0, pair_stop - start),
        }
    payload["snapshot_codes"] = snapshots.tolist()
    payload["pair_codes"] = pairs.tolist()
    return payload


def _training_accumulate(
    value: np.ndarray,
    wet: np.ndarray,
    sums: np.ndarray,
    squares: np.ndarray,
) -> int:
    wet_values = value[:, wet].astype(np.float64, copy=False)
    sums += wet_values.sum(axis=1)
    squares += np.square(wet_values).sum(axis=1)
    return wet_values.shape[1]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_dataset(
    scratch_root: str | Path,
    output: str | Path,
    *,
    spec: DatasetSpec = DatasetSpec(),
    experiments: Iterable[str] = EXPERIMENTS,
    chunk_days: int = 8,
) -> dict[str, Any]:
    """Convert verified S0--S2 daily MDS products into a sealed AF--FNO Zarr store."""

    scratch = Path(scratch_root).resolve()
    destination = Path(output).resolve()
    selected = tuple(experiments)
    channels = state_channels(spec.nr)
    if selected != EXPERIMENTS:
        raise ValueError(f"the shared dataset must contain exactly {EXPERIMENTS}")
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")
    if destination.exists():
        raise FileExistsError(f"dataset already exists: {destination}")
    manifest_path = destination.with_suffix(".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(f"dataset manifest already exists: {manifest_path}")

    inventories = [inventory_trajectory(scratch, experiment, spec) for experiment in selected]
    geometries = [geometry_and_forcing(inventory, spec) for inventory in inventories]
    reference_mask = geometries[0]["wet_mask"]
    if any(not np.array_equal(geometry["wet_mask"], reference_mask) for geometry in geometries[1:]):
        raise ValueError("S0--S2 wet masks differ")
    for name in ("longitude_deg", "latitude_deg"):
        if any(not np.allclose(geometry[name], geometries[0][name]) for geometry in geometries[1:]):
            raise ValueError(f"S0--S2 {name} grids differ")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"incomplete temporary dataset exists: {temporary}")
    store = zarr.DirectoryStore(str(temporary))
    group = zarr.group(store=store, overwrite=False)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    state = group.create_dataset(
        "state",
        shape=(len(selected), spec.expected_records, len(channels), spec.ny, spec.nx),
        chunks=(1, chunk_days, len(channels), spec.ny, spec.nx),
        dtype="f4",
        compressor=compressor,
    )
    static = group.create_dataset(
        "static_features",
        shape=(len(selected), len(STATIC_FEATURES), spec.ny, spec.nx),
        chunks=(1, len(STATIC_FEATURES), spec.ny, spec.nx),
        dtype="f4",
        compressor=compressor,
    )
    iterations = group.create_dataset(
        "iteration", shape=(len(selected), spec.expected_records), chunks=(1, spec.expected_records), dtype="i8"
    )
    group.create_dataset("longitude_deg", data=geometries[0]["longitude_deg"], compressor=compressor)
    group.create_dataset("latitude_deg", data=geometries[0]["latitude_deg"], compressor=compressor)
    group.create_dataset("wet_mask", data=reference_mask.astype("u1"), compressor=compressor)
    split = split_manifest(spec)
    group.create_dataset("snapshot_split", data=np.asarray(split["snapshot_codes"], dtype="u1"))
    group.create_dataset("pair_split", data=np.asarray(split["pair_codes"], dtype="u1"))

    sums = np.zeros(len(channels), dtype=np.float64)
    squares = np.zeros(len(channels), dtype=np.float64)
    count = 0
    finite = True
    max_abs = 0.0
    for experiment_index, (inventory, geometry) in enumerate(zip(inventories, geometries)):
        static[experiment_index] = np.stack(
            [geometry[name] if name != "wet_mask" else geometry[name].astype(np.float32) for name in STATIC_FEATURES]
        )
        iterations[experiment_index] = np.asarray(inventory.iterations, dtype=np.int64)
        for start in range(0, spec.expected_records, chunk_days):
            stop = min(start + chunk_days, spec.expected_records)
            block = np.empty((stop - start, len(channels), spec.ny, spec.nx), dtype=np.float32)
            for offset, record_index in enumerate(range(start, stop)):
                snapshot = _read_field_pair(inventory.dyn_meta[record_index], inventory.surf_meta[record_index], spec)
                snapshot[:, ~reference_mask] = 0.0
                if not np.all(np.isfinite(snapshot)):
                    finite = False
                max_abs = max(max_abs, float(np.max(np.abs(snapshot))))
                if record_index < spec.train_stop:
                    count += _training_accumulate(snapshot, reference_mask, sums, squares)
                block[offset] = snapshot
            state[experiment_index, start:stop] = block

    if not finite:
        raise ValueError("non-finite values found while converting raw trajectories")
    if count == 0:
        raise ValueError("training split has no wet state values")
    means = sums / count
    variances = np.maximum(squares / count - np.square(means), 0.0)
    scales = np.maximum(np.sqrt(variances), 1.0e-6)
    group.create_dataset("state_mean", data=means.astype(np.float32))
    group.create_dataset("state_scale", data=scales.astype(np.float32))
    group.attrs.update(
        {
            "schema_version": 1,
            "project": "adjoint-faithful-fno",
            "experiments": list(selected),
            "wind_stress_n_m2": [WIND_STRESS_N_M2[name] for name in selected],
            "state_channels": list(channels),
            "static_features": list(STATIC_FEATURES),
            "state_units": state_units(spec.nr),
            "u_centering": "0.5*(U(i,j)+U(i+1,j)) on the MITgcm C grid",
            "v_centering": "0.5*(V(i,j)+V(i,j+1)) on the MITgcm C grid",
            "land_value": 0.0,
            "normalization": "wet cells from S0-S2 training snapshots only",
            "split_labels": {"0": "excluded", "1": "train", "2": "validation", "3": "inference"},
            "max_abs_state_value": max_abs,
        }
    )
    zarr.consolidate_metadata(store)
    os.replace(temporary, destination)

    source_segments = []
    for inventory in inventories:
        result = inventory.production_root / "segment_result.json"
        source_segments.append(
            {
                **inventory.summary(),
                "segment_result": str(result),
                "segment_result_sha256": _sha256_file(result) if result.is_file() else None,
                "wind_sha256": _sha256_file(inventory.production_root / "windx_cosy.bin"),
            }
        )
    split_payload = {key: value for key, value in split.items() if key not in {"snapshot_codes", "pair_codes"}}
    split_payload["snapshot_codes_sha256"] = _sha256_array(np.asarray(split["snapshot_codes"], dtype=np.uint8))
    split_payload["pair_codes_sha256"] = _sha256_array(np.asarray(split["pair_codes"], dtype=np.uint8))
    manifest = {
        "dataset": str(destination),
        "dataset_metadata_sha256": _sha256_file(destination / ".zmetadata"),
        "experiments": list(selected),
        "spec": asdict(spec),
        "source_segments": source_segments,
        "wet_cells": int(reference_mask.sum()),
        "wet_fraction": float(reference_mask.mean()),
        "training_values_per_channel": int(count),
        "state_mean_sha256": _sha256_array(means.astype(np.float32)),
        "state_scale_sha256": _sha256_array(scales.astype(np.float32)),
        "split": split_payload,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _variable_groups(nr: int) -> dict[str, slice]:
    return {
        "U_m_s": slice(0, nr),
        "V_m_s": slice(nr, 2 * nr),
        "Theta_degC": slice(2 * nr, 3 * nr),
        "Eta_m": slice(3 * nr, 3 * nr + 1),
    }


def validate_dataset(
    path: str | Path,
    *,
    spec: DatasetSpec = DatasetSpec(),
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Independently check data, split, and training-only normalization integrity."""

    dataset = Path(path).resolve()
    group = zarr.open_consolidated(str(dataset), mode="r")
    channels = state_channels(spec.nr)
    expected_shape = (3, spec.expected_records, len(channels), spec.ny, spec.nx)
    if tuple(group["state"].shape) != expected_shape:
        raise ValueError(f"state shape mismatch: {group['state'].shape} != {expected_shape}")
    if tuple(group["static_features"].shape) != (3, len(STATIC_FEATURES), spec.ny, spec.nx):
        raise ValueError("static feature shape mismatch")
    if group.attrs.get("state_channels") != list(channels) or group.attrs.get("state_units") != state_units(spec.nr):
        raise ValueError("state metadata does not match the declared AF-FNO contract")
    wet = np.asarray(group["wet_mask"], dtype=bool)
    if not np.any(wet) or np.all(wet):
        raise ValueError("invalid wet mask")
    static = np.asarray(group["static_features"])
    if not np.all(np.isfinite(static)) or not np.all(static[:, 3] == wet.astype(np.float32)):
        raise ValueError("invalid static features or wet-mask channel")
    stored_mean = np.asarray(group["state_mean"], dtype=np.float64)
    stored_scale = np.asarray(group["state_scale"], dtype=np.float64)
    if not np.all(np.isfinite(stored_mean)) or not np.all(stored_scale > 0):
        raise ValueError("invalid normalization arrays")
    iterations = np.asarray(group["iteration"])
    if np.any(np.diff(iterations, axis=1) != MODEL_DAY_STEPS):
        raise ValueError("stored iterations are not daily-contiguous")

    declared_split = split_manifest(spec)
    snapshots = np.asarray(group["snapshot_split"], dtype=np.uint8)
    pairs = np.asarray(group["pair_split"], dtype=np.uint8)
    if not np.array_equal(snapshots, np.asarray(declared_split["snapshot_codes"], dtype=np.uint8)):
        raise ValueError("snapshot split does not match the declared chronological contract")
    if not np.array_equal(pairs, np.asarray(declared_split["pair_codes"], dtype=np.uint8)):
        raise ValueError("pair split does not match the declared horizon contract")

    finite = True
    land_zero = True
    spectrum_energy = 0.0
    sums = np.zeros(len(channels), dtype=np.float64)
    squares = np.zeros(len(channels), dtype=np.float64)
    count = 0
    minima = np.full(len(channels), np.inf, dtype=np.float64)
    maxima = np.full(len(channels), -np.inf, dtype=np.float64)
    tendency_abs_sum = np.zeros(len(channels), dtype=np.float64)
    tendency_count = 0
    tendency_abs_max = np.zeros(len(channels), dtype=np.float64)
    for experiment_index in range(3):
        previous: np.ndarray | None = None
        for start in range(0, spec.expected_records, 32):
            stop = min(start + 32, spec.expected_records)
            block = np.asarray(group["state"][experiment_index, start:stop])
            finite &= bool(np.all(np.isfinite(block)))
            land_zero &= bool(np.all(block[:, :, ~wet] == 0.0))
            wet_values = block[:, :, wet].astype(np.float64, copy=False)
            minima = np.minimum(minima, wet_values.min(axis=(0, 2)))
            maxima = np.maximum(maxima, wet_values.max(axis=(0, 2)))
            train_block = block[snapshots[start:stop] == 1]
            if train_block.size:
                train_wet = train_block[:, :, wet].astype(np.float64, copy=False)
                sums += train_wet.sum(axis=(0, 2))
                squares += np.square(train_wet).sum(axis=(0, 2))
                count += train_wet.shape[0] * train_wet.shape[2]
            if previous is not None:
                block_for_difference = np.concatenate((previous[None], block), axis=0)
            else:
                block_for_difference = block
            differences = np.diff(block_for_difference, axis=0)[:, :, wet].astype(np.float64, copy=False)
            if differences.size:
                tendency_abs_sum += np.abs(differences).sum(axis=(0, 2))
                tendency_abs_max = np.maximum(tendency_abs_max, np.abs(differences).max(axis=(0, 2)))
                tendency_count += differences.shape[0] * differences.shape[2]
            previous = block[-1]
        theta_surface = np.asarray(group["state"][experiment_index, 0, 2 * spec.nr])
        spectrum_energy += float(np.square(np.abs(np.fft.rfft2(theta_surface * wet))).sum())
    if not finite or not land_zero or spectrum_energy <= 0.0 or count == 0 or tendency_count == 0:
        raise ValueError("state finite/land/spectral validation failed")

    means = sums / count
    variances = np.maximum(squares / count - np.square(means), 0.0)
    scales = np.maximum(np.sqrt(variances), 1.0e-6)
    mean_error = float(np.max(np.abs(means - stored_mean)))
    scale_error = float(np.max(np.abs(scales - stored_scale)))
    if not np.allclose(means, stored_mean, rtol=2.0e-6, atol=1.0e-6) or not np.allclose(
        scales, stored_scale, rtol=2.0e-6, atol=1.0e-6
    ):
        raise ValueError(
            f"stored training-only normalizers do not reproduce: mean error={mean_error}, scale error={scale_error}"
        )
    ranges = {
        name: {
            "min": float(minima[channel_slice].min()),
            "max": float(maxima[channel_slice].max()),
            "mean_abs_one_day_increment": float((tendency_abs_sum[channel_slice] / tendency_count).mean()),
            "max_abs_one_day_increment": float(tendency_abs_max[channel_slice].max()),
        }
        for name, channel_slice in _variable_groups(spec.nr).items()
    }
    report = {
        "status": "valid",
        "dataset": str(dataset),
        "state_shape": list(expected_shape),
        "wet_cells": int(wet.sum()),
        "spectral_energy": spectrum_energy,
        "training_values_per_channel": count,
        "normalizer_max_abs_error": {"mean": mean_error, "scale": scale_error},
        "state_and_one_day_increment_ranges": ranges,
        "state_metadata_sha256": _sha256_file(dataset / ".zmetadata"),
    }
    if report_path is not None:
        quality_report = Path(report_path).resolve()
        quality_report.parent.mkdir(parents=True, exist_ok=True)
        _write_json(quality_report, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the shared S0-S2 AF-FNO trajectory dataset")
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="inventory, convert, split, and normalize S0-S2")
    convert.add_argument("--scratch-root", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--chunk-days", type=int, default=8)
    validate = commands.add_parser("validate", help="validate an existing AF-FNO Zarr store")
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--report", type=Path, help="optional JSON quality-report destination")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "convert":
        result = build_dataset(args.scratch_root, args.output, chunk_days=args.chunk_days)
    else:
        result = validate_dataset(args.output, report_path=args.report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
