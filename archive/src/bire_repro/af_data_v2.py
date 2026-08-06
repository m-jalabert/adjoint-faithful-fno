"""Build and independently validate the expanded AF--FNO trajectory dataset.

Version 2 concatenates each immutable ten-year v1 trajectory with its exact
ten-year continuation.  The opened v1 validation chronology is excluded from
successor selection, original inference remains sealed, and fresh buffered
validation/inference blocks come only from the extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr
from numcodecs import Blosc

from .af_data import (
    MODEL_DAY_STEPS,
    STATIC_FEATURES,
    STATE_CHANNELS,
    DatasetSpec,
    _read_field_pair,
)
from .af_s0 import STEPS_PER_YEAR, _sha256
from .af_trajectory_expansion import (
    EXPERIMENTS,
    dataset_pair_counts,
    load_expansion_contract,
)


DATASET_VERSION = "trajectories_v2"
OLD_RECORDS = 3600
NEW_RECORDS = 3600
TOTAL_RECORDS = OLD_RECORDS + NEW_RECORDS
STATE_CHANNEL_COUNT = len(STATE_CHANNELS)


class TrajectoryV2DataError(RuntimeError):
    """Raised when the expanded dataset violates its frozen source contract."""


@dataclass(frozen=True)
class ExtensionInventory:
    experiment: str
    run_dir: Path
    manifest_path: Path
    result_path: Path
    dyn_meta: tuple[Path, ...]
    surf_meta: tuple[Path, ...]
    iterations: tuple[int, ...]


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def dataset_v2_split(
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Materialize exact snapshot/pair split codes from the expansion contract."""

    design = contract["dataset_v2_design"]
    records = int(design["raw_records_per_regime"])
    horizon = int(design["horizon_days"])
    if records != TOTAL_RECORDS or horizon != 10:
        raise ValueError("trajectory-v2 dimensions or horizon changed")
    snapshot = np.zeros(records, dtype=np.uint8)
    pair = np.zeros(records, dtype=np.uint8)
    codes = {"training": 1, "validation": 2, "inference": 3}
    counts: dict[str, int] = {}
    for name, code in codes.items():
        count = 0
        for start, stop in design["snapshot_blocks"][name]:
            start, stop = int(start), int(stop)
            if np.any(snapshot[start:stop]):
                raise ValueError("trajectory-v2 split blocks overlap")
            snapshot[start:stop] = code
            pair[start : stop - horizon] = code
            count += stop - start - horizon
        counts[name] = count
    for start, stop in design["snapshot_blocks"]["excluded"]:
        start, stop = int(start), int(stop)
        if np.any(snapshot[start:stop]):
            raise ValueError("trajectory-v2 excluded block overlaps a split")
    expected = {
        key: int(value)
        for key, value in design["pair_counts_per_regime"].items()
    }
    if counts != expected or counts != dataset_pair_counts(contract):
        raise ValueError("trajectory-v2 pair counts changed")
    return snapshot, pair, counts


def _iteration(path: Path, prefix: str) -> int:
    parts = path.name.split(".")
    if len(parts) != 3 or parts[0] != prefix or parts[2] != "meta":
        raise ValueError(f"unexpected {prefix} metadata name: {path.name}")
    return int(parts[1])


def inventory_extension(
    scratch_root: str | Path,
    contract: Mapping[str, Any],
    contract_sha256: str,
    experiment: str,
) -> ExtensionInventory:
    """Verify one complete immutable continuation before conversion."""

    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}")
    spec = contract["experiments"][experiment]
    start = int(spec["local_start_year"])
    stop = int(spec["local_end_year"])
    run_dir = (
        Path(scratch_root).resolve()
        / contract["simulation_design"]["extension_root"]
        / experiment
        / "production"
        / f"years_{start:03d}_{stop:03d}"
    )
    manifest_path = run_dir / "segment_manifest.json"
    result_path = run_dir / "segment_result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"incomplete trajectory-v2 source for {experiment}")
    manifest = json.loads(manifest_path.read_text())
    result = json.loads(result_path.read_text())
    if (
        manifest.get("version") != contract["version"]
        or manifest.get("expansion_contract_sha256") != contract_sha256
        or result.get("returncode") != 0
        or result.get("diagnostics")
        != {"dynState": NEW_RECORDS, "surfState": NEW_RECORDS}
    ):
        raise TrajectoryV2DataError(f"{experiment} extension provenance is invalid")
    dyn_meta = tuple(
        sorted(
            run_dir.glob("dynState.*.meta"),
            key=lambda path: _iteration(path, "dynState"),
        )
    )
    if len(dyn_meta) != NEW_RECORDS:
        raise TrajectoryV2DataError(f"{experiment} has an incomplete dynamic inventory")
    iterations = tuple(_iteration(path, "dynState") for path in dyn_meta)
    expected_first = int(spec["absolute_start_year"]) * STEPS_PER_YEAR
    expected_last = int(spec["absolute_end_year"]) * STEPS_PER_YEAR - MODEL_DAY_STEPS
    if (
        iterations[0] != expected_first
        or iterations[-1] != expected_last
        or any(
            later - earlier != MODEL_DAY_STEPS
            for earlier, later in zip(iterations, iterations[1:])
        )
    ):
        raise TrajectoryV2DataError(f"{experiment} extension is not daily-contiguous")
    surf_meta = tuple(
        run_dir / f"surfState.{iteration:010d}.meta" for iteration in iterations
    )
    missing = [
        path
        for path in (*dyn_meta, *surf_meta)
        if not path.is_file() or not path.with_suffix(".data").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{experiment} extension has missing MDS pairs: {missing[:3]}")
    final_iteration = int(spec["absolute_end_year"]) * STEPS_PER_YEAR
    for suffix in ("meta", "data"):
        if not (run_dir / f"pickup.{final_iteration:010d}.{suffix}").is_file():
            raise FileNotFoundError(f"{experiment} final extension pickup is incomplete")
    return ExtensionInventory(
        experiment=experiment,
        run_dir=run_dir,
        manifest_path=manifest_path,
        result_path=result_path,
        dyn_meta=dyn_meta,
        surf_meta=surf_meta,
        iterations=iterations,
    )


def _accumulate_training(
    block: np.ndarray,
    selected: np.ndarray,
    wet: np.ndarray,
    sums: np.ndarray,
    squares: np.ndarray,
) -> int:
    train = block[selected]
    if train.size == 0:
        return 0
    values = train[:, :, wet].astype(np.float64, copy=False)
    sums += values.sum(axis=(0, 2))
    squares += np.square(values).sum(axis=(0, 2))
    return int(values.shape[0] * values.shape[2])


def build_dataset_v2(
    v1_path: str | Path,
    scratch_root: str | Path,
    expansion_contract_path: str | Path,
    output_path: str | Path,
    *,
    chunk_days: int = 8,
) -> dict[str, Any]:
    """Build the immutable 20-year-per-regime trajectory-v2 Zarr store."""

    if chunk_days <= 0:
        raise ValueError("trajectory-v2 chunk size must be positive")
    v1_path = Path(v1_path).resolve()
    output = Path(output_path).resolve()
    temporary = output.with_name(output.name + ".tmp")
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or temporary.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite trajectory-v2 dataset: {output}")
    contract, resolved_contract, contract_sha = load_expansion_contract(
        expansion_contract_path
    )
    snapshot_codes, pair_codes, pair_counts = dataset_v2_split(contract)
    inventories = [
        inventory_extension(scratch_root, contract, contract_sha, experiment)
        for experiment in EXPERIMENTS
    ]

    v1 = zarr.open_consolidated(str(v1_path), mode="r")
    if (
        tuple(v1["state"].shape) != (3, OLD_RECORDS, STATE_CHANNEL_COUNT, 62, 62)
        or tuple(v1.attrs.get("experiments", ())) != EXPERIMENTS
        or tuple(v1.attrs.get("state_channels", ())) != STATE_CHANNELS
    ):
        raise TrajectoryV2DataError("unexpected trajectories-v1 source contract")
    wet = np.asarray(v1["wet_mask"][:], dtype=bool)
    if wet.shape != (62, 62) or not np.any(wet):
        raise TrajectoryV2DataError("invalid trajectories-v1 wet mask")

    output.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(temporary))
    group = zarr.group(store=store, overwrite=False)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    state = group.create_dataset(
        "state",
        shape=(3, TOTAL_RECORDS, STATE_CHANNEL_COUNT, 62, 62),
        chunks=(1, chunk_days, STATE_CHANNEL_COUNT, 62, 62),
        dtype="f4",
        compressor=compressor,
    )
    static = group.create_dataset(
        "static_features",
        shape=(3, len(STATIC_FEATURES), 62, 62),
        chunks=(1, len(STATIC_FEATURES), 62, 62),
        dtype="f4",
        compressor=compressor,
    )
    iterations = group.create_dataset(
        "iteration",
        shape=(3, TOTAL_RECORDS),
        chunks=(1, TOTAL_RECORDS),
        dtype="i8",
    )
    for name in ("longitude_deg", "latitude_deg", "wet_mask"):
        group.create_dataset(name, data=np.asarray(v1[name][:]), compressor=compressor)
    group.create_dataset("snapshot_split", data=snapshot_codes)
    group.create_dataset("pair_split", data=pair_codes)

    sums = np.zeros(STATE_CHANNEL_COUNT, dtype=np.float64)
    squares = np.zeros(STATE_CHANNEL_COUNT, dtype=np.float64)
    count = 0
    maximum = 0.0
    spec = DatasetSpec()
    for experiment_index, inventory in enumerate(inventories):
        static[experiment_index] = np.asarray(
            v1["static_features"][experiment_index], dtype=np.float32
        )
        old_iterations = np.asarray(
            v1["iteration"][experiment_index], dtype=np.int64
        )
        combined_iterations = np.concatenate(
            (old_iterations, np.asarray(inventory.iterations, dtype=np.int64))
        )
        if np.any(np.diff(combined_iterations) != MODEL_DAY_STEPS):
            raise TrajectoryV2DataError(
                f"{inventory.experiment} v1/v2 iteration join is not daily"
            )
        iterations[experiment_index] = combined_iterations

        for start in range(0, OLD_RECORDS, chunk_days):
            stop = min(start + chunk_days, OLD_RECORDS)
            block = np.asarray(
                v1["state"][experiment_index, start:stop], dtype=np.float32
            )
            if not np.all(np.isfinite(block)) or not np.all(block[:, :, ~wet] == 0):
                raise TrajectoryV2DataError("trajectories-v1 source state is invalid")
            state[experiment_index, start:stop] = block
            maximum = max(maximum, float(np.max(np.abs(block))))
            count += _accumulate_training(
                block,
                snapshot_codes[start:stop] == 1,
                wet,
                sums,
                squares,
            )

        for local_start in range(0, NEW_RECORDS, chunk_days):
            local_stop = min(local_start + chunk_days, NEW_RECORDS)
            block = np.empty(
                (local_stop - local_start, STATE_CHANNEL_COUNT, 62, 62),
                dtype=np.float32,
            )
            for offset, index in enumerate(range(local_start, local_stop)):
                value = _read_field_pair(
                    inventory.dyn_meta[index],
                    inventory.surf_meta[index],
                    spec,
                )
                value[:, ~wet] = 0.0
                block[offset] = value
            if not np.all(np.isfinite(block)):
                raise TrajectoryV2DataError("non-finite state in trajectory extension")
            global_start = OLD_RECORDS + local_start
            global_stop = OLD_RECORDS + local_stop
            state[experiment_index, global_start:global_stop] = block
            maximum = max(maximum, float(np.max(np.abs(block))))
            count += _accumulate_training(
                block,
                snapshot_codes[global_start:global_stop] == 1,
                wet,
                sums,
                squares,
            )

    if count <= 0:
        raise TrajectoryV2DataError("trajectory-v2 training split is empty")
    means = sums / count
    variances = np.maximum(squares / count - np.square(means), 0.0)
    scales = np.maximum(np.sqrt(variances), 1.0e-6)
    group.create_dataset("state_mean", data=means.astype(np.float32))
    group.create_dataset("state_scale", data=scales.astype(np.float32))
    group.attrs.update(dict(v1.attrs))
    group.attrs.update(
        {
            "schema_version": 2,
            "dataset_version": DATASET_VERSION,
            "normalization": (
                "wet cells from both declared trajectory-v2 training blocks only"
            ),
            "split_policy": contract["dataset_v2_design"]["split_policy"],
            "source_v1": str(v1_path),
            "expansion_contract": str(resolved_contract),
            "expansion_contract_sha256": contract_sha,
            "max_abs_state_value": maximum,
        }
    )
    zarr.consolidate_metadata(store)
    os.replace(temporary, output)

    source_extensions = [
        {
            "experiment": inventory.experiment,
            "run_dir": str(inventory.run_dir),
            "manifest": str(inventory.manifest_path),
            "manifest_sha256": _sha256(inventory.manifest_path),
            "result": str(inventory.result_path),
            "result_sha256": _sha256(inventory.result_path),
            "records": len(inventory.iterations),
            "first_iteration": inventory.iterations[0],
            "last_iteration": inventory.iterations[-1],
        }
        for inventory in inventories
    ]
    manifest = {
        "status": "complete",
        "version": DATASET_VERSION,
        "dataset": str(output),
        "dataset_metadata_sha256": _sha256(output / ".zmetadata"),
        "state_shape": [3, TOTAL_RECORDS, STATE_CHANNEL_COUNT, 62, 62],
        "source_v1": str(v1_path),
        "source_v1_metadata_sha256": _sha256(v1_path / ".zmetadata"),
        "expansion_contract": str(resolved_contract),
        "expansion_contract_sha256": contract_sha,
        "source_extensions": source_extensions,
        "split": {
            "snapshot_codes_sha256": _array_sha256(snapshot_codes),
            "pair_codes_sha256": _array_sha256(pair_codes),
            "pair_counts_per_regime": pair_counts,
            "training_pairs_total": pair_counts["training"] * len(EXPERIMENTS),
            "snapshot_blocks": contract["dataset_v2_design"]["snapshot_blocks"],
        },
        "training_values_per_channel": count,
        "state_mean_sha256": _array_sha256(means.astype(np.float32)),
        "state_scale_sha256": _array_sha256(scales.astype(np.float32)),
        "wet_cells": int(wet.sum()),
        "inference_state_metrics_read": False,
    }
    _write_json(manifest_path, manifest)
    return manifest


def validate_dataset_v2(
    dataset_path: str | Path,
    v1_path: str | Path,
    scratch_root: str | Path,
    expansion_contract_path: str | Path,
    report_path: str | Path,
    *,
    chunk_days: int = 16,
    raw_samples_per_extension: int = 3,
) -> dict[str, Any]:
    """Independently verify v2 sources, split, normalizers, and state integrity."""

    if chunk_days <= 0 or raw_samples_per_extension != 3:
        raise ValueError("trajectory-v2 validation settings changed")
    dataset_path = Path(dataset_path).resolve()
    v1_path = Path(v1_path).resolve()
    report_path = Path(report_path).resolve()
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite v2 quality report: {report_path}")
    contract, _, contract_sha = load_expansion_contract(expansion_contract_path)
    snapshot_codes, pair_codes, pair_counts = dataset_v2_split(contract)
    inventories = [
        inventory_extension(scratch_root, contract, contract_sha, experiment)
        for experiment in EXPERIMENTS
    ]
    dataset = zarr.open_consolidated(str(dataset_path), mode="r")
    v1 = zarr.open_consolidated(str(v1_path), mode="r")
    if (
        tuple(dataset["state"].shape)
        != (3, TOTAL_RECORDS, STATE_CHANNEL_COUNT, 62, 62)
        or tuple(dataset["static_features"].shape)
        != (3, len(STATIC_FEATURES), 62, 62)
        or dataset.attrs.get("dataset_version") != DATASET_VERSION
        or dataset.attrs.get("expansion_contract_sha256") != contract_sha
    ):
        raise TrajectoryV2DataError("trajectory-v2 dataset metadata is invalid")
    if not np.array_equal(dataset["snapshot_split"][:], snapshot_codes):
        raise TrajectoryV2DataError("trajectory-v2 snapshot split changed")
    if not np.array_equal(dataset["pair_split"][:], pair_codes):
        raise TrajectoryV2DataError("trajectory-v2 pair split changed")
    if not np.array_equal(
        dataset["static_features"][:], v1["static_features"][:]
    ):
        raise TrajectoryV2DataError("trajectory-v2 static features changed")
    wet = np.asarray(dataset["wet_mask"][:], dtype=bool)
    stored_mean = np.asarray(dataset["state_mean"][:], dtype=np.float64)
    stored_scale = np.asarray(dataset["state_scale"][:], dtype=np.float64)
    sums = np.zeros(STATE_CHANNEL_COUNT, dtype=np.float64)
    squares = np.zeros(STATE_CHANNEL_COUNT, dtype=np.float64)
    count = 0
    finite = True
    land_zero = True
    v1_exact = True
    maximum = 0.0
    for experiment_index in range(len(EXPERIMENTS)):
        iterations = np.asarray(
            dataset["iteration"][experiment_index], dtype=np.int64
        )
        if np.any(np.diff(iterations) != MODEL_DAY_STEPS):
            raise TrajectoryV2DataError("trajectory-v2 iteration continuity failed")
        for start in range(0, TOTAL_RECORDS, chunk_days):
            stop = min(start + chunk_days, TOTAL_RECORDS)
            block = np.asarray(
                dataset["state"][experiment_index, start:stop], dtype=np.float32
            )
            finite &= bool(np.all(np.isfinite(block)))
            land_zero &= bool(np.all(block[:, :, ~wet] == 0))
            maximum = max(maximum, float(np.max(np.abs(block))))
            if start < OLD_RECORDS:
                old_stop = min(stop, OLD_RECORDS)
                old = np.asarray(
                    v1["state"][experiment_index, start:old_stop],
                    dtype=np.float32,
                )
                v1_exact &= bool(np.array_equal(block[: old_stop - start], old))
            count += _accumulate_training(
                block,
                snapshot_codes[start:stop] == 1,
                wet,
                sums,
                squares,
            )
    if not finite or not land_zero or not v1_exact or count <= 0:
        raise TrajectoryV2DataError(
            "trajectory-v2 finite/land/source-copy validation failed"
        )
    means = sums / count
    variances = np.maximum(squares / count - np.square(means), 0.0)
    scales = np.maximum(np.sqrt(variances), 1.0e-6)
    mean_error = float(np.max(np.abs(means - stored_mean)))
    scale_error = float(np.max(np.abs(scales - stored_scale)))
    if not np.allclose(means, stored_mean, rtol=2.0e-6, atol=1.0e-6) or not np.allclose(
        scales, stored_scale, rtol=2.0e-6, atol=1.0e-6
    ):
        raise TrajectoryV2DataError(
            f"trajectory-v2 normalizers do not reproduce: {mean_error}, {scale_error}"
        )

    raw_indices = np.linspace(
        0, NEW_RECORDS - 1, raw_samples_per_extension, dtype=np.int64
    )
    raw_samples_exact = True
    spec = DatasetSpec()
    for experiment_index, inventory in enumerate(inventories):
        for local_index in raw_indices:
            raw = _read_field_pair(
                inventory.dyn_meta[int(local_index)],
                inventory.surf_meta[int(local_index)],
                spec,
            )
            raw[:, ~wet] = 0.0
            stored = np.asarray(
                dataset["state"][
                    experiment_index, OLD_RECORDS + int(local_index)
                ],
                dtype=np.float32,
            )
            raw_samples_exact &= bool(np.array_equal(raw, stored))
    if not raw_samples_exact:
        raise TrajectoryV2DataError("trajectory-v2 raw MDS sample validation failed")

    report = {
        "status": "valid",
        "purpose": "independent_trajectories_v2_quality_validation",
        "dataset": str(dataset_path),
        "dataset_metadata_sha256": _sha256(dataset_path / ".zmetadata"),
        "source_v1": str(v1_path),
        "source_v1_metadata_sha256": _sha256(v1_path / ".zmetadata"),
        "expansion_contract_sha256": contract_sha,
        "state_shape": [3, TOTAL_RECORDS, STATE_CHANNEL_COUNT, 62, 62],
        "pair_counts_per_regime": pair_counts,
        "training_pairs_total": pair_counts["training"] * len(EXPERIMENTS),
        "finite": finite,
        "land_zero": land_zero,
        "v1_state_copy_bitwise_exact": v1_exact,
        "raw_extension_samples_bitwise_exact": raw_samples_exact,
        "raw_sample_indices_per_extension": raw_indices.tolist(),
        "iteration_step": MODEL_DAY_STEPS,
        "training_values_per_channel": count,
        "normalizer_max_abs_error": {
            "mean": mean_error,
            "scale": scale_error,
        },
        "max_abs_state_value": maximum,
        "inference_state_metrics_read": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--v1", type=Path, required=True)
    build.add_argument("--scratch-root", type=Path, required=True)
    build.add_argument("--expansion-contract", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--chunk-days", type=int, default=8)
    validate = commands.add_parser("validate")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--v1", type=Path, required=True)
    validate.add_argument("--scratch-root", type=Path, required=True)
    validate.add_argument("--expansion-contract", type=Path, required=True)
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--chunk-days", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build_dataset_v2(
            args.v1,
            args.scratch_root,
            args.expansion_contract,
            args.output,
            chunk_days=args.chunk_days,
        )
    else:
        result = validate_dataset_v2(
            args.dataset,
            args.v1,
            args.scratch_root,
            args.expansion_contract,
            args.report,
            chunk_days=args.chunk_days,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
