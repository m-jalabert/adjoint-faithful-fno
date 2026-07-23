"""Training-only data and representation audit for Model C.

This stage intentionally reads only split-code-1 snapshots and pairs.  It
quantifies temporal dependence, ten-day increment scales, and the fraction of
tapered state/increment energy represented by the predeclared Fourier-mode
candidates before a Model C training job is allowed to run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .af_data import STATE_CHANNELS
from .af_model_c import GROUP_SLICES, MODE_CANDIDATES


HORIZON_DAYS = 10
EXPERIMENT_COUNT = 3
SAMPLE_COUNT = 96


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integrated_autocorrelation_time(values: Sequence[float]) -> float:
    """Estimate integral time using the initial positive autocorrelation sequence."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 3 or not np.all(np.isfinite(array)):
        raise ValueError("autocorrelation input must be a finite one-dimensional series")
    centered = array - array.mean()
    variance = float(np.dot(centered, centered))
    if variance <= 0:
        return float(array.size)
    transform = np.fft.rfft(centered, n=2 * array.size)
    covariance = np.fft.irfft(transform * transform.conjugate())[: array.size]
    covariance /= np.arange(array.size, 0, -1)
    correlation = covariance / covariance[0]
    nonpositive = np.flatnonzero(correlation[1:] <= 0)
    stop = int(nonpositive[0] + 1) if nonpositive.size else array.size
    return float(max(1.0, 1.0 + 2.0 * correlation[1:stop].sum()))


def balanced_training_records(
    pair_codes: Sequence[int],
    *,
    sample_count: int = SAMPLE_COUNT,
    seed: int = 20260723,
) -> tuple[tuple[int, int], ...]:
    """Select balanced, reproducible, training-only spectral examples."""

    codes = np.asarray(pair_codes, dtype=np.uint8)
    candidates = np.flatnonzero(codes == 1)
    if codes.ndim != 1 or not 20 <= sample_count <= 100 or candidates.size == 0:
        raise ValueError("Model C diagnostic sampling needs 20--100 training pairs")
    counts = [
        sample_count // EXPERIMENT_COUNT + int(i < sample_count % EXPERIMENT_COUNT)
        for i in range(EXPERIMENT_COUNT)
    ]
    records: list[tuple[int, int]] = []
    for experiment, count in enumerate(counts):
        selected = np.sort(
            np.random.default_rng(seed + experiment).choice(candidates, size=count, replace=False)
        )
        records.extend((experiment, int(index)) for index in selected)
    return tuple(records)


def wet_rectangle_bounds(wet_mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the exact wet rectangle used by tapered spectra."""

    wet = np.asarray(wet_mask, dtype=bool)
    if wet.ndim != 2 or not wet.any():
        raise ValueError("Model C diagnostics need a nonempty two-dimensional wet mask")
    rows, columns = np.where(wet)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    rectangle = np.zeros_like(wet)
    rectangle[y0:y1, x0:x1] = True
    if not np.array_equal(wet, rectangle):
        raise ValueError("current tapered-spectrum method requires an exact wet rectangle")
    return y0, y1, x0, x1


def retained_energy_fraction(
    fields: np.ndarray,
    n_modes: tuple[int, int],
) -> float:
    """Return energy inside NeuralOperator's real-FNO mode convention."""

    values = np.asarray(fields, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("spectral fields must be channel,Y,X")
    if n_modes not in MODE_CANDIDATES:
        raise ValueError("unregistered Model C mode candidate")
    values = values - values.mean(axis=(-2, -1), keepdims=True)
    window = np.hanning(values.shape[-2])[:, None] * np.hanning(values.shape[-1])[None, :]
    transformed = np.fft.rfft2(values * window, norm="ortho")
    transformed = np.fft.fftshift(transformed, axes=-2)
    total = float(np.square(np.abs(transformed)).sum())
    y_modes = n_modes[0]
    y_start = values.shape[-2] // 2 - y_modes // 2
    y_stop = y_start + y_modes
    x_stop = n_modes[1] // 2 + 1
    retained = float(
        np.square(np.abs(transformed[:, y_start:y_stop, :x_stop])).sum()
    )
    return retained / total if total > 0 else 1.0


def _contiguous_training_indices(codes: np.ndarray, split_code: int) -> np.ndarray:
    indices = np.flatnonzero(codes == split_code)
    if indices.size == 0 or not np.all(np.diff(indices) == 1):
        raise ValueError(f"Model C expects one contiguous split-code-{split_code} block")
    return indices


def assert_training_pairs_have_training_targets(
    pair_codes: Sequence[int],
    snapshot_codes: Sequence[int],
    *,
    horizon_days: int = HORIZON_DAYS,
) -> None:
    """Verify target snapshots, not later pair starts, remain in training."""

    pairs = np.asarray(pair_codes, dtype=np.uint8)
    snapshots = np.asarray(snapshot_codes, dtype=np.uint8)
    indices = np.flatnonzero(pairs == 1)
    if (
        pairs.ndim != 1
        or snapshots.shape != pairs.shape
        or horizon_days <= 0
        or indices.size == 0
        or int(indices[-1]) + horizon_days >= snapshots.size
        or np.any(snapshots[indices] != 1)
        or np.any(snapshots[indices + horizon_days] != 1)
    ):
        raise ValueError("a nominal Model C training pair crosses the training boundary")


def _normalise(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - mean[None, :, None, None]) / scale[None, :, None, None]


def _training_statistics(
    state: Any,
    pair_indices: np.ndarray,
    snapshot_indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    wet: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream exact increment RMS and group proxy series from the training chronology."""

    increment_squares = np.zeros(len(STATE_CHANNELS), dtype=np.float64)
    increment_count = 0
    state_proxy = np.empty(
        (EXPERIMENT_COUNT, len(GROUP_SLICES), snapshot_indices.size), dtype=np.float64
    )
    increment_proxy = np.empty(
        (EXPERIMENT_COUNT, len(GROUP_SLICES), pair_indices.size), dtype=np.float64
    )
    wet_values = wet[None, None]
    chunk = 32

    for experiment in range(EXPERIMENT_COUNT):
        for start in range(0, snapshot_indices.size, chunk):
            selected = snapshot_indices[start : start + chunk]
            raw = np.asarray(
                state[experiment, int(selected[0]) : int(selected[-1]) + 1],
                dtype=np.float32,
            )
            normalized = _normalise(raw, mean, scale)
            for group_index, channels in enumerate(GROUP_SLICES.values()):
                group = normalized[:, channels]
                state_proxy[experiment, group_index, start : start + selected.size] = np.sqrt(
                    (np.square(group) * wet_values).sum(axis=(1, 2, 3))
                    / (wet.sum() * (channels.stop - channels.start))
                )

        for start in range(0, pair_indices.size, chunk):
            selected = pair_indices[start : start + chunk]
            first, last = int(selected[0]), int(selected[-1])
            present = np.asarray(state[experiment, first : last + 1], dtype=np.float32)
            future = np.asarray(
                state[
                    experiment,
                    first + HORIZON_DAYS : last + HORIZON_DAYS + 1,
                ],
                dtype=np.float32,
            )
            increment = (future - present) / scale[None, :, None, None]
            squared = np.square(increment.astype(np.float64)) * wet_values
            increment_squares += squared.sum(axis=(0, 2, 3))
            increment_count += increment.shape[0] * int(wet.sum())
            for group_index, channels in enumerate(GROUP_SLICES.values()):
                group = increment[:, channels]
                increment_proxy[experiment, group_index, start : start + selected.size] = np.sqrt(
                    (np.square(group) * wet_values).sum(axis=(1, 2, 3))
                    / (wet.sum() * (channels.stop - channels.start))
                )

    if increment_count <= 0 or not np.all(np.isfinite(increment_squares)):
        raise RuntimeError("Model C training statistics did not accumulate finite increments")
    increment_rms = np.sqrt(increment_squares / increment_count)
    if np.any(increment_rms <= 0):
        raise RuntimeError("at least one Model C state channel has zero ten-day increment RMS")
    return increment_rms, state_proxy, increment_proxy


def _spectral_audit(
    state: Any,
    records: Sequence[tuple[int, int]],
    mean: np.ndarray,
    scale: np.ndarray,
    wet_bounds: tuple[int, int, int, int],
) -> dict[str, Any]:
    y0, y1, x0, x1 = wet_bounds
    retained = {
        "state": {
            name: {f"{modes[0]}x{modes[1]}": [] for modes in MODE_CANDIDATES}
            for name in GROUP_SLICES
        },
        "increment": {
            name: {f"{modes[0]}x{modes[1]}": [] for modes in MODE_CANDIDATES}
            for name in GROUP_SLICES
        },
    }
    for experiment, time_index in records:
        present = np.asarray(state[experiment, time_index], dtype=np.float32)
        future = np.asarray(
            state[experiment, time_index + HORIZON_DAYS], dtype=np.float32
        )
        normalized = (present - mean[:, None, None]) / scale[:, None, None]
        increment = (future - present) / scale[:, None, None]
        for group_name, channels in GROUP_SLICES.items():
            group_state = normalized[channels, y0:y1, x0:x1]
            group_increment = increment[channels, y0:y1, x0:x1]
            for modes in MODE_CANDIDATES:
                key = f"{modes[0]}x{modes[1]}"
                retained["state"][group_name][key].append(
                    retained_energy_fraction(group_state, modes)
                )
                retained["increment"][group_name][key].append(
                    retained_energy_fraction(group_increment, modes)
                )
    return {
        family: {
            group: {
                modes: {
                    "mean": float(np.mean(values)),
                    "p10": float(np.percentile(values, 10)),
                    "p90": float(np.percentile(values, 90)),
                }
                for modes, values in candidates.items()
            }
            for group, candidates in groups.items()
        }
        for family, groups in retained.items()
    }


def diagnose_dataset(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    sample_count: int = SAMPLE_COUNT,
    seed: int = 20260723,
) -> dict[str, Any]:
    """Create the sealed training-only Model C data/representation report."""

    import zarr

    dataset_path = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model C diagnostics: {output}")
    started = time.monotonic()
    group = zarr.open_consolidated(str(dataset_path), mode="r")
    state = group["state"]
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    pair_indices = _contiguous_training_indices(pair_codes, 1)
    snapshot_indices = _contiguous_training_indices(snapshot_codes, 1)
    if (
        tuple(state.shape) != (3, 3600, 46, 62, 62)
        or tuple(group.attrs.get("state_channels", ())) != STATE_CHANNELS
    ):
        raise ValueError("Model C diagnostics received an unexpected dataset contract")
    assert_training_pairs_have_training_targets(pair_codes, snapshot_codes)

    mean = np.asarray(group["state_mean"][:], dtype=np.float32)
    scale = np.asarray(group["state_scale"][:], dtype=np.float32)
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    bounds = wet_rectangle_bounds(wet)
    records = balanced_training_records(pair_codes, sample_count=sample_count, seed=seed)
    increment_rms, state_proxy, increment_proxy = _training_statistics(
        state, pair_indices, snapshot_indices, mean, scale, wet
    )
    spectral = _spectral_audit(state, records, mean, scale, bounds)

    autocorrelation: dict[str, Any] = {"state_rms": {}, "increment_rms": {}}
    for family, proxies in (
        ("state_rms", state_proxy),
        ("increment_rms", increment_proxy),
    ):
        sample_length = proxies.shape[-1]
        for group_index, group_name in enumerate(GROUP_SLICES):
            regime_values = []
            for experiment in range(EXPERIMENT_COUNT):
                tau = integrated_autocorrelation_time(proxies[experiment, group_index])
                regime_values.append(
                    {
                        "experiment": str(group.attrs["experiments"][experiment]),
                        "tau_days": tau,
                        "effective_samples": sample_length / tau,
                    }
                )
            autocorrelation[family][group_name] = {
                "by_regime": regime_values,
                "effective_samples_total": float(
                    sum(item["effective_samples"] for item in regime_values)
                ),
            }

    metadata_path = dataset_path / ".zmetadata"
    report = {
        "status": "complete",
        "purpose": "training_only_model_c_data_and_representation_audit",
        "dataset": str(dataset_path),
        "dataset_metadata_sha256": _file_sha256(metadata_path),
        "read_contract": {
            "pair_split_codes_read": [1],
            "snapshot_split_codes_read": [1],
            "validation_read": False,
            "inference_read": False,
            "i1_i2_read": False,
            "response_or_adjoint_read": False,
        },
        "shape": list(state.shape),
        "wet_cells": int(wet.sum()),
        "wet_rectangle": {
            "y_start": bounds[0],
            "y_stop": bounds[1],
            "x_start": bounds[2],
            "x_stop": bounds[3],
        },
        "counts": {
            "training_snapshots_per_regime": int(snapshot_indices.size),
            "training_pairs_per_regime": int(pair_indices.size),
            "training_pairs_total": int(pair_indices.size * EXPERIMENT_COUNT),
            "spectral_samples_total": len(records),
        },
        "overlap_warning": (
            "daily starts overlap within a ten-day target; raw pairs are not independent"
        ),
        "increment_rms_normalized_state_units": increment_rms.tolist(),
        "increment_rms_floor": 1.0e-8,
        "autocorrelation": autocorrelation,
        "spectral_retained_energy": spectral,
        "mode_semantics": (
            "n_modes=(my,mx) maps to my centered fftshift rows and mx//2+1 rfft columns, "
            "matching NeuralOperator 2.0 real-valued SpectralConv"
        ),
        "records": [list(record) for record in records],
        "seed": seed,
        "elapsed_seconds": time.monotonic() - started,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "model_c_training_diagnostics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        output / "model_c_training_diagnostics_arrays.npz",
        increment_rms=increment_rms,
        state_rms_proxy=state_proxy,
        increment_rms_proxy=increment_proxy,
        spectral_records=np.asarray(records, dtype=np.int64),
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sealed training-only Model C diagnostics")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = diagnose_dataset(
        args.dataset,
        args.output_dir,
        sample_count=args.samples,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
