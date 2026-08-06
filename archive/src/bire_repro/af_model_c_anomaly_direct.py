"""Pointwise-anomaly, direct-future-state Model C experiment.

This experiment changes the state representation and prediction
parameterization together, following the consequential choices recovered from
the archived Bire implementation:

* every dynamic channel and wet grid point is centred by its split-1 temporal
  mean and scaled by its split-1 temporal standard deviation;
* quiet points are protected by a per-channel wet-cell quantile floor;
* the FNO predicts the next normalized anomaly state directly rather than a
  residual increment.

The width-128 successor architecture, three-step loss-v1 objective, optimizer,
data order, forcing/static inputs, and ten-day interval remain fixed.  Model
selection uses only the established 540-member split-1 rollout gate.  The
selected checkpoint is then characterized on the prospectively fixed
15-member S2 validation ensemble through day 200; inference, response, and
adjoint states remain sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from .af_a0 import records_for_pair_split
from .af_a0_evaluate import _normalizers
from .af_data import STATIC_FEATURES
from .af_forward_complete import _training_climatology
from .af_model_a import (
    ChunkAwareBatchSampler,
    _checkpoint_state_dict,
    require_model_a_runtime,
    seed_everything,
)
from .af_model_b import records_for_rollout_split, western_boundary_mask
from .af_model_c import (
    MODEL_C_LOSS_V1_CONTRACT_SHA256,
    loss_contract,
    loss_contract_sha256,
    model_c_loss_config,
    model_c_loss_terms,
)
from .af_model_c_bire_figures import (
    FIGURE_3_LEADS,
    FIELDS as BIRE_FIELDS,
    LEAD_DAYS as BIRE_LEAD_DAYS,
    _array_sha256,
    _evaluate_curves,
    _metric_summary,
    complete_figure_starts,
    select_ensemble_starts,
)
from .af_model_c_checkpoint_replay_audit import checkpoint_gate_summary
from .af_model_c_overfit import AUDIT_TERMS, _device, _file_sha256
from .af_model_c_pushforward_objective import select_pushforward_checkpoint
from .af_model_c_rollout_diagnosis import (
    complete_rollout_starts,
    lead_curve_summary,
    select_balanced_training_times,
)
from .af_model_c_validation import (
    _accumulate_physical_errors,
    _finish_physical_errors,
    _metric_accumulator,
)
from .af_model_c_successor import (
    HORIZON_DAYS,
    REFERENCE_DIAGNOSTIC_SEED,
    STATE_CHANNEL_COUNT,
    ModelCSuccessorArchitecture,
    _sample_records_by_regime,
    build_successor,
)
from .af_model_c_successor_validation import (
    LEAD_DAYS,
    _evaluate_baseline_metrics,
    _evaluate_stepper,
    _method_auc_summary,
)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    Dataset = object  # type: ignore[assignment,misc]


VERSION = "model_c_anomaly_direct_v1"
REPORT_NAME = "model_c_anomaly_direct_v1_report.json"
ARRAYS_NAME = "model_c_anomaly_direct_v1_arrays.npz"
CHECKPOINT_NAME = "model_c_anomaly_direct_v1_best.pt"
CHECKPOINT_DIRECTORY = "training_checkpoints"
NORMALIZATION_NAME = "model_c_anomaly_direct_v1_normalization.npz"
EXPECTED_FIGURE_STARTS = (
    6335,
    6353,
    6330,
    6361,
    6358,
    6308,
    6313,
    6346,
    6324,
    6323,
    6319,
    6325,
    6355,
    6366,
    6351,
)


class ModelCAnomalyDirectError(RuntimeError):
    """Raised when the anomaly-direct experiment contract is violated."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _float_array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype=np.float32).tobytes(order="C")
    ).hexdigest()


def _contiguous_runs(indices: np.ndarray) -> tuple[tuple[int, int], ...]:
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("normalizer indices must be a nonempty vector")
    cuts = np.flatnonzero(np.diff(indices) != 1) + 1
    return tuple(
        (int(piece[0]), int(piece[-1]) + 1)
        for piece in np.split(indices, cuts)
    )


def training_pointwise_normalizers(
    group: Any,
    snapshot_codes: np.ndarray,
    *,
    split_code: int = 1,
    floor_quantile: float = 0.05,
    absolute_floor: float = 1.0e-6,
    chunk_days: int = 64,
) -> dict[str, Any]:
    """Compute pooled split-only pointwise mean/std and a robust channel floor."""

    if not 0.0 <= floor_quantile < 0.5:
        raise ValueError("pointwise scale-floor quantile must lie in [0, 0.5)")
    if absolute_floor <= 0.0 or chunk_days <= 0:
        raise ValueError("pointwise absolute floor and chunk size must be positive")
    state = group["state"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    codes = np.asarray(snapshot_codes, dtype=np.uint8)
    selected = np.flatnonzero(codes == split_code)
    if (
        state.shape[2:] != (STATE_CHANNEL_COUNT, *wet.shape)
        or state.shape[1] != codes.size
        or state.shape[0] <= 0
    ):
        raise ValueError("pointwise normalizer received an inconsistent dataset")

    sums = np.zeros((STATE_CHANNEL_COUNT, *wet.shape), dtype=np.float64)
    squares = np.zeros_like(sums)
    count = 0
    for experiment in range(state.shape[0]):
        for run_start, run_stop in _contiguous_runs(selected):
            for start in range(run_start, run_stop, chunk_days):
                stop = min(start + chunk_days, run_stop)
                block = np.asarray(
                    state[experiment, start:stop],
                    dtype=np.float64,
                )
                sums += block.sum(axis=0)
                squares += np.square(block).sum(axis=0)
                count += int(block.shape[0])
    if count != int(selected.size * state.shape[0]):
        raise ModelCAnomalyDirectError("pointwise normalizer count changed")

    mean64 = sums / count
    variance64 = np.maximum(squares / count - np.square(mean64), 0.0)
    raw_scale64 = np.sqrt(variance64)
    floors64 = np.maximum(
        np.quantile(raw_scale64[:, wet], floor_quantile, axis=1),
        absolute_floor,
    )
    scale64 = np.maximum(raw_scale64, floors64[:, None, None])
    mean64[:, ~wet] = 0.0
    raw_scale64[:, ~wet] = 0.0
    scale64[:, ~wet] = 1.0
    mean = np.ascontiguousarray(mean64, dtype=np.float32)
    raw_scale = np.ascontiguousarray(raw_scale64, dtype=np.float32)
    scale = np.ascontiguousarray(scale64, dtype=np.float32)
    floors = np.ascontiguousarray(floors64, dtype=np.float32)
    if (
        not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
        or np.any(floors <= 0.0)
    ):
        raise ModelCAnomalyDirectError("pointwise normalizers are invalid")

    wet_raw = raw_scale[:, wet]
    affected_fraction = np.mean(
        wet_raw < floors[:, None],
        axis=1,
    ).astype(np.float32)
    summary = {
        "training_snapshots_total": count,
        "training_snapshots_per_regime": int(selected.size),
        "wet_cells": int(wet.sum()),
        "floor_quantile": float(floor_quantile),
        "absolute_floor": float(absolute_floor),
        "pointwise_mean_sha256": _float_array_sha256(mean),
        "pointwise_raw_scale_sha256": _float_array_sha256(raw_scale),
        "pointwise_scale_sha256": _float_array_sha256(scale),
        "channel_floor_sha256": _float_array_sha256(floors),
        "channel_floor": floors.tolist(),
        "fraction_wet_cells_floored": affected_fraction.tolist(),
        "maximum_fraction_wet_cells_floored": float(affected_fraction.max()),
        "minimum_wet_raw_scale": np.min(wet_raw, axis=1).astype(float).tolist(),
        "median_wet_raw_scale": np.median(wet_raw, axis=1).astype(float).tolist(),
        "maximum_wet_raw_scale": np.max(wet_raw, axis=1).astype(float).tolist(),
    }
    return {
        "mean": mean,
        "raw_scale": raw_scale,
        "scale": scale,
        "floor": floors,
        "summary": summary,
    }


def pointwise_increment_scale(
    group: Any,
    pair_codes: np.ndarray,
    pointwise_scale: np.ndarray,
    *,
    chunk_days: int = 32,
) -> np.ndarray:
    """RMS normalized ten-day increments under the pointwise coordinates."""

    state = group["state"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    codes = np.asarray(pair_codes, dtype=np.uint8)
    runs = _contiguous_runs(np.flatnonzero(codes == 1))
    if pointwise_scale.shape != (STATE_CHANNEL_COUNT, *wet.shape):
        raise ValueError("pointwise increment scale received invalid normalizers")
    squares = np.zeros(STATE_CHANNEL_COUNT, dtype=np.float64)
    count = 0
    for experiment in range(state.shape[0]):
        for run_start, run_stop in runs:
            for start in range(run_start, run_stop, chunk_days):
                stop = min(start + chunk_days, run_stop)
                present = np.asarray(state[experiment, start:stop], dtype=np.float32)
                future = np.asarray(
                    state[
                        experiment,
                        start + HORIZON_DAYS : stop + HORIZON_DAYS,
                    ],
                    dtype=np.float32,
                )
                increment = (
                    (future - present) / pointwise_scale[None]
                )[:, :, wet]
                squares += np.square(increment.astype(np.float64)).sum(axis=(0, 2))
                count += int(increment.shape[0] * wet.sum())
    result = np.sqrt(squares / count).astype(np.float32)
    if (
        count <= 0
        or result.shape != (STATE_CHANNEL_COUNT,)
        or not np.all(np.isfinite(result))
        or np.any(result <= 0.0)
    ):
        raise ModelCAnomalyDirectError("pointwise increment scales are invalid")
    return result


class ModelCAnomalyRolloutDataset(Dataset):
    """Lazy pointwise-anomaly inputs and direct future-state targets."""

    def __init__(
        self,
        dataset_path: str | Path,
        records: Sequence[tuple[int, int]],
        pointwise_mean: np.ndarray,
        pointwise_scale: np.ndarray,
        *,
        horizon_days: int = HORIZON_DAYS,
        rollout_steps: int = 3,
    ) -> None:
        require_model_a_runtime()
        self.dataset_path = Path(dataset_path).resolve()
        self.records = tuple(
            (int(experiment), int(time_index))
            for experiment, time_index in records
        )
        self.pointwise_mean = np.ascontiguousarray(pointwise_mean, dtype=np.float32)
        self.pointwise_scale = np.ascontiguousarray(pointwise_scale, dtype=np.float32)
        self.horizon_days = int(horizon_days)
        self.rollout_steps = int(rollout_steps)
        if not self.records or self.rollout_steps <= 0:
            raise ValueError("anomaly rollout dataset needs records and positive steps")
        self._group: Any | None = None
        self._state: Any | None = None
        self._static: Any | None = None
        self._open()

    def _open(self) -> None:
        self._group = zarr.open_consolidated(str(self.dataset_path), mode="r")
        self._state = self._group["state"]
        self._static = self._group["static_features"]
        _, _, self.wet, _, self.wind_mean, self.wind_scale = _normalizers(
            self._group
        )
        expected = (STATE_CHANNEL_COUNT, *self.wet.shape)
        if (
            self.pointwise_mean.shape != expected
            or self.pointwise_scale.shape != expected
            or np.any(self.pointwise_scale <= 0.0)
            or self._static.shape[1] != len(STATIC_FEATURES)
        ):
            raise ValueError("anomaly rollout normalizer or static contract changed")
        maximum_offset = self.horizon_days * self.rollout_steps
        if any(
            time_index + maximum_offset >= self._state.shape[1]
            for _, time_index in self.records
        ):
            raise ValueError("an anomaly rollout exceeds the stored chronology")

    def __len__(self) -> int:
        return len(self.records)

    def _normalise_state(self, raw: np.ndarray) -> np.ndarray:
        result = (raw - self.pointwise_mean) / self.pointwise_scale
        result[:, ~self.wet] = 0.0
        return np.ascontiguousarray(result, dtype=np.float32)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        experiment, time_index = self.records[index]
        present = self._normalise_state(
            np.asarray(self._state[experiment, time_index], dtype=np.float32)
        )
        futures = np.stack(
            [
                self._normalise_state(
                    np.asarray(
                        self._state[
                            experiment,
                            time_index + step * self.horizon_days,
                        ],
                        dtype=np.float32,
                    )
                )
                for step in range(1, self.rollout_steps + 1)
            ]
        )
        static = np.asarray(self._static[experiment], dtype=np.float32).copy()
        static[0] = (static[0] - self.wind_mean) / self.wind_scale
        static[0, ~self.wet] = 0.0
        features = np.concatenate((present, static), axis=0)
        return (
            torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(futures, dtype=np.float32)),
        )

    def __getstate__(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["_group"] = result["_state"] = result["_static"] = None
        return result

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open()


def direct_state_unroll(
    model: Any,
    features: Any,
    wet: Any,
    steps: int,
) -> Any:
    """Autoregress a direct normalized future-state map."""

    if steps <= 0:
        raise ValueError("direct-state rollout needs at least one step")
    current = features[:, :STATE_CHANNEL_COUNT]
    static = features[:, STATE_CHANNEL_COUNT:]
    predictions = []
    for _ in range(steps):
        current = model(torch.cat((current, static), dim=1)) * wet
        predictions.append(current)
    return torch.stack(predictions, dim=1)


@dataclass
class PointwiseDirectStepper:
    """Evaluation adapter for a direct map with pointwise normalization."""

    model: Any
    device: Any
    wet: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    wind_mean: float
    wind_scale: float

    def normalized_state(self, physical: np.ndarray) -> Any:
        value = (physical - self.mean[None]) / self.scale[None]
        value[:, :, ~self.wet] = 0.0
        return torch.from_numpy(
            np.ascontiguousarray(value, dtype=np.float32)
        ).to(device=self.device, dtype=torch.float32)

    def normalized_static(self, static: Any, experiments: np.ndarray) -> Any:
        value = np.stack(
            [
                np.asarray(static[int(experiment)], dtype=np.float32)
                for experiment in experiments
            ]
        )
        value[:, 0] = (value[:, 0] - self.wind_mean) / self.wind_scale
        value[:, 0, ~self.wet] = 0.0
        return torch.from_numpy(np.ascontiguousarray(value)).to(
            device=self.device,
            dtype=torch.float32,
        )

    def step(self, current: Any, static: Any) -> Any:
        wet = torch.from_numpy(self.wet.astype(np.float32))[None, None].to(
            self.device
        )
        return self.model(torch.cat((current, static), dim=1)) * wet

    def physical(self, normalized: Any) -> np.ndarray:
        value = normalized.detach().cpu().numpy()
        value = value * self.scale[None] + self.mean[None]
        value[:, :, ~self.wet] = 0.0
        return np.ascontiguousarray(value, dtype=np.float32)


def _one_step_diagnostics(
    model: Any,
    dataset_path: Path,
    records: Sequence[tuple[int, int]],
    pointwise_mean: np.ndarray,
    pointwise_scale: np.ndarray,
    *,
    batch_size: int,
    device: Any,
) -> dict[str, Any]:
    """Physical-unit direct-state RMSE ratios, aggregate and by regime."""

    def score(selected: Sequence[tuple[int, int]]) -> dict[str, Any]:
        dataset = ModelCAnomalyRolloutDataset(
            dataset_path,
            selected,
            pointwise_mean,
            pointwise_scale,
            rollout_steps=1,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        wet = torch.from_numpy(dataset.wet.astype(np.float32))[None, None].to(
            device
        )
        physical_scale = torch.from_numpy(pointwise_scale)[None].to(device)
        accumulator = _metric_accumulator()
        samples = 0
        model.eval()
        with torch.no_grad():
            for features, futures in loader:
                features = features.to(device=device, dtype=torch.float32)
                future = futures[:, 0].to(device=device, dtype=torch.float32)
                present = features[:, :STATE_CHANNEL_COUNT]
                prediction = model(features) * wet
                _accumulate_physical_errors(
                    accumulator,
                    prediction,
                    present,
                    future,
                    physical_scale,
                    wet,
                )
                samples += int(features.shape[0])
        result = _finish_physical_errors(accumulator)
        result["record_count"] = samples
        return result

    by_regime = {
        f"S{experiment}": score(
            [record for record in records if record[0] == experiment]
        )
        for experiment in range(3)
    }
    aggregate = score(records)
    return {
        "aggregate": aggregate,
        "by_regime": by_regime,
        "worst_per_regime_group_ratio": max(
            float(value["worst_group_ratio"]) for value in by_regime.values()
        ),
        "every_regime_and_group_beats_persistence": all(
            bool(value["all_groups_beat_persistence"])
            for value in by_regime.values()
        ),
    }


def load_anomaly_direct_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the source-locked anomaly-direct experiment contract."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    normalization = contract.get("normalization", {})
    prediction = contract.get("prediction", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status")
        != "frozen_before_anomaly_direct_training_or_validation_metrics"
        or normalization.get("centering")
        != "pooled_S0_S1_S2_split1_pointwise_time_mean"
        or normalization.get("scaling")
        != "pooled_S0_S1_S2_split1_pointwise_population_std"
        or float(normalization.get("wet_cell_floor_quantile", -1.0)) != 0.05
        or float(normalization.get("absolute_scale_floor", -1.0)) != 1.0e-6
        or normalization.get("regime_dependent") is not False
        or prediction.get("target") != "direct_normalized_future_anomaly_state"
        or prediction.get("residual_addition") is not False
    ):
        raise ValueError("anomaly-direct representation contract changed")
    architecture = ModelCSuccessorArchitecture(**contract["architecture"])
    if (
        architecture.hidden_channels != 128
        or architecture.n_layers != 4
        or architecture.domain_padding != 0.1
    ):
        raise ValueError("anomaly-direct architecture is not the fixed Model C")
    read = contract.get("read_contract", {})
    if (
        read.get("training_state") is not True
        or read.get("fixed_S2_validation_figure_state") is not True
        or any(
            read.get(name) is not False
            for name in (
                "inference_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
            )
        )
    ):
        raise ValueError("anomaly-direct read contract changed")
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(f"anomaly-direct source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def _verify_dataset(
    contract: Mapping[str, Any],
    dataset: Path,
    quality_report: Path,
) -> None:
    sources = contract["source_artifacts"]
    if (
        not dataset.is_dir()
        or _file_sha256(dataset / ".zmetadata")
        != sources["dataset_metadata_sha256"]
        or not quality_report.is_file()
        or _file_sha256(quality_report) != sources["quality_report_sha256"]
    ):
        raise ModelCAnomalyDirectError("anomaly-direct dataset source changed")
    quality = json.loads(quality_report.read_text())
    if (
        quality.get("status") != "valid"
        or quality.get("inference_state_metrics_read") is not False
    ):
        raise ModelCAnomalyDirectError("anomaly-direct dataset quality gate failed")


def _audit_records(
    pair_codes: np.ndarray,
    snapshot_codes: np.ndarray,
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    complete = complete_rollout_starts(pair_codes, snapshot_codes)
    spec = contract["training_evaluation"]["records"]
    times, bounds = select_balanced_training_times(
        complete,
        starts_per_block=int(spec["starts_per_training_block"]),
        expected_blocks=int(spec["expected_training_blocks"]),
    )
    records = np.asarray(
        [
            (experiment, int(time_index))
            for experiment in range(3)
            for time_index in times
        ],
        dtype=np.int64,
    )
    if (
        records.shape != (int(spec["records_total"]), 2)
        or _array_sha256(times) != spec["training_times_sha256"]
        or _array_sha256(records) != spec["records_sha256"]
    ):
        raise ModelCAnomalyDirectError("anomaly-direct audit records changed")
    return records, times, bounds


def _checkpoint_path(directory: Path, step: int) -> Path:
    return directory / f"model_c_anomaly_direct_step_{step:05d}.pt"


def _verify_normalizer_contract(
    contract: Mapping[str, Any],
    normalizers: Mapping[str, Any],
) -> None:
    expected = contract["normalization"]["expected"]
    observed = normalizers["summary"]
    for key in (
        "training_snapshots_total",
        "wet_cells",
        "pointwise_mean_sha256",
        "pointwise_raw_scale_sha256",
        "pointwise_scale_sha256",
        "channel_floor_sha256",
    ):
        if observed[key] != expected[key]:
            raise ModelCAnomalyDirectError(
                f"pointwise normalizer contract changed: {key}"
            )


def preflight_anomaly_direct(
    dataset_path: str | Path,
    quality_report_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify sources, outputs, and normalizers without opening held states."""

    contract, resolved, digest = load_anomaly_direct_contract(contract_path)
    dataset = Path(dataset_path).resolve()
    quality = Path(quality_report_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"anomaly-direct output already exists: {output}")
    _verify_dataset(contract, dataset, quality)
    group = zarr.open_consolidated(str(dataset), mode="r")
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    normalizers = training_pointwise_normalizers(
        group,
        snapshot_codes,
        floor_quantile=float(
            contract["normalization"]["wet_cell_floor_quantile"]
        ),
        absolute_floor=float(contract["normalization"]["absolute_scale_floor"]),
    )
    _verify_normalizer_contract(contract, normalizers)
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "normalization": normalizers["summary"],
        "validation_state_opened": False,
        "inference_state_opened": False,
    }


def run_anomaly_direct(
    dataset_path: str | Path,
    quality_report_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train, select, and characterize the immutable anomaly-direct Model C."""

    require_model_a_runtime()
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_anomaly_direct_contract(
        contract_path
    )
    dataset = Path(dataset_path).resolve()
    quality = Path(quality_report_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"anomaly-direct output already exists: {output}")
    _verify_dataset(contract, dataset, quality)
    seed = int(contract["training"]["seed"])
    seed_everything(seed)
    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    normalizers = training_pointwise_normalizers(
        group,
        snapshot_codes,
        floor_quantile=float(
            contract["normalization"]["wet_cell_floor_quantile"]
        ),
        absolute_floor=float(contract["normalization"]["absolute_scale_floor"]),
    )
    _verify_normalizer_contract(contract, normalizers)
    point_mean = normalizers["mean"]
    point_scale = normalizers["scale"]
    point_raw_scale = normalizers["raw_scale"]
    channel_floor = normalizers["floor"]
    global_mean, global_scale, wet_array, _, wind_mean, wind_scale = _normalizers(
        group
    )

    loss_config = model_c_loss_config("v1")
    if loss_contract_sha256(loss_config) != MODEL_C_LOSS_V1_CONTRACT_SHA256:
        raise ModelCAnomalyDirectError("base loss-v1 contract changed")
    architecture = ModelCSuccessorArchitecture(**contract["architecture"])
    optimizer_contract = contract["training"]["optimizer"]
    training_records = records_for_rollout_split(pair_codes, 1)
    one_step_records = records_for_pair_split(pair_codes, 1)
    diagnostic_records = _sample_records_by_regime(
        one_step_records,
        count_per_regime=int(
            contract["training_evaluation"]["checkpoint_records_per_regime"]
        ),
        seed=REFERENCE_DIAGNOSTIC_SEED,
    )
    audit_records, audit_times, block_bounds = _audit_records(
        pair_codes,
        snapshot_codes,
        contract,
    )
    increment_values = pointwise_increment_scale(
        group,
        pair_codes,
        point_scale,
    )
    climatology_state, climatology_derived, training_days = _training_climatology(
        state,
        snapshot_codes,
        wet_array,
    )
    baseline_raw = _evaluate_baseline_metrics(
        state,
        audit_records,
        climatology_state,
        climatology_derived,
        wet_array,
        global_mean,
        global_scale,
        batch_size=int(contract["training_evaluation"]["batch_size"]),
    )
    baselines = {
        method: baseline_raw[method]
        for method in ("persistence", "climatology")
    }

    training_dataset = ModelCAnomalyRolloutDataset(
        dataset,
        training_records,
        point_mean,
        point_scale,
        rollout_steps=loss_config.rollout_steps,
    )
    batch_size = int(optimizer_contract["batch_size"])
    loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(training_dataset, batch_size, seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    wet = torch.from_numpy(wet_array.astype(np.float32))[None, None].to(device)
    boundary_array = western_boundary_mask(
        wet_array,
        loss_config.western_boundary_width,
    )
    boundary = torch.from_numpy(boundary_array.astype(np.float32))[
        None, None
    ].to(device)
    increment_scale = torch.from_numpy(increment_values).to(device)
    model = build_successor(architecture).to(device)
    parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_contract["initial_learning_rate"]),
        betas=tuple(float(value) for value in optimizer_contract["adam_betas"]),
        weight_decay=float(optimizer_contract["weight_decay"]),
    )
    maximum_steps = int(optimizer_contract["maximum_steps"])
    decay_step = int(
        round(maximum_steps * float(optimizer_contract["decay_fraction"]))
    )
    checkpoint_steps = tuple(int(value) for value in contract["training"]["checkpoint_steps"])
    if checkpoint_steps[-1] != maximum_steps:
        raise ModelCAnomalyDirectError("anomaly-direct checkpoints must end at maximum steps")

    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    checkpoint_directory = temporary / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir()
    arrays: dict[str, np.ndarray] = {
        "pointwise_mean": point_mean,
        "pointwise_raw_scale": point_raw_scale,
        "pointwise_scale": point_scale,
        "channel_scale_floor": channel_floor,
        "fraction_wet_cells_floored": np.asarray(
            normalizers["summary"]["fraction_wet_cells_floored"],
            dtype=np.float32,
        ),
        "training_records": audit_records.astype(np.int32),
        "training_times": audit_times.astype(np.int32),
        "training_lead_days": np.asarray(LEAD_DAYS, dtype=np.int16),
        "longitude_deg": np.asarray(group["longitude_deg"][:], dtype=np.float32),
        "latitude_deg": np.asarray(group["latitude_deg"][:], dtype=np.float32),
        "wet_mask": wet_array.astype(np.uint8),
    }
    for method, values in baselines.items():
        for name, value in values.items():
            arrays[f"training_{method}__{name}"] = np.asarray(value)

    iterator = iter(loader)
    window_totals = {name: 0.0 for name in AUDIT_TERMS}
    window_samples = 0
    training_history: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []

    for step in range(1, maximum_steps + 1):
        if step == decay_step + 1:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= float(
                    optimizer_contract["decay_factor"]
                )
        try:
            features, futures = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            features, futures = next(iterator)
        features = features.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        futures = futures.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        model.train()
        predictions = direct_state_unroll(
            model,
            features,
            wet,
            loss_config.rollout_steps,
        )
        terms = model_c_loss_terms(
            predictions,
            futures,
            features[:, :STATE_CHANNEL_COUNT],
            wet,
            boundary,
            increment_scale,
            loss_config,
        )
        if not all(
            bool(torch.isfinite(terms[name]).item()) for name in AUDIT_TERMS
        ):
            raise ModelCAnomalyDirectError(
                "anomaly-direct training objective became non-finite"
            )
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        optimizer.step()
        size = int(features.shape[0])
        for name in AUDIT_TERMS:
            window_totals[name] += float(terms[name].detach().cpu()) * size
        window_samples += size
        if step not in checkpoint_steps:
            continue

        training_window = {
            name: window_totals[name] / window_samples
            for name in AUDIT_TERMS
        }
        diagnostic = _one_step_diagnostics(
            model,
            dataset,
            diagnostic_records,
            point_mean,
            point_scale,
            batch_size=16,
            device=device,
        )
        stepper = PointwiseDirectStepper(
            model=model,
            device=device,
            wet=wet_array,
            mean=point_mean,
            scale=point_scale,
            wind_mean=wind_mean,
            wind_scale=wind_scale,
        )
        metrics = _evaluate_stepper(
            stepper,
            state,
            static,
            audit_records,
            climatology_state,
            climatology_derived,
            global_mean,
            global_scale,
            batch_size=int(contract["training_evaluation"]["batch_size"]),
        )
        curves = lead_curve_summary(metrics, baselines, audit_records)
        gate = checkpoint_gate_summary(
            curves,
            diagnostic,
            metrics,
            contract["training_evaluation"]["checkpoint_gate"],
        )
        for name, value in metrics.items():
            arrays[f"training_step_{step}__{name}"] = np.asarray(value)
        history_record = {
            "optimizer_step": step,
            "optimizer_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_window": training_window,
        }
        training_history.append(history_record)
        checkpoint_path = _checkpoint_path(checkpoint_directory, step)
        payload = {
            "version": VERSION,
            "purpose": "pointwise_anomaly_direct_future_state_checkpoint",
            "optimizer_step": step,
            "architecture": architecture.to_dict(),
            "parameter_count": parameter_count,
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "normalization": normalizers["summary"],
            "prediction": contract["prediction"],
            "loss_contract": loss_contract(loss_config),
            "loss_contract_sha256": loss_contract_sha256(loss_config),
            "optimizer_contract": optimizer_contract,
            "training_history_record": history_record,
            "model_state_dict": _checkpoint_state_dict(model),
        }
        torch.save(payload, checkpoint_path)
        checkpoint_summaries.append(
            {
                "fine_tune_step": step,
                "optimizer_step": step,
                "checkpoint": str(
                    output / CHECKPOINT_DIRECTORY / checkpoint_path.name
                ),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
                "training_window": training_window,
                "ten_day_diagnostic": diagnostic,
                "all_field_auc": _method_auc_summary(metrics),
                "lead_curves": curves,
                "checkpoint_gate": gate,
            }
        )
        window_totals = {name: 0.0 for name in AUDIT_TERMS}
        window_samples = 0

    if len(checkpoint_summaries) != len(checkpoint_steps):
        raise ModelCAnomalyDirectError("not every contracted checkpoint was evaluated")
    decision = select_pushforward_checkpoint(checkpoint_summaries)
    selected_step = int(decision["selected_fine_tune_step"])
    selected_source = _checkpoint_path(checkpoint_directory, selected_step)
    selected_payload = torch.load(
        selected_source,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(selected_payload["model_state_dict"])
    model.eval()
    selected_path = temporary / CHECKPOINT_NAME
    torch.save(selected_payload, selected_path)

    full_one_step = _one_step_diagnostics(
        model,
        dataset,
        one_step_records,
        point_mean,
        point_scale,
        batch_size=16,
        device=device,
    )
    reload_dataset = ModelCAnomalyRolloutDataset(
        dataset,
        diagnostic_records[:3],
        point_mean,
        point_scale,
        rollout_steps=1,
    )
    reload_features = torch.stack(
        [reload_dataset[index][0] for index in range(3)]
    ).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        reference = direct_state_unroll(model, reload_features, wet, 9).cpu()
    restored = build_successor(architecture).to(device)
    loaded = torch.load(selected_path, map_location=device, weights_only=False)
    restored.load_state_dict(loaded["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        reloaded = direct_state_unroll(restored, reload_features, wet, 9).cpu()
    reload_exact = bool(torch.equal(reference, reloaded))
    if not reload_exact:
        raise ModelCAnomalyDirectError(
            "selected anomaly-direct checkpoint did not reload exactly"
        )

    figure_candidates = complete_figure_starts(pair_codes, snapshot_codes)
    figure_starts = select_ensemble_starts(
        figure_candidates,
        count=int(contract["validation_figure"]["member_count"]),
        seed=int(contract["validation_figure"]["selection_seed"]),
    )
    figure_contract = contract["validation_figure"]
    if (
        figure_candidates.size != int(figure_contract["candidate_count"])
        or _array_sha256(figure_candidates)
        != figure_contract["candidate_times_sha256"]
        or tuple(int(value) for value in figure_starts)
        != EXPECTED_FIGURE_STARTS
        or _array_sha256(figure_starts)
        != figure_contract["start_draw_order_sha256"]
    ):
        raise ModelCAnomalyDirectError("fixed Figure-4 member selection changed")
    selected_stepper = PointwiseDirectStepper(
        model=restored,
        device=device,
        wet=wet_array,
        mean=point_mean,
        scale=point_scale,
        wind_mean=wind_mean,
        wind_scale=wind_scale,
    )
    figure_arrays = _evaluate_curves(
        selected_stepper,
        state,
        static,
        figure_starts,
        climatology_state,
        climatology_derived,
        wet_array,
        EXPECTED_FIGURE_STARTS[0],
    )
    for name, value in figure_arrays.items():
        arrays[name] = np.asarray(value)
    figure_metrics = _metric_summary(figure_arrays)

    normalization_path = temporary / NORMALIZATION_NAME
    np.savez_compressed(
        normalization_path,
        pointwise_mean=point_mean,
        pointwise_raw_scale=point_raw_scale,
        pointwise_scale=point_scale,
        channel_scale_floor=channel_floor,
    )
    arrays_path = temporary / ARRAYS_NAME
    np.savez_compressed(arrays_path, **arrays)
    selected_summary = next(
        value
        for value in checkpoint_summaries
        if int(value["optimizer_step"]) == selected_step
    )
    report = {
        "status": "complete",
        "purpose": "pointwise_anomaly_direct_future_state_model_c",
        "version": VERSION,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset),
        "dataset_metadata_sha256": _file_sha256(dataset / ".zmetadata"),
        "quality_report": str(quality),
        "quality_report_sha256": _file_sha256(quality),
        "device": str(device),
        "architecture": architecture.to_dict(),
        "parameter_count": parameter_count,
        "normalization": normalizers["summary"],
        "normalization_artifact": str(output / NORMALIZATION_NAME),
        "normalization_artifact_sha256": _file_sha256(normalization_path),
        "prediction": contract["prediction"],
        "loss_contract": loss_contract(loss_config),
        "loss_contract_sha256": loss_contract_sha256(loss_config),
        "optimizer_contract": optimizer_contract,
        "optimizer_decay_step": decay_step,
        "seed": seed,
        "counts": {
            "training_rollouts": len(training_records),
            "training_one_step_pairs": len(one_step_records),
            "checkpoint_diagnostic_pairs": len(diagnostic_records),
            "training_evaluation_rollouts": int(audit_records.shape[0]),
            "training_climatology_snapshots_per_regime": training_days,
            "validation_figure_members": int(figure_starts.size),
        },
        "training_record_contract": {
            "complete_training_blocks": [list(value) for value in block_bounds],
            "training_times_sha256": _array_sha256(audit_times),
            "records_sha256": _array_sha256(audit_records),
        },
        "increment_scale": increment_values.tolist(),
        "increment_scale_sha256": _float_array_sha256(increment_values),
        "training_history": training_history,
        "checkpoint_summary": checkpoint_summaries,
        "selection_decision": decision,
        "selected_training_summary": selected_summary,
        "selected_full_one_step_training_diagnostic": full_one_step,
        "selected_checkpoint": str(output / CHECKPOINT_NAME),
        "selected_checkpoint_sha256": _file_sha256(selected_path),
        "save_reload_nine_step_bitwise_exact": reload_exact,
        "validation_figure": {
            "regime": "S2",
            "lead_days": list(BIRE_LEAD_DAYS),
            "fields": list(BIRE_FIELDS),
            "figure3_leads": list(FIGURE_3_LEADS),
            "start_draw_order": figure_starts.astype(int).tolist(),
            "metrics": figure_metrics,
        },
        "arrays": str(output / ARRAYS_NAME),
        "arrays_sha256": _file_sha256(arrays_path),
        "elapsed_seconds": time.monotonic() - started,
        "read_contract": contract["read_contract"],
        "validation_state_opened": True,
        "inference_state_opened": False,
        "intermediate_wind_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }
    report["report_content_sha256"] = _json_sha256(report)
    (temporary / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, output)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--dataset", type=Path, required=True)
        child.add_argument("--quality-report", type=Path, required=True)
        child.add_argument("--contract", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        if command == "run":
            child.add_argument(
                "--device",
                choices=("auto", "cpu", "cuda"),
                default="auto",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight_anomaly_direct(
            args.dataset,
            args.quality_report,
            args.contract,
            args.output_dir,
        )
    else:
        result = run_anomaly_direct(
            args.dataset,
            args.quality_report,
            args.contract,
            args.output_dir,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
