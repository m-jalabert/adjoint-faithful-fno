"""Train and select Model C Arm R: the reduced-channel causal control.

Arm R preserves the selected Model C representation, direct-state
parameterization, FNO trunk, optimizer, checkpoint schedule, and three-step
objective form.  Its only dynamical-task change is replacement of the dense
46-channel state by the ten recovered Bire-facing channels.  Checkpoint
selection reads split-1 training state only; held S0 evaluation is performed
by a separate, prospectively fixed phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from .af_a0 import records_for_pair_split
from .af_data import STATIC_FEATURES
from .af_forward_complete import _member_acc, _member_rmse
from .af_model_a import (
    ChunkAwareBatchSampler,
    _checkpoint_state_dict,
    require_model_a_runtime,
    seed_everything,
)
from .af_model_b import records_for_rollout_split, western_boundary_mask
from .af_model_c import model_c_loss_config
from .af_model_c_overfit import _device
from .af_model_c_reduced_channels import (
    REDUCED_AUDIT_TERMS,
    REDUCED_CHANNELS,
    REDUCED_CHANNEL_COUNT,
    REDUCED_GROUP_SLICES,
    VERSION as REDUCED_DATASET_VERSION,
    ReducedChannelArchitecture,
    array_sha256,
    build_reduced_model,
    direct_unroll,
    file_sha256,
    json_sha256,
    reduced_fields,
    reduced_loss_contract,
    reduced_loss_contract_sha256,
    reduced_loss_terms,
)
from .af_model_c_rollout_diagnosis import (
    complete_rollout_starts,
    select_balanced_training_times,
)
from .af_model_c_successor import (
    REFERENCE_DIAGNOSTIC_SEED,
    _sample_records_by_regime,
)
from .af_model_c_successor_validation import LEAD_DAYS, curve_auc

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    Dataset = object  # type: ignore[assignment,misc]


VERSION = "model_c_reduced_channel_control_v1"
CONTRACT_STATUS = "frozen_before_arm_r_training_or_held_evaluation"
REPORT_NAME = "model_c_reduced_channel_control_v1_report.json"
ARRAYS_NAME = "model_c_reduced_channel_control_v1_arrays.npz"
CHECKPOINT_NAME = "model_c_reduced_channel_control_v1_best.pt"
CHECKPOINT_DIRECTORY = "training_checkpoints"
NORMALIZATION_NAME = "model_c_reduced_channel_control_v1_normalization.npz"
MANIFEST_NAME = "manifest.json"
HORIZON_DAYS = 10
PRIMARY_FIELDS = ("surface_speed", "sst", "phihyd_surface")
SLOW_PRIMARY_FIELDS = ("sst", "phihyd_surface")


class ModelCReducedControlError(RuntimeError):
    """Raised when Arm R violates its frozen training contract."""


def _float_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype=np.float32).tobytes(order="C")
    ).hexdigest()


def _contiguous_runs(indices: np.ndarray) -> tuple[tuple[int, int], ...]:
    if indices.ndim != 1 or not indices.size:
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
    floor_quantile: float = 0.05,
    absolute_floor: float = 1.0e-6,
    chunk_days: int = 64,
) -> dict[str, Any]:
    """Compute pooled split-1 pointwise statistics for the ten channels."""

    if not 0.0 <= floor_quantile < 0.5:
        raise ValueError("floor quantile must lie in [0,0.5)")
    if absolute_floor <= 0.0 or chunk_days <= 0:
        raise ValueError("normalizer floors and chunks must be positive")
    state = group["state"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    codes = np.asarray(snapshot_codes, dtype=np.uint8)
    selected = np.flatnonzero(codes == 1)
    if (
        state.shape != (3, 7200, REDUCED_CHANNEL_COUNT, 62, 62)
        or wet.shape != (62, 62)
        or codes.shape != (7200,)
    ):
        raise ValueError("reduced normalizer received an inconsistent dataset")
    sums = np.zeros((REDUCED_CHANNEL_COUNT, *wet.shape), dtype=np.float64)
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
    expected_count = int(selected.size * state.shape[0])
    if count != expected_count:
        raise ModelCReducedControlError("normalizer count changed")
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
    wet_raw = raw_scale[:, wet]
    affected = np.mean(
        wet_raw < floors[:, None],
        axis=1,
    ).astype(np.float32)
    if (
        not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise ModelCReducedControlError("invalid reduced normalizers")
    return {
        "mean": mean,
        "raw_scale": raw_scale,
        "scale": scale,
        "floor": floors,
        "summary": {
            "training_snapshots_total": count,
            "training_snapshots_per_regime": int(selected.size),
            "wet_cells": int(wet.sum()),
            "floor_quantile": float(floor_quantile),
            "absolute_floor": float(absolute_floor),
            "pointwise_mean_sha256": _float_sha256(mean),
            "pointwise_raw_scale_sha256": _float_sha256(raw_scale),
            "pointwise_scale_sha256": _float_sha256(scale),
            "channel_floor_sha256": _float_sha256(floors),
            "channel_floor": floors.tolist(),
            "fraction_wet_cells_floored": affected.tolist(),
            "maximum_fraction_wet_cells_floored": float(affected.max()),
        },
    }


def pointwise_increment_scale(
    group: Any,
    pair_codes: np.ndarray,
    pointwise_scale: np.ndarray,
    *,
    chunk_days: int = 32,
) -> np.ndarray:
    """RMS normalized ten-day increments in the reduced coordinates."""

    state = group["state"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    runs = _contiguous_runs(
        np.flatnonzero(np.asarray(pair_codes, dtype=np.uint8) == 1)
    )
    if pointwise_scale.shape != (REDUCED_CHANNEL_COUNT, *wet.shape):
        raise ValueError("reduced increment scale received bad normalizers")
    squares = np.zeros(REDUCED_CHANNEL_COUNT, dtype=np.float64)
    count = 0
    for experiment in range(state.shape[0]):
        for run_start, run_stop in runs:
            for start in range(run_start, run_stop, chunk_days):
                stop = min(start + chunk_days, run_stop)
                present = np.asarray(
                    state[experiment, start:stop],
                    dtype=np.float32,
                )
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
                squares += np.square(
                    increment.astype(np.float64)
                ).sum(axis=(0, 2))
                count += int(increment.shape[0] * wet.sum())
    result = np.sqrt(squares / count).astype(np.float32)
    if (
        count <= 0
        or result.shape != (REDUCED_CHANNEL_COUNT,)
        or not np.all(np.isfinite(result))
        or np.any(result <= 0.0)
    ):
        raise ModelCReducedControlError("invalid reduced increment scales")
    return result


class ReducedRolloutDataset(Dataset):
    """Lazy direct-state rollout examples from the derived ten-channel cache."""

    def __init__(
        self,
        dataset_path: str | Path,
        records: Sequence[tuple[int, int]],
        pointwise_mean: np.ndarray,
        pointwise_scale: np.ndarray,
        *,
        rollout_steps: int = 3,
    ) -> None:
        require_model_a_runtime()
        self.dataset_path = Path(dataset_path).resolve()
        self.records = tuple(
            (int(experiment), int(time_index))
            for experiment, time_index in records
        )
        self.pointwise_mean = np.ascontiguousarray(
            pointwise_mean,
            dtype=np.float32,
        )
        self.pointwise_scale = np.ascontiguousarray(
            pointwise_scale,
            dtype=np.float32,
        )
        self.rollout_steps = int(rollout_steps)
        self._group: Any | None = None
        self._state: Any | None = None
        self._static: Any | None = None
        self._open()

    def _open(self) -> None:
        self._group = zarr.open_consolidated(
            str(self.dataset_path),
            mode="r",
        )
        self._state = self._group["state"]
        self._static = self._group["static_features"]
        self.wet = np.asarray(self._group["wet_mask"][:], dtype=bool)
        wind = np.asarray(self._static[:, 0], dtype=np.float32)
        self.wind_mean = float(wind[:, self.wet].mean())
        self.wind_scale = float(wind[:, self.wet].std())
        expected = (REDUCED_CHANNEL_COUNT, *self.wet.shape)
        if (
            self.pointwise_mean.shape != expected
            or self.pointwise_scale.shape != expected
            or np.any(self.pointwise_scale <= 0.0)
            or self._static.shape[1] != len(STATIC_FEATURES)
        ):
            raise ValueError("Arm-R dataset or normalizers changed")
        if any(
            time_index + HORIZON_DAYS * self.rollout_steps
            >= self._state.shape[1]
            for _, time_index in self.records
        ):
            raise ValueError("Arm-R rollout exceeds the chronology")

    def __len__(self) -> int:
        return len(self.records)

    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        result = (raw - self.pointwise_mean) / self.pointwise_scale
        result[:, ~self.wet] = 0.0
        return np.ascontiguousarray(result, dtype=np.float32)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        experiment, time_index = self.records[index]
        present = self._normalise(
            np.asarray(
                self._state[experiment, time_index],
                dtype=np.float32,
            )
        )
        futures = np.stack(
            [
                self._normalise(
                    np.asarray(
                        self._state[
                            experiment,
                            time_index + step * HORIZON_DAYS,
                        ],
                        dtype=np.float32,
                    )
                )
                for step in range(1, self.rollout_steps + 1)
            ]
        )
        static = np.asarray(
            self._static[experiment],
            dtype=np.float32,
        ).copy()
        static[0] = (static[0] - self.wind_mean) / self.wind_scale
        static[0, ~self.wet] = 0.0
        features = np.concatenate((present, static), axis=0)
        return (
            torch.from_numpy(np.ascontiguousarray(features)),
            torch.from_numpy(np.ascontiguousarray(futures)),
        )

    def __getstate__(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["_group"] = result["_state"] = result["_static"] = None
        return result

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open()


@dataclass
class ReducedDirectStepper:
    """Physical/normalized adapter for the ten-channel direct map."""

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
        ).to(self.device)

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
        wet = torch.from_numpy(
            self.wet.astype(np.float32)
        )[None, None].to(self.device)
        return self.model(torch.cat((current, static), dim=1)) * wet

    def physical(self, normalized: Any) -> np.ndarray:
        value = normalized.detach().cpu().numpy()
        value = value * self.scale[None] + self.mean[None]
        value[:, :, ~self.wet] = 0.0
        return np.ascontiguousarray(value, dtype=np.float32)


def _training_climatology(
    group: Any,
    snapshot_codes: np.ndarray,
    *,
    chunk_days: int = 64,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Return per-regime split-1 pointwise means and nonlinear field means."""

    state = group["state"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    selected = np.flatnonzero(
        np.asarray(snapshot_codes, dtype=np.uint8) == 1
    )
    runs = _contiguous_runs(selected)
    state_mean = np.empty(
        (3, REDUCED_CHANNEL_COUNT, *wet.shape),
        dtype=np.float32,
    )
    field_mean = {
        name: np.empty((3, *wet.shape), dtype=np.float32)
        for name in (
            "surface_speed",
            "surface_u",
            "surface_v",
            "sst",
            "phihyd_surface",
            "streamfunction",
        )
    }
    for experiment in range(3):
        state_sum = np.zeros(
            (REDUCED_CHANNEL_COUNT, *wet.shape),
            dtype=np.float64,
        )
        field_sum = {
            name: np.zeros(wet.shape, dtype=np.float64)
            for name in field_mean
        }
        count = 0
        for run_start, run_stop in runs:
            for start in range(run_start, run_stop, chunk_days):
                stop = min(start + chunk_days, run_stop)
                raw = np.asarray(
                    state[experiment, start:stop],
                    dtype=np.float32,
                )
                state_sum += raw.sum(axis=0, dtype=np.float64)
                fields = reduced_fields(raw, wet)
                for name in field_sum:
                    field_sum[name] += fields[name].sum(
                        axis=0,
                        dtype=np.float64,
                    )
                count += int(raw.shape[0])
        if count != selected.size:
            raise ModelCReducedControlError("climatology count changed")
        state_mean[experiment] = (state_sum / count).astype(np.float32)
        state_mean[experiment, :, ~wet] = 0.0
        for name in field_mean:
            field_mean[name][experiment] = (
                field_sum[name] / count
            ).astype(np.float32)
            field_mean[name][experiment, ~wet] = 0.0
    return state_mean, field_mean, int(selected.size)


def _one_step_score(
    model: Any,
    dataset_path: Path,
    records: Sequence[tuple[int, int]],
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    batch_size: int,
    device: Any,
) -> dict[str, Any]:
    dataset = ReducedRolloutDataset(
        dataset_path,
        records,
        mean,
        scale,
        rollout_steps=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    wet = torch.from_numpy(
        dataset.wet.astype(np.float32)
    )[None, None].to(device)
    physical_scale = torch.from_numpy(scale)[None].to(device)
    accumulator = {
        name: {
            "model_squared_error": 0.0,
            "persistence_squared_error": 0.0,
            "count": 0,
        }
        for name in REDUCED_GROUP_SLICES
    }
    samples = 0
    model.eval()
    with torch.no_grad():
        for features, futures in loader:
            features = features.to(device=device, dtype=torch.float32)
            future = futures[:, 0].to(device=device, dtype=torch.float32)
            present = features[:, :REDUCED_CHANNEL_COUNT]
            prediction = model(features) * wet
            for name, channels in REDUCED_GROUP_SLICES.items():
                model_error = (
                    prediction[:, channels] - future[:, channels]
                ) * physical_scale[:, channels]
                persistence_error = (
                    present[:, channels] - future[:, channels]
                ) * physical_scale[:, channels]
                accumulator[name]["model_squared_error"] += float(
                    (model_error.square() * wet).sum(
                        dtype=torch.float64
                    ).cpu()
                )
                accumulator[name]["persistence_squared_error"] += float(
                    (persistence_error.square() * wet).sum(
                        dtype=torch.float64
                    ).cpu()
                )
                accumulator[name]["count"] += (
                    int(features.shape[0])
                    * (channels.stop - channels.start)
                    * int(dataset.wet.sum())
                )
            samples += int(features.shape[0])
    model_rmse = {}
    persistence_rmse = {}
    ratio = {}
    for name, values in accumulator.items():
        count = int(values["count"])
        model_rmse[name] = math.sqrt(
            values["model_squared_error"] / count
        )
        persistence_rmse[name] = math.sqrt(
            values["persistence_squared_error"] / count
        )
        ratio[name] = model_rmse[name] / persistence_rmse[name]
    return {
        "model_rmse": model_rmse,
        "persistence_rmse": persistence_rmse,
        "ratio_to_persistence": ratio,
        "mean_group_ratio": float(np.mean(tuple(ratio.values()))),
        "worst_group_ratio": float(max(ratio.values())),
        "all_groups_beat_persistence": all(
            value < 1.0 for value in ratio.values()
        ),
        "record_count": samples,
    }


def _one_step_diagnostics(
    model: Any,
    dataset_path: Path,
    records: Sequence[tuple[int, int]],
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    batch_size: int,
    device: Any,
) -> dict[str, Any]:
    by_regime = {
        f"S{experiment}": _one_step_score(
            model,
            dataset_path,
            [
                record
                for record in records
                if record[0] == experiment
            ],
            mean,
            scale,
            batch_size=batch_size,
            device=device,
        )
        for experiment in range(3)
    }
    aggregate = _one_step_score(
        model,
        dataset_path,
        records,
        mean,
        scale,
        batch_size=batch_size,
        device=device,
    )
    return {
        "aggregate": aggregate,
        "by_regime": by_regime,
        "worst_per_regime_group_ratio": max(
            float(value["worst_group_ratio"])
            for value in by_regime.values()
        ),
        "every_regime_and_group_beats_persistence": all(
            bool(value["all_groups_beat_persistence"])
            for value in by_regime.values()
        ),
    }


def _empty_metrics(count: int) -> dict[str, np.ndarray]:
    result = {}
    for field in PRIMARY_FIELDS:
        result[f"rmse__{field}"] = np.empty(
            (count, len(LEAD_DAYS)),
            dtype=np.float32,
        )
        result[f"acc__{field}"] = np.empty(
            (count, len(LEAD_DAYS)),
            dtype=np.float32,
        )
    return result


def _baseline_metrics(
    state: Any,
    records: np.ndarray,
    climatology_state: np.ndarray,
    climatology_fields: Mapping[str, np.ndarray],
    wet: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    count = int(records.shape[0])
    result = {
        method: _empty_metrics(count)
        for method in ("persistence", "climatology")
    }
    for offset in range(0, count, batch_size):
        stop = min(offset + batch_size, count)
        chunk = records[offset:stop]
        experiments = chunk[:, 0].astype(int)
        starts = chunk[:, 1].astype(int)
        initial = np.stack(
            [
                np.asarray(state[e, t], dtype=np.float32)
                for e, t in zip(experiments, starts)
            ]
        )
        initial_fields = reduced_fields(initial, wet)
        climate = reduced_fields(
            climatology_state[experiments],
            wet,
        )
        for name, values in climatology_fields.items():
            climate[name] = values[experiments]
        for lead_index, lead in enumerate(LEAD_DAYS):
            truth = np.stack(
                [
                    np.asarray(
                        state[e, t + lead],
                        dtype=np.float32,
                    )
                    for e, t in zip(experiments, starts)
                ]
            )
            truth_fields = reduced_fields(truth, wet)
            for method, prediction in (
                ("persistence", initial_fields),
                ("climatology", climate),
            ):
                for field in PRIMARY_FIELDS:
                    result[method][f"rmse__{field}"][
                        offset:stop,
                        lead_index,
                    ] = _member_rmse(
                        prediction[field],
                        truth_fields[field],
                        wet,
                    )
                    result[method][f"acc__{field}"][
                        offset:stop,
                        lead_index,
                    ] = _member_acc(
                        prediction[field],
                        truth_fields[field],
                        climate[field],
                        wet,
                    )
    return result


def _model_metrics(
    stepper: ReducedDirectStepper,
    state: Any,
    static: Any,
    records: np.ndarray,
    climatology_fields: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    count = int(records.shape[0])
    result = _empty_metrics(count)
    result["finite"] = np.empty(
        (count, len(LEAD_DAYS)),
        dtype=np.uint8,
    )
    result["normalized_land_max_abs"] = np.empty(
        (count, len(LEAD_DAYS)),
        dtype=np.float32,
    )
    land = ~stepper.wet
    for offset in range(0, count, batch_size):
        stop = min(offset + batch_size, count)
        chunk = records[offset:stop]
        experiments = chunk[:, 0].astype(int)
        starts = chunk[:, 1].astype(int)
        initial = np.stack(
            [
                np.asarray(state[e, t], dtype=np.float32)
                for e, t in zip(experiments, starts)
            ]
        )
        current = stepper.normalized_state(initial)
        forcing = stepper.normalized_static(static, experiments)
        climate = {
            name: values[experiments]
            for name, values in climatology_fields.items()
        }
        with torch.no_grad():
            for lead_index, lead in enumerate(LEAD_DAYS):
                current = stepper.step(current, forcing)
                prediction = stepper.physical(current)
                truth = np.stack(
                    [
                        np.asarray(
                            state[e, t + lead],
                            dtype=np.float32,
                        )
                        for e, t in zip(experiments, starts)
                    ]
                )
                prediction_fields = reduced_fields(
                    prediction,
                    stepper.wet,
                )
                truth_fields = reduced_fields(truth, stepper.wet)
                for field in PRIMARY_FIELDS:
                    result[f"rmse__{field}"][
                        offset:stop,
                        lead_index,
                    ] = _member_rmse(
                        prediction_fields[field],
                        truth_fields[field],
                        stepper.wet,
                    )
                    result[f"acc__{field}"][
                        offset:stop,
                        lead_index,
                    ] = _member_acc(
                        prediction_fields[field],
                        truth_fields[field],
                        climate[field],
                        stepper.wet,
                    )
                result["finite"][offset:stop, lead_index] = np.isfinite(
                    prediction
                ).all(axis=(1, 2, 3))
                if np.any(land):
                    result["normalized_land_max_abs"][
                        offset:stop,
                        lead_index,
                    ] = (
                        torch.amax(
                            torch.abs(current[:, :, land]),
                            dim=(1, 2),
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                else:
                    result["normalized_land_max_abs"][
                        offset:stop,
                        lead_index,
                    ] = 0.0
    return result


def _lead_curves(
    metrics: Mapping[str, np.ndarray],
    baselines: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    result = {}
    for field in PRIMARY_FIELDS:
        result[field] = {}
        model_rmse = np.asarray(metrics[f"rmse__{field}"], dtype=np.float64)
        for baseline, baseline_metrics in baselines.items():
            reference = np.asarray(
                baseline_metrics[f"rmse__{field}"],
                dtype=np.float64,
            )
            model_mean = model_rmse.mean(axis=0)
            reference_mean = reference.mean(axis=0)
            result[field][baseline] = {
                "rmse_ratio_by_lead": np.divide(
                    model_mean,
                    reference_mean,
                    out=np.full_like(model_mean, np.inf),
                    where=reference_mean > 0.0,
                ).tolist(),
                "rmse_auc_ratio": float(
                    curve_auc(model_rmse).mean()
                    / curve_auc(reference).mean()
                ),
            }
    return result


def _checkpoint_gate(
    curves: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    metrics: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    auc_ratios = [
        float(curves[field][baseline]["rmse_auc_ratio"])
        for field in PRIMARY_FIELDS
        for baseline in ("persistence", "climatology")
    ]
    slow_ratios = [
        float(value)
        for field in SLOW_PRIMARY_FIELDS
        for baseline in ("persistence", "climatology")
        for value in curves[field][baseline]["rmse_ratio_by_lead"]
    ]
    ten_day = bool(
        diagnostic["every_regime_and_group_beats_persistence"]
    )
    finite = bool(np.all(np.asarray(metrics["finite"]) == 1))
    land_max = float(
        np.max(np.abs(metrics["normalized_land_max_abs"]))
    )
    auc_pass = all(value < 1.0 for value in auc_ratios)
    slow_pass = all(value < 1.0 for value in slow_ratios)
    return {
        "passed": ten_day and finite and land_max == 0.0 and auc_pass and slow_pass,
        "old_ten_day_regime_group_gate_passed": ten_day,
        "rollout_finite": finite,
        "zero_land_leakage": land_max == 0.0,
        "normalized_land_max_abs": land_max,
        "primary_rmse_auc_passed": auc_pass,
        "slow_field_all_leads_passed": slow_pass,
        "worst_primary_rmse_auc_ratio": float(max(auc_ratios)),
        "worst_slow_field_lead_ratio": float(max(slow_ratios)),
    }


def _selection(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ranking = [
        {
            "optimizer_step": int(summary["optimizer_step"]),
            "passed": bool(summary["checkpoint_gate"]["passed"]),
            "selection_key": [
                float(
                    summary["checkpoint_gate"][
                        "worst_slow_field_lead_ratio"
                    ]
                ),
                float(
                    summary["checkpoint_gate"][
                        "worst_primary_rmse_auc_ratio"
                    ]
                ),
                float(
                    summary["ten_day_diagnostic"][
                        "worst_per_regime_group_ratio"
                    ]
                ),
                int(summary["optimizer_step"]),
            ],
        }
        for summary in summaries
    ]
    ranking.sort(key=lambda value: tuple(value["selection_key"]))
    passing = [value for value in ranking if value["passed"]]
    selected = passing[0] if passing else ranking[0]
    return {
        "classification": (
            "training_only_gate_passed"
            if passing
            else "training_only_gate_failed_diagnostic_checkpoint_only"
        ),
        "passed": bool(passing),
        "selected_optimizer_step": int(selected["optimizer_step"]),
        "ranking": ranking,
    }


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the source-locked Arm-R experiment contract."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or tuple(contract.get("reduced_state", {}).get("channels", ()))
        != REDUCED_CHANNELS
    ):
        raise ValueError("Arm-R experiment contract changed")
    architecture = ReducedChannelArchitecture(**contract["architecture"])
    if architecture.to_dict() != contract["architecture"]:
        raise ValueError("Arm-R architecture changed")
    training = contract["training"]
    if (
        int(training.get("seed", -1)) != 20260724
        or tuple(training.get("checkpoint_steps", ()))
        != (3840, 7680, 11520, 13440, 14400, 14880, 15360)
        or int(training["optimizer"].get("maximum_steps", -1)) != 15360
    ):
        raise ValueError("Arm-R training schedule changed")
    read = contract["read_contract"]
    if (
        read.get("training_state") is not True
        or read.get("held_s0_state_after_selection") is not True
        or any(
            read.get(name) is not False
            for name in (
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
            )
        )
    ):
        raise ValueError("Arm-R read contract changed")
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract["source_hashes"].items():
            source = root / relative
            if not source.is_file() or file_sha256(source) != expected:
                raise ValueError(f"Arm-R source changed: {relative}")
    return contract, resolved, file_sha256(resolved)


def _verify_dataset(
    contract: Mapping[str, Any],
    dataset: Path,
    quality: Path,
) -> None:
    source = contract["source_artifacts"]["reduced_dataset"]
    if (
        not dataset.is_dir()
        or file_sha256(dataset / ".zmetadata")
        != source["metadata_sha256"]
        or not quality.is_file()
        or file_sha256(quality) != source["quality_sha256"]
    ):
        raise ModelCReducedControlError("reduced dataset provenance changed")
    report = json.loads(quality.read_text())
    if (
        report.get("status") != "valid"
        or report.get("version") != REDUCED_DATASET_VERSION
        or tuple(report.get("channels", ())) != REDUCED_CHANNELS
        or report.get("logical_state_sha256")
        != source["logical_state_sha256"]
    ):
        raise ModelCReducedControlError("reduced dataset quality failed")


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
        array_sha256(times) != spec["training_times_sha256"]
        or array_sha256(records) != spec["records_sha256"]
    ):
        raise ModelCReducedControlError("training audit records changed")
    return records, times, bounds


def _verify_normalizers(
    contract: Mapping[str, Any],
    normalizers: Mapping[str, Any],
) -> None:
    expected = contract["normalization"]["expected"]
    observed = normalizers["summary"]
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ModelCReducedControlError(
                f"reduced normalizer changed: {key}"
            )


def preflight(
    contract_path: str | Path,
    dataset_path: str | Path,
    quality_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify immutable sources, normalizers, records, and output absence."""

    contract, resolved, digest = load_contract(contract_path)
    dataset = Path(dataset_path).resolve()
    quality = Path(quality_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise FileExistsError("Arm-R output already exists")
    _verify_dataset(contract, dataset, quality)
    group = zarr.open_consolidated(str(dataset), mode="r")
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    normalizers = training_pointwise_normalizers(
        group,
        snapshot_codes,
        floor_quantile=float(
            contract["normalization"]["wet_cell_floor_quantile"]
        ),
        absolute_floor=float(
            contract["normalization"]["absolute_scale_floor"]
        ),
    )
    _verify_normalizers(contract, normalizers)
    records, times, _ = _audit_records(
        pair_codes,
        snapshot_codes,
        contract,
    )
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "normalization": normalizers["summary"],
        "audit_records": int(records.shape[0]),
        "audit_times": int(times.size),
        "held_s0_state_opened": False,
    }


def run(
    contract_path: str | Path,
    dataset_path: str | Path,
    quality_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train, select, and exactly reload the frozen Arm-R model."""

    require_model_a_runtime()
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    dataset = Path(dataset_path).resolve()
    quality = Path(quality_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError("Arm-R output already exists")
    _verify_dataset(contract, dataset, quality)
    seed = int(contract["training"]["seed"])
    seed_everything(seed)
    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    wet_array = np.asarray(group["wet_mask"][:], dtype=bool)
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    wind = np.asarray(static[:, 0], dtype=np.float32)
    wind_mean = float(wind[:, wet_array].mean())
    wind_scale = float(wind[:, wet_array].std())
    normalizers = training_pointwise_normalizers(
        group,
        snapshot_codes,
        floor_quantile=float(
            contract["normalization"]["wet_cell_floor_quantile"]
        ),
        absolute_floor=float(
            contract["normalization"]["absolute_scale_floor"]
        ),
    )
    _verify_normalizers(contract, normalizers)
    point_mean = normalizers["mean"]
    point_scale = normalizers["scale"]
    increment_values = pointwise_increment_scale(
        group,
        pair_codes,
        point_scale,
    )
    climatology_state, climatology_fields, training_days = (
        _training_climatology(group, snapshot_codes)
    )
    audit_records, audit_times, block_bounds = _audit_records(
        pair_codes,
        snapshot_codes,
        contract,
    )
    baselines = _baseline_metrics(
        state,
        audit_records,
        climatology_state,
        climatology_fields,
        wet_array,
        batch_size=int(contract["training_evaluation"]["batch_size"]),
    )
    training_records = records_for_rollout_split(pair_codes, 1)
    one_step_records = records_for_pair_split(pair_codes, 1)
    diagnostic_records = _sample_records_by_regime(
        one_step_records,
        count_per_regime=int(
            contract["training_evaluation"][
                "checkpoint_records_per_regime"
            ]
        ),
        seed=REFERENCE_DIAGNOSTIC_SEED,
    )

    loss_config = model_c_loss_config("v1")
    if (
        reduced_loss_contract_sha256(loss_config)
        != contract["training"]["loss"]["contract_sha256"]
    ):
        raise ModelCReducedControlError("Arm-R loss contract changed")
    architecture = ReducedChannelArchitecture(**contract["architecture"])
    optimizer_contract = contract["training"]["optimizer"]
    training_dataset = ReducedRolloutDataset(
        dataset,
        training_records,
        point_mean,
        point_scale,
        rollout_steps=loss_config.rollout_steps,
    )
    batch_size = int(optimizer_contract["batch_size"])
    loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            training_dataset,
            batch_size,
            seed,
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    wet = torch.from_numpy(
        wet_array.astype(np.float32)
    )[None, None].to(device)
    boundary_array = western_boundary_mask(
        wet_array,
        loss_config.western_boundary_width,
    )
    boundary = torch.from_numpy(
        boundary_array.astype(np.float32)
    )[None, None].to(device)
    increment_scale = torch.from_numpy(increment_values).to(device)
    model = build_reduced_model(architecture).to(device)
    parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_contract["initial_learning_rate"]),
        betas=tuple(
            float(value)
            for value in optimizer_contract["adam_betas"]
        ),
        weight_decay=float(optimizer_contract["weight_decay"]),
    )
    maximum_steps = int(optimizer_contract["maximum_steps"])
    decay_step = int(
        round(
            maximum_steps
            * float(optimizer_contract["decay_fraction"])
        )
    )
    checkpoint_steps = tuple(
        int(value)
        for value in contract["training"]["checkpoint_steps"]
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    checkpoint_directory = temporary / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir()
    arrays: dict[str, np.ndarray] = {
        "pointwise_mean": point_mean,
        "pointwise_raw_scale": normalizers["raw_scale"],
        "pointwise_scale": point_scale,
        "channel_scale_floor": normalizers["floor"],
        "training_records": audit_records.astype(np.int32),
        "training_times": audit_times.astype(np.int32),
        "training_lead_days": np.asarray(LEAD_DAYS, dtype=np.int16),
        "wet_mask": wet_array.astype(np.uint8),
    }
    for method, values in baselines.items():
        for name, value in values.items():
            arrays[f"training_{method}__{name}"] = np.asarray(value)
    iterator = iter(loader)
    window_totals = {name: 0.0 for name in REDUCED_AUDIT_TERMS}
    window_samples = 0
    history = []
    checkpoint_summaries = []
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
        features = features.to(device=device, dtype=torch.float32)
        futures = futures.to(device=device, dtype=torch.float32)
        model.train()
        predictions = direct_unroll(
            model,
            features,
            wet,
            loss_config.rollout_steps,
        )
        terms = reduced_loss_terms(
            predictions,
            futures,
            features[:, :REDUCED_CHANNEL_COUNT],
            wet,
            boundary,
            increment_scale,
            loss_config,
        )
        if not all(
            bool(torch.isfinite(terms[name]).item())
            for name in REDUCED_AUDIT_TERMS
        ):
            raise ModelCReducedControlError("Arm-R loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        optimizer.step()
        size = int(features.shape[0])
        for name in REDUCED_AUDIT_TERMS:
            window_totals[name] += (
                float(terms[name].detach().cpu()) * size
            )
        window_samples += size
        if step not in checkpoint_steps:
            continue
        training_window = {
            name: window_totals[name] / window_samples
            for name in REDUCED_AUDIT_TERMS
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
        stepper = ReducedDirectStepper(
            model=model,
            device=device,
            wet=wet_array,
            mean=point_mean,
            scale=point_scale,
            wind_mean=wind_mean,
            wind_scale=wind_scale,
        )
        metrics = _model_metrics(
            stepper,
            state,
            static,
            audit_records,
            climatology_fields,
            batch_size=int(
                contract["training_evaluation"]["batch_size"]
            ),
        )
        curves = _lead_curves(metrics, baselines)
        gate = _checkpoint_gate(curves, diagnostic, metrics)
        for name, value in metrics.items():
            arrays[f"training_step_{step}__{name}"] = np.asarray(value)
        record = {
            "optimizer_step": step,
            "optimizer_learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "training_window": training_window,
        }
        history.append(record)
        checkpoint_path = (
            checkpoint_directory
            / f"model_c_reduced_channel_step_{step:05d}.pt"
        )
        payload = {
            "version": VERSION,
            "purpose": "arm_r_reduced_channel_direct_state_checkpoint",
            "optimizer_step": step,
            "architecture": architecture.to_dict(),
            "parameter_count": parameter_count,
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "reduced_channels": list(REDUCED_CHANNELS),
            "normalization": normalizers["summary"],
            "loss_contract": reduced_loss_contract(loss_config),
            "loss_contract_sha256": reduced_loss_contract_sha256(
                loss_config
            ),
            "optimizer_contract": optimizer_contract,
            "training_history_record": record,
            "model_state_dict": _checkpoint_state_dict(model),
        }
        torch.save(payload, checkpoint_path)
        checkpoint_summaries.append(
            {
                "optimizer_step": step,
                "checkpoint": str(
                    output
                    / CHECKPOINT_DIRECTORY
                    / checkpoint_path.name
                ),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "training_window": training_window,
                "ten_day_diagnostic": diagnostic,
                "lead_curves": curves,
                "checkpoint_gate": gate,
            }
        )
        window_totals = {
            name: 0.0 for name in REDUCED_AUDIT_TERMS
        }
        window_samples = 0
    if len(checkpoint_summaries) != len(checkpoint_steps):
        raise ModelCReducedControlError("not every Arm-R checkpoint ran")
    decision = _selection(checkpoint_summaries)
    selected_step = int(decision["selected_optimizer_step"])
    selected_source = (
        checkpoint_directory
        / f"model_c_reduced_channel_step_{selected_step:05d}.pt"
    )
    payload = torch.load(
        selected_source,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    selected_path = temporary / CHECKPOINT_NAME
    torch.save(payload, selected_path)
    reload_dataset = ReducedRolloutDataset(
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
        reference = direct_unroll(
            model,
            reload_features,
            wet,
            9,
        ).cpu()
    restored = build_reduced_model(architecture).to(device)
    loaded = torch.load(
        selected_path,
        map_location=device,
        weights_only=False,
    )
    restored.load_state_dict(loaded["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        reloaded = direct_unroll(
            restored,
            reload_features,
            wet,
            9,
        ).cpu()
    reload_exact = bool(torch.equal(reference, reloaded))
    if not reload_exact:
        raise ModelCReducedControlError("Arm-R reload is not bitwise exact")
    full_one_step = _one_step_diagnostics(
        restored,
        dataset,
        one_step_records,
        point_mean,
        point_scale,
        batch_size=16,
        device=device,
    )
    normalization_path = temporary / NORMALIZATION_NAME
    np.savez_compressed(
        normalization_path,
        pointwise_mean=point_mean,
        pointwise_raw_scale=normalizers["raw_scale"],
        pointwise_scale=point_scale,
        channel_scale_floor=normalizers["floor"],
    )
    arrays_path = temporary / ARRAYS_NAME
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "status": "complete",
        "version": VERSION,
        "purpose": "arm_r_reduced_channel_causal_control",
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset),
        "dataset_metadata_sha256": file_sha256(
            dataset / ".zmetadata"
        ),
        "quality_report": str(quality),
        "quality_report_sha256": file_sha256(quality),
        "device": str(device),
        "architecture": architecture.to_dict(),
        "parameter_count": parameter_count,
        "reduced_channels": list(REDUCED_CHANNELS),
        "normalization": normalizers["summary"],
        "normalization_artifact": str(output / NORMALIZATION_NAME),
        "normalization_artifact_sha256": file_sha256(
            normalization_path
        ),
        "loss_contract": reduced_loss_contract(loss_config),
        "loss_contract_sha256": reduced_loss_contract_sha256(
            loss_config
        ),
        "optimizer_contract": optimizer_contract,
        "optimizer_decay_step": decay_step,
        "seed": seed,
        "counts": {
            "training_rollouts": len(training_records),
            "training_one_step_pairs": len(one_step_records),
            "checkpoint_diagnostic_pairs": len(diagnostic_records),
            "training_evaluation_rollouts": int(
                audit_records.shape[0]
            ),
            "training_climatology_snapshots_per_regime": training_days,
        },
        "training_record_contract": {
            "complete_training_blocks": [
                list(value) for value in block_bounds
            ],
            "training_times_sha256": array_sha256(audit_times),
            "records_sha256": array_sha256(audit_records),
        },
        "increment_scale": increment_values.tolist(),
        "increment_scale_sha256": _float_sha256(increment_values),
        "training_history": history,
        "checkpoint_summary": checkpoint_summaries,
        "selection_decision": decision,
        "selected_full_one_step_training_diagnostic": full_one_step,
        "selected_checkpoint": str(output / CHECKPOINT_NAME),
        "selected_checkpoint_sha256": file_sha256(selected_path),
        "save_reload_nine_step_bitwise_exact": reload_exact,
        "arrays": str(output / ARRAYS_NAME),
        "arrays_sha256": file_sha256(arrays_path),
        "elapsed_seconds": time.monotonic() - started,
        "read_contract": contract["read_contract"],
        "held_s0_state_opened": False,
        "response_or_adjoint_state_opened": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    report["report_content_sha256"] = json_sha256(report)
    report_path = temporary / REPORT_NAME
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "version": VERSION,
        "contract_sha256": contract_sha,
        "report_content_sha256": report["report_content_sha256"],
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        },
    }
    manifest["manifest_content_sha256"] = json_sha256(manifest)
    (temporary / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, output)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        child = commands.add_parser(name)
        child.add_argument("--contract", type=Path, required=True)
        child.add_argument("--dataset", type=Path, required=True)
        child.add_argument("--quality-report", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        if name == "run":
            child.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(
            args.contract,
            args.dataset,
            args.quality_report,
            args.output_dir,
        )
    else:
        result = run(
            args.contract,
            args.dataset,
            args.quality_report,
            args.output_dir,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
