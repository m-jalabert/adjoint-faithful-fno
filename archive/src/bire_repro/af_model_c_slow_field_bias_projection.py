"""Training-only slow-field bias, projection, and predictability audit.

This is the bounded diagnostic that follows the rejected pushforward-duration
and truncated-unroll Model C objectives.  It never trains or alters a
checkpoint.  On the immutable reference width-128 checkpoint it:

* measures teacher-forced signed ten-day increment bias on every split-1 pair;
* compares nine times that bias with the mean day-90 rollout error;
* projects the error onto basin mean plus five training-increment EOFs;
* scores exact SSH-volume and temperature-level-mean output projections,
  with and without subtraction of the static teacher-forced bias field; and
* fits and scores a training-only damped-persistence baseline and estimates
  slow-field anomaly decorrelation times.

Every state read is restricted to split code 1.  Validation and inference
states, intermediate-wind trajectories, responses, and adjoints remain sealed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import zarr

from .af_a0_evaluate import _normalizers
from .af_forward_complete import (
    _member_acc,
    _member_rmse,
    _state_fields,
    _training_climatology,
)
from .af_model_c_overfit import _file_sha256
from .af_model_c_rollout_diagnosis import (
    complete_rollout_starts,
    lead_curve_summary,
    select_balanced_training_times,
)
from .af_model_c_successor_validation import (
    EVALUATION_FIELDS,
    LEAD_DAYS,
    _evaluate_baseline_metrics,
    _load_successor_stepper,
    _method_auc_summary,
)

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


VERSION = "model_c_slow_field_bias_projection_v1"
REPORT_NAME = "model_c_slow_field_bias_projection_report.json"
ARRAYS_NAME = "model_c_slow_field_bias_projection_arrays.npz"
SUMMARY_NAME = "slow_field_bias_projection_summary.json"
MANIFEST_NAME = "figure_manifest.json"
REFERENCE_SEED = 20260723
HORIZON_DAYS = 10
SLOW_CHANNELS = slice(30, 46)
SST_CHANNEL = 30
SSH_CHANNEL = 45
TEMPERATURE_CHANNELS = tuple(range(30, 45))
TEMPERATURE_SLICE = slice(30, 45)
EOF_FIELDS = {"sst": SST_CHANNEL, "ssh": SSH_CHANNEL}
VARIANTS = (
    "raw",
    "ssh_zero_mean",
    "conservation_means",
    "static_bias",
    "static_bias_conservation",
)
CONSERVATION_VARIANTS = (
    "ssh_zero_mean",
    "conservation_means",
    "static_bias_conservation",
)
STATIC_BIAS_VARIANTS = ("static_bias", "static_bias_conservation")
TEMPERATURE_MEAN_VARIANTS = (
    "conservation_means",
    "static_bias_conservation",
)


class SlowFieldBiasProjectionError(RuntimeError):
    """Raised when the frozen audit contract or provenance is violated."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def load_bias_projection_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the contract frozen before any bias/projection metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != VERSION:
        raise ValueError(f"expected contract version {VERSION}")
    if (
        contract.get("contract_status")
        != "frozen_after_truncated_unroll_rejection_before_bias_or_projection_metrics"
    ):
        raise ValueError("slow-field bias/projection contract is not frozen")
    records = contract.get("rollout_records", {})
    if (
        tuple(records.get("lead_days", ())) != LEAD_DAYS
        or int(records.get("starts_per_training_block", -1)) != 90
        or int(records.get("expected_training_blocks", -1)) != 2
        or int(records.get("records_total", -1)) != 540
        or records.get("selection")
        != "same_evenly_spaced_complete_split1_starts_as_job_291102"
    ):
        raise ValueError("rollout-record contract changed")
    projection = contract.get("posthoc_projection", {})
    if (
        tuple(projection.get("variants", ())) != VARIANTS
        or projection.get("ssh_target")
        != "zero_cosine_latitude_weighted_wet_area_mean_increment"
        or projection.get("temperature_target")
        != "per_regime_per_level_training_truth_mean_increment"
        or projection.get("checkpoint_weights_changed") is not False
    ):
        raise ValueError("post-hoc projection contract changed")
    eof = contract.get("eof_analysis", {})
    if (
        tuple(eof.get("fields", ())) != tuple(EOF_FIELDS)
        or int(eof.get("modes", -1)) != 5
        or int(eof.get("oversampling", -1)) != 5
        or int(eof.get("seed", -1)) != 20260728
    ):
        raise ValueError("EOF contract changed")
    predictability = contract.get("predictability", {})
    if (
        int(predictability.get("fit_lag_days", -1)) != 10
        or int(predictability.get("maximum_decorrelation_lag_days", -1))
        != 720
        or predictability.get("alpha_bounds") != [0.0, 1.0]
        or predictability.get("evaluation_records")
        != "same_540_split1_rollouts"
    ):
        raise ValueError("predictability contract changed")
    read = contract.get("read_contract", {})
    if (
        read.get("training_pair_code") != 1
        or read.get("training_state_read") is not True
        or read.get("prior_report_metadata_read") is not True
        or any(
            read.get(name) is not False
            for name in (
                "validation_state_read",
                "inference_state_read",
                "intermediate_wind_read",
                "response_read",
                "adjoint_read",
            )
        )
    ):
        raise ValueError("read contract changed")
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(f"bias/projection source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def wet_area_weights(latitude_deg: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Return normalized regular-lat/lon cell-area weights over wet cells."""

    latitude = np.asarray(latitude_deg, dtype=np.float64)
    wet = np.asarray(wet, dtype=bool)
    if latitude.shape != wet.shape or latitude.ndim != 2:
        raise ValueError("latitude and wet mask must be matching 2-D arrays")
    value = np.cos(np.deg2rad(latitude))
    value[~wet] = 0.0
    total = float(np.sum(value))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("wet-area weights have no positive mass")
    return value / total


def _weighted_field_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Area mean over the final two dimensions."""

    return np.sum(
        np.asarray(values, dtype=np.float64)
        * np.asarray(weights, dtype=np.float64),
        axis=(-2, -1),
    )


def pattern_metrics(
    linear_bias: np.ndarray,
    rollout_error: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Compare a fixed linear-bias prediction with a rollout-error map."""

    left = np.asarray(linear_bias, dtype=np.float64)
    right = np.asarray(rollout_error, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if left.shape != weights.shape or right.shape != weights.shape:
        raise ValueError("pattern fields and weights must have matching shapes")
    left_mean = float(np.sum(left * weights))
    right_mean = float(np.sum(right * weights))
    left_energy = float(np.sum(np.square(left) * weights))
    right_energy = float(np.sum(np.square(right) * weights))
    cross = float(np.sum(left * right * weights))
    epsilon = np.finfo(np.float64).eps
    cosine = cross / max(np.sqrt(left_energy * right_energy), epsilon)
    left_centered = left - left_mean
    right_centered = right - right_mean
    centered_cross = float(np.sum(left_centered * right_centered * weights))
    centered_energy = float(
        np.sqrt(
            np.sum(np.square(left_centered) * weights)
            * np.sum(np.square(right_centered) * weights)
        )
    )
    residual_energy = float(
        np.sum(np.square(right - left) * weights)
    )
    optimal_scale = cross / max(left_energy, epsilon)
    optimal_residual = float(
        np.sum(np.square(right - optimal_scale * left) * weights)
    )
    return {
        "weighted_cosine": float(cosine),
        "weighted_centered_correlation": float(
            centered_cross / max(centered_energy, epsilon)
        ),
        "fixed_amplitude_explained_energy_fraction": float(
            1.0 - residual_energy / max(right_energy, epsilon)
        ),
        "optimal_amplitude_scale": float(optimal_scale),
        "optimal_amplitude_explained_energy_fraction": float(
            1.0 - optimal_residual / max(right_energy, epsilon)
        ),
        "rollout_error_basin_mean": right_mean,
        "linear_bias_basin_mean": left_mean,
        "rollout_error_basin_mean_energy_fraction": float(
            np.square(right_mean) / max(right_energy, epsilon)
        ),
        "rollout_error_weighted_rms": float(np.sqrt(right_energy)),
        "linear_bias_weighted_rms": float(np.sqrt(left_energy)),
    }


def first_efold_time(
    lead_days: np.ndarray,
    correlations: np.ndarray,
) -> float | None:
    """Linearly interpolate the first crossing of exp(-1)."""

    lead = np.asarray(lead_days, dtype=np.float64)
    values = np.asarray(correlations, dtype=np.float64)
    if lead.ndim != 1 or values.shape != lead.shape or lead.size < 2:
        raise ValueError("decorrelation curve must be matching 1-D arrays")
    threshold = float(np.exp(-1.0))
    crossing = np.flatnonzero(values <= threshold)
    if not crossing.size:
        return None
    index = int(crossing[0])
    if index == 0:
        return float(lead[0])
    x0, x1 = float(lead[index - 1]), float(lead[index])
    y0, y1 = float(values[index - 1]), float(values[index])
    if y1 == y0:
        return x1
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def apply_increment_projection(
    increment_normalized: Any,
    experiments: np.ndarray,
    *,
    variant: str,
    state_scale: Any,
    area_weights: Any,
    wet: Any,
    truth_mean_tendency: Any,
    bias_field: Any,
) -> Any:
    """Apply one declared linear/affine correction to a normalized increment."""

    if torch is None:
        raise RuntimeError("projection requires PyTorch")
    if variant not in VARIANTS:
        raise ValueError(f"unknown projection variant {variant}")
    experiments = np.asarray(experiments, dtype=np.int64)
    if increment_normalized.shape[0] != experiments.size:
        raise ValueError("experiment count does not match increment batch")
    physical = increment_normalized * state_scale
    experiment_tensor = torch.as_tensor(
        experiments,
        dtype=torch.long,
        device=physical.device,
    )
    if variant in STATIC_BIAS_VARIANTS:
        physical = physical.clone()
        physical[:, SLOW_CHANNELS] -= bias_field[experiment_tensor]
    if variant in CONSERVATION_VARIANTS:
        if variant not in STATIC_BIAS_VARIANTS:
            physical = physical.clone()
        ssh_mean = torch.sum(
            physical[:, SSH_CHANNEL] * area_weights,
            dim=(-2, -1),
        )
        physical[:, SSH_CHANNEL] -= (
            ssh_mean[:, None, None] * wet
        )
    if variant in TEMPERATURE_MEAN_VARIANTS:
        temperature = physical[:, TEMPERATURE_SLICE]
        present_mean = torch.sum(
            temperature * area_weights[None, None],
            dim=(-2, -1),
        )
        target = truth_mean_tendency[
            experiment_tensor, : len(TEMPERATURE_CHANNELS)
        ]
        temperature += (
            (target - present_mean)[:, :, None, None]
            * wet[None, None]
        )
    return physical / state_scale


def _split_blocks(indices: np.ndarray) -> tuple[np.ndarray, ...]:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not indices.size:
        raise ValueError("split indices must be a nonempty vector")
    cuts = np.flatnonzero(np.diff(indices) != 1) + 1
    return tuple(np.split(indices, cuts))


def _iter_pair_chunks(
    pair_blocks: Sequence[np.ndarray],
    *,
    chunk: int,
) -> Iterable[tuple[int, int]]:
    for block in pair_blocks:
        first = int(block[0])
        stop = int(block[-1]) + 1
        for start in range(first, stop, chunk):
            yield start, min(start + chunk, stop)


def _teacher_forced_audit(
    stepper: Any,
    state: Any,
    static: Any,
    pair_blocks: Sequence[np.ndarray],
    climatology_state: np.ndarray,
    area_weights: np.ndarray,
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Measure split-1 signed bias and fit channelwise AR(1) coefficients."""

    sums_prediction = np.zeros((3, 16, *area_weights.shape), dtype=np.float64)
    sums_truth = np.zeros_like(sums_prediction)
    sums_error = np.zeros_like(sums_prediction)
    error_mse = np.zeros((3, 16), dtype=np.float64)
    alpha_xy = np.zeros((3, 46), dtype=np.float64)
    alpha_xx = np.zeros((3, 46), dtype=np.float64)
    count = np.zeros(3, dtype=np.int64)
    weights = area_weights[None, None]
    scale = stepper.scale[None, :, None, None]

    for experiment in range(3):
        forcing = stepper.normalized_static(
            static,
            np.asarray([experiment], dtype=np.int64),
        )
        for start, stop in _iter_pair_chunks(pair_blocks, chunk=batch_size):
            current_raw = np.asarray(
                state[experiment, start:stop],
                dtype=np.float32,
            )
            future_raw = np.asarray(
                state[
                    experiment,
                    start + HORIZON_DAYS : stop + HORIZON_DAYS,
                ],
                dtype=np.float32,
            )
            current = stepper.normalized_state(current_raw)
            batch_forcing = forcing.expand(current.shape[0], -1, -1, -1)
            with torch.no_grad():
                predicted_increment_z = stepper.model(
                    torch.cat((current, batch_forcing), dim=1)
                )
            predicted_increment = (
                predicted_increment_z.detach().cpu().numpy() * scale
            )
            truth_increment = future_raw - current_raw
            error = predicted_increment[:, SLOW_CHANNELS] - truth_increment[
                :, SLOW_CHANNELS
            ]
            sums_prediction[experiment] += np.sum(
                predicted_increment[:, SLOW_CHANNELS],
                axis=0,
                dtype=np.float64,
            )
            sums_truth[experiment] += np.sum(
                truth_increment[:, SLOW_CHANNELS],
                axis=0,
                dtype=np.float64,
            )
            sums_error[experiment] += np.sum(
                error,
                axis=0,
                dtype=np.float64,
            )
            error_mse[experiment] += np.sum(
                np.square(error, dtype=np.float64) * weights,
                axis=(0, 2, 3),
            )
            anomaly_current = (
                current_raw - climatology_state[experiment][None]
            ).astype(np.float64)
            anomaly_future = (
                future_raw - climatology_state[experiment][None]
            ).astype(np.float64)
            alpha_xy[experiment] += np.sum(
                anomaly_current * anomaly_future * weights,
                axis=(0, 2, 3),
            )
            alpha_xx[experiment] += np.sum(
                np.square(anomaly_current) * weights,
                axis=(0, 2, 3),
            )
            count[experiment] += current_raw.shape[0]

    prediction_mean = sums_prediction / count[:, None, None, None]
    truth_mean = sums_truth / count[:, None, None, None]
    bias = sums_error / count[:, None, None, None]
    mse = error_mse / count[:, None]
    bias_energy = np.sum(
        np.square(bias) * area_weights[None, None],
        axis=(-2, -1),
    )
    alpha_unclipped = np.divide(
        alpha_xy,
        alpha_xx,
        out=np.zeros_like(alpha_xy),
        where=alpha_xx > 0.0,
    )
    alpha = np.clip(alpha_unclipped, 0.0, 1.0)
    arrays = {
        "teacher_prediction_mean": prediction_mean.astype(np.float32),
        "teacher_truth_mean": truth_mean.astype(np.float32),
        "teacher_bias_field": bias.astype(np.float32),
        "teacher_error_mse": mse.astype(np.float64),
        "teacher_bias_energy": bias_energy.astype(np.float64),
        "damped_alpha_unclipped": alpha_unclipped.astype(np.float64),
        "damped_alpha": alpha.astype(np.float64),
        "teacher_pair_count_by_regime": count,
    }
    summary = {
        "pair_count_by_regime": count.tolist(),
        "bias_energy_fraction_by_regime_and_slow_channel": np.divide(
            bias_energy,
            mse,
            out=np.zeros_like(bias_energy),
            where=mse > 0.0,
        ).tolist(),
        "damped_alpha_by_regime_and_channel": alpha.tolist(),
        "damped_alpha_unclipped_by_regime_and_channel": (
            alpha_unclipped.tolist()
        ),
        "truth_ssh_increment_area_mean_by_regime_m": _weighted_field_mean(
            truth_mean[:, -1],
            area_weights,
        ).tolist(),
        "predicted_ssh_increment_area_mean_by_regime_m": (
            _weighted_field_mean(prediction_mean[:, -1], area_weights).tolist()
        ),
        "temperature_truth_area_mean_tendency_by_regime_and_level_degC": (
            _weighted_field_mean(truth_mean[:, :15], area_weights).tolist()
        ),
        "temperature_predicted_area_mean_tendency_by_regime_and_level_degC": (
            _weighted_field_mean(
                prediction_mean[:, :15],
                area_weights,
            ).tolist()
        ),
    }
    return arrays, summary


def _empty_metrics(count: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for field in EVALUATION_FIELDS:
        result[f"rmse__{field}"] = np.empty(
            (count, len(LEAD_DAYS)),
            dtype=np.float32,
        )
        result[f"acc__{field}"] = np.empty(
            (count, len(LEAD_DAYS)),
            dtype=np.float32,
        )
    return result


def _score_fields(
    metrics: dict[str, np.ndarray],
    prediction: np.ndarray,
    truth: np.ndarray,
    climate_fields: Mapping[str, np.ndarray],
    wet: np.ndarray,
    member_slice: slice,
    lead_index: int,
) -> None:
    prediction_fields = _state_fields(prediction, wet)
    truth_fields = _state_fields(truth, wet)
    for field in EVALUATION_FIELDS:
        metrics[f"rmse__{field}"][member_slice, lead_index] = _member_rmse(
            prediction_fields[field],
            truth_fields[field],
            wet,
        )
        metrics[f"acc__{field}"][member_slice, lead_index] = _member_acc(
            prediction_fields[field],
            truth_fields[field],
            climate_fields[field],
            wet,
        )


def _climate_fields(
    climatology_state: np.ndarray,
    experiments: np.ndarray,
    wet: np.ndarray,
) -> dict[str, np.ndarray]:
    return _state_fields(climatology_state[experiments], wet)


def _evaluate_damped_persistence(
    state: Any,
    records: np.ndarray,
    climatology_state: np.ndarray,
    alpha: np.ndarray,
    wet: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    metrics = _empty_metrics(records.shape[0])
    for offset in range(0, records.shape[0], batch_size):
        stop = min(offset + batch_size, records.shape[0])
        chunk = records[offset:stop]
        experiments = chunk[:, 0].astype(int)
        starts = chunk[:, 1].astype(int)
        initial = np.stack(
            [
                np.asarray(state[experiment, start], dtype=np.float32)
                for experiment, start in zip(experiments, starts)
            ]
        )
        climate = climatology_state[experiments]
        climate_fields = _climate_fields(
            climatology_state,
            experiments,
            wet,
        )
        for lead_index, lead in enumerate(LEAD_DAYS):
            factor = np.power(
                alpha[experiments],
                lead_index + 1,
            )[:, :, None, None]
            prediction = climate + factor * (initial - climate)
            prediction[:, :, ~wet] = 0.0
            truth = np.stack(
                [
                    np.asarray(
                        state[experiment, start + lead],
                        dtype=np.float32,
                    )
                    for experiment, start in zip(experiments, starts)
                ]
            )
            _score_fields(
                metrics,
                prediction,
                truth,
                climate_fields,
                wet,
                slice(offset, stop),
                lead_index,
            )
    return metrics


def _evaluate_projection_variants(
    stepper: Any,
    state: Any,
    static: Any,
    records: np.ndarray,
    climatology_state: np.ndarray,
    area_weights: np.ndarray,
    truth_mean_tendency: np.ndarray,
    bias_field: np.ndarray,
    *,
    batch_size: int,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    count = records.shape[0]
    metrics = {variant: _empty_metrics(count) for variant in VARIANTS}
    ssh_increment_mean = {
        variant: np.empty((count, len(LEAD_DAYS)), dtype=np.float32)
        for variant in VARIANTS
    }
    temperature_mean_target_rmse = {
        variant: np.empty((count, len(LEAD_DAYS)), dtype=np.float32)
        for variant in VARIANTS
    }
    day90_error_sum = {
        variant: np.zeros(
            (3, 16, *area_weights.shape),
            dtype=np.float64,
        )
        for variant in VARIANTS
    }
    day90_count = np.zeros(3, dtype=np.int64)
    wet_t = torch.as_tensor(
        stepper.wet,
        dtype=torch.bool,
        device=stepper.device,
    )
    area_t = torch.as_tensor(
        area_weights,
        dtype=torch.float32,
        device=stepper.device,
    )
    scale_t = torch.as_tensor(
        stepper.scale[None, :, None, None],
        dtype=torch.float32,
        device=stepper.device,
    )
    truth_tendency_t = torch.as_tensor(
        _weighted_field_mean(truth_mean_tendency, area_weights),
        dtype=torch.float32,
        device=stepper.device,
    )
    bias_t = torch.as_tensor(
        bias_field,
        dtype=torch.float32,
        device=stepper.device,
    )

    for offset in range(0, count, batch_size):
        stop = min(offset + batch_size, count)
        member_slice = slice(offset, stop)
        chunk = records[offset:stop]
        experiments = chunk[:, 0].astype(int)
        starts = chunk[:, 1].astype(int)
        initial = np.stack(
            [
                np.asarray(state[experiment, start], dtype=np.float32)
                for experiment, start in zip(experiments, starts)
            ]
        )
        initial_z = stepper.normalized_state(initial)
        forcing = stepper.normalized_static(static, experiments)
        current = {
            variant: initial_z.clone()
            for variant in VARIANTS
        }
        climate_fields = _climate_fields(
            climatology_state,
            experiments,
            stepper.wet,
        )
        with torch.no_grad():
            for lead_index, lead in enumerate(LEAD_DAYS):
                truth = np.stack(
                    [
                        np.asarray(
                            state[experiment, start + lead],
                            dtype=np.float32,
                        )
                        for experiment, start in zip(experiments, starts)
                    ]
                )
                for variant in VARIANTS:
                    increment_z = stepper.model(
                        torch.cat((current[variant], forcing), dim=1)
                    )
                    increment_z = apply_increment_projection(
                        increment_z,
                        experiments,
                        variant=variant,
                        state_scale=scale_t,
                        area_weights=area_t,
                        wet=wet_t,
                        truth_mean_tendency=truth_tendency_t,
                        bias_field=bias_t,
                    )
                    current[variant] = current[variant] + increment_z
                    current[variant][:, :, ~wet_t] = 0.0
                    physical_increment = (
                        increment_z * scale_t
                    ).detach().cpu().numpy()
                    ssh_increment_mean[variant][
                        member_slice, lead_index
                    ] = _weighted_field_mean(
                        physical_increment[:, SSH_CHANNEL],
                        area_weights,
                    )
                    present_temperature_mean = _weighted_field_mean(
                        physical_increment[:, TEMPERATURE_SLICE],
                        area_weights,
                    )
                    target = _weighted_field_mean(
                        truth_mean_tendency[
                            experiments, : len(TEMPERATURE_CHANNELS)
                        ],
                        area_weights,
                    )
                    temperature_mean_target_rmse[variant][
                        member_slice, lead_index
                    ] = np.sqrt(
                        np.mean(
                            np.square(present_temperature_mean - target),
                            axis=1,
                        )
                    )
                    prediction = stepper.physical(current[variant])
                    _score_fields(
                        metrics[variant],
                        prediction,
                        truth,
                        climate_fields,
                        stepper.wet,
                        member_slice,
                        lead_index,
                    )
                    if lead_index == len(LEAD_DAYS) - 1:
                        error = (
                            prediction[:, SLOW_CHANNELS]
                            - truth[:, SLOW_CHANNELS]
                        )
                        for local, experiment in enumerate(experiments):
                            day90_error_sum[variant][experiment] += error[
                                local
                            ]
                if lead_index == len(LEAD_DAYS) - 1:
                    for experiment in experiments:
                        day90_count[experiment] += 1

    day90_mean = {
        variant: value / day90_count[:, None, None, None]
        for variant, value in day90_error_sum.items()
    }
    constraint_summary = {}
    for variant in VARIANTS:
        constraint_summary[variant] = {
            "maximum_absolute_ssh_increment_area_mean_m": float(
                np.max(np.abs(ssh_increment_mean[variant]))
            ),
            "temperature_level_mean_target_rmse_degC": float(
                np.sqrt(
                    np.mean(
                        np.square(temperature_mean_target_rmse[variant])
                    )
                )
            ),
        }
    arrays = {
        **{
            f"{variant}__ssh_increment_area_mean": value
            for variant, value in ssh_increment_mean.items()
        },
        **{
            f"{variant}__temperature_mean_target_rmse": value
            for variant, value in temperature_mean_target_rmse.items()
        },
        **{
            f"{variant}__day90_mean_error": value.astype(np.float32)
            for variant, value in day90_mean.items()
        },
        "day90_record_count_by_regime": day90_count,
    }
    return metrics, arrays, constraint_summary


def _truth_increment_chunks(
    state: Any,
    pair_blocks: Sequence[np.ndarray],
    channel: int,
    *,
    chunk: int,
) -> Iterable[np.ndarray]:
    for experiment in range(3):
        for start, stop in _iter_pair_chunks(pair_blocks, chunk=chunk):
            current = np.asarray(
                state[experiment, start:stop, channel],
                dtype=np.float32,
            )
            future = np.asarray(
                state[
                    experiment,
                    start + HORIZON_DAYS : stop + HORIZON_DAYS,
                    channel,
                ],
                dtype=np.float32,
            )
            yield future - current


def randomized_increment_eofs(
    state: Any,
    pair_blocks: Sequence[np.ndarray],
    channel: int,
    temporal_mean: np.ndarray,
    area_weights: np.ndarray,
    *,
    modes: int,
    oversampling: int,
    seed: int,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical EOF maps, weighted orthonormal mean+EOF basis, and variance."""

    wet = area_weights > 0.0
    square_root_weight = np.sqrt(area_weights[wet])
    dimension = int(np.sum(wet))
    rank = modes + oversampling
    if rank >= dimension:
        raise ValueError("randomized EOF rank must be below wet dimension")
    sample_count = 3 * sum(int(block.size) for block in pair_blocks)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((dimension, rank))
    y = np.empty((sample_count, rank), dtype=np.float64)
    total_energy = 0.0
    cursor = 0
    for values in _truth_increment_chunks(
        state,
        pair_blocks,
        channel,
        chunk=chunk,
    ):
        centered = (
            values[:, wet].astype(np.float64) - temporal_mean[wet][None]
        )
        weighted = centered * square_root_weight[None]
        stop = cursor + values.shape[0]
        y[cursor:stop] = weighted @ omega
        total_energy += float(np.sum(np.square(weighted)))
        cursor = stop
    q, _ = np.linalg.qr(y, mode="reduced")
    compressed = np.zeros((rank, dimension), dtype=np.float64)
    cursor = 0
    for values in _truth_increment_chunks(
        state,
        pair_blocks,
        channel,
        chunk=chunk,
    ):
        centered = (
            values[:, wet].astype(np.float64) - temporal_mean[wet][None]
        )
        weighted = centered * square_root_weight[None]
        stop = cursor + values.shape[0]
        compressed += q[cursor:stop].T @ weighted
        cursor = stop
    _, singular, vt = np.linalg.svd(compressed, full_matrices=False)
    weighted_eofs = vt[:modes]
    physical_eofs = np.zeros((modes, *wet.shape), dtype=np.float64)
    physical_eofs[:, wet] = weighted_eofs / square_root_weight[None]
    mean_vector = square_root_weight
    candidates = np.concatenate(
        (mean_vector[None], weighted_eofs),
        axis=0,
    )
    orthonormal, _ = np.linalg.qr(candidates.T, mode="reduced")
    explained = np.square(singular[:modes]) / max(
        total_energy,
        np.finfo(np.float64).eps,
    )
    return (
        physical_eofs.astype(np.float32),
        orthonormal.T.astype(np.float64),
        explained.astype(np.float64),
    )


def _basis_projection(
    field: np.ndarray,
    basis_weighted: np.ndarray,
    area_weights: np.ndarray,
) -> dict[str, Any]:
    wet = area_weights > 0.0
    weighted = (
        np.asarray(field, dtype=np.float64)[wet]
        * np.sqrt(area_weights[wet])
    )
    coefficients = basis_weighted @ weighted
    energy = float(np.sum(np.square(weighted)))
    fractions = np.square(coefficients) / max(
        energy,
        np.finfo(np.float64).eps,
    )
    return {
        "coefficients": coefficients.tolist(),
        "energy_fraction_by_component": fractions.tolist(),
        "cumulative_energy_fraction": np.cumsum(fractions).tolist(),
        "component_labels": [
            "basin_mean",
            *[
                f"increment_eof_{index}"
                for index in range(1, basis_weighted.shape[0])
            ],
        ],
    }


def _decorrelation_curves(
    state: Any,
    snapshot_blocks: Sequence[np.ndarray],
    climatology_state: np.ndarray,
    area_weights: np.ndarray,
    *,
    maximum_lag_days: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    lag_days = np.arange(
        0,
        maximum_lag_days + HORIZON_DAYS,
        HORIZON_DAYS,
        dtype=np.int32,
    )
    summary: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {"decorrelation_lag_days": lag_days}
    wet = area_weights > 0.0
    weights = area_weights[wet]
    for field, channel in EOF_FIELDS.items():
        summary[field] = {}
        for experiment in range(3):
            anomaly_blocks = []
            basin_blocks = []
            for block in snapshot_blocks:
                values = np.asarray(
                    state[
                        experiment,
                        slice(
                            int(block[0]),
                            int(block[-1]) + 1,
                            HORIZON_DAYS,
                        ),
                        channel,
                    ],
                    dtype=np.float32,
                )[:, wet].astype(np.float64)
                anomalies = (
                    values
                    - climatology_state[experiment, channel, wet][None]
                )
                anomaly_blocks.append(anomalies)
                basin_blocks.append(anomalies @ weights)
            spatial = np.empty(lag_days.size, dtype=np.float64)
            basin = np.empty_like(spatial)
            for lag_index in range(lag_days.size):
                numerator = 0.0
                left_energy = 0.0
                right_energy = 0.0
                basin_numerator = 0.0
                basin_left = 0.0
                basin_right = 0.0
                for anomalies, basin_series in zip(
                    anomaly_blocks,
                    basin_blocks,
                ):
                    if lag_index >= anomalies.shape[0]:
                        continue
                    left = anomalies[: anomalies.shape[0] - lag_index or None]
                    right = anomalies[lag_index:]
                    numerator += float(
                        np.sum(left * right * weights[None])
                    )
                    left_energy += float(
                        np.sum(np.square(left) * weights[None])
                    )
                    right_energy += float(
                        np.sum(np.square(right) * weights[None])
                    )
                    left_basin = basin_series[
                        : basin_series.size - lag_index or None
                    ]
                    right_basin = basin_series[lag_index:]
                    basin_numerator += float(
                        np.sum(left_basin * right_basin)
                    )
                    basin_left += float(np.sum(np.square(left_basin)))
                    basin_right += float(np.sum(np.square(right_basin)))
                spatial[lag_index] = numerator / max(
                    np.sqrt(left_energy * right_energy),
                    np.finfo(np.float64).eps,
                )
                basin[lag_index] = basin_numerator / max(
                    np.sqrt(basin_left * basin_right),
                    np.finfo(np.float64).eps,
                )
            arrays[f"{field}__S{experiment}__spatial_correlation"] = spatial
            arrays[f"{field}__S{experiment}__basin_correlation"] = basin
            spatial_efold = first_efold_time(lag_days, spatial)
            basin_efold = first_efold_time(lag_days, basin)
            summary[field][f"S{experiment}"] = {
                "spatial_pooled_efold_days": spatial_efold,
                "basin_mean_efold_days": basin_efold,
                "spatial_pooled_exceeds_720_days": spatial_efold is None,
                "basin_mean_exceeds_720_days": basin_efold is None,
            }
    return summary, arrays


def _pattern_summary(
    bias_field: np.ndarray,
    day90_error: np.ndarray,
    area_weights: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, channel in EOF_FIELDS.items():
        slow_index = channel - SLOW_CHANNELS.start
        field_result = {}
        for experiment in range(3):
            field_result[f"S{experiment}"] = pattern_metrics(
                len(LEAD_DAYS) * bias_field[experiment, slow_index],
                day90_error[experiment, slow_index],
                area_weights,
            )
        field_result["pooled"] = pattern_metrics(
            len(LEAD_DAYS) * np.mean(
                bias_field[:, slow_index],
                axis=0,
            ),
            np.mean(day90_error[:, slow_index], axis=0),
            area_weights,
        )
        result[field] = field_result
    temperature = {}
    for level in range(15):
        level_result = {}
        for experiment in range(3):
            level_result[f"S{experiment}"] = pattern_metrics(
                len(LEAD_DAYS) * bias_field[experiment, level],
                day90_error[experiment, level],
                area_weights,
            )
        level_result["pooled"] = pattern_metrics(
            len(LEAD_DAYS) * np.mean(bias_field[:, level], axis=0),
            np.mean(day90_error[:, level], axis=0),
            area_weights,
        )
        temperature[f"level_{level + 1:02d}"] = level_result
    result["temperature_by_level"] = temperature
    return result


def _projection_decision(
    curves: Mapping[str, Mapping[str, Any]],
    patterns: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = contract["decision_rule"]
    bias_threshold = float(
        thresholds["fixed_bias_explained_energy_fraction_min"]
    )
    fixed_explained = {
        field: float(
            patterns[field]["pooled"][
                "fixed_amplitude_explained_energy_fraction"
            ]
        )
        for field in EOF_FIELDS
    }
    bias_dominated = all(
        value >= bias_threshold for value in fixed_explained.values()
    )
    raw_worst = max(
        float(
            curves["raw"][field]["persistence"]["rmse_auc_ratio"]
        )
        for field in ("sst", "phihyd_surface", "ssh")
    )
    conservation_worst = max(
        float(
            curves["conservation_means"][field]["persistence"][
                "rmse_auc_ratio"
            ]
        )
        for field in ("sst", "phihyd_surface", "ssh")
    )
    combined_worst = max(
        float(
            curves["static_bias_conservation"][field]["persistence"][
                "rmse_auc_ratio"
            ]
        )
        for field in ("sst", "phihyd_surface", "ssh")
    )
    conservation_reduction = 1.0 - conservation_worst / raw_worst
    combined_reduction = 1.0 - combined_worst / raw_worst
    material_threshold = float(
        thresholds["material_worst_slow_auc_reduction_min"]
    )
    if bias_dominated:
        classification = "static_teacher_forced_bias_dominated"
        next_action = (
            "contract_loss_v3_with_conservation_projection_bias_penalty_"
            "and_rollout_conditioned_supervision"
        )
    elif max(conservation_reduction, combined_reduction) >= material_threshold:
        classification = "large_scale_drift_materially_correctable"
        next_action = (
            "contract_loss_v3_with_conservation_projection_bias_penalty_"
            "and_rollout_conditioned_supervision"
        )
    else:
        classification = "feedback_amplification_not_explained_by_static_bias"
        next_action = (
            "retain_constraints_but_prioritize_rollout_conditioned_"
            "supervision_over_static_bias_correction"
        )
    return {
        "classification": classification,
        "next_action": next_action,
        "fixed_bias_explained_energy_fraction": fixed_explained,
        "raw_worst_slow_field_rmse_auc_ratio_to_persistence": raw_worst,
        "conservation_worst_slow_field_rmse_auc_ratio_to_persistence": (
            conservation_worst
        ),
        "combined_worst_slow_field_rmse_auc_ratio_to_persistence": (
            combined_worst
        ),
        "conservation_relative_reduction": conservation_reduction,
        "combined_relative_reduction": combined_reduction,
        "thresholds": thresholds,
    }


def _write_figures(
    figure_dir: Path,
    arrays: Mapping[str, np.ndarray],
    report: Mapping[str, Any],
    area_weights: np.ndarray,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created: list[Path] = []
    wet = area_weights > 0.0
    bias = arrays["teacher_bias_field"]
    raw_error = arrays["raw__day90_mean_error"]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(9.0, 7.3),
        constrained_layout=True,
    )
    for row, (field, channel, unit) in enumerate(
        (("SST", 0, "$^\\circ$C"), ("SSH", 15, "m"))
    ):
        linear = len(LEAD_DAYS) * np.mean(bias[:, channel], axis=0)
        actual = np.mean(raw_error[:, channel], axis=0)
        bound = max(
            float(np.max(np.abs(linear[wet]))),
            float(np.max(np.abs(actual[wet]))),
            np.finfo(float).eps,
        )
        for column, (value, title) in enumerate(
            ((linear, "9 × teacher-forced bias"), (actual, "day-90 rollout error"))
        ):
            shown = value.copy()
            shown[~wet] = np.nan
            image = axes[row, column].imshow(
                shown,
                origin="lower",
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
            )
            axes[row, column].set_title(f"{field}: {title}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            figure.colorbar(
                image,
                ax=axes[row, column],
                shrink=0.82,
                label=unit,
            )
    path = figure_dir / "model_c_bias_vs_day90_error_maps.png"
    figure.savefig(path, dpi=190)
    plt.close(figure)
    created.append(path)

    curves = report["variant_lead_curves"]
    baselines = report["baseline_lead_curves"]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.0, 3.9),
        constrained_layout=True,
        sharex=True,
    )
    colors = {
        "raw": "#222222",
        "ssh_zero_mean": "#5B8E7D",
        "conservation_means": "#2F75B5",
        "static_bias": "#B66D0D",
        "static_bias_conservation": "#8E3B8F",
        "damped_persistence": "#777777",
    }
    labels = {
        "raw": "raw checkpoint",
        "ssh_zero_mean": "SSH volume",
        "conservation_means": "SSH + temperature means",
        "static_bias": "static bias",
        "static_bias_conservation": "bias + constraints",
        "damped_persistence": "damped persistence",
    }
    for axis, field in zip(
        axes,
        ("sst", "phihyd_surface", "ssh"),
    ):
        for variant in VARIANTS:
            ratio = curves[variant][field]["persistence"][
                "rmse_ratio_by_lead"
            ]
            axis.plot(
                LEAD_DAYS,
                ratio,
                marker="o",
                markersize=3,
                linewidth=1.35,
                color=colors[variant],
                label=labels[variant],
            )
        damped_ratio = baselines["damped_persistence"][field][
            "persistence"
        ]["rmse_ratio_by_lead"]
        axis.plot(
            LEAD_DAYS,
            damped_ratio,
            linestyle="--",
            linewidth=1.35,
            color=colors["damped_persistence"],
            label=labels["damped_persistence"],
        )
        axis.axhline(1.0, color="#999999", linewidth=0.9)
        axis.set_title(field.replace("_", " "))
        axis.set_xlabel("lead (days)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("RMSE / persistence RMSE")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    path = figure_dir / "model_c_frozen_projection_rmse_ratios.png"
    figure.savefig(path, dpi=190)
    plt.close(figure)
    created.append(path)

    lag_days = arrays["decorrelation_lag_days"]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 3.8),
        constrained_layout=True,
        sharey=True,
    )
    regime_colors = ("#2F75B5", "#B66D0D", "#5B8E7D")
    for axis, field in zip(axes, ("sst", "ssh")):
        for experiment, color in enumerate(regime_colors):
            axis.plot(
                lag_days,
                arrays[f"{field}__S{experiment}__spatial_correlation"],
                color=color,
                linewidth=1.7,
                label=f"S{experiment} spatial",
            )
            axis.plot(
                lag_days,
                arrays[f"{field}__S{experiment}__basin_correlation"],
                color=color,
                linewidth=1.0,
                linestyle="--",
                label=f"S{experiment} basin mean",
            )
        axis.axhline(np.exp(-1.0), color="#777777", linewidth=0.9)
        axis.axvline(90, color="#999999", linewidth=0.9, linestyle=":")
        axis.set_title(field.upper())
        axis.set_xlabel("lag (days)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("training anomaly correlation")
    axes[-1].legend(fontsize=7, ncol=2, frameon=False)
    path = figure_dir / "model_c_training_slow_field_decorrelation.png"
    figure.savefig(path, dpi=190)
    plt.close(figure)
    created.append(path)

    levels = np.arange(1, 16)
    mse = np.mean(arrays["teacher_error_mse"], axis=0)[:15]
    bias_energy = np.mean(arrays["teacher_bias_energy"], axis=0)[:15]
    fractions = np.divide(
        bias_energy,
        mse,
        out=np.zeros_like(bias_energy),
        where=mse > 0.0,
    )
    correlations = np.asarray(
        [
            report["linear_bias_pattern"]["temperature_by_level"][
                f"level_{level:02d}"
            ]["pooled"]["weighted_cosine"]
            for level in levels
        ]
    )
    figure, axis = plt.subplots(
        figsize=(8.2, 4.2),
        constrained_layout=True,
    )
    axis.bar(
        levels - 0.18,
        fractions,
        width=0.36,
        color="#2F75B5",
        label="one-step MSE in mean bias",
    )
    axis.bar(
        levels + 0.18,
        correlations,
        width=0.36,
        color="#B66D0D",
        label="cosine(9×bias, day-90 error)",
    )
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xticks(levels)
    axis.set_xlabel("temperature level (1 = surface)")
    axis.set_ylabel("fraction or correlation")
    axis.set_title("Temperature signed-bias decomposition by depth")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    path = figure_dir / "model_c_temperature_bias_by_level.png"
    figure.savefig(path, dpi=190)
    plt.close(figure)
    created.append(path)
    return created


def _write_csv(
    path: Path,
    curves: Mapping[str, Mapping[str, Any]],
    baseline_curves: Mapping[str, Mapping[str, Any]],
) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "method",
                "field",
                "lead_days",
                "rmse_ratio_to_persistence",
            )
        )
        for variant in VARIANTS:
            for field in ("sst", "phihyd_surface", "ssh"):
                ratios = curves[variant][field]["persistence"][
                    "rmse_ratio_by_lead"
                ]
                for lead, ratio in zip(LEAD_DAYS, ratios):
                    writer.writerow((variant, field, lead, ratio))
        for field in ("sst", "phihyd_surface", "ssh"):
            ratios = baseline_curves["damped_persistence"][field][
                "persistence"
            ]["rmse_ratio_by_lead"]
            for lead, ratio in zip(LEAD_DAYS, ratios):
                writer.writerow(
                    ("damped_persistence", field, lead, ratio)
                )


def run_bias_projection_audit(
    dataset_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    figure_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run the frozen split-1 diagnostic and publish numerical/plot artifacts."""

    if torch is None:
        raise RuntimeError("slow-field audit requires PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_bias_projection_contract(
        contract_path
    )
    dataset = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    figures = Path(figure_dir).resolve()
    temporary_output = output.with_name(output.name + ".tmp")
    temporary_figures = figures.with_name(figures.name + ".tmp")
    for target in (output, figures, temporary_output, temporary_figures):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite audit output: {target}")

    sources = contract["source_artifacts"]
    expected_files = {
        dataset / ".zmetadata": sources["dataset_metadata_sha256"],
        Path(sources["reference_report"]): sources[
            "reference_report_sha256"
        ],
        Path(sources["reference_checkpoint"]): sources[
            "reference_checkpoint_sha256"
        ],
        Path(sources["rollout_diagnosis_report"]): sources[
            "rollout_diagnosis_report_sha256"
        ],
    }
    for artifact, expected in expected_files.items():
        if not artifact.is_file() or _file_sha256(artifact) != expected:
            raise SlowFieldBiasProjectionError(
                f"source artifact changed: {artifact}"
            )
    reference_report = json.loads(
        Path(sources["reference_report"]).read_text()
    )
    rollout_report = json.loads(
        Path(sources["rollout_diagnosis_report"]).read_text()
    )
    if (
        reference_report.get("status") != "complete"
        or int(reference_report.get("seed", -1)) != REFERENCE_SEED
        or reference_report.get("checkpoint_sha256")
        != sources["reference_checkpoint_sha256"]
        or rollout_report.get("status") != "complete"
        or rollout_report.get("diagnostic_interpretation", {}).get(
            "classification"
        )
        != "training_objective_or_checkpoint_gate_mismatch"
        or rollout_report.get("inference_opened") is not False
    ):
        raise SlowFieldBiasProjectionError(
            "source reports are not the expected sealed results"
        )

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA audit requested without a visible GPU")
    device = torch.device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    pair_indices = np.flatnonzero(pair_codes == 1)
    snapshot_indices = np.flatnonzero(snapshot_codes == 1)
    pair_blocks = _split_blocks(pair_indices)
    snapshot_blocks = _split_blocks(snapshot_indices)
    complete = complete_rollout_starts(pair_codes, snapshot_codes)
    times, block_bounds = select_balanced_training_times(
        complete,
        starts_per_block=int(
            contract["rollout_records"]["starts_per_training_block"]
        ),
        expected_blocks=int(
            contract["rollout_records"]["expected_training_blocks"]
        ),
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
        records.shape != (540, 2)
        or _array_sha256(times)
        != contract["rollout_records"]["training_times_sha256"]
        or _array_sha256(records)
        != contract["rollout_records"]["records_sha256"]
    ):
        raise SlowFieldBiasProjectionError(
            "job-291102 rollout records were not reproduced"
        )
    mean, scale, wet, _, wind_mean, wind_scale = _normalizers(group)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float32)
    area_weights = wet_area_weights(latitude, wet)
    climatology_state, climatology_derived, training_days = (
        _training_climatology(state, snapshot_codes, wet)
    )
    checkpoint = Path(sources["reference_checkpoint"]).resolve()
    stepper, payload = _load_successor_stepper(
        checkpoint,
        device,
        wet,
        mean,
        scale,
        wind_mean,
        wind_scale,
        contract["architecture"],
    )
    if int(payload.get("seed", -1)) != REFERENCE_SEED:
        raise SlowFieldBiasProjectionError("reference checkpoint seed changed")
    batch_size = int(contract["evaluation"]["batch_size"])

    teacher_arrays, teacher_summary = _teacher_forced_audit(
        stepper,
        state,
        static,
        pair_blocks,
        climatology_state,
        area_weights,
        batch_size=batch_size,
    )
    baseline_raw = _evaluate_baseline_metrics(
        state,
        records,
        climatology_state,
        climatology_derived,
        wet,
        mean,
        scale,
        batch_size=batch_size,
    )
    baselines: dict[str, Mapping[str, np.ndarray]] = {
        "persistence": baseline_raw["persistence"],
        "climatology": baseline_raw["climatology"],
    }
    damped_metrics = _evaluate_damped_persistence(
        state,
        records,
        climatology_state,
        teacher_arrays["damped_alpha"],
        wet,
        batch_size=batch_size,
    )
    baselines["damped_persistence"] = damped_metrics
    variant_metrics, projection_arrays, constraint_summary = (
        _evaluate_projection_variants(
            stepper,
            state,
            static,
            records,
            climatology_state,
            area_weights,
            teacher_arrays["teacher_truth_mean"],
            teacher_arrays["teacher_bias_field"],
            batch_size=batch_size,
        )
    )
    raw_day90_error = projection_arrays["raw__day90_mean_error"]
    pattern_summary = _pattern_summary(
        teacher_arrays["teacher_bias_field"],
        raw_day90_error,
        area_weights,
    )

    eof_arrays: dict[str, np.ndarray] = {}
    eof_summary: dict[str, Any] = {}
    for field, channel in EOF_FIELDS.items():
        slow_index = channel - SLOW_CHANNELS.start
        temporal_mean = np.mean(
            teacher_arrays["teacher_truth_mean"][:, slow_index],
            axis=0,
        )
        physical_eofs, basis, explained = randomized_increment_eofs(
            state,
            pair_blocks,
            channel,
            temporal_mean,
            area_weights,
            modes=int(contract["eof_analysis"]["modes"]),
            oversampling=int(contract["eof_analysis"]["oversampling"]),
            seed=int(contract["eof_analysis"]["seed"]),
            chunk=int(contract["eof_analysis"]["chunk_size"]),
        )
        eof_arrays[f"{field}__physical_eofs"] = physical_eofs
        eof_arrays[f"{field}__weighted_mean_eof_basis"] = basis
        eof_arrays[f"{field}__truth_increment_explained_fraction"] = explained
        pooled_error = np.mean(
            raw_day90_error[:, slow_index],
            axis=0,
        )
        eof_summary[field] = {
            "truth_increment_explained_fraction_by_eof": explained.tolist(),
            "raw_day90_error_projection": _basis_projection(
                pooled_error,
                basis,
                area_weights,
            ),
        }

    decorrelation_summary, decorrelation_arrays = _decorrelation_curves(
        state,
        snapshot_blocks,
        climatology_state,
        area_weights,
        maximum_lag_days=int(
            contract["predictability"]["maximum_decorrelation_lag_days"]
        ),
    )
    baseline_curves = {
        name: lead_curve_summary(
            metrics,
            {
                "persistence": baselines["persistence"],
                "climatology": baselines["climatology"],
            },
            records,
        )
        for name, metrics in (
            ("damped_persistence", damped_metrics),
        )
    }
    variant_curves = {
        variant: lead_curve_summary(
            metrics,
            baselines,
            records,
        )
        for variant, metrics in variant_metrics.items()
    }
    decision = _projection_decision(
        variant_curves,
        pattern_summary,
        contract,
    )

    arrays: dict[str, np.ndarray] = {
        "records": records.astype(np.int32),
        "training_times": times.astype(np.int32),
        "lead_days": np.asarray(LEAD_DAYS, dtype=np.int16),
        "area_weights": area_weights.astype(np.float64),
        **teacher_arrays,
        **projection_arrays,
        **eof_arrays,
        **decorrelation_arrays,
    }
    for baseline, metrics in baselines.items():
        for name, value in metrics.items():
            arrays[f"{baseline}__{name}"] = np.asarray(value)
    for variant, metrics in variant_metrics.items():
        for name, value in metrics.items():
            arrays[f"{variant}__{name}"] = np.asarray(value)

    temporary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_figures.parent.mkdir(parents=True, exist_ok=True)
    temporary_output.mkdir(exist_ok=False)
    temporary_figures.mkdir(exist_ok=False)
    arrays_path = temporary_output / ARRAYS_NAME
    np.savez_compressed(arrays_path, **arrays)
    report: dict[str, Any] = {
        "status": "complete",
        "version": VERSION,
        "purpose": (
            "training_only_teacher_forced_bias_frozen_projection_and_"
            "predictability_audit"
        ),
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset),
        "dataset_metadata_sha256": _file_sha256(dataset / ".zmetadata"),
        "reference_seed": REFERENCE_SEED,
        "reference_checkpoint": str(checkpoint),
        "reference_checkpoint_sha256": _file_sha256(checkpoint),
        "device": str(device),
        "read_contract": contract["read_contract"],
        "record_contract": {
            "training_pair_count_per_regime": int(pair_indices.size),
            "training_snapshot_count_per_regime": int(
                snapshot_indices.size
            ),
            "training_blocks": [list(value) for value in block_bounds],
            "training_climatology_snapshots_per_regime": training_days,
            "rollout_records": int(records.shape[0]),
            "records_sha256": _array_sha256(records),
            "training_times_sha256": _array_sha256(times),
        },
        "teacher_forced_bias": teacher_summary,
        "linear_bias_pattern": pattern_summary,
        "eof_analysis": eof_summary,
        "constraint_exactness": constraint_summary,
        "baseline_auc": {
            name: _method_auc_summary(metrics)
            for name, metrics in baselines.items()
        },
        "baseline_lead_curves": baseline_curves,
        "variant_auc": {
            variant: _method_auc_summary(metrics)
            for variant, metrics in variant_metrics.items()
        },
        "variant_lead_curves": variant_curves,
        "predictability": {
            "training_only": True,
            "ten_day_alpha_by_regime_and_channel": (
                teacher_arrays["damped_alpha"].tolist()
            ),
            "decorrelation": decorrelation_summary,
            "full_twenty_year_inference_inclusive_analysis_deferred": True,
            "reason": (
                "preserve_the_preexisting_sealed_inference_state_contract"
            ),
        },
        "decision": decision,
        "arrays": str(output / ARRAYS_NAME),
        "arrays_sha256": _file_sha256(arrays_path),
        "elapsed_seconds": time.monotonic() - started,
        "validation_state_opened": False,
        "inference_state_opened": False,
        "intermediate_wind_opened": False,
        "response_or_adjoint_opened": False,
    }
    report["report_content_sha256"] = _json_sha256(report)
    report_path = temporary_output / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    created = _write_figures(
        temporary_figures,
        arrays,
        report,
        area_weights,
    )
    csv_path = (
        temporary_figures
        / "model_c_slow_field_projection_rmse_ratios.csv"
    )
    _write_csv(
        csv_path,
        variant_curves,
        baseline_curves,
    )
    created.append(csv_path)
    summary = {
        "status": report["status"],
        "version": VERSION,
        "contract_sha256": contract_sha,
        "reference_seed": REFERENCE_SEED,
        "decision": decision,
        "constraint_exactness": constraint_summary,
        "predictability": report["predictability"],
        "linear_bias_pattern": {
            field: pattern_summary[field]["pooled"]
            for field in EOF_FIELDS
        },
        "report": str(output / REPORT_NAME),
        "arrays": str(output / ARRAYS_NAME),
    }
    summary_path = temporary_figures / SUMMARY_NAME
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    created.append(summary_path)
    readme = temporary_figures / "README.md"
    readme.write_text(
        "# Model C slow-field bias and projection audit\n\n"
        "Training-only signed-increment bias, frozen conservation/static-bias "
        "projection tests, increment-EOF decomposition, and damped-persistence "
        "predictability diagnostics for reference seed 20260723. Validation "
        "and inference states, response data, and adjoints were not read.\n"
    )
    created.append(readme)
    manifest = {
        "version": VERSION,
        "contract_sha256": contract_sha,
        "files": {
            path.name: {
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in created
        },
    }
    (temporary_figures / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_output, output)
    os.replace(temporary_figures, figures)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_bias_projection_audit(
        args.dataset,
        args.contract,
        args.output_dir,
        args.figure_dir,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
