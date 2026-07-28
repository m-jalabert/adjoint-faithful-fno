"""Fresh-v2 validation and three-seed freeze gate for the Model C successor.

The module has two deliberately separated roles.  ``resolve-seed`` supports
training-only replication of the already selected width-128 architecture.
``evaluate`` opens only the fresh validation block after the complete metric
contract has been frozen.  Inference and all response/adjoint archives remain
sealed.
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

from .af_a0 import a0_architecture
from .af_a0_evaluate import _normalise, _normalizers
from .af_data import STATE_CHANNELS
from .af_forward_complete import (
    BIRE_FIELDS,
    _member_acc,
    _member_rmse,
    _state_fields,
    _training_climatology,
)
from .af_model_c_successor import (
    REPORT_NAME,
    ModelCSuccessorArchitecture,
    build_successor,
)
from .af_model_c_overfit import _file_sha256
from .fno import build_paper_fno

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


VALIDATION_VERSION = "model_c_successor_validation_v2"
VALIDATION_REPORT_NAME = "model_c_successor_validation_report.json"
VALIDATION_ARRAYS_NAME = "model_c_successor_validation_arrays.npz"
HORIZON_DAYS = 10
LEAD_DAYS = tuple(range(10, 91, 10))
PRIMARY_FIELDS = ("surface_speed", "sst", "phihyd_surface")
EVALUATION_FIELDS = (
    "u",
    "v",
    "temperature",
    "ssh",
    *tuple(name for name in BIRE_FIELDS if name != "ssh"),
)
BASELINES = ("persistence", "climatology", "a0")
GROUP_SLICES = {
    "u": slice(0, 15),
    "v": slice(15, 30),
    "temperature": slice(30, 45),
    "ssh": slice(45, 46),
}


class ModelCSuccessorValidationError(RuntimeError):
    """Raised when fresh validation would violate its frozen contract."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def load_validation_contract(
    path: str | Path,
    *,
    verify_source_files: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the contract frozen before any fresh-v2 validation metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != VALIDATION_VERSION:
        raise ValueError(f"expected validation contract {VALIDATION_VERSION}")
    if (
        contract.get("contract_status")
        != "frozen_after_split_count_correction_before_fresh_v2_validation"
    ):
        raise ValueError("Model C successor validation contract was not frozen")
    read = contract.get("read_contract", {})
    if (
        read.get("training_pair_code") != 1
        or read.get("validation_pair_code") != 2
        or read.get("validation_read_after_contract_freeze") is not True
        or any(
            read.get(name) is not False
            for name in (
                "inference_read",
                "intermediate_wind_read",
                "response_read",
                "adjoint_read",
            )
        )
    ):
        raise ValueError("successor validation read contract changed")
    validation = contract.get("validation", {})
    if (
        tuple(validation.get("lead_days", ())) != LEAD_DAYS
        or tuple(validation.get("primary_fields", ())) != PRIMARY_FIELDS
        or tuple(validation.get("baselines", ())) != BASELINES
        or int(validation.get("expected_complete_starts_per_regime", -1))
        != 180
        or validation.get("auc_definition")
        != "trapezoidal_mean_over_discrete_10_to_90_day_leads"
    ):
        raise ValueError("fresh-v2 validation metric contract changed")
    seeds = contract.get("seed_replication", {})
    if tuple(seeds.get("declared_seeds", ())) != (
        20260723,
        20260724,
        20260725,
    ):
        raise ValueError("successor validation seeds changed")
    if tuple(seeds.get("new_training_seeds", ())) != (20260724, 20260725):
        raise ValueError("successor replication seed set changed")
    bootstrap = contract.get("bootstrap", {})
    if (
        int(bootstrap.get("replicates", -1)) != 4000
        or float(bootstrap.get("confidence_level", -1.0)) != 0.95
        or int(bootstrap.get("seed", -1)) != 20260727
        or tuple(bootstrap.get("block_length_days_by_regime", ()))
        != (21, 122, 19)
    ):
        raise ValueError("successor validation bootstrap contract changed")
    if verify_source_files:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(f"validation source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def resolve_replication_seed(
    contract_path: str | Path,
    *,
    array_index: int,
) -> int:
    """Resolve one scheduler index without reading scientific data."""

    contract, _, _ = load_validation_contract(contract_path)
    seeds = tuple(
        int(seed)
        for seed in contract["seed_replication"]["new_training_seeds"]
    )
    if not 0 <= array_index < len(seeds):
        raise ValueError(
            f"array index {array_index} is outside {len(seeds)} replication seeds"
        )
    return seeds[array_index]


def complete_validation_times(
    pair_codes: np.ndarray,
    snapshot_codes: np.ndarray,
    *,
    pair_code: int = 2,
) -> np.ndarray:
    """Return every daily start with a complete 10--90-day held rollout."""

    pair_codes = np.asarray(pair_codes, dtype=np.uint8)
    snapshot_codes = np.asarray(snapshot_codes, dtype=np.uint8)
    if pair_codes.ndim != 1 or snapshot_codes.ndim != 1:
        raise ValueError("split codes must be one-dimensional")
    selected = []
    for start in range(pair_codes.size):
        pair_indices = start + np.arange(len(LEAD_DAYS)) * HORIZON_DAYS
        snapshot_indices = start + np.arange(
            len(LEAD_DAYS) + 1
        ) * HORIZON_DAYS
        if (
            pair_indices[-1] < pair_codes.size
            and snapshot_indices[-1] < snapshot_codes.size
            and np.all(pair_codes[pair_indices] == pair_code)
            and np.all(snapshot_codes[snapshot_indices] == pair_code)
        ):
            selected.append(start)
    return np.asarray(selected, dtype=np.int64)


def curve_auc(values: np.ndarray) -> np.ndarray:
    """Trapezoidal mean over the frozen, equally spaced 10--90-day leads."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != len(LEAD_DAYS):
        raise ValueError("AUC input does not match the frozen validation leads")
    weights = np.ones(len(LEAD_DAYS), dtype=np.float64)
    weights[[0, -1]] = 0.5
    return np.sum(values * weights, axis=-1) / np.sum(weights)


def circular_block_indices(
    count: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw exactly ``count`` circular-block indices."""

    if count <= 0 or block_length <= 0:
        raise ValueError("block bootstrap counts must be positive")
    block_count = math.ceil(count / block_length)
    starts = rng.integers(0, count, size=block_count)
    offsets = np.arange(block_length)
    return np.concatenate(
        [np.mod(start + offsets, count) for start in starts]
    )[:count]


@dataclass
class ValidationStepper:
    """One frozen ten-day map with its own normalization contract."""

    kind: str
    model: Any
    device: Any
    wet: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    wind_mean: float
    wind_scale: float

    def normalized_state(self, physical: np.ndarray) -> Any:
        value = _normalise(
            physical,
            self.mean,
            self.scale,
            self.wet,
        )
        return torch.from_numpy(value).to(
            device=self.device,
            dtype=torch.float32,
        )

    def normalized_static(
        self,
        static: Any,
        experiments: np.ndarray,
    ) -> Any:
        value = np.stack(
            [
                np.asarray(static[int(experiment)], dtype=np.float32)
                for experiment in experiments
            ]
        )
        value[:, 0] = (value[:, 0] - self.wind_mean) / self.wind_scale
        value[:, 0, ~self.wet] = 0.0
        if self.kind == "a0":
            value = value[:, :1]
        return torch.from_numpy(np.ascontiguousarray(value)).to(
            device=self.device,
            dtype=torch.float32,
        )

    def step(self, current: Any, static: Any) -> Any:
        prediction = self.model(torch.cat((current, static), dim=1))
        if self.kind == "successor":
            prediction = current + prediction
        wet = torch.from_numpy(self.wet).to(self.device)
        prediction[:, :, ~wet] = 0.0
        return prediction

    def physical(self, normalized: Any) -> np.ndarray:
        value = normalized.detach().cpu().numpy()
        value = value * self.scale[None, :, None, None]
        value += self.mean[None, :, None, None]
        value[:, :, ~self.wet] = 0.0
        return value.astype(np.float32)


def _load_a0_stepper(
    checkpoint: Path,
    device: Any,
    wet: np.ndarray,
    expected_sha256: str,
) -> ValidationStepper:
    if _file_sha256(checkpoint) != expected_sha256:
        raise ModelCSuccessorValidationError("frozen A0 checkpoint changed")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture = a0_architecture()
    if payload.get("model_config") != architecture.to_dict():
        raise ModelCSuccessorValidationError("A0 architecture changed")
    normalization = payload.get("normalization", {})
    mean = np.asarray(normalization.get("state_mean"), dtype=np.float32)
    scale = np.asarray(normalization.get("state_scale"), dtype=np.float32)
    if (
        mean.shape != (46,)
        or scale.shape != (46,)
        or np.any(scale <= 0)
    ):
        raise ModelCSuccessorValidationError("invalid A0 normalization")
    model = build_paper_fno(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return ValidationStepper(
        kind="a0",
        model=model,
        device=device,
        wet=wet,
        mean=mean,
        scale=scale,
        wind_mean=float(normalization["wind_mean"]),
        wind_scale=float(normalization["wind_scale"]),
    )


def _load_successor_stepper(
    checkpoint: Path,
    device: Any,
    wet: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    wind_mean: float,
    wind_scale: float,
    expected_architecture: Mapping[str, Any],
) -> tuple[ValidationStepper, Mapping[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != expected_architecture:
        raise ModelCSuccessorValidationError(
            "successor checkpoint architecture changed"
        )
    architecture = ModelCSuccessorArchitecture(**expected_architecture)
    model = build_successor(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return (
        ValidationStepper(
            kind="successor",
            model=model,
            device=device,
            wet=wet,
            mean=mean,
            scale=scale,
            wind_mean=wind_mean,
            wind_scale=wind_scale,
        ),
        payload,
    )


def _climatology_batch_fields(
    experiments: np.ndarray,
    climatology_state: np.ndarray,
    climatology_derived: Mapping[str, np.ndarray],
    wet: np.ndarray,
) -> dict[str, np.ndarray]:
    fields = _state_fields(climatology_state[experiments], wet)
    for name in BIRE_FIELDS:
        fields[name] = climatology_derived[name][experiments, None]
    return fields


def _empty_metric_arrays(count: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in EVALUATION_FIELDS:
        result[f"rmse__{name}"] = np.empty(
            (count, len(LEAD_DAYS)), dtype=np.float32
        )
        result[f"acc__{name}"] = np.empty(
            (count, len(LEAD_DAYS)), dtype=np.float32
        )
    return result


def _evaluate_baseline_metrics(
    state: Any,
    records: np.ndarray,
    climatology_state: np.ndarray,
    climatology_derived: Mapping[str, np.ndarray],
    wet: np.ndarray,
    dataset_mean: np.ndarray,
    dataset_scale: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray] | np.ndarray]:
    count = int(records.shape[0])
    result: dict[str, dict[str, np.ndarray] | np.ndarray] = {
        method: _empty_metric_arrays(count)
        for method in ("persistence", "climatology")
    }
    result["ssh_rms_z__truth"] = np.empty(
        (count, len(LEAD_DAYS)), dtype=np.float32
    )
    result["streamfunction_max_abs__truth"] = np.empty(
        (count, len(LEAD_DAYS)), dtype=np.float32
    )
    for offset in range(0, count, batch_size):
        stop = min(offset + batch_size, count)
        chunk = records[offset:stop]
        experiments = chunk[:, 0].astype(int)
        starts = chunk[:, 1].astype(int)
        initial = np.stack(
            [
                np.asarray(state[experiment, start], dtype=np.float32)
                for experiment, start in zip(experiments, starts)
            ]
        )
        initial_fields = _state_fields(initial, wet)
        climate_fields = _climatology_batch_fields(
            experiments,
            climatology_state,
            climatology_derived,
            wet,
        )
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
            truth_fields = _state_fields(truth, wet)
            for method, prediction_fields in (
                ("persistence", initial_fields),
                ("climatology", climate_fields),
            ):
                metrics = result[method]
                assert isinstance(metrics, dict)
                for name in EVALUATION_FIELDS:
                    metrics[f"rmse__{name}"][offset:stop, lead_index] = (
                        _member_rmse(
                            prediction_fields[name],
                            truth_fields[name],
                            wet,
                        )
                    )
                    metrics[f"acc__{name}"][offset:stop, lead_index] = (
                        _member_acc(
                            prediction_fields[name],
                            truth_fields[name],
                            climate_fields[name],
                            wet,
                        )
                    )
            truth_z = (
                truth[:, 45] - dataset_mean[45]
            ) / dataset_scale[45]
            truth_rms = np.sqrt(
                np.mean(np.square(truth_z[:, wet]), axis=1)
            )
            result["ssh_rms_z__truth"][offset:stop, lead_index] = truth_rms
            result["streamfunction_max_abs__truth"][
                offset:stop, lead_index
            ] = np.max(
                np.abs(truth_fields["streamfunction"][:, 0, wet]),
                axis=1,
            )
    return result


def _evaluate_stepper(
    stepper: ValidationStepper,
    state: Any,
    static: Any,
    records: np.ndarray,
    climatology_state: np.ndarray,
    climatology_derived: Mapping[str, np.ndarray],
    dataset_mean: np.ndarray,
    dataset_scale: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    count = int(records.shape[0])
    result = _empty_metric_arrays(count)
    result["finite"] = np.empty(
        (count, len(LEAD_DAYS)), dtype=np.uint8
    )
    result["normalized_max_abs"] = np.empty(
        (count, len(LEAD_DAYS)), dtype=np.float32
    )
    result["normalized_land_max_abs"] = np.empty(
        (count, len(LEAD_DAYS)), dtype=np.float32
    )
    result["ssh_rms_z"] = np.empty(
        (count, len(LEAD_DAYS)), dtype=np.float32
    )
    result["streamfunction_max_abs"] = np.empty(
        (count, len(LEAD_DAYS)), dtype=np.float32
    )
    for name in GROUP_SLICES:
        result[f"group_mean_bias_z__{name}"] = np.empty(
            (count, len(LEAD_DAYS)), dtype=np.float32
        )
    wet_t = torch.from_numpy(stepper.wet).to(stepper.device)
    land_t = ~wet_t

    for offset in range(0, count, batch_size):
        stop = min(offset + batch_size, count)
        chunk = records[offset:stop]
        experiments = chunk[:, 0].astype(int)
        starts = chunk[:, 1].astype(int)
        initial = np.stack(
            [
                np.asarray(state[experiment, start], dtype=np.float32)
                for experiment, start in zip(experiments, starts)
            ]
        )
        current = stepper.normalized_state(initial)
        forcing = stepper.normalized_static(static, experiments)
        climate_fields = _climatology_batch_fields(
            experiments,
            climatology_state,
            climatology_derived,
            stepper.wet,
        )
        with torch.no_grad():
            for lead_index, lead in enumerate(LEAD_DAYS):
                current = stepper.step(current, forcing)
                prediction = stepper.physical(current)
                truth = np.stack(
                    [
                        np.asarray(
                            state[experiment, start + lead],
                            dtype=np.float32,
                        )
                        for experiment, start in zip(experiments, starts)
                    ]
                )
                prediction_fields = _state_fields(prediction, stepper.wet)
                truth_fields = _state_fields(truth, stepper.wet)
                for name in EVALUATION_FIELDS:
                    result[f"rmse__{name}"][offset:stop, lead_index] = (
                        _member_rmse(
                            prediction_fields[name],
                            truth_fields[name],
                            stepper.wet,
                        )
                    )
                    result[f"acc__{name}"][offset:stop, lead_index] = (
                        _member_acc(
                            prediction_fields[name],
                            truth_fields[name],
                            climate_fields[name],
                            stepper.wet,
                        )
                    )
                finite = torch.isfinite(current).all(dim=(1, 2, 3))
                result["finite"][offset:stop, lead_index] = (
                    finite.detach().cpu().numpy().astype(np.uint8)
                )
                result["normalized_max_abs"][
                    offset:stop, lead_index
                ] = (
                    torch.amax(torch.abs(current), dim=(1, 2, 3))
                    .detach()
                    .cpu()
                    .numpy()
                )
                if bool(torch.any(land_t)):
                    result["normalized_land_max_abs"][
                        offset:stop, lead_index
                    ] = (
                        torch.amax(
                            torch.abs(current[:, :, land_t]),
                            dim=(1, 2),
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                else:
                    result["normalized_land_max_abs"][
                        offset:stop, lead_index
                    ] = 0.0
                for name, channels in GROUP_SLICES.items():
                    error_z = (
                        prediction[:, channels] - truth[:, channels]
                    ) / dataset_scale[channels][None, :, None, None]
                    result[f"group_mean_bias_z__{name}"][
                        offset:stop, lead_index
                    ] = np.mean(
                        error_z[:, :, stepper.wet],
                        axis=(1, 2),
                    )
                ssh_z = (
                    prediction[:, 45] - dataset_mean[45]
                ) / dataset_scale[45]
                result["ssh_rms_z"][offset:stop, lead_index] = np.sqrt(
                    np.mean(np.square(ssh_z[:, stepper.wet]), axis=1)
                )
                result["streamfunction_max_abs"][
                    offset:stop, lead_index
                ] = np.max(
                    np.abs(
                        prediction_fields["streamfunction"][
                            :, 0, stepper.wet
                        ]
                    ),
                    axis=1,
                )
    return result


def _method_auc_summary(
    metrics: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    result = {}
    for name in EVALUATION_FIELDS:
        rmse_auc = curve_auc(metrics[f"rmse__{name}"])
        acc_auc = curve_auc(metrics[f"acc__{name}"])
        result[name] = {
            "rmse_auc_mean": float(np.mean(rmse_auc)),
            "acc_auc_mean": float(np.mean(acc_auc)),
        }
    return result


def _seed_point_summary(
    metrics: Mapping[str, np.ndarray],
    baseline_metrics: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    fields = {}
    passed = True
    for name in PRIMARY_FIELDS:
        model_rmse = curve_auc(metrics[f"rmse__{name}"])
        model_acc = curve_auc(metrics[f"acc__{name}"])
        comparisons = {}
        for baseline in BASELINES:
            baseline_rmse = curve_auc(
                baseline_metrics[baseline][f"rmse__{name}"]
            )
            baseline_acc = curve_auc(
                baseline_metrics[baseline][f"acc__{name}"]
            )
            rmse_ratio = float(
                np.mean(model_rmse) / np.mean(baseline_rmse)
            )
            acc_difference = float(
                np.mean(model_acc) - np.mean(baseline_acc)
            )
            comparisons[baseline] = {
                "rmse_auc_ratio": rmse_ratio,
                "acc_auc_difference": acc_difference,
                "passed": rmse_ratio < 1.0 and acc_difference > 0.0,
            }
            passed = passed and comparisons[baseline]["passed"]
        fields[name] = {
            "model_rmse_auc": float(np.mean(model_rmse)),
            "model_acc_auc": float(np.mean(model_acc)),
            "comparisons": comparisons,
        }
    return {"fields": fields, "passed": bool(passed)}


def _bootstrap_summary(
    seed_metrics: Sequence[Mapping[str, np.ndarray]],
    baseline_metrics: Mapping[str, Mapping[str, np.ndarray]],
    records: np.ndarray,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    bootstrap = contract["bootstrap"]
    replicates = int(bootstrap["replicates"])
    confidence = float(bootstrap["confidence_level"])
    rng = np.random.default_rng(int(bootstrap["seed"]))
    regimes = records[:, 0].astype(int)
    regime_indices = [
        np.flatnonzero(regimes == experiment) for experiment in range(3)
    ]
    block_lengths = tuple(
        int(value)
        for value in bootstrap["block_length_days_by_regime"]
    )
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, Any] = {}
    overall_pass = True
    for name in PRIMARY_FIELDS:
        model_rmse = np.mean(
            np.stack(
                [
                    curve_auc(metrics[f"rmse__{name}"])
                    for metrics in seed_metrics
                ]
            ),
            axis=0,
        )
        model_acc = np.mean(
            np.stack(
                [
                    curve_auc(metrics[f"acc__{name}"])
                    for metrics in seed_metrics
                ]
            ),
            axis=0,
        )
        comparisons = {}
        for baseline in BASELINES:
            baseline_rmse = curve_auc(
                baseline_metrics[baseline][f"rmse__{name}"]
            )
            baseline_acc = curve_auc(
                baseline_metrics[baseline][f"acc__{name}"]
            )
            rmse_samples = np.empty(replicates, dtype=np.float64)
            acc_samples = np.empty(replicates, dtype=np.float64)
            for replicate in range(replicates):
                selected = np.concatenate(
                    [
                        indices[
                            circular_block_indices(
                                indices.size,
                                block_length,
                                rng,
                            )
                        ]
                        for indices, block_length in zip(
                            regime_indices,
                            block_lengths,
                        )
                    ]
                )
                rmse_samples[replicate] = (
                    np.mean(model_rmse[selected])
                    / np.mean(baseline_rmse[selected])
                )
                acc_samples[replicate] = np.mean(
                    model_acc[selected] - baseline_acc[selected]
                )
            rmse_interval = np.quantile(
                rmse_samples,
                (alpha, 1.0 - alpha),
            )
            acc_interval = np.quantile(
                acc_samples,
                (alpha, 1.0 - alpha),
            )
            rmse_point = float(
                np.mean(model_rmse) / np.mean(baseline_rmse)
            )
            acc_point = float(np.mean(model_acc - baseline_acc))
            passed = bool(
                rmse_point < 1.0
                and float(rmse_interval[1]) < 1.0
                and acc_point > 0.0
                and float(acc_interval[0]) > 0.0
            )
            comparisons[baseline] = {
                "rmse_auc_ratio": {
                    "point": rmse_point,
                    "confidence_interval": rmse_interval.tolist(),
                },
                "acc_auc_difference": {
                    "point": acc_point,
                    "confidence_interval": acc_interval.tolist(),
                },
                "passed": passed,
            }
            overall_pass = overall_pass and passed
        result[name] = comparisons
    return {
        "fields": result,
        "confidence_level": confidence,
        "replicates": replicates,
        "passed": bool(overall_pass),
    }


def _stability_summary(
    metrics: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    records: np.ndarray,
    wind_stress: np.ndarray,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    stability = contract["validation"]["stability_gate"]
    final = -1
    experiments = records[:, 0].astype(int)
    all_finite = bool(np.all(metrics["finite"]))
    max_normalized = float(np.max(metrics["normalized_max_abs"]))
    max_land = float(np.max(metrics["normalized_land_max_abs"]))
    final_bias = {
        name: float(
            abs(np.mean(metrics[f"group_mean_bias_z__{name}"][:, final]))
        )
        for name in GROUP_SLICES
    }
    ssh_ratio = float(
        np.mean(metrics["ssh_rms_z"][:, final])
        / np.mean(truth["ssh_rms_z__truth"][:, final])
    )
    stream_ratio = float(
        np.mean(metrics["streamfunction_max_abs"][:, final])
        / np.mean(truth["streamfunction_max_abs__truth"][:, final])
    )
    model_regime = np.asarray(
        [
            np.mean(
                metrics["streamfunction_max_abs"][
                    experiments == experiment,
                    final,
                ]
            )
            for experiment in range(3)
        ]
    )
    truth_regime = np.asarray(
        [
            np.mean(
                truth["streamfunction_max_abs__truth"][
                    experiments == experiment,
                    final,
                ]
            )
            for experiment in range(3)
        ]
    )
    model_slope = float(np.polyfit(wind_stress, model_regime, 1)[0])
    truth_slope = float(np.polyfit(wind_stress, truth_regime, 1)[0])
    slope_ratio = (
        model_slope / truth_slope if truth_slope != 0.0 else float("nan")
    )
    lower, upper = (
        float(value)
        for value in stability["amplitude_ratio_bounds"]
    )
    stream_pass = bool(
        np.isfinite(slope_ratio)
        and np.sign(model_slope) == np.sign(truth_slope)
        and lower <= abs(slope_ratio) <= upper
    )
    passed = bool(
        all_finite
        and max_normalized
        < float(stability["normalized_state_abs_max"])
        and max_land == 0.0
        and max(final_bias.values())
        < float(stability["group_mean_bias_training_sigma_max"])
        and lower <= ssh_ratio <= upper
        and lower <= stream_ratio <= upper
        and stream_pass
    )
    return {
        "all_finite": all_finite,
        "normalized_state_abs_max": max_normalized,
        "normalized_land_abs_max": max_land,
        "day90_group_mean_bias_training_sigma": final_bias,
        "day90_ssh_amplitude_ratio": ssh_ratio,
        "day90_streamfunction_amplitude_ratio": stream_ratio,
        "day90_streamfunction_wind_slope": {
            "model": model_slope,
            "truth": truth_slope,
            "model_over_truth": float(slope_ratio),
            "passed": stream_pass,
        },
        "passed": passed,
    }


def _training_artifacts(
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    artifacts = []
    expected_architecture = contract["architecture"]
    expected_contract_sha = contract["source_artifacts"][
        "successor_training_contract_sha256"
    ]
    reference = contract["seed_replication"]["reference_seed_artifact"]
    for seed in contract["seed_replication"]["declared_seeds"]:
        seed = int(seed)
        if seed == int(reference["seed"]):
            report_path = Path(reference["report"]).resolve()
            expected_report_sha = reference["report_sha256"]
            expected_checkpoint_sha = reference["checkpoint_sha256"]
            expected_source_sha = reference["runtime_source_sha256"]
        else:
            root = Path(
                contract["seed_replication"]["replication_root"]
            ).resolve()
            output = root / f"seed_{seed}"
            report_path = output / REPORT_NAME
            expected_report_sha = None
            expected_checkpoint_sha = None
            expected_source_sha = contract["source_hashes"][
                "src/bire_repro/af_model_c_successor.py"
            ]
        if not report_path.is_file():
            raise ModelCSuccessorValidationError(
                f"missing seed report {report_path}"
            )
        if (
            expected_report_sha is not None
            and _file_sha256(report_path) != expected_report_sha
        ):
            raise ModelCSuccessorValidationError(
                "reference successor report changed"
            )
        report = json.loads(report_path.read_text())
        checkpoint = Path(report.get("checkpoint", "")).resolve()
        if (
            report.get("status") != "complete"
            or int(report.get("seed", -1)) != seed
            or report.get("candidate_id") != "v2_bireprop_w128_mlp4"
            or report.get("architecture") != expected_architecture
            or report.get("contract_sha256") != expected_contract_sha
            or report.get("training_gate", {}).get("passed") is not True
            or report.get("save_reload_three_step_bitwise_exact") is not True
            or report.get("read_contract", {}).get("validation_read")
            is not False
            or report.get("read_contract", {}).get("inference_read") is not False
            or not checkpoint.is_file()
            or _file_sha256(checkpoint) != report.get("checkpoint_sha256")
        ):
            raise ModelCSuccessorValidationError(
                f"seed {seed} violates the frozen training contract"
            )
        if (
            expected_checkpoint_sha is not None
            and report["checkpoint_sha256"] != expected_checkpoint_sha
        ):
            raise ModelCSuccessorValidationError(
                "reference successor checkpoint changed"
            )
        runtime_source = report.get("runtime_source_sha256")
        if seed == int(reference["seed"]):
            runtime_source = expected_source_sha
        if runtime_source != expected_source_sha:
            raise ModelCSuccessorValidationError(
                f"seed {seed} used an unexpected training source"
            )
        artifacts.append(
            {
                "seed": seed,
                "report": str(report_path),
                "report_sha256": _file_sha256(report_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": report["checkpoint_sha256"],
                "selected_optimizer_step": int(
                    report["selected_checkpoint"]["optimizer_step"]
                ),
                "training_worst_per_regime_group_ratio": float(
                    report["full_training_ten_day"][
                        "worst_per_regime_group_ratio"
                    ]
                ),
                "training_report": report,
            }
        )
    shared = {
        (
            artifact["training_report"]["diagnostic_records_sha256"],
            artifact["training_report"]["long_rollout_records_sha256"],
            artifact["training_report"]["increment_scale_sha256"],
        )
        for artifact in artifacts
    }
    if len(shared) != 1:
        raise ModelCSuccessorValidationError(
            "training seeds do not share diagnostic records and scales"
        )
    return artifacts


def _selected_seed(
    seed_summaries: Sequence[Mapping[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    ranking = []
    for summary in seed_summaries:
        ratios = [
            summary["point_gate"]["fields"][field]["comparisons"][baseline][
                "rmse_auc_ratio"
            ]
            for field in PRIMARY_FIELDS
            for baseline in BASELINES
        ]
        ranking.append(
            {
                "seed": int(summary["seed"]),
                "score": float(np.mean(ratios)),
            }
        )
    ranking.sort(key=lambda row: (row["score"], row["seed"]))
    return int(ranking[len(ranking) // 2]["seed"]), ranking


def evaluate_validation(
    dataset_path: str | Path,
    quality_report_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Open fresh split 2 once and apply the complete frozen three-seed gate."""

    if torch is None:
        raise RuntimeError("successor validation requires PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_validation_contract(
        contract_path
    )
    dataset = Path(dataset_path).resolve()
    quality = Path(quality_report_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite successor validation output: {output}"
        )
    source = contract["source_artifacts"]
    if (
        _file_sha256(dataset / ".zmetadata")
        != source["dataset_metadata_sha256"]
        or _file_sha256(quality) != source["quality_report_sha256"]
    ):
        raise ModelCSuccessorValidationError(
            "fresh validation dataset source changed"
        )
    quality_report = json.loads(quality.read_text())
    if (
        quality_report.get("status") != "valid"
        or quality_report.get("inference_state_metrics_read") is not False
    ):
        raise ModelCSuccessorValidationError(
            "trajectory-v2 quality gate is not valid and sealed"
        )
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested without a visible GPU")
    device = torch.device(device_name)

    group = zarr.open_consolidated(str(dataset), mode="r")
    if tuple(group.attrs["state_channels"]) != STATE_CHANNELS:
        raise ModelCSuccessorValidationError(
            "unexpected trajectory-v2 state channels"
        )
    state = group["state"]
    static = group["static_features"]
    mean, scale, wet, _, wind_mean, wind_scale = _normalizers(group)
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(
        group["snapshot_split"][:],
        dtype=np.uint8,
    )
    times = complete_validation_times(pair_codes, snapshot_codes)
    expected_count = int(
        contract["validation"]["expected_complete_starts_per_regime"]
    )
    if times.size != expected_count:
        raise ModelCSuccessorValidationError(
            f"expected {expected_count} complete validation starts, "
            f"found {times.size}"
        )
    records = np.asarray(
        [
            (experiment, int(time_index))
            for experiment in range(3)
            for time_index in times
        ],
        dtype=np.int64,
    )
    climatology_state, climatology_derived, training_days = (
        _training_climatology(state, snapshot_codes, wet)
    )
    batch_size = int(contract["validation"]["batch_size"])
    baseline = _evaluate_baseline_metrics(
        state,
        records,
        climatology_state,
        climatology_derived,
        wet,
        mean,
        scale,
        batch_size=batch_size,
    )
    a0_contract = contract["a0_baseline"]
    a0_stepper = _load_a0_stepper(
        Path(a0_contract["checkpoint"]).resolve(),
        device,
        wet,
        a0_contract["checkpoint_sha256"],
    )
    a0_metrics = _evaluate_stepper(
        a0_stepper,
        state,
        static,
        records,
        climatology_state,
        climatology_derived,
        mean,
        scale,
        batch_size=batch_size,
    )
    baseline_metrics: dict[str, Mapping[str, np.ndarray]] = {
        "persistence": baseline["persistence"],
        "climatology": baseline["climatology"],
        "a0": a0_metrics,
    }
    assert isinstance(baseline["persistence"], dict)
    assert isinstance(baseline["climatology"], dict)

    artifacts = _training_artifacts(contract)
    seed_metrics = []
    seed_summaries = []
    arrays: dict[str, np.ndarray] = {
        "records": records.astype(np.int32),
        "lead_days": np.asarray(LEAD_DAYS, dtype=np.int16),
        "validation_times": times.astype(np.int32),
    }
    for method, metrics in baseline_metrics.items():
        for name, value in metrics.items():
            arrays[f"{method}__{name}"] = np.asarray(value)
    for name in ("ssh_rms_z__truth", "streamfunction_max_abs__truth"):
        arrays[name] = np.asarray(baseline[name])

    for artifact in artifacts:
        stepper, payload = _load_successor_stepper(
            Path(artifact["checkpoint"]),
            device,
            wet,
            mean,
            scale,
            wind_mean,
            wind_scale,
            contract["architecture"],
        )
        if (
            int(payload.get("seed", -1)) != artifact["seed"]
            or payload.get("checkpoint_sha256") is not None
        ):
            raise ModelCSuccessorValidationError(
                "successor checkpoint payload provenance changed"
            )
        metrics = _evaluate_stepper(
            stepper,
            state,
            static,
            records,
            climatology_state,
            climatology_derived,
            mean,
            scale,
            batch_size=batch_size,
        )
        seed_metrics.append(metrics)
        for name, value in metrics.items():
            arrays[f"seed_{artifact['seed']}__{name}"] = value
        point = _seed_point_summary(metrics, baseline_metrics)
        stability = _stability_summary(
            metrics,
            baseline,
            records,
            np.asarray(
                group.attrs["wind_stress_n_m2"],
                dtype=np.float64,
            ),
            contract,
        )
        seed_summaries.append(
            {
                "seed": artifact["seed"],
                "checkpoint": artifact["checkpoint"],
                "checkpoint_sha256": artifact["checkpoint_sha256"],
                "training_worst_per_regime_group_ratio": artifact[
                    "training_worst_per_regime_group_ratio"
                ],
                "point_gate": point,
                "stability_gate": stability,
                "all_field_auc": _method_auc_summary(metrics),
            }
        )
    bootstrap = _bootstrap_summary(
        seed_metrics,
        baseline_metrics,
        records,
        contract,
    )
    prospective_seed, ranking = _selected_seed(seed_summaries)
    all_point = all(
        summary["point_gate"]["passed"] for summary in seed_summaries
    )
    all_stable = all(
        summary["stability_gate"]["passed"] for summary in seed_summaries
    )
    training_pass = all(
        artifact["training_report"]["training_gate"]["passed"]
        for artifact in artifacts
    )
    accepted = bool(
        training_pass and all_point and all_stable and bootstrap["passed"]
    )
    selected_seed = prospective_seed if accepted else None
    selected_artifact = next(
        artifact
        for artifact in artifacts
        if artifact["seed"] == prospective_seed
    )
    gate = {
        "status": (
            "frozen_for_inference"
            if accepted
            else "scientifically_rejected_fresh_v2_validation"
        ),
        "accepted": accepted,
        "every_seed_training_gate_passed": training_pass,
        "every_seed_primary_point_gate_passed": all_point,
        "every_seed_validation_stability_gate_passed": all_stable,
        "paired_block_bootstrap_gate_passed": bool(
            bootstrap["passed"]
        ),
        "prospective_median_ranked_seed": prospective_seed,
        "selected_seed": selected_seed,
        "configuration_frozen": accepted,
        "inference_authorized": accepted,
    }

    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    arrays_path = temporary / VALIDATION_ARRAYS_NAME
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "status": "complete",
        "purpose": "fresh_v2_bire_aligned_three_seed_model_c_validation",
        "version": VALIDATION_VERSION,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset),
        "dataset_metadata_sha256": _file_sha256(
            dataset / ".zmetadata"
        ),
        "quality_report": str(quality),
        "quality_report_sha256": _file_sha256(quality),
        "device": str(device),
        "read_contract": {
            "pair_split_codes_read": [1, 2],
            "snapshot_split_codes_read": [1, 2],
            "validation_read": True,
            "inference_read": False,
            "intermediate_wind_read": False,
            "response_read": False,
            "adjoint_read": False,
        },
        "record_contract": {
            "lead_days": list(LEAD_DAYS),
            "validation_starts_per_regime": int(times.size),
            "validation_records_total": int(records.shape[0]),
            "validation_times_sha256": _array_sha256(times),
            "records_sha256": _array_sha256(records),
            "training_climatology_snapshots_per_regime": training_days,
        },
        "architecture": contract["architecture"],
        "training_seed_artifacts": [
            {
                key: artifact[key]
                for key in (
                    "seed",
                    "report",
                    "report_sha256",
                    "checkpoint",
                    "checkpoint_sha256",
                    "selected_optimizer_step",
                    "training_worst_per_regime_group_ratio",
                )
            }
            for artifact in artifacts
        ],
        "baseline_auc": {
            method: _method_auc_summary(metrics)
            for method, metrics in baseline_metrics.items()
        },
        "seed_validation": seed_summaries,
        "bootstrap_gate": bootstrap,
        "seed_selection": {
            "rule": (
                "median-ranked seed by mean primary-field RMSE-AUC "
                "ratio over persistence, climatology, and A0"
            ),
            "ranking": ranking,
            "prospective_seed": prospective_seed,
            "selected_seed": selected_seed,
            "selected_checkpoint": (
                selected_artifact["checkpoint"] if accepted else None
            ),
            "selected_checkpoint_sha256": (
                selected_artifact["checkpoint_sha256"]
                if accepted
                else None
            ),
        },
        "validation_gate": gate,
        "inference_opened": False,
        "intermediate_wind_opened": False,
        "response_or_adjoint_opened": False,
        "arrays": str(output / VALIDATION_ARRAYS_NAME),
        "arrays_sha256": _file_sha256(arrays_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    report["report_content_sha256"] = _json_sha256(report)
    (temporary / VALIDATION_REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, output)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser("resolve-seed")
    resolve.add_argument("--contract", type=Path, required=True)
    resolve.add_argument("--array-index", type=int, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--quality-report", type=Path, required=True)
    evaluate.add_argument("--contract", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "resolve-seed":
        print(
            resolve_replication_seed(
                args.contract,
                array_index=args.array_index,
            )
        )
        return 0
    result = evaluate_validation(
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
