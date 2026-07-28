"""Rollout-conditioned, conservation-projected Model C loss-v3.

This training-only experiment follows the frozen slow-field bias/projection
audit.  The audit rejected a purely static-bias explanation of the day-90
drift, while showing that signed bias remains material.  The experiment
therefore fine-tunes the diagnostic-best duration checkpoint with:

* the unchanged loss-v1 core, evaluated through the projected map;
* exact area-mean SSH and per-regime temperature-tendency projections at
  every model call;
* a small batch-mean slow-increment bias penalty; and
* one differentiable call from a detached forecast-age state, cycling
  uniformly over target leads from day 10 through day 90.

Only split-1 training states and the already frozen 540-record training gate
are read.  Validation, inference, intermediate-wind, response, and adjoint
states remain sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from .af_a0 import records_for_pair_split
from .af_a0_evaluate import _normalizers
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
from .af_model_c_checkpoint_replay_audit import (
    _audit_records,
    _array_sha256,
    checkpoint_gate_summary,
    load_checkpoint_replay_contract,
)
from .af_model_c_overfit import AUDIT_TERMS, _device, _file_sha256
from .af_model_c_pushforward_duration import load_duration_contract
from .af_model_c_pushforward_objective import (
    HORIZON_DAYS,
    REFERENCE_CANDIDATE,
    ROLLOUT_STEPS,
    STATE_CHANNEL_COUNT,
    ModelCPushforwardDataset,
    select_pushforward_checkpoint,
    slow_field_pushforward_loss,
)
from .af_model_c_rollout_diagnosis import lead_curve_summary
from .af_model_c_slow_field_bias_projection import (
    load_bias_projection_contract,
    wet_area_weights,
)
from .af_model_c_successor import (
    REFERENCE_DIAGNOSTIC_SEED,
    _one_step_diagnostics,
    _sample_records_by_regime,
    architecture_from_candidate,
    build_successor,
    load_successor_contract,
    training_increment_scale,
)
from .af_model_c_successor_validation import (
    ValidationStepper,
    _evaluate_stepper,
    _method_auc_summary,
)
from .af_model_c_truncated_unroll_objective import (
    validate_duration_source_payload,
)

try:
    import torch
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


VERSION = "model_c_rollout_conditioned_loss_v3"
REPORT_NAME = "model_c_rollout_conditioned_loss_v3_report.json"
ARRAYS_NAME = "model_c_rollout_conditioned_loss_v3_arrays.npz"
CHECKPOINT_DIRECTORY = "fine_tune_checkpoints"
SELECTED_CHECKPOINT_NAME = "model_c_rollout_conditioned_loss_v3_best.pt"
SOURCE_FINE_TUNE_STEP = 5760
TEMPERATURE_SLICE = slice(30, 45)
SLOW_SLICE = slice(30, 46)
SSH_CHANNEL = 45
TARGET_STEPS = tuple(range(1, ROLLOUT_STEPS + 1))
TRAINING_TERMS = (
    *AUDIT_TERMS,
    "rollout_conditioned_sst",
    "rollout_conditioned_phihyd_surface",
    "slow_bias_penalty",
)
_BaseModule = torch.nn.Module if torch is not None else object


class ModelCRolloutConditionedError(RuntimeError):
    """Raised when the frozen loss-v3 experiment contract is violated."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ModelCLossV3Dataset(ModelCPushforwardDataset):
    """Nine-step trajectory dataset that also returns its regime index."""

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        features, futures = super().__getitem__(index)
        experiment = int(self.records[index][0])
        return features, futures, torch.tensor(experiment, dtype=torch.long)


def project_normalized_increment(
    increment: Any,
    experiments: Any,
    area_weights: Any,
    wet: Any,
    temperature_target_normalized: Any,
) -> Any:
    """Apply differentiable SSH-volume and temperature-mean projections."""

    if (
        increment.ndim != 4
        or increment.shape[1] != STATE_CHANNEL_COUNT
        or experiments.ndim != 1
        or experiments.shape[0] != increment.shape[0]
        or area_weights.shape != (1, 1, *increment.shape[-2:])
        or wet.shape != area_weights.shape
        or temperature_target_normalized.shape != (3, 15)
    ):
        raise ValueError("loss-v3 projection tensors are inconsistent")
    if torch.any(experiments < 0) or torch.any(experiments > 2):
        raise ValueError("loss-v3 experiment indices must be 0, 1, or 2")
    temperature = increment[:, TEMPERATURE_SLICE]
    temperature_mean = (temperature * area_weights).sum(
        dim=(-2, -1),
        keepdim=True,
    )
    target = temperature_target_normalized[experiments][..., None, None]
    projected_temperature = temperature - temperature_mean + target
    ssh = increment[:, SSH_CHANNEL : SSH_CHANNEL + 1]
    projected_ssh = ssh - (ssh * area_weights).sum(
        dim=(-2, -1),
        keepdim=True,
    )
    return (
        torch.cat(
            (
                increment[:, : TEMPERATURE_SLICE.start],
                projected_temperature,
                projected_ssh,
            ),
            dim=1,
        )
        * wet
    )


def infer_experiments_from_static(
    features: Any,
    wind_signatures: Any,
    area_weights: Any,
) -> Any:
    """Identify the three fixed forcing regimes from normalized wind fields."""

    if (
        features.ndim != 4
        or features.shape[1] != STATE_CHANNEL_COUNT + 5
        or wind_signatures.shape != (3, *features.shape[-2:])
        or area_weights.shape != (1, 1, *features.shape[-2:])
    ):
        raise ValueError("loss-v3 forcing signatures are inconsistent")
    wind = features[:, STATE_CHANNEL_COUNT : STATE_CHANNEL_COUNT + 1]
    squared_distance = (
        (wind - wind_signatures[None]).square()
        * area_weights
    ).sum(dim=(-2, -1))
    return torch.argmin(squared_distance, dim=1)


class ProjectedIncrementModel(_BaseModule):
    """Wrap a raw successor so every returned increment is projected."""

    def __init__(
        self,
        model: Any,
        area_weights: Any,
        wet: Any,
        temperature_target_normalized: Any,
        wind_signatures: Any,
    ) -> None:
        if torch is None:  # pragma: no cover - guarded by runtime checks
            raise RuntimeError("projected Model C requires PyTorch")
        super().__init__()
        self.model = model
        self.register_buffer("area_weights", area_weights)
        self.register_buffer("wet", wet)
        self.register_buffer(
            "temperature_target_normalized",
            temperature_target_normalized,
        )
        self.register_buffer("wind_signatures", wind_signatures)

    def forward(self, features: Any) -> Any:
        experiments = infer_experiments_from_static(
            features,
            self.wind_signatures,
            self.area_weights,
        )
        return project_normalized_increment(
            self.model(features),
            experiments,
            self.area_weights,
            self.wet,
            self.temperature_target_normalized,
        )


def projected_unroll(
    model: Any,
    features: Any,
    experiments: Any,
    area_weights: Any,
    wet: Any,
    temperature_target_normalized: Any,
    steps: int,
) -> Any:
    """Autoregress with the linear output projection at every model call."""

    if steps <= 0:
        raise ValueError("projected rollout needs at least one step")
    current = features[:, :STATE_CHANNEL_COUNT]
    geometry = features[:, STATE_CHANNEL_COUNT:]
    predictions = []
    for _ in range(steps):
        increment = project_normalized_increment(
            model(torch.cat((current, geometry), dim=1)),
            experiments,
            area_weights,
            wet,
            temperature_target_normalized,
        )
        current = (current + increment) * wet
        predictions.append(current)
    return torch.stack(predictions, dim=1)


def rollout_conditioned_endpoint(
    model: Any,
    features: Any,
    experiments: Any,
    area_weights: Any,
    wet: Any,
    temperature_target_normalized: Any,
    *,
    endpoint_step: int,
) -> Any:
    """Take one differentiable call from a detached forecast-age state."""

    if endpoint_step not in TARGET_STEPS:
        raise ValueError("rollout-conditioned endpoint must be day 10--90")
    current = features[:, :STATE_CHANNEL_COUNT]
    geometry = features[:, STATE_CHANNEL_COUNT:]
    with torch.no_grad():
        for _ in range(endpoint_step - 1):
            increment = project_normalized_increment(
                model(torch.cat((current, geometry), dim=1)),
                experiments,
                area_weights,
                wet,
                temperature_target_normalized,
            )
            current = (current + increment) * wet
    current = current.detach()
    increment = project_normalized_increment(
        model(torch.cat((current, geometry), dim=1)),
        experiments,
        area_weights,
        wet,
        temperature_target_normalized,
    )
    return (current + increment) * wet


def slow_increment_bias_penalty(
    prediction_increment: Any,
    target_increment: Any,
    wet: Any,
    increment_scale: Any,
) -> Any:
    """Squared per-cell batch-mean signed error for all slow channels."""

    if (
        prediction_increment.shape != target_increment.shape
        or prediction_increment.ndim != 4
        or prediction_increment.shape[1] != STATE_CHANNEL_COUNT
        or wet.shape != (1, 1, *prediction_increment.shape[-2:])
        or increment_scale.shape != (STATE_CHANNEL_COUNT,)
    ):
        raise ValueError("loss-v3 bias-penalty tensors are inconsistent")
    scale = increment_scale[SLOW_SLICE][None, :, None, None].clamp_min(
        1.0e-12
    )
    signed_mean = (
        (prediction_increment[:, SLOW_SLICE] - target_increment[:, SLOW_SLICE])
        / scale
    ).mean(dim=0, keepdim=True)
    wet_count = wet.sum().clamp_min(1.0)
    return (
        signed_mean.square() * wet
    ).sum() / (wet_count * (SLOW_SLICE.stop - SLOW_SLICE.start))


def projection_residuals(
    projected_increment: Any,
    experiments: Any,
    area_weights: Any,
    temperature_target_normalized: Any,
) -> dict[str, float]:
    """Return maximum mean-constraint residuals for audit reporting."""

    temperature_mean = (
        projected_increment[:, TEMPERATURE_SLICE] * area_weights
    ).sum(dim=(-2, -1))
    target = temperature_target_normalized[experiments]
    ssh_mean = (
        projected_increment[:, SSH_CHANNEL : SSH_CHANNEL + 1]
        * area_weights
    ).sum(dim=(-2, -1))
    return {
        "maximum_absolute_ssh_increment_area_mean_normalized": float(
            torch.max(torch.abs(ssh_mean)).detach().cpu()
        ),
        "maximum_absolute_temperature_target_residual_normalized": float(
            torch.max(torch.abs(temperature_mean - target)).detach().cpu()
        ),
    }


def _forcing_signatures(
    static: Any,
    *,
    wind_mean: float,
    wind_scale: float,
    wet: np.ndarray,
) -> np.ndarray:
    values = np.asarray(static[:, 0], dtype=np.float32)
    values = (values - wind_mean) / wind_scale
    values[:, ~wet] = 0.0
    return np.ascontiguousarray(values, dtype=np.float32)


def load_rollout_conditioned_contract(
    path: str | Path,
    *,
    verify_source_files: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the loss-v3 contract frozen before any new training metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != VERSION:
        raise ValueError(f"expected loss-v3 contract {VERSION}")
    if (
        contract.get("contract_status")
        != "frozen_after_job_292064_before_loss_v3_training_metrics"
    ):
        raise ValueError("loss-v3 contract was not frozen")
    objective = contract.get("objective", {})
    if (
        objective.get("base_loss_version") != "v1"
        or tuple(objective.get("forecast_target_days", ()))
        != tuple(range(10, 91, 10))
        or objective.get("forecast_age_state")
        != "detach_all_preceding_projected_calls"
        or int(objective.get("differentiable_conditioned_calls", -1)) != 1
        or objective.get("target_schedule")
        != "cycle_10_to_90_days_by_optimizer_step"
        or float(objective.get("rollout_conditioned_weight", -1.0))
        != 0.0025
        or float(objective.get("slow_bias_penalty_weight", -1.0)) != 0.01
        or objective.get("projection_application")
        != "every_training_and_evaluation_model_call"
    ):
        raise ValueError("loss-v3 objective changed")
    fine_tune = contract.get("fine_tune", {})
    if (
        int(fine_tune.get("source_fine_tune_step", -1))
        != SOURCE_FINE_TUNE_STEP
        or int(fine_tune.get("maximum_steps", -1)) != 2880
        or tuple(fine_tune.get("checkpoint_steps", ()))
        != (480, 960, 1440, 1920, 2400, 2880)
        or int(fine_tune.get("batch_size", -1)) != 4
        or float(fine_tune.get("learning_rate", -1.0)) != 0.00002
        or tuple(fine_tune.get("adam_betas", ())) != (0.9, 0.95)
        or float(fine_tune.get("weight_decay", -1.0)) != 0.00001
    ):
        raise ValueError("loss-v3 fine-tune schedule changed")
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
        raise ValueError("loss-v3 read contract changed")
    if verify_source_files:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(f"loss-v3 source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def _checkpoint_path(directory: Path, step: int) -> Path:
    return directory / f"model_c_rollout_conditioned_step_{step:04d}.pt"


def _verify_artifacts(
    contract: Mapping[str, Any],
    dataset: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    Path,
    Path,
]:
    sources = contract["source_artifacts"]
    if _file_sha256(dataset / ".zmetadata") != sources[
        "dataset_metadata_sha256"
    ]:
        raise ModelCRolloutConditionedError("trajectory-v2 metadata changed")
    duration_contract, _, duration_sha = load_duration_contract(
        sources["duration_contract"]
    )
    if duration_sha != sources["duration_contract_sha256"]:
        raise ModelCRolloutConditionedError("duration contract changed")
    audit_contract, _, audit_sha = load_checkpoint_replay_contract(
        sources["checkpoint_replay_contract"]
    )
    if audit_sha != sources["checkpoint_replay_contract_sha256"]:
        raise ModelCRolloutConditionedError(
            "checkpoint replay contract changed"
        )
    successor_contract, _, successor_sha = load_successor_contract(
        sources["successor_training_contract"]
    )
    if successor_sha != sources["successor_training_contract_sha256"]:
        raise ModelCRolloutConditionedError(
            "successor-training contract changed"
        )
    _, _, bias_contract_sha = load_bias_projection_contract(
        sources["bias_projection_contract"]
    )
    if bias_contract_sha != sources["bias_projection_contract_sha256"]:
        raise ModelCRolloutConditionedError(
            "bias/projection contract changed"
        )
    duration_report = Path(sources["duration_report"]).resolve()
    duration_arrays = Path(sources["duration_arrays"]).resolve()
    source_checkpoint = Path(sources["duration_selected_checkpoint"]).resolve()
    bias_report = Path(sources["bias_projection_report"]).resolve()
    expected = {
        duration_report: sources["duration_report_sha256"],
        duration_arrays: sources["duration_arrays_sha256"],
        source_checkpoint: sources["duration_selected_checkpoint_sha256"],
        bias_report: sources["bias_projection_report_sha256"],
    }
    for artifact, digest in expected.items():
        if not artifact.is_file() or _file_sha256(artifact) != digest:
            raise ModelCRolloutConditionedError(
                f"loss-v3 source artifact changed: {artifact}"
            )
    duration = json.loads(duration_report.read_text())
    if (
        duration.get("status") != "complete"
        or duration.get("duration_decision", {}).get(
            "selected_fine_tune_step"
        )
        != SOURCE_FINE_TUNE_STEP
        or duration.get("report_content_sha256")
        != sources["duration_report_content_sha256"]
    ):
        raise ModelCRolloutConditionedError(
            "duration result does not authorize loss-v3"
        )
    bias = json.loads(bias_report.read_text())
    if (
        bias.get("status") != "complete"
        or bias.get("decision", {}).get("classification")
        != "feedback_amplification_not_explained_by_static_bias"
        or bias.get("decision", {}).get("next_action")
        != "retain_constraints_but_prioritize_rollout_conditioned_supervision_over_static_bias_correction"
        or bias.get("report_content_sha256")
        != sources["bias_projection_report_content_sha256"]
        or bias.get("validation_state_opened") is not False
        or bias.get("inference_state_opened") is not False
    ):
        raise ModelCRolloutConditionedError(
            "job-292064 decision does not authorize loss-v3"
        )
    if torch is None:
        raise RuntimeError("loss-v3 preflight requires PyTorch")
    payload = torch.load(
        source_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    validate_duration_source_payload(payload)
    del payload
    return (
        audit_contract,
        successor_contract,
        duration_report,
        duration_arrays,
        source_checkpoint,
        bias_report,
    )


def run_rollout_conditioned_loss_v3(
    dataset_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run the frozen training-only rollout-conditioned loss-v3 experiment."""

    if torch is None or DataLoader is None:
        raise RuntimeError("loss-v3 requires PyTorch")
    require_model_a_runtime()
    started = time.monotonic()
    contract, resolved_contract, contract_sha = (
        load_rollout_conditioned_contract(contract_path)
    )
    dataset = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite loss-v3 output: {output}")
    (
        audit_contract,
        successor_contract,
        duration_report_path,
        duration_arrays_path,
        source_checkpoint_path,
        bias_report_path,
    ) = _verify_artifacts(contract, dataset)

    fine_tune = contract["fine_tune"]
    seed = int(fine_tune["batch_order_seed"])
    seed_everything(seed)
    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    audit_records, audit_times, block_bounds = _audit_records(
        pair_codes,
        snapshot_codes,
        audit_contract,
    )
    training_records = records_for_rollout_split(
        pair_codes,
        1,
        rollout_steps=ROLLOUT_STEPS,
    )
    one_step_records = records_for_pair_split(pair_codes, 1)
    diagnostic_records = _sample_records_by_regime(
        one_step_records,
        count_per_regime=int(
            successor_contract["diagnostics"][
                "checkpoint_records_per_regime"
            ]
        ),
        seed=REFERENCE_DIAGNOSTIC_SEED,
    )

    loss_config = model_c_loss_config("v1")
    if loss_contract_sha256(loss_config) != MODEL_C_LOSS_V1_CONTRACT_SHA256:
        raise ModelCRolloutConditionedError("loss-v1 contract changed")
    architecture = architecture_from_candidate(
        successor_contract,
        REFERENCE_CANDIDATE,
    )
    training_dataset = ModelCLossV3Dataset(dataset, training_records)
    loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            training_dataset,
            int(fine_tune["batch_size"]),
            seed,
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    wet_array = training_dataset.wet
    wet = torch.from_numpy(wet_array.astype(np.float32))[None, None].to(device)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float64)
    area_array = wet_area_weights(latitude, wet_array)
    area_weights = torch.from_numpy(area_array.astype(np.float32))[
        None, None
    ].to(device)
    boundary_array = western_boundary_mask(
        wet_array,
        loss_config.western_boundary_width,
    )
    boundary = torch.from_numpy(boundary_array.astype(np.float32))[
        None, None
    ].to(device)
    increment_values = training_increment_scale(group, pair_codes)
    increment_scale = torch.from_numpy(increment_values).to(device)
    physical_scale = torch.from_numpy(
        training_dataset.scale.astype(np.float32)
    ).to(device)
    bias_report = json.loads(bias_report_path.read_text())
    temperature_target_physical = np.asarray(
        bias_report["teacher_forced_bias"][
            "temperature_truth_area_mean_tendency_by_regime_and_level_degC"
        ],
        dtype=np.float32,
    )
    temperature_target_normalized_array = (
        temperature_target_physical
        / training_dataset.scale[TEMPERATURE_SLICE][None]
    ).astype(np.float32)
    temperature_target_normalized = torch.from_numpy(
        temperature_target_normalized_array
    ).to(device)
    wind_signatures_array = _forcing_signatures(
        static,
        wind_mean=training_dataset.wind_mean,
        wind_scale=training_dataset.wind_scale,
        wet=wet_array,
    )
    wind_signatures = torch.from_numpy(wind_signatures_array).to(device)

    source_payload = torch.load(
        source_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    validate_duration_source_payload(
        source_payload,
        architecture=architecture.to_dict(),
    )
    model = build_successor(architecture).to(device)
    model.load_state_dict(source_payload["model_state_dict"])
    parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(fine_tune["learning_rate"]),
        betas=tuple(float(value) for value in fine_tune["adam_betas"]),
        weight_decay=float(fine_tune["weight_decay"]),
    )
    maximum_steps = int(fine_tune["maximum_steps"])
    checkpoint_steps = tuple(
        int(value) for value in fine_tune["checkpoint_steps"]
    )
    rollout_weight = float(
        contract["objective"]["rollout_conditioned_weight"]
    )
    bias_weight = float(contract["objective"]["slow_bias_penalty_weight"])
    climatology_scales = contract["objective"]["climatology_rmse_scales"]

    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    checkpoint_directory = temporary / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir()
    iterator = iter(loader)
    window_totals = {name: 0.0 for name in TRAINING_TERMS}
    window_samples = 0
    history = []
    saved = []

    for step in range(1, maximum_steps + 1):
        try:
            features, futures, experiments = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            features, futures, experiments = next(iterator)
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
        experiments = experiments.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )
        endpoint_step = TARGET_STEPS[(step - 1) % len(TARGET_STEPS)]
        model.train()
        base_predictions = projected_unroll(
            model,
            features,
            experiments,
            area_weights,
            wet,
            temperature_target_normalized,
            loss_config.rollout_steps,
        )
        terms = model_c_loss_terms(
            base_predictions,
            futures[:, : loss_config.rollout_steps],
            features[:, :STATE_CHANNEL_COUNT],
            wet,
            boundary,
            increment_scale,
            loss_config,
        )
        endpoint = rollout_conditioned_endpoint(
            model,
            features,
            experiments,
            area_weights,
            wet,
            temperature_target_normalized,
            endpoint_step=endpoint_step,
        )
        slow = slow_field_pushforward_loss(
            endpoint,
            futures[:, endpoint_step - 1],
            wet,
            physical_scale,
            climatology_scales[str(endpoint_step * HORIZON_DAYS)],
        )
        bias_penalty = slow_increment_bias_penalty(
            base_predictions[:, 0] - features[:, :STATE_CHANNEL_COUNT],
            futures[:, 0] - features[:, :STATE_CHANNEL_COUNT],
            wet,
            increment_scale,
        )
        training_values = {
            **terms,
            "rollout_conditioned_sst": slow["sst"],
            "rollout_conditioned_phihyd_surface": slow[
                "phihyd_surface"
            ],
            "slow_bias_penalty": bias_penalty,
        }
        training_values["total"] = (
            terms["total"]
            + rollout_weight * slow["mean"]
            + bias_weight * bias_penalty
        )
        if not all(
            bool(torch.isfinite(training_values[name]).item())
            for name in TRAINING_TERMS
        ):
            raise ModelCRolloutConditionedError(
                "loss-v3 training objective became non-finite"
            )
        optimizer.zero_grad(set_to_none=True)
        training_values["total"].backward()
        optimizer.step()
        size = int(features.shape[0])
        for name in TRAINING_TERMS:
            window_totals[name] += (
                float(training_values[name].detach().cpu()) * size
            )
        window_samples += size
        if step not in checkpoint_steps:
            continue
        training_window = {
            name: window_totals[name] / window_samples
            for name in TRAINING_TERMS
        }
        history_record = {
            "loss_v3_fine_tune_step": step,
            "source_fine_tune_step": SOURCE_FINE_TUNE_STEP,
            "forecast_target_day_at_checkpoint": endpoint_step * HORIZON_DAYS,
            "optimizer_learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "training_window": training_window,
        }
        history.append(history_record)
        checkpoint_path = _checkpoint_path(checkpoint_directory, step)
        payload = {
            "version": VERSION,
            "purpose": "rollout_conditioned_projected_loss_v3_checkpoint",
            "loss_v3_fine_tune_step": step,
            "source_fine_tune_step": SOURCE_FINE_TUNE_STEP,
            "candidate_id": REFERENCE_CANDIDATE,
            "architecture": architecture.to_dict(),
            "parameter_count": parameter_count,
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "base_loss_contract": loss_contract(loss_config),
            "base_loss_contract_sha256": loss_contract_sha256(loss_config),
            "loss_v3_objective": contract["objective"],
            "fine_tune_contract": fine_tune,
            "temperature_target_normalized": (
                temperature_target_normalized_array.tolist()
            ),
            "training_history_record": history_record,
            "model_state_dict": _checkpoint_state_dict(model),
        }
        torch.save(payload, checkpoint_path)
        saved.append(
            {
                "loss_v3_fine_tune_step": step,
                "fine_tune_step": step,
                "checkpoint": str(
                    output / CHECKPOINT_DIRECTORY / checkpoint_path.name
                ),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
            }
        )
        window_totals = {name: 0.0 for name in TRAINING_TERMS}
        window_samples = 0

    if len(saved) != len(checkpoint_steps):
        raise ModelCRolloutConditionedError(
            "loss-v3 did not save every contracted checkpoint"
        )

    mean, scale, _, _, wind_mean, wind_scale = _normalizers(group)
    climatology_state, climatology_derived, training_days = (
        _training_climatology(state, snapshot_codes, wet_array)
    )
    source_arrays = np.load(duration_arrays_path)
    if not np.array_equal(source_arrays["records"], audit_records):
        raise ModelCRolloutConditionedError(
            "duration evaluation records changed"
        )
    arrays: dict[str, np.ndarray] = {
        "records": audit_records.astype(np.int32),
        "training_times": audit_times.astype(np.int32),
        "lead_days": np.asarray(range(10, 91, 10), dtype=np.int16),
        "area_weights": area_array.astype(np.float32),
        "temperature_target_physical": temperature_target_physical,
        "temperature_target_normalized": (
            temperature_target_normalized_array
        ),
        "wind_signatures": wind_signatures_array,
    }
    for old_prefix, new_prefix in {
        "source_source_persistence": "source_persistence",
        "source_source_climatology": "source_climatology",
        "duration_5760": "source_duration_5760",
    }.items():
        for name in source_arrays.files:
            if name.startswith(old_prefix + "__"):
                arrays[
                    f"{new_prefix}__{name.split('__', 1)[1]}"
                ] = np.asarray(source_arrays[name])
    baselines = {
        method: {
            name.split("__", 1)[1]: np.asarray(source_arrays[name])
            for name in source_arrays.files
            if name.startswith(f"source_source_{method}__")
        }
        for method in ("persistence", "climatology")
    }

    summaries = []
    for saved_record, history_record in zip(saved, history):
        fine_tune_step = int(saved_record["loss_v3_fine_tune_step"])
        checkpoint_path = _checkpoint_path(
            checkpoint_directory,
            fine_tune_step,
        )
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        evaluation_model = build_successor(architecture).to(device)
        evaluation_model.load_state_dict(payload["model_state_dict"])
        evaluation_model.eval()
        projected_model = ProjectedIncrementModel(
            evaluation_model,
            area_weights,
            wet,
            temperature_target_normalized,
            wind_signatures,
        )
        projected_model.eval()
        stepper = ValidationStepper(
            kind="successor",
            model=projected_model,
            device=device,
            wet=wet_array,
            mean=mean,
            scale=scale,
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
            mean,
            scale,
            batch_size=int(contract["evaluation"]["batch_size"]),
        )
        for name, value in metrics.items():
            arrays[f"loss_v3_{fine_tune_step}__{name}"] = np.asarray(value)
        diagnostic = _one_step_diagnostics(
            projected_model,
            dataset,
            diagnostic_records,
            batch_size=16,
            device=device,
        )
        curves = lead_curve_summary(metrics, baselines, audit_records)
        gate = checkpoint_gate_summary(
            curves,
            diagnostic,
            metrics,
            audit_contract["checkpoint_gate"],
        )
        summaries.append(
            {
                **saved_record,
                "training_window": history_record["training_window"],
                "ten_day_diagnostic": diagnostic,
                "all_field_auc": _method_auc_summary(metrics),
                "lead_curves": curves,
                "checkpoint_gate": gate,
            }
        )
        del projected_model, evaluation_model, stepper, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()

    decision = select_pushforward_checkpoint(summaries)
    selected_step = int(decision["selected_fine_tune_step"])
    decision["selected_loss_v3_fine_tune_step"] = selected_step
    selected_source = _checkpoint_path(checkpoint_directory, selected_step)
    selected_payload = torch.load(
        selected_source,
        map_location="cpu",
        weights_only=False,
    )
    selected_path = temporary / SELECTED_CHECKPOINT_NAME
    torch.save(selected_payload, selected_path)

    restored = build_successor(architecture).to(device)
    restored.load_state_dict(selected_payload["model_state_dict"])
    reference_model = build_successor(architecture).to(device)
    reference_model.load_state_dict(selected_payload["model_state_dict"])
    first_examples = [training_dataset[index] for index in (0, 1, 2)]
    sample_features = torch.stack([value[0] for value in first_examples]).to(
        device=device,
        dtype=torch.float32,
    )
    sample_experiments = torch.stack(
        [value[2] for value in first_examples]
    ).to(device=device, dtype=torch.long)
    restored.eval()
    reference_model.eval()
    with torch.no_grad():
        reference = projected_unroll(
            reference_model,
            sample_features,
            sample_experiments,
            area_weights,
            wet,
            temperature_target_normalized,
            9,
        ).cpu()
        reloaded = projected_unroll(
            restored,
            sample_features,
            sample_experiments,
            area_weights,
            wet,
            temperature_target_normalized,
            9,
        ).cpu()
        sample_increment = project_normalized_increment(
            restored(sample_features),
            sample_experiments,
            area_weights,
            wet,
            temperature_target_normalized,
        )
        exactness = projection_residuals(
            sample_increment,
            sample_experiments,
            area_weights,
            temperature_target_normalized,
        )
    reload_exact = bool(torch.equal(reference, reloaded))
    if not reload_exact:
        raise ModelCRolloutConditionedError(
            "selected loss-v3 checkpoint did not reload exactly"
        )

    arrays_path = temporary / ARRAYS_NAME
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "status": "complete",
        "purpose": "training_only_rollout_conditioned_projected_loss_v3",
        "version": VERSION,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset),
        "dataset_metadata_sha256": _file_sha256(dataset / ".zmetadata"),
        "device": str(device),
        "candidate_id": REFERENCE_CANDIDATE,
        "architecture": architecture.to_dict(),
        "parameter_count": parameter_count,
        "base_loss_contract": loss_contract(loss_config),
        "base_loss_contract_sha256": loss_contract_sha256(loss_config),
        "loss_v3_objective": contract["objective"],
        "fine_tune_contract": fine_tune,
        "source_artifacts": {
            "duration_report": str(duration_report_path),
            "duration_report_sha256": _file_sha256(duration_report_path),
            "duration_arrays": str(duration_arrays_path),
            "duration_arrays_sha256": _file_sha256(duration_arrays_path),
            "source_checkpoint": str(source_checkpoint_path),
            "source_checkpoint_sha256": _file_sha256(
                source_checkpoint_path
            ),
            "bias_projection_report": str(bias_report_path),
            "bias_projection_report_sha256": _file_sha256(
                bias_report_path
            ),
        },
        "read_contract": contract["read_contract"],
        "counts": {
            "training_nine_step_rollouts": len(training_records),
            "training_one_step_pairs": len(one_step_records),
            "checkpoint_diagnostic_pairs": len(diagnostic_records),
            "evaluation_rollouts": int(audit_records.shape[0]),
            "training_climatology_snapshots_per_regime": training_days,
        },
        "record_contract": {
            "complete_training_blocks": [
                list(value) for value in block_bounds
            ],
            "training_times_sha256": _array_sha256(audit_times),
            "records_sha256": _array_sha256(audit_records),
        },
        "increment_scale": increment_values.tolist(),
        "temperature_target_physical_degC_per_ten_days": (
            temperature_target_physical.tolist()
        ),
        "projection_exactness": exactness,
        "history": history,
        "checkpoint_summary": summaries,
        "loss_v3_decision": decision,
        "selected_checkpoint": str(output / SELECTED_CHECKPOINT_NAME),
        "selected_checkpoint_sha256": _file_sha256(selected_path),
        "save_reload_nine_step_bitwise_exact": reload_exact,
        "arrays": str(output / ARRAYS_NAME),
        "arrays_sha256": _file_sha256(arrays_path),
        "elapsed_seconds": time.monotonic() - started,
        "validation_state_opened": False,
        "inference_opened": False,
        "intermediate_wind_opened": False,
        "response_or_adjoint_opened": False,
    }
    report["report_content_sha256"] = _json_sha256(report)
    (temporary / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, output)
    return report


def preflight_rollout_conditioned_loss_v3(
    dataset_path: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Verify immutable loss-v3 inputs without opening scientific states."""

    contract, resolved, digest = load_rollout_conditioned_contract(
        contract_path
    )
    dataset = Path(dataset_path).resolve()
    _verify_artifacts(contract, dataset)
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "source_fine_tune_step": SOURCE_FINE_TUNE_STEP,
        "fine_tune_steps": contract["fine_tune"]["maximum_steps"],
        "forecast_target_days": contract["objective"][
            "forecast_target_days"
        ],
        "read_contract": contract["read_contract"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--dataset", type=Path, required=True)
        child.add_argument("--contract", type=Path, required=True)
        if command == "run":
            child.add_argument("--output-dir", type=Path, required=True)
            child.add_argument(
                "--device",
                choices=("auto", "cpu", "cuda"),
                default="auto",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight_rollout_conditioned_loss_v3(
            args.dataset,
            args.contract,
        )
    else:
        result = run_rollout_conditioned_loss_v3(
            args.dataset,
            args.contract,
            args.output_dir,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
