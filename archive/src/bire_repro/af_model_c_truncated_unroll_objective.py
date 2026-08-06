"""Short truncated-unroll correction for Model C long-horizon slow fields.

This training-only experiment starts from the diagnostic-best step-5,760
duration checkpoint.  It preserves the architecture, loss-v1, data order,
slow-field normalization, and correction weight.  The only scientific change
is to backpropagate through three consecutive ten-day model calls and
supervise all three endpoints in alternating day-40--60 and day-70--90
windows.  Validation, inference, response, and adjoint states remain sealed.
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
from .af_model_b import (
    _unroll,
    records_for_rollout_split,
    western_boundary_mask,
)
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

try:
    import torch
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


TRUNCATED_VERSION = "model_c_truncated_unroll_objective_v2"
REPORT_NAME = "model_c_truncated_unroll_v2_report.json"
ARRAYS_NAME = "model_c_truncated_unroll_v2_arrays.npz"
CHECKPOINT_DIRECTORY = "fine_tune_checkpoints"
SELECTED_CHECKPOINT_NAME = "model_c_truncated_unroll_v2_best.pt"
SOURCE_FINE_TUNE_STEP = 5760
WINDOW_STEPS = ((4, 5, 6), (7, 8, 9))
TRAINING_TERMS = (
    *AUDIT_TERMS,
    "truncated_sst",
    "truncated_phihyd_surface",
)


class ModelCTruncatedUnrollError(RuntimeError):
    """Raised when the frozen truncated-unroll experiment is violated."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def truncated_pushforward_window(
    model: Any,
    features: Any,
    wet: Any,
    base_predictions: Any,
    *,
    endpoint_steps: Sequence[int],
) -> Any:
    """Return three consecutive predictions with a three-call gradient graph."""

    steps = tuple(int(value) for value in endpoint_steps)
    if (
        steps not in WINDOW_STEPS
        or base_predictions.ndim != 5
        or base_predictions.shape[1] != 3
    ):
        raise ValueError(
            "truncated window must be days 40--60 or days 70--90"
        )
    geometry = features[:, STATE_CHANNEL_COUNT:]
    current = base_predictions[:, -1].detach()
    with torch.no_grad():
        for _ in range(3, steps[0] - 1):
            current = (
                current + model(torch.cat((current, geometry), dim=1))
            ) * wet
    current = current.detach()
    predictions = []
    for _ in steps:
        current = (
            current + model(torch.cat((current, geometry), dim=1))
        ) * wet
        predictions.append(current)
    return torch.stack(predictions, dim=1)


def truncated_slow_field_loss(
    predictions: Any,
    futures: Any,
    wet: Any,
    state_scale: Any,
    climatology_rmse_scales: Mapping[str, Mapping[str, float]],
    *,
    endpoint_steps: Sequence[int],
) -> dict[str, Any]:
    """Average equal-field climatology-scaled RMSE over three endpoints."""

    steps = tuple(int(value) for value in endpoint_steps)
    if (
        steps not in WINDOW_STEPS
        or predictions.ndim != 5
        or predictions.shape[1] != len(steps)
        or futures.ndim != 5
        or futures.shape[1] < max(steps)
    ):
        raise ValueError("truncated slow-field tensors are inconsistent")
    losses = [
        slow_field_pushforward_loss(
            predictions[:, index],
            futures[:, step - 1],
            wet,
            state_scale,
            climatology_rmse_scales[str(step * HORIZON_DAYS)],
        )
        for index, step in enumerate(steps)
    ]
    sst = torch.stack([value["sst"] for value in losses]).mean()
    phihyd = torch.stack(
        [value["phihyd_surface"] for value in losses]
    ).mean()
    return {
        "mean": 0.5 * (sst + phihyd),
        "sst": sst,
        "phihyd_surface": phihyd,
    }


def load_truncated_contract(
    path: str | Path,
    *,
    verify_source_files: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the objective contract frozen before truncated-unroll metrics."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != TRUNCATED_VERSION:
        raise ValueError(f"expected truncated contract {TRUNCATED_VERSION}")
    if (
        contract.get("contract_status")
        != "frozen_after_operational_source_schema_fix_before_truncated_unroll_metrics"
    ):
        raise ValueError("truncated-unroll contract was not frozen")
    objective = contract.get("objective", {})
    if (
        objective.get("base_loss_version") != "v1"
        or tuple(
            tuple(int(day) for day in window)
            for window in objective.get("supervised_windows_days", ())
        )
        != ((40, 50, 60), (70, 80, 90))
        or objective.get("window_schedule")
        != "alternate_40_to_60_and_70_to_90_by_optimizer_step"
        or int(objective.get("differentiable_model_calls", -1)) != 3
        or objective.get("pre_window_state")
        != "detached_no_grad_rollout_from_day30"
        or objective.get("endpoint_reduction")
        != "equal_mean_over_three_endpoints_and_two_slow_fields"
        or float(objective.get("correction_weight", -1.0)) != 0.0025
        or tuple(objective.get("slow_fields", ()))
        != ("sst", "phihyd_surface")
    ):
        raise ValueError("truncated-unroll objective changed")
    fine_tune = contract.get("fine_tune", {})
    if (
        int(fine_tune.get("source_fine_tune_step", -1))
        != SOURCE_FINE_TUNE_STEP
        or int(fine_tune.get("maximum_steps", -1)) != 1920
        or tuple(fine_tune.get("checkpoint_steps", ()))
        != (480, 960, 1440, 1920)
        or int(fine_tune.get("batch_size", -1)) != 4
        or int(fine_tune.get("effective_batch_size", -1)) != 4
        or float(fine_tune.get("learning_rate", -1.0)) != 0.00002
        or tuple(fine_tune.get("adam_betas", ())) != (0.9, 0.95)
        or float(fine_tune.get("weight_decay", -1.0)) != 0.00001
    ):
        raise ValueError("truncated-unroll fine-tune schedule changed")
    read = contract.get("read_contract", {})
    if (
        read.get("training_pair_code") != 1
        or read.get("training_state_read") is not True
        or any(
            read.get(name) is not False
            for name in (
                "validation_state_read",
                "inference_read",
                "intermediate_wind_read",
                "response_read",
                "adjoint_read",
            )
        )
    ):
        raise ValueError("truncated-unroll read contract changed")
    if verify_source_files:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(
                    f"truncated-unroll source changed: {source}"
                )
    return contract, resolved, _file_sha256(resolved)


def _checkpoint_path(directory: Path, step: int) -> Path:
    return directory / f"model_c_truncated_unroll_step_{step:04d}.pt"


def validate_duration_source_payload(
    payload: Mapping[str, Any],
    *,
    architecture: Mapping[str, Any] | None = None,
) -> None:
    """Validate the duration checkpoint schema used by this continuation."""

    if (
        payload.get("version") != "model_c_pushforward_duration_v1"
        or payload.get("total_fine_tune_step") != SOURCE_FINE_TUNE_STEP
        or payload.get("base_loss_contract_sha256")
        != MODEL_C_LOSS_V1_CONTRACT_SHA256
        or "model_state_dict" not in payload
        or (
            architecture is not None
            and payload.get("architecture") != architecture
        )
    ):
        raise ModelCTruncatedUnrollError(
            "duration source checkpoint contract changed"
        )


def _verify_artifacts(
    contract: Mapping[str, Any],
    dataset: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    Path,
]:
    sources = contract["source_artifacts"]
    if _file_sha256(dataset / ".zmetadata") != sources[
        "dataset_metadata_sha256"
    ]:
        raise ModelCTruncatedUnrollError("trajectory-v2 metadata changed")
    duration_contract, _, duration_contract_sha = load_duration_contract(
        sources["duration_contract"]
    )
    if duration_contract_sha != sources["duration_contract_sha256"]:
        raise ModelCTruncatedUnrollError("duration contract changed")
    audit_contract, _, audit_contract_sha = load_checkpoint_replay_contract(
        sources["checkpoint_replay_contract"]
    )
    if audit_contract_sha != sources["checkpoint_replay_contract_sha256"]:
        raise ModelCTruncatedUnrollError(
            "checkpoint replay contract changed"
        )
    successor_contract, _, successor_contract_sha = load_successor_contract(
        sources["successor_training_contract"]
    )
    if successor_contract_sha != sources[
        "successor_training_contract_sha256"
    ]:
        raise ModelCTruncatedUnrollError(
            "successor-training contract changed"
        )
    report_path = Path(sources["duration_report"]).resolve()
    arrays_path = Path(sources["duration_arrays"]).resolve()
    checkpoint_path = Path(sources["duration_selected_checkpoint"]).resolve()
    expected = {
        report_path: sources["duration_report_sha256"],
        arrays_path: sources["duration_arrays_sha256"],
        checkpoint_path: sources["duration_selected_checkpoint_sha256"],
    }
    for artifact, digest in expected.items():
        if not artifact.is_file() or _file_sha256(artifact) != digest:
            raise ModelCTruncatedUnrollError(
                f"truncated-unroll source artifact changed: {artifact}"
            )
    report = json.loads(report_path.read_text())
    if (
        report.get("status") != "complete"
        or report.get("duration_decision", {}).get("passed") is not False
        or report.get("duration_decision", {}).get(
            "selected_fine_tune_step"
        )
        != SOURCE_FINE_TUNE_STEP
        or report.get("pushforward_v1_replay", {})
        .get("state_dict_comparison", {})
        .get("bitwise_exact")
        is not True
        or report.get("save_reload_nine_step_bitwise_exact") is not True
        or report.get("report_content_sha256")
        != sources["duration_report_content_sha256"]
    ):
        raise ModelCTruncatedUnrollError(
            "duration result does not authorize truncated-unroll correction"
        )
    if torch is None:
        raise RuntimeError("truncated-unroll preflight requires PyTorch")
    source_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    validate_duration_source_payload(source_payload)
    del source_payload
    return (
        audit_contract,
        successor_contract,
        report_path,
        arrays_path,
        checkpoint_path,
    )


def run_truncated_objective(
    dataset_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run the frozen training-only truncated-unroll correction and gate."""

    if torch is None or DataLoader is None:
        raise RuntimeError("truncated-unroll objective requires PyTorch")
    require_model_a_runtime()
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_truncated_contract(
        contract_path
    )
    dataset = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite truncated-unroll output: {output}"
        )
    (
        audit_contract,
        successor_contract,
        duration_report_path,
        duration_arrays_path,
        source_checkpoint_path,
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
        raise ModelCTruncatedUnrollError("loss-v1 contract changed")
    architecture = architecture_from_candidate(
        successor_contract,
        REFERENCE_CANDIDATE,
    )
    training_dataset = ModelCPushforwardDataset(dataset, training_records)
    batch_size = int(fine_tune["batch_size"])
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
    wet = torch.from_numpy(training_dataset.wet.astype(np.float32))[
        None, None
    ].to(device)
    boundary_array = western_boundary_mask(
        training_dataset.wet,
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
    correction_weight = float(contract["objective"]["correction_weight"])
    climatology_scales = contract["objective"][
        "climatology_rmse_scales"
    ]

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
        endpoint_steps = WINDOW_STEPS[(step - 1) % len(WINDOW_STEPS)]
        model.train()
        base_predictions = _unroll(
            model,
            features,
            wet,
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
        truncated_predictions = truncated_pushforward_window(
            model,
            features,
            wet,
            base_predictions,
            endpoint_steps=endpoint_steps,
        )
        slow = truncated_slow_field_loss(
            truncated_predictions,
            futures,
            wet,
            physical_scale,
            climatology_scales,
            endpoint_steps=endpoint_steps,
        )
        training_values = {
            **terms,
            "truncated_sst": slow["sst"],
            "truncated_phihyd_surface": slow["phihyd_surface"],
        }
        training_values["total"] = (
            terms["total"] + correction_weight * slow["mean"]
        )
        if not all(
            bool(torch.isfinite(training_values[name]).item())
            for name in TRAINING_TERMS
        ):
            raise ModelCTruncatedUnrollError(
                "truncated-unroll training loss became non-finite"
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
            "truncated_fine_tune_step": step,
            "total_fine_tune_step": SOURCE_FINE_TUNE_STEP + step,
            "optimizer_learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "training_window": training_window,
        }
        history.append(history_record)
        checkpoint_path = _checkpoint_path(checkpoint_directory, step)
        payload = {
            "version": TRUNCATED_VERSION,
            "purpose": "short_three_call_truncated_unroll_checkpoint",
            "truncated_fine_tune_step": step,
            "fine_tune_step": SOURCE_FINE_TUNE_STEP + step,
            "source_fine_tune_step": SOURCE_FINE_TUNE_STEP,
            "candidate_id": REFERENCE_CANDIDATE,
            "architecture": architecture.to_dict(),
            "parameter_count": parameter_count,
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "base_loss_contract": loss_contract(loss_config),
            "base_loss_contract_sha256": loss_contract_sha256(loss_config),
            "truncated_objective": contract["objective"],
            "fine_tune_contract": fine_tune,
            "training_history_record": history_record,
            "model_state_dict": _checkpoint_state_dict(model),
        }
        torch.save(payload, checkpoint_path)
        saved.append(
            {
                "truncated_fine_tune_step": step,
                "fine_tune_step": SOURCE_FINE_TUNE_STEP + step,
                "checkpoint": str(
                    output / CHECKPOINT_DIRECTORY / checkpoint_path.name
                ),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
            }
        )
        window_totals = {name: 0.0 for name in TRAINING_TERMS}
        window_samples = 0

    if len(saved) != len(checkpoint_steps):
        raise ModelCTruncatedUnrollError(
            "truncated fine-tune did not save every checkpoint"
        )

    mean, scale, wet_array, _, wind_mean, wind_scale = _normalizers(group)
    climatology_state, climatology_derived, training_days = (
        _training_climatology(state, snapshot_codes, wet_array)
    )
    source_arrays = np.load(duration_arrays_path)
    if not np.array_equal(source_arrays["records"], audit_records):
        raise ModelCTruncatedUnrollError(
            "duration evaluation records changed"
        )
    arrays: dict[str, np.ndarray] = {
        "records": audit_records.astype(np.int32),
        "training_times": audit_times.astype(np.int32),
        "lead_days": np.asarray(range(10, 91, 10), dtype=np.int16),
    }
    source_prefixes = {
        "source_source_persistence": "source_persistence",
        "source_source_climatology": "source_climatology",
        "duration_5760": "source_duration_5760",
    }
    for old_prefix, new_prefix in source_prefixes.items():
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
        truncated_step = int(saved_record["truncated_fine_tune_step"])
        checkpoint_path = _checkpoint_path(
            checkpoint_directory,
            truncated_step,
        )
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        evaluation_model = build_successor(architecture).to(device)
        evaluation_model.load_state_dict(payload["model_state_dict"])
        evaluation_model.eval()
        stepper = ValidationStepper(
            kind="successor",
            model=evaluation_model,
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
            arrays[f"truncated_{truncated_step}__{name}"] = np.asarray(value)
        diagnostic = _one_step_diagnostics(
            evaluation_model,
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
        del evaluation_model, stepper, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()

    decision_inputs = [
        {**item, "fine_tune_step": item["truncated_fine_tune_step"]}
        for item in summaries
    ]
    decision = select_pushforward_checkpoint(decision_inputs)
    selected_truncated_step = int(decision["selected_fine_tune_step"])
    decision["selected_truncated_fine_tune_step"] = selected_truncated_step
    decision["selected_total_fine_tune_step"] = (
        SOURCE_FINE_TUNE_STEP + selected_truncated_step
    )
    selected_source = _checkpoint_path(
        checkpoint_directory,
        selected_truncated_step,
    )
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
    features = training_dataset[0][0][None].to(
        device=device,
        dtype=torch.float32,
    )
    restored.eval()
    reference_model.eval()
    with torch.no_grad():
        reference = _unroll(reference_model, features, wet, 9).cpu()
        reloaded = _unroll(restored, features, wet, 9).cpu()
    reload_exact = bool(torch.equal(reference, reloaded))
    if not reload_exact:
        raise ModelCTruncatedUnrollError(
            "truncated selected checkpoint did not reload exactly"
        )

    arrays_path = temporary / ARRAYS_NAME
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "status": "complete",
        "purpose": "training_only_short_three_call_truncated_unroll_correction",
        "version": TRUNCATED_VERSION,
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
        "truncated_objective": contract["objective"],
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
        "history": history,
        "checkpoint_summary": summaries,
        "truncated_decision": decision,
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


def preflight_truncated_objective(
    dataset_path: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Verify immutable training-only sources without loading states."""

    contract, resolved, digest = load_truncated_contract(contract_path)
    dataset = Path(dataset_path).resolve()
    _verify_artifacts(contract, dataset)
    return {
        "status": "ready",
        "version": TRUNCATED_VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "source_fine_tune_step": contract["fine_tune"][
            "source_fine_tune_step"
        ],
        "fine_tune_steps": contract["fine_tune"]["maximum_steps"],
        "supervised_windows_days": contract["objective"][
            "supervised_windows_days"
        ],
        "differentiable_model_calls": contract["objective"][
            "differentiable_model_calls"
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
        result = preflight_truncated_objective(
            args.dataset,
            args.contract,
        )
    else:
        result = run_truncated_objective(
            args.dataset,
            args.contract,
            args.output_dir,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
