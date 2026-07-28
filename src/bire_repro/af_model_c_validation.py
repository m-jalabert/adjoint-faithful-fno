"""Bounded validation-only successive halving for forward-optimized Model C.

The search contract is frozen in ``config/model_c_validation_search_v1.json``
before any validation state metric is read.  Search tasks may read only pair
codes 1 (training) and 2 (validation).  Inference, intermediate-wind, response,
and adjoint data remain sealed until a separate three-seed freeze succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .af_a0 import records_for_pair_split
from .af_model_a import (
    ChunkAwareBatchSampler,
    ModelAPairDataset,
    _checkpoint_state_dict,
    require_model_a_runtime,
    seed_everything,
)
from .af_model_b import (
    ModelBRolloutDataset,
    _unroll,
    rollout_start_indices,
    western_boundary_mask,
)
from .af_model_c import (
    GROUP_SLICES,
    MODEL_C_LOSS_V1_CONTRACT_SHA256,
    ModelCArchitecture,
    build_model_c,
    loss_contract,
    loss_contract_sha256,
    model_c_loss_config,
    model_c_loss_terms,
    tapered_group_spectral_loss,
)
from .af_model_c_overfit import (
    AUDIT_TERMS,
    _device,
    _file_sha256,
    _save_reload_check,
)

try:  # Keep documentation-only imports usable without the optional ML stack.
    import torch
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


SEARCH_VERSION = "model_c_validation_search_v1"
SEARCH_REPORT_NAME = "model_c_validation_report.json"
SEARCH_CHECKPOINT_NAME = "model_c_validation_best.pt"
FINAL_SEED_REPORT_NAME = "model_c_final_seed_report.json"
FINAL_SEED_CHECKPOINT_NAME = "model_c_final_seed_best.pt"
EXPECTED_ROUNDS = (
    (1, 0.25, 1920, 5),
    (2, 0.50, 3840, 3),
    (3, 0.75, 5760, 2),
    (4, 1.00, 7680, 1),
)
EXPECTED_CHECKPOINT_FRACTIONS = (
    0.25,
    0.50,
    0.75,
    0.875,
    0.9375,
    0.96875,
    0.984375,
    1.0,
)
PHYSICS_STEPS = (3, 9, 18)
PHYSICS_LEADS = tuple(step * 10 for step in PHYSICS_STEPS)


class ModelCValidationError(RuntimeError):
    """Raised when a validation candidate violates the frozen search contract."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_search_contract(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    """Load and strictly validate the contract frozen before validation."""

    contract_path = Path(path).resolve()
    contract = json.loads(contract_path.read_text())
    if contract.get("version") != SEARCH_VERSION:
        raise ValueError(f"expected validation search contract {SEARCH_VERSION}")
    if (
        contract.get("contract_status")
        != "predeclared_before_validation_state_metrics_were_read"
    ):
        raise ValueError("Model C validation contract was not predeclared")
    sealed = contract.get("sealed_data", {})
    if (
        sealed.get("training_pair_code") != 1
        or sealed.get("validation_pair_code") != 2
        or sealed.get("inference_pair_code") != 3
        or any(
            sealed.get(name) is not False
            for name in (
                "inference_read",
                "intermediate_wind_read",
                "response_read",
                "adjoint_read",
            )
        )
    ):
        raise ValueError("Model C validation contract does not preserve sealed data")
    if contract.get("loss") != {
        "contract_sha256": MODEL_C_LOSS_V1_CONTRACT_SHA256,
        "version": "v1",
    }:
        raise ValueError("Model C validation search must preserve loss v1")

    candidates = contract.get("candidate_architectures")
    if not isinstance(candidates, list) or len(candidates) != 10:
        raise ValueError("Model C validation search requires exactly ten base candidates")
    candidate_ids = [candidate.get("candidate_id") for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids) or any(
        not isinstance(candidate_id, str) for candidate_id in candidate_ids
    ):
        raise ValueError("Model C validation candidate identifiers must be unique strings")
    expected_pairs = {
        (modes, width)
        for modes in ((12, 12), (16, 16), (16, 24), (24, 16), (24, 24))
        for width in (32, 64)
    }
    actual_pairs = set()
    for candidate in candidates:
        architecture = architecture_from_candidate(candidate)
        if architecture.n_layers != 4 or architecture.domain_padding != 0.10:
            raise ValueError("base Model C candidates require four layers and 10% padding")
        actual_pairs.add((architecture.n_modes, architecture.hidden_channels))
    if actual_pairs != expected_pairs:
        raise ValueError("Model C base candidate grid is incomplete")

    rounds = tuple(
        (
            int(item["stage"]),
            float(item["chronology_fraction"]),
            int(item["maximum_steps"]),
            int(item["survivor_count"]),
        )
        for item in contract.get("successive_halving_rounds", ())
    )
    if rounds != EXPECTED_ROUNDS:
        raise ValueError("Model C successive-halving rounds changed after predeclaration")
    fractions = tuple(float(value) for value in contract.get("checkpoint_fractions", ()))
    if fractions != EXPECTED_CHECKPOINT_FRACTIONS:
        raise ValueError("Model C checkpoint schedule changed after predeclaration")

    optimizer = contract.get("optimizer", {})
    expected_optimizer = {
        "adam_betas": [0.9, 0.95],
        "batch_size": 4,
        "decay_fraction": 0.75,
        "decay_factor": 0.2,
        "full_step_budget": 7680,
        "initial_learning_rate": 0.0005,
        "late_learning_rate": 0.0001,
        "validation_batch_size": 8,
        "weight_decay": 0.00001,
    }
    if optimizer != expected_optimizer:
        raise ValueError("Model C accepted optimizer schedule changed")
    if contract.get("search_seed") != 20260723:
        raise ValueError("Model C search seed changed")
    if contract.get("final_seed_gate", {}).get("seeds") != [
        20260723,
        20260724,
        20260725,
    ]:
        raise ValueError("Model C final seeds changed")
    metrics = contract.get("validation_metrics", {})
    if (
        metrics.get("long_rollout_leads_days") != [30, 90, 180]
        or metrics.get("long_rollout_starts_per_regime") != 16
        or metrics.get("western_boundary_width") != 4
    ):
        raise ValueError("Model C validation metric contract changed")
    return contract, contract_path, _file_sha256(contract_path)


def architecture_from_candidate(candidate: dict[str, Any]) -> ModelCArchitecture:
    """Construct and validate one declared candidate architecture."""

    return ModelCArchitecture(
        n_modes=tuple(int(value) for value in candidate["n_modes"]),
        hidden_channels=int(candidate["hidden_channels"]),
        n_layers=int(candidate["n_layers"]),
        domain_padding=float(candidate["domain_padding"]),
    )


def round_contract(contract: dict[str, Any], stage: int) -> dict[str, Any]:
    """Return one predeclared successive-halving round."""

    matches = [
        item for item in contract["successive_halving_rounds"] if int(item["stage"]) == stage
    ]
    if len(matches) != 1:
        raise ValueError(f"Model C validation stage {stage} is not declared")
    return matches[0]


def checkpoint_steps(contract: dict[str, Any], maximum_steps: int) -> tuple[int, ...]:
    """Convert frozen relative checkpoint locations to exact optimizer steps."""

    result = tuple(
        int(round(float(fraction) * maximum_steps))
        for fraction in contract["checkpoint_fractions"]
    )
    if (
        len(result) != len(set(result))
        or result[-1] != maximum_steps
        or any(step <= 0 or step > maximum_steps for step in result)
    ):
        raise ValueError("Model C validation checkpoint steps are invalid")
    return result


def chronology_records(
    pair_codes: Sequence[int],
    fraction: float,
    *,
    experiment_count: int = 3,
) -> tuple[tuple[int, int], ...]:
    """Select the earliest fraction of complete training rollouts per regime."""

    if not 0 < fraction <= 1:
        raise ValueError("Model C chronology fraction must lie in (0, 1]")
    starts = rollout_start_indices(pair_codes, 1, rollout_steps=3)
    count = int(math.floor(len(starts) * fraction))
    if count <= 0:
        raise ValueError("Model C chronology fraction selected no complete rollouts")
    selected = starts[:count]
    return tuple(
        (experiment, int(time_index))
        for experiment in range(experiment_count)
        for time_index in selected
    )


def long_validation_records(
    pair_codes: Sequence[int],
    starts_per_regime: int,
    *,
    experiment_count: int = 3,
) -> tuple[tuple[int, int], ...]:
    """Select evenly spaced, fixed validation starts supporting 180 days."""

    candidates = np.asarray(
        rollout_start_indices(pair_codes, 2, rollout_steps=max(PHYSICS_STEPS)),
        dtype=np.int64,
    )
    if starts_per_regime <= 0 or candidates.size < starts_per_regime:
        raise ValueError("not enough complete validation starts for Model C physics scores")
    indices = np.linspace(0, candidates.size - 1, starts_per_regime, dtype=np.int64)
    selected = candidates[np.unique(indices)]
    if selected.size != starts_per_regime:
        raise ValueError("Model C long-rollout validation starts are not unique")
    return tuple(
        (experiment, int(time_index))
        for experiment in range(experiment_count)
        for time_index in selected
    )


def verify_split_records(
    pair_codes: Sequence[int],
    snapshot_codes: Sequence[int],
    training_records: Sequence[tuple[int, int]],
    validation_records: Sequence[tuple[int, int]],
    long_records: Sequence[tuple[int, int]],
) -> None:
    """Independently prove that no selected record touches pair code 3."""

    pair = np.asarray(pair_codes, dtype=np.uint8)
    snapshot = np.asarray(snapshot_codes, dtype=np.uint8)
    for _, time_index in training_records:
        if any(pair[time_index + 10 * step] != 1 for step in range(3)):
            raise ModelCValidationError("training rollout crosses its declared pair split")
        if any(snapshot[time_index + 10 * step] != 1 for step in range(4)):
            raise ModelCValidationError("training rollout crosses its snapshot split")
    for _, time_index in validation_records:
        if pair[time_index] != 2:
            raise ModelCValidationError("ten-day validation record has the wrong pair code")
        if snapshot[time_index] != 2 or snapshot[time_index + 10] != 2:
            raise ModelCValidationError("ten-day validation record crosses a snapshot split")
    for _, time_index in long_records:
        if any(pair[time_index + 10 * step] != 2 for step in range(18)):
            raise ModelCValidationError("long validation rollout crosses its pair split")
        if any(snapshot[time_index + 10 * step] != 2 for step in range(19)):
            raise ModelCValidationError("long validation rollout crosses its snapshot split")


def _metric_accumulator() -> dict[str, dict[str, float]]:
    return {
        name: {"model_squared_error": 0.0, "persistence_squared_error": 0.0, "count": 0}
        for name in GROUP_SLICES
    }


def _accumulate_physical_errors(
    accumulator: dict[str, dict[str, float]],
    prediction: Any,
    persistence: Any,
    target: Any,
    scale: Any,
    mask: Any,
) -> None:
    """Accumulate physical-unit squared errors for every equal-status group."""

    wet_count = int(mask.sum().item())
    if wet_count <= 0:
        raise ModelCValidationError("Model C validation metric mask is empty")
    for name, channels in GROUP_SLICES.items():
        physical_scale = scale[:, channels]
        model_error = (prediction[:, channels] - target[:, channels]) * physical_scale
        baseline_error = (persistence[:, channels] - target[:, channels]) * physical_scale
        accumulator[name]["model_squared_error"] += float(
            (model_error.square() * mask).sum(dtype=torch.float64).cpu()
        )
        accumulator[name]["persistence_squared_error"] += float(
            (baseline_error.square() * mask).sum(dtype=torch.float64).cpu()
        )
        accumulator[name]["count"] += (
            int(prediction.shape[0]) * (channels.stop - channels.start) * wet_count
        )


def _finish_physical_errors(
    accumulator: dict[str, dict[str, float]],
) -> dict[str, Any]:
    model_rmse: dict[str, float] = {}
    persistence_rmse: dict[str, float] = {}
    ratios: dict[str, float] = {}
    for name, values in accumulator.items():
        count = int(values["count"])
        if count <= 0:
            raise ModelCValidationError("Model C validation metric has no samples")
        model_rmse[name] = math.sqrt(values["model_squared_error"] / count)
        persistence_rmse[name] = math.sqrt(
            values["persistence_squared_error"] / count
        )
        if persistence_rmse[name] <= 0 or not np.isfinite(persistence_rmse[name]):
            raise ModelCValidationError("Model C persistence RMSE is invalid")
        ratios[name] = model_rmse[name] / persistence_rmse[name]
    values = tuple(ratios.values())
    return {
        "model_rmse": model_rmse,
        "persistence_rmse": persistence_rmse,
        "ratio_to_persistence": ratios,
        "mean_group_ratio": float(np.mean(values)),
        "worst_group_ratio": float(max(values)),
        "all_groups_beat_persistence": all(value < 1.0 for value in values),
    }


def one_step_metrics(
    model: Any,
    loader: Any,
    dataset: ModelAPairDataset,
    *,
    device: Any,
    metric_mask: Any,
) -> dict[str, Any]:
    """Compute physical U/V/temperature/SSH RMSE ratios on fixed records."""

    model.eval()
    accumulator = _metric_accumulator()
    wet = torch.from_numpy(dataset.wet.astype(np.float32))[None, None].to(device)
    scale = torch.from_numpy(dataset.scale.astype(np.float32))[None, :, None, None].to(
        device
    )
    samples = 0
    with torch.no_grad():
        for features, _, future in loader:
            features = features.to(device=device, dtype=torch.float32, non_blocking=True)
            future = future.to(device=device, dtype=torch.float32, non_blocking=True)
            present = features[:, :46]
            prediction = (present + model(features)) * wet
            _accumulate_physical_errors(
                accumulator,
                prediction,
                present,
                future,
                scale,
                metric_mask,
            )
            samples += int(features.shape[0])
    result = _finish_physical_errors(accumulator)
    result["record_count"] = samples
    return result


def _normalise_raw_state(
    raw: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    wet: np.ndarray,
) -> np.ndarray:
    result = (raw - mean[None, :, None, None]) / scale[None, :, None, None]
    result[:, :, ~wet] = 0.0
    return np.ascontiguousarray(result, dtype=np.float32)


def long_rollout_metrics(
    model: Any,
    dataset_path: Path,
    records: Sequence[tuple[int, int]],
    initial_dataset: ModelAPairDataset,
    *,
    batch_size: int,
    device: Any,
    spectral_bins: int,
    boundary_width: int,
) -> dict[str, Any]:
    """Score full-domain, boundary, and local-increment spectra at 30/90/180 days."""

    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    state = group["state"]
    wet_array = initial_dataset.wet
    wet = torch.from_numpy(wet_array.astype(np.float32))[None, None].to(device)
    boundary_array = western_boundary_mask(wet_array, boundary_width)
    boundary = torch.from_numpy(boundary_array.astype(np.float32))[None, None].to(device)
    scale = torch.from_numpy(initial_dataset.scale.astype(np.float32))[
        None, :, None, None
    ].to(device)
    full_accumulators = {lead: _metric_accumulator() for lead in PHYSICS_LEADS}
    boundary_accumulators = {lead: _metric_accumulator() for lead in PHYSICS_LEADS}
    spectral = {
        lead: {"model": 0.0, "persistence": 0.0, "samples": 0}
        for lead in PHYSICS_LEADS
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            examples = [
                initial_dataset[index]
                for index in range(start, min(start + batch_size, len(records)))
            ]
            features = torch.stack([example[0] for example in examples]).to(
                device=device, dtype=torch.float32
            )
            present = features[:, :46]
            current = present
            geometry = features[:, 46:]
            previous_target = present
            for step in range(1, max(PHYSICS_STEPS) + 1):
                previous_prediction = current
                current = (current + model(torch.cat((current, geometry), dim=1))) * wet
                raw_target = np.stack(
                    [
                        np.asarray(
                            state[experiment, time_index + 10 * step],
                            dtype=np.float32,
                        )
                        for experiment, time_index in batch_records
                    ]
                )
                target_array = _normalise_raw_state(
                    raw_target,
                    initial_dataset.mean,
                    initial_dataset.scale,
                    wet_array,
                )
                target = torch.from_numpy(target_array).to(device)
                if step in PHYSICS_STEPS:
                    lead = step * 10
                    _accumulate_physical_errors(
                        full_accumulators[lead],
                        current,
                        present,
                        target,
                        scale,
                        wet,
                    )
                    _accumulate_physical_errors(
                        boundary_accumulators[lead],
                        current,
                        present,
                        target,
                        scale,
                        boundary,
                    )
                    target_increment = target - previous_target
                    model_loss = tapered_group_spectral_loss(
                        current - previous_prediction,
                        target_increment,
                        wet,
                        bins=spectral_bins,
                    )
                    persistence_loss = tapered_group_spectral_loss(
                        torch.zeros_like(target_increment),
                        target_increment,
                        wet,
                        bins=spectral_bins,
                    )
                    size = len(batch_records)
                    spectral[lead]["model"] += float(model_loss.cpu()) * size
                    spectral[lead]["persistence"] += (
                        float(persistence_loss.cpu()) * size
                    )
                    spectral[lead]["samples"] += size
                previous_target = target

    lead_metrics: dict[str, Any] = {}
    physics_components: list[float] = []
    for lead in PHYSICS_LEADS:
        full = _finish_physical_errors(full_accumulators[lead])
        boundary_result = _finish_physical_errors(boundary_accumulators[lead])
        samples = int(spectral[lead]["samples"])
        model_spectral = spectral[lead]["model"] / samples
        persistence_spectral = spectral[lead]["persistence"] / samples
        if persistence_spectral <= 0 or not np.isfinite(persistence_spectral):
            raise ModelCValidationError("Model C persistence spectral score is invalid")
        spectral_ratio = model_spectral / persistence_spectral
        lead_metrics[str(lead)] = {
            "full_domain": full,
            "western_boundary": boundary_result,
            "tapered_increment_spectrum": {
                "model_loss": model_spectral,
                "persistence_loss": persistence_spectral,
                "ratio_to_persistence": spectral_ratio,
            },
        }
        physics_components.extend(
            (
                full["mean_group_ratio"],
                boundary_result["mean_group_ratio"],
                spectral_ratio,
            )
        )
    return {
        "record_count": len(records),
        "lead_metrics": lead_metrics,
        "physics_score": float(np.mean(physics_components)),
        "physics_score_components": physics_components,
    }


def selection_key(
    ten_day: dict[str, Any],
    physics: dict[str, Any],
    parameter_count: int,
    candidate_id: str,
) -> list[Any]:
    """Return the frozen lexicographic candidate/checkpoint ranking key."""

    return [
        0 if ten_day["all_groups_beat_persistence"] else 1,
        float(ten_day["worst_group_ratio"]),
        float(physics["physics_score"]),
        int(parameter_count),
        candidate_id,
    ]


def _average_training_window(
    totals: dict[str, float],
    samples: int,
) -> dict[str, float]:
    if samples <= 0:
        raise ModelCValidationError("Model C training window contains no samples")
    return {name: totals[name] / samples for name in AUDIT_TERMS}


def _source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        name: _file_sha256(package / name)
        for name in (
            "af_model_a.py",
            "af_model_b.py",
            "af_model_c.py",
            "af_model_c_validation.py",
        )
    }


def _verify_upstream_contracts(
    contract: dict[str, Any],
    dataset_path: Path,
    diagnostics_path: Path,
    calibration_path: Path,
    optimization_path: Path,
    loss_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = contract["source_contracts"]
    observed = {
        "dataset_metadata_sha256": _file_sha256(dataset_path / ".zmetadata"),
        "training_diagnostics_sha256": _file_sha256(diagnostics_path),
        "calibration_report_sha256": _file_sha256(calibration_path),
        "optimization_manifest_sha256": _file_sha256(optimization_path),
        "loss_manifest_sha256": _file_sha256(loss_manifest_path),
    }
    if observed != source:
        raise ModelCValidationError("Model C validation upstream provenance changed")
    diagnostics = json.loads(diagnostics_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    optimization = json.loads(optimization_path.read_text())
    loss_manifest = json.loads(loss_manifest_path.read_text())
    for name, report in (("diagnostics", diagnostics), ("calibration", calibration)):
        read_contract = report.get("read_contract", {})
        if (
            read_contract.get("validation_read") is not False
            or read_contract.get("inference_read") is not False
        ):
            raise ModelCValidationError(
                f"Model C {name} did not preserve pre-validation sealing"
            )
    if (
        optimization.get("status") != "accepted_for_bounded_validation_search"
        or optimization.get("loss_contract_sha256")
        != MODEL_C_LOSS_V1_CONTRACT_SHA256
    ):
        raise ModelCValidationError("Model C optimizer was not accepted for validation")
    if loss_manifest.get("status") != "frozen_training_only":
        raise ModelCValidationError("Model C loss-v1 manifest is not frozen")
    return diagnostics, optimization


def _eligible_candidates(
    contract: dict[str, Any],
    stage: int,
    selection_manifest_path: str | Path | None,
    contract_sha256: str,
) -> tuple[dict[str, Any], ...]:
    if stage == 1:
        if selection_manifest_path is not None:
            raise ValueError("stage 1 must not receive a survivor manifest")
        return tuple(contract["candidate_architectures"])
    if selection_manifest_path is None:
        raise ValueError(f"stage {stage} requires the preceding survivor manifest")
    path = Path(selection_manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if (
        manifest.get("version") != SEARCH_VERSION
        or int(manifest.get("stage", -1)) != stage - 1
        or manifest.get("contract_sha256") != contract_sha256
        or manifest.get("status") not in (
            "complete",
            "architecture_selected_pending_three_seed_gate",
        )
    ):
        raise ModelCValidationError("invalid Model C preceding-stage selection manifest")
    candidate_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in contract["candidate_architectures"]
    }
    survivors = tuple(manifest.get("survivors", ()))
    if not survivors or any(candidate_id not in candidate_by_id for candidate_id in survivors):
        raise ModelCValidationError("survivor manifest contains an unknown candidate")
    return tuple(candidate_by_id[candidate_id] for candidate_id in survivors)


def _selected_final_candidate(
    contract: dict[str, Any],
    selection_manifest_path: str | Path,
    contract_sha256: str,
) -> tuple[dict[str, Any], Path, str]:
    """Verify and return the architecture selected by the complete search."""

    path = Path(selection_manifest_path).resolve()
    manifest = json.loads(path.read_text())
    diagnostics = manifest.get("conditional_diagnostics", {})
    if (
        manifest.get("version") != SEARCH_VERSION
        or int(manifest.get("stage", -1)) != 4
        or manifest.get("contract_sha256") != contract_sha256
        or manifest.get("status")
        != "architecture_selected_pending_three_seed_gate"
        or manifest.get("all_candidate_reports_present") is not True
        or manifest.get("inference_opened") is not False
        or manifest.get("response_or_adjoint_opened") is not False
        or diagnostics.get("wall_leakage", {}).get("triggered") is not False
        or diagnostics.get("insufficient_capacity", {}).get("triggered") is not False
    ):
        raise ModelCValidationError(
            "invalid Model C final architecture-selection manifest"
        )
    survivors = manifest.get("survivors")
    ranking = manifest.get("ranking")
    if (
        not isinstance(survivors, list)
        or len(survivors) != 1
        or not isinstance(ranking, list)
        or not ranking
        or ranking[0].get("candidate_id") != survivors[0]
    ):
        raise ModelCValidationError(
            "Model C final architecture selection is not unique"
        )
    candidate_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in contract["candidate_architectures"]
    }
    candidate = candidate_by_id.get(survivors[0])
    if candidate is None:
        raise ModelCValidationError(
            "Model C final architecture selection is not declared"
        )
    selected_entry = ranking[0]
    report_path = Path(selected_entry.get("report", "")).resolve()
    checkpoint_path = Path(selected_entry.get("checkpoint", "")).resolve()
    if (
        not report_path.is_file()
        or _file_sha256(report_path) != selected_entry.get("report_sha256")
        or not checkpoint_path.is_file()
        or _file_sha256(checkpoint_path) != selected_entry.get("checkpoint_sha256")
    ):
        raise ModelCValidationError(
            "Model C selected search artifact does not match its frozen hash"
        )
    report = json.loads(report_path.read_text())
    if (
        report.get("status") != "complete"
        or int(report.get("stage", -1)) != 4
        or report.get("candidate_id") != survivors[0]
        or report.get("search_contract_sha256") != contract_sha256
        or report.get("architecture")
        != architecture_from_candidate(candidate).to_dict()
        or report.get("checkpoint") != str(checkpoint_path)
        or report.get("checkpoint_sha256") != selected_entry.get("checkpoint_sha256")
        or report.get("read_contract", {}).get("inference_read") is not False
        or report.get("read_contract", {}).get("response_read") is not False
        or report.get("read_contract", {}).get("adjoint_read") is not False
        or report.get("save_reload_three_step_bitwise_exact") is not True
    ):
        raise ModelCValidationError(
            "Model C selected search report violates the frozen contract"
        )
    return candidate, path, _file_sha256(path)


def resolve_candidate_id(
    contract_path: str | Path,
    *,
    stage: int,
    array_index: int,
    selection_manifest_path: str | Path | None = None,
) -> str:
    """Resolve one scheduler-array index without reading any scientific data."""

    contract, _, contract_sha = load_search_contract(contract_path)
    candidates = _eligible_candidates(
        contract,
        stage,
        selection_manifest_path,
        contract_sha,
    )
    if not 0 <= array_index < len(candidates):
        raise ValueError(
            f"array index {array_index} is outside the {len(candidates)} stage candidates"
        )
    return str(candidates[array_index]["candidate_id"])


def resolve_final_seed(contract_path: str | Path, *, array_index: int) -> int:
    """Resolve one final-gate scheduler index to a predeclared seed."""

    contract, _, _ = load_search_contract(contract_path)
    seeds = tuple(int(seed) for seed in contract["final_seed_gate"]["seeds"])
    if not 0 <= array_index < len(seeds):
        raise ValueError(
            f"array index {array_index} is outside the {len(seeds)} final seeds"
        )
    return seeds[array_index]


def run_validation_candidate(
    dataset_path: str | Path,
    diagnostics_path: str | Path,
    calibration_path: str | Path,
    optimization_path: str | Path,
    loss_manifest_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    stage: int,
    candidate_id: str,
    selection_manifest_path: str | Path | None = None,
    device_name: str = "auto",
    training_seed: int | None = None,
) -> dict[str, Any]:
    """Train and score one immutable search or final-seed candidate."""

    require_model_a_runtime()
    contract, resolved_contract_path, contract_sha = load_search_contract(contract_path)
    round_spec = round_contract(contract, stage)
    final_seed_run = training_seed is not None
    if final_seed_run:
        if stage != 4 or selection_manifest_path is None:
            raise ValueError(
                "Model C final-seed runs require stage 4 and its selection manifest"
            )
        candidate, _, _ = _selected_final_candidate(
            contract,
            selection_manifest_path,
            contract_sha,
        )
        if candidate["candidate_id"] != candidate_id:
            raise ModelCValidationError(
                f"candidate {candidate_id!r} was not selected for the final seed gate"
            )
        seed = int(training_seed)
        if seed not in {
            int(value) for value in contract["final_seed_gate"]["seeds"]
        }:
            raise ModelCValidationError(
                f"seed {seed} is not declared for the Model C final gate"
            )
        run_phase = "final_seed_gate"
        purpose = "validation_only_model_c_three_seed_gate"
        checkpoint_name = FINAL_SEED_CHECKPOINT_NAME
        report_name = FINAL_SEED_REPORT_NAME
    else:
        eligible = _eligible_candidates(
            contract,
            stage,
            selection_manifest_path,
            contract_sha,
        )
        matches = [
            candidate
            for candidate in eligible
            if candidate["candidate_id"] == candidate_id
        ]
        if len(matches) != 1:
            raise ModelCValidationError(
                f"candidate {candidate_id!r} is not eligible for validation "
                f"stage {stage}"
            )
        candidate = matches[0]
        seed = int(contract["search_seed"])
        run_phase = "successive_halving"
        purpose = "bounded_validation_only_model_c_successive_halving"
        checkpoint_name = SEARCH_CHECKPOINT_NAME
        report_name = SEARCH_REPORT_NAME
    architecture = architecture_from_candidate(candidate)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model C output: {output}")
    dataset_path = Path(dataset_path).resolve()
    diagnostics_path = Path(diagnostics_path).resolve()
    calibration_path = Path(calibration_path).resolve()
    optimization_path = Path(optimization_path).resolve()
    loss_manifest_path = Path(loss_manifest_path).resolve()
    diagnostics, _ = _verify_upstream_contracts(
        contract,
        dataset_path,
        diagnostics_path,
        calibration_path,
        optimization_path,
        loss_manifest_path,
    )
    increment_values = np.asarray(
        diagnostics["increment_rms_normalized_state_units"], dtype=np.float32
    )
    if increment_values.shape != (46,) or np.any(increment_values <= 0):
        raise ModelCValidationError("Model C validation increment scales are invalid")

    seed_everything(seed)
    device = _device(device_name)
    loss_config = model_c_loss_config("v1")
    if loss_contract_sha256(loss_config) != MODEL_C_LOSS_V1_CONTRACT_SHA256:
        raise ModelCValidationError("Model C loss-v1 code contract changed")

    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    fraction = float(round_spec["chronology_fraction"])
    training_records = chronology_records(pair_codes, fraction)
    validation_records = records_for_pair_split(pair_codes, 2)
    long_records = long_validation_records(
        pair_codes,
        int(contract["validation_metrics"]["long_rollout_starts_per_regime"]),
    )
    verify_split_records(
        pair_codes,
        snapshot_codes,
        training_records,
        validation_records,
        long_records,
    )

    training_dataset = ModelBRolloutDataset(dataset_path, training_records)
    validation_dataset = ModelAPairDataset(dataset_path, validation_records)
    long_initial_dataset = ModelAPairDataset(dataset_path, long_records)
    normalization = {
        "state_mean_sha256": _array_sha256(
            np.asarray(training_dataset.mean, dtype=np.float32)
        ),
        "state_scale_sha256": _array_sha256(
            np.asarray(training_dataset.scale, dtype=np.float32)
        ),
        "wind_mean": float(training_dataset.wind_mean),
        "wind_scale": float(training_dataset.wind_scale),
    }
    normalization["contract_sha256"] = _json_sha256(normalization)
    optimizer_contract = contract["optimizer"]
    train_loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            training_dataset,
            int(optimizer_contract["batch_size"]),
            seed,
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(optimizer_contract["validation_batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    training_metric_dataset = ModelAPairDataset(dataset_path, training_records)
    training_metric_loader = DataLoader(
        training_metric_dataset,
        batch_size=int(optimizer_contract["validation_batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    wet = torch.from_numpy(training_dataset.wet.astype(np.float32))[None, None].to(device)
    boundary_array = western_boundary_mask(
        training_dataset.wet, loss_config.western_boundary_width
    )
    boundary = torch.from_numpy(boundary_array.astype(np.float32))[None, None].to(device)
    increment_scale = torch.from_numpy(increment_values).to(device)
    model = build_model_c(architecture).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_contract["initial_learning_rate"]),
        betas=tuple(float(value) for value in optimizer_contract["adam_betas"]),
        weight_decay=float(optimizer_contract["weight_decay"]),
    )

    maximum_steps = int(round_spec["maximum_steps"])
    decay_step = int(round(maximum_steps * float(optimizer_contract["decay_fraction"])))
    evaluation_steps = checkpoint_steps(contract, maximum_steps)
    iterator = iter(train_loader)
    window_totals = {name: 0.0 for name in AUDIT_TERMS}
    window_samples = 0
    history: list[dict[str, Any]] = []
    best_key: tuple[Any, ...] | None = None
    best_state: dict[str, Any] | None = None
    best_record: dict[str, Any] | None = None
    started = time.monotonic()

    for step in range(1, maximum_steps + 1):
        if step == decay_step + 1:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= float(optimizer_contract["decay_factor"])
        try:
            features, futures = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            features, futures = next(iterator)
        model.train()
        features = features.to(device=device, dtype=torch.float32, non_blocking=True)
        futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
        predictions = _unroll(model, features, wet, loss_config.rollout_steps)
        terms = model_c_loss_terms(
            predictions,
            futures,
            features[:, :46],
            wet,
            boundary,
            increment_scale,
            loss_config,
        )
        if not all(bool(torch.isfinite(terms[name]).item()) for name in AUDIT_TERMS):
            raise ModelCValidationError("Model C validation training loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        optimizer.step()
        size = int(features.shape[0])
        for name in AUDIT_TERMS:
            window_totals[name] += float(terms[name].detach().cpu()) * size
        window_samples += size

        if step not in evaluation_steps:
            continue
        ten_day = one_step_metrics(
            model,
            validation_loader,
            validation_dataset,
            device=device,
            metric_mask=wet,
        )
        physics = long_rollout_metrics(
            model,
            dataset_path,
            long_records,
            long_initial_dataset,
            batch_size=int(optimizer_contract["validation_batch_size"]),
            device=device,
            spectral_bins=loss_config.spectral_bins,
            boundary_width=loss_config.western_boundary_width,
        )
        key = selection_key(ten_day, physics, parameter_count, candidate_id)
        record = {
            "optimizer_step": step,
            "optimizer_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_window": _average_training_window(
                window_totals,
                window_samples,
            ),
            "validation_ten_day": ten_day,
            "validation_physics": physics,
            "selection_key": key,
        }
        history.append(record)
        if best_key is None or tuple(key) < best_key:
            best_key = tuple(key)
            best_state = _checkpoint_state_dict(model)
            best_record = record
        window_totals = {name: 0.0 for name in AUDIT_TERMS}
        window_samples = 0

    if best_state is None or best_record is None:
        raise ModelCValidationError("Model C validation run selected no checkpoint")
    model.load_state_dict(best_state)
    training_ten_day = one_step_metrics(
        model,
        training_metric_loader,
        training_metric_dataset,
        device=device,
        metric_mask=wet,
    )
    selected_checkpoint = dict(best_record)
    selected_checkpoint["training_ten_day"] = training_ten_day

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    checkpoint_path = output / checkpoint_name
    features, _, _ = validation_dataset[0]
    reload_features = features[None].to(device=device, dtype=torch.float32)
    payload = {
        "model_class": "ModelCForwardOptimizedResidualFNO",
        "purpose": purpose,
        "run_phase": run_phase,
        "search_version": SEARCH_VERSION,
        "search_contract": str(resolved_contract_path),
        "search_contract_sha256": contract_sha,
        "stage": stage,
        "candidate_id": candidate_id,
        "architecture": architecture.to_dict(),
        "parameter_count": parameter_count,
        "search_seed": int(contract["search_seed"]),
        "training_seed": seed,
        "loss_contract": loss_contract(loss_config),
        "loss_contract_sha256": loss_contract_sha256(loss_config),
        "optimizer_contract": optimizer_contract,
        "round_contract": round_spec,
        "dataset": str(dataset_path),
        "dataset_metadata_sha256": _file_sha256(dataset_path / ".zmetadata"),
        "normalization": normalization,
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": _file_sha256(diagnostics_path),
        "calibration": str(calibration_path),
        "calibration_sha256": _file_sha256(calibration_path),
        "optimization_manifest": str(optimization_path),
        "optimization_manifest_sha256": _file_sha256(optimization_path),
        "loss_manifest": str(loss_manifest_path),
        "loss_manifest_sha256": _file_sha256(loss_manifest_path),
        "selection_manifest": (
            str(Path(selection_manifest_path).resolve())
            if selection_manifest_path is not None
            else None
        ),
        "selection_manifest_sha256": (
            _file_sha256(Path(selection_manifest_path).resolve())
            if selection_manifest_path is not None
            else None
        ),
        "training_records": [list(record) for record in training_records],
        "validation_records": [list(record) for record in validation_records],
        "long_validation_records": [list(record) for record in long_records],
        "selected_checkpoint": selected_checkpoint,
        "history": history,
        "model_state_dict": _checkpoint_state_dict(model),
    }
    bitwise_exact = _save_reload_check(
        model,
        architecture,
        reload_features,
        wet,
        checkpoint_path,
        payload,
        device,
    )
    if not bitwise_exact:
        raise ModelCValidationError("Model C validation checkpoint reload is not exact")
    report = {
        "status": "complete",
        "purpose": purpose,
        "run_phase": run_phase,
        "search_version": SEARCH_VERSION,
        "search_contract": str(resolved_contract_path),
        "search_contract_sha256": contract_sha,
        "stage": stage,
        "candidate_id": candidate_id,
        "architecture": architecture.to_dict(),
        "parameter_count": parameter_count,
        "search_seed": int(contract["search_seed"]),
        "training_seed": seed,
        "round_contract": round_spec,
        "optimizer_contract": optimizer_contract,
        "optimizer_decay_step": decay_step,
        "evaluation_steps": list(evaluation_steps),
        "loss_contract": loss_contract(loss_config),
        "loss_contract_sha256": loss_contract_sha256(loss_config),
        "dataset": str(dataset_path),
        "dataset_metadata_sha256": _file_sha256(dataset_path / ".zmetadata"),
        "normalization": normalization,
        "source_hashes": _source_hashes(),
        "read_contract": {
            "pair_split_codes_read": [1, 2],
            "snapshot_split_codes_read": [1, 2],
            "validation_read": True,
            "inference_read": False,
            "intermediate_wind_read": False,
            "response_read": False,
            "adjoint_read": False,
        },
        "record_counts": {
            "training": len(training_records),
            "validation_ten_day": len(validation_records),
            "validation_long_rollout": len(long_records),
        },
        "training_records_sha256": _json_sha256(training_records),
        "validation_records_sha256": _json_sha256(validation_records),
        "long_validation_records_sha256": _json_sha256(long_records),
        "selection_manifest": payload["selection_manifest"],
        "selection_manifest_sha256": payload["selection_manifest_sha256"],
        "history": history,
        "selected_checkpoint": selected_checkpoint,
        "ten_day_validation_eligible": bool(
            selected_checkpoint["validation_ten_day"][
                "all_groups_beat_persistence"
            ]
        ),
        "save_reload_three_step_bitwise_exact": bitwise_exact,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "device": str(device),
        "device_metadata": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "neuraloperator_version": metadata.version("neuraloperator"),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / report_name).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def run_final_seed_candidate(
    dataset_path: str | Path,
    diagnostics_path: str | Path,
    calibration_path: str | Path,
    optimization_path: str | Path,
    loss_manifest_path: str | Path,
    contract_path: str | Path,
    selection_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Retrain the selected architecture from scratch for one final seed."""

    contract, _, contract_sha = load_search_contract(contract_path)
    candidate, _, _ = _selected_final_candidate(
        contract,
        selection_manifest_path,
        contract_sha,
    )
    return run_validation_candidate(
        dataset_path,
        diagnostics_path,
        calibration_path,
        optimization_path,
        loss_manifest_path,
        contract_path,
        output_dir,
        stage=4,
        candidate_id=str(candidate["candidate_id"]),
        selection_manifest_path=selection_manifest_path,
        device_name=device_name,
        training_seed=seed,
    )


def _report_selection_key(report: dict[str, Any]) -> tuple[Any, ...]:
    key = report.get("selected_checkpoint", {}).get("selection_key")
    if (
        not isinstance(key, list)
        or len(key) != 5
        or key[0] not in (0, 1)
        or not all(np.isfinite(float(value)) for value in key[1:4])
        or not isinstance(key[4], str)
    ):
        raise ModelCValidationError("candidate report has an invalid selection key")
    return tuple(key)


def _conditional_diagnostics(ranked_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    finalists = tuple(ranked_reports[:2])
    wall_rows = []
    capacity_rows = []
    for report in finalists:
        selected = report["selected_checkpoint"]
        lead = selected["validation_physics"]["lead_metrics"]["180"]
        full_ratio = float(lead["full_domain"]["mean_group_ratio"])
        boundary_ratio = float(lead["western_boundary"]["mean_group_ratio"])
        leakage_ratio = boundary_ratio / max(full_ratio, 1.0e-30)
        wall_rows.append(
            {
                "candidate_id": report["candidate_id"],
                "full_domain_mean_ratio": full_ratio,
                "boundary_mean_ratio": boundary_ratio,
                "boundary_to_full_ratio": leakage_ratio,
                "triggered": boundary_ratio > 1.0 and leakage_ratio >= 1.25,
            }
        )
        training_worst = float(
            selected["training_ten_day"]["worst_group_ratio"]
        )
        capacity_rows.append(
            {
                "candidate_id": report["candidate_id"],
                "training_worst_group_ratio": training_worst,
                "triggered": training_worst >= 1.0,
            }
        )
    return {
        "wall_leakage": {
            "triggered": any(row["triggered"] for row in wall_rows),
            "thresholds": {
                "boundary_to_full_ratio": 1.25,
                "boundary_mean_ratio_must_exceed": 1.0,
            },
            "finalists": wall_rows,
        },
        "insufficient_capacity": {
            "triggered": any(row["triggered"] for row in capacity_rows),
            "threshold": 1.0,
            "finalists": capacity_rows,
        },
    }


def select_validation_stage(
    contract_path: str | Path,
    stage_root: str | Path,
    output_path: str | Path,
    *,
    stage: int,
    eligible_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rank one complete round and write its immutable survivor manifest."""

    contract, resolved_contract_path, contract_sha = load_search_contract(contract_path)
    round_spec = round_contract(contract, stage)
    eligible = _eligible_candidates(
        contract,
        stage,
        eligible_manifest_path,
        contract_sha,
    )
    root = Path(stage_root).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model C selection manifest: {output}")
    reports: list[dict[str, Any]] = []
    report_entries = []
    for candidate in eligible:
        candidate_id = candidate["candidate_id"]
        report_path = root / candidate_id / SEARCH_REPORT_NAME
        report = json.loads(report_path.read_text())
        checkpoint_path = Path(report.get("checkpoint", "")).resolve()
        if (
            report.get("status") != "complete"
            or report.get("search_version") != SEARCH_VERSION
            or report.get("search_contract_sha256") != contract_sha
            or int(report.get("stage", -1)) != stage
            or report.get("candidate_id") != candidate_id
            or report.get("read_contract", {}).get("inference_read") is not False
            or report.get("save_reload_three_step_bitwise_exact") is not True
            or report.get("architecture") != architecture_from_candidate(candidate).to_dict()
            or not checkpoint_path.is_file()
            or _file_sha256(checkpoint_path) != report.get("checkpoint_sha256")
        ):
            raise ModelCValidationError(
                f"candidate report {report_path} violates the validation contract"
            )
        expected_key = selection_key(
            report["selected_checkpoint"]["validation_ten_day"],
            report["selected_checkpoint"]["validation_physics"],
            int(report["parameter_count"]),
            candidate_id,
        )
        if expected_key != list(_report_selection_key(report)):
            raise ModelCValidationError(
                f"candidate report {report_path} has a stale selection key"
            )
        reports.append(report)
        report_entries.append(
            {
                "candidate_id": candidate_id,
                "report": str(report_path),
                "report_sha256": _file_sha256(report_path),
                "checkpoint": report["checkpoint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "selection_key": list(_report_selection_key(report)),
                "ten_day_validation_eligible": report[
                    "ten_day_validation_eligible"
                ],
            }
        )
    reports.sort(key=_report_selection_key)
    entries_by_id = {entry["candidate_id"]: entry for entry in report_entries}
    ranking = [entries_by_id[report["candidate_id"]] for report in reports]
    survivor_count = int(round_spec["survivor_count"])
    survivors = [report["candidate_id"] for report in reports[:survivor_count]]
    diagnostics = _conditional_diagnostics(reports) if stage == 4 else None
    conditional_required = bool(
        diagnostics is not None
        and (
            diagnostics["wall_leakage"]["triggered"]
            or diagnostics["insufficient_capacity"]["triggered"]
        )
    )
    status = (
        "conditional_architecture_tests_required"
        if conditional_required
        else "architecture_selected_pending_three_seed_gate"
        if stage == 4
        else "complete"
    )
    result = {
        "status": status,
        "purpose": "immutable_model_c_successive_halving_selection",
        "version": SEARCH_VERSION,
        "contract": str(resolved_contract_path),
        "contract_sha256": contract_sha,
        "stage": stage,
        "round_contract": round_spec,
        "eligible_candidates": [candidate["candidate_id"] for candidate in eligible],
        "ranking": ranking,
        "survivors": survivors,
        "eliminated": [
            report["candidate_id"] for report in reports[survivor_count:]
        ],
        "all_candidate_reports_present": len(reports) == len(eligible),
        "conditional_diagnostics": diagnostics,
        "inference_opened": False,
        "response_or_adjoint_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def final_seed_gate_acceptance(
    reports: Sequence[dict[str, Any]],
    declared_seeds: Sequence[int],
) -> dict[str, Any]:
    """Evaluate the strict, predeclared independent-seed validation gate."""

    expected = tuple(int(seed) for seed in declared_seeds)
    observed: dict[int, dict[str, Any]] = {}
    duplicate_seeds: list[int] = []
    for report in reports:
        seed = int(report.get("training_seed", -1))
        if seed in observed:
            duplicate_seeds.append(seed)
        else:
            observed[seed] = report
    missing = [seed for seed in expected if seed not in observed]
    unexpected = sorted(seed for seed in observed if seed not in set(expected))
    per_seed = []
    for seed in expected:
        report = observed.get(seed)
        if report is None:
            continue
        selected = report.get("selected_checkpoint", {})
        ten_day = selected.get("validation_ten_day", {})
        ratios = ten_day.get("ratio_to_persistence", {})
        group_pass = {
            name: bool(
                name in ratios
                and np.isfinite(float(ratios[name]))
                and float(ratios[name]) < 1.0
            )
            for name in GROUP_SLICES
        }
        reload_pass = (
            report.get("save_reload_three_step_bitwise_exact") is True
        )
        per_seed.append(
            {
                "training_seed": seed,
                "ratio_to_persistence": {
                    name: float(ratios[name])
                    for name in GROUP_SLICES
                    if name in ratios
                },
                "group_pass": group_pass,
                "every_group_below_persistence": all(group_pass.values()),
                "three_step_reload_bitwise_exact": reload_pass,
                "accepted": all(group_pass.values()) and reload_pass,
            }
        )
    all_groups = (
        not missing
        and not unexpected
        and not duplicate_seeds
        and len(per_seed) == len(expected)
        and all(row["every_group_below_persistence"] for row in per_seed)
    )
    all_reloads = (
        not missing
        and not unexpected
        and not duplicate_seeds
        and len(per_seed) == len(expected)
        and all(row["three_step_reload_bitwise_exact"] for row in per_seed)
    )
    return {
        "declared_seeds": list(expected),
        "observed_seeds": sorted(observed),
        "missing_seeds": missing,
        "unexpected_seeds": unexpected,
        "duplicate_seeds": sorted(set(duplicate_seeds)),
        "per_seed": per_seed,
        "every_seed_every_group_below_persistence": all_groups,
        "every_seed_three_step_reload_bitwise_exact": all_reloads,
        "accepted": all_groups and all_reloads,
    }


def freeze_final_seed_gate(
    contract_path: str | Path,
    selection_manifest_path: str | Path,
    seed_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Verify all final-seed artifacts and write one immutable freeze decision."""

    contract, resolved_contract_path, contract_sha = load_search_contract(contract_path)
    candidate, resolved_selection_path, selection_sha = _selected_final_candidate(
        contract,
        selection_manifest_path,
        contract_sha,
    )
    architecture = architecture_from_candidate(candidate)
    round_spec = round_contract(contract, 4)
    expected_steps = list(
        checkpoint_steps(contract, int(round_spec["maximum_steps"]))
    )
    expected_sources = _source_hashes()
    expected_read_contract = {
        "pair_split_codes_read": [1, 2],
        "snapshot_split_codes_read": [1, 2],
        "validation_read": True,
        "inference_read": False,
        "intermediate_wind_read": False,
        "response_read": False,
        "adjoint_read": False,
    }
    root = Path(seed_root).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite Model C final-seed decision: {output}"
        )

    reports: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    record_fingerprints: set[tuple[Any, ...]] = set()
    normalization_contracts: set[str] = set()
    parameter_counts: set[int] = set()
    declared_seeds = tuple(
        int(seed) for seed in contract["final_seed_gate"]["seeds"]
    )
    for seed in declared_seeds:
        report_path = root / f"seed_{seed}" / FINAL_SEED_REPORT_NAME
        report = json.loads(report_path.read_text())
        checkpoint_path = Path(report.get("checkpoint", "")).resolve()
        selected = report.get("selected_checkpoint", {})
        expected_key = selection_key(
            selected.get("validation_ten_day", {}),
            selected.get("validation_physics", {}),
            int(report.get("parameter_count", -1)),
            str(candidate["candidate_id"]),
        )
        if (
            report.get("status") != "complete"
            or report.get("purpose") != "validation_only_model_c_three_seed_gate"
            or report.get("run_phase") != "final_seed_gate"
            or report.get("search_version") != SEARCH_VERSION
            or report.get("search_contract_sha256") != contract_sha
            or int(report.get("stage", -1)) != 4
            or report.get("candidate_id") != candidate["candidate_id"]
            or report.get("architecture") != architecture.to_dict()
            or int(report.get("search_seed", -1)) != int(contract["search_seed"])
            or int(report.get("training_seed", -1)) != seed
            or report.get("round_contract") != round_spec
            or report.get("optimizer_contract") != contract["optimizer"]
            or report.get("evaluation_steps") != expected_steps
            or report.get("loss_contract_sha256")
            != contract["loss"]["contract_sha256"]
            or report.get("dataset_metadata_sha256")
            != contract["source_contracts"]["dataset_metadata_sha256"]
            or report.get("selection_manifest") != str(resolved_selection_path)
            or report.get("selection_manifest_sha256") != selection_sha
            or report.get("read_contract") != expected_read_contract
            or report.get("source_hashes") != expected_sources
            or list(selected.get("selection_key", ())) != expected_key
            or not checkpoint_path.is_file()
            or _file_sha256(checkpoint_path) != report.get("checkpoint_sha256")
        ):
            raise ModelCValidationError(
                f"final-seed report {report_path} violates the frozen contract"
            )
        normalization = report.get("normalization", {})
        normalizer_hash = normalization.get("contract_sha256")
        if (
            not isinstance(normalizer_hash, str)
            or len(normalizer_hash) != 64
            or _json_sha256(
                {
                    key: normalization[key]
                    for key in (
                        "state_mean_sha256",
                        "state_scale_sha256",
                        "wind_mean",
                        "wind_scale",
                    )
                }
            )
            != normalizer_hash
        ):
            raise ModelCValidationError(
                f"final-seed report {report_path} has an invalid normalizer hash"
            )
        normalization_contracts.add(normalizer_hash)
        parameter_counts.add(int(report["parameter_count"]))
        record_fingerprints.add(
            (
                report.get("training_records_sha256"),
                report.get("validation_records_sha256"),
                report.get("long_validation_records_sha256"),
                tuple(sorted(report.get("record_counts", {}).items())),
            )
        )
        reports.append(report)
        entries.append(
            {
                "training_seed": seed,
                "report": str(report_path),
                "report_sha256": _file_sha256(report_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": report["checkpoint_sha256"],
                "selected_optimizer_step": int(selected["optimizer_step"]),
                "validation_ten_day_ratio_to_persistence": selected[
                    "validation_ten_day"
                ]["ratio_to_persistence"],
                "validation_physics_score": float(
                    selected["validation_physics"]["physics_score"]
                ),
                "ten_day_validation_eligible": bool(
                    report["ten_day_validation_eligible"]
                ),
                "save_reload_three_step_bitwise_exact": bool(
                    report["save_reload_three_step_bitwise_exact"]
                ),
            }
        )
    if (
        len(record_fingerprints) != 1
        or len(normalization_contracts) != 1
        or len(parameter_counts) != 1
    ):
        raise ModelCValidationError(
            "Model C final seeds do not share one data and model contract"
        )

    gate = final_seed_gate_acceptance(reports, declared_seeds)
    accepted = bool(gate["accepted"])
    record_fingerprint = next(iter(record_fingerprints))
    freeze_contract = {
        "version": SEARCH_VERSION,
        "search_contract_sha256": contract_sha,
        "architecture_selection_sha256": selection_sha,
        "candidate_id": candidate["candidate_id"],
        "architecture": architecture.to_dict(),
        "parameter_count": next(iter(parameter_counts)),
        "optimizer_contract": contract["optimizer"],
        "loss_contract_sha256": contract["loss"]["contract_sha256"],
        "dataset_metadata_sha256": contract["source_contracts"][
            "dataset_metadata_sha256"
        ],
        "normalization_contract_sha256": next(iter(normalization_contracts)),
        "source_hashes": expected_sources,
        "training_records_sha256": record_fingerprint[0],
        "validation_records_sha256": record_fingerprint[1],
        "long_validation_records_sha256": record_fingerprint[2],
        "declared_seeds": list(declared_seeds),
        "seed_artifacts": [
            {
                "training_seed": entry["training_seed"],
                "report_sha256": entry["report_sha256"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
            }
            for entry in entries
        ],
    }
    result = {
        "status": (
            "frozen_for_inference"
            if accepted
            else "scientifically_rejected_three_seed_gate"
        ),
        "purpose": "immutable_model_c_three_seed_freeze_decision",
        "version": SEARCH_VERSION,
        "search_contract": str(resolved_contract_path),
        "search_contract_sha256": contract_sha,
        "architecture_selection_manifest": str(resolved_selection_path),
        "architecture_selection_manifest_sha256": selection_sha,
        "candidate_id": candidate["candidate_id"],
        "architecture": architecture.to_dict(),
        "parameter_count": next(iter(parameter_counts)),
        "normalization_contract_sha256": next(iter(normalization_contracts)),
        "source_hashes": expected_sources,
        "seed_gate": gate,
        "seed_artifacts": entries,
        "freeze_contract": freeze_contract,
        "freeze_contract_sha256": _json_sha256(freeze_contract),
        "configuration_frozen": accepted,
        "inference_authorized": accepted,
        "inference_opened": False,
        "intermediate_wind_opened": False,
        "response_or_adjoint_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or select a bounded Model C validation-search stage"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="train and score one stage candidate")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--diagnostics", type=Path, required=True)
    run.add_argument("--calibration", type=Path, required=True)
    run.add_argument("--optimization-manifest", type=Path, required=True)
    run.add_argument("--loss-manifest", type=Path, required=True)
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--stage", type=int, choices=(1, 2, 3, 4), required=True)
    run.add_argument("--candidate-id", required=True)
    run.add_argument("--selection-manifest", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    select = commands.add_parser("select", help="rank one complete stage")
    select.add_argument("--contract", type=Path, required=True)
    select.add_argument("--stage", type=int, choices=(1, 2, 3, 4), required=True)
    select.add_argument("--stage-root", type=Path, required=True)
    select.add_argument("--eligible-manifest", type=Path)
    select.add_argument("--output", type=Path, required=True)

    resolve = commands.add_parser(
        "resolve", help="map one scheduler-array index to a candidate identifier"
    )
    resolve.add_argument("--contract", type=Path, required=True)
    resolve.add_argument("--stage", type=int, choices=(1, 2, 3, 4), required=True)
    resolve.add_argument("--array-index", type=int, required=True)
    resolve.add_argument("--selection-manifest", type=Path)

    resolve_final = commands.add_parser(
        "resolve-final-seed",
        help="map one scheduler-array index to a declared final seed",
    )
    resolve_final.add_argument("--contract", type=Path, required=True)
    resolve_final.add_argument("--array-index", type=int, required=True)

    run_final = commands.add_parser(
        "run-final",
        help="retrain the selected architecture for one declared final seed",
    )
    run_final.add_argument("--dataset", type=Path, required=True)
    run_final.add_argument("--diagnostics", type=Path, required=True)
    run_final.add_argument("--calibration", type=Path, required=True)
    run_final.add_argument("--optimization-manifest", type=Path, required=True)
    run_final.add_argument("--loss-manifest", type=Path, required=True)
    run_final.add_argument("--contract", type=Path, required=True)
    run_final.add_argument("--selection-manifest", type=Path, required=True)
    run_final.add_argument("--seed", type=int, required=True)
    run_final.add_argument("--output-dir", type=Path, required=True)
    run_final.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )

    freeze_final = commands.add_parser(
        "freeze-final",
        help="verify three final seeds and write the immutable freeze decision",
    )
    freeze_final.add_argument("--contract", type=Path, required=True)
    freeze_final.add_argument("--selection-manifest", type=Path, required=True)
    freeze_final.add_argument("--seed-root", type=Path, required=True)
    freeze_final.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run_validation_candidate(
            args.dataset,
            args.diagnostics,
            args.calibration,
            args.optimization_manifest,
            args.loss_manifest,
            args.contract,
            args.output_dir,
            stage=args.stage,
            candidate_id=args.candidate_id,
            selection_manifest_path=args.selection_manifest,
            device_name=args.device,
        )
    elif args.command == "select":
        result = select_validation_stage(
            args.contract,
            args.stage_root,
            args.output,
            stage=args.stage,
            eligible_manifest_path=args.eligible_manifest,
        )
    elif args.command == "resolve":
        print(
            resolve_candidate_id(
                args.contract,
                stage=args.stage,
                array_index=args.array_index,
                selection_manifest_path=args.selection_manifest,
            )
        )
        return 0
    elif args.command == "resolve-final-seed":
        print(
            resolve_final_seed(
                args.contract,
                array_index=args.array_index,
            )
        )
        return 0
    elif args.command == "run-final":
        result = run_final_seed_candidate(
            args.dataset,
            args.diagnostics,
            args.calibration,
            args.optimization_manifest,
            args.loss_manifest,
            args.contract,
            args.selection_manifest,
            args.output_dir,
            seed=args.seed,
            device_name=args.device,
        )
    else:
        result = freeze_final_seed_gate(
            args.contract,
            args.selection_manifest,
            args.seed_root,
            args.output,
        )
    summary = {
        key: result[key]
        for key in ("status", "stage")
        if key in result
    }
    if "candidate_id" in result:
        summary["candidate_id"] = result["candidate_id"]
    if "survivors" in result:
        summary["survivors"] = result["survivors"]
    if "training_seed" in result:
        summary["training_seed"] = result["training_seed"]
    if "inference_authorized" in result:
        summary["inference_authorized"] = result["inference_authorized"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
