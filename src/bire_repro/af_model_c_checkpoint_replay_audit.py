"""Exact-replay late-checkpoint audit for the trajectory-v2 Model C successor.

The selected width-128 successor beats persistence at ten days but accumulates
large SST and pressure errors through day 90.  This module replays the original
reference-seed optimization exactly, saves every predeclared late checkpoint,
and evaluates each checkpoint on a fixed, chronology-balanced subset of
split-1 rollouts.  It never reads validation or inference states and it does
not alter the architecture, loss, optimizer, batch order, or training length.
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
    seed_everything,
)
from .af_model_b import (
    ModelBRolloutDataset,
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
from .af_model_c_overfit import AUDIT_TERMS, _device, _file_sha256
from .af_model_c_rollout_diagnosis import (
    complete_rollout_starts,
    lead_curve_summary,
    load_rollout_diagnosis_contract,
    select_balanced_training_times,
)
from .af_model_c_successor import (
    REFERENCE_DIAGNOSTIC_SEED,
    STATE_CHANNEL_COUNT,
    _one_step_diagnostics,
    _sample_records_by_regime,
    architecture_from_candidate,
    build_successor,
    load_successor_contract,
    training_increment_scale,
)
from .af_model_c_successor_validation import (
    LEAD_DAYS,
    PRIMARY_FIELDS,
    ValidationStepper,
    _evaluate_baseline_metrics,
    _evaluate_stepper,
    _method_auc_summary,
)

try:
    import torch
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


AUDIT_VERSION = "model_c_checkpoint_replay_audit_v1"
REPORT_NAME = "model_c_checkpoint_replay_audit_report.json"
ARRAYS_NAME = "model_c_checkpoint_replay_audit_arrays.npz"
CHECKPOINT_DIRECTORY = "late_checkpoints"
REFERENCE_CANDIDATE = "v2_bireprop_w128_mlp4"
SLOW_PRIMARY_FIELDS = ("sst", "phihyd_surface")


class ModelCCheckpointReplayAuditError(RuntimeError):
    """Raised when exact replay or the frozen audit contract is violated."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def load_checkpoint_replay_contract(
    path: str | Path,
    *,
    verify_source_files: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the audit contract frozen before any checkpoint replay metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != AUDIT_VERSION:
        raise ValueError(f"expected checkpoint replay contract {AUDIT_VERSION}")
    if (
        contract.get("contract_status")
        != "frozen_after_training_rollout_drift_before_late_checkpoint_replay"
    ):
        raise ValueError("Model C checkpoint replay contract was not frozen")
    replay = contract.get("replay", {})
    if (
        replay.get("candidate_id") != REFERENCE_CANDIDATE
        or int(replay.get("seed", -1)) != REFERENCE_DIAGNOSTIC_SEED
        or tuple(replay.get("checkpoint_steps", ()))
        != (11520, 13440, 14400, 14880, 15120, 15360)
        or int(replay.get("reference_selected_step", -1)) != 14880
        or replay.get("architecture_change") is not False
        or replay.get("loss_change") is not False
        or replay.get("optimizer_change") is not False
    ):
        raise ValueError("exact-replay schedule changed")
    records = contract.get("records", {})
    if (
        tuple(records.get("lead_days", ())) != LEAD_DAYS
        or int(records.get("starts_per_training_block", -1)) != 90
        or int(records.get("expected_training_blocks", -1)) != 2
        or int(records.get("records_total", -1)) != 540
        or records.get("selection")
        != "evenly_spaced_complete_90_day_starts_per_training_block"
    ):
        raise ValueError("checkpoint audit record contract changed")
    gate = contract.get("checkpoint_gate", {})
    if (
        tuple(gate.get("primary_fields", ())) != PRIMARY_FIELDS
        or tuple(gate.get("slow_primary_fields", ()))
        != SLOW_PRIMARY_FIELDS
        or tuple(gate.get("baselines", ()))
        != ("persistence", "climatology")
        or float(gate.get("strict_rmse_ratio_threshold", -1.0)) != 1.0
        or gate.get("require_primary_rmse_auc_below_both_baselines")
        is not True
        or gate.get("require_slow_field_rmse_below_both_at_every_lead")
        is not True
        or gate.get("require_old_ten_day_regime_group_gate") is not True
        or gate.get("require_finite_rollout") is not True
        or gate.get("require_zero_land_leakage") is not True
    ):
        raise ValueError("checkpoint audit scientific gate changed")
    read = contract.get("read_contract", {})
    if (
        read.get("training_pair_code") != 1
        or read.get("training_state_read") is not True
        or read.get("prior_report_metadata_read") is not True
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
        raise ValueError("checkpoint audit read contract changed")
    if verify_source_files:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(f"checkpoint audit source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def _verify_source_artifacts(
    contract: Mapping[str, Any],
    dataset: Path,
) -> tuple[
    dict[str, Any],
    Path,
    str,
    dict[str, Any],
    Path,
    dict[str, Any],
]:
    sources = contract["source_artifacts"]
    if _file_sha256(dataset / ".zmetadata") != sources[
        "dataset_metadata_sha256"
    ]:
        raise ModelCCheckpointReplayAuditError(
            "trajectory-v2 dataset metadata changed"
        )
    successor_contract_path = Path(
        sources["successor_training_contract"]
    ).resolve()
    successor_contract, _, successor_contract_sha = load_successor_contract(
        successor_contract_path
    )
    if successor_contract_sha != sources[
        "successor_training_contract_sha256"
    ]:
        raise ModelCCheckpointReplayAuditError(
            "successor training contract changed"
        )
    reference_report_path = Path(
        sources["reference_training_report"]
    ).resolve()
    reference_checkpoint_path = Path(
        sources["reference_selected_checkpoint"]
    ).resolve()
    rollout_report_path = Path(sources["training_rollout_report"]).resolve()
    rollout_contract_path = Path(
        sources["training_rollout_contract"]
    ).resolve()
    _, _, rollout_contract_sha = load_rollout_diagnosis_contract(
        rollout_contract_path
    )
    if rollout_contract_sha != sources["training_rollout_contract_sha256"]:
        raise ModelCCheckpointReplayAuditError(
            "training-rollout diagnosis contract changed"
        )
    expected_hashes = {
        reference_report_path: sources["reference_training_report_sha256"],
        reference_checkpoint_path: sources[
            "reference_selected_checkpoint_sha256"
        ],
        rollout_report_path: sources["training_rollout_report_sha256"],
    }
    for artifact, expected in expected_hashes.items():
        if not artifact.is_file() or _file_sha256(artifact) != expected:
            raise ModelCCheckpointReplayAuditError(
                f"checkpoint replay source changed: {artifact}"
            )
    reference_report = json.loads(reference_report_path.read_text())
    rollout_report = json.loads(rollout_report_path.read_text())
    if (
        reference_report.get("status") != "complete"
        or reference_report.get("candidate_id") != REFERENCE_CANDIDATE
        or int(reference_report.get("seed", -1))
        != REFERENCE_DIAGNOSTIC_SEED
        or int(
            reference_report.get("selected_checkpoint", {}).get(
                "optimizer_step", -1
            )
        )
        != 14880
        or reference_report.get("checkpoint_sha256")
        != sources["reference_selected_checkpoint_sha256"]
        or reference_report.get("contract_sha256")
        != successor_contract_sha
    ):
        raise ModelCCheckpointReplayAuditError(
            "reference training report provenance changed"
        )
    if (
        rollout_report.get("status") != "complete"
        or rollout_report.get("diagnostic_interpretation", {}).get(
            "classification"
        )
        != "training_objective_or_checkpoint_gate_mismatch"
        or rollout_report.get("inference_opened") is not False
        or rollout_report.get("read_contract", {}).get(
            "validation_state_read"
        )
        is not False
        or rollout_report.get("report_content_sha256")
        != sources["training_rollout_report_content_sha256"]
    ):
        raise ModelCCheckpointReplayAuditError(
            "training-rollout diagnosis is not the expected sealed result"
        )
    return (
        successor_contract,
        successor_contract_path,
        successor_contract_sha,
        reference_report,
        reference_checkpoint_path,
        rollout_report,
    )


def _portable_state_dict(model: Any) -> dict[str, Any]:
    """Copy a loadable state dictionary to CPU for immutable checkpointing."""

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key != "_metadata"
    }


def compare_state_dicts(
    replayed: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two model state dictionaries tensor by tensor."""

    replayed_keys = set(replayed)
    reference_keys = set(reference)
    missing = sorted(reference_keys - replayed_keys)
    unexpected = sorted(replayed_keys - reference_keys)
    mismatched = []
    max_abs_difference = 0.0
    for key in sorted(replayed_keys & reference_keys):
        left = replayed[key].detach().cpu()
        right = reference[key].detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            mismatched.append(key)
            continue
        if torch.equal(left, right):
            continue
        mismatched.append(key)
        if left.is_complex():
            difference = torch.abs(left - right)
        else:
            difference = torch.abs(
                left.to(torch.float64) - right.to(torch.float64)
            )
        if difference.numel():
            max_abs_difference = max(
                max_abs_difference,
                float(torch.max(difference).item()),
            )
    exact = not missing and not unexpected and not mismatched
    return {
        "bitwise_exact": exact,
        "parameter_keys": len(reference_keys),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "mismatched_key_count": len(mismatched),
        "max_abs_difference": max_abs_difference,
    }


def numeric_tree_max_abs_difference(left: Any, right: Any) -> float:
    """Return the largest numeric difference in two equal JSON-like trees."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return float("inf")
        return max(
            (
                numeric_tree_max_abs_difference(left[key], right[key])
                for key in left
            ),
            default=0.0,
        )
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes))
    ):
        if len(left) != len(right):
            return float("inf")
        return max(
            (
                numeric_tree_max_abs_difference(a, b)
                for a, b in zip(left, right)
            ),
            default=0.0,
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def checkpoint_gate_summary(
    lead_curves: Mapping[str, Any],
    ten_day_diagnostic: Mapping[str, Any],
    metrics: Mapping[str, np.ndarray],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the predeclared training-only long-horizon checkpoint gate."""

    threshold = float(gate["strict_rmse_ratio_threshold"])
    baselines = tuple(gate["baselines"])
    primary_fields = tuple(gate["primary_fields"])
    slow_fields = tuple(gate["slow_primary_fields"])
    auc_checks: dict[str, dict[str, bool]] = {}
    all_lead_checks: dict[str, dict[str, bool]] = {}
    auc_ratios = []
    slow_lead_ratios = []
    for field in primary_fields:
        auc_checks[field] = {}
        for baseline in baselines:
            raw_ratio = float(
                lead_curves[field][baseline]["rmse_auc_ratio"]
            )
            ratio = raw_ratio if np.isfinite(raw_ratio) else float("inf")
            auc_ratios.append(ratio)
            auc_checks[field][baseline] = ratio < threshold
    for field in slow_fields:
        all_lead_checks[field] = {}
        for baseline in baselines:
            ratios = np.asarray(
                lead_curves[field][baseline]["rmse_ratio_by_lead"],
                dtype=np.float64,
            )
            scoring_ratios = np.where(
                np.isfinite(ratios),
                ratios,
                np.inf,
            )
            slow_lead_ratios.extend(scoring_ratios.tolist())
            all_lead_checks[field][baseline] = bool(
                np.all(ratios < threshold)
            )
    finite_pass = bool(np.all(np.asarray(metrics["finite"]) == 1))
    land_max = float(
        np.max(np.abs(np.asarray(metrics["normalized_land_max_abs"])))
    )
    land_pass = land_max == 0.0
    ten_day_pass = bool(
        ten_day_diagnostic[
            "every_regime_and_group_beats_persistence"
        ]
    )
    auc_pass = all(
        value for checks in auc_checks.values() for value in checks.values()
    )
    slow_pass = all(
        value
        for checks in all_lead_checks.values()
        for value in checks.values()
    )
    passed = ten_day_pass and finite_pass and land_pass and auc_pass and slow_pass
    return {
        "passed": passed,
        "old_ten_day_regime_group_gate_passed": ten_day_pass,
        "rollout_finite": finite_pass,
        "normalized_land_max_abs": land_max,
        "zero_land_leakage": land_pass,
        "primary_rmse_auc_checks": auc_checks,
        "primary_rmse_auc_passed": auc_pass,
        "slow_field_all_lead_checks": all_lead_checks,
        "slow_field_all_leads_passed": slow_pass,
        "worst_primary_rmse_auc_ratio": float(max(auc_ratios)),
        "worst_slow_field_lead_ratio": float(max(slow_lead_ratios)),
        "day90_ratios": {
            field: {
                baseline: float(
                    lead_curves[field][baseline]["rmse_ratio_by_lead"][-1]
                )
                for baseline in baselines
            }
            for field in slow_fields
        },
    }


def checkpoint_audit_decision(
    checkpoint_summaries: Sequence[Mapping[str, Any]],
    *,
    exact_replay_passed: bool,
) -> dict[str, Any]:
    """Classify checkpoint selection versus objective correction."""

    ranking = []
    for summary in checkpoint_summaries:
        gate = summary["checkpoint_gate"]
        ranking.append(
            {
                "optimizer_step": int(summary["optimizer_step"]),
                "passed": bool(gate["passed"]),
                "selection_key": [
                    float(gate["worst_slow_field_lead_ratio"]),
                    float(gate["worst_primary_rmse_auc_ratio"]),
                    float(
                        summary["ten_day_diagnostic"][
                            "worst_per_regime_group_ratio"
                        ]
                    ),
                    int(summary["optimizer_step"]),
                ],
            }
        )
    ranking.sort(key=lambda row: tuple(row["selection_key"]))
    eligible = [row for row in ranking if row["passed"]]
    if not exact_replay_passed:
        return {
            "status": "blocked_untrusted_exact_replay",
            "classification": "replay_provenance_failed",
            "selected_optimizer_step": None,
            "checkpoint_only_correction_supported": False,
            "objective_correction_required": False,
            "ranking": ranking,
            "next_action": (
                "resolve exact-replay provenance before interpreting "
                "checkpoint skill"
            ),
        }
    if eligible:
        selected = eligible[0]
        return {
            "status": "complete",
            "classification": (
                "checkpoint_selection_only_correction_supported"
            ),
            "selected_optimizer_step": selected["optimizer_step"],
            "checkpoint_only_correction_supported": True,
            "objective_correction_required": False,
            "ranking": ranking,
            "next_action": (
                "replicate the frozen checkpoint-selection rule across seeds "
                "before independent validation"
            ),
        }
    return {
        "status": "complete",
        "classification": "objective_correction_required",
        "selected_optimizer_step": None,
        "diagnostic_best_optimizer_step": (
            ranking[0]["optimizer_step"] if ranking else None
        ),
        "checkpoint_only_correction_supported": False,
        "objective_correction_required": True,
        "ranking": ranking,
        "next_action": (
            "freeze a bounded pushforward or unrolled slow-field objective "
            "test; keep architecture and sealed data unchanged"
        ),
    }


def _audit_records(
    pair_codes: np.ndarray,
    snapshot_codes: np.ndarray,
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    complete = complete_rollout_starts(pair_codes, snapshot_codes)
    records_contract = contract["records"]
    times, bounds = select_balanced_training_times(
        complete,
        starts_per_block=int(records_contract["starts_per_training_block"]),
        expected_blocks=int(records_contract["expected_training_blocks"]),
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
        records.shape != (int(records_contract["records_total"]), 2)
        or _array_sha256(times)
        != records_contract["training_times_sha256"]
        or _array_sha256(records) != records_contract["records_sha256"]
    ):
        raise ModelCCheckpointReplayAuditError(
            "fixed training-rollout records changed"
        )
    return records, times, bounds


def _checkpoint_path(directory: Path, step: int) -> Path:
    return directory / f"model_c_checkpoint_step_{step:05d}.pt"


def run_checkpoint_replay_audit(
    dataset_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Replay, checkpoint, and score the reference Model C training run."""

    if torch is None or DataLoader is None:
        raise RuntimeError("checkpoint replay audit requires PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = (
        load_checkpoint_replay_contract(contract_path)
    )
    dataset = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite checkpoint replay output: {output}"
        )
    (
        successor_contract,
        successor_contract_path,
        successor_contract_sha,
        reference_report,
        reference_checkpoint_path,
        rollout_report,
    ) = _verify_source_artifacts(contract, dataset)
    replay = contract["replay"]
    optimizer_contract = successor_contract["optimizer"]
    architecture = architecture_from_candidate(
        successor_contract,
        str(replay["candidate_id"]),
    )
    loss_config = model_c_loss_config("v1")
    if (
        loss_contract_sha256(loss_config)
        != MODEL_C_LOSS_V1_CONTRACT_SHA256
        or reference_report.get("loss_contract_sha256")
        != MODEL_C_LOSS_V1_CONTRACT_SHA256
    ):
        raise ModelCCheckpointReplayAuditError(
            "loss-v1 contract changed before exact replay"
        )

    seed = int(replay["seed"])
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
        contract,
    )
    training_records = records_for_rollout_split(pair_codes, 1)
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
    increment_values = training_increment_scale(group, pair_codes)
    if (
        _array_sha256(increment_values)
        != reference_report["increment_scale_sha256"]
    ):
        raise ModelCCheckpointReplayAuditError(
            "training increment scale changed before exact replay"
        )

    mean, scale, wet_array, _, wind_mean, wind_scale = _normalizers(group)
    climatology_state, climatology_derived, training_days = (
        _training_climatology(state, snapshot_codes, wet_array)
    )
    evaluation_batch_size = int(contract["evaluation"]["batch_size"])
    baseline_raw = _evaluate_baseline_metrics(
        state,
        audit_records,
        climatology_state,
        climatology_derived,
        wet_array,
        mean,
        scale,
        batch_size=evaluation_batch_size,
    )
    baselines: dict[str, Mapping[str, np.ndarray]] = {
        "persistence": baseline_raw["persistence"],
        "climatology": baseline_raw["climatology"],
    }

    training_dataset = ModelBRolloutDataset(dataset, training_records)
    batch_size = int(optimizer_contract["batch_size"])
    train_loader = DataLoader(
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
    increment_scale = torch.from_numpy(increment_values).to(device)
    model = build_successor(architecture).to(device)
    parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    if parameter_count != int(reference_report["parameter_count"]):
        raise ModelCCheckpointReplayAuditError(
            "reference architecture parameter count changed"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_contract["initial_learning_rate"]),
        betas=tuple(
            float(value) for value in optimizer_contract["adam_betas"]
        ),
        weight_decay=float(optimizer_contract["weight_decay"]),
    )
    maximum_steps = int(optimizer_contract["maximum_steps"])
    decay_step = int(
        round(maximum_steps * float(optimizer_contract["decay_fraction"]))
    )
    checkpoint_steps = tuple(int(value) for value in replay["checkpoint_steps"])
    if checkpoint_steps[-1] != maximum_steps:
        raise ModelCCheckpointReplayAuditError(
            "audit checkpoints do not end at the original maximum step"
        )

    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    checkpoint_directory = temporary / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir()
    history: list[dict[str, Any]] = []
    saved_checkpoints: list[dict[str, Any]] = []
    best_key: tuple[float, float, int] | None = None
    best_state: dict[str, Any] | None = None
    window_totals = {name: 0.0 for name in AUDIT_TERMS}
    window_samples = 0
    iterator = iter(train_loader)

    for step in range(1, maximum_steps + 1):
        if step == decay_step + 1:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= float(
                    optimizer_contract["decay_factor"]
                )
        try:
            features, futures = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
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
        predictions = _unroll(
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
            raise ModelCCheckpointReplayAuditError(
                "exact-replay training loss became non-finite"
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
            batch_size=16,
            device=device,
        )
        key = (
            float(diagnostic["worst_per_regime_group_ratio"]),
            float(training_window["total"]),
            step,
        )
        history_record = {
            "optimizer_step": step,
            "optimizer_learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "training_window": training_window,
            "training_only_ten_day_diagnostic": diagnostic,
            "selection_key": list(key),
        }
        history.append(history_record)
        if best_key is None or key < best_key:
            best_key = key
            best_state = _checkpoint_state_dict(model)
        checkpoint_path = _checkpoint_path(checkpoint_directory, step)
        payload = {
            "version": AUDIT_VERSION,
            "purpose": "exact_replay_late_checkpoint_snapshot",
            "optimizer_step": step,
            "seed": seed,
            "candidate_id": REFERENCE_CANDIDATE,
            "architecture": architecture.to_dict(),
            "parameter_count": parameter_count,
            "audit_contract": str(resolved_contract),
            "audit_contract_sha256": contract_sha,
            "successor_training_contract": str(
                successor_contract_path
            ),
            "successor_training_contract_sha256": successor_contract_sha,
            "loss_contract": loss_contract(loss_config),
            "loss_contract_sha256": loss_contract_sha256(loss_config),
            "optimizer_contract": optimizer_contract,
            "training_history_record": history_record,
            "model_state_dict": _portable_state_dict(model),
        }
        torch.save(payload, checkpoint_path)
        saved_checkpoints.append(
            {
                "optimizer_step": step,
                "checkpoint": str(
                    output
                    / CHECKPOINT_DIRECTORY
                    / checkpoint_path.name
                ),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
            }
        )
        window_totals = {name: 0.0 for name in AUDIT_TERMS}
        window_samples = 0

    if len(history) != len(checkpoint_steps):
        raise ModelCCheckpointReplayAuditError(
            "exact replay did not save every predeclared checkpoint"
        )
    if best_state is None or best_key is None:
        raise ModelCCheckpointReplayAuditError(
            "exact replay selected no reference checkpoint"
        )
    history_difference = numeric_tree_max_abs_difference(
        history,
        reference_report["history"],
    )
    reference_payload = torch.load(
        reference_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    replayed_reference = torch.load(
        _checkpoint_path(checkpoint_directory, 14880),
        map_location="cpu",
        weights_only=False,
    )
    state_comparison = compare_state_dicts(
        replayed_reference["model_state_dict"],
        reference_payload["model_state_dict"],
    )
    exact_replay = bool(
        state_comparison["bitwise_exact"] and history_difference == 0.0
    )
    replay_verification = {
        "passed": exact_replay,
        "reference_optimizer_step": 14880,
        "reference_checkpoint": str(reference_checkpoint_path),
        "reference_checkpoint_sha256": _file_sha256(
            reference_checkpoint_path
        ),
        "selected_state_dict_comparison": state_comparison,
        "history_bitwise_numeric_exact": history_difference == 0.0,
        "history_max_abs_difference": history_difference,
    }

    arrays: dict[str, np.ndarray] = {
        "records": audit_records.astype(np.int32),
        "training_times": audit_times.astype(np.int32),
        "lead_days": np.asarray(LEAD_DAYS, dtype=np.int16),
    }
    for baseline, metrics in baselines.items():
        for name, value in metrics.items():
            arrays[f"{baseline}__{name}"] = np.asarray(value)

    checkpoint_summaries = []
    for saved, history_record in zip(saved_checkpoints, history):
        step = int(saved["optimizer_step"])
        checkpoint_payload = torch.load(
            _checkpoint_path(checkpoint_directory, step),
            map_location=device,
            weights_only=False,
        )
        evaluation_model = build_successor(architecture).to(device)
        evaluation_model.load_state_dict(
            checkpoint_payload["model_state_dict"]
        )
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
            batch_size=evaluation_batch_size,
        )
        for name, value in metrics.items():
            arrays[f"step_{step}__{name}"] = np.asarray(value)
        curves = lead_curve_summary(metrics, baselines, audit_records)
        checkpoint_gate = checkpoint_gate_summary(
            curves,
            history_record["training_only_ten_day_diagnostic"],
            metrics,
            contract["checkpoint_gate"],
        )
        checkpoint_summaries.append(
            {
                **saved,
                "ten_day_diagnostic": history_record[
                    "training_only_ten_day_diagnostic"
                ],
                "training_window": history_record["training_window"],
                "all_field_auc": _method_auc_summary(metrics),
                "lead_curves": curves,
                "checkpoint_gate": checkpoint_gate,
            }
        )
        del evaluation_model, stepper, checkpoint_payload
        if device.type == "cuda":
            torch.cuda.empty_cache()

    decision = checkpoint_audit_decision(
        checkpoint_summaries,
        exact_replay_passed=exact_replay,
    )
    arrays_path = temporary / ARRAYS_NAME
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "status": "complete",
        "purpose": (
            "training_only_exact_replay_late_checkpoint_selection_audit"
        ),
        "version": AUDIT_VERSION,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset),
        "dataset_metadata_sha256": _file_sha256(dataset / ".zmetadata"),
        "device": str(device),
        "seed": seed,
        "candidate_id": REFERENCE_CANDIDATE,
        "architecture": architecture.to_dict(),
        "parameter_count": parameter_count,
        "loss_contract": loss_contract(loss_config),
        "loss_contract_sha256": loss_contract_sha256(loss_config),
        "optimizer_contract": optimizer_contract,
        "optimizer_decay_step": decay_step,
        "source_artifacts": {
            "reference_training_report": contract["source_artifacts"][
                "reference_training_report"
            ],
            "reference_training_report_sha256": _file_sha256(
                Path(
                    contract["source_artifacts"][
                        "reference_training_report"
                    ]
                )
            ),
            "reference_selected_checkpoint": str(
                reference_checkpoint_path
            ),
            "reference_selected_checkpoint_sha256": _file_sha256(
                reference_checkpoint_path
            ),
            "training_rollout_report": contract["source_artifacts"][
                "training_rollout_report"
            ],
            "training_rollout_report_sha256": _file_sha256(
                Path(
                    contract["source_artifacts"][
                        "training_rollout_report"
                    ]
                )
            ),
            "training_rollout_classification": rollout_report[
                "diagnostic_interpretation"
            ]["classification"],
        },
        "read_contract": {
            "training_pair_code": 1,
            "training_state_read": True,
            "prior_report_metadata_read": True,
            "validation_state_read": False,
            "inference_read": False,
            "intermediate_wind_read": False,
            "response_read": False,
            "adjoint_read": False,
        },
        "record_contract": {
            "lead_days": list(LEAD_DAYS),
            "complete_training_blocks": [
                list(value) for value in block_bounds
            ],
            "starts_per_training_block": int(
                contract["records"]["starts_per_training_block"]
            ),
            "records_total": int(audit_records.shape[0]),
            "training_times_sha256": _array_sha256(audit_times),
            "records_sha256": _array_sha256(audit_records),
            "training_climatology_snapshots_per_regime": training_days,
        },
        "baseline_auc": {
            method: _method_auc_summary(metrics)
            for method, metrics in baselines.items()
        },
        "training_history": history,
        "saved_checkpoints": saved_checkpoints,
        "exact_replay_verification": replay_verification,
        "checkpoint_summary": checkpoint_summaries,
        "audit_decision": decision,
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


def preflight_checkpoint_replay_audit(
    dataset_path: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Verify the frozen contract and immutable sources without training."""

    contract, resolved, digest = load_checkpoint_replay_contract(
        contract_path
    )
    dataset = Path(dataset_path).resolve()
    _verify_source_artifacts(contract, dataset)
    return {
        "status": "ready",
        "version": AUDIT_VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "checkpoint_steps": contract["replay"]["checkpoint_steps"],
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
        result = preflight_checkpoint_replay_audit(
            args.dataset,
            args.contract,
        )
    else:
        result = run_checkpoint_replay_audit(
            args.dataset,
            args.contract,
            args.output_dir,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
