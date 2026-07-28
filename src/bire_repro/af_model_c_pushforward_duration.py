"""Replay-verified low-rate duration extension of Model C pushforward v1.

The first 1,920 fine-tune steps are replayed from the exact step-14,400
checkpoint and must reproduce the pushforward-v1 selected state bit for bit.
Training then continues with the identical objective and already-decayed
learning rate.  Only split-1 states and the frozen 540-record gate are used.
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
from .af_model_b import _unroll, records_for_rollout_split, western_boundary_mask
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
    compare_state_dicts,
    load_checkpoint_replay_contract,
)
from .af_model_c_overfit import _device, _file_sha256
from .af_model_c_pushforward_objective import (
    REFERENCE_CANDIDATE,
    ROLLOUT_STEPS,
    STATE_CHANNEL_COUNT,
    TRAINING_TERMS,
    ModelCPushforwardDataset,
    detached_pushforward_endpoint,
    load_pushforward_contract,
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


DURATION_VERSION = "model_c_pushforward_duration_v1"
REPORT_NAME = "model_c_pushforward_duration_report.json"
ARRAYS_NAME = "model_c_pushforward_duration_arrays.npz"
CHECKPOINT_DIRECTORY = "duration_checkpoints"
SELECTED_CHECKPOINT_NAME = "model_c_pushforward_duration_best.pt"
HORIZON_DAYS = 10
REPLAY_END_STEP = 1920


class ModelCPushforwardDurationError(RuntimeError):
    """Raised when the duration extension violates its frozen contract."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_duration_contract(
    path: str | Path,
    *,
    verify_source_files: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the duration contract frozen before extension metrics."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != DURATION_VERSION:
        raise ValueError(f"expected duration contract {DURATION_VERSION}")
    if (
        contract.get("contract_status")
        != "frozen_after_pushforward_v1_rejection_before_duration_extension"
    ):
        raise ValueError("pushforward duration contract was not frozen")
    replay = contract.get("replay", {})
    if (
        int(replay.get("source_optimizer_step", -1)) != 14400
        or int(replay.get("required_exact_step", -1)) != REPLAY_END_STEP
        or replay.get("same_batch_order") is not True
        or replay.get("same_optimizer") is not True
        or replay.get("same_objective") is not True
    ):
        raise ValueError("pushforward duration replay contract changed")
    extension = contract.get("extension", {})
    if (
        int(extension.get("maximum_total_steps", -1)) != 5760
        or int(extension.get("absolute_decay_step", -1)) != 1440
        or float(extension.get("initial_learning_rate", -1.0)) != 0.0001
        or float(extension.get("decayed_learning_rate", -1.0)) != 0.00002
        or tuple(extension.get("checkpoint_steps", ()))
        != (2400, 2880, 3360, 3840, 4320, 4800, 5280, 5760)
        or int(extension.get("batch_size", -1)) != 4
        or tuple(extension.get("adam_betas", ())) != (0.9, 0.95)
        or float(extension.get("weight_decay", -1.0)) != 0.00001
    ):
        raise ValueError("pushforward duration schedule changed")
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
        raise ValueError("pushforward duration read contract changed")
    if verify_source_files:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(f"pushforward duration source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def _verify_sources(
    contract: Mapping[str, Any],
    dataset: Path,
) -> tuple[
    dict[str, Any],
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
        raise ModelCPushforwardDurationError("trajectory-v2 metadata changed")
    push_contract, _, push_contract_sha = load_pushforward_contract(
        sources["pushforward_v1_contract"]
    )
    if push_contract_sha != sources["pushforward_v1_contract_sha256"]:
        raise ModelCPushforwardDurationError(
            "pushforward-v1 contract changed"
        )
    audit_contract, _, audit_contract_sha = load_checkpoint_replay_contract(
        sources["checkpoint_replay_contract"]
    )
    if audit_contract_sha != sources["checkpoint_replay_contract_sha256"]:
        raise ModelCPushforwardDurationError(
            "checkpoint-replay contract changed"
        )
    successor_contract, _, successor_contract_sha = load_successor_contract(
        sources["successor_training_contract"]
    )
    if successor_contract_sha != sources[
        "successor_training_contract_sha256"
    ]:
        raise ModelCPushforwardDurationError(
            "successor-training contract changed"
        )
    report_path = Path(sources["pushforward_v1_report"]).resolve()
    arrays_path = Path(sources["pushforward_v1_arrays"]).resolve()
    selected_path = Path(
        sources["pushforward_v1_selected_checkpoint"]
    ).resolve()
    original_path = Path(sources["source_checkpoint_step14400"]).resolve()
    expected = {
        report_path: sources["pushforward_v1_report_sha256"],
        arrays_path: sources["pushforward_v1_arrays_sha256"],
        selected_path: sources[
            "pushforward_v1_selected_checkpoint_sha256"
        ],
        original_path: sources["source_checkpoint_step14400_sha256"],
    }
    for artifact, digest in expected.items():
        if not artifact.is_file() or _file_sha256(artifact) != digest:
            raise ModelCPushforwardDurationError(
                f"duration source artifact changed: {artifact}"
            )
    report = json.loads(report_path.read_text())
    if (
        report.get("status") != "complete"
        or report.get("objective_decision", {}).get("passed") is not False
        or report.get("objective_decision", {}).get(
            "selected_fine_tune_step"
        )
        != REPLAY_END_STEP
        or report.get("save_reload_nine_step_bitwise_exact") is not True
        or report.get("report_content_sha256")
        != sources["pushforward_v1_report_content_sha256"]
    ):
        raise ModelCPushforwardDurationError(
            "pushforward-v1 result does not authorize duration diagnosis"
        )
    return (
        push_contract,
        audit_contract,
        successor_contract,
        arrays_path,
        selected_path,
        original_path,
    )


def _checkpoint_path(directory: Path, step: int) -> Path:
    return directory / f"model_c_pushforward_duration_step_{step:04d}.pt"


def run_duration_extension(
    dataset_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Replay pushforward v1 exactly, extend it, and apply the same gate."""

    if torch is None or DataLoader is None:
        raise RuntimeError("pushforward duration requires PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_duration_contract(
        contract_path
    )
    dataset = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite duration output: {output}"
        )
    (
        push_contract,
        audit_contract,
        successor_contract,
        source_arrays_path,
        push_selected_path,
        source_checkpoint_path,
    ) = _verify_sources(contract, dataset)

    seed = int(push_contract["fine_tune"]["batch_order_seed"])
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
        raise ModelCPushforwardDurationError("loss-v1 contract changed")
    architecture = architecture_from_candidate(
        successor_contract,
        REFERENCE_CANDIDATE,
    )
    training_dataset = ModelCPushforwardDataset(dataset, training_records)
    extension = contract["extension"]
    loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            training_dataset,
            int(extension["batch_size"]),
            seed,
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    wet = torch.from_numpy(training_dataset.wet.astype(np.float32))[
        None, None
    ].to(device)
    boundary = torch.from_numpy(
        western_boundary_mask(
            training_dataset.wet,
            loss_config.western_boundary_width,
        ).astype(np.float32)
    )[None, None].to(device)
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
    model = build_successor(architecture).to(device)
    model.load_state_dict(source_payload["model_state_dict"])
    parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(extension["initial_learning_rate"]),
        betas=tuple(float(value) for value in extension["adam_betas"]),
        weight_decay=float(extension["weight_decay"]),
    )
    endpoint_steps = tuple(
        int(value // HORIZON_DAYS)
        for value in push_contract["objective"][
            "pushforward_endpoint_days"
        ]
    )
    pushforward_weight = float(
        push_contract["objective"]["pushforward_weight"]
    )
    climatology_scales = push_contract["objective"][
        "climatology_rmse_scales"
    ]
    maximum_steps = int(extension["maximum_total_steps"])
    decay_step = int(extension["absolute_decay_step"])
    checkpoint_steps = tuple(
        int(value) for value in extension["checkpoint_steps"]
    )

    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    checkpoint_directory = temporary / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir()
    iterator = iter(loader)
    window_totals = {name: 0.0 for name in TRAINING_TERMS}
    window_samples = 0
    history = []
    saved = []
    replay_comparison: dict[str, Any] | None = None

    for step in range(1, maximum_steps + 1):
        if step == decay_step + 1:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= float(
                    push_contract["fine_tune"]["decay_factor"]
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
        endpoint_step = endpoint_steps[(step - 1) % len(endpoint_steps)]
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
        endpoint = detached_pushforward_endpoint(
            model,
            features,
            wet,
            base_predictions,
            endpoint_step=endpoint_step,
        )
        slow = slow_field_pushforward_loss(
            endpoint,
            futures[:, endpoint_step - 1],
            wet,
            physical_scale,
            climatology_scales[str(endpoint_step * HORIZON_DAYS)],
        )
        training_values = {
            **terms,
            "pushforward_sst": slow["sst"],
            "pushforward_phihyd_surface": slow["phihyd_surface"],
        }
        training_values["total"] = (
            terms["total"] + pushforward_weight * slow["mean"]
        )
        if not all(
            bool(torch.isfinite(training_values[name]).item())
            for name in TRAINING_TERMS
        ):
            raise ModelCPushforwardDurationError(
                "duration training loss became non-finite"
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

        if step == REPLAY_END_STEP:
            selected_payload = torch.load(
                push_selected_path,
                map_location="cpu",
                weights_only=False,
            )
            replay_comparison = compare_state_dicts(
                _checkpoint_state_dict(model),
                selected_payload["model_state_dict"],
            )
            if not replay_comparison["bitwise_exact"]:
                raise ModelCPushforwardDurationError(
                    "pushforward-v1 replay was not bitwise exact"
                )
            window_totals = {name: 0.0 for name in TRAINING_TERMS}
            window_samples = 0
            continue
        if step not in checkpoint_steps:
            continue
        training_window = {
            name: window_totals[name] / window_samples
            for name in TRAINING_TERMS
        }
        history_record = {
            "total_fine_tune_step": step,
            "extension_step": step - REPLAY_END_STEP,
            "optimizer_learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "training_window": training_window,
        }
        history.append(history_record)
        checkpoint_path = _checkpoint_path(checkpoint_directory, step)
        payload = {
            "version": DURATION_VERSION,
            "purpose": "replay_verified_pushforward_low_rate_duration",
            "total_fine_tune_step": step,
            "extension_step": step - REPLAY_END_STEP,
            "source_optimizer_step": 14400,
            "candidate_id": REFERENCE_CANDIDATE,
            "architecture": architecture.to_dict(),
            "parameter_count": parameter_count,
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "base_loss_contract": loss_contract(loss_config),
            "base_loss_contract_sha256": loss_contract_sha256(loss_config),
            "pushforward_objective": push_contract["objective"],
            "duration_contract": extension,
            "training_history_record": history_record,
            "model_state_dict": _checkpoint_state_dict(model),
        }
        torch.save(payload, checkpoint_path)
        saved.append(
            {
                "fine_tune_step": step,
                "extension_step": step - REPLAY_END_STEP,
                "checkpoint": str(
                    output / CHECKPOINT_DIRECTORY / checkpoint_path.name
                ),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
            }
        )
        window_totals = {name: 0.0 for name in TRAINING_TERMS}
        window_samples = 0

    if replay_comparison is None or len(saved) != len(checkpoint_steps):
        raise ModelCPushforwardDurationError(
            "duration run did not complete replay and checkpoints"
        )

    mean, scale, wet_array, _, wind_mean, wind_scale = _normalizers(group)
    climatology_state, climatology_derived, training_days = (
        _training_climatology(state, snapshot_codes, wet_array)
    )
    source_arrays = np.load(source_arrays_path)
    if not np.array_equal(source_arrays["records"], audit_records):
        raise ModelCPushforwardDurationError(
            "pushforward-v1 evaluation records changed"
        )
    arrays: dict[str, np.ndarray] = {
        "records": audit_records.astype(np.int32),
        "training_times": audit_times.astype(np.int32),
        "lead_days": np.asarray(range(10, 91, 10), dtype=np.int16),
    }
    for prefix in (
        "source_persistence",
        "source_climatology",
        "fine_tune_1920",
    ):
        for name in source_arrays.files:
            if name.startswith(prefix + "__"):
                arrays[f"source_{prefix}__{name.split('__', 1)[1]}"] = (
                    np.asarray(source_arrays[name])
                )
    baselines = {
        method: {
            name.split("__", 1)[1]: np.asarray(source_arrays[name])
            for name in source_arrays.files
            if name.startswith(f"source_{method}__")
        }
        for method in ("persistence", "climatology")
    }

    summaries = []
    for saved_record, history_record in zip(saved, history):
        step = int(saved_record["fine_tune_step"])
        checkpoint_path = _checkpoint_path(checkpoint_directory, step)
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
            arrays[f"duration_{step}__{name}"] = np.asarray(value)
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

    decision = select_pushforward_checkpoint(summaries)
    selected_step = int(decision["selected_fine_tune_step"])
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
        raise ModelCPushforwardDurationError(
            "duration selected checkpoint did not reload exactly"
        )

    arrays_path = temporary / ARRAYS_NAME
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "status": "complete",
        "purpose": "training_only_pushforward_low_rate_duration_diagnosis",
        "version": DURATION_VERSION,
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
        "pushforward_objective": push_contract["objective"],
        "duration_contract": extension,
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
        "pushforward_v1_replay": {
            "required_exact_step": REPLAY_END_STEP,
            "state_dict_comparison": replay_comparison,
        },
        "history": history,
        "checkpoint_summary": summaries,
        "duration_decision": decision,
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


def preflight_duration(
    dataset_path: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Verify the duration sources without opening scientific states."""

    contract, resolved, digest = load_duration_contract(contract_path)
    dataset = Path(dataset_path).resolve()
    _verify_sources(contract, dataset)
    return {
        "status": "ready",
        "version": DURATION_VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "required_exact_step": REPLAY_END_STEP,
        "maximum_total_steps": contract["extension"][
            "maximum_total_steps"
        ],
        "checkpoint_steps": contract["extension"]["checkpoint_steps"],
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
        result = preflight_duration(args.dataset, args.contract)
    else:
        result = run_duration_extension(
            args.dataset,
            args.contract,
            args.output_dir,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
