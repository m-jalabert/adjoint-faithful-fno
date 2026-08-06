"""Training-only Bire LayerNorm/dropout controls for anomaly-direct Model C."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from .af_a0_evaluate import _normalizers
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
from .af_model_c_anomaly_direct import (
    ModelCAnomalyRolloutDataset,
    direct_state_unroll,
    pointwise_increment_scale,
)
from . import (
    af_model_c_anomaly_direct_training_spectral_attribution_v2 as attribution,
)
from .af_model_c_anomaly_direct_deep_pressure_spectral_regularization import (
    summarize_evaluation,
)
from .af_model_c_overfit import AUDIT_TERMS, _device, _file_sha256
from .af_model_c_successor import (
    STATE_CHANNEL_COUNT,
    ModelCSuccessorArchitecture,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from neuralop.models import FNO
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]
    FNO = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


VERSION = "model_c_anomaly_direct_bire_regularization_controls_v1"
CHECKPOINT_STEPS = (3840, 7680, 11520, 13440, 14400, 14880, 15360)
ARM_IDS = ("layernorm", "dropout", "layernorm_dropout")
REPORT_NAME = "bire_regularization_control_report.json"
ARRAYS_NAME = "bire_regularization_control_arrays.npz"
FIGURE_NAME = "model_c_bire_regularization_control_selection.png"
README_NAME = "README.md"
MANIFEST_NAME = "manifest.json"
CHECKPOINT_DIRECTORY = "training_checkpoints"
BEST_NAME = "model_c_bire_regularization_control_best_diagnostic.pt"


class BireRegularizationControlError(RuntimeError):
    """Raised when a frozen Bire regularization control changes."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RegularizationArm:
    """One factorial arm in the frozen LayerNorm/dropout comparison."""

    arm_id: str
    pointwise_layer_norm: bool
    channel_mlp_dropout: float

    def __post_init__(self) -> None:
        if self.arm_id not in ARM_IDS:
            raise ValueError("unknown Bire regularization arm")
        if self.channel_mlp_dropout not in (0.0, 0.5):
            raise ValueError("dropout control must use zero or archived default 0.5")


if nn is not None:

    class PointwiseChannelLayerNorm(nn.Module):
        """LayerNorm over channels independently at every grid point."""

        def __init__(self, channels: int, epsilon: float = 1.0e-5) -> None:
            super().__init__()
            self.channels = int(channels)
            self.epsilon = float(epsilon)
            self.weight = nn.Parameter(torch.ones(self.channels))
            self.bias = nn.Parameter(torch.zeros(self.channels))

        def forward(self, value: Any) -> Any:
            if value.ndim != 4 or value.shape[1] != self.channels:
                raise ValueError("pointwise LayerNorm expects N,C,Y,X")
            channels_last = value.permute(0, 2, 3, 1)
            normalized = functional.layer_norm(
                channels_last,
                (self.channels,),
                self.weight,
                self.bias,
                self.epsilon,
            )
            return normalized.permute(0, 3, 1, 2)


    class BireRegularizedDirectFNO(nn.Module):
        """The fixed width-128 Model C with explicit Bire regularizers."""

        def __init__(
            self,
            architecture: ModelCSuccessorArchitecture,
            arm: RegularizationArm,
        ) -> None:
            super().__init__()
            self.architecture = architecture
            self.arm = arm
            self.fno = FNO(
                n_modes=architecture.n_modes,
                in_channels=architecture.in_channels,
                out_channels=architecture.out_channels,
                hidden_channels=architecture.hidden_channels,
                n_layers=architecture.n_layers,
                lifting_channel_ratio=architecture.lifting_channel_ratio,
                projection_channel_ratio=architecture.projection_channel_ratio,
                positional_embedding=architecture.positional_embedding,
                use_channel_mlp=architecture.use_channel_mlp,
                channel_mlp_dropout=arm.channel_mlp_dropout,
                channel_mlp_expansion=architecture.channel_mlp_expansion,
                domain_padding=architecture.domain_padding,
                fno_block_precision=architecture.fno_block_precision,
                factorization=architecture.factorization,
            )
            if arm.pointwise_layer_norm:
                self.fno.fno_blocks.norm = nn.ModuleList(
                    [
                        PointwiseChannelLayerNorm(
                            architecture.hidden_channels,
                        )
                        for _ in range(architecture.n_layers * 2)
                    ]
                )
            self.local = nn.Conv2d(
                architecture.in_channels,
                architecture.out_channels,
                kernel_size=architecture.local_kernel_size,
                padding=architecture.local_kernel_size // 2,
            )

        def forward(self, features: Any) -> Any:
            if (
                features.ndim != 4
                or features.shape[1] != self.architecture.in_channels
            ):
                raise ValueError("regularized Model C expects N,51,Y,X")
            return self.fno(features) + self.local(features)

else:  # pragma: no cover
    PointwiseChannelLayerNorm = None  # type: ignore[assignment,misc]
    BireRegularizedDirectFNO = None  # type: ignore[assignment,misc]


def build_regularized_model(
    architecture: ModelCSuccessorArchitecture,
    arm: RegularizationArm,
) -> Any:
    require_model_a_runtime()
    if BireRegularizedDirectFNO is None:  # pragma: no cover
        raise RuntimeError("Bire regularization controls require PyTorch")
    return BireRegularizedDirectFNO(architecture, arm)


def arm_from_contract(
    contract: Mapping[str, Any],
    arm_index: int,
) -> RegularizationArm:
    arms = contract.get("arms", ())
    if arm_index < 0 or arm_index >= len(arms):
        raise ValueError("Bire regularization arm index is out of range")
    record = arms[arm_index]
    arm = RegularizationArm(
        arm_id=str(record["arm_id"]),
        pointwise_layer_norm=bool(record["pointwise_layer_norm"]),
        channel_mlp_dropout=float(record["channel_mlp_dropout"]),
    )
    expected = (
        ("layernorm", True, 0.0),
        ("dropout", False, 0.5),
        ("layernorm_dropout", True, 0.5),
    )
    if (
        arm.arm_id,
        arm.pointwise_layer_norm,
        arm.channel_mlp_dropout,
    ) != expected[arm_index]:
        raise ValueError("Bire regularization factorial changed")
    return arm


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    training = contract.get("training", {})
    read = contract.get("read_contract", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status")
        != "frozen_after_failed_spectral_fine_tune_before_regularization_control_metrics"
        or len(contract.get("arms", ())) != 3
        or int(training.get("seed", -1)) != 20260724
        or int(training.get("maximum_steps", -1)) != 15360
        or tuple(training.get("checkpoint_steps", ())) != CHECKPOINT_STEPS
        or int(training.get("batch_size", -1)) != 4
        or float(training.get("initial_learning_rate", -1.0)) != 5.0e-4
        or read.get("training_state") is not True
        or any(
            read.get(name) is not False
            for name in (
                "validation_state",
                "inference_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
                "long_term_state",
            )
        )
    ):
        raise ValueError("Bire regularization control contract changed")
    for index in range(3):
        arm_from_contract(contract, index)
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(f"Bire regularization control source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def _verify_artifacts(
    contract: Mapping[str, Any],
    dataset: Path,
) -> tuple[Path, dict[str, Any]]:
    sources = contract["sources"]
    if (
        dataset != Path(sources["dataset"]["path"]).resolve()
        or _file_sha256(dataset / ".zmetadata")
        != sources["dataset"]["metadata_sha256"]
    ):
        raise BireRegularizationControlError("dataset source changed")
    for name in (
        "normalization",
        "attribution_report",
        "attribution_arrays",
        "attribution_manifest",
        "failed_spectral_report",
        "failed_spectral_manifest",
    ):
        record = sources[name]
        path = Path(record["path"]).resolve()
        if not path.is_file() or _file_sha256(path) != record["sha256"]:
            raise BireRegularizationControlError(f"source artifact changed: {name}")
    attribution_contract, _, digest = attribution.load_contract(
        sources["attribution_contract"]["path"]
    )
    if digest != sources["attribution_contract"]["sha256"]:
        raise BireRegularizationControlError("attribution contract changed")
    attribution._verify_sources(attribution_contract, dataset)
    failed = json.loads(Path(sources["failed_spectral_report"]["path"]).read_text())
    if (
        failed.get("selection_decision", {}).get("status")
        != "no_fine_tune_checkpoint_passed"
        or failed.get("selection_decision", {}).get("selected_fine_tune_step") != 0
        or failed.get("content_sha256")
        != sources["failed_spectral_report"]["content_sha256"]
    ):
        raise BireRegularizationControlError(
            "failed spectral fine-tune result changed"
        )
    return Path(sources["normalization"]["path"]).resolve(), attribution_contract


def select_arm_checkpoint(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select a passing arm checkpoint or retain a diagnostic only."""

    if not summaries:
        raise ValueError("arm checkpoint selection needs summaries")
    passing = [
        value for value in summaries if value.get("gate", {}).get("pass") is True
    ]
    pool = passing if passing else list(summaries)
    selected = min(
        pool,
        key=lambda value: (
            float(value["worst_mid_bottom_modewise_ratio_all_leads"]),
            float(value["worst_primary_relative_to_source"]),
            int(value["fine_tune_step"]),
        ),
    )
    return {
        "status": (
            "arm_training_gate_passed"
            if passing
            else "no_arm_checkpoint_passed"
        ),
        "arm_training_gate_passed": bool(passing),
        "selected_optimizer_step": int(selected["optimizer_step"]),
        "selected_fine_tune_step": int(selected["fine_tune_step"]),
        "best_diagnostic_optimizer_step": int(selected["optimizer_step"]),
        "next_action": (
            "wait_for_all_frozen_factorial_arms_then_aggregate"
        ),
    }


def _checkpoint_path(directory: Path, step: int) -> Path:
    return directory / f"model_c_bire_regularization_step_{step:05d}.pt"


def _source_evaluation(sources: Mapping[str, Any]) -> dict[str, Any]:
    with np.load(sources["attribution_arrays"]["path"], allow_pickle=False) as data:
        seeds = np.asarray(data["seeds"])
        index = int(np.flatnonzero(seeds == 20260724)[0])
        return {
            "optimizer_step": 13440,
            "ratio": np.asarray(data["frozen_median_modewise_ratio"][index]),
            "integrated": np.asarray(data["integrated_energy_ratio"][index]),
            "tail_model": np.asarray(data["tail_model_fraction"][index]),
            "tail_truth": np.asarray(data["tail_truth_fraction"][index]),
            "model_rmse": np.asarray(data["primary_model_rmse"][index]),
            "persistence_rmse": np.asarray(data["primary_persistence_rmse"][index]),
        }


def _evaluate_checkpoint(
    checkpoint: Path,
    optimizer_step: int,
    normalization: Path,
    arm: RegularizationArm,
    **kwargs: Any,
) -> dict[str, Any]:
    record = {
        "seed": 20260724,
        "optimizer_step": optimizer_step,
        "checkpoint": {"path": str(checkpoint)},
        "normalization": {"path": str(normalization)},
    }
    original_builder = attribution.build_successor
    attribution.build_successor = lambda architecture: build_regularized_model(
        architecture,
        arm,
    )
    try:
        return attribution._evaluate_seed(record, **kwargs)
    finally:
        attribution.build_successor = original_builder


def _plot(
    path: Path,
    summaries: Sequence[Mapping[str, Any]],
    selected_step: int,
    arm: RegularizationArm,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([value["fine_tune_step"] for value in summaries])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(
        steps,
        [value["day360_mid_modewise_ratio"] for value in summaries],
        "o-",
        label="PHIHYD mid",
    )
    axes[0].plot(
        steps,
        [value["day360_bottom_modewise_ratio"] for value in summaries],
        "s-",
        label="PHIHYD bottom",
    )
    axes[0].axhline(4.0, color="black", linestyle="--", label="factor-four")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Day-360 median modewise ratio")
    axes[0].legend()
    for field in ("surface_speed", "sst", "phihyd_surface"):
        axes[1].plot(
            steps,
            [
                value["primary_10_to_90_rmse_ratio_to_persistence"][field]
                for value in summaries
            ],
            "o-",
            label=field.replace("_", " "),
        )
    axes[1].axhline(1.0, color="black", linestyle="--", label="persistence")
    axes[1].set_ylabel("10--90-day RMSE ratio")
    axes[1].legend()
    for axis in axes:
        axis.axvline(selected_step, color="#d95f02", alpha=0.7)
        axis.set_xlabel("Optimizer step (0 = fixed source)")
        axis.grid(alpha=0.25)
    figure.suptitle(f"Bire regularization control: {arm.arm_id}")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def preflight(
    contract_path: str | Path,
    arm_index: int,
) -> dict[str, Any]:
    contract, resolved, digest = load_contract(contract_path)
    arm = arm_from_contract(contract, arm_index)
    dataset = Path(contract["sources"]["dataset"]["path"]).resolve()
    _, attribution_contract = _verify_artifacts(contract, dataset)
    group = zarr.open_consolidated(str(dataset), mode="r")
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    records = attribution.training_records(attribution_contract, split)
    return {
        "status": "ready",
        "version": VERSION,
        "arm": arm.arm_id,
        "contract": str(resolved),
        "contract_sha256": digest,
        "selection_records": int(records.shape[0]),
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }


def run(
    contract_path: str | Path,
    arm_index: int,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train and evaluate one frozen factorial arm."""

    if torch is None or DataLoader is None:  # pragma: no cover
        raise RuntimeError("Bire regularization controls require PyTorch")
    require_model_a_runtime()
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    arm = arm_from_contract(contract, arm_index)
    dataset = Path(contract["sources"]["dataset"]["path"]).resolve()
    normalization, attribution_contract = _verify_artifacts(contract, dataset)
    scratch = Path(contract["output"]["scratch_root"]).resolve() / arm.arm_id
    project = Path(contract["output"]["project_root"]).resolve() / arm.arm_id
    scratch_tmp = scratch.with_name(scratch.name + ".tmp")
    project_tmp = project.with_name(project.name + ".tmp")
    if any(path.exists() for path in (scratch, project, scratch_tmp, project_tmp)):
        raise FileExistsError("refusing to overwrite Bire regularization output")

    training = contract["training"]
    seed_everything(int(training["seed"]))
    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    _, _, wet_array, _, wind_mean, wind_scale = _normalizers(group)
    wet_array = np.asarray(wet_array, dtype=bool)
    with np.load(normalization, allow_pickle=False) as values:
        point_mean = np.asarray(values["pointwise_mean"], dtype=np.float32)
        point_scale = np.asarray(values["pointwise_scale"], dtype=np.float32)
    training_records = records_for_rollout_split(pair_codes, 1, rollout_steps=3)
    training_dataset = ModelCAnomalyRolloutDataset(
        dataset,
        training_records,
        point_mean,
        point_scale,
        rollout_steps=3,
    )
    loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            training_dataset,
            int(training["batch_size"]),
            int(training["seed"]),
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    loss_config = model_c_loss_config("v1")
    if loss_contract_sha256(loss_config) != MODEL_C_LOSS_V1_CONTRACT_SHA256:
        raise BireRegularizationControlError("loss-v1 changed")
    architecture = ModelCSuccessorArchitecture(**contract["architecture"])
    model = build_regularized_model(architecture, arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["initial_learning_rate"]),
        betas=tuple(float(value) for value in training["adam_betas"]),
        weight_decay=float(training["weight_decay"]),
    )
    wet = torch.from_numpy(wet_array.astype(np.float32))[None, None].to(device)
    boundary_array = western_boundary_mask(
        wet_array,
        loss_config.western_boundary_width,
    )
    boundary = torch.from_numpy(boundary_array.astype(np.float32))[
        None, None
    ].to(device)
    increment_scale = torch.from_numpy(
        pointwise_increment_scale(group, pair_codes, point_scale)
    ).to(device)
    maximum_steps = int(training["maximum_steps"])
    decay_step = int(round(maximum_steps * float(training["decay_fraction"])))
    scratch_tmp.parent.mkdir(parents=True, exist_ok=True)
    project_tmp.parent.mkdir(parents=True, exist_ok=True)
    scratch_tmp.mkdir()
    project_tmp.mkdir()
    checkpoint_directory = scratch_tmp / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir()
    iterator = iter(loader)
    totals = {name: 0.0 for name in AUDIT_TERMS}
    samples = 0
    history = []
    checkpoints = []
    for step in range(1, maximum_steps + 1):
        if step == decay_step + 1:
            for group_record in optimizer.param_groups:
                group_record["lr"] *= float(training["decay_factor"])
        try:
            features, futures = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            features, futures = next(iterator)
        features = features.to(device=device, dtype=torch.float32, non_blocking=True)
        futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
        model.train()
        predictions = direct_state_unroll(model, features, wet, 3)
        terms = model_c_loss_terms(
            predictions,
            futures,
            features[:, :STATE_CHANNEL_COUNT],
            wet,
            boundary,
            increment_scale,
            loss_config,
        )
        if not all(bool(torch.isfinite(terms[name]).item()) for name in AUDIT_TERMS):
            raise BireRegularizationControlError("training objective became non-finite")
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        optimizer.step()
        batch = int(features.shape[0])
        for name in AUDIT_TERMS:
            totals[name] += float(terms[name].detach().cpu()) * batch
        samples += batch
        if step not in CHECKPOINT_STEPS:
            continue
        window = {name: totals[name] / samples for name in AUDIT_TERMS}
        history_record = {
            "optimizer_step": step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_window": window,
        }
        history.append(history_record)
        checkpoint_path = _checkpoint_path(checkpoint_directory, step)
        payload = {
            "version": VERSION,
            "arm": {
                "arm_id": arm.arm_id,
                "pointwise_layer_norm": arm.pointwise_layer_norm,
                "channel_mlp_dropout": arm.channel_mlp_dropout,
            },
            "optimizer_step": step,
            "fine_tune_step": step,
            "architecture": architecture.to_dict(),
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "base_loss_contract": loss_contract(loss_config),
            "base_loss_contract_sha256": MODEL_C_LOSS_V1_CONTRACT_SHA256,
            "training_history_record": history_record,
            "model_state_dict": _checkpoint_state_dict(model),
        }
        torch.save(payload, checkpoint_path)
        checkpoints.append(
            {
                "optimizer_step": step,
                "fine_tune_step": step,
                "checkpoint": str(
                    scratch / CHECKPOINT_DIRECTORY / checkpoint_path.name
                ),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
            }
        )
        totals = {name: 0.0 for name in AUDIT_TERMS}
        samples = 0
    del model, optimizer, loader, training_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    records = attribution.training_records(attribution_contract, split)
    initial = attribution.base._gather_states(state, records, 0)
    raw_static = attribution.base._gather_static(static, records)
    common = {
        "device": device,
        "initial": initial,
        "raw_static": raw_static,
        "experiments": records[:, 0],
        "state": state,
        "records": records,
        "wet": wet_array,
        "wind_mean": float(wind_mean),
        "wind_scale": float(wind_scale),
        "modes": np.arange(1, 31, dtype=np.float32),
    }
    source = _source_evaluation(contract["sources"])
    source_summary = summarize_evaluation(source)
    source_primary = source_summary[
        "primary_10_to_90_rmse_ratio_to_persistence"
    ]
    evaluated = []
    summaries = []
    for record in checkpoints:
        value = _evaluate_checkpoint(
            scratch_tmp / CHECKPOINT_DIRECTORY / Path(record["checkpoint"]).name,
            int(record["optimizer_step"]),
            normalization,
            arm,
            **common,
        )
        evaluated.append(value)
        summary = summarize_evaluation(
            value,
            source_primary_ratios=source_primary,
            selection=contract["selection"],
        )
        summary["fine_tune_step"] = int(record["optimizer_step"])
        summaries.append(summary)
    decision = select_arm_checkpoint(summaries)
    selected_step = int(decision["selected_optimizer_step"])
    shutil.copy2(
        _checkpoint_path(checkpoint_directory, selected_step),
        scratch_tmp / BEST_NAME,
    )

    steps = np.asarray([value["optimizer_step"] for value in summaries], dtype=np.int32)
    arrays_path = scratch_tmp / ARRAYS_NAME
    np.savez_compressed(
        arrays_path,
        optimizer_steps=steps,
        lead_days=np.arange(10, 361, 10, dtype=np.int16),
        frozen_median_modewise_ratio=np.stack(
            [value["ratio"] for value in evaluated]
        ).astype(np.float32),
        integrated_energy_ratio=np.stack(
            [value["integrated"] for value in evaluated]
        ).astype(np.float32),
        primary_model_rmse=np.stack(
            [value["model_rmse"] for value in evaluated]
        ).astype(np.float32),
        primary_persistence_rmse=np.asarray(
            source["persistence_rmse"],
            dtype=np.float32,
        ),
        source_frozen_median_modewise_ratio=np.asarray(
            source["ratio"],
            dtype=np.float32,
        ),
        selection_records=records.astype(np.int32),
    )
    report = {
        "status": "complete",
        "version": VERSION,
        "arm": {
            "arm_id": arm.arm_id,
            "pointwise_layer_norm": arm.pointwise_layer_norm,
            "channel_mlp_dropout": arm.channel_mlp_dropout,
        },
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "architecture": architecture.to_dict(),
        "parameter_count": int(
            sum(
                value.numel()
                for value in torch.load(
                    _checkpoint_path(checkpoint_directory, selected_step),
                    map_location="cpu",
                    weights_only=False,
                )["model_state_dict"].values()
            )
        ),
        "training_history": history,
        "checkpoints": checkpoints,
        "source_summary": source_summary,
        "evaluation_summaries": summaries,
        "selection_decision": decision,
        "best_diagnostic_checkpoint": str(scratch / BEST_NAME),
        "best_diagnostic_checkpoint_sha256": _file_sha256(scratch_tmp / BEST_NAME),
        "arrays": str(scratch / ARRAYS_NAME),
        "arrays_sha256": _file_sha256(arrays_path),
        "read_contract": contract["read_contract"],
        "validation_state_opened": False,
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
        "long_term_state_opened": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    report["content_sha256"] = _json_sha256(report)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (scratch_tmp / REPORT_NAME).write_text(rendered)
    (project_tmp / REPORT_NAME).write_text(rendered)
    shutil.copy2(arrays_path, project_tmp / ARRAYS_NAME)
    combined_summaries = [
        {**source_summary, "fine_tune_step": 0},
        *summaries,
    ]
    _plot(
        project_tmp / FIGURE_NAME,
        combined_summaries,
        selected_step,
        arm,
    )
    (project_tmp / README_NAME).write_text(
        "# Model C Bire regularization control\n\n"
        f"Arm: `{arm.arm_id}`. Decision: `{decision['status']}`. "
        "Training split only; all later archives remained sealed.\n"
    )
    artifacts = {
        name: _file_sha256(project_tmp / name)
        for name in (REPORT_NAME, ARRAYS_NAME, FIGURE_NAME, README_NAME)
    }
    manifest = {
        "status": "complete",
        "version": VERSION,
        "arm": arm.arm_id,
        "contract_sha256": contract_sha,
        "artifacts": artifacts,
        "content_sha256": _json_sha256(artifacts),
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }
    (project_tmp / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(scratch_tmp, scratch)
    os.replace(project_tmp, project)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        child.add_argument("--arm-index", type=int, required=True)
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
        result = preflight(args.contract, args.arm_index)
    else:
        result = run(
            args.contract,
            args.arm_index,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
