"""Bire-aligned architecture with the incumbent group-balanced Model C objective.

The Bire-aligned arms established two things.  The \\(10^{-2}\\) learning rate, not
the architecture, caused the collapse to climatology; and at \\(5\\times10^{-4}\\) the
three-block map is *dynamically bounded* -- day-2,000 error flat at 3--5 times
climatology with per-call gain at or below 1.001 -- but converges on the wrong
invariant distribution, at roughly half the incumbent's day-200 anomaly
correlation.  The diagnosis is bounded dynamics with an incorrect attractor,
not an unstable autoregressive Jacobian.

Closing the remaining code-fidelity gaps to ``oceanfourcast`` did not help: the
faithfulness bundle improved the training gate and lost held-S0 skill on the
field the base arm was strongest at.  The remaining suspect is the objective.

Bire's ``MSE + w MAE`` averages over all 46 normalized channels, which gives the
physical groups effective multiplicities \\(U:V:\\Theta:\\eta = 15:15:15:1\\).  The
free surface therefore receives \\(1/46\\) of the channel-averaged loss while each
of \\(U, V, \\Theta\\) receives \\(15/46\\).  That is not equivalent to Bire's problem:
they predicted several PHIHYD levels and the streamfunction directly, whereas
this model predicts only \\((U, V, \\Theta, \\eta)\\) and reconstructs pressure and
circulation afterwards.  The objective consequently leaves the free surface,
barotropic circulation, hydrostatic pressure, and the slow basin-scale modes
weakly constrained -- exactly the quantities that decide which attractor a long
rollout approaches.

This arm is the architecture-fixed loss-recovery control.  It keeps every
architectural and optimizer choice of the working \\(5\\times10^{-4}\\) arm and
changes only the objective and the rollout exposure, restoring the incumbent
three-step group-balanced Model C loss v1::

    L_state = (L_U + L_V + L_Theta + L_eta) / 4

together with its increment, rollout, spectral, and western-boundary terms.  It
answers one question: is the bounded behaviour coming from the Bire-aligned
architecture while the loss of skill comes from the Bire objective?

Frozen against ``model_c_bire_aligned_full_state_lr5e4_v1``: the architecture
dataclass, the 49 external inputs, the Bire position fields, the six pointwise
LayerNorms, the absent external 3x3 branch, 10% padding, seed, batch size 8,
Adam with betas (0.9, 0.95), zero weight decay, absent gradient clipping,
lr0 = 5e-4, and the 7,680-step budget.  Only the objective and the rollout
exposure move, so the optimizer-step and sequence-exposure budgets stay directly
comparable with all three Bire-aligned arms.

Training split only.  The held S0 figure suite opens through its own contract.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from . import af_model_c_anomaly_direct_training_spectral_attribution_v2 as attribution
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
from .af_model_c_anomaly_direct_bire_regularization_controls import (
    select_arm_checkpoint,
)
from .af_model_c_anomaly_direct_deep_pressure_spectral_regularization import (
    summarize_evaluation,
)
from .af_model_c_bire_aligned_full_state import (
    ARRAYS_NAME,
    CHECKPOINT_DIRECTORY,
    EXTERNAL_INPUT_CHANNELS,
    LIFTING_INPUT_CHANNELS,
    MANIFEST_NAME,
    MAXIMUM_STEPS,
    README_NAME,
    BireAlignedArchitecture,
    BireAlignedDivergenceError,
    BireAlignedFullStateError,
    _evaluate_checkpoint,
    _json_sha256,
    _plot,
    _source_evaluation,
    _verify_artifacts,
    build_bire_aligned_model,
    retained_features,
)
from .af_model_c_overfit import AUDIT_TERMS, _device, _file_sha256
from .af_model_c_successor import STATE_CHANNEL_COUNT

try:
    import torch
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


VERSION = "model_c_bire_aligned_loss_recovery_v1"
CONTRACT_STATUS = "frozen_before_any_bire_aligned_loss_recovery_metric"

LEARNING_RATE = 5.0e-4
ROLLOUT_STEPS = 3
CHECKPOINT_STEPS = (1920, 3840, 5760, 7680)
PARENT_VERSION = "model_c_bire_aligned_full_state_lr5e4_v1"

REPORT_NAME = "bire_aligned_loss_recovery_report.json"
FIGURE_NAME = "model_c_bire_aligned_loss_recovery_selection.png"
DIVERGENCE_NAME = "bire_aligned_loss_recovery_divergence.json"

#: Everything that must still match the 5e-4 arm.  The objective and the rollout
#: exposure are deliberately absent: they are the declared change.
FROZEN_TRAINING_FIELDS = (
    "seed",
    "optimizer",
    "batch_size",
    "adam_betas",
    "weight_decay",
    "gradient_clipping",
    "maximum_steps",
    "initial_learning_rate",
)


class BireAlignedLossRecoveryError(BireAlignedFullStateError):
    """Raised when the loss-recovery control violates its contract."""


def _assert_single_change(contract: Mapping[str, Any]) -> None:
    """Fail unless only the objective and rollout exposure moved from the parent."""

    record = contract["sources"]["parent_contract"]
    path = Path(record["path"]).resolve()
    if not path.is_file() or _file_sha256(path) != record["sha256"]:
        raise BireAlignedLossRecoveryError("the parent 5e-4 contract changed")
    parent = json.loads(path.read_text())
    if (
        parent.get("version") != PARENT_VERSION
        or float(parent["training"]["initial_learning_rate"]) != LEARNING_RATE
    ):
        raise BireAlignedLossRecoveryError("the parent arm is not the 5e-4 control")
    if contract["architecture"] != parent["architecture"]:
        raise BireAlignedLossRecoveryError(
            "the loss-recovery control must keep the parent architecture exactly"
        )
    for field in FROZEN_TRAINING_FIELDS:
        if contract["training"].get(field) != parent["training"].get(field):
            raise BireAlignedLossRecoveryError(
                f"the loss-recovery control moved a third quantity: {field}"
            )
    if contract["loss"] == parent["loss"]:
        raise BireAlignedLossRecoveryError("the objective change is missing")


def _assert_objective_description_is_consistent(contract: Mapping[str, Any]) -> None:
    """Reject a contract whose prose contradicts its authoritative fields.

    The faithfulness arm shipped stage strings quoting an MAE weight of 0.01
    while its authoritative ``loss.mae_weight`` was 0.05, because an equality
    assertion against the parent forced the stale prose through.  The run was
    numerically correct --- the code reads the authoritative field --- but the
    description was wrong, so the check is now explicit.
    """

    loss = contract["loss"]
    declared = str(loss.get("objective", ""))
    if "mse_mae" in declared or "mae" in declared.lower():
        raise BireAlignedLossRecoveryError(
            "this arm restores the group-balanced Model C loss, not MSE+MAE"
        )
    for stage in contract.get("stages", ()):
        text = str(stage.get("objective", ""))
        if "MAE" in text or "0.01" in text or "0.05" in text:
            raise BireAlignedLossRecoveryError(
                "a stage objective string still describes the Bire MSE+MAE loss"
            )
        if str(stage.get("rollout_steps")) != str(ROLLOUT_STEPS):
            raise BireAlignedLossRecoveryError(
                "every stage of this arm uses the three-step rollout"
            )


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the contract frozen before any loss-recovery metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    architecture = contract.get("architecture", {})
    training = contract.get("training", {})
    loss = contract.get("loss", {})
    read = contract.get("read_contract", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or int(architecture.get("in_channels", -1)) != EXTERNAL_INPUT_CHANNELS
        or int(architecture.get("lifting_in_channels", -1))
        != LIFTING_INPUT_CHANNELS
        or int(architecture.get("out_channels", -1)) != STATE_CHANNEL_COUNT
        or int(architecture.get("n_layers", -1)) != 3
        or architecture.get("local_kernel_size") is not None
        or architecture.get("positional_embedding") is not None
        or int(training.get("seed", -1)) != 20260724
        or int(training.get("batch_size", -1)) != 8
        or float(training.get("initial_learning_rate", -1.0)) != LEARNING_RATE
        or tuple(float(v) for v in training.get("adam_betas", ())) != (0.9, 0.95)
        or float(training.get("weight_decay", -1.0)) != 0.0
        or training.get("optimizer") != "adam"
        or training.get("gradient_clipping") is not False
        or int(training.get("maximum_steps", -1)) != MAXIMUM_STEPS
        or tuple(training.get("checkpoint_steps", ())) != CHECKPOINT_STEPS
        or int(training.get("rollout_steps", -1)) != ROLLOUT_STEPS
        or loss.get("objective") != "incumbent_group_balanced_model_c_loss_v1"
        or loss.get("contract_sha256") != MODEL_C_LOSS_V1_CONTRACT_SHA256
        or read.get("training_state") is not True
        or any(
            read.get(name) is not False
            for name in (
                "validation_state",
                "inference_state",
                "held_s0_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
            )
        )
    ):
        raise BireAlignedLossRecoveryError("Bire-aligned loss-recovery contract changed")
    BireAlignedArchitecture(**architecture)
    _assert_objective_description_is_consistent(contract)
    _assert_single_change(contract)
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise BireAlignedLossRecoveryError(
                    f"Bire-aligned loss-recovery source changed: {source}"
                )
    return contract, resolved, _file_sha256(resolved)


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify sources, the single change, and the restored objective."""

    contract, resolved, digest = load_contract(contract_path)
    dataset = Path(contract["sources"]["dataset"]["path"]).resolve()
    _, attribution_contract = _verify_artifacts(contract, dataset)
    loss_config = model_c_loss_config("v1")
    if loss_contract_sha256(loss_config) != MODEL_C_LOSS_V1_CONTRACT_SHA256:
        raise BireAlignedLossRecoveryError("loss-v1 changed")
    group = zarr.open_consolidated(str(dataset), mode="r")
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    records = attribution.training_records(attribution_contract, split)
    architecture = BireAlignedArchitecture(**contract["architecture"])
    model = build_bire_aligned_model(architecture)
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "objective": "incumbent_group_balanced_model_c_loss_v1",
        "loss_contract_sha256": MODEL_C_LOSS_V1_CONTRACT_SHA256,
        "rollout_steps": loss_config.rollout_steps,
        "group_weighting": "U:V:Theta:eta equally at one quarter each",
        "architecture_unchanged_from_parent": True,
        "fno_blocks": architecture.n_layers,
        "layer_norm_modules": len(model.fno.fno_blocks.norm),
        "external_local_branch": False,
        "parameter_count": int(sum(v.numel() for v in model.parameters())),
        "selection_records": int(records.shape[0]),
        "inference_state_opened": False,
        "held_s0_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }


def run(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train the Bire-aligned map under the incumbent objective and select."""

    if torch is None or DataLoader is None:  # pragma: no cover
        raise RuntimeError("the loss-recovery control requires PyTorch")
    require_model_a_runtime()
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    dataset = Path(contract["sources"]["dataset"]["path"]).resolve()
    normalization, attribution_contract = _verify_artifacts(contract, dataset)
    scratch = Path(contract["output"]["scratch_root"]).resolve()
    project = Path(contract["output"]["project_root"]).resolve()
    scratch_tmp = scratch.with_name(scratch.name + ".tmp")
    project_tmp = project.with_name(project.name + ".tmp")
    if any(p.exists() for p in (scratch, project, scratch_tmp, project_tmp)):
        raise FileExistsError("refusing to overwrite loss-recovery output")

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

    loss_config = model_c_loss_config("v1")
    if loss_contract_sha256(loss_config) != MODEL_C_LOSS_V1_CONTRACT_SHA256:
        raise BireAlignedLossRecoveryError("loss-v1 changed")

    training_records = records_for_rollout_split(pair_codes, 1, rollout_steps=3)
    training_dataset = ModelCAnomalyRolloutDataset(
        dataset,
        training_records,
        point_mean,
        point_scale,
        rollout_steps=loss_config.rollout_steps,
    )
    batch_size = int(training["batch_size"])
    loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            training_dataset,
            batch_size,
            int(training["seed"]),
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    architecture = BireAlignedArchitecture(**contract["architecture"])
    model = build_bire_aligned_model(architecture).to(device)
    parameter_count = int(sum(v.numel() for v in model.parameters()))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["initial_learning_rate"]),
        betas=tuple(float(v) for v in training["adam_betas"]),
        weight_decay=float(training["weight_decay"]),
    )
    wet = torch.from_numpy(wet_array.astype(np.float32))[None, None].to(device)
    boundary = torch.from_numpy(
        western_boundary_mask(wet_array, loss_config.western_boundary_width).astype(
            np.float32
        )
    )[None, None].to(device)
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
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    def _diverged(step: int, reason: str) -> None:
        (project_tmp / DIVERGENCE_NAME).write_text(
            json.dumps(
                {
                    "status": "diverged",
                    "version": VERSION,
                    "contract_sha256": contract_sha,
                    "reason": reason,
                    "optimizer_step": int(step),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "gradient_clipping": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        shutil.rmtree(scratch_tmp, ignore_errors=True)
        os.replace(project_tmp, project)
        raise BireAlignedDivergenceError(f"{reason} at optimizer step {step}")

    for step in range(1, maximum_steps + 1):
        if step == decay_step + 1:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= float(training["decay_factor"])
        try:
            raw_features, futures = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_features, futures = next(iterator)
        raw_features = raw_features.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
        features = retained_features(raw_features)
        model.train()
        predictions = direct_state_unroll(
            model, features, wet, loss_config.rollout_steps
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
        if not all(bool(torch.isfinite(terms[n]).item()) for n in AUDIT_TERMS):
            _diverged(step, "training objective became non-finite")
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        if not all(
            bool(torch.isfinite(p.grad).all().item())
            for p in model.parameters()
            if p.grad is not None
        ):
            _diverged(step, "training gradients became non-finite")
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
        checkpoint_path = (
            checkpoint_directory / f"model_c_bire_loss_recovery_step_{step:05d}.pt"
        )
        torch.save(
            {
                "version": VERSION,
                "optimizer_step": step,
                "fine_tune_step": step,
                "architecture": architecture.to_dict(),
                "contract": str(resolved_contract),
                "contract_sha256": contract_sha,
                "base_loss_contract": loss_contract(loss_config),
                "base_loss_contract_sha256": MODEL_C_LOSS_V1_CONTRACT_SHA256,
                "rollout_steps": loss_config.rollout_steps,
                "training_history_record": history_record,
                "model_state_dict": _checkpoint_state_dict(model),
            },
            checkpoint_path,
        )
        checkpoints.append(
            {
                "optimizer_step": step,
                "fine_tune_step": step,
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": _file_sha256(checkpoint_path),
            }
        )
        totals = {name: 0.0 for name in AUDIT_TERMS}
        samples = 0

    if len(checkpoints) != len(CHECKPOINT_STEPS):
        raise BireAlignedLossRecoveryError("not every declared checkpoint was written")
    del model, optimizer, loader, training_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    records = attribution.training_records(attribution_contract, split)
    common = {
        "device": device,
        "initial": attribution.base._gather_states(state, records, 0),
        "raw_static": attribution.base._gather_static(static, records),
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
    source_primary = source_summary["primary_10_to_90_rmse_ratio_to_persistence"]
    evaluated = []
    summaries = []
    for record in checkpoints:
        value = _evaluate_checkpoint(
            checkpoint_directory / record["checkpoint"],
            int(record["optimizer_step"]),
            normalization,
            **common,
        )
        evaluated.append(value)
        summary = summarize_evaluation(
            value,
            source_primary_ratios=source_primary,
            selection=contract["selection"],
        )
        summary["fine_tune_step"] = int(record["optimizer_step"])
        summary["stage_id"] = f"step_{record['optimizer_step']}"
        summaries.append(summary)

    decision = select_arm_checkpoint(summaries)
    selected_step = int(decision["selected_optimizer_step"])
    selected_name = next(
        r["checkpoint"] for r in checkpoints if r["optimizer_step"] == selected_step
    )
    shutil.copy2(checkpoint_directory / selected_name, scratch_tmp / "selected.pt")
    published = {
        "optimizer_step": selected_step,
        "checkpoint": str(scratch / "selected.pt"),
        "checkpoint_sha256": _file_sha256(scratch_tmp / "selected.pt"),
    }

    arrays_path = scratch_tmp / ARRAYS_NAME
    np.savez_compressed(
        arrays_path,
        optimizer_steps=np.asarray(
            [s["optimizer_step"] for s in summaries], dtype=np.int32
        ),
        lead_days=np.arange(10, 361, 10, dtype=np.int16),
        frozen_median_modewise_ratio=np.stack(
            [v["ratio"] for v in evaluated]
        ).astype(np.float32),
        integrated_energy_ratio=np.stack(
            [v["integrated"] for v in evaluated]
        ).astype(np.float32),
        primary_model_rmse=np.stack(
            [v["model_rmse"] for v in evaluated]
        ).astype(np.float32),
        primary_persistence_rmse=np.asarray(
            source["persistence_rmse"], dtype=np.float32
        ),
        source_frozen_median_modewise_ratio=np.asarray(
            source["ratio"], dtype=np.float32
        ),
        selection_records=records.astype(np.int32),
    )
    report = {
        "status": "complete",
        "version": VERSION,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "architecture": architecture.to_dict(),
        "parameter_count": parameter_count,
        "changed_from_parent": {
            "objective": {
                "was": "wet_cell_mse_plus_0p01_mae",
                "now": "incumbent_group_balanced_model_c_loss_v1",
            },
            "rollout_exposure": {
                "was": "one_step_pretraining_then_two_step_fine_tuning",
                "now": "three_step_unrolled_throughout",
            },
        },
        "loss": contract["loss"],
        "base_loss_contract": loss_contract(loss_config),
        "base_loss_contract_sha256": MODEL_C_LOSS_V1_CONTRACT_SHA256,
        "group_weighting": "U:V:Theta:eta equally at one quarter each",
        "optimizer": {
            "name": "adam",
            "initial_learning_rate": float(training["initial_learning_rate"]),
            "adam_betas": [float(v) for v in training["adam_betas"]],
            "weight_decay": float(training["weight_decay"]),
            "batch_size": batch_size,
            "gradient_clipping": False,
            "decay_step": decay_step,
            "decay_factor": float(training["decay_factor"]),
        },
        "training_history": history,
        "checkpoints": checkpoints,
        "selection_decision": decision,
        "published_checkpoint": published,
        "source_summary": source_summary,
        "evaluation_summaries": summaries,
        "arrays": str(scratch / ARRAYS_NAME),
        "arrays_sha256": _file_sha256(arrays_path),
        "read_contract": contract["read_contract"],
        "validation_state_opened": False,
        "inference_state_opened": False,
        "held_s0_state_opened": False,
        "response_or_adjoint_state_opened": False,
        "elapsed_seconds": time.monotonic() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    report["content_sha256"] = _json_sha256(report)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (scratch_tmp / REPORT_NAME).write_text(rendered)
    (project_tmp / REPORT_NAME).write_text(rendered)
    shutil.copy2(arrays_path, project_tmp / ARRAYS_NAME)
    _plot(
        project_tmp / FIGURE_NAME,
        [{**source_summary, "fine_tune_step": 0, "stage_id": "source"}, *summaries],
    )
    (project_tmp / README_NAME).write_text(_readme(report))
    artifacts = {
        name: _file_sha256(project_tmp / name)
        for name in (REPORT_NAME, ARRAYS_NAME, FIGURE_NAME, README_NAME)
    }
    (project_tmp / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "status": "complete",
                "version": VERSION,
                "contract_sha256": contract_sha,
                "artifacts": artifacts,
                "content_sha256": _json_sha256(artifacts),
                "inference_state_opened": False,
                "held_s0_state_opened": False,
                "response_or_adjoint_state_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(scratch_tmp, scratch)
    os.replace(project_tmp, project)
    return report


def _readme(report: Mapping[str, Any]) -> str:
    lines = [
        "# Bire-aligned architecture with the incumbent Model C objective",
        "",
        "Architecture-fixed loss-recovery control.  Everything architectural and",
        "optimizer-side is frozen against",
        "`model_c_bire_aligned_full_state_lr5e4_v1`; only the objective and the",
        "rollout exposure change.",
        "",
        "| quantity | 5e-4 Bire arm | this arm |",
        "| --- | --- | --- |",
        "| objective | wet-cell `MSE + 0.01 MAE` over 46 channels | **group-balanced Model C loss v1** |",
        "| group weighting | `U:V:Theta:eta = 15:15:15:1` | **equal quarters** |",
        "| rollout exposure | 1-step pretrain then 2-step fine-tune | **3-step unrolled throughout** |",
        "",
        "Three FNO blocks, six pointwise LayerNorms, Bire PosEmbed, no external",
        "3x3 branch, 24x16 modes, width 128, 10% padding, Adam(5e-4, betas",
        "0.9/0.95, weight decay 0), batch size 8, and the 7,680-step budget are",
        "all unchanged, so the optimizer-step and sequence-exposure budgets stay",
        "comparable with every Bire-aligned arm.",
        "",
        "It answers one question: is the bounded behaviour coming from the",
        "Bire-aligned architecture while the loss of skill comes from the Bire",
        "objective?",
        "",
        "Gate instrument (unchanged 360-day split-1 spectral/primary summary):",
        "",
        "| step | worst 10--90-day RMSE ratio | worst mid/bottom modewise | gate |",
        "| --- | --- | --- | --- |",
    ]
    for value in report["evaluation_summaries"]:
        lines.append(
            "| {step} | {primary:.4f} | {spectral:.3f} | {gate} |".format(
                step=value["optimizer_step"],
                primary=value["worst_primary_10_to_90_ratio"],
                spectral=value["worst_mid_bottom_modewise_ratio_all_leads"],
                gate=value["gate"]["pass"],
            )
        )
    lines += [
        "",
        f"Selected: step {report['published_checkpoint']['optimizer_step']} "
        f"(`{report['selection_decision']['status']}`).",
        "",
        "Training split only; validation, inference, held S0, response, and",
        "adjoint archives remained sealed for this run.",
        "",
        f"Report content SHA-256: `{report['content_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        if command == "run":
            child.add_argument(
                "--device", choices=("auto", "cpu", "cuda"), default="auto"
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.contract)
    else:
        result = run(args.contract, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
