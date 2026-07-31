"""Bire-faithful training protocol for the Bire-aligned full-state FNO.

Three divergences from the public ``oceanfourcast`` implementation survived the
earlier arms.  None was deliberate; each is corrected here, as one bundle, on
top of the working 5e-4 base:

===========================  ==========================  ======================
quantity                     earlier arms                this arm
===========================  ==========================  ======================
MAE weight in the objective  0.01                        **0.05**
learning-rate schedule       step decay, x0.2 at 75%     **CosineAnnealingLR**
                                                         (T_max 3, eta_min 1e-5)
checkpoint selection         fixed optimizer steps       **lowest validation
                                                         loss within each stage**
===========================  ==========================  ======================

``train.py`` computes ``criterion(out, y) + 0.05 * criterion2(out, y)`` in all
four of its training and validation functions, so 0.05 -- not the 0.01 carried
by the earlier contracts -- is the reference weight.  The scheduler is
``CosineAnnealingLR(optimizer, 3, eta_min=1e-5)`` stepped once per epoch, and
the checkpoint retained is the one minimising validation loss over a random 10%
holdout, with the running best reset when fine-tuning begins.

Everything else is frozen against ``model_c_bire_aligned_full_state_lr5e4_v1``:
the architecture, the 49 external inputs, the Bire position fields, the six
pointwise LayerNorms, the absent external 3x3 branch, the two-stage protocol and
its 3,840/3,840 split, seed, batch size 8, Adam betas (0.9, 0.95), zero weight
decay, absent gradient clipping, and lr0 = 5e-4.

ChannelMLP dropout stays at zero.  The repository defaults it to 0.5, but the
paper's hyperparameter table does not specify it, and 0.5 is a strong
regulariser rather than a faithfulness correction -- it belongs to its own arm.

Two declared consequences of adopting validation-based selection:

* a random 10% of the split-1 training records is held out, so this arm trains
  on 90% of the records the earlier arms saw.  The holdout is drawn from
  training-split records only; no validation, inference, or held S0 archive is
  opened.
* "epoch" is declared as a fixed 1,920-optimizer-step period, four of which
  make up the run.  This keeps the optimizer-step and sequence-exposure budget
  identical to the earlier Bire-aligned arms -- so the comparison stays clean on
  that axis -- while giving the scheduler and the validation probe the per-epoch
  cadence they have in the reference implementation.

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

from .af_a0_evaluate import _normalizers
from .af_model_a import (
    ChunkAwareBatchSampler,
    _checkpoint_state_dict,
    require_model_a_runtime,
    seed_everything,
)
from .af_model_b import records_for_rollout_split
from .af_model_c_anomaly_direct import (
    ModelCAnomalyRolloutDataset,
    direct_state_unroll,
)
from .af_model_c_anomaly_direct_deep_pressure_spectral_regularization import (
    summarize_evaluation,
)
from .af_model_c_bire_aligned_full_state import (
    ARRAYS_NAME,
    CHECKPOINT_DIRECTORY,
    EXTERNAL_INPUT_CHANNELS,
    LIFTING_INPUT_CHANNELS,
    LOSS_TERMS,
    MANIFEST_NAME,
    MAXIMUM_STEPS,
    README_NAME,
    STAGE_NAMES,
    BireAlignedArchitecture,
    BireAlignedDivergenceError,
    BireAlignedFullStateError,
    _evaluate_checkpoint,
    _json_sha256,
    _plot,
    _source_evaluation,
    _verify_artifacts,
    bire_loss_terms,
    build_bire_aligned_model,
    retained_features,
)
from .af_model_c_overfit import _device, _file_sha256
from .af_model_c_successor import STATE_CHANNEL_COUNT
from . import af_model_c_anomaly_direct_training_spectral_attribution_v2 as attribution

try:
    import torch
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


VERSION = "model_c_bire_aligned_faithful_v1"
CONTRACT_STATUS = "frozen_before_any_bire_aligned_faithful_metric"

MAE_WEIGHT = 0.05
PARENT_MAE_WEIGHT = 0.01
LEARNING_RATE = 5.0e-4
COSINE_T_MAX = 3
COSINE_ETA_MIN = 1.0e-5
EPOCH_STEPS = 1920
EPOCH_COUNT = 4
EPOCH_BOUNDARIES = tuple(EPOCH_STEPS * (index + 1) for index in range(EPOCH_COUNT))
STAGE_LAST_STEP = {"pretrained": 3840, "finetuned": MAXIMUM_STEPS}
VALIDATION_FRACTION = 0.1
VALIDATION_SPLIT_SEED = 20260730

REPORT_NAME = "bire_aligned_faithful_report.json"
FIGURE_NAME = "model_c_bire_aligned_faithful_selection.png"


class BireAlignedFaithfulError(BireAlignedFullStateError):
    """Raised when the Bire-faithful protocol arm violates its contract."""


def epoch_of_step(step: int) -> int:
    """One-based scheduler/validation period containing ``step``."""

    if not 1 <= step <= MAXIMUM_STEPS:
        raise ValueError("optimizer step is outside the declared budget")
    return (step - 1) // EPOCH_STEPS + 1


def stage_of_step(step: int) -> str:
    """``pretrained`` for the one-step half, ``finetuned`` for the two-step half."""

    return "pretrained" if step <= STAGE_LAST_STEP["pretrained"] else "finetuned"


def autoregressive_steps(stage: str) -> int:
    if stage not in STAGE_NAMES:
        raise ValueError(f"unknown stage: {stage}")
    return 1 if stage == "pretrained" else 2


def split_validation_records(
    records: Sequence[tuple[int, int]],
    *,
    fraction: float = VALIDATION_FRACTION,
    seed: int = VALIDATION_SPLIT_SEED,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Hold out a seeded random fraction, as ``random_split`` does upstream.

    The holdout is drawn from split-1 training records only, so no sealed
    archive is touched.  Training order is preserved for the retained records so
    that the chunk-aware sampler still reads contiguous Zarr blocks.
    """

    total = len(records)
    count = int(round(total * float(fraction)))
    if not 0 < count < total:
        raise BireAlignedFaithfulError("validation holdout must be a strict subset")
    order = np.random.default_rng(int(seed)).permutation(total)
    held = set(int(index) for index in order[:count])
    validation = tuple(records[index] for index in sorted(held))
    training = tuple(
        record for index, record in enumerate(records) if index not in held
    )
    if len(validation) + len(training) != total:
        raise BireAlignedFaithfulError("validation holdout lost records")
    return training, validation


def _records_sha256(records: Sequence[tuple[int, int]]) -> str:
    return _json_sha256([[int(a), int(b)] for a, b in records])


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the contract frozen before any Bire-faithful metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    architecture = contract.get("architecture", {})
    training = contract.get("training", {})
    loss = contract.get("loss", {})
    schedule = contract.get("learning_rate_schedule", {})
    selection = contract.get("checkpoint_selection", {})
    read = contract.get("read_contract", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or int(architecture.get("in_channels", -1)) != EXTERNAL_INPUT_CHANNELS
        or int(architecture.get("lifting_in_channels", -1))
        != LIFTING_INPUT_CHANNELS
        or int(architecture.get("out_channels", -1)) != STATE_CHANNEL_COUNT
        or int(architecture.get("n_layers", -1)) != 3
        or float(architecture.get("channel_mlp_dropout", -1.0)) != 0.0
        or architecture.get("local_kernel_size") is not None
        or architecture.get("positional_embedding") is not None
        or int(training.get("seed", -1)) != 20260724
        or int(training.get("batch_size", -1)) != 8
        or float(training.get("initial_learning_rate", -1.0)) != LEARNING_RATE
        or tuple(float(value) for value in training.get("adam_betas", ()))
        != (0.9, 0.95)
        or float(training.get("weight_decay", -1.0)) != 0.0
        or training.get("optimizer") != "adam"
        or training.get("gradient_clipping") is not False
        or int(training.get("maximum_steps", -1)) != MAXIMUM_STEPS
        or int(training.get("epoch_steps", -1)) != EPOCH_STEPS
        or tuple(training.get("epoch_boundaries", ())) != EPOCH_BOUNDARIES
        or loss.get("objective") != "wet_cell_mse_plus_0p05_mae"
        or float(loss.get("mae_weight", -1.0)) != MAE_WEIGHT
        or schedule.get("kind") != "cosine_annealing"
        or int(schedule.get("t_max", -1)) != COSINE_T_MAX
        or float(schedule.get("eta_min", -1.0)) != COSINE_ETA_MIN
        or selection.get("rule") != "lowest_validation_loss_within_each_stage"
        or float(selection.get("validation_fraction", -1.0)) != VALIDATION_FRACTION
        or int(selection.get("validation_split_seed", -1)) != VALIDATION_SPLIT_SEED
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
        raise BireAlignedFaithfulError("Bire-faithful contract changed")
    BireAlignedArchitecture(**architecture)
    _assert_bundle_against_parent(contract)
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise BireAlignedFaithfulError(
                    f"Bire-faithful source changed: {source}"
                )
    return contract, resolved, _file_sha256(resolved)


#: Training fields that must still match the 5e-4 base arm.  ``decay_fraction``
#: and ``decay_factor`` are absent because the step schedule they parameterised
#: is replaced wholesale by the cosine schedule.
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


def _assert_bundle_against_parent(contract: Mapping[str, Any]) -> None:
    """Fail unless only the three declared quantities moved from the 5e-4 arm."""

    record = contract["sources"]["parent_contract"]
    path = Path(record["path"]).resolve()
    if not path.is_file() or _file_sha256(path) != record["sha256"]:
        raise BireAlignedFaithfulError("the parent 5e-4 contract changed")
    parent = json.loads(path.read_text())
    if (
        parent.get("version") != "model_c_bire_aligned_full_state_lr5e4_v1"
        or float(parent["training"]["initial_learning_rate"]) != LEARNING_RATE
        or float(parent["loss"]["mae_weight"]) != PARENT_MAE_WEIGHT
    ):
        raise BireAlignedFaithfulError("the parent arm is not the 5e-4 control")
    if contract["architecture"] != parent["architecture"]:
        raise BireAlignedFaithfulError(
            "the faithfulness bundle must keep the parent architecture exactly"
        )
    if contract["stages"] != parent["stages"]:
        raise BireAlignedFaithfulError(
            "the faithfulness bundle must keep the parent two-stage protocol"
        )
    if contract["selection"] != parent["selection"]:
        raise BireAlignedFaithfulError(
            "the faithfulness bundle must keep the parent gate instrument"
        )
    for field in FROZEN_TRAINING_FIELDS:
        if contract["training"].get(field) != parent["training"].get(field):
            raise BireAlignedFaithfulError(
                f"the faithfulness bundle moved a fourth quantity: {field}"
            )
    if float(contract["loss"]["mae_weight"]) == PARENT_MAE_WEIGHT:
        raise BireAlignedFaithfulError("the MAE weight correction is missing")


def _stage_loss(
    model: Any,
    features: Any,
    futures: Any,
    wet: Any,
    steps: int,
    mae_weight: float,
) -> dict[str, Any]:
    """Summed ``MSE + w MAE`` over the stage's autoregressive steps."""

    predictions = direct_state_unroll(model, features, wet, steps)
    accumulated: dict[str, Any] = {name: None for name in LOSS_TERMS}
    for index in range(steps):
        terms = bire_loss_terms(
            predictions[:, index],
            futures[:, index],
            wet,
            mae_weight=mae_weight,
        )
        for name in LOSS_TERMS:
            accumulated[name] = (
                terms[name]
                if accumulated[name] is None
                else accumulated[name] + terms[name]
            )
    return accumulated


def _validation_loss(
    model: Any,
    loader: Any,
    wet: Any,
    device: Any,
    steps: int,
    mae_weight: float,
) -> dict[str, float]:
    """Mean per-sample validation loss under the active stage's objective."""

    totals = {name: 0.0 for name in LOSS_TERMS}
    samples = 0
    model.eval()
    with torch.no_grad():
        for raw_features, futures in loader:
            raw_features = raw_features.to(device=device, dtype=torch.float32)
            futures = futures.to(device=device, dtype=torch.float32)
            features = retained_features(raw_features)
            accumulated = _stage_loss(
                model,
                features,
                futures,
                wet,
                steps,
                mae_weight,
            )
            batch = int(features.shape[0])
            for name in LOSS_TERMS:
                totals[name] += float(accumulated[name].detach().cpu()) * batch
            samples += batch
    model.train()
    return {name: totals[name] / samples for name in LOSS_TERMS}


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify sources, the bundle, and the record split without training."""

    contract, resolved, digest = load_contract(contract_path)
    dataset = Path(contract["sources"]["dataset"]["path"]).resolve()
    _, attribution_contract = _verify_artifacts(contract, dataset)
    group = zarr.open_consolidated(str(dataset), mode="r")
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    records = records_for_rollout_split(pair_codes, 1, rollout_steps=3)
    training, validation = split_validation_records(records)
    selection_records = attribution.training_records(attribution_contract, split)
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "mae_weight": MAE_WEIGHT,
        "parent_mae_weight": PARENT_MAE_WEIGHT,
        "learning_rate_schedule": "cosine_annealing",
        "checkpoint_selection": "lowest_validation_loss_within_each_stage",
        "channel_mlp_dropout": 0.0,
        "total_records": len(records),
        "training_records": len(training),
        "validation_records": len(validation),
        "validation_records_sha256": _records_sha256(validation),
        "epoch_boundaries": list(EPOCH_BOUNDARIES),
        "gate_selection_records": int(selection_records.shape[0]),
        "inference_state_opened": False,
        "held_s0_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }


def run(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train under the corrected protocol and publish the two selected stages."""

    if torch is None or DataLoader is None:  # pragma: no cover
        raise RuntimeError("the Bire-faithful arm requires PyTorch")
    require_model_a_runtime()
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    dataset = Path(contract["sources"]["dataset"]["path"]).resolve()
    normalization, attribution_contract = _verify_artifacts(contract, dataset)
    scratch = Path(contract["output"]["scratch_root"]).resolve()
    project = Path(contract["output"]["project_root"]).resolve()
    scratch_tmp = scratch.with_name(scratch.name + ".tmp")
    project_tmp = project.with_name(project.name + ".tmp")
    if any(path.exists() for path in (scratch, project, scratch_tmp, project_tmp)):
        raise FileExistsError("refusing to overwrite Bire-faithful output")

    training_contract = contract["training"]
    mae_weight = float(contract["loss"]["mae_weight"])
    seed_everything(int(training_contract["seed"]))
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

    all_records = records_for_rollout_split(pair_codes, 1, rollout_steps=3)
    train_records, validation_records = split_validation_records(all_records)
    batch_size = int(training_contract["batch_size"])
    train_dataset = ModelCAnomalyRolloutDataset(
        dataset,
        train_records,
        point_mean,
        point_scale,
        rollout_steps=2,
    )
    validation_dataset = ModelCAnomalyRolloutDataset(
        dataset,
        validation_records,
        point_mean,
        point_scale,
        rollout_steps=2,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            train_dataset,
            batch_size,
            int(training_contract["seed"]),
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    architecture = BireAlignedArchitecture(**contract["architecture"])
    model = build_bire_aligned_model(architecture).to(device)
    parameter_count = int(sum(value.numel() for value in model.parameters()))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_contract["initial_learning_rate"]),
        betas=tuple(float(value) for value in training_contract["adam_betas"]),
        weight_decay=float(training_contract["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        int(contract["learning_rate_schedule"]["t_max"]),
        eta_min=float(contract["learning_rate_schedule"]["eta_min"]),
    )
    wet = torch.from_numpy(wet_array.astype(np.float32))[None, None].to(device)

    scratch_tmp.parent.mkdir(parents=True, exist_ok=True)
    project_tmp.parent.mkdir(parents=True, exist_ok=True)
    scratch_tmp.mkdir()
    project_tmp.mkdir()
    checkpoint_directory = scratch_tmp / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir()

    iterator = iter(loader)
    totals = {name: 0.0 for name in LOSS_TERMS}
    samples = 0
    history: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []

    def _diverged(step: int, reason: str) -> None:
        record = {
            "status": "diverged",
            "version": VERSION,
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "reason": reason,
            "optimizer_step": int(step),
            "epoch": epoch_of_step(step),
            "stage": stage_of_step(step),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_clipping": False,
        }
        (project_tmp / "bire_aligned_faithful_divergence.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        shutil.rmtree(scratch_tmp, ignore_errors=True)
        os.replace(project_tmp, project)
        raise BireAlignedDivergenceError(
            f"{reason} at optimizer step {step} ({record['stage']} stage)"
        )

    for step in range(1, MAXIMUM_STEPS + 1):
        stage = stage_of_step(step)
        steps_this = autoregressive_steps(stage)
        try:
            raw_features, futures = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_features, futures = next(iterator)
        raw_features = raw_features.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
        features = retained_features(raw_features)
        model.train()
        accumulated = _stage_loss(
            model,
            features,
            futures,
            wet,
            steps_this,
            mae_weight,
        )
        if not all(
            bool(torch.isfinite(accumulated[name]).item()) for name in LOSS_TERMS
        ):
            _diverged(step, "training objective became non-finite")
        optimizer.zero_grad(set_to_none=True)
        accumulated["total"].backward()
        if not all(
            bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        ):
            _diverged(step, "training gradients became non-finite")
        optimizer.step()

        batch = int(features.shape[0])
        for name in LOSS_TERMS:
            totals[name] += float(accumulated[name].detach().cpu()) * batch
        samples += batch
        if step not in EPOCH_BOUNDARIES:
            continue

        epoch = epoch_of_step(step)
        window = {name: totals[name] / samples for name in LOSS_TERMS}
        learning_rate = float(optimizer.param_groups[0]["lr"])
        validation = _validation_loss(
            model,
            validation_loader,
            wet,
            device,
            steps_this,
            mae_weight,
        )
        history_record = {
            "epoch": epoch,
            "optimizer_step": step,
            "stage_id": stage,
            "autoregressive_steps": steps_this,
            "learning_rate": learning_rate,
            "training_window": window,
            "validation": validation,
        }
        history.append(history_record)
        checkpoint_path = checkpoint_directory / f"model_c_bire_faithful_epoch_{epoch}.pt"
        torch.save(
            {
                "version": VERSION,
                "stage_id": stage,
                "epoch": epoch,
                "autoregressive_steps": steps_this,
                "optimizer_step": step,
                "fine_tune_step": step,
                "architecture": architecture.to_dict(),
                "contract": str(resolved_contract),
                "contract_sha256": contract_sha,
                "loss": contract["loss"],
                "learning_rate_schedule": contract["learning_rate_schedule"],
                "training_history_record": history_record,
                "model_state_dict": _checkpoint_state_dict(model),
            },
            checkpoint_path,
        )
        epoch_records.append(
            {
                "epoch": epoch,
                "optimizer_step": step,
                "stage_id": stage,
                "learning_rate": learning_rate,
                "validation_total": float(validation["total"]),
                "checkpoint": checkpoint_path.name,
            }
        )
        # Stepped once per epoch, as in the reference implementation.
        scheduler.step()
        totals = {name: 0.0 for name in LOSS_TERMS}
        samples = 0

    # Bire resets the running best when fine-tuning begins, so each stage keeps
    # its own lowest-validation-loss checkpoint.
    selection: dict[str, Any] = {}
    for stage in STAGE_NAMES:
        candidates = [r for r in epoch_records if r["stage_id"] == stage]
        if not candidates:
            raise BireAlignedFaithfulError(f"no checkpoint recorded for {stage}")
        best = min(candidates, key=lambda r: (r["validation_total"], r["epoch"]))
        selection[stage] = {
            "selected_epoch": best["epoch"],
            "selected_optimizer_step": best["optimizer_step"],
            "validation_total": best["validation_total"],
            "candidates": [
                {
                    "epoch": r["epoch"],
                    "optimizer_step": r["optimizer_step"],
                    "validation_total": r["validation_total"],
                }
                for r in candidates
            ],
        }

    del model, optimizer, loader, train_dataset, validation_loader, validation_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    published: dict[str, Any] = {}
    for stage in STAGE_NAMES:
        best = selection[stage]
        source_name = next(
            r["checkpoint"]
            for r in epoch_records
            if r["epoch"] == best["selected_epoch"]
        )
        destination = scratch_tmp / f"{stage}.pt"
        shutil.copy2(checkpoint_directory / source_name, destination)
        published[stage] = {
            "optimizer_step": int(best["selected_optimizer_step"]),
            "epoch": int(best["selected_epoch"]),
            "checkpoint": str(scratch / f"{stage}.pt"),
            "checkpoint_sha256": _file_sha256(destination),
        }

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
    for stage in STAGE_NAMES:
        value = _evaluate_checkpoint(
            scratch_tmp / f"{stage}.pt",
            int(published[stage]["optimizer_step"]),
            normalization,
            **common,
        )
        evaluated.append(value)
        summary = summarize_evaluation(
            value,
            source_primary_ratios=source_primary,
            selection=contract["selection"],
        )
        summary["fine_tune_step"] = int(published[stage]["optimizer_step"])
        summary["stage_id"] = stage
        summaries.append(summary)

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
        validation_records=np.asarray(validation_records, dtype=np.int32),
    )
    report = {
        "status": "complete",
        "version": VERSION,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "architecture": architecture.to_dict(),
        "parameter_count": parameter_count,
        "corrected_divergences": {
            "mae_weight": {"was": PARENT_MAE_WEIGHT, "now": MAE_WEIGHT},
            "learning_rate_schedule": {
                "was": "step_decay_0p2_at_75_percent",
                "now": "cosine_annealing_t_max_3_eta_min_1e_5",
            },
            "checkpoint_selection": {
                "was": "fixed_optimizer_steps",
                "now": "lowest_validation_loss_within_each_stage",
            },
        },
        "channel_mlp_dropout": 0.0,
        "dropout_note": (
            "left at zero deliberately; the repository default of 0.5 is a "
            "regulariser the paper does not specify and belongs to its own arm"
        ),
        "loss": contract["loss"],
        "learning_rate_schedule": contract["learning_rate_schedule"],
        "optimizer": {
            "name": "adam",
            "initial_learning_rate": float(
                training_contract["initial_learning_rate"]
            ),
            "adam_betas": [
                float(v) for v in training_contract["adam_betas"]
            ],
            "weight_decay": float(training_contract["weight_decay"]),
            "batch_size": batch_size,
            "gradient_clipping": False,
        },
        "record_split": {
            "total_records": len(all_records),
            "training_records": len(train_records),
            "validation_records": len(validation_records),
            "validation_fraction": VALIDATION_FRACTION,
            "validation_split_seed": VALIDATION_SPLIT_SEED,
            "validation_records_sha256": _records_sha256(validation_records),
            "drawn_from": "split_1_training_records_only",
        },
        "epoch_definition": {
            "epoch_steps": EPOCH_STEPS,
            "epoch_count": EPOCH_COUNT,
            "epoch_boundaries": list(EPOCH_BOUNDARIES),
            "note": (
                "a fixed 1920-step period, so the optimizer-step and sequence "
                "exposure budget matches the earlier Bire-aligned arms exactly"
            ),
        },
        "training_history": history,
        "epoch_records": epoch_records,
        "checkpoint_selection": selection,
        "published_checkpoints": published,
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
    manifest = {
        "status": "complete",
        "version": VERSION,
        "contract_sha256": contract_sha,
        "artifacts": artifacts,
        "content_sha256": _json_sha256(artifacts),
        "inference_state_opened": False,
        "held_s0_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }
    (project_tmp / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(scratch_tmp, scratch)
    os.replace(project_tmp, project)
    return report


def _readme(report: Mapping[str, Any]) -> str:
    lines = [
        "# Bire-faithful protocol arm (training split only)",
        "",
        "Three unintended divergences from the public `oceanfourcast`",
        "implementation, corrected as one bundle on the working 5e-4 base:",
        "",
        "| quantity | earlier arms | this arm |",
        "| --- | --- | --- |",
        f"| MAE weight | {PARENT_MAE_WEIGHT} | **{MAE_WEIGHT}** |",
        "| LR schedule | step decay x0.2 at 75% | **cosine, T_max 3, eta_min 1e-5** |",
        "| checkpoint selection | fixed steps | **lowest validation loss per stage** |",
        "",
        "Architecture, inputs, positional encoder, two-stage protocol, seed,",
        "batch size 8, betas, weight decay, absent gradient clipping, and",
        "lr0 = 5e-4 are frozen against `model_c_bire_aligned_full_state_lr5e4_v1`.",
        "ChannelMLP dropout stays at zero.",
        "",
        "Validation is a seeded random 10% of the split-1 training records; no",
        "sealed archive is opened. Epoch = a fixed 1,920-step period, so the",
        "optimizer-step and sequence-exposure budget matches the earlier arms.",
        "",
        "Per-epoch validation loss and selection:",
        "",
        "| epoch | step | stage | lr | train total | valid total |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in report["training_history"]:
        lines.append(
            "| {epoch} | {step} | {stage} | {lr:.3g} | {tr:.5f} | {va:.5f} |".format(
                epoch=record["epoch"],
                step=record["optimizer_step"],
                stage=record["stage_id"],
                lr=record["learning_rate"],
                tr=record["training_window"]["total"],
                va=record["validation"]["total"],
            )
        )
    lines += ["", "Selected per stage:", ""]
    for stage, record in report["checkpoint_selection"].items():
        lines.append(
            f"* `{stage}` -- epoch {record['selected_epoch']}, "
            f"step {record['selected_optimizer_step']}, "
            f"validation {record['validation_total']:.5f}"
        )
    lines += [
        "",
        "Gate instrument (unchanged 360-day split-1 spectral/primary summary):",
        "",
        "| stage | step | worst 10--90-day RMSE ratio | worst mid/bottom modewise | gate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for value in report["evaluation_summaries"]:
        lines.append(
            "| {stage} | {step} | {primary:.4f} | {spectral:.3f} | {gate} |".format(
                stage=value["stage_id"],
                step=value["optimizer_step"],
                primary=value["worst_primary_10_to_90_ratio"],
                spectral=value["worst_mid_bottom_modewise_ratio_all_leads"],
                gate=value["gate"]["pass"],
            )
        )
    lines += [
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
                "--device",
                choices=("auto", "cpu", "cuda"),
                default="auto",
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
