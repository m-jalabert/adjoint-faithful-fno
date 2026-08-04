"""Loss-recovery model under the Bire et al. Section 3.2 data arrangement.

Same model as :mod:`af_model_c_bire_aligned_v3` --- three FNO blocks, six
pointwise LayerNorms, modes (24,16), width 128, Bire positional encoding, 10%
padding, no external local branch, Model C loss v1 over a three-step rollout,
Adam 5e-4 decaying to 1e-4 at 75%, batch 8, 7,680 steps, from scratch on seed
20260724, pooled over S0+S1+S2.

Only the data arrangement changes, to the paper's own::

    train       0--5999      6000 d   "final 6000 timesteps"
    validation  6000--7199   1200 d   "the next 1200"
    inference   6200--7199   1000 d   "the final 1000", nested in validation

Training grows from 5,040 to 6,000 days per regime (17,910 pooled three-step
records against 15,030), and the buffers are dropped because the paper has none
--- leakage is prevented instead by requiring a training rollout's whole target
sequence to stay inside the training block, so the latest start is 5969.

Days 7200--8999 lie beyond Bire's 7,200-day record.  They are never trained,
validated, or used as an inference start; they exist only so the 2,000-day
inference rollouts keep lead-matched truth, which the paper could not have.

Training and validation only.  The inference set opens through the figure
contract.
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
from .af_bire_protocol_split import (
    INFERENCE_RANGE,
    STORE_DAYS,
    TRAIN_RANGE,
    VALIDATION_RANGE,
    split_codes,
)
from .af_bire_protocol_split import store_codes
from .af_bire_protocol_split import validation_records as protocol_validation_records
from .af_bire_protocol_split import validation_starts as protocol_validation_starts
from .af_bire_protocol_split import verify as verify_split
from .af_data_v3 import DATASET_VERSION, EXPERIMENTS, PRODUCTION_DAYS
from .af_data_v3 import split_codes as v3_split_codes
from .af_forward_complete import derived_fields
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
    training_pointwise_normalizers,
)
from .af_model_c_bire_aligned_chronological import (
    ARRAYS_NAME,
    FIGURE_NAME,
    PRIMARY_FIELDS,
    REPORT_NAME,
    _plot,
    select_by_validation,
    validate_checkpoint,
)
from .af_model_c_bire_aligned_full_state import (
    CHECKPOINT_DIRECTORY,
    EXTERNAL_INPUT_CHANNELS,
    LIFTING_INPUT_CHANNELS,
    MANIFEST_NAME,
    MAXIMUM_STEPS,
    README_NAME,
    BireAlignedArchitecture,
    BireAlignedDivergenceError,
    BireAlignedFullStateError,
    BireAlignedStepper,
    _json_sha256,
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


VERSION = "model_c_bire_protocol_pooled_v1"
CONTRACT_STATUS = "frozen_before_any_bire_protocol_metric"
PARENT_VERSION = "model_c_bire_aligned_v3_pooled_v1"

LEARNING_RATE = 5.0e-4
ROLLOUT_STEPS = 3
CHECKPOINT_STEPS = (1920, 3840, 5760, 7680)
VALIDATION_ROLLOUT_DAYS = 360
VALIDATION_START_STRIDE = 30

NORMALIZATION_NAME = "model_c_bire_protocol_train_only_normalization.npz"
DIVERGENCE_NAME = "bire_protocol_divergence.json"

FROZEN_TRAINING_FIELDS = (
    "seed",
    "optimizer",
    "batch_size",
    "adam_betas",
    "weight_decay",
    "gradient_clipping",
    "maximum_steps",
    "initial_learning_rate",
    "decay_fraction",
    "decay_factor",
    "rollout_steps",
)


class BireProtocolArmError(BireAlignedFullStateError):
    """Raised when the Bire-protocol pooled arm violates its contract."""


validation_starts = protocol_validation_starts
validation_records = protocol_validation_records


def _assert_store_is_v3(group: Any) -> None:
    """Confirm the store is trajectory-v3, then override its split in memory.

    The v3 store carries the project's own three-block split (train 0--5039,
    validation 5130--6389, test 6480--8999).  This arm deliberately applies the
    Bire Section 3.2 arrangement instead, so the stored arrays must *not* match
    the declaration and cannot be used as the check.  What is checked is that we
    are on the expected store: the v3 three-block codes, in the expected shape.
    The Bire codes are then computed in memory; the store is never modified.
    """

    if str(group.attrs.get("version")) != DATASET_VERSION:
        raise BireProtocolArmError("the store is not trajectory-v3")
    stored = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    if stored.size != PRODUCTION_DAYS:
        raise BireProtocolArmError("the v3 store has an unexpected record length")
    expected_v3, _ = v3_split_codes()
    if not np.array_equal(stored, expected_v3):
        raise BireProtocolArmError(
            "the v3 store no longer carries its own three-block split"
        )


def train_only_climatology(
    state: Any,
    wet: np.ndarray,
    *,
    chunk_days: int = 60,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Per-regime pointwise climatology over the Bire training block 0--5999 only."""

    start_index, stop_index = TRAIN_RANGE
    experiments = int(state.shape[0])
    state_sum = np.zeros((experiments, STATE_CHANNEL_COUNT, *wet.shape), dtype=np.float64)
    derived_sum = {
        name: np.zeros((experiments, *wet.shape), dtype=np.float64)
        for name in ("surface_speed", "phihyd_surface", "sst", "streamfunction")
    }
    count = 0
    for experiment in range(experiments):
        count = 0
        for begin in range(start_index, stop_index, chunk_days):
            end = min(begin + chunk_days, stop_index)
            raw = np.asarray(state[experiment, begin:end], dtype=np.float32)
            state_sum[experiment] += raw.sum(axis=0, dtype=np.float64)
            fields = derived_fields(raw, wet)
            for name in derived_sum:
                derived_sum[name][experiment] += fields[name].sum(axis=0, dtype=np.float64)
            count += int(raw.shape[0])
    if count != stop_index - start_index:
        raise BireProtocolArmError("Bire-protocol train-only climatology count changed")
    state_mean = (state_sum / count).astype(np.float32)
    state_mean[:, :, ~wet] = 0.0
    derived_mean = {}
    for name, value in derived_sum.items():
        mean = (value / count).astype(np.float32)
        mean[:, ~wet] = 0.0
        derived_mean[name] = mean
    return state_mean, derived_mean, int(count)


def _assert_model_matches_parent(contract: Mapping[str, Any]) -> None:
    record = contract["sources"]["parent_contract"]
    path = Path(record["path"]).resolve()
    if not path.is_file() or _file_sha256(path) != record["sha256"]:
        raise BireProtocolArmError("the parent loss-recovery contract changed")
    parent = json.loads(path.read_text())
    if parent.get("version") != PARENT_VERSION:
        raise BireProtocolArmError("the parent arm is not the loss-recovery control")
    if contract["architecture"] != parent["architecture"]:
        raise BireProtocolArmError("the Bire-protocol arm must keep the parent architecture exactly")
    if contract["loss"] != parent["loss"]:
        raise BireProtocolArmError("the Bire-protocol arm must keep the parent objective exactly")
    for field in FROZEN_TRAINING_FIELDS:
        if contract["training"].get(field) != parent["training"].get(field):
            raise BireProtocolArmError(f"the Bire-protocol arm moved a model quantity: {field}")


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the contract frozen before any Bire-protocol metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    architecture = contract.get("architecture", {})
    training = contract.get("training", {})
    dataset = contract.get("dataset", {})
    selection = contract.get("checkpoint_selection", {})
    read = contract.get("read_contract", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or dataset.get("version") != DATASET_VERSION
        or int(dataset.get("production_days", -1)) != PRODUCTION_DAYS
        or tuple(dataset.get("train", ())) != TRAIN_RANGE
        or tuple(dataset.get("validation", ())) != VALIDATION_RANGE
        or tuple(dataset.get("inference", ())) != INFERENCE_RANGE
        or dataset.get("pooled_regimes") is not True
        or int(architecture.get("in_channels", -1)) != EXTERNAL_INPUT_CHANNELS
        or int(architecture.get("lifting_in_channels", -1)) != LIFTING_INPUT_CHANNELS
        or int(architecture.get("n_layers", -1)) != 3
        or architecture.get("local_kernel_size") is not None
        or architecture.get("positional_embedding") is not None
        or int(training.get("seed", -1)) != 20260724
        or int(training.get("batch_size", -1)) != 8
        or float(training.get("initial_learning_rate", -1.0)) != LEARNING_RATE
        or int(training.get("maximum_steps", -1)) != MAXIMUM_STEPS
        or tuple(training.get("checkpoint_steps", ())) != CHECKPOINT_STEPS
        or int(training.get("rollout_steps", -1)) != ROLLOUT_STEPS
        or contract.get("loss", {}).get("contract_sha256") != MODEL_C_LOSS_V1_CONTRACT_SHA256
        or contract.get("normalization", {}).get("recomputed_from") != "bire_protocol_train_only_0_5999"
        or selection.get("rule") != "min_worst_long_climatology_ratio_subject_to_short_guard"
        or selection.get("evaluated_on") != "pooled_bire_protocol_validation_6000_6199"
        or read.get("training_state") is not True
        or read.get("validation_state") is not True
        or any(
            read.get(name) is not False
            for name in ("inference_state", "intermediate_wind_state", "response_state", "adjoint_state")
        )
    ):
        raise BireProtocolArmError("Bire-protocol pooled contract changed")
    BireAlignedArchitecture(**architecture)
    _assert_model_matches_parent(contract)
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise BireProtocolArmError(f"Bire-protocol source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def _verify_dataset(contract: Mapping[str, Any]) -> Path:
    record = contract["sources"]["dataset"]
    dataset = Path(record["path"]).resolve()
    if not dataset.is_dir() or _file_sha256(dataset / ".zmetadata") != record["metadata_sha256"]:
        raise BireProtocolArmError("trajectory-v3 dataset source changed")
    return dataset


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify the dataset, the split, and the model without training."""

    contract, resolved, digest = load_contract(contract_path)
    dataset = _verify_dataset(contract)
    group = zarr.open_consolidated(str(dataset), mode="r")
    _assert_store_is_v3(group)
    _, declared_pairs = store_codes()
    records = records_for_rollout_split(declared_pairs, 1, rollout_steps=ROLLOUT_STEPS)
    architecture = BireAlignedArchitecture(**contract["architecture"])
    model = build_bire_aligned_model(architecture)
    winds = [
        float(np.abs(np.asarray(group["static_features"][index, 0])).max())
        for index in range(len(EXPERIMENTS))
    ]
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "dataset_version": str(group.attrs["version"]),
        "split": verify_split(),
        "regime_wind_stress_max_abs": winds,
        "training_rollout_records": len(records),
        "training_starts_per_regime": len(records) // len(EXPERIMENTS),
        "validation_records": int(validation_records().shape[0]),
        "inference_range": list(INFERENCE_RANGE),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    """Train on the pooled Bire-protocol training blocks and select on validation."""

    if torch is None or DataLoader is None:  # pragma: no cover
        raise RuntimeError("the Bire-protocol arm requires PyTorch")
    require_model_a_runtime()
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    dataset = _verify_dataset(contract)
    split_summary = verify_split()
    scratch = Path(contract["output"]["scratch_root"]).resolve()
    project = Path(contract["output"]["project_root"]).resolve()
    scratch_tmp = scratch.with_name(scratch.name + ".tmp")
    project_tmp = project.with_name(project.name + ".tmp")
    if any(p.exists() for p in (scratch, project, scratch_tmp, project_tmp)):
        raise FileExistsError("refusing to overwrite Bire-protocol arm output")

    training = contract["training"]
    seed_everything(int(training["seed"]))
    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    _assert_store_is_v3(group)
    # Store-length codes: identical to the arrangement on 0-7199 and code 0
    # beyond, so the shared readers see the full store while nothing at or past
    # index 7200 is ever selected for training or validation.
    snapshot_split, pair_split = store_codes()
    _, _, wet_array, _, wind_mean, wind_scale = _normalizers(group)
    wet_array = np.asarray(wet_array, dtype=bool)

    normalizers = training_pointwise_normalizers(
        group, snapshot_split, split_code=1, floor_quantile=0.05, absolute_floor=1.0e-6
    )
    point_mean = normalizers["mean"]
    point_scale = normalizers["scale"]
    increment_values = pointwise_increment_scale(group, pair_split, point_scale)
    climatology_state, climatology_derived, climatology_days = train_only_climatology(
        state, wet_array
    )

    loss_config = model_c_loss_config("v1")
    if loss_contract_sha256(loss_config) != MODEL_C_LOSS_V1_CONTRACT_SHA256:
        raise BireProtocolArmError("loss-v1 changed")

    training_records = records_for_rollout_split(
        pair_split, 1, rollout_steps=loss_config.rollout_steps
    )
    training_dataset = ModelCAnomalyRolloutDataset(
        dataset, training_records, point_mean, point_scale,
        rollout_steps=loss_config.rollout_steps,
    )
    batch_size = int(training["batch_size"])
    loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(training_dataset, batch_size, int(training["seed"])),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    architecture = BireAlignedArchitecture(**contract["architecture"])
    model = build_bire_aligned_model(architecture).to(device)
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["initial_learning_rate"]),
        betas=tuple(float(v) for v in training["adam_betas"]),
        weight_decay=float(training["weight_decay"]),
    )
    wet = torch.from_numpy(wet_array.astype(np.float32))[None, None].to(device)
    boundary = torch.from_numpy(
        western_boundary_mask(wet_array, loss_config.western_boundary_width).astype(np.float32)
    )[None, None].to(device)
    increment_scale = torch.from_numpy(increment_values).to(device)
    maximum_steps = int(training["maximum_steps"])
    decay_step = int(round(maximum_steps * float(training["decay_fraction"])))

    scratch_tmp.parent.mkdir(parents=True, exist_ok=True)
    project_tmp.parent.mkdir(parents=True, exist_ok=True)
    scratch_tmp.mkdir()
    project_tmp.mkdir()
    checkpoint_directory = scratch_tmp / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir()
    normalization_path = scratch_tmp / NORMALIZATION_NAME
    np.savez_compressed(
        normalization_path,
        pointwise_mean=point_mean,
        pointwise_raw_scale=normalizers["raw_scale"],
        pointwise_scale=point_scale,
        channel_scale_floor=normalizers["floor"],
    )

    iterator = iter(loader)
    totals = {name: 0.0 for name in AUDIT_TERMS}
    samples = 0
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    def _diverged(step: int, reason: str) -> None:
        (project_tmp / DIVERGENCE_NAME).write_text(
            json.dumps(
                {"status": "diverged", "version": VERSION, "reason": reason,
                 "optimizer_step": int(step),
                 "learning_rate": float(optimizer.param_groups[0]["lr"])},
                indent=2, sort_keys=True) + "\n"
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
        raw_features = raw_features.to(device=device, dtype=torch.float32, non_blocking=True)
        futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
        features = retained_features(raw_features)
        model.train()
        predictions = direct_state_unroll(model, features, wet, loss_config.rollout_steps)
        terms = model_c_loss_terms(
            predictions, futures, features[:, :STATE_CHANNEL_COUNT],
            wet, boundary, increment_scale, loss_config,
        )
        if not all(bool(torch.isfinite(terms[n]).item()) for n in AUDIT_TERMS):
            _diverged(step, "training objective became non-finite")
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        if not all(
            bool(torch.isfinite(p.grad).all().item())
            for p in model.parameters() if p.grad is not None
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
        path = checkpoint_directory / f"model_c_bire_protocol_step_{step:05d}.pt"
        torch.save(
            {
                "version": VERSION,
                "optimizer_step": step,
                "fine_tune_step": step,
                "architecture": architecture.to_dict(),
                "contract": str(resolved_contract),
                "contract_sha256": contract_sha,
                "dataset_version": DATASET_VERSION,
                "base_loss_contract": loss_contract(loss_config),
                "base_loss_contract_sha256": MODEL_C_LOSS_V1_CONTRACT_SHA256,
                "rollout_steps": loss_config.rollout_steps,
                "training_history_record": history_record,
                "model_state_dict": _checkpoint_state_dict(model),
            },
            path,
        )
        checkpoints.append(
            {"optimizer_step": step, "checkpoint": path.name,
             "checkpoint_sha256": _file_sha256(path)}
        )
        totals = {name: 0.0 for name in AUDIT_TERMS}
        samples = 0

    if len(checkpoints) != len(CHECKPOINT_STEPS):
        raise BireProtocolArmError("not every declared checkpoint was written")

    records = validation_records()
    summaries = []
    evaluated_arrays = []
    for record in checkpoints:
        payload = torch.load(
            checkpoint_directory / record["checkpoint"], map_location=device, weights_only=False
        )
        probe = build_bire_aligned_model(architecture).to(device)
        probe.load_state_dict(payload["model_state_dict"])
        probe.eval()
        stepper = BireAlignedStepper(
            model=probe, device=device, wet=wet_array, mean=point_mean, scale=point_scale,
            wind_mean=float(wind_mean), wind_scale=float(wind_scale),
        )
        value = validate_checkpoint(
            stepper, state, static, records, climatology_state, climatology_derived, wet_array
        )
        evaluated_arrays.append(value.pop("arrays"))
        value["optimizer_step"] = int(record["optimizer_step"])
        summaries.append(value)
        del probe, stepper
        if device.type == "cuda":
            torch.cuda.empty_cache()

    decision = select_by_validation(summaries)
    selected_step = int(decision["selected_optimizer_step"])
    selected_name = next(
        r["checkpoint"] for r in checkpoints if r["optimizer_step"] == selected_step
    )
    shutil.copy2(checkpoint_directory / selected_name, scratch_tmp / "selected.pt")
    published = {
        "optimizer_step": selected_step,
        "checkpoint": str(scratch / "selected.pt"),
        "checkpoint_sha256": _file_sha256(scratch_tmp / "selected.pt"),
        "normalization": str(scratch / NORMALIZATION_NAME),
        "normalization_sha256": _file_sha256(normalization_path),
    }

    arrays_path = scratch_tmp / ARRAYS_NAME
    np.savez_compressed(
        arrays_path,
        optimizer_steps=np.asarray([s["optimizer_step"] for s in summaries], dtype=np.int32),
        lead_days=np.asarray(summaries[0]["lead_days"], dtype=np.int16),
        validation_records=records.astype(np.int32),
        snapshot_codes=snapshot_split,
        pair_codes=pair_split,
        **{
            f"validation_rmse__{method}__{field}": np.stack(
                [a[method][field] for a in evaluated_arrays]
            ).astype(np.float32)
            for method in ("model", "persistence", "climatology")
            for field in PRIMARY_FIELDS
        },
    )
    report = {
        "status": "complete",
        "version": VERSION,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": {"path": str(dataset), "version": DATASET_VERSION,
                    "metadata_sha256": _file_sha256(dataset / ".zmetadata")},
        "split": split_summary,
        "architecture": architecture.to_dict(),
        "parameter_count": parameter_count,
        "normalization": {
            "recomputed_from": "bire_protocol_train_only_0_5999",
            "summary": normalizers["summary"],
            "artifact": str(scratch / NORMALIZATION_NAME),
            "artifact_sha256": _file_sha256(normalization_path),
        },
        "climatology": {"source": "per_regime_pointwise_mean_over_bire_protocol_train_only_0_5999",
                        "days_per_regime": climatology_days},
        "increment_scale": increment_values.tolist(),
        "loss": contract["loss"],
        "base_loss_contract_sha256": MODEL_C_LOSS_V1_CONTRACT_SHA256,
        "optimizer": {
            "name": "adam",
            "initial_learning_rate": float(training["initial_learning_rate"]),
            "adam_betas": [float(v) for v in training["adam_betas"]],
            "weight_decay": float(training["weight_decay"]),
            "batch_size": batch_size, "gradient_clipping": False,
            "decay_step": decay_step, "decay_factor": float(training["decay_factor"]),
        },
        "counts": {
            "training_rollout_records": len(training_records),
            "training_starts_per_regime": len(training_records) // len(EXPERIMENTS),
            "validation_records": int(records.shape[0]),
            "validation_starts_per_regime": int(validation_starts().size),
        },
        "training_history": history,
        "checkpoints": checkpoints,
        "validation_summaries": summaries,
        "selection_decision": decision,
        "published_checkpoint": published,
        "arrays": str(scratch / ARRAYS_NAME),
        "arrays_sha256": _file_sha256(arrays_path),
        "read_contract": contract["read_contract"],
        "validation_state_opened": True,
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
        "elapsed_seconds": time.monotonic() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    report["content_sha256"] = _json_sha256(report)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (scratch_tmp / REPORT_NAME).write_text(rendered)
    (project_tmp / REPORT_NAME).write_text(rendered)
    shutil.copy2(arrays_path, project_tmp / ARRAYS_NAME)
    _plot(project_tmp / FIGURE_NAME, summaries, selected_step)
    (project_tmp / README_NAME).write_text(_readme(report))
    artifacts = {
        name: _file_sha256(project_tmp / name)
        for name in (REPORT_NAME, ARRAYS_NAME, FIGURE_NAME, README_NAME)
    }
    (project_tmp / MANIFEST_NAME).write_text(
        json.dumps(
            {"status": "complete", "version": VERSION, "contract_sha256": contract_sha,
             "artifacts": artifacts, "content_sha256": _json_sha256(artifacts),
             "inference_state_opened": False},
            indent=2, sort_keys=True) + "\n"
    )
    os.replace(scratch_tmp, scratch)
    os.replace(project_tmp, project)
    return report


def _readme(report: Mapping[str, Any]) -> str:
    split = report["split"]
    lines = [
        "# Pooled loss-recovery model under the Bire et al. Section 3.2 arrangement",
        "",
        "Same model as `model_c_bire_aligned_v3_pooled_v1`: three FNO blocks, six",
        "pointwise LayerNorms, modes 24x16, width 128, Bire positional encoding,",
        "10% padding, no external local branch, Model C loss v1 over a three-step",
        "rollout, Adam 5e-4 decaying to 1e-4 at 75%, batch 8, 7,680 steps, from",
        "scratch on seed 20260724. Only the data arrangement changes.",
        "",
        "| split | indices | days per regime |",
        "| --- | --- | --- |",
        f"| train (pooled) | {split['train'][0]}--{split['train'][1]} | {split['train_days']} |",
        f"| validation (pooled) | {split['validation'][0]}--{split['validation'][1]} | {split['validation_days']} |",
        f"| inference (nested in validation) | {split['inference'][0]}--{split['inference'][1] - 1} | {split['inference_days']} |",
        "",
        "6,000 + 1,200 = 7,200 is Bire's entire record, and the 1,000 inference",
        "days are the final 1,000 of validation, not a third block -- the paper",
        "states it uses no third held-out set. No buffers: the paper has none, and",
        "leakage is prevented instead by requiring a training rollout's whole",
        f"target sequence to stay inside training, so the latest start is",
        f"{split['latest_training_rollout_start']}.",
        "",
        f"The model sees nothing at or beyond index {split['model_visible_days'][1] + 1}. Days",
        f"{split['evaluation_truth_only_days'][0]}--{split['evaluation_truth_only_days'][1]} are evaluation truth only, never trained on,",
        "validated on, or used as an inference start. Bire obtain the same",
        "separation across simulations (runs 2 and 4 are entirely held out); we",
        "obtain it along time, which is what allows the day-2,000 ground-truth",
        "column their Figure 7 also shows.",
        "",
        "Selection used the pooled validation block under the declared rule:",
        "minimise the worst 90--360-day RMSE-AUC relative to climatology subject",
        "to each field's 10--90-day AUC staying within 5% of the best checkpoint.",
        "",
        "| step | short AUC / persistence | long AUC / climatology | day-200 ACC (U/SST/P) |",
        "| --- | --- | --- | --- |",
    ]
    for value in report["validation_summaries"]:
        s = value["short_ratio_to_persistence"]
        l = value["long_ratio_to_climatology"]
        a = value["acc_day200"]
        lines.append(
            "| {step} | {s0:.3f}/{s1:.3f}/{s2:.3f} | {l0:.3f}/{l1:.3f}/{l2:.3f} | "
            "{a0:+.3f}/{a1:+.3f}/{a2:+.3f} |".format(
                step=value["optimizer_step"],
                s0=s["surface_speed"], s1=s["sst"], s2=s["phihyd_surface"],
                l0=l["surface_speed"], l1=l["sst"], l2=l["phihyd_surface"],
                a0=a["surface_u"], a1=a["sst"], a2=a["phihyd_surface"],
            )
        )
    decision = report["selection_decision"]
    lines += [
        "",
        f"Selected step {decision['selected_optimizer_step']} via `{decision['branch']}`.",
        "",
        "Training and validation only; the inference set opens through the figure",
        "contract, S0 only.",
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
            child.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = preflight(args.contract) if args.command == "preflight" else run(
        args.contract, device_name=args.device
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
