"""Loss-recovery model on trajectory-v3: independent equilibria, pooled regimes.

Same model as :mod:`af_model_c_bire_aligned_loss_recovery` --- three FNO blocks,
six pointwise LayerNorms, modes (24,16), width 128, Bire positional encoding, 10%
padding, no external local branch, direct 46-channel future state, Model C loss
v1 over a three-step rollout, Adam at 5e-4 decaying to 1e-4 at 75%, batch 8,
7,680 optimizer steps, trained from scratch on seed 20260724.

What changes is the data underneath it.  Trajectory-v3 replaces the branch-based
S1 and S2 --- which restarted from the S0 year-100 state and carried its
equilibrium --- with regimes equilibrated independently for 100 years from the
tutorial initial condition under their own wind, and extends every regime to 25
production years::

    code  split        indices      days   per regime
    1     train        0--5039      5040   pooled across S0, S1, S2
    0     buffer       5040--5129     90
    2     validation   5130--6389   1260   pooled across S0, S1, S2
    0     buffer       6390--6479     90
    3     test         6480--8999   2520   S0 primary, S1/S2 robustness

One FNO is trained on all three training blocks and selected on all three
validation blocks, so the regimes are pooled rather than modelled separately.
The pointwise normalizer, the per-regime climatology, and the increment scale are
all recomputed from the v3 training block; nothing is inherited from v2, whose
training interval and whose S1/S2 physics both differ.

Checkpoint selection reuses the declared rule from the chronological arm:
minimise the worst 90--360-day RMSE-AUC relative to climatology subject to each
primary field's 10--90-day AUC staying within 5% of the best checkpoint's, with a
declared fallback if no checkpoint satisfies the guard.

Training and validation only.  The test block stays sealed; the figure suites
open through their own contract, with S0 as the primary Bire-style comparison and
S1/S2 reported as wind-regime robustness.
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
from .af_data_v3 import (
    DATASET_VERSION,
    EXPERIMENTS,
    PRODUCTION_DAYS,
    TEST_START_WINDOW,
    TRAIN_RANGE,
    VALIDATION_RANGE,
    split_codes,
    verify_split,
)
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


VERSION = "model_c_bire_aligned_v3_pooled_v1"
CONTRACT_STATUS = "frozen_before_any_trajectory_v3_metric"
PARENT_VERSION = "model_c_bire_aligned_loss_recovery_v1"

LEARNING_RATE = 5.0e-4
ROLLOUT_STEPS = 3
CHECKPOINT_STEPS = (1920, 3840, 5760, 7680)
VALIDATION_ROLLOUT_DAYS = 360
VALIDATION_START_STRIDE = 30

NORMALIZATION_NAME = "model_c_v3_train_only_normalization.npz"
DIVERGENCE_NAME = "bire_aligned_v3_divergence.json"

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


class V3ArmError(BireAlignedFullStateError):
    """Raised when the trajectory-v3 pooled arm violates its contract."""


def validation_starts() -> np.ndarray:
    """Declared 360-day validation rollout starts inside 5130--6389."""

    latest = VALIDATION_RANGE[1] - 1 - VALIDATION_ROLLOUT_DAYS
    starts = np.arange(
        VALIDATION_RANGE[0], latest + 1, VALIDATION_START_STRIDE, dtype=np.int64
    )
    if starts.size == 0 or int(starts[-1]) + VALIDATION_ROLLOUT_DAYS > latest + VALIDATION_ROLLOUT_DAYS:
        raise V3ArmError("validation starts left the validation block")
    return starts


def validation_records() -> np.ndarray:
    """Pooled ``(regime, start)`` validation rollouts across all three regimes."""

    starts = validation_starts()
    return np.asarray(
        [
            (experiment, int(start))
            for experiment in range(len(EXPERIMENTS))
            for start in starts
        ],
        dtype=np.int64,
    )


def train_only_climatology(
    state: Any,
    wet: np.ndarray,
    *,
    chunk_days: int = 60,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Per-regime pointwise climatology over the v3 training block only."""

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
        raise V3ArmError("v3 train-only climatology count changed")
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
        raise V3ArmError("the parent loss-recovery contract changed")
    parent = json.loads(path.read_text())
    if parent.get("version") != PARENT_VERSION:
        raise V3ArmError("the parent arm is not the loss-recovery control")
    if contract["architecture"] != parent["architecture"]:
        raise V3ArmError("the v3 arm must keep the parent architecture exactly")
    if contract["loss"] != parent["loss"]:
        raise V3ArmError("the v3 arm must keep the parent objective exactly")
    for field in FROZEN_TRAINING_FIELDS:
        if contract["training"].get(field) != parent["training"].get(field):
            raise V3ArmError(f"the v3 arm moved a model quantity: {field}")


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the contract frozen before any trajectory-v3 metric."""

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
        or contract.get("normalization", {}).get("recomputed_from") != "v3_train_only_0_5039"
        or selection.get("rule") != "min_worst_long_climatology_ratio_subject_to_short_guard"
        or selection.get("evaluated_on") != "pooled_v3_validation_block_5130_6389"
        or read.get("training_state") is not True
        or read.get("validation_state") is not True
        or any(
            read.get(name) is not False
            for name in ("test_state", "intermediate_wind_state", "response_state", "adjoint_state")
        )
    ):
        raise V3ArmError("trajectory-v3 pooled contract changed")
    BireAlignedArchitecture(**architecture)
    _assert_model_matches_parent(contract)
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise V3ArmError(f"trajectory-v3 source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def _verify_dataset(contract: Mapping[str, Any]) -> Path:
    record = contract["sources"]["dataset"]
    dataset = Path(record["path"]).resolve()
    if not dataset.is_dir() or _file_sha256(dataset / ".zmetadata") != record["metadata_sha256"]:
        raise V3ArmError("trajectory-v3 dataset source changed")
    return dataset


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify the dataset, the split, and the model without training."""

    contract, resolved, digest = load_contract(contract_path)
    dataset = _verify_dataset(contract)
    group = zarr.open_consolidated(str(dataset), mode="r")
    stored_snapshots = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    stored_pairs = np.asarray(group["pair_split"][:], dtype=np.uint8)
    declared_snapshots, declared_pairs = split_codes()
    if not np.array_equal(stored_snapshots, declared_snapshots) or not np.array_equal(
        stored_pairs, declared_pairs
    ):
        raise V3ArmError("the stored v3 split does not match the declaration")
    records = records_for_rollout_split(stored_pairs, 1, rollout_steps=ROLLOUT_STEPS)
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
        "test_start_window": list(TEST_START_WINDOW),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "test_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    """Train on the pooled v3 training blocks and select on pooled validation."""

    if torch is None or DataLoader is None:  # pragma: no cover
        raise RuntimeError("the trajectory-v3 arm requires PyTorch")
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
        raise FileExistsError("refusing to overwrite trajectory-v3 arm output")

    training = contract["training"]
    seed_everything(int(training["seed"]))
    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    snapshot_split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    pair_split = np.asarray(group["pair_split"][:], dtype=np.uint8)
    declared_snapshots, declared_pairs = split_codes()
    if not np.array_equal(snapshot_split, declared_snapshots) or not np.array_equal(
        pair_split, declared_pairs
    ):
        raise V3ArmError("the stored v3 split does not match the declaration")
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
        raise V3ArmError("loss-v1 changed")

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
        path = checkpoint_directory / f"model_c_v3_step_{step:05d}.pt"
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
        raise V3ArmError("not every declared checkpoint was written")

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
            "recomputed_from": "v3_train_only_0_5039",
            "summary": normalizers["summary"],
            "artifact": str(scratch / NORMALIZATION_NAME),
            "artifact_sha256": _file_sha256(normalization_path),
        },
        "climatology": {"source": "per_regime_pointwise_mean_over_v3_train_only",
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
        "test_state_opened": False,
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
             "test_state_opened": False},
            indent=2, sort_keys=True) + "\n"
    )
    os.replace(scratch_tmp, scratch)
    os.replace(project_tmp, project)
    return report


def _readme(report: Mapping[str, Any]) -> str:
    split = report["split"]
    lines = [
        "# Loss-recovery model on trajectory-v3 (independent equilibria, pooled regimes)",
        "",
        "Same model as `model_c_bire_aligned_loss_recovery_v1`: three FNO blocks,",
        "six pointwise LayerNorms, modes 24x16, width 128, Bire positional encoding,",
        "10% padding, no external local branch, Model C loss v1 over a three-step",
        "rollout, Adam 5e-4 decaying to 1e-4 at 75%, batch 8, 7,680 steps, from",
        "scratch on seed 20260724.",
        "",
        "The data underneath it is new. S1 and S2 are no longer branches of the S0",
        "year-100 state: each regime was equilibrated independently for 100 years",
        "from the tutorial initial condition under its own wind, then run 25",
        "production years.",
        "",
        "| split | indices | days per regime |",
        "| --- | --- | --- |",
        f"| train (pooled) | {split['train'][0]}--{split['train'][1]} | {split['train_days']} |",
        f"| validation (pooled) | {split['validation'][0]}--{split['validation'][1]} | {split['validation_days']} |",
        f"| test (sealed here) | {split['test'][0]}--{split['test'][1]} | {split['test_days']} |",
        "",
        "One FNO trained on all three training blocks and selected on all three",
        "validation blocks. Pointwise normalizer, per-regime climatology, and",
        "increment scale all recomputed from the v3 training block.",
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
        "Training and validation only; the test block remained sealed.",
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
