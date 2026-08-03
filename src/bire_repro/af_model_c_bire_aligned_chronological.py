"""Loss-recovery model retrained under a strictly chronological split.

The loss-recovery arm established a promising combination --- Bire-aligned
three-block architecture with the incumbent group-balanced Model C objective ---
under the *stored* split, whose training block is interleaved with the later
record (0--2519 and 3690--6209).  That layout supports an interpolation claim.
This arm repeats the identical model under the clean prospective protocol
declared in :mod:`af_model_c_chronological_split`::

    train 0--5039  >  buffer  >  validation 5130--5759  >  buffer  >  test 5850--7199

and answers whether the conclusion survives.

Nothing about the model changes: same seed, three FNO blocks, six pointwise
LayerNorms, modes (24,16), width 128, Bire positional encoding, 10% padding, no
external local branch, direct 46-channel future state, Model C loss v1 over a
three-step unrolled rollout, Adam at 5e-4 decaying to 1e-4 at 75%, batch size 8,
7,680 optimizer steps, trained from scratch.

Three things do change, and all three are consequences of the split rather than
free choices:

* **the normalizer is recomputed from indices 0--5039 only.**  Reusing the
  incumbent's would import information from 5040--6209, which is validation or
  test here.  The pointwise mean, the pointwise scale, the per-channel
  fifth-percentile wet-cell floor, the per-regime climatology, and the pointwise
  increment scale are all rebuilt from the new training interval.
* **checkpoint selection moves to the held validation block.**  The stored-split
  arms selected on a 360-day instrument evaluated over *training* records; here
  selection uses 90 declared 360-day rollouts inside 5130--5759, which the model
  never trains on.
* **the training set itself moves.**  Both sets contain 5,040 days but only
  3,870 overlap: 5040--6209 is exchanged for 2520--3689, so 23.2% of the
  training snapshots change.  This arm therefore tests the chronological
  protocol *and* sensitivity to the training period, and must not be reported as
  a split-order ablation alone.

The declared selection rule minimises the worst long-lead error relative to
climatology subject to a short-skill guard::

    minimise   max_q  AUC_90:360(RMSE_q) / AUC_90:360(RMSE_q, climatology)
    subject to AUC_10:90(RMSE_q) <= 1.05 * min_theta AUC_10:90(RMSE_q)

over speed, SST, and surface pressure.  If no checkpoint satisfies the guard the
declared fallback is the checkpoint minimising the worst short-skill ratio, and
the report records which branch fired.

Training and validation only.  The test block stays sealed: the held S0 figure
suite opens through its own contract, on the same 15 starts in 6660--7199 that
are held out under both splits.
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
from .af_forward_complete import _member_acc, _member_rmse, derived_fields
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
from .af_model_c_chronological_split import (
    TRAIN_RANGE,
    VALIDATION_ROLLOUT_DAYS,
)
from .af_model_c_chronological_split import VERSION as SPLIT_VERSION
from .af_model_c_chronological_split import (
    pair_codes as chronological_pair_codes,
)
from .af_model_c_chronological_split import (
    snapshot_codes as chronological_snapshot_codes,
)
from .af_model_c_chronological_split import (
    train_only_normalizers,
    training_overlap,
    validation_records,
    verify as verify_split,
)
from .af_model_c_overfit import AUDIT_TERMS, _device, _file_sha256
from .af_model_c_successor import STATE_CHANNEL_COUNT

try:
    import torch
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


VERSION = "model_c_bire_aligned_loss_recovery_chronological_v1"
CONTRACT_STATUS = "frozen_before_any_chronological_split_metric"
PARENT_VERSION = "model_c_bire_aligned_loss_recovery_v1"

LEARNING_RATE = 5.0e-4
ROLLOUT_STEPS = 3
CHECKPOINT_STEPS = (1920, 3840, 5760, 7680)
LEAD_DAYS = tuple(range(10, VALIDATION_ROLLOUT_DAYS + 1, 10))
SHORT_LEADS = tuple(lead for lead in LEAD_DAYS if 10 <= lead <= 90)
LONG_LEADS = tuple(lead for lead in LEAD_DAYS if 90 <= lead <= 360)
ACC_LEAD_LIMIT = 200
PRIMARY_FIELDS = ("surface_speed", "sst", "phihyd_surface")
ACC_FIELDS = ("surface_u", "surface_v", "phihyd_surface", "sst")
BIAS_FIELDS = ("sst", "eta", "phihyd_surface", "streamfunction")
SHORT_SKILL_TOLERANCE = 1.05

REPORT_NAME = "bire_aligned_chronological_report.json"
ARRAYS_NAME = "bire_aligned_chronological_arrays.npz"
FIGURE_NAME = "model_c_bire_aligned_chronological_selection.png"
NORMALIZATION_NAME = "model_c_chronological_train_only_normalization.npz"
DIVERGENCE_NAME = "bire_aligned_chronological_divergence.json"

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


class ChronologicalArmError(BireAlignedFullStateError):
    """Raised when the chronological-split arm violates its contract."""


def _assert_model_matches_parent(contract: Mapping[str, Any]) -> None:
    """Fail unless only the split and its derived statistics moved."""

    record = contract["sources"]["parent_contract"]
    path = Path(record["path"]).resolve()
    if not path.is_file() or _file_sha256(path) != record["sha256"]:
        raise ChronologicalArmError("the parent loss-recovery contract changed")
    parent = json.loads(path.read_text())
    if parent.get("version") != PARENT_VERSION:
        raise ChronologicalArmError("the parent arm is not the loss-recovery control")
    if contract["architecture"] != parent["architecture"]:
        raise ChronologicalArmError(
            "the chronological arm must keep the parent architecture exactly"
        )
    if contract["loss"] != parent["loss"]:
        raise ChronologicalArmError(
            "the chronological arm must keep the parent objective exactly"
        )
    for field in FROZEN_TRAINING_FIELDS:
        if contract["training"].get(field) != parent["training"].get(field):
            raise ChronologicalArmError(
                f"the chronological arm moved a model quantity: {field}"
            )


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the contract frozen before any chronological-split metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    architecture = contract.get("architecture", {})
    training = contract.get("training", {})
    split = contract.get("split", {})
    selection = contract.get("checkpoint_selection", {})
    read = contract.get("read_contract", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or split.get("module_version") != SPLIT_VERSION
        or tuple(split.get("train", ())) != TRAIN_RANGE
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
        or contract.get("loss", {}).get("contract_sha256")
        != MODEL_C_LOSS_V1_CONTRACT_SHA256
        or contract.get("normalization", {}).get("recomputed_from") != "train_only_0_5039"
        or selection.get("rule") != "min_worst_long_climatology_ratio_subject_to_short_guard"
        or float(selection.get("short_skill_tolerance", -1.0)) != SHORT_SKILL_TOLERANCE
        or selection.get("evaluated_on") != "held_validation_block_5130_5759"
        or read.get("training_state") is not True
        or read.get("validation_state") is not True
        or any(
            read.get(name) is not False
            for name in (
                "test_state",
                "held_s0_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
            )
        )
    ):
        raise ChronologicalArmError("chronological-split contract changed")
    BireAlignedArchitecture(**architecture)
    _assert_model_matches_parent(contract)
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ChronologicalArmError(
                    f"chronological-split source changed: {source}"
                )
    return contract, resolved, _file_sha256(resolved)


def _verify_dataset(contract: Mapping[str, Any]) -> Path:
    record = contract["sources"]["dataset"]
    dataset = Path(record["path"]).resolve()
    if (
        not dataset.is_dir()
        or _file_sha256(dataset / ".zmetadata") != record["metadata_sha256"]
    ):
        raise ChronologicalArmError("dataset source changed")
    return dataset


def _gather(state: Any, records: np.ndarray, offset: int) -> np.ndarray:
    """Truth states at ``start + offset`` for every ``(experiment, start)`` record."""

    return np.stack(
        [
            np.asarray(state[int(experiment), int(start) + int(offset)], dtype=np.float32)
            for experiment, start in records
        ]
    )


def train_only_climatology(
    state: Any,
    wet: np.ndarray,
    *,
    chunk_days: int = 60,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Per-regime pointwise climatology from the new training interval only.

    Nonlinear fields are averaged *after* derivation, matching the frozen S0
    figure convention so the two climatologies remain comparable.
    """

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
        raise ChronologicalArmError("train-only climatology count changed")
    state_mean = (state_sum / count).astype(np.float32)
    state_mean[:, :, ~wet] = 0.0
    derived_mean = {}
    for name, value in derived_sum.items():
        mean = (value / count).astype(np.float32)
        mean[:, ~wet] = 0.0
        derived_mean[name] = mean
    return state_mean, derived_mean, int(count)


def lead_auc(curve: Sequence[float], leads: Sequence[float], window: Sequence[int]) -> float:
    """Trapezoidal RMSE-AUC of ``curve`` over the leads inside ``window``.

    Module level and separately tested: as a closure this was unreachable from
    the suite, and a NumPy 2 removal (``np.trapz``) reached production instead.
    """

    lead_array = np.asarray(leads, dtype=np.float64)
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != lead_array.shape:
        raise ValueError("RMSE curve and lead axis must have the same length")
    mask = np.isin(lead_array, np.asarray(window, dtype=np.float64))
    if int(mask.sum()) < 2:
        raise ValueError("an AUC window needs at least two leads")
    integrate = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed it
    return float(integrate(values[mask], lead_array[mask]))


def _evaluation_fields(states: np.ndarray, wet: np.ndarray) -> dict[str, np.ndarray]:
    fields = derived_fields(states, wet)
    fields["surface_u"] = np.asarray(states[:, 0], dtype=np.float32)
    fields["surface_v"] = np.asarray(states[:, 15], dtype=np.float32)
    fields["eta"] = np.asarray(states[:, 45], dtype=np.float32)
    return fields


def validate_checkpoint(
    stepper: BireAlignedStepper,
    state: Any,
    static: Any,
    records: np.ndarray,
    climatology_state: np.ndarray,
    climatology_derived: Mapping[str, np.ndarray],
    wet: np.ndarray,
) -> dict[str, Any]:
    """360-day validation rollout metrics for one checkpoint."""

    experiments = records[:, 0]
    initial = _gather(state, records, 0)
    current = stepper.normalized_state(initial)
    forcing = stepper.normalized_static(static, experiments)
    initial_fields = _evaluation_fields(initial, wet)
    climate_state = np.stack([climatology_state[int(e)] for e in experiments])
    climate_fields = _evaluation_fields(climate_state, wet)
    for name, value in climatology_derived.items():
        climate_fields[name] = np.stack([value[int(e)] for e in experiments])

    leads = len(LEAD_DAYS)
    rmse = {
        method: {field: np.empty((records.shape[0], leads), dtype=np.float64)
                 for field in PRIMARY_FIELDS}
        for method in ("model", "persistence", "climatology")
    }
    acc = {field: [] for field in ACC_FIELDS}
    amplitude = np.empty(leads, dtype=np.float64)
    bias: dict[str, float] = {}
    wet_tensor = torch.from_numpy(wet).to(stepper.device)

    with torch.no_grad():
        for index, lead in enumerate(LEAD_DAYS):
            current = stepper.step(current, forcing)
            prediction = stepper.physical(current)
            truth = _gather(state, records, lead)
            predicted = _evaluation_fields(prediction, wet)
            observed = _evaluation_fields(truth, wet)
            for field in PRIMARY_FIELDS:
                rmse["model"][field][:, index] = _member_rmse(
                    predicted[field], observed[field], wet
                )
                rmse["persistence"][field][:, index] = _member_rmse(
                    initial_fields[field], observed[field], wet
                )
                rmse["climatology"][field][:, index] = _member_rmse(
                    climate_fields[field], observed[field], wet
                )
            amplitude[index] = float(
                torch.amax(torch.abs(current[:, :, wet_tensor])).detach().cpu()
            )
            if lead <= ACC_LEAD_LIMIT:
                for field in ACC_FIELDS:
                    acc[field].append(
                        float(
                            np.mean(
                                _member_acc(
                                    predicted[field],
                                    observed[field],
                                    climate_fields[field],
                                    wet,
                                )
                            )
                        )
                    )
            if lead == LEAD_DAYS[-1]:
                for field in BIAS_FIELDS:
                    error = (predicted[field] - observed[field])[:, wet]
                    bias[field] = float(np.mean(error))

    mean_rmse = {
        method: {field: rmse[method][field].mean(axis=0) for field in PRIMARY_FIELDS}
        for method in rmse
    }
    short_auc = {
        f: lead_auc(mean_rmse["model"][f], LEAD_DAYS, SHORT_LEADS)
        for f in PRIMARY_FIELDS
    }
    long_auc = {
        f: lead_auc(mean_rmse["model"][f], LEAD_DAYS, LONG_LEADS)
        for f in PRIMARY_FIELDS
    }
    long_clim = {
        f: lead_auc(mean_rmse["climatology"][f], LEAD_DAYS, LONG_LEADS)
        for f in PRIMARY_FIELDS
    }
    short_persistence = {
        f: lead_auc(mean_rmse["persistence"][f], LEAD_DAYS, SHORT_LEADS)
        for f in PRIMARY_FIELDS
    }
    gain = {}
    for field in PRIMARY_FIELDS:
        curve = mean_rmse["model"][field]
        gain[field] = float((curve[-1] / curve[-4]) ** (1.0 / 3.0)) if curve[-4] > 0 else float("nan")
    return {
        "lead_days": list(LEAD_DAYS),
        "mean_rmse": {m: {f: v.tolist() for f, v in d.items()} for m, d in mean_rmse.items()},
        "short_auc_10_90": short_auc,
        "long_auc_90_360": long_auc,
        "long_auc_90_360_climatology": long_clim,
        "long_ratio_to_climatology": {
            f: float(long_auc[f] / long_clim[f]) for f in PRIMARY_FIELDS
        },
        "short_ratio_to_persistence": {
            f: float(short_auc[f] / short_persistence[f]) for f in PRIMARY_FIELDS
        },
        "acc_through_day200": {f: acc[f] for f in ACC_FIELDS},
        "acc_day200": {f: float(acc[f][-1]) for f in ACC_FIELDS},
        "per_call_gain_330_360": gain,
        "maximum_normalized_amplitude": float(np.max(amplitude)),
        "slow_field_bias_day360": bias,
        "arrays": {
            method: {f: rmse[method][f] for f in PRIMARY_FIELDS} for method in rmse
        },
    }


def select_by_validation(
    summaries: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = SHORT_SKILL_TOLERANCE,
) -> dict[str, Any]:
    """Apply the declared validation checkpoint rule."""

    if not summaries:
        raise ChronologicalArmError("validation selection needs summaries")
    best_short = {
        field: min(float(s["short_auc_10_90"][field]) for s in summaries)
        for field in PRIMARY_FIELDS
    }
    feasible = [
        s
        for s in summaries
        if all(
            float(s["short_auc_10_90"][field]) <= tolerance * best_short[field]
            for field in PRIMARY_FIELDS
        )
    ]
    if feasible:
        selected = min(
            feasible,
            key=lambda s: (
                max(float(v) for v in s["long_ratio_to_climatology"].values()),
                int(s["optimizer_step"]),
            ),
        )
        branch = "primary_rule"
    else:
        selected = min(
            summaries,
            key=lambda s: (
                max(
                    float(s["short_auc_10_90"][f]) / best_short[f]
                    for f in PRIMARY_FIELDS
                ),
                int(s["optimizer_step"]),
            ),
        )
        branch = "declared_fallback_no_checkpoint_met_the_short_guard"
    return {
        "branch": branch,
        "short_skill_tolerance": float(tolerance),
        "best_short_auc_10_90": best_short,
        "feasible_steps": [int(s["optimizer_step"]) for s in feasible],
        "selected_optimizer_step": int(selected["optimizer_step"]),
        "selected_worst_long_ratio_to_climatology": max(
            float(v) for v in selected["long_ratio_to_climatology"].values()
        ),
    }


def _plot(path: Path, summaries: Sequence[Mapping[str, Any]], selected_step: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(s["optimizer_step"]) for s in summaries]
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    for field in PRIMARY_FIELDS:
        axes[0].plot(
            steps,
            [s["short_ratio_to_persistence"][field] for s in summaries],
            "o-",
            label=field.replace("_", " "),
        )
        axes[1].plot(
            steps,
            [s["long_ratio_to_climatology"][field] for s in summaries],
            "o-",
            label=field.replace("_", " "),
        )
    axes[0].axhline(1.0, color="black", linestyle="--")
    axes[0].set_ylabel("10--90-day RMSE-AUC / persistence")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_ylabel("90--360-day RMSE-AUC / climatology")
    for field in ACC_FIELDS:
        axes[2].plot(
            steps,
            [s["acc_day200"][field] for s in summaries],
            "o-",
            label=field.replace("_", " "),
        )
    axes[2].set_ylabel("Day-200 ACC")
    axes[2].set_ylim(-0.1, 1.0)
    for axis in axes:
        axis.axvline(selected_step, color="#d95f02", alpha=0.7)
        axis.set_xlabel("Optimizer step")
        axis.set_xticks(steps)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Chronological split: held-validation checkpoint selection "
        f"(selected step {selected_step})"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify the split, the sources, and the model without training."""

    contract, resolved, digest = load_contract(contract_path)
    dataset = _verify_dataset(contract)
    split = verify_split()
    group = zarr.open_consolidated(str(dataset), mode="r")
    stored = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    overlap = training_overlap(stored)
    records = records_for_rollout_split(
        chronological_pair_codes(), 1, rollout_steps=ROLLOUT_STEPS
    )
    architecture = BireAlignedArchitecture(**contract["architecture"])
    model = build_bire_aligned_model(architecture)
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "split": split,
        "training_overlap_with_stored_split": overlap,
        "training_rollout_records": len(records),
        "training_starts_per_regime": len(records) // 3,
        "validation_records": int(validation_records().shape[0]),
        "normalization": "recomputed_from_train_only_0_5039",
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "test_state_opened": False,
        "held_s0_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }


def run(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train from scratch on the chronological split and select on validation."""

    if torch is None or DataLoader is None:  # pragma: no cover
        raise RuntimeError("the chronological arm requires PyTorch")
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
        raise FileExistsError("refusing to overwrite chronological-arm output")

    training = contract["training"]
    seed_everything(int(training["seed"]))
    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    stored_snapshots = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    overlap = training_overlap(stored_snapshots)
    _, _, wet_array, _, wind_mean, wind_scale = _normalizers(group)
    wet_array = np.asarray(wet_array, dtype=bool)

    # Every train-derived statistic is rebuilt from indices 0--5039.
    normalizers = train_only_normalizers(group)
    point_mean = normalizers["mean"]
    point_scale = normalizers["scale"]
    pair_split = chronological_pair_codes()
    increment_values = pointwise_increment_scale(group, pair_split, point_scale)
    climatology_state, climatology_derived, climatology_days = train_only_climatology(
        state, wet_array
    )

    loss_config = model_c_loss_config("v1")
    if loss_contract_sha256(loss_config) != MODEL_C_LOSS_V1_CONTRACT_SHA256:
        raise ChronologicalArmError("loss-v1 changed")

    training_records = records_for_rollout_split(
        pair_split, 1, rollout_steps=loss_config.rollout_steps
    )
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
            training_dataset, batch_size, int(training["seed"])
        ),
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
        western_boundary_mask(wet_array, loss_config.western_boundary_width).astype(
            np.float32
        )
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
                {
                    "status": "diverged",
                    "version": VERSION,
                    "contract_sha256": contract_sha,
                    "reason": reason,
                    "optimizer_step": int(step),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
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
        raw_features = raw_features.to(device=device, dtype=torch.float32, non_blocking=True)
        futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
        features = retained_features(raw_features)
        model.train()
        predictions = direct_state_unroll(model, features, wet, loss_config.rollout_steps)
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
        path = checkpoint_directory / f"model_c_chronological_step_{step:05d}.pt"
        torch.save(
            {
                "version": VERSION,
                "optimizer_step": step,
                "fine_tune_step": step,
                "architecture": architecture.to_dict(),
                "contract": str(resolved_contract),
                "contract_sha256": contract_sha,
                "split_version": SPLIT_VERSION,
                "base_loss_contract": loss_contract(loss_config),
                "base_loss_contract_sha256": MODEL_C_LOSS_V1_CONTRACT_SHA256,
                "rollout_steps": loss_config.rollout_steps,
                "training_history_record": history_record,
                "model_state_dict": _checkpoint_state_dict(model),
            },
            path,
        )
        checkpoints.append(
            {
                "optimizer_step": step,
                "checkpoint": path.name,
                "checkpoint_sha256": _file_sha256(path),
            }
        )
        totals = {name: 0.0 for name in AUDIT_TERMS}
        samples = 0

    if len(checkpoints) != len(CHECKPOINT_STEPS):
        raise ChronologicalArmError("not every declared checkpoint was written")

    # Held validation block opens here; the test block stays sealed.
    records = validation_records()
    summaries = []
    evaluated_arrays = []
    for record in checkpoints:
        payload = torch.load(
            checkpoint_directory / record["checkpoint"],
            map_location=device,
            weights_only=False,
        )
        probe = build_bire_aligned_model(architecture).to(device)
        probe.load_state_dict(payload["model_state_dict"])
        probe.eval()
        stepper = BireAlignedStepper(
            model=probe,
            device=device,
            wet=wet_array,
            mean=point_mean,
            scale=point_scale,
            wind_mean=float(wind_mean),
            wind_scale=float(wind_scale),
        )
        value = validate_checkpoint(
            stepper,
            state,
            static,
            records,
            climatology_state,
            climatology_derived,
            wet_array,
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
        lead_days=np.asarray(LEAD_DAYS, dtype=np.int16),
        validation_records=records.astype(np.int32),
        snapshot_codes=chronological_snapshot_codes().astype(np.uint8),
        pair_codes=pair_split.astype(np.uint8),
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
        "split": split_summary,
        "training_overlap_with_stored_split": overlap,
        "architecture": architecture.to_dict(),
        "parameter_count": parameter_count,
        "normalization": {
            "recomputed_from": "train_only_0_5039",
            "summary": normalizers["summary"],
            "artifact": str(scratch / NORMALIZATION_NAME),
            "artifact_sha256": _file_sha256(normalization_path),
            "wind_normalization_note": (
                "static_features has no time axis, so wind statistics are a "
                "property of the forcing regimes and cannot depend on the split"
            ),
        },
        "climatology": {
            "source": "per_regime_pointwise_mean_over_train_only_0_5039",
            "days_per_regime": climatology_days,
        },
        "increment_scale": increment_values.tolist(),
        "loss": contract["loss"],
        "base_loss_contract_sha256": MODEL_C_LOSS_V1_CONTRACT_SHA256,
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
        "counts": {
            "training_rollout_records": len(training_records),
            "training_starts_per_regime": len(training_records) // 3,
            "validation_records": int(records.shape[0]),
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
    _plot(project_tmp / FIGURE_NAME, summaries, selected_step)
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
                "test_state_opened": False,
                "held_s0_state_opened": False,
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
    split = report["split"]
    overlap = report["training_overlap_with_stored_split"]
    lines = [
        "# Loss-recovery model under a strictly chronological split",
        "",
        "Identical model to `model_c_bire_aligned_loss_recovery_v1` --- same seed,",
        "three FNO blocks, six pointwise LayerNorms, modes 24x16, width 128, Bire",
        "positional encoding, 10% padding, no external local branch, Model C loss",
        "v1 over a three-step rollout, Adam 5e-4 decaying to 1e-4 at 75%, batch 8,",
        "7,680 steps, trained from scratch.",
        "",
        "| split | indices | days |",
        "| --- | --- | --- |",
        f"| train | {split['train'][0]}--{split['train'][1]} | {split['train_days']} |",
        f"| validation | {split['validation'][0]}--{split['validation'][1]} | {split['validation_days']} |",
        f"| test (sealed here) | {split['test'][0]}--{split['test'][1]} | {split['test_days']} |",
        "",
        f"Training strictly precedes validation, which strictly precedes test, with",
        f"90-day buffers at both boundaries ({split['buffer_days']} buffer days).",
        "",
        "**This is not a pure split-order ablation.** Both training sets contain",
        f"5,040 days but only {overlap['shared_days']:,} overlap: "
        f"{overlap['dropped_from_training']} is exchanged for "
        f"{overlap['added_to_training']}, changing "
        f"{overlap['changed_fraction']:.1%} of the training snapshots. The arm",
        "tests the chronological protocol *and* sensitivity to the training period.",
        "",
        "All train-derived statistics were recomputed from 0--5039: pointwise mean,",
        "pointwise scale, channel scale floors, per-regime climatology, and the",
        "pointwise increment scale. Wind normalization is unchanged because",
        "`static_features` has no time axis.",
        "",
        "Checkpoints were selected on 90 held 360-day validation rollouts inside",
        "5130--5759 using the declared rule: minimise the worst 90--360-day",
        "RMSE-AUC relative to climatology, subject to each field's 10--90-day",
        "RMSE-AUC staying within 5% of the best checkpoint's.",
        "",
        "| step | short AUC / persistence | long AUC / climatology | day-200 ACC (U/SST/P) |",
        "| --- | --- | --- | --- |",
    ]
    for value in report["validation_summaries"]:
        short = value["short_ratio_to_persistence"]
        long = value["long_ratio_to_climatology"]
        acc = value["acc_day200"]
        lines.append(
            "| {step} | {s0:.3f}/{s1:.3f}/{s2:.3f} | {l0:.3f}/{l1:.3f}/{l2:.3f} | "
            "{a0:+.3f}/{a1:+.3f}/{a2:+.3f} |".format(
                step=value["optimizer_step"],
                s0=short["surface_speed"], s1=short["sst"], s2=short["phihyd_surface"],
                l0=long["surface_speed"], l1=long["sst"], l2=long["phihyd_surface"],
                a0=acc["surface_u"], a1=acc["sst"], a2=acc["phihyd_surface"],
            )
        )
    decision = report["selection_decision"]
    lines += [
        "",
        f"Selected step {decision['selected_optimizer_step']} via `{decision['branch']}`.",
        "",
        "Training and validation only. The test block and the held S0 archive",
        "remained sealed for this run.",
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
    if args.command == "preflight":
        result = preflight(args.contract)
    else:
        result = run(args.contract, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
