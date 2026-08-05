"""Six-step rollout fine-tune of the matched-budget Bire-protocol checkpoint.

This is a fine-tuning arm, not a from-scratch arm.  It starts from the weights
:mod:`af_model_c_bire_protocol_duration` selected at optimizer step 15,360 and
continues training them for 3,840 further steps with the autoregressive rollout
deepened from three ten-day calls to six::

    t -> t+10 -> t+20 -> t+30 -> t+40 -> t+50 -> t+60

The model prediction is fed back at steps two through six; there is no teacher
forcing after the initial state, exactly as in the three-step arm.

Everything that defines the model and its coordinates is held fixed against the
checkpoint being fine-tuned --- the architecture, the Fourier modes, the static
inputs, the Bire positional encoding, the 46-channel output, the Bire Section
3.2 split, and the train-only pointwise normalizers, which are **reused from the
parent package rather than recomputed** so the fine-tuned model lives in exactly
the same normalized state space as its initialization.

Three quantities move, and they move together because they are one change:

============================  =================  =================
quantity                      step-15,360 arm    this arm
============================  =================  =================
``rollout_steps``             3                  6
``rollout_weight``            0.15               0.50
initial learning rate         5e-4               2e-5
============================  =================  =================

with batch 8 -> 4 so that activation memory is unchanged (4 x 6 = 8 x 3), a
3,840-step budget, and the same 0.75/0.2 schedule shape, which puts the decay at
step 2,880 and the tail 960 steps at 4e-6.  The optimizer is reset: Adam state
from the parent run is deliberately **not** loaded, because its second moments
were accumulated under a different objective.

The objective is the group-balanced Model C loss with the rollout extended::

    L = L_state^(1)
      + 0.001  L_increment^(1)
      + 0.50   (1/5) sum_{k=2..6} L_state^(k)
      + 1e-5   (1/6) sum_{k=1..6} L_spectral^(k)
      + 0.065  (1/6) sum_{k=1..6} L_boundary^(k)

:func:`af_model_c.model_c_loss_terms` already computes every one of those five
groups as a function of ``config.rollout_steps``, so no loss *code* changes.
What changes is the configuration, and the certified three-step
:class:`~af_model_c.ModelCLossConfig` refuses ``rollout_steps != 3`` by design.
:class:`BireProtocolRolloutFineTuneLossConfig` is therefore declared here, in
this arm's own module, leaving ``af_model_c`` and the v1 contract SHA-256
untouched --- the six-step objective gets its own contract and its own hash,
and the two can never be confused in a checkpoint header.

Why this arm does not rebind into the parent's training loop
------------------------------------------------------------
Derived arms in this project usually reuse
:mod:`af_model_c_bire_protocol`'s ``run`` by rebinding module globals.  That
idiom fits a one-factor control; it does not fit here.  A fine-tune differs from
its parent in *how it starts* (warm, from a named checkpoint), in *what it reads*
(published normalizers instead of a fresh pass over the training block), and in
*which objective it optimizes*.  Expressing those through global rebinding would
mean silently replacing ``build_bire_aligned_model`` with something that loads
weights --- which would also fire for the four validation probes --- and pointing
the v1 loss-contract constant at a non-v1 hash.  The orchestration is written out
instead.  Every component it calls is the shared, already-certified one: the same
dataset class, the same unroll, the same loss terms, the same validation pass,
the same selection rule, the same plot.

Training and validation only.  The S0 figure suite and the 2,000-day acceptance
gate open through :mod:`af_model_c_bire_protocol_rollout_ft_figures`.
"""

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
from .af_bire_protocol_split import (
    INFERENCE_RANGE,
    TRAIN_RANGE,
    VALIDATION_RANGE,
    store_codes,
)
from .af_bire_protocol_split import validation_records as protocol_validation_records
from .af_bire_protocol_split import validation_starts as protocol_validation_starts
from .af_bire_protocol_split import verify as verify_split
from .af_data_v3 import DATASET_VERSION, EXPERIMENTS, PRODUCTION_DAYS
from .af_model_a import (
    ChunkAwareBatchSampler,
    _checkpoint_state_dict,
    require_model_a_runtime,
    seed_everything,
)
from .af_model_b import records_for_rollout_split, western_boundary_mask
from .af_model_c import (
    GROUP_SLICES,
    MODEL_C_LOSS_V1_CONTRACT_SHA256,
    ModelCLossConfig,
    model_c_loss_terms,
)
from .af_model_c_anomaly_direct import ModelCAnomalyRolloutDataset, direct_state_unroll
from .af_model_c_bire_aligned_chronological import (
    PRIMARY_FIELDS,
    SHORT_SKILL_TOLERANCE,
    _plot,
    select_by_validation,
    validate_checkpoint,
)
from .af_model_c_bire_aligned_full_state import (
    CHECKPOINT_DIRECTORY,
    EXTERNAL_INPUT_CHANNELS,
    LIFTING_INPUT_CHANNELS,
    MANIFEST_NAME,
    README_NAME,
    BireAlignedArchitecture,
    BireAlignedDivergenceError,
    BireAlignedStepper,
    _json_sha256,
    build_bire_aligned_model,
    retained_features,
)
from .af_model_c_bire_protocol import _assert_store_is_v3, train_only_climatology
from .af_model_c_overfit import AUDIT_TERMS, _device, _file_sha256
from .af_model_c_successor import STATE_CHANNEL_COUNT

try:
    import torch
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


VERSION = "model_c_bire_protocol_rollout_ft_v1"
CONTRACT_STATUS = "frozen_before_any_bire_protocol_rollout_ft_metric"

#: The arm whose selected checkpoint this one starts from.
PARENT_VERSION = "model_c_bire_protocol_duration_v1"
#: The optimizer step of that checkpoint, and the comparator for every gate.
BASELINE_OPTIMIZER_STEP = 15360

#: The deepened rollout: six ten-day calls, prediction fed back from step two.
ROLLOUT_STEPS = 6
#: Rollout weight raised with the depth; the three-step arm uses 0.15.
ROLLOUT_WEIGHT = 0.50
#: Held at the certified v1 values.
INCREMENT_WEIGHT = 0.001
SPECTRAL_WEIGHT = 1.0e-5
BOUNDARY_WEIGHT = 0.065
SPECTRAL_BINS = 12
WESTERN_BOUNDARY_WIDTH = 4

LEARNING_RATE = 2.0e-5
BATCH_SIZE = 4
MAXIMUM_STEPS = 3840
CHECKPOINT_STEPS = (960, 1920, 2880, 3840)
DECAY_FRACTION = 0.75
DECAY_FACTOR = 0.2
SEED = 20260724

#: Starts whose whole six-step target sequence stays inside training 0--5999.
TRAINING_STARTS_PER_REGIME = 5940
TRAINING_RECORDS = TRAINING_STARTS_PER_REGIME * len(EXPERIMENTS)

#: Final acceptance gate, section 6 of the arm declaration.
SHORT_AUC_TOLERANCE_TO_BASELINE = 1.05
WORST_LONG_RATIO_CEILING = 0.85

NORMALIZATION_NAME = "model_c_bire_protocol_rollout_ft_train_only_normalization.npz"
DIVERGENCE_NAME = "bire_protocol_rollout_ft_divergence.json"
CHECKPOINT_STEM = "model_c_bire_protocol_rollout_ft_step"
REPORT_NAME = "bire_protocol_rollout_ft_report.json"
ARRAYS_NAME = "bire_protocol_rollout_ft_arrays.npz"
FIGURE_NAME = "model_c_bire_protocol_rollout_ft_selection.png"

#: Every training field the fine-tune is permitted to move, with the parent's
#: value and this arm's.  Anything not listed here must equal the parent's.
DECLARED_CHANGES: dict[str, tuple[Any, Any]] = {
    "batch_size": (8, BATCH_SIZE),
    "initial_learning_rate": (5.0e-4, LEARNING_RATE),
    "maximum_steps": (BASELINE_OPTIMIZER_STEP, MAXIMUM_STEPS),
    "rollout_steps": (3, ROLLOUT_STEPS),
}
#: Training fields that must be identical to the parent's, field for field.
HELD_TRAINING_FIELDS = (
    "seed",
    "optimizer",
    "adam_betas",
    "weight_decay",
    "gradient_clipping",
    "decay_fraction",
    "decay_factor",
)

validation_starts = protocol_validation_starts
validation_records = protocol_validation_records


class BireProtocolRolloutFineTuneError(RuntimeError):
    """Raised when the six-step fine-tuning arm violates its contract."""


@dataclass(frozen=True)
class BireProtocolRolloutFineTuneLossConfig(ModelCLossConfig):
    """Model C loss v1 with the rollout deepened to six steps.

    Deliberately does **not** call :meth:`ModelCLossConfig.__post_init__`: that
    validator exists to pin the certified objective to three steps, and this
    class is the one declared exception to it.  Every other coefficient is the
    v1 value, and the whole configuration is pinned by equality below, so the
    exception cannot widen into a second free parameter.
    """

    rollout_steps: int = ROLLOUT_STEPS
    rollout_weight: float = ROLLOUT_WEIGHT

    def __post_init__(self) -> None:
        expected = {
            "rollout_steps": ROLLOUT_STEPS,
            "increment_weight": INCREMENT_WEIGHT,
            "rollout_weight": ROLLOUT_WEIGHT,
            "spectral_weight": SPECTRAL_WEIGHT,
            "boundary_weight": BOUNDARY_WEIGHT,
            "spectral_bins": SPECTRAL_BINS,
            "western_boundary_width": WESTERN_BOUNDARY_WIDTH,
        }
        if self.to_dict() != expected:
            raise ValueError(
                "the rollout fine-tune loss changes only the rollout depth and weight"
            )


def fine_tune_loss_contract(config: ModelCLossConfig) -> dict[str, Any]:
    """Machine-readable semantics of the six-step objective.

    Not :func:`af_model_c.loss_contract`: that function's component strings name
    the three-step leads literally ("at 20 and 30 days", "at 10, 20, 30 days"),
    so reusing it would record a description that contradicts the configuration
    it carries.  The structure and the hashing are identical, so the two
    contracts are comparable field for field.
    """

    if not isinstance(config, BireProtocolRolloutFineTuneLossConfig):
        raise BireProtocolRolloutFineTuneError(
            "the six-step loss contract describes only the fine-tune configuration"
        )
    return {
        "version": "v1_rollout6",
        "status": "frozen_before_any_bire_protocol_rollout_ft_metric",
        "supersedes_nothing": True,
        "derived_from_loss_contract_sha256": MODEL_C_LOSS_V1_CONTRACT_SHA256,
        "state": "equal_mean_U_V_temperature_SSH_masked_relative_l2_at_10_days",
        "increment": (
            "equal_mean_group_rmse_of_10_day_increment_error_scaled_by_"
            "training_only_per_channel_increment_rms"
        ),
        "rollout": (
            "equal_mean_U_V_temperature_SSH_masked_relative_l2_at_"
            "20_30_40_50_and_60_days"
        ),
        "spectral": (
            "equal_mean_group_12_bin_amplitude_relative_l2_of_10_day_increments_"
            "on_exact_wet_rectangle_after_hann_taper_over_all_six_calls"
        ),
        "boundary": (
            "equal_mean_U_V_temperature_SSH_relative_l2_at_10_20_30_40_50_60_days_"
            "on_first_4_wet_cells_east_of_western_wall"
        ),
        "total": (
            "state + increment_weight*increment + rollout_weight*rollout + "
            "spectral_weight*spectral + boundary_weight*boundary"
        ),
        "weight_basis": (
            "v1_coefficients_retained_with_rollout_weight_raised_from_0.15_to_0.50_"
            "so_the_five_self_generated_steps_carry_weight_comparable_to_the_"
            "single_teacher_forced_step"
        ),
        "groups": {name: 0.25 for name in GROUP_SLICES},
        "config": config.to_dict(),
    }


def fine_tune_loss_contract_sha256(config: ModelCLossConfig) -> str:
    encoded = json.dumps(
        fine_tune_loss_contract(config), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def fine_tune_loss_config() -> BireProtocolRolloutFineTuneLossConfig:
    """The one explicit, immutable six-step configuration."""

    return BireProtocolRolloutFineTuneLossConfig()


#: Hash of the six-step objective, written into every checkpoint header so a
#: three-step and a six-step checkpoint can never be mistaken for each other.
FINE_TUNE_LOSS_CONTRACT_SHA256 = fine_tune_loss_contract_sha256(fine_tune_loss_config())


def _assert_only_the_declared_changes(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    """Model, split and normalizers must equal the checkpoint being fine-tuned."""

    record = contract["sources"]["parent_contract"]
    path = Path(record["path"]).resolve()
    if not path.is_file() or _file_sha256(path) != record["sha256"]:
        raise BireProtocolRolloutFineTuneError("the parent duration contract changed")
    parent = json.loads(path.read_text())
    if parent.get("version") != PARENT_VERSION:
        raise BireProtocolRolloutFineTuneError(
            "the initialization does not come from the matched-budget duration arm"
        )
    if contract["architecture"] != parent["architecture"]:
        raise BireProtocolRolloutFineTuneError(
            "a fine-tune cannot change the architecture it loads weights into"
        )
    if contract["dataset"] != parent["dataset"]:
        raise BireProtocolRolloutFineTuneError("the fine-tune must keep the parent split")
    if contract["normalization"] != parent["normalization"]:
        raise BireProtocolRolloutFineTuneError(
            "the fine-tune must keep the parent normalized state space"
        )
    for field in HELD_TRAINING_FIELDS:
        if contract["training"].get(field) != parent["training"].get(field):
            raise BireProtocolRolloutFineTuneError(
                f"the fine-tune moved a held training quantity: {field}"
            )
    for field, (parent_value, mine) in DECLARED_CHANGES.items():
        if parent["training"].get(field) != parent_value:
            raise BireProtocolRolloutFineTuneError(
                f"the parent's {field} is not {parent_value!r}"
            )
        if contract["training"].get(field) != mine:
            raise BireProtocolRolloutFineTuneError(
                f"the fine-tune's {field} is not {mine!r}"
            )
    parent_loss = parent["loss"]
    mine = contract["loss"]
    moved = [key for key in parent_loss["coefficients"] if parent_loss["coefficients"][key] != mine["coefficients"][key]]
    if moved != ["rollout"]:
        raise BireProtocolRolloutFineTuneError(
            f"the fine-tune must move exactly the rollout coefficient, moved {moved!r}"
        )
    if int(parent_loss["rollout_steps"]) != 3 or int(mine["rollout_steps"]) != ROLLOUT_STEPS:
        raise BireProtocolRolloutFineTuneError("the rollout depth is not 3 -> 6")
    return parent


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the fine-tuning contract frozen before any metric of this arm."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    architecture = contract.get("architecture", {})
    training = contract.get("training", {})
    dataset = contract.get("dataset", {})
    loss = contract.get("loss", {})
    selection = contract.get("checkpoint_selection", {})
    initialization = contract.get("initialization", {})
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
        or int(architecture.get("out_channels", -1)) != STATE_CHANNEL_COUNT
        or int(architecture.get("n_layers", -1)) != 3
        or architecture.get("local_kernel_size") is not None
        or architecture.get("positional_embedding") is not None
        or int(training.get("seed", -1)) != SEED
        or int(training.get("batch_size", -1)) != BATCH_SIZE
        or float(training.get("initial_learning_rate", -1.0)) != LEARNING_RATE
        or int(training.get("maximum_steps", -1)) != MAXIMUM_STEPS
        or tuple(training.get("checkpoint_steps", ())) != CHECKPOINT_STEPS
        or int(training.get("rollout_steps", -1)) != ROLLOUT_STEPS
        or training.get("from_scratch") is not False
        or training.get("load_optimizer_state") is not False
        or loss.get("contract_sha256") != FINE_TUNE_LOSS_CONTRACT_SHA256
        or int(loss.get("rollout_steps", -1)) != ROLLOUT_STEPS
        or contract.get("normalization", {}).get("recomputed_from")
        != "bire_protocol_train_only_0_5999"
        or initialization.get("load_only") != "model_state_dict"
        or int(initialization.get("optimizer_step", -1)) != BASELINE_OPTIMIZER_STEP
        or selection.get("rule") != "min_worst_long_climatology_ratio_subject_to_short_guard"
        or selection.get("evaluated_on") != "pooled_bire_protocol_validation_6000_6199"
        or float(selection.get("short_skill_tolerance", -1.0)) != SHORT_SKILL_TOLERANCE
        or read.get("training_state") is not True
        or read.get("validation_state") is not True
        or any(
            read.get(name) is not False
            for name in (
                "inference_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
            )
        )
    ):
        raise BireProtocolRolloutFineTuneError("Bire-protocol rollout fine-tune contract changed")
    # Parent equality first: a moved architecture is this arm's violation to
    # report, and its message names the arm, not the shared validator.
    _assert_only_the_declared_changes(contract)
    BireAlignedArchitecture(**architecture)
    if round(MAXIMUM_STEPS * float(training["decay_fraction"])) != 2880:
        raise BireProtocolRolloutFineTuneError("the decay does not fall at step 2,880")
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise BireProtocolRolloutFineTuneError(f"fine-tune source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def _verify_file(record: Mapping[str, Any], label: str) -> Path:
    path = Path(record["path"]).resolve()
    if not path.exists():
        raise BireProtocolRolloutFineTuneError(f"{label} is missing: {path}")
    digest = _file_sha256(path / ".zmetadata" if path.is_dir() else path)
    if digest != record["sha256"]:
        raise BireProtocolRolloutFineTuneError(f"{label} changed on disk: {path}")
    return path


def _verify_dataset(contract: Mapping[str, Any]) -> Path:
    record = contract["sources"]["dataset"]
    dataset = Path(record["path"]).resolve()
    if not dataset.is_dir() or _file_sha256(dataset / ".zmetadata") != record["metadata_sha256"]:
        raise BireProtocolRolloutFineTuneError("trajectory-v3 dataset source changed")
    return dataset


def reused_normalization(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Read the parent's published normalizers instead of recomputing them.

    The fine-tuned weights must see the same normalized coordinates as the
    weights they start from, so the pointwise mean and scale come from the
    parent's ``.npz`` verbatim.  The per-channel increment scale is not stored
    in that file, but it is a deterministic function of the same training block
    and the same pointwise scale, and the parent's report records the 46 values
    it used; they are read from there rather than recomputed over 18,000
    snapshots for a second time.
    """

    artifact = _verify_file(contract["sources"]["parent_normalization"], "parent normalization")
    with np.load(artifact) as stored:
        mean = np.ascontiguousarray(stored["pointwise_mean"], dtype=np.float32)
        raw_scale = np.ascontiguousarray(stored["pointwise_raw_scale"], dtype=np.float32)
        scale = np.ascontiguousarray(stored["pointwise_scale"], dtype=np.float32)
        floor = np.ascontiguousarray(stored["channel_scale_floor"], dtype=np.float32)
    report = json.loads(_verify_file(contract["sources"]["parent_report"], "parent report").read_text())
    increment = np.asarray(report["increment_scale"], dtype=np.float32)
    if (
        mean.shape[0] != STATE_CHANNEL_COUNT
        or scale.shape != mean.shape
        or floor.shape != (STATE_CHANNEL_COUNT,)
        or increment.shape != (STATE_CHANNEL_COUNT,)
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
        or not np.all(np.isfinite(increment))
        or np.any(increment <= 0.0)
    ):
        raise BireProtocolRolloutFineTuneError("the reused normalizers are not the 46-channel set")
    summary = dict(report["normalization"]["summary"])
    return {
        "mean": mean,
        "raw_scale": raw_scale,
        "scale": scale,
        "floor": floor,
        "increment_scale": increment,
        "summary": summary,
        "artifact": artifact,
        "report": report,
    }


def load_initial_state_dict(contract: Mapping[str, Any], device: Any) -> dict[str, Any]:
    """Return the parent's ``model_state_dict`` and its provenance.

    Only the weights are taken.  The optimizer state is not read: its second
    moments were accumulated under the three-step objective at 5e-4, and reusing
    them would carry that objective's curvature estimate into a differently
    weighted loss at a learning rate twenty-five times smaller.
    """

    path = _verify_file(contract["sources"]["initialization_checkpoint"], "initialization checkpoint")
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("version") != PARENT_VERSION
        or int(payload.get("optimizer_step", -1)) != BASELINE_OPTIMIZER_STEP
        or payload.get("dataset_version") != DATASET_VERSION
        or payload.get("base_loss_contract_sha256") != MODEL_C_LOSS_V1_CONTRACT_SHA256
        or payload.get("architecture") != contract["architecture"]
    ):
        raise BireProtocolRolloutFineTuneError(
            "the initialization checkpoint is not the selected step-15,360 duration model"
        )
    if "model_state_dict" not in payload:
        raise BireProtocolRolloutFineTuneError("the initialization checkpoint has no weights")
    return {
        "state_dict": payload["model_state_dict"],
        "provenance": {
            "path": str(path),
            "sha256": _file_sha256(path),
            "version": PARENT_VERSION,
            "optimizer_step": BASELINE_OPTIMIZER_STEP,
            "loaded": "model_state_dict",
            "optimizer_state_loaded": False,
            "base_loss_contract_sha256": MODEL_C_LOSS_V1_CONTRACT_SHA256,
        },
    }


def baseline_validation_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """The step-15,360 checkpoint's own metrics on the same 102 rollouts.

    Recomputing them here would be identical work: the validation records, the
    train-only climatology, the normalizers and the 360-day protocol are all
    unchanged, so the parent's summary is the comparison the acceptance gate
    wants, taken from an artifact whose SHA-256 the contract pins.
    """

    for summary in report["validation_summaries"]:
        if int(summary["optimizer_step"]) == BASELINE_OPTIMIZER_STEP:
            return dict(summary)
    raise BireProtocolRolloutFineTuneError(
        "the parent report carries no step-15,360 validation summary"
    )


def acceptance_gate(
    selected: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """The two validation-measurable conditions of the final acceptance gate.

    The 2,000-day conditions and the visual criterion are evaluated by the
    figure package, which is the only stage that runs a 2,000-day rollout; this
    block records their status as deferred rather than silently passing them.
    """

    short = {
        field: float(selected["short_auc_10_90"][field]) / float(baseline["short_auc_10_90"][field])
        for field in PRIMARY_FIELDS
    }
    worst_long = max(float(v) for v in selected["long_ratio_to_climatology"].values())
    short_pass = all(value <= SHORT_AUC_TOLERANCE_TO_BASELINE for value in short.values())
    long_pass = worst_long <= WORST_LONG_RATIO_CEILING
    return {
        "short_auc_10_90_ratio_to_baseline": short,
        "short_auc_tolerance": SHORT_AUC_TOLERANCE_TO_BASELINE,
        "short_auc_no_field_worsens_by_more_than_5_percent": bool(short_pass),
        "worst_long_ratio_to_climatology": worst_long,
        "worst_long_ratio_ceiling": WORST_LONG_RATIO_CEILING,
        "worst_long_ratio_at_or_below_ceiling": bool(long_pass),
        "validation_conditions_pass": bool(short_pass and long_pass),
        "baseline_optimizer_step": BASELINE_OPTIMIZER_STEP,
        "baseline_short_auc_10_90": {
            field: float(baseline["short_auc_10_90"][field]) for field in PRIMARY_FIELDS
        },
        "baseline_worst_long_ratio_to_climatology": max(
            float(v) for v in baseline["long_ratio_to_climatology"].values()
        ),
        "deferred_to_the_figure_package": [
            "2000_day_all_values_finite",
            "2000_day_maximum_normalized_magnitude_at_most_8",
            "2000_day_streamfunction_minimum_at_least_minus_33_sv",
            "day_2000_spatial_standard_deviation_ratio_within_0.80_1.25",
            "western_boundary_sharp_and_gyre_identifiable_by_inspection",
        ],
    }


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify the contract, the sources, the initialization and the record counts."""

    contract, resolved, digest = load_contract(contract_path)
    dataset = _verify_dataset(contract)
    group = zarr.open_consolidated(str(dataset), mode="r")
    _assert_store_is_v3(group)
    _, pair_split = store_codes()
    records = records_for_rollout_split(pair_split, 1, rollout_steps=ROLLOUT_STEPS)
    if len(records) != TRAINING_RECORDS:
        raise BireProtocolRolloutFineTuneError(
            f"the six-step training set is {len(records)} records, not {TRAINING_RECORDS}"
        )
    normalization = reused_normalization(contract)
    baseline = baseline_validation_summary(normalization["report"])
    architecture = BireAlignedArchitecture(**contract["architecture"])
    result: dict[str, Any] = {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "dataset_version": str(group.attrs["version"]),
        "split": verify_split(),
        "loss_contract_sha256": FINE_TUNE_LOSS_CONTRACT_SHA256,
        "rollout_steps": ROLLOUT_STEPS,
        "rollout_weight": ROLLOUT_WEIGHT,
        "training_rollout_records": len(records),
        "training_starts_per_regime": len(records) // len(EXPERIMENTS),
        "latest_training_start": int(max(t for _, t in records)),
        "validation_records": int(validation_records().shape[0]),
        "inference_range": list(INFERENCE_RANGE),
        "normalization_reused_from": str(normalization["artifact"]),
        "baseline_worst_long_ratio_to_climatology": max(
            float(v) for v in baseline["long_ratio_to_climatology"].values()
        ),
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }
    if torch is not None:
        device = _device("cpu")
        model = build_bire_aligned_model(architecture)
        initialization = load_initial_state_dict(contract, device)
        model.load_state_dict(initialization["state_dict"])
        result["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
        result["initialization"] = initialization["provenance"]
    return result


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    """Fine-tune the step-15,360 weights over six calls and select on validation."""

    if torch is None or DataLoader is None:  # pragma: no cover - environment dependent
        raise RuntimeError("the rollout fine-tuning arm requires PyTorch")
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
        raise FileExistsError("refusing to overwrite rollout fine-tune output")

    training = contract["training"]
    seed_everything(int(training["seed"]))
    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    _assert_store_is_v3(group)
    snapshot_split, pair_split = store_codes()
    _, _, wet_array, _, wind_mean, wind_scale = _normalizers(group)
    wet_array = np.asarray(wet_array, dtype=bool)

    normalization = reused_normalization(contract)
    point_mean = normalization["mean"]
    point_scale = normalization["scale"]
    increment_values = normalization["increment_scale"]
    baseline = baseline_validation_summary(normalization["report"])
    climatology_state, climatology_derived, climatology_days = train_only_climatology(
        state, wet_array
    )

    loss_config = fine_tune_loss_config()
    if fine_tune_loss_contract_sha256(loss_config) != FINE_TUNE_LOSS_CONTRACT_SHA256:
        raise BireProtocolRolloutFineTuneError("the six-step objective changed")
    if loss_config.rollout_steps != ROLLOUT_STEPS or loss_config.rollout_weight != ROLLOUT_WEIGHT:
        raise BireProtocolRolloutFineTuneError("the fine-tune objective is not the six-step one")

    training_records = records_for_rollout_split(
        pair_split, 1, rollout_steps=loss_config.rollout_steps
    )
    if len(training_records) != TRAINING_RECORDS:
        raise BireProtocolRolloutFineTuneError("the six-step training record count changed")
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
    initialization = load_initial_state_dict(contract, device)
    model.load_state_dict(initialization["state_dict"])
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    # A fresh Adam: no moment estimate from the three-step run is carried over.
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
    # Republished, not recomputed: the package is self-contained and the bytes
    # describe the same coordinates the initialization was trained in.
    normalization_path = scratch_tmp / NORMALIZATION_NAME
    np.savez_compressed(
        normalization_path,
        pointwise_mean=point_mean,
        pointwise_raw_scale=normalization["raw_scale"],
        pointwise_scale=point_scale,
        channel_scale_floor=normalization["floor"],
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
        # Six calls, the prediction fed back from the second onward.
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
        path = checkpoint_directory / f"{CHECKPOINT_STEM}_{step:05d}.pt"
        torch.save(
            {
                "version": VERSION,
                "optimizer_step": step,
                "fine_tune_step": step,
                "architecture": architecture.to_dict(),
                "contract": str(resolved_contract),
                "contract_sha256": contract_sha,
                "dataset_version": DATASET_VERSION,
                "base_loss_contract": fine_tune_loss_contract(loss_config),
                "base_loss_contract_sha256": FINE_TUNE_LOSS_CONTRACT_SHA256,
                "rollout_steps": loss_config.rollout_steps,
                "initialized_from": initialization["provenance"],
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
        raise BireProtocolRolloutFineTuneError("not every declared checkpoint was written")

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
    selected_summary = next(s for s in summaries if int(s["optimizer_step"]) == selected_step)
    shutil.copy2(checkpoint_directory / selected_name, scratch_tmp / "selected.pt")
    published = {
        "optimizer_step": selected_step,
        "checkpoint": str(scratch / "selected.pt"),
        "checkpoint_sha256": _file_sha256(scratch_tmp / "selected.pt"),
        "normalization": str(scratch / NORMALIZATION_NAME),
        "normalization_sha256": _file_sha256(normalization_path),
    }
    comparison = {
        int(s["optimizer_step"]): acceptance_gate(s, baseline) for s in summaries
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
        "initialization": initialization["provenance"],
        "normalization": {
            "recomputed_from": "bire_protocol_train_only_0_5999",
            "reused_without_recomputation": True,
            "reused_from": str(normalization["artifact"]),
            "summary": normalization["summary"],
            "artifact": str(scratch / NORMALIZATION_NAME),
            "artifact_sha256": _file_sha256(normalization_path),
        },
        "climatology": {"source": "per_regime_pointwise_mean_over_bire_protocol_train_only_0_5999",
                        "days_per_regime": climatology_days},
        "increment_scale": increment_values.tolist(),
        "loss": contract["loss"],
        "base_loss_contract": fine_tune_loss_contract(loss_config),
        "base_loss_contract_sha256": FINE_TUNE_LOSS_CONTRACT_SHA256,
        "optimizer": {
            "name": "adam",
            "initial_learning_rate": float(training["initial_learning_rate"]),
            "adam_betas": [float(v) for v in training["adam_betas"]],
            "weight_decay": float(training["weight_decay"]),
            "batch_size": batch_size, "gradient_clipping": False,
            "decay_step": decay_step, "decay_factor": float(training["decay_factor"]),
            "state_loaded_from_parent": False,
        },
        "counts": {
            "training_rollout_records": len(training_records),
            "training_starts_per_regime": len(training_records) // len(EXPERIMENTS),
            "latest_training_start": int(max(t for _, t in training_records)),
            "validation_records": int(records.shape[0]),
            "validation_starts_per_regime": int(validation_starts().size),
        },
        "training_history": history,
        "checkpoints": checkpoints,
        "validation_summaries": summaries,
        "baseline_validation_summary": baseline,
        "checkpoint_comparison_to_baseline": {str(k): v for k, v in comparison.items()},
        "selection_decision": decision,
        "acceptance_gate": acceptance_gate(selected_summary, baseline),
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
    decision = report["selection_decision"]
    gate = report["acceptance_gate"]
    comparison = report["checkpoint_comparison_to_baseline"]
    rows = "\n".join(
        "| {step} | {short} | {long} | {ratio} |".format(
            step=f"{int(s['optimizer_step']):,}",
            short="/".join(f"{s['short_auc_10_90'][f]:.3f}" for f in PRIMARY_FIELDS),
            long="/".join(f"{s['long_ratio_to_climatology'][f]:.3f}" for f in PRIMARY_FIELDS),
            ratio="/".join(
                f"{comparison[str(int(s['optimizer_step']))]['short_auc_10_90_ratio_to_baseline'][f]:.3f}"
                for f in PRIMARY_FIELDS
            ),
        )
        for s in report["validation_summaries"]
    )
    baseline = report["baseline_validation_summary"]
    return f"""# Six-step rollout fine-tune of the Bire-protocol step-15,360 model

Warm start from `{PARENT_VERSION}`'s selected checkpoint
(`selected.pt`, optimizer step {BASELINE_OPTIMIZER_STEP:,}); only
`model_state_dict` was loaded and the optimizer was reset.

| quantity | step-{BASELINE_OPTIMIZER_STEP:,} arm | this arm |
| --- | --- | --- |
| rollout calls | 3 | {ROLLOUT_STEPS} |
| rollout weight | 0.15 | {ROLLOUT_WEIGHT} |
| initial learning rate | 5e-4 | {LEARNING_RATE:g} |
| batch size | 8 | {BATCH_SIZE} |
| optimizer steps | {BASELINE_OPTIMIZER_STEP:,} | {MAXIMUM_STEPS:,} |
| decay step (0.75 x budget) | 11,520 | {int(report['optimizer']['decay_step']):,} |

Batch 4 over six unrolled calls carries the same activation load as batch 8 over
three ({BATCH_SIZE} x {ROLLOUT_STEPS} = 8 x 3). The architecture, Fourier modes,
static inputs, positional encoding, 46-channel output, Bire Section 3.2 split,
and train-only pointwise normalizers are unchanged and asserted field by field
against the parent contract; the normalizers are **reused from the parent
package rather than recomputed**, so the fine-tuned weights stay in exactly the
normalized coordinates they started in.

The objective adds three self-generated steps to the rollout term:

    L = L_state + 0.001 L_increment + {ROLLOUT_WEIGHT} (1/5) sum_(k=2..6) L_state^(k)
      + 1e-5 (1/6) sum_(k=1..6) L_spectral^(k) + 0.065 (1/6) sum_(k=1..6) L_boundary^(k)

with U, V, temperature and SSH weighted 0.25 each, as before. The prediction is
fed back at steps two through six; there is no teacher forcing after the initial
state. Six-step objective SHA-256: `{FINE_TUNE_LOSS_CONTRACT_SHA256}`.

Training draws {report['counts']['training_starts_per_regime']:,} starts per regime
({report['counts']['training_rollout_records']:,} pooled), the latest being
{report['counts']['latest_training_start']:,}, so every six-step target sequence stays inside
training 0--5999.

Selection is unchanged: minimise the worst 90--360-day RMSE-AUC relative to
climatology subject to each field's 10--90-day AUC staying within 5% of the best
fine-tuning checkpoint, on {report['counts']['validation_records']} pooled validation rollouts.

| step | short AUC 10--90 | long AUC / climatology | short AUC / step-{BASELINE_OPTIMIZER_STEP:,} |
| --- | --- | --- | --- |
{rows}

The step-{BASELINE_OPTIMIZER_STEP:,} baseline scores
{"/".join(f"{baseline['short_auc_10_90'][f]:.3f}" for f in PRIMARY_FIELDS)} short and
{"/".join(f"{baseline['long_ratio_to_climatology'][f]:.3f}" for f in PRIMARY_FIELDS)} long on the
same rollouts, in the order {", ".join(PRIMARY_FIELDS)}.

Selected step {selected_step_text(decision)} via `{decision['branch']}`.

Acceptance gate, validation half: no 10--90-day field worsens by more than 5%
against the baseline -- **{'pass' if gate['short_auc_no_field_worsens_by_more_than_5_percent'] else 'fail'}**;
worst 90--360-day climatology ratio {gate['worst_long_ratio_to_climatology']:.3f} <= {WORST_LONG_RATIO_CEILING}
-- **{'pass' if gate['worst_long_ratio_at_or_below_ceiling'] else 'fail'}**. The 2,000-day and
visual conditions are evaluated by the figure package, which is the only stage
that runs a 2,000-day rollout.

Training and validation only; the inference set opens through the figure
contract, S0 only.

Report content SHA-256: `{report['content_sha256']}`.
"""


def selected_step_text(decision: Mapping[str, Any]) -> str:
    return f"{int(decision['selected_optimizer_step']):,}"


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


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
