"""Canonical six-step fine-tuning pipeline for the two-in / one-out Model C.

The one-input arm learned ``x_t -> x_{t+10}``. This arm learns

    (x_{t-10}, x_t) -> x_{t+10},

and autoregresses by sliding the pair forward, so the operator is given two
consecutive time levels --- and therefore an empirical tendency
``(x_t - x_{t-10}) / 10 days`` --- instead of the current state alone. Nothing
else moves: the 32x32 modes, the trained bias-free local 3x3 branch, the
position encoder, the dataset, the normalizers, the six-step objective, the
optimizer schedule, the seed and the checkpoint-selection rule are all
inherited from the 32x32 parent this arm warm-starts.

The raw contract inherits the frozen dataset, objective, schedule and selection
fields from that parent, which is itself compact and inherits them from Y32,
which inherits them from local24. This module resolves that inheritance chain to
its root, audits it, performs the function-preserving history-channel migration,
then trains and selects the two-input arm.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import zarr

from .runtime import DataLoader, torch
from .runtime import AUDIT_TERMS, ChunkAwareBatchSampler, GROUP_SLICES, STATE_CHANNEL_COUNT, _checkpoint_state_dict, _device, _file_sha256, _json_sha256, require_model_a_runtime, seed_everything
from .dataset import DATASET_VERSION, EXPERIMENTS, HORIZON_DAYS, INFERENCE_RANGE, ModelCTwoInRolloutDataset, TRAIN_RANGE, _normalizers, records_for_two_in_rollout_split, store_codes, validation_records, validation_starts, verify, western_boundary_mask
from .objective import MODEL_C_LOSS_V1_CONTRACT_SHA256, ModelCLossConfig, model_c_loss_terms
from .model import BireAlignedDivergenceError, BireTwoInOneOutArchitecture, BireTwoInStepper, BireY32X32Architecture, CHECKPOINT_DIRECTORY, INPUT_LAG_DAYS, INPUT_STATE_COUNT, MANIFEST_NAME, PRESENT_SLICE, README_NAME, TWO_IN_EXTERNAL_INPUT_CHANNELS, TWO_IN_LIFTING_INPUT_CHANNELS, build_bire_two_in_one_out_model, migrate_y32_x32_state_dict, retained_two_in_features, two_in_state_unroll
from .validation import PRIMARY_FIELDS, _assert_store_is_v3, _plot, select_by_validation, train_only_climatology, validate_checkpoint

VERSION = "model_c_2in_1out_v1"

CONTRACT_STATUS = "frozen_before_any_model_c_2in_1out_metric"

PARENT_VERSION = "model_c_bire_protocol_rollout_ft_y32_x32_v1"

PARENT_CONTRACT_SHA256 = (
    "edb9058740e8d6fe25e2b0693d35503045a9e23452675af3b8ea317e97fcd553"
)

PARENT_MATERIALIZED_CONTRACT_SHA256 = (
    "21e663937ef638283ccd3b825abea969d69a39bcfbf5c3136f0d94fedbccc1df"
)

BASELINE_OPTIMIZER_STEP = 3840

ROLLOUT_STEPS = 6

ROLLOUT_WEIGHT = 0.50

INCREMENT_WEIGHT = 0.001

SPECTRAL_WEIGHT = 1.0e-5

BOUNDARY_WEIGHT = 0.065

SPECTRAL_BINS = 12

WESTERN_BOUNDARY_WIDTH = 4

LEARNING_RATE = 2.0e-5

BATCH_SIZE = 4

MAXIMUM_STEPS = 3840

CHECKPOINT_STEPS = (960, 1920, 2880, 3840)

SEED = 20260724

#: Ten starts per regime are lost relative to the one-input arm: days 0--9 have
#: no t - 10 history inside the record. Starts are consecutive days, not
#: multiples of the ten-day horizon, so the loss is ten days and not one start.
TRAINING_STARTS_PER_REGIME = 5930

TRAINING_RECORDS = TRAINING_STARTS_PER_REGIME * len(EXPERIMENTS)

SHORT_AUC_TOLERANCE_TO_BASELINE = 1.05

WORST_LONG_RATIO_CEILING = 0.85

LONG_AUC_TOLERANCE_TO_BASELINE = 1.0

SLUG = "model_c_2in_1out"

NORMALIZATION_NAME = f"{SLUG}_train_only_normalization.npz"

DIVERGENCE_NAME = f"{SLUG}_divergence.json"

CHECKPOINT_STEM = f"{SLUG}_step"

REPORT_NAME = f"{SLUG}_report.json"

ARRAYS_NAME = f"{SLUG}_arrays.npz"

FIGURE_NAME = f"{SLUG}_selection.png"

MODES = (32, 32)

LOCAL_KERNEL_SIZE = 3

PARENT_INPUT_STATES = 1

INHERITED_FIELDS = (
    "dataset",
    "normalization",
    "training",
    "loss",
    "checkpoint_selection",
)

#: The only architecture fields that may differ from the 32x32 parent.
DECLARED_ARCHITECTURE_CHANGES = (
    "in_channels",
    "lifting_in_channels",
    "input_states",
    "input_lag_days",
)

INPUT_MIGRATION = (
    "prepend_46_zero_history_input_channels_to_the_lifting_and_local_weights_"
    "and_copy_the_parent_state_static_and_position_columns_into_the_trailing_slice"
)

OUTPUT_ARTIFACTS = (
    REPORT_NAME,
    ARRAYS_NAME,
    FIGURE_NAME,
    README_NAME,
    MANIFEST_NAME,
)

REQUIRED_SOURCE_HASHES = frozenset(
    {
        "src/oceanfno/dataset.py",
        "src/oceanfno/model.py",
        "src/oceanfno/train.py",
    }
)

class ModelCTwoInTrainingError(RuntimeError):
    """Raised when the two-in / one-out arm violates its contract."""


# Historical callers used the generic arm name. Keep it as an exact alias so
# every canonical training failure is catchable through either public API.
BireProtocolRolloutFineTuneError = ModelCTwoInTrainingError

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

    The strings below are also *unchanged* from the one-input arm, which is the
    point: the objective this arm optimizes is byte-for-byte the incumbent's, so
    the two runs' losses are directly comparable.  Only what the model is handed
    at each call moved.
    """

    if not isinstance(config, BireProtocolRolloutFineTuneLossConfig):
        raise ModelCTwoInTrainingError(
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

FINE_TUNE_LOSS_CONTRACT_SHA256 = fine_tune_loss_contract_sha256(fine_tune_loss_config())

def _materialize(
    raw: Mapping[str, Any],
    parent: Mapping[str, Any],
    fields: Sequence[str] = INHERITED_FIELDS,
) -> dict[str, Any]:
    """Overlay a compact declaration onto its parent's resolved document.

    Everything the child states explicitly wins; the inherited blocks are always
    taken from the parent, so a child cannot silently redefine them by restating
    them with a different value.
    """

    contract = copy.deepcopy(dict(parent))
    for key, value in raw.items():
        if key not in fields:
            contract[key] = copy.deepcopy(value)
    for field in fields:
        contract[field] = copy.deepcopy(parent[field])
    return contract


def _resolve_contract(path: Path, *, depth: int = 0) -> dict[str, Any]:
    """Resolve a compact declaration against its ancestors, to the root.

    The chain this arm sits on is three deep --- 32x32 inherits from Y32, Y32
    from local24, and local24 states everything --- so comparing this arm's
    inherited fields against the raw parent bytes would compare them against
    fields that file does not contain. Each hop's bytes are pinned by the
    child's own ``sources.parent_contract.sha256``, so no level can move
    unnoticed, and a contract without ``inherit_parent_fields`` terminates the
    walk.
    """

    if depth > 8:
        raise ModelCTwoInTrainingError("the contract inheritance chain does not terminate")
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ModelCTwoInTrainingError(f"an ancestor contract is missing: {resolved}")
    raw = json.loads(resolved.read_text())
    fields = tuple(raw.get("inherit_parent_fields", ()))
    if not fields:
        return raw
    record = raw.get("sources", {}).get("parent_contract", {})
    ancestor = Path(str(record.get("path", ""))).resolve()
    if not ancestor.is_file() or _file_sha256(ancestor) != record.get("sha256"):
        raise ModelCTwoInTrainingError(
            f"the contract {resolved.name} no longer pins its own parent's bytes"
        )
    parent = _resolve_contract(ancestor, depth=depth + 1)
    for field in fields:
        if field in raw and raw[field] != parent[field]:
            raise ModelCTwoInTrainingError(
                f"{resolved.name} overrides inherited field {field}"
            )
    return _materialize(raw, parent, fields)


def _parent_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the 32x32 parent, including its own two levels of inheritance."""

    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or _file_sha256(path) != record.get("sha256")
        or record.get("sha256") != PARENT_CONTRACT_SHA256
    ):
        raise ModelCTwoInTrainingError("the selected 32x32 parent contract changed")
    raw_parent = json.loads(path.read_text())
    if raw_parent.get("version") != PARENT_VERSION:
        raise ModelCTwoInTrainingError("the parent is not the 32x32 one-input arm")
    if tuple(raw_parent.get("inherit_parent_fields", ())) != INHERITED_FIELDS:
        raise ModelCTwoInTrainingError(
            "the 32x32 parent's own inherited-field declaration changed"
        )
    parent = _resolve_contract(path)
    materialized = _json_sha256(parent)
    if (
        materialized != PARENT_MATERIALIZED_CONTRACT_SHA256
        or record.get("materialized_sha256") != materialized
    ):
        raise ModelCTwoInTrainingError("the resolved 32x32 parent contract changed")
    return parent


def _assert_only_the_declared_changes(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Prove that temporal context is the sole scientific change."""

    parent = _parent_contract(contract["sources"]["parent_contract"])
    parent_architecture = dict(parent["architecture"])
    if (
        tuple(parent_architecture.get("n_modes", ())) != MODES
        or parent_architecture.get("local_kernel_size") != LOCAL_KERNEL_SIZE
        or "input_states" in parent_architecture
    ):
        raise ModelCTwoInTrainingError(
            "the parent is not the audited one-input 32x32 model with a trained 3x3 branch"
        )
    BireY32X32Architecture(**parent_architecture)
    expected = dict(parent_architecture)
    expected["in_channels"] = TWO_IN_EXTERNAL_INPUT_CHANNELS
    expected["lifting_in_channels"] = TWO_IN_LIFTING_INPUT_CHANNELS
    expected["input_states"] = INPUT_STATE_COUNT
    expected["input_lag_days"] = INPUT_LAG_DAYS
    if contract.get("architecture") != expected:
        raise ModelCTwoInTrainingError(
            "only the input contract may move from the 32x32 parent: "
            f"{', '.join(DECLARED_ARCHITECTURE_CHANGES)}"
        )
    for field in INHERITED_FIELDS:
        if contract.get(field) != parent.get(field):
            raise ModelCTwoInTrainingError(
                f"the two-input arm moved the parent's {field} contract"
            )
    return parent

def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the two-in / one-out declaration and its pinned inherited fields."""

    resolved = Path(path).resolve()
    raw = json.loads(resolved.read_text())
    if tuple(raw.get("inherit_parent_fields", ())) != INHERITED_FIELDS:
        raise ModelCTwoInTrainingError("the exact inherited-field declaration changed")
    sources = raw.get("sources", {})
    parent = _parent_contract(sources.get("parent_contract", {}))
    for field in INHERITED_FIELDS:
        if field in raw and raw[field] != parent[field]:
            raise ModelCTwoInTrainingError(
                f"the raw two-input declaration overrides inherited field {field}"
            )

    contract = _materialize(raw, parent)

    architecture = contract.get("architecture", {})
    training = contract.get("training", {})
    initialization = contract.get("initialization", {})
    read = contract.get("read_contract", {})
    output = contract.get("output", {})
    report_record = sources.get("parent_report", {})
    report_path = Path(str(report_record.get("path", ""))).resolve()
    if not report_path.is_file() or _file_sha256(report_path) != report_record.get(
        "sha256"
    ):
        raise ModelCTwoInTrainingError("the 32x32 parent report changed")
    parent_report = json.loads(report_path.read_text())
    published = parent_report.get("published_checkpoint", {})
    normalization_record = sources.get("parent_normalization", {})
    checkpoint_record = sources.get("initialization_checkpoint", {})
    hashes = contract.get("source_hashes", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or tuple(architecture.get("n_modes", ())) != MODES
        or architecture.get("local_kernel_size") != LOCAL_KERNEL_SIZE
        or int(architecture.get("input_states", -1)) != INPUT_STATE_COUNT
        or int(architecture.get("input_lag_days", -1)) != INPUT_LAG_DAYS
        or int(architecture.get("in_channels", -1))
        != TWO_IN_EXTERNAL_INPUT_CHANNELS
        or int(architecture.get("lifting_in_channels", -1))
        != TWO_IN_LIFTING_INPUT_CHANNELS
        or initialization.get("load_only") != "model_state_dict"
        or int(initialization.get("optimizer_step", -1))
        != BASELINE_OPTIMIZER_STEP
        or initialization.get("version") != PARENT_VERSION
        or initialization.get("input_migration") != INPUT_MIGRATION
        or initialization.get("history_branch_initialization") != "zeros"
        or initialization.get("local_branch_initialization")
        != "copied_from_parent"
        or initialization.get("local_branch_bias") is not False
        or initialization.get("optimizer_state_loaded") is not False
        or initialization.get("normalization_reused") is not True
        or training.get("load_optimizer_state") is not False
        or training.get("from_scratch") is not False
        or int(training.get("rollout_steps", -1)) != ROLLOUT_STEPS
        or tuple(training.get("checkpoint_steps", ())) != CHECKPOINT_STEPS
        or int(training.get("maximum_steps", -1)) != MAXIMUM_STEPS
        or contract.get("loss", {}).get("contract_sha256")
        != FINE_TUNE_LOSS_CONTRACT_SHA256
        or read != parent.get("read_contract")
        or initialization.get("checkpoint") != checkpoint_record.get("path")
        or not str(output.get("project_root", "")).endswith(f"{SLUG}_v1")
        or not str(output.get("scratch_root", "")).endswith(f"{SLUG}_v1")
        or output.get("project_root") == parent.get("output", {}).get("project_root")
        or output.get("scratch_root") == parent.get("output", {}).get("scratch_root")
        or tuple(output.get("artifacts", ())) != OUTPUT_ARTIFACTS
        or not REQUIRED_SOURCE_HASHES.issubset(hashes)
        or parent_report.get("version") != PARENT_VERSION
        or int(
            parent_report.get("selection_decision", {}).get(
                "selected_optimizer_step", -1
            )
        )
        != BASELINE_OPTIMIZER_STEP
        or int(published.get("optimizer_step", -1)) != BASELINE_OPTIMIZER_STEP
        or published.get("checkpoint") != checkpoint_record.get("path")
        or published.get("checkpoint_sha256") != checkpoint_record.get("sha256")
        or published.get("normalization") != normalization_record.get("path")
        or published.get("normalization_sha256")
        != normalization_record.get("sha256")
    ):
        raise ModelCTwoInTrainingError("the two-in / one-out training contract changed")

    _assert_only_the_declared_changes(contract)
    BireTwoInOneOutArchitecture(**architecture)
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in hashes.items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ModelCTwoInTrainingError(f"two-input source changed: {source}")
    return contract, resolved, _file_sha256(resolved)

def _verify_file(record: Mapping[str, Any], label: str) -> Path:
    path = Path(record["path"]).resolve()
    if not path.exists():
        raise ModelCTwoInTrainingError(f"{label} is missing: {path}")
    digest = _file_sha256(path / ".zmetadata" if path.is_dir() else path)
    if digest != record["sha256"]:
        raise ModelCTwoInTrainingError(f"{label} changed on disk: {path}")
    return path

def _verify_dataset(contract: Mapping[str, Any]) -> Path:
    record = contract["sources"]["dataset"]
    dataset = Path(record["path"]).resolve()
    if not dataset.is_dir() or _file_sha256(dataset / ".zmetadata") != record["metadata_sha256"]:
        raise ModelCTwoInTrainingError("trajectory-v3 dataset source changed")
    return dataset

def reused_normalization(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Read the parent's published normalizers instead of recomputing them.

    The fine-tuned weights must see the same normalized coordinates as the
    weights they start from, so the pointwise mean and scale come from the
    parent's ``.npz`` verbatim.  The history state is normalized with those same
    pointwise fields, which is what makes ``x_t - x_{t-10}`` a meaningful
    difference inside the network.  The per-channel increment scale is not
    stored in that file, but it is a deterministic function of the same training
    block and the same pointwise scale, and the parent's report records the 46
    values it used; they are read from there rather than recomputed over 18,000
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
        raise ModelCTwoInTrainingError("the reused normalizers are not the 46-channel set")
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

def load_initial_state_dict(
    contract: Mapping[str, Any],
    device: Any,
    target_model: Any,
) -> dict[str, Any]:
    """Load and audit the function-preserving one-input -> two-input migration."""

    path = _verify_file(contract["sources"]["initialization_checkpoint"], "initialization checkpoint")
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("version") != PARENT_VERSION
        or int(payload.get("optimizer_step", -1)) != BASELINE_OPTIMIZER_STEP
        or payload.get("dataset_version") != DATASET_VERSION
        or payload.get("base_loss_contract_sha256") != FINE_TUNE_LOSS_CONTRACT_SHA256
    ):
        raise ModelCTwoInTrainingError(
            "the initialization checkpoint is not the 32x32 one-input model"
        )
    if "model_state_dict" not in payload:
        raise ModelCTwoInTrainingError("the initialization checkpoint has no weights")
    parent_contract = _assert_only_the_declared_changes(contract)
    if payload.get("architecture") != parent_contract["architecture"]:
        raise ModelCTwoInTrainingError(
            "the initialization architecture does not match its archived contract"
        )
    migration = migrate_y32_x32_state_dict(payload["model_state_dict"], target_model)
    return {
        "state_dict": migration["state_dict"],
        "provenance": {
            "path": str(path),
            "sha256": _file_sha256(path),
            "version": PARENT_VERSION,
            "optimizer_step": BASELINE_OPTIMIZER_STEP,
            "loaded": "model_state_dict",
            "optimizer_state_loaded": False,
            "base_loss_contract_sha256": FINE_TUNE_LOSS_CONTRACT_SHA256,
            "architecture_migration": migration["provenance"],
        },
    }

def baseline_validation_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """The incumbent one-input checkpoint's metrics on the same 102 rollouts.

    Recomputing them here would be identical work: the validation records, the
    train-only climatology, the normalizers and the 360-day protocol are all
    unchanged, so the parent's summary is the comparison the acceptance gate
    wants, taken from an artifact whose SHA-256 the contract pins.
    """

    for summary in report["validation_summaries"]:
        if int(summary["optimizer_step"]) == BASELINE_OPTIMIZER_STEP:
            return dict(summary)
    raise ModelCTwoInTrainingError(
        "the parent report carries no selected step-3,840 validation summary"
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
    baseline_worst_long = max(
        float(v) for v in baseline["long_ratio_to_climatology"].values()
    )
    short_pass = all(value <= SHORT_AUC_TOLERANCE_TO_BASELINE for value in short.values())
    long_pass = worst_long <= WORST_LONG_RATIO_CEILING
    no_worse_than_parent = worst_long <= baseline_worst_long
    return {
        "short_auc_10_90_ratio_to_baseline": short,
        "short_auc_tolerance": SHORT_AUC_TOLERANCE_TO_BASELINE,
        "short_auc_no_field_worsens_by_more_than_5_percent": bool(short_pass),
        "worst_long_ratio_to_climatology": worst_long,
        "worst_long_ratio_ceiling": WORST_LONG_RATIO_CEILING,
        "worst_long_ratio_at_or_below_ceiling": bool(long_pass),
        "worst_long_ratio_no_worse_than_parent": bool(no_worse_than_parent),
        "validation_conditions_pass": bool(
            short_pass and long_pass and no_worse_than_parent
        ),
        "baseline_optimizer_step": BASELINE_OPTIMIZER_STEP,
        "baseline_short_auc_10_90": {
            field: float(baseline["short_auc_10_90"][field]) for field in PRIMARY_FIELDS
        },
        "baseline_worst_long_ratio_to_climatology": baseline_worst_long,
        "deferred_to_the_figure_package": [
            "2000_day_all_values_finite",
            "2000_day_maximum_normalized_magnitude_at_most_8",
            "2000_day_streamfunction_minimum_at_least_minus_33_sv",
            "day_2000_streamfunction_anomaly_rms_ratio_near_one",
            "day_2000_western_band_to_interior_anomaly_ratio_controlled",
            "day_2000_directional_spectrum_and_gradient_sharpness",
            "western_boundary_sharp_and_gyre_identifiable_by_inspection",
        ],
    }


def fine_tune_split_summary() -> dict[str, Any]:
    """Return the shared split summary with the two-input start bounds."""

    summary = dict(verify())
    summary["training_rollout_steps"] = ROLLOUT_STEPS
    summary["input_states"] = INPUT_STATE_COUNT
    summary["input_lag_days"] = INPUT_LAG_DAYS
    summary["earliest_training_rollout_start"] = INPUT_LAG_DAYS
    summary["latest_training_rollout_start"] = TRAIN_RANGE[1] - 1 - 10 * ROLLOUT_STEPS
    summary["history_note"] = (
        "the record's time index is still the present state t; the pair adds the "
        "t-10 state as an initial condition, so no target moves and days 0-9 are "
        "the only training starts the history requirement removes"
    )
    summary["validation_history_note"] = (
        "a validation rollout starting at day 6000 reads day 5990 as its second "
        "initial condition; that day is model-visible training input, never a "
        "scored target, exactly as persistence reads the initial state"
    )
    return summary


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify the contract, the sources, the initialization and the record counts."""

    contract, resolved, digest = load_contract(contract_path)
    dataset = _verify_dataset(contract)
    group = zarr.open_consolidated(str(dataset), mode="r")
    _assert_store_is_v3(group)
    _, pair_split = store_codes()
    records = records_for_two_in_rollout_split(
        pair_split, 1, rollout_steps=ROLLOUT_STEPS
    )
    if len(records) != TRAINING_RECORDS:
        raise ModelCTwoInTrainingError(
            f"the two-input training set is {len(records)} records, not {TRAINING_RECORDS}"
        )
    normalization = reused_normalization(contract)
    baseline = baseline_validation_summary(normalization["report"])
    architecture = BireTwoInOneOutArchitecture(**contract["architecture"])
    result: dict[str, Any] = {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "dataset_version": str(group.attrs["version"]),
        "split": fine_tune_split_summary(),
        "loss_contract_sha256": FINE_TUNE_LOSS_CONTRACT_SHA256,
        "rollout_steps": ROLLOUT_STEPS,
        "rollout_weight": ROLLOUT_WEIGHT,
        "input_states": INPUT_STATE_COUNT,
        "input_lag_days": INPUT_LAG_DAYS,
        "external_input_channels": TWO_IN_EXTERNAL_INPUT_CHANNELS,
        "lifting_input_channels": TWO_IN_LIFTING_INPUT_CHANNELS,
        "training_rollout_records": len(records),
        "training_starts_per_regime": len(records) // len(EXPERIMENTS),
        "earliest_training_start": int(min(t for _, t in records)),
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
        model = build_bire_two_in_one_out_model(architecture)
        initialization = load_initial_state_dict(contract, device, model)
        model.load_state_dict(initialization["state_dict"])
        result["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
        result["initialization"] = initialization["provenance"]
    return result

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


def _readme(report: Mapping[str, Any]) -> str:
    """Describe the canonical two-in / one-out training package."""

    decision = report["selection_decision"]
    gate = report["acceptance_gate"]
    rows = "\n".join(
        "| {step:,} | {short} | {long} |".format(
            step=int(summary["optimizer_step"]),
            short=" / ".join(
                f"{summary['short_auc_10_90'][field]:.3f}"
                for field in PRIMARY_FIELDS
            ),
            long=" / ".join(
                f"{summary['long_ratio_to_climatology'][field]:.3f}"
                for field in PRIMARY_FIELDS
            ),
        )
        for summary in report["validation_summaries"]
    )
    baseline = report["baseline_validation_summary"]
    return f"""# Two-in / one-out continuation of the 32 x 32 Model C

This model warm-starts `{PARENT_VERSION}` at optimizer step
{BASELINE_OPTIMIZER_STEP:,}. Only the input contract changes:

    one-in / one-out:   x_t                -> x_(t+10)
    two-in / one-out:  (x_(t-10), x_t)     -> x_(t+10)

and autoregression slides the pair forward, so a self-generated rollout reads
`(x_t, xhat_(t+10)) -> xhat_(t+20)` and then
`(xhat_(t+10), xhat_(t+20)) -> xhat_(t+30)`. The pair gives the operator an
empirical tendency `(x_t - x_(t-10)) / 10 days`, which is the multistep idea
behind Adams--Bashforth in spirit; it is not AB-II's algebra, since MITgcm
extrapolates from two stored *tendencies* rather than two states.

The external input block therefore grows from 46 + 3 = 49 channels to
2 x 46 + 3 = {TWO_IN_EXTERNAL_INPUT_CHANNELS}, and lifting from 51 to
{TWO_IN_LIFTING_INPUT_CHANNELS}. The two input-facing tensors --- the lifting
weight and the bias-free local 3 x 3 weight --- gain 46 leading input channels
that begin at exact zero, and the parent's state, static and position columns
are copied into the trailing slice. At initialization the model therefore
*ignores* the history state and reproduces the parent's map on `x_t` for any
history whatsoever, up to float32 summation order.

The 32 x 32 Fourier modes, the trained local branch, the deterministic
sine/cosine position encoder, the dataset, the normalizers, the six-step
autoregressive loss, the optimizer reset, the schedule, the seed, the validation
starts and the checkpoint-selection rule are inherited unchanged, so temporal
context is the only thing this arm tests.

Training draws {report['counts']['training_starts_per_regime']:,} starts per
regime ({report['counts']['training_rollout_records']:,} pooled), from
{report['counts']['earliest_training_start']:,} to
{report['counts']['latest_training_start']:,}: days 0--9 are the only starts the
one-input arm had that the history requirement removes, and no target moved.

| step | short AUC 10--90 (speed / SST / pressure) | long / climatology |
| --- | --- | --- |
{rows}

The step-{BASELINE_OPTIMIZER_STEP:,} one-input baseline scores
{" / ".join(f"{baseline['short_auc_10_90'][f]:.3f}" for f in PRIMARY_FIELDS)} short and
{" / ".join(f"{baseline['long_ratio_to_climatology'][f]:.3f}" for f in PRIMARY_FIELDS)} long on the
same {report['counts']['validation_records']} pooled rollouts, in the order
{", ".join(PRIMARY_FIELDS)}.

Selected step {int(decision['selected_optimizer_step']):,} via
`{decision['branch']}`. Validation gate:
**{'pass' if gate['validation_conditions_pass'] else 'fail'}**. The S0-only
2,000-day figures and anomaly diagnostics are evaluated by the canonical
held-inference package.

Parameter count: {int(report['parameter_count']):,}.
Report content SHA-256: `{report['content_sha256']}`.
"""

def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    """Fine-tune the function-preserving two-input migration and select it."""

    if torch is None or DataLoader is None:  # pragma: no cover - environment dependent
        raise RuntimeError("the rollout fine-tuning arm requires PyTorch")
    require_model_a_runtime()
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    dataset = _verify_dataset(contract)
    split_summary = fine_tune_split_summary()
    scratch = Path(contract["output"]["scratch_root"]).resolve()
    project = Path(contract["output"]["project_root"]).resolve()
    scratch_tmp = scratch.with_name(scratch.name + ".tmp")
    project_tmp = project.with_name(project.name + ".tmp")
    if any(p.exists() for p in (scratch, project, scratch_tmp, project_tmp)):
        raise FileExistsError("refusing to overwrite two-input fine-tune output")

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
        raise ModelCTwoInTrainingError("the six-step objective changed")
    if loss_config.rollout_steps != ROLLOUT_STEPS or loss_config.rollout_weight != ROLLOUT_WEIGHT:
        raise ModelCTwoInTrainingError("the fine-tune objective is not the six-step one")

    training_records = records_for_two_in_rollout_split(
        pair_split, 1, rollout_steps=loss_config.rollout_steps
    )
    if len(training_records) != TRAINING_RECORDS:
        raise ModelCTwoInTrainingError("the two-input training record count changed")
    training_dataset = ModelCTwoInRolloutDataset(
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
    architecture = BireTwoInOneOutArchitecture(**contract["architecture"])
    model = build_bire_two_in_one_out_model(architecture).to(device)
    initialization = load_initial_state_dict(contract, device, model)
    model.load_state_dict(initialization["state_dict"])
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    # A fresh Adam keeps the temporal-context ablation independent of parent moments.
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
        features = retained_two_in_features(raw_features)
        model.train()
        # Six calls; the pair is self-generated from the second onward.
        predictions = two_in_state_unroll(model, features, wet, loss_config.rollout_steps)
        terms = model_c_loss_terms(
            predictions, futures, features[:, PRESENT_SLICE],
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
                "input_states": INPUT_STATE_COUNT,
                "input_lag_days": INPUT_LAG_DAYS,
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
        raise ModelCTwoInTrainingError("not every declared checkpoint was written")

    records = validation_records()
    summaries = []
    evaluated_arrays = []
    for record in checkpoints:
        payload = torch.load(
            checkpoint_directory / record["checkpoint"], map_location=device, weights_only=False
        )
        probe = build_bire_two_in_one_out_model(architecture).to(device)
        probe.load_state_dict(payload["model_state_dict"])
        probe.eval()
        stepper = BireTwoInStepper(
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
        "temporal_context": {
            "input_states": INPUT_STATE_COUNT,
            "input_lag_days": INPUT_LAG_DAYS,
            "map": "(x_t_minus_10, x_t) -> x_t_plus_10",
            "autoregression": "the_pair_slides_forward_so_no_step_after_the_first_sees_truth",
            "external_input_channels": TWO_IN_EXTERNAL_INPUT_CHANNELS,
            "lifting_input_channels": TWO_IN_LIFTING_INPUT_CHANNELS,
        },
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
            "earliest_training_start": int(min(t for _, t in training_records)),
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

def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = preflight(args.contract) if args.command == "preflight" else run(
        args.contract, device_name=args.device
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0

if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
