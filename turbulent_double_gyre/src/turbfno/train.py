"""Train the production emulator from random initialization and select one checkpoint.

    F_theta: [x_t (46), S (5)] -> x_{t+10} (46)

Six-step autoregressive training with the complete physics-aware objective active
from optimizer step one. Nothing is inherited: the weights are randomly
initialized, the pointwise normalizers and the increment RMS scales are
recomputed over training days 0--5999 of S0/S1/S2, and Adam starts cold. There is
no parent checkpoint, no migration, no staged fine-tuning.

**This arm constrains the operator, not the trajectory.** The per-mode spectral
cap remains active: the largest singular value of the ``128 x 128`` complex
channel-mixing matrix at *every* Fourier mode is capped at one,

    R_k <- R_k * min(1, 1 / sigma_max(R_k)),

following McCabe et al., arXiv:2306.10619. Those matrices hold 99.46 % of the
turbulent model's parameters.

The production update is confined to the 10 % latent FFT halo. NeuralOperator's
constant-zero halo produced a sharp hidden-to-zero edge; the first trained
turbulent rollout then developed narrowband zonal stripes next to the boundary.
A zero-shot ablation left the stripe fraction unchanged when the useful local
3x3 branch was disabled (0.478 to 0.479), but reduced it to 0.261 when the
lifted field was replicated into the halo and brought smoothly to zero with a
raised-cosine taper. Architecture size, data, split, objective, schedule,
optimizer, batch and seed stay fixed, and the model is retrained from scratch.

The contraction *penalty* of the two preceding arms is removed entirely, so this
run answers one question: does controlling the spectral operator itself fix the
instability? Perturbation growth is still measured --- ``g`` per training window
and ``lambda_hat`` per checkpoint --- but contributes no gradient.

Preceding measurements of ``lambda_hat`` at the selected checkpoint: v1 1.0168
(no constraint), v2 1.01126 (squared hinge, 0.01), v3 1.01361 (log hinge, 0.1).

Schedule::

    Adam, batch 8 (microbatch 1 x accumulation 8), betas (0.9, 0.95)
    lr 5e-4 for steps 1--5760, 1e-4 for steps 5761--7680
    checkpoints at 1,920 / 3,840 / 5,760 / 7,680

7,680 optimizer steps x 8 samples x 6 autoregressive calls = 368,640 state
transitions, matching the earlier full-training exposure at twice the unroll
depth.

Entry points::

    python -m turbfno.train preflight --contract config/...json
    python -m turbfno.train run       --contract config/...json [--device cuda]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from .runtime import (
    AUDIT_TERMS,
    ChunkAwareBatchSampler,
    DataLoader,
    STATE_CHANNEL_COUNT,
    _device,
    _file_sha256,
    _json_sha256,
    json_safe,
    require_runtime,
    seed_everything,
    torch,
)
from .distributed import (
    ShardedBatchSampler,
    initialize as initialize_distributed,
    require_divisible,
)
from .dataset import (
    DATASET_VERSION,
    EXPERIMENTS,
    GRID_SHAPE,
    INFERENCE_RANGE,
    RolloutDataset,
    STATIC_FEATURES,
    TRAIN_CODE,
    TRAIN_RANGE,
    assert_store_is_v3,
    normalizer_summary,
    read_mitgcm_2d,
    records_for_rollout_split,
    static_block,
    store_codes,
    store_wind_normalization,
    training_increment_scale,
    training_pointwise_normalizers,
    validation_records,
    validation_starts,
    verify,
    western_boundary_mask,
)
from .objective import (
    LOSS_CONTRACT_SHA256,
    ProductionLossConfig,
    loss_contract,
    production_loss_config,
    production_loss_terms,
)
from .model import (
    CHECKPOINT_DIRECTORY,
    DivergenceError,
    EXPECTED_PARAMETER_COUNT,
    MANIFEST_NAME,
    README_NAME,
    ProductionArchitecture,
    ProductionStepper,
    build_model,
    parameter_count,
    state_unroll,
)
from .barotropic_transport import barotropic_transport_relative_l2
from .continuity import ContinuityContext, continuity_relative_l2
from .pressure_gradient import PressureGradientContext, pressure_gradient_relative_l2
from .perturbation_growth import (
    DIAGNOSTIC_CALLS,
    EPSILON_RELATIVE,
    GROWTH_RATE_CEILING,
    growth_rate_summary,
    initial_direction,
    single_call_amplification,
)
from .spectral_norm import (
    POWER_ITERATIONS,
    WARMUP_ITERATIONS,
    apply_mode_spectral_norm,
    materialized_state_dict,
    mode_sigma_summary,
)
from .validation import (
    PRIMARY_FIELDS,
    _plot,
    select_by_validation,
    train_only_climatology,
    validate_checkpoint,
)

VERSION = "turb_forward_control_v1"

SLUG = "turb_forward_control_v1"

#: The unconstrained arm this is measured against. Not a weight ancestor --- this
#: run is from scratch --- only the number the change is aimed at.
REFERENCE_VERSION = "model_c_production_1in_1out_v1"

REFERENCE_GROWTH_RATE = 1.0168

#: Growth rates the two contraction-penalty arms reached, for the record.
PENALTY_ARM_GROWTH_RATES = {
    "model_c_production_1in_1out_v2": 1.01126,
    "model_c_production_1in_1out_v3": 1.01361,
}

CONTRACT_STATUS = "frozen_for_the_production_padding_revision"

PREVIOUS_PADDING_MODE = "constant_zero"

PREVIOUS_STRIPE_POWER_FRACTION = 0.47765

TAPERED_PADDING_STRIPE_POWER_FRACTION = 0.26071

ROLLOUT_STEPS = 6

#: The effective batch, unchanged from the 1-degree arm. It is realized as
#: ``MICROBATCH_SIZE`` samples accumulated ``GRADIENT_ACCUMULATION_STEPS``
#: times, which is arithmetically identical --- every loss term is a per-sample
#: quantity averaged over the batch and the microbatches are exactly equal in
#: size. Under a data-parallel launch the accumulation is divided among the
#: ranks and the gradients averaged, which changes nothing but the summation
#: order; see :mod:`turbfno.distributed`.
BATCH_SIZE = 8

#: One sample. The six-step recurrent activation load at 248 x 248 with 64 x 64
#: modes peaks at 20.8 GiB on a 32 GB V100; two samples need roughly 35 GiB, so
#: this is forced by the hardware rather than chosen.
MICROBATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICROBATCH_SIZE

#: Per-mode matrices the spectral cap holds: three blocks x 64 * (64//2 + 1).
SPECTRAL_MATRICES_TOTAL = 3 * 64 * (64 // 2 + 1)

LEARNING_RATE = 5.0e-4

ADAM_BETAS = (0.9, 0.95)

WEIGHT_DECAY = 0.0

MAXIMUM_STEPS = 7680

CHECKPOINT_STEPS = (1920, 3840, 5760, 7680)

DECAY_FRACTION = 0.75

DECAY_FACTOR = 0.2

SEED = 20260911

#: One time level, so no start is lost to a history requirement: every day from
#: 0 to 5,939 admits the full six-step target sequence inside the training block.
TRAINING_STARTS_PER_REGIME = 5940

TRAINING_RECORDS = TRAINING_STARTS_PER_REGIME * len(EXPERIMENTS)

STATE_TRANSITIONS = MAXIMUM_STEPS * BATCH_SIZE * ROLLOUT_STEPS

SHORT_AUC_TOLERANCE = 1.05

WORST_LONG_RATIO_CEILING = 0.85

NORMALIZATION_NAME = f"{SLUG}_train_only_normalization.npz"

DIVERGENCE_NAME = f"{SLUG}_divergence.json"

CHECKPOINT_STEM = f"{SLUG}_step"

REPORT_NAME = f"{SLUG}_report.json"

ARRAYS_NAME = f"{SLUG}_arrays.npz"

FIGURE_NAME = f"{SLUG}_selection.png"

OUTPUT_ARTIFACTS = (REPORT_NAME, ARRAYS_NAME, FIGURE_NAME, README_NAME, MANIFEST_NAME)

REQUIRED_SOURCE_HASHES = frozenset(
    {
        "src/turbfno/barotropic_transport.py",
        "src/turbfno/continuity.py",
        "src/turbfno/dataset.py",
        "src/turbfno/distributed.py",
        "src/turbfno/model.py",
        "src/turbfno/objective.py",
        "src/turbfno/pressure_gradient.py",
        "src/turbfno/perturbation_growth.py",
        "src/turbfno/runtime.py",
        "src/turbfno/spectral_norm.py",
        "src/turbfno/train.py",
        "src/turbfno/validation.py",
    }
)

#: The simulation's own inputs three of the five static channels are built from.
#: They are not in the trajectory store, so each is pinned by digest and the
#: declaration is parsed as well.
REQUIRED_MITGCM_SOURCES = (
    "mitgcm_zonal_spacing",
    "mitgcm_sst_relaxation",
    "mitgcm_declaration",
)


class TrainingContractError(RuntimeError):
    """Raised when the production training contract is violated."""


def _verify_file(record: Mapping[str, Any], label: str) -> Path:
    path = Path(record["path"]).resolve()
    if not path.exists():
        raise TrainingContractError(f"{label} is missing: {path}")
    digest = _file_sha256(path / ".zmetadata" if path.is_dir() else path)
    if digest != record["sha256"]:
        raise TrainingContractError(f"{label} changed on disk: {path}")
    return path


def _verify_dataset(contract: Mapping[str, Any]) -> Path:
    record = contract["sources"]["dataset"]
    dataset = Path(record["path"]).resolve()
    if (
        not dataset.is_dir()
        or _file_sha256(dataset / ".zmetadata") != record["metadata_sha256"]
    ):
        raise TrainingContractError("the trajectory-v3 dataset source changed")
    return dataset


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the self-contained production declaration and audit every field.

    The contract states everything: there is no parent to inherit from and no
    inheritance machinery to resolve. What is checked here is that the file
    declares exactly the frozen architecture, schedule, objective and
    from-scratch initialization this module implements, and that every pinned
    input still hashes to the recorded bytes.
    """

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    architecture = contract.get("architecture", {})
    initialization = contract.get("initialization", {})
    training = contract.get("training", {})
    loss = contract.get("loss", {})
    selection = contract.get("checkpoint_selection", {})
    normalization = contract.get("normalization", {})
    output = contract.get("output", {})
    sources = contract.get("sources", {})
    hashes = contract.get("source_hashes", {})
    config = production_loss_config()

    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or architecture != ProductionArchitecture().to_dict()
    ):
        raise TrainingContractError("the production architecture declaration changed")

    # The single most important property of this arm: nothing is inherited.
    if (
        initialization.get("from_scratch") is not True
        or initialization.get("load_model_state") is not False
        or initialization.get("load_optimizer_state") is not False
        or initialization.get("normalization_reused") is not False
        or initialization.get("parent_checkpoint") is not None
        or initialization.get("local_branch_initialization") != "zeros"
        or initialization.get("local_branch_bias") is not False
        or "checkpoint" in initialization
    ):
        raise TrainingContractError(
            "the production arm trains from random initialization; the contract "
            "must declare no checkpoint, no parent and no reused normalization"
        )
    for forbidden in ("parent_checkpoint", "parent_normalization", "parent_report",
                      "initialization_checkpoint"):
        if forbidden in sources:
            raise TrainingContractError(
                f"the production arm has no lineage; remove sources.{forbidden}"
            )

    if (
        int(training.get("rollout_steps", -1)) != ROLLOUT_STEPS
        or int(training.get("batch_size", -1)) != BATCH_SIZE
        or int(training.get("microbatch_size", -1)) != MICROBATCH_SIZE
        or int(training.get("gradient_accumulation_steps", -1))
        != GRADIENT_ACCUMULATION_STEPS
        or float(training.get("initial_learning_rate", -1.0)) != LEARNING_RATE
        or tuple(float(v) for v in training.get("adam_betas", ())) != ADAM_BETAS
        or float(training.get("weight_decay", -1.0)) != WEIGHT_DECAY
        or int(training.get("maximum_steps", -1)) != MAXIMUM_STEPS
        or tuple(training.get("checkpoint_steps", ())) != CHECKPOINT_STEPS
        or float(training.get("decay_fraction", -1.0)) != DECAY_FRACTION
        or float(training.get("decay_factor", -1.0)) != DECAY_FACTOR
        or int(training.get("seed", -1)) != SEED
        or training.get("optimizer") != "adam"
        or training.get("gradient_clipping") is not False
        or int(training.get("state_transitions", -1)) != STATE_TRANSITIONS
    ):
        raise TrainingContractError("the production optimizer schedule changed")

    if (
        loss.get("contract_sha256") != LOSS_CONTRACT_SHA256
        or loss.get("all_terms_active_from_step_1") is not True
        or {
            key: float(loss.get(key, -1.0))
            for key in (
                "increment_weight",
                "rollout_weight",
                "spectral_weight",
                "boundary_weight",
                "pressure_gradient_weight",
                "continuity_weight",
                "barotropic_transport_weight",
            )
        }
        != {
            key: float(getattr(config, key))
            for key in (
                "increment_weight",
                "rollout_weight",
                "spectral_weight",
                "boundary_weight",
                "pressure_gradient_weight",
                "continuity_weight",
                "barotropic_transport_weight",
            )
        }
        or loss.get("contraction_penalty") is not False
    ):
        raise TrainingContractError("the production objective changed")

    spectral = contract.get("spectral_normalization", {})
    if (
        spectral.get("applied") is not True
        or spectral.get("form") != "R_k <- R_k * min(1, 1 / sigma_max(R_k))"
        or spectral.get("applies_to") != "spectral_convolutions_only"
        or int(spectral.get("power_iterations_per_forward", -1)) != POWER_ITERATIONS
        or int(spectral.get("warmup_iterations", -1)) != WARMUP_ITERATIONS
        or int(spectral.get("matrices_total", -1)) != SPECTRAL_MATRICES_TOTAL
        or spectral.get("checkpoints_materialized") is not True
        or spectral.get("adds_parameters") != 0
    ):
        raise TrainingContractError("the spectral-normalization declaration changed")

    if (
        normalization.get("recomputed_from_training_days_only") is not True
        or tuple(normalization.get("train_days", ())) != TRAIN_RANGE
        or normalization.get("reused_from_a_previous_run") is not False
    ):
        raise TrainingContractError("the production normalization declaration changed")

    if (
        float(selection.get("short_auc_tolerance", -1.0)) != SHORT_AUC_TOLERANCE
        or float(selection.get("worst_long_ratio_ceiling", -1.0))
        != WORST_LONG_RATIO_CEILING
        or tuple(selection.get("primary_fields", ())) != PRIMARY_FIELDS
        or int(selection.get("rollout_days", -1)) != 360
        or float(selection.get("growth_rate_ceiling", -1.0)) != GROWTH_RATE_CEILING
        or int(selection.get("growth_rate_calls", -1)) != DIAGNOSTIC_CALLS
    ):
        raise TrainingContractError("the checkpoint-selection rule changed")

    if (
        not str(output.get("project_root", "")).endswith(VERSION)
        or not str(output.get("scratch_root", "")).endswith(VERSION)
        or tuple(output.get("artifacts", ())) != OUTPUT_ARTIFACTS
    ):
        raise TrainingContractError("the production output declaration changed")

    if not REQUIRED_SOURCE_HASHES.issubset(hashes) or not set(
        REQUIRED_MITGCM_SOURCES
    ).issubset(sources):
        raise TrainingContractError("the production source declaration is incomplete")

    ProductionArchitecture(**architecture)
    if verify_sources:
        _verify_dataset(contract)
        for key in REQUIRED_MITGCM_SOURCES:
            _verify_file(sources[key], key)
        root = resolved.parents[1]
        for relative, expected in hashes.items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise TrainingContractError(f"a pinned source changed: {relative}")
    return contract, resolved, _file_sha256(resolved)


def physical_static_block(
    sources: Mapping[str, Any],
    group: Any,
    pointwise_mean: np.ndarray,
    pointwise_scale: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the five physical static channels from pinned MITgcm sources.

    Three of the five do not exist in the trajectory store, so they are derived
    from the simulation's own inputs: the zonal grid spacing MITgcm dumped, the
    ``thetaClimFile`` it restores towards, and the Coriolis parameter its
    spherical-polar defaults imply. All three files are pinned by SHA-256 and the
    ``data`` declaration is parsed, so a setup change cannot silently alter what
    the network is told about the ocean.
    """

    spacing = _verify_file(sources["mitgcm_zonal_spacing"], "MITgcm zonal spacing")
    relaxation = _verify_file(
        sources["mitgcm_sst_relaxation"], "MITgcm SST relaxation target"
    )
    declaration = _verify_file(sources["mitgcm_declaration"], "MITgcm declaration")
    block, provenance = static_block(
        group,
        zonal_spacing_path=spacing,
        sst_relax_path=relaxation,
        data_path=declaration,
        pointwise_mean=pointwise_mean,
        pointwise_scale=pointwise_scale,
    )
    if tuple(provenance["channels"]) != STATIC_FEATURES:
        raise TrainingContractError("the derived static channel set changed")
    provenance["sha256"] = {
        "zonal_grid_spacing": _file_sha256(spacing),
        "sst_relaxation_target": _file_sha256(relaxation),
        "mitgcm_declaration": _file_sha256(declaration),
    }
    return block, provenance


def physics_contexts(
    group: Any,
    pointwise_mean: np.ndarray,
    pointwise_scale: np.ndarray,
    zonal_spacing_path: str | Path,
) -> tuple[PressureGradientContext, ContinuityContext]:
    """Both physics terms read the same normalizers and the same MITgcm metric."""

    fields = {
        "pointwise_mean": np.asarray(pointwise_mean, dtype=np.float32),
        "pointwise_scale": np.asarray(pointwise_scale, dtype=np.float32),
        "zonal_spacing_m": np.asarray(
            read_mitgcm_2d(zonal_spacing_path), dtype=np.float32
        ),
        "wet_mask": np.asarray(group["wet_mask"][:], dtype=bool),
    }
    return PressureGradientContext(**fields), ContinuityContext(**fields)


def evaluate_loss(
    predictions: Any,
    targets: Any,
    present: Any,
    wet: Any,
    boundary: Any,
    increment_scale: Any,
    config: ProductionLossConfig,
    pressure_context: PressureGradientContext,
    continuity_context: ContinuityContext,
    *,
    model: Any,
    static: Any,
    direction: Any,
) -> tuple[dict[str, Any], Any, Any]:
    """The eight-term objective, plus the diagnostic amplification.

    Returns ``(terms, mean amplification g, next direction)``. ``g`` carries no
    gradient in this arm --- long-horizon growth is constrained by the spectral
    reparameterization, not by a penalty --- so it costs one forward pass and no
    backward. ``predictions[:, 0]`` is ``F(x_t, S) * wet``, reused as its baseline.
    """

    growth, advanced = single_call_amplification(
        model,
        present,
        static,
        wet,
        predictions[:, 0],
        direction,
        epsilon_relative=EPSILON_RELATIVE,
    )
    auxiliary = {
        "pressure_gradient": pressure_gradient_relative_l2(
            predictions, targets, pressure_context
        ),
        "continuity": continuity_relative_l2(
            predictions, targets, present, continuity_context
        ),
        "barotropic_transport": barotropic_transport_relative_l2(
            predictions, targets, present, continuity_context
        ),
    }
    terms = production_loss_terms(
        predictions,
        targets,
        present,
        wet,
        boundary,
        increment_scale,
        config,
        auxiliary,
    )
    return terms, growth, advanced


def split_summary() -> dict[str, Any]:
    """The shared split summary with this arm's one-input start bounds."""

    summary = dict(verify())
    summary["training_rollout_steps"] = ROLLOUT_STEPS
    summary["input_states"] = 1
    summary["earliest_training_rollout_start"] = 0
    summary["latest_training_rollout_start"] = TRAINING_STARTS_PER_REGIME - 1
    summary["training_starts_per_regime"] = TRAINING_STARTS_PER_REGIME
    summary["training_sequences"] = TRAINING_RECORDS
    summary["static_channels"] = list(STATIC_FEATURES)
    summary["history_note"] = (
        "the map reads one time level, so day 0 is a valid training start and no "
        "start is lost to a history requirement"
    )
    return summary


def acceptance_gate(
    selected: Mapping[str, Any],
    best_short: Mapping[str, float],
) -> dict[str, Any]:
    """The two validation-measurable conditions of the acceptance gate.

    There is no predecessor to compare against, so the short-horizon condition is
    the selection rule's own guard: the chosen checkpoint must be within 5 % of
    the best short AUC achieved by any checkpoint of this run, in every primary
    field. The 2,000-day conditions are evaluated by the figure package, which is
    the only stage that runs a 2,000-day rollout; they are recorded as deferred
    rather than silently passing.
    """

    short = {
        field: float(selected["short_auc_10_90"][field]) / float(best_short[field])
        for field in PRIMARY_FIELDS
    }
    worst_long = max(float(v) for v in selected["long_ratio_to_climatology"].values())
    growth = selected.get("perturbation_growth", {})
    measured = growth.get("worst_growth_rate_per_call")
    worst_growth = float("inf") if measured is None else float(measured)
    short_pass = all(value <= SHORT_AUC_TOLERANCE for value in short.values())
    long_pass = worst_long <= WORST_LONG_RATIO_CEILING
    growth_pass = worst_growth <= GROWTH_RATE_CEILING
    return {
        "short_auc_10_90_ratio_to_best_checkpoint": short,
        "short_auc_tolerance": SHORT_AUC_TOLERANCE,
        "short_auc_within_5_percent_of_best_in_every_field": bool(short_pass),
        "worst_long_ratio_to_climatology": worst_long,
        "worst_long_ratio_ceiling": WORST_LONG_RATIO_CEILING,
        "worst_long_ratio_at_or_below_ceiling": bool(long_pass),
        "worst_perturbation_growth_rate_per_call": worst_growth,
        "growth_rate_ceiling": GROWTH_RATE_CEILING,
        "growth_rate_at_or_below_ceiling": bool(growth_pass),
        "reference_growth_rate_per_call": REFERENCE_GROWTH_RATE,
        "perturbation_growth": dict(growth),
        "validation_conditions_pass": bool(short_pass and long_pass and growth_pass),
        "long_horizon_window_note": (
            "worst_long_ratio_to_climatology integrates 90-360 days only, which "
            "is why the growth-rate condition above exists: the unconstrained "
            f"arm passed every RMSE-based condition with {REFERENCE_GROWTH_RATE} "
            "per call and then amplified 28x over the 200-call inference"
        ),
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


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify the contract, the sources, the record counts and the architecture.

    Deliberately does *not* recompute the 18,000-snapshot normalization: that is
    a training-run cost. It does build the model, assert the exact parameter
    count, and prove each physics term is identically zero when truth is scored
    against itself.
    """

    contract, resolved, digest = load_contract(contract_path)
    dataset = _verify_dataset(contract)
    group = zarr.open_consolidated(str(dataset), mode="r")
    assert_store_is_v3(group)
    _, pair_split = store_codes()
    records = records_for_rollout_split(
        pair_split, TRAIN_CODE, rollout_steps=ROLLOUT_STEPS
    )
    if len(records) != TRAINING_RECORDS:
        raise TrainingContractError(
            f"the training set is {len(records)} records, not {TRAINING_RECORDS}"
        )
    architecture = ProductionArchitecture(**contract["architecture"])
    result: dict[str, Any] = {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "dataset_version": str(group.attrs["version"]),
        "split": split_summary(),
        "loss_contract_sha256": LOSS_CONTRACT_SHA256,
        "from_scratch": True,
        "parent_checkpoint": None,
        "rollout_steps": ROLLOUT_STEPS,
        "static_channels": list(STATIC_FEATURES),
        "external_input_channels": architecture.in_channels,
        "lifting_input_channels": architecture.lifting_in_channels,
        "training_rollout_records": len(records),
        "training_starts_per_regime": len(records) // len(EXPERIMENTS),
        "earliest_training_start": int(min(t for _, t in records)),
        "latest_training_start": int(max(t for _, t in records)),
        "validation_records": int(validation_records().shape[0]),
        "inference_range": list(INFERENCE_RANGE),
        "state_transitions": STATE_TRANSITIONS,
        "inference_state_opened": False,
    }
    if torch is not None:
        seed_everything(SEED)
        model = build_model(architecture)
        count = parameter_count(model)
        if count != EXPECTED_PARAMETER_COUNT:
            raise TrainingContractError(
                f"the architecture builds {count:,} parameters, not "
                f"{EXPECTED_PARAMETER_COUNT:,}"
            )
        if float(model.local.weight.abs().max()) != 0.0:
            raise TrainingContractError("the local 3x3 branch is not zero-initialized")
        result["parameter_count"] = count
        result["local_branch_zero_initialized"] = True
        # Mathematical sanity: truth scored against itself must cost nothing.
        pressure, continuity = physics_contexts(
            group,
            np.zeros((STATE_CHANNEL_COUNT, *GRID_SHAPE), dtype=np.float32),
            np.ones((STATE_CHANNEL_COUNT, *GRID_SHAPE), dtype=np.float32),
            contract["sources"]["mitgcm_zonal_spacing"]["path"],
        )
        dummy = torch.zeros(
            (1, ROLLOUT_STEPS, STATE_CHANNEL_COUNT, *GRID_SHAPE),
            dtype=torch.float32,
        )
        identity = {
            "pressure_gradient": float(
                pressure_gradient_relative_l2(dummy, dummy, pressure).detach()
            ),
            "continuity": float(
                continuity_relative_l2(dummy, dummy, dummy[:, 0], continuity).detach()
            ),
            "barotropic_transport": float(
                barotropic_transport_relative_l2(
                    dummy, dummy, dummy[:, 0], continuity
                ).detach()
            ),
        }
        if any(value != 0.0 for value in identity.values()):
            raise TrainingContractError(f"a physics identity loss is non-zero: {identity}")
        result["physics_identity_losses"] = identity
        padding = model.fno.domain_padding
        probe = torch.ones((1, 1, *GRID_SHAPE), dtype=torch.float32)
        padded = padding.pad(probe)
        restored = padding.unpad(padded)
        if (
            tuple(padded.shape[-2:]) != (298, 298)
            or not bool(torch.equal(restored, probe))
            or float(padded[..., 0, :].abs().max()) != 0.0
            or float(padded[..., -1, :].abs().max()) != 0.0
            or float(padded[..., :, 0].abs().max()) != 0.0
            or float(padded[..., :, -1].abs().max()) != 0.0
        ):
            raise TrainingContractError(
                "raised-cosine padding did not preserve the physical field and "
                "close the outer FFT boundary"
            )
        result["domain_padding"] = {
            "mode": architecture.domain_padding_mode,
            "fraction": architecture.domain_padding,
            "halo_cells_per_side": [padding.pad_y, padding.pad_x],
            "padded_shape": list(padded.shape[-2:]),
            "physical_field_preserved_exactly": True,
            "outer_fft_boundary_exactly_zero": True,
        }
        # The retained spectral cap is verified against an exact SVD rather
        # than the power-iteration estimate that installs it.
        spectral = apply_mode_spectral_norm(model)
        conv = model.fno.fno_blocks.convs[0]
        with torch.no_grad():
            free = conv.weight.original.detach()
            capped = conv.weight.tensor.detach()

            def _exact(tensor: Any) -> Any:
                matrices = tensor.reshape(
                    tensor.shape[0], tensor.shape[1], -1
                ).permute(2, 0, 1)
                return torch.linalg.matrix_norm(matrices, ord=2)

            before, after = _exact(free), _exact(capped)
            untouched = before <= 1.0
            preserved = bool(
                torch.allclose(
                    free.reshape(free.shape[0], free.shape[1], -1).permute(2, 0, 1)[untouched],
                    capped.reshape(capped.shape[0], capped.shape[1], -1).permute(2, 0, 1)[untouched],
                    atol=1.0e-7,
                )
            )
        if float(after.max()) > 1.0 + 1.0e-3 or not preserved:
            raise TrainingContractError(
                "per-mode spectral normalization did not cap sigma_max at one: "
                f"max {float(after.max()):.6f}, below-one modes preserved {preserved}"
            )
        if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
            raise TrainingContractError("spectral normalization changed the parameter count")
        result["spectral_normalization"] = spectral
        result["spectral_norm_check"] = {
            "block_0_sigma_max_before": float(before.max()),
            "block_0_sigma_mean_before": float(before.mean()),
            "block_0_sigma_max_after": float(after.max()),
            "block_0_modes_above_one_before": int((before > 1.0).sum()),
            "below_one_modes_left_untouched": preserved,
            "verified_against": "exact_svd_torch_linalg_matrix_norm_ord_2",
        }
        result["growth_rate_ceiling"] = GROWTH_RATE_CEILING
        result["contraction_penalty"] = False
        result["reference_arm"] = {
            "version": REFERENCE_VERSION,
            "measured_growth_rate_per_call": REFERENCE_GROWTH_RATE,
        }
        result["penalty_arm_growth_rates"] = PENALTY_ARM_GROWTH_RATES
    return result


def _readme(report: Mapping[str, Any]) -> str:
    decision = report["selection_decision"]
    gate = report["acceptance_gate"]
    rows = "\n".join(
        "| {step:,} | {short} | {long} | {growth} |".format(
            step=int(summary["optimizer_step"]),
            short=" / ".join(
                f"{summary['short_auc_10_90'][field]:.3f}" for field in PRIMARY_FIELDS
            ),
            long=" / ".join(
                f"{summary['long_ratio_to_climatology'][field]:.3f}"
                for field in PRIMARY_FIELDS
            ),
            growth=(
                f"{summary['perturbation_growth']['worst_growth_rate_per_call']:.4f}"
            ),
        )
        for summary in report["validation_summaries"]
    )
    decision_growth = float(decision.get("selected_growth_rate_per_call", float("nan")))
    decision_growth = float(decision.get("selected_growth_rate_per_call", float("nan")))
    spectral = report["spectral_normalization"]
    initial = spectral["sigma_at_initialization"]["per_block"][0]
    final = report["training_history"][-1]["spectral_norm"]["per_block"][0]
    return f"""# Production emulator, smooth FFT-boundary padding

    F_theta: [x_t, S] -> x_(t+10)

`x_t` is the 46-channel prognostic state at one time only; `S` is the five
physical static channels. 51 external channels, plus two deterministic
sine/cosine position channels inside the model, so lifting sees 53 and the
output is 46. A 64 x 64-mode, width-128, three-block FNO with six pointwise
LayerNorms, a 4C Channel MLP, 10 % raised-cosine tapered replicate padding and a
parallel bias-free 3 x 3 local correction. Parameter count:
{int(report['parameter_count']):,}.

## Production update

The first turbulent rollout developed narrowband zonal stripes next to the
domain edge. Disabling the local 3 x 3 path did not suppress them and degraded
SST and pressure skill. Replacing the constant-zero latent FFT halo with a
replicated halo brought smoothly to zero reduced the diagnostic stripe-band
power fraction from {PREVIOUS_STRIPE_POWER_FRACTION:.3f} to
{TAPERED_PADDING_STRIPE_POWER_FRACTION:.3f}. The outer FFT boundary is exactly
zero and periodically continuous, while the original lifted field is untouched.

This is the only production revision. The local branch, Fourier modes, width,
blocks, loss, data, optimizer, learning-rate schedule and seed are unchanged;
the weights and training-only normalizers are recomputed from scratch.

## Retained spectral normalization

The channel mixing at Fourier mode `k` is a dense complex matrix
`R_k` in `C^(128 x 128)`; there are {spectral['matrices_total']:,} of them across
the three blocks, holding 99.46 % of the parameters. Each remains capped:

    R_k <- R_k * min(1, 1 / sigma_max(R_k))

estimated by a persistent alternating power iteration, one step per forward.
Modes already at or below one are left exactly alone. This is a
reparameterization of the operator, **not** a bound on the emulator Jacobian:
each block is `GELU(K_R h + W h)` with its own skips, so legitimate transient
amplification remains representable.

At initialization every mode of block 0 had `sigma_max` above one
(mean {initial['sigma_mean']:.3f}, max {initial['sigma_max']:.3f}); by the final
checkpoint the free parameter's mean was {final['sigma_mean']:.3f} with
{final['modes_above_one']:,} of {final['modes']:,} modes above one and therefore
actively held down.

Checkpoints are written **materialized** --- the normalized weights are baked
into the tensor --- so a published checkpoint loads into an unmodified model and
the inference layer is simply `y_hat(k) = R_k_tilde x_hat(k)`, whose adjoint is
`R_k_tilde^H`.

The contraction *penalty* of the two preceding arms is removed. Perturbation
growth is still measured and still gates checkpoint selection, but contributes
no gradient, so this run isolates the operator constraint.

## Training

From random initialization: no parent checkpoint, no migration, no inherited
normalization. Both normalizers are recomputed over training days
{TRAIN_RANGE[0]}--{TRAIN_RANGE[1] - 1} of S0, S1 and S2 only.

Six-step autoregressive, no teacher forcing after the initial state:
{report['counts']['training_rollout_records']:,} sequences
({report['counts']['training_starts_per_regime']:,} starts per regime),
{MAXIMUM_STEPS:,} optimizer steps at batch {BATCH_SIZE}
(microbatch {MICROBATCH_SIZE} x accumulation {GRADIENT_ACCUMULATION_STEPS}) =
{STATE_TRANSITIONS:,} state transitions. Learning rate {LEARNING_RATE:g} through
step {int(report['optimizer']['decay_step']):,}, then
{LEARNING_RATE * DECAY_FACTOR:g}.

    L = L_state + 0.001 L_inc + 0.50 L_rollout + 1e-5 L_spectral
        + 0.065 L_boundary + 0.05 L_pressure + 0.05 L_continuity
        + 0.05 L_barotropic

## Selection

| step | short AUC 10--90 (speed / SST / pressure) | long / climatology | growth rate |
| --- | --- | --- | --- |
{rows}

Selected step {int(decision['selected_optimizer_step']):,} via
`{decision['branch']}`, growth rate {decision_growth:.5f} per call
(ceiling {GROWTH_RATE_CEILING}; the unconstrained arm measured
{REFERENCE_GROWTH_RATE}). Validation gate:
**{'pass' if gate['validation_conditions_pass'] else 'fail'}**.

Evaluation uses the nested validation/inference protocol; there is no
independent third test split.

Report content SHA-256: `{report['content_sha256']}`.
"""


def _dataloader_workers() -> int:
    """Loader processes per rank. Overridable so a smoke run can stay in-process."""

    return max(0, int(os.environ.get("AF_DATALOADER_WORKERS", "4")))


def run(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
    resume_validation: bool = False,
) -> dict[str, Any]:
    """Train from random initialization, validate every checkpoint, publish one.

    ``resume_validation`` picks up a run whose training loop completed but whose
    validation did not: the four checkpoints and the normalization artifact are
    read back from the unfinished scratch tree instead of being recomputed, and
    everything from the validation stage onward proceeds unchanged. It exists
    because the loop is a day of compute and the stage after it is an hour.
    """

    if torch is None or DataLoader is None:  # pragma: no cover - environment dependent
        raise RuntimeError("training the production emulator requires PyTorch")
    require_runtime()
    started = time.monotonic()
    topology = initialize_distributed(device_name)
    if topology.distributed:
        device_name = f"cuda:{topology.local_rank}"
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    dataset = _verify_dataset(contract)
    summary_of_split = split_summary()
    scratch = Path(contract["output"]["scratch_root"]).resolve()
    project = Path(contract["output"]["project_root"]).resolve()
    scratch_tmp = scratch.with_name(scratch.name + ".tmp")
    project_tmp = project.with_name(project.name + ".tmp")
    if resume_validation:
        if scratch.exists() or project.exists():
            raise FileExistsError(
                "this run is already published; there is nothing to resume"
            )
        if not (scratch_tmp / CHECKPOINT_DIRECTORY).is_dir():
            raise TrainingContractError(
                f"no unfinished training tree to resume at {scratch_tmp}"
            )
        # Everything the resume depends on, checked now rather than after the
        # climatology scan: a structural problem should cost a second, not half
        # an hour.
        missing = [
            name
            for name in (
                NORMALIZATION_NAME,
                *(
                    f"{CHECKPOINT_DIRECTORY}/{CHECKPOINT_STEM}_{step:05d}.pt"
                    for step in CHECKPOINT_STEPS
                ),
            )
            if not (scratch_tmp / name).is_file()
        ]
        if missing:
            raise TrainingContractError(
                f"the tree at {scratch_tmp} cannot be resumed; missing {missing}"
            )
        if topology.is_primary:
            project_tmp.mkdir(parents=True, exist_ok=True)
    elif any(p.exists() for p in (scratch, project, scratch_tmp, project_tmp)):
        raise FileExistsError("refusing to overwrite production training output")

    training = contract["training"]
    seed_everything(int(training["seed"]))
    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    store_static = group["static_features"]
    assert_store_is_v3(group)
    snapshot_split, pair_split = store_codes()
    wet_array, _, _ = store_wind_normalization(group)
    wet_array = np.asarray(wet_array, dtype=bool)

    # Both normalization artifacts are computed here, from training days only.
    # A resumed run reads back the ones the interrupted run already wrote and
    # hashed: recomputing them would take an hour and, being a deterministic
    # function of the same training days, would reproduce the same bytes.
    if resume_validation:
        with np.load(scratch_tmp / NORMALIZATION_NAME) as stored:
            normalizers = {
                "mean": stored["pointwise_mean"],
                "scale": stored["pointwise_scale"],
                "raw_scale": stored["pointwise_raw_scale"],
                "floor": stored["channel_scale_floor"],
            }
            increment_values = stored["increment_scale"]
        point_mean = normalizers["mean"]
        point_scale = normalizers["scale"]
        normalizers["summary"] = normalizer_summary(
            point_mean,
            normalizers["raw_scale"],
            point_scale,
            normalizers["floor"],
            wet_array,
            snapshots_total=(TRAIN_RANGE[1] - TRAIN_RANGE[0]) * len(EXPERIMENTS),
            snapshots_per_regime=TRAIN_RANGE[1] - TRAIN_RANGE[0],
        )
    else:
        normalizers = training_pointwise_normalizers(group, snapshot_split)
        point_mean = normalizers["mean"]
        point_scale = normalizers["scale"]
        increment_values = training_increment_scale(group, pair_split, point_scale)
    statics, static_provenance = physical_static_block(
        contract["sources"], group, point_mean, point_scale
    )
    pressure_context, continuity_context = physics_contexts(
        group, point_mean, point_scale, contract["sources"]["mitgcm_zonal_spacing"]["path"]
    )
    climatology_state, climatology_derived, climatology_days = train_only_climatology(
        state, wet_array
    )

    loss_config = production_loss_config()
    training_records = records_for_rollout_split(
        pair_split, TRAIN_CODE, rollout_steps=loss_config.rollout_steps
    )
    if len(training_records) != TRAINING_RECORDS:
        raise TrainingContractError("the training record count changed")
    training_dataset = RolloutDataset(
        dataset,
        training_records,
        point_mean,
        point_scale,
        statics,
        rollout_steps=loss_config.rollout_steps,
    )
    microbatch = int(training["microbatch_size"])
    # The contract declares the *global* microbatches per optimizer step; each
    # rank runs its own share and the gradients are averaged, so the effective
    # batch is the contract's regardless of how many GPUs the run was given.
    accumulation_total = int(training["gradient_accumulation_steps"])
    accumulation = require_divisible(accumulation_total, topology)
    loader = DataLoader(
        training_dataset,
        batch_sampler=ShardedBatchSampler(
            ChunkAwareBatchSampler(
                training_dataset, microbatch, int(training["seed"])
            ),
            topology,
        ),
        # Each sample pulls seven 11.3 MB day slices, and unlike the 1-degree
        # store this one is far too large to sit in page cache, so the read is
        # real work that has to overlap the step rather than serialize with it.
        num_workers=_dataloader_workers(),
        prefetch_factor=4 if _dataloader_workers() else None,
        pin_memory=device.type == "cuda",
    )
    architecture = ProductionArchitecture(**contract["architecture"])
    model = build_model(architecture).to(device)
    # The retained spectral constraint is installed before the optimizer sees
    # the parameters; it replaces the weight holder and adds no parameters.
    spectral_provenance = apply_mode_spectral_norm(model)
    spectral_provenance["sigma_at_initialization"] = mode_sigma_summary(model)
    count = parameter_count(model)
    if count != EXPECTED_PARAMETER_COUNT:
        raise TrainingContractError(
            f"the architecture builds {count:,} parameters, not {EXPECTED_PARAMETER_COUNT:,}"
        )
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

    # Artifacts have one owner. Non-primary ranks contribute gradients and
    # nothing else; they never create, write or read the output tree.
    checkpoint_directory = scratch_tmp / CHECKPOINT_DIRECTORY
    if topology.is_primary:
        scratch_tmp.parent.mkdir(parents=True, exist_ok=True)
        project_tmp.parent.mkdir(parents=True, exist_ok=True)
        # A resumed run inherits the interrupted run's tree rather than making
        # one: the checkpoints it is about to validate are already inside it.
        scratch_tmp.mkdir(exist_ok=resume_validation)
        project_tmp.mkdir(exist_ok=resume_validation)
        checkpoint_directory.mkdir(exist_ok=resume_validation)
    # Also rank 0's alone. Every rank recomputes the normalizers -- they are
    # cheap to derive and expensive to communicate -- but only the owner of the
    # output tree writes them. The complete list of writes in this function is
    # this one, the checkpoints in the loop, the divergence record, and
    # everything after the non-primary ranks have returned; all are guarded.
    normalization_path = scratch_tmp / NORMALIZATION_NAME
    # Not rewritten on resume: it was read back from this same path, and
    # np.savez stores zip timestamps, so rewriting identical arrays would still
    # change the file's hash and with it the published provenance.
    if topology.is_primary and not resume_validation:
        np.savez_compressed(
            normalization_path,
            pointwise_mean=point_mean,
            pointwise_raw_scale=normalizers["raw_scale"],
            pointwise_scale=point_scale,
            channel_scale_floor=normalizers["floor"],
            increment_scale=increment_values,
        )

    # The power-iteration direction is persistent across the whole run: it is
    # advanced one step per microbatch, so it tracks the operator's leading
    # growing direction instead of resampling an uninformative random one.
    direction = initial_direction(
        (1, STATE_CHANNEL_COUNT, *wet_array.shape),
        wet,
        device,
        seed=int(training["seed"]),
    )
    iterator = iter(loader)
    totals = {name: 0.0 for name in AUDIT_TERMS}
    growth_total = 0.0
    samples = 0
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    learning_rate_for_divergence = float(training["initial_learning_rate"])

    def _diverged(step: int, reason: str) -> None:
        # A rank can only detect divergence in its own microbatches, and it does
        # so before the gradients are averaged, so a non-primary rank may get
        # here alone. It owns no output tree; it raises and the launcher takes
        # the whole job down with it.
        if not topology.is_primary:
            raise DivergenceError(
                f"{reason} at optimizer step {step} on rank {topology.rank}"
            )
        (project_tmp / DIVERGENCE_NAME).write_text(
            json.dumps(
                {
                    "status": "diverged",
                    "version": VERSION,
                    "reason": reason,
                    "optimizer_step": int(step),
                    "learning_rate": learning_rate_for_divergence,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        shutil.rmtree(scratch_tmp, ignore_errors=True)
        os.replace(project_tmp, project)
        raise DivergenceError(f"{reason} at optimizer step {step}")

    for step in ([] if resume_validation else range(1, maximum_steps + 1)):
        if step == decay_step + 1:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= float(training["decay_factor"])
            learning_rate_for_divergence = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)
        model.train()
        step_samples = 0
        for micro in range(accumulation):
            try:
                raw_features, futures = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_features, futures = next(iterator)
            # The dataset already emits the final 51-channel layout, so there is
            # nothing to select here.
            features = raw_features.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
            # Six calls; the state is self-generated from the second onward.
            predictions = state_unroll(model, features, wet, loss_config.rollout_steps)
            terms, growth, direction = evaluate_loss(
                predictions,
                futures,
                features[:, :STATE_CHANNEL_COUNT],
                wet,
                boundary,
                increment_scale,
                loss_config,
                pressure_context,
                continuity_context,
                model=model,
                static=features[:, STATE_CHANNEL_COUNT:],
                direction=direction,
            )
            if not all(bool(torch.isfinite(terms[n]).item()) for n in AUDIT_TERMS):
                _diverged(step, "training objective became non-finite")
            (terms["total"] / accumulation).backward()
            batch = int(features.shape[0])
            for name in AUDIT_TERMS:
                totals[name] += float(terms[name].detach().cpu()) * batch
            growth_total += float(growth.cpu()) * batch
            step_samples += batch
        # Every rank has now run its own share of this step's microbatches and
        # accumulated into .grad. Averaging here makes the update exactly the one
        # a single process would have made from the whole batch: each term of the
        # objective is a per-sample mean and every rank carries the same number
        # of samples, so the mean of the per-rank means is the mean over the batch.
        topology.average_gradients(model)
        if not all(
            bool(torch.isfinite(p.grad).all().item())
            for p in model.parameters()
            if p.grad is not None
        ):
            _diverged(step, "training gradients became non-finite")
        optimizer.step()
        direction = topology.average_direction(direction)
        samples += step_samples

        if step not in CHECKPOINT_STEPS:
            continue
        window = {
            name: topology.average_scalar(totals[name] / samples)
            for name in AUDIT_TERMS
        }
        history_record = {
            "optimizer_step": step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_window": window,
            "mean_single_call_amplification": topology.average_scalar(
                growth_total / samples
            ),
            "spectral_norm": mode_sigma_summary(model),
        }
        history.append(history_record)
        path = checkpoint_directory / f"{CHECKPOINT_STEM}_{step:05d}.pt"
        if not topology.is_primary:
            totals = {name: 0.0 for name in AUDIT_TERMS}
            growth_total = 0.0
            samples = 0
            continue
        torch.save(
            {
                "version": VERSION,
                "optimizer_step": step,
                "architecture": architecture.to_dict(),
                "contract": str(resolved_contract),
                "contract_sha256": contract_sha,
                "dataset_version": DATASET_VERSION,
                "loss_contract": loss_contract(loss_config),
                "loss_contract_sha256": LOSS_CONTRACT_SHA256,
                "rollout_steps": loss_config.rollout_steps,
                "input_states": 1,
                "static_channels": list(STATIC_FEATURES),
                "from_scratch": True,
                "spectral_normalization": "per_mode_sigma_max_capped_at_one",
                "training_history_record": history_record,
                # Materialized: the normalized weights are baked in, so this
                # loads strictly into a plain ProductionFNO and the published
                # operator is exactly the one that was trained.
                "model_state_dict": materialized_state_dict(model),
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
        growth_total = 0.0
        samples = 0

    resumed_contract_hashes: set[str] = set()
    if resume_validation:
        # Every checkpoint carries the training-window record that was appended
        # to the history when it was written, so the history is recoverable in
        # full from the files themselves.
        for optimizer_step in CHECKPOINT_STEPS:
            path = checkpoint_directory / f"{CHECKPOINT_STEM}_{optimizer_step:05d}.pt"
            if not path.is_file():
                raise TrainingContractError(f"the resumed run is missing {path.name}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if int(payload["optimizer_step"]) != int(optimizer_step):
                raise TrainingContractError(f"{path.name} is not step {optimizer_step}")
            if payload.get("loss_contract_sha256") != LOSS_CONTRACT_SHA256:
                raise TrainingContractError(
                    f"{path.name} was trained against a different objective"
                )
            resumed_contract_hashes.add(str(payload.get("contract_sha256")))
            history.append(payload["training_history_record"])
            checkpoints.append(
                {
                    "optimizer_step": int(optimizer_step),
                    "checkpoint": path.name,
                    "checkpoint_sha256": _file_sha256(path),
                }
            )
            del payload

    if not topology.is_primary:
        topology.shutdown()
        return {
            "status": "complete_contributing_rank",
            "version": VERSION,
            "rank": topology.rank,
            "world_size": topology.world_size,
            "optimizer_steps": maximum_steps,
            "note": (
                "this rank supplied gradients only; every artifact belongs to "
                "rank 0, which validates and publishes alone"
            ),
        }

    if len(checkpoints) != len(CHECKPOINT_STEPS):
        raise TrainingContractError("not every declared checkpoint was written")

    # The trained model, Adam state and final six-call autograd graph have no use
    # in validation, which loads each materialized checkpoint into a fresh probe.
    # Releasing the graph and loader explicitly is essential: deleting only the
    # model left ~31 GiB live on a V100 and the first validation FFT then OOMed.
    if not resume_validation:
        del predictions
        del terms
        del growth
        del features
        del futures
    del direction
    del iterator
    del loader
    del training_dataset
    del optimizer
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    records = validation_records()
    # Two S0 starts from the validation block for the growth-rate diagnostic.
    # The rollout is autonomous -- it reads no truth after the initial state --
    # so running it past day 7,200 leaks nothing.
    growth_starts = [int(validation_starts()[0]), int(validation_starts()[-1])]
    growth_states = []
    for start in growth_starts:
        raw = np.asarray(state[0, start], dtype=np.float32)
        normalized = (raw - point_mean) / point_scale
        normalized[:, ~wet_array] = 0.0
        growth_states.append(
            torch.from_numpy(np.ascontiguousarray(normalized))[None].to(device)
        )
    growth_static = torch.from_numpy(np.ascontiguousarray(statics[0]))[None].to(device)

    summaries = []
    evaluated_arrays = []
    for record in checkpoints:
        payload = torch.load(
            checkpoint_directory / record["checkpoint"],
            map_location=device,
            weights_only=False,
        )
        probe = build_model(architecture).to(device)
        probe.load_state_dict(payload["model_state_dict"], strict=True)
        probe.eval()
        stepper = ProductionStepper(
            model=probe,
            device=device,
            wet=wet_array,
            mean=point_mean,
            scale=point_scale,
            statics=statics,
        )
        value = validate_checkpoint(
            stepper,
            state,
            store_static,
            records,
            climatology_state,
            climatology_derived,
            wet_array,
        )
        # The quantity v1 was blind to, measured before this checkpoint can be
        # considered for selection.
        value["perturbation_growth"] = growth_rate_summary(
            probe, growth_states, growth_static, wet
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
    selected_summary = next(
        s for s in summaries if int(s["optimizer_step"]) == selected_step
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
        optimizer_steps=np.asarray(
            [s["optimizer_step"] for s in summaries], dtype=np.int32
        ),
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
        "resumed": (
            None
            if not resume_validation
            else {
                "resumed_from": "checkpoints written by an earlier run of this contract",
                "training_loop_rerun": False,
                "normalization_recomputed": False,
                "normalization_read_back_from": str(scratch / NORMALIZATION_NAME),
                "contract_sha256_at_training_time": sorted(resumed_contract_hashes),
                "contract_sha256_at_publication": contract_sha,
                "why_they_differ": (
                    "the first attempt finished all 7,680 optimizer steps and then "
                    "exhausted the GPU in validation, which was rolling all 102 "
                    "members as one batch. The fix chunks that rollout over members "
                    "and frees the trained model before the first probe is built. "
                    "Both edits are downstream of training: validation.py's "
                    "reductions over members are taken after every chunk has "
                    "contributed, so the metrics are unchanged, and no weight in "
                    "any checkpoint was produced by the edited code"
                ),
                "unchanged_and_verified": [
                    "loss_contract_sha256, checked against every checkpoint",
                    "the optimizer step each checkpoint declares",
                    "the dataset and its metadata hash",
                ],
            }
        ),
        "data_parallel": {
            "world_size": topology.world_size,
            "microbatches_per_optimizer_step_total": accumulation_total,
            "microbatches_per_optimizer_step_per_rank": accumulation,
            "microbatch_size": microbatch,
            "effective_batch": accumulation_total * microbatch,
            "gradient_reduction": "mean over ranks after every rank's accumulation",
            "exactness": (
                "every objective term is a per-sample mean and the ranks carry "
                "equal sample counts, so the averaged gradient is the whole "
                "batch's gradient up to float32 summation order"
            ),
            "amplification_diagnostic": (
                "each rank advances its own power-iteration direction through "
                "its own microbatches; the directions are averaged and "
                "renormalized once per optimizer step"
            ),
            "artifacts": "written by rank 0 only, which also validates and selects",
        },
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": {
            "path": str(dataset),
            "version": DATASET_VERSION,
            "metadata_sha256": _file_sha256(dataset / ".zmetadata"),
        },
        "split": summary_of_split,
        "architecture": architecture.to_dict(),
        "parameter_count": count,
        "initialization": {
            "from_scratch": True,
            "parent_checkpoint": None,
            "load_model_state": False,
            "load_optimizer_state": False,
            "normalization_reused": False,
            "fno_weights": "neuraloperator_default_random_initialization",
            "local_branch_initialization": "zeros",
            "local_branch_bias": False,
            "layer_norm_scale": 1.0,
            "layer_norm_bias": 0.0,
            "seed": int(training["seed"]),
        },
        "normalization": {
            "recomputed_from": "train_only_days_0_5999_of_S0_S1_S2",
            "reused_from_a_previous_run": False,
            "summary": normalizers["summary"],
            "artifact": str(scratch / NORMALIZATION_NAME),
            "artifact_sha256": _file_sha256(normalization_path),
        },
        "climatology": {
            "source": "per_regime_pointwise_mean_over_train_only_0_5999",
            "days_per_regime": climatology_days,
        },
        "increment_scale": increment_values.tolist(),
        "loss": contract["loss"],
        "loss_contract": loss_contract(loss_config),
        "loss_contract_sha256": LOSS_CONTRACT_SHA256,
        "production_padding_update": {
            "previous_mode": PREVIOUS_PADDING_MODE,
            "mode": architecture.domain_padding_mode,
            "fraction": architecture.domain_padding,
            "weight_lineage": "none; this run is retrained from scratch",
            "diagnostic_ablation": {
                "previous_stripe_power_fraction": PREVIOUS_STRIPE_POWER_FRACTION,
                "tapered_padding_stripe_power_fraction": (
                    TAPERED_PADDING_STRIPE_POWER_FRACTION
                ),
                "local_branch_removal_suppressed_stripes": False,
            },
            "unchanged": [
                "parameter_count", "Fourier_modes", "hidden_width", "blocks",
                "local_branch", "dataset", "split", "objective", "optimizer",
                "batch", "learning_rate_schedule", "seed", "validation_protocol",
            ],
        },
        "constrained_relative_to": {
            "version": REFERENCE_VERSION,
            "measured_growth_rate_per_call": REFERENCE_GROWTH_RATE,
            "penalty_arm_growth_rates": PENALTY_ARM_GROWTH_RATES,
            "weight_lineage": "none; this run is from scratch",
            "changed": [
                "per_mode_spectral_normalization_of_the_three_fourier_convolutions",
                "contraction_penalty_removed_from_the_objective",
                "constant_zero_padding_replaced_by_raised_cosine_tapered_replicate_padding",
            ],
            "unchanged": [
                "architecture", "static_channels", "modes", "width", "blocks",
                "local_branch", "dataset", "split", "normalization_procedure",
                "rollout_steps", "batch_size", "maximum_steps", "learning_rate",
                "schedule", "seed", "the_eight_loss_terms",
                "validation_protocol", "inference_protocol",
            ],
            "success_criterion": (
                "lambda_hat at or below 1.0 with 0-360 day RMSE and ACC "
                "essentially unchanged, day-2000 anomaly amplitude order one, "
                "western-boundary structure retained and no excessive spectral "
                "damping"
            ),
        },
        "spectral_normalization": spectral_provenance,
        "contraction_penalty": False,
        "temporal_context": {
            "input_states": 1,
            "map": "x_t -> x_t_plus_10",
            "prediction": "direct_state_not_residual",
            "autoregression": "no_teacher_forcing_after_the_initial_state",
        },
        "static_channels": {
            "channels": list(STATIC_FEATURES),
            "external_input_channels": architecture.in_channels,
            "lifting_input_channels": architecture.lifting_in_channels,
            "provenance": static_provenance,
        },
        "optimizer": {
            "name": "adam",
            "initial_learning_rate": float(training["initial_learning_rate"]),
            "adam_betas": [float(v) for v in training["adam_betas"]],
            "weight_decay": float(training["weight_decay"]),
            "batch_size": int(training["batch_size"]),
            "microbatch_size": microbatch,
            "gradient_accumulation_steps": accumulation,
            "gradient_clipping": False,
            "decay_step": decay_step,
            "decay_factor": float(training["decay_factor"]),
            "state_transitions": STATE_TRANSITIONS,
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
        "selection_decision": decision,
        "acceptance_gate": acceptance_gate(
            selected_summary, decision["best_short_auc_10_90"]
        ),
        "published_checkpoint": published,
        "arrays": str(scratch / ARRAYS_NAME),
        "arrays_sha256": _file_sha256(arrays_path),
        "read_contract": contract["read_contract"],
        "validation_state_opened": True,
        "inference_state_opened": False,
        "elapsed_seconds": time.monotonic() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    report = json_safe(report)
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
                "inference_state_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(scratch_tmp, scratch)
    os.replace(project_tmp, project)
    topology.shutdown()
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        if command == "run":
            child.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
            child.add_argument(
                "--resume-validation",
                action="store_true",
                help=(
                    "skip the training loop and validate the checkpoints an "
                    "interrupted run already wrote"
                ),
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        preflight(args.contract)
        if args.command == "preflight"
        else run(
            args.contract,
            device_name=args.device,
            resume_validation=args.resume_validation,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
