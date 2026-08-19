"""Ninety-day autoregressive fine-tune of the published production operator.

    F_theta: [x_t (46), S (5)] -> x_{t+10} (46)

This is the first **staged** run in the tree. Every preceding arm trained from
random initialization with all eight losses active from optimizer step one; this
one starts from a frozen, published parent and does nothing but extend the
autonomous rollout the operator is asked to survive:

    parent  model_c_production_1in_1out_spectralnorm_v1/selected.pt (step 7,680)
      |
      v
    child   model_c_production_1in_1out_spectralnorm_ft90_v1

**Zero architecture changes.** 51 -> 53 -> 256 -> 128 -> three 32x32-mode FNO
blocks -> 128 -> 256 -> 46, width 128, ChannelMLP 128->512->128, six pointwise
LayerNorms, the bias-free 3x3 local branch, per-mode spectral normalization at
``rho = 1``, 46 prognostic outputs. No new parameters, no new channels, no
change to the modes, the vertical levels or the ten-day interval.

Exactly four things move:

===============  ==================================  ==========================
                 parent                              this run
===============  ==================================  ==========================
initialization   random                              parent ``selected.pt``
rollout steps    6  (60 days)                        9  (90 days)
learning rate    5e-4 -> 1e-4 at step 5,760          5e-5, constant
optimizer steps  7,680                               1,920
===============  ==================================  ==========================

Everything else --- dataset, split, normalization, the eight loss weights, the
spectral cap, batch 8, Adam betas (0.9, 0.95), no weight decay, no gradient
clipping, the validation protocol and the selection rule --- is the parent's,
unchanged.

**Why ninety days and not five hundred.** Bire et al. estimate the double gyre's
decorrelation time at roughly 90 days; beyond that the emulator is not expected
to reproduce a particular MITgcm trajectory, only to stay on the same
statistically stationary attractor. Training against exact ``x_{t+500}`` would
therefore optimize an impossible deterministic target, and the cheapest way to
minimize it is to suppress variability --- the opposite of what this run wants.
Ninety days is the longest horizon at which pointwise truth is still a
defensible objective.

**Why the loss is untouched.** The eight terms and their weights are the
parent's exactly. The only change is that the rollout they are evaluated over is
nine calls instead of six, so

    L_state   = E_10                    (unchanged)
    L_rollout = (1/8) sum_{k=2}^{9} E_{10k}      i.e. 20, 30, ..., 90 days

and the spectral, boundary, pressure-gradient, continuity and barotropic terms
each run through all nine calls. Changing the training protocol and the
objective in the same run would leave the result unattributable.

**Deliberately not changed here.** The spectral cap stays one-sided at
``rho = 1``; tightening it to 0.99 is the *next* single-variable experiment if
this one still rises, not a second change bundled into this one. Modes stay
32x32 --- the day-2000 high-k power fraction is already 0.0004 against truth
0.028, so the operator is too smooth, not too rough. The state stays all fifteen
levels, because the point of this emulator is a self-contained prognostic map
whose Jacobian can serve as a tangent/adjoint operator. And no ``tanh``
stabilizer: saturating regions drive ``tanh'`` toward zero, buying bounded
trajectories by destroying the sensitivities the whole programme exists to
study.

Budget: 1,920 steps x 8 samples x 9 autoregressive calls = 138,240 additional
state transitions, 37.5 % of the parent's 368,640, at microbatch 2 x
accumulation 4 so the deeper unroll's retained activations fit one GPU.

Entry points::

    python -m oceanfno.finetune preflight --contract config/...json
    python -m oceanfno.finetune run       --contract config/...json [--device cuda]
"""
from __future__ import annotations

import argparse
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
from .dataset import (
    DATASET_VERSION,
    EXPERIMENTS,
    HORIZON_DAYS,
    INFERENCE_RANGE,
    RolloutDataset,
    STATIC_FEATURES,
    TRAIN_CODE,
    TRAIN_RANGE,
    assert_store_is_v3,
    records_for_rollout_split,
    store_codes,
    store_wind_normalization,
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
from .perturbation_growth import (
    DIAGNOSTIC_CALLS,
    GROWTH_RATE_CEILING,
    growth_rate_summary,
    initial_direction,
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
from .train import (
    TrainingContractError,
    _verify_dataset,
    _verify_file,
    evaluate_loss,
    physical_static_block,
    physics_contexts,
)

VERSION = "model_c_production_1in_1out_spectralnorm_ft90_v1"

SLUG = VERSION

#: The frozen, published parent. Its weights are the initialization, its
#: normalization is reused byte-for-byte, and its selected checkpoint's
#: short-horizon skill is the reference this run must not degrade.
PARENT_VERSION = "model_c_production_1in_1out_spectralnorm_v1"

PARENT_OPTIMIZER_STEP = 7680

#: Measured on the parent's selected checkpoint under this exact validation
#: protocol (102 pooled S0/S1/S2 rollouts, 360 days, ten-day steps). Pinned here
#: so the acceptance gate has a reference that does not depend on rereading a
#: report at gate time; the report is verified by digest regardless.
PARENT_SHORT_AUC_10_90 = {
    "surface_speed": 0.08082015981855482,
    "sst": 0.8210369892984483,
    "phihyd_surface": 0.5791260363698443,
}

PARENT_GROWTH_RATE = 1.0132248253968004

CONTRACT_STATUS = "frozen_before_any_fine_tuned_metric"

#: The one substantive training change: nine autonomous calls, 90 days.
ROLLOUT_STEPS = 9

PARENT_ROLLOUT_STEPS = 6

#: The effective batch is the parent's. It is realized as smaller microbatches
#: because nine retained unrolls cost more activation memory than six: the
#: product that sets the peak is microbatch x calls, 2 x 9 = 18 against the
#: parent's 4 x 6 = 24, so this is the cheaper of the two.
BATCH_SIZE = 8

MICROBATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICROBATCH_SIZE

#: An order of magnitude below the parent's terminal rate. The parent already
#: knows the ten-day propagator; this stage adjusts where its autonomous
#: trajectory settles, and a rate that could relearn the propagator would
#: dissolve the thing being fine-tuned.
LEARNING_RATE = 5.0e-5

#: Constant. The parent's two-stage 5e-4 -> 1e-4 schedule exists to anneal a
#: from-scratch run; 1,920 steps at a rate already five times below its terminal
#: value has nothing left to anneal, and a decay would confound "the fine-tune
#: converged" with "the fine-tune stopped moving".
LEARNING_RATE_SCHEDULE = "constant"

ADAM_BETAS = (0.9, 0.95)

WEIGHT_DECAY = 0.0

MAXIMUM_STEPS = 1920

CHECKPOINT_STEPS = (480, 960, 1440, 1920)

#: Fresh, and different from the parent's 20260724: the batch order of a
#: fine-tune that reused it would replay the parent's own final epoch ordering.
SEED = 20260817

#: Nine calls need the full 90-day target sequence inside the training block, so
#: the latest usable start is day 5,909 (its last target is day 5,999). The
#: parent's six-call contract reached day 5,939.
TRAINING_STARTS_PER_REGIME = 5910

TRAINING_RECORDS = TRAINING_STARTS_PER_REGIME * len(EXPERIMENTS)

STATE_TRANSITIONS = MAXIMUM_STEPS * BATCH_SIZE * ROLLOUT_STEPS

PARENT_STATE_TRANSITIONS = 368640

#: Inherited unchanged from the parent's selection rule.
SHORT_AUC_TOLERANCE = 1.05

WORST_LONG_RATIO_CEILING = 0.85

#: New, and the point of the run: the fine-tuned model must not buy long-horizon
#: behaviour with short-horizon skill. Its selected checkpoint's 10--90-day
#: RMSE-AUC must be within 5 % of the parent's in every primary field.
PARENT_SHORT_AUC_TOLERANCE = 1.05

#: The 90--360-day RMSE curve's late secant slope divided by its early one,
#: both over 90-day windows: ``(E_360 - E_270) / (E_180 - E_90)``. Below one is
#: a curve that is flattening; above one is a curve that is still accelerating
#: away. The parent measures 0.95 / 1.73 / 1.65 (pressure / SST / speed) ---
#: already steepening in two of the three fields, which is the behaviour this
#: run exists to change.
FLATTENING_CEILING = 1.0

NORMALIZATION_NAME = f"{SLUG}_train_only_normalization.npz"

DIVERGENCE_NAME = f"{SLUG}_divergence.json"

CHECKPOINT_STEM = f"{SLUG}_step"

REPORT_NAME = f"{SLUG}_report.json"

ARRAYS_NAME = f"{SLUG}_arrays.npz"

FIGURE_NAME = f"{SLUG}_selection.png"

OUTPUT_ARTIFACTS = (REPORT_NAME, ARRAYS_NAME, FIGURE_NAME, README_NAME, MANIFEST_NAME)

REQUIRED_SOURCE_HASHES = frozenset(
    {
        "src/oceanfno/barotropic_transport.py",
        "src/oceanfno/continuity.py",
        "src/oceanfno/dataset.py",
        "src/oceanfno/finetune.py",
        "src/oceanfno/model.py",
        "src/oceanfno/objective.py",
        "src/oceanfno/pressure_gradient.py",
        "src/oceanfno/perturbation_growth.py",
        "src/oceanfno/runtime.py",
        "src/oceanfno/spectral_norm.py",
        "src/oceanfno/train.py",
        "src/oceanfno/validation.py",
    }
)

REQUIRED_MITGCM_SOURCES = (
    "mitgcm_zonal_spacing",
    "mitgcm_sst_relaxation",
    "mitgcm_declaration",
)

#: Everything inherited from the parent run, each pinned by digest.
REQUIRED_PARENT_SOURCES = (
    "parent_checkpoint",
    "parent_normalization",
    "parent_report",
)

#: Keys of the parent's normalization archive, reused verbatim.
NORMALIZATION_KEYS = (
    "pointwise_mean",
    "pointwise_raw_scale",
    "pointwise_scale",
    "channel_scale_floor",
    "increment_scale",
)


# ---------------------------------------------------------------------------
# the objective: the parent's eight terms over a nine-call rollout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FineTuneLossConfig(ProductionLossConfig):
    """The production objective, evaluated over nine autoregressive calls.

    Every coefficient is the parent's. ``rollout_steps`` is the only field that
    differs, and it is frozen at nine exactly the way the parent freezes six, so
    a fine-tune cannot silently drift to some other horizon.
    """

    rollout_steps: int = ROLLOUT_STEPS

    def __post_init__(self) -> None:
        expected = dict(_frozen_parent_fields())
        expected["rollout_steps"] = ROLLOUT_STEPS
        if self.to_dict() != expected:
            raise ValueError(
                "the fine-tuning objective is the production objective over nine "
                "calls; every coefficient is fixed and only the horizon differs"
            )


def _frozen_parent_fields() -> dict[str, Any]:
    """The parent's frozen coefficient set, read from the parent's own config."""

    return production_loss_config().to_dict()


def finetune_loss_config() -> FineTuneLossConfig:
    """Return the single frozen fine-tuning configuration."""

    return FineTuneLossConfig()


#: The five strings of the parent's loss contract that name the horizon. They
#: are prose, not behaviour --- the behaviour follows ``rollout_steps`` --- but a
#: published checkpoint that describes itself as six-call when it is nine-call
#: is a provenance defect, so they are restated rather than inherited.
_NINE_CALL_SEMANTICS = {
    "rollout": (
        "equal_mean_U_V_temperature_SSH_masked_relative_l2_at_"
        "20_30_40_50_60_70_80_and_90_days"
    ),
    "spectral": (
        "equal_mean_group_12_bin_amplitude_relative_l2_of_10_day_increments_"
        "on_exact_wet_rectangle_after_hann_taper_over_all_nine_calls"
    ),
    "boundary": (
        "equal_mean_U_V_temperature_SSH_relative_l2_at_10_20_30_40_50_60_70_80_"
        "90_days_on_first_4_wet_cells_east_of_western_wall"
    ),
    "pressure_gradient": (
        "relative_l2_of_the_horizontal_gradient_of_PHIHYD_reconstructed_from_"
        "THETA_and_ETAN_equal_weight_x_y_15_levels_and_9_rollout_calls"
    ),
    "continuity": (
        "truth_referenced_relative_squared_l2_of_the_free_surface_residual_"
        "d_eta_dt_plus_divergence_of_the_time_centered_depth_integrated_"
        "transport_chained_like_the_rollout_over_9_calls"
    ),
    "barotropic_transport": (
        "truth_referenced_relative_squared_l2_of_the_ten_day_tendency_of_the_"
        "depth_integrated_transport_equal_weight_zonal_and_meridional_over_9_calls"
    ),
}


def finetune_loss_contract(config: FineTuneLossConfig) -> dict[str, Any]:
    """Machine-readable semantics of the nine-call objective.

    Derived from the parent's contract rather than rewritten, so a change to any
    coefficient there propagates here and cannot be forgotten. Only the horizon
    prose and the activation clause are restated.
    """

    if not isinstance(config, FineTuneLossConfig):
        raise ValueError("the fine-tuning loss contract describes only the ft90 objective")
    contract = dict(loss_contract(config))
    contract.update(_NINE_CALL_SEMANTICS)
    contract["version"] = "production_v1_terms_over_a_ninety_day_rollout"
    contract["activation"] = (
        "every_term_active_from_fine_tuning_step_one_over_nine_autoregressive_calls"
    )
    contract["stage"] = "staged_fine_tune_of_a_published_parent_no_teacher_forcing"
    contract["parent_loss_contract_sha256"] = LOSS_CONTRACT_SHA256
    contract["weights_identical_to_parent"] = True
    return contract


def finetune_loss_contract_sha256(config: FineTuneLossConfig) -> str:
    encoded = json.dumps(
        finetune_loss_contract(config), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


FINETUNE_LOSS_CONTRACT_SHA256 = finetune_loss_contract_sha256(finetune_loss_config())


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the fine-tuning declaration and audit every field.

    The parent arm's loader exists to prove that *nothing* is inherited. This
    one proves the opposite, and proves it precisely: exactly one parent, named
    and hashed; its normalization reused rather than recomputed; its optimizer
    state explicitly not loaded; the architecture byte-identical to the parent's;
    and only the four declared training fields different.
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
    config = finetune_loss_config()

    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or architecture != ProductionArchitecture().to_dict()
    ):
        raise TrainingContractError(
            "the fine-tuning declaration must carry the parent's exact architecture"
        )

    # The defining property of this arm: exactly one parent, fully specified.
    if (
        initialization.get("from_scratch") is not False
        or initialization.get("load_model_state") is not True
        or initialization.get("load_optimizer_state") is not False
        or initialization.get("normalization_reused") is not True
        or initialization.get("parent_version") != PARENT_VERSION
        or int(initialization.get("parent_optimizer_step", -1)) != PARENT_OPTIMIZER_STEP
        or initialization.get("strict_state_dict_load") is not True
        or initialization.get("architecture_changed") is not False
    ):
        raise TrainingContractError(
            "the fine-tuning arm loads one named parent checkpoint strictly, "
            "reuses its normalization and starts Adam cold"
        )
    if str(initialization.get("parent_checkpoint", "")) != str(
        sources.get("parent_checkpoint", {}).get("path", "")
    ):
        raise TrainingContractError(
            "initialization.parent_checkpoint and sources.parent_checkpoint disagree"
        )
    missing_parents = [k for k in REQUIRED_PARENT_SOURCES if k not in sources]
    if missing_parents:
        raise TrainingContractError(
            f"the fine-tuning arm must pin every inherited artifact: {missing_parents}"
        )

    if (
        int(training.get("rollout_steps", -1)) != ROLLOUT_STEPS
        or int(training.get("parent_rollout_steps", -1)) != PARENT_ROLLOUT_STEPS
        or int(training.get("batch_size", -1)) != BATCH_SIZE
        or int(training.get("microbatch_size", -1)) != MICROBATCH_SIZE
        or int(training.get("gradient_accumulation_steps", -1))
        != GRADIENT_ACCUMULATION_STEPS
        or float(training.get("learning_rate", -1.0)) != LEARNING_RATE
        or training.get("learning_rate_schedule") != LEARNING_RATE_SCHEDULE
        or tuple(float(v) for v in training.get("adam_betas", ())) != ADAM_BETAS
        or float(training.get("weight_decay", -1.0)) != WEIGHT_DECAY
        or int(training.get("maximum_steps", -1)) != MAXIMUM_STEPS
        or tuple(training.get("checkpoint_steps", ())) != CHECKPOINT_STEPS
        or int(training.get("seed", -1)) != SEED
        or training.get("optimizer") != "adam"
        or training.get("fresh_optimizer_state") is not True
        or training.get("gradient_clipping") is not False
        or training.get("teacher_forcing_after_the_initial_state") is not False
        or int(training.get("state_transitions", -1)) != STATE_TRANSITIONS
        or int(training.get("training_starts_per_regime", -1))
        != TRAINING_STARTS_PER_REGIME
    ):
        raise TrainingContractError("the fine-tuning schedule changed")
    # A decayed rate would confound convergence with a shrinking step.
    for forbidden in ("decay_fraction", "decay_factor", "initial_learning_rate"):
        if forbidden in training:
            raise TrainingContractError(
                f"the fine-tuning rate is constant; remove training.{forbidden}"
            )

    if (
        loss.get("contract_sha256") != FINETUNE_LOSS_CONTRACT_SHA256
        or loss.get("parent_contract_sha256") != LOSS_CONTRACT_SHA256
        or loss.get("all_terms_active_from_step_1") is not True
        or loss.get("staged_fine_tuning") is not True
        or loss.get("weights_identical_to_parent") is not True
        or loss.get("contraction_penalty") is not False
        or int(loss.get("rollout_steps", -1)) != ROLLOUT_STEPS
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
    ):
        raise TrainingContractError("the fine-tuning objective changed")

    spectral = contract.get("spectral_normalization", {})
    if (
        spectral.get("applied") is not True
        or spectral.get("form") != "R_k <- R_k * min(1, 1 / sigma_max(R_k))"
        or spectral.get("applies_to") != "spectral_convolutions_only"
        or float(spectral.get("rho", -1.0)) != 1.0
        or int(spectral.get("power_iterations_per_forward", -1)) != POWER_ITERATIONS
        or int(spectral.get("warmup_iterations", -1)) != WARMUP_ITERATIONS
        or int(spectral.get("matrices_total", -1)) != 1632
        or spectral.get("checkpoints_materialized") is not True
        or spectral.get("adds_parameters") != 0
        or spectral.get("changed_from_parent") is not False
    ):
        raise TrainingContractError("the spectral-normalization declaration changed")

    if (
        normalization.get("recomputed_from_training_days_only") is not False
        or normalization.get("reused_from_a_previous_run") is not True
        or normalization.get("reused_from") != PARENT_VERSION
        or tuple(normalization.get("train_days", ())) != TRAIN_RANGE
    ):
        raise TrainingContractError(
            "the fine-tune continues the parent's map on the parent's dataset and "
            "must reuse the parent's normalization unchanged"
        )

    if (
        float(selection.get("short_auc_tolerance", -1.0)) != SHORT_AUC_TOLERANCE
        or float(selection.get("parent_short_auc_tolerance", -1.0))
        != PARENT_SHORT_AUC_TOLERANCE
        or float(selection.get("worst_long_ratio_ceiling", -1.0))
        != WORST_LONG_RATIO_CEILING
        or float(selection.get("flattening_ceiling", -1.0)) != FLATTENING_CEILING
        or tuple(selection.get("primary_fields", ())) != PRIMARY_FIELDS
        or int(selection.get("rollout_days", -1)) != 360
        or float(selection.get("growth_rate_ceiling", -1.0)) != GROWTH_RATE_CEILING
        or int(selection.get("growth_rate_calls", -1)) != DIAGNOSTIC_CALLS
        or selection.get("parent_short_auc_10_90") != PARENT_SHORT_AUC_10_90
    ):
        raise TrainingContractError("the checkpoint-selection rule changed")

    if (
        not str(output.get("project_root", "")).endswith(VERSION)
        or not str(output.get("scratch_root", "")).endswith(VERSION)
        or tuple(output.get("artifacts", ())) != OUTPUT_ARTIFACTS
    ):
        raise TrainingContractError("the fine-tuning output declaration changed")

    if not REQUIRED_SOURCE_HASHES.issubset(hashes) or not set(
        REQUIRED_MITGCM_SOURCES
    ).issubset(sources):
        raise TrainingContractError("the fine-tuning source declaration is incomplete")

    ProductionArchitecture(**architecture)
    if verify_sources:
        _verify_dataset(contract)
        for key in (*REQUIRED_MITGCM_SOURCES, *REQUIRED_PARENT_SOURCES):
            _verify_file(sources[key], key)
        root = resolved.parents[1]
        for relative, expected in hashes.items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise TrainingContractError(f"a pinned source changed: {relative}")
    return contract, resolved, _file_sha256(resolved)


# ---------------------------------------------------------------------------
# inheritance
# ---------------------------------------------------------------------------


def inherited_normalizers(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Load the parent's normalization archive, unchanged.

    This is the one place inheritance is *desirable*: the fine-tune continues
    exactly the same map on exactly the same dataset, and recomputing the
    normalizers --- even from the same days, even to the same recipe --- would
    move the coordinate system the trained operator already lives in.
    """

    path = _verify_file(contract["sources"]["parent_normalization"], "parent normalization")
    with np.load(path) as archive:
        missing = [key for key in NORMALIZATION_KEYS if key not in archive]
        if missing:
            raise TrainingContractError(
                f"the parent normalization archive is missing {missing}"
            )
        values = {key: np.ascontiguousarray(archive[key]) for key in NORMALIZATION_KEYS}
    expected = (STATE_CHANNEL_COUNT, 62, 62)
    if (
        values["pointwise_mean"].shape != expected
        or values["pointwise_scale"].shape != expected
        or values["increment_scale"].shape != (STATE_CHANNEL_COUNT,)
        or not np.all(np.isfinite(values["pointwise_scale"]))
        or np.any(values["pointwise_scale"] <= 0.0)
        or np.any(values["increment_scale"] <= 0.0)
    ):
        raise TrainingContractError("the parent normalization archive is not usable")
    values["path"] = path
    values["summary"] = {
        "source": f"reused_unchanged_from_{PARENT_VERSION}",
        "recomputed": False,
        "reason": (
            "the fine-tune continues the same operator on the same dataset; "
            "recomputing would alter the coordinate system the parent was "
            "trained in for no gain"
        ),
        "train_days": list(TRAIN_RANGE),
        "archive_sha256": _file_sha256(path),
    }
    return values


def load_parent(
    contract: Mapping[str, Any],
    architecture: ProductionArchitecture,
    model: Any,
    device: Any,
) -> dict[str, Any]:
    """Load the parent's published weights strictly into a freshly built model.

    The parent's checkpoint is *materialized* --- its spectral weights already
    carry the cap baked in --- so it loads into a plain
    :class:`~oceanfno.model.ProductionFNO` with no reparameterization attached,
    which is exactly what is built here. The cap is reinstalled afterwards, on
    weights that already satisfy it, so the fine-tune begins from an operator
    functionally identical to the published one.

    Nothing but ``model_state_dict`` is taken. Adam's moments are not in the
    checkpoint and are not reconstructed: a fine-tune that inherited second
    moments accumulated at 5e-4 would take its first steps at an effective rate
    set by the parent's schedule rather than by this contract.
    """

    path = _verify_file(contract["sources"]["parent_checkpoint"], "parent checkpoint")
    payload = torch.load(path, map_location=device, weights_only=False)
    declared = payload.get("architecture", {})
    if (
        payload.get("version") != PARENT_VERSION
        or int(payload.get("optimizer_step", -1)) != PARENT_OPTIMIZER_STEP
        or declared != architecture.to_dict()
        or int(payload.get("input_states", -1)) != 1
        or int(payload.get("rollout_steps", -1)) != PARENT_ROLLOUT_STEPS
        or payload.get("loss_contract_sha256") != LOSS_CONTRACT_SHA256
        or payload.get("spectral_normalization") != "per_mode_sigma_max_capped_at_one"
        or payload.get("from_scratch") is not True
    ):
        raise TrainingContractError(
            "the parent checkpoint is not the declared published production step"
        )
    # ``optimizer_step`` is a counter and is expected; what must not be present
    # is a moment estimate, which this stage would refuse to inherit.
    carried = [
        key
        for key in payload
        if key in ("optimizer_state_dict", "optimizer_state", "optimizer_states")
    ]
    if carried:
        raise TrainingContractError(
            f"the parent checkpoint carries optimizer state {carried}; "
            "this arm starts Adam cold"
        )
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise TrainingContractError(
            "the parent state dict did not load strictly into the declared architecture"
        )
    if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise TrainingContractError("loading the parent changed the parameter count")
    return {
        "version": PARENT_VERSION,
        "checkpoint": str(path),
        "checkpoint_sha256": _file_sha256(path),
        "optimizer_step": PARENT_OPTIMIZER_STEP,
        "loss_contract_sha256": payload.get("loss_contract_sha256"),
        "rollout_steps": PARENT_ROLLOUT_STEPS,
        "loaded": "model_state_dict_only_strict",
        "load_optimizer_state": False,
        "architecture_changed": False,
        "migration": None,
        "function_preserving_load": (
            "the parent checkpoint is materialized, so the plain model built here "
            "reproduces the published operator exactly before the cap is reinstalled"
        ),
    }


def parent_reference(contract: Mapping[str, Any]) -> dict[str, Any]:
    """The parent's published validation numbers, read from its pinned report."""

    path = _verify_file(contract["sources"]["parent_report"], "parent report")
    report = json.loads(path.read_text())
    if (
        report.get("version") != PARENT_VERSION
        or report.get("status") != "complete"
        or int(report["published_checkpoint"]["optimizer_step"]) != PARENT_OPTIMIZER_STEP
    ):
        raise TrainingContractError("the parent report is not the published production run")
    summary = next(
        s
        for s in report["validation_summaries"]
        if int(s["optimizer_step"]) == PARENT_OPTIMIZER_STEP
    )
    short = {field: float(summary["short_auc_10_90"][field]) for field in PRIMARY_FIELDS}
    if any(
        abs(short[field] - PARENT_SHORT_AUC_10_90[field]) > 1.0e-9
        for field in PRIMARY_FIELDS
    ):
        raise TrainingContractError(
            "the parent report's short AUC disagrees with the pinned reference"
        )
    return {
        "version": PARENT_VERSION,
        "report": str(path),
        "report_sha256": _file_sha256(path),
        "optimizer_step": PARENT_OPTIMIZER_STEP,
        "short_auc_10_90": short,
        "long_ratio_to_climatology": {
            field: float(summary["long_ratio_to_climatology"][field])
            for field in PRIMARY_FIELDS
        },
        "acc_day200": dict(summary["acc_day200"]),
        "maximum_normalized_amplitude": float(summary["maximum_normalized_amplitude"]),
        "growth_rate_per_call": PARENT_GROWTH_RATE,
        "rmse_curve_flattening_90_360": rmse_curve_flattening(summary),
        "measured_under": "the identical 360-day 102-start validation protocol",
    }


# ---------------------------------------------------------------------------
# diagnostics the fine-tune adds
# ---------------------------------------------------------------------------


def rmse_curve_flattening(summary: Mapping[str, Any]) -> dict[str, float]:
    """Late over early secant slope of the 90--360-day RMSE curve, per field.

        (E_360 - E_270) / (E_180 - E_90)

    Both windows are 90 days wide, so this is a pure ratio of slopes. Below one
    is a curve that is decelerating toward a plateau --- the behaviour a model
    on the right attractor shows once past the decorrelation time. Above one is
    a curve still accelerating away, which is what the parent does.

    Returns ``inf`` where the early window did not grow, so an undefined ratio
    fails the condition rather than passing it silently.
    """

    leads = np.asarray(summary["lead_days"], dtype=np.int64)
    result: dict[str, float] = {}
    for field in PRIMARY_FIELDS:
        curve = np.asarray(summary["mean_rmse"]["model"][field], dtype=np.float64)
        if curve.shape != leads.shape:
            raise ValueError("the RMSE curve and the lead axis disagree")

        def at(day: int) -> float:
            index = np.flatnonzero(leads == day)
            if index.size != 1:
                raise ValueError(f"the validation curve has no lead at day {day}")
            return float(curve[int(index[0])])

        early = at(180) - at(90)
        late = at(360) - at(270)
        result[field] = float("inf") if early <= 0.0 else float(late / early)
    return result


def split_summary() -> dict[str, Any]:
    """The shared split summary with this arm's nine-step start bounds."""

    summary = dict(verify())
    summary["training_rollout_steps"] = ROLLOUT_STEPS
    summary["input_states"] = 1
    summary["earliest_training_rollout_start"] = 0
    summary["latest_training_rollout_start"] = TRAINING_STARTS_PER_REGIME - 1
    summary["training_starts_per_regime"] = TRAINING_STARTS_PER_REGIME
    summary["training_sequences"] = TRAINING_RECORDS
    summary["static_channels"] = list(STATIC_FEATURES)
    summary["history_note"] = (
        "the map still reads one time level, so day 0 remains a valid start; the "
        "nine-call target sequence is what moves the latest start from day 5,939 "
        "to day 5,909, whose final target is day 5,999"
    )
    summary["parent_training_starts_per_regime"] = 5940
    summary["starts_lost_to_the_longer_rollout"] = 5940 - TRAINING_STARTS_PER_REGIME
    return summary


def acceptance_gate(
    selected: Mapping[str, Any],
    best_short: Mapping[str, float],
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    """The five validation-measurable conditions declared for this run.

    Three are the parent's, inherited unchanged. Two are new and are the reason
    the run exists: the fine-tune must not pay for its longer horizon with
    short-horizon skill, and its 90--360-day error curve must begin to flatten
    rather than steepen.

    The 2,000-day conditions --- amplitude, streamfunction, western-boundary
    structure, spectral power, the ratio to climatology at long lead --- are
    where the attractor question is actually settled, and only the figure
    package runs a 2,000-day rollout. They are recorded as deferred rather than
    silently passing.
    """

    short_best = {
        field: float(selected["short_auc_10_90"][field]) / float(best_short[field])
        for field in PRIMARY_FIELDS
    }
    short_parent = {
        field: float(selected["short_auc_10_90"][field])
        / float(parent["short_auc_10_90"][field])
        for field in PRIMARY_FIELDS
    }
    worst_long = max(float(v) for v in selected["long_ratio_to_climatology"].values())
    flattening = rmse_curve_flattening(selected)
    worst_flattening = max(flattening.values())
    growth = selected.get("perturbation_growth", {})
    measured = growth.get("worst_growth_rate_per_call")
    worst_growth = float("inf") if measured is None else float(measured)

    best_pass = all(value <= SHORT_AUC_TOLERANCE for value in short_best.values())
    parent_pass = all(
        value <= PARENT_SHORT_AUC_TOLERANCE for value in short_parent.values()
    )
    long_pass = worst_long <= WORST_LONG_RATIO_CEILING
    flattening_pass = worst_flattening <= FLATTENING_CEILING
    growth_pass = worst_growth <= GROWTH_RATE_CEILING
    return {
        "short_auc_10_90_ratio_to_best_checkpoint": short_best,
        "short_auc_tolerance": SHORT_AUC_TOLERANCE,
        "short_auc_within_5_percent_of_best_in_every_field": bool(best_pass),
        "short_auc_10_90_ratio_to_parent": short_parent,
        "parent_short_auc_10_90": dict(parent["short_auc_10_90"]),
        "parent_short_auc_tolerance": PARENT_SHORT_AUC_TOLERANCE,
        "short_auc_within_5_percent_of_the_parent_in_every_field": bool(parent_pass),
        "worst_long_ratio_to_climatology": worst_long,
        "worst_long_ratio_ceiling": WORST_LONG_RATIO_CEILING,
        "worst_long_ratio_at_or_below_ceiling": bool(long_pass),
        "rmse_curve_flattening_90_360": flattening,
        "parent_rmse_curve_flattening_90_360": dict(
            parent["rmse_curve_flattening_90_360"]
        ),
        "flattening_ceiling": FLATTENING_CEILING,
        "rmse_curve_flattening_at_or_below_one_in_every_field": bool(flattening_pass),
        "worst_perturbation_growth_rate_per_call": worst_growth,
        "growth_rate_ceiling": GROWTH_RATE_CEILING,
        "growth_rate_at_or_below_ceiling": bool(growth_pass),
        "parent_growth_rate_per_call": float(parent["growth_rate_per_call"]),
        "perturbation_growth": dict(growth),
        "validation_conditions_pass": bool(
            best_pass and parent_pass and long_pass and flattening_pass and growth_pass
        ),
        "flattening_definition": (
            "(E_360 - E_270) / (E_180 - E_90) on the mean model RMSE curve; both "
            "windows are 90 days wide, so below one is a decelerating curve"
        ),
        "long_horizon_window_note": (
            "worst_long_ratio_to_climatology integrates 90-360 days only. Past "
            "the roughly 90-day decorrelation time the requirement is not "
            "pointwise agreement with MITgcm but staying on the same stationary "
            "attractor, which is why the flattening and growth-rate conditions "
            "exist alongside it and why the day-2000 conditions below are "
            "deferred rather than approximated here"
        ),
        "deferred_to_the_figure_package": [
            "2000_day_all_values_finite",
            "2000_day_maximum_normalized_magnitude_at_most_8",
            "2000_day_streamfunction_minimum_at_least_minus_33_sv",
            "day_2000_streamfunction_anomaly_rms_ratio_near_one",
            "day_2000_western_band_to_interior_anomaly_ratio_controlled",
            "day_2000_directional_spectrum_and_gradient_sharpness",
            "day_2000_rmse_ratio_to_climatology_plateaus_rather_than_rising",
            "western_boundary_sharp_and_gyre_identifiable_by_inspection",
        ],
    }


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify the contract, the parent, the record counts and the load.

    Beyond the parent arm's checks this proves the two properties a staged run
    lives or dies by: that the parent's weights load strictly into the declared
    architecture, and that reinstalling the spectral cap on those already-capped
    weights leaves the operator where it was. If the second check drifts, the
    fine-tune would be starting from a different model than the published one.
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
    if max(t for _, t in records) != TRAINING_STARTS_PER_REGIME - 1:
        raise TrainingContractError("the latest nine-call training start moved")
    architecture = ProductionArchitecture(**contract["architecture"])
    parent = parent_reference(contract)
    result: dict[str, Any] = {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "dataset_version": str(group.attrs["version"]),
        "split": split_summary(),
        "loss_contract_sha256": FINETUNE_LOSS_CONTRACT_SHA256,
        "parent_loss_contract_sha256": LOSS_CONTRACT_SHA256,
        "from_scratch": False,
        "parent": parent,
        "rollout_steps": ROLLOUT_STEPS,
        "parent_rollout_steps": PARENT_ROLLOUT_STEPS,
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
        "parent_state_transitions": PARENT_STATE_TRANSITIONS,
        "inference_state_opened": False,
    }
    normalizers = inherited_normalizers(contract)
    result["normalization"] = normalizers["summary"]
    if torch is not None:
        seed_everything(SEED)
        device = _device("cpu")
        model = build_model(architecture).to(device)
        result["parent_load"] = load_parent(contract, architecture, model, device)
        model.eval()
        probe = torch.randn(
            (2, architecture.in_channels, 62, 62),
            generator=torch.Generator().manual_seed(SEED),
        )
        with torch.no_grad():
            published = model(probe).detach().clone()
        # Reinstalling the cap on materialized weights must be a near-identity:
        # the parent's sigma already sits at one up to the power-iteration lag.
        spectral = apply_mode_spectral_norm(model)
        with torch.no_grad():
            reinstalled = model(probe).detach()
            reference = float(published.square().mean().sqrt())
            deviation = float((reinstalled - published).abs().max()) / max(
                reference, 1.0e-12
            )
        if deviation > 1.0e-2:
            raise TrainingContractError(
                "reinstalling the spectral cap moved the parent operator by "
                f"{deviation:.4g} of its own rms; the fine-tune would not be "
                "starting from the published model"
            )
        if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
            raise TrainingContractError("spectral normalization changed the parameter count")
        result["spectral_normalization"] = spectral
        result["spectral_reinstall_check"] = {
            "maximum_relative_deviation_from_the_published_operator": deviation,
            "tolerance": 1.0e-2,
            "sigma_after_reinstall": mode_sigma_summary(model),
            "note": (
                "the parent checkpoint is materialized, so its spectral weights "
                "already satisfy the cap; a fresh warmup re-estimates sigma and "
                "may shave it by the power-iteration lag, nothing more"
            ),
        }
        result["parameter_count"] = parameter_count(model)
        # Both physics terms must still be identically zero on the nine-call
        # shape truth is scored against itself in.
        pressure, continuity = physics_contexts(
            group,
            normalizers["pointwise_mean"],
            normalizers["pointwise_scale"],
            contract["sources"]["mitgcm_zonal_spacing"]["path"],
        )
        from .barotropic_transport import barotropic_transport_relative_l2
        from .continuity import continuity_relative_l2
        from .pressure_gradient import pressure_gradient_relative_l2

        dummy = torch.zeros(
            (1, ROLLOUT_STEPS, STATE_CHANNEL_COUNT, 62, 62), dtype=torch.float32
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
        result["growth_rate_ceiling"] = GROWTH_RATE_CEILING
        result["flattening_ceiling"] = FLATTENING_CEILING
    return result


# ---------------------------------------------------------------------------
# the README
# ---------------------------------------------------------------------------


def _number(value: Any, digits: int = 3) -> str:
    """Format a diagnostic that may legitimately be missing or non-finite.

    ``_readme`` renders the report *after* :func:`json_safe` has replaced every
    non-finite float with ``None`` --- and a failed growth-rate fit or an
    undefined flattening ratio is a real, expected outcome, not a bug. Formatting
    one of those directly raises ``TypeError`` at the very end of a completed
    run, which is how a v3 job lost 2 h 46 m of finished work. Nothing rendered
    here is ever allowed to do that again.
    """

    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{digits}f}"


def _worst(values: Mapping[str, Any]) -> str:
    """The largest of a per-field diagnostic, tolerating missing entries."""

    finite = [float(v) for v in values.values() if v is not None and np.isfinite(v)]
    if len(finite) < len(values):
        # A missing or infinite entry *is* the worst case; say so rather than
        # quietly reporting the largest of the ones that happened to be finite.
        return "n/a"
    return _number(max(finite), 2)


def _readme(report: Mapping[str, Any]) -> str:
    decision = report["selection_decision"]
    gate = report["acceptance_gate"]
    parent = report["parent"]
    rows = "\n".join(
        "| {step:,} | {short} | {long} | {flat} | {growth} |".format(
            step=int(summary["optimizer_step"]),
            short=" / ".join(
                _number(summary["short_auc_10_90"][field]) for field in PRIMARY_FIELDS
            ),
            long=" / ".join(
                _number(summary["long_ratio_to_climatology"][field])
                for field in PRIMARY_FIELDS
            ),
            flat=" / ".join(
                _number(rmse_curve_flattening(summary)[field], 2)
                for field in PRIMARY_FIELDS
            ),
            growth=_number(
                summary["perturbation_growth"]["worst_growth_rate_per_call"], 4
            ),
        )
        for summary in report["validation_summaries"]
    )
    parent_short = " / ".join(
        _number(parent["short_auc_10_90"][field]) for field in PRIMARY_FIELDS
    )
    parent_long = " / ".join(
        _number(parent["long_ratio_to_climatology"][field]) for field in PRIMARY_FIELDS
    )
    parent_flat = " / ".join(
        _number(parent["rmse_curve_flattening_90_360"][field], 2)
        for field in PRIMARY_FIELDS
    )
    ratios = " / ".join(
        _number(gate["short_auc_10_90_ratio_to_parent"][field]) for field in PRIMARY_FIELDS
    )
    return f"""# Ninety-day fine-tune of the production emulator

    F_theta: [x_t, S] -> x_(t+10)

A staged second training stage on the published production operator. **No
architecture change**: the same 32 x 32-mode, width-128, three-block FNO with
six pointwise LayerNorms, the 4C Channel MLP, 10 % domain padding, the bias-free
3 x 3 local branch and per-mode spectral normalization at rho = 1.
Parameter count {int(report['parameter_count']):,}, unchanged.

## Lineage

    {PARENT_VERSION}
            |  selected.pt, optimizer step {PARENT_OPTIMIZER_STEP:,}
            v
    {VERSION}

The parent is frozen and stays published. Its `model_state_dict` is loaded
strictly into a plain model --- the checkpoint is materialized, so that plain
model *is* the published operator --- and the per-mode cap is then reinstalled
on weights that already satisfy it. Adam starts cold; the parent's second
moments, accumulated at 5e-4, are not inherited. The parent's pointwise
normalizers and increment scales are reused byte-for-byte, because this stage
continues the same map on the same dataset and recomputing them would move the
coordinate system the operator already lives in.

## What changed

| | parent | this run |
| --- | --- | --- |
| initialization | random | parent `selected.pt` |
| autoregressive calls | {PARENT_ROLLOUT_STEPS} (60 days) | **{ROLLOUT_STEPS} (90 days)** |
| learning rate | 5e-4 -> 1e-4 | **{LEARNING_RATE:g}, constant** |
| optimizer steps | 7,680 | **{MAXIMUM_STEPS:,}** |
| microbatch x accumulation | 4 x 2 | 2 x 4 |
| state transitions | {PARENT_STATE_TRANSITIONS:,} | {STATE_TRANSITIONS:,} |

Nothing else. The eight loss terms and their weights are identical:

    L = L_state + 0.001 L_inc + 0.50 L_rollout + 1e-5 L_spectral
        + 0.065 L_boundary + 0.05 L_pressure + 0.05 L_continuity
        + 0.05 L_barotropic

with `L_state = E_10` and `L_rollout = (1/8) sum_(k=2)^(9) E_(10k)`, so the
model must remain usable on its own generated states through 20, 30, ..., 90
days. The spectral, boundary and three physics terms likewise run through all
nine calls. Ninety days rather than 200 or 2,000 because the double gyre
decorrelates in roughly 90 days: past that, exact pointwise truth stops being a
defensible target and minimizing it rewards suppressing variability.

## Training

Nine-step autoregressive, no teacher forcing after the initial state:
{report['counts']['training_rollout_records']:,} sequences
({report['counts']['training_starts_per_regime']:,} starts per regime --- 30
fewer than the parent's, since the last usable start must still fit a 90-day
target sequence inside the training block), {MAXIMUM_STEPS:,} optimizer steps at
batch {BATCH_SIZE} (microbatch {MICROBATCH_SIZE} x accumulation
{GRADIENT_ACCUMULATION_STEPS}) = {STATE_TRANSITIONS:,} state transitions.

## Selection

| step | short AUC 10--90 (speed / SST / pressure) | long / climatology | flattening | growth rate |
| --- | --- | --- | --- | --- |
{rows}
| *parent* | *{parent_short}* | *{parent_long}* | *{parent_flat}* | *{_number(parent['growth_rate_per_call'], 4)}* |

Selected step {int(decision['selected_optimizer_step']):,} via
`{decision['branch']}`.

Short-horizon skill against the parent: {ratios} (tolerance
{PARENT_SHORT_AUC_TOLERANCE}) ---
**{'within' if gate['short_auc_within_5_percent_of_the_parent_in_every_field'] else 'outside'}**
the declared 5 % budget. The 90--360-day curve's late/early slope ratio is
{_worst(gate['rmse_curve_flattening_90_360'])} at worst against the
parent's {_worst(gate['parent_rmse_curve_flattening_90_360'])};
below one is a curve flattening toward a plateau. Growth rate
{_number(gate['worst_perturbation_growth_rate_per_call'], 5)} per call against the
parent's {_number(parent['growth_rate_per_call'], 5)} (ceiling {GROWTH_RATE_CEILING}).

Validation gate: **{'pass' if gate['validation_conditions_pass'] else 'fail'}**.
The day-2000 conditions are deferred to the figure package, which is the only
stage that runs a 2,000-day rollout, and is where the attractor question is
actually settled: the requirement there is that RMSE *plateaus* near the error
between two independent states of the same climate, not that it reproduces
MITgcm pointwise.

Evaluation uses the nested validation/inference protocol; there is no
independent third test split.

Report content SHA-256: `{report['content_sha256']}`.
"""


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    """Fine-tune from the parent, validate every checkpoint, publish one."""

    if torch is None or DataLoader is None:  # pragma: no cover - environment dependent
        raise RuntimeError("fine-tuning the production emulator requires PyTorch")
    require_runtime()
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    dataset = _verify_dataset(contract)
    summary_of_split = split_summary()
    scratch = Path(contract["output"]["scratch_root"]).resolve()
    project = Path(contract["output"]["project_root"]).resolve()
    scratch_tmp = scratch.with_name(scratch.name + ".tmp")
    project_tmp = project.with_name(project.name + ".tmp")
    if any(p.exists() for p in (scratch, project, scratch_tmp, project_tmp)):
        raise FileExistsError("refusing to overwrite fine-tuning output")

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

    # Inherited, not recomputed. This is the one deliberate inheritance.
    normalizers = inherited_normalizers(contract)
    point_mean = normalizers["pointwise_mean"]
    point_scale = normalizers["pointwise_scale"]
    increment_values = normalizers["increment_scale"]
    parent = parent_reference(contract)
    statics, static_provenance = physical_static_block(
        contract["sources"], group, point_mean, point_scale
    )
    pressure_context, continuity_context = physics_contexts(
        group, point_mean, point_scale, contract["sources"]["mitgcm_zonal_spacing"]["path"]
    )
    climatology_state, climatology_derived, climatology_days = train_only_climatology(
        state, wet_array
    )

    loss_config = finetune_loss_config()
    training_records = records_for_rollout_split(
        pair_split, TRAIN_CODE, rollout_steps=loss_config.rollout_steps
    )
    if len(training_records) != TRAINING_RECORDS:
        raise TrainingContractError("the fine-tuning record count changed")
    training_dataset = RolloutDataset(
        dataset,
        training_records,
        point_mean,
        point_scale,
        statics,
        rollout_steps=loss_config.rollout_steps,
    )
    microbatch = int(training["microbatch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            training_dataset, microbatch, int(training["seed"])
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    architecture = ProductionArchitecture(**contract["architecture"])
    model = build_model(architecture).to(device)
    parent_load = load_parent(contract, architecture, model, device)
    # Reinstalled on weights that already satisfy it, so the run starts from the
    # published operator rather than from a re-scaled copy of it.
    spectral_provenance = apply_mode_spectral_norm(model)
    spectral_provenance["sigma_at_parent_load"] = mode_sigma_summary(model)
    spectral_provenance["rho"] = 1.0
    spectral_provenance["changed_from_parent"] = False
    count = parameter_count(model)
    if count != EXPECTED_PARAMETER_COUNT:
        raise TrainingContractError(
            f"the architecture builds {count:,} parameters, not {EXPECTED_PARAMETER_COUNT:,}"
        )
    # Cold Adam. Nothing about the parent's moment estimates survives.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
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

    scratch_tmp.parent.mkdir(parents=True, exist_ok=True)
    project_tmp.parent.mkdir(parents=True, exist_ok=True)
    scratch_tmp.mkdir()
    project_tmp.mkdir()
    checkpoint_directory = scratch_tmp / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir()
    # Copied, not rewritten: the published child normalization is byte-identical
    # to the parent's, so the two digests agree and the reuse is checkable.
    normalization_path = scratch_tmp / NORMALIZATION_NAME
    shutil.copy2(normalizers["path"], normalization_path)
    if _file_sha256(normalization_path) != normalizers["summary"]["archive_sha256"]:
        raise TrainingContractError("the inherited normalization did not copy intact")

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

    def _diverged(step: int, reason: str) -> None:
        (project_tmp / DIVERGENCE_NAME).write_text(
            json.dumps(
                {
                    "status": "diverged",
                    "version": VERSION,
                    "parent": PARENT_VERSION,
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
        raise DivergenceError(f"{reason} at optimizer step {step}")

    for step in range(1, maximum_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        model.train()
        step_samples = 0
        for micro in range(accumulation):
            try:
                raw_features, futures = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_features, futures = next(iterator)
            features = raw_features.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
            # Nine calls; the state is self-generated from the second onward.
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
                _diverged(step, "fine-tuning objective became non-finite")
            (terms["total"] / accumulation).backward()
            batch = int(features.shape[0])
            for name in AUDIT_TERMS:
                totals[name] += float(terms[name].detach().cpu()) * batch
            growth_total += float(growth.cpu()) * batch
            step_samples += batch
        if not all(
            bool(torch.isfinite(p.grad).all().item())
            for p in model.parameters()
            if p.grad is not None
        ):
            _diverged(step, "fine-tuning gradients became non-finite")
        optimizer.step()
        samples += step_samples

        if step not in CHECKPOINT_STEPS:
            continue
        window = {name: totals[name] / samples for name in AUDIT_TERMS}
        history_record = {
            "optimizer_step": step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_window": window,
            "mean_single_call_amplification": growth_total / samples,
            "spectral_norm": mode_sigma_summary(model),
        }
        history.append(history_record)
        path = checkpoint_directory / f"{CHECKPOINT_STEM}_{step:05d}.pt"
        torch.save(
            {
                "version": VERSION,
                "optimizer_step": step,
                "architecture": architecture.to_dict(),
                "contract": str(resolved_contract),
                "contract_sha256": contract_sha,
                "dataset_version": DATASET_VERSION,
                "loss_contract": finetune_loss_contract(loss_config),
                "loss_contract_sha256": FINETUNE_LOSS_CONTRACT_SHA256,
                "rollout_steps": loss_config.rollout_steps,
                "input_states": 1,
                "static_channels": list(STATIC_FEATURES),
                "from_scratch": False,
                "parent_version": PARENT_VERSION,
                "parent_checkpoint_sha256": parent_load["checkpoint_sha256"],
                "parent_optimizer_step": PARENT_OPTIMIZER_STEP,
                "spectral_normalization": "per_mode_sigma_max_capped_at_one",
                "training_history_record": history_record,
                # Materialized exactly as the parent's was, so a published
                # fine-tuned checkpoint loads into a plain ProductionFNO and the
                # inference operator is the one that was trained.
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

    if len(checkpoints) != len(CHECKPOINT_STEPS):
        raise TrainingContractError("not every declared checkpoint was written")

    records = validation_records()
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
        value["perturbation_growth"] = growth_rate_summary(
            probe, growth_states, growth_static, wet
        )
        evaluated_arrays.append(value.pop("arrays"))
        value["optimizer_step"] = int(record["optimizer_step"])
        value["rmse_curve_flattening_90_360"] = rmse_curve_flattening(value)
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
        "normalization_identical_to_parent": True,
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
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": {
            "path": str(dataset),
            "version": DATASET_VERSION,
            "metadata_sha256": _file_sha256(dataset / ".zmetadata"),
        },
        "split": summary_of_split,
        "architecture": architecture.to_dict(),
        "architecture_changed_from_parent": False,
        "parameter_count": count,
        "stage": "staged_ninety_day_autoregressive_fine_tune",
        "parent": parent,
        "initialization": {
            "from_scratch": False,
            "parent_version": PARENT_VERSION,
            "parent_checkpoint": parent_load["checkpoint"],
            "parent_checkpoint_sha256": parent_load["checkpoint_sha256"],
            "parent_optimizer_step": PARENT_OPTIMIZER_STEP,
            "load_model_state": True,
            "strict_state_dict_load": True,
            "load_optimizer_state": False,
            "normalization_reused": True,
            "architecture_changed": False,
            "seed": int(training["seed"]),
            "parent_load": parent_load,
        },
        "normalization": {
            "reused_from": PARENT_VERSION,
            "recomputed": False,
            "reused_from_a_previous_run": True,
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
        "loss_contract": finetune_loss_contract(loss_config),
        "loss_contract_sha256": FINETUNE_LOSS_CONTRACT_SHA256,
        "parent_loss_contract_sha256": LOSS_CONTRACT_SHA256,
        "fine_tuned_relative_to": {
            "version": PARENT_VERSION,
            "parent_growth_rate_per_call": PARENT_GROWTH_RATE,
            "weight_lineage": f"{PARENT_VERSION}/selected.pt step {PARENT_OPTIMIZER_STEP}",
            "changed": [
                "initialization_is_the_parent_checkpoint_not_random",
                "rollout_steps_6_to_9_sixty_to_ninety_days",
                "learning_rate_5e-4_decayed_to_a_constant_5e-5",
                "maximum_steps_7680_to_1920",
                "microbatch_4x2_to_2x4_for_the_deeper_unroll",
            ],
            "unchanged": [
                "architecture", "static_channels", "modes", "width", "blocks",
                "local_branch", "spectral_normalization", "spectral_cap_rho",
                "dataset", "split", "normalization", "the_eight_loss_terms",
                "loss_weights", "adam_betas", "weight_decay", "batch_size",
                "gradient_clipping", "validation_protocol", "selection_rule",
                "inference_protocol",
            ],
            "success_criterion": (
                "0-90 day skill within 5 percent of the parent, 90-360 day RMSE "
                "still below climatology and beginning to flatten rather than "
                "steepen, and at day 360-2000 a model that settles onto the "
                "correct climate attractor -- mean, std, RMS anomaly, "
                "streamfunction amplitude, western-boundary structure and "
                "spectral power -- rather than one that reproduces MITgcm "
                "pointwise"
            ),
            "next_experiment_if_this_still_rises": (
                "the same ninety-day protocol with the spectral cap tightened to "
                "rho = 0.99; deliberately not combined with this run"
            ),
        },
        "spectral_normalization": spectral_provenance,
        "contraction_penalty": False,
        "temporal_context": {
            "input_states": 1,
            "map": "x_t -> x_t_plus_10",
            "prediction": "direct_state_not_residual",
            "autoregression": "no_teacher_forcing_after_the_initial_state",
            "training_horizon_days": ROLLOUT_STEPS * HORIZON_DAYS,
            "parent_training_horizon_days": PARENT_ROLLOUT_STEPS * HORIZON_DAYS,
        },
        "static_channels": {
            "channels": list(STATIC_FEATURES),
            "external_input_channels": architecture.in_channels,
            "lifting_input_channels": architecture.lifting_in_channels,
            "provenance": static_provenance,
        },
        "optimizer": {
            "name": "adam",
            "learning_rate": float(training["learning_rate"]),
            "learning_rate_schedule": LEARNING_RATE_SCHEDULE,
            "fresh_optimizer_state": True,
            "adam_betas": [float(v) for v in training["adam_betas"]],
            "weight_decay": float(training["weight_decay"]),
            "batch_size": int(training["batch_size"]),
            "microbatch_size": microbatch,
            "gradient_accumulation_steps": accumulation,
            "gradient_clipping": False,
            "maximum_steps": maximum_steps,
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
            selected_summary, decision["best_short_auc_10_90"], parent
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
                "parent_version": PARENT_VERSION,
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
    return report


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
    result = (
        preflight(args.contract)
        if args.command == "preflight"
        else run(args.contract, device_name=args.device)
    )
    print(json.dumps(json_safe(result), indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
