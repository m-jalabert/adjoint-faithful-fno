"""Tests for the ninety-day staged fine-tune of the production operator.

Two things are under test here, and the second matters as much as the first.

1. **The fine-tune declares what it does.** Exactly one parent, named and
   hashed; the parent's normalization reused rather than recomputed; Adam cold;
   nine autoregressive calls; and the architecture, the eight loss weights and
   the spectral cap byte-identical to the parent's.

2. **The parent is undisturbed.** This arm was built by *adding* a module, not
   by generalizing ``train.py``, so every source the parent's frozen contract
   pins must still hash to the recorded bytes. If that ever fails, the parent
   has stopped being reproducible and the lineage claim is void.

Runnable without a GPU and without the trajectory store: everything that would
need the cluster reads the contract with ``verify_sources=False``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oceanfno.barotropic_transport import barotropic_transport_relative_l2
from oceanfno.continuity import ContinuityContext, continuity_relative_l2
from oceanfno.dataset import (
    TRAIN_CODE,
    records_for_rollout_split,
    store_codes,
)
from oceanfno.model import (
    EXPECTED_PARAMETER_COUNT,
    ProductionArchitecture,
    build_model,
    parameter_count,
    state_unroll,
)
from oceanfno.objective import (
    LOSS_CONTRACT_SHA256,
    ProductionLossConfig,
    production_loss_config,
    production_loss_terms,
)
from oceanfno.perturbation_growth import GROWTH_RATE_CEILING
from oceanfno.pressure_gradient import PressureGradientContext, pressure_gradient_relative_l2
from oceanfno.runtime import AUDIT_TERMS, _file_sha256, torch
from oceanfno.spectral_norm import apply_mode_spectral_norm, materialized_state_dict
from oceanfno.validation import PRIMARY_FIELDS
import oceanfno.finetune as finetune
import oceanfno.train as train

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_production_1in_1out_spectralnorm_ft90_v1.json"
PARENT_CONTRACT = ROOT / "config/model_c_production_1in_1out_spectralnorm_v1.json"
SBATCH = ROOT / "slurm/models/c/train_production_1in_1out_spectralnorm_ft90_v1.sbatch"

requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch is optional")


# ---------------------------------------------------------------------------
# the parent must survive untouched
# ---------------------------------------------------------------------------


def test_every_source_the_parent_contract_pins_is_unchanged() -> None:
    """The fine-tune adds a module; it must not have edited one the parent pins.

    This is the load-bearing test of the whole arm. ``finetune.py`` imports
    ``train.py``, ``objective.py`` and the rest rather than generalizing them,
    precisely so the parent stays reproducible from its own frozen contract.
    """

    pinned = json.loads(PARENT_CONTRACT.read_text())["source_hashes"]
    assert pinned, "the parent contract pins no sources"
    changed = {
        relative: (expected, _file_sha256(ROOT / relative))
        for relative, expected in pinned.items()
        if _file_sha256(ROOT / relative) != expected
    }
    assert not changed, f"the fine-tune modified sources the parent pins: {changed}"


def test_the_parent_objective_is_untouched_by_the_child() -> None:
    """Importing the fine-tune must not mutate the frozen production config."""

    assert production_loss_config().rollout_steps == 6
    assert LOSS_CONTRACT_SHA256 != finetune.FINETUNE_LOSS_CONTRACT_SHA256
    with pytest.raises(ValueError):
        ProductionLossConfig(rollout_steps=9)


def test_the_parent_training_module_still_declares_six_steps() -> None:
    assert train.ROLLOUT_STEPS == 6
    assert train.MAXIMUM_STEPS == 7680
    assert train.LEARNING_RATE == 5e-4
    assert train.MICROBATCH_SIZE == 4


# ---------------------------------------------------------------------------
# exactly four changes
# ---------------------------------------------------------------------------


def test_only_the_four_declared_fields_differ_from_the_parent() -> None:
    child = json.loads(CONTRACT.read_text())
    parent = json.loads(PARENT_CONTRACT.read_text())
    # The architecture is the single strongest claim: no rewrite of any kind.
    assert child["architecture"] == parent["architecture"]
    assert child["architecture"] == ProductionArchitecture().to_dict()
    assert child["expected_parameter_count"] == parent["expected_parameter_count"]
    assert child["expected_parameter_count"] == EXPECTED_PARAMETER_COUNT == 27_297_960
    # And the four that do move.
    assert parent["training"]["rollout_steps"] == 6
    assert child["training"]["rollout_steps"] == 9
    assert parent["training"]["initial_learning_rate"] == 5e-4
    assert child["training"]["learning_rate"] == 5e-5
    assert parent["training"]["maximum_steps"] == 7680
    assert child["training"]["maximum_steps"] == 1920
    assert parent["initialization"]["from_scratch"] is True
    assert child["initialization"]["from_scratch"] is False


def test_the_batch_is_the_parents_and_the_microbatch_is_smaller() -> None:
    training = json.loads(CONTRACT.read_text())["training"]
    assert training["batch_size"] == 8 == finetune.BATCH_SIZE
    assert training["microbatch_size"] == 2
    assert training["gradient_accumulation_steps"] == 4
    assert (
        finetune.MICROBATCH_SIZE * finetune.GRADIENT_ACCUMULATION_STEPS
        == finetune.BATCH_SIZE
    )
    # Equal-size microbatches are what make accumulation exactly a batch of 8.
    assert finetune.TRAINING_STARTS_PER_REGIME % finetune.MICROBATCH_SIZE == 0
    # Retained activations scale as microbatch x calls, so this is cheaper than
    # the parent's 4 x 6 despite the deeper unroll.
    assert finetune.MICROBATCH_SIZE * finetune.ROLLOUT_STEPS < 4 * 6


def test_the_learning_rate_is_constant_with_no_decay_stage() -> None:
    training = json.loads(CONTRACT.read_text())["training"]
    assert training["learning_rate"] == finetune.LEARNING_RATE == 5e-5
    assert training["learning_rate_schedule"] == "constant"
    assert training["fresh_optimizer_state"] is True
    for forbidden in ("decay_fraction", "decay_factor", "initial_learning_rate"):
        assert forbidden not in training


def test_the_budget_is_declared_and_arithmetically_consistent() -> None:
    training = json.loads(CONTRACT.read_text())["training"]
    assert finetune.MAXIMUM_STEPS == 1920
    assert finetune.CHECKPOINT_STEPS == (480, 960, 1440, 1920)
    assert training["checkpoint_steps"] == [480, 960, 1440, 1920]
    assert (
        finetune.STATE_TRANSITIONS
        == finetune.MAXIMUM_STEPS * finetune.BATCH_SIZE * finetune.ROLLOUT_STEPS
        == 138_240
    )
    assert training["state_transitions"] == 138_240
    # A real second stage, not a rerun: well under the parent's exposure.
    assert finetune.STATE_TRANSITIONS < finetune.PARENT_STATE_TRANSITIONS
    assert finetune.STATE_TRANSITIONS / finetune.PARENT_STATE_TRANSITIONS == pytest.approx(
        0.375
    )


# ---------------------------------------------------------------------------
# the lineage
# ---------------------------------------------------------------------------


def test_the_contract_names_and_pins_exactly_one_parent() -> None:
    contract = json.loads(CONTRACT.read_text())
    initialization = contract["initialization"]
    sources = contract["sources"]
    assert initialization["from_scratch"] is False
    assert initialization["load_model_state"] is True
    assert initialization["strict_state_dict_load"] is True
    assert initialization["load_optimizer_state"] is False
    assert initialization["normalization_reused"] is True
    assert initialization["architecture_changed"] is False
    assert initialization["parent_version"] == finetune.PARENT_VERSION
    assert initialization["parent_optimizer_step"] == 7680
    # The named checkpoint and the pinned source must be the same file.
    assert initialization["parent_checkpoint"] == sources["parent_checkpoint"]["path"]
    assert initialization["parent_checkpoint"].endswith(
        f"{finetune.PARENT_VERSION}/selected.pt"
    )
    for key in ("parent_checkpoint", "parent_normalization", "parent_report"):
        assert key in sources
        assert len(sources[key]["sha256"]) == 64


def test_a_contract_without_a_parent_is_rejected(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text())
    del contract["sources"]["parent_checkpoint"]
    target = tmp_path / "no_parent.json"
    target.write_text(json.dumps(contract))
    with pytest.raises(train.TrainingContractError):
        finetune.load_contract(target, verify_sources=False)


def test_a_contract_that_inherits_optimizer_state_is_rejected(tmp_path: Path) -> None:
    """Inheriting 5e-4-scale moments would undo the point of lowering the rate."""

    contract = json.loads(CONTRACT.read_text())
    contract["initialization"]["load_optimizer_state"] = True
    target = tmp_path / "warm_adam.json"
    target.write_text(json.dumps(contract))
    with pytest.raises(train.TrainingContractError):
        finetune.load_contract(target, verify_sources=False)


def test_a_contract_that_recomputes_normalization_is_rejected(tmp_path: Path) -> None:
    """The child continues the parent's map, so it must keep its coordinates."""

    contract = json.loads(CONTRACT.read_text())
    contract["normalization"]["reused_from_a_previous_run"] = False
    contract["normalization"]["recomputed_from_training_days_only"] = True
    target = tmp_path / "recomputed.json"
    target.write_text(json.dumps(contract))
    with pytest.raises(train.TrainingContractError):
        finetune.load_contract(target, verify_sources=False)


def test_a_contract_whose_checkpoint_and_source_disagree_is_rejected(
    tmp_path: Path,
) -> None:
    contract = json.loads(CONTRACT.read_text())
    contract["initialization"]["parent_checkpoint"] = "/elsewhere/selected.pt"
    target = tmp_path / "mismatched.json"
    target.write_text(json.dumps(contract))
    with pytest.raises(train.TrainingContractError):
        finetune.load_contract(target, verify_sources=False)


def test_the_declared_parent_short_auc_matches_the_published_report() -> None:
    """The gate's reference must be the number the parent actually published."""

    contract = json.loads(CONTRACT.read_text())
    report_path = Path(contract["sources"]["parent_report"]["path"])
    if not report_path.is_file():  # the cluster's outputs are not always present
        pytest.skip("the parent report is not on this filesystem")
    report = json.loads(report_path.read_text())
    summary = next(
        s for s in report["validation_summaries"] if int(s["optimizer_step"]) == 7680
    )
    for field in PRIMARY_FIELDS:
        assert summary["short_auc_10_90"][field] == pytest.approx(
            finetune.PARENT_SHORT_AUC_10_90[field], rel=1e-12
        )


def test_contract_loads_without_cluster_source_verification() -> None:
    contract, resolved, digest = finetune.load_contract(CONTRACT, verify_sources=False)
    assert contract["version"] == finetune.VERSION
    assert contract["version"].endswith("_ft90_v1")
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64


# ---------------------------------------------------------------------------
# the ninety-day objective
# ---------------------------------------------------------------------------


def test_the_objective_is_the_parents_weights_over_nine_calls() -> None:
    config = finetune.finetune_loss_config()
    parent = production_loss_config()
    assert config.rollout_steps == 9
    assert parent.rollout_steps == 6
    for field in (
        "increment_weight",
        "rollout_weight",
        "spectral_weight",
        "boundary_weight",
        "pressure_gradient_weight",
        "continuity_weight",
        "barotropic_transport_weight",
        "spectral_bins",
        "western_boundary_width",
    ):
        assert getattr(config, field) == getattr(parent, field), field


def test_the_finetuning_objective_configuration_is_frozen() -> None:
    with pytest.raises(ValueError):
        finetune.FineTuneLossConfig(rollout_steps=12)
    with pytest.raises(ValueError):
        finetune.FineTuneLossConfig(rollout_weight=0.15)
    with pytest.raises(ValueError):
        finetune.FineTuneLossConfig(continuity_weight=0.0)


def test_the_loss_contract_declares_nine_calls_and_pins_its_parent() -> None:
    contract = finetune.finetune_loss_contract(finetune.finetune_loss_config())
    assert contract["config"]["rollout_steps"] == 9
    assert contract["parent_loss_contract_sha256"] == LOSS_CONTRACT_SHA256
    assert contract["weights_identical_to_parent"] is True
    # The prose must not still describe the parent's six-call horizon.
    assert "20_30_40_50_60_70_80_and_90_days" in contract["rollout"]
    assert "nine_calls" in contract["spectral"]
    assert "9_rollout_calls" in contract["pressure_gradient"]
    for value in contract.values():
        if isinstance(value, str):
            assert "six" not in value and "_6_" not in value


def test_the_declared_loss_hash_matches_the_contract_file() -> None:
    loss = json.loads(CONTRACT.read_text())["loss"]
    assert loss["contract_sha256"] == finetune.FINETUNE_LOSS_CONTRACT_SHA256
    assert loss["parent_contract_sha256"] == LOSS_CONTRACT_SHA256
    assert loss["staged_fine_tuning"] is True
    assert loss["weights_identical_to_parent"] is True
    assert loss["all_terms_active_from_step_1"] is True
    assert loss["contraction_penalty"] is False
    assert loss["rollout_steps"] == 9


@requires_torch
def test_the_rollout_term_averages_the_eight_leads_from_20_to_90_days() -> None:
    """L_rollout = (1/8) sum_(k=2)^(9) E_(10k): the eight non-first calls."""

    config = finetune.finetune_loss_config()
    wet = torch.ones((1, 1, 8, 8))
    boundary = torch.ones((1, 1, 8, 8))
    present = torch.zeros((1, 46, 8, 8))
    targets = torch.zeros((1, 9, 46, 8, 8))
    targets[:] = 1.0
    predictions = targets.clone()
    # Exactly one of the eight rollout leads is wrong, by a known amount.
    predictions[:, 4] = 0.0
    auxiliary = {
        name: torch.zeros(())
        for name in ("pressure_gradient", "continuity", "barotropic_transport")
    }
    terms = production_loss_terms(
        predictions, targets, present, wet, boundary, torch.ones(46), config, auxiliary
    )
    # Call 0 (day 10) is L_state and is exact here, so L_state must be zero and
    # the single wrong lead must carry 1/8 of the rollout term.
    assert float(terms["state"]) == pytest.approx(0.0, abs=1e-6)
    assert float(terms["rollout"]) == pytest.approx(1.0 / 8.0, rel=1e-5)


@requires_torch
def test_every_physics_term_runs_through_all_nine_calls() -> None:
    """A defect on the ninth call must be visible, not truncated at the sixth."""

    n = 8
    mean = np.zeros((46, n, n), dtype=np.float32)
    scale = np.ones_like(mean)
    dx = np.full((n, n), 1.0e5, dtype=np.float32)
    wet = np.ones((n, n), dtype=bool)
    pressure = PressureGradientContext(mean, scale, dx, wet)
    continuity = ContinuityContext(mean, scale, dx, wet)
    present = torch.zeros((1, 46, n, n))
    truth = torch.zeros((1, 9, 46, n, n))
    ramp = torch.linspace(0.0, 1.0, n)[None, :].expand(n, n)
    for step in range(9):
        truth[:, step, 45] = (step + 1) * 0.5
        truth[:, step, 30] = ramp * (step + 1)
        truth[:, step, 0:15] = (step + 1) * 0.01
    prediction = truth.clone()
    # Corrupt only the ninth call, which a six-call loss could not see.
    prediction[:, 8] = 0.0
    assert prediction.shape[1] == 9
    for value in (
        pressure_gradient_relative_l2(prediction, truth, pressure),
        continuity_relative_l2(prediction, truth, present, continuity),
        barotropic_transport_relative_l2(prediction, truth, present, continuity),
    ):
        assert bool(torch.isfinite(value).item())
        assert float(value) > 0.0


@requires_torch
def test_the_nine_call_total_is_the_declared_weighted_sum() -> None:
    config = finetune.finetune_loss_config()
    generator = torch.Generator().manual_seed(0)
    predictions = torch.randn((2, 9, 46, 8, 8), generator=generator)
    targets = torch.randn((2, 9, 46, 8, 8), generator=generator)
    present = torch.randn((2, 46, 8, 8), generator=generator)
    wet = torch.ones((1, 1, 8, 8))
    boundary = torch.zeros((1, 1, 8, 8))
    boundary[..., :4] = 1.0
    auxiliary = {
        name: torch.zeros(())
        for name in ("pressure_gradient", "continuity", "barotropic_transport")
    }
    terms = production_loss_terms(
        predictions, targets, present, wet, boundary, torch.ones(46), config, auxiliary
    )
    assert set(AUDIT_TERMS).issubset(terms)
    expected = (
        terms["state"]
        + config.increment_weight * terms["increment"]
        + config.rollout_weight * terms["rollout"]
        + config.spectral_weight * terms["spectral"]
        + config.boundary_weight * terms["boundary"]
    )
    assert float(terms["total"]) == pytest.approx(float(expected), rel=1.0e-6)


# ---------------------------------------------------------------------------
# the ninety-day rollout and the split
# ---------------------------------------------------------------------------


def test_nine_calls_cost_thirty_starts_per_regime() -> None:
    _, pair_codes = store_codes()
    records = records_for_rollout_split(pair_codes, TRAIN_CODE, rollout_steps=9)
    assert len(records) == finetune.TRAINING_RECORDS == 17_730
    assert len(records) // 3 == finetune.TRAINING_STARTS_PER_REGIME == 5910
    assert min(time for _, time in records) == 0
    # The last start's ninetieth day is 5,999: the final training day, and not
    # one step into validation.
    assert max(time for _, time in records) == 5909
    assert 5909 + 9 * 10 == 5999
    parent_records = records_for_rollout_split(pair_codes, TRAIN_CODE, rollout_steps=6)
    assert len(parent_records) // 3 - len(records) // 3 == 30


def test_the_split_summary_reports_the_longer_horizon() -> None:
    summary = finetune.split_summary()
    assert summary["training_rollout_steps"] == 9
    assert summary["latest_training_rollout_start"] == 5909
    assert summary["starts_lost_to_the_longer_rollout"] == 30
    assert summary["train"] == [0, 5999]
    assert summary["validation"] == [6000, 7199]
    assert summary["inference_nested_in_validation"] is True


@requires_torch
def test_the_nine_step_rollout_is_self_generated_after_the_first_call() -> None:
    seen: list[Any] = []

    class Recorder:
        architecture = ProductionArchitecture()

        def __call__(self, features: Any) -> Any:
            seen.append(features[:, :46].clone())
            return features[:, :46] + 1.0

    features = torch.zeros((1, 51, 8, 8))
    wet = torch.ones((1, 1, 8, 8))
    predictions = state_unroll(Recorder(), features, wet, finetune.ROLLOUT_STEPS)
    assert predictions.shape == (1, 9, 46, 8, 8)
    assert float(seen[0].abs().max()) == 0.0
    for step in range(1, 9):
        assert torch.equal(seen[step], predictions[:, step - 1])


# ---------------------------------------------------------------------------
# the spectral cap is inherited, not changed
# ---------------------------------------------------------------------------


def test_the_spectral_cap_is_declared_unchanged_at_rho_one() -> None:
    child = json.loads(CONTRACT.read_text())["spectral_normalization"]
    parent = json.loads(PARENT_CONTRACT.read_text())["spectral_normalization"]
    assert child["applied"] is True
    assert child["rho"] == 1.0
    assert child["changed_from_parent"] is False
    assert child["form"] == parent["form"]
    assert child["applies_to"] == parent["applies_to"]
    assert child["matrices_total"] == parent["matrices_total"] == 1632
    assert child["power_iterations_per_forward"] == parent["power_iterations_per_forward"]
    assert child["warmup_iterations"] == parent["warmup_iterations"]
    assert child["adds_parameters"] == 0
    assert child["checkpoints_materialized"] is True


def test_a_contract_that_tightens_the_cap_is_rejected(tmp_path: Path) -> None:
    """rho = 0.99 is the declared *next* experiment, not part of this one."""

    contract = json.loads(CONTRACT.read_text())
    contract["spectral_normalization"]["rho"] = 0.99
    target = tmp_path / "tightened.json"
    target.write_text(json.dumps(contract))
    with pytest.raises(train.TrainingContractError):
        finetune.load_contract(target, verify_sources=False)


@requires_torch
def test_reinstalling_the_cap_on_capped_weights_preserves_the_function() -> None:
    """The property preflight asserts against the real parent, in miniature.

    A materialized checkpoint's spectral weights already satisfy the cap, so
    installing it again must be close to the identity -- otherwise the fine-tune
    would start from a re-scaled copy of the published operator rather than from
    the operator itself.
    """

    architecture = ProductionArchitecture()
    trained = build_model(architecture)
    apply_mode_spectral_norm(trained, warmup_iterations=200)
    published = build_model(architecture)
    published.load_state_dict(materialized_state_dict(trained), strict=True)
    published.eval()
    probe = torch.randn((1, 51, 62, 62), generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        before = published(probe).clone()
    apply_mode_spectral_norm(published, warmup_iterations=200)
    published.eval()
    with torch.no_grad():
        after = published(probe)
    reference = float(before.square().mean().sqrt())
    deviation = float((after - before).abs().max()) / reference
    assert deviation < 1.0e-2
    assert parameter_count(published) == EXPECTED_PARAMETER_COUNT


# ---------------------------------------------------------------------------
# the flattening diagnostic and the acceptance gate
# ---------------------------------------------------------------------------


def _curve(values: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "lead_days": list(range(10, 361, 10)),
        "mean_rmse": {"model": values},
    }


def _linear(slope: float) -> list[float]:
    return [slope * lead for lead in range(10, 361, 10)]


def _accelerating() -> list[float]:
    return [1.0e-5 * lead**2 for lead in range(10, 361, 10)]


def _saturating() -> list[float]:
    return [1.0 - np.exp(-lead / 120.0) for lead in range(10, 361, 10)]


def test_a_straight_rmse_curve_has_flattening_exactly_one() -> None:
    summary = _curve({field: _linear(1.0e-4) for field in PRIMARY_FIELDS})
    for value in finetune.rmse_curve_flattening(summary).values():
        assert value == pytest.approx(1.0, rel=1.0e-9)


def test_an_accelerating_curve_is_above_one_and_a_saturating_one_below() -> None:
    fast = _curve({field: _accelerating() for field in PRIMARY_FIELDS})
    slow = _curve({field: _saturating() for field in PRIMARY_FIELDS})
    for value in finetune.rmse_curve_flattening(fast).values():
        assert value > 1.0
    for value in finetune.rmse_curve_flattening(slow).values():
        assert value < 1.0


def test_a_flat_early_window_reports_infinity_rather_than_passing() -> None:
    """An undefined ratio must fail the condition, not slip through as zero."""

    summary = _curve({field: [0.01] * 36 for field in PRIMARY_FIELDS})
    for value in finetune.rmse_curve_flattening(summary).values():
        assert value == float("inf")


def test_the_parents_measured_flattening_is_reproduced_from_its_report() -> None:
    """The parent's curve steepens in two of three fields; that is the target."""

    contract = json.loads(CONTRACT.read_text())
    report_path = Path(contract["sources"]["parent_report"]["path"])
    if not report_path.is_file():
        pytest.skip("the parent report is not on this filesystem")
    report = json.loads(report_path.read_text())
    summary = next(
        s for s in report["validation_summaries"] if int(s["optimizer_step"]) == 7680
    )
    flattening = finetune.rmse_curve_flattening(summary)
    assert flattening["sst"] > 1.0
    assert flattening["surface_speed"] > 1.0
    assert max(flattening.values()) > finetune.FLATTENING_CEILING


def _summary(
    step: int,
    short: float,
    long_ratio: float,
    growth: float,
    curve: list[float] | None = None,
) -> dict[str, Any]:
    values = curve if curve is not None else _saturating()
    return {
        "optimizer_step": step,
        "short_auc_10_90": {field: short for field in PRIMARY_FIELDS},
        "long_ratio_to_climatology": {field: long_ratio for field in PRIMARY_FIELDS},
        "perturbation_growth": {"worst_growth_rate_per_call": growth},
        "lead_days": list(range(10, 361, 10)),
        "mean_rmse": {"model": {field: values for field in PRIMARY_FIELDS}},
    }


def _parent(short: float = 1.0) -> dict[str, Any]:
    return {
        "short_auc_10_90": {field: short for field in PRIMARY_FIELDS},
        "growth_rate_per_call": finetune.PARENT_GROWTH_RATE,
        "rmse_curve_flattening_90_360": {field: 1.5 for field in PRIMARY_FIELDS},
    }


def test_the_gate_passes_only_when_all_five_conditions_hold() -> None:
    selected = _summary(1920, short=1.0, long_ratio=0.3, growth=0.99)
    gate = finetune.acceptance_gate(
        selected, {field: 1.0 for field in PRIMARY_FIELDS}, _parent()
    )
    assert gate["short_auc_within_5_percent_of_best_in_every_field"] is True
    assert gate["short_auc_within_5_percent_of_the_parent_in_every_field"] is True
    assert gate["worst_long_ratio_at_or_below_ceiling"] is True
    assert gate["rmse_curve_flattening_at_or_below_one_in_every_field"] is True
    assert gate["growth_rate_at_or_below_ceiling"] is True
    assert gate["validation_conditions_pass"] is True


def test_the_gate_fails_a_fine_tune_that_traded_away_short_horizon_skill() -> None:
    """The condition this arm adds: 10 % worse than the parent is a rejection.

    Everything else about this checkpoint is excellent -- it is the best in its
    own run, well below climatology, stable and flattening -- which is exactly
    the trade the gate exists to refuse.
    """

    selected = _summary(1920, short=1.10, long_ratio=0.2, growth=0.98)
    gate = finetune.acceptance_gate(
        selected, {field: 1.10 for field in PRIMARY_FIELDS}, _parent(short=1.0)
    )
    assert gate["short_auc_within_5_percent_of_best_in_every_field"] is True
    assert gate["short_auc_within_5_percent_of_the_parent_in_every_field"] is False
    assert gate["validation_conditions_pass"] is False
    for field in PRIMARY_FIELDS:
        assert gate["short_auc_10_90_ratio_to_parent"][field] == pytest.approx(1.10)


def test_the_gate_fails_a_curve_that_is_still_steepening() -> None:
    selected = _summary(1920, short=1.0, long_ratio=0.3, growth=0.99, curve=_accelerating())
    gate = finetune.acceptance_gate(
        selected, {field: 1.0 for field in PRIMARY_FIELDS}, _parent()
    )
    assert gate["rmse_curve_flattening_at_or_below_one_in_every_field"] is False
    assert gate["validation_conditions_pass"] is False


def test_the_gate_fails_an_unstable_checkpoint_and_reports_the_parent() -> None:
    selected = _summary(1920, short=1.0, long_ratio=0.3, growth=1.0132)
    gate = finetune.acceptance_gate(
        selected, {field: 1.0 for field in PRIMARY_FIELDS}, _parent()
    )
    assert gate["growth_rate_at_or_below_ceiling"] is False
    assert gate["validation_conditions_pass"] is False
    assert gate["growth_rate_ceiling"] == GROWTH_RATE_CEILING == 1.0
    assert gate["parent_growth_rate_per_call"] == finetune.PARENT_GROWTH_RATE


def test_the_gate_defers_the_day_2000_conditions_rather_than_passing_them() -> None:
    selected = _summary(1920, short=1.0, long_ratio=0.3, growth=0.99)
    gate = finetune.acceptance_gate(
        selected, {field: 1.0 for field in PRIMARY_FIELDS}, _parent()
    )
    deferred = gate["deferred_to_the_figure_package"]
    assert "2000_day_all_values_finite" in deferred
    assert "day_2000_rmse_ratio_to_climatology_plateaus_rather_than_rising" in deferred
    assert "day_2000_streamfunction_anomaly_rms_ratio_near_one" in deferred


def test_a_failed_growth_measurement_is_treated_as_unstable() -> None:
    selected = _summary(1920, short=1.0, long_ratio=0.3, growth=0.99)
    selected["perturbation_growth"]["worst_growth_rate_per_call"] = None
    gate = finetune.acceptance_gate(
        selected, {field: 1.0 for field in PRIMARY_FIELDS}, _parent()
    )
    assert gate["worst_perturbation_growth_rate_per_call"] == float("inf")
    assert gate["growth_rate_at_or_below_ceiling"] is False


# ---------------------------------------------------------------------------
# selection, outputs and the job
# ---------------------------------------------------------------------------


def test_the_selection_rule_is_the_parents_plus_two_new_conditions() -> None:
    child = json.loads(CONTRACT.read_text())["checkpoint_selection"]
    parent = json.loads(PARENT_CONTRACT.read_text())["checkpoint_selection"]
    for key in (
        "rollout_days",
        "short_auc_window_days",
        "short_auc_tolerance",
        "long_auc_window_days",
        "worst_long_ratio_ceiling",
        "primary_fields",
        "growth_rate_ceiling",
        "growth_rate_calls",
    ):
        assert child[key] == parent[key], key
    assert child["parent_short_auc_tolerance"] == 1.05
    assert child["flattening_ceiling"] == 1.0
    assert child["parent_short_auc_10_90"] == finetune.PARENT_SHORT_AUC_10_90


def test_outputs_are_written_under_the_c_roots_beside_the_parent() -> None:
    output = json.loads(CONTRACT.read_text())["output"]
    assert output["project_root"].endswith("outputs/af_fno/C/" + finetune.VERSION)
    assert output["scratch_root"].endswith("af_fno/models/C/" + finetune.VERSION)
    assert tuple(output["artifacts"]) == finetune.OUTPUT_ARTIFACTS
    # A distinct root: the parent stays published where it is.
    parent_output = json.loads(PARENT_CONTRACT.read_text())["output"]
    assert output["project_root"] != parent_output["project_root"]
    assert output["scratch_root"] != parent_output["scratch_root"]


def test_the_contract_records_why_the_tempting_changes_were_not_made() -> None:
    rejected = json.loads(CONTRACT.read_text())["rejected_alternatives"]
    for name in (
        "reduce_modes_32_to_24",
        "reduce_15_levels_to_3",
        "change_the_ten_day_interval",
        "tanh_stabilizer",
        "tighten_the_spectral_cap_to_0p99_in_the_same_run",
        "train_against_day_500_or_day_2000_truth",
    ):
        assert name in rejected
        assert len(rejected[name]) > 40


# ---------------------------------------------------------------------------
# a completed run must never be lost to a non-finite diagnostic
# ---------------------------------------------------------------------------


def test_the_readme_renders_a_report_whose_diagnostics_failed() -> None:
    """A failed growth fit is a supported outcome, not a crash at the finish.

    ``_readme`` runs on the ``json_safe``-sanitized report, where every
    non-finite float has already become ``None``. Formatting one directly would
    raise ``TypeError`` after the whole fine-tune had completed and every
    checkpoint had been written -- the way a v3 job lost 2 h 46 m.
    """

    from oceanfno.runtime import json_safe

    summaries = []
    for step in finetune.CHECKPOINT_STEPS:
        summary = _summary(step, short=1.0, long_ratio=0.3, growth=0.99)
        summaries.append(summary)
    # Every way a diagnostic can legitimately go missing, all at once.
    summaries[0]["perturbation_growth"]["worst_growth_rate_per_call"] = None
    summaries[1]["perturbation_growth"]["worst_growth_rate_per_call"] = float("inf")
    summaries[2]["mean_rmse"]["model"] = {f: [0.01] * 36 for f in PRIMARY_FIELDS}
    selected = summaries[-1]
    gate = finetune.acceptance_gate(
        selected, {f: 1.0 for f in PRIMARY_FIELDS}, _parent()
    )
    gate["rmse_curve_flattening_90_360"] = {f: float("inf") for f in PRIMARY_FIELDS}
    report = json_safe(
        {
            "selection_decision": {
                "selected_optimizer_step": 1920,
                "branch": "primary_rule",
            },
            "acceptance_gate": gate,
            "parent": _parent(),
            "validation_summaries": summaries,
            "counts": {
                "training_rollout_records": finetune.TRAINING_RECORDS,
                "training_starts_per_regime": finetune.TRAINING_STARTS_PER_REGIME,
            },
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "content_sha256": "0" * 64,
        }
    )
    text = finetune._readme(report)
    assert "n/a" in text
    assert "Ninety-day fine-tune" in text
    # The table still has one row per checkpoint plus the parent reference.
    assert text.count("| 480 ") + text.count("| 960 ") == 2


def test_the_number_formatter_never_raises_on_a_missing_diagnostic() -> None:
    assert finetune._number(None) == "n/a"
    assert finetune._number(float("inf")) == "inf"
    assert finetune._number(float("-inf")) == "-inf"
    assert finetune._number(float("nan")) == "n/a"
    assert finetune._number(1.23456, 3) == "1.235"
    # A worst-case summary over an incomplete set must not report the best of
    # the survivors as if it were the worst.
    assert finetune._worst({"a": 1.0, "b": None}) == "n/a"
    assert finetune._worst({"a": 1.0, "b": 2.0}) == "2.00"


def test_slurm_uses_the_finetune_entrypoint_and_this_contract() -> None:
    text = SBATCH.read_text()
    assert "-m oceanfno.finetune preflight" in text
    assert "-m oceanfno.finetune run" in text
    assert "model_c_production_1in_1out_spectralnorm_ft90_v1.json" in text
    # A fine-tune must not be pointed at the parent's training entry point.
    assert "-m oceanfno.train run" not in text
