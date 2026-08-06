"""Tests for the six-step rollout fine-tuning arm.

Three things here are worth more than the rest.

The **objective algebra** test evaluates the boxed loss by hand and compares it
with what the shared ``model_c_loss_terms`` returns under the six-step
configuration.  The arm's entire claim is that the deepened rollout needed no
loss code, only a configuration; that is only true if the 1/5 and 1/6 means come
out where the declaration says they do.

The **certified configuration** test asserts that the three-step
``ModelCLossConfig`` still refuses six steps and still carries 0.15.  The new
configuration is an exception to that validator, and an exception that quietly
became the rule would silently redefine every earlier arm's objective.

The **README** test renders against a synthetic report.  The duration arm lost a
completed 1h34m job to a ``KeyError`` in its README, which is the last thing
written before promotion, so this one is exercised before the job is ever
submitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oceanfno import validation as protocol
from oceanfno.objective import (
    MODEL_C_LOSS_V1_CONTRACT_SHA256,
    ModelCLossConfig,
    model_c_loss_config,
)
from oceanfno.train import (
    BASELINE_OPTIMIZER_STEP,
    BATCH_SIZE,
    BOUNDARY_WEIGHT,
    CHECKPOINT_STEPS,
    CONTRACT_STATUS,
    DECLARED_CHANGES,
    FINE_TUNE_LOSS_CONTRACT_SHA256,
    HELD_TRAINING_FIELDS,
    INCREMENT_WEIGHT,
    LEARNING_RATE,
    MAXIMUM_STEPS,
    PARENT_VERSION,
    ROLLOUT_STEPS,
    ROLLOUT_WEIGHT,
    SPECTRAL_WEIGHT,
    TRAINING_RECORDS,
    TRAINING_STARTS_PER_REGIME,
    VERSION,
    WORST_LONG_RATIO_CEILING,
    BireProtocolRolloutFineTuneError,
    BireProtocolRolloutFineTuneLossConfig,
    _readme,
    acceptance_gate,
    baseline_validation_summary,
    fine_tune_loss_config,
    fine_tune_loss_contract,
    fine_tune_loss_contract_sha256,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_bire_protocol_rollout_ft_v2.json"
PARENT = ROOT / "archive/config/model_c_bire_protocol_duration_v1.json"
SBATCH = ROOT / "slurm/models/c/train.sbatch"

torch = pytest.importorskip("torch", reason="the objective algebra needs PyTorch")


def _tampered(mutate, directory: Path) -> Path:
    """A contract copy with one field moved.

    Written under pytest's ``tmp_path`` rather than beside the real contracts:
    ``load_contract`` is called with ``verify_sources=False`` here, so nothing
    needs the repository layout, and earlier arms' tests left hundreds of
    stray directories in ``config/``.
    """

    contract = json.loads(CONTRACT.read_text())
    mutate(contract)
    path = directory / "tampered.json"
    path.write_text(json.dumps(contract))
    return path


# --------------------------------------------------------------------------
# The objective
# --------------------------------------------------------------------------


def test_the_six_step_configuration_moves_only_depth_and_rollout_weight() -> None:
    config = fine_tune_loss_config()
    v1 = model_c_loss_config("v1")
    assert config.rollout_steps == ROLLOUT_STEPS == 6
    assert config.rollout_weight == ROLLOUT_WEIGHT == 0.50
    moved = {
        key
        for key, value in config.to_dict().items()
        if v1.to_dict()[key] != value
    }
    assert moved == {"rollout_steps", "rollout_weight"}
    assert config.increment_weight == INCREMENT_WEIGHT == 0.001
    assert config.spectral_weight == SPECTRAL_WEIGHT == 1.0e-5
    assert config.boundary_weight == BOUNDARY_WEIGHT == 0.065
    assert config.spectral_bins == 12 and config.western_boundary_width == 4


def test_the_certified_three_step_configuration_is_untouched() -> None:
    """The exception must not widen into a second free parameter."""

    v1 = model_c_loss_config("v1")
    assert v1.rollout_steps == 3 and v1.rollout_weight == 0.15
    with pytest.raises(ValueError):
        ModelCLossConfig(rollout_steps=6)
    with pytest.raises(ValueError):
        BireProtocolRolloutFineTuneLossConfig(rollout_weight=0.15)
    with pytest.raises(ValueError):
        BireProtocolRolloutFineTuneLossConfig(rollout_steps=3)


def test_the_six_step_objective_has_its_own_hash() -> None:
    assert FINE_TUNE_LOSS_CONTRACT_SHA256 != MODEL_C_LOSS_V1_CONTRACT_SHA256
    assert FINE_TUNE_LOSS_CONTRACT_SHA256 == fine_tune_loss_contract_sha256(
        fine_tune_loss_config()
    )
    contract = fine_tune_loss_contract(fine_tune_loss_config())
    assert contract["config"]["rollout_steps"] == 6
    assert contract["derived_from_loss_contract_sha256"] == MODEL_C_LOSS_V1_CONTRACT_SHA256
    assert set(contract["groups"].values()) == {0.25}
    with pytest.raises(BireProtocolRolloutFineTuneError):
        fine_tune_loss_contract(model_c_loss_config("v1"))


def test_the_total_is_the_declared_objective() -> None:
    """L = L_state + 0.001 L_inc + 0.50 mean_{k=2..6} + 1e-5 mean_{k=1..6} + 0.065 mean_{k=1..6}."""

    from oceanfno.objective import (
        group_increment_nrmse_terms,
        group_relative_l2_terms,
        model_c_loss_terms,
        tapered_group_spectral_loss,
    )

    torch.manual_seed(0)
    grid, channels, batch = 16, 46, 2
    config = fine_tune_loss_config()
    predictions = torch.randn(batch, ROLLOUT_STEPS, channels, grid, grid)
    targets = torch.randn(batch, ROLLOUT_STEPS, channels, grid, grid)
    present = torch.randn(batch, channels, grid, grid)
    wet = torch.ones(1, 1, grid, grid)
    boundary = torch.zeros(1, 1, grid, grid)
    boundary[..., :4] = 1.0
    increment_scale = torch.rand(channels) + 0.5

    terms = model_c_loss_terms(
        predictions, targets, present, wet, boundary, increment_scale, config
    )

    state = group_relative_l2_terms(predictions[:, 0], targets[:, 0], wet)["mean"]
    increment = group_increment_nrmse_terms(
        predictions[:, 0] - present, targets[:, 0] - present, wet, increment_scale
    )["mean"]
    rollout = torch.stack(
        [
            group_relative_l2_terms(predictions[:, k], targets[:, k], wet)["mean"]
            for k in range(1, ROLLOUT_STEPS)
        ]
    ).mean()
    spectral = torch.stack(
        [
            tapered_group_spectral_loss(
                predictions[:, k] - (present if k == 0 else predictions[:, k - 1]),
                targets[:, k] - (present if k == 0 else targets[:, k - 1]),
                wet,
                bins=config.spectral_bins,
            )
            for k in range(ROLLOUT_STEPS)
        ]
    ).mean()
    boundary_term = torch.stack(
        [
            group_relative_l2_terms(predictions[:, k], targets[:, k], boundary)["mean"]
            for k in range(ROLLOUT_STEPS)
        ]
    ).mean()

    # The rollout mean covers five steps, the spectral and boundary means six.
    assert len([k for k in range(1, ROLLOUT_STEPS)]) == 5
    for name, expected in (
        ("state", state),
        ("increment", increment),
        ("rollout", rollout),
        ("spectral", spectral),
        ("boundary", boundary_term),
    ):
        assert torch.allclose(terms[name], expected), name
    expected_total = (
        state
        + INCREMENT_WEIGHT * increment
        + ROLLOUT_WEIGHT * rollout
        + SPECTRAL_WEIGHT * spectral
        + BOUNDARY_WEIGHT * boundary_term
    )
    assert torch.allclose(terms["total"], expected_total)


def test_the_rollout_term_is_not_the_three_step_one() -> None:
    """A six-step tensor scored under the v1 config would be silently truncated."""

    from oceanfno.objective import model_c_loss_terms

    torch.manual_seed(1)
    grid, channels, batch = 16, 46, 2
    predictions = torch.randn(batch, ROLLOUT_STEPS, channels, grid, grid)
    targets = torch.randn(batch, ROLLOUT_STEPS, channels, grid, grid)
    present = torch.randn(batch, channels, grid, grid)
    wet = torch.ones(1, 1, grid, grid)
    boundary = torch.zeros(1, 1, grid, grid)
    boundary[..., :4] = 1.0
    increment_scale = torch.rand(channels) + 0.5
    with pytest.raises(ValueError):
        model_c_loss_terms(
            predictions, targets, present, wet, boundary, increment_scale,
            model_c_loss_config("v1"),
        )


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the fine-tune contract is absent")
def test_the_contract_moves_only_the_declared_quantities() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    parent = json.loads(PARENT.read_text())
    assert contract["version"] == VERSION != parent["version"]
    assert contract["contract_status"] == CONTRACT_STATUS
    assert contract["architecture"] == parent["architecture"]
    assert contract["dataset"] == parent["dataset"]
    assert contract["normalization"] == parent["normalization"]
    assert contract["checkpoint_selection"] == parent["checkpoint_selection"]
    for field in HELD_TRAINING_FIELDS:
        assert contract["training"][field] == parent["training"][field], field
    for field, (before, after) in DECLARED_CHANGES.items():
        assert parent["training"][field] == before, field
        assert contract["training"][field] == after, field
    moved = {
        field
        for field in set(contract["training"]) & set(parent["training"])
        if contract["training"][field] != parent["training"][field]
    }
    assert set(DECLARED_CHANGES) <= moved
    # Nothing outside the declared changes plus the bookkeeping fields moved.
    assert moved - set(DECLARED_CHANGES) <= {
        "records", "exposure_budget", "from_scratch", "checkpoint_steps",
    }


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the fine-tune contract is absent")
def test_the_optimizer_matches_the_declaration() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    training = contract["training"]
    assert training["optimizer"] == "adam"
    assert float(training["initial_learning_rate"]) == LEARNING_RATE == 2.0e-5
    assert tuple(training["adam_betas"]) == (0.9, 0.95)
    assert float(training["weight_decay"]) == 0.0
    assert training["gradient_clipping"] is False
    assert int(training["batch_size"]) == BATCH_SIZE == 4
    assert int(training["seed"]) == 20260724
    assert int(training["maximum_steps"]) == MAXIMUM_STEPS == 3840
    assert tuple(training["checkpoint_steps"]) == CHECKPOINT_STEPS == (960, 1920, 2880, 3840)
    decay = round(training["maximum_steps"] * training["decay_fraction"])
    assert decay == 2880
    schedule = training["learning_rate_schedule"]
    assert schedule["steps_1_to_2880"] == 2.0e-5
    assert schedule["steps_2881_to_3840"] == pytest.approx(4.0e-6)
    assert BATCH_SIZE * ROLLOUT_STEPS == 8 * 3


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the fine-tune contract is absent")
def test_the_initialization_is_the_step_15360_checkpoint_weights_only() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    initialization = contract["initialization"]
    assert initialization["version"] == PARENT_VERSION
    assert int(initialization["optimizer_step"]) == BASELINE_OPTIMIZER_STEP == 15360
    assert initialization["load_only"] == "model_state_dict"
    assert initialization["optimizer_state_loaded"] is False
    assert initialization["normalization_reused"] is True
    assert contract["training"]["from_scratch"] is False
    assert contract["training"]["load_optimizer_state"] is False
    assert initialization["checkpoint"].endswith("bire_protocol_duration_v1/selected.pt")
    assert (
        contract["sources"]["initialization_checkpoint"]["path"]
        == initialization["checkpoint"]
    )
    assert (
        contract["sources"]["parent_normalization"]["path"]
        .endswith("model_c_bire_protocol_duration_train_only_normalization.npz")
    )


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the fine-tune contract is absent")
def test_the_loss_block_declares_the_six_step_objective() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    parent = json.loads(PARENT.read_text())
    loss = contract["loss"]
    assert loss["contract_sha256"] == FINE_TUNE_LOSS_CONTRACT_SHA256
    assert loss["derived_from_contract_sha256"] == MODEL_C_LOSS_V1_CONTRACT_SHA256
    assert int(loss["rollout_steps"]) == 6 and int(parent["loss"]["rollout_steps"]) == 3
    moved = {
        key
        for key, value in loss["coefficients"].items()
        if parent["loss"]["coefficients"][key] != value
    }
    assert moved == {"rollout"}
    assert loss["coefficients"]["rollout"] == 0.50
    assert loss["state"] == parent["loss"]["state"]


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the fine-tune contract is absent")
def test_outputs_do_not_collide_with_anything_already_published() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    for key in ("scratch_root", "project_root"):
        root = contract["output"][key]
        assert root.endswith("bire_protocol_rollout_ft_v2")
        assert not Path(root).exists(), "a re-run would refuse to overwrite this"
    superseded = contract["supersedes"]
    assert superseded["version"] == "model_c_bire_protocol_rollout_ft_v1"
    assert len(superseded["contract_sha256"]) == 64



@pytest.mark.skipif(not CONTRACT.is_file(), reason="the fine-tune contract is absent")
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda c: c["training"].update(batch_size=8), id="batch_reverted"),
        pytest.param(lambda c: c["training"].update(rollout_steps=3), id="depth_reverted"),
        pytest.param(lambda c: c["training"].update(seed=1), id="seed_moved"),
        pytest.param(lambda c: c["training"].update(weight_decay=0.01), id="decay_added"),
        pytest.param(lambda c: c["training"].update(adam_betas=[0.9, 0.999]), id="betas_moved"),
        pytest.param(lambda c: c["loss"].update(contract_sha256="0" * 64), id="objective_moved"),
        pytest.param(
            lambda c: c["loss"]["coefficients"].update(boundary=0.1), id="second_coefficient_moved"
        ),
        pytest.param(
            lambda c: c["architecture"].update(hidden_channels=64), id="architecture_moved"
        ),
        pytest.param(
            lambda c: c["normalization"].update(recomputed_from="something_else"),
            id="normalizers_moved",
        ),
        pytest.param(
            lambda c: c["initialization"].update(optimizer_step=11520), id="initialization_moved"
        ),
        pytest.param(
            lambda c: c["training"].update(load_optimizer_state=True), id="optimizer_state_loaded"
        ),
    ],
)
def test_a_tampered_contract_is_rejected(mutate, tmp_path) -> None:
    with pytest.raises(BireProtocolRolloutFineTuneError):
        load_contract(_tampered(mutate, tmp_path), verify_sources=False)


# --------------------------------------------------------------------------
# The training set
# --------------------------------------------------------------------------


def test_six_step_starts_keep_the_whole_target_sequence_inside_training() -> None:
    from oceanfno.dataset import store_codes
    from oceanfno.dataset import records_for_rollout_split

    _, pair_codes = store_codes()
    records = records_for_rollout_split(pair_codes, 1, rollout_steps=ROLLOUT_STEPS)
    starts = sorted({time_index for _, time_index in records})
    assert len(records) == TRAINING_RECORDS == 17820
    assert len(starts) == TRAINING_STARTS_PER_REGIME == 5940
    assert starts[0] == 0 and starts[-1] == 5939
    # t + 60 is the last target and must still be a training day.
    assert starts[-1] + 10 * ROLLOUT_STEPS == 5999
    three = records_for_rollout_split(pair_codes, 1, rollout_steps=3)
    assert len(three) == 17910 > len(records)


# --------------------------------------------------------------------------
# The gate and the report
# --------------------------------------------------------------------------


def _summary(step: int, short: float, long: float) -> dict:
    fields = ("surface_speed", "sst", "phihyd_surface")
    return {
        "optimizer_step": step,
        "short_auc_10_90": {f: short for f in fields},
        "long_ratio_to_climatology": {f: long for f in fields},
        "lead_days": [0, 10],
    }


def test_the_acceptance_gate_reads_both_conditions() -> None:
    baseline = _summary(15360, 1.0, 0.93)
    passing = acceptance_gate(_summary(3840, 0.95, 0.80), baseline)
    assert passing["short_auc_no_field_worsens_by_more_than_5_percent"]
    assert passing["worst_long_ratio_at_or_below_ceiling"]
    assert passing["validation_conditions_pass"]
    assert passing["worst_long_ratio_ceiling"] == WORST_LONG_RATIO_CEILING == 0.85

    short_breach = acceptance_gate(_summary(3840, 1.06, 0.80), baseline)
    assert not short_breach["short_auc_no_field_worsens_by_more_than_5_percent"]
    assert not short_breach["validation_conditions_pass"]

    # Exactly 5% worse is inside the tolerance; a hair more is not.
    assert acceptance_gate(_summary(3840, 1.05, 0.80), baseline)["validation_conditions_pass"]

    long_breach = acceptance_gate(_summary(3840, 0.95, 0.86), baseline)
    assert not long_breach["worst_long_ratio_at_or_below_ceiling"]
    assert not long_breach["validation_conditions_pass"]
    assert "2000_day_all_values_finite" in long_breach["deferred_to_the_figure_package"]


def test_the_baseline_summary_must_be_the_step_15360_one() -> None:
    report = {"validation_summaries": [_summary(11520, 1.0, 1.0), _summary(15360, 0.9, 0.93)]}
    assert baseline_validation_summary(report)["optimizer_step"] == 15360
    with pytest.raises(BireProtocolRolloutFineTuneError):
        baseline_validation_summary({"validation_summaries": [_summary(11520, 1.0, 1.0)]})


def _report() -> dict:
    summaries = [
        _summary(step, 0.9 + index * 0.01, 0.95 - index * 0.05)
        for index, step in enumerate(CHECKPOINT_STEPS)
    ]
    baseline = _summary(BASELINE_OPTIMIZER_STEP, 1.0, 0.932)
    selected = summaries[-1]
    return {
        "content_sha256": "a" * 64,
        "optimizer": {"decay_step": 2880},
        "counts": {
            "training_rollout_records": TRAINING_RECORDS,
            "training_starts_per_regime": TRAINING_STARTS_PER_REGIME,
            "latest_training_start": 5939,
            "validation_records": 102,
        },
        "validation_summaries": summaries,
        "baseline_validation_summary": baseline,
        "checkpoint_comparison_to_baseline": {
            str(int(s["optimizer_step"])): acceptance_gate(s, baseline) for s in summaries
        },
        "selection_decision": {
            "selected_optimizer_step": int(selected["optimizer_step"]),
            "branch": "primary_rule",
        },
        "acceptance_gate": acceptance_gate(selected, baseline),
    }


def test_the_readme_renders_before_the_job_is_ever_submitted() -> None:
    """Regression guard: the duration arm lost a finished job to a README KeyError."""

    text = _readme(_report())
    assert "Six-step rollout fine-tune" in text
    assert f"{BASELINE_OPTIMIZER_STEP:,}" in text
    assert "Selected step 3,840" in text and "primary_rule" in text
    assert "17,820" in text and "5,940" in text and "5,939" in text
    assert FINE_TUNE_LOSS_CONTRACT_SHA256 in text
    assert "2e-05" in text or "2.0e-05" in text or "2e-5" in text
    for step in CHECKPOINT_STEPS:
        assert f"{step:,}" in text
    flat = " ".join(text.split())          # the sentences wrap across lines
    assert "no teacher forcing after the initial state" in flat
    assert "reused from the parent package rather than recomputed" in flat
    for stale in ("7,680", "from scratch", "three-step rollout"):
        assert stale not in text


def test_the_readme_rejects_a_report_missing_the_gate() -> None:
    report = _report()
    del report["acceptance_gate"]
    with pytest.raises(KeyError):
        _readme(report)


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------


def test_artifact_names_are_distinct_and_name_this_arm() -> None:
    """Two arms wrote the same checkpoint filename with different weights once."""

    from oceanfno import train as arm

    names = [getattr(arm, n) for n in ("REPORT_NAME", "ARRAYS_NAME", "FIGURE_NAME",
                                       "NORMALIZATION_NAME", "DIVERGENCE_NAME",
                                       "CHECKPOINT_STEM")]
    assert len(set(names)) == len(names)
    assert all("rollout_ft" in n for n in names)
    checkpoints = {f"{arm.CHECKPOINT_STEM}_{s:05d}.pt" for s in CHECKPOINT_STEPS}
    assert len(checkpoints) == len(CHECKPOINT_STEPS)



@pytest.mark.skipif(not SBATCH.is_file(), reason="launcher absent")
def test_launcher_invokes_its_own_module_and_contract() -> None:
    text = SBATCH.read_text()
    invoked = {
        line.split("-m", 1)[1].strip().split()[0]
        for line in text.splitlines()
        if " -m " in f" {line} " and "oceanfno." in line
    }
    assert invoked == {"oceanfno.train"}
    assert "model_c_bire_protocol_rollout_ft_v2.json" in text
    assert "_v1.json" not in text


def test_the_package_carries_no_module_rebinding_machinery() -> None:
    """The consolidated tree calls its collaborators directly.

    The previous layout reused a parent arm's training loop by swapping module
    globals in and out around the call. That is the mechanism this refactor
    removed, and its absence is what makes the call graph readable.
    """

    from oceanfno import anomaly, figures, train

    for module in (train, figures, anomaly):
        assert not hasattr(module, "_ParentBinding")
        assert not hasattr(module, "_SuiteBinding")
        assert not hasattr(module, "PARENT_BINDINGS")

