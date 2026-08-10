"""Tests for the canonical meridional-32 continuation of the local24 arm.

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
    FINE_TUNE_LOSS_CONTRACT_SHA256,
    INCREMENT_WEIGHT,
    LEARNING_RATE,
    LOCAL_KERNEL_SIZE,
    MAXIMUM_STEPS,
    MODE_MIGRATION,
    PARENT_VERSION,
    PARENT_MODES,
    ROLLOUT_STEPS,
    ROLLOUT_WEIGHT,
    SPECTRAL_WEIGHT,
    TRAINING_RECORDS,
    TRAINING_STARTS_PER_REGIME,
    VERSION,
    WORST_LONG_RATIO_CEILING,
    BireProtocolRolloutFineTuneError,
    BireProtocolRolloutFineTuneLossConfig,
    BireY32TrainingError,
    Y32_MODES,
    _readme,
    acceptance_gate,
    baseline_validation_summary,
    fine_tune_loss_config,
    fine_tune_loss_contract,
    fine_tune_loss_contract_sha256,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_bire_protocol_rollout_ft_local24_y32_v1.json"
PARENT = ROOT / "config/model_c_bire_protocol_rollout_ft_local24_v1.json"
SBATCH = ROOT / "slurm/models/c/train.sbatch"

torch = pytest.importorskip("torch", reason="the objective algebra needs PyTorch")


def _portable_state_dict(model) -> dict:
    """Mirror the checkpoint writer without coupling this test to its helper."""

    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key != "_metadata"
    }


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
# The active Y32 architecture and exact local24 warm migration
# --------------------------------------------------------------------------


def test_y32_has_32_meridional_modes_and_the_trained_local_branch() -> None:
    pytest.importorskip("neuralop")
    from oceanfno.model import BireY32Architecture, build_bire_y32_model

    architecture = BireY32Architecture()
    assert architecture.n_modes == Y32_MODES == (32, 24)
    assert architecture.local_kernel_size == 3
    model = build_bire_y32_model(architecture)

    convolutions = model.fno.fno_blocks.convs
    assert len(convolutions) == 3
    for index, convolution in enumerate(convolutions):
        assert tuple(convolution.n_modes) == (32, 13), index
        weight = model.state_dict()[
            f"fno.fno_blocks.convs.{index}.weight.tensor"
        ]
        assert tuple(weight.shape) == (128, 128, 32, 13)

    local = model.local
    assert tuple(local.weight.shape) == (46, 49, 3, 3)
    assert local.padding == (1, 1)
    assert local.bias is None
    assert sum(parameter.numel() for parameter in model.parameters()) == 21_005_164


def test_y32_migration_is_centered_strict_and_bitwise_function_preserving() -> None:
    pytest.importorskip("neuralop")
    from oceanfno.model import (
        BireLocal24Architecture,
        BireY32Architecture,
        build_bire_local24_model,
        build_bire_y32_model,
        migrate_local24_state_dict,
    )

    torch.manual_seed(19)
    parent = build_bire_local24_model(BireLocal24Architecture()).eval()
    # A real retained parent has a trained, nonzero local correction.  Seeding
    # it here catches a migration that silently re-zeroes that learned branch.
    with torch.no_grad():
        parent.local.weight.normal_(mean=0.0, std=0.02)
    parent_state = _portable_state_dict(parent)
    parent_before = {key: value.clone() for key, value in parent_state.items()}
    target = build_bire_y32_model(BireY32Architecture()).eval()
    result = migrate_local24_state_dict(parent_state, target)
    state = result["state_dict"]
    provenance = result["provenance"]

    assert provenance["source_n_modes_tensor_order_y_x"] == [24, 24]
    assert provenance["target_n_modes_tensor_order_y_x"] == [32, 24]
    assert len(provenance["spectral_expansions"]) == 3
    assert all(
        record["copied_meridional_slice"] == [4, 28]
        for record in provenance["spectral_expansions"]
    )
    assert provenance["strict_load"] is True
    assert provenance["missing_keys"] == []
    assert provenance["unexpected_keys"] == []
    assert provenance["initial_map_preserved_by_centered_zero_extension"] is True

    target_keys = {key for key in target.state_dict() if key != "_metadata"}
    assert set(state) == set(parent_state) == target_keys
    spectral = {
        f"fno.fno_blocks.convs.{index}.weight.tensor" for index in range(3)
    }
    for key, parent_value in parent_state.items():
        if key not in spectral:
            assert torch.equal(state[key], parent_value), key
    for key in spectral:
        assert tuple(parent_state[key].shape) == (128, 128, 24, 13)
        assert tuple(state[key].shape) == (128, 128, 32, 13)
        assert torch.equal(state[key][..., 4:28, :], parent_state[key]), key
        assert torch.count_nonzero(state[key][..., :4, :]).item() == 0, key
        assert torch.count_nonzero(state[key][..., 28:, :]).item() == 0, key
    assert torch.count_nonzero(parent_state["local.weight"]).item() > 0
    assert torch.equal(state["local.weight"], parent_state["local.weight"])
    assert all(torch.equal(parent_state[key], parent_before[key]) for key in parent_state)

    target.load_state_dict(state, strict=True)
    features = torch.randn(1, 49, 62, 62)
    with torch.inference_mode():
        expected = parent(features)
        actual = target(features)
    assert torch.equal(actual, expected)


def test_y32_migration_rejects_missing_unexpected_and_wrong_shape_state() -> None:
    pytest.importorskip("neuralop")
    from oceanfno.model import (
        BireAlignedFullStateError,
        BireLocal24Architecture,
        BireY32Architecture,
        build_bire_local24_model,
        build_bire_y32_model,
        migrate_local24_state_dict,
    )

    parent = build_bire_local24_model(BireLocal24Architecture())
    state = _portable_state_dict(parent)
    target = build_bire_y32_model(BireY32Architecture())
    spectral = "fno.fno_blocks.convs.0.weight.tensor"

    missing = dict(state)
    missing.pop("fno.projection.fcs.1.bias")
    unexpected = dict(state)
    unexpected["undeclared.weight"] = torch.zeros(1)
    wrong_shape = dict(state)
    wrong_shape[spectral] = state[spectral][..., :23, :]
    for tampered in (missing, unexpected, wrong_shape):
        with pytest.raises(BireAlignedFullStateError):
            migrate_local24_state_dict(tampered, target)


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the fine-tune contract is absent")
def test_the_contract_moves_only_the_declared_quantities() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    parent = json.loads(PARENT.read_text())
    assert contract["version"] == VERSION != parent["version"]
    assert contract["contract_status"] == CONTRACT_STATUS
    for field in ("dataset", "normalization", "training", "loss", "checkpoint_selection"):
        assert contract[field] == parent[field], field

    architecture_changes = {
        field
        for field in set(contract["architecture"]) | set(parent["architecture"])
        if contract["architecture"].get(field) != parent["architecture"].get(field)
    }
    assert architecture_changes == {"n_modes"}
    assert tuple(parent["architecture"]["n_modes"]) == PARENT_MODES == (24, 24)
    assert tuple(contract["architecture"]["n_modes"]) == Y32_MODES == (32, 24)
    assert parent["architecture"]["local_kernel_size"] == (
        contract["architecture"]["local_kernel_size"]
    ) == LOCAL_KERNEL_SIZE == 3


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
def test_initialization_is_the_local24_checkpoint_with_centered_zero_extension() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    initialization = contract["initialization"]
    assert initialization["version"] == PARENT_VERSION
    assert int(initialization["optimizer_step"]) == BASELINE_OPTIMIZER_STEP == 3840
    assert initialization["load_only"] == "model_state_dict"
    assert initialization["optimizer_state_loaded"] is False
    assert initialization["normalization_reused"] is True
    assert contract["training"]["from_scratch"] is False
    assert contract["training"]["load_optimizer_state"] is False
    assert initialization["checkpoint"].endswith(
        "bire_protocol_rollout_ft_local24_v1/selected.pt"
    )
    assert initialization["mode_migration"] == MODE_MIGRATION
    assert initialization["local_branch_initialization"] == "copied_from_parent"
    assert initialization["local_branch_bias"] is False
    assert (
        contract["sources"]["initialization_checkpoint"]["path"]
        == initialization["checkpoint"]
    )
    assert (
        contract["sources"]["parent_normalization"]["path"]
        .endswith(
            "model_c_bire_protocol_rollout_ft_local24_train_only_normalization.npz"
        )
    )


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the fine-tune contract is absent")
def test_the_loss_block_declares_the_six_step_objective() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    parent = json.loads(PARENT.read_text())
    loss = contract["loss"]
    assert loss["contract_sha256"] == FINE_TUNE_LOSS_CONTRACT_SHA256
    assert loss["derived_from_contract_sha256"] == MODEL_C_LOSS_V1_CONTRACT_SHA256
    assert int(loss["rollout_steps"]) == int(parent["loss"]["rollout_steps"]) == 6
    assert loss == parent["loss"]
    assert loss["coefficients"]["rollout"] == 0.50
    assert loss["state"] == parent["loss"]["state"]


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the fine-tune contract is absent")
def test_completed_outputs_are_pinned_and_a_rerun_would_be_refused() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    scratch = Path(contract["output"]["scratch_root"])
    project = Path(contract["output"]["project_root"])
    assert scratch.name == project.name == "bire_protocol_rollout_ft_local24_y32_v1"
    assert (scratch / "selected.pt").is_file()
    assert (project / "manifest.json").is_file()
    assert scratch.exists(), "the completed scratch root is the overwrite guard"
    assert project != Path(json.loads(PARENT.read_text())["output"]["project_root"])



@pytest.mark.skipif(not CONTRACT.is_file(), reason="the fine-tune contract is absent")
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda c: c["architecture"].update(hidden_channels=64), id="architecture_moved"
        ),
        pytest.param(
            lambda c: c["architecture"].update(n_modes=[24, 24]),
            id="meridional_modes_reverted",
        ),
        pytest.param(
            lambda c: c["architecture"].update(n_modes=[24, 32]),
            id="axis_order_reversed",
        ),
        pytest.param(
            lambda c: c["architecture"].update(local_kernel_size=None), id="local_removed"
        ),
        pytest.param(
            lambda c: c["initialization"].update(
                mode_migration="prefix_copy_into_first_24_meridional_indices"
            ),
            id="uncentered_migration",
        ),
        pytest.param(
            lambda c: c["initialization"].update(local_branch_initialization="zeros"),
            id="trained_local_branch_zeroed",
        ),
        pytest.param(
            lambda c: c["training"].update(load_optimizer_state=True),
            id="optimizer_state_loaded",
        ),
    ],
)
def test_a_tampered_contract_is_rejected(mutate, tmp_path) -> None:
    with pytest.raises(BireY32TrainingError):
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
    baseline = _summary(BASELINE_OPTIMIZER_STEP, 1.0, 0.93)
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

    # Passing the absolute 0.85 ceiling is insufficient if the architecture
    # fine-tune regresses against its archived six-step parent.
    stronger_parent = _summary(BASELINE_OPTIMIZER_STEP, 1.0, 0.70)
    parent_breach = acceptance_gate(_summary(3840, 0.95, 0.80), stronger_parent)
    assert parent_breach["worst_long_ratio_at_or_below_ceiling"]
    assert not parent_breach["worst_long_ratio_no_worse_than_parent"]
    assert not parent_breach["validation_conditions_pass"]


def test_the_baseline_summary_must_be_the_parent_step_3840_one() -> None:
    report = {
        "validation_summaries": [
            _summary(1920, 1.0, 1.0),
            _summary(BASELINE_OPTIMIZER_STEP, 0.9, 0.93),
        ]
    }
    assert baseline_validation_summary(report)["optimizer_step"] == BASELINE_OPTIMIZER_STEP
    with pytest.raises(BireProtocolRolloutFineTuneError):
        baseline_validation_summary({"validation_summaries": [_summary(1920, 1.0, 1.0)]})


def _report() -> dict:
    summaries = [
        _summary(step, 0.9 + index * 0.01, 0.95 - index * 0.05)
        for index, step in enumerate(CHECKPOINT_STEPS)
    ]
    baseline = _summary(BASELINE_OPTIMIZER_STEP, 1.0, 0.932)
    selected = summaries[-1]
    return {
        "content_sha256": "a" * 64,
        "parameter_count": 21_005_164,
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
    assert "Canonical meridional-32" in text
    assert f"{BASELINE_OPTIMIZER_STEP:,}" in text
    assert "Selected step 3,840" in text and "primary_rule" in text
    assert "21,005,164" in text
    for step in CHECKPOINT_STEPS:
        assert f"{step:,}" in text
    flat = " ".join(text.split())
    assert "24 x 24 to 32 x 24" in flat
    assert "embedded in indices 4:28" in flat
    assert "local 3 x 3 branch is copied unchanged" in flat
    for stale in ("7,680", "from scratch", "24 x 16", "zero-initialized local"):
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
    assert all("rollout_ft_local24_y32" in n for n in names)
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
    assert "model_c_bire_protocol_rollout_ft_local24_y32_v1.json" in text


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
        assert not hasattr(module, "_y32_runner")
        assert not hasattr(module, "_y32_figures_runner")
        assert not hasattr(module, "_y32_anomaly_runner")
