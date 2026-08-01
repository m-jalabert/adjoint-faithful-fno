from __future__ import annotations

import json
from pathlib import Path

import pytest

from bire_repro.af_model_c import MODEL_C_LOSS_V1_CONTRACT_SHA256, model_c_loss_config
from bire_repro.af_model_c_bire_aligned_loss_recovery import (
    CHECKPOINT_STEPS,
    CONTRACT_STATUS,
    FROZEN_TRAINING_FIELDS,
    LEARNING_RATE,
    ROLLOUT_STEPS,
    VERSION,
    BireAlignedLossRecoveryError,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config/model_c_bire_aligned_loss_recovery_v1.json"
PARENT = ROOT / "config/model_c_bire_aligned_full_state_lr5e4_v1.json"
SBATCH = ROOT / "slurm/models/c/af_model_c_bire_aligned_loss_recovery.sbatch"


def test_contract_restores_the_incumbent_objective() -> None:
    contract, resolved, digest = load_contract(CONTRACT)
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64
    assert contract["version"] == VERSION
    assert contract["contract_status"] == CONTRACT_STATUS
    loss = contract["loss"]
    assert loss["objective"] == "incumbent_group_balanced_model_c_loss_v1"
    assert loss["contract_sha256"] == MODEL_C_LOSS_V1_CONTRACT_SHA256
    assert loss["rollout_steps"] == ROLLOUT_STEPS == 3
    assert set(loss["components"]) == {
        "state",
        "increment",
        "rollout",
        "spectral",
        "western_boundary",
    }


def test_declared_coefficients_match_the_live_loss_v1_config() -> None:
    """The contract must describe the objective the code will actually build."""

    contract, _, _ = load_contract(CONTRACT)
    config = model_c_loss_config("v1")
    coefficients = contract["loss"]["coefficients"]
    assert config.rollout_steps == contract["loss"]["rollout_steps"]
    assert config.increment_weight == coefficients["increment"]
    assert config.rollout_weight == coefficients["rollout"]
    assert config.spectral_weight == coefficients["spectral"]
    assert config.boundary_weight == coefficients["boundary"]
    assert config.spectral_bins == contract["loss"]["spectral_bins"]
    assert config.western_boundary_width == contract["loss"]["western_boundary_width"]


def test_architecture_and_optimizer_are_frozen_against_the_parent() -> None:
    contract, _, _ = load_contract(CONTRACT)
    parent = json.loads(PARENT.read_text())
    assert contract["architecture"] == parent["architecture"]
    for field in FROZEN_TRAINING_FIELDS:
        assert contract["training"][field] == parent["training"][field], field
    assert contract["training"]["initial_learning_rate"] == LEARNING_RATE == 5.0e-4
    assert contract["training"]["batch_size"] == 8
    assert contract["training"]["maximum_steps"] == 7680
    assert tuple(contract["training"]["checkpoint_steps"]) == CHECKPOINT_STEPS
    # Same optimizer-step and sequence-exposure budget as every Bire-aligned arm.
    assert 7680 * 8 == parent["training"]["maximum_steps"] * parent["training"]["batch_size"]


def test_objective_prose_cannot_contradict_the_authoritative_fields() -> None:
    """Regression guard for the stale MAE-weight prose the faithful arm shipped."""

    for stage in load_contract(CONTRACT)[0]["stages"]:
        assert "MAE" not in stage["objective"]
        assert stage["rollout_steps"] == ROLLOUT_STEPS
    contract = json.loads(CONTRACT.read_text())
    contract["stages"][0]["objective"] = "MSE(x, y) + 0.01 MAE(x, y)"
    written = ROOT / "config" / "_tmp_reject.json"
    try:
        written.write_text(json.dumps(contract))
        with pytest.raises(BireAlignedLossRecoveryError):
            load_contract(written, verify_sources=False)
    finally:
        written.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("training", "initial_learning_rate"), 0.01),
        (("training", "batch_size"), 4),
        (("training", "maximum_steps"), 15360),
        (("training", "rollout_steps"), 2),
        (("architecture", "n_layers"), 4),
        (("architecture", "local_kernel_size"), 3),
        (("loss", "objective"), "wet_cell_mse_plus_0p01_mae"),
        (("loss", "contract_sha256"), "0" * 64),
    ],
)
def test_rejects_drift_in_any_frozen_or_declared_quantity(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    contract = json.loads(CONTRACT.read_text())
    target = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    written = tmp_path / "contract.json"
    written.write_text(json.dumps(contract))
    with pytest.raises(BireAlignedLossRecoveryError):
        load_contract(written, verify_sources=False)


def test_rejects_a_contract_that_did_not_change_the_objective(tmp_path: Path) -> None:
    """A control that changes nothing answers nothing."""

    parent = json.loads(PARENT.read_text())
    contract = json.loads(CONTRACT.read_text())
    contract["loss"] = parent["loss"]
    written = tmp_path / "contract.json"
    written.write_text(json.dumps(contract))
    with pytest.raises(BireAlignedLossRecoveryError):
        load_contract(written, verify_sources=False)


def test_output_does_not_collide_with_any_earlier_arm() -> None:
    contract, _, _ = load_contract(CONTRACT)
    parent = json.loads(PARENT.read_text())
    assert contract["output"]["project_root"] != parent["output"]["project_root"]
    assert contract["output"]["scratch_root"] != parent["output"]["scratch_root"]
    assert contract["output"]["project_root"].endswith("bire_aligned_loss_recovery_v1")


def test_held_state_stays_sealed() -> None:
    read = load_contract(CONTRACT)[0]["read_contract"]
    assert read["training_state"] is True
    for sealed in (
        "validation_state",
        "inference_state",
        "held_s0_state",
        "intermediate_wind_state",
        "response_state",
        "adjoint_state",
    ):
        assert read[sealed] is False


def test_decision_logic_is_declared_before_the_result() -> None:
    logic = load_contract(CONTRACT)[0]["decision_logic"]
    assert set(logic) == {
        "short_term_ACC_recovers_and_day2000_flat",
        "short_term_ACC_still_poor_but_rollout_flat",
        "short_term_improves_but_long_term_growth_returns",
        "no_meaningful_improvement",
    }


def test_launcher_invokes_this_arms_module_and_contract() -> None:
    text = SBATCH.read_text()
    assert text.count("bire_repro.af_model_c_bire_aligned_loss_recovery") == 2
    assert "bire_repro.af_model_c_bire_aligned_lr_control" not in text
    assert "bire_repro.af_model_c_bire_aligned_faithful" not in text
    assert CONTRACT.name in text
