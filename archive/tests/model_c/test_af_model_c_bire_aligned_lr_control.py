from __future__ import annotations

import json
from pathlib import Path

import pytest

from bire_repro import af_model_c_bire_aligned_full_state as aligned
from bire_repro.af_model_c_bire_aligned_lr_control import (
    CONTRACT_STATUS,
    CONTROL_LEARNING_RATE,
    FROZEN_TRAINING_FIELDS,
    PARENT_LEARNING_RATE,
    VERSION,
    BireAlignedLearningRateControlError,
    _ParentBinding,
    _readme,
    load_contract,
)

CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "model_c_bire_aligned_full_state_lr5e4_v1.json"
)
PARENT = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "model_c_bire_aligned_full_state_v1.json"
)


def test_contract_declares_the_learning_rate_control() -> None:
    contract, resolved, digest = load_contract(CONTRACT)
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64
    assert contract["version"] == VERSION
    assert contract["contract_status"] == CONTRACT_STATUS
    assert contract["training"]["initial_learning_rate"] == CONTROL_LEARNING_RATE
    assert CONTROL_LEARNING_RATE == 5.0e-4
    assert PARENT_LEARNING_RATE == 1.0e-2


def test_learning_rate_is_the_only_difference_from_the_parent() -> None:
    contract, _, _ = load_contract(CONTRACT)
    parent = json.loads(PARENT.read_text())
    assert contract["architecture"] == parent["architecture"]
    assert contract["loss"] == parent["loss"]
    assert contract["stages"] == parent["stages"]
    assert contract["selection"] == parent["selection"]
    for field in FROZEN_TRAINING_FIELDS:
        assert contract["training"][field] == parent["training"][field], field
    differing = {
        key
        for key in set(contract["training"]) | set(parent["training"])
        if contract["training"].get(key) != parent["training"].get(key)
    }
    assert differing == {"initial_learning_rate"}


def test_output_does_not_collide_with_the_parent_arm() -> None:
    contract, _, _ = load_contract(CONTRACT)
    parent = json.loads(PARENT.read_text())
    assert contract["output"]["project_root"] != parent["output"]["project_root"]
    assert contract["output"]["scratch_root"] != parent["output"]["scratch_root"]
    assert contract["output"]["project_root"].endswith("lr5e4_v1")


def test_held_state_stays_sealed_for_the_training_run() -> None:
    contract, _, _ = load_contract(CONTRACT)
    read = contract["read_contract"]
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


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("training", "initial_learning_rate"), 0.01),
        (("training", "batch_size"), 4),
        (("training", "weight_decay"), 1e-05),
        (("training", "adam_betas"), [0.9, 0.999]),
        (("training", "decay_factor"), 0.5),
        (("architecture", "n_layers"), 4),
        (("loss", "mae_weight"), 0.1),
    ],
)
def test_rejects_a_second_moving_quantity(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    """A control that moves two things answers nothing, so it must not load."""

    contract = json.loads(CONTRACT.read_text())
    target = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    written = tmp_path / "contract.json"
    written.write_text(json.dumps(contract))
    with pytest.raises(BireAlignedLearningRateControlError):
        load_contract(written, verify_sources=False)


def test_rejects_a_contract_whose_parent_reference_is_not_the_1e2_arm(
    tmp_path: Path,
) -> None:
    contract = json.loads(CONTRACT.read_text())
    contract["sources"]["parent_contract"]["sha256"] = "0" * 64
    written = tmp_path / "contract.json"
    written.write_text(json.dumps(contract))
    with pytest.raises(BireAlignedLearningRateControlError):
        load_contract(written, verify_sources=False)


def test_parent_binding_installs_and_restores() -> None:
    before = {
        name: getattr(aligned, name)
        for name in ("load_contract", "_readme", "VERSION")
    }
    with _ParentBinding():
        assert aligned.load_contract is load_contract
        assert aligned._readme is _readme
        assert aligned.VERSION == VERSION
    for name, value in before.items():
        assert getattr(aligned, name) is value


def test_contract_still_loads_while_the_binding_is_active() -> None:
    """The loader runs *inside* the binding, where ``aligned.VERSION`` is ours.

    Reading the parent's expected version through that global made the parent
    reference check fail only under the real preflight/run path.
    """

    with _ParentBinding():
        contract, _, _ = load_contract(CONTRACT)
        assert contract["version"] == VERSION
        assert aligned.load_contract(CONTRACT)[0]["version"] == VERSION


def test_parent_rejects_this_contract_without_the_binding() -> None:
    """The two arms' loaders must not accept each other's contracts."""

    with pytest.raises(aligned.BireAlignedFullStateError):
        aligned.load_contract(CONTRACT, verify_sources=False)
    with pytest.raises(BireAlignedLearningRateControlError):
        load_contract(PARENT, verify_sources=False)


def test_launcher_invokes_this_arms_module_and_contract() -> None:
    """The launcher was derived from the parent's by substitution.

    A missed substitution left it invoking the parent module with this arm's
    contract, which the parent loader rejected only at submission time.
    """

    sbatch = (
        Path(__file__).resolve().parents[2]
        / "slurm"
        / "models"
        / "c"
        / "af_model_c_bire_aligned_lr_control.sbatch"
    )
    text = sbatch.read_text()
    assert "bire_repro.af_model_c_bire_aligned_lr_control" in text
    assert "bire_repro.af_model_c_bire_aligned_full_state" not in text
    assert CONTRACT.name in text
    assert text.count("bire_repro.af_model_c_bire_aligned_lr_control") == 2


def test_published_readme_names_the_control_not_the_parent() -> None:
    text = _readme(
        {
            "content_sha256": "0" * 64,
            "evaluation_summaries": [
                {
                    "stage_id": "pretrained",
                    "optimizer_step": 3840,
                    "worst_primary_10_to_90_ratio": 0.5,
                    "worst_mid_bottom_modewise_ratio_all_leads": 2.0,
                    "gate": {"pass": True},
                }
            ],
        }
    )
    assert "5e-4" in text
    assert "learning-rate control" in text
    assert "pretrained" in text
