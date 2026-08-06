from __future__ import annotations

import json
from pathlib import Path

import pytest

from bire_repro import af_model_c_bire_aligned_s0_figures as aligned_figures
from bire_repro.af_model_c import MODEL_C_LOSS_V1_CONTRACT_SHA256
from bire_repro.af_model_c_bire_aligned_loss_recovery_s0_figures import (
    CONTRACT_STATUS,
    STAGE_NAMES,
    VERSION,
    BireAlignedLossRecoveryS0FigureError,
    _LossRecoveryBinding,
    _readme,
    load_contract,
    selected_steps,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config/model_c_bire_aligned_loss_recovery_s0_figures_v1.json"
PARTNER = ROOT / "config/model_c_bire_aligned_full_state_lr5e4_s0_figures_v1.json"
INCUMBENT = ROOT / "config/model_c_single_position_layernorm_s0_figures_v1.json"
SBATCH = ROOT / "slurm/models/c/af_model_c_bire_aligned_loss_recovery_s0_figures.sbatch"

pytestmark = pytest.mark.skipif(
    not CONTRACT.is_file(),
    reason="the figure contract is written from the completed training report",
)


def test_contract_targets_one_gate_selected_checkpoint() -> None:
    contract, resolved, digest = load_contract(CONTRACT, verify_sources=False)
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64
    raw = json.loads(CONTRACT.read_text())
    assert raw["version"] == VERSION
    assert raw["contract_status"] == CONTRACT_STATUS
    assert tuple(raw["stage_order"]) == STAGE_NAMES == ("selected",)
    model = raw["stages"]["selected"]["model"]
    assert model["objective"] == "incumbent_group_balanced_model_c_loss_v1"
    assert model["base_loss_contract_sha256"] == MODEL_C_LOSS_V1_CONTRACT_SHA256
    assert model["rollout_steps"] == 3
    assert contract["active_stage"] == "selected"


def test_architecture_matches_the_objective_partner() -> None:
    """The control moves only the objective, so the maps must match."""

    raw = json.loads(CONTRACT.read_text())
    partner = json.loads(PARTNER.read_text())
    assert (
        raw["stages"]["selected"]["model"]["architecture"]
        == partner["stages"]["finetuned"]["model"]["architecture"]
    )


def test_protocol_matches_every_other_published_package() -> None:
    raw = json.loads(CONTRACT.read_text())
    incumbent = json.loads(INCUMBENT.read_text())
    assert raw["protocol"] == incumbent["protocol"]
    assert raw["baselines"] == incumbent["baselines"]
    assert raw["truth"] == incumbent["truth"]
    assert raw["prior_model"] == incumbent["prior_model"]
    assert (
        raw["stages"]["selected"]["output"]["required"]
        == incumbent["output"]["required"]
    )


def test_output_is_a_new_folder() -> None:
    raw = json.loads(CONTRACT.read_text())
    partner = json.loads(PARTNER.read_text())
    output = raw["stages"]["selected"]["output"]
    assert output["project"].endswith("selected")
    assert "loss_recovery" in output["project"] and "loss_recovery" in output["scratch"]
    for stage in ("pretrained", "finetuned"):
        assert output["project"] != partner["stages"][stage]["output"]["project"]


def test_selected_step_is_read_from_the_contract() -> None:
    raw = json.loads(CONTRACT.read_text())
    assert selected_steps(CONTRACT) == (
        raw["stages"]["selected"]["model"]["optimizer_step"],
    )


def test_binding_installs_single_stage_vocabulary_and_restores(tmp_path: Path) -> None:
    before = {
        name: getattr(aligned_figures, name)
        for name in (
            "VERSION",
            "CONTRACT_STATUS",
            "_readme",
            "CHECKPOINT_STEPS",
            "STAGE_NAMES",
            "_selected_stepper",
        )
    }
    with _LossRecoveryBinding(CONTRACT):
        assert aligned_figures.STAGE_NAMES == ("selected",)
        assert aligned_figures.CHECKPOINT_STEPS == selected_steps(CONTRACT)
        assert aligned_figures.VERSION == VERSION
    for name, value in before.items():
        assert getattr(aligned_figures, name) == value
    # The two-stage runner must still reject this arm's contract on its own.
    with pytest.raises(aligned_figures.BireAlignedS0FigureError):
        aligned_figures.load_contract(CONTRACT, verify_sources=False)


def test_rejects_a_contract_with_more_than_one_stage(tmp_path: Path) -> None:
    raw = json.loads(CONTRACT.read_text())
    raw["stage_order"] = ["pretrained", "finetuned"]
    written = tmp_path / "c.json"
    written.write_text(json.dumps(raw))
    with pytest.raises(BireAlignedLossRecoveryS0FigureError):
        selected_steps(written)


def test_readme_names_the_restored_objective() -> None:
    with _LossRecoveryBinding(CONTRACT):
        with aligned_figures._FigureBinding("selected"):
            text = _readme({"report_content_sha256": "0" * 64})
    assert "loss-recovery" in text
    assert "15:15:15:1" in text
    assert "L_state" in text
    assert "7,680" in text


def test_launcher_invokes_this_arms_module_and_contract() -> None:
    text = SBATCH.read_text()
    assert text.count("bire_repro.af_model_c_bire_aligned_loss_recovery_s0_figures") == 2
    assert "lr_control" not in text
    assert CONTRACT.name in text
