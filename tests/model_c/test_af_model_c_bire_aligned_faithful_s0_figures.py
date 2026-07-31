from __future__ import annotations

import json
from pathlib import Path

import pytest

from bire_repro import af_model_c_bire_aligned_s0_figures as aligned_figures
from bire_repro.af_model_c_bire_aligned_faithful import MAE_WEIGHT
from bire_repro.af_model_c_bire_aligned_faithful_s0_figures import (
    CONTRACT_STATUS,
    VERSION,
    BireAlignedFaithfulS0FigureError,
    _FaithfulBinding,
    _readme,
    load_contract,
    selected_steps,
)
from bire_repro.af_model_c_bire_aligned_full_state import STAGE_NAMES

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config/model_c_bire_aligned_faithful_s0_figures_v1.json"
PARTNER = ROOT / "config/model_c_bire_aligned_full_state_lr5e4_s0_figures_v1.json"
INCUMBENT = ROOT / "config/model_c_single_position_layernorm_s0_figures_v1.json"
SBATCH = ROOT / "slurm/models/c/af_model_c_bire_aligned_faithful_s0_figures.sbatch"

pytestmark = pytest.mark.skipif(
    not CONTRACT.is_file(),
    reason="the figure contract is written from the completed training report",
)


def test_contract_targets_the_faithful_checkpoints() -> None:
    contract, resolved, digest = load_contract(CONTRACT, verify_sources=False)
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64
    raw = json.loads(CONTRACT.read_text())
    assert raw["version"] == VERSION
    assert raw["contract_status"] == CONTRACT_STATUS
    for stage in STAGE_NAMES:
        model = raw["stages"][stage]["model"]
        assert model["mae_weight"] == MAE_WEIGHT == 0.05
        assert model["selected_by"] == "lowest_validation_loss_within_the_stage"
        assert "faithful" in raw["stages"][stage]["checkpoint"]["path"]


def test_architecture_matches_the_one_factor_partner() -> None:
    raw = json.loads(CONTRACT.read_text())
    partner = json.loads(PARTNER.read_text())
    for stage in STAGE_NAMES:
        assert (
            raw["stages"][stage]["model"]["architecture"]
            == partner["stages"][stage]["model"]["architecture"]
        )


def test_protocol_matches_every_other_published_package() -> None:
    raw = json.loads(CONTRACT.read_text())
    incumbent = json.loads(INCUMBENT.read_text())
    assert raw["protocol"] == incumbent["protocol"]
    assert raw["baselines"] == incumbent["baselines"]
    assert raw["truth"] == incumbent["truth"]
    assert raw["prior_model"] == incumbent["prior_model"]


def test_each_stage_writes_its_own_new_folder() -> None:
    raw = json.loads(CONTRACT.read_text())
    partner = json.loads(PARTNER.read_text())
    for stage in STAGE_NAMES:
        output = raw["stages"][stage]["output"]
        assert output["project"].endswith(stage)
        assert "faithful" in output["project"] and "faithful" in output["scratch"]
        assert output["project"] != partner["stages"][stage]["output"]["project"]


def test_selected_steps_are_read_from_the_contract_not_assumed() -> None:
    """Validation-based selection means the step is not known in advance."""

    steps = selected_steps(CONTRACT)
    raw = json.loads(CONTRACT.read_text())
    assert steps == tuple(
        raw["stages"][s]["model"]["optimizer_step"] for s in STAGE_NAMES
    )


def test_binding_rebinds_the_steps_and_restores_everything(tmp_path: Path) -> None:
    before = {
        name: getattr(aligned_figures, name)
        for name in ("VERSION", "CONTRACT_STATUS", "_readme", "CHECKPOINT_STEPS")
    }
    # A contract whose stages were selected at non-default epochs must still
    # load; without rebinding CHECKPOINT_STEPS it would only work by luck.
    raw = json.loads(CONTRACT.read_text())
    raw["stages"]["pretrained"]["model"]["optimizer_step"] = 1920
    moved = tmp_path / "moved.json"
    moved.write_text(json.dumps(raw))
    assert selected_steps(moved) == (1920, 7680)
    with _FaithfulBinding(moved):
        assert aligned_figures.CHECKPOINT_STEPS == (1920, 7680)
        assert aligned_figures.VERSION == VERSION
    for name, value in before.items():
        assert getattr(aligned_figures, name) == value
    load_contract(moved, verify_sources=False)


def test_rejects_a_contract_whose_stage_order_changed(tmp_path: Path) -> None:
    raw = json.loads(CONTRACT.read_text())
    raw["stage_order"] = ["finetuned", "pretrained"]
    written = tmp_path / "c.json"
    written.write_text(json.dumps(raw))
    with pytest.raises(BireAlignedFaithfulS0FigureError):
        selected_steps(written)


def test_readme_names_the_arm_and_the_three_corrections() -> None:
    with _FaithfulBinding(CONTRACT):
        with aligned_figures._FigureBinding("finetuned"):
            text = _readme({"report_content_sha256": "0" * 64})
    assert "Bire-faithful" in text
    assert "0.05" in text
    assert "cosine" in text
    assert "validation loss" in text


def test_launcher_invokes_this_arms_module_and_contract() -> None:
    text = SBATCH.read_text()
    assert text.count("bire_repro.af_model_c_bire_aligned_faithful_s0_figures") == 2
    assert "lr_control" not in text
    assert CONTRACT.name in text
