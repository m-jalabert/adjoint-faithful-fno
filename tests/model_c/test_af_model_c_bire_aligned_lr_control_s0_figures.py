from __future__ import annotations

import json
from pathlib import Path

import pytest

from bire_repro import af_model_c_bire_aligned_s0_figures as aligned_figures
from bire_repro.af_model_c_bire_aligned_full_state import (
    CHECKPOINT_STEPS,
    STAGE_NAMES,
)
from bire_repro.af_model_c_bire_aligned_lr_control_s0_figures import (
    CONTRACT_STATUS,
    VERSION,
    _ControlBinding,
    _readme,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config/model_c_bire_aligned_full_state_lr5e4_s0_figures_v1.json"
PARENT = ROOT / "config/model_c_bire_aligned_full_state_s0_figures_v1.json"
INCUMBENT = ROOT / "config/model_c_single_position_layernorm_s0_figures_v1.json"
SBATCH = ROOT / "slurm/models/c/af_model_c_bire_aligned_lr_control_s0_figures.sbatch"

pytestmark = pytest.mark.skipif(
    not CONTRACT.is_file(),
    reason="the figure contract is written from the completed training report",
)


def test_contract_targets_the_learning_rate_control_checkpoints() -> None:
    contract, resolved, digest = load_contract(CONTRACT, verify_sources=False)
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64
    raw = json.loads(CONTRACT.read_text())
    assert raw["version"] == VERSION
    assert raw["contract_status"] == CONTRACT_STATUS
    for stage, step in zip(STAGE_NAMES, CHECKPOINT_STEPS):
        model = raw["stages"][stage]["model"]
        assert model["optimizer_step"] == step
        assert model["initial_learning_rate"] == 5.0e-4
        assert "lr5e4" in raw["stages"][stage]["checkpoint"]["path"]
    assert contract["active_stage"] == "finetuned"


def test_architecture_is_identical_to_the_one_factor_partner() -> None:
    """The control moves only the learning rate, so the maps must match."""

    raw = json.loads(CONTRACT.read_text())
    partner = json.loads(PARENT.read_text())
    for stage in STAGE_NAMES:
        assert (
            raw["stages"][stage]["model"]["architecture"]
            == partner["stages"][stage]["model"]["architecture"]
        )


def test_protocol_matches_the_incumbent_and_the_partner_package() -> None:
    raw = json.loads(CONTRACT.read_text())
    incumbent = json.loads(INCUMBENT.read_text())
    partner = json.loads(PARENT.read_text())
    assert raw["protocol"] == incumbent["protocol"] == partner["protocol"]
    assert raw["baselines"] == incumbent["baselines"]
    assert raw["truth"] == incumbent["truth"]
    assert raw["prior_model"] == incumbent["prior_model"]


def test_each_stage_writes_a_new_folder_that_collides_with_nothing() -> None:
    raw = json.loads(CONTRACT.read_text())
    partner = json.loads(PARENT.read_text())
    incumbent = json.loads(INCUMBENT.read_text())
    seen = {incumbent["output"]["project"]}
    for stage in STAGE_NAMES:
        for source in (partner, raw):
            seen.add(source["stages"][stage]["output"]["project"])
        output = raw["stages"][stage]["output"]
        assert output["project"].endswith(stage)
        assert "lr5e4" in output["project"] and "lr5e4" in output["scratch"]
    # incumbent + 2 partner stages + 2 control stages, all distinct
    assert len(seen) == 5


def test_binding_installs_and_restores_the_partner_runner() -> None:
    before = {
        name: getattr(aligned_figures, name)
        for name in ("VERSION", "CONTRACT_STATUS", "_readme")
    }
    with _ControlBinding():
        assert aligned_figures.VERSION == VERSION
        assert aligned_figures.CONTRACT_STATUS == CONTRACT_STATUS
        assert aligned_figures._readme is _readme
    for name, value in before.items():
        assert getattr(aligned_figures, name) is value
    # The partner runner must still reject this arm's contract on its own.
    with pytest.raises(aligned_figures.BireAlignedS0FigureError):
        aligned_figures.load_contract(CONTRACT, verify_sources=False)


def test_readme_names_the_control_and_its_stage() -> None:
    with _ControlBinding():
        for stage, step in zip(STAGE_NAMES, CHECKPOINT_STEPS):
            with aligned_figures._FigureBinding(stage):
                text = _readme({"report_content_sha256": "0" * 64})
            assert f"{step:,}" in text
            assert stage in text
            assert "learning-rate control" in text
            assert "0.0005" in text or "5e-04" in text


def test_launcher_invokes_this_arms_module_and_contract() -> None:
    text = SBATCH.read_text()
    assert text.count("bire_repro.af_model_c_bire_aligned_lr_control_s0_figures") == 2
    assert "bire_repro.af_model_c_bire_aligned_s0_figures" not in text
    assert CONTRACT.name in text
