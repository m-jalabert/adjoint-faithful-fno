from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bire_repro import af_model_c_bire_aligned_s0_figures as aligned_figures
from bire_repro import af_model_c_bire_s0_figures as frozen_figures
from bire_repro.af_model_c import MODEL_C_LOSS_V1_CONTRACT_SHA256
from bire_repro.af_model_c_bire_aligned_chronological_s0_figures import (
    CONTRACT_STATUS,
    STAGE_NAMES,
    VERSION,
    ChronologicalS0FigureError,
    _ChronologicalBinding,
    _readme,
    _train_only_s0_climatology,
    load_contract,
    selected_steps,
)
from bire_repro.af_model_c_bire_s0_long_truth import EXPECTED_STARTS
from bire_repro.af_model_c_chronological_split import VERSION as SPLIT_VERSION
from bire_repro.af_model_c_chronological_split import snapshot_codes

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config/model_c_bire_aligned_chronological_s0_figures_v1.json"
PARTNER = ROOT / "config/model_c_bire_aligned_loss_recovery_s0_figures_v1.json"
SBATCH = ROOT / "slurm/models/c/af_model_c_bire_aligned_chronological_s0_figures.sbatch"

pytestmark = pytest.mark.skipif(
    not CONTRACT.is_file(),
    reason="the figure contract is written from the completed training report",
)


def test_contract_targets_the_validation_selected_checkpoint() -> None:
    contract, resolved, digest = load_contract(CONTRACT, verify_sources=False)
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64
    raw = json.loads(CONTRACT.read_text())
    assert raw["version"] == VERSION
    assert raw["contract_status"] == CONTRACT_STATUS
    assert tuple(raw["stage_order"]) == STAGE_NAMES
    model = raw["stages"]["selected"]["model"]
    assert model["split_version"] == SPLIT_VERSION
    assert model["base_loss_contract_sha256"] == MODEL_C_LOSS_V1_CONTRACT_SHA256
    assert model["rollout_steps"] == 3
    assert "held_validation_block" in model["selected_by"]


def test_normalization_is_this_arms_train_only_artifact() -> None:
    """The shared seed-20260724 normalizer would leak 5040--6209."""

    raw = json.loads(CONTRACT.read_text())
    partner = json.loads(PARTNER.read_text())
    ours = raw["artifacts"]["selected_normalization"]["path"]
    assert "chronological" in ours
    assert ours != partner["artifacts"]["selected_normalization"]["path"]


def test_climatology_baseline_is_declared_train_only() -> None:
    raw = json.loads(CONTRACT.read_text())
    assert "0_5039" in raw["baselines"]["climatology"]
    assert "climatology_note" in raw["baselines"]


def test_the_climatology_binding_actually_changes_the_interval() -> None:
    """Both intervals hold 5,040 days, so the runner's count check cannot catch this."""

    stored = np.zeros(7200, dtype=np.uint8)
    stored[0:2520] = 1
    stored[3690:6210] = 1
    new = snapshot_codes()
    assert int((stored == 1).sum()) == int((new == 1).sum()) == 5040
    # ... yet they select different days, 360 of which are test here.
    leaked = np.flatnonzero((stored == 1) & (new == 3))
    assert leaked.size == 360
    assert leaked.min() == 5850 and leaked.max() == 6209


def test_train_only_climatology_ignores_the_codes_it_is_handed(monkeypatch) -> None:
    captured = {}

    def fake(state, codes, wet, *, chunk_days=60):
        captured["codes"] = np.asarray(codes).copy()
        return np.zeros((46, 2, 2), np.float32), {}, int((np.asarray(codes) == 1).sum())

    monkeypatch.setattr(
        "bire_repro.af_model_c_bire_aligned_chronological_s0_figures._FROZEN_CLIMATOLOGY",
        fake,
    )
    stored = np.zeros(7200, dtype=np.uint8)
    stored[0:2520] = 1
    stored[3690:6210] = 1
    _train_only_s0_climatology(None, stored, np.ones((2, 2), bool))
    assert np.array_equal(captured["codes"], snapshot_codes())


def test_binding_installs_and_restores_both_modules() -> None:
    before = {
        n: getattr(aligned_figures, n)
        for n in ("VERSION", "CONTRACT_STATUS", "_readme", "CHECKPOINT_STEPS",
                  "STAGE_NAMES", "_selected_stepper", "_ACTIVE_STAGE")
    }
    frozen_before = frozen_figures._s0_training_climatology
    with _ChronologicalBinding(CONTRACT):
        assert aligned_figures.STAGE_NAMES == ("selected",)
        assert aligned_figures.VERSION == VERSION
        assert frozen_figures._s0_training_climatology is _train_only_s0_climatology
    for name, value in before.items():
        assert getattr(aligned_figures, name) == value
    assert frozen_figures._s0_training_climatology is frozen_before


def test_the_fifteen_starts_are_held_out_under_this_split() -> None:
    raw = json.loads(CONTRACT.read_text())
    codes = snapshot_codes()
    starts = np.asarray(EXPECTED_STARTS)
    assert np.all(codes[starts] == 3)
    assert tuple(raw["protocol"]["start_draw_order"]) == EXPECTED_STARTS
    assert "6660_7199" in raw["comparison"]["valid_comparison_block"]
    assert "5850_6209" in raw["comparison"]["invalid_comparison_block"]


def test_output_is_a_new_folder() -> None:
    raw = json.loads(CONTRACT.read_text())
    partner = json.loads(PARTNER.read_text())
    output = raw["stages"]["selected"]["output"]
    assert "chronological" in output["project"] and "chronological" in output["scratch"]
    assert output["project"] != partner["stages"]["selected"]["output"]["project"]


def test_readme_names_the_protocol_change() -> None:
    with _ChronologicalBinding(CONTRACT):
        with aligned_figures._FigureBinding("selected"):
            text = _readme({"report_content_sha256": "0" * 64})
    assert "chronological" in text.lower()
    assert "23.2%" in text
    assert "0--5039" in text


def test_rejects_more_than_one_stage(tmp_path: Path) -> None:
    raw = json.loads(CONTRACT.read_text())
    raw["stage_order"] = ["pretrained", "finetuned"]
    written = tmp_path / "c.json"
    written.write_text(json.dumps(raw))
    with pytest.raises(ChronologicalS0FigureError):
        selected_steps(written)


def test_launcher_invokes_this_arms_module_and_contract() -> None:
    text = SBATCH.read_text()
    assert text.count("bire_repro.af_model_c_bire_aligned_chronological_s0_figures") == 2
    assert "loss_recovery" not in text
    assert CONTRACT.name in text
