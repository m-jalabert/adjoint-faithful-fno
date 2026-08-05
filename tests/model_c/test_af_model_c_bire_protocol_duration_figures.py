"""Tests for the duration arm's S0 figure suite.

The package's whole value is that it is comparable with the parent figure
package, so most of these assert that the evaluation protocol is byte-identical
to the parent's and only the checkpoint differs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bire_repro import af_model_c_bire_protocol_figures as suite
from bire_repro.af_model_c_bire_protocol_duration_figures import (
    COMPARATOR_STEP,
    CONTRACT_STATUS,
    MEMBER_COUNT,
    PARENT_BINDINGS,
    SELECTED_STEP,
    VERSION,
    BireProtocolDurationFigureError,
    _SuiteBinding,
    _readme,
    declared_inference_starts,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config/model_c_bire_protocol_duration_s0_figures_v1.json"
PARENT = ROOT / "config/model_c_bire_protocol_s0_figures_v1.json"
SBATCH = ROOT / "slurm/models/c/af_model_c_bire_protocol_duration_figures.sbatch"

pytestmark = pytest.mark.skipif(
    not CONTRACT.is_file(), reason="the duration figure contract is absent"
)


def test_the_evaluation_protocol_is_identical_to_the_parents() -> None:
    """Only the checkpoint may differ, or the packages are not comparable."""

    mine = json.loads(CONTRACT.read_text())
    parent = json.loads(PARENT.read_text())
    for key in ("member_count", "maximum_lead_days", "prediction_interval_days",
                "start_draw_order", "figure_names", "rmse_fields", "acc_fields",
                "regimes", "primary_regime", "inference_set",
                "figure3_lead_days", "figure7_lead_days"):
        assert mine["protocol"][key] == parent["protocol"][key], key
    assert mine["baselines"] == parent["baselines"]
    assert mine["truth"] == parent["truth"]
    assert mine["dataset"] == parent["dataset"]
    assert mine["artifacts"]["dataset_metadata"] == parent["artifacts"]["dataset_metadata"]
    assert mine["comparability"]["byte_comparable_with_the_parent_figure_package"] is True


def test_it_scores_the_15360_step_checkpoint_against_its_own_7680() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    assert SELECTED_STEP == 15360 and COMPARATOR_STEP == 7680
    assert int(contract["selected_model"]["optimizer_step"]) == SELECTED_STEP
    assert int(contract["figure6"]["comparator_optimizer_step"]) == COMPARATOR_STEP
    assert contract["figure6"]["literal_pretrain_finetune_pair"] is False
    # The comparator must be this run's checkpoint, not the parent arm's.
    assert "bire_protocol_duration_v1" in contract["artifacts"]["comparator_checkpoint"]["path"]
    assert "bire_protocol_duration_v1" in contract["artifacts"]["selected_checkpoint"]["path"]
    assert "not_the_parents_checkpoint" in contract["figure6"]


def test_outputs_do_not_collide_with_the_parent_package() -> None:
    mine = json.loads(CONTRACT.read_text())
    parent = json.loads(PARENT.read_text())
    for key in ("project_root", "scratch_root"):
        assert mine["output"][key] != parent["output"][key]
        assert "duration" in mine["output"][key]


def test_starts_are_the_parents_and_admit_2000_days_of_truth() -> None:
    starts = declared_inference_starts()
    assert starts.size == MEMBER_COUNT == 15
    assert np.array_equal(starts, suite.declared_inference_starts())
    assert starts.min() >= 6200 and int(starts.max()) + 2000 < 9000
    assert np.array_equal(starts, np.sort(starts))


def test_a_changed_comparator_is_rejected() -> None:
    import tempfile

    contract = json.loads(CONTRACT.read_text())
    contract["figure6"]["comparator_optimizer_step"] = 11520
    with tempfile.TemporaryDirectory(dir=ROOT / "config") as directory:
        path = Path(directory) / "tampered.json"
        path.write_text(json.dumps(contract))
        with pytest.raises(BireProtocolDurationFigureError):
            load_contract(path, verify_sources=False)


def test_binding_swaps_and_restores_the_suite_globals() -> None:
    before = {name: getattr(suite, name) for name in PARENT_BINDINGS}
    with _SuiteBinding():
        assert suite.VERSION == VERSION
        assert suite.COMPARATOR_STEP == COMPARATOR_STEP
        assert suite.CONTRACT_STATUS == CONTRACT_STATUS
        assert suite.load_contract is load_contract
    for name, value in before.items():
        assert getattr(suite, name) == value, f"{name} was not restored"


def test_binding_restores_on_exception() -> None:
    before = {name: getattr(suite, name) for name in PARENT_BINDINGS}
    with pytest.raises(RuntimeError):
        with _SuiteBinding():
            raise RuntimeError("boom")
    for name, value in before.items():
        assert getattr(suite, name) == value


def test_readme_renders_and_names_this_arm_not_the_parent() -> None:
    text = _readme("S0", {"selected_optimizer_step": SELECTED_STEP, "tau0_n_m2": 0.1,
                          "report_content_sha256": "0" * 64})
    assert "15,360" in text and "7,680" in text
    assert "6263--6979" in text or ("6263" in text and "6979" in text)
    for stale in ("5039", "5130", "6389", "6480", "held test block", "Pooled v3"):
        assert stale not in text
    flat = " ".join(text.split())          # the phrase wraps across lines
    assert "not the parent's checkpoint" in flat


def test_launcher_invokes_its_own_module_and_contract() -> None:
    text = SBATCH.read_text()
    invoked = {
        line.split("-m", 1)[1].strip().split()[0]
        for line in text.splitlines()
        if " -m " in f" {line} " and "bire_repro." in line
    }
    assert invoked == {"bire_repro.af_model_c_bire_protocol_duration_figures"}
    assert "model_c_bire_protocol_duration_s0_figures_v1.json" in text
