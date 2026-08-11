"""Tests for the canonical retained pressure-gradient S0 figure package."""
from __future__ import annotations

import json
from pathlib import Path

import oceanfno.figures as figures

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_2in_1out_new_channels_pressure_gradient_s0_figures_v1.json"
SBATCH = ROOT / "slurm/models/c/figures_2in_1out_new_channels_pressure_gradient.sbatch"


def test_figure_comparison_changes_only_training_loss() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["version"] == figures.VERSION
    assert contract["selected_model"]["architecture"] == contract["comparator_model"]["architecture"]
    assert contract["selected_model"]["version"] == figures.TRAINING_VERSION
    assert contract["comparator_model"]["version"] == figures.COMPARATOR_VERSION
    assert contract["figure6"]["literal_pretrain_finetune_pair"] is True
    assert contract["protocol"]["maximum_lead_days"] == 2000
    assert contract["protocol"]["member_count"] == 15


def test_figure_contract_is_finalized_to_the_accepted_checkpoint() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["selected_model"]["optimizer_step"] == 3840
    for key in ("selected_checkpoint", "selected_normalization", "selected_report"):
        digest = contract["artifacts"][key]["sha256"]
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert not digest.startswith("PENDING")


def test_declared_inference_protocol_is_unchanged() -> None:
    starts = figures.declared_inference_starts()
    assert len(starts) == 15
    assert int(starts.min()) >= 6200
    assert int(starts.max()) < 7000


def test_slurm_uses_canonical_figure_entrypoint() -> None:
    text = SBATCH.read_text()
    assert "-m oceanfno.figures" in text
    assert "oceanfno.figures_pressure_gradient" not in text
    assert "model_c_2in_1out_new_channels_pressure_gradient_s0_figures_v1.json" in text
