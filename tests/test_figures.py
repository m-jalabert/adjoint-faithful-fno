"""Tests for the canonical retained continuity S0 figure package."""
from __future__ import annotations

import json
from pathlib import Path

import oceanfno.figures as figures

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_2in_1out_new_channels_pressure_gradient_continuity_s0_figures_v1.json"
SBATCH = ROOT / "slurm/models/c/figures_2in_1out_new_channels_pressure_gradient_continuity.sbatch"


def test_figure_comparison_changes_only_training_loss() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["version"] == figures.VERSION
    assert contract["selected_model"]["architecture"] == contract["comparator_model"]["architecture"]
    assert contract["selected_model"]["version"] == figures.TRAINING_VERSION
    assert contract["comparator_model"]["version"] == figures.COMPARATOR_VERSION
    assert contract["figure6"]["literal_pretrain_finetune_pair"] is True
    assert contract["protocol"]["maximum_lead_days"] == 2000
    assert contract["protocol"]["member_count"] == 15


def test_comparator_is_the_retained_pressure_gradient_parent() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["comparator_model"]["optimizer_step"] == figures.COMPARATOR_STEP
    assert "model_c_2in_1out_new_channels_pressure_gradient_v1" in (
        contract["artifacts"]["comparator_checkpoint"]["path"]
    )


def test_figure_contract_digests_are_pending_or_sealed() -> None:
    # ``figures finalize`` stamps these after training; before that they carry
    # the declared sentinel and never a partially written digest.
    contract = json.loads(CONTRACT.read_text())
    step = contract["selected_model"]["optimizer_step"]
    assert step == figures.PENDING or int(step) == 3840
    for key in ("selected_checkpoint", "selected_normalization", "selected_report"):
        digest = contract["artifacts"][key]["sha256"]
        assert isinstance(digest, str)
        assert digest == figures.PENDING or len(digest) == 64


def test_declared_inference_protocol_is_unchanged() -> None:
    starts = figures.declared_inference_starts()
    assert len(starts) == 15
    assert int(starts.min()) >= 6200
    assert int(starts.max()) < 7000


def test_slurm_uses_canonical_figure_entrypoint() -> None:
    text = SBATCH.read_text()
    assert "-m oceanfno.figures" in text
    assert "oceanfno.figures_pressure_gradient" not in text
    assert "model_c_2in_1out_new_channels_pressure_gradient_continuity_s0_figures_v1.json" in text
