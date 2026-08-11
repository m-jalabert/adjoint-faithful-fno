"""Tests for the canonical retained pressure-gradient anomaly package."""
from __future__ import annotations

import json
from pathlib import Path

import oceanfno.anomaly as anomaly

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_2in_1out_new_channels_pressure_gradient_s0_anomaly_v1.json"
SBATCH = ROOT / "slurm/models/c/anomaly_2in_1out_new_channels_pressure_gradient.sbatch"


def test_anomaly_reads_sealed_figure_arrays_and_no_weights() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["version"] == anomaly.VERSION
    assert contract["protocol"]["reads_model_weights"] is False
    assert contract["protocol"]["rolls_nothing_out"] is True
    assert contract["modifies_published_figures"] is False
    assert contract["adds_only"] is True


def test_anomaly_contract_is_finalized() -> None:
    contract = json.loads(CONTRACT.read_text())
    for key in (
        "figure_package_contract",
        "figure_package_report",
        "figure_package_arrays",
        "figure_package_manifest",
    ):
        digest = contract["artifacts"][key]["sha256"]
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert not digest.startswith("PENDING")
    assert "pressure_gradient_s0_figures_v1" in contract["artifacts"]["figure_package_arrays"]["path"]


def test_anomaly_protocol_is_scientifically_unchanged() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["reference"]["source"] == "mitgcm"
    assert contract["reference"]["days"] == [0, 6000]
    assert contract["reference"]["subtracted_from"] == "both_truth_and_prediction"
    assert contract["protocol"]["figure3_lead_days"] == [0, 10, 20, 30, 40]
    assert contract["protocol"]["figure7_lead_days"] == [60, 2000]


def test_slurm_uses_canonical_anomaly_entrypoint() -> None:
    text = SBATCH.read_text()
    assert "-m oceanfno.anomaly" in text
    assert "oceanfno.anomaly_pressure_gradient" not in text
    assert "model_c_2in_1out_new_channels_pressure_gradient_s0_anomaly_v1.json" in text
