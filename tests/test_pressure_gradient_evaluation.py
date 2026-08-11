"""Static wiring checks for pressure-gradient figures and anomaly packages."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "config/model_c_2in_1out_new_channels_pressure_gradient_s0_figures_v1.json"
ANOM = ROOT / "config/model_c_2in_1out_new_channels_pressure_gradient_s0_anomaly_v1.json"
FIG_SBATCH = ROOT / "slurm/models/c/figures_2in_1out_new_channels_pressure_gradient.sbatch"
ANOM_SBATCH = ROOT / "slurm/models/c/anomaly_2in_1out_new_channels_pressure_gradient.sbatch"


def test_figure_comparison_changes_only_training_loss() -> None:
    contract = json.loads(FIG.read_text())
    assert contract["selected_model"]["architecture"] == contract["comparator_model"]["architecture"]
    assert contract["selected_model"]["version"] == "model_c_2in_1out_new_channels_pressure_gradient_v1"
    assert contract["comparator_model"]["version"] == "model_c_2in_1out_new_channels_v1"
    assert contract["figure6"]["literal_pretrain_finetune_pair"] is True
    assert contract["protocol"]["maximum_lead_days"] == 2000
    assert contract["protocol"]["member_count"] == 15


def test_anomaly_reads_figure_arrays_and_no_weights() -> None:
    contract = json.loads(ANOM.read_text())
    assert contract["protocol"]["reads_model_weights"] is False
    assert contract["protocol"]["rolls_nothing_out"] is True
    assert contract["modifies_published_figures"] is False
    assert contract["adds_only"] is True
    assert contract["artifacts"]["figure_package_arrays"]["sha256"] == "PENDING_AFTER_FIGURES"


def test_jobs_call_dedicated_runners() -> None:
    assert "oceanfno.figures_pressure_gradient" in FIG_SBATCH.read_text()
    assert "oceanfno.anomaly_pressure_gradient" in ANOM_SBATCH.read_text()
