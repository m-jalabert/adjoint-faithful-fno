"""Tests for the production streamfunction-anomaly package."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oceanfno import anomaly, plots
from oceanfno.dataset import TRAIN_RANGE

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_production_1in_1out_spectralnorm_v1_s0_anomaly_v1.json"
SBATCH = ROOT / "slurm/models/c/anomaly_production_1in_1out_spectralnorm_v1.sbatch"


def test_anomaly_reads_sealed_figure_arrays_and_no_weights() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["version"] == anomaly.VERSION
    assert contract["protocol"]["reads_model_weights"] is False
    assert contract["protocol"]["rolls_nothing_out"] is True
    assert contract["modifies_published_figures"] is False
    assert contract["adds_only"] is True


def test_anomaly_is_bound_to_the_production_figure_package() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert anomaly.FIGURE_PACKAGE_VERSION == "model_c_production_1in_1out_spectralnorm_v1_s0_figures_v1"
    for key in ("figure_package_contract", "figure_package_arrays"):
        assert "model_c_production_1in_1out_spectralnorm_v1_s0_figures_v1" in (
            contract["artifacts"][key]["path"]
        )


def test_anomaly_contract_digests_are_pending_or_sealed() -> None:
    # ``anomaly finalize`` stamps these after the figure package is published.
    contract = json.loads(CONTRACT.read_text())
    for key in anomaly.SEALED_KEYS:
        digest = contract["artifacts"][key]["sha256"]
        assert isinstance(digest, str)
        assert digest == anomaly.PENDING or len(digest) == 64
    # All four are deferred, the figure contract included: `figures finalize`
    # rewrites that file as a declared step, so its bytes are not knowable here.
    assert set(anomaly.SEALED_KEYS) == {path[-2] for path in anomaly.PENDING_PATHS}


def test_an_unfinalized_contract_refuses_to_load() -> None:
    contract = json.loads(CONTRACT.read_text())
    if anomaly.unfilled_fields(contract):
        with pytest.raises(anomaly.AnomalyContractError):
            anomaly.load_contract(CONTRACT, verify_sources=False)


def test_the_reference_is_the_mitgcm_training_mean_removed_from_both_sides() -> None:
    contract = json.loads(CONTRACT.read_text())
    reference = contract["reference"]
    assert reference["source"] == "mitgcm"
    assert reference["days"] == list(TRAIN_RANGE) == [0, 6000]
    assert reference["subtracted_from"] == "both_truth_and_prediction"
    assert reference["model_own_mean_used"] is False
    assert reference["is_two_dimensional_field"] is True
    assert reference["not_a_scalar_spatial_mean"] is True


def test_the_anomaly_protocol_lead_days_match_the_figure_suite() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["protocol"]["figure3_lead_days"] == list(plots.FIGURE_3_LEADS)
    assert contract["protocol"]["figure7_lead_days"] == list(plots.FIGURE_7_LEADS)
    assert contract["protocol"]["figure3_lead_days"] == [0, 10, 20, 30, 40]
    assert contract["protocol"]["figure7_lead_days"] == [60, 2000]
    assert contract["protocol"]["member"] == 0


def test_the_three_declared_structure_diagnostics_are_computed() -> None:
    contract = json.loads(CONTRACT.read_text())
    declared = contract["protocol"]["day2000_structure_diagnostics"]
    assert declared == list(anomaly._EXPECTED_DIAGNOSTICS)
    wet = np.zeros((16, 16), dtype=bool)
    wet[1:15, 1:15] = True
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(16, 16))
    truth[~wet] = 0.0
    summary = anomaly.day2000_structure_summary(truth, 0.5 * truth, wet)
    assert set(summary["normalized_first_difference_rms"]) == {
        "truth_meridional",
        "model_meridional",
        "truth_zonal",
        "model_zonal",
    }
    boundary = summary["western_first_4_wet_cells"]
    # Halving the field halves the boundary RMS but leaves its shape ratios alone.
    assert boundary["model_to_truth_rms_ratio"] == pytest.approx(0.5)
    assert boundary["model_boundary_to_interior_rms_ratio"] == pytest.approx(
        boundary["truth_boundary_to_interior_rms_ratio"]
    )


def test_the_anomaly_rms_ratio_detects_damped_and_manufactured_transients() -> None:
    wet = np.ones((8, 8), dtype=bool)
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(2, 8, 8))
    damped = anomaly.variability_summary(truth, 0.25 * truth, (60, 2000), wet)
    inflated = anomaly.variability_summary(truth, 3.0 * truth, (60, 2000), wet)
    assert damped["2000"]["anomaly_rms_ratio"] == pytest.approx(0.25)
    assert inflated["2000"]["anomaly_rms_ratio"] == pytest.approx(3.0)
    assert damped["60"]["anomaly_error_rms_sv"] > 0.0


def test_outputs_are_written_under_the_production_roots() -> None:
    output = json.loads(CONTRACT.read_text())["output"]
    assert output["project_root"].endswith("outputs/af_fno/C/" + anomaly.VERSION)
    assert output["scratch_root"].endswith("af_fno/models/C/" + anomaly.VERSION)
    assert output["overwrite"] is False
    assert tuple(output["required"]) == anomaly._EXPECTED_REQUIRED


def test_slurm_uses_the_production_anomaly_entrypoint() -> None:
    text = SBATCH.read_text()
    assert "-m oceanfno.anomaly finalize" in text
    assert "-m oceanfno.anomaly preflight" in text
    assert "-m oceanfno.anomaly run" in text
    assert "model_c_production_1in_1out_spectralnorm_v1_s0_anomaly_v1.json" in text
