"""Tests for the streamfunction-anomaly companion plates.

The scientific content of this package is one subtraction, so the tests are
mostly about the ways that subtraction can be got wrong:

* subtracting a scalar instead of the full two-dimensional field;
* subtracting the model's own time mean, which would hide a bias in the mean
  circulation rather than reveal it;
* letting validation or inference days into the reference average;
* quietly replacing the published total-field plates instead of adding to them.

The variability statistic gets boundary tests of its own, because it is the
diagnostic the total-field standard-deviation ratio could not provide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oceanfno.dataset import TRAIN_RANGE, VALIDATION_RANGE
from oceanfno.plots import (
    ARRAYS_NAME as SUITE_ARRAYS_NAME,
)
from oceanfno.plots import (
    FIGURE_3_LEADS,
    FIGURE_7_LEADS,
    FIGURE_NAMES as PUBLISHED_FIGURE_NAMES,
)
from oceanfno.anomaly import (
    ANOMALY_LABEL,
    CONTRACT_STATUS,
    FIGURE_3A,
    FIGURE_7A,
    FIGURE_NAMES,
    REGIME,
    VERSION,
    BireProtocolRolloutFineTuneAnomalyError,
    load_contract,
    variability_summary,
    wet_rms,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_bire_protocol_rollout_ft_s0_anomaly_v2.json"
SBATCH = ROOT / "slurm/models/c/anomaly.sbatch"
PACKAGE = ROOT / "outputs/af_fno/C/bire_protocol_rollout_ft_s0_anomaly_v1/S0"

pytestmark = pytest.mark.skipif(
    not CONTRACT.is_file(), reason="the anomaly contract is absent"
)


def _written(contract: dict, directory: Path) -> Path:
    path = directory / "anomaly.json"
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return path


# --------------------------------------------------------------------------
# The reference field
# --------------------------------------------------------------------------


def test_the_reference_is_a_training_only_mitgcm_field() -> None:
    contract = json.loads(CONTRACT.read_text())
    reference = contract["reference"]
    assert reference["source"] == "mitgcm"
    assert tuple(reference["days"]) == TRAIN_RANGE == (0, 6000)
    assert reference["regime"] == REGIME == "S0"
    # No validation or inference day may enter the average.
    assert reference["days"][1] <= VALIDATION_RANGE[0]


def test_the_reference_is_the_full_two_dimensional_field() -> None:
    """Not a scalar spatial mean -- one value per grid point."""

    contract = json.loads(CONTRACT.read_text())
    assert contract["reference"]["is_two_dimensional_field"] is True
    assert contract["reference"]["not_a_scalar_spatial_mean"] is True


def test_the_same_reference_is_used_for_truth_and_prediction() -> None:
    contract = json.loads(CONTRACT.read_text())
    reference = contract["reference"]
    assert reference["subtracted_from"] == "both_truth_and_prediction"
    assert reference["model_own_mean_used"] is False
    assert "bias" in reference["model_own_mean_reason"]


@pytest.mark.skipif(
    not (PACKAGE / "model_c_bire_rollout_ft_anomaly_arrays.npz").is_file(),
    reason="the anomaly package has not been produced",
)
def test_the_published_arrays_really_carry_one_subtraction_of_one_field() -> None:
    """truth' - model' must equal truth - model: the reference cancels exactly."""

    from oceanfno.plots import ARRAYS_NAME

    with np.load(PACKAGE / "model_c_bire_rollout_ft_anomaly_arrays.npz") as mine:
        anomaly_truth = np.asarray(mine["figure7_truth"], dtype=np.float64)
        anomaly_model = np.asarray(mine["figure7_model"], dtype=np.float64)
        reference = np.asarray(mine["reference_time_mean_streamfunction"], dtype=np.float64)
        wet = np.asarray(mine["wet_mask"], dtype=bool)
    published = ROOT / "outputs/af_fno/C/bire_protocol_rollout_ft_s0_figures_v1/S0" / ARRAYS_NAME
    with np.load(published) as theirs:
        total_truth = np.asarray(theirs["figure7_truth_streamfunction"], dtype=np.float64)
        total_model = np.asarray(theirs["figure7_model_streamfunction"], dtype=np.float64)

    # The difference field is invariant: whatever was removed, it was the same.
    assert np.allclose(anomaly_truth - anomaly_model, total_truth - total_model, atol=1e-4)
    # And each side really is total minus the one reference field.
    assert np.allclose(anomaly_truth, total_truth - reference[None], atol=1e-4)
    assert np.allclose(anomaly_model, total_model - reference[None], atol=1e-4)
    # A scalar subtraction would leave the reference spatially constant.
    assert float(reference[wet].std()) > 1.0


# --------------------------------------------------------------------------
# It adds; it does not replace
# --------------------------------------------------------------------------


def test_the_new_plates_do_not_collide_with_the_published_ones() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["modifies_published_figures"] is False
    assert contract["adds_only"] is True
    assert set(FIGURE_NAMES).isdisjoint(set(PUBLISHED_FIGURE_NAMES))
    assert "anomaly" in FIGURE_3A and "anomaly" in FIGURE_7A
    for key in ("project_root", "scratch_root"):
        assert "anomaly" in contract["output"][key]


def test_the_total_field_diagnostics_are_explicitly_retained() -> None:
    """The -32.90 Sv minimum measures mean-circulation intensity, not variability."""

    contract = json.loads(CONTRACT.read_text())
    retained = contract["retained_on_the_total_field"]
    assert retained["acceptance_gate_unaffected"] is True
    assert retained["day2000_streamfunction_minimum_sv"] == pytest.approx(-32.8997, abs=1e-3)


def test_the_colour_bar_says_anomaly() -> None:
    assert "anomaly" in ANOMALY_LABEL
    assert "Sv" in ANOMALY_LABEL
    assert ANOMALY_LABEL != "Barotropic streamfunction (Sv)"


def test_the_package_reads_no_model_weights() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["protocol"]["reads_model_weights"] is False
    assert contract["protocol"]["rolls_nothing_out"] is True
    assert contract["read_contract"]["promotes_no_checkpoint"] is True
    # It consumes the sealed figure package rather than recomputing rollouts.
    assert Path(
        contract["artifacts"]["figure_package_arrays"]["path"]
    ).name == SUITE_ARRAYS_NAME


# --------------------------------------------------------------------------
# Contract loading
# --------------------------------------------------------------------------


def test_the_contract_loads() -> None:
    contract, resolved, digest = load_contract(CONTRACT, verify_sources=False)
    assert contract["version"] == VERSION
    assert contract["contract_status"] == CONTRACT_STATUS
    assert tuple(contract["protocol"]["figure3_lead_days"]) == FIGURE_3_LEADS
    assert tuple(contract["protocol"]["figure7_lead_days"]) == FIGURE_7_LEADS
    assert len(digest) == 64 and resolved.is_file()


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda c: c["reference"].update(model_own_mean_used=True), id="model_own_mean"
        ),
        pytest.param(
            lambda c: c["reference"].update(days=[0, 7200]), id="validation_days_in_reference"
        ),
        pytest.param(
            lambda c: c["reference"].update(is_two_dimensional_field=False), id="scalar_mean"
        ),
        pytest.param(
            lambda c: c["reference"].update(subtracted_from="prediction_only"), id="one_sided"
        ),
        pytest.param(
            lambda c: c["reference"].update(source="fno"), id="not_mitgcm"
        ),
        pytest.param(
            lambda c: c.update(modifies_published_figures=True), id="replaces_figures"
        ),
        pytest.param(
            lambda c: c["protocol"].update(figure7_lead_days=[60, 1000]), id="leads_moved"
        ),
    ],
)
def test_a_tampered_anomaly_contract_is_rejected(mutate, tmp_path) -> None:
    contract = json.loads(CONTRACT.read_text())
    mutate(contract)
    with pytest.raises(BireProtocolRolloutFineTuneAnomalyError):
        load_contract(_written(contract, tmp_path), verify_sources=False)


# --------------------------------------------------------------------------
# The variability statistic
# --------------------------------------------------------------------------


def test_wet_rms_ignores_land() -> None:
    field = np.ones((4, 4))
    wet = np.zeros((4, 4), dtype=bool)
    wet[:2] = True
    field[2:] = 1000.0
    assert wet_rms(field, wet) == pytest.approx(1.0)


def test_the_ratio_separates_damped_from_manufactured_variability() -> None:
    wet = np.ones((8, 8), dtype=bool)
    rng = np.random.default_rng(0)
    truth = rng.normal(0.0, 1.0, size=(2, 8, 8))

    damped = variability_summary(truth, truth * 0.4, (60, 2000), wet)
    assert damped["60"]["anomaly_rms_ratio"] == pytest.approx(0.4, abs=1e-6)

    manufactured = variability_summary(truth, truth * 3.0, (60, 2000), wet)
    assert manufactured["60"]["anomaly_rms_ratio"] == pytest.approx(3.0, abs=1e-6)

    faithful = variability_summary(truth, truth.copy(), (60, 2000), wet)
    assert faithful["2000"]["anomaly_rms_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert faithful["2000"]["anomaly_error_rms_sv"] == pytest.approx(0.0, abs=1e-12)


def test_the_ratio_catches_what_a_total_field_ratio_would_miss() -> None:
    """A large shared mean makes a 7x variability error look like a 1.0 ratio."""

    wet = np.ones((16, 16), dtype=bool)
    rng = np.random.default_rng(1)
    mean_field = rng.normal(0.0, 12.0, size=(16, 16))
    truth_anomaly = rng.normal(0.0, 0.2, size=(1, 16, 16))
    model_anomaly = truth_anomaly * 7.0

    total_ratio = (
        np.std((mean_field[None] + model_anomaly)[:, wet])
        / np.std((mean_field[None] + truth_anomaly)[:, wet])
    )
    anomaly_ratio = variability_summary(
        truth_anomaly, model_anomaly, (2000,), wet
    )["2000"]["anomaly_rms_ratio"]

    assert anomaly_ratio == pytest.approx(7.0, abs=1e-6)
    # The total-field ratio sits inside the gate's 0.80-1.25 window regardless.
    assert 0.80 <= float(total_ratio) <= 1.25


@pytest.mark.skipif(
    not (PACKAGE / "model_c_bire_rollout_ft_anomaly_report.json").is_file(),
    reason="the anomaly package has not been produced",
)
def test_the_published_report_records_both_lead_sets() -> None:
    report = json.loads((PACKAGE / "model_c_bire_rollout_ft_anomaly_report.json").read_text())
    assert report["status"] == "complete"
    assert report["reference"]["days_averaged"] == TRAIN_RANGE[1] - TRAIN_RANGE[0] == 6000
    assert report["reference"]["model_own_mean_used"] is False
    assert report["modifies_published_figures"] is False
    assert set(report["variability"]["figure3"]) == {str(v) for v in FIGURE_3_LEADS}
    assert set(report["variability"]["figure7"]) == {str(v) for v in FIGURE_7_LEADS}
    for block in report["variability"].values():
        for record in block.values():
            assert record["truth_anomaly_rms_sv"] > 0.0
            assert np.isfinite(record["anomaly_rms_ratio"])


def test_launcher_invokes_its_own_module_and_contract() -> None:
    text = SBATCH.read_text()
    invoked = {
        line.split("-m", 1)[1].strip().split()[0]
        for line in text.splitlines()
        if " -m " in f" {line} " and "oceanfno." in line
    }
    assert invoked == {"oceanfno.anomaly"}
    assert "model_c_bire_protocol_rollout_ft_s0_anomaly_v2.json" in text
    assert "--gres=gpu" not in text
