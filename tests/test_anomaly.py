"""Contract and metric tests for the canonical Y32 S0 anomaly package.

This package must change only the model values it reads.  The reference field,
plot definitions, member, leads, and diagnostic formulas remain the retained
local24 definitions, while every input artifact must belong to the sealed Y32
figure package.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "config/model_c_bire_protocol_rollout_ft_local24_y32_s0_anomaly_v1.json"
)
COMPARED = (
    ROOT / "config/model_c_bire_protocol_rollout_ft_local24_s0_anomaly_v1.json"
)
FIGURE_CONTRACT = (
    ROOT
    / "config/model_c_bire_protocol_rollout_ft_local24_y32_s0_figures_v1.json"
)
FIGURE_PACKAGE = (
    ROOT
    / "outputs/af_fno/C/bire_protocol_rollout_ft_local24_y32_s0_figures_v1/S0"
)
FIGURE_REPORT = FIGURE_PACKAGE / "model_c_bire_s0_figures_report.json"
FIGURE_ARRAYS = FIGURE_PACKAGE / "model_c_bire_s0_figures_arrays.npz"
FIGURE_MANIFEST = FIGURE_PACKAGE / "manifest.json"
MODULE = ROOT / "src/oceanfno/anomaly.py"
SBATCH = ROOT / "slurm/models/c/anomaly.sbatch"
PACKAGE = (
    ROOT
    / "outputs/af_fno/C/bire_protocol_rollout_ft_local24_y32_s0_anomaly_v1/S0"
)

VERSION = "model_c_bire_protocol_rollout_ft_local24_y32_s0_anomaly_v1"
CONTRACT_STATUS = (
    "frozen_after_the_rollout_ft_local24_y32_figure_package_and_before_any_"
    "anomaly_metric"
)
FIGURE_VERSION = "model_c_bire_protocol_rollout_ft_local24_y32_s0_figures_v1"
PENDING = "PENDING_AFTER_FIGURES"

FIGURE_3_LEADS = (0, 10, 20, 30, 40)
FIGURE_7_LEADS = (60, 2000)
FIGURE_NAMES = (
    "model_c_bire_figure3a_streamfunction_anomaly_1deg_s0_dt10.png",
    "model_c_bire_figure7a_streamfunction_anomaly_day060_day2000_s0.png",
)
REFERENCE_FIGURE = "model_c_bire_reference_time_mean_streamfunction_s0.png"
PNG_NAMES = (*FIGURE_NAMES, REFERENCE_FIGURE)
REPORT_NAME = "model_c_bire_rollout_ft_anomaly_report.json"
ARRAYS_NAME = "model_c_bire_rollout_ft_anomaly_arrays.npz"

DIAGNOSTICS = (
    "anomaly_rms_ratio",
    "amplitude_normalized_meridional_and_zonal_first_difference_rms",
    "hann_directional_power_fraction_above_0p2_cycles_per_cell",
    "western_first_4_wet_cells_rms_and_boundary_to_interior_ratio",
)

PROJECT_ROOT = (
    "/home/mjalabert314/bire_james25_repro/outputs/af_fno/C/"
    "bire_protocol_rollout_ft_local24_y32_s0_anomaly_v1"
)
SCRATCH_ROOT = (
    "/bigscratch/mjalabert314/bire_james25_repro/af_fno/models/C/"
    "bire_protocol_rollout_ft_local24_y32_s0_anomaly_v1"
)

PROVENANCE = {
    "figure_package_contract": FIGURE_CONTRACT,
    "figure_package_report": FIGURE_REPORT,
    "figure_package_arrays": FIGURE_ARRAYS,
    "figure_package_manifest": FIGURE_MANIFEST,
}
PENDING_ARTIFACTS = (
    "figure_package_report",
    "figure_package_arrays",
    "figure_package_manifest",
)

EXPECTED_FIGURE7 = {
    "60": {
        "truth_anomaly_rms_sv": 0.24119470400991888,
        "model_anomaly_rms_sv": 0.2623884478417753,
        "anomaly_rms_ratio": 1.0878698556788577,
        "anomaly_error_rms_sv": 0.10967285763810991,
        "truth_anomaly_range_sv": (-5.053138732910156, 6.039175033569336),
        "model_anomaly_range_sv": (-4.805507659912109, 6.34630012512207),
    },
    "2000": {
        "truth_anomaly_rms_sv": 0.1944885857667538,
        "model_anomaly_rms_sv": 0.6737965126291088,
        "anomaly_rms_ratio": 3.4644527336795963,
        "anomaly_error_rms_sv": 0.6444222960271173,
        "truth_anomaly_range_sv": (-3.9899673461914062, 3.315977096557617),
        "model_anomaly_range_sv": (-10.034830093383789, 3.070181503981871),
    },
}
EXPECTED_STRUCTURE = {
    "normalized_first_difference_rms": {
        "truth_meridional": 1.2212329258247157,
        "model_meridional": 0.42742369204102065,
        "truth_zonal": 0.5526219434406737,
        "model_zonal": 0.7912927409799007,
    },
    "hann_directional_power_fraction_above_0p2_cycles_per_cell": {
        "truth_meridional": 0.02763470465687459,
        "model_meridional": 0.002304569765306466,
        "truth_zonal": 0.005084964649634374,
        "model_zonal": 0.07616236495715499,
    },
    "western_first_4_wet_cells": {
        "truth_rms_sv": 0.7435585690750866,
        "model_rms_sv": 1.5897058761923932,
        "model_to_truth_rms_ratio": 2.1379699492533994,
        "truth_boundary_to_interior_rms_ratio": 23.098211684067127,
        "model_boundary_to_interior_rms_ratio": 2.8741788594173783,
    },
}

requires_contract = pytest.mark.skipif(
    not CONTRACT.is_file(), reason="the Y32 anomaly contract is absent"
)
requires_module = pytest.mark.skipif(
    not MODULE.is_file(), reason="the canonical Y32 anomaly module is absent"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw() -> dict:
    return json.loads(CONTRACT.read_text())


def _pending() -> dict:
    contract = _raw()
    contract["artifacts"]["figure_package_contract"]["sha256"] = _sha256(
        FIGURE_CONTRACT
    )
    for key in PENDING_ARTIFACTS:
        contract["artifacts"][key]["sha256"] = PENDING
    return contract


def _filled() -> dict:
    contract = _pending()
    for key, path in PROVENANCE.items():
        contract["artifacts"][key]["sha256"] = _sha256(path)
    return contract


def _written(contract: dict, directory: Path) -> Path:
    path = directory / "y32_anomaly.json"
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return path


def _suite():
    return importlib.import_module("oceanfno.anomaly")


@requires_contract
def test_y32_uses_the_same_three_png_definitions_as_local24() -> None:
    mine = _raw()
    compared = json.loads(COMPARED.read_text())

    assert mine["protocol"] == compared["protocol"]
    assert mine["dataset"] == compared["dataset"]
    assert tuple(mine["protocol"]["figure3_lead_days"]) == FIGURE_3_LEADS
    assert tuple(mine["protocol"]["figure7_lead_days"]) == FIGURE_7_LEADS
    assert tuple(mine["protocol"]["figure_names"]) == FIGURE_NAMES
    assert tuple(mine["protocol"]["day2000_structure_diagnostics"]) == DIAGNOSTICS
    assert tuple(
        name for name in mine["output"]["required"] if name.endswith(".png")
    ) == PNG_NAMES
    assert mine["output"]["required"] == compared["output"]["required"]

    semantic_reference = (
        "source",
        "days",
        "definition",
        "is_two_dimensional_field",
        "not_a_scalar_spatial_mean",
        "model_own_mean_used",
        "subtracted_from",
        "regime",
    )
    for key in semantic_reference:
        assert mine["reference"][key] == compared["reference"][key], key


@requires_module
def test_canonical_y32_module_exposes_the_frozen_metric_and_plot_functions() -> None:
    suite = _suite()
    names = (
        "training_mean_streamfunction",
        "wet_rms",
        "variability_summary",
        "day2000_structure_summary",
        "_normalized_first_difference",
        "_directional_high_wavenumber_fraction",
        "_plot_anomaly_grid",
        "_plot_anomaly_long",
    )
    for name in names:
        assert callable(getattr(suite, name)), name
    assert suite.FIGURE_NAMES == FIGURE_NAMES
    assert suite.REPORT_NAME == REPORT_NAME
    assert suite.ARRAYS_NAME == ARRAYS_NAME


@requires_contract
def test_y32_reads_only_the_completed_y32_figure_package() -> None:
    contract = _filled()
    artifacts = contract["artifacts"]
    report = json.loads(FIGURE_REPORT.read_text())
    manifest = json.loads(FIGURE_MANIFEST.read_text())

    for key, path in PROVENANCE.items():
        assert Path(artifacts[key]["path"]) == path.resolve(), key
        assert artifacts[key]["sha256"] == _sha256(path), key
    assert report["version"] == manifest["version"] == FIGURE_VERSION
    assert Path(report["contract"]) == FIGURE_CONTRACT.resolve()
    assert report["contract_sha256"] == _sha256(FIGURE_CONTRACT)
    assert manifest["contract_sha256"] == report["contract_sha256"]
    assert report["arrays_sha256"] == _sha256(FIGURE_ARRAYS)
    assert manifest["artifacts"][FIGURE_ARRAYS.name]["sha256"] == _sha256(
        FIGURE_ARRAYS
    )
    assert manifest["artifacts"][FIGURE_REPORT.name]["sha256"] == _sha256(
        FIGURE_REPORT
    )
    assert contract["protocol"]["reads_model_weights"] is False
    assert contract["protocol"]["rolls_nothing_out"] is True
    assert contract["read_contract"]["promotes_no_checkpoint"] is True


@requires_contract
def test_y32_is_s0_only_and_has_exact_noncolliding_output_roots() -> None:
    mine = _raw()
    compared = json.loads(COMPARED.read_text())

    assert mine["protocol"]["primary_regime"] == "S0"
    assert mine["protocol"]["member"] == 0
    assert mine["reference"]["regime"] == "S0"
    assert mine["dataset"]["tau0_n_m2"] == {"S0": 0.1}
    assert mine["output"]["project_root"] == PROJECT_ROOT
    assert mine["output"]["scratch_root"] == SCRATCH_ROOT
    assert mine["output"]["overwrite"] is False
    assert mine["output"]["one_folder_per_regime"] is True
    for key in ("project_root", "scratch_root"):
        assert mine["output"][key] != compared["output"][key]
    assert mine["retained_on_the_total_field"]["measurement_source"] == (
        "bire_protocol_rollout_ft_local24_y32_acceptance_gate.json"
    )


@requires_contract
@requires_module
def test_a_filled_y32_anomaly_contract_loads_strictly(tmp_path) -> None:
    suite = _suite()
    contract, resolved, digest = suite.load_contract(
        _written(_filled(), tmp_path), verify_sources=False
    )
    assert contract["version"] == suite.VERSION == VERSION
    assert contract["contract_status"] == suite.CONTRACT_STATUS == CONTRACT_STATUS
    assert resolved.is_file()
    assert len(digest) == 64


@requires_contract
@requires_module
def test_pending_provenance_refuses_to_load_and_finalize_fills_it(tmp_path) -> None:
    suite = _suite()
    pending = _pending()
    for key in PENDING_ARTIFACTS:
        assert pending["artifacts"][key]["sha256"] == PENDING
    path = _written(pending, tmp_path)
    with pytest.raises(suite.BireY32AnomalyError):
        suite.load_contract(path, verify_sources=False)

    result = suite.finalize(path)
    assert result["status"] == "filled"
    written = json.loads(path.read_text())
    assert written == _filled()
    suite.load_contract(path, verify_sources=False)

    before = path.read_text()
    again = suite.finalize(path)
    assert again["status"] == "already_complete"
    assert path.read_text() == before


@requires_contract
@requires_module
def test_finalize_refuses_to_overwrite_bound_provenance(tmp_path) -> None:
    suite = _suite()
    contract = _pending()
    contract["artifacts"]["figure_package_arrays"]["sha256"] = "0" * 64
    with pytest.raises(suite.BireY32AnomalyError, match="refus|mismatch|changed"):
        suite.finalize(_written(contract, tmp_path))


@requires_contract
@requires_module
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda c: c["protocol"].update(primary_regime="S1"),
            id="regime_changed",
        ),
        pytest.param(
            lambda c: c["protocol"].update(member=1), id="member_changed"
        ),
        pytest.param(
            lambda c: c["protocol"].update(figure7_lead_days=[60, 1000]),
            id="lead_changed",
        ),
        pytest.param(
            lambda c: c["protocol"].update(figure_names=[FIGURE_NAMES[0]]),
            id="plot_removed",
        ),
        pytest.param(
            lambda c: c["protocol"].update(
                day2000_structure_diagnostics=list(DIAGNOSTICS[:-1])
            ),
            id="directional_metric_removed",
        ),
        pytest.param(
            lambda c: c["protocol"].update(reads_model_weights=True),
            id="weights_enabled",
        ),
        pytest.param(
            lambda c: c["reference"].update(days=[0, 7200]),
            id="reference_leaks_validation",
        ),
        pytest.param(
            lambda c: c["reference"].update(model_own_mean_used=True),
            id="model_mean_substituted",
        ),
        pytest.param(
            lambda c: c["reference"].update(subtracted_from="prediction_only"),
            id="one_sided_subtraction",
        ),
        pytest.param(
            lambda c: c["artifacts"]["figure_package_contract"].update(
                sha256="0" * 64
            ),
            id="figure_contract_changed",
        ),
        pytest.param(
            lambda c: c["artifacts"]["figure_package_report"].update(
                path="/tmp/not-the-y32-report.json"
            ),
            id="figure_report_replaced",
        ),
        pytest.param(
            lambda c: c["artifacts"]["figure_package_arrays"].update(
                path="/tmp/not-the-y32-arrays.npz"
            ),
            id="figure_arrays_replaced",
        ),
        pytest.param(
            lambda c: c["artifacts"]["figure_package_manifest"].update(
                sha256="0" * 64
            ),
            id="figure_manifest_changed",
        ),
        pytest.param(
            lambda c: c["artifacts"]["dataset_metadata"].update(sha256="0" * 64),
            id="dataset_changed",
        ),
        pytest.param(
            lambda c: c["output"].update(
                project_root=PROJECT_ROOT.replace("local24_y32", "local24")
            ),
            id="project_root_collides",
        ),
        pytest.param(
            lambda c: c["output"].update(scratch_root=SCRATCH_ROOT + "_other"),
            id="scratch_root_changed",
        ),
        pytest.param(
            lambda c: c["output"].update(overwrite=True),
            id="overwrite_enabled",
        ),
        pytest.param(
            lambda c: c.update(modifies_published_figures=True),
            id="replaces_figures",
        ),
        pytest.param(lambda c: c.update(adds_only=False), id="not_additive"),
    ],
)
def test_y32_anomaly_contract_rejects_science_provenance_and_output_tampering(
    mutate, tmp_path
) -> None:
    suite = _suite()
    contract = _filled()
    mutate(contract)
    with pytest.raises(suite.BireY32AnomalyError):
        suite.load_contract(_written(contract, tmp_path), verify_sources=False)


@requires_module
def test_metric_schema_and_direction_names_are_exact() -> None:
    suite = _suite()
    wet = np.ones((16, 16), dtype=bool)
    y, x = np.indices(wet.shape)
    truth = np.sin(2.0 * np.pi * 5.0 * y / wet.shape[0])
    model = 2.0 * truth + 0.25 * np.cos(2.0 * np.pi * 5.0 * x / wet.shape[1])

    variability = suite.variability_summary(
        truth[None], model[None], (2000,), wet
    )["2000"]
    assert set(variability) == {
        "truth_anomaly_rms_sv",
        "model_anomaly_rms_sv",
        "anomaly_rms_ratio",
        "anomaly_error_rms_sv",
        "truth_anomaly_range_sv",
        "model_anomaly_range_sv",
    }
    structure = suite.day2000_structure_summary(truth, model, wet)
    assert set(structure) == {"basis", *EXPECTED_STRUCTURE}
    for key, expected in EXPECTED_STRUCTURE.items():
        assert set(structure[key]) == set(expected), key
    assert structure["basis"] == (
        "member_0_day_2000_streamfunction_anomaly_about_the_S0_MITgcm_training_mean"
    )


@requires_module
def test_wet_rms_ignores_land() -> None:
    field = np.ones((4, 4))
    wet = np.zeros((4, 4), dtype=bool)
    wet[:2] = True
    field[2:] = 1000.0
    assert _suite().wet_rms(field, wet) == pytest.approx(1.0)


@requires_module
def test_anomaly_ratio_separates_damped_and_manufactured_variability() -> None:
    wet = np.ones((8, 8), dtype=bool)
    truth = np.random.default_rng(0).normal(0.0, 1.0, size=(2, 8, 8))

    damped = _suite().variability_summary(truth, truth * 0.4, (60, 2000), wet)
    manufactured = _suite().variability_summary(
        truth, truth * 3.0, (60, 2000), wet
    )
    faithful = _suite().variability_summary(truth, truth.copy(), (60, 2000), wet)

    assert damped["60"]["anomaly_rms_ratio"] == pytest.approx(0.4, abs=1e-6)
    assert manufactured["60"]["anomaly_rms_ratio"] == pytest.approx(
        3.0, abs=1e-6
    )
    assert faithful["2000"]["anomaly_rms_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert faithful["2000"]["anomaly_error_rms_sv"] == pytest.approx(
        0.0, abs=1e-12
    )


@requires_module
def test_anomaly_ratio_catches_variability_hidden_by_the_total_mean() -> None:
    wet = np.ones((16, 16), dtype=bool)
    rng = np.random.default_rng(1)
    mean_field = rng.normal(0.0, 12.0, size=(16, 16))
    truth_anomaly = rng.normal(0.0, 0.2, size=(1, 16, 16))
    model_anomaly = truth_anomaly * 7.0

    total_ratio = np.std((mean_field[None] + model_anomaly)[:, wet]) / np.std(
        (mean_field[None] + truth_anomaly)[:, wet]
    )
    anomaly_ratio = _suite().variability_summary(
        truth_anomaly, model_anomaly, (2000,), wet
    )["2000"]["anomaly_rms_ratio"]

    assert anomaly_ratio == pytest.approx(7.0, abs=1e-6)
    assert 0.80 <= float(total_ratio) <= 1.25


@pytest.mark.skipif(
    not (PACKAGE / REPORT_NAME).is_file(),
    reason="the Y32 anomaly package has not been produced",
)
def test_published_y32_anomaly_metrics_match_the_sealed_figure_arrays() -> None:
    report = json.loads((PACKAGE / REPORT_NAME).read_text())
    assert report["version"] == VERSION
    assert report["status"] == "complete"
    assert report["regime"] == "S0"
    assert tuple(report["figures"]) == FIGURE_NAMES

    for lead, expected in EXPECTED_FIGURE7.items():
        actual = report["variability"]["figure7"][lead]
        assert set(actual) == set(expected)
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value, rel=1e-7, abs=1e-9), (
                lead,
                key,
            )
    structure = report["day2000_structure"]
    for family, expected in EXPECTED_STRUCTURE.items():
        assert set(structure[family]) == set(expected)
        for key, value in expected.items():
            assert structure[family][key] == pytest.approx(
                value, rel=1e-7, abs=1e-9
            ), (family, key)


@pytest.mark.skipif(
    not (PACKAGE / ARRAYS_NAME).is_file(),
    reason="the Y32 anomaly package has not been produced",
)
def test_published_arrays_are_one_shared_field_subtraction() -> None:
    with np.load(PACKAGE / ARRAYS_NAME) as anomaly:
        reference = np.asarray(
            anomaly["reference_time_mean_streamfunction"], dtype=np.float64
        )
        truth = np.asarray(anomaly["figure7_truth"], dtype=np.float64)
        model = np.asarray(anomaly["figure7_model"], dtype=np.float64)
    with np.load(FIGURE_ARRAYS) as total:
        total_truth = np.asarray(
            total["figure7_truth_streamfunction"], dtype=np.float64
        )
        total_model = np.asarray(
            total["figure7_model_streamfunction"], dtype=np.float64
        )
    assert np.allclose(truth, total_truth - reference[None], atol=1e-4)
    assert np.allclose(model, total_model - reference[None], atol=1e-4)
    assert np.allclose(truth - model, total_truth - total_model, atol=1e-4)


@requires_module
def test_canonical_anomaly_has_no_y32_wrapper_context() -> None:
    suite = _suite()
    assert suite.VERSION == VERSION
    assert suite.CONTRACT_STATUS == CONTRACT_STATUS
    assert not hasattr(suite, "_y32_anomaly_runner")


@pytest.mark.skipif(not SBATCH.is_file(), reason="the Y32 anomaly launcher is absent")
def test_launcher_finalizes_preflights_then_runs_only_the_canonical_module() -> None:
    text = SBATCH.read_text()
    invoked = {
        line.split("-m", 1)[1].strip().split()[0]
        for line in text.splitlines()
        if " -m " in f" {line} " and "oceanfno." in line
    }
    assert invoked == {"oceanfno.anomaly"}
    assert "model_c_bire_protocol_rollout_ft_local24_y32_s0_anomaly_v1.json" in text
    assert text.index("finalize") < text.index("  preflight") < text.index("  run")
    assert "--gres=gpu" not in text
