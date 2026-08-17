"""Tests for the production S0 figure package and its held-evaluation contract."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from oceanfno import figures, plots
from oceanfno.dataset import INFERENCE_RANGE, INFERENCE_START_RANGE
from oceanfno.model import ProductionArchitecture

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_production_1in_1out_spectralnorm_v1_s0_figures_v1.json"
SBATCH = ROOT / "slurm/models/c/figures_production_1in_1out_spectralnorm_v1.sbatch"


def test_the_figure_contract_names_the_production_model() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["version"] == figures.VERSION
    assert contract["selected_model"]["version"] == figures.TRAINING_VERSION
    assert contract["selected_model"]["from_scratch"] is True
    assert contract["selected_model"]["rollout_steps"] == 6
    assert contract["selected_model"]["architecture"] == ProductionArchitecture().to_dict()
    assert contract["protocol"]["maximum_lead_days"] == 2000
    assert contract["protocol"]["member_count"] == 15


def test_the_suite_carries_no_comparator_model() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert "comparator_model" not in contract
    assert "figure6" not in contract
    assert contract["protocol"]["comparator_model"] is None
    # The ACC plate reads a single model series; the comparator keys are gone.
    source = inspect.getsource(plots)
    assert "acc__prior__" not in source
    assert "acc__selected__" not in source
    assert "acc__model__" in source


def test_the_protocol_is_nested_validation_inference() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["protocol"]["inference_set"] == list(INFERENCE_RANGE)
    assert contract["protocol"]["start_window"] == list(INFERENCE_START_RANGE)
    assert contract["protocol"]["nesting"] == (
        "nested_validation_inference_protocol_no_independent_third_test_split"
    )


def test_figure_contract_digests_are_pending_or_sealed() -> None:
    # ``figures finalize`` stamps these after training; before that they carry
    # the declared sentinel and never a partially written digest.
    contract = json.loads(CONTRACT.read_text())
    step = contract["selected_model"]["optimizer_step"]
    assert step == figures.PENDING or int(step) in (1920, 3840, 5760, 7680)
    for key in ("selected_checkpoint", "selected_normalization", "selected_report"):
        digest = contract["artifacts"][key]["sha256"]
        assert isinstance(digest, str)
        assert digest == figures.PENDING or len(digest) == 64


def test_an_unfinalized_contract_refuses_to_load() -> None:
    contract = json.loads(CONTRACT.read_text())
    if figures.unfilled_fields(contract):
        with pytest.raises(figures.FigureContractError):
            figures.load_contract(CONTRACT, verify_sources=False)


def test_declared_inference_protocol_is_unchanged() -> None:
    starts = figures.declared_inference_starts()
    assert len(starts) == figures.MEMBER_COUNT == 15
    assert figures.START_SEED == 20260802
    assert int(starts.min()) >= 6200
    assert int(starts.max()) < 7000
    # Reusing the seed keeps the members comparable with the earlier experiments.
    assert [int(v) for v in starts] == [
        6263, 6293, 6331, 6389, 6579, 6593, 6598, 6601,
        6651, 6661, 6694, 6707, 6711, 6968, 6979,
    ]
    assert json.loads(CONTRACT.read_text())["protocol"]["start_draw_order"] == [
        int(v) for v in starts
    ]


def test_the_six_frozen_figures_are_declared() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["protocol"]["figure_names"] == list(plots.FIGURE_NAMES)
    assert len(plots.FIGURE_NAMES) == 6
    assert contract["protocol"]["figure3_lead_days"] == [0, 10, 20, 30, 40]
    assert contract["protocol"]["figure7_lead_days"] == [60, 2000]


def test_outputs_are_written_under_the_production_roots() -> None:
    output = json.loads(CONTRACT.read_text())["output"]
    assert output["project_root"].endswith("outputs/af_fno/C/" + figures.VERSION)
    assert output["scratch_root"].endswith("af_fno/models/C/" + figures.VERSION)
    assert output["overwrite"] is False


def test_the_long_rollout_gate_reads_the_published_arrays() -> None:
    wet = np.zeros((62, 62), dtype=bool)
    wet[1:61, 1:61] = True
    truth = np.zeros((2, 62, 62), dtype=np.float64)
    truth[1] = np.linspace(-20.0, 5.0, 62)[None, :]
    arrays = {
        "wet_mask": wet.astype(np.uint8),
        "figure7_truth_streamfunction": truth,
        "figure7_model_streamfunction": truth.copy(),
    }
    summary = {
        "all_selected_states_finite": True,
        "maximum_selected_normalized_abs": 3.0,
        "rmse": {
            field: {
                "model": {"day2000_mean": 1.0},
                "climatology": {"day2000_mean": 2.0},
            }
            for field in plots.RMSE_FIELDS
        },
    }
    gate = figures.long_rollout_gate(arrays, summary)
    # A perfect day-2,000 field passes every measurable condition.
    assert gate["long_rollout_conditions_pass"] is True
    assert gate["measured"]["day2000_spatial_std_ratio_to_truth"] == pytest.approx(1.0)
    # The day-2,000 collapse ratio is reported but deliberately not gated.
    assert set(gate["advisory_day2000_rmse_ratio_to_climatology"]) == set(
        plots.RMSE_FIELDS
    )
    assert "advisory_day2000_rmse_ratio_to_climatology" not in gate["conditions"]


def test_the_gate_rejects_a_collapsed_day_2000_field() -> None:
    wet = np.ones((8, 8), dtype=bool)
    truth = np.zeros((2, 8, 8), dtype=np.float64)
    truth[1] = np.linspace(-20.0, 5.0, 8)[None, :]
    arrays = {
        "wet_mask": wet.astype(np.uint8),
        "figure7_truth_streamfunction": truth,
        # A flat field: no spatial structure left at all.
        "figure7_model_streamfunction": np.zeros_like(truth),
    }
    summary = {
        "all_selected_states_finite": True,
        "maximum_selected_normalized_abs": 3.0,
        "rmse": {
            field: {
                "model": {"day2000_mean": 1.0},
                "climatology": {"day2000_mean": 1.0},
            }
            for field in plots.RMSE_FIELDS
        },
    }
    gate = figures.long_rollout_gate(arrays, summary)
    assert gate["conditions"]["day2000_spatial_std_ratio_in_range"] is False
    assert gate["long_rollout_conditions_pass"] is False


def test_slurm_uses_the_production_figure_entrypoint() -> None:
    text = SBATCH.read_text()
    assert "-m oceanfno.figures finalize" in text
    assert "-m oceanfno.figures preflight" in text
    assert "-m oceanfno.figures run" in text
    assert "model_c_production_1in_1out_spectralnorm_v1_s0_figures_v1.json" in text
