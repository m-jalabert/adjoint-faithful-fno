from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bire_repro.af_model_c_slow_field_bias_projection import (
    VARIANTS,
    apply_increment_projection,
    first_efold_time,
    load_bias_projection_contract,
    pattern_metrics,
    wet_area_weights,
)


def test_wet_area_weights_are_normalized_and_masked() -> None:
    latitude = np.asarray([[0.0, 0.0], [60.0, 60.0]])
    wet = np.asarray([[True, False], [True, True]])
    weights = wet_area_weights(latitude, wet)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0, 1] == 0.0
    assert weights[0, 0] == pytest.approx(2.0 * weights[1, 0])


def test_pattern_metrics_distinguish_fixed_and_scaled_explanation() -> None:
    weights = np.full((2, 2), 0.25)
    result = pattern_metrics(
        np.ones((2, 2)),
        2.0 * np.ones((2, 2)),
        weights,
    )
    assert result["weighted_cosine"] == pytest.approx(1.0)
    assert result["fixed_amplitude_explained_energy_fraction"] == pytest.approx(
        0.75
    )
    assert result[
        "optimal_amplitude_explained_energy_fraction"
    ] == pytest.approx(1.0)
    assert result["optimal_amplitude_scale"] == pytest.approx(2.0)


def test_first_efold_time_interpolates_and_reports_lower_bound() -> None:
    leads = np.asarray([0.0, 10.0, 20.0])
    assert first_efold_time(
        leads,
        np.asarray([1.0, 0.6, 0.2]),
    ) == pytest.approx(15.8030139707)
    assert first_efold_time(
        leads,
        np.asarray([1.0, 0.8, 0.6]),
    ) is None


def test_increment_projection_enforces_declared_means() -> None:
    torch = pytest.importorskip("torch")
    increment = torch.zeros((2, 46, 2, 2), dtype=torch.float32)
    increment[:, 30:45] = 3.0
    increment[:, 45] = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[4.0, 3.0], [2.0, 1.0]]]
    )
    weights = torch.full((2, 2), 0.25)
    wet = torch.ones((2, 2), dtype=torch.bool)
    scale = torch.ones((1, 46, 1, 1))
    targets = torch.zeros((3, 16))
    targets[0, :15] = 0.25
    targets[1, :15] = -0.5
    bias = torch.zeros((3, 16, 2, 2))
    corrected = apply_increment_projection(
        increment,
        np.asarray([0, 1]),
        variant="conservation_means",
        state_scale=scale,
        area_weights=weights,
        wet=wet,
        truth_mean_tendency=targets,
        bias_field=bias,
    )
    physical = corrected * scale
    assert torch.max(
        torch.abs(torch.sum(physical[:, 45] * weights, dim=(-2, -1)))
    ).item() < 1.0e-6
    means = torch.sum(
        physical[:, 30:45] * weights[None, None],
        dim=(-2, -1),
    )
    assert torch.allclose(means[0], torch.full((15,), 0.25))
    assert torch.allclose(means[1], torch.full((15,), -0.5))


def test_contract_loader_rejects_variant_change(tmp_path: Path) -> None:
    contract = {
        "version": "model_c_slow_field_bias_projection_v1",
        "contract_status": (
            "frozen_after_truncated_unroll_rejection_before_bias_or_projection_metrics"
        ),
        "rollout_records": {
            "lead_days": list(range(10, 91, 10)),
            "starts_per_training_block": 90,
            "expected_training_blocks": 2,
            "records_total": 540,
            "selection": (
                "same_evenly_spaced_complete_split1_starts_as_job_291102"
            ),
        },
        "posthoc_projection": {
            "variants": list(VARIANTS[:-1]),
            "ssh_target": (
                "zero_cosine_latitude_weighted_wet_area_mean_increment"
            ),
            "temperature_target": (
                "per_regime_per_level_training_truth_mean_increment"
            ),
            "checkpoint_weights_changed": False,
        },
        "eof_analysis": {
            "fields": ["sst", "ssh"],
            "modes": 5,
            "oversampling": 5,
            "seed": 20260728,
        },
        "predictability": {
            "fit_lag_days": 10,
            "maximum_decorrelation_lag_days": 720,
            "alpha_bounds": [0.0, 1.0],
            "evaluation_records": "same_540_split1_rollouts",
        },
        "read_contract": {
            "training_pair_code": 1,
            "training_state_read": True,
            "prior_report_metadata_read": True,
            "validation_state_read": False,
            "inference_state_read": False,
            "intermediate_wind_read": False,
            "response_read": False,
            "adjoint_read": False,
        },
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="projection contract changed"):
        load_bias_projection_contract(path, verify_sources=False)
