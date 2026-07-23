"""Numerical contracts for the complete frozen forward diagnostics."""

import numpy as np
import pytest

from bire_repro.af_forward_compare import _check_common_protocol
from bire_repro.af_forward_complete import (
    ALL_FIELDS,
    BIRE_FIELDS,
    _choose_starts,
    _member_acc,
    _member_rmse,
    _training_climatology,
    derived_fields,
    radial_spectrum,
)
from bire_repro.af_pressure import GRAVITY_M_S2, PHIHYD_LEVELS, T_REF_C, phihyd_from_theta_eta


def test_metric_field_registry_deduplicates_ssh() -> None:
    assert ALL_FIELDS.count("ssh") == 1
    assert tuple(name for name in BIRE_FIELDS if name.startswith("phihyd_")) == tuple(
        PHIHYD_LEVELS
    )


def test_member_metrics_flatten_channels_and_wet_cells_per_member() -> None:
    wet = np.array([[True, True], [True, False]])
    truth = np.zeros((2, 2, 2, 2), dtype=np.float32)
    prediction = truth.copy()
    prediction[0, :, wet] = 2.0
    prediction[1, :, wet] = 3.0
    assert np.allclose(_member_rmse(prediction, truth, wet), (2.0, 3.0))

    climatology = np.zeros_like(truth)
    truth[:, :, wet] = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
    assert np.allclose(_member_acc(truth, truth, climatology, wet), 1.0)


def test_derived_fields_and_spectrum_have_declared_shapes() -> None:
    wet = np.ones((6, 6), dtype=bool)
    states = np.zeros((2, 46, 6, 6), dtype=np.float32)
    states[:, 0] = 3.0
    states[:, 15] = 4.0
    states[:, 30] = np.arange(36, dtype=np.float32).reshape(6, 6)
    fields = derived_fields(states, wet)
    assert set(fields) == {
        "surface_speed",
        "sst",
        "ssh",
        "phihyd_surface",
        "phihyd_mid",
        "phihyd_bottom",
        "streamfunction",
    }
    assert np.all(fields["surface_speed"] == 5.0)
    modes, power = radial_spectrum(fields["sst"], wet)
    assert modes.ndim == 1
    assert power.shape == (2, modes.size)
    assert np.all(np.isfinite(power))


def test_phihyd_reference_column_is_free_surface_geopotential() -> None:
    theta = np.broadcast_to(T_REF_C[None, :, None, None], (2, 15, 3, 4))
    eta = np.asarray([0.25, -0.5], dtype=np.float32)[:, None, None]
    eta = np.broadcast_to(eta, (2, 3, 4))
    wet = np.ones((3, 4), dtype=bool)
    result = phihyd_from_theta_eta(theta, eta, wet)
    assert np.allclose(result, GRAVITY_M_S2 * eta[:, None])


def test_phihyd_masks_land_and_checks_shapes() -> None:
    theta = np.broadcast_to(T_REF_C[:, None, None], (15, 2, 2))
    eta = np.zeros((2, 2))
    wet = np.asarray([[True, False], [True, True]])
    result = phihyd_from_theta_eta(theta, eta, wet)
    assert result.shape == theta.shape
    assert np.all(result[:, ~wet] == 0.0)
    with pytest.raises(ValueError, match="15"):
        phihyd_from_theta_eta(theta[:14], eta, wet)


def test_training_climatology_keeps_regimes_separate() -> None:
    state = np.zeros((3, 4, 46, 3, 3), dtype=np.float32)
    for experiment in range(3):
        state[experiment] = float(experiment + 1)
    wet = np.ones((3, 3), dtype=bool)
    means, derived, count = _training_climatology(
        state, np.array([1, 1, 0, 0], dtype=np.uint8), wet, chunk=1
    )
    assert count == 2
    assert np.allclose(means[:, 0, 0, 0], (1.0, 2.0, 3.0))
    assert np.allclose(derived["sst"][:, 0, 0], (1.0, 2.0, 3.0))


def test_one_year_starts_are_fixed_and_regime_major() -> None:
    pair_codes = np.zeros(500, dtype=np.uint8)
    pair_codes[10:490] = 3
    selected, starts = _choose_starts(pair_codes, 500)
    assert selected.size == 15
    assert len(starts) == 45
    assert starts[0] == (0, int(selected[0]))
    assert starts[15] == (1, int(selected[0]))


def test_comparison_rejects_different_ensemble_starts() -> None:
    first = (
        "A0",
        {"protocol": {"ensemble_starts": [[0, 10]], "rollout_days": 360}},
        {"lead_days": np.array([10, 20])},
    )
    second = (
        "A",
        {"protocol": {"ensemble_starts": [[0, 11]], "rollout_days": 360}},
        {"lead_days": np.array([10, 20])},
    )
    with pytest.raises(ValueError, match="common ensemble"):
        _check_common_protocol((first, second))
