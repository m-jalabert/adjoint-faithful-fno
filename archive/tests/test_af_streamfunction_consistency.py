from __future__ import annotations

import numpy as np
import pytest

from bire_repro.af_forward_complete import _barotropic_streamfunction
from bire_repro.af_streamfunction_consistency import (
    depth_integrated_velocity,
    transport_divergence_volume,
    u_derived_streamfunction,
    uv_streamfunction_metrics,
)
from bire_repro.af_tutorial_analysis import EARTH_RADIUS_M


def test_centered_project_operator_matches_centered_face_integral() -> None:
    rng = np.random.default_rng(20260728)
    raw_u = rng.normal(scale=1.0e-3, size=(15, 62, 62)).astype(
        np.float32
    )
    centered_u = 0.5 * (raw_u + np.roll(raw_u, -1, axis=-1))
    states = np.zeros((1, 46, 62, 62), dtype=np.float32)
    states[0, :15] = centered_u
    wet = np.ones((62, 62), dtype=bool)
    project = _barotropic_streamfunction(states, wet)[0]

    dy = EARTH_RADIUS_M * np.deg2rad(1.0)
    raw_face = u_derived_streamfunction(
        depth_integrated_velocity(raw_u),
        np.full((62, 62), dy),
        center_x=True,
    )
    np.testing.assert_allclose(project, raw_face, rtol=2.0e-7, atol=2.0e-7)


def test_uv_metrics_remove_only_the_additive_gauge() -> None:
    rng = np.random.default_rng(7)
    psi_u = rng.normal(size=(62, 62))
    psi_v = psi_u - 4.25
    wet = np.ones((62, 62), dtype=bool)
    result = uv_streamfunction_metrics(psi_u, psi_v, wet)
    assert result["gauge_offset_sv"] == pytest.approx(4.25)
    assert result["rmse_sv"] == pytest.approx(0.0, abs=1.0e-14)
    assert result["relative_rmse"] == pytest.approx(0.0, abs=1.0e-14)
    assert result["correlation"] == pytest.approx(1.0)
    assert result["amplitude_ratio_v_to_u"] == pytest.approx(1.0)


def test_c_grid_transport_from_one_potential_is_divergence_free() -> None:
    rng = np.random.default_rng(14)
    potential = rng.normal(size=(63, 63))
    u_volume = -(
        potential[1:, :62] - potential[:62, :62]
    )
    v_volume = potential[:62, 1:] - potential[:62, :62]
    ones = np.ones((62, 62))
    divergence = transport_divergence_volume(
        u_volume,
        v_volume,
        ones,
        ones,
    )
    np.testing.assert_allclose(divergence[:-1, :-1], 0.0, atol=1.0e-14)
