import numpy as np
import pytest

from bire_repro.af_tutorial_analysis import (
    DRF_M,
    HEAT_CAPACITY_CP,
    RHO_CONST,
    THETA_RELAX_SECONDS,
    barotropic_streamfunction,
    reconstruct_trelax,
)


def test_reconstruct_trelax_matches_core_linear_restoring_law() -> None:
    theta = np.full((62, 62), 20.0)
    target = np.full((62, 62), 19.0)
    wet = np.ones((62, 62), dtype=bool)
    result = reconstruct_trelax(theta, target, wet)
    expected = -DRF_M[0] * RHO_CONST * HEAT_CAPACITY_CP / THETA_RELAX_SECONDS
    assert np.all(result == pytest.approx(expected))


def test_barotropic_streamfunction_has_tutorial_sign_and_sv_units() -> None:
    u = np.zeros((15, 62, 62))
    u[:, :, :] = 1.0
    result = barotropic_streamfunction(u, 100_000.0)
    expected_first_row = -DRF_M.sum() * 100_000.0 / 1.0e6
    assert result.shape == (62, 62)
    assert np.all(result[0] == pytest.approx(expected_first_row))
    assert np.all(result[1] == pytest.approx(2 * expected_first_row))
