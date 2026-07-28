import numpy as np

from bire_repro.af_streamfunction_consistency import (
    transport_divergence_volume,
)
from bire_repro.af_streamfunction_consistency_v2 import (
    collocated_metrics,
    u_corner_streamfunction,
    v_corner_streamfunction,
)


def test_collocated_corner_paths_match_discrete_streamfunction():
    y, x = np.mgrid[:62, :62]
    psi = (
        np.sin(np.pi * x / 61.0)
        * np.sin(np.pi * y / 61.0)
        * 1.0e6
    )
    u_volume = np.zeros((62, 62), dtype=np.float64)
    v_volume = np.zeros_like(u_volume)
    u_volume[:-1] = -(psi[1:] - psi[:-1])
    v_volume[:, :-1] = psi[:, 1:] - psi[:, :-1]
    u = u_volume / 2.0
    v = v_volume / 3.0
    derived_u = u_corner_streamfunction(u, np.full((62, 62), 2.0))
    derived_v = v_corner_streamfunction(v, np.full((62, 62), 3.0))
    expected = (psi - psi[0, 0]) / 1.0e6
    assert np.allclose(derived_u, expected)
    assert np.allclose(derived_v, expected)


def test_collocated_metrics_remove_only_constant_gauge():
    y, x = np.mgrid[:62, :62]
    psi = x + 2.0 * y
    mask = np.zeros((62, 62), dtype=bool)
    mask[1:61, 1:61] = True
    metrics = collocated_metrics(psi, psi - 7.0, mask)
    assert metrics["gauge_offset_sv"] == 7.0
    assert metrics["rmse_sv"] == 0.0
    assert metrics["relative_rmse"] == 0.0
    assert np.isclose(metrics["correlation"], 1.0)


def test_corner_path_difference_is_discrete_divergence_prefix():
    rng = np.random.default_rng(13)
    u = rng.normal(size=(62, 62))
    v = rng.normal(size=(62, 62))
    u[:, 0] = 0.0
    v[0, :] = 0.0
    dyg = np.ones((62, 62))
    dxg = np.ones((62, 62))
    psi_u = u_corner_streamfunction(u, dyg)
    psi_v = v_corner_streamfunction(v, dxg)
    path_difference = (psi_u - psi_v) * 1.0e6
    divergence = transport_divergence_volume(u, v, dyg, dxg)
    divergence_prefix = np.cumsum(
        np.cumsum(divergence[:-1, :-1], axis=0),
        axis=1,
    )
    assert np.allclose(path_difference[1:, 1:], -divergence_prefix)
