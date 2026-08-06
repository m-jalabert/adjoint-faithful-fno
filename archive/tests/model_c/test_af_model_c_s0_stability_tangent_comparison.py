from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bire_repro.af_model_c_s0_stability_tangent_comparison import (  # noqa: E402
    _band_projector,
    _spatial_stats,
    dominant_singular_and_tangent_gain,
)


class LinearStepper:
    def __init__(self, multiplier: float, wet: np.ndarray) -> None:
        self.multiplier = multiplier
        self.wet = wet

    def step(self, current: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        del static
        return self.multiplier * current


def test_band_projector_preserves_shape_and_reality() -> None:
    wet = np.ones((10, 10), dtype=bool)
    value = torch.randn(2, 3, 10, 10)
    projected = _band_projector(wet, (1, 3))(value)
    assert projected.shape == value.shape
    assert torch.isfinite(projected).all()
    assert torch.linalg.vector_norm(projected) > 0


def test_singular_and_finite_time_gain_for_scalar_map() -> None:
    wet = np.ones((8, 8), dtype=bool)
    stepper = LinearStepper(1.03, wet)
    current = torch.randn(1, 2, 8, 8)
    static = torch.empty(1, 0, 8, 8)
    result = dominant_singular_and_tangent_gain(
        stepper,  # type: ignore[arg-type]
        current,
        static,
        _band_projector(wet, None),
        seed=7,
    )
    assert np.isclose(result["dominant_one_step_singular_gain"], 1.03, rtol=1e-5)
    assert np.isclose(
        result["ten_call_tangent_geometric_gain_per_call"],
        1.03,
        rtol=1e-5,
    )
    assert np.isclose(result["ten_call_tangent_total_gain"], 1.03**10, rtol=1e-5)


def test_spatial_stats_include_streamfunction_extrema() -> None:
    wet = np.ones((2, 2), dtype=bool)
    fields = {
        "surface_speed": np.asarray([[[1.0, 2.0], [3.0, 4.0]]]),
        "phihyd_surface": np.asarray([[[2.0, 4.0], [6.0, 8.0]]]),
        "sst": np.asarray([[[3.0, 6.0], [9.0, 12.0]]]),
        "streamfunction": np.asarray([[[-4.0, 1.0], [5.0, 9.0]]]),
    }
    result = _spatial_stats(fields, wet)
    assert result["minimum__streamfunction"][0] == -4.0
    assert result["maximum__streamfunction"][0] == 9.0
    assert "minimum__sst" not in result
