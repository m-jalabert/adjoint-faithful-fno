"""Focused tests for the MITgcm-consistent pressure-gradient auxiliary loss."""
from __future__ import annotations

import numpy as np
import pytest

from oceanfno.runtime import torch
from oceanfno.pressure_gradient import (
    PressureGradientContext,
    phihyd_from_normalized_state,
    pressure_gradient_relative_l2,
)

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is optional")


def _context(n: int = 8) -> PressureGradientContext:
    mean = np.zeros((46, n, n), dtype=np.float32)
    scale = np.ones_like(mean)
    dx = np.full((n, n), 1.0e5, dtype=np.float32)
    wet = np.ones((n, n), dtype=bool)
    return PressureGradientContext(mean, scale, dx, wet)


def test_phihyd_shape_and_identity_loss() -> None:
    context = _context()
    state = torch.zeros((2, 3, 46, 8, 8), dtype=torch.float32)
    phi = phihyd_from_normalized_state(state, context)
    assert phi.shape == (2, 3, 15, 8, 8)
    value = pressure_gradient_relative_l2(state, state, context)
    assert float(value) == 0.0


def test_uniform_eta_offset_is_dynamically_irrelevant() -> None:
    context = _context()
    truth = torch.zeros((1, 2, 46, 8, 8), dtype=torch.float32)
    prediction = truth.clone()
    prediction[:, :, 45] += 0.25
    value = pressure_gradient_relative_l2(prediction, truth, context)
    assert float(value) == pytest.approx(0.0, abs=1.0e-7)


def test_pressure_slope_is_seen_and_backpropagates() -> None:
    context = _context()
    truth = torch.zeros((1, 2, 46, 8, 8), dtype=torch.float32)
    ramp = torch.linspace(0.0, 0.1, 8, dtype=torch.float32)[None, :].expand(8, 8)
    truth[:, :, 45] = ramp
    prediction = torch.zeros_like(truth, requires_grad=True)
    value = pressure_gradient_relative_l2(prediction, truth, context)
    assert bool(torch.isfinite(value).item())
    assert float(value) > 0.0
    value.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all().item())
    assert float(prediction.grad[:, :, 45].abs().sum()) > 0.0


def test_temperature_gradient_contributes_to_pressure_force() -> None:
    context = _context()
    truth = torch.zeros((1, 1, 46, 8, 8), dtype=torch.float32)
    ramp = torch.linspace(0.0, 2.0, 8, dtype=torch.float32)[None, :].expand(8, 8)
    truth[:, :, 30] = ramp
    prediction = torch.zeros_like(truth)
    value = pressure_gradient_relative_l2(prediction, truth, context)
    assert bool(torch.isfinite(value).item())
    assert float(value) > 0.0
