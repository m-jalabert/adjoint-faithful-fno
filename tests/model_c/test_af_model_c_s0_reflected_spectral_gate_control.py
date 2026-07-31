from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bire_repro.af_model_c_s0_reflected_spectral_gate_control import (  # noqa: E402
    reflected_spectral_gate,
)


def test_zero_gate_is_exact_and_constant_is_preserved() -> None:
    wet = np.ones((12, 12), dtype=bool)
    value = torch.randn(2, 3, 12, 12)
    zero = reflected_spectral_gate(value, wet, torch.zeros(2))
    constant = reflected_spectral_gate(
        torch.full_like(value, 4.0),
        wet,
        torch.ones(2),
    )
    assert torch.equal(zero, value)
    assert torch.allclose(constant, torch.full_like(value, 4.0), atol=1.0e-6)


def test_full_gate_damps_high_modes_but_preserves_smooth_field() -> None:
    wet = np.ones((32, 32), dtype=bool)
    y, x = torch.meshgrid(
        torch.arange(32, dtype=torch.float32),
        torch.arange(32, dtype=torch.float32),
        indexing="ij",
    )
    smooth = torch.cos(torch.pi * x / 31.0)[None, None]
    checker = ((x.to(torch.int64) + y.to(torch.int64)) % 2).float()[None, None]
    value = torch.cat((smooth, checker), dim=1)
    filtered = reflected_spectral_gate(value, wet, torch.ones(1))
    smooth_error = torch.sqrt(torch.mean((filtered[:, :1] - smooth) ** 2))
    assert smooth_error < 1.0e-5
    assert torch.std(filtered[:, 1:]) < 0.05 * torch.std(checker)


def test_land_is_reset_to_zero() -> None:
    wet = np.ones((10, 10), dtype=bool)
    wet[[0, -1], :] = False
    wet[:, [0, -1]] = False
    value = torch.randn(2, 4, 10, 10)
    filtered = reflected_spectral_gate(value, wet, torch.ones(2) * 0.5)
    assert torch.count_nonzero(filtered[:, :, ~torch.from_numpy(wet)]) == 0
