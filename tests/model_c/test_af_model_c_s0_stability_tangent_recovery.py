from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from bire_repro.af_forward_complete import (  # noqa: E402
    derived_fields,
    radial_spectrum,
)
from bire_repro.af_model_c_s0_stability_tangent_recovery import (  # noqa: E402
    derived_fields64,
    member_rmse64,
    radial_spectrum64,
    safe_log_gain,
)


def test_float64_reductions_do_not_overflow_large_float32_states() -> None:
    wet = np.ones((8, 8), dtype=bool)
    prediction = np.full((2, 8, 8), np.float32(1.0e30), dtype=np.float32)
    truth = np.zeros_like(prediction)
    rmse = member_rmse64(prediction, truth, wet)
    assert np.all(np.isfinite(rmse))
    assert np.allclose(rmse, 1.0e30, rtol=1.0e-6)


def test_derived_fields_and_spectrum_remain_float64() -> None:
    wet = np.ones((8, 8), dtype=bool)
    states = np.full((1, 46, 8, 8), np.float32(1.0e20), dtype=np.float32)
    fields = derived_fields64(states, wet)
    assert fields["surface_speed"].dtype == np.float64
    assert np.isfinite(fields["surface_speed"]).all()
    _, spectrum = radial_spectrum64(fields["sst"], wet)
    assert spectrum.dtype == np.float64
    assert np.isfinite(spectrum).all()


def test_float64_diagnostics_match_original_at_physical_amplitudes() -> None:
    generator = np.random.default_rng(7)
    wet = np.ones((12, 12), dtype=bool)
    states = generator.normal(size=(2, 46, 12, 12)).astype(np.float32)
    recovered = derived_fields64(states, wet)
    original = derived_fields(states, wet)
    for field in recovered:
        assert np.allclose(recovered[field], original[field], rtol=2.0e-6, atol=1.0e-6)
    modes64, spectrum64 = radial_spectrum64(recovered["sst"], wet)
    modes32, spectrum32 = radial_spectrum(original["sst"], wet)
    assert np.array_equal(modes64, modes32)
    assert np.allclose(spectrum64, spectrum32, rtol=2.0e-6, atol=1.0e-6)


def test_safe_log_gain_censors_at_first_incomplete_lead() -> None:
    leads = np.arange(300, 610, 10)
    calls = np.arange(leads.size)
    members = np.repeat((1.05**calls)[None], 4, axis=0)
    members[0, 20:] = np.nan
    result = safe_log_gain(leads, members, (300, 600))
    assert result["status"] == "censored_at_first_nonfinite"
    assert result["fit_lead_days"] == [300, 490]
    assert np.isclose(result["gain"], 1.05)
    assert result["bootstrap_95_percent_interval"] is None
