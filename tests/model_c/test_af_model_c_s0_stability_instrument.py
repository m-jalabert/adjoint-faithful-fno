from __future__ import annotations

import numpy as np

from bire_repro.af_model_c_s0_stability_instrument import (
    bootstrap_gain_interval,
    first_and_sustained_crossing,
    fit_log_gain,
)


def test_fit_log_gain_recovers_declared_multiplier() -> None:
    leads = np.arange(0, 601, 10)
    expected = 1.026
    curve = 0.2 * expected ** (leads / 10)
    gain, e_folding = fit_log_gain(leads, curve, (300, 600))
    assert np.isclose(gain, expected, rtol=1e-12)
    assert np.isclose(e_folding, 10 / np.log(expected), rtol=1e-12)


def test_bootstrap_gain_interval_contains_identical_member_gain() -> None:
    leads = np.arange(0, 601, 10)
    curve = 0.2 * 1.03 ** (leads / 10)
    members = np.repeat(curve[None], 15, axis=0)
    lower, upper = bootstrap_gain_interval(
        leads,
        members,
        (300, 600),
        replicates=100,
    )
    assert np.isclose(lower, 1.03)
    assert np.isclose(upper, 1.03)


def test_first_and_sustained_crossing_distinguishes_reentry() -> None:
    leads = np.arange(0, 70, 10)
    model = np.asarray([0, 1, 3, 1, 4, 5, 6], dtype=float)
    baseline = np.asarray([0, 2, 2, 2, 2, 2, 2], dtype=float)
    assert first_and_sustained_crossing(leads, model, baseline) == {
        "first_day": 20,
        "sustained_day": 40,
    }
