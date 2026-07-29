from __future__ import annotations

import numpy as np
import pytest

from bire_repro.analysis.af_model_c_anomaly_direct_spectral_diagnosis import (
    diagnose_field,
)


def test_diagnose_field_separates_integrated_energy_from_tiny_tail() -> None:
    modes = np.arange(1, 18, dtype=np.float64)
    truth = np.where(modes < 10, 1.0, 2.0e-8)
    model = truth.copy()
    model[modes >= 10] *= 100.0
    result = diagnose_field(
        modes,
        np.stack((truth, truth)),
        np.stack((model, model)),
    )
    assert result["frozen_median_modewise_energy_ratio"] == pytest.approx(1.0)
    assert result["integrated_energy_ratio"] == pytest.approx(1.0, rel=2.0e-6)
    assert (
        result["tail_k10_plus"]["model_fraction_of_valid_energy"]
        < 2.0e-6
    )
    assert result["tail_k10_plus"]["median_modewise_energy_ratio"] == pytest.approx(
        100.0
    )


def test_diagnose_field_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError, match="do not match"):
        diagnose_field(
            np.arange(3),
            np.ones((2, 3)),
            np.ones((2, 4)),
        )
