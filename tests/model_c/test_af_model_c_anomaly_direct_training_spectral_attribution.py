from __future__ import annotations

import numpy as np
import pytest

from bire_repro.af_model_c_anomaly_direct_training_spectral_attribution import (
    spectral_summary,
)


def test_spectral_summary_preserves_frozen_ratio_and_tail_scale() -> None:
    modes = np.arange(1, 18, dtype=np.float64)
    truth = np.where(modes < 10, 1.0, 2.0e-8)
    model = truth.copy()
    model[modes >= 10] *= 100.0
    result = spectral_summary(
        modes,
        np.stack((truth, truth)),
        np.stack((model, model)),
    )
    assert result["frozen_median_modewise_ratio"] == pytest.approx(1.0)
    assert result["frozen_factor_four_pass"] is True
    assert result["integrated_energy_ratio"] == pytest.approx(1.0, rel=2.0e-6)
    assert result["tail_model_fraction"] < 2.0e-6
    assert result["tail_integrated_ratio"] == pytest.approx(100.0)


def test_spectral_summary_rejects_missing_tail() -> None:
    with pytest.raises(ValueError, match="band is empty"):
        spectral_summary(
            np.arange(1, 5),
            np.ones((2, 4)),
            np.ones((2, 4)),
        )
