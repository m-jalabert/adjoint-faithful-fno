"""Small numerical contracts for the paper-style frozen A0 diagnostics."""

import numpy as np

from bire_repro.af_a0_paper_plots import anomaly_correlation_samples


def test_anomaly_correlation_is_one_for_identical_members() -> None:
    wet = np.array([[True, True], [True, False]])
    truth = np.array([[[[2.0, 3.0], [4.0, 0.0]]]], dtype=np.float32)
    value = anomaly_correlation_samples(truth, truth.copy(), np.array([1.0], dtype=np.float32), slice(0, 1), wet)
    assert value.shape == (1,)
    assert value[0] == 1.0


def test_anomaly_correlation_handles_a_zero_variance_member() -> None:
    wet = np.ones((2, 2), dtype=bool)
    constant = np.ones((1, 1, 2, 2), dtype=np.float32)
    value = anomaly_correlation_samples(constant, constant, np.array([1.0], dtype=np.float32), slice(0, 1), wet)
    assert value[0] == 0.0
