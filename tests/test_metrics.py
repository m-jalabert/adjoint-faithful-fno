import numpy as np

from bire_repro.metrics import anomaly_correlation, latitude_weights, legacy_unweighted_mse, weighted_rmse


def test_weighted_metrics():
    truth = np.arange(12, dtype=float).reshape(1, 3, 4)
    prediction = truth + 2.0
    weights = latitude_weights(np.asarray([10.0, 20.0, 30.0]), nx=4)
    np.testing.assert_allclose(weighted_rmse(prediction, truth, weights), [2.0])
    np.testing.assert_allclose(legacy_unweighted_mse(prediction, truth), [4.0])
    np.testing.assert_allclose(anomaly_correlation(truth, truth, np.zeros((3, 4)), weights), [1.0])

