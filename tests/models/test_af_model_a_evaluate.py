"""Small contract checks for frozen Model A evaluation helpers."""

import numpy as np

from bire_repro.analysis.af_model_a_evaluate import _features


def test_model_a_features_have_dynamic_and_static_contract() -> None:
    class FakeState:
        def __getitem__(self, key):
            return np.full((46, 2, 2), float(key[1]), dtype=np.float32)

    class FakeStatic:
        def __getitem__(self, key):
            value = np.zeros((5, 2, 2), dtype=np.float32)
            value[0] = float(key + 1)
            return value

    wet = np.array([[True, False], [True, True]])
    features, raw, geometry = _features(
        FakeState(), FakeStatic(), ((0, 1), (1, 2)), np.zeros(46, dtype=np.float32), np.ones(46, dtype=np.float32), wet, 1.0, 2.0
    )
    assert features.shape == (2, 51, 2, 2)
    assert raw.shape == (2, 46, 2, 2)
    assert geometry.shape == (2, 5, 2, 2)
    assert np.all(geometry[:, 0, ~wet] == 0.0)
