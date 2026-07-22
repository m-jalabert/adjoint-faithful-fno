"""Small helper checks for frozen Model B evaluation."""

import numpy as np
import pytest

from bire_repro.af_model_b_evaluate import _normalized_prediction


def test_model_b_evaluation_adds_residual_to_normalized_present() -> None:
    torch = pytest.importorskip("torch")

    class UnitResidual(torch.nn.Module):
        def forward(self, value):
            return torch.ones((value.shape[0], 46, *value.shape[-2:]), dtype=value.dtype)

    features = np.zeros((2, 51, 3, 4), dtype=np.float32)
    features[:, :46] = 2.0
    prediction = _normalized_prediction(UnitResidual(), features, torch.device("cpu"))
    np.testing.assert_array_equal(prediction, np.full((2, 46, 3, 4), 3.0, dtype=np.float32))
