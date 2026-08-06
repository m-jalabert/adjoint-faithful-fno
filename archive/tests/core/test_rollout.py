import json

import numpy as np
import pytest

from bire_repro.core.rollout import (
    rollout_autoregressive,
    rollout_days,
    spatial_subsample,
    stride_for_resolution,
    write_rollout,
)
from bire_repro.core.training import PointwiseNormalizer, SequenceSource


def test_rollout_day_contract_and_stride_aliases():
    assert np.array_equal(rollout_days(40, 10), [0, 10, 20, 30, 40])
    assert rollout_days(2000, 30)[-1] == 1980
    assert stride_for_resolution("full") == 1
    assert stride_for_resolution("2deg") == 8
    assert stride_for_resolution("low") == 8
    values = np.arange(64).reshape(8, 8)
    assert np.array_equal(spatial_subsample(values, 2), values[::2, ::2])


def test_autoregressive_rollout_retains_static_forcing():
    torch = pytest.importorskip("torch")

    class AddForcing(torch.nn.Module):
        def forward(self, x):
            return x[:, :2] + x[:, 2:3]

    initial = torch.zeros(1, 2, 3, 3)
    forcing = torch.full((1, 1, 3, 3), 2.0)
    result = rollout_autoregressive(AddForcing(), initial, forcing, n_steps=3)
    assert result.shape == (1, 4, 2, 3, 3)
    assert torch.all(result[:, 1] == 2)
    assert torch.all(result[:, 3] == 6)


def test_npz_rollout_schema_contains_prediction_truth_and_attrs(tmp_path):
    torch = pytest.importorskip("torch")

    class IdentityState(torch.nn.Module):
        in_channels = 3
        out_channels = 2

        def forward(self, x):
            return x[:, :2]

    data = np.zeros((1, 5, 3, 4, 4), dtype=np.float32)
    data[0, :, 0] = np.arange(5)[:, None, None]
    data[0, :, 1] = 2 * np.arange(5)[:, None, None]
    data[0, :, 2] = 0.1
    source_path = tmp_path / "state.npy"
    np.save(source_path, data)
    source = SequenceSource(source_path)
    normalizer = PointwiseNormalizer(
        np.zeros((3, 4, 4), dtype=np.float32),
        np.ones((3, 4, 4), dtype=np.float32),
    )
    checkpoint = tmp_path / "dummy.pt"
    checkpoint.write_bytes(b"test checkpoint")
    output = tmp_path / "rollout.npz"
    summary = write_rollout(
        model=IdentityState(),
        checkpoint_path=checkpoint,
        source=source,
        normalizer=normalizer,
        experiment_id=1,
        initial_indices=(0,),
        lag_days=1,
        horizon_days=3,
        output_path=output,
        state_channels=2,
        wind_channel=2,
        config_hash="abc123",
    )
    assert summary["missing_truth_member_times"] == 0
    with np.load(output, allow_pickle=False) as payload:
        assert payload["prediction"].shape == (1, 4, 2, 4, 4)
        assert payload["truth"].shape == (1, 4, 2, 4, 4)
        assert np.array_equal(payload["initial_index"], [0])
        assert np.array_equal(payload["day"], [0, 1, 2, 3])
        attrs = json.loads(str(payload["attrs_json"]))
        assert attrs["experiment_id"] == 1
        assert attrs["lag_days"] == 1
        assert attrs["config_sha256"] == "abc123"
