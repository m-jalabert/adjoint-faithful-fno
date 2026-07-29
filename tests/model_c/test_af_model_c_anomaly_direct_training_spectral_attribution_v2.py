from __future__ import annotations

import numpy as np
import pytest

from bire_repro.af_model_c_anomaly_direct_training_spectral_attribution_v2 import (
    target_snapshot_offset,
    training_records,
)


def test_target_snapshot_offset_uses_ten_day_stride() -> None:
    assert target_snapshot_offset(1) == 10
    assert target_snapshot_offset(9) == 90
    assert target_snapshot_offset(36) == 360
    with pytest.raises(ValueError, match="outside"):
        target_snapshot_offset(0)


def test_training_records_require_complete_daily_split_window() -> None:
    times = np.asarray([0], dtype=np.int32)
    records = np.asarray([(experiment, 0) for experiment in range(3)], dtype=np.int32)
    contract = {
        "protocol": {
            "start_times": times.tolist(),
            "start_times_sha256": (
                "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
            ),
            "records_sha256": (
                "ab25350e3e65efebe24584461683ecda68725576e825e550038b90e7b1479946"
            ),
        }
    }
    assert records.shape == (3, 2)
    split = np.ones(361, dtype=np.uint8)
    assert training_records(contract, split).tolist() == records.tolist()
    split[200] = 3
    with pytest.raises(RuntimeError, match="leaves split 1"):
        training_records(contract, split)
