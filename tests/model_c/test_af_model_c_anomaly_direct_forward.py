from __future__ import annotations

import numpy as np
import pytest

from bire_repro.af_model_c_anomaly_direct_forward import (
    HORIZON_DAYS,
    ROLLOUT_STEPS,
    ForwardContractError,
    _declared_starts,
    curve_auc,
)


def test_curve_auc_preserves_a_constant_curve() -> None:
    leads = tuple(range(10, 91, 10))
    entries = [{"mean": 2.5} for _ in leads]
    assert curve_auc(entries, leads) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="AUC inputs changed"):
        curve_auc(entries[:-1], leads)


def test_declared_starts_must_remain_inside_inference() -> None:
    contract = {"protocol": {"start_times": [10, 12]}}
    pair = np.full(100, 3, dtype=np.uint8)
    snapshot = np.full(
        100 + HORIZON_DAYS * ROLLOUT_STEPS, 3, dtype=np.uint8
    )
    assert _declared_starts(contract, pair, snapshot) == (
        (0, 10),
        (0, 12),
        (1, 10),
        (1, 12),
        (2, 10),
        (2, 12),
    )
    snapshot[12 + HORIZON_DAYS * ROLLOUT_STEPS] = 0
    with pytest.raises(ForwardContractError, match="leaves the fresh"):
        _declared_starts(contract, pair, snapshot)
