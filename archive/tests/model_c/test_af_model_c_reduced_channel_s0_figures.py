from __future__ import annotations

import numpy as np

from bire_repro.af_model_c_bire_s0_figures import (
    LEAD_DAYS,
    RMSE_FIELDS,
)
from bire_repro.af_model_c_reduced_channel_s0_figures import (
    _deterministic_gate,
)


def test_arm_r_deterministic_gate_uses_days_10_through_90() -> None:
    arrays: dict[str, np.ndarray] = {}
    for field in RMSE_FIELDS:
        model = np.ones((15, len(LEAD_DAYS)), dtype=np.float32)
        persistence = np.full_like(model, 2.0)
        climatology = np.full_like(model, 3.0)
        arrays[f"rmse__model__{field}"] = model
        arrays[f"rmse__persistence__{field}"] = persistence
        arrays[f"rmse__climatology__{field}"] = climatology
    gate = _deterministic_gate(arrays)
    assert gate["passed"] is True
    for field in RMSE_FIELDS:
        assert (
            gate["fields"][field]["persistence"]["rmse_auc_ratio_10_90"]
            == 0.5
        )
