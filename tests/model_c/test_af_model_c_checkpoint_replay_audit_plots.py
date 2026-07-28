from __future__ import annotations

import numpy as np

from bire_repro.af_model_c_checkpoint_replay_audit import (
    SLOW_PRIMARY_FIELDS,
)
from bire_repro.analysis.af_model_c_checkpoint_replay_audit_plots import (
    audit_csv_rows,
    mean_rmse_curve,
    ratio_curve,
)
from bire_repro.af_model_c_successor_validation import LEAD_DAYS


STEPS = (11520, 13440)


def _synthetic_arrays() -> dict[str, np.ndarray]:
    count = 5
    arrays: dict[str, np.ndarray] = {}
    for field_index, field in enumerate(SLOW_PRIMARY_FIELDS, start=1):
        base = np.full(
            (count, len(LEAD_DAYS)),
            field_index,
            dtype=np.float64,
        )
        arrays[f"persistence__rmse__{field}"] = base
        arrays[f"climatology__rmse__{field}"] = base * 2.0
        arrays[f"step_{STEPS[0]}__rmse__{field}"] = base * 0.5
        arrays[f"step_{STEPS[1]}__rmse__{field}"] = base * 1.5
    return arrays


def test_checkpoint_and_baseline_mean_curves() -> None:
    arrays = _synthetic_arrays()
    assert np.all(
        mean_rmse_curve(arrays, "persistence", "sst") == 1.0
    )
    assert np.all(
        mean_rmse_curve(arrays, str(STEPS[0]), "sst") == 0.5
    )
    assert np.all(
        ratio_curve(
            arrays,
            STEPS[0],
            "sst",
            "persistence",
        )
        == 0.5
    )
    assert np.all(
        ratio_curve(
            arrays,
            STEPS[0],
            "sst",
            "climatology",
        )
        == 0.25
    )


def test_audit_csv_rows_cover_fields_methods_and_leads() -> None:
    rows = audit_csv_rows(_synthetic_arrays(), STEPS)
    expected_methods = len(STEPS) + 2
    assert len(rows) == (
        len(SLOW_PRIMARY_FIELDS) * expected_methods * len(LEAD_DAYS)
    )
    selected = next(
        row
        for row in rows
        if row["field"] == "phihyd_surface"
        and row["method"] == f"step_{STEPS[1]}"
        and row["lead_days"] == 90
    )
    assert selected["mean_rmse"] == 3.0
    assert selected["ratio_to_persistence"] == 1.5
    assert selected["ratio_to_climatology"] == 0.75
