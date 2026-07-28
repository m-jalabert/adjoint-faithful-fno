from __future__ import annotations

import numpy as np

from bire_repro.analysis.af_model_c_pushforward_duration_plots import (
    FIELDS,
    mean_rmse_curve,
    publication_rows,
)
from bire_repro.af_model_c_successor_validation import LEAD_DAYS


STEPS = (2400, 2880)


def _arrays() -> dict[str, np.ndarray]:
    count = 4
    result: dict[str, np.ndarray] = {}
    for index, field in enumerate(FIELDS, start=1):
        base = np.full((count, len(LEAD_DAYS)), float(index))
        result[f"source_source_persistence__rmse__{field}"] = base
        result[f"source_source_climatology__rmse__{field}"] = 2.0 * base
        result[f"source_fine_tune_1920__rmse__{field}"] = 1.5 * base
        result[f"duration_{STEPS[0]}__rmse__{field}"] = 0.5 * base
        result[f"duration_{STEPS[1]}__rmse__{field}"] = 0.75 * base
    return result


def test_duration_curves_resolve_source_and_extensions() -> None:
    arrays = _arrays()
    assert np.all(mean_rmse_curve(arrays, "persistence", "sst") == 1.0)
    assert np.all(mean_rmse_curve(arrays, "pushforward_v1", "sst") == 1.5)
    assert np.all(mean_rmse_curve(arrays, "2400", "sst") == 0.5)


def test_publication_rows_cover_all_duration_values() -> None:
    rows = publication_rows(_arrays(), STEPS)
    assert len(rows) == len(FIELDS) * (len(STEPS) + 3) * len(LEAD_DAYS)
    selected = next(
        row
        for row in rows
        if row["field"] == "phihyd_surface"
        and row["method"] == "fine_tune_2880"
        and row["lead_days"] == 90
    )
    assert selected["mean_rmse"] == 1.5
    assert selected["ratio_to_persistence"] == 0.75
    assert selected["ratio_to_climatology"] == 0.375
