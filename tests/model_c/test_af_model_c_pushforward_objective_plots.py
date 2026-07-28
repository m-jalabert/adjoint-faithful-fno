from __future__ import annotations

import numpy as np

from bire_repro.analysis.af_model_c_pushforward_objective_plots import (
    FIELDS,
    mean_rmse_curve,
    publication_rows,
)
from bire_repro.af_model_c_successor_validation import LEAD_DAYS


STEPS = (480, 960)


def _arrays() -> dict[str, np.ndarray]:
    count = 4
    result: dict[str, np.ndarray] = {}
    for index, field in enumerate(FIELDS, start=1):
        base = np.full(
            (count, len(LEAD_DAYS)),
            float(index),
        )
        result[f"source_persistence__rmse__{field}"] = base
        result[f"source_climatology__rmse__{field}"] = 2.0 * base
        result[f"source_step_14400__rmse__{field}"] = 1.5 * base
        result[f"fine_tune_{STEPS[0]}__rmse__{field}"] = 0.5 * base
        result[f"fine_tune_{STEPS[1]}__rmse__{field}"] = 0.75 * base
    return result


def test_pushforward_curves_resolve_source_and_fine_tunes() -> None:
    arrays = _arrays()
    assert np.all(mean_rmse_curve(arrays, "persistence", "sst") == 1.0)
    assert np.all(mean_rmse_curve(arrays, "source", "sst") == 1.5)
    assert np.all(mean_rmse_curve(arrays, "480", "sst") == 0.5)


def test_publication_rows_cover_all_plotted_values() -> None:
    rows = publication_rows(_arrays(), STEPS)
    assert len(rows) == len(FIELDS) * (len(STEPS) + 3) * len(LEAD_DAYS)
    selected = next(
        row
        for row in rows
        if row["field"] == "phihyd_surface"
        and row["method"] == "fine_tune_960"
        and row["lead_days"] == 90
    )
    assert selected["mean_rmse"] == 1.5
    assert selected["ratio_to_persistence"] == 0.75
    assert selected["ratio_to_climatology"] == 0.375
