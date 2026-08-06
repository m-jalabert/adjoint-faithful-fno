from __future__ import annotations

import numpy as np

from bire_repro.af_model_c_bire_figures import (
    FIGURE_3_NAME,
    FIGURE_4_NAME,
    FIELDS,
    LEAD_DAYS,
    METHODS,
    _plot_figure3,
    _plot_figure4,
    complete_figure_starts,
    percentile_curve,
    select_ensemble_starts,
)


def test_complete_figure_starts() -> None:
    pairs = np.full(41, 2, dtype=np.uint8)
    snapshots = np.full(41, 2, dtype=np.uint8)
    starts = complete_figure_starts(
        pairs,
        snapshots,
        maximum_lead_days=20,
    )
    assert np.array_equal(starts, np.arange(21))
    pairs[10] = 3
    starts = complete_figure_starts(
        pairs,
        snapshots,
        maximum_lead_days=20,
    )
    assert 0 not in starts
    assert 10 not in starts


def test_deterministic_start_selection() -> None:
    candidates = np.arange(70, dtype=np.int64) + 6300
    first = select_ensemble_starts(candidates, count=15, seed=20260727)
    second = select_ensemble_starts(candidates, count=15, seed=20260727)
    assert np.array_equal(first, second)
    assert np.unique(first).size == 15
    assert np.all(np.isin(first, candidates))


def test_percentile_curve() -> None:
    values = np.arange(15, dtype=np.float64)[:, None] + np.arange(3)
    summary = percentile_curve(values)
    assert np.array_equal(summary["mean"], np.asarray((7.0, 8.0, 9.0)))
    assert np.allclose(summary["p10"], (1.4, 2.4, 3.4))
    assert np.allclose(summary["p90"], (12.6, 13.6, 14.6))


def test_plotters(tmp_path) -> None:
    arrays: dict[str, np.ndarray] = {
        "figure3_truth_streamfunction": np.ones((5, 5, 5)),
        "figure3_prediction_streamfunction": np.full((5, 5, 5), 0.8),
    }
    for method_index, method in enumerate(METHODS, start=1):
        for field_index, field in enumerate(FIELDS, start=1):
            curve = np.linspace(0, 0.01 * field_index, len(LEAD_DAYS))
            arrays[f"{method}__rmse__{field}"] = np.tile(
                method_index * curve,
                (15, 1),
            )
    coordinate = np.arange(5, dtype=np.float64)
    longitude, latitude = np.meshgrid(coordinate, coordinate)
    wet = np.ones((5, 5), dtype=bool)
    _plot_figure3(tmp_path, arrays, longitude, latitude, wet)
    _plot_figure4(tmp_path, arrays)
    assert (tmp_path / FIGURE_3_NAME).stat().st_size > 0
    assert (tmp_path / FIGURE_4_NAME).stat().st_size > 0
