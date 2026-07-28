import numpy as np

from bire_repro.analysis.af_model_c_truncated_unroll_plots import (
    mean_rmse_curve,
    publication_rows,
)


def _arrays():
    arrays = {
        "source_persistence__rmse__sst": np.full((540, 9), 2.0),
        "source_climatology__rmse__sst": np.full((540, 9), 4.0),
        "source_duration_5760__rmse__sst": np.full((540, 9), 3.0),
        "truncated_480__rmse__sst": np.full((540, 9), 1.0),
        "source_persistence__rmse__phihyd_surface": np.full(
            (540, 9), 2.0
        ),
        "source_climatology__rmse__phihyd_surface": np.full(
            (540, 9), 4.0
        ),
        "source_duration_5760__rmse__phihyd_surface": np.full(
            (540, 9), 3.0
        ),
        "truncated_480__rmse__phihyd_surface": np.full((540, 9), 1.0),
    }
    return arrays


def test_mean_rmse_curve_routes_truncated_key():
    assert np.array_equal(
        mean_rmse_curve(_arrays(), "480", "sst"),
        np.ones(9),
    )


def test_publication_rows_include_both_baseline_ratios():
    rows = publication_rows(_arrays(), (480,))
    selected = next(
        row
        for row in rows
        if row["field"] == "sst"
        and row["method"] == "truncated_480"
        and row["lead_days"] == 10
    )
    assert selected["ratio_to_persistence"] == 0.5
    assert selected["ratio_to_climatology"] == 0.25
