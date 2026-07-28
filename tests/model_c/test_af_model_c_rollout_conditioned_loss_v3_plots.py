import numpy as np

from bire_repro.analysis.af_model_c_rollout_conditioned_loss_v3_plots import (
    FIELDS,
    mean_rmse_curve,
    publication_rows,
)
from bire_repro.af_model_c_successor_validation import LEAD_DAYS


def _arrays() -> dict[str, np.ndarray]:
    arrays = {}
    methods = (
        "source_persistence",
        "source_climatology",
        "source_duration_5760",
        "loss_v3_960",
    )
    scales = {
        "source_persistence": 2.0,
        "source_climatology": 4.0,
        "source_duration_5760": 3.0,
        "loss_v3_960": 1.0,
    }
    for field in FIELDS:
        for method in methods:
            arrays[f"{method}__rmse__{field}"] = np.full(
                (540, len(LEAD_DAYS)),
                scales[method],
                dtype=np.float32,
            )
    return arrays


def test_mean_rmse_curve_uses_member_mean() -> None:
    curve = mean_rmse_curve(_arrays(), "loss_v3_960", "sst")
    np.testing.assert_allclose(curve, np.ones(len(LEAD_DAYS)))


def test_publication_rows_include_exact_ratios() -> None:
    rows = publication_rows(_arrays(), [960])
    selected = next(
        row
        for row in rows
        if row["field"] == "sst"
        and row["method"] == "loss_v3_960"
        and row["lead_days"] == 90
    )
    assert selected["ratio_to_persistence"] == 0.5
    assert selected["ratio_to_climatology"] == 0.25
