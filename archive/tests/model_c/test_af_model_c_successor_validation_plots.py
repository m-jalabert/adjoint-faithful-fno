from __future__ import annotations

import numpy as np

from bire_repro.af_model_c_successor_validation import (
    EVALUATION_FIELDS,
    LEAD_DAYS,
)
from bire_repro.analysis.af_model_c_successor_validation_plots import (
    BASELINES,
    SEEDS,
    all_field_auc_ratios,
    baseline_curve,
    mean_model_curve,
)


def _synthetic_arrays() -> dict[str, np.ndarray]:
    count = 6
    arrays: dict[str, np.ndarray] = {}
    for field_index, field in enumerate(EVALUATION_FIELDS, start=1):
        base = np.full(
            (count, len(LEAD_DAYS)),
            float(field_index),
            dtype=np.float64,
        )
        for baseline_index, baseline in enumerate(BASELINES, start=1):
            arrays[f"{baseline}__rmse__{field}"] = base * baseline_index
            arrays[f"{baseline}__acc__{field}"] = np.zeros_like(base)
        for seed_index, seed in enumerate(SEEDS, start=1):
            arrays[f"seed_{seed}__rmse__{field}"] = base * seed_index
            arrays[f"seed_{seed}__acc__{field}"] = np.ones_like(base)
    return arrays


def test_model_and_baseline_curves() -> None:
    arrays = _synthetic_arrays()
    mean, low, high = mean_model_curve(
        arrays,
        metric="rmse",
        field="sst",
    )
    field_index = EVALUATION_FIELDS.index("sst") + 1
    assert np.all(mean == 2 * field_index)
    assert np.all(low == field_index)
    assert np.all(high == 3 * field_index)
    assert np.all(
        baseline_curve(
            arrays,
            baseline="climatology",
            metric="rmse",
            field="sst",
        )
        == 2 * field_index
    )


def test_all_field_auc_ratios() -> None:
    ratios = all_field_auc_ratios(_synthetic_arrays())
    for field in EVALUATION_FIELDS:
        assert np.isclose(ratios[field]["persistence"], 2.0)
        assert np.isclose(ratios[field]["climatology"], 1.0)
        assert np.isclose(ratios[field]["a0"], 2.0 / 3.0)
