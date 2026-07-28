from __future__ import annotations

import numpy as np

from bire_repro.analysis.af_model_c_rollout_diagnosis_plots import (
    BASELINES,
    SEEDS,
    baseline_curve,
    field_auc_ratios,
    model_curve,
    ratio_curve,
)
from bire_repro.af_model_c_successor_validation import (
    EVALUATION_FIELDS,
    LEAD_DAYS,
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
        for seed_index, seed in enumerate(SEEDS, start=1):
            arrays[f"seed_{seed}__rmse__{field}"] = base * seed_index
    return arrays


def test_model_baseline_and_ratio_curves() -> None:
    arrays = _synthetic_arrays()
    field_index = EVALUATION_FIELDS.index("sst") + 1
    mean, low, high = model_curve(arrays, "sst")
    assert np.all(mean == 2 * field_index)
    assert np.all(low == field_index)
    assert np.all(high == 3 * field_index)
    assert np.all(
        baseline_curve(arrays, "climatology", "sst") == 2 * field_index
    )
    ratio_mean, ratio_low, ratio_high = ratio_curve(arrays, "sst")
    assert np.all(ratio_mean == 2.0)
    assert np.all(ratio_low == 1.0)
    assert np.all(ratio_high == 3.0)


def test_all_field_auc_ratios() -> None:
    ratios = field_auc_ratios(_synthetic_arrays())
    for field in EVALUATION_FIELDS:
        assert np.isclose(ratios[field]["persistence"], 2.0)
        assert np.isclose(ratios[field]["climatology"], 1.0)
