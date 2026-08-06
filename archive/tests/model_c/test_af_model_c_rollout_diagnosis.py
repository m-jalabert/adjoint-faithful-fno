from __future__ import annotations

from pathlib import Path

import numpy as np

from bire_repro.af_model_c_rollout_diagnosis import (
    complete_rollout_starts,
    lead_curve_summary,
    load_rollout_diagnosis_contract,
    select_balanced_training_times,
)
from bire_repro.af_model_c_successor_validation import (
    EVALUATION_FIELDS,
    LEAD_DAYS,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config" / "model_c_rollout_diagnosis_v1.json"


def test_frozen_rollout_diagnosis_contract() -> None:
    contract, path, digest = load_rollout_diagnosis_contract(CONTRACT)
    assert path == CONTRACT
    assert len(digest) == 64
    assert contract["records"]["records_total"] == 540
    assert contract["read_contract"]["inference_read"] is False


def test_complete_rollout_starts_and_balanced_selection() -> None:
    pair_codes = np.zeros(340, dtype=np.uint8)
    snapshot_codes = np.zeros(341, dtype=np.uint8)
    pair_codes[0:150] = 1
    snapshot_codes[0:160] = 1
    pair_codes[180:330] = 1
    snapshot_codes[180:340] = 1

    complete = complete_rollout_starts(pair_codes, snapshot_codes)
    assert np.array_equal(
        complete,
        np.concatenate((np.arange(70), np.arange(180, 250))),
    )

    selected, bounds = select_balanced_training_times(
        complete,
        starts_per_block=10,
        expected_blocks=2,
    )
    assert bounds == ((0, 69), (180, 249))
    assert selected.shape == (20,)
    assert np.unique(selected).size == 20
    assert np.sum(selected <= 69) == 10
    assert np.sum(selected >= 180) == 10


def test_lead_curve_summary_ratios_and_crossing() -> None:
    count = 6
    records = np.asarray(
        [(experiment, index) for experiment in range(3) for index in range(2)],
        dtype=np.int64,
    )
    metrics: dict[str, np.ndarray] = {}
    persistence: dict[str, np.ndarray] = {}
    climatology: dict[str, np.ndarray] = {}
    ratios = np.linspace(0.5, 1.5, len(LEAD_DAYS))
    for field in EVALUATION_FIELDS:
        persistence[f"rmse__{field}"] = np.ones(
            (count, len(LEAD_DAYS)),
            dtype=np.float64,
        )
        persistence[f"acc__{field}"] = np.zeros(
            (count, len(LEAD_DAYS)),
            dtype=np.float64,
        )
        climatology[f"rmse__{field}"] = np.full(
            (count, len(LEAD_DAYS)),
            2.0,
            dtype=np.float64,
        )
        climatology[f"acc__{field}"] = np.full(
            (count, len(LEAD_DAYS)),
            -0.5,
            dtype=np.float64,
        )
        metrics[f"rmse__{field}"] = np.broadcast_to(
            ratios,
            (count, len(LEAD_DAYS)),
        ).copy()
        metrics[f"acc__{field}"] = np.ones(
            (count, len(LEAD_DAYS)),
            dtype=np.float64,
        )

    summary = lead_curve_summary(
        metrics,
        {"persistence": persistence, "climatology": climatology},
        records,
    )
    surface = summary["sst"]
    assert np.allclose(
        surface["persistence"]["rmse_ratio_by_lead"],
        ratios,
    )
    assert surface["persistence"]["first_rmse_crossing_day"] == 50
    assert np.isclose(surface["persistence"]["rmse_auc_ratio"], 1.0)
    assert np.isclose(surface["climatology"]["rmse_auc_ratio"], 0.5)
    assert np.isclose(surface["climatology"]["acc_auc_difference"], 1.5)
