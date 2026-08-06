"""Pure contracts for the fresh-v2 Model C successor validation gate."""

from pathlib import Path

import numpy as np

from bire_repro.af_model_c_successor_validation import (
    LEAD_DAYS,
    PRIMARY_FIELDS,
    _bootstrap_summary,
    circular_block_indices,
    complete_validation_times,
    curve_auc,
    load_validation_contract,
    resolve_replication_seed,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config" / "model_c_successor_validation_v2.json"


def test_complete_validation_times_requires_every_ten_day_transition() -> None:
    pair_codes = np.zeros(130, dtype=np.uint8)
    snapshot_codes = np.zeros(140, dtype=np.uint8)
    pair_codes[10:120] = 2
    snapshot_codes[10:130] = 2
    selected = complete_validation_times(pair_codes, snapshot_codes)
    assert selected.tolist() == list(range(10, 40))

    pair_codes[50] = 0
    selected = complete_validation_times(pair_codes, snapshot_codes)
    assert 10 not in selected
    assert 20 not in selected
    assert 30 not in selected


def test_curve_auc_is_the_normalized_trapezoidal_area() -> None:
    constant = np.full((3, len(LEAD_DAYS)), 2.5)
    assert np.allclose(curve_auc(constant), 2.5)
    linear = np.asarray(LEAD_DAYS, dtype=float)[None]
    assert np.allclose(curve_auc(linear), 50.0)


def test_circular_block_indices_are_reproducible_and_exact_length() -> None:
    first = circular_block_indices(17, 6, np.random.default_rng(42))
    second = circular_block_indices(17, 6, np.random.default_rng(42))
    assert first.shape == (17,)
    assert np.array_equal(first, second)
    assert np.all((0 <= first) & (first < 17))


def test_block_bootstrap_accepts_uniform_strong_improvement() -> None:
    records = np.asarray(
        [(experiment, time) for experiment in range(3) for time in range(20)]
    )
    model = {}
    baselines = {
        method: {} for method in ("persistence", "climatology", "a0")
    }
    for field in PRIMARY_FIELDS:
        model[f"rmse__{field}"] = np.full((60, 9), 0.5)
        model[f"acc__{field}"] = np.full((60, 9), 0.8)
        for method in baselines:
            baselines[method][f"rmse__{field}"] = np.ones((60, 9))
            baselines[method][f"acc__{field}"] = np.full((60, 9), 0.3)
    contract = {
        "bootstrap": {
            "replicates": 50,
            "confidence_level": 0.95,
            "seed": 20260727,
            "block_length_days_by_regime": [4, 7, 5],
        }
    }
    result = _bootstrap_summary(
        [model, model, model],
        baselines,
        records,
        contract,
    )
    assert result["passed"]
    for field in PRIMARY_FIELDS:
        for comparison in result["fields"][field].values():
            assert comparison["rmse_auc_ratio"]["point"] == 0.5
            assert np.isclose(
                comparison["acc_auc_difference"]["point"],
                0.5,
            )


def test_frozen_contract_and_replication_seed_resolution() -> None:
    contract, path, digest = load_validation_contract(CONTRACT)
    assert path == CONTRACT
    assert len(digest) == 64
    assert contract["validation"]["lead_days"] == list(LEAD_DAYS)
    assert resolve_replication_seed(CONTRACT, array_index=0) == 20260724
    assert resolve_replication_seed(CONTRACT, array_index=1) == 20260725
