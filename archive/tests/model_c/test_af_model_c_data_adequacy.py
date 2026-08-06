"""Pure decision and bootstrap tests for the Model C data-adequacy audit."""

import numpy as np

from bire_repro.diagnostics.af_model_c_data_adequacy import (
    circular_block_bootstrap_ratio,
    data_expansion_decision,
    summarize_record_mse,
)


def test_record_mse_summary_preserves_regime_boundaries() -> None:
    records = ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2))
    errors = {
        "ssh": {
            "model": np.asarray([1.0, 1.0, 4.0, 4.0, 9.0, 9.0]),
            "persistence": np.asarray([4.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
        }
    }
    summary = summarize_record_mse(records, errors, ("S0", "S1", "S2"))
    assert summary["all_regimes"]["record_count"] == 6
    assert summary["by_regime"]["S0"]["groups"]["ssh"]["ratio_to_persistence"] == 0.5
    assert summary["by_regime"]["S1"]["groups"]["ssh"]["ratio_to_persistence"] == 1.0
    assert summary["by_regime"]["S2"]["groups"]["ssh"]["ratio_to_persistence"] == 1.5


def test_circular_bootstrap_is_reproducible_and_respects_exact_scaling() -> None:
    persistence = [np.linspace(1.0, 2.0, 40), np.linspace(2.0, 3.0, 40)]
    model = [0.81 * values for values in persistence]
    kwargs = {
        "replicates": 200,
        "seed": 17,
        "confidence_level": 0.95,
    }
    first = circular_block_bootstrap_ratio(model, persistence, (5, 7), **kwargs)
    second = circular_block_bootstrap_ratio(model, persistence, (5, 7), **kwargs)
    assert first == second
    assert np.isclose(first["lower"], 0.9)
    assert np.isclose(first["median"], 0.9)
    assert np.isclose(first["upper"], 0.9)
    assert first["probability_below_persistence"] == 1.0


def test_expansion_decision_requires_every_frozen_gate() -> None:
    contract = {
        "decision_rules": {
            "chronology": {"require_strictly_decreasing_worst_group_ratio": True},
            "effective_independence": {
                "groups": ["temperature", "ssh"],
                "maximum_effective_state_samples_total_per_group": 20.0,
            },
            "per_regime_generalization": {
                "group": "ssh",
                "minimum_regimes_with_seed_median_validation_ratio_above_training_ratio": 2,
            },
            "training_fit": {
                "require_all_final_seeds_all_aggregate_groups_below_persistence": True
            },
        }
    }
    chronology = [
        {"validation_worst_group_ratio": 2.0},
        {"validation_worst_group_ratio": 1.5},
        {"validation_worst_group_ratio": 1.2},
        {"validation_worst_group_ratio": 1.0},
    ]
    per_seed = []
    for _ in range(3):
        per_seed.append(
            {
                "training": {
                    "by_regime": {
                        name: {
                            "groups": {"ssh": {"ratio_to_persistence": training}}
                        }
                        for name, training in zip(("S0", "S1", "S2"), (0.8, 0.9, 1.0))
                    }
                },
                "validation": {
                    "by_regime": {
                        name: {
                            "groups": {"ssh": {"ratio_to_persistence": validation}}
                        }
                        for name, validation in zip(("S0", "S1", "S2"), (0.9, 1.1, 0.9))
                    }
                },
            }
        )
    autocorrelation = {
        "state_rms": {
            "temperature": {"effective_samples_total": 8.0},
            "ssh": {"effective_samples_total": 7.0},
        }
    }
    decision = data_expansion_decision(
        contract,
        chronology,
        [True, True, True],
        per_seed,
        autocorrelation,
        ("S0", "S1", "S2"),
    )
    assert decision["expansion_authorized"]
    assert decision["positive_generalization_regime_count"] == 2

    chronology[-1] = {"validation_worst_group_ratio": 1.25}
    rejected = data_expansion_decision(
        contract,
        chronology,
        [True, True, True],
        per_seed,
        autocorrelation,
        ("S0", "S1", "S2"),
    )
    assert not rejected["expansion_authorized"]
    assert not rejected["checks"]["strictly_improving_chronology"]
