"""Contracts for the immutable trajectories-v2 MITgcm extensions."""

from pathlib import Path

import pytest

from bire_repro.af_trajectory_expansion import (
    dataset_pair_counts,
    load_expansion_contract,
    resolve_experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_trajectory_v2_contract_reproduces_split_counts() -> None:
    contract, path, digest = load_expansion_contract(
        PROJECT_ROOT / "config/trajectories_v2_expansion.json"
    )
    assert path.is_absolute()
    assert len(digest) == 64
    assert dataset_pair_counts(contract) == {
        "training": 5020,
        "validation": 260,
        "inference": 1150,
    }
    assert contract["dataset_v2_design"]["training_pairs_total"] == 15060
    assert contract["data_adequacy_decision"]["status"].startswith(
        "authorize_trajectories_v2"
    )


def test_trajectory_v2_array_mapping_is_explicit() -> None:
    assert [resolve_experiment(index) for index in range(3)] == ["S0", "S1", "S2"]
    with pytest.raises(ValueError, match="0, 1, or 2"):
        resolve_experiment(3)
