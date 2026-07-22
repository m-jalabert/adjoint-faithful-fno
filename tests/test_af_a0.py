"""Contract checks for the adapted Bire-architecture A0 gate."""

import pytest

from bire_repro.af_a0 import (
    A0DevelopmentConfig,
    A0FinalConfig,
    A0OverfitConfig,
    a0_architecture,
    records_for_pair_split,
    stratified_training_records,
)


def test_a0_architecture_contains_only_declared_benchmark_adaptations() -> None:
    architecture = a0_architecture()
    assert architecture.in_channels == 47
    assert architecture.out_channels == 46
    assert architecture.n_modes == (16, 16)
    assert architecture.hidden_channels == 128
    assert architecture.n_layers == 3


def test_a0_overfit_samples_are_deterministic_balanced_and_training_only() -> None:
    pair_codes = [1] * 12 + [2] * 3 + [3] * 2
    records = stratified_training_records(pair_codes, sample_count=20, seed=7)
    assert records == stratified_training_records(pair_codes, sample_count=20, seed=7)
    assert [sum(experiment == index for experiment, _ in records) for index in range(3)] == [7, 7, 6]
    assert all(pair_codes[time_index] == 1 for _, time_index in records)


def test_a0_overfit_contract_rejects_undeclared_sample_sizes() -> None:
    with pytest.raises(ValueError, match="20--100"):
        A0OverfitConfig(sample_count=101)


def test_development_records_use_the_sealed_chronological_split() -> None:
    pair_codes = [1, 1, 2, 2, 3]
    assert records_for_pair_split(pair_codes, 1) == ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))
    assert records_for_pair_split(pair_codes, 2) == ((0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3))
    with pytest.raises(ValueError, match="split_code"):
        records_for_pair_split(pair_codes, 4)
    assert A0DevelopmentConfig().epochs == 12


def test_final_a0_config_is_immutable_after_development_selection() -> None:
    assert A0FinalConfig().epochs == 10
    with pytest.raises(ValueError, match="frozen A0"):
        A0FinalConfig(epochs=12)
