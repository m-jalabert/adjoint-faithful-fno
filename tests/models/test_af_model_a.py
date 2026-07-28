"""Contract checks for the modern state-only Model A baseline."""

import pytest

from bire_repro.af_model_a import (
    MODEL_A_INPUT_CHANNELS,
    ModelAArchitecture,
    ModelADevelopmentConfig,
    ModelAFinalConfig,
    ModelAOverfitConfig,
    STATE_CHANNEL_COUNT,
    model_a_architecture,
    stratified_training_records,
)


def test_model_a_architecture_is_the_declared_neuraloperator_starting_point() -> None:
    architecture = model_a_architecture()
    assert architecture.in_channels == MODEL_A_INPUT_CHANNELS == 51
    assert architecture.out_channels == STATE_CHANNEL_COUNT == 46
    assert architecture.n_modes == (16, 16)
    assert architecture.hidden_channels == 32
    assert architecture.n_layers == 4
    assert architecture.domain_padding == 0.10
    assert architecture.local_kernel_size == 3
    assert architecture.positional_embedding == "grid"
    assert architecture.use_channel_mlp
    assert architecture.factorization is None


def test_model_a_starting_architecture_does_not_silently_change_the_declared_width() -> None:
    with pytest.raises(ValueError, match="32 hidden"):
        ModelAArchitecture(hidden_channels=64)


def test_model_a_overfit_records_are_balanced_and_training_only() -> None:
    pair_codes = [1] * 12 + [2] * 3 + [3] * 2
    records = stratified_training_records(pair_codes, sample_count=20, seed=7)
    assert records == stratified_training_records(pair_codes, sample_count=20, seed=7)
    assert [sum(experiment == index for experiment, _ in records) for index in range(3)] == [7, 7, 6]
    assert all(pair_codes[time_index] == 1 for _, time_index in records)


def test_model_a_overfit_rejects_out_of_contract_sample_count() -> None:
    with pytest.raises(ValueError, match="20--100"):
        ModelAOverfitConfig(sample_count=101)


def test_model_a_development_configuration_is_sealed_to_the_chronological_gate() -> None:
    assert ModelADevelopmentConfig().epochs == 12
    with pytest.raises(ValueError, match="learning_rate"):
        ModelADevelopmentConfig(learning_rate=0.0)


def test_model_a_final_configuration_is_locked_to_the_development_minimum() -> None:
    assert ModelAFinalConfig().epochs == 10
    with pytest.raises(ValueError, match="frozen Model A"):
        ModelAFinalConfig(epochs=12)


def test_model_a_checkpoint_state_is_reloadable_without_neuraloperator_metadata() -> None:
    torch = pytest.importorskip("torch")
    from bire_repro.af_model_a import _checkpoint_state_dict, build_model_a

    model = build_model_a()
    state = _checkpoint_state_dict(model)
    assert "_metadata" not in state
    clone = build_model_a()
    clone.load_state_dict(state)
    assert torch.equal(next(model.parameters()), next(clone.parameters()))
