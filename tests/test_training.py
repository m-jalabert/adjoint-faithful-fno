from pathlib import Path

import numpy as np
import pytest

from bire_repro.training import (
    AutoregressivePairDataset,
    PointwiseNormalizer,
    SequenceSource,
    TrainingConfig,
    _architecture_from_config,
    _load_configuration,
    _training_from_config,
    forecast_loss,
)


def test_pointwise_normalizer_round_trip_and_stride():
    mean = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6)
    std = np.full_like(mean, 2.0)
    normalizer = PointwiseNormalizer(mean, std, epsilon=1.0e-5)
    values = mean + 4.0
    normalized = normalizer.normalize(values)
    assert np.allclose(normalizer.denormalize(normalized), values)
    coarse = normalizer.for_stride(2)
    assert coarse.mean.shape == (3, 2, 3)
    assert np.array_equal(coarse.mean, mean[..., ::2, ::2])


def test_training_protocol_defaults_are_locked():
    config = TrainingConfig()
    assert config.batch_size == 8
    assert config.seed == 1024
    assert config.learning_rates == (0.01, 0.001, 0.0005)
    assert config.adam_betas == (0.9, 0.95)
    assert config.pretrain_max_epochs == 30
    assert config.finetune_max_epochs == 15
    assert config.weight_decay == 0


def test_canonical_toml_maps_to_paper_fno_defaults():
    manifest = Path(__file__).resolve().parents[1] / "config" / "bire_a0_reference.toml"
    config, _ = _load_configuration(manifest)
    architecture = _architecture_from_config(config)
    training = _training_from_config(config)
    assert architecture.in_channels == 11
    assert architecture.hidden_channels == 128
    assert architecture.channel_mlp_channels == 512
    assert architecture.n_modes == (64, 64)
    assert training.learning_rates == (0.01, 0.001, 0.0005)


def test_lazy_pair_dataset_does_not_cross_split(tmp_path):
    torch = pytest.importorskip("torch")
    data = np.zeros((2, 12, 3, 2, 2), dtype=np.float32)
    for experiment in range(2):
        for day in range(12):
            data[experiment, day, 0] = day
            data[experiment, day, 1] = 10 + day
            data[experiment, day, 2] = 100 + experiment
    path = tmp_path / "state.npy"
    np.save(path, data)
    source = SequenceSource(path)
    stats = PointwiseNormalizer(
        np.zeros((3, 2, 2), dtype=np.float32),
        np.ones((3, 2, 2), dtype=np.float32),
    )
    dataset = AutoregressivePairDataset(
        source,
        stats,
        experiment_ids=(1, 2),
        start=2,
        stop=10,
        lag_days=2,
        steps=2,
        state_channels=2,
        wind_channel=2,
    )
    assert len(dataset) == 2 * (8 - 4)
    x, (first, second) = dataset[0]
    assert x.shape == (3, 2, 2)
    assert first.shape == second.shape == (2, 2, 2)
    assert torch.allclose(first[0], torch.full((2, 2), 4.0 / 1.00001))
    assert torch.allclose(second[0], torch.full((2, 2), 6.0 / 1.00001))
    assert dataset.samples[-1][1] + 4 < 10


def test_composite_loss_exact_formula():
    torch = pytest.importorskip("torch")
    prediction = torch.tensor([0.0, 2.0])
    target = torch.tensor([1.0, 0.0])
    # MSE = 2.5, MAE = 1.5.
    assert forecast_loss(prediction, target).item() == pytest.approx(2.515)
