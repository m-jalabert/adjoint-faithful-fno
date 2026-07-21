import math

import pytest

from bire_repro import fno


def test_paper_architecture_defaults_are_locked():
    config = fno.PaperFNOConfig()
    assert config.in_channels == 11
    assert config.out_channels == 10
    assert config.lifting_channels == 256
    assert config.hidden_channels == 128
    assert config.channel_mlp_channels == 512
    assert config.n_layers == 3
    assert config.n_modes == (64, 64)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((248, 248), (64, 64)), ((31, 31), (30, 30)), ((32, 17), (32, 16))],
)
def test_effective_modes_explicitly_crop_for_resolution(shape, expected):
    assert fno.effective_n_modes((64, 64), shape) == expected


def test_missing_dependency_message_is_actionable():
    if fno.torch is not None and fno.SpectralConv is not None:
        pytest.skip("ML dependencies are available")
    with pytest.raises(RuntimeError, match="PyTorch and NeuralOperator 2.0.0"):
        fno.build_paper_fno()


def test_tiny_cpu_model_shape_gradient_position_and_coarse_crop():
    torch = pytest.importorskip("torch")
    pytest.importorskip("neuralop")

    config = fno.PaperFNOConfig(
        lifting_channels=8,
        hidden_channels=4,
        projection_channels=8,
        channel_mlp_channels=16,
        n_layers=2,
        n_modes=(8, 8),
    )
    model = fno.PaperFNO2d(config)
    values = torch.randn(2, 11, 12, 10, requires_grad=True)
    output = model(values)
    assert output.shape == (2, 10, 12, 10)
    output.square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert all(block.spectral.separable for block in model.blocks)

    coarse = model(torch.randn(1, 11, 7, 7))
    assert coarse.shape == (1, 10, 7, 7)
    assert all(block.last_effective_n_modes == (6, 6) for block in model.blocks)

    x_coord, y_coord = fno.AlternatingSineCosinePosition2d.coordinates_like(
        torch.zeros(1, 11, 4, 4)
    )
    assert x_coord[0, 0, 0, 0].item() == pytest.approx(0.0)
    assert x_coord[0, 0, 0, 1].item() == pytest.approx(math.cos(math.pi / 8))
    assert x_coord[0, 0, 0, 2].item() == pytest.approx(math.sin(math.pi / 4))
    assert torch.equal(x_coord[0, 0, 0], x_coord[0, 0, -1])
    assert torch.equal(y_coord[0, 0, :, 0], y_coord[0, 0, :, -1])

