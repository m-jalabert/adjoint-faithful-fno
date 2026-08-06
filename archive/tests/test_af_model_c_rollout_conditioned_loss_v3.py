from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from bire_repro.af_model_c_rollout_conditioned_loss_v3 import (  # noqa: E402
    ProjectedIncrementModel,
    infer_experiments_from_static,
    load_rollout_conditioned_contract,
    project_normalized_increment,
    projected_unroll,
    rollout_conditioned_endpoint,
    slow_increment_bias_penalty,
)


class CountingIncrement(torch.nn.Module):
    def __init__(self, channels: int = 46) -> None:
        super().__init__()
        self.channels = channels
        self.calls = 0
        self.gain = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.gain * features[:, : self.channels]


def _projection_inputs(
    batch: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    wet = torch.ones(1, 1, 4, 5)
    area = wet / wet.sum()
    experiments = torch.arange(batch, dtype=torch.long) % 3
    targets = torch.stack(
        (
            torch.linspace(-0.03, 0.01, 15),
            torch.linspace(-0.02, 0.02, 15),
            torch.linspace(-0.01, 0.03, 15),
        )
    )
    return experiments, area, wet, targets


def test_projection_enforces_means_and_retains_gradients() -> None:
    experiments, area, wet, targets = _projection_inputs()
    increment = torch.randn(3, 46, 4, 5, requires_grad=True)
    projected = project_normalized_increment(
        increment,
        experiments,
        area,
        wet,
        targets,
    )
    temperature_mean = (projected[:, 30:45] * area).sum((-2, -1))
    ssh_mean = (projected[:, 45:46] * area).sum((-2, -1))
    assert torch.allclose(
        temperature_mean,
        targets[experiments],
        atol=2.0e-7,
        rtol=0.0,
    )
    assert torch.allclose(ssh_mean, torch.zeros_like(ssh_mean), atol=2.0e-7)
    projected.square().mean().backward()
    assert increment.grad is not None
    assert torch.isfinite(increment.grad).all()


def test_static_signature_inference_and_projected_wrapper() -> None:
    experiments, area, wet, targets = _projection_inputs()
    signatures = torch.stack(
        (
            torch.full((4, 5), -1.0),
            torch.zeros(4, 5),
            torch.full((4, 5), 1.0),
        )
    )
    features = torch.zeros(3, 51, 4, 5)
    features[:, 46] = signatures
    inferred = infer_experiments_from_static(features, signatures, area)
    assert torch.equal(inferred, experiments)
    raw = CountingIncrement()
    wrapped = ProjectedIncrementModel(raw, area, wet, targets, signatures)
    projected = wrapped(features)
    temperature_mean = (projected[:, 30:45] * area).sum((-2, -1))
    assert torch.allclose(
        temperature_mean,
        targets[experiments],
        atol=2.0e-7,
        rtol=0.0,
    )
    assert raw.calls == 1


def test_rollout_conditioned_endpoint_has_one_gradient_bearing_call() -> None:
    experiments, area, wet, targets = _projection_inputs(batch=2)
    model = CountingIncrement()
    features = torch.randn(2, 51, 4, 5)
    endpoint = rollout_conditioned_endpoint(
        model,
        features,
        experiments,
        area,
        wet,
        targets,
        endpoint_step=7,
    )
    assert model.calls == 7
    endpoint.mean().backward()
    assert model.gain.grad is not None
    assert torch.isfinite(model.gain.grad)


def test_projected_unroll_and_bias_penalty_semantics() -> None:
    experiments, area, wet, targets = _projection_inputs(batch=2)
    model = CountingIncrement()
    features = torch.randn(2, 51, 4, 5)
    rollout = projected_unroll(
        model,
        features,
        experiments,
        area,
        wet,
        targets,
        3,
    )
    assert rollout.shape == (2, 3, 46, 4, 5)
    assert model.calls == 3

    predicted = torch.zeros(2, 46, 4, 5)
    target = torch.zeros_like(predicted)
    predicted[:, 30:] = 0.25
    scale = torch.ones(46)
    positive = slow_increment_bias_penalty(
        predicted,
        target,
        wet,
        scale,
    )
    assert float(positive) == pytest.approx(0.25**2)
    predicted[1, 30:] = -0.25
    cancelled = slow_increment_bias_penalty(
        predicted,
        target,
        wet,
        scale,
    )
    assert float(cancelled) == pytest.approx(0.0, abs=1.0e-12)


def test_contract_freezes_the_primary_one_state_loss_v3() -> None:
    root = Path(__file__).resolve().parents[1]
    contract, _, _ = load_rollout_conditioned_contract(
        root / "config" / "model_c_rollout_conditioned_loss_v3.json",
        verify_source_files=False,
    )
    assert contract["objective"]["forecast_target_days"] == list(
        range(10, 91, 10)
    )
    assert contract["objective"]["differentiable_conditioned_calls"] == 1
    assert contract["objective"]["slow_bias_penalty_weight"] == 0.01
    assert contract["fine_tune"]["maximum_steps"] == 2880
    assert contract["read_contract"]["validation_state_read"] is False
