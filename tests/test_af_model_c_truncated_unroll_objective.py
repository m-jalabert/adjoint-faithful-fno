from __future__ import annotations

import json

import pytest
import torch

from bire_repro.af_model_c_pushforward_objective import STATE_CHANNEL_COUNT
from bire_repro.af_model_c_truncated_unroll_objective import (
    TRUNCATED_VERSION,
    load_truncated_contract,
    truncated_pushforward_window,
    truncated_slow_field_loss,
    validate_duration_source_payload,
)


class _Increment(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.01))
        self.calls = 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.weight * features[:, :STATE_CHANNEL_COUNT]


def test_truncated_window_detaches_start_and_keeps_three_call_graph() -> None:
    model = _Increment()
    features = torch.ones(2, STATE_CHANNEL_COUNT + 5, 3, 4)
    wet = torch.ones(1, 1, 3, 4)
    base = torch.ones(
        2,
        3,
        STATE_CHANNEL_COUNT,
        3,
        4,
        requires_grad=True,
    )
    predictions = truncated_pushforward_window(
        model,
        features,
        wet,
        base,
        endpoint_steps=(7, 8, 9),
    )
    assert predictions.shape == (
        2,
        3,
        STATE_CHANNEL_COUNT,
        3,
        4,
    )
    assert model.calls == 6
    predictions.sum().backward()
    assert model.weight.grad is not None
    assert model.weight.grad.item() > 0
    assert base.grad is None


def test_truncated_slow_loss_averages_three_finite_endpoints() -> None:
    predictions = torch.zeros(2, 3, STATE_CHANNEL_COUNT, 3, 4)
    futures = torch.ones(2, 9, STATE_CHANNEL_COUNT, 3, 4)
    wet = torch.ones(1, 1, 3, 4)
    state_scale = torch.ones(STATE_CHANNEL_COUNT)
    scales = {
        str(day): {"sst": 2.0, "phihyd_surface": 3.0}
        for day in range(10, 100, 10)
    }
    losses = truncated_slow_field_loss(
        predictions,
        futures,
        wet,
        state_scale,
        scales,
        endpoint_steps=(4, 5, 6),
    )
    assert set(losses) == {"mean", "sst", "phihyd_surface"}
    assert torch.isfinite(losses["mean"])
    assert losses["sst"].item() == pytest.approx(0.5)
    assert losses["mean"].item() == pytest.approx(
        0.5 * (losses["sst"].item() + losses["phihyd_surface"].item())
    )


def test_duration_source_uses_total_fine_tune_step_schema() -> None:
    payload = {
        "version": "model_c_pushforward_duration_v1",
        "total_fine_tune_step": 5760,
        "base_loss_contract_sha256": (
            "19000a1426ea928db7799c82a73ce071a"
            "874911eb7e1df50bd276582ec30b5f9"
        ),
        "model_state_dict": {},
        "architecture": {"hidden_channels": 128},
    }
    validate_duration_source_payload(
        payload,
        architecture={"hidden_channels": 128},
    )
    changed = dict(payload)
    changed["fine_tune_step"] = changed.pop("total_fine_tune_step")
    with pytest.raises(
        RuntimeError,
        match="source checkpoint contract changed",
    ):
        validate_duration_source_payload(changed)


def test_truncated_contract_rejects_changed_gradient_horizon(
    tmp_path,
) -> None:
    contract = {
        "version": TRUNCATED_VERSION,
        "contract_status": (
            "frozen_after_operational_source_schema_fix_before_truncated_unroll_metrics"
        ),
        "objective": {
            "base_loss_version": "v1",
            "supervised_windows_days": [[40, 50, 60], [70, 80, 90]],
            "window_schedule": (
                "alternate_40_to_60_and_70_to_90_by_optimizer_step"
            ),
            "differentiable_model_calls": 2,
            "pre_window_state": "detached_no_grad_rollout_from_day30",
            "endpoint_reduction": (
                "equal_mean_over_three_endpoints_and_two_slow_fields"
            ),
            "correction_weight": 0.0025,
            "slow_fields": ["sst", "phihyd_surface"],
        },
        "fine_tune": {
            "source_fine_tune_step": 5760,
            "maximum_steps": 1920,
            "checkpoint_steps": [480, 960, 1440, 1920],
            "batch_size": 4,
            "effective_batch_size": 4,
            "learning_rate": 0.00002,
            "adam_betas": [0.9, 0.95],
            "weight_decay": 0.00001,
        },
        "read_contract": {
            "training_pair_code": 1,
            "training_state_read": True,
            "validation_state_read": False,
            "inference_read": False,
            "intermediate_wind_read": False,
            "response_read": False,
            "adjoint_read": False,
        },
        "source_hashes": {},
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="objective changed"):
        load_truncated_contract(path)
