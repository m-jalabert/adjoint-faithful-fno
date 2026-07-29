from __future__ import annotations

import torch

from bire_repro.af_model_c_anomaly_direct_bire_regularization_controls import (
    PointwiseChannelLayerNorm,
    RegularizationArm,
    select_arm_checkpoint,
)


def test_pointwise_layer_norm_normalizes_channels_per_grid_point() -> None:
    layer = PointwiseChannelLayerNorm(4)
    value = torch.arange(2 * 4 * 3 * 2, dtype=torch.float32).reshape(2, 4, 3, 2)
    result = layer(value)
    assert result.shape == value.shape
    assert torch.allclose(result.mean(dim=1), torch.zeros((2, 3, 2)), atol=1.0e-6)
    assert torch.allclose(
        result.var(dim=1, unbiased=False),
        torch.ones((2, 3, 2)),
        atol=1.0e-5,
    )


def test_regularization_arms_fix_archived_dropout() -> None:
    assert RegularizationArm("layernorm", True, 0.0).channel_mlp_dropout == 0.0
    assert RegularizationArm("dropout", False, 0.5).channel_mlp_dropout == 0.5
    assert (
        RegularizationArm("layernorm_dropout", True, 0.5).pointwise_layer_norm
        is True
    )


def test_arm_selection_uses_best_passing_spectrum() -> None:
    summaries = (
        {
            "fine_tune_step": 3840,
            "optimizer_step": 3840,
            "worst_mid_bottom_modewise_ratio_all_leads": 3.7,
            "worst_primary_relative_to_source": 1.01,
            "gate": {"pass": True},
        },
        {
            "fine_tune_step": 7680,
            "optimizer_step": 7680,
            "worst_mid_bottom_modewise_ratio_all_leads": 2.8,
            "worst_primary_relative_to_source": 1.05,
            "gate": {"pass": True},
        },
    )
    decision = select_arm_checkpoint(summaries)
    assert decision["arm_training_gate_passed"] is True
    assert decision["selected_optimizer_step"] == 7680


def test_arm_selection_keeps_best_diagnostic_when_none_pass() -> None:
    summaries = (
        {
            "fine_tune_step": 3840,
            "optimizer_step": 3840,
            "worst_mid_bottom_modewise_ratio_all_leads": 8.0,
            "worst_primary_relative_to_source": 0.9,
            "gate": {"pass": False},
        },
        {
            "fine_tune_step": 7680,
            "optimizer_step": 7680,
            "worst_mid_bottom_modewise_ratio_all_leads": 6.0,
            "worst_primary_relative_to_source": 0.95,
            "gate": {"pass": False},
        },
    )
    decision = select_arm_checkpoint(summaries)
    assert decision["arm_training_gate_passed"] is False
    assert decision["best_diagnostic_optimizer_step"] == 7680
