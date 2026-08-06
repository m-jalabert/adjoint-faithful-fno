from __future__ import annotations

import torch

from bire_repro.af_model_c_anomaly_direct_group_heads import (
    GROUP_SIZES,
    GroupHeadsArchitecture,
    GroupHeadsControl,
    build_group_head_model,
    select_control_checkpoint,
)


def test_group_head_architecture_preserves_source_trunk() -> None:
    architecture = GroupHeadsArchitecture()
    assert architecture.n_layers == 4
    assert architecture.domain_padding == 0.1
    assert architecture.hidden_channels == 128
    assert architecture.head_group_sizes == GROUP_SIZES
    assert sum(architecture.head_group_sizes) == 46


def test_group_head_forward_concatenates_state_order() -> None:
    model = build_group_head_model(
        GroupHeadsArchitecture(),
        GroupHeadsControl(),
    ).eval()
    value = torch.zeros(1, 51, 8, 8)
    with torch.no_grad():
        result = model(value)
    assert result.shape == (1, 46, 8, 8)
    assert torch.isfinite(result).all()
    assert [head[-1].out_channels for head in model.heads] == list(GROUP_SIZES)


def test_group_head_selection_preserves_failed_diagnostic_only() -> None:
    summaries = (
        {
            "fine_tune_step": 13440,
            "optimizer_step": 13440,
            "worst_mid_bottom_modewise_ratio_all_leads": 7.0,
            "worst_primary_relative_to_source": 1.04,
            "gate": {"pass": False},
        },
        {
            "fine_tune_step": 14400,
            "optimizer_step": 14400,
            "worst_mid_bottom_modewise_ratio_all_leads": 5.0,
            "worst_primary_relative_to_source": 1.07,
            "gate": {"pass": False},
        },
    )
    decision = select_control_checkpoint(summaries)
    assert decision["status"] == "no_group_head_checkpoint_passed"
    assert decision["best_diagnostic_optimizer_step"] == 14400
    assert (
        decision["next_action"]
        == "retain_original_model_and_freeze_slow_fast_split_control"
    )
