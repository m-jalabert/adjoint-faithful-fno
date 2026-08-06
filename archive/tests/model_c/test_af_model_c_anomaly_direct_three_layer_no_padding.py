from __future__ import annotations

from bire_repro.af_model_c_anomaly_direct_three_layer_no_padding import (
    architecture_from_contract,
    select_control_checkpoint,
)


def test_architecture_control_fixes_three_layers_and_no_padding() -> None:
    architecture = architecture_from_contract(
        {
            "architecture": {
                "in_channels": 51,
                "out_channels": 46,
                "n_modes": [24, 16],
                "hidden_channels": 128,
                "n_layers": 3,
                "lifting_channel_ratio": 2,
                "projection_channel_ratio": 2,
                "channel_mlp_expansion": 4.0,
                "domain_padding": None,
                "positional_embedding": "grid",
                "use_channel_mlp": True,
                "local_kernel_size": 3,
                "fno_block_precision": "full",
                "factorization": None,
            }
        }
    )
    assert architecture.n_layers == 3
    assert architecture.domain_padding is None


def test_control_selection_keeps_best_diagnostic_when_none_pass() -> None:
    summaries = (
        {
            "fine_tune_step": 13440,
            "optimizer_step": 13440,
            "worst_mid_bottom_modewise_ratio_all_leads": 8.0,
            "worst_primary_relative_to_source": 1.05,
            "gate": {"pass": False},
        },
        {
            "fine_tune_step": 14400,
            "optimizer_step": 14400,
            "worst_mid_bottom_modewise_ratio_all_leads": 6.0,
            "worst_primary_relative_to_source": 1.08,
            "gate": {"pass": False},
        },
    )
    decision = select_control_checkpoint(summaries)
    assert decision["status"] == "no_architecture_control_checkpoint_passed"
    assert decision["control_training_gate_passed"] is False
    assert decision["best_diagnostic_optimizer_step"] == 14400


def test_control_selection_requires_full_inherited_gate() -> None:
    summaries = (
        {
            "fine_tune_step": 14880,
            "optimizer_step": 14880,
            "worst_mid_bottom_modewise_ratio_all_leads": 3.5,
            "worst_primary_relative_to_source": 1.04,
            "gate": {"pass": True},
        },
    )
    decision = select_control_checkpoint(summaries)
    assert decision["status"] == "architecture_control_training_gate_passed"
    assert decision["control_training_gate_passed"] is True
    assert (
        decision["next_action"]
        == "freeze_replication_before_any_later_archive_read"
    )
