from __future__ import annotations

from bire_repro.analysis.af_model_c_bire_regularization_aggregate import (
    select_cross_arm,
)


def _report(arm: str, step: int, spectrum: float, primary: float, passed: bool):
    return {
        "status": "complete",
        "arm": {"arm_id": arm},
        "selection_decision": {
            "arm_training_gate_passed": passed,
            "selected_optimizer_step": step,
        },
        "evaluation_summaries": [
            {
                "optimizer_step": step,
                "worst_mid_bottom_modewise_ratio_all_leads": spectrum,
                "worst_primary_relative_to_source": primary,
                "gate": {"pass": passed},
            }
        ],
    }


def test_cross_arm_selects_best_passing_arm() -> None:
    reports = (
        _report("layernorm", 14400, 3.8, 1.08, True),
        _report("dropout", 14880, 3.1, 1.09, True),
        _report("layernorm_dropout", 13440, 6.0, 1.01, False),
    )
    decision = select_cross_arm(reports)
    assert decision["status"] == "regularization_arm_selected"
    assert decision["selected_arm"] == "dropout"
    assert decision["selected_optimizer_step"] == 14880
    assert decision["retain_original_model"] is False


def test_cross_arm_retains_source_and_advances_when_none_pass() -> None:
    reports = (
        _report("layernorm", 14400, 9.7, 1.19, False),
        _report("dropout", 14400, 37.6, 1.58, False),
        _report("layernorm_dropout", 14880, 19.5, 1.64, False),
    )
    decision = select_cross_arm(reports)
    assert decision["status"] == "no_regularization_arm_passed"
    assert decision["selected_arm"] is None
    assert decision["best_diagnostic_arm"] == "layernorm"
    assert decision["retain_original_model"] is True
    assert (
        decision["next_action"]
        == "freeze_three_layer_no_padding_training_only_control"
    )
