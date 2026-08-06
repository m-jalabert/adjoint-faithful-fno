from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from bire_repro.af_model_c_checkpoint_replay_audit import (
    checkpoint_audit_decision,
    checkpoint_gate_summary,
    load_checkpoint_replay_contract,
    numeric_tree_max_abs_difference,
)
from bire_repro.af_model_c_successor_validation import (
    LEAD_DAYS,
    PRIMARY_FIELDS,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config" / "model_c_checkpoint_replay_audit_v1.json"


def test_checkpoint_replay_contract_freezes_exact_training_only_audit() -> None:
    contract, path, digest = load_checkpoint_replay_contract(CONTRACT)
    assert path == CONTRACT
    assert len(digest) == 64
    assert contract["replay"]["checkpoint_steps"] == [
        11520,
        13440,
        14400,
        14880,
        15120,
        15360,
    ]
    assert contract["records"]["records_total"] == 540
    assert contract["read_contract"]["validation_state_read"] is False
    assert contract["read_contract"]["inference_read"] is False
    assert contract["consequences"]["fresh_validation_authorized"] is False


def _passing_curves() -> dict[str, dict[str, dict[str, object]]]:
    curves: dict[str, dict[str, dict[str, object]]] = {}
    for field in PRIMARY_FIELDS:
        curves[field] = {}
        for baseline in ("persistence", "climatology"):
            curves[field][baseline] = {
                "rmse_auc_ratio": 0.8,
                "rmse_ratio_by_lead": [0.8] * len(LEAD_DAYS),
            }
    return curves


def _diagnostic(worst: float = 0.7) -> dict[str, object]:
    return {
        "every_regime_and_group_beats_persistence": True,
        "worst_per_regime_group_ratio": worst,
    }


def test_checkpoint_gate_requires_slow_fields_below_both_at_every_lead() -> None:
    contract, _, _ = load_checkpoint_replay_contract(
        CONTRACT,
        verify_source_files=False,
    )
    metrics = {
        "finite": np.ones((6, len(LEAD_DAYS)), dtype=np.uint8),
        "normalized_land_max_abs": np.zeros(
            (6, len(LEAD_DAYS)),
            dtype=np.float32,
        ),
    }
    passed = checkpoint_gate_summary(
        _passing_curves(),
        _diagnostic(),
        metrics,
        contract["checkpoint_gate"],
    )
    assert passed["passed"]
    assert passed["worst_primary_rmse_auc_ratio"] == 0.8
    assert passed["worst_slow_field_lead_ratio"] == 0.8

    failed_curves = _passing_curves()
    failed_curves["sst"]["persistence"]["rmse_ratio_by_lead"] = [
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        1.01,
        0.8,
        0.8,
        0.8,
    ]
    failed = checkpoint_gate_summary(
        failed_curves,
        _diagnostic(),
        metrics,
        contract["checkpoint_gate"],
    )
    assert not failed["passed"]
    assert not failed["slow_field_all_lead_checks"]["sst"]["persistence"]


def test_checkpoint_decision_distinguishes_selection_and_objective() -> None:
    base = {
        "optimizer_step": 13440,
        "ten_day_diagnostic": _diagnostic(0.8),
        "checkpoint_gate": {
            "passed": False,
            "worst_slow_field_lead_ratio": 1.2,
            "worst_primary_rmse_auc_ratio": 1.1,
        },
    }
    passing = deepcopy(base)
    passing["optimizer_step"] = 14400
    passing["checkpoint_gate"] = {
        "passed": True,
        "worst_slow_field_lead_ratio": 0.9,
        "worst_primary_rmse_auc_ratio": 0.85,
    }
    selection = checkpoint_audit_decision(
        [base, passing],
        exact_replay_passed=True,
    )
    assert (
        selection["classification"]
        == "checkpoint_selection_only_correction_supported"
    )
    assert selection["selected_optimizer_step"] == 14400

    objective = checkpoint_audit_decision(
        [base],
        exact_replay_passed=True,
    )
    assert objective["classification"] == "objective_correction_required"
    assert objective["diagnostic_best_optimizer_step"] == 13440

    blocked = checkpoint_audit_decision(
        [passing],
        exact_replay_passed=False,
    )
    assert blocked["classification"] == "replay_provenance_failed"
    assert blocked["selected_optimizer_step"] is None


def test_numeric_tree_exactness_detects_nested_difference() -> None:
    left = {"history": [{"step": 1, "loss": 0.5}], "passed": True}
    assert numeric_tree_max_abs_difference(left, deepcopy(left)) == 0.0
    right = {"history": [{"step": 1, "loss": 0.5001}], "passed": True}
    assert np.isclose(
        numeric_tree_max_abs_difference(left, right),
        0.0001,
    )
