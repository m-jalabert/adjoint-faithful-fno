from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bire_repro.af_model_c_pushforward_objective import (
    load_pushforward_contract,
    select_pushforward_checkpoint,
    slow_field_pushforward_loss,
    surface_phihyd_error,
)
from bire_repro.af_pressure import (
    GRAVITY_M_S2,
    THERMAL_EXPANSION_PER_C,
)


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config" / "model_c_pushforward_objective_v1.json"


def test_surface_phihyd_error_matches_top_level_linear_eos() -> None:
    prediction = torch.zeros((1, 46, 2, 2))
    target = torch.zeros_like(prediction)
    scale = torch.ones(46)
    prediction[:, 30] = 2.0
    prediction[:, 45] = 0.5
    expected = (
        GRAVITY_M_S2 * 0.5
        - 25.0 * GRAVITY_M_S2 * THERMAL_EXPANSION_PER_C * 2.0
    )
    actual = surface_phihyd_error(prediction, target, scale)
    assert torch.allclose(actual, torch.full_like(actual, expected))


def test_slow_field_loss_is_equal_dimensionless_mean() -> None:
    prediction = torch.zeros((1, 46, 2, 2))
    target = torch.zeros_like(prediction)
    scale = torch.ones(46)
    wet = torch.ones((1, 1, 2, 2))
    prediction[:, 30] = 2.0
    prediction[:, 45] = (
        3.0
        + 25.0
        * GRAVITY_M_S2
        * THERMAL_EXPANSION_PER_C
        * 2.0
    ) / GRAVITY_M_S2
    terms = slow_field_pushforward_loss(
        prediction,
        target,
        wet,
        scale,
        {"sst": 2.0, "phihyd_surface": 3.0},
    )
    assert torch.isclose(terms["sst"], torch.tensor(1.0))
    assert torch.isclose(
        terms["phihyd_surface"],
        torch.tensor(1.0),
    )
    assert torch.isclose(terms["mean"], torch.tensor(1.0))


def test_checkpoint_selection_requires_complete_gate() -> None:
    def summary(step: int, passed: bool, slow: float) -> dict[str, object]:
        return {
            "fine_tune_step": step,
            "checkpoint_gate": {
                "passed": passed,
                "worst_slow_field_lead_ratio": slow,
                "worst_primary_rmse_auc_ratio": slow - 0.1,
            },
            "ten_day_diagnostic": {
                "worst_per_regime_group_ratio": 0.8,
            },
        }

    passed = select_pushforward_checkpoint(
        [summary(480, False, 1.2), summary(960, True, 0.9)]
    )
    assert passed["passed"]
    assert passed["selected_fine_tune_step"] == 960
    assert (
        passed["classification"]
        == "training_only_pushforward_gate_passed"
    )

    rejected = select_pushforward_checkpoint(
        [summary(480, False, 1.2), summary(960, False, 1.1)]
    )
    assert not rejected["passed"]
    assert rejected["selected_fine_tune_step"] == 960
    assert (
        rejected["selected_for"]
        == "diagnostic_only_no_validation_authorized"
    )


def test_pushforward_contract_freezes_one_training_only_change() -> None:
    contract, path, digest = load_pushforward_contract(CONTRACT)
    assert path == CONTRACT
    assert len(digest) == 64
    assert contract["objective"]["pushforward_endpoint_days"] == [60, 90]
    assert contract["objective"]["pushforward_weight"] == 0.0025
    assert contract["fine_tune"]["source_optimizer_step"] == 14400
    assert contract["read_contract"]["validation_state_read"] is False
    assert contract["read_contract"]["inference_read"] is False
    assert np.isclose(
        contract["objective"]["climatology_rmse_scales"]["90"]["sst"],
        0.039431508630514145,
    )
