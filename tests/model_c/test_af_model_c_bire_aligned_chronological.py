from __future__ import annotations

import json
from pathlib import Path

import pytest

from bire_repro.af_model_c import MODEL_C_LOSS_V1_CONTRACT_SHA256
from bire_repro.af_model_c_bire_aligned_chronological import (
    CHECKPOINT_STEPS,
    CONTRACT_STATUS,
    FROZEN_TRAINING_FIELDS,
    LEAD_DAYS,
    LONG_LEADS,
    PRIMARY_FIELDS,
    SHORT_LEADS,
    SHORT_SKILL_TOLERANCE,
    VERSION,
    ChronologicalArmError,
    lead_auc,
    load_contract,
    select_by_validation,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config/model_c_bire_aligned_loss_recovery_chronological_v1.json"
PARENT = ROOT / "config/model_c_bire_aligned_loss_recovery_v1.json"
SBATCH = ROOT / "slurm/models/c/af_model_c_bire_aligned_chronological.sbatch"


def test_contract_declares_the_chronological_protocol() -> None:
    contract, resolved, digest = load_contract(CONTRACT)
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64
    assert contract["version"] == VERSION
    assert contract["contract_status"] == CONTRACT_STATUS
    split = contract["split"]
    assert split["train"] == [0, 5040]
    assert split["validation"] == [5130, 5760]
    assert split["test"] == [5850, 7200]
    assert split["training_starts_per_regime"] == 5010
    assert contract["normalization"]["recomputed_from"] == "train_only_0_5039"
    assert contract["checkpoint_selection"]["short_skill_tolerance"] == 1.05


def test_the_model_is_identical_to_the_parent() -> None:
    contract, _, _ = load_contract(CONTRACT)
    parent = json.loads(PARENT.read_text())
    assert contract["architecture"] == parent["architecture"]
    assert contract["loss"] == parent["loss"]
    assert contract["loss"]["contract_sha256"] == MODEL_C_LOSS_V1_CONTRACT_SHA256
    for field in FROZEN_TRAINING_FIELDS:
        assert contract["training"][field] == parent["training"][field], field
    assert tuple(contract["training"]["checkpoint_steps"]) == CHECKPOINT_STEPS


def test_normalizer_must_be_recomputed_not_inherited() -> None:
    """Reusing the parent normalizer would leak 5040--6209 into training."""

    contract, _, _ = load_contract(CONTRACT)
    assert "normalization" not in contract["sources"]
    recomputed = contract["normalization"]["recomputed_quantities"]
    for quantity in (
        "pointwise_mean",
        "pointwise_scale",
        "channel_scale_floors",
        "per_regime_pointwise_climatology",
        "pointwise_increment_scale",
    ):
        assert quantity in recomputed
    assert "wind_stress_normalization" in contract["normalization"]["not_recomputed"]


def test_read_contract_opens_validation_and_seals_test() -> None:
    read = load_contract(CONTRACT)[0]["read_contract"]
    assert read["training_state"] is True
    assert read["validation_state"] is True
    for sealed in (
        "test_state",
        "held_s0_state",
        "intermediate_wind_state",
        "response_state",
        "adjoint_state",
    ):
        assert read[sealed] is False


def test_duration_is_explicitly_not_extended() -> None:
    contract, _, _ = load_contract(CONTRACT)
    assert contract["training"]["maximum_steps"] == 7680
    assert "duration_not_extended" in contract["training"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("split", "train"), [0, 6000]),
        (("split", "module_version"), "other"),
        (("normalization", "recomputed_from"), "shared_seed_normalizer"),
        (("checkpoint_selection", "short_skill_tolerance"), 1.5),
        (("checkpoint_selection", "evaluated_on"), "training_split_records"),
        (("training", "initial_learning_rate"), 0.01),
        (("training", "rollout_steps"), 2),
        (("architecture", "n_layers"), 4),
        (("read_contract", "test_state"), True),
    ],
)
def test_rejects_drift_in_any_declared_quantity(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    contract = json.loads(CONTRACT.read_text())
    target = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    written = tmp_path / "contract.json"
    written.write_text(json.dumps(contract))
    with pytest.raises(ChronologicalArmError):
        load_contract(written, verify_sources=False)


def test_lead_windows_are_the_declared_ones() -> None:
    assert SHORT_LEADS[0] == 10 and SHORT_LEADS[-1] == 90
    assert LONG_LEADS[0] == 90 and LONG_LEADS[-1] == 360
    assert PRIMARY_FIELDS == ("surface_speed", "sst", "phihyd_surface")
    assert SHORT_SKILL_TOLERANCE == 1.05


def test_lead_auc_integrates_the_declared_windows() -> None:
    """This ran only inside a closure and shipped a NumPy 2 removal to the GPU."""

    leads = list(LEAD_DAYS)
    flat = [2.0] * len(leads)
    # constant 2.0 over 10..90 -> 2 * 80
    assert lead_auc(flat, leads, SHORT_LEADS) == pytest.approx(160.0)
    # constant 2.0 over 90..360 -> 2 * 270
    assert lead_auc(flat, leads, LONG_LEADS) == pytest.approx(540.0)
    # a linear ramp integrates to the trapezoid area
    ramp = [float(l) for l in leads]
    assert lead_auc(ramp, leads, SHORT_LEADS) == pytest.approx((10 + 90) / 2 * 80)
    with pytest.raises(ValueError):
        lead_auc(flat[:-1], leads, SHORT_LEADS)
    with pytest.raises(ValueError):
        lead_auc(flat, leads, (10,))


def _summary(step, short, long):
    return {
        "optimizer_step": step,
        "short_auc_10_90": dict(zip(PRIMARY_FIELDS, short)),
        "long_ratio_to_climatology": dict(zip(PRIMARY_FIELDS, long)),
    }


def test_selection_rule_minimises_long_error_within_the_short_guard() -> None:
    summaries = [
        # best short skill everywhere, but poor long-lead behaviour
        _summary(1920, [1.00, 1.00, 1.00], [3.0, 3.0, 3.0]),
        # 4% worse short skill -> inside the 5% guard, much better long-lead
        _summary(3840, [1.04, 1.04, 1.04], [1.5, 1.5, 1.5]),
        # 20% worse short skill -> excluded despite the best long-lead value
        _summary(5760, [1.20, 1.20, 1.20], [1.0, 1.0, 1.0]),
    ]
    decision = select_by_validation(summaries)
    assert decision["branch"] == "primary_rule"
    assert decision["selected_optimizer_step"] == 3840
    assert decision["feasible_steps"] == [1920, 3840]
    assert decision["selected_worst_long_ratio_to_climatology"] == pytest.approx(1.5)


def test_selection_rule_uses_the_worst_field_not_the_mean() -> None:
    summaries = [
        _summary(1920, [1.0, 1.0, 1.0], [1.0, 1.0, 9.0]),
        _summary(3840, [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]),
    ]
    assert select_by_validation(summaries)["selected_optimizer_step"] == 3840


def test_selection_falls_back_when_no_checkpoint_meets_the_guard() -> None:
    """The guard is per field, so the feasible set can be empty."""

    summaries = [
        _summary(1920, [1.00, 2.00, 1.00], [1.0, 1.0, 1.0]),
        _summary(3840, [2.00, 1.00, 1.00], [2.0, 2.0, 2.0]),
    ]
    decision = select_by_validation(summaries)
    assert decision["branch"].startswith("declared_fallback")
    assert decision["feasible_steps"] == []
    assert decision["selected_optimizer_step"] in (1920, 3840)


def test_launcher_invokes_this_arms_module_and_contract() -> None:
    text = SBATCH.read_text()
    assert text.count("bire_repro.af_model_c_bire_aligned_chronological") == 2
    assert "bire_repro.af_model_c_bire_aligned_loss_recovery" not in text
    assert CONTRACT.name in text
