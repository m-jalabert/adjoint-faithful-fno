"""Tests for the rollout fine-tune's S0 figure suite and acceptance gate.

The package's value is that it is comparable with the step-15,360 package it was
fine-tuned from, so most of these assert that the evaluation protocol is
identical and only the checkpoint differs.

Three tests carry the weight this suite adds over the earlier ones:

* ``_stepper`` is rewritten here because figure 6 pairs two checkpoints with
  *different* objective hashes for the first time in this project.  It is
  exercised against the real comparator checkpoint on disk.
* ``finalize`` fills the four contract fields training cannot know in advance.
  It must be idempotent and must refuse to overwrite a filled field, or it
  becomes a way to edit a frozen contract.
* ``long_rollout_gate`` is the only implementation of the 2,000-day half of the
  acceptance gate, so each threshold is checked at its boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bire_repro import af_model_c_bire_protocol_figures as suite
from bire_repro import af_model_c_bire_s0_figures as figures
from bire_repro.af_model_c import MODEL_C_LOSS_V1_CONTRACT_SHA256
from bire_repro.af_model_c_bire_protocol_rollout_ft import (
    BASELINE_OPTIMIZER_STEP,
    CHECKPOINT_STEPS,
    FINE_TUNE_LOSS_CONTRACT_SHA256,
    ROLLOUT_STEPS,
)
from bire_repro.af_model_c_bire_protocol_rollout_ft_figures import (
    COMPARATOR_STEP,
    CONTRACT_STATUS,
    DAY2000_STD_RATIO_RANGE,
    GATE_NAME,
    MAXIMUM_NORMALIZED_MAGNITUDE,
    MEMBER_COUNT,
    MINIMUM_STREAMFUNCTION_SV,
    MODEL_LABEL,
    PARENT_BINDINGS,
    PENDING,
    PENDING_PATHS,
    VERSION,
    BireProtocolRolloutFineTuneFigureError,
    FineTuneLabels,
    _SuiteBinding,
    _readme,
    declared_inference_starts,
    finalize,
    load_contract,
    long_rollout_gate,
    unfilled_fields,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config/model_c_bire_protocol_rollout_ft_s0_figures_v1.json"
COMPARED = ROOT / "config/model_c_bire_protocol_duration_s0_figures_v1.json"
TRAINING = ROOT / "config/model_c_bire_protocol_rollout_ft_v1.json"
SBATCH = ROOT / "slurm/models/c/af_model_c_bire_protocol_rollout_ft_figures.sbatch"

pytestmark = pytest.mark.skipif(
    not CONTRACT.is_file(), reason="the rollout fine-tune figure contract is absent"
)

SELECTED_STEP = CHECKPOINT_STEPS[-1]


def _filled(step: int = SELECTED_STEP) -> dict:
    """The contract as `finalize` would leave it after a completed run."""

    contract = json.loads(CONTRACT.read_text())
    contract["selected_model"]["optimizer_step"] = step
    for key in ("selected_checkpoint", "selected_normalization", "selected_report"):
        contract["artifacts"][key]["sha256"] = "b" * 64
    return contract


def _written(contract: dict, directory: Path, name: str = "filled.json") -> Path:
    """Write a contract copy under pytest's ``tmp_path``.

    Every caller loads it with ``verify_sources=False``, so nothing needs the
    repository layout; earlier arms' tests wrote these beside the real contracts
    and left hundreds of stray directories in ``config/``.
    """

    path = directory / name
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return path


# --------------------------------------------------------------------------
# Comparability with the package this model was fine-tuned from
# --------------------------------------------------------------------------


def test_the_evaluation_protocol_is_identical_to_the_compared_package() -> None:
    """Only the checkpoint may differ, or the two packages are not comparable."""

    mine = json.loads(CONTRACT.read_text())
    compared = json.loads(COMPARED.read_text())
    for key in ("member_count", "maximum_lead_days", "prediction_interval_days",
                "start_draw_order", "figure_names", "rmse_fields", "acc_fields",
                "regimes", "primary_regime", "inference_set", "start_seed",
                "figure3_lead_days", "figure7_lead_days", "long_lead_days"):
        assert mine["protocol"][key] == compared["protocol"][key], key
    assert mine["baselines"] == compared["baselines"]
    assert mine["truth"] == compared["truth"]
    assert mine["dataset"] == compared["dataset"]
    assert mine["artifacts"]["dataset_metadata"] == compared["artifacts"]["dataset_metadata"]
    assert mine["output"]["required"] == compared["output"]["required"]
    assert mine["comparability"]["byte_comparable_with_the_step_15360_figure_package"] is True


def test_starts_are_the_compared_packages_and_admit_2000_days_of_truth() -> None:
    starts = declared_inference_starts()
    assert starts.size == MEMBER_COUNT == 15
    assert np.array_equal(starts, suite.declared_inference_starts())
    assert starts.min() >= 6200 and int(starts.max()) + 2000 < 9000
    assert np.array_equal(starts, np.sort(starts))


def test_outputs_do_not_collide_with_the_compared_package() -> None:
    mine = json.loads(CONTRACT.read_text())
    compared = json.loads(COMPARED.read_text())
    for key in ("project_root", "scratch_root"):
        assert mine["output"][key] != compared["output"][key]
        assert "rollout_ft" in mine["output"][key]


# --------------------------------------------------------------------------
# The literal pre-train / fine-tune pair
# --------------------------------------------------------------------------


def test_figure6_is_a_literal_pretrain_finetune_pair(tmp_path) -> None:
    contract, _, _ = load_contract(_written(_filled(), tmp_path), verify_sources=False)
    assert contract["figure6"]["literal_pretrain_finetune_pair"] is True
    assert int(contract["figure6"]["comparator_optimizer_step"]) == COMPARATOR_STEP
    assert COMPARATOR_STEP == BASELINE_OPTIMIZER_STEP == 15360
    # The comparator is the checkpoint the fine-tune actually started from.
    assert "bire_protocol_duration_v1/selected.pt" in contract["artifacts"]["comparator_checkpoint"]["path"]
    assert "bire_protocol_rollout_ft_v1/selected.pt" in contract["artifacts"]["selected_checkpoint"]["path"]
    training = json.loads(TRAINING.read_text())
    assert (
        contract["artifacts"]["comparator_checkpoint"]["path"]
        == training["sources"]["initialization_checkpoint"]["path"]
    )
    assert (
        contract["artifacts"]["comparator_checkpoint"]["sha256"]
        == training["sources"]["initialization_checkpoint"]["sha256"]
    )
    # And every earlier package in this project could not say this.
    assert json.loads(COMPARED.read_text())["figure6"]["literal_pretrain_finetune_pair"] is False


def test_the_two_checkpoints_declare_different_objectives(tmp_path) -> None:
    contract, _, _ = load_contract(_written(_filled(), tmp_path), verify_sources=False)
    selected = contract["selected_model"]
    comparator = contract["comparator_model"]
    assert selected["base_loss_contract_sha256"] == FINE_TUNE_LOSS_CONTRACT_SHA256
    assert comparator["base_loss_contract_sha256"] == MODEL_C_LOSS_V1_CONTRACT_SHA256
    assert selected["base_loss_contract_sha256"] != comparator["base_loss_contract_sha256"]
    assert int(selected["rollout_steps"]) == ROLLOUT_STEPS == 6
    assert int(comparator["rollout_steps"]) == 3
    assert selected["architecture"] == json.loads(TRAINING.read_text())["architecture"]


@pytest.mark.skipif(
    not Path(
        "/bigscratch/mjalabert314/bire_james25_repro/af_fno/models/C/"
        "bire_protocol_duration_v1/selected.pt"
    ).is_file(),
    reason="the comparator checkpoint is not on this filesystem",
)
def test_the_stepper_verifies_each_checkpoint_against_its_own_objective() -> None:
    """The suite's single-hash check cannot serve a pre-train / fine-tune pair."""

    torch = pytest.importorskip("torch")
    from bire_repro.af_model_c_bire_protocol_rollout_ft_figures import _stepper

    contract = _filled()
    # The fine-tune's own normalizers are not written until it runs, and this
    # test is about the identity check, not the file layout; the parent's are
    # the same 46-channel arrays by construction.
    parent_normalization = (
        "/bigscratch/mjalabert314/bire_james25_repro/af_fno/models/C/"
        "bire_protocol_duration_v1/model_c_bire_protocol_duration_train_only_normalization.npz"
    )
    contract["artifacts"]["selected_normalization"]["path"] = parent_normalization
    device = torch.device("cpu")
    with np.load(parent_normalization) as stored:
        wet = np.asarray(stored["pointwise_scale"][0] > 0, dtype=bool)
    stepper = _stepper(contract, "comparator_checkpoint", device, wet, 0.0, 1.0)
    assert stepper.model is not None

    # The same file offered as the six-step selected checkpoint must be refused.
    contract["artifacts"]["selected_checkpoint"] = dict(
        contract["artifacts"]["comparator_checkpoint"]
    )
    contract["selected_model"]["optimizer_step"] = COMPARATOR_STEP
    with pytest.raises(BireProtocolRolloutFineTuneFigureError):
        _stepper(contract, "selected_checkpoint", device, wet, 0.0, 1.0)


def test_the_captions_name_the_fine_tune_pairing() -> None:
    with _SuiteBinding():
        with suite.RegimeLabels("S0", 0.1, SELECTED_STEP) as labels:
            assert isinstance(labels, FineTuneLabels)
            assert figures.METHOD_LABELS["model"] == MODEL_LABEL
            assert (
                labels.rewrite("S0 architecture-direction comparison")
                == "S0 three-step model vs six-step fine-tune"
            )
            assert labels.rewrite("Prior residual Model C") == (
                f"Before fine-tuning (step {COMPARATOR_STEP:,})"
            )
            assert labels.rewrite("Selected anomaly-direct Model C") == (
                f"After {ROLLOUT_STEPS}-step fine-tune (step {SELECTED_STEP:,})"
            )
            # The regime and wind rules the suite supplies must survive.
            assert r"$\tau_0=0.1$" in labels.rewrite(
                r"Control wind $\tau_0=0.1$ N m$^{-2}$"
            )
    assert figures.METHOD_LABELS["model"] == "Selected Model C"
    # Outside the binding the suite's own captions are back.
    with suite.RegimeLabels("S0", 0.1, 7680) as labels:
        assert labels.rewrite("S0 architecture-direction comparison") == (
            "S0 training-progress comparison"
        )


def test_the_red_curve_is_labelled_distinctly_from_the_compared_package() -> None:
    assert MODEL_LABEL != suite.MODEL_LABEL
    assert "6-step fine-tune" in MODEL_LABEL
    assert "MODEL_LABEL" in PARENT_BINDINGS


# --------------------------------------------------------------------------
# Contract loading and the pending fields
# --------------------------------------------------------------------------


def test_the_shipped_contract_refuses_to_load_until_training_has_run() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert unfilled_fields(contract) == [".".join(path) for path in PENDING_PATHS]
    with pytest.raises(BireProtocolRolloutFineTuneFigureError) as raised:
        load_contract(CONTRACT, verify_sources=False)
    assert "finalize" in str(raised.value)
    assert contract["pending_after_training"]["sentinel"] == PENDING


def test_a_filled_contract_loads(tmp_path) -> None:
    contract, resolved, digest = load_contract(_written(_filled(), tmp_path), verify_sources=False)
    assert contract["version"] == VERSION
    assert contract["contract_status"] == CONTRACT_STATUS
    assert int(contract["selected_model"]["optimizer_step"]) in CHECKPOINT_STEPS
    assert len(digest) == 64 and resolved.is_file()


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda c: c["figure6"].update(literal_pretrain_finetune_pair=False),
            id="pairing_denied",
        ),
        pytest.param(
            lambda c: c["figure6"].update(comparator_optimizer_step=11520),
            id="comparator_moved",
        ),
        pytest.param(
            lambda c: c["selected_model"].update(rollout_steps=3), id="depth_reverted"
        ),
        pytest.param(
            lambda c: c["selected_model"].update(base_loss_contract_sha256="0" * 64),
            id="objective_moved",
        ),
        pytest.param(
            lambda c: c["comparator_model"].update(base_loss_contract_sha256="0" * 64),
            id="comparator_objective_moved",
        ),
        pytest.param(
            lambda c: c["selected_model"].update(optimizer_step=1234), id="step_not_a_checkpoint"
        ),
        pytest.param(
            lambda c: c["protocol"].update(member_count=10), id="members_moved"
        ),
    ],
)
def test_a_tampered_figure_contract_is_rejected(mutate, tmp_path) -> None:
    contract = _filled()
    mutate(contract)
    with pytest.raises(BireProtocolRolloutFineTuneFigureError):
        load_contract(_written(contract, tmp_path), verify_sources=False)


# --------------------------------------------------------------------------
# finalize
# --------------------------------------------------------------------------


def _staged_report(directory: Path, step: int = SELECTED_STEP) -> tuple[Path, dict]:
    """A training report on disk, plus a contract pointing at it."""

    from bire_repro.af_model_c_bire_protocol_rollout_ft import REPORT_NAME

    report_path = directory / REPORT_NAME
    contract = _filled(step)
    published = {
        "optimizer_step": step,
        "checkpoint": contract["artifacts"]["selected_checkpoint"]["path"],
        "checkpoint_sha256": "c" * 64,
        "normalization": contract["artifacts"]["selected_normalization"]["path"],
        "normalization_sha256": "d" * 64,
    }
    report_path.write_text(
        json.dumps(
            {"version": "model_c_bire_protocol_rollout_ft_v1", "published_checkpoint": published},
            indent=2, sort_keys=True,
        )
    )
    contract = json.loads(CONTRACT.read_text())
    contract["artifacts"]["selected_report"]["path"] = str(report_path)
    return report_path, contract


def test_finalize_fills_the_pending_fields_from_the_runs_own_report(tmp_path) -> None:
    _, contract = _staged_report(tmp_path, step=2880)
    path = _written(contract, tmp_path, "contract.json")
    result = finalize(path)
    assert result["status"] == "filled"
    assert result["selected_optimizer_step"] == 2880
    assert set(result["applied"]) == {".".join(p) for p in PENDING_PATHS}
    written = json.loads(path.read_text())
    assert unfilled_fields(written) == []
    assert written["selected_model"]["optimizer_step"] == 2880
    assert written["artifacts"]["selected_checkpoint"]["sha256"] == "c" * 64
    assert written["artifacts"]["selected_normalization"]["sha256"] == "d" * 64
    loaded, _, _ = load_contract(path, verify_sources=False)
    assert int(loaded["selected_model"]["optimizer_step"]) == 2880


def test_finalize_is_idempotent(tmp_path) -> None:
    _, contract = _staged_report(tmp_path)
    path = _written(contract, tmp_path, "contract.json")
    finalize(path)
    before = path.read_text()
    again = finalize(path)
    assert again["status"] == "already_complete" and again["applied"] == {}
    assert path.read_text() == before


def test_finalize_refuses_to_overwrite_a_filled_field(tmp_path) -> None:
    _, contract = _staged_report(tmp_path, step=1920)
    contract["selected_model"]["optimizer_step"] = 3840
    path = _written(contract, tmp_path, "contract.json")
    with pytest.raises(BireProtocolRolloutFineTuneFigureError) as raised:
        finalize(path)
    assert "refusing to overwrite" in str(raised.value)


def test_finalize_refuses_a_report_from_another_arm(tmp_path) -> None:
    report_path, contract = _staged_report(tmp_path)
    payload = json.loads(report_path.read_text())
    payload["version"] = "model_c_bire_protocol_duration_v1"
    report_path.write_text(json.dumps(payload))
    with pytest.raises(BireProtocolRolloutFineTuneFigureError):
        finalize(_written(contract, tmp_path, "contract.json"))


def test_finalize_refuses_a_report_that_names_a_different_checkpoint(tmp_path) -> None:
    report_path, contract = _staged_report(tmp_path)
    payload = json.loads(report_path.read_text())
    payload["published_checkpoint"]["checkpoint"] = "/somewhere/else/selected.pt"
    report_path.write_text(json.dumps(payload))
    with pytest.raises(BireProtocolRolloutFineTuneFigureError):
        finalize(_written(contract, tmp_path, "contract.json"))


def test_finalize_reports_a_missing_training_report(tmp_path) -> None:
    contract = json.loads(CONTRACT.read_text())
    contract["artifacts"]["selected_report"]["path"] = str(
        tmp_path / "bire_protocol_rollout_ft_report.json"
    )
    with pytest.raises(BireProtocolRolloutFineTuneFigureError) as raised:
        finalize(_written(contract, tmp_path, "contract.json"))
    assert "not on disk yet" in str(raised.value)


# --------------------------------------------------------------------------
# The 2,000-day half of the acceptance gate
# --------------------------------------------------------------------------


def _gate_inputs(
    *,
    finite: bool = True,
    magnitude: float = 3.0,
    minimum: float = -20.0,
    std_scale: float = 1.0,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(0)
    grid = 32
    wet = np.ones((grid, grid), dtype=np.uint8)
    truth = rng.normal(0.0, 5.0, size=(grid, grid))
    model = truth * std_scale
    # Plant the requested extreme without disturbing the spread much.
    model = model - model.min() + minimum
    model = (model - model.mean()) * (
        (truth.std() * std_scale) / max(model.std(), 1e-12)
    ) + model.mean()
    model[0, 0] = minimum
    arrays = {
        "wet_mask": wet,
        "figure7_model_streamfunction": np.stack([model * 0.5, model]),
        "figure7_truth_streamfunction": np.stack([truth * 0.5, truth]),
    }
    summary = {
        "all_selected_states_finite": finite,
        "maximum_selected_normalized_abs": magnitude,
        "rmse": {
            field: {
                "model": {"day2000_mean": 1.0},
                "climatology": {"day2000_mean": 1.25},
            }
            for field in figures.RMSE_FIELDS
        },
    }
    return arrays, summary


def test_the_long_rollout_gate_passes_a_healthy_rollout() -> None:
    arrays, summary = _gate_inputs()
    gate = long_rollout_gate(arrays, summary)
    assert gate["long_rollout_conditions_pass"]
    assert all(gate["conditions"].values())
    assert gate["measured"]["day2000_streamfunction_minimum_sv"] == pytest.approx(-20.0)
    low, high = DAY2000_STD_RATIO_RANGE
    assert low <= gate["measured"]["day2000_spatial_std_ratio_to_truth"] <= high


@pytest.mark.parametrize(
    "kwargs,failing",
    [
        ({"finite": False}, "all_values_finite"),
        ({"magnitude": 8.5}, "maximum_normalized_magnitude_at_most_8"),
        ({"minimum": -40.0}, "streamfunction_minimum_at_least_minus_33_sv"),
        ({"std_scale": 0.5}, "day2000_spatial_std_ratio_in_range"),
        ({"std_scale": 1.5}, "day2000_spatial_std_ratio_in_range"),
    ],
)
def test_each_long_rollout_condition_can_fail_on_its_own(kwargs, failing) -> None:
    arrays, summary = _gate_inputs(**kwargs)
    gate = long_rollout_gate(arrays, summary)
    assert gate["conditions"][failing] is False, failing
    assert not gate["long_rollout_conditions_pass"]
    assert sum(1 for value in gate["conditions"].values() if not value) == 1


def test_the_thresholds_are_the_declared_ones() -> None:
    assert MAXIMUM_NORMALIZED_MAGNITUDE == 8.0
    assert MINIMUM_STREAMFUNCTION_SV == -33.0
    assert DAY2000_STD_RATIO_RANGE == (0.80, 1.25)
    arrays, summary = _gate_inputs(magnitude=8.0, minimum=-33.0)
    gate = long_rollout_gate(arrays, summary)
    assert gate["conditions"]["maximum_normalized_magnitude_at_most_8"]
    assert gate["conditions"]["streamfunction_minimum_at_least_minus_33_sv"]


def test_the_gate_reports_the_collapse_indicator_without_gating_on_it() -> None:
    """A model that has relaxed onto climatology can satisfy every bound above."""

    arrays, summary = _gate_inputs()
    for field in figures.RMSE_FIELDS:
        summary["rmse"][field]["model"]["day2000_mean"] = 1.25
    gate = long_rollout_gate(arrays, summary)
    ratios = gate["advisory_day2000_rmse_ratio_to_climatology"]
    assert set(ratios) == set(figures.RMSE_FIELDS)
    assert all(value == pytest.approx(1.0) for value in ratios.values())
    # Advisory only: the arm declaration does not require day-2,000 skill.
    assert gate["long_rollout_conditions_pass"]
    assert "not gated" in gate["advisory_note"]


# --------------------------------------------------------------------------
# Bindings, README, launcher
# --------------------------------------------------------------------------


def test_binding_swaps_and_restores_the_suite_globals() -> None:
    before = {name: getattr(suite, name) for name in PARENT_BINDINGS}
    with _SuiteBinding():
        assert suite.VERSION == VERSION
        assert suite.COMPARATOR_STEP == COMPARATOR_STEP
        assert suite.CONTRACT_STATUS == CONTRACT_STATUS
        assert suite.load_contract is load_contract
        assert suite.RegimeLabels is FineTuneLabels
        assert suite._stepper is not before["_stepper"]
    for name, value in before.items():
        assert getattr(suite, name) is value, f"{name} was not restored"


def test_binding_restores_on_exception() -> None:
    before = {name: getattr(suite, name) for name in PARENT_BINDINGS}
    with pytest.raises(RuntimeError):
        with _SuiteBinding():
            raise RuntimeError("boom")
    for name, value in before.items():
        assert getattr(suite, name) is value


def test_the_stepper_and_captions_are_bound_not_only_the_constants() -> None:
    """The suite assumes one objective hash and a within-run comparison."""

    assert "_stepper" in PARENT_BINDINGS
    assert "RegimeLabels" in PARENT_BINDINGS


def test_a_binding_this_arm_cannot_supply_leaves_the_suite_untouched() -> None:
    """A half-applied binding never runs __exit__, so it would leak process-wide."""

    from bire_repro import af_model_c_bire_protocol_rollout_ft_figures as arm

    names = ("VERSION", "MODEL_LABEL", "_stepper", "RegimeLabels")
    before = {name: getattr(suite, name) for name in names}
    original = arm.PARENT_BINDINGS
    arm.PARENT_BINDINGS = original + ("a_name_this_arm_does_not_define",)
    try:
        with pytest.raises(BireProtocolRolloutFineTuneFigureError):
            with _SuiteBinding():
                pass
    finally:
        arm.PARENT_BINDINGS = original
    for name, value in before.items():
        assert getattr(suite, name) is value, f"{name} leaked out of a failed binding"


def test_readme_renders_and_names_this_arm() -> None:
    text = _readme(
        "S0",
        {
            "selected_optimizer_step": SELECTED_STEP,
            "tau0_n_m2": 0.1,
            "report_content_sha256": "0" * 64,
        },
    )
    assert f"{SELECTED_STEP:,}" in text and f"{COMPARATOR_STEP:,}" in text
    assert "6263" in text and "6979" in text
    assert GATE_NAME in text
    flat = " ".join(text.split())
    assert "literal pre-train / fine-tune pair" in flat
    assert "three ten-day calls to six" in flat
    for stale in ("5039", "5130", "6389", "6480", "held test block", "7,680"):
        assert stale not in text


def test_launcher_finalizes_then_runs_its_own_module_and_contract() -> None:
    text = SBATCH.read_text()
    invoked = {
        line.split("-m", 1)[1].strip().split()[0]
        for line in text.splitlines()
        if " -m " in f" {line} " and "bire_repro." in line
    }
    assert invoked == {"bire_repro.af_model_c_bire_protocol_rollout_ft_figures"}
    assert "model_c_bire_protocol_rollout_ft_s0_figures_v1.json" in text
    # finalize must run before preflight, or preflight meets a pending contract.
    assert text.index("finalize") < text.index("  preflight") < text.index("  run")
