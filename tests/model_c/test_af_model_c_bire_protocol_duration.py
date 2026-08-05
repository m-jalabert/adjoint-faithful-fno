"""Tests for the matched-budget duration control.

The arm is a one-factor control, so most of these assert what must *not* have
moved.  The binding test matters because this arm reuses the parent's training
loop by rebinding module globals: if a binding is dropped, the run silently
trains the parent's budget and writes into the parent's package.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bire_repro import af_model_c_bire_protocol as protocol
from bire_repro.af_model_c_bire_protocol_duration import (
    CHECKPOINT_STEPS,
    CONTRACT_STATUS,
    DECLARED_CHANGE,
    MAXIMUM_STEPS,
    PARENT_MAXIMUM_STEPS,
    PARENT_VERSION,
    PARENT_BINDINGS,
    VERSION,
    BireProtocolDurationError,
    _ParentBinding,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config/model_c_bire_protocol_duration_v1.json"
PARENT = ROOT / "config/model_c_bire_protocol_pooled_v1.json"
SBATCH = ROOT / "slurm/models/c/af_model_c_bire_protocol_duration.sbatch"


def test_the_budget_is_doubled_and_checkpoints_are_its_quarters() -> None:
    assert MAXIMUM_STEPS == 15360 == 2 * PARENT_MAXIMUM_STEPS
    assert CHECKPOINT_STEPS == (3840, 7680, 11520, 15360)
    assert CHECKPOINT_STEPS[-1] == MAXIMUM_STEPS
    assert all(
        step == (index + 1) * MAXIMUM_STEPS // 4
        for index, step in enumerate(CHECKPOINT_STEPS)
    )
    assert CHECKPOINT_STEPS != protocol.CHECKPOINT_STEPS


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the duration contract is absent")
def test_contract_moves_exactly_the_budget() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    parent = json.loads(PARENT.read_text())
    moved = [
        field
        for field in protocol.FROZEN_TRAINING_FIELDS
        if contract["training"].get(field) != parent["training"].get(field)
    ]
    assert moved == [DECLARED_CHANGE] == ["maximum_steps"]
    assert contract["architecture"] == parent["architecture"]
    assert contract["loss"] == parent["loss"]
    assert contract["dataset"] == parent["dataset"]
    assert contract["checkpoint_selection"]["rule"] == parent["checkpoint_selection"]["rule"]
    assert contract["version"] == VERSION != parent["version"]
    assert contract["contract_status"] == CONTRACT_STATUS


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the duration contract is absent")
def test_the_decay_moves_with_the_budget_and_is_declared() -> None:
    """Holding decay_fraction moves the decay step; that must be recorded."""

    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    training = contract["training"]
    assert float(training["decay_fraction"]) == 0.75
    assert round(training["maximum_steps"] * training["decay_fraction"]) == 11520
    declared = contract["changed_relative_to_the_parent"]["decay_step_consequence"]
    assert declared["parent"] == 5760 and declared["this_arm"] == 11520
    assert "not_comparable" in declared["note"]


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the duration contract is absent")
def test_outputs_do_not_collide_with_the_parent_package() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    parent = json.loads(PARENT.read_text())
    for key in ("scratch_root", "project_root"):
        assert contract["output"][key] != parent["output"][key]
        assert "duration" in contract["output"][key]
    assert set(contract["output"]["artifacts"]).isdisjoint(
        set(parent["output"]["artifacts"]) - {"README.md", "manifest.json"}
    )


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the duration contract is absent")
def test_a_moved_second_quantity_is_rejected() -> None:
    import tempfile

    contract = json.loads(CONTRACT.read_text())
    contract["training"]["batch_size"] = 16
    with tempfile.TemporaryDirectory(dir=ROOT / "config") as directory:
        path = Path(directory) / "tampered.json"
        path.write_text(json.dumps(contract))
        with pytest.raises(BireProtocolDurationError):
            load_contract(path, verify_sources=False)


@pytest.mark.skipif(not CONTRACT.is_file(), reason="the duration contract is absent")
def test_reverting_the_budget_is_rejected() -> None:
    """With maximum_steps back at the parent's, nothing moved and the arm is void."""

    import tempfile

    contract = json.loads(CONTRACT.read_text())
    contract["training"]["maximum_steps"] = PARENT_MAXIMUM_STEPS
    with tempfile.TemporaryDirectory(dir=ROOT / "config") as directory:
        path = Path(directory) / "tampered.json"
        path.write_text(json.dumps(contract))
        with pytest.raises(BireProtocolDurationError):
            load_contract(path, verify_sources=False)


def test_binding_swaps_every_global_the_parent_resolves_and_restores_them() -> None:
    before = {name: getattr(protocol, name) for name in PARENT_BINDINGS}
    with _ParentBinding():
        assert protocol.VERSION == VERSION
        assert protocol.MAXIMUM_STEPS == MAXIMUM_STEPS
        assert protocol.CHECKPOINT_STEPS == CHECKPOINT_STEPS
        assert protocol.load_contract is load_contract
        for name in PARENT_BINDINGS:
            assert getattr(protocol, name) != before[name] or name == "CONTRACT_STATUS"
    for name, value in before.items():
        assert getattr(protocol, name) == value, f"{name} was not restored"


def test_binding_restores_on_exception() -> None:
    before = {name: getattr(protocol, name) for name in PARENT_BINDINGS}
    with pytest.raises(RuntimeError):
        with _ParentBinding():
            raise RuntimeError("boom")
    for name, value in before.items():
        assert getattr(protocol, name) == value


def test_artifact_names_do_not_collide_with_the_parents() -> None:
    from bire_repro import af_model_c_bire_protocol_duration as arm

    for name in ("REPORT_NAME", "ARRAYS_NAME", "FIGURE_NAME",
                 "NORMALIZATION_NAME", "DIVERGENCE_NAME"):
        assert getattr(arm, name) != getattr(protocol, name)
        assert "duration" in getattr(arm, name)


@pytest.mark.skipif(not SBATCH.is_file(), reason="launcher absent")
def test_launcher_invokes_its_own_module_and_contract() -> None:
    text = SBATCH.read_text()
    invoked = {
        line.split("-m", 1)[1].strip().split()[0]
        for line in text.splitlines()
        if " -m " in f" {line} " and "bire_repro." in line
    }
    assert invoked == {"bire_repro.af_model_c_bire_protocol_duration"}
    assert "model_c_bire_protocol_duration_v1.json" in text
    assert "model_c_bire_protocol_pooled_v1.json" not in text


@pytest.mark.skipif(
    not (ROOT / "outputs/af_fno/C/bire_protocol_duration_v1/bire_protocol_duration_report.json").is_file(),
    reason="the run has not completed",
)
def test_readme_renders_against_a_real_report() -> None:
    """Regression: `_readme` read `selected_optimizer_step` at the top level.

    It lives under `selection_decision`, so the run crashed after 1h34m of
    completed training and validation, with every numerical artifact already
    written.  Rendering the README is the last step before promotion, so a key
    error there costs the whole job.
    """

    from bire_repro.af_model_c_bire_protocol_duration import _readme

    report = json.loads(
        (ROOT / "outputs/af_fno/C/bire_protocol_duration_v1/bire_protocol_duration_report.json").read_text()
    )
    text = _readme(report)
    selected = int(report["selection_decision"]["selected_optimizer_step"])
    assert f"Selected step {selected:,}" in text
    assert report["selection_decision"]["branch"] in text
    assert f"{MAXIMUM_STEPS:,}" in text and f"{PARENT_MAXIMUM_STEPS:,}" in text
    for summary in report["validation_summaries"]:
        assert f"{int(summary['optimizer_step']):,}" in text


def test_readme_rejects_a_report_missing_the_selection_block() -> None:
    from bire_repro.af_model_c_bire_protocol_duration import _readme

    with pytest.raises(KeyError):
        _readme({"validation_summaries": [], "optimizer": {"decay_step": 11520}})
