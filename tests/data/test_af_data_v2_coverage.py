"""Contract and decision tests for the training-only v2 coverage audit."""

from pathlib import Path

from bire_repro.diagnostics.af_data_v2_coverage import (
    assess_effective_coverage,
    load_coverage_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_coverage_contract_loads() -> None:
    contract, path, digest = load_coverage_contract(
        ROOT / "config" / "trajectories_v2_coverage_audit.json"
    )
    assert path.name == "trajectories_v2_coverage_audit.json"
    assert len(digest) == 64
    assert contract["read_contract"]["validation_read"] is False
    assert contract["read_contract"]["inference_read"] is False


def test_effective_coverage_assessment_distinguishes_target_and_material_gain() -> None:
    multipliers = {
        "state_rms": {"temperature": 2.1, "ssh": 2.0},
        "increment_rms": {"temperature": 1.8, "ssh": 1.9},
    }
    target = assess_effective_coverage(
        multipliers,
        target=2.0,
        minimum_material=1.5,
        slow_groups=("temperature", "ssh"),
    )
    assert target["target_met"]
    assert target["material_gain"]
    assert target["status"] == "two_times_effective_target_met"

    multipliers["state_rms"]["ssh"] = 1.7
    material = assess_effective_coverage(
        multipliers,
        target=2.0,
        minimum_material=1.5,
        slow_groups=("temperature", "ssh"),
    )
    assert not material["target_met"]
    assert material["material_gain"]

    multipliers["state_rms"]["ssh"] = 1.4
    insufficient = assess_effective_coverage(
        multipliers,
        target=2.0,
        minimum_material=1.5,
        slow_groups=("temperature", "ssh"),
    )
    assert not insufficient["material_gain"]
