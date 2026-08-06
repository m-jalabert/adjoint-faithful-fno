from pathlib import Path

import pytest

import bire_repro.af_s0 as af_s0_module
from bire_repro.af_s0 import (
    S0_PRODUCTION_YEARS,
    S0_SPINUP_YEARS,
    STEPS_PER_YEAR,
    prepare_segment,
    render_data,
    simulation_inventory,
)


def test_render_data_uses_tutorial_timestep_and_restart_iterations() -> None:
    rendered = render_data(10 * STEPS_PER_YEAR, 10 * STEPS_PER_YEAR)
    assert f"nIter0={10 * STEPS_PER_YEAR}" in rendered
    assert f"nTimeSteps={10 * STEPS_PER_YEAR}" in rendered
    assert "deltaT=1200." in rendered
    assert "pChkptFreq=31104000." in rendered
    assert "viscAh=5000." in rendered
    assert "diffKhT=1000." in rendered


def test_inventory_exposes_literal_plan_arithmetic() -> None:
    inventory = simulation_inventory()
    assert inventory["scientific_integrations"]["minimum_total"] == 247
    assert inventory["scientific_integrations"]["trajectory_v2_extensions"] == 3
    assert inventory["adjoint_ground_truth_integrations"] == 12
    assert inventory["model_time"]["trajectory_years"] == 172
    assert S0_SPINUP_YEARS + S0_PRODUCTION_YEARS == 110


def test_prepare_segment_rejects_out_of_phase_years(tmp_path: Path) -> None:
    executable = tmp_path / "mitgcmuv"
    executable.write_bytes(b"executable")
    with pytest.raises(ValueError, match="cannot pass year 100"):
        prepare_segment(tmp_path, tmp_path / "scratch", executable, "spinup", 95, 10)
    with pytest.raises(ValueError, match="years 100 through 110"):
        prepare_segment(tmp_path, tmp_path / "scratch", executable, "production", 90, 10)


def test_git_revision_falls_back_to_checkout_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "MITgcm"
    reference = source / ".git" / "refs" / "heads" / "main"
    reference.parent.mkdir(parents=True)
    (source / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    reference.write_text(f"{af_s0_module.MITGCM_COMMIT}\n")
    monkeypatch.setattr(af_s0_module.shutil, "which", lambda _command: None)
    assert af_s0_module._git_revision(source) == af_s0_module.MITGCM_COMMIT
