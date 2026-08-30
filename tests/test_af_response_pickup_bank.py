"""No-compute tests for the validation pickup-bank bridge (plan step 6).

These tests exercise ``bire_repro.af_response_pickup_bank`` entirely against
local fixtures: a synthetic source pickup, a monkeypatched MITgcm git
revision, and (for ``run_segment``) a tiny fake ``mitgcmuv`` shell script.
Nothing here submits a real Slurm job or touches ``/bigscratch``.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SRC = PROJECT_ROOT / "archive" / "src"
if str(ARCHIVE_SRC) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_SRC))

from bire_repro import af_response_pickup_bank as bank  # noqa: E402
from bire_repro.af_s0 import MITGCM_COMMIT, render_data  # noqa: E402
from bire_repro.af_s0_twin import PICKUP_DTYPE, PICKUP_FIELD_LAYOUT, PICKUP_GRID  # noqa: E402


NRECORDS = sum(count for _, count in PICKUP_FIELD_LAYOUT)


def _write_pickup(directory: Path, iteration: int) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / f"pickup.{iteration:010d}.meta"
    data_path = meta_path.with_suffix(".data")
    fields = " ".join(f"'{name:<8}'" for name, _ in PICKUP_FIELD_LAYOUT)
    meta_path.write_text(
        " nDims = [   2 ];\n"
        " dimList = [\n"
        "  62,    1,   62,\n"
        "  62,    1,   62\n"
        " ];\n"
        " dataprec = [ 'float64' ];\n"
        f" nrecords = [ {NRECORDS} ];\n"
        f" timeStepNumber = [ {iteration} ];\n"
        f" fldList = {{ {fields} }};\n"
    )
    values = np.zeros((NRECORDS, *PICKUP_GRID), dtype=PICKUP_DTYPE)
    values.tofile(data_path)
    return meta_path, data_path


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source_run_dir(tmp_path: Path) -> Path:
    """A stand-in for one existing trajectory-v3 chain segment directory."""

    directory = tmp_path / "source_segment"
    directory.mkdir()
    iteration = bank.day_to_iteration(bank.SOURCE_DAY)
    _write_pickup(directory, iteration)
    for name in (
        "data.pkg",
        "eedata",
        "data.diagnostics",
        "bathy.bin",
        "windx_cosy.bin",
        "SST_relax.bin",
    ):
        (directory / name).write_bytes(name.encode("utf-8"))
    return directory


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "external" / "MITgcm").mkdir(parents=True)
    return root


@pytest.fixture
def executable(tmp_path: Path) -> Path:
    path = tmp_path / "mitgcmuv"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture(autouse=True)
def _patch_git_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bank, "_git_revision", lambda source: MITGCM_COMMIT)


# ---------------------------------------------------------------------------
# Constants and physics-file rendering


def test_bridge_covers_exactly_the_declared_320_days_in_32_ten_day_steps() -> None:
    assert bank.BASE_ITERATION == 2_592_000
    assert bank.TOTAL_DAYS == 320
    assert bank.N_SEGMENTS == 32
    assert bank.PCHKPT_FREQ_SECONDS == 864_000
    assert len(bank.ARCHIVED_CHECKPOINTS) == 32
    assert bank.ARCHIVED_CHECKPOINTS[0] == (5770, bank.day_to_iteration(5770))
    assert bank.ARCHIVED_CHECKPOINTS[-1] == (6080, bank.day_to_iteration(6080))
    assert bank.day_to_iteration(6080) - bank.day_to_iteration(5760) == bank.N_TIME_STEPS
    assert set(bank.RETAINED_ANCHOR_DAYS) <= {day for day, _ in bank.ARCHIVED_CHECKPOINTS}


def test_render_bridge_data_changes_only_pchkpt_freq() -> None:
    start_iteration = bank.day_to_iteration(bank.SOURCE_DAY)
    baseline = render_data(start_iteration, bank.N_TIME_STEPS).splitlines()
    bridge = bank.render_bridge_data().splitlines()
    assert len(baseline) == len(bridge)
    differing = [(a, b) for a, b in zip(baseline, bridge) if a != b]
    assert differing == [(" pChkptFreq=31104000.,", " pChkptFreq=864000.,")]
    assert f"nIter0={start_iteration}" in bank.render_bridge_data()
    assert f"nTimeSteps={bank.N_TIME_STEPS}" in bank.render_bridge_data()


def test_pchkpt_freq_override_fails_closed_when_needle_is_absent() -> None:
    with pytest.raises(bank.PickupBankError, match="exactly one"):
        bank._pchkpt_freq_override("no such field here")


def test_pchkpt_freq_override_fails_closed_when_needle_repeats() -> None:
    rendered = "pChkptFreq=31104000.,\npChkptFreq=31104000.,\n"
    with pytest.raises(bank.PickupBankError, match="exactly one"):
        bank._pchkpt_freq_override(rendered)


# ---------------------------------------------------------------------------
# prepare_segment


def _prepare(
    project_root: Path, tmp_path: Path, executable: Path, source_run_dir: Path, regime: str = "S0"
):
    iteration = bank.day_to_iteration(bank.SOURCE_DAY)
    meta_path = source_run_dir / f"pickup.{iteration:010d}.meta"
    data_path = meta_path.with_suffix(".data")
    return bank.prepare_segment(
        project_root,
        tmp_path / "scratch",
        executable,
        regime,
        meta_path,
        data_path,
        _sha256(meta_path),
        _sha256(data_path),
    )


def test_prepare_segment_stages_inputs_and_hashes_them(
    project_root: Path, tmp_path: Path, executable: Path, source_run_dir: Path
) -> None:
    manifest = _prepare(project_root, tmp_path, executable, source_run_dir)
    run_dir = Path(manifest["run_dir"])
    assert run_dir.is_dir()
    assert (run_dir / "data").read_text() == bank.render_bridge_data()
    for name in (
        "data.pkg",
        "eedata",
        "data.diagnostics",
        "bathy.bin",
        "windx_cosy.bin",
        "SST_relax.bin",
    ):
        assert (run_dir / name).read_bytes() == name.encode("utf-8")
    assert (run_dir / "mitgcmuv").resolve() == executable.resolve()
    assert manifest["configuration_sha256"]["data"] == _sha256(run_dir / "data")
    assert manifest["mitgcm_commit"] == MITGCM_COMMIT
    assert len(manifest["archived_checkpoints"]) == 32
    assert manifest["retained_anchor_days"] == list(bank.RETAINED_ANCHOR_DAYS)


def test_prepare_segment_is_idempotent(
    project_root: Path, tmp_path: Path, executable: Path, source_run_dir: Path
) -> None:
    first = _prepare(project_root, tmp_path, executable, source_run_dir)
    second = _prepare(project_root, tmp_path, executable, source_run_dir)
    assert first == second


def test_prepare_segment_detects_identity_drift(
    project_root: Path, tmp_path: Path, executable: Path, source_run_dir: Path
) -> None:
    _prepare(project_root, tmp_path, executable, source_run_dir)
    other_executable = tmp_path / "mitgcmuv_v2"
    other_executable.write_text("#!/bin/sh\n# a different build\nexit 0\n")
    other_executable.chmod(other_executable.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(bank.PickupBankError, match="identity changed"):
        _prepare(project_root, tmp_path, other_executable, source_run_dir)


def test_prepare_segment_refuses_a_partial_directory(
    project_root: Path, tmp_path: Path, executable: Path, source_run_dir: Path
) -> None:
    run_dir = (
        tmp_path / "scratch" / bank.ROOT_NAME / "S0" / f"bridge_{bank.SOURCE_DAY}_{bank.END_DAY}"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "stray_file").write_text("leftover")
    with pytest.raises(bank.PickupBankError, match="without a manifest"):
        _prepare(project_root, tmp_path, executable, source_run_dir)


def test_prepare_segment_rejects_a_hash_mismatch_before_any_write(
    project_root: Path, tmp_path: Path, executable: Path, source_run_dir: Path
) -> None:
    iteration = bank.day_to_iteration(bank.SOURCE_DAY)
    meta_path = source_run_dir / f"pickup.{iteration:010d}.meta"
    data_path = meta_path.with_suffix(".data")
    with pytest.raises(bank.PickupBankError, match="hash mismatch"):
        bank.prepare_segment(
            project_root,
            tmp_path / "scratch",
            executable,
            "S0",
            meta_path,
            data_path,
            "0" * 64,
            _sha256(data_path),
        )
    assert not (tmp_path / "scratch").exists()


def test_prepare_segment_rejects_wrong_regime() -> None:
    with pytest.raises(bank.PickupBankError, match="regime must be one of"):
        bank.prepare_segment(
            Path("/nonexistent"),
            Path("/nonexistent"),
            Path("/nonexistent"),
            "S3",
            Path("/nonexistent"),
            Path("/nonexistent"),
            "0" * 64,
            "0" * 64,
        )


# ---------------------------------------------------------------------------
# run_segment, against a small synthetic manifest (not the real 32 checkpoints)


def _fake_mitgcmuv(
    run_dir: Path, archived_checkpoints: list[dict[str, int]], total_days: int
) -> None:
    script = run_dir / "mitgcmuv"
    lines = ["#!/bin/sh", "set -e"]
    for day in range(1, total_days + 1):
        lines.append(f"touch dynState.{day:010d}.meta")
        lines.append(f"touch surfState.{day:010d}.meta")
    for entry in archived_checkpoints:
        iteration = entry["iteration"]
        lines.append(f"touch pickup.{iteration:010d}.meta")
        lines.append(f"touch pickup.{iteration:010d}.data")
    script.write_text("\n".join(lines) + "\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def _small_manifest(run_dir: Path) -> dict:
    source_day, end_day, segment_days = 0, 20, 10
    checkpoints = [
        {
            "day": source_day + segment_days * k,
            "iteration": bank.day_to_iteration(source_day + segment_days * k),
        }
        for k in (1, 2)
    ]
    manifest = {
        "version": bank.VERSION,
        "regime": "S0",
        "source_day": source_day,
        "end_day": end_day,
        "start_iteration": bank.day_to_iteration(source_day),
        "end_iteration": bank.day_to_iteration(end_day),
        "run_dir": str(run_dir),
        "archived_checkpoints": checkpoints,
        "retained_anchor_days": [checkpoints[-1]["day"]],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "segment_manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_run_segment_archives_and_hashes_every_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest = _small_manifest(run_dir)
    _fake_mitgcmuv(
        run_dir, manifest["archived_checkpoints"], manifest["end_day"] - manifest["source_day"]
    )

    result = bank.run_segment(manifest, launcher=[])

    assert result["returncode"] == 0
    assert len(result["archived_pickups"]) == 2
    assert {entry["day"] for entry in result["retained_pickups"]} == {20}
    for entry in result["archived_pickups"]:
        assert len(entry["meta_sha256"]) == 64
        assert len(entry["data_sha256"]) == 64
    assert (run_dir / "segment_result.json").is_file()


def test_run_segment_is_idempotent(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest = _small_manifest(run_dir)
    _fake_mitgcmuv(
        run_dir, manifest["archived_checkpoints"], manifest["end_day"] - manifest["source_day"]
    )
    first = bank.run_segment(manifest, launcher=[])
    second = bank.run_segment(manifest, launcher=[])
    assert first == second


def test_run_segment_refuses_to_overwrite_partial_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest = _small_manifest(run_dir)
    (run_dir / "run.log").write_text("a previous attempt died mid-run")
    with pytest.raises(bank.PickupBankError, match="incomplete bridge output"):
        bank.run_segment(manifest, launcher=[])


def test_run_segment_detects_a_missing_archived_pickup(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest = _small_manifest(run_dir)
    # Only produce diagnostics and the first checkpoint, not the second.
    _fake_mitgcmuv(
        run_dir, manifest["archived_checkpoints"][:1], manifest["end_day"] - manifest["source_day"]
    )
    with pytest.raises(bank.PickupBankError, match="missing archived pickup"):
        bank.run_segment(manifest, launcher=[])


def test_run_segment_detects_a_diagnostic_count_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest = _small_manifest(run_dir)
    _fake_mitgcmuv(
        run_dir, manifest["archived_checkpoints"], manifest["end_day"] - manifest["source_day"] - 1
    )
    with pytest.raises(bank.PickupBankError, match="diagnostic count mismatch"):
        bank.run_segment(manifest, launcher=[])
