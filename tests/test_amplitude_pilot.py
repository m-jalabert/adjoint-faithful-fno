"""No-compute tests for the amplitude pilot (plan section 7.2/8/10, step 7).

Kernel/RMS-scale correctness is the highest-risk part of this step (a subtle
bug there would silently corrupt every perturbation amplitude), so it gets
direct numerical tests here against small synthetic grids/normalizers, in
addition to the live validation already run by hand against the real grid
and parent normalizer (all 24 real pilot directions checked to
RMS == 1.000000, and the one real SSH-cap failure case at S0/day3600/alpha
0.10 confirmed). MITgcm mechanics (``af_pilot_segment``) are tested the same
way as the step-6 pickup bank: local fixtures and a fake executable.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _extra in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "archive" / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import build_amplitude_pilot as pilot  # noqa: E402
from bire_repro import af_pilot_segment as segment  # noqa: E402
from bire_repro.af_s0 import MITGCM_COMMIT  # noqa: E402


# ---------------------------------------------------------------------------
# Kernel / RMS-scale math


def _direction(**overrides):
    base = {
        "regime": "S0",
        "anchor_day": 720,
        "family": "U",
        "kernel": "gaussian_5x5_sigma1",
        "region": "interior",
        "levels": [1],
        "j": 10,
        "i": 10,
        "long": False,
    }
    base.update(overrides)
    return base


@pytest.fixture
def wet_mask() -> np.ndarray:
    return np.ones((62, 62), dtype=bool)


@pytest.fixture
def uniform_sigma() -> np.ndarray:
    # Uniform sigma=1 isolates the kernel-shape/centering math from any
    # heterogeneous-roughness effect, which is exercised separately below.
    return np.ones((46, 62, 62), dtype=np.float64)


def test_gaussian_weights_are_unit_l2_and_symmetric() -> None:
    weights = pilot._GAUSSIAN_5X5
    assert weights.shape == (5, 5)
    assert weights == pytest.approx(weights.T)
    assert float(np.sum(weights**2)) == pytest.approx(1.0)
    assert weights[2, 2] == weights.max()


@pytest.mark.parametrize(
    ("family", "kernel", "expected_support"),
    [
        ("U", "gaussian_5x5_sigma1", 30),
        ("V", "gaussian_5x5_sigma1", 30),
        ("Theta", "gaussian_5x5_sigma1", 25),
    ],
)
def test_direction_vector_has_unit_rms_and_expected_support(
    wet_mask: np.ndarray, uniform_sigma: np.ndarray, family: str, kernel: str, expected_support: int
) -> None:
    row = _direction(family=family, kernel=kernel)
    v_q = pilot.direction_vector(row, wet_mask, uniform_sigma)
    centred = pilot._centred_projection(family, v_q)
    channel = pilot.channel_index(family, row["levels"])
    support = (centred != 0.0) & wet_mask
    assert int(support.sum()) == expected_support
    standardized = centred[support] / uniform_sigma[channel][support]
    assert float(np.sqrt(np.mean(standardized**2))) == pytest.approx(1.0, abs=1e-9)


def test_ssh_point_direction_equals_sigma_at_its_cell(wet_mask: np.ndarray) -> None:
    sigma = np.ones((46, 62, 62), dtype=np.float64)
    sigma[45, 10, 10] = 0.03
    row = _direction(family="SSH", kernel="point", levels=[])
    v_q = pilot.direction_vector(row, wet_mask, sigma)
    assert v_q[10, 10] == pytest.approx(0.03)
    assert int(np.count_nonzero(v_q)) == 1


def test_direction_vector_scale_responds_to_heterogeneous_sigma(wet_mask: np.ndarray) -> None:
    sigma = np.ones((46, 62, 62), dtype=np.float64)
    sigma[0, 8:13, 8:13] = 2.0  # channel 0 = U level 1
    row = _direction(family="U", levels=[1])
    v_q_uniform = pilot.direction_vector(row, wet_mask, np.ones((46, 62, 62)))
    v_q_scaled = pilot.direction_vector(row, wet_mask, sigma)
    # A larger local sigma means a larger raw native edit is needed to reach
    # the same unit-RMS *normalized* target.
    assert np.abs(v_q_scaled).max() > np.abs(v_q_uniform).max()


def test_direction_vector_rejects_a_centre_too_close_to_the_edge(
    wet_mask: np.ndarray, uniform_sigma: np.ndarray
) -> None:
    row = _direction(j=1, i=10)
    with pytest.raises(pilot.ContractError, match="grid edge"):
        pilot.direction_vector(row, wet_mask, uniform_sigma)


def test_pickup_edits_for_reports_the_ssh_cap_correctly() -> None:
    sigma = np.ones((46, 62, 62), dtype=np.float64)
    row = _direction(family="SSH", kernel="point", levels=[])
    v_q = pilot.direction_vector(row, np.ones((62, 62), dtype=bool), sigma)
    _field, _edits, peak_small = pilot.pickup_edits_for(row, v_q, 0.005, 1)
    _field, _edits, peak_large = pilot.pickup_edits_for(row, v_q, 5.0, 1)
    assert peak_small <= pilot.SSH_PEAK_METERS_MAX
    assert peak_large > pilot.SSH_PEAK_METERS_MAX


def test_pickup_edits_for_uses_the_correct_record_and_is_signed() -> None:
    sigma = np.ones((46, 62, 62), dtype=np.float64)
    row = _direction(family="Theta", levels=[9])
    v_q = pilot.direction_vector(row, np.ones((62, 62), dtype=bool), sigma)
    field, edits_plus, _ = pilot.pickup_edits_for(row, v_q, 0.05, 1)
    _field, edits_minus, _ = pilot.pickup_edits_for(row, v_q, 0.05, -1)
    assert field == "Theta"
    expected_record = pilot.pickup_record_index("Theta", 9)
    assert all(edit.record == expected_record for edit in edits_plus)
    by_cell_plus = {(edit.j, edit.i): edit.value for edit in edits_plus}
    by_cell_minus = {(edit.j, edit.i): edit.value for edit in edits_minus}
    assert set(by_cell_plus) == set(by_cell_minus)
    for cell, value in by_cell_plus.items():
        assert value == pytest.approx(-by_cell_minus[cell])


# ---------------------------------------------------------------------------
# af_pilot_segment MITgcm mechanics (mirrors test_af_response_pickup_bank.py)


@pytest.fixture
def source_run_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "source_segment"
    directory.mkdir()
    iteration = segment.day_to_iteration(720)
    meta = directory / f"pickup.{iteration:010d}.meta"
    meta.write_text("synthetic pickup meta\n")
    meta.with_suffix(".data").write_bytes(b"\x00" * 64)
    for name in segment.FORCING_FILES:
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
    monkeypatch.setattr(segment, "_git_revision", lambda source: MITGCM_COMMIT)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_render_segment_data_supports_an_arbitrary_horizon() -> None:
    rendered_10 = segment.render_segment_data(720, 10)
    rendered_90 = segment.render_segment_data(720, 90)
    assert "nTimeSteps=720" in rendered_10  # 10 days * 72 steps/day
    assert "nTimeSteps=6480" in rendered_90  # 90 days * 72 steps/day
    assert "pChkptFreq=864000." in rendered_10
    assert len(segment.archived_checkpoints(720, 10)) == 1
    assert len(segment.archived_checkpoints(720, 90)) == 9


def test_prepare_and_run_segment_short_horizon(
    project_root: Path, tmp_path: Path, executable: Path, source_run_dir: Path
) -> None:
    iteration = segment.day_to_iteration(720)
    meta = source_run_dir / f"pickup.{iteration:010d}.meta"
    data = meta.with_suffix(".data")
    manifest = segment.prepare_segment(
        project_root,
        tmp_path / "scratch",
        executable,
        "S0_d0720_test",
        source_run_dir,
        meta,
        data,
        _sha256(meta),
        _sha256(data),
        720,
        10,
    )
    run_dir = Path(manifest["run_dir"])
    assert (run_dir / "data").is_file()
    assert len(manifest["archived_checkpoints"]) == 1

    fake_iteration = manifest["archived_checkpoints"][0]["iteration"]
    lines = ["#!/bin/sh", "set -e"]
    for day in range(1, 11):
        lines.append(f"touch dynState.{day:010d}.meta")
        lines.append(f"touch surfState.{day:010d}.meta")
    lines.append(f"touch pickup.{fake_iteration:010d}.meta")
    lines.append(f"touch pickup.{fake_iteration:010d}.data")
    script = run_dir / "mitgcmuv"
    script.write_text("\n".join(lines) + "\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    result = segment.run_segment(manifest, launcher=[])
    assert result["returncode"] == 0
    assert len(result["archived_pickups"]) == 1
    # Idempotent replay
    assert segment.run_segment(manifest, launcher=[]) == result
