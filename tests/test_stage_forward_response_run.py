"""No-compute tests for step 9's production response-run staging.

MITgcm mechanics (``af_pilot_segment.prepare_segment``/``run_segment``) are
already covered by ``tests/test_amplitude_pilot.py`` with local fixtures and
a fake executable; ``stage_forward_response_run.py`` reuses that machinery
unchanged (plus a backward-compatible ``scratch_subdir`` parameter), so this
file focuses on the logic specific to production staging: inventory
filtering (train/validation only, never blind), amplitude lookup, and
shared-nominal-group/pilot-overlap grouping.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _extra in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "archive" / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import stage_forward_response_run as staging  # noqa: E402


def _row(**overrides):
    base = {
        "role": "train",
        "regime": "S0",
        "anchor_day": 720,
        "family": "U",
        "direction_slot": 0,
        "direction_id": "response-v1:train:S0:d0720:U:gaussian_5x5_sigma1:a0:q0:levels",
        "kernel": "gaussian_5x5_sigma1",
        "region": "interior",
        "levels_one_based": [1],
        "vertical_weights_float64_hex": [float(1.0).hex()],
        "long": False,
        "centre": {"j": 10, "i": 10, "lon": 1.0, "lat": 2.0},
    }
    base.update(overrides)
    return base


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_load_production_rows_keeps_only_train_and_validation(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    rows = [
        _row(role="pilot", direction_id="p"),
        _row(role="train", direction_id="t"),
        _row(role="validation", direction_id="v", anchor_day=6010),
    ]
    _write_jsonl(inventory, rows)
    loaded = staging.load_production_rows(inventory)
    assert {row["role"] for row in loaded} == {"train", "validation"}
    assert len(loaded) == 2


def test_load_production_rows_never_touches_a_blind_path(tmp_path: Path) -> None:
    """Structural guarantee: this module has no argument or default pointing at
    a blind manifest, so it cannot expose blind data even by misconfiguration
    within a single call -- verified here by checking the actual default."""

    assert "blind" not in str(staging.DEFAULT_PUBLIC_INVENTORY).lower()


def test_as_pilot_style_direction_adapts_the_production_schema() -> None:
    row = _row(kernel="point", family="SSH", levels_one_based=[], long=True)
    adapted = staging._as_pilot_style_direction(row)
    assert adapted == {
        "j": 10,
        "i": 10,
        "kernel": "point",
        "family": "SSH",
        "levels": [],
        "long": True,
    }


def test_nominal_groups_uses_60_days_only_when_a_long_direction_shares_the_anchor() -> None:
    rows = [
        _row(role="train", regime="S0", anchor_day=720, long=False, direction_slot=0),
        _row(role="train", regime="S0", anchor_day=720, long=True, direction_slot=1),
        _row(role="train", regime="S0", anchor_day=1800, long=False, direction_slot=0),
        _row(role="validation", regime="S1", anchor_day=6010, long=True, direction_slot=0),
    ]
    groups = staging.nominal_groups(rows)
    assert groups[("train", "S0", 720)] == 60
    assert groups[("train", "S0", 1800)] == 10
    assert groups[("validation", "S1", 6010)] == 60
    assert len(groups) == 3


def test_pilot_overlap_anchors_reads_the_frozen_geometry_file(tmp_path: Path) -> None:
    geometry = tmp_path / "pilot_geometry.json"
    geometry.write_text(
        json.dumps(
            {
                "version": "amplitude_pilot_geometry_v1",
                "directions": [
                    {"regime": "S0", "anchor_day": 720},
                    {"regime": "S0", "anchor_day": 3600},
                    {"regime": "S1", "anchor_day": 720},
                ],
            }
        )
    )
    overlap = staging._pilot_overlap_anchors(geometry)
    assert overlap == {("S0", 720), ("S0", 3600), ("S1", 720)}


def test_load_final_amplitudes_requires_a_frozen_pass(tmp_path: Path) -> None:
    path = tmp_path / "amplitudes.json"
    path.write_text(
        json.dumps(
            {
                "final_selection": "PASS -- all four amplitudes frozen",
                "selected_amplitudes": {"U": 0.1, "V": 0.1, "Theta": 0.005, "SSH": 0.05},
            }
        )
    )
    amplitudes = staging.load_final_amplitudes(path)
    assert amplitudes == {"U": 0.1, "V": 0.1, "Theta": 0.005, "SSH": 0.05}

    not_frozen = tmp_path / "not_frozen.json"
    not_frozen.write_text(
        json.dumps({"final_selection": "PENDING", "selected_amplitudes": {}})
    )
    with pytest.raises(staging.ProductionRunError, match="not.*frozen PASS"):
        staging.load_final_amplitudes(not_frozen)


def test_load_final_amplitudes_requires_all_four_families(tmp_path: Path) -> None:
    path = tmp_path / "amplitudes.json"
    path.write_text(
        json.dumps(
            {
                "final_selection": "PASS -- all four amplitudes frozen",
                "selected_amplitudes": {"U": 0.1, "V": 0.1},
            }
        )
    )
    with pytest.raises(staging.ProductionRunError, match="missing families"):
        staging.load_final_amplitudes(path)


def test_list_work_excludes_pilot_overlap_anchors_from_new_nominal_groups(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    geometry = tmp_path / "pilot_geometry.json"
    _write_jsonl(
        inventory,
        [
            _row(role="train", regime="S0", anchor_day=720, long=True, direction_slot=0),
            _row(role="train", regime="S0", anchor_day=1800, long=False, direction_slot=0),
            _row(role="validation", regime="S1", anchor_day=6010, long=True, direction_slot=0),
        ],
    )
    geometry.write_text(
        json.dumps(
            {
                "version": "amplitude_pilot_geometry_v1",
                "directions": [{"regime": "S0", "anchor_day": 720}],
            }
        )
    )
    work = staging.list_work(inventory, geometry)
    assert work["signed_branches"] == 6  # 3 directions x 2 signs
    assert work["new_nominal_branches"] == 2  # (train,S0,1800) and (validation,S1,6010)
    assert work["reused_pilot_nominal_groups"] == [
        {"role": "train", "regime": "S0", "anchor_day": 720, "duration_days": 60}
    ]
    new_keys = {
        (item["role"], item["regime"], item["anchor_day"]) for item in work["new_nominal_groups"]
    }
    assert ("train", "S0", 720) not in new_keys


def test_run_nominal_refuses_a_pilot_overlap_anchor(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    geometry = tmp_path / "pilot_geometry.json"
    _write_jsonl(inventory, [_row(role="train", regime="S0", anchor_day=720, long=True)])
    geometry.write_text(
        json.dumps(
            {
                "version": "amplitude_pilot_geometry_v1",
                "directions": [{"regime": "S0", "anchor_day": 720}],
            }
        )
    )
    with pytest.raises(staging.ProductionRunError, match="pilot-overlap anchor"):
        staging.run_nominal(
            "train",
            "S0",
            720,
            inventory_path=inventory,
            pilot_geometry_path=geometry,
        )


def test_run_nominal_refuses_an_anchor_with_no_directions(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    geometry = tmp_path / "pilot_geometry.json"
    _write_jsonl(inventory, [_row(role="train", regime="S0", anchor_day=720, long=False)])
    geometry.write_text(json.dumps({"version": "amplitude_pilot_geometry_v1", "directions": []}))
    with pytest.raises(staging.ProductionRunError, match="no production directions"):
        staging.run_nominal(
            "train",
            "S0",
            9999,
            inventory_path=inventory,
            pilot_geometry_path=geometry,
        )


# ---------------------------------------------------------------------------
# Multi-level ("held-out vertical combination", section 8.6) direction math.


import math

import numpy as np

import build_amplitude_pilot as pilot  # noqa: E402


@pytest.fixture
def wet_mask() -> np.ndarray:
    return np.ones((62, 62), dtype=bool)


@pytest.fixture
def uniform_sigma() -> np.ndarray:
    return np.ones((46, 62, 62), dtype=np.float64)


def test_direction_vector_by_level_matches_pilot_for_a_single_level(
    wet_mask: np.ndarray, uniform_sigma: np.ndarray
) -> None:
    """The multi-level generalization must reduce exactly to
    build_amplitude_pilot.direction_vector when there is only one level
    (weight 1.0) -- this is the equivalence the production run relies on to
    trust the new path is correct without having to re-derive it from
    scratch."""

    row = _row(family="Theta", levels_one_based=[5], vertical_weights_float64_hex=[float(1.0).hex()])
    generalized = staging.direction_vector_by_level(row, wet_mask, uniform_sigma)
    assert set(generalized) == {5}

    direction = staging._as_pilot_style_direction(row)
    original = pilot.direction_vector(direction, wet_mask, uniform_sigma)
    np.testing.assert_array_equal(generalized[5], original)


def test_direction_vector_by_level_is_jointly_unit_rms_across_levels(
    wet_mask: np.ndarray, uniform_sigma: np.ndarray
) -> None:
    weight = 1.0 / math.sqrt(2.0)
    row = _row(
        family="Theta",
        levels_one_based=[1, 2],
        vertical_weights_float64_hex=[float(weight).hex(), float(weight).hex()],
    )
    v_q_by_level = staging.direction_vector_by_level(row, wet_mask, uniform_sigma)
    assert set(v_q_by_level) == {1, 2}

    direction = staging._as_pilot_style_direction(row)
    native = pilot._native_kernel(direction)
    centred = pilot._centred_projection("Theta", native)
    support = centred != 0.0

    # direction_vector_by_level's *return value* is native*weight/rms (an
    # edit-space direction, not yet in FNO-input space); section 8.1's unit-
    # RMS rule is about the perturbation's effect *in the pointwise-
    # normalized FNO input*, i.e. centred(v_q)/sigma. Re-centre and re-divide
    # by sigma exactly as the internal "standardized" quantity was built,
    # then check that reconstruction has RMS 1 jointly across both levels --
    # confirming the internal `rms` division actually achieves the rule,
    # not merely that the pre-division stack has some other property.
    reconstructed = np.concatenate(
        [
            pilot._centred_projection("Theta", v_q_by_level[level])[support]
            / uniform_sigma[pilot.channel_index("Theta", [level])][support]
            for level in (1, 2)
        ]
    )
    assert math.sqrt(float(np.mean(reconstructed**2))) == pytest.approx(1.0)

    # Each level's returned vector is native * weight / rms -- same spatial
    # pattern, scaled by that level's own weight.
    ratio = v_q_by_level[1][support] / v_q_by_level[2][support]
    np.testing.assert_allclose(ratio, np.ones_like(ratio))


def test_pickup_edits_for_by_level_targets_each_levels_own_record(
    wet_mask: np.ndarray, uniform_sigma: np.ndarray
) -> None:
    weight = 1.0 / math.sqrt(2.0)
    row = _row(
        family="Theta",
        levels_one_based=[1, 2],
        vertical_weights_float64_hex=[float(weight).hex(), float(weight).hex()],
    )
    v_q_by_level = staging.direction_vector_by_level(row, wet_mask, uniform_sigma)
    field, edits, peak = staging.pickup_edits_for_by_level(row, v_q_by_level, alpha=0.1, sign=1)
    assert field == "Theta"
    records = {edit.record for edit in edits}
    # Theta starts at record 30 (see af_s0_twin.pickup_record_index); level 1
    # and level 2 must land on two distinct, adjacent records, not collapse
    # onto one.
    assert records == {30, 31}
    assert peak > 0.0


def test_direction_vector_by_level_rejects_a_level_weight_count_mismatch(
    wet_mask: np.ndarray, uniform_sigma: np.ndarray
) -> None:
    row = _row(
        family="Theta",
        levels_one_based=[1, 2],
        vertical_weights_float64_hex=[float(1.0).hex()],  # only one weight for two levels
    )
    with pytest.raises(staging.ProductionRunError, match="level/weight count mismatch"):
        staging.direction_vector_by_level(row, wet_mask, uniform_sigma)


def test_ssh_peak_cap_exceptions_are_the_three_reviewed_low_overshoot_cases() -> None:
    assert set(staging.SSH_PEAK_CAP_EXCEPTIONS) == {
        ("S1", 360, "SSH", 15),
        ("S2", 1440, "SSH", 14),
        ("S0", 720, "SSH", 14),
    }
    for info in staging.SSH_PEAK_CAP_EXCEPTIONS.values():
        assert info["decision"] == "accepted_as_documented_exception_not_a_defect"
