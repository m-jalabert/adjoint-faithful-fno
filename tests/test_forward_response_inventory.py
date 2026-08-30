"""No-compute tests for the frozen forward-response inventory machinery."""

from __future__ import annotations

import math
import stat
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_forward_response_inventory as inventory  # noqa: E402


DATASET_CONTRACT = PROJECT_ROOT / "config" / "forward_response_dataset_v3.json"
PILOT_CONTRACT = PROJECT_ROOT / "config" / "forward_response_amplitude_pilot_v1.json"


@pytest.fixture(scope="module")
def contract() -> dict:
    return inventory.load_json_strict(DATASET_CONTRACT)


@pytest.fixture(scope="module")
def pilot_contract() -> dict:
    return inventory.load_json_strict(PILOT_CONTRACT)


@pytest.fixture(scope="module")
def direction_slots(contract: dict, pilot_contract: dict) -> list[inventory.Direction]:
    slots = inventory.build_direction_slots(contract, pilot_contract)
    return inventory.assign_region_slots(slots, contract)


@pytest.fixture(scope="module")
def repaired_direction_slots(
    direction_slots: list[inventory.Direction], contract: dict
) -> tuple[list[inventory.Direction], dict]:
    return inventory.repair_region_slots_for_long_feasibility(direction_slots, contract)


def _full_grid(ny: int = 9, nx: int = 10) -> inventory.Grid:
    wet = np.ones((ny, nx), dtype=np.float64)
    active = np.ones((inventory.NR, ny, nx), dtype=np.float64)
    lon = np.broadcast_to(np.arange(nx, dtype=np.float64), (ny, nx)).copy()
    lat = np.broadcast_to(np.arange(ny, dtype=np.float64)[:, None], (ny, nx)).copy()
    return inventory.Grid(wet, active, active.copy(), active.copy(), lon, lat, lon, lat)


def _write_pickup(root: Path, day: int, *, marker: bytes = b"") -> None:
    iteration = inventory.BASE_ITERATION + inventory.STEPS_PER_DAY * day
    stem = root / f"pickup.{iteration:010d}"
    metadata = (
        " nDims = [ 2 ];\n"
        " dimList = [ 62, 1, 62, 62, 1, 62 ];\n"
        " dataprec = [ 'float64' ];\n"
        " nrecords = [ 108 ];\n"
        f" timeStepNumber = [ {iteration} ];\n"
    )
    Path(str(stem) + ".meta").write_text(metadata)
    payload = bytearray(108 * 62 * 62 * 8)
    payload[: len(marker)] = marker
    Path(str(stem) + ".data").write_bytes(payload)


def _candidate(row: inventory.Direction, j: int, i: int, speed: float = 0.0) -> inventory.Candidate:
    assert row.region is not None
    return inventory.Candidate(
        j=j,
        i=i,
        lon=float(i),
        lat=float(j),
        region=row.region,
        subregion="q0",
        centroid_lon=float(i),
        centroid_lat=float(j),
        tertiary_distance_km=0.0,
        tie_sha256=inventory.tie_sha(
            row.role,
            row.regime,
            row.family,
            row.support_name,
            row.region,
            j,
            i,
        ),
        wbc_speed=speed,
    )


def test_strict_json_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"outer":{"same":1,"same":2}}')
    with pytest.raises(inventory.ContractError, match="duplicate JSON key 'same'"):
        inventory.load_json_strict(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_loader_rejects_nonfinite_constants(tmp_path: Path, constant: str) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text(f'{{"value":{constant}}}')
    with pytest.raises(inventory.ContractError, match="non-finite JSON constant"):
        inventory.load_json_strict(path)


def test_current_contract_freezes_every_algorithmic_convention(contract: dict) -> None:
    inventory.validate_frozen_algorithm_contract(contract)
    assert contract["joint_spatial_allocator"]["tie_hash_grammar"] == inventory.TIE_GRAMMAR
    assert contract["selected_amplitudes"] is None


def test_generic_mds_reader_uses_metadata_dimensions_and_precision(tmp_path: Path) -> None:
    meta = tmp_path / "odd.meta"
    meta.write_text(
        " nDims = [ 3 ];\n"
        " dimList = [ 4, 1, 4, 3, 1, 3, 2, 1, 2 ];\n"
        " dataprec = [ 'float64' ];\n"
        " nrecords = [ 2 ];\n"
        " timeStepNumber = [ 123 ];\n"
    )
    source = np.arange(2 * 2 * 3 * 4, dtype=">f8")
    source.tofile(meta.with_suffix(".data"))
    parsed, value = inventory.read_mds(meta)
    assert parsed.dimensions == (4, 3, 2)
    assert parsed.timestep == 123
    assert value.shape == (2, 2, 3, 4)
    assert value.dtype == np.dtype(">f8")
    assert float(value[-1, -1, -1, -1]) == 47.0


def test_mds_reader_rejects_trailing_or_short_data(tmp_path: Path) -> None:
    meta = tmp_path / "bad.meta"
    meta.write_text("dimList = [ 2,1,2, 2,1,2 ];\ndataprec = [ 'float32' ];\nnrecords = [ 1 ];\n")
    np.arange(5, dtype=">f4").tofile(meta.with_suffix(".data"))
    with pytest.raises(inventory.SourceError, match="expected 4 values, got 5"):
        inventory.read_mds(meta)


def test_source_resolver_uses_only_supplied_canonical_roots_and_checks_duplicates(
    tmp_path: Path,
) -> None:
    first = tmp_path / "canonical-a"
    second = tmp_path / "canonical-b"
    unrelated = tmp_path / "global-scratch-lookalike"
    first.mkdir()
    second.mkdir()
    unrelated.mkdir()
    _write_pickup(first, 3600, marker=b"same")
    _write_pickup(second, 3600, marker=b"same")
    _write_pickup(unrelated, 3600, marker=b"different")
    resolved = inventory.resolve_annual_pickup("S0", 3600, {"S0": (first, second)})
    assert len(resolved.candidates) == 2
    assert resolved.canonical.segment == str(first)
    assert "source-manifest" in resolved.canonical_choice_reason
    data = Path(resolved.candidates[1].data_path)
    payload = bytearray(data.read_bytes())
    payload[10] = 1
    data.write_bytes(payload)
    with pytest.raises(inventory.SourceError, match="conflicting canonical-chain"):
        inventory.resolve_annual_pickup("S0", 3600, {"S0": (first, second)})


def test_pickup_projection_casts_before_trusted_centering_and_resets_land(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pickup"
    root.mkdir()
    _write_pickup(root, 0)
    iteration = inventory.BASE_ITERATION
    meta = root / f"pickup.{iteration:010d}.meta"
    data = root / f"pickup.{iteration:010d}.data"
    records = np.zeros((108, 62, 62), dtype=">f8")
    records[0, 4, 4:7] = (1.0, 3.0, 7.0)
    records[15, 4:7, 4] = (2.0, 6.0, 10.0)
    records[30, 4, 4] = 11.0
    records[105, 4, 4] = 13.0
    records.tofile(data)
    wet = np.ones((62, 62), dtype=bool)
    wet[4, 5] = False
    projected = inventory.pickup_to_trajectory_p32(meta, wet)
    assert projected.shape == (46, 62, 62)
    assert projected.dtype == np.float32
    assert projected[0, 4, 4] == np.float32(2.0)
    assert projected[15, 4, 4] == np.float32(4.0)
    assert projected[30, 4, 4] == np.float32(11.0)
    assert projected[45, 4, 4] == np.float32(13.0)
    assert np.all(projected[:, 4, 5] == 0.0)


def test_region_precedence_is_exact_on_overlapping_boundary_rows() -> None:
    wet = np.zeros((24, 14), dtype=bool)
    wet[:, 1:13] = True
    masks = inventory.region_masks(wet)
    assert masks["WBC"][:, 1:5].all()
    assert masks["eastern"][:, 9:13].all()
    assert masks["southern"][:10, 5:9].all()
    assert masks["northern"][-10:, 5:9].all()
    assert masks["interior"][10:14, 5:9].all()
    assert not any(
        np.any(masks[a] & masks[b])
        for a, b in __import__("itertools").combinations(inventory.REGIONS, 2)
    )
    assert np.array_equal(np.logical_or.reduce(tuple(masks.values())), wet)


def test_projected_c_grid_footprints_match_the_centering_equations() -> None:
    u = inventory.projected_footprint("U", "gaussian_5x5_sigma1", 5, 5)
    v = inventory.projected_footprint("V", "gaussian_5x5_sigma1", 5, 5)
    theta = inventory.projected_footprint("Theta", "gaussian_5x5_sigma1", 5, 5)
    assert len(u) == 30 and {i for _j, i in u} == set(range(2, 8))
    assert len(v) == 30 and {j for j, _i in v} == set(range(2, 8))
    assert len(theta) == 25


def test_full_support_and_phase_a_exclusion_are_applied_to_projected_footprints() -> None:
    grid = _full_grid(25, 25)
    without_target = inventory.candidate_mask(
        grid, "U", (1,), "gaussian_5x5_sigma1", phase_a_target=()
    )
    with_target = inventory.candidate_mask(
        grid,
        "U",
        (1,),
        "gaussian_5x5_sigma1",
        phase_a_target=((12, 5),),
    )
    assert int(without_target.sum()) == 21 * 21
    for j, i in np.argwhere(with_target):
        assert (12, 5) not in inventory.projected_footprint(
            "U", "gaussian_5x5_sigma1", int(j), int(i)
        )
    assert with_target.sum() < without_target.sum()


def test_u_v_carrier_tracer_must_be_active_at_every_requested_level() -> None:
    grid = _full_grid(9, 9)
    grid.hfac_c[4, 4, 4] = 0.0
    level_one = inventory.candidate_mask(grid, "U", (1,), "gaussian_5x5_sigma1", phase_a_target=())
    level_five = inventory.candidate_mask(grid, "U", (5,), "gaussian_5x5_sigma1", phase_a_target=())
    assert level_one[4, 4]
    assert not level_five[4, 4]


def test_level_support_token_uses_pinned_binary64_hex() -> None:
    two = 1.0 / math.sqrt(2.0)
    token = inventory.level_support_name((1, 2), (two, two))
    assert token == (
        '{"levels":[1,2],"weights_hex":["0x1.6a09e667f3bccp-1","0x1.6a09e667f3bccp-1"]}'
    )
    assert inventory.level_support_name((), ()) == '{"levels":[],"weights_hex":[]}'


def test_vertical_anchor_formulas_and_direction_counts_are_exact(
    direction_slots: list[inventory.Direction], contract: dict
) -> None:
    report = inventory.validate_direction_contract(direction_slots, contract)
    assert report["role_counts"] == {
        "blind_test": 216,
        "pilot": 24,
        "train": 672,
        "validation": 216,
    }
    for regime in inventory.REGIMES:
        for family in ("U", "V", "Theta"):
            validation = [
                row
                for row in direction_slots
                if row.role == "validation" and row.regime == regime and row.family == family
            ]
            assert Counter(row.levels[0] for row in validation if len(row.levels) == 1) == Counter(
                range(1, 16)
            )
            assert Counter(row.anchor_slot for row in validation) == Counter({0: 6, 1: 6, 2: 6})


def test_region_slot_assignment_is_hash_ordered_quota_exact_and_repeatable(
    contract: dict, pilot_contract: dict
) -> None:
    raw = inventory.build_direction_slots(contract, pilot_contract)
    first = inventory.assign_region_slots(raw, contract)
    second = inventory.assign_region_slots(list(reversed(raw)), contract)
    assert {row.slot_id: row.region for row in first} == {row.slot_id: row.region for row in second}
    group = [
        row for row in first if row.role == "train" and row.regime == "S0" and row.family == "U"
    ]
    assert Counter(row.region for row in group) == Counter(
        {"WBC": 20, "interior": 9, "eastern": 9, "northern": 9, "southern": 9}
    )


def test_minimum_hamming_repair_is_general_exact_and_quota_preserving(
    repaired_direction_slots: tuple[list[inventory.Direction], dict], contract: dict
) -> None:
    repaired, report = repaired_direction_slots
    assert report["initial_complete_long_feasible"] is False
    assert report["failing_initial_components"] == ["validation/joint/Theta"]
    assert report["minimum_hamming_changes"] == 2
    theta = next(
        component
        for component in report["component_reports"]
        if component["component"] == "validation/joint/Theta"
    )
    assert theta["public_changes"] == [
        {
            "anchor_day": 6050,
            "anchor_index": 1,
            "direction_slot": 12,
            "family": "Theta",
            "from_region": "WBC",
            "level_support_token": ('{"levels":[1],"weights_hex":["0x1.0000000000000p+0"]}'),
            "regime": "S0",
            "role": "validation",
            "to_region": "southern",
        },
        {
            "anchor_day": 6050,
            "anchor_index": 1,
            "direction_slot": 15,
            "family": "Theta",
            "from_region": "southern",
            "level_support_token": ('{"levels":[7],"weights_hex":["0x1.0000000000000p+0"]}'),
            "regime": "S0",
            "role": "validation",
            "to_region": "WBC",
        },
    ]
    inventory.validate_direction_contract(repaired, contract)
    validation_theta = [
        row
        for row in repaired
        if row.role == "validation" and row.regime == "S0" and row.family == "Theta"
    ]
    assert Counter(row.region for row in validation_theta) == Counter(
        {"WBC": 6, "interior": 3, "eastern": 3, "northern": 3, "southern": 3}
    )


def test_repair_retains_every_component_whose_sha_zip_was_feasible(
    direction_slots: list[inventory.Direction],
    repaired_direction_slots: tuple[list[inventory.Direction], dict],
) -> None:
    repaired, report = repaired_direction_slots
    changed = {
        row.slot_id
        for row in repaired
        if row.region
        != next(original.region for original in direction_slots if original.slot_id == row.slot_id)
    }
    certified = {
        f"{change['role']}:{change['regime']}:d{change['anchor_day']:04d}:"
        f"{change['family']}:{change['direction_slot']}"
        for component in report["component_reports"]
        for change in component["public_changes"]
    }
    assert len(changed) == len(certified) == 2
    assert all("validation:S0" in value and "Theta" in value for value in certified)


def test_complete_long_membership_constraints_pass_after_the_general_repair(
    repaired_direction_slots: tuple[list[inventory.Direction], dict], contract: dict
) -> None:
    repaired, _report = repaired_direction_slots
    counters: Counter[tuple[str, str, str]] = Counter()
    centred: list[inventory.Direction] = []
    for row in repaired:
        key = (row.role, row.regime, row.family)
        index = counters[key]
        counters[key] += 1
        centred.append(
            inventory.replace(
                row,
                j=index // 62,
                i=index % 62,
                lon=float(index % 62),
                lat=float(index // 62),
            )
        )
    solved, report = inventory.solve_long_membership(centred, contract)
    assert report["long_counts"] == {
        "blind_test": 36,
        "pilot": 12,
        "train": 96,
        "validation": 36,
    }
    for role in ("validation", "blind_test"):
        for family in inventory.FAMILIES:
            selected = [
                row for row in solved if row.long and row.role == role and row.family == family
            ]
            assert {row.region for row in selected} == set(inventory.REGIONS)


def test_candidate_sha_uses_literal_frozen_preimage() -> None:
    token = '{"levels":[1],"weights_hex":["0x1.0000000000000p+0"]}'
    preimage = f"response-v1|train|S0|U|{token}|WBC|14|3"
    import hashlib

    assert (
        inventory.tie_sha("train", "S0", "U", token, "WBC", 14, 3)
        == hashlib.sha256(preimage.encode("utf-8")).hexdigest()
    )


def test_region_repair_sha_uses_the_exact_frozen_grammar(
    direction_slots: list[inventory.Direction],
) -> None:
    import hashlib

    row = next(
        row
        for row in direction_slots
        if row.role == "validation" and row.regime == "S0" and row.family == "Theta"
    )
    preimage = (
        "response-v1|region-repair|"
        f"{row.role}|{row.regime}|{row.family}|{row.kernel}|{row.anchor_slot}|"
        f"{row.direction_slot}|{row.support_name}|southern"
    )
    assert (
        inventory._region_repair_sha(row, "southern")
        == hashlib.sha256(preimage.encode("utf-8")).hexdigest()
    )


def test_contract_freezes_ssh_cross_kernel_ownership_after_physical_objectives(
    contract: dict,
) -> None:
    rule = contract["joint_spatial_allocator"]["joint_objective_scope"][
        "ssh_cross_kernel_ownership_tie_break"
    ]
    assert "(candidate tie SHA-256, region-slot SHA-256)" in rule
    assert "(1 before 0)" in rule


def test_equal_rank_quartiles_use_candidate_sha_for_coordinate_ties() -> None:
    row = inventory.Direction(
        "train", "S0", 0, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "WBC"
    )
    raw = [(index, 2, float(index), float(index // 2), "unused") for index in range(8)]
    labels, centroids = inventory._candidate_subregions(raw, row=row)
    assert Counter(labels.values()) == Counter({"q0": 2, "q1": 2, "q2": 2, "q3": 2})
    assert set(centroids) == {"q0", "q1", "q2", "q3"}


def test_interior_midpoint_belongs_to_west_and_south() -> None:
    row = inventory.Direction(
        "train", "S0", 0, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "interior"
    )
    raw = [
        (0, 0, -1.0, -1.0, "a"),
        (0, 1, -1.0, 1.0, "b"),
        (1, 0, 1.0, -1.0, "c"),
        (1, 1, 1.0, 1.0, "d"),
        (2, 2, 0.0, 0.0, "e"),
    ]
    labels, _centroids = inventory._candidate_subregions(raw, row=row)
    assert labels[(2, 2)] == "ws"


def test_spherical_centroid_handles_longitude_wrap() -> None:
    lon, lat = inventory.spherical_centroid((179.0, -179.0), (0.0, 0.0))
    assert abs(abs(lon) - 180.0) < 1e-12
    assert lat == pytest.approx(0.0, abs=1e-12)


def test_wbc_speed_reads_already_centered_channels_without_a_second_roll() -> None:
    state = np.zeros((1, 4, 46, 2, 4), dtype=np.float32)
    state[0, :, 0] = np.asarray([[1.0, 2.0, 4.0, 8.0], [1.0, 2.0, 4.0, 8.0]])
    state[0, :, 15] = 0.0
    speed = inventory.mean_surface_speed_already_centered(
        state, 0, start_day=0, stop_day=4, chunk_days=2
    )
    assert np.array_equal(speed, state[0, 0, 0].astype(np.float64))
    assert not np.array_equal(speed, 0.5 * (speed + np.roll(speed, -1, axis=-1)))


def test_real_grid_masks_match_the_frozen_capacity_counts() -> None:
    root = Path(
        "/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm/S0/production/years_100_110"
    )
    if not (root / "hFacC.meta").is_file():
        pytest.skip("canonical scratch grid is unavailable")
    grid = inventory.read_grid(root)
    masks = inventory.region_masks(grid.wet)
    expected = {
        "U": {"WBC": 47, "interior": 2080, "eastern": 112, "northern": 416, "southern": 416},
        "V": {"WBC": 100, "interior": 2080, "eastern": 110, "northern": 416, "southern": 364},
        "Theta": {"WBC": 103, "interior": 2080, "eastern": 112, "northern": 416, "southern": 416},
    }
    for family, regional in expected.items():
        eligible = inventory.candidate_mask(grid, family, (1,), "gaussian_5x5_sigma1")
        assert {
            region: int((eligible & masks[region]).sum()) for region in inventory.REGIONS
        } == regional


def test_greedy_farthest_point_solver_is_deterministic_on_a_small_problem() -> None:
    rows = [
        inventory.Direction(
            "train", "S0", 0, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "WBC"
        ),
        inventory.Direction(
            "train", "S0", 0, 0, 1, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "WBC"
        ),
        inventory.Direction(
            "validation", "S0", 6010, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "WBC"
        ),
        inventory.Direction(
            "validation", "S0", 6010, 0, 1, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "WBC"
        ),
    ]
    candidates = {
        row.slot_id: [_candidate(row, 5, i, speed=float(i * i)) for i in range(6)] for row in rows
    }
    first, first_objective = inventory.allocate_centres_greedy_farthest_point(rows, candidates)
    second, second_objective = inventory.allocate_centres_greedy_farthest_point(rows, candidates)
    assert [(row.slot_id, row.j, row.i) for row in first] == [
        (row.slot_id, row.j, row.i) for row in second
    ]
    assert first_objective == second_objective
    assert len({(row.j, row.i) for row in first}) == 4


def test_maximin_long_subset_uses_sha_only_after_distance() -> None:
    rows = [
        inventory.Direction(
            "train",
            "S0",
            day,
            index,
            0,
            "Theta",
            "gaussian_5x5_sigma1",
            (1,),
            (1.0,),
            "interior",
            j=2,
            i=index,
            lon=float(index),
            lat=0.0,
        )
        for index, day in enumerate((0, 720, 1800, 2520))
    ]
    model = inventory._MixedIntegerModel(4)
    model.add_constraint({index: 1.0 for index in range(4)}, 2.0, 2.0)
    selected, distance = inventory._solve_maximin_subset(rows, model)
    assert selected == {0, 3}
    assert distance == pytest.approx(float(inventory.great_circle_km(0.0, 0.0, 3.0, 0.0)))


def _has_pairwise_upper_bound(model: inventory._MixedIntegerModel, var_a: int, var_b: int) -> bool:
    columns_by_row: dict[int, dict[int, float]] = defaultdict(dict)
    for row, column, coefficient in zip(
        model.row_indices, model.column_indices, model.coefficients
    ):
        columns_by_row[row][column] = coefficient
    for row, columns in columns_by_row.items():
        if (
            set(columns) == {var_a, var_b}
            and columns[var_a] == 1.0
            and columns[var_b] == 1.0
            and model.constraint_upper[row] == 1.0
        ):
            return True
    return False


def test_train_is_exempt_from_the_distance_three_exclusion() -> None:
    pilot = inventory.Direction(
        "pilot", "S0", 0, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "interior"
    )
    train = inventory.Direction(
        "train", "S0", 720, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "interior"
    )
    validation = inventory.Direction(
        "validation", "S0", 1800, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "interior"
    )
    candidates = {
        pilot.slot_id: [_candidate(pilot, 5, 5)],
        train.slot_id: [_candidate(train, 5, 6)],
        validation.slot_id: [_candidate(validation, 5, 6)],
    }
    problem = inventory._build_centre_problem([pilot, train], candidates)
    pilot_variable = problem.y_variables[("pilot", 5, 5)]
    train_variable = problem.y_variables[("train", 5, 6)]
    assert not _has_pairwise_upper_bound(problem.model, pilot_variable, train_variable)

    problem = inventory._build_centre_problem([pilot, validation], candidates)
    pilot_variable = problem.y_variables[("pilot", 5, 5)]
    validation_variable = problem.y_variables[("validation", 5, 6)]
    assert _has_pairwise_upper_bound(problem.model, pilot_variable, validation_variable)


def test_apply_fixed_centres_collapses_pilot_rows_to_their_frozen_choice() -> None:
    pilot = inventory.Direction(
        "pilot", "S0", 720, 0, 0, "U", "gaussian_5x5_sigma1", (1,), (1.0,), "WBC"
    )
    train = inventory.Direction(
        "train", "S0", 720, 0, 0, "U", "gaussian_5x5_sigma1", (1,), (1.0,), "WBC"
    )
    candidates = {
        pilot.slot_id: [_candidate(pilot, 3, 4), _candidate(pilot, 3, 5)],
        train.slot_id: [_candidate(train, 3, 4), _candidate(train, 3, 5)],
    }
    fixed = {("S0", 720, "U"): (3, 5)}
    reduced = inventory.apply_fixed_centres([pilot, train], candidates, fixed)
    assert reduced[pilot.slot_id] == (candidates[pilot.slot_id][1],)
    assert reduced[train.slot_id] == candidates[train.slot_id]


def test_apply_fixed_centres_rejects_a_frozen_centre_no_longer_enumerated() -> None:
    pilot = inventory.Direction(
        "pilot", "S0", 720, 0, 0, "U", "gaussian_5x5_sigma1", (1,), (1.0,), "WBC"
    )
    candidates = {pilot.slot_id: [_candidate(pilot, 3, 4)]}
    fixed = {("S0", 720, "U"): (9, 9)}
    with pytest.raises(inventory.ContractError):
        inventory.apply_fixed_centres([pilot], candidates, fixed)


def test_prove_hard_capacity_stays_fast_with_a_large_unreduced_candidate_pool() -> None:
    grid = _full_grid(ny=64, nx=64)
    masks = {region: np.zeros((64, 64), dtype=bool) for region in inventory.REGIONS}
    masks["interior"][:, :] = True
    rows = [
        inventory.Direction(
            role,
            "S0",
            day,
            0,
            index,
            "Theta",
            "gaussian_5x5_sigma1",
            (1,),
            (1.0,),
            "interior",
        )
        for role, day in (("train", 0), ("validation", 6010))
        for index in range(2)
    ]
    start = time.monotonic()
    report = inventory.prove_hard_capacity(grid, rows, masks)
    elapsed = time.monotonic() - start
    assert report["witness_rows"] == len(rows)
    assert report["centre_ids_distinct_across_roles"] is True
    assert elapsed < 60.0


def test_prove_hard_capacity_lets_pilot_and_train_share_close_centres() -> None:
    """A witness where pilot and train end up adjacent must still pass.

    Only 5 candidates exist in "interior", collinear one cell apart. Fitting
    pilot, train, and validation requires validation to sit >= 3 from *both*
    earlier picks while pilot and train are free to land next to each other
    -- no arrangement fits if pilot-train were (incorrectly) also held to
    distance >= 3, so this is only satisfiable, and only verifiable without
    a false CapacityError, when the pilot/train exemption is applied
    identically by both the solver and this function's post-hoc check.
    """

    grid = _full_grid(ny=9, nx=9)
    masks = {region: np.zeros((9, 9), dtype=bool) for region in inventory.REGIONS}
    for i in range(2, 7):
        masks["interior"][4, i] = True
    rows = [
        inventory.Direction(
            "pilot", "S0", 0, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "interior"
        ),
        inventory.Direction(
            "train", "S0", 360, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "interior"
        ),
        inventory.Direction(
            "validation",
            "S0",
            6010,
            0,
            0,
            "Theta",
            "gaussian_5x5_sigma1",
            (1,),
            (1.0,),
            "interior",
        ),
    ]
    report = inventory.prove_hard_capacity(grid, rows, masks)
    assert report["witness_rows"] == 3
    assert report["centre_ids_distinct_across_roles"] is True
    assert report["non_wbc_cross_role_chebyshev_min"] == 3


def test_load_pilot_fixed_centres_reads_the_frozen_geometry_file(tmp_path: Path) -> None:
    geometry = tmp_path / "amplitude_pilot_geometry_v1.json"
    geometry.write_text(
        '{"version": "amplitude_pilot_geometry_v1", "grid_wet_tracer_cells": 1, '
        '"directions": [{"role": "pilot", "regime": "S0", "anchor_day": 720, '
        '"family": "U", "j": 3, "i": 5, "region": "WBC", "lon": 1.0, "lat": 1.0, '
        '"kernel": "gaussian_5x5_sigma1", "levels": [1], "long": true}]}'
    )
    fixed = inventory.load_pilot_fixed_centres(geometry)
    assert fixed == {("S0", 720, "U"): (3, 5)}


@pytest.mark.parametrize("final_mode", [0o400, 0o444])
def test_exclusive_inventory_writer_seals_mode_and_refuses_overwrite(
    tmp_path: Path, final_mode: int
) -> None:
    output = tmp_path / "inventory.jsonl"
    rows = [{"b": 2, "a": 1}]
    digest = inventory._write_jsonl_exclusive(output, rows, final_mode=final_mode)
    assert output.read_bytes() == b'{"a":1,"b":2}\n'
    assert digest == inventory.hashlib.sha256(output.read_bytes()).hexdigest()
    assert stat.S_IMODE(output.stat().st_mode) == final_mode
    with pytest.raises(FileExistsError):
        inventory._write_jsonl_exclusive(output, rows, final_mode=final_mode)


def test_geometry_rows_are_separate_and_contain_no_numeric_response() -> None:
    row = inventory.Direction(
        "blind_test",
        "S0",
        7560,
        0,
        0,
        "SSH",
        "point",
        (),
        (),
        "WBC",
        j=4,
        i=2,
        lon=1.0,
        lat=2.0,
        long=True,
    )
    serialized = inventory.inventory_row(row)
    assert serialized["numeric_response_present"] is False
    assert "response" not in serialized
    assert "alpha" not in serialized
    assert serialized["role"] == "blind_test"
    assert serialized["horizon_days"] == 90


def _far_candidate(
    row: inventory.Direction, j: int, i: int, lon: float, lat: float, tertiary: float = 1.0
) -> inventory.Candidate:
    return inventory.Candidate(
        j=j,
        i=i,
        lon=lon,
        lat=lat,
        region=row.region,
        subregion="q0",
        centroid_lon=lon,
        centroid_lat=lat,
        tertiary_distance_km=tertiary,
        tie_sha256=inventory.tie_sha(
            row.role, row.regime, row.family, row.support_name, row.region, j, i
        ),
    )


def test_fix_pins_variable_via_bounds_not_a_new_constraint_row() -> None:
    model = inventory._MixedIntegerModel(2)
    rows_before = len(model.constraint_lower)
    model.fix(0, 1)
    assert len(model.constraint_lower) == rows_before
    assert model.variable_lower[0] == 1.0
    assert model.variable_upper[0] == 1.0
    result = model.solve()
    assert result is not None
    assert np.rint(result.x[0]) == 1.0


def test_pick_farthest_candidate_maximizes_cross_role_distance_first() -> None:
    """With an already-placed other-role point, farthest-point must win over the tertiary score."""

    train = inventory.Direction(
        "train", "S0", 0, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "interior"
    )
    validation = inventory.Direction(
        "validation", "S0", 6010, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "interior"
    )
    placed_train = _far_candidate(train, 0, 0, 0.0, 0.0)
    near = _far_candidate(validation, 1, 1, 1.0, 1.0, tertiary=1.0)  # close, but preferred tertiary
    far = _far_candidate(validation, 10, 10, 10.0, 10.0, tertiary=100.0)  # far, disfavoured tertiary
    winner = inventory._pick_farthest_candidate(
        [near, far], "validation", {"train": [placed_train]}
    )
    assert (winner.j, winner.i) == (10, 10)


def test_pick_farthest_candidate_falls_back_to_tertiary_when_nothing_placed() -> None:
    validation = inventory.Direction(
        "validation", "S0", 6010, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "interior"
    )
    close_to_centroid = _far_candidate(validation, 1, 1, 1.0, 1.0, tertiary=1.0)
    far_from_centroid = _far_candidate(validation, 10, 10, 10.0, 10.0, tertiary=100.0)
    winner = inventory._pick_farthest_candidate([close_to_centroid, far_from_centroid], "validation", {})
    assert (winner.j, winner.i) == (1, 1)


def test_allocate_centres_greedy_farthest_point_respects_distance_three_and_distinctness() -> None:
    train = inventory.Direction(
        "train", "S0", 0, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "interior"
    )
    validation = inventory.Direction(
        "validation", "S0", 6010, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "interior"
    )
    candidates = {
        train.slot_id: [_far_candidate(train, 10, 10, 10.0, 10.0)],
        validation.slot_id: [
            _far_candidate(validation, 11, 11, 11.0, 11.0, tertiary=1.0),  # blocked: <3 from train
            _far_candidate(validation, 20, 20, 20.0, 20.0, tertiary=100.0),  # only legal choice
        ],
    }
    mapped, _objective = inventory.allocate_centres_greedy_farthest_point([train, validation], candidates)
    validation_row = next(row for row in mapped if row.role == "validation")
    assert (validation_row.j, validation_row.i) == (20, 20)


def test_allocate_centres_lexicographically_by_region_places_two_independent_regions() -> None:
    """Two regions with no boundary conflict: distinct cells, no repair needed, real separation achieved."""

    rows = []
    candidates: dict[str, list[inventory.Candidate]] = {}
    # Direction.slot_id deliberately omits region (it is the role/regime/day/
    # family/kernel/support identity), so two regions sharing an anchor_day
    # would collide in the candidates dict keyed by slot_id -- offset each
    # region's anchor_day to keep the two independent test rows apart.
    for offset, (region, base_j, base_i) in enumerate((("eastern", 0, 0), ("northern", 50, 50))):
        day_shift = offset * 100_000
        train_a = inventory.Direction(
            "train", "S0", 0 + day_shift, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), region
        )
        train_b = inventory.Direction(
            "train", "S0", 360 + day_shift, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), region
        )
        validation = inventory.Direction(
            "validation",
            "S0",
            6010 + day_shift,
            0,
            0,
            "Theta",
            "gaussian_5x5_sigma1",
            (1,),
            (1.0,),
            region,
        )
        candidates[train_a.slot_id] = [
            _far_candidate(train_a, base_j, base_i, float(base_i), float(base_j)),
            _far_candidate(train_a, base_j, base_i + 1, float(base_i + 1), float(base_j)),
        ]
        candidates[train_b.slot_id] = [
            _far_candidate(train_b, base_j + 1, base_i, float(base_i), float(base_j + 1)),
            _far_candidate(train_b, base_j + 1, base_i + 1, float(base_i + 1), float(base_j + 1)),
        ]
        candidates[validation.slot_id] = [
            _far_candidate(validation, base_j + 5, base_i + 5, float(base_i + 5), float(base_j + 5))
        ]
        rows.extend([train_a, train_b, validation])

    mapped, objective = inventory.allocate_centres_lexicographically_by_region(rows, candidates)
    assert sorted(row.slot_id for row in mapped) == sorted(row.slot_id for row in rows)
    assert len({(row.j, row.i) for row in mapped}) == len(mapped)
    assert objective["cross_region_repair_attempts"] == 0
    assert all(value > 0.0 for value in objective["cross_role_region_minima_sorted_km"])
    assert all(value > 0.0 for value in objective["within_role_region_minima_sorted_km"])


def test_non_wbc_chebyshev_violations_flags_a_cross_region_boundary_conflict() -> None:
    train = inventory.Direction(
        "train", "S0", 0, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "eastern",
        j=10, i=10, lon=10.0, lat=10.0,
    )
    validation_close = inventory.Direction(
        "validation", "S0", 6010, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "northern",
        j=11, i=11, lon=11.0, lat=11.0,
    )
    violations = inventory._non_wbc_chebyshev_violations([train, validation_close])
    assert [row.slot_id for row in violations] == [validation_close.slot_id]

    validation_far = inventory.Direction(
        "validation", "S0", 6010, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "northern",
        j=99, i=99, lon=99.0, lat=99.0,
    )
    assert inventory._non_wbc_chebyshev_violations([train, validation_far]) == []


def test_allocate_centres_lexicographically_by_region_repairs_a_cross_region_conflict() -> None:
    """A validation centre whose unconstrained-optimal choice conflicts across a region
    boundary must be excluded and re-solved onto its remaining, farther candidate."""

    train = inventory.Direction(
        "train", "S0", 0, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "eastern"
    )
    validation = inventory.Direction(
        "validation", "S0", 6010, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "northern"
    )
    candidates = {
        train.slot_id: [_far_candidate(train, 10, 10, 10.0, 10.0)],
        validation.slot_id: [
            _far_candidate(validation, 11, 11, 11.0, 11.0, tertiary=1.0),  # preferred, conflicts
            _far_candidate(validation, 20, 20, 20.0, 20.0, tertiary=100.0),  # safe, disfavoured
        ],
    }
    mapped, objective = inventory.allocate_centres_lexicographically_by_region(
        [train, validation], candidates
    )
    validation_row = next(row for row in mapped if row.role == "validation")
    assert (validation_row.j, validation_row.i) == (20, 20)
    assert objective["cross_region_repair_attempts"] >= 1
    assert inventory._non_wbc_chebyshev_violations(mapped) == []


def test_allocate_centres_lexicographically_by_region_raises_when_repair_cannot_converge() -> None:
    train = inventory.Direction(
        "train", "S0", 0, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "eastern"
    )
    validation = inventory.Direction(
        "validation", "S0", 6010, 0, 0, "Theta", "gaussian_5x5_sigma1", (1,), (1.0,), "northern"
    )
    candidates = {
        train.slot_id: [_far_candidate(train, 10, 10, 10.0, 10.0)],
        validation.slot_id: [_far_candidate(validation, 11, 11, 11.0, 11.0)],
    }
    with pytest.raises(inventory.CapacityError, match="no candidates remain"):
        inventory.allocate_centres_lexicographically_by_region([train, validation], candidates)


def test_materialize_inventory_requires_both_output_parents_before_writing(tmp_path: Path) -> None:
    """Reproduces the fixed ordering bug: if only blind's parent existed, blind used to get
    sealed (O_EXCL, mode 0400) before the missing public parent was ever noticed."""

    blind_parent = tmp_path / "blind_only"
    blind_parent.mkdir()
    missing_public_parent = tmp_path / "missing_public"
    with pytest.raises(inventory.InventoryError, match="output parent does not exist"):
        inventory.materialize_inventory(
            dataset_contract_path=tmp_path / "irrelevant_dataset.json",
            pilot_contract_path=tmp_path / "irrelevant_pilot.json",
            pilot_geometry_path=tmp_path / "irrelevant_geometry.json",
            development_output=missing_public_parent / "public.jsonl",
            blind_output=blind_parent / "blind.jsonl",
        )
    assert list(blind_parent.iterdir()) == []
