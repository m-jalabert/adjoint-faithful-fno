"""Tests for the MITgcm adjoint ground-truth setup.

Runnable without MITgcm, without TAF, and without a GPU.  These guard the
parts of docs/mitgcm_adjoint_ground_truth_plan.md that can be wrong silently:
the day-to-iteration map, the frozen target contract, the wet-area constant,
the tile decomposition identity inside cost_test.F, and the compile-time
options that the adjoint build depends on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from make_cost_weight import MDS_DTYPE, build_weight  # noqa: E402
from select_adjoint_target import (  # noqa: E402
    BASE_ITERATION,
    CONTRACT_VERSION,
    STEPS_PER_DAY,
    centred_surface_speed,
    read_mds_2d,
    search_region,
)

CODE_AD = PROJECT_ROOT / "af_fno" / "mitgcm" / "code_ad"
INPUT_AD = PROJECT_ROOT / "af_fno" / "mitgcm" / "input_ad"
CONTRACT_PATH = PROJECT_ROOT / "config" / f"{CONTRACT_VERSION}.json"


@pytest.fixture(scope="module")
def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


# --- 1. time bookkeeping ---------------------------------------------------


@pytest.mark.parametrize(
    "day,iteration",
    [(0, 2_592_000), (7200, 3_110_400), (7210, 3_111_120), (7220, 3_111_840)],
)
def test_iteration_of_day(day: int, iteration: int) -> None:
    assert BASE_ITERATION + STEPS_PER_DAY * day == iteration


def test_contract_records_the_anchor_iterations(contract: dict) -> None:
    anchors = contract["iteration_of_day"]
    assert anchors["base_iteration"] == BASE_ITERATION
    assert anchors["steps_per_day"] == STEPS_PER_DAY
    assert anchors["day_7200"] == 3_110_400
    assert anchors["day_7210"] == 3_111_120
    assert anchors["day_7220"] == 3_111_840


# --- 2. the frozen target --------------------------------------------------


def test_contract_target_is_wet_and_in_the_search_region(contract: dict) -> None:
    zarr = pytest.importorskip("zarr")
    store = zarr.open(
        "/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/trajectories_v3.zarr",
        mode="r",
    )
    wet = np.asarray(store["wet_mask"]).astype(bool)
    j0, i0 = contract["j_index0"], contract["i_index0"]
    assert wet[j0, i0], "the frozen target cell is not wet"
    assert search_region(wet)[j0, i0], "the frozen target is outside its own search region"
    assert contract["i_global"] == i0 + 1
    assert contract["j_global"] == j0 + 1


def test_centred_surface_speed_averages_c_grid_faces_to_centres() -> None:
    # UVEL sits on the western face, VVEL on the southern face, so the centred
    # value needs the eastern / northern neighbour.
    u = np.zeros((1, 3, 3))
    v = np.zeros((1, 3, 3))
    u[0, 1, 0], u[0, 1, 1] = 2.0, 4.0
    speed = centred_surface_speed(u, v)
    assert speed[0, 1, 0] == pytest.approx(3.0)

    u[:] = 0.0
    v[0, 0, 1], v[0, 1, 1] = 1.0, 5.0
    speed = centred_surface_speed(u, v)
    assert speed[0, 0, 1] == pytest.approx(3.0)


# --- 3. the wet-area constant ----------------------------------------------


def test_wet_area_matches_the_grid_files(contract: dict) -> None:
    zarr = pytest.importorskip("zarr")
    store = zarr.open(
        "/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/trajectories_v3.zarr",
        mode="r",
    )
    wet = np.asarray(store["wet_mask"]).astype(bool)
    rac = read_mds_2d(Path(contract["grid"]["rac_source"]))
    recomputed = float((rac * wet).sum())
    assert recomputed == pytest.approx(contract["wet_area_m2"], rel=1e-12)
    assert int(wet.sum()) == contract["grid"]["wet_cell_count"] == 3600


# --- 4. the tile decomposition identity ------------------------------------


def test_tile_decomposition_reproduces_the_anomaly() -> None:
    """The identity cost_test.F relies on.

    COST_FINAL sums objf_test over tiles and then globally, so accumulating
    ``sum_ij w_ij * eta_ij`` per tile must reproduce the whole functional.
    Verified here on a 2x2 tiling of a synthetic basin, which is the same
    decomposition the real run uses (sNx=sNy=31, nPx=nPy=2).
    """

    rng = np.random.default_rng(20260812)
    ny = nx = 8
    wet = np.ones((ny, nx), dtype=bool)
    wet[0, :] = wet[-1, :] = wet[:, 0] = wet[:, -1] = False  # land rim
    rac = rng.uniform(0.5, 1.5, size=(ny, nx))
    eta = rng.normal(size=(ny, nx))
    wet_area = float((rac * wet).sum())
    j0, i0 = 3, 2

    weight = build_weight("ssh_anomaly", wet, rac, wet_area, j0, i0)

    # what COST_FINAL assembles: per-tile partial sums, then a global sum
    half_y, half_x = ny // 2, nx // 2
    objf_test = []
    for tile_j in (slice(0, half_y), slice(half_y, ny)):
        for tile_i in (slice(0, half_x), slice(half_x, nx)):
            objf_test.append(float((weight[tile_j, tile_i] * eta[tile_j, tile_i]).sum()))
    assembled = sum(objf_test)

    expected = eta[j0, i0] - float((rac * wet * eta).sum()) / wet_area
    assert assembled == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert len(objf_test) == 4


def test_mean_only_weight_sums_to_minus_one() -> None:
    """Gate G3's premise: the mean functional has total weight -1."""

    rng = np.random.default_rng(7)
    wet = np.ones((6, 6), dtype=bool)
    wet[0, :] = wet[-1, :] = False
    rac = rng.uniform(0.5, 1.5, size=(6, 6))
    wet_area = float((rac * wet).sum())
    weight = build_weight("mean_only", wet, rac, wet_area, 2, 2)
    assert weight.sum() == pytest.approx(-1.0, rel=1e-12)


# --- 5. the weight field on disk -------------------------------------------


@pytest.mark.parametrize("qoi", ["ssh_anomaly", "mean_only"])
def test_weight_file_is_written_and_self_consistent(qoi: str, contract: dict) -> None:
    path = PROJECT_ROOT / "work" / f"costWeight_{qoi}.bin"
    if not path.is_file():
        pytest.skip(f"{path.name} not built yet; run scripts/make_cost_weight.py --qoi {qoi}")

    zarr = pytest.importorskip("zarr")
    store = zarr.open(
        "/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/trajectories_v3.zarr",
        mode="r",
    )
    wet = np.asarray(store["wet_mask"]).astype(bool)
    stored = np.fromfile(path, dtype=MDS_DTYPE).reshape(wet.shape).astype(np.float64)

    assert stored.shape == (62, 62)
    assert np.all(stored[~wet] == 0.0), "weight must be exactly zero on land"
    assert np.isfinite(stored).all()

    j0, i0 = contract["j_index0"], contract["i_index0"]
    rac = read_mds_2d(Path(contract["grid"]["rac_source"]))
    mean_term = -rac[j0, i0] / contract["wet_area_m2"]
    if qoi == "ssh_anomaly":
        assert stored[j0, i0] == pytest.approx(1.0 + mean_term, rel=1e-6)
        # mean-free by construction, to float32 round-off
        assert abs(stored[wet].sum()) < 1e-6
    else:
        assert stored[j0, i0] == pytest.approx(mean_term, rel=1e-6)
        assert stored[wet].sum() == pytest.approx(-1.0, rel=1e-6)


# --- 6. compile-time options the adjoint build depends on ------------------


def test_code_ad_has_every_required_file() -> None:
    required = {
        "SIZE.h",
        "DIAGNOSTICS_SIZE.h",
        "CPP_OPTIONS.h",
        "AUTODIFF_OPTIONS.h",
        "COST_OPTIONS.h",
        "CTRL_OPTIONS.h",
        "CTRL_SIZE.h",
        "tamc.h",
        "cost.h",
        "cost_test.F",
        "cost_readparms.F",
        "packages.conf",
    }
    present = {p.name for p in CODE_AD.iterdir() if p.is_file()}
    assert required <= present, f"missing from code_ad: {sorted(required - present)}"


@pytest.mark.parametrize(
    "filename,token",
    [
        ("COST_OPTIONS.h", "#define ALLOW_COST_TEST"),
        ("CTRL_OPTIONS.h", "#define ALLOW_ETAN0_CONTROL"),
        ("AUTODIFF_OPTIONS.h", "#define ALLOW_AUTODIFF_TAMC"),
        ("AUTODIFF_OPTIONS.h", "#define ALLOW_TAMC_CHECKPOINTING"),
        # ADJetan dumps at adjDumpFreq come from pkg/autodiff/addummy_for_etan.F,
        # which is guarded by ALLOW_AUTODIFF_MONITOR
        ("AUTODIFF_OPTIONS.h", "#define ALLOW_AUTODIFF_MONITOR"),
        # useSingleCpuIO is .TRUE., so tape I/O must go through WHTAPEIO
        ("AUTODIFF_OPTIONS.h", "#define ALLOW_AUTODIFF_WHTAPEIO"),
    ],
)
def test_required_cpp_option_is_defined(filename: str, token: str) -> None:
    assert token in (CODE_AD / filename).read_text()


def test_divided_adjoint_stays_off() -> None:
    text = (CODE_AD / "AUTODIFF_OPTIONS.h").read_text()
    assert "#undef ALLOW_DIVIDED_ADJOINT" in text
    # the 3-level checkpoint form is what tamc.h is sized for
    assert "#define AUTODIFF_2_LEVEL_CHECKPOINT" not in text


def test_checkpoint_levels_cover_the_planned_horizon() -> None:
    """AUTODIFF_CHECK enforces nchklev_1*nchklev_2*nchklev_3 >= nTimeSteps."""

    text = (CODE_AD / "tamc.h").read_text()
    levels = {}
    # Only the ALLOW_TAMC_CHECKPOINTING branch is live; the #else branch
    # declares its own nchklev_1 and would otherwise shadow the real value.
    active = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#ifdef ALLOW_TAMC_CHECKPOINTING"):
            active = True
            continue
        if active and stripped.startswith(("#else", "#endif")):
            active = False
            continue
        if active and stripped.startswith("parameter( nchklev_"):
            name, _, value = stripped[len("parameter( ") :].partition("=")
            levels[name.strip()] = int(value.strip().rstrip(")").strip())

    assert set(levels) == {"nchklev_1", "nchklev_2", "nchklev_3"}, levels

    product = levels["nchklev_1"] * levels["nchklev_2"] * levels["nchklev_3"]
    assert levels["nchklev_1"] == STEPS_PER_DAY, "level 1 should be one model day"
    assert product >= 1440, "must cover the 20-day Run B"
    assert product == STEPS_PER_DAY * 200, "sized for 200 days per the plan"


def test_packages_conf_carries_the_ad_stack() -> None:
    text = (CODE_AD / "packages.conf").read_text()
    entries = {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}
    assert {"gfd", "diagnostics", "autodiff", "cost", "ctrl", "grdchk"} <= entries


def test_size_h_matches_the_forward_build() -> None:
    """The adjoint must run the same decomposition as the forward trajectory."""

    forward = (PROJECT_ROOT / "af_fno" / "mitgcm" / "code" / "SIZE.h").read_text()
    assert (CODE_AD / "SIZE.h").read_text() == forward


# --- 7. the runtime namelists ----------------------------------------------


def test_cost_namelist_wires_the_weight_field() -> None:
    text = (INPUT_AD / "data.cost").read_text()
    assert "mult_test        = 1." in text, "fc must equal J with no rescaling"
    assert "afCostWeightFile" in text


def test_cost_test_reads_the_weight_passively() -> None:
    """READ_REC_XY_RL, not ACTIVE_READ_XY: w is a constant, not a control."""

    text = (CODE_AD / "cost_test.F").read_text()
    # fixed-form Fortran: a comment is C/c/*/! in column 1
    code = "\n".join(
        line for line in text.splitlines() if not line[:1] in ("C", "c", "*", "!")
    )
    assert "READ_REC_XY_RL" in code
    assert "ACTIVE_READ_XY" not in code, "w is a constant, not a control"
    assert "objf_test(bi,bj)" in code
    # MDSREADFIELD is retired in c68j: it STOPs at runtime unless
    # USE_OBSOLETE_MDS_RW_FIELD is defined
    assert "MDSREADFIELD" not in code


def test_grdchk_targets_the_etan_control() -> None:
    text = (INPUT_AD / "data.grdchk").read_text()
    assert "grdchkvarindex = 29" in text, "29 is xx_etan (ctrl_init.F, grdchk_getxx.F:602)"


def test_grdchk_position_is_tile_local_and_in_range() -> None:
    """grdchk_readparms.F:116 rejects iGloPos > sNx."""

    size_text = (CODE_AD / "SIZE.h").read_text()
    snx = int(size_text.split("sNx =", 1)[1].split(",", 1)[0])
    sny = int(size_text.split("sNy =", 1)[1].split(",", 1)[0])

    grdchk = (INPUT_AD / "data.grdchk").read_text()
    values = {}
    for line in grdchk.splitlines():
        stripped = line.strip()
        if stripped.startswith(("iGloPos", "jGloPos", "iGloTile", "jGloTile")):
            name, _, value = stripped.partition("=")
            values[name.strip()] = int(value.strip().rstrip(","))

    assert 1 <= values["iGloPos"] <= snx
    assert 1 <= values["jGloPos"] <= sny
    # tile (1,1) covers global i,j = 1..31, so local == global there
    assert values["iGloTile"] == 1 and values["jGloTile"] == 1


# --- 8. run staging --------------------------------------------------------


def test_staged_physics_is_identical_to_the_forward_renderer() -> None:
    """The adjoint must linearise about the trajectory the FNO was trained on.

    Only PARM03 (nIter0 / nTimeSteps / pChkptFreq / adjDumpFreq) may differ from
    archive/src/bire_repro/af_s0.py::render_data, which rendered every S0
    forward segment.  Anything else would differentiate a different model.
    """

    import re

    sys.path.insert(0, str(PROJECT_ROOT / "archive" / "src"))
    forward_render = pytest.importorskip("bire_repro.af_s0").render_data
    from stage_adjoint_run import render_data as adjoint_render

    def blocks(text: str) -> dict[str, str]:
        found = {}
        for name in ("PARM01", "PARM02", "PARM04", "PARM05"):
            match = re.search(r"&" + name + r"\n(.*?)\n &", text, re.S)
            assert match is not None, f"{name} not found in rendered data"
            found[name] = match.group(1)
        return found

    assert blocks(forward_render(3_110_400, 1440)) == blocks(
        adjoint_render(3_110_400, 1440, 864000.0, None)
    )


@pytest.mark.parametrize(
    "mode,start_day,days,adjoint",
    [("pickup", 7200, 20, False), ("grdchk", 7210, 10, True), ("runA", 7210, 10, True), ("runB", 7200, 20, True)],
)
def test_staging_modes_target_the_right_window(mode: str, start_day: int, days: int, adjoint: bool) -> None:
    from stage_adjoint_run import MODES, iteration_of_day

    settings = MODES[mode]
    assert settings["start_day"] == start_day
    assert settings["days"] == days
    assert settings["adjoint"] is adjoint
    # every window must end at day 7220, where the cost is evaluated
    assert start_day + days == 7220
    assert iteration_of_day(start_day + days) == 3_111_840


def test_adjoint_runs_dump_daily_and_fit_the_checkpoint_budget() -> None:
    from stage_adjoint_run import MODES

    text = (CODE_AD / "tamc.h").read_text()
    active, levels = False, {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#ifdef ALLOW_TAMC_CHECKPOINTING"):
            active = True
            continue
        if active and stripped.startswith(("#else", "#endif")):
            active = False
            continue
        if active and stripped.startswith("parameter( nchklev_"):
            name, _, value = stripped[len("parameter( ") :].partition("=")
            levels[name.strip()] = int(value.strip().rstrip(")").strip())
    budget = levels["nchklev_1"] * levels["nchklev_2"] * levels["nchklev_3"]

    for mode, settings in MODES.items():
        steps = settings["days"] * STEPS_PER_DAY
        if settings["adjoint"]:
            assert steps <= budget, f"{mode} needs {steps} steps, tamc.h allows {budget}"
        if mode in ("runA", "runB"):
            assert settings["adj_dump_freq"] == 86400.0, "one ADJetan map per model day"


def test_pickup_mode_writes_the_pickups_the_adjoint_runs_need() -> None:
    """Run A starts at day 7210, so pChkptFreq must land a pickup there."""

    from stage_adjoint_run import MODES, iteration_of_day, render_data

    settings = MODES["pickup"]
    assert settings["adjoint"] is False, "the pickup run uses the forward executable"
    freq = settings["pchkpt_freq"]
    delta_t = 1200.0
    for day in (7210, 7220):
        assert (iteration_of_day(day) * delta_t) % freq == 0.0, f"no pickup would land on day {day}"

    text = render_data(iteration_of_day(7200), 1440, freq, None)
    assert "adjDumpFreq" not in text, "the forward executable has no adjoint to dump"


def test_grdchk_mode_starts_where_run_a_starts() -> None:
    from stage_adjoint_run import MODES

    assert MODES["grdchk"]["start_day"] == MODES["runA"]["start_day"]
    assert MODES["grdchk"]["days"] == MODES["runA"]["days"]
    assert MODES["grdchk"]["grdchk"] is True


def test_grdchk_point_is_the_frozen_target(contract: dict) -> None:
    grdchk = (INPUT_AD / "data.grdchk").read_text()
    values = {}
    for line in grdchk.splitlines():
        stripped = line.strip()
        if stripped.startswith(("iGloPos", "jGloPos")):
            name, _, value = stripped.partition("=")
            values[name.strip()] = int(value.strip().rstrip(","))
    assert values["iGloPos"] == contract["i_global"]
    assert values["jGloPos"] == contract["j_global"]
