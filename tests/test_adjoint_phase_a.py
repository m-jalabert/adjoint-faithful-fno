"""Tests for the Phase A adjoint study.

Runnable without MITgcm, without TAF, without a GPU and without the trajectory
store.  These guard the parts of docs/Adjoint_study_Phase_A.md that can be
wrong silently:

* the contract's own arithmetic --- every run's window must close on its
  declared cost day, or the two sides are differentiating different things;
* the smooth kernel --- its stencil must be wet, normalized, and centred on the
  frozen target, because a kernel that quietly renormalizes over land measures
  a different place than the one p* was chosen for;
* the checkpoint budget --- ninety days must fit the existing tape, because if
  it does not the study needs a TAF submission and the plan is wrong;
* the precision defect --- neuralop's spectral buffer is hard-coded to
  complex64, and the promotion that fixes it has to stay effective.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from make_cost_weight import (  # noqa: E402
    DEFAULT_KERNEL_RADIUS,
    DEFAULT_KERNEL_SIGMA,
    CostWeightError,
    build_kernel,
    build_weight,
)
from select_adjoint_target import BASE_ITERATION, STEPS_PER_DAY  # noqa: E402
from stage_adjoint_run import WEIGHT_FILES, phase_a_modes  # noqa: E402

PLAN_CONTRACT = "adjoint_phase_a_v1"
CONTRACT_PATH = PROJECT_ROOT / "config" / f"{PLAN_CONTRACT}.json"
CODE_AD = PROJECT_ROOT / "af_fno" / "mitgcm" / "code_ad"


@pytest.fixture(scope="module")
def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


@pytest.fixture(scope="module")
def basin() -> tuple[np.ndarray, tuple[int, int]]:
    """The real wet mask, recovered from the mean-only weight field on disk.

    Read rather than reconstructed: the whole study turns on both sides using
    the same basin, and a locally rebuilt mask would defeat the point.
    """

    path = PROJECT_ROOT / "work" / "costWeight_mean_only.bin"
    if not path.is_file():
        pytest.skip("costWeight_mean_only.bin not built yet")
    field = np.fromfile(path, dtype=">f4").reshape(62, 62)
    target = json.loads(CONTRACT_PATH.read_text())["target"]
    return field != 0.0, (int(target["j_index0"]), int(target["i_index0"]))


# --- 1. the contract's own arithmetic --------------------------------------


def test_every_run_closes_on_its_declared_cost_day(contract: dict) -> None:
    """nIter0 + nTimeSteps must land exactly on the cost day.

    An off-by-one here means MITgcm evaluates J on a different day than the
    emulator does, and every comparison silently becomes a comparison of two
    different functionals.
    """

    for run in contract["mitgcm_runs"]:
        expected = BASE_ITERATION + STEPS_PER_DAY * int(run["cost_day"])
        assert int(run["n_iter0"]) + int(run["n_time_steps"]) == expected, run["name"]


def test_the_primary_sweep_holds_the_source_day_fixed(contract: dict) -> None:
    """Sweep (B): only the cost day moves.  V10 and G90 are the two exceptions."""

    source = BASE_ITERATION + STEPS_PER_DAY * int(contract["window"]["source_day"])
    for run in contract["mitgcm_runs"]:
        if run["name"] == "V10":
            continue  # gate G2b deliberately starts ten days before the cost
        assert int(run["n_iter0"]) == source, run["name"]


def test_v10_starts_ten_days_before_the_ninety_day_cost(contract: dict) -> None:
    """Gate G2b compares a fresh ten-day run against a ninety-day tape."""

    v10 = next(r for r in contract["mitgcm_runs"] if r["name"] == "V10")
    p90 = next(r for r in contract["mitgcm_runs"] if r["name"] == "P90")
    assert v10["cost_day"] == p90["cost_day"]
    assert int(v10["n_iter0"]) == BASE_ITERATION + STEPS_PER_DAY * 7280
    assert int(v10["n_time_steps"]) == 10 * STEPS_PER_DAY


def test_declared_leads_are_all_whole_emulator_calls(contract: dict) -> None:
    """The emulator advances in ten-day steps; a lead it cannot reach is not comparable."""

    for lead in contract["window"]["lead_days"]:
        assert int(lead) % 10 == 0


def test_forward_run_covers_the_window_and_one_tail_day(contract: dict) -> None:
    """data.diagnostics sets dumpAtLast=.FALSE., so day 7290 needs a tail step."""

    forward = contract["mitgcm_forward_runs"][0]
    assert int(forward["n_time_steps"]) == (int(forward["days"]) + int(forward["tail_days"])) * STEPS_PER_DAY
    assert 7280 in forward["pickups_at_days"], "V10 has no pickup without this"
    assert 7290 in forward["pickups_at_days"]


def test_every_objective_names_a_weight_file_the_stager_knows(contract: dict) -> None:
    for run in contract["mitgcm_runs"]:
        assert run["weight"] in WEIGHT_FILES, run["name"]


def test_weight_digests_match_the_files_on_disk(contract: dict) -> None:
    """Gate F6's premise.  Two sides weighting eta differently is the largest
    silent-failure risk in the study, so the contract pins the bytes."""

    import hashlib

    for name, declared in contract["objectives"].items():
        if not isinstance(declared, dict) or "weight_file" not in declared:
            continue
        path = PROJECT_ROOT / declared["weight_file"]
        if not path.is_file():
            pytest.skip(f"{path.name} not built yet")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == declared["weight_sha256"], name


# --- 2. the smooth kernel --------------------------------------------------


def test_kernel_is_normalized_and_centred_on_the_target(basin) -> None:
    wet, (j0, i0) = basin
    kernel, provenance = build_kernel(wet, j0, i0)
    assert kernel.sum() == pytest.approx(1.0, rel=1e-12)
    assert provenance["wet_cells_used"] == provenance["stencil_cells"] == 5
    assert provenance["cells_on_land"] == []
    assert provenance["cells_off_grid"] == []
    assert provenance["centroid_displacement_cells"] == pytest.approx(0.0, abs=1e-12)


def test_kernel_profile_matches_the_frozen_contract(basin, contract: dict) -> None:
    wet, (j0, i0) = basin
    kernel, _ = build_kernel(wet, j0, i0)
    declared = np.asarray(contract["objectives"]["ssh_anomaly_kernel"]["kernel"]["weights"])
    live = np.sort(kernel[kernel > 0.0])
    assert np.allclose(live, np.sort(declared), atol=1e-12)


def test_kernel_is_symmetric_about_the_target(basin) -> None:
    wet, (j0, i0) = basin
    kernel, _ = build_kernel(wet, j0, i0)
    for offset in (1, 2):
        assert kernel[j0 - offset, i0] == pytest.approx(kernel[j0 + offset, i0], rel=1e-14)


def test_isotropic_stencil_is_refused_because_it_would_leave_the_basin(basin) -> None:
    """p* is in the first wet column.  A symmetric 5x5 there loses ten of its
    twenty-five cells and renormalizing moves the effective target off the jet,
    so the default must refuse rather than quietly measure somewhere else."""

    wet, (j0, i0) = basin
    with pytest.raises(CostWeightError, match="leaves the basin"):
        build_kernel(wet, j0, i0, axis="isotropic")


def test_the_rejected_isotropic_stencil_displaces_the_centroid(basin) -> None:
    """The number the plan cites for rejecting it, reproduced rather than quoted."""

    wet, (j0, i0) = basin
    _, provenance = build_kernel(wet, j0, i0, axis="isotropic", allow_land=True)
    assert provenance["wet_cells_used"] == 15
    assert provenance["centroid_displacement_cells"] == pytest.approx(0.504, abs=0.002)


def test_kernel_weight_field_is_mean_free_and_zero_on_land(basin) -> None:
    wet, (j0, i0) = basin
    rng = np.random.default_rng(11)
    rac = rng.uniform(0.5, 1.5, size=wet.shape) * wet
    wet_area = float(rac.sum())
    weight, _ = build_weight("ssh_anomaly_kernel", wet, rac, wet_area, j0, i0)
    assert abs(float(weight[wet].sum())) < 1e-12
    assert np.all(weight[~wet] == 0.0)


def test_kernel_reduces_to_the_point_objective_as_sigma_vanishes(basin) -> None:
    """A sanity check on the parametrization: a very narrow Gaussian is a delta."""

    wet, (j0, i0) = basin
    kernel, _ = build_kernel(wet, j0, i0, sigma=0.05)
    assert kernel[j0, i0] == pytest.approx(1.0, abs=1e-12)


# --- 3. the checkpoint budget ----------------------------------------------


def test_ninety_days_fits_the_existing_tape() -> None:
    """If this fails the study needs a new TAF submission and the plan is wrong.

    The ground-truth plan sized nchklev_2 for 200 days rather than for its own
    20-day first experiment, explicitly so that a longer lead would cost disk
    and not a licence.  This is that decision being cashed in.
    """

    text = (CODE_AD / "tamc.h").read_text()
    levels = {}
    for name in ("nchklev_1", "nchklev_2", "nchklev_3"):
        for line in text.splitlines():
            if f"parameter( {name}" in line:
                levels[name] = int(line.split("=")[1].split(")")[0])
                break
    budget = levels["nchklev_1"] * levels["nchklev_2"] * levels["nchklev_3"]
    assert budget >= 90 * STEPS_PER_DAY, f"{budget} < {90 * STEPS_PER_DAY}"


def test_no_phase_a_run_exceeds_the_checkpoint_budget(contract: dict) -> None:
    text = (CODE_AD / "tamc.h").read_text()
    levels = [
        int(line.split("=")[1].split(")")[0])
        for name in ("nchklev_1", "nchklev_2", "nchklev_3")
        for line in text.splitlines()
        if f"parameter( {name}" in line
    ]
    budget = levels[0] * levels[1] * levels[2]
    for run in contract["mitgcm_runs"]:
        assert int(run["n_time_steps"]) <= budget, run["name"]


# --- 4. the staging table --------------------------------------------------


def test_staging_derives_every_declared_run(contract: dict) -> None:
    modes = phase_a_modes(PROJECT_ROOT)
    for run in contract["mitgcm_runs"]:
        assert run["name"] in modes
        settings = modes[run["name"]]
        assert settings["start_day"] * STEPS_PER_DAY + BASE_ITERATION == int(run["n_iter0"])
        assert settings["days"] * STEPS_PER_DAY == int(run["n_time_steps"])
        assert settings["weight"] == run["weight"]


def test_only_g90_runs_the_gradient_check() -> None:
    modes = phase_a_modes(PROJECT_ROOT)
    checking = {name for name, s in modes.items() if s.get("grdchk")}
    assert checking == {"G90"}


def test_every_adjoint_run_dumps_daily() -> None:
    """91 ADJetan maps from the ninety-day run is the whole companion sweep,
    for the price of the run that was happening anyway."""

    modes = phase_a_modes(PROJECT_ROOT)
    for name, settings in modes.items():
        if settings["adjoint"]:
            assert settings["adj_dump_freq"] == 86400.0, name


def test_phase_a_does_not_write_into_the_validated_v1_products() -> None:
    from stage_adjoint_run import DEFAULT_SCRATCH, DEFAULT_SCRATCH_PHASE_A

    assert DEFAULT_SCRATCH_PHASE_A != DEFAULT_SCRATCH


def test_v1_modes_are_unchanged() -> None:
    """The v1 study stays reproducible from the same script."""

    from stage_adjoint_run import MODES

    assert sorted(MODES) == ["grdchk", "pickup", "runA", "runB", "runG3"]
    assert MODES["runA"]["start_day"] == 7210 and MODES["runA"]["days"] == 10
    assert MODES["runB"]["start_day"] == 7200 and MODES["runB"]["days"] == 20


# --- 5. the emulator side --------------------------------------------------


def test_lead_days_map_to_whole_numbers_of_calls() -> None:
    pytest.importorskip("torch")
    from fno_adjoint_ft90 import LEAD_DAYS, calls_for_lead

    assert [calls_for_lead(lead) for lead in LEAD_DAYS] == [1, 2, 3, 9]


def test_eta_is_the_last_state_channel() -> None:
    pytest.importorskip("torch")
    from fno_adjoint_ft90 import ETA_CHANNEL, STATE_CHANNEL_COUNT

    assert ETA_CHANNEL == STATE_CHANNEL_COUNT - 1 == 45


def test_the_emulator_is_markov_with_fifty_one_inputs() -> None:
    """The predecessor was two-input with 97 channels, and its present-slot
    derivative was a partial one with no MITgcm counterpart.  This asserts the
    property that makes Phase A's comparison well-posed."""

    pytest.importorskip("torch")
    from fno_adjoint_ft90 import EXTERNAL_INPUT_CHANNELS, STATE_CHANNEL_COUNT

    assert EXTERNAL_INPUT_CHANNELS == 51
    assert STATE_CHANNEL_COUNT == 46


def test_spectral_buffer_is_single_precision_without_the_promotion() -> None:
    """The upstream defect this study had to find, pinned so it cannot silently
    change under a neuralop upgrade.

    neuralop 2.0.0's SpectralConv.forward hard-codes ``out_dtype =
    torch.cfloat`` when fno_block_precision is 'full', so the Fourier working
    buffer is complex64 whatever dtype the model is in.  97.95 % of this
    operator's parameters live in that path, so casting to double without the
    promotion leaves essentially all of it in single precision.
    """

    torch = pytest.importorskip("torch")
    pytest.importorskip("neuralop")
    from neuralop.layers.spectral_convolution import SpectralConv

    layer = SpectralConv(in_channels=4, out_channels=4, n_modes=(8, 8))
    layer._apply(lambda t: t.to(torch.complex128) if t.is_complex() else (t.to(torch.float64) if t.is_floating_point() else t))
    observed = []
    original = torch.zeros

    def spy(*args, **kwargs):
        tensor = original(*args, **kwargs)
        if tensor.is_complex():
            observed.append(tensor.dtype)
        return tensor

    torch.zeros = spy
    try:
        with torch.no_grad():
            layer(torch.zeros(1, 4, 16, 16, dtype=torch.float64))
    finally:
        torch.zeros = original

    assert observed, "no complex working buffer was allocated"
    assert torch.complex64 in observed, (
        "neuralop no longer truncates the spectral buffer to single precision -- "
        "re-check whether DoublePrecisionSpectralConv is still needed"
    )


def test_the_promotion_restores_the_adjoint_identity() -> None:
    """<v, J u> = <J^T v, u> to float64 round-off, on a layer that is linear.

    Before the promotion this residual is ~7e-07, which is single precision and
    far too large to call the result a float64 adjoint.  Exercised on a small
    layer so the test runs in a second.
    """

    torch = pytest.importorskip("torch")
    pytest.importorskip("neuralop")
    from neuralop.layers.spectral_convolution import SpectralConv

    from fno_adjoint_ft90 import DoublePrecisionSpectralConv

    def residual(promote: bool) -> float:
        torch.manual_seed(20260819)
        layer = SpectralConv(in_channels=6, out_channels=6, n_modes=(8, 8))
        layer._apply(
            lambda t: t.to(torch.complex128)
            if t.is_complex()
            else (t.to(torch.float64) if t.is_floating_point() else t)
        )
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
        if promote:
            layer.__class__ = DoublePrecisionSpectralConv
        generator = torch.Generator().manual_seed(5)
        x = torch.randn((1, 6, 16, 16), generator=generator, dtype=torch.float64)
        u = torch.randn((1, 6, 16, 16), generator=generator, dtype=torch.float64)
        with torch.no_grad():
            exact = layer(u) - layer(torch.zeros_like(u))  # J u: the layer is linear
        v = torch.randn(tuple(exact.shape), generator=generator, dtype=torch.float64)
        leaf = x.clone().requires_grad_(True)
        (transpose,) = torch.autograd.grad((v * layer(leaf)).sum(), leaf)
        left = float((v * exact).sum())
        right = float((transpose * u).sum())
        return abs(left - right) / max(abs(left), 1e-300)

    before = residual(promote=False)
    after = residual(promote=True)
    assert before > 1e-9, f"expected a single-precision residual, got {before:.3e}"
    assert after < 1e-12, f"the promotion did not restore float64: {after:.3e}"
    assert after < before / 1000.0
