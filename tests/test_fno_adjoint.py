"""Tests for the FNO adjoint generation and comparison.

Runnable without a GPU, without the 216 MB checkpoint and without MITgcm.
These guard the parts of docs/fno_adjoint_plan.md that can be wrong silently:
the frozen contract constants, the channel layout the whole study rests on, the
cost-weight files being *read* rather than rebuilt, and the comparison metrics'
behaviour on cases whose answers are known by hand.

The parts that need the checkpoint --- the gradient itself, gates F1 to F5 ---
are exercised by running the script, which prints every gate and writes them to
``report.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import adjoint_metrics as metrics  # noqa: E402
from make_cost_weight import MDS_DTYPE, build_weight  # noqa: E402
from select_adjoint_target import CONTRACT_VERSION  # noqa: E402

CONTRACT_PATH = PROJECT_ROOT / "config" / f"{CONTRACT_VERSION}.json"


@pytest.fixture(scope="module")
def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


@pytest.fixture(scope="module")
def basin() -> tuple[np.ndarray, tuple[int, int]]:
    """A 62x62 basin with the one-cell land rim, and a western target cell.

    Deliberately built here rather than read from the zarr, so the metric tests
    run anywhere; the real mask's shape is asserted separately against the
    frozen contract.
    """

    wet = np.zeros((62, 62), dtype=bool)
    wet[1:61, 1:61] = True
    return wet, (16, 1)


# ---------------------------------------------------------------------------
# the frozen contract  (plan section 1)
# ---------------------------------------------------------------------------


def test_window_and_channels_match_the_plan() -> None:
    """The constants the two sides must agree on, spelled out once more here.

    If someone edits ``fno_adjoint.py``'s window or channel indices, this fails
    rather than the study quietly answering a different question.
    """

    import fno_adjoint as adjoint

    assert adjoint.DAY_HISTORY_E1 == 7200
    assert adjoint.DAY_PRESENT_E1 == 7210
    assert adjoint.DAY_TARGET == 7220
    assert adjoint.DAY_TARGET - adjoint.DAY_PRESENT_E1 == adjoint.HORIZON_DAYS
    # E3 differentiates at day 7200 and reaches the same target in two calls
    assert adjoint.DAY_PRESENT_E3 == 7200
    assert adjoint.DAY_PRESENT_E3 + 2 * adjoint.HORIZON_DAYS == adjoint.DAY_TARGET
    # the pair is stacked history-then-present ahead of the five statics
    assert adjoint.HISTORY_ETA_CHANNEL == 45
    assert adjoint.PRESENT_ETA_CHANNEL == 91
    assert adjoint.EXTERNAL_INPUT_CHANNELS == 2 * 46 + 5
    assert adjoint.EXPECTED_PARAMETER_COUNT == 27_328_780
    assert adjoint.EXPECTED_OPTIMIZER_STEP == 3840
    assert adjoint.EXPECTED_CHECKPOINT_SHA256.startswith("bf3ccc70")


def test_channel_indices_agree_with_the_dataset_layout() -> None:
    """Channel 45 and 91 are eta because eta is last in the 46-channel state."""

    import fno_adjoint as adjoint
    from oceanfno.runtime import STATE_CHANNELS

    assert STATE_CHANNELS[-1] == "Eta"
    assert STATE_CHANNELS.index("Eta") == adjoint.ETA_CHANNEL
    assert len(STATE_CHANNELS) == adjoint.STATE_CHANNELS
    assert adjoint.PRESENT_ETA_CHANNEL == adjoint.STATE_CHANNELS + adjoint.ETA_CHANNEL


def test_lead_sweep_contains_the_twenty_day_experiment() -> None:
    """E3 is two calls, so the sweep must contain 2 or the internal check is lost."""

    import fno_adjoint as adjoint

    assert 2 in adjoint.DEFAULT_LEAD_CALLS
    assert adjoint.DEFAULT_LEAD_CALLS == (1, 2, 3, 6, 20)


def test_target_cell_is_the_frozen_one(contract: dict) -> None:
    assert (contract["j_index0"], contract["i_index0"]) == (16, 1)
    assert contract["grid"]["wet_cell_count"] == 3600
    assert contract["i_global"] == contract["i_index0"] + 1


# ---------------------------------------------------------------------------
# the weight field must be read, never rebuilt  (plan section 8, "Risks")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quantity", ["ssh_anomaly", "mean_only"])
def test_cost_weight_files_exist_and_are_land_free(quantity: str, contract: dict) -> None:
    """The bytes MITgcm reads are the bytes this study reads."""

    path = PROJECT_ROOT / "work" / f"costWeight_{quantity}.bin"
    if not path.is_file():
        pytest.skip(f"{path} not built; run scripts/make_cost_weight.py --qoi {quantity}")
    field = np.fromfile(path, dtype=MDS_DTYPE)
    assert field.size == 62 * 62
    field = field.reshape(62, 62).astype(np.float64)
    assert np.all(np.isfinite(field))
    if quantity == "ssh_anomaly":
        # a delta at p* on top of the area-weighted mean term, so the wet sum is zero
        assert field[contract["j_index0"], contract["i_index0"]] > 0.9
        assert abs(field.sum()) < 1.0e-4
    else:
        assert field.max() <= 0.0


def test_the_stored_weight_matches_its_own_builder(contract: dict) -> None:
    """A stale ``.bin`` would silently answer a different question.

    This verifies the file; the adjoint script never substitutes a locally built
    weight for the one on disk.
    """

    path = PROJECT_ROOT / "work" / "costWeight_ssh_anomaly.bin"
    rac_source = Path(contract["grid"]["rac_source"])
    if not path.is_file() or not rac_source.with_suffix(".data").is_file():
        pytest.skip("the weight field or the MITgcm grid dump is not available here")
    from select_adjoint_target import read_mds_2d

    wet = np.zeros((62, 62), dtype=bool)
    wet[1:61, 1:61] = True
    rac = read_mds_2d(rac_source)
    rebuilt = build_weight(
        "ssh_anomaly", wet, rac, float(contract["wet_area_m2"]),
        int(contract["j_index0"]), int(contract["i_index0"]),
    ).astype(MDS_DTYPE)
    stored = np.fromfile(path, dtype=MDS_DTYPE).reshape(62, 62)
    assert np.array_equal(stored, rebuilt)


# ---------------------------------------------------------------------------
# comparison metrics  (plan section 6)
# ---------------------------------------------------------------------------


def test_primary_metrics_on_an_identical_pair(basin) -> None:
    """A map compared with itself must score perfectly, by construction."""

    wet, _ = basin
    generator = np.random.default_rng(0)
    field = generator.normal(size=wet.shape) * wet
    result = metrics.primary_metrics(field, field, wet)
    assert result["pattern_correlation"] == pytest.approx(1.0)
    assert result["relative_l2"] == pytest.approx(0.0, abs=1e-15)
    assert result["amplitude_ratio"] == pytest.approx(1.0)
    assert result["sign_agreement"] == pytest.approx(1.0)


def test_amplitude_ratio_and_correlation_separate_scale_from_pattern(basin) -> None:
    """Doubling a map leaves the correlation at one and doubles the amplitude."""

    wet, _ = basin
    generator = np.random.default_rng(1)
    reference = generator.normal(size=wet.shape) * wet
    result = metrics.primary_metrics(2.0 * reference, reference, wet)
    assert result["pattern_correlation"] == pytest.approx(1.0)
    assert result["amplitude_ratio"] == pytest.approx(2.0)
    assert result["relative_l2"] == pytest.approx(1.0)
    assert result["sign_agreement"] == pytest.approx(1.0)

    flipped = metrics.primary_metrics(-reference, reference, wet)
    assert flipped["pattern_correlation"] == pytest.approx(-1.0)
    assert flipped["sign_agreement"] == pytest.approx(0.0)


def test_land_leakage_counts_the_dry_rim(basin) -> None:
    """MITgcm is exactly zero on the 244 dry cells; anything else is spurious."""

    wet, _ = basin
    field = np.zeros(wet.shape)
    field[wet] = 1.0
    clean = metrics.land_leakage(field, wet)
    assert clean["dry_cell_count"] == 244
    assert clean["max_abs_dry"] == 0.0

    field[0, 0] = 0.25
    leaky = metrics.land_leakage(field, wet)
    assert leaky["max_abs_dry"] == pytest.approx(0.25)
    assert leaky["dry_to_wet_max_ratio"] == pytest.approx(0.25)


def test_western_band_split_uses_the_project_convention(basin) -> None:
    """The band is the first four wet cells east of each row's western wall."""

    wet, _ = basin
    field = np.zeros(wet.shape)
    field[:, 1:5] = 1.0  # exactly the four-cell band
    field *= wet
    split = metrics.boundary_interior_split(field, wet, width=4)
    assert split["boundary_cell_count"] == 60 * 4
    assert split["interior_cell_count"] == 3600 - 240
    assert split["boundary_rms"] == pytest.approx(1.0)
    assert split["interior_rms"] == pytest.approx(0.0)


def test_radial_decay_recovers_a_known_e_folding_length(basin) -> None:
    """Feed the diagnostic an exact exponential and check it reads it back."""

    wet, target = basin
    rows, columns = np.indices(wet.shape)
    distance = np.sqrt((rows - target[0]) ** 2.0 + (columns - target[1]) ** 2.0)
    field = np.exp(-distance / 7.0) * wet
    decay = metrics.radial_decay(field, wet, target)
    assert decay["e_folding_cells"] == pytest.approx(7.0, rel=0.05)
    assert decay["fit_r_squared"] > 0.99


def test_radial_spectrum_is_absolute_power_in_twelve_bins(basin) -> None:
    """Twelve bins, and doubling the field quadruples every bin's power.

    ``local-branch-gamma-ablation`` warns that the high-wavenumber *fraction*
    misleads, so the diagnostic must report absolute power; this is what makes
    that true.
    """

    wet, _ = basin
    generator = np.random.default_rng(2)
    field = generator.normal(size=wet.shape) * wet
    single = metrics.radial_power_spectrum(field, wet)
    doubled = metrics.radial_power_spectrum(2.0 * field, wet)
    assert single["bins"] == 12
    assert len(single["power_per_bin"]) == 12
    assert np.allclose(
        np.asarray(doubled["power_per_bin"]), 4.0 * np.asarray(single["power_per_bin"])
    )
    assert metrics.spectrum_ratio(doubled, single) == pytest.approx([4.0] * 12)


def test_metrics_reject_mismatched_grids(basin) -> None:
    wet, _ = basin
    with pytest.raises(metrics.AdjointMetricError):
        metrics.pattern_correlation(np.zeros((62, 62)), np.zeros((10, 10)), wet)
    with pytest.raises(metrics.AdjointMetricError):
        metrics.relative_l2(np.zeros((62, 62)), np.zeros((62, 62)), wet)
    with pytest.raises(metrics.AdjointMetricError):
        metrics.pattern_correlation(np.full((62, 62), np.nan), np.zeros((62, 62)), wet)


# ---------------------------------------------------------------------------
# the comparison refuses to run on an unvalidated adjoint  (plan section 8)
# ---------------------------------------------------------------------------


def test_comparison_requires_mitgcm_gate_g1() -> None:
    """An unvalidated ``adxx_etan`` is not ground truth, so the script must refuse."""

    import compare_adjoint_maps as compare

    wet = np.zeros((62, 62), dtype=bool)
    wet[1:61, 1:61] = True
    arrays = {
        "target_ij": np.asarray([16, 1]),
        "wet_mask": wet.astype(np.uint8),
        "rA": np.ones(wet.shape),
    }
    fno_report = {"shared_contract": {"cost_weight_sha256": {"ssh_anomaly": "abc"}}}

    for gates in ({}, {"G1": {"passed": False}}):
        with pytest.raises(compare.AdjointComparisonError, match="G1"):
            compare.check_shared_contract(arrays, arrays, fno_report, {"gates": gates})

    passing = {"gates": {"G1": {"passed": True}}}
    assert compare.check_shared_contract(arrays, arrays, fno_report, passing)[
        "mitgcm_gate_g1_passed"
    ]


def test_comparison_refuses_a_different_weight_field() -> None:
    """If the two sides read different ``w``, the comparison is a convention test."""

    import compare_adjoint_maps as compare

    wet = np.zeros((62, 62), dtype=bool)
    wet[1:61, 1:61] = True
    arrays = {
        "target_ij": np.asarray([16, 1]),
        "wet_mask": wet.astype(np.uint8),
        "rA": np.ones(wet.shape),
    }
    fno_report = {"shared_contract": {"cost_weight_sha256": {"ssh_anomaly": "abc"}}}
    mitgcm_report = {
        "gates": {"G1": {"passed": True}},
        "shared_contract": {"cost_weight_sha256": {"ssh_anomaly": "def"}},
    }
    with pytest.raises(compare.AdjointComparisonError, match="different cost weight"):
        compare.check_shared_contract(arrays, arrays, fno_report, mitgcm_report)


def test_comparison_refuses_a_different_target_cell() -> None:
    import compare_adjoint_maps as compare

    wet = np.zeros((62, 62), dtype=bool)
    wet[1:61, 1:61] = True
    base = {"target_ij": np.asarray([16, 1]), "wet_mask": wet.astype(np.uint8), "rA": np.ones(wet.shape)}
    moved = dict(base, target_ij=np.asarray([16, 2]))
    with pytest.raises(compare.AdjointComparisonError, match="p\\*"):
        compare.check_shared_contract(base, moved, {"shared_contract": {"cost_weight_sha256": {}}}, {})
