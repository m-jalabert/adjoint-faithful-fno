from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bire_repro import af_model_c_bire_s0_figures as figures
from bire_repro.af_data_v3 import EXPERIMENTS, TEST_START_WINDOW
from bire_repro.af_model_c_bire_aligned_v3_figures import (
    COMPARATOR_STEP,
    MEMBER_COUNT,
    REGIME_WIND_LABEL,
    RegimeLabels,
    load_contract,
    declared_test_starts,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config/model_c_bire_aligned_v3_figures_v1.json"

pytestmark = pytest.mark.skipif(
    not CONTRACT.is_file(), reason="the figure contract is written after training"
)

#: The exact caption strings the frozen plotters embed.
FROZEN_CAPTIONS = (
    r"Control wind $\tau_0=0.1$ N m$^{-2}$; Model C $\Delta t=10$ days; native $1^\circ$ grid",
    r"$\tau_0=0.1$ N m$^{-2}$; $\Delta t=10$ days; 15 inference initial conditions",
    r"One S0 inference member; $\tau_0=0.1$ N m$^{-2}$; $\Delta t=10$ days",
    r"S0 architecture-direction comparison; $\Delta t=10$ days",
    r"Control wind $\tau_0=0.1$ N m$^{-2}$; $\Delta t=10$ days",
)


def test_starts_lie_in_the_declared_window_and_admit_2000_days() -> None:
    starts = declared_test_starts()
    assert starts.size == MEMBER_COUNT == 15
    assert starts.min() >= TEST_START_WINDOW[0]
    assert starts.max() < TEST_START_WINDOW[1]
    assert starts.max() + 2000 <= 8999
    assert np.array_equal(starts, np.sort(starts))
    assert np.unique(starts).size == starts.size
    assert np.array_equal(starts, declared_test_starts())


@pytest.mark.parametrize(
    ("regime", "tau"), [("S0", "0.1"), ("S1", "0.075"), ("S2", "0.125")]
)
def test_every_frozen_caption_is_rewritten_for_its_regime(regime, tau) -> None:
    """S1 and S2 packages must never carry S0's control wind."""

    labels = RegimeLabels(regime, float(tau), 7680)
    for caption in FROZEN_CAPTIONS:
        out = labels.rewrite(caption)
        if regime != "S0":
            assert r"$\tau_0=0.1$" not in out
        assert "One S0 inference member" not in out
        assert "S0 architecture-direction comparison" not in out
        if r"\tau_0" in out:
            assert rf"$\tau_0={tau}$" in out
    assert f"One {regime} test member" in labels.rewrite(FROZEN_CAPTIONS[2])
    assert f"{regime} training-progress comparison" in labels.rewrite(FROZEN_CAPTIONS[3])
    assert REGIME_WIND_LABEL[regime] in labels.rewrite(FROZEN_CAPTIONS[0])


def test_legend_labels_name_this_runs_checkpoints() -> None:
    labels = RegimeLabels("S1", 0.075, 7680)
    assert labels.rewrite("Prior residual Model C") == f"Step {COMPARATOR_STEP:,} checkpoint"
    assert (
        labels.rewrite("Selected anomaly-direct Model C")
        == "Selected step 7,680 checkpoint"
    )
    assert labels.rewrite(None) is None


def test_binding_patches_and_restores_matplotlib_and_method_labels() -> None:
    import matplotlib.axes
    import matplotlib.figure

    before = (
        matplotlib.axes.Axes.set_title,
        matplotlib.figure.Figure.suptitle,
        matplotlib.axes.Axes.plot,
        figures.METHOD_LABELS,
    )
    with RegimeLabels("S2", 0.125, 7680):
        assert matplotlib.axes.Axes.set_title is not before[0]
        assert figures.METHOD_LABELS["model"] == "Pooled v3 Model C"
        assert figures.METHOD_LABELS["persistence"] == "Persistence"
    assert matplotlib.axes.Axes.set_title is before[0]
    assert matplotlib.figure.Figure.suptitle is before[1]
    assert matplotlib.axes.Axes.plot is before[2]
    assert figures.METHOD_LABELS is before[3]


def test_rendered_titles_carry_the_right_wind() -> None:
    """End to end through matplotlib, not just the substitution table."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for regime, tau in (("S1", "0.075"), ("S2", "0.125")):
        with RegimeLabels(regime, float(tau), 7680):
            figure, axis = plt.subplots()
            axis.set_title(FROZEN_CAPTIONS[1])
            figure.suptitle(FROZEN_CAPTIONS[0])
            axis.plot([0, 1], [0, 1], label="Prior residual Model C")
            title = axis.get_title()
            suptitle = figure._suptitle.get_text()
            legend = axis.get_legend_handles_labels()[1]
            plt.close(figure)
        assert rf"$\tau_0={tau}$" in title and r"$\tau_0=0.1$" not in title
        assert rf"$\tau_0={tau}$" in suptitle and r"$\tau_0=0.1$" not in suptitle
        assert REGIME_WIND_LABEL[regime] in suptitle
        assert legend == [f"Step {COMPARATOR_STEP:,} checkpoint"]


def test_contract_declares_all_three_regimes_with_their_own_wind() -> None:
    raw = json.loads(CONTRACT.read_text())
    assert tuple(raw["protocol"]["regimes"]) == EXPERIMENTS
    assert raw["protocol"]["primary_regime"] == "S0"
    assert raw["dataset"]["tau0_n_m2"] == {"S0": 0.1, "S1": 0.075, "S2": 0.125}
    assert raw["comparability"]["byte_comparable_with_v2_packages"] is False
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    assert contract["figure6"]["comparator_optimizer_step"] == COMPARATOR_STEP
