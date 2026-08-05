"""Tests for the Bire et al. Section 3.2 protocol arm.

Two of these are regressions for descriptions that were wrong in a published
package: :func:`test_verify_does_not_claim_truth_is_missing` and
:func:`test_readme_describes_the_bire_split_not_the_chronological_one`.  Both
bugs were pure metadata --- no metric moved --- which is exactly why nothing
caught them until the package was read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bire_repro.af_bire_protocol_split import (
    INFERENCE_RANGE,
    INFERENCE_START_RANGE,
    MAXIMUM_INFERENCE_ROLLOUT_DAYS,
    RECORD_DAYS,
    STORE_DAYS,
    TRAIN_RANGE,
    VALIDATION_RANGE,
    BireProtocolSplitError,
    assert_model_visible,
    assert_truth_available,
    inference_starts,
    split_codes,
    store_codes,
    validation_starts,
    verify,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config/model_c_bire_protocol_pooled_v1.json"
FIGURE_CONTRACT = ROOT / "config/model_c_bire_protocol_s0_figures_v1.json"
SBATCH = ROOT / "slurm/models/c/af_model_c_bire_protocol.sbatch"
FIGURE_SBATCH = ROOT / "slurm/models/c/af_model_c_bire_protocol_figures.sbatch"


# --------------------------------------------------------------------------
# The arrangement itself
# --------------------------------------------------------------------------


def test_the_arrangement_is_6000_1200_1000_with_inference_nested() -> None:
    report = verify()
    assert report["train"] == [0, 5999]
    assert report["validation"] == [6000, 7199]
    assert report["inference"] == [6200, 7200]
    assert (report["train_days"], report["validation_days"]) == (6000, 1200)
    assert report["inference_days"] == 1000
    assert report["train_days"] + report["validation_days"] == RECORD_DAYS == 7200
    assert report["inference_nested_in_validation"] is True


def test_no_buffers_and_no_training_pair_reaches_validation() -> None:
    """The paper has no buffers; the rollout window is what prevents leakage."""

    report = verify()
    assert "none" in report["buffers"]
    _, pairs = split_codes()
    latest = int(np.flatnonzero(pairs == 1).max())
    assert latest + 10 < VALIDATION_RANGE[0]
    assert report["latest_training_rollout_start"] == 5969
    # A three-step rollout from 5969 lands its last target on 5999, the final
    # training index -- so nothing reaches validation and no buffer is needed.
    assert 5969 + 3 * 10 == VALIDATION_RANGE[0] - 1 == TRAIN_RANGE[1] - 1


def test_selection_starts_never_fall_inside_the_inference_set() -> None:
    starts = validation_starts()
    assert starts.max() < INFERENCE_RANGE[0]
    assert starts.min() >= VALIDATION_RANGE[0]
    assert starts.max() + 360 < VALIDATION_RANGE[1]
    assert np.intersect1d(starts, np.arange(*INFERENCE_RANGE)).size == 0


def test_store_codes_agree_on_the_record_and_zero_the_truth_tail() -> None:
    """The tail is not a split; it carries no code at all."""

    snapshots, pairs = split_codes()
    store_snapshots, store_pairs = store_codes()
    assert store_snapshots.size == store_pairs.size == STORE_DAYS
    assert np.array_equal(store_snapshots[:RECORD_DAYS], snapshots)
    assert np.array_equal(store_pairs[:RECORD_DAYS], pairs)
    assert not store_snapshots[RECORD_DAYS:].any()
    assert not store_pairs[RECORD_DAYS:].any()


# --------------------------------------------------------------------------
# Model visibility versus evaluation truth --- the distinction that was
# mis-stated in both directions before it settled
# --------------------------------------------------------------------------


def test_nothing_the_model_sees_reaches_past_bires_record() -> None:
    snapshots, _ = split_codes()
    assert_model_visible(np.flatnonzero(snapshots), "split")
    assert_model_visible(validation_starts() + 360, "validation rollouts")
    assert_model_visible(inference_starts(15, 20260802), "inference starts")
    with pytest.raises(BireProtocolSplitError):
        assert_model_visible(np.array([RECORD_DAYS]), "past the record")


def test_every_member_has_lead_matched_truth_for_the_full_2000_days() -> None:
    starts = inference_starts(15, 20260802)
    assert starts.min() >= INFERENCE_START_RANGE[0] == INFERENCE_RANGE[0]
    assert starts.max() < INFERENCE_START_RANGE[1] == STORE_DAYS - 2000
    assert_truth_available(starts + MAXIMUM_INFERENCE_ROLLOUT_DAYS, "truth")
    assert int(starts.max()) + MAXIMUM_INFERENCE_ROLLOUT_DAYS < STORE_DAYS
    with pytest.raises(BireProtocolSplitError):
        assert_truth_available(np.array([STORE_DAYS]), "past the store")


def test_verify_does_not_claim_truth_is_missing() -> None:
    """Regression: the published report asserted no lead-matched truth existed.

    It does exist --- days 7200--8999 are held out as truth precisely so the
    day-2,000 ground-truth column of the paper's Figure 7 is reproducible --- so
    a description saying otherwise contradicts the code that draws the starts.
    """

    report = verify()
    reference = report["long_term_reference"].lower()
    assert "no lead-matched truth" not in reference
    assert "lead-matched" in reference and "truth" in reference
    assert report["evaluation_truth_only_days"] == [RECORD_DAYS, STORE_DAYS - 1]
    assert report["model_visible_days"] == [0, RECORD_DAYS - 1]
    # Nothing in the emitted description may deny the truth window.
    blob = json.dumps(report).lower()
    for denial in ("no lead-matched", "no truth", "truth is unavailable"):
        assert denial not in blob


def test_inference_starts_are_deterministic_and_ordered() -> None:
    first = inference_starts(15, 20260802)
    assert np.array_equal(first, inference_starts(15, 20260802))
    assert np.array_equal(first, np.sort(first))
    assert np.unique(first).size == first.size == 15


# --------------------------------------------------------------------------
# Package identity: the artifacts must be named and described after this arm
# --------------------------------------------------------------------------


def test_artifact_names_are_not_inherited_from_the_chronological_parent() -> None:
    from bire_repro import af_model_c_bire_aligned_chronological as parent
    from bire_repro import af_model_c_bire_protocol as arm

    for name in ("REPORT_NAME", "ARRAYS_NAME", "FIGURE_NAME"):
        mine, theirs = getattr(arm, name), getattr(parent, name)
        assert mine != theirs, f"{name} still names the chronological arm"
        assert "chronological" not in mine
        assert "bire_protocol" in mine or "bire_protocol" in mine.replace("_", "")


def test_readme_describes_the_bire_split_not_the_chronological_one() -> None:
    """Regression: the published figure README described the v3 arm's split."""

    from bire_repro.af_model_c_bire_protocol_figures import _readme

    text = _readme(
        "S0",
        {
            "selected_optimizer_step": 7680,
            "tau0_n_m2": 0.1,
            "report_content_sha256": "0" * 64,
        },
    )
    for stale in ("5039", "5130", "6389", "6480", "2,520-day", "held test block"):
        assert stale not in text, f"README still carries {stale!r}"
    for expected in ("0--5999", "6000--7199", "6200--6999", "6263--6979", "7200--8999", "no buffers"):
        assert expected in text.replace("There are no buffers", "no buffers")
    assert "Bire" in text


def test_figure_captions_keep_bires_word_for_the_inference_set() -> None:
    from bire_repro.af_model_c_bire_protocol_figures import RegimeLabels

    labels = RegimeLabels("S0", 0.1, 7680)
    assert labels.rewrite("One S0 inference member") == "One S0 inference member"
    assert (
        labels.rewrite("$\\tau_0=0.1$ N m$^{-2}$; $\\Delta t=10$ days; "
                       "15 inference initial conditions")
        == "$\\tau_0=0.1$ N m$^{-2}$; $\\Delta t=10$ days; 15 inference initial conditions"
    )
    assert (
        labels.rewrite("S0 architecture-direction comparison")
        == "S0 training-progress comparison"
    )


def test_figures_legend_names_this_arm() -> None:
    from bire_repro import af_model_c_bire_s0_figures as figures
    from bire_repro.af_model_c_bire_protocol_figures import RegimeLabels

    with RegimeLabels("S0", 0.1, 7680):
        assert "v3" not in figures.METHOD_LABELS["model"]
        assert figures.METHOD_LABELS["model"] == "Bire-protocol Model C"


# --------------------------------------------------------------------------
# Launchers --- a prior arm shipped an sbatch invoking its parent module
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("script", "module"),
    [
        (SBATCH, "bire_repro.af_model_c_bire_protocol"),
        (FIGURE_SBATCH, "bire_repro.af_model_c_bire_protocol_figures"),
    ],
)
def test_launcher_invokes_its_own_module(script: Path, module: str) -> None:
    if not script.is_file():
        pytest.skip(f"{script.name} is absent")
    text = script.read_text()
    invoked = {
        line.split("-m", 1)[1].strip().split()[0]
        for line in text.splitlines()
        if " -m " in f" {line} " and "bire_repro." in line
    }
    assert invoked == {module}, f"{script.name} invokes {invoked}"


@pytest.mark.skipif(not CONTRACT.is_file(), reason="contract written by training")
def test_contract_declares_the_bire_arrangement() -> None:
    dataset = json.loads(CONTRACT.read_text())["dataset"]
    assert dataset["train"] == list(TRAIN_RANGE)
    assert dataset["validation"] == list(VALIDATION_RANGE)
    assert dataset["inference"] == list(INFERENCE_RANGE)
    assert dataset["bire_record_days"] == RECORD_DAYS
    assert dataset["model_visible_days"] == [0, RECORD_DAYS - 1]
    assert dataset["evaluation_truth_only_days"] == [RECORD_DAYS, STORE_DAYS - 1]
    assert dataset["inference_nested_in_validation"] is True
    # The paper has no buffers, so no buffer width may survive from the
    # chronological arm alongside the empty buffer list.
    assert dataset["buffers"] == []
    assert "buffer_days" not in dataset
    assert "no lead-matched truth" not in json.dumps(dataset)
