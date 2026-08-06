from __future__ import annotations

import numpy as np
import pytest

from bire_repro.af_model_b import records_for_rollout_split
from bire_repro.af_model_c_bire_s0_long_truth import EXPECTED_STARTS
from bire_repro.af_model_c_chronological_split import (
    BUFFER_RANGES,
    HORIZON_DAYS,
    TEST_RANGE,
    TRAIN_RANGE,
    VALIDATION_RANGE,
    VALIDATION_ROLLOUT_DAYS,
    ChronologicalSplitError,
    pair_codes,
    snapshot_codes,
    training_overlap,
    validation_records,
    validation_starts,
    verify,
)


def test_declared_boundaries_match_the_requested_layout() -> None:
    assert TRAIN_RANGE == (0, 5040)
    assert VALIDATION_RANGE == (5130, 5760)
    assert TEST_RANGE == (5850, 7200)
    assert BUFFER_RANGES == ((5040, 5130), (5760, 5850))
    summary = verify()
    assert summary["train"] == [0, 5039] and summary["train_days"] == 5040
    assert summary["validation"] == [5130, 5759] and summary["validation_days"] == 630
    assert summary["test"] == [5850, 7199] and summary["test_days"] == 1350
    assert summary["buffer_days"] == 180
    assert summary["strictly_ordered"] is True


def test_splits_are_disjoint_ordered_and_cover_the_record() -> None:
    codes = snapshot_codes()
    assert codes.size == 7200
    train = np.flatnonzero(codes == 1)
    validation = np.flatnonzero(codes == 2)
    test = np.flatnonzero(codes == 3)
    assert train.max() < validation.min() < validation.max() < test.min()
    assert train.size + validation.size + test.size + 180 == 7200
    # 90-day buffers, exactly
    assert validation.min() - train.max() - 1 == 90
    assert test.min() - validation.max() - 1 == 90


def test_no_pair_or_rollout_crosses_a_split_boundary() -> None:
    pairs = pair_codes()
    for code, (start, stop) in ((1, TRAIN_RANGE), (2, VALIDATION_RANGE), (3, TEST_RANGE)):
        index = np.flatnonzero(pairs == code)
        assert index.min() == start
        assert index.max() + HORIZON_DAYS == stop - 1
    training = records_for_rollout_split(pairs, 1, rollout_steps=3)
    assert len(training) == 3 * 5010
    latest = max(t for _, t in training)
    # a three-step rollout from the latest start must stay inside training
    assert latest + 3 * HORIZON_DAYS <= TRAIN_RANGE[1] - 1


def test_validation_rollouts_stay_inside_the_validation_block() -> None:
    starts = validation_starts()
    assert starts.size == 30
    assert starts.min() >= VALIDATION_RANGE[0]
    assert starts.max() + VALIDATION_ROLLOUT_DAYS <= VALIDATION_RANGE[1] - 1
    records = validation_records()
    assert records.shape == (90, 2)
    assert set(records[:, 0].tolist()) == {0, 1, 2}


def test_the_split_is_not_a_pure_reordering() -> None:
    """Both training sets hold 5,040 days but only 3,870 overlap."""

    stored = np.zeros(7200, dtype=np.uint8)
    stored[0:2520] = 1
    stored[3690:6210] = 1
    overlap = training_overlap(stored)
    assert overlap["stored_training_days"] == 5040
    assert overlap["new_training_days"] == 5040
    assert overlap["shared_days"] == 3870
    assert overlap["changed_days"] == 1170
    assert overlap["changed_fraction"] == pytest.approx(1170 / 5040)


def test_the_fifteen_s0_test_starts_are_held_out_under_both_splits() -> None:
    """This is what makes the old and new runs comparable."""

    stored = np.zeros(7200, dtype=np.uint8)
    stored[0:2520] = 1
    stored[3690:6210] = 1
    stored[6300:6570] = 2
    stored[2970:3600] = 3
    stored[6660:7200] = 3
    new = snapshot_codes()
    starts = np.asarray(EXPECTED_STARTS)
    assert np.all(new[starts] == 3)
    assert np.all(stored[starts] == 3)
    # and every start is outside the new training and validation blocks
    assert np.all(new[starts] != 1) and np.all(new[starts] != 2)


def test_verify_rejects_a_short_record() -> None:
    with pytest.raises((ChronologicalSplitError, ValueError, IndexError)):
        verify(record_count=4000)
