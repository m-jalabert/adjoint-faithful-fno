"""Strictly chronological train/validation/test split and its train-only statistics.

The trajectory-v2 store carries a split whose training block is *interleaved*
with the later data: snapshot code 1 covers 0--2519 and 3690--6209, so the
training set surrounds the 6300--6569 validation block in time.  Every result so
far was obtained under that layout, which supports an interpolation claim rather
than a prospective one.

This module declares the clean alternative::

    code  split        indices      days
    1     train        0--5039      5040
    0     buffer       5040--5129     90
    2     validation   5130--5759    630
    0     buffer       5760--5849     90
    3     test         5850--7199   1350

so that training strictly precedes validation, which strictly precedes test,
with the project's 90-day buffers at both boundaries.

This is deliberately **not** a pure reordering.  Both training sets contain 5,040
days but only 3,870 overlap: the new split trades 5040--6209 for 2520--3689, so
1,170 snapshots --- 23.2% of the training set --- change identity.  Any arm run
under it therefore tests two things at once, the chronological protocol and
sensitivity to which stretch of trajectory is used for training, and must not be
reported as a split-order ablation alone.

Because the training interval moves, every train-derived statistic must be
recomputed.  Reusing the incumbent normalizer would import information from
5040--6209, which is validation or test here.  :func:`train_only_normalizers`
recomputes the pointwise mean, the pointwise scale, and the per-channel
fifth-percentile wet-cell floor from indices 0--5039 only.

Two quantities do *not* change, and the reasons are worth recording rather than
recomputing silently:

* wind-stress normalization.  ``static_features`` has shape ``(3, 5, 62, 62)``
  with no time axis, so the wind statistics are a property of the three forcing
  regimes and cannot depend on a chronological split.
* the day-2000 S0 test starts.  They lie in 6660--7199, which is test under both
  the stored split and this one, so the two protocols remain directly comparable
  on exactly that block.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .af_data import DatasetSpec
from .af_model_c_anomaly_direct import training_pointwise_normalizers

VERSION = "model_c_chronological_split_v1"

TRAIN_RANGE = (0, 5040)
VALIDATION_RANGE = (5130, 5760)
TEST_RANGE = (5850, 7200)
BUFFER_RANGES = ((5040, 5130), (5760, 5850))
HORIZON_DAYS = 10
RECORD_COUNT = 7200

TRAIN_CODE, VALIDATION_CODE, TEST_CODE, BUFFER_CODE = 1, 2, 3, 0

#: Fixed 360-day validation rollout starts: a deterministic stride-9 sample of
#: the 270 daily starts satisfying ``t + 360 <= 5759``, giving 30 per regime.
VALIDATION_START_STRIDE = 9
VALIDATION_ROLLOUT_DAYS = 360


class ChronologicalSplitError(RuntimeError):
    """Raised when the declared chronological split is violated."""


def snapshot_codes(record_count: int = RECORD_COUNT) -> np.ndarray:
    """Per-snapshot split codes for the strictly chronological layout."""

    codes = np.zeros(int(record_count), dtype=np.uint8)
    for (start, stop), code in (
        (TRAIN_RANGE, TRAIN_CODE),
        (VALIDATION_RANGE, VALIDATION_CODE),
        (TEST_RANGE, TEST_CODE),
    ):
        codes[start:stop] = code
    return codes


def pair_codes(
    record_count: int = RECORD_COUNT,
    *,
    horizon_days: int = HORIZON_DAYS,
) -> np.ndarray:
    """Per-pair split codes: a pair belongs to a split only if both ends do."""

    codes = np.zeros(int(record_count), dtype=np.uint8)
    for (start, stop), code in (
        (TRAIN_RANGE, TRAIN_CODE),
        (VALIDATION_RANGE, VALIDATION_CODE),
        (TEST_RANGE, TEST_CODE),
    ):
        pair_stop = stop - int(horizon_days)
        if pair_stop > start:
            codes[start:pair_stop] = code
    return codes


def validation_starts() -> np.ndarray:
    """Declared 360-day validation rollout starts, one set shared by all regimes."""

    latest = VALIDATION_RANGE[1] - 1 - VALIDATION_ROLLOUT_DAYS
    starts = np.arange(
        VALIDATION_RANGE[0],
        latest + 1,
        VALIDATION_START_STRIDE,
        dtype=np.int64,
    )
    if starts.size == 0 or int(starts[-1]) + VALIDATION_ROLLOUT_DAYS > latest + VALIDATION_ROLLOUT_DAYS:
        raise ChronologicalSplitError("validation starts left the validation block")
    return starts


def validation_records(experiment_count: int = 3) -> np.ndarray:
    """``(experiment, start)`` pairs for the declared validation rollouts."""

    starts = validation_starts()
    return np.asarray(
        [
            (experiment, int(start))
            for experiment in range(int(experiment_count))
            for start in starts
        ],
        dtype=np.int64,
    )


def verify(record_count: int = RECORD_COUNT) -> dict[str, Any]:
    """Assert the split is ordered, buffered, disjoint, and leak-free."""

    snapshots = snapshot_codes(record_count)
    pairs = pair_codes(record_count)
    train = np.flatnonzero(snapshots == TRAIN_CODE)
    validation = np.flatnonzero(snapshots == VALIDATION_CODE)
    test = np.flatnonzero(snapshots == TEST_CODE)
    if not (train.max() < validation.min() and validation.max() < test.min()):
        raise ChronologicalSplitError("splits are not strictly ordered in time")
    for start, stop in BUFFER_RANGES:
        if np.any(snapshots[start:stop] != BUFFER_CODE):
            raise ChronologicalSplitError("a declared buffer is not empty")
    if validation.min() - train.max() - 1 != 90 or test.min() - validation.max() - 1 != 90:
        raise ChronologicalSplitError("buffers are not the project's 90 days")
    # A training *pair* must not reach across the buffer into validation.
    training_pairs = np.flatnonzero(pairs == TRAIN_CODE)
    if int(training_pairs.max()) + HORIZON_DAYS >= VALIDATION_RANGE[0]:
        raise ChronologicalSplitError("a training pair reaches into validation")
    validation_pairs = np.flatnonzero(pairs == VALIDATION_CODE)
    if int(validation_pairs.max()) + HORIZON_DAYS >= TEST_RANGE[0]:
        raise ChronologicalSplitError("a validation pair reaches into test")
    starts = validation_starts()
    if int(starts[-1]) + VALIDATION_ROLLOUT_DAYS > int(validation.max()):
        raise ChronologicalSplitError("a validation rollout leaves the validation block")
    stored = DatasetSpec()
    return {
        "version": VERSION,
        "train": [int(train.min()), int(train.max())],
        "validation": [int(validation.min()), int(validation.max())],
        "test": [int(test.min()), int(test.max())],
        "train_days": int(train.size),
        "validation_days": int(validation.size),
        "test_days": int(test.size),
        "buffer_days": int(np.count_nonzero(snapshots == BUFFER_CODE)),
        "validation_rollout_starts_per_regime": int(starts.size),
        "validation_rollout_days": VALIDATION_ROLLOUT_DAYS,
        "strictly_ordered": True,
        "expected_records": int(stored.expected_records),
    }


def training_overlap(stored_snapshot_codes: np.ndarray) -> dict[str, Any]:
    """Quantify how much of the training set actually changes.

    This is the number the arm must be reported against: the split is not a
    reordering, it exchanges 1,170 training snapshots for a different 1,170.
    """

    stored = np.flatnonzero(np.asarray(stored_snapshot_codes) == TRAIN_CODE)
    new = np.flatnonzero(snapshot_codes(len(stored_snapshot_codes)) == TRAIN_CODE)
    shared = np.intersect1d(stored, new)
    return {
        "stored_training_days": int(stored.size),
        "new_training_days": int(new.size),
        "shared_days": int(shared.size),
        "changed_days": int(new.size - shared.size),
        "changed_fraction": float((new.size - shared.size) / new.size),
        "dropped_from_training": "5040-6209",
        "added_to_training": "2520-3689",
        "note": "not a pure split-order ablation; the training period itself moves",
    }


def train_only_normalizers(
    group: Any,
    *,
    floor_quantile: float = 0.05,
    absolute_floor: float = 1.0e-6,
) -> Mapping[str, Any]:
    """Pointwise mean/scale and channel floors from indices 0--5039 only.

    Delegates to the unchanged
    :func:`~bire_repro.af_model_c_anomaly_direct.training_pointwise_normalizers`
    with this split's snapshot codes, so the construction is identical to the
    incumbent's and only the interval differs.
    """

    return training_pointwise_normalizers(
        group,
        snapshot_codes(),
        split_code=TRAIN_CODE,
        floor_quantile=float(floor_quantile),
        absolute_floor=float(absolute_floor),
    )
