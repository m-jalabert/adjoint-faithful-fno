"""Bire et al. (2025) Section 3.2 data arrangement, applied to trajectory-v3.

The paper's split, quoted: *"Overall, 7,200 snapshots from each simulation are
saved. Final 6000 timesteps from simulations 1, 3, and 5 are allocated for
training. The next 1200 timesteps are allocated for validation. The final 1000
timesteps are for inference (testing set)."*  Two paragraphs later: *"We do not
use a third held-out test data set."*

So 6000 + 1200 = 7,200 is their entire record, and the 1,000 "inference"
timesteps are the **last 1,000 of the validation block**, not a third partition.
This module reproduces that arrangement::

    split       indices      days   note
    train       0--5999      6000   Bire's "final 6000 timesteps"
    validation  6000--7199   1200   Bire's "next 1200"
    inference   6200--7199   1000   Bire's "final 1000", nested in validation

No buffers: the paper has none.  Leakage is prevented structurally instead ---
a training rollout qualifies only if its whole three-step target sequence stays
inside the training block, so the latest training start is 5969.

What the **model** may see is strictly Bire's 7,200 days: training 0--5999 and
validation 6000--7199, and nothing beyond index 7199 is ever trained on,
validated on, or used as an inference start.

Evaluation **truth** is a separate matter, and the paper is explicit that it has
some.  Figure 7's caption reads *"Streamfunction … on the 60th day (top row) and
2000th day for the ground truth (first column), and FNO ensemble members …"* ---
a ground-truth column at day 2,000.  Bire have five simulations; two (2 and 4)
are *entirely* dedicated to inference, so a long rollout started early in one of
those has 2,000 days of MITgcm truth after it while remaining wholly unseen by
the model.

We pool all three regimes for training, so the same effect is obtained from the
time axis instead: the trajectory-v3 store holds 9,000 days, and days
7200--8999 are used **only as evaluation truth** --- never trained on, never
validated on, never an inference start.  That is the same separation the paper
has, expressed along time rather than across simulations, and it is what makes
the day-2,000 ground-truth comparison possible for us as it was for them.

Inference starts are therefore drawn from 6200--6999, so every member completes
a 2,000-day rollout with lead-matched truth (6999 + 2000 = 8999).

"""

from __future__ import annotations

from typing import Any

import numpy as np

VERSION = "af_bire_protocol_split_v1"

#: Total days in the trajectory-v3 store; the tail serves only as evaluation truth.
STORE_DAYS = 9000
#: Bire's record length: the model may see nothing at or beyond this index.
RECORD_DAYS = 7200
TRAIN_RANGE = (0, 6000)
VALIDATION_RANGE = (6000, 7200)
INFERENCE_RANGE = (6200, 7200)
HORIZON_DAYS = 10

TRAIN_CODE, VALIDATION_CODE, UNUSED_CODE = 1, 2, 0

VALIDATION_ROLLOUT_DAYS = 360
VALIDATION_START_STRIDE = 6
VALIDATION_START_RANGE = (VALIDATION_RANGE[0], INFERENCE_RANGE[0])

MAXIMUM_INFERENCE_ROLLOUT_DAYS = 2000
SHORT_TERM_LEAD_DAYS = 200
#: Starts admitting a full 2,000-day lead-matched truth window inside the store.
INFERENCE_START_RANGE = (
    INFERENCE_RANGE[0],
    STORE_DAYS - MAXIMUM_INFERENCE_ROLLOUT_DAYS,
)


class BireProtocolSplitError(RuntimeError):
    """Raised when the Bire Section 3.2 arrangement is violated."""


def split_codes(record_days: int = RECORD_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """Snapshot and pair codes for the Bire arrangement.

    The arrays are 7,200 long, not 9,000: indices beyond Bire's record simply do
    not exist for this protocol, so an out-of-range read fails rather than
    silently returning code 0.
    """

    snapshots = np.zeros(int(record_days), dtype=np.uint8)
    pairs = np.zeros(int(record_days), dtype=np.uint8)
    for (start, stop), code in (
        (TRAIN_RANGE, TRAIN_CODE),
        (VALIDATION_RANGE, VALIDATION_CODE),
    ):
        snapshots[start:stop] = code
        pair_stop = stop - HORIZON_DAYS
        if pair_stop > start:
            pairs[start:pair_stop] = code
    return snapshots, pairs


def store_codes() -> tuple[np.ndarray, np.ndarray]:
    """The same arrangement, padded to the full 9,000-day store.

    Several shared readers require code arrays spanning the whole store and
    validate their length against it.  Days 7200--8999 carry code 0 --- not
    training, not validation --- which is exactly their role: evaluation truth
    only.  Restricted to 0--7199 these arrays are identical to
    :func:`split_codes`, and :func:`verify` asserts that.
    """

    snapshots = np.zeros(STORE_DAYS, dtype=np.uint8)
    pairs = np.zeros(STORE_DAYS, dtype=np.uint8)
    model_snapshots, model_pairs = split_codes()
    snapshots[:RECORD_DAYS] = model_snapshots
    pairs[:RECORD_DAYS] = model_pairs
    return snapshots, pairs


def validation_starts() -> np.ndarray:
    """360-day selection rollouts from the validation days outside the inference set."""

    low, high = VALIDATION_START_RANGE
    starts = np.arange(low, high, VALIDATION_START_STRIDE, dtype=np.int64)
    if int(starts[-1]) + VALIDATION_ROLLOUT_DAYS >= VALIDATION_RANGE[1]:
        raise BireProtocolSplitError("a validation rollout leaves the validation block")
    return starts


def validation_records(experiment_count: int = 3) -> np.ndarray:
    starts = validation_starts()
    return np.asarray(
        [
            (experiment, int(start))
            for experiment in range(int(experiment_count))
            for start in starts
        ],
        dtype=np.int64,
    )


def inference_starts(count: int, seed: int) -> np.ndarray:
    """``count`` members drawn from the inference days admitting a 2,000-day rollout."""

    low, high = INFERENCE_START_RANGE
    candidates = np.arange(low, high, dtype=np.int64)
    if candidates.size < count:
        raise BireProtocolSplitError("not enough inference starts for the ensemble")
    starts = np.sort(
        np.random.default_rng(int(seed)).choice(candidates, size=count, replace=False)
    )
    if int(starts[-1]) + MAXIMUM_INFERENCE_ROLLOUT_DAYS >= STORE_DAYS:
        raise BireProtocolSplitError(
            "an inference start cannot complete a 2,000-day truth window"
        )
    return starts


def verify(record_days: int = RECORD_DAYS) -> dict[str, Any]:
    """Assert the arrangement matches the paper and leaks nothing into validation."""

    snapshots, pairs = split_codes(record_days)
    train = np.flatnonzero(snapshots == TRAIN_CODE)
    validation = np.flatnonzero(snapshots == VALIDATION_CODE)
    if train.size != 6000 or validation.size != 1200:
        raise BireProtocolSplitError("the 6000/1200 arrangement changed")
    if train.max() + 1 != validation.min():
        raise BireProtocolSplitError("validation must follow training immediately")
    if train.size + validation.size != RECORD_DAYS:
        raise BireProtocolSplitError("train plus validation must equal Bire's 7,200")
    inference = np.arange(*INFERENCE_RANGE)
    if inference.size != 1000 or not np.all(snapshots[inference] == VALIDATION_CODE):
        raise BireProtocolSplitError(
            "the inference set must be the final 1,000 days of validation"
        )
    training_pairs = np.flatnonzero(pairs == TRAIN_CODE)
    if int(training_pairs.max()) + HORIZON_DAYS >= VALIDATION_RANGE[0]:
        raise BireProtocolSplitError("a training pair reaches into validation")
    selection = validation_starts()
    if np.any(selection >= INFERENCE_RANGE[0]):
        raise BireProtocolSplitError("a selection start falls inside the inference set")
    store_snapshots, store_pairs = store_codes()
    if not np.array_equal(store_snapshots[:RECORD_DAYS], snapshots) or not np.array_equal(
        store_pairs[:RECORD_DAYS], pairs
    ):
        raise BireProtocolSplitError("store-padded codes disagree with the arrangement")
    if np.any(store_snapshots[RECORD_DAYS:] != UNUSED_CODE) or np.any(
        store_pairs[RECORD_DAYS:] != UNUSED_CODE
    ):
        raise BireProtocolSplitError("evaluation-truth days must carry no split code")
    unused = np.flatnonzero(snapshots == UNUSED_CODE)
    return {
        "version": VERSION,
        "train": [int(train.min()), int(train.max())],
        "validation": [int(validation.min()), int(validation.max())],
        "inference": list(INFERENCE_RANGE),
        "train_days": int(train.size),
        "validation_days": int(validation.size),
        "inference_days": int(inference.size),
        "bire_record_days": RECORD_DAYS,
        "buffers": "none; the paper has none, leakage is prevented by the rollout window",
        "latest_training_rollout_start": int(training_pairs.max()) - 2 * HORIZON_DAYS,
        "validation_selection_starts": [
            int(selection.min()),
            int(selection.max()),
            int(selection.size),
        ],
        "inference_start_range": list(INFERENCE_START_RANGE),
        "usable_record_days": int(RECORD_DAYS),
        "store_days": int(STORE_DAYS),
        "model_visible_days": [0, int(RECORD_DAYS) - 1],
        "evaluation_truth_only_days": [int(RECORD_DAYS), int(STORE_DAYS) - 1],
        "separation_note": (
            "the model sees nothing at or beyond index 7200; days 7200-8999 are "
            "evaluation truth only, never trained on, validated on, or used as an "
            "inference start. Bire obtain the same separation across simulations "
            "(2 and 4 are entirely held out), we obtain it along time"
        ),
        "short_term_lead_days": int(SHORT_TERM_LEAD_DAYS),
        "long_term_lead_days": int(MAXIMUM_INFERENCE_ROLLOUT_DAYS),
        "long_term_reference": (
            "lead-matched MITgcm truth covers the whole 2,000-day rollout: starts "
            "are drawn from 6200-6999 and the store runs to 8999, so every member "
            "is scored against truth at every lead, with climatology and "
            "persistence as the two reference curves"
        ),
        "unused_within_record": int(unused.size),
        "inference_nested_in_validation": True,
        "nesting_note": (
            "Bire's inference set is the final 1,000 days of validation and the "
            "paper states no third held-out set is used; selection starts are drawn "
            "from the 200 validation days outside it so no start is shared"
        ),
    }


def assert_model_visible(indices: Any, label: str) -> None:
    """Fail if anything the *model* sees reaches past Bire's 7,200-day record.

    Applies to training targets, validation rollouts, and inference starts --- not
    to evaluation truth, which the paper also has beyond what the model saw.
    """

    values = np.asarray(indices)
    if values.size and int(values.max()) >= RECORD_DAYS:
        raise BireProtocolSplitError(
            f"{label} reaches index {int(values.max())}, past Bire's "
            f"{RECORD_DAYS}-day model record"
        )


def assert_truth_available(indices: Any, label: str) -> None:
    """Fail if evaluation truth reaches past the end of the store."""

    values = np.asarray(indices)
    if values.size and int(values.max()) >= STORE_DAYS:
        raise BireProtocolSplitError(
            f"{label} reaches index {int(values.max())}, past the {STORE_DAYS}-day store"
        )
