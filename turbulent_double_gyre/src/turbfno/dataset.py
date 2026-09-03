"""Trajectory store, the Bire Section 3.2 split, normalizers and the rollout set.

Three things live here:

* the **split** --- train ``[0, 6000)``, validation ``[6000, 7200)`` with the
  inference block ``[6200, 7200)`` nested inside it, and days 7200--8999 present
  only as lead-matched MITgcm evaluation truth;
* the **normalization**, computed from the training block alone: a pointwise
  ``(46, 248, 248)`` mean and scale, and the per-channel RMS of the normalized
  ten-day increment the objective's increment term divides by;
* the **five physical static channels** and the one-input rollout dataset that
  hands the operator ``[x_t, S]``.

Nothing here is inherited from a previous run. Both normalization artifacts are
recomputed by the training entry point over training days only.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from .runtime import (
    Dataset,
    HORIZON_DAYS,
    STATE_CHANNELS,
    STATE_CHANNEL_COUNT,
    STORE_STATIC_FEATURES,
    WIND_FEATURE_INDEX,
    _float_sha256,
    require_runtime,
    torch,
)

DATASET_VERSION = "trajectories_turb_v1"

SPLIT_VERSION = "af_bire_protocol_split_v1"

EXPERIMENTS = ("S0_turb", "S1_turb", "S2_turb")

GRID_SHAPE = (248, 248)

#: Degrees per grid cell. The domain is the same 62 x 62 degree basin as the
#: 1-degree store, so this is the only geometric quantity that changed.
DEGREES_PER_CELL = 0.25

#: Days physically present in the store.
STORE_DAYS = 9000

#: Days Bire's protocol makes visible to the model at all.
RECORD_DAYS = 7200

TRAIN_RANGE = (0, 6000)

VALIDATION_RANGE = (6000, 7200)

INFERENCE_RANGE = (6200, 7200)

TRAIN_CODE, VALIDATION_CODE, UNUSED_CODE = 1, 2, 0

VALIDATION_ROLLOUT_DAYS = 360

VALIDATION_START_STRIDE = 6

#: Selection rollouts are drawn from the validation days *outside* the nested
#: inference block, so no checkpoint is selected on a start it is later scored on.
VALIDATION_START_RANGE = (VALIDATION_RANGE[0], INFERENCE_RANGE[0])

MAXIMUM_INFERENCE_ROLLOUT_DAYS = 2000

SHORT_TERM_LEAD_DAYS = 200

INFERENCE_START_RANGE = (
    INFERENCE_RANGE[0],
    STORE_DAYS - MAXIMUM_INFERENCE_ROLLOUT_DAYS,
)

#: The store's own three-block chronological split, used only to confirm the
#: dataset's identity before this protocol's split is applied in memory.
V3_TRAIN_RANGE = (0, 5040)

V3_VALIDATION_RANGE = (5130, 6390)

V3_TEST_RANGE = (6480, 9000)


class BireProtocolSplitError(RuntimeError):
    """Raised when the Bire Section 3.2 arrangement is violated."""


class StaticChannelError(RuntimeError):
    """Raised when the physical static channels cannot be trusted."""


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

    Days 7200--8999 carry code 0 --- not training, not validation --- which is
    exactly their role: evaluation truth only.
    """

    snapshots = np.zeros(STORE_DAYS, dtype=np.uint8)
    pairs = np.zeros(STORE_DAYS, dtype=np.uint8)
    model_snapshots, model_pairs = split_codes()
    snapshots[:RECORD_DAYS] = model_snapshots
    pairs[:RECORD_DAYS] = model_pairs
    return snapshots, pairs


def v3_split_codes() -> tuple[np.ndarray, np.ndarray]:
    """Chronological snapshot and pair codes the v3 store itself carries."""

    snapshots = np.zeros(STORE_DAYS, dtype=np.uint8)
    pairs = np.zeros(STORE_DAYS, dtype=np.uint8)
    for (start, stop), code in (
        (V3_TRAIN_RANGE, 1),
        (V3_VALIDATION_RANGE, 2),
        (V3_TEST_RANGE, 3),
    ):
        snapshots[start:stop] = code
        pair_stop = stop - HORIZON_DAYS
        if pair_stop > start:
            pairs[start:pair_stop] = code
    return snapshots, pairs


def assert_store_is_v3(group: Any) -> None:
    """Confirm the store is the turbulent one, then override its split in memory.

    The store carries the project's own three-block split. This protocol
    deliberately applies the Bire Section 3.2 arrangement instead, so the stored
    arrays must *not* match the declaration and cannot be used as the check.
    What is checked is that we are on the expected store; the Bire codes are then
    computed in memory and the store is never modified.
    """

    if str(group.attrs.get("version")) != DATASET_VERSION:
        raise BireProtocolSplitError(f"the store is not {DATASET_VERSION}")
    stored = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    if stored.size != STORE_DAYS:
        raise BireProtocolSplitError("the store has an unexpected record length")
    expected, _ = v3_split_codes()
    if not np.array_equal(stored, expected):
        raise BireProtocolSplitError(
            "the store no longer carries its own three-block split"
        )


def validation_starts() -> np.ndarray:
    """360-day selection rollouts from the validation days outside inference."""

    low, high = VALIDATION_START_RANGE
    starts = np.arange(low, high, VALIDATION_START_STRIDE, dtype=np.int64)
    if int(starts[-1]) + VALIDATION_ROLLOUT_DAYS >= VALIDATION_RANGE[1]:
        raise BireProtocolSplitError("a validation rollout leaves the validation block")
    return starts


def validation_records(experiment_count: int = len(EXPERIMENTS)) -> np.ndarray:
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
    """``count`` members drawn from the inference days admitting 2,000 days."""

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


def assert_model_visible(indices: Any, label: str) -> None:
    """Fail if anything the *model* sees reaches past Bire's 7,200-day record."""

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
    return {
        "version": SPLIT_VERSION,
        "train": [int(train.min()), int(train.max())],
        "validation": [int(validation.min()), int(validation.max())],
        "inference": list(INFERENCE_RANGE),
        "train_days": int(train.size),
        "validation_days": int(validation.size),
        "inference_days": int(inference.size),
        "bire_record_days": RECORD_DAYS,
        "buffers": "none; the paper has none, leakage is prevented by the rollout window",
        "validation_selection_starts": [
            int(selection.min()),
            int(selection.max()),
            int(selection.size),
        ],
        "inference_start_range": list(INFERENCE_START_RANGE),
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
        "inference_nested_in_validation": True,
        "nesting_note": (
            "nested validation/inference protocol; there is no independent third "
            "test split. Bire's inference set is the final 1,000 days of "
            "validation and the paper states no third held-out set is used; "
            "selection starts are drawn from the 200 validation days outside it "
            "so no start is shared"
        ),
    }


def rollout_start_indices(
    pair_codes: Sequence[int],
    split_code: int,
    *,
    horizon_days: int = HORIZON_DAYS,
    rollout_steps: int = 6,
) -> tuple[int, ...]:
    """Starts whose complete autoregressive target sequence stays in one split."""

    codes = np.asarray(pair_codes, dtype=np.uint8)
    if codes.ndim != 1 or horizon_days <= 0 or rollout_steps <= 0:
        raise ValueError("split codes must be one-dimensional and horizons positive")
    latest = codes.size - rollout_steps * horizon_days
    if latest <= 0:
        return ()
    return tuple(
        time_index
        for time_index in range(latest)
        if all(
            codes[time_index + step * horizon_days] == split_code
            for step in range(rollout_steps)
        )
    )


def records_for_rollout_split(
    pair_codes: Sequence[int],
    split_code: int,
    *,
    experiment_count: int = len(EXPERIMENTS),
    horizon_days: int = HORIZON_DAYS,
    rollout_steps: int = 6,
) -> tuple[tuple[int, int], ...]:
    """``(regime, start)`` records for every complete rollout in one split.

    The one-input contract needs no history state, so every start from day 0
    onward that admits the full target sequence is usable.
    """

    starts = rollout_start_indices(
        pair_codes, split_code, horizon_days=horizon_days, rollout_steps=rollout_steps
    )
    if not starts:
        raise ValueError(f"dataset has no complete rollouts for pair split {split_code}")
    return tuple(
        (experiment, time_index)
        for experiment in range(experiment_count)
        for time_index in starts
    )


def _contiguous_runs(indices: np.ndarray) -> tuple[tuple[int, int], ...]:
    if indices.ndim != 1 or not indices.size:
        raise ValueError("normalizer indices must be a nonempty vector")
    cuts = np.flatnonzero(np.diff(indices) != 1) + 1
    return tuple(
        (int(piece[0]), int(piece[-1]) + 1) for piece in np.split(indices, cuts)
    )


def store_wind_normalization(group: Any) -> tuple[np.ndarray, float, float]:
    """The wet mask and the pooled wet-cell wind standardization of the store."""

    if tuple(group.attrs.get("state_channels", ())) != STATE_CHANNELS:
        raise ValueError("the store's 46-channel state contract changed")
    if tuple(group.attrs.get("static_features", ())) != STORE_STATIC_FEATURES:
        raise ValueError("the store's static-feature contract changed")
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    wind = np.asarray(group["static_features"][:, WIND_FEATURE_INDEX], dtype=np.float32)
    values = wind[:, wet]
    mean, scale = float(values.mean()), float(values.std())
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("the store's wind forcing has no usable spread")
    return wet, mean, scale


def training_pointwise_normalizers(
    group: Any,
    snapshot_codes: np.ndarray,
    *,
    floor_quantile: float = 0.05,
    absolute_floor: float = 1.0e-6,
    chunk_days: int = 64,
) -> dict[str, Any]:
    """Pointwise mean and scale over the training block of every regime.

    ``x_hat_c(y, x) = (x_c(y, x) - mu_c(y, x)) / sigma_c(y, x)``, with land reset
    to zero. The scale is floored per channel at its own 5th percentile over wet
    cells so a point where a field barely varies --- pinned surface temperature,
    for instance --- cannot divide by an almost-zero standard deviation and
    manufacture a channel two orders of magnitude larger than every other input.

    Computed from training days only, in one streaming pass; nothing is reused
    from any earlier run.
    """

    if not 0.0 <= floor_quantile < 0.5:
        raise ValueError("floor quantile must lie in [0, 0.5)")
    if absolute_floor <= 0.0 or chunk_days <= 0:
        raise ValueError("normalizer floors and chunks must be positive")
    state = group["state"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    codes = np.asarray(snapshot_codes, dtype=np.uint8)
    selected = np.flatnonzero(codes == TRAIN_CODE)
    if (
        state.ndim != 5
        or state.shape[2] != STATE_CHANNEL_COUNT
        or state.shape[3:] != GRID_SHAPE
        or wet.shape != GRID_SHAPE
        or codes.size != state.shape[1]
    ):
        raise ValueError("the normalizer received an inconsistent dataset")

    sums = np.zeros((STATE_CHANNEL_COUNT, *GRID_SHAPE), dtype=np.float64)
    squares = np.zeros_like(sums)
    count = 0
    for experiment in range(state.shape[0]):
        for run_start, run_stop in _contiguous_runs(selected):
            for start in range(run_start, run_stop, chunk_days):
                stop = min(start + chunk_days, run_stop)
                block = np.asarray(state[experiment, start:stop], dtype=np.float64)
                sums += block.sum(axis=0)
                squares += np.square(block).sum(axis=0)
                count += int(block.shape[0])
    expected = int(selected.size * state.shape[0])
    if count != expected:
        raise ValueError(f"the normalizer read {count} snapshots, not {expected}")

    mean64 = sums / count
    variance64 = np.maximum(squares / count - np.square(mean64), 0.0)
    raw_scale64 = np.sqrt(variance64)
    floors64 = np.maximum(
        np.quantile(raw_scale64[:, wet], floor_quantile, axis=1), absolute_floor
    )
    scale64 = np.maximum(raw_scale64, floors64[:, None, None])
    mean64[:, ~wet] = 0.0
    raw_scale64[:, ~wet] = 0.0
    scale64[:, ~wet] = 1.0
    mean = np.ascontiguousarray(mean64, dtype=np.float32)
    raw_scale = np.ascontiguousarray(raw_scale64, dtype=np.float32)
    scale = np.ascontiguousarray(scale64, dtype=np.float32)
    floors = np.ascontiguousarray(floors64, dtype=np.float32)
    floored = np.mean(raw_scale[:, wet] < floors[:, None], axis=1).astype(np.float32)
    if (
        not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise ValueError("the recomputed normalizers are not finite and positive")
    return {
        "mean": mean,
        "raw_scale": raw_scale,
        "scale": scale,
        "floor": floors,
        "summary": normalizer_summary(
            mean,
            raw_scale,
            scale,
            floors,
            wet,
            snapshots_total=count,
            snapshots_per_regime=int(selected.size),
            floor_quantile=float(floor_quantile),
            absolute_floor=float(absolute_floor),
        ),
    }


def normalizer_summary(
    mean: np.ndarray,
    raw_scale: np.ndarray,
    scale: np.ndarray,
    floors: np.ndarray,
    wet: np.ndarray,
    *,
    snapshots_total: int,
    snapshots_per_regime: int,
    floor_quantile: float = 0.05,
    absolute_floor: float = 1.0e-6,
) -> dict[str, Any]:
    """Describe a set of normalizers, from the arrays alone.

    Every field is a function of the four arrays and the wet mask, so a run that
    reads normalizers back from disk describes them exactly as the run that
    computed them did --- there is no second implementation to drift.
    """

    floored = np.mean(
        np.asarray(raw_scale)[:, wet] < np.asarray(floors)[:, None], axis=1
    ).astype(np.float32)
    return {
        "source": "train_only_days_0_5999_of_every_regime",
        "reused_from_a_previous_run": False,
        "training_snapshots_total": int(snapshots_total),
        "training_snapshots_per_regime": int(snapshots_per_regime),
        "wet_cells": int(np.asarray(wet).sum()),
        "floor_quantile": float(floor_quantile),
        "absolute_floor": float(absolute_floor),
        "pointwise_mean_sha256": _float_sha256(mean),
        "pointwise_raw_scale_sha256": _float_sha256(raw_scale),
        "pointwise_scale_sha256": _float_sha256(scale),
        "channel_floor_sha256": _float_sha256(floors),
        "channel_floor": np.asarray(floors).tolist(),
        "fraction_wet_cells_floored": floored.tolist(),
        "maximum_fraction_wet_cells_floored": float(floored.max()),
    }


def training_increment_scale(
    group: Any,
    pair_codes: np.ndarray,
    pointwise_scale: np.ndarray,
    *,
    chunk_days: int = 32,
) -> np.ndarray:
    """Per-channel RMS of the normalized ten-day increment over training pairs.

    This is the divisor of the objective's increment term, so it too is a
    training-only quantity and is recomputed here rather than read from a report.
    """

    state = group["state"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    scale = np.asarray(pointwise_scale, dtype=np.float32)
    if scale.shape != (STATE_CHANNEL_COUNT, *wet.shape):
        raise ValueError("the increment scale received the wrong pointwise scale")
    runs = _contiguous_runs(
        np.flatnonzero(np.asarray(pair_codes, dtype=np.uint8) == TRAIN_CODE)
    )
    squares = np.zeros(STATE_CHANNEL_COUNT, dtype=np.float64)
    count = 0
    for experiment in range(state.shape[0]):
        for run_start, run_stop in runs:
            for start in range(run_start, run_stop, chunk_days):
                stop = min(start + chunk_days, run_stop)
                present = np.asarray(state[experiment, start:stop], dtype=np.float32)
                future = np.asarray(
                    state[experiment, start + HORIZON_DAYS : stop + HORIZON_DAYS],
                    dtype=np.float32,
                )
                increment = ((future - present) / scale[None])[:, :, wet]
                squares += np.square(increment.astype(np.float64)).sum(axis=(0, 2))
                count += int(increment.shape[0] * wet.sum())
    result = np.sqrt(squares / count).astype(np.float32)
    if (
        count <= 0
        or result.shape != (STATE_CHANNEL_COUNT,)
        or not np.all(np.isfinite(result))
        or np.any(result <= 0.0)
    ):
        raise ValueError("the recomputed increment scales are not finite and positive")
    return result


#: MITgcm's own defaults, which this setup does not override. ``data`` sets
#: ``usingSphericalPolarGrid=.TRUE.`` and declares neither ``rotationPeriod`` nor
#: an f-plane/beta-plane pair, so the model integrates fCori = 2 omega sin(phi)
#: with omega = 2 pi / 86164 s.
MITGCM_ROTATION_PERIOD_SECONDS = 86164.0

MITGCM_OMEGA = 2.0 * math.pi / MITGCM_ROTATION_PERIOD_SECONDS

#: The relaxation timescale the setup declares, kept only as an audited
#: constant: the *target* field is a model input, the timescale is not.
MITGCM_THETA_RELAX_SECONDS = 2592000.0

#: Index of Theta_01, kept so the rejected normalization alternative for the SST
#: relaxation target can be measured and recorded rather than merely asserted.
SURFACE_TEMPERATURE_INDEX = STATE_CHANNELS.index("Theta_01")

#: The production static block: two forcing/geometry fields taken from the
#: store, and three coefficients of the governing equations.
STATIC_FEATURES = (
    "wind_stress_x",
    "wet_mask",
    "coriolis_parameter",
    "zonal_grid_spacing",
    "sst_relaxation_target",
)

STATIC_CHANNEL_COUNT = len(STATIC_FEATURES)

_STORE_SOURCED_STATIC_INDICES = {
    "wind_stress_x": STORE_STATIC_FEATURES.index("wind_stress_x"),
    "wet_mask": STORE_STATIC_FEATURES.index("wet_mask"),
}

_REQUIRED_MITGCM_SETTINGS = (
    ("usingSphericalPolarGrid", ".TRUE."),
    ("thetaClimFile", "'SST_relax.bin'"),
    ("tauThetaClimRelax", f"{MITGCM_THETA_RELAX_SECONDS:.0f}."),
)

_FORBIDDEN_MITGCM_SETTINGS = ("rotationPeriod", "f0", "beta", "fPrime")


def read_mitgcm_2d(path: str | Path) -> np.ndarray:
    """Read one big-endian float32 248x248 MITgcm record."""

    resolved = Path(path)
    values = np.fromfile(resolved, dtype=">f4")
    if values.size != GRID_SHAPE[0] * GRID_SHAPE[1]:
        raise StaticChannelError(
            f"{resolved} holds {values.size} float32 values, not one 248x248 record"
        )
    result = np.ascontiguousarray(values.reshape(GRID_SHAPE), dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise StaticChannelError(f"{resolved} contains non-finite values")
    return result


def assert_mitgcm_coriolis_defaults(data_path: str | Path) -> dict[str, Any]:
    """Confirm the run really integrates the coefficients this module derives.

    Coriolis is not dumped by MITgcm, so it is computed here from latitude. That
    is only correct while the setup keeps the spherical-polar grid and MITgcm's
    default rotation period, and while the relaxation target is the file being
    read. Rather than trust that, the declaration is parsed and any override is
    a hard failure.
    """

    resolved = Path(data_path)
    stripped = "".join(resolved.read_text().split())
    missing = [
        f"{key}={value}"
        for key, value in _REQUIRED_MITGCM_SETTINGS
        if f"{key}={value}" not in stripped
    ]
    overridden = [key for key in _FORBIDDEN_MITGCM_SETTINGS if f"{key}=" in stripped]
    if missing or overridden:
        raise StaticChannelError(
            "the MITgcm declaration no longer matches the derived static channels: "
            f"missing={missing!r}, overridden_defaults={overridden!r}"
        )
    return {
        "data_file": str(resolved),
        "rotation_period_seconds": MITGCM_ROTATION_PERIOD_SECONDS,
        "omega_rad_per_second": MITGCM_OMEGA,
        "coriolis_definition": "2*omega*sin(latitude), MITgcm spherical-polar default",
        "theta_relax_seconds": MITGCM_THETA_RELAX_SECONDS,
        "asserted_settings": [f"{k}={v}" for k, v in _REQUIRED_MITGCM_SETTINGS],
        "asserted_absent": list(_FORBIDDEN_MITGCM_SETTINGS),
    }


def coriolis_parameter(latitude_deg: np.ndarray) -> np.ndarray:
    """f = 2 omega sin(phi), the coefficient MITgcm itself builds from YC."""

    latitude = np.asarray(latitude_deg, dtype=np.float64)
    return np.ascontiguousarray(
        2.0 * MITGCM_OMEGA * np.sin(np.deg2rad(latitude)), dtype=np.float32
    )


def _standardize(field: np.ndarray, wet: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Center and scale a static coefficient over wet cells, then mask the land.

    These fields are geometry and boundary forcing, not model state, so there is
    no train/validation distinction to preserve: the same deterministic array is
    present at every day of the record.
    """

    values = np.asarray(field, dtype=np.float64)
    wet_values = values[wet]
    mean = float(wet_values.mean())
    scale = float(wet_values.std())
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0.0:
        raise StaticChannelError("a derived static channel has no usable spread")
    result = ((values - mean) / scale).astype(np.float32)
    result[~wet] = 0.0
    return np.ascontiguousarray(result), mean, scale


def static_block(
    group: Any,
    *,
    zonal_spacing_path: str | Path,
    sst_relax_path: str | Path,
    data_path: str | Path,
    pointwise_mean: np.ndarray,
    pointwise_scale: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the normalized ``(experiments, 5, Y, X)`` physical static block.

    ``wind_stress_x`` and ``wet_mask`` are taken from the store. The three added
    channels are coefficients of the governing equations:

    ``coriolis_parameter``
        derived from the store's own latitude, asserted equal to MITgcm's ``YC``;
    ``zonal_grid_spacing``
        MITgcm's own ``DXF`` at tracer points, read rather than recomputed, so
        the spherical-grid convergence the model integrates is the one the
        network sees;
    ``sst_relaxation_target``
        the ``thetaClimFile`` field, standardized over wet cells like the others.

    A note on that last choice, because the elegant alternative is wrong here.
    Normalizing the target in the *surface temperature channel's* pointwise
    coordinates would make the relaxation difference exactly representable: with
    both divided by the same ``scale(x, y)``, one lifting weight pair forms
    ``theta_hat - theta_clim_hat``. But that pointwise scale falls to ~0.011 degC
    where the simulated SST is pinned and barely varies, and dividing a fixed
    climatology by it yields a channel two orders of magnitude above every other
    input into the same 1x1 lifting. Standardizing on its own keeps the channel
    at unit variance; the network reconstructs the difference through the
    lifting's 128 output features instead of through one weight pair. The size of
    the rejected alternative is measured and recorded below.
    """

    provenance = assert_mitgcm_coriolis_defaults(data_path)
    store_static = np.asarray(group["static_features"][:], dtype=np.float32)
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float32)
    mean = np.asarray(pointwise_mean, dtype=np.float32)
    scale = np.asarray(pointwise_scale, dtype=np.float32)
    experiments = int(store_static.shape[0])
    if (
        store_static.shape[1] != len(STORE_STATIC_FEATURES)
        or wet.shape != GRID_SHAPE
        or latitude.shape != GRID_SHAPE
        or mean.shape != (STATE_CHANNEL_COUNT, *GRID_SHAPE)
        or scale.shape != mean.shape
    ):
        raise StaticChannelError("the store or normalizer contract changed")

    spacing = read_mitgcm_2d(zonal_spacing_path)
    target = read_mitgcm_2d(sst_relax_path)
    # MITgcm builds DXF from the same latitudes the store carries; if the two
    # ever describe different grids the channel would be silently wrong.
    predicted = (
        6.370e6
        * np.cos(np.deg2rad(latitude.astype(np.float64)))
        * np.deg2rad(DEGREES_PER_CELL)
    )
    if float(np.abs(spacing - predicted).max()) > 1.0:
        raise StaticChannelError(
            "DXF disagrees with R cos(phi) d(lambda) on the store's own latitudes"
        )

    coriolis, coriolis_mean, coriolis_scale = _standardize(
        coriolis_parameter(latitude), wet
    )
    spacing_normalized, spacing_mean, spacing_scale = _standardize(spacing, wet)
    relax, relax_mean, relax_scale = _standardize(target, wet)
    rejected = (
        (target.astype(np.float64) - mean[SURFACE_TEMPERATURE_INDEX])
        / scale[SURFACE_TEMPERATURE_INDEX]
    )[wet]

    wind_source = store_static[:, _STORE_SOURCED_STATIC_INDICES["wind_stress_x"]]
    wind_values = wind_source[:, wet]
    wind_mean = float(wind_values.mean())
    wind_scale = float(wind_values.std())
    if not np.isfinite(wind_scale) or wind_scale <= 0.0:
        raise StaticChannelError("the store's wind forcing has no usable spread")

    block = np.zeros(
        (experiments, STATIC_CHANNEL_COUNT, *GRID_SHAPE), dtype=np.float32
    )
    for experiment in range(experiments):
        wind = (wind_source[experiment] - wind_mean) / wind_scale
        wind[~wet] = 0.0
        block[experiment, 0] = wind
        block[experiment, 1] = store_static[
            experiment, _STORE_SOURCED_STATIC_INDICES["wet_mask"]
        ]
        block[experiment, 2] = coriolis
        block[experiment, 3] = spacing_normalized
        block[experiment, 4] = relax
    if not np.all(np.isfinite(block)):
        raise StaticChannelError("the derived static block is not finite")

    provenance.update(
        {
            "channels": list(STATIC_FEATURES),
            "sources": {
                "wind_stress_x": "trajectory_v3_static_features_index_0",
                "wet_mask": "trajectory_v3_static_features_index_3",
                "coriolis_parameter": "derived_from_trajectory_v3_latitude_deg",
                "zonal_grid_spacing": str(Path(zonal_spacing_path)),
                "sst_relaxation_target": str(Path(sst_relax_path)),
            },
            "normalization": {
                "wind_stress_x": {
                    "method": "pooled_wet_cell_standardization",
                    "mean": wind_mean,
                    "scale": wind_scale,
                },
                "wet_mask": {"method": "raw_zero_one_indicator"},
                "coriolis_parameter": {
                    "method": "wet_cell_standardization",
                    "mean": coriolis_mean,
                    "scale": coriolis_scale,
                },
                "zonal_grid_spacing": {
                    "method": "wet_cell_standardization",
                    "mean": spacing_mean,
                    "scale": spacing_scale,
                },
                "sst_relaxation_target": {
                    "method": "wet_cell_standardization",
                    "mean": relax_mean,
                    "scale": relax_scale,
                    "rejected_alternative": (
                        "pointwise_surface_temperature_coordinates_would_make_the_"
                        "relaxation_difference_exactly_representable_but_the_"
                        "pointwise_scale_falls_to_0p011_degC_where_SST_is_pinned"
                    ),
                    "rejected_alternative_channel_std": float(rejected.std()),
                    "rejected_alternative_channel_range": [
                        float(rejected.min()),
                        float(rejected.max()),
                    ],
                    "compared_state_channel": STATE_CHANNELS[SURFACE_TEMPERATURE_INDEX],
                },
            },
            "collinearity_note": {
                "measured_pearson_correlation_f_and_zonal_spacing": float(
                    np.corrcoef(
                        coriolis_parameter(latitude)[wet].astype(np.float64),
                        spacing[wet].astype(np.float64),
                    )[0, 1]
                ),
                "reason": (
                    "the basin spans 14.5N to 75.5N, symmetric about 45N, so "
                    "cos(phi) traverses the same value set as sin(phi) reversed "
                    "and the two coefficients are near mirror images"
                ),
                "consequence": (
                    "both are declared because both appear in the governing "
                    "equations, but a gain cannot be attributed to one of them "
                    "without a further ablation"
                ),
            },
            "physical_ranges": {
                "coriolis_parameter_per_second": [
                    float(coriolis_parameter(latitude)[wet].min()),
                    float(coriolis_parameter(latitude)[wet].max()),
                ],
                "zonal_grid_spacing_metres": [
                    float(spacing[wet].min()),
                    float(spacing[wet].max()),
                ],
                "sst_relaxation_target_degc": [
                    float(target[wet].min()),
                    float(target[wet].max()),
                ],
            },
        }
    )
    return block, provenance


def western_boundary_mask(wet_mask: np.ndarray, width: int = 4) -> np.ndarray:
    """Select the first ``width`` wet cells east of each row's western wall."""

    wet = np.asarray(wet_mask, dtype=bool)
    if wet.ndim != 2 or width <= 0:
        raise ValueError(
            "western boundary construction needs a two-dimensional wet mask "
            "and a positive width"
        )
    result = np.zeros_like(wet)
    for row in range(wet.shape[0]):
        columns = np.flatnonzero(wet[row])
        if columns.size:
            stop = min(columns[0] + width, wet.shape[1])
            result[row, columns[0] : stop] = wet[row, columns[0] : stop]
    if not np.any(result):
        raise ValueError(
            "wet mask has no cells from which to construct the western boundary band"
        )
    return result


class RolloutDataset(Dataset):
    """One-input rollout examples: ``[x_t, S]`` and the six future truth states.

    ``features`` is the model's final 51-channel layout --- the normalized
    present state followed by the five physical statics --- and ``futures`` is
    ``(rollout_steps, 46, Y, X)`` of normalized truth at ``t + 10 ... t + 60``.
    Three of the five statics are not in the store, so the block is built once
    up front and simply indexed by regime here.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        records: Sequence[tuple[int, int]],
        pointwise_mean: np.ndarray,
        pointwise_scale: np.ndarray,
        physical_statics: np.ndarray,
        *,
        horizon_days: int = HORIZON_DAYS,
        rollout_steps: int = 6,
    ) -> None:
        require_runtime()
        self.dataset_path = Path(dataset_path).resolve()
        self.records = tuple(
            (int(experiment), int(time_index)) for experiment, time_index in records
        )
        self.pointwise_mean = np.ascontiguousarray(pointwise_mean, dtype=np.float32)
        self.pointwise_scale = np.ascontiguousarray(pointwise_scale, dtype=np.float32)
        self.physical_statics = np.ascontiguousarray(physical_statics, dtype=np.float32)
        self.horizon_days = int(horizon_days)
        self.rollout_steps = int(rollout_steps)
        if not self.records or self.rollout_steps <= 0:
            raise ValueError("the rollout dataset needs records and positive steps")
        self._group: Any | None = None
        self._state: Any | None = None
        self._open()

    def _open(self) -> None:
        self._group = zarr.open_consolidated(str(self.dataset_path), mode="r")
        self._state = self._group["state"]
        self.wet = np.asarray(self._group["wet_mask"][:], dtype=bool)
        expected = (STATE_CHANNEL_COUNT, *self.wet.shape)
        if (
            self.pointwise_mean.shape != expected
            or self.pointwise_scale.shape != expected
            or np.any(self.pointwise_scale <= 0.0)
        ):
            raise ValueError("the rollout dataset's normalizer contract changed")
        if self.physical_statics.ndim != 4 or self.physical_statics.shape[1:] != (
            STATIC_CHANNEL_COUNT,
            *self.wet.shape,
        ):
            raise StaticChannelError("the physical static block has the wrong shape")
        if self.physical_statics.shape[0] <= max(e for e, _ in self.records):
            raise StaticChannelError(
                "the physical static block does not cover the requested regimes"
            )
        if any(
            time_index + self.horizon_days * self.rollout_steps >= self._state.shape[1]
            for _, time_index in self.records
        ):
            raise ValueError("a rollout exceeds the stored chronology")

    def __len__(self) -> int:
        return len(self.records)

    def _normalise_state(self, raw: np.ndarray) -> np.ndarray:
        result = (raw - self.pointwise_mean) / self.pointwise_scale
        result[:, ~self.wet] = 0.0
        return np.ascontiguousarray(result, dtype=np.float32)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        experiment, time_index = self.records[index]
        present = self._normalise_state(
            np.asarray(self._state[experiment, time_index], dtype=np.float32)
        )
        futures = np.stack(
            [
                self._normalise_state(
                    np.asarray(
                        self._state[experiment, time_index + step * self.horizon_days],
                        dtype=np.float32,
                    )
                )
                for step in range(1, self.rollout_steps + 1)
            ]
        )
        features = np.concatenate((present, self.physical_statics[experiment]), axis=0)
        return (
            torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(futures, dtype=np.float32)),
        )

    def __getstate__(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["_group"] = result["_state"] = None
        return result

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(state)
        self._open()
