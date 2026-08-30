"""Response-training data path for arm C (plan sections 14/15.2, step 12/13).

Reads the curated development response store
(``forward_response_v1.zarr``'s ``train``/``validation`` roles, built by
``scripts/extract_forward_response_dataset.py``, step 10) and builds the
deterministic auxiliary-update schedule section 15.2 specifies: on exactly
every fourth optimizer update, one response direction's (nominal, minus,
plus) triplet is trained against the model's own forward response.

**The schedule's isolation requirement.** Section 15.2: "The response
sampler uses an isolated counter/hash stream and cannot consume Python,
NumPy, or Torch RNG state used by initialization or the nominal sampler."
The schedule below is built entirely from SHA-256 digests of
``(seed, purpose, direction_id)`` strings -- never from ``random``,
``np.random``, or ``torch`` -- so it is reproducible independent of, and
never interleaved with, any global RNG draw the nominal path makes.

**Schedule construction**, exactly matching section 15.2's declared
invariants over a full 7,680-step (1,920 response-update) C run:

- pattern ``short, short, short, long`` repeating -- 1,440 short + 480 long;
- 96 long directions make exactly 5 complete passes (5*96=480);
- 576 short directions make 2 complete passes (2*576=1,152) plus one frozen
  288-direction balanced half-pass (72 per family, 24 per family/regime),
  totalling 1,152+288=1,440;
- "no replacement within a pass": every direction in a pass is used
  exactly once before any repeats.

Each pass is built by round-robin cycling through the 12
``(input_family, regime)`` strata (576/12=48 short or 96/12=8 long per
stratum), one direction per stratum per round, each stratum itself ordered
by a hash-keyed deterministic shuffle. This is a stronger, literal reading
of "input families are exactly balanced in blocks; regimes are exactly
balanced in blocks" than a single whole-pass shuffle would give: any window
of 12 consecutive picks from one pass contains exactly one direction from
every stratum, not just every 576 (long: every 12 picks, one per stratum;
short and long are separately round-robined). The 288-direction half-pass
uses the same round-robin construction restricted to a fixed half of each
stratum, chosen as the 24 (short) or -- not applicable, long has no
half-pass -- lowest-hash directions in that stratum (deterministic, and
"tie-broken by the direction hash" is then automatic since the ordering
key already is the hash).

The lambda screen (step 12, ``config/forward_response_lambda_screen_v1.json``)
uses exactly the first 480 entries of this same 1,920-entry schedule
("response_schedule_is_prefix_of_full_run_schedule": true) -- built once
here, not a separately constructed shorter schedule.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_PATH = Path(
    "/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/forward_response_v1.zarr"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "af_fno" / "response" / "forward_response_v1"

GROUPS = ("U", "V", "Theta", "SSH")
GROUP_SLICES = {"U": slice(0, 15), "V": slice(15, 30), "Theta": slice(30, 45), "SSH": slice(45, 46)}
REGIMES = ("S0", "S1", "S2")

FULL_RUN_RESPONSE_UPDATES = 1920
SCREEN_RESPONSE_UPDATES = 480
LONG_PASSES = 5
SHORT_FULL_PASSES = 2
SHORT_HALF_PASS_PER_STRATUM = 24  # half of 48 per (family, regime) stratum
STRATA = tuple((family, regime) for family in GROUPS for regime in REGIMES)


class ResponseDatasetError(RuntimeError):
    """Raised when the response-training data path cannot be legitimately used."""


@dataclass(frozen=True)
class ResponseDirection:
    direction_id: str
    role: str
    array_group: str  # "short" | "long"
    array_row: int
    anchor_row: int
    input_family: str
    regime: str
    region: str
    alpha: float
    long: bool


def _regime_of(direction_id: str) -> str:
    regime = direction_id.split(":")[2]
    if regime not in REGIMES:
        raise ResponseDatasetError(f"cannot parse regime from direction_id: {direction_id}")
    return regime


def load_direction_table(role: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> list[ResponseDirection]:
    path = output_root / f"{role}_direction_table.jsonl"
    if not path.is_file():
        raise ResponseDatasetError(f"direction table missing: {path}")
    directions = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            directions.append(
                ResponseDirection(
                    direction_id=row["direction_id"],
                    role=row["role"],
                    array_group=row["array_group"],
                    array_row=int(row["array_row"]),
                    anchor_row=int(row["anchor_row"]),
                    input_family=row["input_family"],
                    regime=_regime_of(row["direction_id"]),
                    region=row["region"],
                    alpha=float(row["alpha"]),
                    long=bool(row["long"]),
                )
            )
    return directions


def _hash_key(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _stratified_round_robin(
    directions: Sequence[ResponseDirection], *, seed: int, purpose: str
) -> list[ResponseDirection]:
    """One pass: round-robin over the 12 (family, regime) strata, each
    internally ordered by a hash-keyed deterministic shuffle."""

    by_stratum: dict[tuple[str, str], list[ResponseDirection]] = {stratum: [] for stratum in STRATA}
    for direction in directions:
        by_stratum[(direction.input_family, direction.regime)].append(direction)
    for stratum, members in by_stratum.items():
        members.sort(key=lambda d: _hash_key(seed, purpose, "shuffle", stratum, d.direction_id))
    sizes = {len(members) for members in by_stratum.values()}
    if len(sizes) != 1:
        raise ResponseDatasetError(f"strata are not equally sized for {purpose}: {[len(m) for m in by_stratum.values()]}")
    (per_stratum,) = sizes
    stratum_order = sorted(STRATA, key=lambda s: _hash_key(seed, purpose, "stratum_order", s))
    ordered: list[ResponseDirection] = []
    for round_index in range(per_stratum):
        for stratum in stratum_order:
            ordered.append(by_stratum[stratum][round_index])
    return ordered


def _short_half_pass(directions: Sequence[ResponseDirection], *, seed: int) -> list[ResponseDirection]:
    by_stratum: dict[tuple[str, str], list[ResponseDirection]] = {stratum: [] for stratum in STRATA}
    for direction in directions:
        by_stratum[(direction.input_family, direction.regime)].append(direction)
    half: list[ResponseDirection] = []
    for stratum, members in by_stratum.items():
        members = sorted(members, key=lambda d: _hash_key(seed, "short_half_pass_select", stratum, d.direction_id))
        if len(members) < SHORT_HALF_PASS_PER_STRATUM:
            raise ResponseDatasetError(f"stratum {stratum} has only {len(members)} directions, need {SHORT_HALF_PASS_PER_STRATUM}")
        half.extend(members[:SHORT_HALF_PASS_PER_STRATUM])
    return _stratified_round_robin(half, seed=seed, purpose="short_half_pass_order")


def build_auxiliary_schedule(
    seed: int,
    train_directions: Sequence[ResponseDirection],
    *,
    total_updates: int = FULL_RUN_RESPONSE_UPDATES,
) -> list[ResponseDirection]:
    """The full deterministic 1,920-entry (direction per response update)
    schedule for one seed. ``total_updates`` exists only so a caller can
    assert the expected length; the schedule itself is always built at full
    length and (for the screen) sliced afterward, since
    "response_schedule_is_prefix_of_full_run_schedule" must hold exactly."""

    short = [d for d in train_directions if d.array_group == "short"]
    long = [d for d in train_directions if d.array_group == "long"]
    if len(short) != 576 or len(long) != 96:
        raise ResponseDatasetError(f"expected 576 short / 96 long train directions, found {len(short)}/{len(long)}")

    long_stream: list[ResponseDirection] = []
    for pass_index in range(LONG_PASSES):
        long_stream.extend(_stratified_round_robin(long, seed=seed, purpose=f"long_pass_{pass_index}"))

    short_stream: list[ResponseDirection] = []
    for pass_index in range(SHORT_FULL_PASSES):
        short_stream.extend(_stratified_round_robin(short, seed=seed, purpose=f"short_pass_{pass_index}"))
    short_stream.extend(_short_half_pass(short, seed=seed))

    if len(long_stream) != 480 or len(short_stream) != 1440:
        raise ResponseDatasetError(f"schedule stream lengths wrong: short={len(short_stream)} long={len(long_stream)}")

    schedule: list[ResponseDirection] = []
    for block in range(480):
        schedule.extend(short_stream[3 * block : 3 * block + 3])
        schedule.append(long_stream[block])
    if len(schedule) != FULL_RUN_RESPONSE_UPDATES:
        raise ResponseDatasetError(f"schedule has {len(schedule)} entries, expected {FULL_RUN_RESPONSE_UPDATES}")
    if total_updates != FULL_RUN_RESPONSE_UPDATES:
        raise ResponseDatasetError("build_auxiliary_schedule always returns the full 1920-entry schedule; slice it")
    return schedule


def schedule_sha256(schedule: Sequence[ResponseDirection]) -> str:
    payload = "\n".join(d.direction_id for d in schedule).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ResponseStore:
    """Per-role zarr accessor: the nominal anchor state and one direction's
    (minus, plus) initial states and raw response differences."""

    def __init__(self, role: str, *, dataset_path: Path = DEFAULT_DATASET_PATH):
        self.role = role
        self._store = zarr.open_consolidated(str(dataset_path), mode="r")
        self._group = self._store[role]
        self._lead_days = tuple(int(v) for v in np.asarray(self._group["lead_days"]))

    @property
    def lead_days(self) -> tuple[int, ...]:
        return self._lead_days

    def anchor_state_p32(self, anchor_row: int) -> np.ndarray:
        return np.asarray(self._group["anchors"]["state_p32"][anchor_row], dtype=np.float32)

    def branch_inputs_p32(self, direction: ResponseDirection) -> tuple[np.ndarray, np.ndarray]:
        """Returns (minus, plus) initial P32 states, each (46,62,62)."""

        array = self._group[direction.array_group]["input_state_p32"]
        pair = np.asarray(array[direction.array_row], dtype=np.float32)
        return pair[0], pair[1]

    def raw_response(self, direction: ResponseDirection) -> tuple[np.ndarray, np.ndarray]:
        """Returns (minus, plus) raw P64 response differences
        ``P64[perturbed]-P64[nominal]``, each ``(L,46,62,62)`` -- ``L=1`` for
        short, ``L=6`` for long (train/validation leads 10..60)."""

        array = self._group[direction.array_group]["response_p64"]
        pair = np.asarray(array[direction.array_row], dtype=np.float64)
        return pair[0], pair[1]
