"""Audit and materialize response-v1 geometry, sealing the blind inventory.

This is execution step 4 of
``docs/Adjoint_faithful_response_training_plan.md``.  The default ``audit``
mode is deliberately non-secret and read-only: it verifies the frozen source,
grid, direction, support, quota, and capacity contracts, but never prints or
writes exact blind centres.  ``materialize`` is a separate capability: it
writes the public (pilot+train+validation) and blind (blind_test) geometry
manifests to two distinct, exclusively-created files (blind mode 0400, public
mode 0444), refusing to run if either path already exists. Per the roadmap's
2026-08-24 amendment, blind/development isolation is enforced by this
write-once file-mode convention and the separate evaluator-only output path
alone -- the earlier design (a live OS-identity/mount-namespace firewall) was
retired as disproportionate for a single-researcher project; nothing here
checks the calling process's identity.

The inventory contains geometry and restart-edit declarations only.  This
program never creates a response array, a perturbed pickup, or any other
numeric blind data.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os

# Must be set before numpy/scipy (and therefore OpenBLAS) are ever imported
# anywhere in the process, including this module's own import below: OpenBLAS
# reads its thread count at library-load time, and a fork()ed worker inherits
# whatever the parent already decided, so setting these later (e.g. inside a
# worker function) is too late to have any effect. Materializing the response
# inventory runs twelve (regime,family) groups in separate worker processes
# (materialize_inventory's ProcessPoolExecutor); OpenBLAS defaults to using
# every core it can see (measured MAX_THREADS=64 on this build) with no
# awareness of the other eleven siblings also trying to do the same on a
# shared machine. Verified directly: the real materialize run stalled for
# over 90 minutes on a centre-allocation MIP that an identical, single-
# process, non-parallel run of the same (regime,family,region) case had
# solved in under 8 minutes -- a >10x regression from thread oversubscription
# (up to 12 processes x 64 threads contending for far fewer physical cores,
# shared with other users' jobs), not an algorithmic one. Each worker's own
# MIP is already small once decomposed by region; it does not need or
# benefit from BLAS's own internal parallelism, so pinning to one thread per
# process and getting real parallelism from the twelve OS processes instead
# is strictly better here. Respects any value the caller already set.
for _blas_thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_blas_thread_env, "1")

import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# v3 replaces the exact MIP-based leximax centre-placement objective (v2,
# region-decomposed but still impractically slow at production scale) with a
# deterministic greedy farthest-point placement -- see the roadmap's
# 2026-08-26 amendment. v1 (broader all-five-regions solve scope) and v2
# never materialized anything and are kept only as historical records.
DEFAULT_DATASET_CONTRACT = PROJECT_ROOT / "config" / "forward_response_dataset_v3.json"
DEFAULT_PILOT_CONTRACT = PROJECT_ROOT / "config" / "forward_response_amplitude_pilot_v1.json"
DEFAULT_PILOT_GEOMETRY = (
    PROJECT_ROOT
    / "outputs"
    / "af_fno"
    / "response"
    / "forward_response_v1"
    / "amplitude_pilot_geometry_v1.json"
)
DEFAULT_DEVELOPMENT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "af_fno"
    / "response"
    / "forward_response_v1"
    / "forward_response_inventory_v1.jsonl"
)
DEFAULT_BLIND_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "af_fno"
    / "response"
    / "forward_response_blind_v1"
    / "forward_response_blind_inventory_v1.jsonl"
)

REGIMES = ("S0", "S1", "S2")
FAMILIES = ("U", "V", "Theta", "SSH")
REGIONS = ("WBC", "interior", "eastern", "northern", "southern")
REGION_PRECEDENCE = ("WBC", "eastern", "southern", "northern", "interior")
ROLE_ORDER = ("pilot", "train", "validation", "blind_test")
BASE_ITERATION = 2_592_000
STEPS_PER_DAY = 72
NR = 15
RADIUS = 2
EARTH_RADIUS_KM = 6371.0
PHASE_A_TARGET = tuple((j, 1) for j in range(14, 19))
TIE_GRAMMAR = "response-v1|split|regime|family|level-support|region|j|i"


class InventoryError(RuntimeError):
    """Base class for fail-closed inventory errors."""


class ContractError(InventoryError):
    """A frozen input contract is malformed or does not match its bytes."""


class SourceError(InventoryError):
    """A canonical source/grid fact cannot be established."""


class CapacityError(InventoryError):
    """The requested inventory cannot satisfy a hard capacity constraint."""


class UnderdeterminedError(InventoryError):
    """The contract does not uniquely determine a scientific inventory."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ContractError(f"non-finite JSON constant {value!r}")


def load_json_strict(path: str | Path) -> dict[str, Any]:
    """Load a JSON object while rejecting duplicate keys at every depth."""

    source = Path(path)
    try:
        value = json.loads(
            source.read_text(),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except OSError as error:
        raise ContractError(f"cannot read {source}: {error}") from error
    except json.JSONDecodeError as error:
        raise ContractError(f"cannot parse {source}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{source} must contain one JSON object")
    return value


FORWARD_RESPONSE_INVENTORY_PROGRESS = os.environ.get(
    "FORWARD_RESPONSE_INVENTORY_PROGRESS", "1"
) not in ("0", "", "false", "False")


def _progress(label: str, message: str) -> None:
    """Best-effort progress line to stdout, silent by default only if disabled.

    The centre-allocation MIP for a large region (e.g. "interior") can run
    for many minutes with no other output, which is a real operational gap
    for a background job with no other visibility -- verified directly: a
    materialize run gave zero signal for over an hour of legitimate,
    non-stuck computation. Worker processes spawned by ProcessPoolExecutor
    inherit the parent's stdout on the default fork start method, so these
    lines land in the same redirected log as the parent's own output; each
    is a single flushed write, so lines from different workers interleave
    but never truncate each other. Set
    FORWARD_RESPONSE_INVENTORY_PROGRESS=0 to silence.
    """

    if not FORWARD_RESPONSE_INVENTORY_PROGRESS:
        return
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] [pid {os.getpid()}] [{label}] {message}", file=sys.stdout, flush=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tie_sha(
    split: str,
    regime: str,
    family: str,
    level_support: str,
    region: str,
    j: int,
    i: int,
) -> str:
    """Return the exact section-9 SHA tie key."""

    payload = f"response-v1|{split}|{regime}|{family}|{level_support}|{region}|{j}|{i}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MDSMeta:
    dimensions: tuple[int, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    nrecords: int
    dtype: np.dtype
    fields: tuple[str, ...]
    timestep: int | None


def parse_mds_meta(path: str | Path) -> MDSMeta:
    """Parse a global MITgcm MDS pair without assuming a 62x62 grid."""

    metadata_path = Path(path)
    text = metadata_path.read_text()
    dimension_match = re.search(r"dimList\s*=\s*\[(.*?)\];", text, re.DOTALL)
    if not dimension_match:
        raise SourceError(f"missing dimList in {metadata_path}")
    values = [int(value) for value in re.findall(r"[-+]?\d+", dimension_match.group(1))]
    if not values or len(values) % 3:
        raise SourceError(f"invalid dimList in {metadata_path}")
    dimensions = tuple(values[index] for index in range(0, len(values), 3))
    starts = tuple(values[index] for index in range(1, len(values), 3))
    ends = tuple(values[index] for index in range(2, len(values), 3))
    if any(
        start != 1 or end != dimension for dimension, start, end in zip(dimensions, starts, ends)
    ):
        raise SourceError(f"tiled/non-global dimList is unsupported in {metadata_path}")

    record_match = re.search(r"nrecords\s*=\s*\[\s*(\d+)\s*\]", text)
    nrecords = int(record_match.group(1)) if record_match else 1
    precision_match = re.search(r"dataprec\s*=\s*\[\s*'([^']+)'", text)
    precision = precision_match.group(1).strip().lower() if precision_match else "float32"
    precision_map = {
        "float32": np.dtype(">f4"),
        "real*4": np.dtype(">f4"),
        "float64": np.dtype(">f8"),
        "real*8": np.dtype(">f8"),
    }
    if precision not in precision_map:
        raise SourceError(f"unsupported MDS precision {precision!r} in {metadata_path}")
    field_match = re.search(r"fldList\s*=\s*\{(.*?)\};", text, re.DOTALL)
    fields = (
        tuple(value.strip() for value in re.findall(r"'([^']+)'", field_match.group(1)))
        if field_match
        else ()
    )
    timestep_match = re.search(r"timeStepNumber\s*=\s*\[\s*(\d+)\s*\]", text)
    timestep = int(timestep_match.group(1)) if timestep_match else None
    return MDSMeta(
        dimensions=dimensions,
        starts=starts,
        ends=ends,
        nrecords=nrecords,
        dtype=precision_map[precision],
        fields=fields,
        timestep=timestep,
    )


def read_mds(path: str | Path) -> tuple[MDSMeta, np.ndarray]:
    """Read a global pair as ``(record, ..., y, x)`` in native precision."""

    metadata_path = Path(path)
    if metadata_path.suffix == ".data":
        metadata_path = metadata_path.with_suffix(".meta")
    meta = parse_mds_meta(metadata_path)
    count = meta.nrecords * math.prod(meta.dimensions)
    data_path = metadata_path.with_suffix(".data")
    values = np.fromfile(data_path, dtype=meta.dtype, count=count + 1)
    if values.size != count:
        raise SourceError(
            f"MDS size mismatch for {data_path}: expected {count} values, got {values.size}"
        )
    return meta, values.reshape((meta.nrecords, *reversed(meta.dimensions)))


def _read_one_grid_field(root: Path, name: str) -> np.ndarray:
    meta, value = read_mds(root / f"{name}.meta")
    if meta.nrecords != 1:
        raise SourceError(f"grid field {name} has {meta.nrecords} records, expected one")
    return np.asarray(value[0])


@dataclass(frozen=True)
class Grid:
    depth: np.ndarray
    hfac_c: np.ndarray
    hfac_w: np.ndarray
    hfac_s: np.ndarray
    xc: np.ndarray
    yc: np.ndarray
    xg: np.ndarray
    yg: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.depth.shape)  # type: ignore[return-value]

    @property
    def wet(self) -> np.ndarray:
        return np.asarray(self.depth > 0.0, dtype=bool)


def read_grid(root: str | Path) -> Grid:
    """Read the contract grid from generic MDS metadata, then cross-check shapes."""

    source = Path(root)
    grid = Grid(
        depth=_read_one_grid_field(source, "Depth"),
        hfac_c=_read_one_grid_field(source, "hFacC"),
        hfac_w=_read_one_grid_field(source, "hFacW"),
        hfac_s=_read_one_grid_field(source, "hFacS"),
        xc=_read_one_grid_field(source, "XC"),
        yc=_read_one_grid_field(source, "YC"),
        xg=_read_one_grid_field(source, "XG"),
        yg=_read_one_grid_field(source, "YG"),
    )
    if grid.depth.ndim != 2:
        raise SourceError(f"Depth must be two-dimensional, got {grid.depth.shape}")
    if any(value.shape != grid.depth.shape for value in (grid.xc, grid.yc, grid.xg, grid.yg)):
        raise SourceError("horizontal grid-coordinate shapes disagree")
    expected_3d = (NR, *grid.depth.shape)
    if any(value.shape != expected_3d for value in (grid.hfac_c, grid.hfac_w, grid.hfac_s)):
        raise SourceError(f"hFac shapes must all be {expected_3d}")
    if not np.array_equal(grid.wet, grid.hfac_c[0] > 0.0):
        raise SourceError("Depth>0 and surface hFacC>0 wet masks disagree")
    return grid


@dataclass(frozen=True)
class SourceCopy:
    segment: str
    meta_path: str
    data_path: str
    meta_sha256: str
    data_sha256: str


@dataclass(frozen=True)
class SourceResolution:
    regime: str
    day: int
    iteration: int
    canonical: SourceCopy
    candidates: tuple[SourceCopy, ...]
    canonical_choice_reason: str


def _verified_chain_roots(contract: Mapping[str, Any]) -> dict[str, tuple[Path, ...]]:
    sources = contract.get("sources", {})
    declared = sources.get("canonical_chain_roots")
    if not isinstance(declared, Mapping) or tuple(declared) != REGIMES:
        raise ContractError("sources.canonical_chain_roots must declare ordered S0/S1/S2 roots")
    manifest_path = Path(str(sources.get("trajectory_source_manifest", "")))
    expected_manifest_hash = str(sources.get("trajectory_source_manifest_sha256", ""))
    if sha256_file(manifest_path) != expected_manifest_hash:
        raise SourceError("trajectory-v3 source-manifest SHA-256 mismatch")
    manifest = load_json_strict(manifest_path)
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list):
        raise SourceError("trajectory-v3 manifest has no experiment inventory")
    manifest_roots: dict[str, tuple[Path, ...]] = {}
    for entry in experiments:
        if not isinstance(entry, Mapping):
            raise SourceError("malformed trajectory-v3 experiment entry")
        regime = str(entry.get("experiment"))
        segments = entry.get("segments")
        if regime not in REGIMES or not isinstance(segments, list):
            raise SourceError("malformed trajectory-v3 segment inventory")
        manifest_roots[regime] = tuple(Path(str(segment["run_dir"])) for segment in segments)
    roots = {regime: tuple(Path(str(path)) for path in declared[regime]) for regime in REGIMES}
    if roots != manifest_roots:
        raise SourceError(
            "canonical_chain_roots differ from the hash-pinned trajectory-v3 source manifest"
        )
    return roots


def resolve_annual_pickup(
    regime: str,
    day: int,
    chain_roots: Mapping[str, Sequence[Path]],
) -> SourceResolution:
    """Resolve one pickup only inside the canonical trajectory-v3 chain roots."""

    if regime not in REGIMES or regime not in chain_roots:
        raise SourceError(f"unknown regime {regime!r}")
    iteration = BASE_ITERATION + STEPS_PER_DAY * int(day)
    filename = f"pickup.{iteration:010d}"
    copies: list[SourceCopy] = []
    for root in chain_roots[regime]:
        meta_path = Path(root) / f"{filename}.meta"
        data_path = Path(root) / f"{filename}.data"
        meta_exists = meta_path.is_file()
        data_exists = data_path.is_file()
        if meta_exists != data_exists:
            raise SourceError(f"incomplete pickup pair at {root / filename}")
        if not meta_exists:
            continue
        meta = parse_mds_meta(meta_path)
        if meta.timestep != iteration:
            raise SourceError(f"pickup timestep mismatch at {meta_path}")
        if meta.dimensions != (62, 62) or meta.nrecords != 108 or meta.dtype != np.dtype(">f8"):
            raise SourceError(f"unexpected pickup layout at {meta_path}")
        copies.append(
            SourceCopy(
                segment=str(root),
                meta_path=str(meta_path),
                data_path=str(data_path),
                meta_sha256=sha256_file(meta_path),
                data_sha256=sha256_file(data_path),
            )
        )
    if not copies:
        raise SourceError(
            f"no complete {regime} day-{day} pickup in canonical trajectory-v3 chain roots"
        )
    if (
        len({copy.meta_sha256 for copy in copies}) != 1
        or len({copy.data_sha256 for copy in copies}) != 1
    ):
        raise SourceError(f"conflicting canonical-chain pickup copies for {regime} day {day}")
    # The source-manifest segment order is authoritative.  Never prefer a
    # downstream copy merely because a day lies on a segment boundary.
    canonical = copies[0]
    reason = (
        "only complete copy in canonical trajectory-v3 chain"
        if len(copies) == 1
        else "first source-manifest segment; all boundary-copy hashes agree"
    )
    return SourceResolution(regime, day, iteration, canonical, tuple(copies), reason)


def pickup_to_trajectory_p32(meta_path: str | Path, wet_mask: np.ndarray) -> np.ndarray:
    """Apply the trusted trajectory-v3 P32 projection to a complete pickup."""

    meta, records = read_mds(meta_path)
    if (
        meta.dimensions != (62, 62)
        or meta.nrecords != 108
        or meta.dtype != np.dtype(">f8")
        or records.shape != (108, 62, 62)
    ):
        raise SourceError(f"unexpected complete-pickup layout at {meta_path}")
    wet = np.asarray(wet_mask, dtype=bool)
    if wet.shape != (62, 62):
        raise SourceError(f"unexpected wet-mask shape {wet.shape}")
    # This is deliberately cast-before-centre, matching af_data._read_field_pair
    # and Gate D0.  The P64 response path is a different later operation.
    u = np.asarray(records[0:15], dtype=np.float32)
    v = np.asarray(records[15:30], dtype=np.float32)
    theta = np.asarray(records[30:45], dtype=np.float32)
    eta_n = np.asarray(records[105], dtype=np.float32)[None]
    u_center = 0.5 * (u + np.roll(u, -1, axis=-1))
    v_center = 0.5 * (v + np.roll(v, -1, axis=-2))
    state = np.concatenate((u_center, v_center, theta, eta_n), axis=0).astype(
        np.float32, copy=False
    )
    state[:, ~wet] = 0.0
    return state


def pickup_to_trajectory_p64(meta_path: str | Path, wet_mask: np.ndarray) -> np.ndarray:
    """The response path's own projection: identical face-to-centre averaging
    to :func:`pickup_to_trajectory_p32`, but never cast to float32 -- section
    10.2 requires MITgcm responses to be "differenced after float64 native-grid
    extraction and face-to-centre projection", and section 13's
    ``anchors/state_p64``/``response_p64`` arrays are declared ``<f8``. This is
    the "P64 response path" the docstring above once deferred to "a different
    later operation" (execution step 10).
    """

    meta, records = read_mds(meta_path)
    if (
        meta.dimensions != (62, 62)
        or meta.nrecords != 108
        or meta.dtype != np.dtype(">f8")
        or records.shape != (108, 62, 62)
    ):
        raise SourceError(f"unexpected complete-pickup layout at {meta_path}")
    wet = np.asarray(wet_mask, dtype=bool)
    if wet.shape != (62, 62):
        raise SourceError(f"unexpected wet-mask shape {wet.shape}")
    u = np.asarray(records[0:15], dtype=np.float64)
    v = np.asarray(records[15:30], dtype=np.float64)
    theta = np.asarray(records[30:45], dtype=np.float64)
    eta_n = np.asarray(records[105], dtype=np.float64)[None]
    u_center = 0.5 * (u + np.roll(u, -1, axis=-1))
    v_center = 0.5 * (v + np.roll(v, -1, axis=-2))
    state = np.concatenate((u_center, v_center, theta, eta_n), axis=0).astype(
        np.float64, copy=False
    )
    state[:, ~wet] = 0.0
    return state


def region_masks(wet_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Build the exact section-9 tracer masks in their frozen precedence."""

    wet = np.asarray(wet_mask, dtype=bool)
    if wet.ndim != 2 or not wet.any():
        raise ContractError("region construction needs a nonempty 2-D wet mask")
    result = {name: np.zeros_like(wet) for name in REGIONS}
    occupied = np.zeros_like(wet)
    for j in range(wet.shape[0]):
        columns = np.flatnonzero(wet[j])
        if columns.size:
            chosen = columns[:4]
            result["WBC"][j, chosen] = True
    occupied |= result["WBC"]
    for j in range(wet.shape[0]):
        columns = np.flatnonzero(wet[j])
        if columns.size:
            chosen = columns[-4:]
            result["eastern"][j, chosen] = True
    result["eastern"] &= ~occupied
    occupied |= result["eastern"]
    wet_rows = np.flatnonzero(wet.any(axis=1))
    for name, rows in (("southern", wet_rows[:10]), ("northern", wet_rows[-10:])):
        result[name][rows, :] = wet[rows, :] & ~occupied[rows, :]
        occupied |= result[name]
    result["interior"] = wet & ~occupied
    if not np.array_equal(np.logical_or.reduce(tuple(result.values())), wet):
        raise ContractError("region masks do not partition the wet tracer mask")
    if any(np.any(result[a] & result[b]) for a, b in itertools.combinations(REGIONS, 2)):
        raise ContractError("region masks overlap")
    return result


def _full_square_support(active: np.ndarray, radius: int = RADIUS) -> np.ndarray:
    active = np.asarray(active, dtype=bool)
    if active.ndim != 2:
        raise ValueError("support erosion requires a two-dimensional mask")
    result = np.zeros_like(active)
    size = 2 * radius + 1
    if min(active.shape) < size:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(active, (size, size))
    result[radius : active.shape[0] - radius, radius : active.shape[1] - radius] = windows.all(
        axis=(-2, -1)
    )
    return result


def projected_footprint(family: str, kernel: str, j: int, i: int) -> set[tuple[int, int]]:
    """Tracer-grid footprint seen by the FNO after native-grid projection."""

    if family == "SSH" and kernel == "point":
        return {(j, i)}
    if family == "U":
        return {(row, column) for row in range(j - 2, j + 3) for column in range(i - 3, i + 3)}
    if family == "V":
        return {(row, column) for row in range(j - 3, j + 3) for column in range(i - 2, i + 3)}
    if family in {"Theta", "SSH"}:
        return {(row, column) for row in range(j - 2, j + 3) for column in range(i - 2, i + 3)}
    raise ContractError(f"unknown family/kernel {family}/{kernel}")


def candidate_mask(
    grid: Grid,
    family: str,
    levels: Sequence[int],
    kernel: str,
    *,
    phase_a_target: Sequence[tuple[int, int]] = PHASE_A_TARGET,
) -> np.ndarray:
    """Return exact full-support carrier centres after Phase-A exclusion."""

    if family not in FAMILIES:
        raise ContractError(f"unknown family {family!r}")
    level_indices = tuple(int(level) - 1 for level in levels)
    if family != "SSH" and (
        not level_indices or min(level_indices) < 0 or max(level_indices) >= NR
    ):
        raise ContractError(f"invalid one-based levels {tuple(levels)} for {family}")
    if family == "SSH" and levels:
        raise ContractError("SSH directions must not declare vertical levels")
    if kernel not in {"point", "gaussian_5x5_sigma1"}:
        raise ContractError(f"unknown kernel {kernel!r}")
    if family != "SSH" and kernel != "gaussian_5x5_sigma1":
        raise ContractError(f"{family} requires the smooth 5x5 kernel")

    if family == "SSH" and kernel == "point":
        eligible = grid.wet.copy()
    else:
        active_3d = {"U": grid.hfac_w, "V": grid.hfac_s, "Theta": grid.hfac_c, "SSH": grid.hfac_c}[
            family
        ]
        requested = level_indices if level_indices else (0,)
        active = np.logical_and.reduce(tuple(active_3d[index] > 0.0 for index in requested))
        eligible = _full_square_support(active)
        # U/V region labels are supplied by the immediately-east/north tracer
        # carrier at the same native (j,i), active at every requested level.
        if family in {"U", "V"}:
            carrier = np.logical_and.reduce(tuple(grid.hfac_c[index] > 0.0 for index in requested))
            eligible &= carrier

    target = set((int(j), int(i)) for j, i in phase_a_target)
    for j, i in np.argwhere(eligible):
        if projected_footprint(family, kernel, int(j), int(i)) & target:
            eligible[j, i] = False
    return eligible


def family_coordinates(grid: Grid, family: str) -> tuple[np.ndarray, np.ndarray]:
    if family == "U":
        return grid.xg, grid.yc
    if family == "V":
        return grid.xc, grid.yg
    if family in {"Theta", "SSH"}:
        return grid.xc, grid.yc
    raise ContractError(f"unknown family {family!r}")


def level_support_name(levels: Sequence[int], weights: Sequence[float] = ()) -> str:
    """Serialize level support exactly as frozen in the corrected contract."""

    if len(levels) != len(weights):
        if levels or weights:
            raise ContractError("levels and vertical weights must have equal length")
    return canonical_json(
        {
            "levels": [int(level) for level in levels],
            "weights_hex": [float(weight).hex() for weight in weights],
        }
    )


@dataclass(frozen=True)
class Direction:
    role: str
    regime: str
    anchor_day: int
    anchor_slot: int
    direction_slot: int
    family: str
    kernel: str
    levels: tuple[int, ...]
    weights: tuple[float, ...]
    region: str | None = None
    j: int | None = None
    i: int | None = None
    lon: float | None = None
    lat: float | None = None
    long: bool = False

    @property
    def support_name(self) -> str:
        return level_support_name(self.levels, self.weights)

    @property
    def slot_id(self) -> str:
        return (
            f"response-v1:{self.role}:{self.regime}:d{self.anchor_day:04d}:"
            f"{self.family}:{self.kernel}:a{self.anchor_slot}:q{self.direction_slot}:"
            f"{self.support_name}"
        )


def _unit_weights(levels: Sequence[int]) -> tuple[float, ...]:
    if len(levels) == 1:
        return (1.0,)
    if len(levels) == 2:
        value = 1.0 / math.sqrt(2.0)
        return (value, value)
    if len(levels) == 3:
        edge = math.exp(-0.5)
        denominator = math.sqrt(1.0 + 2.0 * math.exp(-1.0))
        return (edge / denominator, 1.0 / denominator, edge / denominator)
    raise ContractError(f"unsupported vertical combination {tuple(levels)}")


VALIDATION_COMBINATIONS = ((1, 2), (7, 8, 9), (14, 15))
BLIND_COMBINATIONS = ((2, 3), (6, 7, 8), (13, 14))
VARIABLE_OFFSETS = {"U": 0, "V": 5, "Theta": 10}


def build_direction_slots(
    dataset_contract: Mapping[str, Any], pilot_contract: Mapping[str, Any]
) -> list[Direction]:
    """Build every non-spatial slot from the exact anchor/vertical formulas."""

    directions: list[Direction] = []
    pilot_cases = pilot_contract["anchors"]["ordered_cases"]
    pilot_regions = pilot_contract["anchors"]["region_sequence"]
    pilot_levels = pilot_contract["anchors"]["one_based_levels"]
    pilot_kernels = pilot_contract["anchors"]["ssh_kernel_by_day"]
    pilot_long = pilot_contract["long_assignment"]["by_regime_and_day"]
    for case_index, ((regime, day), region) in enumerate(zip(pilot_cases, pilot_regions)):
        anchor_slot = list(dataset_contract["roles"]["pilot"]["anchor_days_per_regime"]).index(day)
        for family_index, family in enumerate(FAMILIES):
            levels = () if family == "SSH" else (int(pilot_levels[family][case_index]),)
            kernel = str(pilot_kernels[str(day)]) if family == "SSH" else "gaussian_5x5_sigma1"
            directions.append(
                Direction(
                    role="pilot",
                    regime=str(regime),
                    anchor_day=int(day),
                    anchor_slot=anchor_slot,
                    direction_slot=family_index,
                    family=family,
                    kernel=kernel,
                    levels=levels,
                    weights=() if not levels else (1.0,),
                    region=str(region),
                    long=family in pilot_long[str(regime)][str(day)],
                )
            )

    train_days = tuple(
        int(day) for day in dataset_contract["roles"]["train"]["anchor_days_per_regime"]
    )
    for regime_index, regime in enumerate(REGIMES):
        for anchor_slot, day in enumerate(train_days):
            for family_index, family in enumerate(("U", "V", "Theta")):
                for direction_slot in range(4):
                    q = 4 * anchor_slot + direction_slot
                    level = 1 + ((q + 5 * regime_index + VARIABLE_OFFSETS[family]) % NR)
                    directions.append(
                        Direction(
                            role="train",
                            regime=regime,
                            anchor_day=day,
                            anchor_slot=anchor_slot,
                            direction_slot=4 * family_index + direction_slot,
                            family=family,
                            kernel="gaussian_5x5_sigma1",
                            levels=(level,),
                            weights=(1.0,),
                        )
                    )
            for kernel_index, kernel in enumerate(("point", "gaussian_5x5_sigma1")):
                for kernel_slot in range(2):
                    directions.append(
                        Direction(
                            role="train",
                            regime=regime,
                            anchor_day=day,
                            anchor_slot=anchor_slot,
                            direction_slot=12 + 2 * kernel_index + kernel_slot,
                            family="SSH",
                            kernel=kernel,
                            levels=(),
                            weights=(),
                        )
                    )

    for role, combinations, shift in (
        ("validation", VALIDATION_COMBINATIONS, 0),
        ("blind_test", BLIND_COMBINATIONS, 1),
    ):
        days = tuple(int(day) for day in dataset_contract["roles"][role]["anchor_days_per_regime"])
        for regime_index, regime in enumerate(REGIMES):
            per_anchor: dict[int, list[Direction]] = {index: [] for index in range(3)}
            for family in ("U", "V", "Theta"):
                offset = VARIABLE_OFFSETS[family]
                for level in range(1, NR + 1):
                    anchor_slot = ((level - 1) + 2 * regime_index + offset + shift) % 3
                    per_anchor[anchor_slot].append(
                        Direction(
                            role=role,
                            regime=regime,
                            anchor_day=days[anchor_slot],
                            anchor_slot=anchor_slot,
                            direction_slot=-1,
                            family=family,
                            kernel="gaussian_5x5_sigma1",
                            levels=(level,),
                            weights=(1.0,),
                        )
                    )
                for type_index, levels in enumerate(combinations):
                    anchor_slot = (type_index + regime_index + offset + shift) % 3
                    per_anchor[anchor_slot].append(
                        Direction(
                            role=role,
                            regime=regime,
                            anchor_day=days[anchor_slot],
                            anchor_slot=anchor_slot,
                            direction_slot=-1,
                            family=family,
                            kernel="gaussian_5x5_sigma1",
                            levels=tuple(levels),
                            weights=_unit_weights(levels),
                        )
                    )
            for anchor_slot, day in enumerate(days):
                for kernel in ("point", "gaussian_5x5_sigma1"):
                    for _ in range(3):
                        per_anchor[anchor_slot].append(
                            Direction(
                                role=role,
                                regime=regime,
                                anchor_day=day,
                                anchor_slot=anchor_slot,
                                direction_slot=-1,
                                family="SSH",
                                kernel=kernel,
                                levels=(),
                                weights=(),
                            )
                        )
                ordered = sorted(
                    per_anchor[anchor_slot],
                    key=lambda row: (
                        FAMILIES.index(row.family),
                        row.levels,
                        0 if row.kernel == "point" else 1,
                    ),
                )
                if len(ordered) != 24:
                    raise ContractError(
                        f"{role}/{regime}/anchor {anchor_slot} has {len(ordered)} slots"
                    )
                directions.extend(
                    replace(row, direction_slot=index) for index, row in enumerate(ordered)
                )
    return directions


def _region_quota(
    contract: Mapping[str, Any], role: str, family: str, kernel: str
) -> dict[str, int]:
    tables = contract["regional_quotas_per_regime"]
    if role == "train":
        key = (
            "train_U_V_Theta_each"
            if family != "SSH"
            else f"train_SSH_{kernel.replace('_5x5_sigma1', '')}_each"
        )
    elif role in {"validation", "blind_test"}:
        key = (
            "validation_blind_U_V_Theta_each"
            if family != "SSH"
            else f"validation_blind_SSH_{kernel.replace('_5x5_sigma1', '')}_each"
        )
    else:
        raise ContractError(f"pilot regions are fixed, not quota-derived ({family}/{kernel})")
    if key not in tables:
        raise ContractError(f"missing regional quota table {key}")
    quota = {str(region): int(value) for region, value in tables[key].items()}
    if set(quota) != set(REGIONS) or any(value < 0 for value in quota.values()):
        raise ContractError(f"malformed regional quota table {key}")
    return quota


def region_slot_preimage(row: Direction) -> str:
    return (
        "response-v1|region-slot|"
        f"{row.role}|{row.regime}|{row.family}|{row.kernel}|"
        f"{row.anchor_slot}|{row.direction_slot}|{row.support_name}"
    )


def region_slot_sha(row: Direction) -> str:
    return hashlib.sha256(region_slot_preimage(row).encode("utf-8")).hexdigest()


def assign_region_slots(rows: Sequence[Direction], contract: Mapping[str, Any]) -> list[Direction]:
    """Apply the frozen region-slot SHA/round-robin quota assignment."""

    assigned = [row for row in rows if row.role == "pilot"]
    if any(row.region not in REGIONS for row in assigned):
        raise ContractError("every pilot row must carry its exact declared region")
    production = [row for row in rows if row.role != "pilot"]
    strata: dict[tuple[str, str, str, str], list[Direction]] = defaultdict(list)
    for row in production:
        kernel_quota_family = row.kernel if row.family == "SSH" else "gaussian_5x5_sigma1"
        strata[(row.role, row.regime, row.family, kernel_quota_family)].append(row)
    for (role, _regime, family, kernel), members in sorted(strata.items()):
        quota = _region_quota(contract, role, family, kernel)
        if sum(quota.values()) != len(members):
            raise ContractError(
                f"{role}/{family}/{kernel} quotas sum to {sum(quota.values())}, "
                f"but there are {len(members)} slots"
            )
        ordered = sorted(members, key=lambda row: (region_slot_sha(row), row.slot_id))
        remaining = dict(quota)
        labels: list[str] = []
        while sum(remaining.values()):
            emitted = False
            for region in REGIONS:
                if remaining[region] > 0:
                    labels.append(region)
                    remaining[region] -= 1
                    emitted = True
            if not emitted:  # pragma: no cover - defensive invariant
                raise ContractError("region quota stream stalled")
        assigned.extend(replace(row, region=region) for row, region in zip(ordered, labels))
    return sorted(
        assigned,
        key=lambda row: (
            ROLE_ORDER.index(row.role),
            REGIMES.index(row.regime),
            row.anchor_day,
            row.direction_slot,
        ),
    )


def validate_direction_contract(
    rows: Sequence[Direction], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove all non-spatial count, formula, and chronology invariants."""

    expected_totals = {"pilot": 24, "train": 672, "validation": 216, "blind_test": 216}
    role_counts = Counter(row.role for row in rows)
    if role_counts != Counter(expected_totals):
        raise ContractError(f"direction totals differ: {dict(role_counts)} != {expected_totals}")
    for row in rows:
        if row.role not in ROLE_ORDER or row.regime not in REGIMES or row.family not in FAMILIES:
            raise ContractError(f"invalid direction identity {row.slot_id}")
        if row.region not in REGIONS:
            raise ContractError(f"unassigned/invalid region for {row.slot_id}")
        if row.family == "SSH":
            if row.levels or row.weights:
                raise ContractError(f"SSH row carries vertical support: {row.slot_id}")
        else:
            if len(row.levels) != len(row.weights) or not math.isclose(
                sum(weight * weight for weight in row.weights), 1.0, rel_tol=2e-15, abs_tol=2e-15
            ):
                raise ContractError(f"vertical weights are not unit L2: {row.slot_id}")
        # Exercise and thereby freeze token formatting on every row.
        json.loads(row.support_name)

    by_anchor = Counter((row.role, row.regime, row.anchor_day) for row in rows)
    expected_per_anchor = {"pilot": 4, "train": 16, "validation": 24, "blind_test": 24}
    if any(
        count != expected_per_anchor[role] for (role, _regime, _day), count in by_anchor.items()
    ):
        raise ContractError("directions-per-anchor count changed")

    for role in ("train", "validation", "blind_test"):
        for regime in REGIMES:
            for family in FAMILIES:
                family_rows = [
                    row
                    for row in rows
                    if row.role == role and row.regime == regime and row.family == family
                ]
                if family != "SSH":
                    singles = Counter(row.levels[0] for row in family_rows if len(row.levels) == 1)
                    if role == "train":
                        if set(singles) != set(range(1, 16)) or min(singles.values()) < 3:
                            raise ContractError(
                                f"training level coverage failed for {regime}/{family}"
                            )
                    elif singles != Counter({level: 1 for level in range(1, 16)}):
                        raise ContractError(
                            f"held-out single-level coverage failed for {role}/{regime}/{family}"
                        )
                quota_groups = (
                    [("gaussian_5x5_sigma1", family_rows)]
                    if family != "SSH"
                    else [
                        (kernel, [row for row in family_rows if row.kernel == kernel])
                        for kernel in ("point", "gaussian_5x5_sigma1")
                    ]
                )
                for kernel, group in quota_groups:
                    expected = _region_quota(contract, role, family, kernel)
                    actual = Counter(row.region for row in group)
                    if actual != Counter(expected):
                        raise ContractError(
                            f"region quota failed for {role}/{regime}/{family}/{kernel}: "
                            f"{dict(actual)} != {expected}"
                        )

    horizon = {"pilot": 90, "train": 60, "validation": 60, "blind_test": 90}
    for role in ("train", "validation", "blind_test"):
        lo, hi = (int(value) for value in contract["roles"][role]["chronology_half_open"])
        for row in rows:
            if row.role == role and not (
                lo <= row.anchor_day and row.anchor_day + horizon[role] < hi
            ):
                raise ContractError(f"chronology violation for {row.slot_id}")
    return {
        "role_counts": dict(sorted(role_counts.items())),
        "anchor_counts": {
            role: sum(1 for key in by_anchor if key[0] == role) for role in ROLE_ORDER
        },
        "level_support_tokens": len({row.support_name for row in rows}),
        "region_quotas_exact": True,
        "chronology_exact": True,
    }


@dataclass(frozen=True)
class Candidate:
    j: int
    i: int
    lon: float
    lat: float
    region: str
    subregion: str
    centroid_lon: float
    centroid_lat: float
    tertiary_distance_km: float
    tie_sha256: str
    wbc_speed: float = 0.0


def great_circle_km(
    lon_a: float | np.ndarray,
    lat_a: float | np.ndarray,
    lon_b: float | np.ndarray,
    lat_b: float | np.ndarray,
    radius_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """Stable vectorized great-circle distance on the frozen spherical Earth."""

    lon1 = np.deg2rad(np.asarray(lon_a, dtype=np.float64))
    lat1 = np.deg2rad(np.asarray(lat_a, dtype=np.float64))
    lon2 = np.deg2rad(np.asarray(lon_b, dtype=np.float64))
    lat2 = np.deg2rad(np.asarray(lat_b, dtype=np.float64))
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    haversine = (
        np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    return radius_km * 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def spherical_centroid(lon: Sequence[float], lat: Sequence[float]) -> tuple[float, float]:
    longitude = np.deg2rad(np.asarray(lon, dtype=np.float64))
    latitude = np.deg2rad(np.asarray(lat, dtype=np.float64))
    if not longitude.size or longitude.shape != latitude.shape:
        raise CapacityError("a subregion centroid needs nonempty paired coordinates")
    vectors = np.stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ),
        axis=1,
    )
    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise CapacityError("spherical subregion centroid is undefined")
    mean /= norm
    return float(np.rad2deg(np.arctan2(mean[1], mean[0]))), float(
        np.rad2deg(np.arctan2(mean[2], math.hypot(mean[0], mean[1])))
    )


def _candidate_subregions(
    coordinates: Sequence[tuple[int, int, float, float, str]],
    *,
    row: Direction,
) -> tuple[dict[tuple[int, int], str], dict[str, tuple[float, float]]]:
    """Apply frozen equal-rank quartiles/geographic quadrants and centroids."""

    if not coordinates or row.region not in REGIONS:
        raise CapacityError(f"no eligible candidates for {row.slot_id}/{row.region}")
    labels: dict[tuple[int, int], str] = {}
    region = str(row.region)
    if region in {"WBC", "eastern", "northern", "southern"}:
        coordinate_index = 3 if region in {"WBC", "eastern"} else 2
        ordered = sorted(
            coordinates,
            key=lambda item: (
                item[coordinate_index],
                tie_sha(
                    row.role, row.regime, row.family, row.support_name, region, item[0], item[1]
                ),
            ),
        )
        for rank, (j, i, _lon, _lat, _tie) in enumerate(ordered):
            quartile = min(3, (4 * rank) // len(ordered))
            labels[(j, i)] = f"q{quartile}"
    else:
        lon_values = [item[2] for item in coordinates]
        lat_values = [item[3] for item in coordinates]
        lon_midpoint = 0.5 * (min(lon_values) + max(lon_values))
        lat_midpoint = 0.5 * (min(lat_values) + max(lat_values))
        for j, i, lon, lat, _tie in coordinates:
            east = lon > lon_midpoint  # midpoint belongs west
            north = lat > lat_midpoint  # midpoint belongs south
            labels[(j, i)] = ("e" if east else "w") + ("n" if north else "s")
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for j, i, lon, lat, _tie in coordinates:
        grouped[labels[(j, i)]].append((lon, lat))
    expected = {"q0", "q1", "q2", "q3"} if region != "interior" else {"ws", "wn", "es", "en"}
    if set(grouped) != expected:
        raise CapacityError(
            f"{row.slot_id}/{region} has empty frozen subregion(s): {sorted(expected - set(grouped))}"
        )
    centroids = {
        name: spherical_centroid([value[0] for value in values], [value[1] for value in values])
        for name, values in grouped.items()
    }
    return labels, centroids


def enumerate_candidates(
    grid: Grid,
    row: Direction,
    masks: Mapping[str, np.ndarray],
    *,
    wbc_speed: np.ndarray | None = None,
) -> list[Candidate]:
    if row.region not in REGIONS:
        raise ContractError(f"row has no region: {row.slot_id}")
    eligible = candidate_mask(grid, row.family, row.levels, row.kernel) & masks[row.region]
    lon_grid, lat_grid = family_coordinates(grid, row.family)
    raw: list[tuple[int, int, float, float, str]] = []
    for j, i in np.argwhere(eligible):
        digest = tie_sha(
            row.role,
            row.regime,
            row.family,
            row.support_name,
            row.region,
            int(j),
            int(i),
        )
        raw.append((int(j), int(i), float(lon_grid[j, i]), float(lat_grid[j, i]), digest))
    labels, centroids = _candidate_subregions(raw, row=row)
    candidates: list[Candidate] = []
    all_centroids = tuple(centroids.values())
    for j, i, lon, lat, digest in raw:
        subregion = labels[(j, i)]
        own_lon, own_lat = centroids[subregion]
        nearest = min(
            float(great_circle_km(lon, lat, centre_lon, centre_lat))
            for centre_lon, centre_lat in all_centroids
        )
        candidates.append(
            Candidate(
                j=j,
                i=i,
                lon=lon,
                lat=lat,
                region=str(row.region),
                subregion=subregion,
                centroid_lon=own_lon,
                centroid_lat=own_lat,
                tertiary_distance_km=nearest,
                tie_sha256=digest,
                wbc_speed=0.0 if wbc_speed is None else float(wbc_speed[j, i]),
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.tie_sha256)


def mean_surface_speed_already_centered(
    state: Any,
    regime_index: int,
    *,
    start_day: int = 0,
    stop_day: int = 6000,
    chunk_days: int = 64,
) -> np.ndarray:
    """Mean ``hypot(U_01,V_01)`` with no second face-centering operation."""

    shape = tuple(int(value) for value in state.shape)
    if len(shape) != 5 or shape[2] < 16 or not (0 <= regime_index < shape[0]):
        raise SourceError(f"unexpected trajectory state shape {shape}")
    if not (0 <= start_day < stop_day <= shape[1]) or chunk_days <= 0:
        raise SourceError("invalid mean-speed chronology/chunk")
    total = np.zeros(shape[-2:], dtype=np.float64)
    count = 0
    for first in range(start_day, stop_day, chunk_days):
        last = min(stop_day, first + chunk_days)
        # Channels 0 and 15 are already face-to-centre projected in v3.  A
        # roll/average here would be an explicit contract violation.
        u = np.asarray(state[regime_index, first:last, 0], dtype=np.float64)
        v = np.asarray(state[regime_index, first:last, 15], dtype=np.float64)
        total += np.hypot(u, v).sum(axis=0, dtype=np.float64)
        count += last - first
    return total / float(count)


def _candidate_indices(
    grid: Grid, masks: Mapping[str, np.ndarray], row: Direction
) -> list[tuple[int, int, str]]:
    if row.region not in REGIONS:
        raise ContractError(f"unassigned region for {row.slot_id}")
    eligible = candidate_mask(grid, row.family, row.levels, row.kernel) & masks[row.region]
    return sorted(
        (
            int(j),
            int(i),
            tie_sha(
                row.role,
                row.regime,
                row.family,
                row.support_name,
                row.region,
                int(j),
                int(i),
            ),
        )
        for j, i in np.argwhere(eligible)
    )


def prove_hard_capacity(
    grid: Grid, rows: Sequence[Direction], masks: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Construct a private witness for counts/disjointness/distance capacity.

    The witness is intentionally not an inventory and is never serialized.
    It proves the hard constraints (counts, distinctness, the non-WBC
    distance-three rule) have capacity at all; materialization still runs
    the frozen deterministic farthest-point allocator
    (``allocate_centres_greedy_farthest_point``) to actually choose cells.
    """

    selected: dict[str, tuple[int, int]] = {}
    role_centres: dict[tuple[str, str], list[tuple[str, str, int, int]]] = defaultdict(list)
    lon_by_family = {family: family_coordinates(grid, family)[0] for family in FAMILIES}
    lat_by_family = {family: family_coordinates(grid, family)[1] for family in FAMILIES}
    for regime in REGIMES:
        for family in FAMILIES:
            family_rows = [row for row in rows if row.regime == regime and row.family == family]
            if not family_rows:
                continue
            cached: dict[tuple[str, str, str, str], list[Candidate]] = {}
            candidate_lists: dict[str, Sequence[Candidate]] = {}
            for row in family_rows:
                key = (row.role, row.kernel, row.support_name, str(row.region))
                if key not in cached:
                    candidates: list[Candidate] = []
                    for j, i, digest in _candidate_indices(grid, masks, row):
                        candidates.append(
                            Candidate(
                                j=j,
                                i=i,
                                lon=float(lon_by_family[family][j, i]),
                                lat=float(lat_by_family[family][j, i]),
                                region=str(row.region),
                                subregion="capacity-only",
                                centroid_lon=0.0,
                                centroid_lat=0.0,
                                tertiary_distance_km=0.0,
                                tie_sha256=digest,
                            )
                        )
                    cached[key] = candidates
                candidate_lists[row.slot_id] = cached[key]
            # At production scale a large region (e.g. "interior") gives some
            # rows thousands of eligible candidates; building this witness
            # MIP from the raw pool makes it too large for HiGHS to solve in
            # practice, the same scaling problem `allocate_centres_lexico-
            # graphically` hits and fixes with `_reduce_candidate_pool`. That
            # fix needed an objective-convergence check because it is
            # searching for an *optimum*; this witness only needs *any*
            # feasible assignment, and a solution built from a subset of the
            # true candidates is automatically feasible for the full
            # problem too -- so it is safe to just try increasing pool caps
            # and stop at the first one that solves, falling back to the
            # unreduced pool (cap=None) only if every reduction is
            # infeasible.
            result = None
            problem = None
            for cap in (*_POOL_CAP_LADDER, None):
                reduced_lists = (
                    candidate_lists
                    if cap is None
                    else {
                        slot_id: _reduce_candidate_pool(row_candidates, cap)
                        for slot_id, row_candidates in candidate_lists.items()
                    }
                )
                problem = _build_centre_problem(family_rows, reduced_lists)
                result = problem.model.solve()
                if result is not None:
                    break
            if result is None:
                raise CapacityError(f"hard-capacity MILP is infeasible for {regime}/{family}")
            rounded = np.rint(result.x)
            for row_index, variable, candidate in problem.x_variables:
                if rounded[variable] > 0.5:
                    row = problem.rows[row_index]
                    selected[row.slot_id] = (candidate.j, candidate.i)
                    role_centres[(regime, family)].append(
                        (row.role, candidate.region, candidate.j, candidate.i)
                    )

    if len(selected) != len(rows):
        raise CapacityError(f"capacity MILP selected {len(selected)} of {len(rows)} rows")
    cross_chebyshev: list[int] = []
    wbc_distinct = True
    for regime in REGIMES:
        for family in FAMILIES:
            entries = role_centres[(regime, family)]
            for left, right in itertools.combinations(entries, 2):
                left_role, left_region, left_j, left_i = left
                right_role, right_region, right_j, right_i = right
                if left_role == right_role:
                    continue
                if (left_j, left_i) == (right_j, right_i):
                    wbc_distinct = False
                # Distinct-ID uniqueness (above) is unconditional across all
                # roles, but the distance-three requirement itself only ever
                # binds when the LATER role (in ROLE_ORDER) is validation or
                # blind_test -- see the matching comment in
                # `_build_centre_problem`. Checking it against every
                # cross-role pair here (including pilot/train, which the
                # solver never constrained) would reject valid witnesses.
                later_role = max((left_role, right_role), key=ROLE_ORDER.index)
                if later_role not in ("validation", "blind_test"):
                    continue
                if left_region != "WBC" and right_region != "WBC":
                    cross_chebyshev.append(max(abs(left_j - right_j), abs(left_i - right_i)))
    if not wbc_distinct or (cross_chebyshev and min(cross_chebyshev) < 3):
        raise CapacityError("private witness does not satisfy cross-role hard constraints")
    return {
        "witness_rows": len(selected),
        "exact_rows_emitted": False,
        "centre_ids_distinct_across_roles": wbc_distinct,
        "non_wbc_cross_role_chebyshev_min": min(cross_chebyshev) if cross_chebyshev else None,
        "wbc_distance_three_exception_used": True,
    }


def _verify_plan_and_contract(contract: Mapping[str, Any]) -> dict[str, str]:
    if contract.get("version") != "forward_response_dataset_v3":
        raise ContractError(f"unexpected dataset contract version {contract.get('version')!r}")
    approved = contract.get("approved_plan", {})
    plan = PROJECT_ROOT / str(approved.get("path", ""))
    digest = sha256_file(plan)
    if digest != approved.get("sha256"):
        raise ContractError(f"approved-plan hash mismatch: {digest} != {approved.get('sha256')}")
    sources = contract.get("sources", {})
    store = Path(str(sources.get("trajectory_store", "")))
    metadata = store / ".zmetadata"
    metadata_digest = sha256_file(metadata)
    if metadata_digest != sources.get("trajectory_metadata_sha256"):
        raise SourceError("trajectory-v3 consolidated metadata hash mismatch")
    if sources.get("global_scratch_iteration_search_forbidden") is not True:
        raise ContractError("global scratch iteration search must remain forbidden")
    return {
        "approved_plan_sha256": digest,
        "trajectory_metadata_sha256": metadata_digest,
        "trajectory_source_manifest_sha256": str(sources["trajectory_source_manifest_sha256"]),
    }


def verify_grid_hashes(
    contract: Mapping[str, Any], roots: Mapping[str, Sequence[Path]]
) -> dict[str, Any]:
    grid_contract = contract["sources"]["grid"]
    canonical_root = Path(str(grid_contract["canonical_root"]))
    if canonical_root != roots["S0"][0]:
        raise SourceError("grid canonical root is not the first trajectory-v3 S0 chain root")
    files = grid_contract["files"]
    canonical_hashes: dict[str, tuple[str, str]] = {}
    for name, declared in files.items():
        meta = canonical_root / f"{name}.meta"
        data = canonical_root / f"{name}.data"
        observed = (sha256_file(meta), sha256_file(data))
        expected = (str(declared["meta_sha256"]), str(declared["data_sha256"]))
        if observed != expected:
            raise SourceError(f"canonical {name} grid hashes changed")
        canonical_hashes[str(name)] = observed
    segments_checked = 0
    for regime in REGIMES:
        for root in roots[regime]:
            segments_checked += 1
            for name, expected in canonical_hashes.items():
                observed = (sha256_file(root / f"{name}.meta"), sha256_file(root / f"{name}.data"))
                if observed != expected:
                    raise SourceError(f"grid {name} differs in canonical segment {root}")
    return {
        "canonical_root": str(canonical_root),
        "fields_checked": sorted(canonical_hashes),
        "canonical_segments_checked": segments_checked,
        "byte_identical_across_segments": True,
    }


def verify_trajectory_store(contract: Mapping[str, Any], grid: Grid) -> tuple[Any, dict[str, Any]]:
    try:
        import zarr
    except ImportError as error:  # pragma: no cover - project environment pins zarr
        raise SourceError("zarr is required to audit trajectories_v3") from error
    path = Path(str(contract["sources"]["trajectory_store"]))
    try:
        group = zarr.open_consolidated(str(path), mode="r")
    except Exception as error:
        raise SourceError(f"cannot open consolidated trajectory store {path}: {error}") from error
    state = group["state"]
    if tuple(state.shape) != (3, 9000, 46, 62, 62) or np.dtype(state.dtype) != np.dtype("float32"):
        raise SourceError(f"unexpected trajectory-v3 state contract {state.shape}/{state.dtype}")
    attrs = dict(group.attrs)
    expected_channels = tuple(
        [f"U_{level:02d}" for level in range(1, 16)]
        + [f"V_{level:02d}" for level in range(1, 16)]
        + [f"Theta_{level:02d}" for level in range(1, 16)]
        + ["Eta"]
    )
    if (
        tuple(attrs.get("experiments", ())) != REGIMES
        or tuple(attrs.get("state_channels", ())) != expected_channels
    ):
        raise SourceError("trajectory-v3 regime/channel ordering changed")
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    if not np.array_equal(wet, grid.wet):
        raise SourceError("trajectory-v3 wet mask differs from canonical MDS grid")
    iteration = np.asarray(group["iteration"][:])
    expected_iteration = BASE_ITERATION + STEPS_PER_DAY * np.arange(9000, dtype=np.int64)
    if iteration.shape != (3, 9000) or not all(
        np.array_equal(iteration[index], expected_iteration) for index in range(3)
    ):
        raise SourceError("trajectory-v3 day/iteration relation changed")
    return state, {
        "path": str(path),
        "state_shape": list(state.shape),
        "state_dtype": str(state.dtype),
        "wet_tracer_cells": int(wet.sum()),
        "state_channels_verified": True,
        "wbc_speed_channels": ["U_01", "V_01"],
        "wbc_speed_state_already_centered": True,
        "second_face_centering_permitted": False,
    }


def audit_pickup_sources(
    contract: Mapping[str, Any],
    roots: Mapping[str, Sequence[Path]],
    grid: Grid,
    trajectory_state: Any,
) -> dict[str, Any]:
    annual_days: set[int] = set()
    for role in ("pilot", "train", "blind_test"):
        annual_days.update(int(day) for day in contract["roles"][role]["anchor_days_per_regime"])
    annual_days.add(int(contract["validation_pickup_bank"]["source_day"]))
    resolutions = [
        resolve_annual_pickup(regime, day, roots)
        for regime in REGIMES
        for day in sorted(annual_days)
    ]
    projection_failures: list[str] = []
    for resolution in resolutions:
        projected = pickup_to_trajectory_p32(resolution.canonical.meta_path, grid.wet)
        truth = np.asarray(
            trajectory_state[REGIMES.index(resolution.regime), resolution.day],
            dtype=np.float32,
        )
        if not np.array_equal(projected, truth):
            projection_failures.append(f"{resolution.regime}/day{resolution.day}")
    if projection_failures:
        raise SourceError(
            "annual pickup P32 projections differ from trajectory-v3: "
            + ", ".join(projection_failures[:6])
        )
    duplicate_resolutions = sum(len(resolution.candidates) > 1 for resolution in resolutions)
    validation_days = tuple(
        int(day) for day in contract["roles"]["validation"]["anchor_days_per_regime"]
    )
    validation_present = 0
    # Off-cycle pickup-bank products are not searched globally.  They can only
    # become canonical through their future hash-pinned bank manifest; at this
    # no-compute step their absence is expected and explicitly reported.
    for regime in REGIMES:
        for day in validation_days:
            iteration = BASE_ITERATION + STEPS_PER_DAY * day
            if any((root / f"pickup.{iteration:010d}.meta").is_file() for root in roots[regime]):
                validation_present += 1
    return {
        "annual_anchor_resolutions": len(resolutions),
        "unique_annual_days_per_regime": len(annual_days),
        "boundary_duplicate_resolutions": duplicate_resolutions,
        "all_duplicate_hashes_agree": True,
        "annual_pickup_p32_projections_bit_identical": len(resolutions),
        "projection_cast_before_centering": True,
        "projection_land_reset_applied": True,
        "global_scratch_search_used": False,
        "validation_pickup_bank_expected_sources": 3 * len(validation_days),
        "validation_pickup_bank_sources_in_trajectory_chains": validation_present,
        "validation_pickup_bank_status": "pending_execution_step_6",
    }


def candidate_count_report(
    grid: Grid, rows: Sequence[Direction], masks: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    counts: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (row.family, row.kernel, row.support_name, str(row.region))
        if key in seen:
            continue
        seen.add(key)
        count = int(
            (
                candidate_mask(grid, row.family, row.levels, row.kernel) & masks[str(row.region)]
            ).sum()
        )
        if count <= 0:
            raise CapacityError(f"zero candidates for {key}")
        counts[(row.family, row.kernel, str(row.region))].add(count)
    return {
        f"{family}/{kernel}/{region}": {
            "minimum_across_level_supports": min(values),
            "maximum_across_level_supports": max(values),
        }
        for (family, kernel, region), values in sorted(counts.items())
    }


class _MixedIntegerModel:
    """Small sparse builder around SciPy/HiGHS with zero MIP gap."""

    def __init__(self, variable_count: int) -> None:
        self.variable_count = int(variable_count)
        self.variable_lower = [0.0] * self.variable_count
        self.variable_upper = [1.0] * self.variable_count
        self.integrality = [1] * self.variable_count
        self.row_indices: list[int] = []
        self.column_indices: list[int] = []
        self.coefficients: list[float] = []
        self.constraint_lower: list[float] = []
        self.constraint_upper: list[float] = []

    def copy(self) -> _MixedIntegerModel:
        other = _MixedIntegerModel(0)
        other.variable_count = self.variable_count
        other.variable_lower = self.variable_lower.copy()
        other.variable_upper = self.variable_upper.copy()
        other.integrality = self.integrality.copy()
        other.row_indices = self.row_indices.copy()
        other.column_indices = self.column_indices.copy()
        other.coefficients = self.coefficients.copy()
        other.constraint_lower = self.constraint_lower.copy()
        other.constraint_upper = self.constraint_upper.copy()
        return other

    def add_variables(
        self,
        count: int,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
        integral: bool = True,
    ) -> tuple[int, ...]:
        first = self.variable_count
        self.variable_count += int(count)
        self.variable_lower.extend([float(lower)] * int(count))
        self.variable_upper.extend([float(upper)] * int(count))
        self.integrality.extend([1 if integral else 0] * int(count))
        return tuple(range(first, self.variable_count))

    def add_constraint(
        self,
        terms: Mapping[int, float] | Iterable[tuple[int, float]],
        lower: float = -math.inf,
        upper: float = math.inf,
    ) -> None:
        row = len(self.constraint_lower)
        items = terms.items() if isinstance(terms, Mapping) else terms
        nonzero = 0
        for column, coefficient in items:
            if coefficient:
                self.row_indices.append(row)
                self.column_indices.append(int(column))
                self.coefficients.append(float(coefficient))
                nonzero += 1
        if not nonzero and not (lower <= 0.0 <= upper):
            raise CapacityError("an empty linear constraint is infeasible")
        self.constraint_lower.append(float(lower))
        self.constraint_upper.append(float(upper))

    def fix(self, variable: int, value: int | float) -> None:
        # Fixing by tightening the variable's own bounds is exactly equivalent
        # to the old `add_constraint({variable: 1.0}, value, value)` -- both
        # pin the same single value -- but does not grow the constraint
        # matrix, which matters here: the SHA-tie loop calls this up to once
        # per row of a group, and a per-region model is rebuilt/copied on
        # every such fix.
        self.variable_lower[int(variable)] = float(value)
        self.variable_upper[int(variable)] = float(value)

    def solve(self, objective: np.ndarray | None = None) -> Any | None:
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import coo_array
        except ImportError as error:  # pragma: no cover - pinned project dependency
            raise CapacityError(
                "SciPy MILP support is required for exact inventory solves"
            ) from error
        if objective is None:
            objective = np.zeros(self.variable_count, dtype=np.float64)
        objective = np.asarray(objective, dtype=np.float64)
        if objective.shape != (self.variable_count,):
            raise ValueError("MILP objective has the wrong shape")
        matrix = coo_array(
            (
                np.asarray(self.coefficients, dtype=np.float64),
                (
                    np.asarray(self.row_indices, dtype=np.int32),
                    np.asarray(self.column_indices, dtype=np.int32),
                ),
            ),
            shape=(len(self.constraint_lower), self.variable_count),
        ).tocsr()
        result = milp(
            c=objective,
            integrality=np.asarray(self.integrality, dtype=np.uint8),
            bounds=Bounds(
                np.asarray(self.variable_lower, dtype=np.float64),
                np.asarray(self.variable_upper, dtype=np.float64),
            ),
            constraints=LinearConstraint(
                matrix,
                np.asarray(self.constraint_lower, dtype=np.float64),
                np.asarray(self.constraint_upper, dtype=np.float64),
            ),
            options={"mip_rel_gap": 0.0, "presolve": True},
        )
        return result if result.success and result.x is not None else None


@dataclass(frozen=True)
class _RegionLongProblem:
    model: _MixedIntegerModel
    rows: tuple[Direction, ...]
    region_variables: Mapping[tuple[int, str], int]
    long_variables: Mapping[int, int]
    joint_variables: Mapping[tuple[int, str], int]


def _region_repair_sha(row: Direction, region: str) -> str:
    preimage = (
        "response-v1|region-repair|"
        f"{row.role}|{row.regime}|{row.family}|{row.kernel}|"
        f"{row.anchor_slot}|{row.direction_slot}|{row.support_name}|{region}"
    )
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def _region_long_components(rows: Sequence[Direction]) -> list[tuple[str, tuple[Direction, ...]]]:
    components: list[tuple[str, tuple[Direction, ...]]] = []
    for family in FAMILIES:
        members = tuple(row for row in rows if row.role == "train" and row.family == family)
        components.append((f"train/joint/{family}", members))
    for role in ("validation", "blind_test"):
        for family in FAMILIES:
            members = tuple(row for row in rows if row.role == role and row.family == family)
            components.append((f"{role}/joint/{family}", members))
    return components


def _build_region_long_problem(
    rows: Sequence[Direction], contract: Mapping[str, Any]
) -> _RegionLongProblem:
    members = tuple(rows)
    if not members:
        raise ContractError("region/long feasibility component is empty")
    role = members[0].role
    family = members[0].family
    if role not in {"train", "validation", "blind_test"} or any(
        row.role != role or row.family != family for row in members
    ):
        raise ContractError("region/long component mixes roles or families")
    count = len(members)
    region_variables = {
        (index, region): 5 * index + region_index
        for index in range(count)
        for region_index, region in enumerate(REGIONS)
    }
    long_offset = 5 * count
    long_variables = {index: long_offset + index for index in range(count)}
    joint_offset = 6 * count
    joint_variables = {
        (index, region): joint_offset + 5 * index + region_index
        for index in range(count)
        for region_index, region in enumerate(REGIONS)
    }
    model = _MixedIntegerModel(11 * count)
    for index in range(count):
        model.add_constraint(
            ((region_variables[(index, region)], 1.0) for region in REGIONS), 1.0, 1.0
        )
        long_variable = long_variables[index]
        for region in REGIONS:
            assigned = region_variables[(index, region)]
            joint = joint_variables[(index, region)]
            model.add_constraint({joint: 1.0, assigned: -1.0}, upper=0.0)
            model.add_constraint({joint: 1.0, long_variable: -1.0}, upper=0.0)
            model.add_constraint({joint: 1.0, assigned: -1.0, long_variable: -1.0}, lower=-1.0)

    quota_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(members):
        kernel = row.kernel if family == "SSH" else "gaussian_5x5_sigma1"
        quota_groups[(row.regime, kernel)].append(index)
    for (regime, kernel), indices in quota_groups.items():
        quota = _region_quota(contract, role, family, kernel)
        for region in REGIONS:
            model.add_constraint(
                ((region_variables[(index, region)], 1.0) for index in indices),
                float(quota[region]),
                float(quota[region]),
            )

    if role == "train":
        long_days = set(int(day) for day in contract["roles"]["train"]["long_anchor_days"])
        for index, row in enumerate(members):
            if row.anchor_day not in long_days:
                model.fix(long_variables[index], 0)
        for regime in REGIMES:
            regime_indices = [index for index, row in enumerate(members) if row.regime == regime]
            for day in sorted(long_days):
                indices = [index for index in regime_indices if members[index].anchor_day == day]
                model.add_constraint(((long_variables[index], 1.0) for index in indices), 1.0, 1.0)
            for region in REGIONS:
                terms = ((joint_variables[(index, region)], 1.0) for index in regime_indices)
                if region == "WBC":
                    model.add_constraint(terms, 2.0, 2.0)
                else:
                    model.add_constraint(terms, lower=1.0)
            if family == "SSH":
                for kernel in ("point", "gaussian_5x5_sigma1"):
                    indices = [index for index in regime_indices if members[index].kernel == kernel]
                    model.add_constraint(
                        ((long_variables[index], 1.0) for index in indices), 4.0, 4.0
                    )
            else:
                for band in ("upper", "middle", "deep"):
                    indices = [
                        index for index in regime_indices if _depth_band(members[index]) == band
                    ]
                    model.add_constraint(
                        ((long_variables[index], 1.0) for index in indices), lower=2.0
                    )
    else:
        shift = 0 if role == "validation" else 1
        combinations = VALIDATION_COMBINATIONS if role == "validation" else BLIND_COMBINATIONS
        ssh_sequence = (
            ("point", "gaussian_5x5_sigma1", "point")
            if role == "validation"
            else ("gaussian_5x5_sigma1", "point", "gaussian_5x5_sigma1")
        )
        for regime_index, regime in enumerate(REGIMES):
            regime_indices = [index for index, row in enumerate(members) if row.regime == regime]
            model.add_constraint(
                ((long_variables[index], 1.0) for index in regime_indices), 3.0, 3.0
            )
            for anchor_slot in range(3):
                indices = [
                    index
                    for index, row in enumerate(members)
                    if row.regime == regime and row.anchor_slot == anchor_slot
                ]
                model.add_constraint(((long_variables[index], 1.0) for index in indices), 1.0, 1.0)
            if family == "SSH":
                for anchor_slot, kernel in enumerate(ssh_sequence):
                    indices = [
                        index
                        for index, row in enumerate(members)
                        if row.regime == regime
                        and row.anchor_slot == anchor_slot
                        and row.kernel == kernel
                    ]
                    model.add_constraint(
                        ((long_variables[index], 1.0) for index in indices), 1.0, 1.0
                    )
            else:
                singles = [
                    index
                    for index, row in enumerate(members)
                    if row.regime == regime and len(row.levels) == 1
                ]
                multis = [
                    index
                    for index, row in enumerate(members)
                    if row.regime == regime and len(row.levels) > 1
                ]
                model.add_constraint(((long_variables[index], 1.0) for index in singles), 2.0, 2.0)
                model.add_constraint(((long_variables[index], 1.0) for index in multis), 1.0, 1.0)
                required_type = (regime_index + VARIABLE_OFFSETS[family] + shift) % 3
                required_levels = tuple(combinations[required_type])
                required = [
                    index
                    for index, row in enumerate(members)
                    if row.regime == regime and row.levels == required_levels
                ]
                model.add_constraint(((long_variables[index], 1.0) for index in required), 1.0, 1.0)
                for band in ("upper", "middle", "deep"):
                    indices = [
                        index
                        for index, row in enumerate(members)
                        if row.regime == regime and _depth_band(row) == band
                    ]
                    model.add_constraint(
                        ((long_variables[index], 1.0) for index in indices), lower=1.0
                    )
                for region in REGIONS:
                    model.add_constraint(
                        ((joint_variables[(index, region)], 1.0) for index in regime_indices),
                        upper=1.0,
                    )
        for region in REGIONS:
            model.add_constraint(
                ((joint_variables[(index, region)], 1.0) for index in range(count)),
                lower=1.0,
            )
    return _RegionLongProblem(
        model=model,
        rows=members,
        region_variables=region_variables,
        long_variables=long_variables,
        joint_variables=joint_variables,
    )


def _freeze_linear_optimum(
    model: _MixedIntegerModel, coefficients: np.ndarray, *, maximize: bool
) -> tuple[_MixedIntegerModel, float]:
    """Solve a linear objective over ``model``, then freeze its exact binary64 value.

    Used by ``_repair_one_region_long_component`` to minimize the Hamming
    distance of a region-label repair. Recomputes the optimum in fixed
    variable order and freezes it as an equality constraint rather than
    trusting an epsilon-weighted scalarization. Only genuinely *integer*
    solver variables are rounded to their clean 0/1 value here --
    blanket-rounding every variable would be wrong for any caller with a
    real-valued decision variable, corrupting the frozen value and making
    the re-solve below provably infeasible against the model's own true
    optimum.
    """

    objective = -coefficients if maximize else coefficients
    result = model.solve(objective)
    if result is None:
        raise CapacityError("linear lexicographic objective is infeasible")
    integrality = np.asarray(model.integrality, dtype=bool)
    solution = np.where(integrality, np.rint(result.x), result.x)
    value = float(np.dot(coefficients, solution))
    frozen = model.copy()
    frozen.add_constraint(
        (
            (index, float(coefficient))
            for index, coefficient in enumerate(coefficients)
            if coefficient
        ),
        value,
        value,
    )
    if frozen.solve() is None:
        raise CapacityError("HiGHS could not reproduce its frozen binary64 optimum")
    return frozen, value


def _repair_one_region_long_component(
    label: str, rows: Sequence[Direction], contract: Mapping[str, Any]
) -> tuple[list[Direction], dict[str, Any]]:
    problem = _build_region_long_problem(rows, contract)
    initial = problem.model.copy()
    for index, row in enumerate(problem.rows):
        if row.region not in REGIONS:
            raise ContractError(f"initial SHA region is missing for {row.slot_id}")
        initial.fix(problem.region_variables[(index, row.region)], 1)
    if initial.solve() is not None:
        return list(problem.rows), {
            "component": label,
            "initial_sha_zip_feasible": True,
            "minimum_hamming_changes": 0,
            "public_changes": [],
            "blind_changes_redacted": 0,
        }

    hamming = np.zeros(problem.model.variable_count, dtype=np.float64)
    for index, row in enumerate(problem.rows):
        for region in REGIONS:
            if region != row.region:
                hamming[problem.region_variables[(index, region)]] = 1.0
    minimized, minimum_float = _freeze_linear_optimum(problem.model, hamming, maximize=False)
    minimum = int(round(minimum_float))
    if not math.isclose(minimum_float, minimum, abs_tol=1e-9):
        raise CapacityError(f"nonintegral region-repair Hamming optimum {minimum_float}")

    slot_hashes = [region_slot_sha(row) for row in problem.rows]
    if len(set(slot_hashes)) != len(slot_hashes):
        raise ContractError(f"region-slot SHA collision in {label}")
    work = minimized
    ordered_indices = sorted(range(len(problem.rows)), key=lambda index: slot_hashes[index])
    for index in ordered_indices:
        row = problem.rows[index]
        ordered_regions = sorted(REGIONS, key=lambda region: _region_repair_sha(row, region))
        chosen: _MixedIntegerModel | None = None
        for region in ordered_regions:
            trial = work.copy()
            trial.fix(problem.region_variables[(index, region)], 1)
            if trial.solve() is not None:
                chosen = trial
                break
        if chosen is None:
            raise CapacityError(f"region-repair SHA tie became infeasible at {row.slot_id}")
        work = chosen
    result = work.solve()
    if result is None:
        raise CapacityError(f"fully frozen region repair is infeasible for {label}")
    repaired: list[Direction] = []
    public_changes: list[dict[str, Any]] = []
    blind_changes = 0
    for index, row in enumerate(problem.rows):
        labels = [
            region
            for region in REGIONS
            if result.x[problem.region_variables[(index, region)]] > 0.5
        ]
        if len(labels) != 1:
            raise CapacityError(f"region repair did not assign one label to {row.slot_id}")
        new_region = labels[0]
        repaired.append(replace(row, region=new_region))
        if new_region != row.region:
            if row.role == "blind_test":
                blind_changes += 1
            else:
                public_changes.append(
                    {
                        "anchor_day": row.anchor_day,
                        "anchor_index": row.anchor_slot,
                        "direction_slot": row.direction_slot,
                        "family": row.family,
                        "from_region": row.region,
                        "level_support_token": row.support_name,
                        "regime": row.regime,
                        "role": row.role,
                        "to_region": new_region,
                    }
                )
    if len(public_changes) + blind_changes != minimum:
        raise CapacityError(f"region repair changed the wrong number of labels in {label}")
    return repaired, {
        "component": label,
        "initial_sha_zip_feasible": False,
        "minimum_hamming_changes": minimum,
        "public_changes": public_changes,
        "blind_changes_redacted": blind_changes,
        "repair_tie_grammar": (
            "response-v1|region-repair|role|regime|family|kernel|anchor-index|"
            "direction-slot|level-support|region"
        ),
    }


def repair_region_slots_for_long_feasibility(
    rows: Sequence[Direction], contract: Mapping[str, Any]
) -> tuple[list[Direction], dict[str, Any]]:
    """Minimally repair SHA-zipped labels only when long constraints demand it."""

    by_slot = {row.slot_id: row for row in rows}
    if len(by_slot) != len(rows):
        raise ContractError("direction slot IDs are not unique")
    component_reports: list[dict[str, Any]] = []
    for label, members in _region_long_components(rows):
        repaired, report = _repair_one_region_long_component(label, members, contract)
        by_slot.update({row.slot_id: row for row in repaired})
        component_reports.append(report)
    result = [by_slot[row.slot_id] for row in rows]
    mapping_payload = [
        {"direction_id": row.slot_id, "region": by_slot[row.slot_id].region}
        for row in sorted(rows, key=lambda item: item.slot_id)
    ]
    changes = sum(int(report["minimum_hamming_changes"]) for report in component_reports)
    return result, {
        "initial_complete_long_feasible": all(
            bool(report["initial_sha_zip_feasible"]) for report in component_reports
        ),
        "minimum_hamming_changes": changes,
        "components_checked": len(component_reports),
        "failing_initial_components": [
            report["component"]
            for report in component_reports
            if not report["initial_sha_zip_feasible"]
        ],
        "component_reports": component_reports,
        "repaired_region_mapping_sha256": hashlib.sha256(
            canonical_json(mapping_payload).encode("utf-8")
        ).hexdigest(),
        "exact_blind_rows_emitted": False,
    }


@dataclass(frozen=True)
class _CentreProblem:
    model: _MixedIntegerModel
    rows: tuple[Direction, ...]
    x_variables: tuple[tuple[int, int, Candidate], ...]
    y_variables: Mapping[tuple[str, int, int], int]
    centres: Mapping[tuple[int, int], tuple[float, float, str]]


def _unit_sphere_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Exact unit-sphere Cartesian embedding, shape ``(N, 3)`` for ``N`` paired points.

    Chord length in this embedding is a strictly monotonic function of
    great-circle distance, so nearest-neighbour/farthest-point order here is
    identical to ``great_circle_km`` order -- used by
    ``_reduce_candidate_pool`` for its farthest-point-sampling reduction.
    """

    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    return np.stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1
    )


#: Candidate-pool-size escalation ladder used by ``prove_hard_capacity``'s
#: feasibility witness: a large region's raw candidate pool (thousands of
#: eligible cells) can make even a pure feasibility MILP too large to build
#: quickly, so that witness escalates through these caps and stops at the
#: first one that solves. Unlike an optimum search, a feasible witness found
#: under a reduced pool is automatically feasible for the full problem too,
#: so no convergence-matching check is needed here.
_POOL_CAP_LADDER = (150, 300, 600, 1200, 2400)


def _reduce_candidate_pool(candidates: Sequence[Candidate], cap: int) -> tuple[Candidate, ...]:
    """A spatially-diverse, scalar-objective-preserving subset of size <= cap.

    ``_build_centre_problem`` gives every row one MIP variable per eligible
    candidate; production regions like "interior" have thousands of eligible
    cells, which is what makes even a pure feasibility model too large for
    HiGHS to resolve repeatedly at cold start. Used by
    ``prove_hard_capacity``'s escalating-cap witness search
    (``_POOL_CAP_LADDER``): since that search only needs *any* feasible
    assignment, not an optimum, a witness found under a reduced pool is
    automatically valid for the unreduced problem too, so no convergence
    check against a larger cap is required there.

    The subset always keeps: the SHA-tie-break-first candidate (so the
    common case where the objective doesn't discriminate between many
    candidates still resolves to the same choice as the unreduced pool), the
    two scalar-objective extremes (max wbc_speed, min tertiary_distance_km),
    and then a greedy farthest-point-sampled spread of the rest -- maximizing
    the minimum pairwise separation representable in the kept subset is
    exactly what the leximax objective needs, so this is the reduction least
    likely to discard the true optimum, though it is a heuristic, not a
    proof.
    """

    if len(candidates) <= cap:
        return tuple(candidates)
    ordered = sorted(candidates, key=lambda c: c.tie_sha256)
    chosen: list[Candidate] = [ordered[0]]
    chosen_ids = {(ordered[0].j, ordered[0].i)}
    for extreme in (
        max(candidates, key=lambda c: c.wbc_speed),
        min(candidates, key=lambda c: c.tertiary_distance_km),
    ):
        if (extreme.j, extreme.i) not in chosen_ids:
            chosen.append(extreme)
            chosen_ids.add((extreme.j, extreme.i))
    remaining = [c for c in candidates if (c.j, c.i) not in chosen_ids]
    if not remaining or len(chosen) >= cap:
        return tuple(chosen[:cap])

    remaining_xyz = _unit_sphere_xyz(
        np.asarray([c.lon for c in remaining], dtype=np.float64),
        np.asarray([c.lat for c in remaining], dtype=np.float64),
    )
    chosen_xyz = _unit_sphere_xyz(
        np.asarray([c.lon for c in chosen], dtype=np.float64),
        np.asarray([c.lat for c in chosen], dtype=np.float64),
    )
    min_sq_dist = np.min(
        ((remaining_xyz[:, None, :] - chosen_xyz[None, :, :]) ** 2).sum(axis=2), axis=1
    )
    remaining = list(remaining)
    while len(chosen) < cap and remaining:
        farthest = int(np.argmax(min_sq_dist))
        winner = remaining.pop(farthest)
        chosen.append(winner)
        min_sq_dist = np.delete(min_sq_dist, farthest)
        remaining_xyz = np.delete(remaining_xyz, farthest, axis=0)
        if not remaining:
            break
        winner_xyz = _unit_sphere_xyz(
            np.asarray([winner.lon], dtype=np.float64), np.asarray([winner.lat], dtype=np.float64)
        )
        new_sq_dist = ((remaining_xyz - winner_xyz) ** 2).sum(axis=1)
        min_sq_dist = np.minimum(min_sq_dist, new_sq_dist)
    return tuple(chosen)


def _build_centre_problem(
    rows: Sequence[Direction],
    candidate_lists: Mapping[str, Sequence[Candidate]],
) -> _CentreProblem:
    ordered_rows = tuple(rows)
    x_specs: list[tuple[int, int, Candidate]] = []
    centres: dict[tuple[int, int], tuple[float, float, str]] = {}
    for row_index, row in enumerate(ordered_rows):
        candidates = candidate_lists[row.slot_id]
        if not candidates:
            raise CapacityError(f"no candidates for {row.slot_id}")
        for candidate in candidates:
            key = (candidate.j, candidate.i)
            value = (candidate.lon, candidate.lat, candidate.region)
            if key in centres and centres[key] != value:
                raise ContractError(f"candidate coordinate/region mismatch at {key}")
            centres[key] = value
            x_specs.append((row_index, -1, candidate))

    roles_by_centre: dict[tuple[str, int, int], None] = {}
    for row_index, _unused, candidate in x_specs:
        row = ordered_rows[row_index]
        roles_by_centre[(row.role, candidate.j, candidate.i)] = None
    x_count = len(x_specs)
    y_keys = tuple(
        sorted(roles_by_centre, key=lambda key: (ROLE_ORDER.index(key[0]), key[1], key[2]))
    )
    y_variables = {key: x_count + index for index, key in enumerate(y_keys)}
    model = _MixedIntegerModel(x_count + len(y_keys))

    x_specs = [
        (row_index, variable, candidate)
        for variable, (row_index, _v, candidate) in enumerate(x_specs)
    ]
    x_by_row: dict[int, list[int]] = defaultdict(list)
    x_by_role_centre: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for row_index, variable, candidate in x_specs:
        row = ordered_rows[row_index]
        x_by_row[row_index].append(variable)
        x_by_role_centre[(row.role, candidate.j, candidate.i)].append(variable)
    for row_index in range(len(ordered_rows)):
        model.add_constraint(((variable, 1.0) for variable in x_by_row[row_index]), 1.0, 1.0)
    for key, y_variable in y_variables.items():
        terms = [(variable, 1.0) for variable in x_by_role_centre[key]]
        terms.append((y_variable, -1.0))
        model.add_constraint(terms, 0.0, 0.0)
    y_by_centre: dict[tuple[int, int], list[int]] = defaultdict(list)
    for (_role, j, i), variable in y_variables.items():
        y_by_centre[(j, i)].append(variable)
    for variables in y_by_centre.values():
        # Centre IDs form sets within roles and are distinct across roles.
        model.add_constraint(((variable, 1.0) for variable in variables), upper=1.0)

    # Global cross-region distance-three coupling for every pair of non-WBC
    # centres in different roles. Section 9.3 step 4 names only "non-WBC
    # validation and blind centres" as needing distance >= 3 from every
    # centre assigned to an EARLIER role -- train is never the constrained
    # (subject) role, but it IS a valid earlier-role target: validation and
    # blind must still stay >= 3 from train's centres, exactly as they must
    # from pilot's. The only pair this rule never touches is (pilot, train),
    # since neither role is ever "validation or blind". `roles_present` is
    # already in ROLE_ORDER (itertools.combinations preserves input order),
    # so `right_role` is always the later of the pair, and skipping exactly
    # the pairs where the later role is "train" is equivalent to "apply only
    # when the later role is validation or blind_test" (train is the only
    # ROLE_ORDER member besides pilot that can never be the later element
    # paired with something that also isn't validation/blind_test). A pair
    # involving WBC is the other, separately-declared exception.
    roles_present = [role for role in ROLE_ORDER if any(key[0] == role for key in y_variables)]
    for left_role, right_role in itertools.combinations(roles_present, 2):
        if right_role == "train":
            continue
        left_keys = [key for key in y_variables if key[0] == left_role]
        for _role, left_j, left_i in left_keys:
            if centres[(left_j, left_i)][2] == "WBC":
                continue
            left_variable = y_variables[(left_role, left_j, left_i)]
            for right_j in range(left_j - 2, left_j + 3):
                for right_i in range(left_i - 2, left_i + 3):
                    right_key = (right_role, right_j, right_i)
                    if right_key not in y_variables or centres[(right_j, right_i)][2] == "WBC":
                        continue
                    right_variable = y_variables[right_key]
                    model.add_constraint({left_variable: 1.0, right_variable: 1.0}, upper=1.0)

    return _CentreProblem(
        model=model,
        rows=ordered_rows,
        x_variables=tuple(x_specs),
        y_variables=y_variables,
        centres=centres,
    )


def _brute_force_region_minima(rows: Sequence[Direction], *, within: bool) -> tuple[float, ...]:
    """Direct O(k^2) minimum pairwise separation over an already-placed set.

    ``k`` is the number of rows in one (regime, family, region) stratum
    (tens, never the raw candidate pool), so this is always cheap. Used to
    report the achieved cross-role (``within=False``) or within-role
    (``within=True``) separation after
    ``allocate_centres_greedy_farthest_point`` places its cells -- an
    observed value, not a claim that no denser packing exists.
    """

    by_region: dict[str, list[Direction]] = defaultdict(list)
    for row in rows:
        by_region[row.region].append(row)
    minima: list[float] = []
    for region, members in by_region.items():
        best = math.inf
        for left, right in itertools.combinations(members, 2):
            if (left.j, left.i) == (right.j, right.i):
                continue
            same_role = left.role == right.role
            if within and not same_role:
                continue
            if not within and same_role:
                continue
            distance = float(great_circle_km(left.lon, left.lat, right.lon, right.lat))
            best = min(best, distance)
        if math.isfinite(best):
            minima.append(best)
    return tuple(sorted(minima))


_MAX_CROSS_REGION_REPAIR_ATTEMPTS = 32


def _pick_farthest_candidate(
    eligible: Sequence[Candidate],
    role: str,
    placed_by_role: Mapping[str, Sequence[Candidate]],
) -> Candidate:
    """Four-level deterministic score used by ``allocate_centres_greedy_farthest_point``.

    Picks, among ``eligible``, the candidate maximizing (i) the minimum
    great-circle distance to every already-placed cell of a DIFFERENT role,
    then (ii) the minimum distance to already-placed cells of the SAME
    role, then (iii) the existing physical/geographic preference (WBC:
    higher training-chronology surface speed; elsewhere: smaller distance
    to the subregion centroid), then (iv) ascending candidate SHA-256. A
    level with nothing placed yet scores every candidate ``+inf`` (no
    preference), falling through to the next level -- this is what makes
    the very first placement in a stratum fall straight to (iii)/(iv).
    """

    cross_placed = [
        candidate
        for other_role, placed in placed_by_role.items()
        if other_role != role
        for candidate in placed
    ]
    same_placed = list(placed_by_role.get(role, ()))
    eligible_lon = np.asarray([candidate.lon for candidate in eligible], dtype=np.float64)
    eligible_lat = np.asarray([candidate.lat for candidate in eligible], dtype=np.float64)

    def _min_distance_to(placed: Sequence[Candidate]) -> np.ndarray:
        if not placed:
            return np.full(len(eligible), np.inf, dtype=np.float64)
        placed_lon = np.asarray([candidate.lon for candidate in placed], dtype=np.float64)
        placed_lat = np.asarray([candidate.lat for candidate in placed], dtype=np.float64)
        pairwise = great_circle_km(
            eligible_lon[:, None], eligible_lat[:, None], placed_lon[None, :], placed_lat[None, :]
        )
        return pairwise.min(axis=1)

    cross_score = _min_distance_to(cross_placed)
    same_score = _min_distance_to(same_placed)
    if eligible[0].region == "WBC":
        tertiary_score = np.asarray([candidate.wbc_speed for candidate in eligible], dtype=np.float64)
    else:
        tertiary_score = np.asarray(
            [-candidate.tertiary_distance_km for candidate in eligible], dtype=np.float64
        )

    best_index = 0
    best_key: tuple[float, float, float, str] | None = None
    for index, candidate in enumerate(eligible):
        key = (
            -float(cross_score[index]),
            -float(same_score[index]),
            -float(tertiary_score[index]),
            candidate.tie_sha256,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_index = index
    return eligible[best_index]


def allocate_centres_greedy_farthest_point(
    rows: Sequence[Direction],
    candidate_lists: Mapping[str, Sequence[Candidate]],
    *,
    label: str = "",
) -> tuple[list[Direction], dict[str, Any]]:
    """Deterministic farthest-point placement for one (regime,family,region) stratum.

    2026-08-26 amendment: replaces the exact MIP-based leximax search this
    module previously used (``allocate_centres_lexicographically``, now
    removed). That search remained impractically slow even after decomposing
    by region: S0/U/WBC -- 1,505 candidates, *smaller* than several strata
    that solved in minutes -- ran over 45 minutes without finishing its
    first sub-stage on a real production run. The exact search's cost comes
    specifically from *proving* it found the mathematically maximal
    worst-case separation (an NP-hard max-min dispersion search); nothing
    about the study's scientific validity depends on that proof, only on
    the achieved separation and the hard constraints actually holding (see
    the amended section 9.3).

    Every row in this stratum is processed in one fixed, deterministic order
    (role order, then the existing region-slot SHA grammar), and for each
    row the eligible candidate is chosen by ``_pick_farthest_candidate`` --
    see its docstring for the four-level score. This is a standard,
    analyzable greedy heuristic for max-min dispersion (farthest-point
    placement), not an ad hoc choice.

    Eligibility enforces the same hard constraints the exact solver did:
    a cell already assigned to any earlier row in this stratum (any role)
    is excluded (centre IDs are globally distinct within a region), and,
    for non-WBC validation/blind_test rows, any cell within native-index
    Chebyshev distance 3 of a cell already placed for an earlier role is
    excluded too (section 9.3 step 4's same-region half; the cross-region
    half is handled by the caller's repair loop, reused unchanged).

    Returns the same objective-dict shape ``_merge_region_objectives``
    already expects, computed by brute force
    (``_brute_force_region_minima``) on the small final selection --
    reporting what separation was achieved, not claiming it is optimal.
    """

    if (
        len({row.regime for row in rows}) != 1
        or len({row.family for row in rows}) != 1
        or len({row.region for row in rows}) != 1
    ):
        raise ContractError(
            "greedy centre placement must contain exactly one regime/family/region"
        )

    order = sorted(rows, key=lambda row: (ROLE_ORDER.index(row.role), region_slot_sha(row)))
    region = order[0].region
    taken: set[tuple[int, int]] = set()
    placed_by_role: dict[str, list[Candidate]] = defaultdict(list)
    mapped: list[Direction] = []
    speed_sum = 0.0
    negdist_sum = 0.0

    for row in order:
        pool = candidate_lists[row.slot_id]
        if not pool:
            raise CapacityError(f"no candidates for {row.slot_id}")
        eligible = [candidate for candidate in pool if (candidate.j, candidate.i) not in taken]
        if region != "WBC" and row.role in ("validation", "blind_test"):
            blocked: set[tuple[int, int]] = set()
            for earlier_role in ROLE_ORDER[: ROLE_ORDER.index(row.role)]:
                for placed in placed_by_role.get(earlier_role, ()):
                    for delta_j in range(-2, 3):
                        for delta_i in range(-2, 3):
                            blocked.add((placed.j + delta_j, placed.i + delta_i))
            eligible = [
                candidate for candidate in eligible if (candidate.j, candidate.i) not in blocked
            ]
        if not eligible:
            raise CapacityError(
                f"no eligible candidates remain for {row.slot_id} after "
                "distinctness/distance-three exclusion"
            )
        winner = _pick_farthest_candidate(eligible, row.role, placed_by_role)
        taken.add((winner.j, winner.i))
        placed_by_role[row.role].append(winner)
        mapped.append(replace(row, j=winner.j, i=winner.i, lon=winner.lon, lat=winner.lat))
        if region == "WBC":
            speed_sum += winner.wbc_speed
        else:
            negdist_sum += -winner.tertiary_distance_km

    _progress(label, f"placed {len(mapped)} rows in region {region}")
    return mapped, {
        "cross_role_region_minima_sorted_km": list(_brute_force_region_minima(mapped, within=False)),
        "within_role_region_minima_sorted_km": list(_brute_force_region_minima(mapped, within=True)),
        "wbc_training_chronology_speed_sum": speed_sum,
        "non_wbc_negative_centroid_distance_sums_sorted_km": (
            [] if region == "WBC" else [negdist_sum]
        ),
        "achieved_not_proven_optimal": True,
    }


def _merge_region_objectives(per_region: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Reassemble the group-level achieved-separation report from independent per-region solves.

    Each ``(regime,family,region)`` call to
    ``allocate_centres_greedy_farthest_point`` returns cross/within vectors
    of length <= 1 (its own region's achieved minimum) and a
    negative-distance-sum list of length <= 1 (empty for WBC, one scalar
    otherwise). Concatenating those single entries across all five regions
    and re-sorting ascending gives the group-level achieved vector in the
    same layout the exact solver used to freeze -- these are now measured
    outcomes of the greedy placement, not a proof of optimality. The WBC
    speed sum is the only region with a nonzero contribution (every other
    region's rows never touch WBC candidates), so summing across regions is
    exact, not an aggregation choice.
    """

    cross = sorted(
        value
        for objective in per_region.values()
        for value in objective["cross_role_region_minima_sorted_km"]
    )
    within = sorted(
        value
        for objective in per_region.values()
        for value in objective["within_role_region_minima_sorted_km"]
    )
    negdist = sorted(
        value
        for objective in per_region.values()
        for value in objective["non_wbc_negative_centroid_distance_sums_sorted_km"]
    )
    speed = sum(
        float(objective["wbc_training_chronology_speed_sum"]) for objective in per_region.values()
    )
    return {
        "cross_role_region_minima_sorted_km": cross,
        "within_role_region_minima_sorted_km": within,
        "wbc_training_chronology_speed_sum": speed,
        "non_wbc_negative_centroid_distance_sums_sorted_km": negdist,
        "achieved_not_proven_optimal": True,
        "solve_unit": "per_region_greedy_farthest_point_v3",
        "per_region_objectives": dict(per_region),
    }


def _non_wbc_chebyshev_violations(rows: Sequence[Direction]) -> list[Direction]:
    """Find every later-role row whose centre is < 3 native cells from an earlier-role centre.

    Mirrors ``_build_centre_problem``'s own cross-role coupling exactly: only
    non-WBC centres are checked, WBC is a declared distance exception, and a
    pair only binds when the LATER role (in ``ROLE_ORDER``) is
    ``validation`` or ``blind_test`` (train is never the constrained/subject
    role -- see the matching comment there). This is the one place a
    per-region decomposition of the objective can still disagree with a true
    joint solve: two regions never share an objective term, but a
    validation/blind centre in one region can still sit within Chebyshev
    distance 3 of an earlier-role centre in a *different* region at a shared
    boundary. Returns the offending later-role rows (deduplicated), which
    ``allocate_centres_lexicographically_by_region`` excludes and retries.
    """

    violations: dict[str, Direction] = {}
    non_wbc = [row for row in rows if row.region != "WBC" and row.j is not None]
    non_wbc.sort(key=lambda row: (row.region, row.j, row.i))
    by_role: dict[str, list[Direction]] = defaultdict(list)
    for row in non_wbc:
        by_role[row.role].append(row)
    for left_role, right_role in itertools.combinations(ROLE_ORDER, 2):
        later_role = right_role  # combinations() preserves ROLE_ORDER, so right is later
        if later_role not in ("validation", "blind_test"):
            continue
        for later_row in by_role.get(later_role, ()):
            for earlier_row in by_role.get(left_role, ()):
                if earlier_row.region == later_row.region:
                    continue  # same-region pairs are already enforced inside that region's solve
                if (
                    max(abs(later_row.j - earlier_row.j), abs(later_row.i - earlier_row.i)) < 3
                ):
                    violations[later_row.slot_id] = later_row
                    break
    return list(violations.values())


def allocate_centres_lexicographically_by_region(
    rows: Sequence[Direction],
    candidate_lists: Mapping[str, Sequence[Candidate]],
    *,
    label: str = "",
) -> tuple[list[Direction], dict[str, Any]]:
    """Place one frozen ``(regime,family)`` centre allocation, decomposed per region.

    Section 9.3 step 3 of the approved plan specifies allocating centres
    "jointly within every ``(regime,family,region)`` stratum" -- i.e. the
    four roles are placed *within* each region independently, not across
    all five regions of a ``(regime,family)`` group at once. Each region is
    handed to ``allocate_centres_greedy_farthest_point`` (see its docstring
    for the 2026-08-26 amendment: a deterministic farthest-point heuristic
    replacing the exact MIP-based leximax search this function used
    earlier the same day, which proved impractically slow even after this
    same per-region decomposition).

    The one genuine cross-stratum coupling is the hard non-WBC Chebyshev-3
    constraint, which the approved plan states without a same-region
    qualifier. This is handled by solve-then-verify: after every region
    places its cells independently, ``_non_wbc_chebyshev_violations`` checks
    the assembled selection for boundary conflicts; any offending later-role
    centre is excluded from its own region's candidate pool and that region
    alone is re-placed. Convergence is not assumed -- exhausting
    ``_MAX_CROSS_REGION_REPAIR_ATTEMPTS`` raises rather than emitting an
    unverified geometry, matching the existing region-slot repair's
    "stop before inventory materialization" convention.
    """

    if len({row.regime for row in rows}) != 1 or len({row.family for row in rows}) != 1:
        raise ContractError("centre solve must contain exactly one regime/family")
    excluded: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for attempt in range(_MAX_CROSS_REGION_REPAIR_ATTEMPTS):
        per_region_mapped: dict[str, list[Direction]] = {}
        per_region_objective: dict[str, dict[str, Any]] = {}
        for region in REGION_PRECEDENCE:
            region_rows = [row for row in rows if row.region == region]
            if not region_rows:
                continue
            region_candidates: dict[str, Sequence[Candidate]] = {}
            for row in region_rows:
                pool = candidate_lists[row.slot_id]
                banned = excluded.get(row.slot_id)
                if banned:
                    pool = tuple(c for c in pool if (c.j, c.i) not in banned)
                    if not pool:
                        raise CapacityError(
                            f"no candidates remain for {row.slot_id} after "
                            "cross-region distance-three repair"
                        )
                region_candidates[row.slot_id] = pool
            region_label = f"{label}/{region}" if label else region
            region_start = time.monotonic()
            mapped, objective = allocate_centres_greedy_farthest_point(
                region_rows, region_candidates, label=region_label
            )
            _progress(
                region_label, f"attempt={attempt} done in {time.monotonic() - region_start:.1f}s"
            )
            per_region_mapped[region] = mapped
            per_region_objective[region] = objective
        all_mapped = [row for group in per_region_mapped.values() for row in group]
        if len(all_mapped) != len(rows):
            raise CapacityError(
                f"region-decomposed solve mapped {len(all_mapped)} of {len(rows)} rows"
            )
        violations = _non_wbc_chebyshev_violations(all_mapped)
        if not violations:
            merged = _merge_region_objectives(per_region_objective)
            merged["cross_region_repair_attempts"] = attempt
            return all_mapped, merged
        _progress(label, f"attempt={attempt}: {len(violations)} cross-region violations, repairing")
        for row in violations:
            excluded[row.slot_id].add((row.j, row.i))
    raise CapacityError(
        f"cross-region distance-three repair did not converge in "
        f"{_MAX_CROSS_REGION_REPAIR_ATTEMPTS} attempts; stop before materialization"
    )


def load_pilot_fixed_centres(
    geometry_path: Path,
) -> dict[tuple[str, int, str], tuple[int, int]]:
    """The already-frozen 24 pilot centres, keyed by (regime, anchor_day, family).

    Pilot is solved and its response campaign already run (section 25 steps
    6-8) before train/validation/blind are materialized -- out of the
    "jointly" order section 9.3 step 3 literally describes, because pilot's
    own sub-objective is provably independent of the other roles (see
    ``build_amplitude_pilot.py``'s ``_select_pilot_centre``). Its 24 centres
    must therefore be supplied to the remaining joint solve as fixed data,
    not re-decided -- reopening them would either contradict the frozen
    amplitude-calibration campaign or require redoing it. Every pilot row's
    (regime, anchor_day, family) triple is unique (one row per case per
    family), so it is a complete key.
    """

    payload = load_json_strict(geometry_path)
    return {
        (row["regime"], int(row["anchor_day"]), row["family"]): (int(row["j"]), int(row["i"]))
        for row in payload["directions"]
    }


def apply_fixed_centres(
    rows: Sequence[Direction],
    candidate_lists: Mapping[str, Sequence[Candidate]],
    fixed_centres: Mapping[tuple[str, int, str], tuple[int, int]],
) -> dict[str, Sequence[Candidate]]:
    """Collapse fixed rows' candidate list to their single known choice.

    No change to any solver internals is needed: a row with exactly one
    candidate has no freedom left, so the existing "exactly one candidate
    per row" constraint pins it, and every other mechanism (region quotas,
    the section-9.3-step-4 distance-three exclusion, the cross/within-role
    objective) automatically accounts for its real, fixed position with
    zero new code. Raises if a row's declared fixed centre is not actually
    among its enumerated candidates (e.g. grid/mask drift since pilot was
    solved) rather than silently accepting an unverified position.
    """

    reduced = dict(candidate_lists)
    for row in rows:
        key = (row.regime, row.anchor_day, row.family)
        target = fixed_centres.get(key)
        if target is None or row.role != "pilot":
            continue
        matches = [c for c in candidate_lists[row.slot_id] if (c.j, c.i) == target]
        if len(matches) != 1:
            raise ContractError(
                f"frozen pilot centre {target} for {row.slot_id} is not among its "
                "current enumerated candidates"
            )
        reduced[row.slot_id] = (matches[0],)
    return reduced


def _row_centre_tie(row: Direction) -> str:
    if row.region not in REGIONS or row.j is None or row.i is None:
        raise ContractError(f"long-subset candidate lacks a frozen centre: {row.slot_id}")
    return tie_sha(
        row.role,
        row.regime,
        row.family,
        row.support_name,
        row.region,
        row.j,
        row.i,
    )


def _solve_maximin_subset(
    rows: Sequence[Direction], model: _MixedIntegerModel
) -> tuple[set[int], float]:
    if model.variable_count != len(rows):
        raise ValueError("long-subset model must have one variable per direction")
    pairs: list[tuple[int, int, float]] = []
    for left, right in itertools.combinations(range(len(rows)), 2):
        a = rows[left]
        b = rows[right]
        if a.lon is None or a.lat is None or b.lon is None or b.lat is None:
            raise ContractError("long-subset solve requires physical centre coordinates")
        pairs.append((left, right, float(great_circle_km(a.lon, a.lat, b.lon, b.lat))))
    thresholds = np.unique(np.asarray([pair[2] for pair in pairs], dtype=np.float64))
    if not thresholds.size:
        raise CapacityError("long-subset maximin solve has fewer than two candidates")
    low = 0
    high = int(thresholds.size) - 1
    best: tuple[int, _MixedIntegerModel] | None = None
    while low <= high:
        middle = (low + high) // 2
        threshold = float(thresholds[middle])
        trial = model.copy()
        for left, right, distance in pairs:
            if distance < threshold:
                trial.add_constraint({left: 1.0, right: 1.0}, upper=1.0)
        if trial.solve() is not None:
            best = (middle, trial)
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise CapacityError("long-subset constraints are infeasible")
    work = best[1]
    selected: set[int] = set()
    for index in sorted(
        range(len(rows)), key=lambda item: (_row_centre_tie(rows[item]), rows[item].slot_id)
    ):
        trial = work.copy()
        trial.fix(index, 1)
        result = trial.solve()
        if result is not None:
            work = trial
            selected.add(index)
        else:
            work.fix(index, 0)
    result = work.solve()
    if result is None:
        raise CapacityError("SHA-tied long-subset solve became infeasible")
    selected = {index for index, value in enumerate(result.x[: len(rows)]) if value > 0.5}
    return selected, float(thresholds[best[0]])


def _depth_band(row: Direction) -> str:
    if not row.levels:
        raise ContractError("SSH has no vertical depth band")
    if max(row.levels) <= 5:
        return "upper"
    if min(row.levels) >= 11:
        return "deep"
    if min(row.levels) >= 6 and max(row.levels) <= 10:
        return "middle"
    raise ContractError(f"vertical support crosses frozen depth bands: {row.support_name}")


def solve_long_membership(
    rows: Sequence[Direction], contract: Mapping[str, Any]
) -> tuple[list[Direction], dict[str, Any]]:
    """Solve every sparse-long subset before response magnitude can exist."""

    updated = [replace(row, long=False) if row.role != "pilot" else row for row in rows]
    row_index = {row.slot_id: index for index, row in enumerate(updated)}
    objective_reports: dict[str, float] = {}
    train_long_days = set(int(day) for day in contract["roles"]["train"]["long_anchor_days"])
    for regime in REGIMES:
        for family in FAMILIES:
            pool = [
                row
                for row in updated
                if row.role == "train"
                and row.regime == regime
                and row.family == family
                and row.anchor_day in train_long_days
            ]
            model = _MixedIntegerModel(len(pool))
            model.add_constraint(((index, 1.0) for index in range(len(pool))), 8.0, 8.0)
            for day in sorted(train_long_days):
                indices = [index for index, row in enumerate(pool) if row.anchor_day == day]
                model.add_constraint(((index, 1.0) for index in indices), 1.0, 1.0)
            for region in REGIONS:
                indices = [index for index, row in enumerate(pool) if row.region == region]
                if region == "WBC":
                    model.add_constraint(((index, 1.0) for index in indices), 2.0, 2.0)
                else:
                    model.add_constraint(((index, 1.0) for index in indices), lower=1.0)
            if family == "SSH":
                for kernel in ("point", "gaussian_5x5_sigma1"):
                    indices = [index for index, row in enumerate(pool) if row.kernel == kernel]
                    model.add_constraint(((index, 1.0) for index in indices), 4.0, 4.0)
            else:
                for band in ("upper", "middle", "deep"):
                    indices = [index for index, row in enumerate(pool) if _depth_band(row) == band]
                    model.add_constraint(((index, 1.0) for index in indices), lower=2.0)
            selected, distance = _solve_maximin_subset(pool, model)
            for index in selected:
                original_index = row_index[pool[index].slot_id]
                updated[original_index] = replace(updated[original_index], long=True)
            objective_reports[f"train/{regime}/{family}"] = distance

    for role, shift, ssh_sequence in (
        ("validation", 0, ("point", "gaussian_5x5_sigma1", "point")),
        ("blind_test", 1, ("gaussian_5x5_sigma1", "point", "gaussian_5x5_sigma1")),
    ):
        combinations = VALIDATION_COMBINATIONS if role == "validation" else BLIND_COMBINATIONS
        for family in FAMILIES:
            pool = [row for row in updated if row.role == role and row.family == family]
            model = _MixedIntegerModel(len(pool))
            model.add_constraint(((index, 1.0) for index in range(len(pool))), 9.0, 9.0)
            for regime_index, regime in enumerate(REGIMES):
                regime_indices = [index for index, row in enumerate(pool) if row.regime == regime]
                model.add_constraint(((index, 1.0) for index in regime_indices), 3.0, 3.0)
                for anchor_slot in range(3):
                    indices = [
                        index
                        for index, row in enumerate(pool)
                        if row.regime == regime and row.anchor_slot == anchor_slot
                    ]
                    model.add_constraint(((index, 1.0) for index in indices), 1.0, 1.0)
                if family == "SSH":
                    for anchor_slot, kernel in enumerate(ssh_sequence):
                        indices = [
                            index
                            for index, row in enumerate(pool)
                            if row.regime == regime
                            and row.anchor_slot == anchor_slot
                            and row.kernel == kernel
                        ]
                        model.add_constraint(((index, 1.0) for index in indices), 1.0, 1.0)
                else:
                    singles = [
                        index
                        for index, row in enumerate(pool)
                        if row.regime == regime and len(row.levels) == 1
                    ]
                    multis = [
                        index
                        for index, row in enumerate(pool)
                        if row.regime == regime and len(row.levels) > 1
                    ]
                    model.add_constraint(((index, 1.0) for index in singles), 2.0, 2.0)
                    model.add_constraint(((index, 1.0) for index in multis), 1.0, 1.0)
                    required_type = (regime_index + VARIABLE_OFFSETS[family] + shift) % 3
                    required_levels = tuple(combinations[required_type])
                    required = [
                        index
                        for index, row in enumerate(pool)
                        if row.regime == regime and row.levels == required_levels
                    ]
                    model.add_constraint(((index, 1.0) for index in required), 1.0, 1.0)
                    for band in ("upper", "middle", "deep"):
                        indices = [
                            index
                            for index, row in enumerate(pool)
                            if row.regime == regime and _depth_band(row) == band
                        ]
                        model.add_constraint(((index, 1.0) for index in indices), lower=1.0)
                    for region in REGIONS:
                        indices = [
                            index
                            for index, row in enumerate(pool)
                            if row.regime == regime and row.region == region
                        ]
                        model.add_constraint(((index, 1.0) for index in indices), upper=1.0)
            for region in REGIONS:
                indices = [index for index, row in enumerate(pool) if row.region == region]
                model.add_constraint(((index, 1.0) for index in indices), lower=1.0)
            selected, distance = _solve_maximin_subset(pool, model)
            for index in selected:
                original_index = row_index[pool[index].slot_id]
                updated[original_index] = replace(updated[original_index], long=True)
            objective_reports[f"{role}/joint/{family}"] = distance

    long_counts = Counter(row.role for row in updated if row.long)
    expected = Counter({"pilot": 12, "train": 96, "validation": 36, "blind_test": 36})
    if long_counts != expected:
        raise CapacityError(f"long counts differ: {dict(long_counts)} != {dict(expected)}")
    return updated, {
        "long_counts": dict(sorted(long_counts.items())),
        "maximin_km": dict(sorted(objective_reports.items())),
        "magnitude_used": False,
        "candidate_sha_tie_applied": True,
    }


def validate_frozen_algorithm_contract(contract: Mapping[str, Any]) -> None:
    """Reject a nearby-but-different allocator before using any source data."""

    if tuple(contract["regions"]["precedence"]) != REGION_PRECEDENCE:
        raise ContractError("region precedence changed")
    allocator = contract["joint_spatial_allocator"]
    if tuple(allocator["role_order"]) != ROLE_ORDER:
        raise ContractError("allocator role order changed")
    if allocator.get("tie_hash_grammar") != TIE_GRAMMAR:
        raise ContractError("candidate tie grammar changed")
    if int(allocator.get("non_wbc_cross_role_chebyshev_min", -1)) != 3:
        raise ContractError("non-WBC cross-role distance changed")
    if allocator.get("wbc_hard_distance_exception") is not True:
        raise ContractError("WBC hard-distance exception changed")
    declared_target = tuple(
        (int(value[0]), int(value[1]))
        for value in allocator["phase_a_exclusion_zero_based"]["tracer_cells"]
    )
    if declared_target != PHASE_A_TARGET:
        raise ContractError("Phase-A exclusion stencil changed")
    scope = allocator.get("joint_objective_scope", {})
    if "one (regime,family,region)" not in str(scope.get("solve_unit", "")):
        raise ContractError("joint objective solve unit is not frozen")
    if tuple(scope.get("distance_vector_region_order_before_sorting", ())) != REGIONS:
        raise ContractError("joint objective region vector changed")
    expected_objective_method = (
        "deterministic_farthest_point_greedy_v3: process every row in one fixed order "
        "(role order, then region-slot SHA); for each row pick the eligible candidate "
        "maximizing, in order, (i) minimum distance to already-placed cells of a "
        "different role, (ii) minimum distance to already-placed cells of the same "
        "role, (iii) the physical/geographic tertiary preference, (iv) ascending "
        "candidate SHA-256. Achieved separation is reported, not proven optimal."
    )
    if scope.get("objective_method") != expected_objective_method:
        raise ContractError("centre-placement objective method is not the frozen greedy allocator")
    expected_ssh_ownership = (
        "after the complete physical objective vector ties, order SSH (slot,candidate) "
        "ownership variables by the ascending tuple (candidate tie SHA-256, region-slot "
        "SHA-256) and lexicographically maximize their binary selection vector (1 before "
        "0); this assigns point-versus-gaussian ownership without changing the candidate "
        "grammar or selected global centre set"
    )
    if scope.get("ssh_cross_kernel_ownership_tie_break") != expected_ssh_ownership:
        raise ContractError("SSH cross-kernel ownership tie is absent or changed")
    if "post_selection_row_mapping" not in allocator or "region_slot_assignment" not in allocator:
        raise ContractError("region-slot/post-selection mapping conventions are absent")
    repair = allocator["region_slot_assignment"].get("repair_if_long_membership_infeasible", {})
    expected_repair_grammar = (
        "response-v1|region-repair|role|regime|family|kernel|anchor-index|"
        "direction-slot|level-support|region"
    )
    if (
        repair.get("repair_hash_grammar") != expected_repair_grammar
        or repair.get("zero_hamming_assignment_must_be_retained_when_feasible") is not True
        or repair.get("quota_preservation_required") is not True
        or repair.get("centre_coordinates_response_values_and_model_values_forbidden") is not True
    ):
        raise ContractError("minimum-Hamming region-repair convention is absent or changed")
    serialization = contract["vertical_support"]["canonical_serialization"]
    expected_tokens = {
        "single_level_weight_hex": float(1.0).hex(),
        "two_level_weight_hex": (1.0 / math.sqrt(2.0)).hex(),
        "three_level_outer_weight_hex": (
            math.exp(-0.5) / math.sqrt(1.0 + 2.0 * math.exp(-1.0))
        ).hex(),
        "three_level_centre_weight_hex": (1.0 / math.sqrt(1.0 + 2.0 * math.exp(-1.0))).hex(),
        "ssh_token": level_support_name((), ()),
    }
    for key, expected in expected_tokens.items():
        if serialization.get(key) != expected:
            raise ContractError(f"vertical-support serialization {key} changed")
    if contract.get("selected_amplitudes") is not None:
        raise ContractError("Step-4 inventory must precede numeric amplitude selection")
    gates = contract.get("generation_gates", {})
    blind_requirements = gates.get("blind_test", {}).get("requires", [])
    if (
        "manual_confirmation_blind_data_not_yet_generated" not in blind_requirements
        or gates.get("adjoint_or_taf_generation") != "forbidden_for_all_response_dataset_phases"
    ):
        raise ContractError("numeric/adjoint generation gates changed")


def _support_counts(row: Direction) -> tuple[int, int]:
    vertical = max(1, len(row.levels))
    if row.family == "SSH" and row.kernel == "point":
        return 1, 1
    native = 25 * vertical
    centred = (30 if row.family in {"U", "V"} else 25) * vertical
    return native, centred


def inventory_row(row: Direction) -> dict[str, Any]:
    """Canonical geometry-only JSON row; it cannot carry response numerics."""

    if row.j is None or row.i is None or row.lon is None or row.lat is None or row.region is None:
        raise ContractError(f"cannot serialize an unmaterialized row: {row.slot_id}")
    native_count, centred_count = _support_counts(row)
    horizon = 90 if row.role in {"pilot", "blind_test"} and row.long else 60 if row.long else 10
    return {
        "anchor_day": row.anchor_day,
        "anchor_index": row.anchor_slot,
        "centre": {"i": row.i, "j": row.j, "lat": row.lat, "lon": row.lon},
        "centred_support_count": centred_count,
        "direction_id": row.slot_id,
        "direction_slot": row.direction_slot,
        "family": row.family,
        "horizon_days": horizon,
        "inventory_version": "forward_response_inventory_v1",
        "iteration": BASE_ITERATION + STEPS_PER_DAY * row.anchor_day,
        "kernel": row.kernel,
        "level_support_token": row.support_name,
        "levels_one_based": list(row.levels),
        "long": row.long,
        "native_grid": {"U": "W", "V": "S", "Theta": "C", "SSH": "C"}[row.family],
        "native_support_count": native_count,
        "numeric_response_present": False,
        "phase_a_exclusion_applied": True,
        "regime": row.regime,
        "region": row.region,
        "role": row.role,
        "vertical_weights_float64_hex": [float(weight).hex() for weight in row.weights],
    }


def _reject_symlinked_output_path(path: Path) -> None:
    """Refuse to materialize through a symlinked directory or file.

    Kept from the retired OS-identity firewall design as the one part of it
    that is not about proving process identity: a write-once manifest is
    only actually write-once if its route to disk cannot be redirected.
    """

    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise InventoryError(f"output route contains symlink {parent}")
        if parent == parent.parent:
            break


def _write_jsonl_exclusive(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    final_mode: int,
) -> str:
    """Create and fsync one canonical, write-once geometry manifest."""

    if not path.parent.is_dir():
        raise InventoryError(f"output parent does not exist: {path.parent}")
    if final_mode not in {0o400, 0o444}:
        raise InventoryError("sealed inventory mode must be 0400 or 0444")
    payload = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), final_mode)
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _prepare_inventory_context(
    dataset_contract_path: Path, pilot_contract_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, tuple[Path, ...]],
    Grid,
    Any,
    list[Direction],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    contract = load_json_strict(dataset_contract_path)
    pilot = load_json_strict(pilot_contract_path)
    validate_frozen_algorithm_contract(contract)
    hashes = _verify_plan_and_contract(contract)
    roots = _verified_chain_roots(contract)
    grid_hash_report = verify_grid_hashes(contract, roots)
    grid = read_grid(contract["sources"]["grid"]["canonical_root"])
    state, trajectory_report = verify_trajectory_store(contract, grid)
    masks = region_masks(grid.wet)
    base_rows = assign_region_slots(build_direction_slots(contract, pilot), contract)
    rows, region_repair_report = repair_region_slots_for_long_feasibility(base_rows, contract)
    direction_report = validate_direction_contract(rows, contract)
    context_report = {
        "contract_hashes": hashes,
        "grid": grid_hash_report,
        "trajectory": trajectory_report,
        "directions": direction_report,
        "region_long_feasibility": region_repair_report,
    }
    return contract, pilot, roots, grid, state, rows, masks, context_report


def run_audit(
    dataset_contract_path: Path = DEFAULT_DATASET_CONTRACT,
    pilot_contract_path: Path = DEFAULT_PILOT_CONTRACT,
) -> dict[str, Any]:
    """Return only aggregate, non-secret Step-4 evidence."""

    contract, _pilot, roots, grid, _state, rows, masks, context = _prepare_inventory_context(
        dataset_contract_path, pilot_contract_path
    )
    region_count = {name: int(mask.sum()) for name, mask in region_masks(grid.wet).items()}
    report = {
        "mode": "audit",
        "version": "forward_response_inventory_audit_v1",
        "pass": True,
        "exact_direction_rows_emitted": False,
        "blind_centres_emitted": False,
        "numeric_data_created": False,
        **context,
        "sources": audit_pickup_sources(contract, roots, grid, _state),
        "region_raw_counts": region_count,
        "active_native_faces_per_level": {
            "C": [int(value) for value in (grid.hfac_c > 0.0).sum(axis=(1, 2))],
            "W": [int(value) for value in (grid.hfac_w > 0.0).sum(axis=(1, 2))],
            "S": [int(value) for value in (grid.hfac_s > 0.0).sum(axis=(1, 2))],
        },
        "candidate_capacity": candidate_count_report(grid, rows, masks),
        "hard_capacity": prove_hard_capacity(grid, rows, masks),
        "phase_a_exclusion": {
            "tracer_cells_zero_based": [list(value) for value in PHASE_A_TARGET],
            "projected_U_V_Theta_footprints_excluded": True,
            "SSH_centres_and_supports_excluded": True,
        },
        "materialization": {
            "attempted": False,
            "blind_output_visibility": "evaluator_only",
            "blind_seal_mechanism": "write_once_exclusive_create_mode_0400_separate_path",
        },
    }
    return report


def _solve_one_group(
    regime: str,
    family: str,
    rows: Sequence[Direction],
    grid: Grid,
    masks: Mapping[str, np.ndarray],
    wbc_speed: np.ndarray,
    fixed_centres: Mapping[tuple[str, int, str], tuple[int, int]],
) -> tuple[str, list[Direction], dict[str, Any]]:
    """One ``(regime,family)`` group's full region-decomposed centre solve.

    A standalone, picklable unit of work so ``materialize_inventory`` can run
    every one of the twelve ``(regime,family)`` groups in a separate process:
    each group only ever enumerates candidates and solves
    ``allocate_centres_lexicographically_by_region`` over its own rows, with
    no state shared across groups, so there is nothing to protect against
    concurrent access.
    """

    label = f"{regime}/{family}"
    solve_rows = [row for row in rows if row.regime == regime and row.family == family]
    cache: dict[tuple[str, str, str, str], list[Candidate]] = {}
    candidates: dict[str, Sequence[Candidate]] = {}
    for row in solve_rows:
        key = (row.role, row.kernel, row.support_name, str(row.region))
        if key not in cache:
            cache[key] = enumerate_candidates(grid, row, masks, wbc_speed=wbc_speed)
        candidates[row.slot_id] = cache[key]
    candidates = apply_fixed_centres(solve_rows, candidates, fixed_centres)
    _progress(label, f"worker started, {len(solve_rows)} rows")
    group_start = time.monotonic()
    solved, objective = allocate_centres_lexicographically_by_region(
        solve_rows, candidates, label=label
    )
    _progress(label, f"worker finished in {time.monotonic() - group_start:.1f}s")
    return label, solved, objective


def materialize_inventory(
    *,
    dataset_contract_path: Path = DEFAULT_DATASET_CONTRACT,
    pilot_contract_path: Path = DEFAULT_PILOT_CONTRACT,
    pilot_geometry_path: Path = DEFAULT_PILOT_GEOMETRY,
    development_output: Path = DEFAULT_DEVELOPMENT_OUTPUT,
    blind_output: Path = DEFAULT_BLIND_OUTPUT,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Materialize public (pilot+train+validation) and blind geometry.

    Blind isolation is the write-once convention described in the roadmap's
    2026-08-24 amendment: the two manifests are exclusively created (O_EXCL)
    at distinct paths with distinct file modes (blind 0400, public 0444), and
    materialization refuses outright if either already exists. There is no
    live process-identity check -- see the module docstring.

    The twelve ``(regime,family)`` groups are solved in parallel worker
    processes (``_solve_one_group``, one process per group up to
    ``max_workers``, default the number of groups): each group only ever
    touches its own rows/candidates, so there is no shared state to
    serialize around. This is a wall-clock optimization only -- every
    group's result is exactly what the same call would have produced run
    serially.
    """

    if development_output.absolute() == blind_output.absolute():
        raise InventoryError("public and blind manifests must be separate files")
    if development_output.exists() or blind_output.exists():
        raise InventoryError("write-once inventory output already exists")
    _reject_symlinked_output_path(development_output)
    _reject_symlinked_output_path(blind_output)
    # Both destination directories must exist BEFORE either write-once file is
    # created. _write_jsonl_exclusive writes blind first, then public; if only
    # public's parent were missing, blind would already be sealed (O_EXCL,
    # mode 0400) by the time the public write failed on its own parent check,
    # and every retry would then hit "write-once inventory output already
    # exists" above with public never created -- a stuck partial seal that
    # this write-once code path never overwrites or deletes by design, so
    # recovering would require manual intervention outside the contract.
    for output_path in (development_output, blind_output):
        if not output_path.parent.is_dir():
            raise InventoryError(f"output parent does not exist: {output_path.parent}")

    contract, _pilot, _roots, grid, state, rows, masks, context = _prepare_inventory_context(
        dataset_contract_path, pilot_contract_path
    )
    fixed_centres = load_pilot_fixed_centres(pilot_geometry_path)
    allocated: list[Direction] = []
    objective_reports: dict[str, Any] = {}
    speed_by_regime = {
        regime: mean_surface_speed_already_centered(state, index)
        for index, regime in enumerate(REGIMES)
    }
    group_keys = [(regime, family) for regime in REGIMES for family in FAMILIES]
    workers = max_workers or len(group_keys)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _solve_one_group,
                regime,
                family,
                rows,
                grid,
                masks,
                speed_by_regime[regime],
                fixed_centres,
            ): (regime, family)
            for regime, family in group_keys
        }
        for future in as_completed(futures):
            label, solved, objective = future.result()
            allocated.extend(solved)
            objective_reports[label] = objective
    allocated, long_report = solve_long_membership(allocated, contract)
    public_rows = [inventory_row(row) for row in allocated if row.role != "blind_test"]
    blind_rows = [inventory_row(row) for row in allocated if row.role == "blind_test"]
    if len(public_rows) != 912 or len(blind_rows) != 216:
        raise CapacityError(
            f"public/blind materialized counts differ: {len(public_rows)}/{len(blind_rows)}"
        )

    # Stage public bytes in memory and create the evaluator-only path with
    # O_EXCL.  Neither manifest contains responses or amplitudes.
    blind_digest = _write_jsonl_exclusive(blind_output, blind_rows, final_mode=0o400)
    public_digest = _write_jsonl_exclusive(development_output, public_rows, final_mode=0o444)
    return {
        "mode": "materialize",
        "version": "forward_response_inventory_materialization_v1",
        "pass": True,
        "numeric_data_created": False,
        "public": {
            "path": str(development_output),
            "rows": len(public_rows),
            "sha256": public_digest,
            "mode": "0444",
        },
        "blind": {
            "path_redacted_from_development_report": True,
            "rows": len(blind_rows),
            "sha256": blind_digest,
            "visibility": "evaluator_only",
            "mode": "0400",
        },
        "objectives": objective_reports,
        "long_membership": long_report,
        "audit_context": context,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    audit_parser = subparsers.add_parser("audit", help="aggregate read-only capacity audit")
    audit_parser.add_argument("--dataset-contract", type=Path, default=DEFAULT_DATASET_CONTRACT)
    audit_parser.add_argument("--pilot-contract", type=Path, default=DEFAULT_PILOT_CONTRACT)
    audit_parser.add_argument("--report", type=Path, default=None)
    materialize_parser = subparsers.add_parser(
        "materialize",
        help="exact geometry sealing, public + write-once blind (no numeric responses)",
    )
    materialize_parser.add_argument(
        "--dataset-contract", type=Path, default=DEFAULT_DATASET_CONTRACT
    )
    materialize_parser.add_argument("--pilot-contract", type=Path, default=DEFAULT_PILOT_CONTRACT)
    materialize_parser.add_argument("--pilot-geometry", type=Path, default=DEFAULT_PILOT_GEOMETRY)
    materialize_parser.add_argument(
        "--development-output", type=Path, default=DEFAULT_DEVELOPMENT_OUTPUT
    )
    materialize_parser.add_argument("--blind-output", type=Path, default=DEFAULT_BLIND_OUTPUT)
    materialize_parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="parallel (regime,family) group solves; default one worker per group (12)",
    )
    arguments = parser.parse_args(argv)
    try:
        if arguments.mode == "audit":
            result = run_audit(arguments.dataset_contract, arguments.pilot_contract)
            if arguments.report is not None:
                arguments.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        else:
            result = materialize_inventory(
                dataset_contract_path=arguments.dataset_contract,
                pilot_contract_path=arguments.pilot_contract,
                pilot_geometry_path=arguments.pilot_geometry,
                development_output=arguments.development_output,
                blind_output=arguments.blind_output,
                max_workers=arguments.max_workers,
            )
    except InventoryError as error:
        print(f"FORWARD RESPONSE INVENTORY: FAIL -- {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
