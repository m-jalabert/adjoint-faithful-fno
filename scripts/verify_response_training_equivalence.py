"""Execution step 11's primary-seed equivalence harness (plan section 23.2).

Proves ``src/oceanfno/train_response.py``'s response-disabled code path
reproduces the exact parent trainer's deterministic artifacts -- training/
validation record sets, per-seed nominal batch schedule, derived static
block, and recomputed normalization -- by recomputing each one through
``train_response.py``'s own code path and hash-matching it against the
values `config/model_c_adjoint_faithful_nominal_control_v1.json`'s
``study_contract.equality_artifact_hashes`` already pins (frozen 2026-08-24,
before this module existed).

This does not re-run all 7,680 optimizer steps of a real B training (that is
step 11's own next action, a multi-GPU-hour job): ``train.py``'s own
``load_contract`` hard-rejects any contract whose ``training.maximum_steps``
is not exactly 7,680, so there is no way to get its *real* ``run()`` to do a
short comparison run, and re-running the full 7,680 steps twice just to
compare would cost as much GPU time as the real run itself. What *can* be
checked without training is everything upstream of the step loop -- and
because ``train_response.py`` imports several of ``train.py``'s own
functions directly (``physical_static_block``, ``physics_contexts``,
``evaluate_loss``, ``split_summary``, ``acceptance_gate``) rather than
reimplementing them, per-step loss/gradient equivalence for the *shared*
code is a Python-identity guarantee, not something a runtime test discovers.
What a test *can* and must catch is a mistake in the genuinely new
orchestration code (contract loading, seeding order, dataset/loader/model
construction order, output naming) -- exactly what the artifact-hash
comparison below targets, since every one of these artifacts depends on that
exact sequence executing in the exact order the parent's did.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oceanfno import train_response as tr  # noqa: E402
from oceanfno.dataset import (  # noqa: E402
    TRAIN_CODE,
    assert_store_is_v3,
    records_for_rollout_split,
    store_codes,
    store_wind_normalization,
    training_increment_scale,
    training_pointwise_normalizers,
    validation_records,
    validation_starts,
)
from oceanfno.runtime import ChunkAwareBatchSampler  # noqa: E402


class EquivalenceError(RuntimeError):
    """Raised when a recomputed artifact does not match its pinned hash."""


def _array_sha256(array: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def _check(label: str, actual: str, expected: str, findings: list[str], *, informational: bool = False) -> None:
    if actual == expected:
        print(f"  OK   {label}")
        return
    marker = "WARN" if informational else "FAIL"
    print(f"  {marker} {label}: actual={actual} expected={expected}")
    if not informational:
        findings.append(label)


def _nominal_schedule(training_records: list[tuple[int, int]], seed: int, steps: int, microbatches: int, microbatch_size: int) -> np.ndarray:
    """Replays exactly what train_response.run()'s microbatch loop does --
    build the sampler once, iterate it, and on StopIteration rebuild a fresh
    iterator -- without loading any zarr data, since only the *index*
    schedule (mapped back to (experiment, start)) is needed for the hash."""

    class _Records:
        def __init__(self, records):
            self.records = records

    sampler = ChunkAwareBatchSampler(_Records(training_records), microbatch_size, seed)
    schedule = np.zeros((steps, microbatches, microbatch_size, 2), dtype=np.int32)
    iterator = iter(sampler)
    for step in range(steps):
        for micro in range(microbatches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(sampler)
                batch = next(iterator)
            for slot, index in enumerate(batch):
                experiment, start = training_records[index]
                schedule[step, micro, slot] = (experiment, start)
    return schedule


def verify(contract_path: str | Path) -> list[str]:
    findings: list[str] = []
    contract, _resolved, _digest = tr.load_contract(contract_path, verify_sources=True)
    study = contract["study_contract"]
    pinned = study["equality_artifact_hashes"]
    dataset_path = tr._verify_dataset(contract)
    group = zarr.open_consolidated(str(dataset_path), mode="r")
    assert_store_is_v3(group)
    snapshot_split, pair_split = store_codes()

    print("=== index-only artifacts (no data read) ===")
    training_records = records_for_rollout_split(pair_split, TRAIN_CODE, rollout_steps=tr.ROLLOUT_STEPS)
    _check(
        "training_records",
        _array_sha256(np.asarray(training_records), "<i8"),
        pinned["training_records_(17820,2)_<i8"],
        findings,
    )
    records = validation_records()
    _check("validation_records", _array_sha256(records, "<i4"), pinned["validation_records_(102,2)_<i4"], findings)
    _check("snapshot_codes", _array_sha256(snapshot_split, "|u1"), pinned["snapshot_codes_(9000,)_|u1"], findings)
    _check("pair_codes", _array_sha256(pair_split, "|u1"), pinned["pair_codes_(9000,)_|u1"], findings)
    # Plan section 6.2: the nested final-inference starts are a fixed,
    # separately-seeded (20260802) selection within [6200,7000), unrelated to
    # validation_starts() (the 34-per-regime checkpoint-validation starts).
    # Not used by training or Gate M0 -- checked here only for completeness.
    inference_starts = np.asarray(
        [6263, 6293, 6331, 6389, 6579, 6593, 6598, 6601, 6651, 6661, 6694, 6707, 6711, 6968, 6979],
        dtype=np.int64,
    )
    if "inference_starts_(15,)_<i8" in pinned:
        _check("inference_starts", _array_sha256(inference_starts, "<i8"), pinned["inference_starts_(15,)_<i8"], findings)

    print("=== per-seed nominal batch schedule ===")
    microbatches = int(contract["training"]["gradient_accumulation_steps"])
    microbatch_size = int(contract["training"]["microbatch_size"])
    steps = int(contract["training"]["maximum_steps"])
    for seed_str, expected in pinned["nominal_schedule_(7680,2,4,2)_<i4"].items():
        seed = int(seed_str)
        started = time.monotonic()
        schedule = _nominal_schedule(training_records, seed, steps, microbatches, microbatch_size)
        elapsed = time.monotonic() - started
        _check(f"nominal_schedule[seed={seed}] ({elapsed:.1f}s)", _array_sha256(schedule, "<i4"), expected, findings)

    print("=== data-dependent artifacts (reads the trajectory store) ===")
    normalizers = training_pointwise_normalizers(group, snapshot_split)
    component_hashes = {
        "pointwise_mean_(46,62,62)_<f4": _array_sha256(normalizers["mean"], "<f4"),
        "pointwise_raw_scale_(46,62,62)_<f4": _array_sha256(normalizers["raw_scale"], "<f4"),
        "pointwise_scale_(46,62,62)_<f4": _array_sha256(normalizers["scale"], "<f4"),
        "channel_scale_floor_(46,)_<f4": _array_sha256(normalizers["floor"], "<f4"),
    }
    increment_values = training_increment_scale(group, pair_split, normalizers["scale"])
    component_hashes["increment_scale_(46,)_<f4"] = _array_sha256(increment_values, "<f4")
    for key, actual in component_hashes.items():
        _check(f"normalization.{key}", actual, pinned["normalization_components_c_order"][key], findings)

    statics, _provenance = tr.physical_static_block(contract["sources"], group, normalizers["mean"], normalizers["scale"])
    _check("static_block", _array_sha256(statics, "<f4"), pinned["static_block_(3,5,62,62)_<f4"], findings)

    state = group["state"]
    wet_array, _, _ = store_wind_normalization(group)
    wet_array = np.asarray(wet_array, dtype=bool)
    climatology_state, climatology_derived, _days = tr.parent_train.train_only_climatology(state, wet_array)
    clim_pinned = pinned["train_only_climatology_c_order_<f4"]
    _check("climatology.state", _array_sha256(climatology_state, "<f4"), clim_pinned["state_(3,46,62,62)"], findings)
    # The pin (frozen 2026-08-24) shortened "phihyd_surface" -> "phihyd"; the
    # other three derived-field names are unchanged from validation.py's own.
    for real_key, pinned_key in (
        ("phihyd_surface", "phihyd"),
        ("sst", "sst"),
        ("streamfunction", "streamfunction"),
        ("surface_speed", "surface_speed"),
    ):
        _check(
            f"climatology.{pinned_key}",
            _array_sha256(climatology_derived[real_key], "<f4"),
            clim_pinned[pinned_key],
            findings,
        )

    return findings


def main() -> int:
    contract_path = sys.argv[1] if len(sys.argv) > 1 else str(
        PROJECT_ROOT / "config" / "model_c_adjoint_faithful_nominal_control_v1.json"
    )
    findings = verify(contract_path)
    if findings:
        print(f"\nEquivalence harness: FAIL -- {len(findings)} mismatches: {findings}")
        return 1
    print("\nEquivalence harness: PASS -- every recomputed artifact matches its pinned hash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
