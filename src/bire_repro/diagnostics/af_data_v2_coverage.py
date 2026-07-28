"""Training-only effective-coverage audit for trajectory dataset version 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from ..af_model_c import GROUP_SLICES
from .af_model_c_diagnostics import integrated_autocorrelation_time
from ..af_s0 import _sha256


COVERAGE_VERSION = "trajectories_v2_coverage_audit_v1"
REPORT_NAME = "trajectories_v2_coverage_report.json"
ARRAY_NAME = "trajectories_v2_coverage_arrays.npz"
TRAINING_BLOCKS = ((0, 2520), (3690, 6210))
BLOCK_NAMES = ("v1_training", "extension_training")
EXPERIMENTS = ("S0", "S1", "S2")
HORIZON_DAYS = 10


class TrajectoryV2CoverageError(RuntimeError):
    """Raised when the training-only coverage audit crosses its contract."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_coverage_contract(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    """Load the contract frozen after v2 QC and before coverage metrics."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != COVERAGE_VERSION:
        raise ValueError(f"expected {COVERAGE_VERSION}")
    if (
        contract.get("contract_status")
        != "frozen_after_v2_quality_validation_before_coverage_metrics"
    ):
        raise ValueError("trajectory-v2 coverage contract was not frozen")
    if contract.get("training_blocks") != [list(block) for block in TRAINING_BLOCKS]:
        raise ValueError("trajectory-v2 coverage training blocks changed")
    read = contract.get("read_contract", {})
    if (
        read.get("snapshot_split_codes_read") != [1]
        or read.get("pair_split_codes_read") != [1]
        or any(
            read.get(name) is not False
            for name in (
                "validation_read",
                "inference_read",
                "intermediate_wind_read",
                "response_read",
                "adjoint_read",
            )
        )
    ):
        raise ValueError("trajectory-v2 coverage contract does not preserve sealed data")
    assessment = contract.get("assessment", {})
    if (
        assessment.get("target_effective_multiplier") != 2.0
        or assessment.get("minimum_material_multiplier") != 1.5
        or assessment.get("slow_state_groups") != ["temperature", "ssh"]
    ):
        raise ValueError("trajectory-v2 coverage assessment changed")
    return contract, resolved, _sha256(resolved)


def assess_effective_coverage(
    multipliers: Mapping[str, Mapping[str, float]],
    *,
    target: float,
    minimum_material: float,
    slow_groups: Sequence[str],
) -> dict[str, Any]:
    """Apply the predeclared descriptive target and material-gain threshold."""

    state_values = {
        group: float(multipliers["state_rms"][group]) for group in slow_groups
    }
    target_met = all(value >= target for value in state_values.values())
    material_gain = all(
        value >= minimum_material for value in state_values.values()
    )
    return {
        "slow_state_effective_multipliers": state_values,
        "target_effective_multiplier": target,
        "minimum_material_multiplier": minimum_material,
        "target_met": target_met,
        "material_gain": material_gain,
        "status": (
            "two_times_effective_target_met"
            if target_met
            else (
                "material_but_less_than_two_times_effective_gain"
                if material_gain
                else "insufficient_effective_gain"
            )
        ),
    }


def _block_proxies(
    state: Any,
    mean: np.ndarray,
    scale: np.ndarray,
    wet: np.ndarray,
    *,
    experiment: int,
    start: int,
    stop: int,
    chunk_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute state/increment RMS proxies for one contiguous training block."""

    state_proxy = np.empty((len(GROUP_SLICES), stop - start), dtype=np.float64)
    pair_count = stop - start - HORIZON_DAYS
    increment_proxy = np.empty((len(GROUP_SLICES), pair_count), dtype=np.float64)
    wet_values = wet[None, None]
    wet_count = int(wet.sum())
    for offset in range(0, stop - start, chunk_days):
        size = min(chunk_days, stop - start - offset)
        raw = np.asarray(
            state[experiment, start + offset : start + offset + size],
            dtype=np.float32,
        )
        normalized = (raw - mean[None, :, None, None]) / scale[
            None, :, None, None
        ]
        for group_index, channels in enumerate(GROUP_SLICES.values()):
            values = normalized[:, channels]
            state_proxy[group_index, offset : offset + size] = np.sqrt(
                (np.square(values) * wet_values).sum(axis=(1, 2, 3))
                / (wet_count * (channels.stop - channels.start))
            )
    for offset in range(0, pair_count, chunk_days):
        size = min(chunk_days, pair_count - offset)
        present = np.asarray(
            state[experiment, start + offset : start + offset + size],
            dtype=np.float32,
        )
        future = np.asarray(
            state[
                experiment,
                start + offset + HORIZON_DAYS : start + offset + HORIZON_DAYS + size,
            ],
            dtype=np.float32,
        )
        increment = (future - present) / scale[None, :, None, None]
        for group_index, channels in enumerate(GROUP_SLICES.values()):
            values = increment[:, channels]
            increment_proxy[group_index, offset : offset + size] = np.sqrt(
                (np.square(values) * wet_values).sum(axis=(1, 2, 3))
                / (wet_count * (channels.stop - channels.start))
            )
    return state_proxy, increment_proxy


def run_coverage_audit(
    dataset_path: str | Path,
    quality_report_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_days: int = 32,
) -> dict[str, Any]:
    """Measure effective training coverage without reading split codes 2 or 3."""

    dataset_path = Path(dataset_path).resolve()
    quality_path = Path(quality_report_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite coverage output: {output}")
    if chunk_days <= 0:
        raise ValueError("coverage chunk size must be positive")
    contract, resolved_contract, contract_sha = load_coverage_contract(contract_path)
    quality_sha = _sha256(quality_path)
    if (
        _sha256(dataset_path / ".zmetadata")
        != contract["source_hashes"]["dataset_metadata_sha256"]
        or quality_sha != contract["source_hashes"]["quality_report_sha256"]
    ):
        raise TrajectoryV2CoverageError("trajectory-v2 coverage sources changed")
    quality = json.loads(quality_path.read_text())
    if quality.get("status") != "valid":
        raise TrajectoryV2CoverageError("trajectory-v2 quality gate did not pass")

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    state = group["state"]
    if tuple(state.shape) != (3, 7200, 46, 62, 62):
        raise TrajectoryV2CoverageError("trajectory-v2 state shape changed")
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    for start, stop in TRAINING_BLOCKS:
        if (
            np.any(snapshot_codes[start:stop] != 1)
            or np.any(pair_codes[start : stop - HORIZON_DAYS] != 1)
        ):
            raise TrajectoryV2CoverageError("declared coverage block is not training-only")
    mean = np.asarray(group["state_mean"][:], dtype=np.float32)
    scale = np.asarray(group["state_scale"][:], dtype=np.float32)
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    arrays: dict[str, np.ndarray] = {}
    evidence: dict[str, Any] = {"state_rms": {}, "increment_rms": {}}
    started = time.monotonic()

    proxy_store: dict[tuple[str, str, int], np.ndarray] = {}
    for experiment_index, experiment in enumerate(EXPERIMENTS):
        for block_index, ((start, stop), block_name) in enumerate(
            zip(TRAINING_BLOCKS, BLOCK_NAMES)
        ):
            state_proxy, increment_proxy = _block_proxies(
                state,
                mean,
                scale,
                wet,
                experiment=experiment_index,
                start=start,
                stop=stop,
                chunk_days=chunk_days,
            )
            proxy_store[("state_rms", experiment, block_index)] = state_proxy
            proxy_store[("increment_rms", experiment, block_index)] = increment_proxy
            arrays[f"{experiment}_{block_name}_state_rms"] = state_proxy
            arrays[f"{experiment}_{block_name}_increment_rms"] = increment_proxy

    multipliers: dict[str, dict[str, float]] = {
        "state_rms": {},
        "increment_rms": {},
    }
    for family in ("state_rms", "increment_rms"):
        for group_index, group_name in enumerate(GROUP_SLICES):
            by_regime = []
            baseline_total = 0.0
            combined_total = 0.0
            for experiment in EXPERIMENTS:
                by_block = []
                for block_index, block_name in enumerate(BLOCK_NAMES):
                    proxy = proxy_store[(family, experiment, block_index)][group_index]
                    tau = integrated_autocorrelation_time(proxy)
                    effective = float(proxy.size / tau)
                    by_block.append(
                        {
                            "block": block_name,
                            "sample_count": int(proxy.size),
                            "tau_days": tau,
                            "effective_samples": effective,
                        }
                    )
                    combined_total += effective
                    if block_index == 0:
                        baseline_total += effective
                by_regime.append(
                    {
                        "experiment": experiment,
                        "by_block": by_block,
                        "combined_effective_samples": float(
                            sum(item["effective_samples"] for item in by_block)
                        ),
                        "effective_multiplier": float(
                            sum(item["effective_samples"] for item in by_block)
                            / by_block[0]["effective_samples"]
                        ),
                    }
                )
            multiplier = combined_total / baseline_total
            multipliers[family][group_name] = float(multiplier)
            evidence[family][group_name] = {
                "by_regime": by_regime,
                "baseline_effective_samples_total": baseline_total,
                "combined_effective_samples_total": combined_total,
                "effective_multiplier": float(multiplier),
            }

    assessment = assess_effective_coverage(
        multipliers,
        target=float(contract["assessment"]["target_effective_multiplier"]),
        minimum_material=float(
            contract["assessment"]["minimum_material_multiplier"]
        ),
        slow_groups=contract["assessment"]["slow_state_groups"],
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    arrays_path = temporary / ARRAY_NAME
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "status": "complete",
        "purpose": "training_only_trajectories_v2_effective_coverage_audit",
        "version": COVERAGE_VERSION,
        "dataset": str(dataset_path),
        "dataset_metadata_sha256": _sha256(dataset_path / ".zmetadata"),
        "quality_report": str(quality_path),
        "quality_report_sha256": quality_sha,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "read_contract": {
            "snapshot_split_codes_read": [1],
            "pair_split_codes_read": [1],
            "validation_read": False,
            "inference_read": False,
            "intermediate_wind_read": False,
            "response_read": False,
            "adjoint_read": False,
        },
        "training_blocks": [list(block) for block in TRAINING_BLOCKS],
        "autocorrelation": evidence,
        "effective_multipliers": multipliers,
        "assessment": assessment,
        "arrays": ARRAY_NAME,
        "arrays_sha256": _sha256(arrays_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    report["report_content_sha256"] = _json_sha256(report)
    (temporary / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, output)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-days", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_coverage_audit(
        args.dataset,
        args.quality_report,
        args.contract,
        args.output_dir,
        chunk_days=args.chunk_days,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
