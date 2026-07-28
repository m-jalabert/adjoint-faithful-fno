"""Training-only 10--90-day rollout diagnosis after successor validation.

The fresh-v2 validation gate showed good ten-day skill followed by accumulating
slow-field error.  This module tests whether the same lead-time structure is
already present on a fixed, chronology-balanced subset of split-1 training
starts.  It never reads inference or response/adjoint archives and does not
train or alter a checkpoint.
"""

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

from .af_a0_evaluate import _normalizers
from .af_forward_complete import _training_climatology
from .af_model_c_overfit import _file_sha256
from .af_model_c_successor_validation import (
    EVALUATION_FIELDS,
    LEAD_DAYS,
    _evaluate_baseline_metrics,
    _evaluate_stepper,
    _load_successor_stepper,
    _method_auc_summary,
    _training_artifacts,
    curve_auc,
    load_validation_contract,
)

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


ROLLOUT_DIAGNOSIS_VERSION = "model_c_rollout_diagnosis_v1"
REPORT_NAME = "model_c_rollout_diagnosis_report.json"
ARRAYS_NAME = "model_c_rollout_diagnosis_arrays.npz"
SLOW_FIELDS = ("sst", "phihyd_surface", "ssh")


class ModelCRolloutDiagnosisError(RuntimeError):
    """Raised when the frozen training-only diagnosis contract is violated."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def load_rollout_diagnosis_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the contract frozen before training-rollout metrics are computed."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != ROLLOUT_DIAGNOSIS_VERSION:
        raise ValueError(
            f"expected rollout diagnosis {ROLLOUT_DIAGNOSIS_VERSION}"
        )
    if (
        contract.get("contract_status")
        != "frozen_after_fresh_v2_rejection_before_training_rollout_metrics"
    ):
        raise ValueError("Model C rollout diagnosis contract was not frozen")
    records = contract.get("records", {})
    if (
        int(records.get("starts_per_training_block", -1)) != 90
        or int(records.get("expected_training_blocks", -1)) != 2
        or tuple(records.get("lead_days", ())) != LEAD_DAYS
        or records.get("selection")
        != "evenly_spaced_complete_90_day_starts_per_training_block"
    ):
        raise ValueError("training-rollout record contract changed")
    read = contract.get("read_contract", {})
    if (
        read.get("training_pair_code") != 1
        or read.get("training_state_read") is not True
        or read.get("validation_decision_metadata_read") is not True
        or any(
            read.get(name) is not False
            for name in (
                "validation_state_read",
                "inference_read",
                "intermediate_wind_read",
                "response_read",
                "adjoint_read",
            )
        )
    ):
        raise ValueError("training-rollout read contract changed")
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(f"rollout diagnosis source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def complete_rollout_starts(
    pair_codes: np.ndarray,
    snapshot_codes: np.ndarray,
    *,
    split_code: int = 1,
) -> np.ndarray:
    """Return starts with every ten-day pair and state through day 90."""

    pair_codes = np.asarray(pair_codes, dtype=np.uint8)
    snapshot_codes = np.asarray(snapshot_codes, dtype=np.uint8)
    if pair_codes.ndim != 1 or snapshot_codes.ndim != 1:
        raise ValueError("split codes must be one-dimensional")
    selected = []
    offsets = np.arange(len(LEAD_DAYS) + 1) * 10
    for start in range(pair_codes.size):
        pair_indices = start + offsets[:-1]
        snapshot_indices = start + offsets
        if (
            pair_indices[-1] < pair_codes.size
            and snapshot_indices[-1] < snapshot_codes.size
            and np.all(pair_codes[pair_indices] == split_code)
            and np.all(snapshot_codes[snapshot_indices] == split_code)
        ):
            selected.append(start)
    return np.asarray(selected, dtype=np.int64)


def select_balanced_training_times(
    complete_starts: np.ndarray,
    *,
    starts_per_block: int = 90,
    expected_blocks: int = 2,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Select deterministic, evenly spaced starts from every training block."""

    starts = np.asarray(complete_starts, dtype=np.int64)
    if starts.ndim != 1 or starts.size == 0 or starts_per_block <= 0:
        raise ValueError("complete starts and block sample count are required")
    cuts = np.flatnonzero(np.diff(starts) != 1) + 1
    blocks = tuple(np.split(starts, cuts))
    if len(blocks) != expected_blocks:
        raise ValueError(
            f"expected {expected_blocks} complete training blocks, "
            f"found {len(blocks)}"
        )
    selected = []
    bounds = []
    for block in blocks:
        if block.size < starts_per_block:
            raise ValueError("training block is too short for declared sampling")
        positions = np.linspace(
            0,
            block.size - 1,
            starts_per_block,
            dtype=np.int64,
        )
        values = block[positions]
        if np.unique(values).size != starts_per_block:
            raise ValueError("evenly spaced training starts are not unique")
        selected.append(values)
        bounds.append((int(block[0]), int(block[-1])))
    return np.concatenate(selected), tuple(bounds)


def lead_curve_summary(
    metrics: Mapping[str, np.ndarray],
    baselines: Mapping[str, Mapping[str, np.ndarray]],
    records: np.ndarray,
) -> dict[str, Any]:
    """Summarize lead curves and AUC ratios overall and by wind regime."""

    records = np.asarray(records, dtype=np.int64)
    regimes = records[:, 0]
    result: dict[str, Any] = {}
    for field in EVALUATION_FIELDS:
        model_rmse = np.asarray(metrics[f"rmse__{field}"], dtype=np.float64)
        model_acc = np.asarray(metrics[f"acc__{field}"], dtype=np.float64)
        comparisons = {}
        for baseline_name, baseline_metrics in baselines.items():
            baseline_rmse = np.asarray(
                baseline_metrics[f"rmse__{field}"],
                dtype=np.float64,
            )
            baseline_acc = np.asarray(
                baseline_metrics[f"acc__{field}"],
                dtype=np.float64,
            )
            mean_model_rmse = model_rmse.mean(axis=0)
            mean_baseline_rmse = baseline_rmse.mean(axis=0)
            lead_ratio = np.divide(
                mean_model_rmse,
                mean_baseline_rmse,
                out=np.full_like(mean_model_rmse, np.inf),
                where=mean_baseline_rmse > 0,
            )
            lead_acc_difference = (
                model_acc.mean(axis=0) - baseline_acc.mean(axis=0)
            )
            by_regime = {}
            for experiment in range(3):
                selected = regimes == experiment
                regime_model = model_rmse[selected].mean(axis=0)
                regime_baseline = baseline_rmse[selected].mean(axis=0)
                by_regime[f"S{experiment}"] = {
                    "rmse_ratio_by_lead": np.divide(
                        regime_model,
                        regime_baseline,
                        out=np.full_like(regime_model, np.inf),
                        where=regime_baseline > 0,
                    ).tolist(),
                    "rmse_auc_ratio": float(
                        curve_auc(model_rmse[selected]).mean()
                        / curve_auc(baseline_rmse[selected]).mean()
                    ),
                }
            crossing = np.flatnonzero(lead_ratio >= 1.0)
            comparisons[baseline_name] = {
                "rmse_ratio_by_lead": lead_ratio.tolist(),
                "acc_difference_by_lead": lead_acc_difference.tolist(),
                "rmse_auc_ratio": float(
                    curve_auc(model_rmse).mean()
                    / curve_auc(baseline_rmse).mean()
                ),
                "acc_auc_difference": float(
                    curve_auc(model_acc).mean()
                    - curve_auc(baseline_acc).mean()
                ),
                "first_rmse_crossing_day": (
                    int(LEAD_DAYS[int(crossing[0])])
                    if crossing.size
                    else None
                ),
                "by_regime": by_regime,
            }
        result[field] = comparisons
    return result


def _diagnostic_interpretation(
    seed_curves: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the predeclared descriptive hypothesis classification."""

    one_step = {}
    auc = {}
    day90 = {}
    for field in SLOW_FIELDS:
        one_step[field] = [
            float(seed[field]["persistence"]["rmse_ratio_by_lead"][0])
            for seed in seed_curves
        ]
        auc[field] = [
            float(seed[field]["persistence"]["rmse_auc_ratio"])
            for seed in seed_curves
        ]
        day90[field] = [
            float(seed[field]["persistence"]["rmse_ratio_by_lead"][-1])
            for seed in seed_curves
        ]
    per_seed_drift = []
    for index in range(len(seed_curves)):
        per_seed_drift.append(
            any(
                one_step[field][index] < 1.0
                and auc[field][index] > 1.0
                and day90[field][index] > 1.0
                for field in SLOW_FIELDS
            )
        )
    reproduced = bool(all(per_seed_drift))
    return {
        "slow_fields": list(SLOW_FIELDS),
        "one_step_ratio_to_persistence": one_step,
        "rmse_auc_ratio_to_persistence": auc,
        "day90_ratio_to_persistence": day90,
        "every_seed_reproduces_good_one_step_then_failed_slow_rollout": bool(
            reproduced
        ),
        "classification": (
            "training_objective_or_checkpoint_gate_mismatch"
            if reproduced
            else "validation_generalization_gap_not_reproduced_on_training"
        ),
    }


def evaluate_rollout_diagnosis(
    dataset_path: str | Path,
    validation_report_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate immutable checkpoints on fixed split-1 10--90-day starts."""

    if torch is None:
        raise RuntimeError("rollout diagnosis requires PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = (
        load_rollout_diagnosis_contract(contract_path)
    )
    dataset = Path(dataset_path).resolve()
    validation_report_path = Path(validation_report_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite rollout diagnosis output: {output}"
        )
    sources = contract["source_artifacts"]
    if (
        _file_sha256(dataset / ".zmetadata")
        != sources["dataset_metadata_sha256"]
        or _file_sha256(validation_report_path)
        != sources["fresh_validation_report_sha256"]
    ):
        raise ModelCRolloutDiagnosisError(
            "rollout diagnosis source artifact changed"
        )
    validation_report = json.loads(validation_report_path.read_text())
    if (
        validation_report.get("validation_gate", {}).get("status")
        != "scientifically_rejected_fresh_v2_validation"
        or validation_report.get("validation_gate", {}).get(
            "inference_authorized"
        )
        is not False
        or validation_report.get("inference_opened") is not False
    ):
        raise ModelCRolloutDiagnosisError(
            "fresh validation is not the expected sealed rejection"
        )
    validation_contract, _, validation_contract_sha = (
        load_validation_contract(sources["successor_validation_contract"])
    )
    if validation_contract_sha != sources[
        "successor_validation_contract_sha256"
    ]:
        raise ModelCRolloutDiagnosisError("successor validation contract changed")
    artifacts = _training_artifacts(validation_contract)

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnosis requested without a visible GPU")
    device = torch.device(device_name)

    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    complete = complete_rollout_starts(pair_codes, snapshot_codes)
    times, block_bounds = select_balanced_training_times(
        complete,
        starts_per_block=int(
            contract["records"]["starts_per_training_block"]
        ),
        expected_blocks=int(contract["records"]["expected_training_blocks"]),
    )
    records = np.asarray(
        [
            (experiment, int(time_index))
            for experiment in range(3)
            for time_index in times
        ],
        dtype=np.int64,
    )
    mean, scale, wet, _, wind_mean, wind_scale = _normalizers(group)
    climatology_state, climatology_derived, training_days = (
        _training_climatology(state, snapshot_codes, wet)
    )
    batch_size = int(contract["evaluation"]["batch_size"])
    baseline_raw = _evaluate_baseline_metrics(
        state,
        records,
        climatology_state,
        climatology_derived,
        wet,
        mean,
        scale,
        batch_size=batch_size,
    )
    baselines: dict[str, Mapping[str, np.ndarray]] = {
        "persistence": baseline_raw["persistence"],
        "climatology": baseline_raw["climatology"],
    }
    arrays: dict[str, np.ndarray] = {
        "records": records.astype(np.int32),
        "lead_days": np.asarray(LEAD_DAYS, dtype=np.int16),
        "training_times": times.astype(np.int32),
    }
    for method, metrics in baselines.items():
        for name, value in metrics.items():
            arrays[f"{method}__{name}"] = np.asarray(value)

    seed_summaries = []
    seed_curves = []
    for artifact in artifacts:
        stepper, payload = _load_successor_stepper(
            Path(artifact["checkpoint"]),
            device,
            wet,
            mean,
            scale,
            wind_mean,
            wind_scale,
            validation_contract["architecture"],
        )
        if int(payload.get("seed", -1)) != int(artifact["seed"]):
            raise ModelCRolloutDiagnosisError(
                "successor checkpoint seed provenance changed"
            )
        metrics = _evaluate_stepper(
            stepper,
            state,
            static,
            records,
            climatology_state,
            climatology_derived,
            mean,
            scale,
            batch_size=batch_size,
        )
        for name, value in metrics.items():
            arrays[f"seed_{artifact['seed']}__{name}"] = np.asarray(value)
        curves = lead_curve_summary(metrics, baselines, records)
        seed_curves.append(curves)
        seed_summaries.append(
            {
                "seed": int(artifact["seed"]),
                "checkpoint": artifact["checkpoint"],
                "checkpoint_sha256": artifact["checkpoint_sha256"],
                "all_field_auc": _method_auc_summary(metrics),
                "lead_curves": curves,
            }
        )

    interpretation = _diagnostic_interpretation(seed_curves)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    arrays_path = temporary / ARRAYS_NAME
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "status": "complete",
        "purpose": "training_only_10_to_90_day_rollout_objective_diagnosis",
        "version": ROLLOUT_DIAGNOSIS_VERSION,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset),
        "dataset_metadata_sha256": _file_sha256(dataset / ".zmetadata"),
        "fresh_validation_report": str(validation_report_path),
        "fresh_validation_report_sha256": _file_sha256(
            validation_report_path
        ),
        "device": str(device),
        "read_contract": {
            "training_pair_code": 1,
            "training_state_read": True,
            "validation_decision_metadata_read": True,
            "validation_state_read": False,
            "inference_read": False,
            "intermediate_wind_read": False,
            "response_read": False,
            "adjoint_read": False,
        },
        "record_contract": {
            "lead_days": list(LEAD_DAYS),
            "complete_training_blocks": [list(value) for value in block_bounds],
            "starts_per_training_block": int(
                contract["records"]["starts_per_training_block"]
            ),
            "selected_times_per_regime": int(times.size),
            "records_total": int(records.shape[0]),
            "training_times_sha256": _array_sha256(times),
            "records_sha256": _array_sha256(records),
            "training_climatology_snapshots_per_regime": training_days,
        },
        "training_seed_artifacts": [
            {
                key: artifact[key]
                for key in (
                    "seed",
                    "report",
                    "report_sha256",
                    "checkpoint",
                    "checkpoint_sha256",
                    "selected_optimizer_step",
                )
            }
            for artifact in artifacts
        ],
        "baseline_auc": {
            method: _method_auc_summary(metrics)
            for method, metrics in baselines.items()
        },
        "seed_diagnosis": seed_summaries,
        "diagnostic_interpretation": interpretation,
        "arrays": str(output / ARRAYS_NAME),
        "arrays_sha256": _file_sha256(arrays_path),
        "elapsed_seconds": time.monotonic() - started,
        "inference_opened": False,
        "intermediate_wind_opened": False,
        "response_or_adjoint_opened": False,
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
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_rollout_diagnosis(
        args.dataset,
        args.validation_report,
        args.contract,
        args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
