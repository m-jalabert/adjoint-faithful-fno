"""Post-search, validation-aware data-adequacy decision for Model C.

This audit is deliberately narrower than another validation search.  It
evaluates the already-selected Model C checkpoints by wind regime, combines
those results with the previously sealed training-only autocorrelation report,
and applies an immutable expansion rule.  Pair code 3 and all later datasets
remain unread.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..af_model_a import ModelAPairDataset, require_model_a_runtime
from ..af_model_c import GROUP_SLICES, ModelCArchitecture, build_model_c
from ..af_model_c_overfit import _device, _file_sha256

try:  # Keep documentation-only imports usable without the optional ML stack.
    import torch
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


CONTRACT_VERSION = "model_c_data_adequacy_v1"
REPORT_NAME = "model_c_data_adequacy_report.json"
ARRAY_NAME = "model_c_data_adequacy_arrays.npz"


class ModelCDataAdequacyError(RuntimeError):
    """Raised when the post-search audit violates its immutable contract."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_data_adequacy_contract(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    """Load and validate the contract frozen before per-regime evaluation."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != CONTRACT_VERSION:
        raise ValueError(f"expected {CONTRACT_VERSION}")
    if (
        contract.get("contract_status")
        != "predeclared_before_per_regime_model_metrics_were_computed"
    ):
        raise ValueError("data-adequacy contract was not predeclared")
    read = contract.get("read_contract", {})
    if (
        read.get("pair_split_codes_read") != [1, 2]
        or read.get("inference_pair_code") != 3
        or any(
            read.get(name) is not False
            for name in (
                "inference_read",
                "intermediate_wind_read",
                "response_read",
                "adjoint_read",
            )
        )
    ):
        raise ValueError("data-adequacy contract does not preserve sealed data")
    bootstrap = contract.get("bootstrap", {})
    if (
        bootstrap.get("replicates") != 4000
        or bootstrap.get("seed") != 20260727
        or bootstrap.get("confidence_level") != 0.95
    ):
        raise ValueError("data-adequacy bootstrap contract changed")
    if contract.get("candidate_id") != "m24_16_w64":
        raise ValueError("data-adequacy audit must retain the selected Model C candidate")
    return contract, resolved, _file_sha256(resolved)


def _architecture_from_mapping(values: Mapping[str, Any]) -> ModelCArchitecture:
    return ModelCArchitecture(
        in_channels=int(values["in_channels"]),
        out_channels=int(values["out_channels"]),
        n_modes=tuple(int(value) for value in values["n_modes"]),
        hidden_channels=int(values["hidden_channels"]),
        n_layers=int(values["n_layers"]),
        domain_padding=float(values["domain_padding"]),
        positional_embedding=str(values["positional_embedding"]),
        use_channel_mlp=bool(values["use_channel_mlp"]),
        local_kernel_size=int(values["local_kernel_size"]),
        precision=str(values["precision"]),
        factorization=values["factorization"],
    )


def _record_mse(
    model: Any,
    dataset: ModelAPairDataset,
    *,
    batch_size: int,
    device: Any,
) -> dict[str, dict[str, np.ndarray]]:
    """Return physical model and persistence MSE for every ordered record."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    wet = torch.from_numpy(dataset.wet.astype(np.float32))[None, None].to(device)
    scale = torch.from_numpy(dataset.scale.astype(np.float32))[None, :, None, None].to(
        device
    )
    wet_count = int(dataset.wet.sum())
    result = {
        group: {"model": [], "persistence": []}
        for group in GROUP_SLICES
    }
    model.eval()
    with torch.no_grad():
        for features, _, future in loader:
            features = features.to(device=device, dtype=torch.float32, non_blocking=True)
            future = future.to(device=device, dtype=torch.float32, non_blocking=True)
            present = features[:, :46]
            prediction = (present + model(features)) * wet
            for group, channels in GROUP_SLICES.items():
                denominator = wet_count * (channels.stop - channels.start)
                physical_scale = scale[:, channels]
                model_error = (prediction[:, channels] - future[:, channels]) * physical_scale
                persistence_error = (present[:, channels] - future[:, channels]) * physical_scale
                model_mse = (
                    (model_error.square() * wet)
                    .sum(dim=(1, 2, 3), dtype=torch.float64)
                    .div(denominator)
                )
                persistence_mse = (
                    (persistence_error.square() * wet)
                    .sum(dim=(1, 2, 3), dtype=torch.float64)
                    .div(denominator)
                )
                result[group]["model"].append(model_mse.cpu().numpy())
                result[group]["persistence"].append(persistence_mse.cpu().numpy())
    return {
        group: {
            baseline: np.concatenate(chunks).astype(np.float64, copy=False)
            for baseline, chunks in values.items()
        }
        for group, values in result.items()
    }


def summarize_record_mse(
    records: Sequence[tuple[int, int]],
    errors: Mapping[str, Mapping[str, np.ndarray]],
    experiments: Sequence[str],
) -> dict[str, Any]:
    """Summarize ordered record-level MSE globally and by wind regime."""

    experiment_index = np.asarray([record[0] for record in records], dtype=np.int64)

    def summarize(indices: np.ndarray) -> dict[str, Any]:
        groups: dict[str, Any] = {}
        for group, values in errors.items():
            model_mse = float(np.mean(np.asarray(values["model"])[indices]))
            persistence_mse = float(
                np.mean(np.asarray(values["persistence"])[indices])
            )
            if model_mse < 0 or persistence_mse <= 0:
                raise ModelCDataAdequacyError("record-level physical MSE is invalid")
            groups[group] = {
                "model_rmse": math.sqrt(model_mse),
                "persistence_rmse": math.sqrt(persistence_mse),
                "ratio_to_persistence": math.sqrt(model_mse / persistence_mse),
            }
        return {
            "record_count": int(indices.size),
            "groups": groups,
        }

    summary = {"all_regimes": summarize(np.arange(len(records), dtype=np.int64))}
    summary["by_regime"] = {
        experiment: summarize(np.flatnonzero(experiment_index == index))
        for index, experiment in enumerate(experiments)
    }
    return summary


def circular_block_bootstrap_ratio(
    model_by_regime: Sequence[np.ndarray],
    persistence_by_regime: Sequence[np.ndarray],
    block_lengths: Sequence[int],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float]:
    """Bootstrap an aggregate RMSE ratio using independent circular regime blocks."""

    if (
        len(model_by_regime) != len(persistence_by_regime)
        or len(model_by_regime) != len(block_lengths)
        or not model_by_regime
        or replicates <= 0
        or not 0 < confidence_level < 1
    ):
        raise ValueError("invalid circular block-bootstrap inputs")
    rng = np.random.default_rng(seed)
    ratios = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        model_sum = 0.0
        persistence_sum = 0.0
        count = 0
        for model, persistence, requested_length in zip(
            model_by_regime, persistence_by_regime, block_lengths
        ):
            model = np.asarray(model, dtype=np.float64)
            persistence = np.asarray(persistence, dtype=np.float64)
            if (
                model.ndim != 1
                or model.shape != persistence.shape
                or model.size == 0
                or np.any(model < 0)
                or np.any(persistence <= 0)
            ):
                raise ValueError("bootstrap MSE arrays are invalid")
            block_length = min(max(1, int(requested_length)), model.size)
            block_count = math.ceil(model.size / block_length)
            starts = rng.integers(0, model.size, size=block_count)
            offsets = np.arange(block_length, dtype=np.int64)
            indices = ((starts[:, None] + offsets[None, :]) % model.size).ravel()[
                : model.size
            ]
            model_sum += float(model[indices].sum())
            persistence_sum += float(persistence[indices].sum())
            count += int(indices.size)
        ratios[replicate] = math.sqrt((model_sum / count) / (persistence_sum / count))
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "lower": float(np.quantile(ratios, alpha)),
        "median": float(np.median(ratios)),
        "upper": float(np.quantile(ratios, 1.0 - alpha)),
        "probability_below_persistence": float(np.mean(ratios < 1.0)),
    }


def data_expansion_decision(
    contract: Mapping[str, Any],
    chronology: Sequence[Mapping[str, Any]],
    aggregate_training_fit: Sequence[bool],
    per_seed: Sequence[Mapping[str, Any]],
    autocorrelation: Mapping[str, Any],
    experiments: Sequence[str],
) -> dict[str, Any]:
    """Apply the frozen expansion rules to already-computed evidence."""

    rules = contract["decision_rules"]
    chronology_ratios = [
        float(item["validation_worst_group_ratio"]) for item in chronology
    ]
    chronology_pass = all(
        later < earlier
        for earlier, later in zip(chronology_ratios, chronology_ratios[1:])
    )
    training_pass = bool(aggregate_training_fit) and all(aggregate_training_fit)

    group = str(rules["per_regime_generalization"]["group"])
    gaps: dict[str, Any] = {}
    positive_regimes = 0
    for experiment in experiments:
        training = np.asarray(
            [
                seed_result["training"]["by_regime"][experiment]["groups"][group][
                    "ratio_to_persistence"
                ]
                for seed_result in per_seed
            ],
            dtype=np.float64,
        )
        validation = np.asarray(
            [
                seed_result["validation"]["by_regime"][experiment]["groups"][group][
                    "ratio_to_persistence"
                ]
                for seed_result in per_seed
            ],
            dtype=np.float64,
        )
        training_median = float(np.median(training))
        validation_median = float(np.median(validation))
        gap = validation_median - training_median
        positive = gap > 0
        positive_regimes += int(positive)
        gaps[experiment] = {
            "training_seed_median_ratio": training_median,
            "validation_seed_median_ratio": validation_median,
            "validation_minus_training": gap,
            "positive_generalization_gap": positive,
        }
    required_regimes = int(
        rules["per_regime_generalization"][
            "minimum_regimes_with_seed_median_validation_ratio_above_training_ratio"
        ]
    )
    per_regime_pass = positive_regimes >= required_regimes

    effective_evidence: dict[str, Any] = {}
    effective_pass = True
    maximum = float(
        rules["effective_independence"][
            "maximum_effective_state_samples_total_per_group"
        ]
    )
    for slow_group in rules["effective_independence"]["groups"]:
        observed = float(
            autocorrelation["state_rms"][slow_group]["effective_samples_total"]
        )
        passed = observed <= maximum
        effective_pass &= passed
        effective_evidence[slow_group] = {
            "effective_samples_total": observed,
            "maximum_for_data_limited_decision": maximum,
            "data_limited": passed,
        }

    checks = {
        "aggregate_training_fit": training_pass,
        "strictly_improving_chronology": chronology_pass,
        "per_regime_generalization_gap": per_regime_pass,
        "limited_effective_slow_state_coverage": effective_pass,
    }
    authorized = all(checks.values())
    return {
        "checks": checks,
        "chronology_ratios": chronology_ratios,
        "per_regime_ssh_generalization": gaps,
        "positive_generalization_regime_count": positive_regimes,
        "required_generalization_regime_count": required_regimes,
        "effective_independence": effective_evidence,
        "expansion_authorized": authorized,
        "status": (
            "authorize_trajectories_v2_effective_coverage_expansion"
            if authorized
            else "retain_trajectories_v1_and_diagnose_non_data_bottleneck"
        ),
    }


def _chronology_evidence(
    final_decision: Mapping[str, Any],
    candidate_id: str,
) -> list[dict[str, Any]]:
    root = Path(str(final_decision["architecture_selection_manifest"])).resolve().parent
    result = []
    for stage in range(1, 5):
        selection_path = root / f"stage_{stage}_selection.json"
        selection = json.loads(selection_path.read_text())
        matches = [
            item
            for item in selection["ranking"]
            if item["candidate_id"] == candidate_id
        ]
        if len(matches) != 1:
            raise ModelCDataAdequacyError(
                f"{candidate_id} is missing from validation stage {stage}"
            )
        item = matches[0]
        report_path = Path(item["report"]).resolve()
        if _file_sha256(report_path) != item["report_sha256"]:
            raise ModelCDataAdequacyError("validation-stage report hash changed")
        report = json.loads(report_path.read_text())
        result.append(
            {
                "stage": stage,
                "chronology_fraction": float(
                    report["round_contract"]["chronology_fraction"]
                ),
                "optimizer_steps": int(report["round_contract"]["maximum_steps"]),
                "validation_worst_group_ratio": float(
                    report["selected_checkpoint"]["validation_ten_day"][
                        "worst_group_ratio"
                    ]
                ),
                "report": str(report_path),
                "report_sha256": item["report_sha256"],
            }
        )
    return result


def run_data_adequacy_audit(
    dataset_path: str | Path,
    training_diagnostics_path: str | Path,
    final_decision_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 16,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate fixed checkpoints and decide whether v2 data are authorized."""

    require_model_a_runtime()
    contract, resolved_contract, contract_sha = load_data_adequacy_contract(
        contract_path
    )
    dataset_path = Path(dataset_path).resolve()
    diagnostics_path = Path(training_diagnostics_path).resolve()
    final_path = Path(final_decision_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite data-adequacy output: {output}")
    if batch_size <= 0:
        raise ValueError("data-adequacy batch size must be positive")

    source_hashes = {
        "dataset_metadata_sha256": _file_sha256(dataset_path / ".zmetadata"),
        "training_diagnostics_sha256": _file_sha256(diagnostics_path),
        "final_seed_decision_sha256": _file_sha256(final_path),
        "search_contract_sha256": _file_sha256(
            Path(json.loads(final_path.read_text())["search_contract"]).resolve()
        ),
    }
    if source_hashes != contract["source_hashes"]:
        raise ModelCDataAdequacyError("data-adequacy source hashes changed")
    diagnostics = json.loads(diagnostics_path.read_text())
    final_decision = json.loads(final_path.read_text())
    if (
        final_decision.get("status") != "scientifically_rejected_three_seed_gate"
        or final_decision.get("configuration_frozen") is not False
        or final_decision.get("inference_authorized") is not False
    ):
        raise ModelCDataAdequacyError("unexpected final-seed decision state")

    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    if set(np.unique(pair_codes)) != {0, 1, 2, 3}:
        raise ModelCDataAdequacyError("dataset pair-split contract changed")
    experiments = tuple(str(value) for value in group.attrs["experiments"])
    if experiments != ("S0", "S1", "S2"):
        raise ModelCDataAdequacyError("unexpected training wind regimes")

    chronology = _chronology_evidence(final_decision, contract["candidate_id"])
    device = _device(device_name)
    started = time.monotonic()
    arrays: dict[str, np.ndarray] = {}
    per_seed: list[dict[str, Any]] = []
    aggregate_training_fit: list[bool] = []

    for artifact in final_decision["seed_artifacts"]:
        report_path = Path(artifact["report"]).resolve()
        checkpoint_path = Path(artifact["checkpoint"]).resolve()
        if (
            _file_sha256(report_path) != artifact["report_sha256"]
            or _file_sha256(checkpoint_path) != artifact["checkpoint_sha256"]
        ):
            raise ModelCDataAdequacyError("final-seed artifact hash changed")
        report = json.loads(report_path.read_text())
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if (
            checkpoint["candidate_id"] != contract["candidate_id"]
            or int(checkpoint["training_seed"]) != int(artifact["training_seed"])
            or checkpoint["dataset_metadata_sha256"]
            != source_hashes["dataset_metadata_sha256"]
        ):
            raise ModelCDataAdequacyError("final-seed checkpoint contract changed")
        architecture = _architecture_from_mapping(checkpoint["architecture"])
        model = build_model_c(architecture).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        training_records = tuple(
            (int(record[0]), int(record[1]))
            for record in checkpoint["training_records"]
        )
        validation_records = tuple(
            (int(record[0]), int(record[1]))
            for record in checkpoint["validation_records"]
        )
        if (
            any(pair_codes[time_index] != 1 for _, time_index in training_records)
            or any(pair_codes[time_index] != 2 for _, time_index in validation_records)
        ):
            raise ModelCDataAdequacyError("checkpoint record list crosses a sealed split")
        training_dataset = ModelAPairDataset(dataset_path, training_records)
        validation_dataset = ModelAPairDataset(dataset_path, validation_records)
        training_errors = _record_mse(
            model,
            training_dataset,
            batch_size=batch_size,
            device=device,
        )
        validation_errors = _record_mse(
            model,
            validation_dataset,
            batch_size=batch_size,
            device=device,
        )
        seed = int(artifact["training_seed"])
        prefix = f"seed_{seed}"
        arrays[f"{prefix}_training_records"] = np.asarray(
            training_records, dtype=np.int64
        )
        arrays[f"{prefix}_validation_records"] = np.asarray(
            validation_records, dtype=np.int64
        )
        for split_name, errors in (
            ("training", training_errors),
            ("validation", validation_errors),
        ):
            for state_group, values in errors.items():
                arrays[f"{prefix}_{split_name}_{state_group}_model_mse"] = values[
                    "model"
                ]
                arrays[
                    f"{prefix}_{split_name}_{state_group}_persistence_mse"
                ] = values["persistence"]
        training_summary = summarize_record_mse(
            training_records,
            training_errors,
            experiments,
        )
        validation_summary = summarize_record_mse(
            validation_records,
            validation_errors,
            experiments,
        )
        aggregate_fit = bool(
            report["selected_checkpoint"]["training_ten_day"][
                "all_groups_beat_persistence"
            ]
        )
        aggregate_training_fit.append(aggregate_fit)

        bootstrap: dict[str, Any] = {}
        validation_experiments = np.asarray(
            [record[0] for record in validation_records], dtype=np.int64
        )
        for group_name in GROUP_SLICES:
            block_lengths = [
                math.ceil(
                    float(
                        diagnostics["autocorrelation"]["increment_rms"][group_name][
                            "by_regime"
                        ][index]["tau_days"]
                    )
                )
                for index in range(len(experiments))
            ]
            model_by_regime = [
                validation_errors[group_name]["model"][
                    validation_experiments == index
                ]
                for index in range(len(experiments))
            ]
            persistence_by_regime = [
                validation_errors[group_name]["persistence"][
                    validation_experiments == index
                ]
                for index in range(len(experiments))
            ]
            bootstrap[group_name] = {
                "block_lengths_days": block_lengths,
                "all_regimes": circular_block_bootstrap_ratio(
                    model_by_regime,
                    persistence_by_regime,
                    block_lengths,
                    replicates=int(contract["bootstrap"]["replicates"]),
                    seed=int(contract["bootstrap"]["seed"]) + seed,
                    confidence_level=float(
                        contract["bootstrap"]["confidence_level"]
                    ),
                ),
            }
        per_seed.append(
            {
                "training_seed": seed,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": artifact["checkpoint_sha256"],
                "aggregate_training_fit_from_frozen_report": aggregate_fit,
                "training": training_summary,
                "validation": validation_summary,
                "validation_block_bootstrap": bootstrap,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    decision = data_expansion_decision(
        contract,
        chronology,
        aggregate_training_fit,
        per_seed,
        diagnostics["autocorrelation"],
        experiments,
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    arrays_path = temporary / ARRAY_NAME
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "status": "complete",
        "purpose": "post_search_model_c_data_adequacy_decision",
        "version": CONTRACT_VERSION,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "source_hashes": source_hashes,
        "candidate_id": contract["candidate_id"],
        "experiments": list(experiments),
        "read_contract": {
            "pair_split_codes_read": [1, 2],
            "validation_read": True,
            "inference_read": False,
            "intermediate_wind_read": False,
            "response_read": False,
            "adjoint_read": False,
        },
        "chronology_evidence": chronology,
        "training_autocorrelation": diagnostics["autocorrelation"],
        "per_seed": per_seed,
        "decision": decision,
        "arrays": ARRAY_NAME,
        "arrays_sha256": _file_sha256(arrays_path),
        "device": str(device),
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
    parser.add_argument("--training-diagnostics", type=Path, required=True)
    parser.add_argument("--final-decision", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_data_adequacy_audit(
        args.dataset,
        args.training_diagnostics,
        args.final_decision,
        args.contract,
        args.output_dir,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
