"""Training-only learning-history and late-checkpoint audit for Model C.

This audit is deliberately downstream of the rejected 96-rollout memorization
runs.  It reads the exact record list and diagnostic checkpoint saved by one
run, independently verifies that every 30-day rollout remains inside the
training split, and measures full-subset component gradients.  Validation,
inference, intermediate-wind, response, and adjoint data are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..af_model_a import require_model_a_runtime, seed_everything
from ..af_model_b import ModelBRolloutDataset, _unroll, western_boundary_mask
from ..af_model_c import (
    GROUP_SLICES,
    ModelCArchitecture,
    ModelCLossConfig,
    build_model_c,
    loss_config_from_contract,
    loss_contract_sha256,
    model_c_loss_terms,
)
from ..af_model_c_overfit import (
    AUDIT_TERMS,
    ModelCOverfitConfig,
    materialize_rollouts,
    overfit_acceptance,
)

try:  # Keep documentation-only imports usable without the optional ML stack.
    import torch
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


COMPONENT_TERMS = ("state", "increment", "rollout", "spectral", "boundary")
INCREMENT_GROUP_TERMS = tuple(f"increment_{name}" for name in GROUP_SLICES)
GRADIENT_TERMS = ("total", *COMPONENT_TERMS, *INCREMENT_GROUP_TERMS)


class ModelCObjectiveAuditError(RuntimeError):
    """Raised when a late-checkpoint audit cannot establish its provenance."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1.0e-9, atol=1.0e-12))


def _weighted_loss_contributions(
    metrics: dict[str, float],
    loss_config: ModelCLossConfig,
) -> dict[str, dict[str, float]]:
    weights = {
        "state": 1.0,
        "increment": loss_config.increment_weight,
        "rollout": loss_config.rollout_weight,
        "spectral": loss_config.spectral_weight,
        "boundary": loss_config.boundary_weight,
    }
    weighted = {name: float(metrics[name] * weight) for name, weight in weights.items()}
    reconstructed = sum(weighted.values())
    return {
        name: {
            "raw_value": float(metrics[name]),
            "weight": weight,
            "weighted_value": weighted[name],
            "weighted_value_fraction_of_total": weighted[name]
            / max(float(metrics["total"]), 1.0e-30),
            "weighted_value_ratio_to_state": weighted[name]
            / max(weighted["state"], 1.0e-30),
        }
        for name, weight in weights.items()
    } | {
        "reconstruction": {
            "reported_total": float(metrics["total"]),
            "component_sum": reconstructed,
            "absolute_difference": abs(reconstructed - float(metrics["total"])),
        }
    }


def _extreme_epoch(
    evaluated: list[dict[str, Any]],
    key: str,
    *,
    minimum: bool = True,
) -> dict[str, Any]:
    operation = min if minimum else max
    selected = operation(evaluated, key=lambda item: float(item[key]))
    return {"epoch": int(selected["epoch"]), "value": float(selected[key])}


def summarize_learning_history(overfit_report: dict[str, Any]) -> dict[str, Any]:
    """Re-evaluate every saved epoch against every predeclared gate criterion."""

    attempts = overfit_report.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("Model C overfit report contains no attempts")
    config = ModelCOverfitConfig(**overfit_report["config"])
    loss_config = loss_config_from_contract(overfit_report["loss_contract"])
    persistence = {
        name: float(value)
        for name, value in overfit_report["persistence_increment_nrmse"].items()
    }
    attempt_summaries = []
    for attempt in attempts:
        history = attempt.get("history", [])
        expected_epochs = list(
            range(config.evaluation_interval, config.epochs + 1, config.evaluation_interval)
        )
        evaluated_rows = [row for row in history if "evaluation" in row]
        actual_epochs = [int(row["epoch"]) for row in evaluated_rows]
        if len(history) != config.epochs or actual_epochs != expected_epochs:
            raise ModelCObjectiveAuditError(
                "Model C history does not contain every declared epoch/evaluation"
            )

        initial = {name: float(attempt["initial"][name]) for name in AUDIT_TERMS}
        evaluated: list[dict[str, Any]] = []
        for row in evaluated_rows:
            metrics = {name: float(row["evaluation"][name]) for name in AUDIT_TERMS}
            acceptance = overfit_acceptance(initial, metrics, persistence, config)
            ratios = acceptance["diagnostics"]["increment_group_ratios_to_persistence"]
            evaluated.append(
                {
                    "epoch": int(row["epoch"]),
                    "metrics": metrics,
                    "criteria": acceptance["criteria"],
                    "accepted": bool(acceptance["accepted"]),
                    "increment_group_ratios_to_persistence": ratios,
                    "worst_increment_group_ratio": float(max(ratios.values())),
                    "weighted_loss_contributions": _weighted_loss_contributions(
                        metrics, loss_config
                    ),
                }
            )

        best_total = min(evaluated, key=lambda item: item["metrics"]["total"])
        best_balanced = min(
            evaluated,
            key=lambda item: (
                item["worst_increment_group_ratio"],
                item["metrics"]["total"],
                item["epoch"],
            ),
        )
        if (
            int(attempt["best_epoch"]) != best_total["epoch"]
            or any(
                not _metric_close(float(attempt["best"][name]), best_total["metrics"][name])
                for name in AUDIT_TERMS
            )
        ):
            raise ModelCObjectiveAuditError(
                "saved Model C best checkpoint disagrees with its complete history"
            )
        saved_acceptance = attempt["acceptance"]
        recomputed_acceptance = best_total["accepted"]
        if bool(saved_acceptance["accepted"]) != recomputed_acceptance:
            raise ModelCObjectiveAuditError(
                "saved Model C acceptance disagrees with the recomputed best epoch"
            )

        criterion_names = tuple(best_total["criteria"])
        criterion_summary = {}
        for name in criterion_names:
            passing = [row["epoch"] for row in evaluated if row["criteria"][name]]
            failing = [row["epoch"] for row in evaluated if not row["criteria"][name]]
            criterion_summary[name] = {
                "pass_count": len(passing),
                "fail_count": len(failing),
                "first_pass_epoch": passing[0] if passing else None,
                "last_fail_epoch": failing[-1] if failing else None,
            }

        group_minima = {}
        for group_name in GROUP_SLICES:
            key = f"increment_{group_name}"
            selected = min(
                evaluated,
                key=lambda item: item["increment_group_ratios_to_persistence"][group_name],
            )
            group_minima[group_name] = {
                "epoch": selected["epoch"],
                "nrmse": selected["metrics"][key],
                "ratio_to_persistence": selected[
                    "increment_group_ratios_to_persistence"
                ][group_name],
            }
        all_non_increment_pass = [
            row["epoch"]
            for row in evaluated
            if all(
                passed
                for name, passed in row["criteria"].items()
                if name != "increment_group_ratio_to_persistence"
            )
        ]
        attempt_summaries.append(
            {
                "learning_rate": float(attempt["learning_rate"]),
                "training_epoch_count": len(history),
                "evaluation_count": len(evaluated),
                "evaluated_epoch_range": [actual_epochs[0], actual_epochs[-1]],
                "evaluated_epoch_stride": config.evaluation_interval,
                "criterion_summary": criterion_summary,
                "all_non_increment_criteria_pass_count": len(all_non_increment_pass),
                "first_all_non_increment_criteria_pass_epoch": (
                    all_non_increment_pass[0] if all_non_increment_pass else None
                ),
                "any_epoch_accepted": any(row["accepted"] for row in evaluated),
                "best_total": best_total,
                "best_balanced_increment": best_balanced,
                "increment_group_minima": group_minima,
                "total_minimum": _extreme_epoch(
                    [
                        {"epoch": row["epoch"], "value": row["metrics"]["total"]}
                        for row in evaluated
                    ],
                    "value",
                ),
                "evaluated_epochs": evaluated,
            }
        )

    return {
        "status": "complete",
        "predeclared_criteria": {
            "minimum_total_reduction_fraction": config.minimum_total_reduction_fraction,
            "minimum_spectral_reduction_fraction": (
                config.minimum_spectral_reduction_fraction
            ),
            "maximum_state_group": config.maximum_state_group,
            "maximum_increment_group_ratio_to_persistence": (
                config.maximum_increment_group_ratio_to_persistence
            ),
            "maximum_rollout": config.maximum_rollout,
            "maximum_boundary": config.maximum_boundary,
            "save_reload_three_step_bitwise_exact": True,
        },
        "persistence_increment_nrmse": persistence,
        "save_reload_three_step_bitwise_exact": bool(
            overfit_report["save_reload_three_step_bitwise_exact"]
        ),
        "attempts": attempt_summaries,
    }


def gradient_inner_product(
    left: Iterable[Any | None],
    right: Iterable[Any | None],
    device: Any,
) -> Any:
    """Return the real parameter-space inner product, including complex weights."""

    result = torch.zeros((), dtype=torch.float64, device=device)
    for left_value, right_value in zip(left, right):
        if left_value is None or right_value is None:
            continue
        product = left_value.conj() * right_value
        result = result + product.real.sum(dtype=torch.float64)
    return result


def gradient_l2_norm(gradient: Iterable[Any | None], device: Any) -> Any:
    """Return a complex-safe float64 norm for an aggregated parameter gradient."""

    squared = gradient_inner_product(gradient, gradient, device)
    norm = torch.sqrt(squared.clamp_min(0.0))
    if not bool(torch.isfinite(norm).item()):
        raise ModelCObjectiveAuditError("late-checkpoint gradient norm is non-finite")
    return norm


def _scaled_gradient(
    gradient: tuple[Any | None, ...],
    scale: float,
) -> tuple[Any | None, ...]:
    return tuple(None if value is None else value * scale for value in gradient)


def _sum_gradients(
    gradients: Iterable[tuple[tuple[Any | None, ...], float]],
) -> tuple[Any | None, ...]:
    items = tuple(gradients)
    if not items:
        raise ValueError("cannot sum an empty gradient collection")
    length = len(items[0][0])
    result: list[Any | None] = []
    for index in range(length):
        values = [
            gradient[index] * weight
            for gradient, weight in items
            if gradient[index] is not None
        ]
        result.append(sum(values[1:], values[0].clone()) if values else None)
    return tuple(result)


def _relative_gradient_difference(
    left: tuple[Any | None, ...],
    right: tuple[Any | None, ...],
    device: Any,
) -> float:
    difference = _sum_gradients(((left, 1.0), (right, -1.0)))
    denominator = gradient_l2_norm(right, device).clamp_min(1.0e-30)
    return float((gradient_l2_norm(difference, device) / denominator).cpu())


def _component_gradient_audit(
    model: Any,
    loader: Any,
    *,
    device: Any,
    wet: Any,
    boundary: Any,
    increment_scale: Any,
    loss_config: ModelCLossConfig,
) -> dict[str, Any]:
    """Aggregate gradients of full-subset mean losses before taking their norms."""

    model.eval()
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    accumulated: dict[str, list[Any | None]] = {
        name: [None] * len(parameters) for name in GRADIENT_TERMS
    }
    raw_totals = {name: 0.0 for name in AUDIT_TERMS}
    samples = 0
    for features, futures in loader:
        features = features.to(device=device, dtype=torch.float32, non_blocking=True)
        futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
        predictions = _unroll(model, features, wet, loss_config.rollout_steps)
        terms = model_c_loss_terms(
            predictions,
            futures,
            features[:, :46],
            wet,
            boundary,
            increment_scale,
            loss_config,
        )
        if not all(bool(torch.isfinite(terms[name]).item()) for name in AUDIT_TERMS):
            raise ModelCObjectiveAuditError("late checkpoint produced a non-finite loss")
        size = int(features.shape[0])
        for name in AUDIT_TERMS:
            raw_totals[name] += float(terms[name].detach().cpu()) * size
        for term_index, name in enumerate(GRADIENT_TERMS):
            gradients = torch.autograd.grad(
                terms[name],
                parameters,
                retain_graph=term_index < len(GRADIENT_TERMS) - 1,
                allow_unused=True,
            )
            for parameter_index, gradient in enumerate(gradients):
                if gradient is None:
                    continue
                audit_dtype = torch.complex128 if gradient.is_complex() else torch.float64
                contribution = gradient.detach().to(audit_dtype) * size
                current = accumulated[name][parameter_index]
                accumulated[name][parameter_index] = (
                    contribution if current is None else current + contribution
                )
        samples += size
    if not samples:
        raise ModelCObjectiveAuditError("late-checkpoint audit loader produced no samples")

    averaged = {
        name: tuple(
            None if value is None else value / samples for value in accumulated[name]
        )
        for name in GRADIENT_TERMS
    }
    raw_metrics = {name: value / samples for name, value in raw_totals.items()}
    norms = {
        name: float(gradient_l2_norm(averaged[name], device).cpu())
        for name in GRADIENT_TERMS
    }
    cosines: dict[str, dict[str, float | None]] = {}
    for left_name in GRADIENT_TERMS:
        cosines[left_name] = {}
        for right_name in GRADIENT_TERMS:
            denominator = norms[left_name] * norms[right_name]
            cosines[left_name][right_name] = (
                float(
                    (
                        gradient_inner_product(
                            averaged[left_name], averaged[right_name], device
                        )
                        / denominator
                    ).cpu()
                )
                if denominator > 0
                else None
            )

    weights = {
        "state": 1.0,
        "increment": loss_config.increment_weight,
        "rollout": loss_config.rollout_weight,
        "spectral": loss_config.spectral_weight,
        "boundary": loss_config.boundary_weight,
    }
    weighted_gradient = {
        name: {
            "raw_gradient_norm": norms[name],
            "weight": weight,
            "weighted_gradient_norm": norms[name] * weight,
            "weighted_gradient_ratio_to_state": norms[name]
            * weight
            / max(norms["state"], 1.0e-30),
            "weighted_gradient_ratio_to_total": norms[name]
            * weight
            / max(norms["total"], 1.0e-30),
        }
        for name, weight in weights.items()
    }
    group_weight = loss_config.increment_weight / len(GROUP_SLICES)
    increment_group_gradients = {
        group_name: {
            "raw_gradient_norm": norms[f"increment_{group_name}"],
            "effective_weight_in_total": group_weight,
            "weighted_gradient_norm": norms[f"increment_{group_name}"] * group_weight,
            "weighted_gradient_ratio_to_state": norms[f"increment_{group_name}"]
            * group_weight
            / max(norms["state"], 1.0e-30),
            "cosine_with_state": cosines[f"increment_{group_name}"]["state"],
            "cosine_with_total": cosines[f"increment_{group_name}"]["total"],
        }
        for group_name in GROUP_SLICES
    }
    reconstructed_total = _sum_gradients(
        (averaged[name], weight) for name, weight in weights.items()
    )
    reconstructed_increment = _sum_gradients(
        (averaged[f"increment_{name}"], 1.0 / len(GROUP_SLICES))
        for name in GROUP_SLICES
    )
    return {
        "sample_count": samples,
        "aggregation": (
            "gradient_of_the_full_96_record_mean_loss_not_mean_of_batch_gradient_norms"
        ),
        "raw_metrics": raw_metrics,
        "loss_contributions": _weighted_loss_contributions(raw_metrics, loss_config),
        "raw_gradient_norms": norms,
        "weighted_component_gradients": weighted_gradient,
        "increment_group_gradients": increment_group_gradients,
        "gradient_cosines": cosines,
        "linearity_checks": {
            "total_gradient_relative_reconstruction_error": (
                _relative_gradient_difference(
                    reconstructed_total, averaged["total"], device
                )
            ),
            "increment_gradient_relative_group_reconstruction_error": (
                _relative_gradient_difference(
                    reconstructed_increment, averaged["increment"], device
                )
            ),
        },
    }


def _verify_training_records(dataset_path: Path, records: tuple[tuple[int, int], ...]) -> None:
    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    if len(records) != 96 or {experiment for experiment, _ in records} != {0, 1, 2}:
        raise ModelCObjectiveAuditError("late-checkpoint audit requires the exact 96 records")
    for experiment, time_index in records:
        if experiment not in (0, 1, 2):
            raise ModelCObjectiveAuditError("record uses an undeclared forcing regime")
        if any(pair_codes[time_index + 10 * step] != 1 for step in range(3)):
            raise ModelCObjectiveAuditError("record crosses a non-training pair split")
        if any(snapshot_codes[time_index + 10 * step] != 1 for step in range(4)):
            raise ModelCObjectiveAuditError("record crosses a non-training snapshot split")


def audit_late_checkpoint(
    overfit_report_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 4,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Audit one rejected late checkpoint and its complete learning history."""

    require_model_a_runtime()
    report_path = Path(overfit_report_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model C objective audit: {output}")
    if batch_size <= 0:
        raise ValueError("Model C objective audit batch size must be positive")
    overfit_report = json.loads(report_path.read_text())
    read_contract = overfit_report.get("read_contract", {})
    if (
        read_contract.get("validation_read") is not False
        or read_contract.get("inference_read") is not False
        or read_contract.get("response_or_adjoint_read") is not False
    ):
        raise ModelCObjectiveAuditError("source overfit report did not preserve sealed data")
    if overfit_report.get("status") != "rejected":
        raise ModelCObjectiveAuditError("late-checkpoint diagnosis expects a rejected run")
    if not overfit_report.get("save_reload_three_step_bitwise_exact"):
        raise ModelCObjectiveAuditError("source checkpoint did not pass exact reload")

    dataset_path = Path(overfit_report["dataset"]).resolve()
    diagnostics_path = Path(overfit_report["diagnostics"]).resolve()
    checkpoint_path = Path(overfit_report["diagnostic_checkpoint"]).resolve()
    if _file_sha256(dataset_path / ".zmetadata") != overfit_report[
        "dataset_metadata_sha256"
    ]:
        raise ModelCObjectiveAuditError("dataset metadata changed after the source run")
    if _file_sha256(diagnostics_path) != overfit_report["diagnostics_sha256"]:
        raise ModelCObjectiveAuditError("Model C diagnostics changed after the source run")
    if _file_sha256(checkpoint_path) != overfit_report["diagnostic_checkpoint_sha256"]:
        raise ModelCObjectiveAuditError("Model C diagnostic checkpoint hash changed")
    if (
        loss_contract_sha256(loss_config_from_contract(overfit_report["loss_contract"]))
        != overfit_report["loss_contract_sha256"]
    ):
        raise ModelCObjectiveAuditError("Model C v1 loss contract hash changed")

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Model C CUDA objective audit requested without a visible GPU")
    device = torch.device(device_name)
    seed_everything(int(overfit_report["config"]["seed"]))

    records = tuple(
        (int(experiment), int(time_index))
        for experiment, time_index in overfit_report["records"]
    )
    _verify_training_records(dataset_path, records)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (
        checkpoint.get("records") != overfit_report["records"]
        or int(checkpoint.get("selected_epoch", -1))
        != int(overfit_report["diagnostic_checkpoint_epoch"])
        or float(checkpoint.get("selected_learning_rate", -1.0))
        != float(overfit_report["diagnostic_checkpoint_learning_rate"])
    ):
        raise ModelCObjectiveAuditError("checkpoint provenance disagrees with its report")

    architecture = ModelCArchitecture(**overfit_report["architecture"])
    loss_config = loss_config_from_contract(overfit_report["loss_contract"])
    model = build_model_c(architecture).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    source_dataset = ModelBRolloutDataset(dataset_path, records)
    wet = torch.from_numpy(source_dataset.wet.astype(np.float32))[None, None].to(device)
    boundary_array = western_boundary_mask(
        source_dataset.wet, loss_config.western_boundary_width
    )
    boundary = torch.from_numpy(boundary_array.astype(np.float32))[None, None].to(device)
    dataset = materialize_rollouts(source_dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    diagnostics = json.loads(diagnostics_path.read_text())
    increment_values = np.asarray(
        diagnostics["increment_rms_normalized_state_units"], dtype=np.float32
    )
    if increment_values.shape != (46,) or np.any(increment_values <= 0):
        raise ModelCObjectiveAuditError("diagnostic increment scales are invalid")
    increment_scale = torch.from_numpy(increment_values).to(device)

    started = time.monotonic()
    history_audit = summarize_learning_history(overfit_report)
    component_audit = _component_gradient_audit(
        model,
        loader,
        device=device,
        wet=wet,
        boundary=boundary,
        increment_scale=increment_scale,
        loss_config=loss_config,
    )
    selected_metrics = overfit_report["attempts"][0]["best"]
    metric_reproduction = {
        name: {
            "source": float(selected_metrics[name]),
            "audit": float(component_audit["raw_metrics"][name]),
            "absolute_difference": abs(
                float(selected_metrics[name]) - float(component_audit["raw_metrics"][name])
            ),
        }
        for name in AUDIT_TERMS
    }
    result = {
        "status": "complete",
        "purpose": "training_only_model_c_late_checkpoint_objective_diagnosis",
        "source_overfit_report": str(report_path),
        "source_overfit_report_sha256": _file_sha256(report_path),
        "diagnostic_checkpoint": str(checkpoint_path),
        "diagnostic_checkpoint_sha256": _file_sha256(checkpoint_path),
        "diagnostic_checkpoint_epoch": int(
            overfit_report["diagnostic_checkpoint_epoch"]
        ),
        "dataset": str(dataset_path),
        "dataset_metadata_sha256": _file_sha256(dataset_path / ".zmetadata"),
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": _file_sha256(diagnostics_path),
        "read_contract": {
            "pair_split_codes_read": [1],
            "snapshot_split_codes_read": [1],
            "validation_read": False,
            "inference_read": False,
            "intermediate_wind_read": False,
            "response_or_adjoint_read": False,
        },
        "records": [list(record) for record in records],
        "architecture": architecture.to_dict(),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "loss_contract": overfit_report["loss_contract"],
        "loss_contract_sha256": overfit_report["loss_contract_sha256"],
        "device": str(device),
        "device_metadata": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "batch_size": batch_size,
        "history_audit": history_audit,
        "checkpoint_component_audit": component_audit,
        "selected_metric_reproduction": metric_reproduction,
        "elapsed_seconds": time.monotonic() - started,
    }
    output.mkdir(parents=True, exist_ok=False)
    report_output = output / "model_c_late_checkpoint_objective_audit.json"
    report_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a rejected Model C late checkpoint using training only"
    )
    parser.add_argument("--overfit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_late_checkpoint(
        args.overfit_report,
        args.output_dir,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    summary = {
        "status": result["status"],
        "source_overfit_report": result["source_overfit_report"],
        "diagnostic_checkpoint_epoch": result["diagnostic_checkpoint_epoch"],
        "output_dir": str(args.output_dir.resolve()),
        "elapsed_seconds": result["elapsed_seconds"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
