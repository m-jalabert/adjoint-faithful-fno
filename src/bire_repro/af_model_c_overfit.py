"""Training-only memorization and reload gate for forward-optimized Model C."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .af_model_a import _checkpoint_state_dict, require_model_a_runtime, seed_everything
from .af_model_b import (
    ModelBRolloutDataset,
    _unroll,
    stratified_rollout_records,
    western_boundary_mask,
)
from .af_model_c import (
    GROUP_SLICES,
    ModelCArchitecture,
    ModelCLossConfig,
    build_model_c,
    group_increment_nrmse_terms,
    loss_contract,
    loss_contract_sha256,
    model_c_architecture,
    model_c_loss_terms,
)

try:  # Keep documentation imports usable without the optional ML stack.
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    TensorDataset = None  # type: ignore[assignment,misc]


AUDIT_TERMS = (
    "total",
    "state",
    "increment",
    "rollout",
    "spectral",
    "boundary",
    *(f"state_{name}" for name in GROUP_SLICES),
    *(f"increment_{name}" for name in GROUP_SLICES),
)


class ModelCOverfitError(RuntimeError):
    """Raised when the Model C memorization gate is not trustworthy."""


@dataclass(frozen=True)
class ModelCOverfitConfig:
    """Predeclared stronger gate on 96 balanced training rollouts."""

    sample_count: int = 96
    batch_size: int = 4
    epochs: int = 160
    evaluation_interval: int = 5
    learning_rates: tuple[float, ...] = (1.0e-3, 5.0e-4)
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1.0e-5
    seed: int = 20260723
    minimum_total_reduction_fraction: float = 0.90
    minimum_spectral_reduction_fraction: float = 0.50
    maximum_state_group: float = 0.08
    maximum_increment_group_ratio_to_persistence: float = 1.00
    maximum_rollout: float = 0.10
    maximum_boundary: float = 0.15

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "learning_rates", tuple(float(value) for value in self.learning_rates)
        )
        if not 20 <= self.sample_count <= 100:
            raise ValueError("Model C overfit uses 20--100 training rollouts")
        if min(self.batch_size, self.epochs, self.evaluation_interval) <= 0:
            raise ValueError("Model C overfit counts must be positive")
        if self.epochs % self.evaluation_interval:
            raise ValueError("Model C overfit epochs must align with evaluation intervals")
        if not self.learning_rates or any(value <= 0 for value in self.learning_rates):
            raise ValueError("Model C overfit learning rates must be positive")
        if len(self.adam_betas) != 2 or any(not 0 <= value < 1 for value in self.adam_betas):
            raise ValueError("Model C Adam betas must lie in [0, 1)")
        if self.weight_decay < 0 or self.seed < 0:
            raise ValueError("Model C optimizer settings are invalid")
        fractions = (
            self.minimum_total_reduction_fraction,
            self.minimum_spectral_reduction_fraction,
        )
        if any(not 0 < value < 1 for value in fractions):
            raise ValueError("Model C reduction thresholds must lie in (0, 1)")
        maxima = (
            self.maximum_state_group,
            self.maximum_increment_group_ratio_to_persistence,
            self.maximum_rollout,
            self.maximum_boundary,
        )
        if any(value <= 0 for value in maxima):
            raise ValueError("Model C acceptance maxima must be positive")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overfit_acceptance(
    initial: dict[str, float],
    best: dict[str, float],
    persistence_increment: dict[str, float],
    config: ModelCOverfitConfig,
) -> dict[str, Any]:
    """Evaluate the groupwise memorization criteria without hiding a weak field."""

    required = set(AUDIT_TERMS)
    if not required <= initial.keys() or not required <= best.keys():
        raise ValueError("Model C overfit metrics are incomplete")
    values = np.asarray(
        [initial[name] for name in AUDIT_TERMS] + [best[name] for name in AUDIT_TERMS]
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Model C overfit metrics are non-finite")
    total_reduction = 1.0 - best["total"] / max(initial["total"], 1.0e-12)
    spectral_reduction = 1.0 - best["spectral"] / max(initial["spectral"], 1.0e-12)
    state_group_max = max(best[f"state_{name}"] for name in GROUP_SLICES)
    if set(persistence_increment) != set(GROUP_SLICES) or any(
        value <= 0 or not np.isfinite(value) for value in persistence_increment.values()
    ):
        raise ValueError("Model C persistence increment baseline is invalid")
    increment_group_ratios = {
        name: best[f"increment_{name}"] / persistence_increment[name]
        for name in GROUP_SLICES
    }
    increment_group_ratio_max = max(increment_group_ratios.values())
    criteria = {
        "total_reduction": total_reduction >= config.minimum_total_reduction_fraction,
        "spectral_reduction": (
            spectral_reduction >= config.minimum_spectral_reduction_fraction
        ),
        "state_group_max": state_group_max <= config.maximum_state_group,
        "increment_group_ratio_to_persistence": (
            increment_group_ratio_max
            <= config.maximum_increment_group_ratio_to_persistence
        ),
        "rollout": best["rollout"] <= config.maximum_rollout,
        "boundary": best["boundary"] <= config.maximum_boundary,
    }
    return {
        "accepted": all(criteria.values()),
        "criteria": criteria,
        "diagnostics": {
            "total_reduction_fraction": total_reduction,
            "spectral_reduction_fraction": spectral_reduction,
            "state_group_max": state_group_max,
            "increment_group_ratios_to_persistence": increment_group_ratios,
            "increment_group_ratio_max": increment_group_ratio_max,
            "rollout": best["rollout"],
            "boundary": best["boundary"],
        },
        "thresholds": {
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
        },
    }


def _device(name: str) -> Any:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Model C CUDA overfit requested without a visible GPU")
    return torch.device(name)


def _masks(
    dataset: ModelBRolloutDataset,
    device: Any,
    loss_config: ModelCLossConfig,
) -> tuple[Any, Any]:
    wet = torch.from_numpy(dataset.wet.astype(np.float32))[None, None].to(device)
    boundary_array = western_boundary_mask(
        dataset.wet,
        loss_config.western_boundary_width,
    )
    boundary = torch.from_numpy(boundary_array.astype(np.float32))[None, None].to(device)
    return wet, boundary


def materialize_rollouts(dataset: ModelBRolloutDataset) -> Any:
    """Cache the small memorization subset once instead of rereading Zarr each epoch."""

    if TensorDataset is None:
        raise RuntimeError("Model C rollout materialization requires PyTorch")
    examples = [dataset[index] for index in range(len(dataset))]
    if not examples:
        raise ModelCOverfitError("cannot materialize an empty Model C dataset")
    features = torch.stack([example[0] for example in examples])
    futures = torch.stack([example[1] for example in examples])
    if features.shape[0] != len(dataset) or futures.shape[0] != len(dataset):
        raise ModelCOverfitError("materialized Model C rollout count changed")
    return TensorDataset(features, futures)


def _epoch(
    model: Any,
    loader: Any,
    *,
    device: Any,
    wet: Any,
    boundary: Any,
    increment_scale: Any,
    loss_config: ModelCLossConfig,
    optimizer: Any | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals = {name: 0.0 for name in AUDIT_TERMS}
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
            raise ModelCOverfitError("Model C overfit encountered a non-finite loss")
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            optimizer.step()
        size = int(features.shape[0])
        for name in AUDIT_TERMS:
            totals[name] += float(terms[name].detach().cpu()) * size
        samples += size
    if not samples:
        raise ModelCOverfitError("Model C overfit loader produced no samples")
    return {name: value / samples for name, value in totals.items()}


def _persistence_increment_baseline(
    loader: Any,
    *,
    device: Any,
    wet: Any,
    increment_scale: Any,
) -> dict[str, float]:
    """Score zero increment on the exact sampled records used by the gate."""

    totals = {name: 0.0 for name in GROUP_SLICES}
    samples = 0
    for features, futures in loader:
        features = features.to(device=device, dtype=torch.float32)
        futures = futures.to(device=device, dtype=torch.float32)
        target_increment = futures[:, 0] - features[:, :46]
        terms = group_increment_nrmse_terms(
            torch.zeros_like(target_increment),
            target_increment,
            wet,
            increment_scale,
        )
        size = int(features.shape[0])
        for name in GROUP_SLICES:
            totals[name] += float(terms[name].detach().cpu()) * size
        samples += size
    if not samples:
        raise ModelCOverfitError("Model C persistence loader produced no samples")
    return {name: value / samples for name, value in totals.items()}


def _save_reload_check(
    model: Any,
    architecture: Any,
    features: Any,
    wet: Any,
    checkpoint_path: Path,
    payload: dict[str, Any],
    device: Any,
) -> bool:
    model.eval()
    with torch.no_grad():
        reference = _unroll(model, features, wet, 3).detach().cpu()
    torch.save(payload, checkpoint_path)
    restored = build_model_c(architecture).to(device)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored.load_state_dict(loaded["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        reloaded = _unroll(restored, features, wet, 3).detach().cpu()
    return bool(torch.equal(reference, reloaded))


def run_overfit(
    dataset_path: str | Path,
    diagnostics_path: str | Path,
    calibration_path: str | Path,
    output_dir: str | Path,
    *,
    config: ModelCOverfitConfig = ModelCOverfitConfig(),
    loss_config: ModelCLossConfig = ModelCLossConfig(),
    architecture: ModelCArchitecture = ModelCArchitecture(),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run the frozen-loss memorization and three-step reload gate."""

    require_model_a_runtime()
    seed_everything(config.seed)
    dataset_path = Path(dataset_path).resolve()
    diagnostics_path = Path(diagnostics_path).resolve()
    calibration_path = Path(calibration_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model C overfit output: {output}")
    diagnostics = json.loads(diagnostics_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    for name, report in (("diagnostics", diagnostics), ("calibration", calibration)):
        read_contract = report.get("read_contract", {})
        if (
            read_contract.get("validation_read") is not False
            or read_contract.get("inference_read") is not False
        ):
            raise ValueError(f"Model C {name} report does not preserve sealed splits")
    if Path(diagnostics.get("dataset", "")).resolve() != dataset_path:
        raise ValueError("Model C diagnostics and overfit datasets differ")
    if Path(calibration.get("dataset", "")).resolve() != dataset_path:
        raise ValueError("Model C calibration and overfit datasets differ")
    increment_values = np.asarray(
        diagnostics["increment_rms_normalized_state_units"],
        dtype=np.float32,
    )
    if increment_values.shape != (46,) or np.any(increment_values <= 0):
        raise ValueError("Model C overfit received invalid increment scales")

    device = _device(device_name)
    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    records = stratified_rollout_records(
        np.asarray(group["pair_split"][:], dtype=np.uint8),
        config.sample_count,
        config.seed,
    )
    source_dataset = ModelBRolloutDataset(dataset_path, records)
    wet, boundary = _masks(source_dataset, device, loss_config)
    dataset = materialize_rollouts(source_dataset)
    materialized_bytes = sum(
        int(tensor.numel() * tensor.element_size()) for tensor in dataset.tensors
    )
    evaluation_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    increment_scale = torch.from_numpy(increment_values).to(device)
    persistence_increment = _persistence_increment_baseline(
        evaluation_loader,
        device=device,
        wet=wet,
        increment_scale=increment_scale,
    )
    attempts: list[dict[str, Any]] = []
    rejected_best: tuple[float, dict[str, Any], dict[str, Any], int, float] | None = None
    started = time.monotonic()

    for learning_rate in config.learning_rates:
        # Keep initialization and batch order identical so the bounded fallback
        # changes only the learning rate.
        seed_everything(config.seed)
        train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(config.seed),
        )
        model = build_model_c(architecture).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=config.adam_betas,
            weight_decay=config.weight_decay,
        )
        initial = _epoch(
            model,
            evaluation_loader,
            device=device,
            wet=wet,
            boundary=boundary,
            increment_scale=increment_scale,
            loss_config=loss_config,
            optimizer=None,
        )
        history: list[dict[str, Any]] = []
        best_metrics: dict[str, float] | None = None
        best_epoch: int | None = None
        best_state: dict[str, Any] | None = None
        for epoch in range(1, config.epochs + 1):
            training = _epoch(
                model,
                train_loader,
                device=device,
                wet=wet,
                boundary=boundary,
                increment_scale=increment_scale,
                loss_config=loss_config,
                optimizer=optimizer,
            )
            record: dict[str, Any] = {"epoch": epoch, "training": training}
            if epoch % config.evaluation_interval == 0:
                evaluation = _epoch(
                    model,
                    evaluation_loader,
                    device=device,
                    wet=wet,
                    boundary=boundary,
                    increment_scale=increment_scale,
                    loss_config=loss_config,
                    optimizer=None,
                )
                record["evaluation"] = evaluation
                if best_metrics is None or evaluation["total"] < best_metrics["total"]:
                    best_metrics = evaluation
                    best_epoch = epoch
                    best_state = _checkpoint_state_dict(model)
            history.append(record)
        if best_metrics is None or best_state is None or best_epoch is None:
            raise ModelCOverfitError("Model C overfit did not evaluate a checkpoint")
        acceptance = overfit_acceptance(
            initial,
            best_metrics,
            persistence_increment,
            config,
        )
        attempts.append(
            {
                "learning_rate": learning_rate,
                "initial": initial,
                "best_epoch": best_epoch,
                "best": best_metrics,
                "acceptance": acceptance,
                "history": history,
            }
        )
        if not acceptance["accepted"]:
            if rejected_best is None or best_metrics["total"] < rejected_best[0]:
                rejected_best = (
                    best_metrics["total"],
                    best_state,
                    best_metrics,
                    best_epoch,
                    learning_rate,
                )
            continue

        model.load_state_dict(best_state)
        output.mkdir(parents=True, exist_ok=False)
        checkpoint_path = output / "model_c_overfit_best.pt"
        features, _ = dataset[0]
        features = features[None].to(device=device, dtype=torch.float32)
        payload = {
            "model_class": "ModelCForwardOptimizedResidualFNO",
            "architecture": architecture.to_dict(),
            "overfit_config": asdict(config),
            "loss_contract": loss_contract(loss_config),
            "loss_contract_sha256": loss_contract_sha256(loss_config),
            "dataset": str(dataset_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _file_sha256(diagnostics_path),
            "calibration": str(calibration_path),
            "calibration_sha256": _file_sha256(calibration_path),
            "records": [list(record) for record in records],
            "training_subset_cache": {
                "materialized_once": True,
                "samples": len(dataset),
                "bytes": materialized_bytes,
                "source": "immutable Zarr records listed in this checkpoint",
            },
            "persistence_increment_nrmse": persistence_increment,
            "model_state_dict": _checkpoint_state_dict(model),
            "history": history,
        }
        bitwise_exact = _save_reload_check(
            model,
            architecture,
            features,
            wet,
            checkpoint_path,
            payload,
            device,
        )
        if not bitwise_exact:
            raise ModelCOverfitError("Model C three-step checkpoint reload is not exact")
        report = {
            "status": "accepted",
            "purpose": "training_only_model_c_memorization_and_reload_gate",
            "dataset": str(dataset_path),
            "dataset_metadata_sha256": _file_sha256(dataset_path / ".zmetadata"),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _file_sha256(diagnostics_path),
            "calibration": str(calibration_path),
            "calibration_sha256": _file_sha256(calibration_path),
            "read_contract": {
                "pair_split_codes_read": [1],
                "validation_read": False,
                "inference_read": False,
                "response_or_adjoint_read": False,
            },
            "device": str(device),
            "neuraloperator_version": metadata.version("neuraloperator"),
            "architecture": architecture.to_dict(),
            "parameter_count": int(sum(value.numel() for value in model.parameters())),
            "config": asdict(config),
            "loss_contract": loss_contract(loss_config),
            "loss_contract_sha256": loss_contract_sha256(loss_config),
            "records": [list(record) for record in records],
            "training_subset_cache": {
                "materialized_once": True,
                "samples": len(dataset),
                "bytes": materialized_bytes,
                "source": "immutable Zarr records listed in this report",
            },
            "persistence_increment_nrmse": persistence_increment,
            "attempts": attempts,
            "selected_learning_rate": learning_rate,
            "selected_epoch": best_epoch,
            "save_reload_three_step_bitwise_exact": bitwise_exact,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "elapsed_seconds": time.monotonic() - started,
        }
        (output / "model_c_overfit_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        return report

    if rejected_best is None:
        raise ModelCOverfitError("Model C rejected all attempts without a checkpoint")
    _, rejected_state, rejected_metrics, rejected_epoch, rejected_learning_rate = rejected_best
    model = build_model_c(architecture).to(device)
    model.load_state_dict(rejected_state)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "model_c_overfit_rejected_best.pt"
    features, _ = dataset[0]
    features = features[None].to(device=device, dtype=torch.float32)
    payload = {
        "model_class": "ModelCForwardOptimizedResidualFNO",
        "status": "rejected_diagnostic_checkpoint",
        "architecture": architecture.to_dict(),
        "overfit_config": asdict(config),
        "loss_contract": loss_contract(loss_config),
        "loss_contract_sha256": loss_contract_sha256(loss_config),
        "dataset": str(dataset_path),
        "diagnostics": str(diagnostics_path),
        "calibration": str(calibration_path),
        "records": [list(record) for record in records],
        "training_subset_cache": {
            "materialized_once": True,
            "samples": len(dataset),
            "bytes": materialized_bytes,
            "source": "immutable Zarr records listed in this checkpoint",
        },
        "persistence_increment_nrmse": persistence_increment,
        "selected_learning_rate": rejected_learning_rate,
        "selected_epoch": rejected_epoch,
        "selected_metrics": rejected_metrics,
        "model_state_dict": _checkpoint_state_dict(model),
    }
    bitwise_exact = _save_reload_check(
        model,
        architecture,
        features,
        wet,
        checkpoint_path,
        payload,
        device,
    )
    if not bitwise_exact:
        raise ModelCOverfitError("rejected Model C checkpoint reload is not exact")
    rejected_report = {
        "status": "rejected",
        "purpose": "training_only_model_c_memorization_and_reload_gate",
        "dataset": str(dataset_path),
        "dataset_metadata_sha256": _file_sha256(dataset_path / ".zmetadata"),
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": _file_sha256(diagnostics_path),
        "calibration": str(calibration_path),
        "calibration_sha256": _file_sha256(calibration_path),
        "read_contract": {
            "pair_split_codes_read": [1],
            "validation_read": False,
            "inference_read": False,
            "response_or_adjoint_read": False,
        },
        "device": str(device),
        "architecture": architecture.to_dict(),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "config": asdict(config),
        "loss_contract": loss_contract(loss_config),
        "loss_contract_sha256": loss_contract_sha256(loss_config),
        "records": [list(record) for record in records],
        "training_subset_cache": {
            "materialized_once": True,
            "samples": len(dataset),
            "bytes": materialized_bytes,
            "source": "immutable Zarr records listed in this report",
        },
        "persistence_increment_nrmse": persistence_increment,
        "attempts": attempts,
        "diagnostic_checkpoint": str(checkpoint_path),
        "diagnostic_checkpoint_sha256": _file_sha256(checkpoint_path),
        "diagnostic_checkpoint_learning_rate": rejected_learning_rate,
        "diagnostic_checkpoint_epoch": rejected_epoch,
        "save_reload_three_step_bitwise_exact": bitwise_exact,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "model_c_overfit_rejected_report.json").write_text(
        json.dumps(rejected_report, indent=2, sort_keys=True) + "\n"
    )
    raise ModelCOverfitError(
        f"Model C failed the groupwise memorization gate; see {output}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Model C memorization gate")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--modes-y", type=int, default=16)
    parser.add_argument("--modes-x", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, choices=(32, 64), default=32)
    parser.add_argument("--layers", type=int, choices=(4, 6), default=4)
    parser.add_argument("--padding", type=float, choices=(0.10, 0.20), default=0.10)
    parser.add_argument(
        "--learning-rate",
        type=float,
        action="append",
        dest="learning_rates",
        help="bounded learning rate; repeat to declare controlled fallbacks",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_overfit(
        args.dataset,
        args.diagnostics,
        args.calibration,
        args.output_dir,
        config=ModelCOverfitConfig(
            sample_count=args.samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rates=(
                tuple(args.learning_rates)
                if args.learning_rates is not None
                else ModelCOverfitConfig().learning_rates
            ),
        ),
        architecture=model_c_architecture(
            n_modes=(args.modes_y, args.modes_x),
            hidden_channels=args.hidden_channels,
            n_layers=args.layers,
            domain_padding=args.padding,
        ),
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
