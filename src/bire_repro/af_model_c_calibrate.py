"""Training-only loss-scale calibration for Model C.

The calibration is deliberately separate from candidate selection.  It warms
the width-32, 16-mode starting model on 96 training rollouts using only the
state, increment, and rollout core, then measures raw component values and
gradient norms.  It proposes auxiliary weights for human review; it does not
open validation or save a candidate checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .af_model_a import _checkpoint_state_dict, require_model_a_runtime, seed_everything
from .af_model_b import (
    ModelBRolloutDataset,
    _unroll,
    stratified_rollout_records,
    western_boundary_mask,
)
from .af_model_c import (
    ModelCLossConfig,
    build_model_c,
    loss_contract,
    model_c_architecture,
    model_c_loss_terms,
)

try:  # Keep documentation imports usable without the optional ML stack.
    import torch
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


RAW_TERMS = ("state", "increment", "rollout", "spectral", "boundary")


class ModelCCalibrationError(RuntimeError):
    """Raised when the training-only loss calibration is not trustworthy."""


def gradient_l2_norm(gradients: Iterable[Any], device: Any) -> Any:
    """Accumulate a global gradient norm in float64 to avoid float32 overflow."""

    squared = torch.zeros((), device=device, dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            audit_dtype = torch.complex128 if gradient.is_complex() else torch.float64
            local_norm = torch.linalg.vector_norm(gradient.detach().to(audit_dtype))
            squared = squared + local_norm.square()
    norm = torch.sqrt(squared)
    if not bool(torch.isfinite(norm).item()):
        raise ModelCCalibrationError("Model C gradient norm is non-finite in float64")
    return norm


@dataclass(frozen=True)
class ModelCCalibrationConfig:
    sample_count: int = 96
    batch_size: int = 4
    warmup_epochs: int = 20
    calibration_batches: int = 4
    learning_rate: float = 1.0e-3
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1.0e-5
    seed: int = 20260723

    def __post_init__(self) -> None:
        if not 20 <= self.sample_count <= 100:
            raise ValueError("Model C calibration uses 20--100 training samples")
        if min(self.batch_size, self.warmup_epochs, self.calibration_batches) <= 0:
            raise ValueError("Model C calibration counts must be positive")
        if self.calibration_batches * self.batch_size > self.sample_count:
            raise ValueError("Model C calibration batches exceed the sample set")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.seed < 0:
            raise ValueError("Model C calibration optimizer settings are invalid")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def propose_loss_weights(
    gradient_norms: dict[str, float],
    *,
    targets: dict[str, float] | None = None,
    lower: float = 0.01,
    upper: float = 10.0,
) -> dict[str, float]:
    """Scale auxiliary gradients to declared fractions of the state gradient."""

    targets = targets or {
        "increment": 0.5,
        "rollout": 0.5,
        "spectral": 0.25,
        "boundary": 0.25,
    }
    if gradient_norms.get("state", 0.0) <= 0 or lower <= 0 or upper < lower:
        raise ValueError("Model C calibration needs a positive state gradient and bounds")
    result: dict[str, float] = {}
    for name, target in targets.items():
        norm = gradient_norms.get(name, 0.0)
        if norm <= 0 or target <= 0:
            raise ValueError(f"Model C calibration has invalid {name} gradient/target")
        result[name] = float(
            np.clip(target * gradient_norms["state"] / norm, lower, upper)
        )
    return result


def _masks(dataset: ModelBRolloutDataset, device: Any) -> tuple[Any, Any]:
    wet = torch.from_numpy(dataset.wet.astype(np.float32))[None, None].to(device)
    boundary_array = western_boundary_mask(dataset.wet, ModelCLossConfig().western_boundary_width)
    boundary = torch.from_numpy(boundary_array.astype(np.float32))[None, None].to(device)
    return wet, boundary


def _batch_terms(
    model: Any,
    features: Any,
    futures: Any,
    *,
    wet: Any,
    boundary: Any,
    increment_scale: Any,
) -> dict[str, Any]:
    predictions = _unroll(model, features, wet, ModelCLossConfig().rollout_steps)
    return model_c_loss_terms(
        predictions,
        futures,
        features[:, :46],
        wet,
        boundary,
        increment_scale,
        ModelCLossConfig(),
    )


def _warmup_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    *,
    device: Any,
    wet: Any,
    boundary: Any,
    increment_scale: Any,
) -> dict[str, float]:
    model.train()
    totals = {name: 0.0 for name in RAW_TERMS}
    samples = 0
    for features, futures in loader:
        features = features.to(device=device, dtype=torch.float32, non_blocking=True)
        futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
        terms = _batch_terms(
            model,
            features,
            futures,
            wet=wet,
            boundary=boundary,
            increment_scale=increment_scale,
        )
        core = terms["state"] + 0.5 * terms["increment"] + 0.5 * terms["rollout"]
        if not bool(torch.isfinite(core).item()):
            raise ModelCCalibrationError("Model C warmup encountered a non-finite core loss")
        optimizer.zero_grad(set_to_none=True)
        core.backward()
        optimizer.step()
        size = int(features.shape[0])
        for name in RAW_TERMS:
            totals[name] += float(terms[name].detach().cpu()) * size
        samples += size
    if not samples:
        raise ModelCCalibrationError("Model C calibration loader produced no samples")
    return {name: value / samples for name, value in totals.items()}


def _gradient_audit(
    model: Any,
    batches: Iterable[tuple[Any, Any]],
    *,
    device: Any,
    wet: Any,
    boundary: Any,
    increment_scale: Any,
) -> dict[str, dict[str, float]]:
    model.eval()
    raw_totals = {name: 0.0 for name in RAW_TERMS}
    gradient_totals = {name: 0.0 for name in RAW_TERMS}
    batch_count = 0
    for features, futures in batches:
        features = features.to(device=device, dtype=torch.float32)
        futures = futures.to(device=device, dtype=torch.float32)
        terms = _batch_terms(
            model,
            features,
            futures,
            wet=wet,
            boundary=boundary,
            increment_scale=increment_scale,
        )
        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        for name in RAW_TERMS:
            gradients = torch.autograd.grad(
                terms[name],
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            raw_totals[name] += float(terms[name].detach().cpu())
            gradient_totals[name] += float(gradient_l2_norm(gradients, device).cpu())
        batch_count += 1
    if not batch_count:
        raise ModelCCalibrationError("Model C gradient audit received no batches")
    return {
        "raw_terms": {name: value / batch_count for name, value in raw_totals.items()},
        "gradient_norms": {
            name: value / batch_count for name, value in gradient_totals.items()
        },
    }


def calibrate(
    dataset_path: str | Path,
    diagnostics_path: str | Path,
    output_dir: str | Path,
    *,
    config: ModelCCalibrationConfig = ModelCCalibrationConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run warmup and gradient audit using training data only."""

    require_model_a_runtime()
    seed_everything(config.seed)
    dataset_path = Path(dataset_path).resolve()
    diagnostics_path = Path(diagnostics_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model C calibration: {output}")
    diagnostics = json.loads(diagnostics_path.read_text())
    if diagnostics.get("read_contract", {}).get("validation_read") is not False:
        raise ValueError("Model C calibration requires the sealed training-only diagnostic")
    if Path(diagnostics.get("dataset", "")).resolve() != dataset_path:
        raise ValueError("Model C calibration dataset does not match its diagnostics")
    if diagnostics.get("dataset_metadata_sha256") != _file_sha256(dataset_path / ".zmetadata"):
        raise ValueError("Model C dataset metadata changed after diagnostics")
    increment_values = np.asarray(
        diagnostics["increment_rms_normalized_state_units"], dtype=np.float32
    )
    if increment_values.shape != (46,) or np.any(increment_values <= 0):
        raise ValueError("Model C diagnostics contain invalid increment scales")

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Model C CUDA calibration requested without a visible GPU")
    device = torch.device(device_name)

    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    records = stratified_rollout_records(pair_codes, config.sample_count, config.seed)
    dataset = ModelBRolloutDataset(dataset_path, records)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
    )
    audit_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    wet, boundary = _masks(dataset, device)
    increment_scale = torch.from_numpy(increment_values).to(device)
    architecture = model_c_architecture()
    model = build_model_c(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        weight_decay=config.weight_decay,
    )
    fixed_batches = []
    for index, batch in enumerate(audit_loader):
        if index >= config.calibration_batches:
            break
        fixed_batches.append(batch)

    started = time.monotonic()
    initial = _gradient_audit(
        model,
        fixed_batches,
        device=device,
        wet=wet,
        boundary=boundary,
        increment_scale=increment_scale,
    )
    history = []
    for epoch in range(1, config.warmup_epochs + 1):
        values = _warmup_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            wet=wet,
            boundary=boundary,
            increment_scale=increment_scale,
        )
        history.append({"epoch": epoch, **values})
    calibrated = _gradient_audit(
        model,
        fixed_batches,
        device=device,
        wet=wet,
        boundary=boundary,
        increment_scale=increment_scale,
    )
    proposed = propose_loss_weights(calibrated["gradient_norms"])
    if not all(np.isfinite(list(proposed.values()))):
        raise ModelCCalibrationError("Model C proposed non-finite loss weights")

    report = {
        "status": "complete_requires_manual_weight_freeze",
        "purpose": "training_only_model_c_loss_scale_and_gradient_audit",
        "dataset": str(dataset_path),
        "dataset_metadata_sha256": _file_sha256(dataset_path / ".zmetadata"),
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": _file_sha256(diagnostics_path),
        "read_contract": {
            "pair_split_codes_read": [1],
            "validation_read": False,
            "inference_read": False,
            "response_or_adjoint_read": False,
        },
        "architecture": architecture.to_dict(),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "config": asdict(config),
        "provisional_loss_contract": loss_contract(ModelCLossConfig()),
        "records": [list(record) for record in records],
        "initial_audit": initial,
        "warmup_history": history,
        "post_warmup_audit": calibrated,
        "gradient_targets_relative_to_state": {
            "increment": 0.5,
            "rollout": 0.5,
            "spectral": 0.25,
            "boundary": 0.25,
        },
        "proposed_weights_unfrozen": proposed,
        "warmup_state_dict_sha256": hashlib.sha256(
            b"".join(
                value.detach().cpu().numpy().tobytes()
                for value in _checkpoint_state_dict(model).values()
            )
        ).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "model_c_loss_calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate Model C losses on training only")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = calibrate(
        args.dataset,
        args.diagnostics,
        args.output_dir,
        config=ModelCCalibrationConfig(warmup_epochs=args.warmup_epochs),
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
