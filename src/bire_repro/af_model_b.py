"""Forward-loss FNO (Model B) for the AF--FNO ladder.

Model B deliberately reuses Model A's architecture, residual target, inputs,
normalizers, and chronological splits.  Its only scientific change is the
training objective: a three-step autoregressive rollout term, a radial spectral
term, and an explicitly western-boundary term are added to the one-step state
loss.  Forward perturbation responses remain excluded until Model C.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .af_data import STATIC_FEATURES, STATE_CHANNELS
from .af_model_a import (
    ChunkAwareBatchSampler,
    ModelAArchitecture,
    _checkpoint_state_dict,
    _normalization_from_group,
    build_model_a,
    masked_relative_l2,
    model_a_architecture,
    require_model_a_runtime,
    seed_everything,
)

try:  # Keep documentation-only imports working without the optional ML stack.
    import torch
    from torch.utils.data import DataLoader, Dataset
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]


STATE_CHANNEL_COUNT = len(STATE_CHANNELS)
STATIC_CHANNEL_COUNT = len(STATIC_FEATURES)
MODEL_B_INPUT_CHANNELS = STATE_CHANNEL_COUNT + STATIC_CHANNEL_COUNT
LOSS_PROFILES = ("rollout", "rollout_spectral", "full")


class ModelBTrainingError(RuntimeError):
    """Raised when a Model B training or reproducibility gate fails."""


@dataclass(frozen=True)
class ModelBLossConfig:
    """Declared forward-loss contract, with natural-scale starting weights."""

    rollout_steps: int = 3
    rollout_weight: float = 0.5
    spectral_weight: float = 0.05
    boundary_weight: float = 0.25
    spectral_bins: int = 12
    western_boundary_width: int = 4

    def __post_init__(self) -> None:
        if self.rollout_steps != 3:
            raise ValueError("Model B is fixed to a three-step (30-day) rollout")
        if min(self.rollout_weight, self.spectral_weight, self.boundary_weight) < 0:
            raise ValueError("Model B loss weights must be non-negative")
        if self.rollout_weight <= 0:
            raise ValueError("Model B requires a positive rollout weight")
        if self.spectral_bins <= 1 or self.western_boundary_width <= 0:
            raise ValueError(
                "Model B needs at least two spectral bins and a positive boundary width"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def loss_config_for_profile(profile: str) -> ModelBLossConfig:
    """Return one incremental development profile without changing loss scales."""

    if profile == "rollout":
        return ModelBLossConfig(spectral_weight=0.0, boundary_weight=0.0)
    if profile == "rollout_spectral":
        return ModelBLossConfig(boundary_weight=0.0)
    if profile == "full":
        return ModelBLossConfig()
    raise ValueError(f"unknown Model B loss profile {profile!r}; choose from {LOSS_PROFILES}")


def loss_contract(config: ModelBLossConfig) -> dict[str, Any]:
    """Machine-readable semantics stored with every Model B checkpoint."""

    return {
        "state": "masked_relative_l2_at_10_days",
        "rollout": "mean_masked_relative_l2_at_20_and_30_days",
        "spectral": "mean_12_bin_radial_anomaly_amplitude_relative_l2_at_10_20_30_days",
        "boundary": "mean_masked_relative_l2_at_10_20_30_days_on_first_4_wet_cells_east_of_western_wall",
        "total": "state + rollout_weight*rollout + spectral_weight*spectral + boundary_weight*boundary",
        "config": config.to_dict(),
    }


def loss_contract_sha256(config: ModelBLossConfig) -> str:
    encoded = json.dumps(loss_contract(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelBOverfitConfig:
    """Small-sample gate for the complete Model B objective."""

    sample_count: int = 96
    horizon_days: int = 10
    batch_size: int = 4
    epochs: int = 160
    learning_rates: tuple[float, ...] = (1.0e-3, 5.0e-4)
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1.0e-5
    seed: int = 20260721

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "learning_rates", tuple(float(value) for value in self.learning_rates)
        )
        if not 20 <= self.sample_count <= 100:
            raise ValueError("Model B overfit sample_count must be in the declared 20--100 range")
        if self.horizon_days != 10:
            raise ValueError("Model B is fixed to the shared ten-day state map")
        if self.batch_size <= 0 or self.epochs <= 0 or self.seed < 0:
            raise ValueError("Model B batch_size, epochs, and seed must be positive")
        if not self.learning_rates or any(value <= 0 for value in self.learning_rates):
            raise ValueError("Model B learning rates must be positive")
        if len(self.adam_betas) != 2 or any(not 0 <= value < 1 for value in self.adam_betas):
            raise ValueError("Model B Adam betas must lie in [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("Model B weight decay must be non-negative")


@dataclass(frozen=True)
class ModelBDevelopmentConfig:
    """Sealed-split development run for one incremental loss profile."""

    batch_size: int = 8
    epochs: int = 12
    learning_rate: float = 1.0e-3
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1.0e-5
    seed: int = 20260721

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0 or self.learning_rate <= 0 or self.seed < 0:
            raise ValueError("Model B development settings must be positive")
        if len(self.adam_betas) != 2 or any(not 0 <= value < 1 for value in self.adam_betas):
            raise ValueError("Model B Adam betas must lie in [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("Model B weight decay must be non-negative")


@dataclass(frozen=True)
class ModelBFinalConfig:
    """Frozen schedule selected from the complete-loss development profile."""

    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 1.0e-3
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1.0e-5
    seed: int = 20260721

    def __post_init__(self) -> None:
        if self.epochs != 12 or self.batch_size != 8 or self.learning_rate != 1.0e-3:
            raise ValueError(
                "frozen Model B final configuration is epochs=12, batch=8, learning_rate=0.001"
            )
        if self.adam_betas != (0.9, 0.95) or self.weight_decay != 1.0e-5 or self.seed != 20260721:
            raise ValueError("Model B final optimizer and seed must match the development protocol")


def model_b_architecture() -> ModelAArchitecture:
    """Return the exact frozen Model A architecture used for the loss ablation."""

    return model_a_architecture()


def build_model_b(architecture: ModelAArchitecture = ModelAArchitecture()) -> Any:
    """Build Model B; architecture equality with Model A is intentional."""

    return build_model_a(architecture)


def rollout_start_indices(
    pair_codes: Sequence[int],
    split_code: int,
    *,
    horizon_days: int = 10,
    rollout_steps: int = 3,
) -> tuple[int, ...]:
    """Find starts whose complete autoregressive target sequence stays in one split."""

    codes = np.asarray(pair_codes, dtype=np.uint8)
    if codes.ndim != 1 or horizon_days <= 0 or rollout_steps <= 0:
        raise ValueError("Model B split codes must be one-dimensional and horizons positive")
    latest = codes.size - rollout_steps * horizon_days
    if latest <= 0:
        return ()
    result = []
    for time_index in range(latest):
        if all(
            codes[time_index + step * horizon_days] == split_code for step in range(rollout_steps)
        ):
            result.append(time_index)
    return tuple(result)


def records_for_rollout_split(
    pair_codes: Sequence[int],
    split_code: int,
    *,
    experiment_count: int = 3,
    horizon_days: int = 10,
    rollout_steps: int = 3,
) -> tuple[tuple[int, int], ...]:
    starts = rollout_start_indices(
        pair_codes, split_code, horizon_days=horizon_days, rollout_steps=rollout_steps
    )
    if not starts:
        raise ValueError(f"dataset has no complete Model B rollouts for pair split {split_code}")
    return tuple(
        (experiment, time_index) for experiment in range(experiment_count) for time_index in starts
    )


def stratified_rollout_records(
    pair_codes: Sequence[int], sample_count: int, seed: int
) -> tuple[tuple[int, int], ...]:
    """Select balanced, training-only, complete rollout records for overfitting."""

    candidates = np.asarray(rollout_start_indices(pair_codes, 1), dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("dataset has no complete Model B training rollouts")
    counts = [sample_count // 3 + int(index < sample_count % 3) for index in range(3)]
    if any(count > candidates.size for count in counts):
        raise ValueError("requested Model B sample count exceeds training rollouts")
    records: list[tuple[int, int]] = []
    for experiment, count in enumerate(counts):
        selected = np.sort(
            np.random.default_rng(seed + experiment).choice(candidates, size=count, replace=False)
        )
        records.extend((experiment, int(time_index)) for time_index in selected)
    return tuple(records)


class ModelBRolloutDataset(Dataset):
    """Lazy Model A features plus three normalized ten-day rollout targets."""

    def __init__(
        self,
        dataset_path: str | Path,
        records: Sequence[tuple[int, int]],
        *,
        horizon_days: int = 10,
        rollout_steps: int = 3,
    ) -> None:
        require_model_a_runtime()
        self.dataset_path = Path(dataset_path).resolve()
        self.records = tuple(
            (int(experiment), int(time_index)) for experiment, time_index in records
        )
        self.horizon_days = int(horizon_days)
        self.rollout_steps = int(rollout_steps)
        if not self.records or self.horizon_days <= 0 or self.rollout_steps != 3:
            raise ValueError("Model B dataset needs records and the declared three-step horizon")
        self._group: Any | None = None
        self._state: Any | None = None
        self._static: Any | None = None
        self._open()

    def _open(self) -> None:
        import zarr

        self._group = zarr.open_consolidated(str(self.dataset_path), mode="r")
        self._state, self._static = self._group["state"], self._group["static_features"]
        self.mean, self.scale, self.wet, self.wind_mean, self.wind_scale = (
            _normalization_from_group(self._group)
        )
        if (
            self._state.shape[2] != STATE_CHANNEL_COUNT
            or self._static.shape[1] != STATIC_CHANNEL_COUNT
        ):
            raise ValueError("dataset channel count does not match Model B")
        if any(
            time + self.horizon_days * self.rollout_steps >= self._state.shape[1]
            for _, time in self.records
        ):
            raise ValueError("a Model B rollout exceeds the available trajectory length")

    def __len__(self) -> int:
        return len(self.records)

    def _normalise_state(self, value: np.ndarray) -> np.ndarray:
        result = (value - self.mean[:, None, None]) / self.scale[:, None, None]
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
        geometry = np.asarray(self._static[experiment], dtype=np.float32).copy()
        geometry[0] = (geometry[0] - self.wind_mean) / self.wind_scale
        geometry[0, ~self.wet] = 0.0
        features = np.ascontiguousarray(
            np.concatenate((present, geometry), axis=0), dtype=np.float32
        )
        return torch.from_numpy(features), torch.from_numpy(futures)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_group"] = state["_state"] = state["_static"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open()


def western_boundary_mask(wet_mask: np.ndarray, width: int = 4) -> np.ndarray:
    """Select the first ``width`` wet cells east of each row's western wall."""

    wet = np.asarray(wet_mask, dtype=bool)
    if wet.ndim != 2 or width <= 0:
        raise ValueError(
            "western boundary construction needs a two-dimensional wet mask and positive width"
        )
    result = np.zeros_like(wet)
    for row in range(wet.shape[0]):
        columns = np.flatnonzero(wet[row])
        if columns.size:
            result[row, columns[0] : min(columns[0] + width, wet.shape[1])] = wet[
                row, columns[0] : min(columns[0] + width, wet.shape[1])
            ]
    if not np.any(result):
        raise ValueError("wet mask has no cells from which to construct the western boundary band")
    return result


def binned_spectral_loss(
    prediction: Any,
    target: Any,
    wet_mask: Any,
    *,
    bins: int = 12,
    epsilon: float = 1.0e-12,
) -> Any:
    """Relative L2 mismatch of radial, wet-anomaly Fourier amplitudes."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Model B spectral prediction and target must share N,C,Y,X shape")
    if wet_mask.shape != (1, 1, *prediction.shape[-2:]) or bins <= 1:
        raise ValueError("Model B spectral mask or bin count is invalid")
    wet_count = wet_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)

    def amplitudes(value: Any) -> Any:
        mean = (value * wet_mask).sum(dim=(-2, -1), keepdim=True) / wet_count
        anomaly = (value - mean) * wet_mask
        power = torch.fft.rfft2(anomaly, norm="ortho").abs().square()
        ny, nx_half = power.shape[-2:]
        fy = torch.fft.fftfreq(prediction.shape[-2], device=value.device)
        fx = torch.fft.rfftfreq(prediction.shape[-1], device=value.device)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        valid = radius > 0
        indices = (
            torch.floor(radius[valid] / radius.max() * bins).to(torch.long).clamp_max(bins - 1)
        )
        flattened = power[..., valid]
        spectrum = torch.zeros((*power.shape[:2], bins), device=value.device, dtype=value.dtype)
        spectrum.scatter_add_(2, indices[None, None].expand_as(flattened), flattened)
        counts = torch.bincount(indices, minlength=bins).to(dtype=value.dtype).clamp_min(1.0)
        return torch.sqrt(spectrum / counts[None, None] + epsilon)

    predicted_amplitude, target_amplitude = amplitudes(prediction), amplitudes(target)
    numerator = (predicted_amplitude - target_amplitude).square().sum(dim=(1, 2))
    denominator = target_amplitude.square().sum(dim=(1, 2)).clamp_min(epsilon)
    return torch.sqrt(numerator / denominator).mean()


def _unroll(model: Any, features: Any, wet: Any, steps: int) -> Any:
    current = features[:, :STATE_CHANNEL_COUNT]
    geometry = features[:, STATE_CHANNEL_COUNT:]
    predictions = []
    for _ in range(steps):
        current = (current + model(torch.cat((current, geometry), dim=1))) * wet
        predictions.append(current)
    return torch.stack(predictions, dim=1)


def model_b_loss_terms(
    predictions: Any,
    targets: Any,
    wet: Any,
    western_boundary: Any,
    config: ModelBLossConfig,
) -> dict[str, Any]:
    """Compute every named Model B term and their declared weighted total."""

    if predictions.shape != targets.shape or predictions.ndim != 5:
        raise ValueError("Model B rollout predictions and targets must share N,S,C,Y,X shape")
    if predictions.shape[1] != config.rollout_steps:
        raise ValueError("Model B loss received the wrong number of rollout steps")
    state = masked_relative_l2(predictions[:, 0], targets[:, 0], wet)
    rollout = torch.stack(
        [
            masked_relative_l2(predictions[:, step], targets[:, step], wet)
            for step in range(1, config.rollout_steps)
        ]
    ).mean()
    spectral = torch.stack(
        [
            binned_spectral_loss(
                predictions[:, step], targets[:, step], wet, bins=config.spectral_bins
            )
            for step in range(config.rollout_steps)
        ]
    ).mean()
    boundary = torch.stack(
        [
            masked_relative_l2(predictions[:, step], targets[:, step], western_boundary)
            for step in range(config.rollout_steps)
        ]
    ).mean()
    total = (
        state
        + config.rollout_weight * rollout
        + config.spectral_weight * spectral
        + config.boundary_weight * boundary
    )
    return {
        "total": total,
        "state": state,
        "rollout": rollout,
        "spectral": spectral,
        "boundary": boundary,
    }


def _epoch(
    model: Any,
    loader: Any,
    *,
    device: Any,
    wet: Any,
    western_boundary: Any,
    loss_config: ModelBLossConfig,
    optimizer: Any | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("total", "state", "rollout", "spectral", "boundary")}
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, futures in loader:
            features = features.to(device=device, dtype=torch.float32, non_blocking=True)
            futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
            predictions = _unroll(model, features, wet, loss_config.rollout_steps)
            terms = model_b_loss_terms(predictions, futures, wet, western_boundary, loss_config)
            if not all(bool(torch.isfinite(value).item()) for value in terms.values()):
                raise ModelBTrainingError("Model B encountered a non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                terms["total"].backward()
                optimizer.step()
            size = int(features.shape[0])
            for name, value in terms.items():
                totals[name] += float(value.detach().cpu()) * size
            samples += size
    if not samples:
        raise ModelBTrainingError("Model B data loader produced no samples")
    return {name: value / samples for name, value in totals.items()}


def _persistence_terms(
    loader: Any,
    *,
    device: Any,
    wet: Any,
    western_boundary: Any,
    loss_config: ModelBLossConfig,
) -> dict[str, float]:
    totals = {name: 0.0 for name in ("total", "state", "rollout", "spectral", "boundary")}
    samples = 0
    with torch.no_grad():
        for features, futures in loader:
            features = features.to(device=device, dtype=torch.float32, non_blocking=True)
            futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
            predictions = features[:, None, :STATE_CHANNEL_COUNT].expand_as(futures)
            terms = model_b_loss_terms(predictions, futures, wet, western_boundary, loss_config)
            size = int(features.shape[0])
            for name, value in terms.items():
                totals[name] += float(value.detach().cpu()) * size
            samples += size
    if not samples:
        raise ModelBTrainingError("Model B persistence loader produced no samples")
    return {name: value / samples for name, value in totals.items()}


def _device(device_name: str) -> Any:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Model B CUDA run requested but no CUDA device is visible")
    return torch.device(device_name)


def _masks(
    dataset: ModelBRolloutDataset, device: Any, loss_config: ModelBLossConfig
) -> tuple[Any, Any]:
    wet = torch.from_numpy(dataset.wet.astype(np.float32))[None, None].to(device)
    boundary_array = western_boundary_mask(dataset.wet, loss_config.western_boundary_width)
    boundary = torch.from_numpy(boundary_array.astype(np.float32))[None, None].to(device)
    return wet, boundary


def _normalization(dataset: ModelBRolloutDataset) -> dict[str, Any]:
    return {
        "state_mean": dataset.mean.tolist(),
        "state_scale": dataset.scale.tolist(),
        "wind_mean": dataset.wind_mean,
        "wind_scale": dataset.wind_scale,
    }


def _save_and_reload_check(
    model: Any,
    architecture: ModelAArchitecture,
    features: Any,
    checkpoint_path: Path,
    payload: dict[str, Any],
    device: Any,
) -> bool:
    model.eval()
    with torch.no_grad():
        reference = model(features).detach().cpu()
    torch.save(payload, checkpoint_path)
    restored = build_model_b(architecture).to(device)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored.load_state_dict(loaded["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        reloaded = restored(features).detach().cpu()
    return bool(torch.equal(reference, reloaded))


def run_overfit(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    config: ModelBOverfitConfig = ModelBOverfitConfig(),
    loss_config: ModelBLossConfig = ModelBLossConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run the complete-loss small-sample and checkpoint-reload gate."""

    require_model_a_runtime()
    if loss_config.spectral_weight <= 0 or loss_config.boundary_weight <= 0:
        raise ValueError(
            "Model B overfit gate must exercise rollout, spectral, and boundary losses"
        )
    seed_everything(config.seed)
    dataset_path, output = Path(dataset_path).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite or reuse Model B overfit directory: {output}")
    device = _device(device_name)
    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    records = stratified_rollout_records(
        np.asarray(group["pair_split"][:], dtype=np.uint8), config.sample_count, config.seed
    )
    dataset = ModelBRolloutDataset(dataset_path, records)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
    )
    evaluation_loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    wet, boundary = _masks(dataset, device, loss_config)
    architecture = model_b_architecture()
    attempts: list[dict[str, Any]] = []
    started = time.monotonic()
    for attempt, learning_rate in enumerate(config.learning_rates):
        seed_everything(config.seed + attempt)
        model = build_model_b(architecture).to(device)
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
            western_boundary=boundary,
            loss_config=loss_config,
            optimizer=None,
        )
        history: list[dict[str, Any]] = []
        best_total, best_state, best_model_state = float("inf"), float("inf"), None
        for epoch in range(1, config.epochs + 1):
            train = _epoch(
                model,
                train_loader,
                device=device,
                wet=wet,
                western_boundary=boundary,
                loss_config=loss_config,
                optimizer=optimizer,
            )
            evaluation = _epoch(
                model,
                evaluation_loader,
                device=device,
                wet=wet,
                western_boundary=boundary,
                loss_config=loss_config,
                optimizer=None,
            )
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "train": train,
                    "evaluation": evaluation,
                }
            )
            if evaluation["total"] < best_total:
                best_total, best_state, best_model_state = (
                    evaluation["total"],
                    evaluation["state"],
                    _checkpoint_state_dict(model),
                )
        accepted = bool(
            best_model_state is not None
            and best_total < initial["total"]
            and best_state < initial["state"]
        )
        attempts.append(
            {
                "learning_rate": learning_rate,
                "initial": initial,
                "best_total": best_total,
                "best_state": best_state,
                "final": history[-1]["evaluation"],
                "accepted": accepted,
            }
        )
        if not accepted:
            continue
        model.load_state_dict(best_model_state)
        output.mkdir(parents=True, exist_ok=False)
        checkpoint_path = output / "model_b_overfit_best.pt"
        features, _ = dataset[0]
        features = features[None].to(device)
        payload = {
            "model_class": "ModelBResidualFNO_same_architecture_as_ModelA",
            "model_config": architecture.to_dict(),
            "model_b_overfit_config": asdict(config),
            "model_b_loss_contract": loss_contract(loss_config),
            "model_b_loss_contract_sha256": loss_contract_sha256(loss_config),
            "dataset": str(dataset_path),
            "records": [list(record) for record in records],
            "normalization": _normalization(dataset),
            "state_target": "normalized_future_minus_normalized_present",
            "model_state_dict": _checkpoint_state_dict(model),
            "history": history,
        }
        bitwise_exact = _save_and_reload_check(
            model, architecture, features, checkpoint_path, payload, device
        )
        if not bitwise_exact:
            raise ModelBTrainingError(
                "Model B overfit checkpoint reload changed deterministic inference"
            )
        report = {
            "status": "accepted",
            "dataset": str(dataset_path),
            "device": str(device),
            "neuraloperator_version": metadata.version("neuraloperator"),
            "architecture": architecture.to_dict(),
            "config": asdict(config),
            "loss_contract": loss_contract(loss_config),
            "loss_contract_sha256": loss_contract_sha256(loss_config),
            "records": [list(record) for record in records],
            "attempts": attempts,
            "selected_learning_rate": learning_rate,
            "save_reload_bitwise_exact": bitwise_exact,
            "elapsed_seconds": time.monotonic() - started,
        }
        (output / "model_b_overfit_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        return report
    raise ModelBTrainingError(
        "Model B did not lower its complete and state losses: "
        + json.dumps(attempts, sort_keys=True)
    )


def run_development(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    config: ModelBDevelopmentConfig = ModelBDevelopmentConfig(),
    loss_config: ModelBLossConfig = ModelBLossConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train one incremental profile on train data and score held validation rollouts."""

    require_model_a_runtime()
    seed_everything(config.seed)
    dataset_path, output = Path(dataset_path).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite or reuse Model B development directory: {output}"
        )
    device = _device(device_name)
    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    train_records = records_for_rollout_split(pair_codes, 1)
    validation_records = records_for_rollout_split(pair_codes, 2)
    train_dataset = ModelBRolloutDataset(dataset_path, train_records)
    validation_dataset = ModelBRolloutDataset(dataset_path, validation_records)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=ChunkAwareBatchSampler(train_dataset, config.batch_size, config.seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    wet, boundary = _masks(train_dataset, device, loss_config)
    architecture = model_b_architecture()
    model = build_model_b(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        weight_decay=config.weight_decay,
    )
    persistence = _persistence_terms(
        validation_loader,
        device=device,
        wet=wet,
        western_boundary=boundary,
        loss_config=loss_config,
    )
    history: list[dict[str, Any]] = []
    best_validation, best_state, best_epoch = float("inf"), None, None
    started = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        train = _epoch(
            model,
            train_loader,
            device=device,
            wet=wet,
            western_boundary=boundary,
            loss_config=loss_config,
            optimizer=optimizer,
        )
        validation = _epoch(
            model,
            validation_loader,
            device=device,
            wet=wet,
            western_boundary=boundary,
            loss_config=loss_config,
            optimizer=None,
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": config.learning_rate,
                "train": train,
                "validation": validation,
            }
        )
        if validation["total"] < best_validation:
            best_validation, best_state, best_epoch = (
                validation["total"],
                _checkpoint_state_dict(model),
                epoch,
            )
    if best_state is None or best_epoch is None:
        raise ModelBTrainingError("Model B development run never produced a checkpoint")
    model.load_state_dict(best_state)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "model_b_development_best.pt"
    features, _ = validation_dataset[0]
    features = features[None].to(device)
    payload = {
        "model_class": "ModelBResidualFNO_same_architecture_as_ModelA",
        "model_config": architecture.to_dict(),
        "model_b_development_config": asdict(config),
        "model_b_loss_contract": loss_contract(loss_config),
        "model_b_loss_contract_sha256": loss_contract_sha256(loss_config),
        "dataset": str(dataset_path),
        "split_contract": {
            "train_pair_code": 1,
            "validation_pair_code": 2,
            "complete_rollout_required": True,
        },
        "pair_counts": {"train": len(train_records), "validation": len(validation_records)},
        "normalization": _normalization(train_dataset),
        "state_target": "normalized_future_minus_normalized_present",
        "model_state_dict": _checkpoint_state_dict(model),
        "history": history,
    }
    bitwise_exact = _save_and_reload_check(
        model, architecture, features, checkpoint_path, payload, device
    )
    if not bitwise_exact:
        raise ModelBTrainingError(
            "Model B development checkpoint reload changed deterministic inference"
        )
    best_terms = history[best_epoch - 1]["validation"]
    report = {
        "status": "accepted"
        if best_validation < persistence["total"]
        else "completed_without_persistence_skill",
        "dataset": str(dataset_path),
        "device": str(device),
        "neuraloperator_version": metadata.version("neuraloperator"),
        "architecture": architecture.to_dict(),
        "config": asdict(config),
        "loss_contract": loss_contract(loss_config),
        "loss_contract_sha256": loss_contract_sha256(loss_config),
        "pair_counts": {"train": len(train_records), "validation": len(validation_records)},
        "persistence_validation": persistence,
        "best_validation": best_terms,
        "final_validation": history[-1]["validation"],
        "best_epoch": best_epoch,
        "save_reload_bitwise_exact": bitwise_exact,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "model_b_development_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def run_final(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    development_report_path: str | Path,
    config: ModelBFinalConfig,
    loss_config: ModelBLossConfig = ModelBLossConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train a selected full-loss schedule and freeze one Model B realization."""

    require_model_a_runtime()
    if loss_config.spectral_weight <= 0 or loss_config.boundary_weight <= 0:
        raise ValueError("frozen Model B must include rollout, spectral, and boundary losses")
    development_report_path = Path(development_report_path).resolve()
    development_report = json.loads(development_report_path.read_text())
    if development_report.get("loss_contract_sha256") != loss_contract_sha256(loss_config):
        raise ValueError("Model B final requires the complete-loss development report")
    if development_report.get("best_epoch") != config.epochs:
        raise ValueError("Model B final epoch must equal the complete-loss development minimum")
    seed_everything(config.seed)
    dataset_path, output = Path(dataset_path).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite or reuse frozen Model B directory: {output}")
    device = _device(device_name)
    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    records = {
        "train": records_for_rollout_split(pair_codes, 1),
        "validation": records_for_rollout_split(pair_codes, 2),
        "inference": records_for_rollout_split(pair_codes, 3),
    }
    datasets = {name: ModelBRolloutDataset(dataset_path, value) for name, value in records.items()}
    train_loader = DataLoader(
        datasets["train"],
        batch_sampler=ChunkAwareBatchSampler(datasets["train"], config.batch_size, config.seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    evaluation_loaders = {
        name: DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        for name, dataset in datasets.items()
        if name != "train"
    }
    wet, boundary = _masks(datasets["train"], device, loss_config)
    architecture = model_b_architecture()
    model = build_model_b(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        train = _epoch(
            model,
            train_loader,
            device=device,
            wet=wet,
            western_boundary=boundary,
            loss_config=loss_config,
            optimizer=optimizer,
        )
        history.append({"epoch": epoch, "learning_rate": config.learning_rate, "train": train})
    evaluation = {
        name: {
            "model_b": _epoch(
                model,
                loader,
                device=device,
                wet=wet,
                western_boundary=boundary,
                loss_config=loss_config,
                optimizer=None,
            ),
            "persistence": _persistence_terms(
                loader,
                device=device,
                wet=wet,
                western_boundary=boundary,
                loss_config=loss_config,
            ),
        }
        for name, loader in evaluation_loaders.items()
    }
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "model_b_final.pt"
    features, _ = datasets["inference"][0]
    features = features[None].to(device)
    payload = {
        "model_class": "ModelBResidualFNO_same_architecture_as_ModelA",
        "model_config": architecture.to_dict(),
        "model_b_final_config": asdict(config),
        "model_b_loss_contract": loss_contract(loss_config),
        "model_b_loss_contract_sha256": loss_contract_sha256(loss_config),
        "development_report": str(development_report_path),
        "development_report_sha256": _file_sha256(development_report_path),
        "dataset": str(dataset_path),
        "split_contract": {
            "train_pair_code": 1,
            "validation_pair_code": 2,
            "inference_pair_code": 3,
            "complete_rollout_required": True,
        },
        "pair_counts": {name: len(value) for name, value in records.items()},
        "normalization": _normalization(datasets["train"]),
        "state_target": "normalized_future_minus_normalized_present",
        "model_state_dict": _checkpoint_state_dict(model),
        "history": history,
    }
    bitwise_exact = _save_and_reload_check(
        model, architecture, features, checkpoint_path, payload, device
    )
    if not bitwise_exact:
        raise ModelBTrainingError(
            "frozen Model B checkpoint reload changed deterministic inference"
        )
    report = {
        "status": "frozen",
        "dataset": str(dataset_path),
        "device": str(device),
        "neuraloperator_version": metadata.version("neuraloperator"),
        "architecture": architecture.to_dict(),
        "config": asdict(config),
        "loss_contract": loss_contract(loss_config),
        "loss_contract_sha256": loss_contract_sha256(loss_config),
        "development_report": str(development_report_path),
        "development_report_sha256": _file_sha256(development_report_path),
        "pair_counts": {name: len(value) for name, value in records.items()},
        "evaluation": evaluation,
        "save_reload_bitwise_exact": bitwise_exact,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "model_b_final_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the forward-loss Model B FNO")
    commands = parser.add_subparsers(dest="command", required=True)
    overfit = commands.add_parser(
        "overfit", help="exercise the complete Model B loss on 20--100 samples"
    )
    overfit.add_argument("--dataset", type=Path, required=True)
    overfit.add_argument("--output-dir", type=Path, required=True)
    overfit.add_argument("--samples", type=int, default=96)
    overfit.add_argument("--epochs", type=int, default=160)
    overfit.add_argument("--batch-size", type=int, default=4)
    overfit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    develop = commands.add_parser("develop", help="run one incremental sealed-split loss profile")
    develop.add_argument("--dataset", type=Path, required=True)
    develop.add_argument("--output-dir", type=Path, required=True)
    develop.add_argument("--profile", choices=LOSS_PROFILES, default="full")
    develop.add_argument("--epochs", type=int, default=12)
    develop.add_argument("--batch-size", type=int, default=8)
    develop.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    final = commands.add_parser(
        "final", help="train one final full-loss realization after development"
    )
    final.add_argument("--dataset", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, required=True)
    final.add_argument("--development-report", type=Path, required=True)
    final.add_argument(
        "--epochs", type=int, default=12, help="frozen full-loss development minimum"
    )
    final.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "overfit":
        result = run_overfit(
            args.dataset,
            args.output_dir,
            config=ModelBOverfitConfig(
                sample_count=args.samples, epochs=args.epochs, batch_size=args.batch_size
            ),
            device_name=args.device,
        )
    elif args.command == "develop":
        result = run_development(
            args.dataset,
            args.output_dir,
            config=ModelBDevelopmentConfig(epochs=args.epochs, batch_size=args.batch_size),
            loss_config=loss_config_for_profile(args.profile),
            device_name=args.device,
        )
    elif args.command == "final":
        result = run_final(
            args.dataset,
            args.output_dir,
            development_report_path=args.development_report,
            config=ModelBFinalConfig(epochs=args.epochs),
            device_name=args.device,
        )
    else:  # pragma: no cover - argparse enforces the command
        raise ValueError(f"unsupported Model B command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
