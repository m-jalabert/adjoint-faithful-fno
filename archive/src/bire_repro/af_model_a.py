"""Modern state-residual FNO baseline (Model A) for the AF--FNO ladder.

Model A is intentionally not a repair of the historical Bire architecture.  It
uses NeuralOperator 2.0's dense FNO, an explicit 3-by-3 local correction, the
five declared static fields, and predicts the normalized ten-day state change.
It is the state-only comparison for the later forward-loss (B) and
response-supervised (C) models.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .af_a0 import records_for_pair_split
from .af_data import STATIC_FEATURES, STATE_CHANNELS

try:  # Keep documentation-only imports working on login nodes without ML packages.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from neuralop.models import FNO
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    FNO = None  # type: ignore[assignment]


STATE_CHANNEL_COUNT = len(STATE_CHANNELS)
STATIC_CHANNEL_COUNT = len(STATIC_FEATURES)
MODEL_A_INPUT_CHANNELS = STATE_CHANNEL_COUNT + STATIC_CHANNEL_COUNT


class ModelATrainingError(RuntimeError):
    """Raised when the Model A acceptance gate cannot establish a valid model."""


@dataclass(frozen=True)
class ModelAArchitecture:
    """The fixed initial NeuralOperator 2.0 architecture from the project plan."""

    in_channels: int = MODEL_A_INPUT_CHANNELS
    out_channels: int = STATE_CHANNEL_COUNT
    n_modes: tuple[int, int] = (16, 16)
    hidden_channels: int = 32
    n_layers: int = 4
    domain_padding: float = 0.10
    positional_embedding: str = "grid"
    use_channel_mlp: bool = True
    local_kernel_size: int = 3
    precision: str = "full"
    factorization: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_modes", tuple(int(value) for value in self.n_modes))
        if self.in_channels != MODEL_A_INPUT_CHANNELS or self.out_channels != STATE_CHANNEL_COUNT:
            raise ValueError("Model A channel contract is 46 dynamic + 5 static -> 46 residual channels")
        if len(self.n_modes) != 2 or any(value <= 0 for value in self.n_modes):
            raise ValueError("Model A needs two positive Fourier-mode counts")
        if self.hidden_channels != 32 or self.n_layers != 4:
            raise ValueError("initial Model A is fixed at 32 hidden channels and four FNO layers")
        if self.domain_padding != 0.10 or self.local_kernel_size != 3:
            raise ValueError("initial Model A uses 10% padding and a 3-by-3 local branch")
        if self.positional_embedding != "grid" or not self.use_channel_mlp:
            raise ValueError("initial Model A requires grid embedding and the channel MLP")
        if self.precision != "full" or self.factorization is not None:
            raise ValueError("initial Model A is dense float32 FNO")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_modes"] = list(self.n_modes)
        return result


@dataclass(frozen=True)
class ModelAOverfitConfig:
    """Small-sample, training-only acceptance gate before full Model A training."""

    sample_count: int = 96
    horizon_days: int = 10
    batch_size: int = 4
    epochs: int = 160
    learning_rates: tuple[float, ...] = (1.0e-3, 5.0e-4)
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1.0e-5
    seed: int = 20260721

    def __post_init__(self) -> None:
        object.__setattr__(self, "learning_rates", tuple(float(value) for value in self.learning_rates))
        if not 20 <= self.sample_count <= 100:
            raise ValueError("Model A overfit sample_count must be in the declared 20--100 range")
        if self.horizon_days != 10:
            raise ValueError("Model A is fixed to the shared ten-day state map")
        if self.batch_size <= 0 or self.epochs <= 0 or self.seed < 0:
            raise ValueError("Model A batch_size, epochs, and seed must be positive")
        if not self.learning_rates or any(value <= 0 for value in self.learning_rates):
            raise ValueError("Model A learning_rates must be positive")
        if len(self.adam_betas) != 2 or any(not 0 <= value < 1 for value in self.adam_betas):
            raise ValueError("Model A Adam betas must lie in [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("Model A weight_decay must be non-negative")


@dataclass(frozen=True)
class ModelADevelopmentConfig:
    """Sealed-split Model A development run before one frozen final realization."""

    batch_size: int = 8
    epochs: int = 12
    learning_rate: float = 1.0e-3
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1.0e-5
    seed: int = 20260721

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0 or self.learning_rate <= 0 or self.seed < 0:
            raise ValueError("Model A batch_size, epochs, learning_rate, and seed must be positive")
        if len(self.adam_betas) != 2 or any(not 0 <= value < 1 for value in self.adam_betas):
            raise ValueError("Model A Adam betas must lie in [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("Model A weight_decay must be non-negative")


@dataclass(frozen=True)
class ModelAFinalConfig:
    """One frozen Model A realization selected from the sealed development run."""

    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1.0e-3
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1.0e-5
    seed: int = 20260721

    def __post_init__(self) -> None:
        if self.batch_size != 8 or self.epochs != 10 or self.learning_rate != 1.0e-3:
            raise ValueError("frozen Model A final configuration is batch=8, epochs=10, learning_rate=0.001")
        if self.adam_betas != (0.9, 0.95) or self.weight_decay != 1.0e-5 or self.seed != 20260721:
            raise ValueError("frozen Model A optimizer and seed must match the selected development schedule")


def require_model_a_runtime() -> None:
    """Verify the maintained, pinned NeuralOperator runtime used by Model A."""

    if torch is None or nn is None or FNO is None or DataLoader is None:
        raise RuntimeError("Model A requires the project PyTorch and NeuralOperator 2.0.0 environment")
    try:
        version = metadata.version("neuraloperator")
    except metadata.PackageNotFoundError as exc:  # pragma: no cover - installation failure
        raise RuntimeError("Model A requires the project PyTorch and NeuralOperator 2.0.0 environment") from exc
    if version != "2.0.0":
        raise RuntimeError(f"Model A is pinned to neuraloperator 2.0.0, found {version}")


def seed_everything(seed: int) -> None:
    """Make acceptance-gate initialization and batch order reproducible."""

    require_model_a_runtime()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


if nn is not None:

    class ModelAResidualFNO(nn.Module):
        """Dense NeuralOperator FNO with a declared 3-by-3 local residual branch."""

        def __init__(self, architecture: ModelAArchitecture = ModelAArchitecture()) -> None:
            super().__init__()
            require_model_a_runtime()
            self.architecture = architecture
            self.fno = FNO(
                n_modes=architecture.n_modes,
                in_channels=architecture.in_channels,
                out_channels=architecture.out_channels,
                hidden_channels=architecture.hidden_channels,
                n_layers=architecture.n_layers,
                positional_embedding=architecture.positional_embedding,
                use_channel_mlp=architecture.use_channel_mlp,
                domain_padding=architecture.domain_padding,
                fno_block_precision=architecture.precision,
                factorization=architecture.factorization,
            )
            self.local = nn.Conv2d(
                architecture.in_channels,
                architecture.out_channels,
                kernel_size=architecture.local_kernel_size,
                padding=architecture.local_kernel_size // 2,
            )

        def forward(self, features: Any) -> Any:
            if features.ndim != 4 or features.shape[1] != self.architecture.in_channels:
                raise ValueError(
                    f"Model A expects N,{self.architecture.in_channels},Y,X inputs; got {tuple(features.shape)}"
                )
            return self.fno(features) + self.local(features)

else:  # pragma: no cover - only relevant without the optional ML environment
    ModelAResidualFNO = None  # type: ignore[assignment]


def model_a_architecture() -> ModelAArchitecture:
    """Return the declared, untuned starting architecture."""

    return ModelAArchitecture()


def build_model_a(architecture: ModelAArchitecture = ModelAArchitecture()) -> Any:
    require_model_a_runtime()
    return ModelAResidualFNO(architecture)


def stratified_training_records(pair_codes: Sequence[int], sample_count: int, seed: int) -> tuple[tuple[int, int], ...]:
    """Deterministically select balanced Model A overfit pairs from training only."""

    codes = np.asarray(pair_codes, dtype=np.uint8)
    candidates = np.flatnonzero(codes == 1)
    if candidates.size == 0:
        raise ValueError("dataset has no declared Model A training pairs")
    counts = [sample_count // 3 + int(index < sample_count % 3) for index in range(3)]
    if any(count > candidates.size for count in counts):
        raise ValueError("requested Model A sample count exceeds training pairs")
    records: list[tuple[int, int]] = []
    for experiment, count in enumerate(counts):
        selected = np.sort(np.random.default_rng(seed + experiment).choice(candidates, size=count, replace=False))
        records.extend((experiment, int(time_index)) for time_index in selected)
    return tuple(records)


def _normalization_from_group(group: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if tuple(group.attrs.get("state_channels", ())) != STATE_CHANNELS:
        raise ValueError("dataset state-channel order is not the declared 46-channel AF contract")
    if tuple(group.attrs.get("static_features", ())) != STATIC_FEATURES:
        raise ValueError("dataset static-feature order is not the declared Model A contract")
    mean = np.asarray(group["state_mean"][:], dtype=np.float32)
    scale = np.asarray(group["state_scale"][:], dtype=np.float32)
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    if mean.shape != (STATE_CHANNEL_COUNT,) or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError("invalid Model A state normalizers")
    wind = np.asarray(group["static_features"][:, 0], dtype=np.float32)
    values = wind[:, wet]
    wind_mean, wind_scale = float(values.mean()), float(values.std())
    if not np.isfinite(wind_mean) or not np.isfinite(wind_scale) or wind_scale <= 0:
        raise ValueError("invalid Model A wind normalization")
    return mean, scale, wet, wind_mean, wind_scale


class ModelAPairDataset(Dataset):
    """Lazy features and normalized ten-day residual targets for Model A."""

    def __init__(self, dataset_path: str | Path, records: Sequence[tuple[int, int]], *, horizon_days: int = 10) -> None:
        require_model_a_runtime()
        self.dataset_path = Path(dataset_path).resolve()
        self.records = tuple((int(experiment), int(time_index)) for experiment, time_index in records)
        self.horizon_days = int(horizon_days)
        if not self.records:
            raise ValueError("Model A dataset needs at least one pair")
        self._group: Any | None = None
        self._state: Any | None = None
        self._static: Any | None = None
        self._open()

    def _open(self) -> None:
        import zarr

        self._group = zarr.open_consolidated(str(self.dataset_path), mode="r")
        self._state, self._static = self._group["state"], self._group["static_features"]
        self.mean, self.scale, self.wet, self.wind_mean, self.wind_scale = _normalization_from_group(self._group)
        if self._state.shape[2] != STATE_CHANNEL_COUNT or self._static.shape[1] != STATIC_CHANNEL_COUNT:
            raise ValueError("dataset channel count does not match Model A")
        if any(time + self.horizon_days >= self._state.shape[1] for _, time in self.records):
            raise ValueError("a Model A pair exceeds the available trajectory length")

    def __len__(self) -> int:
        return len(self.records)

    def _normalise_state(self, value: np.ndarray) -> np.ndarray:
        result = (value - self.mean[:, None, None]) / self.scale[:, None, None]
        result[:, ~self.wet] = 0.0
        return np.ascontiguousarray(result, dtype=np.float32)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        experiment, time_index = self.records[index]
        present = self._normalise_state(np.asarray(self._state[experiment, time_index], dtype=np.float32))
        future = self._normalise_state(
            np.asarray(self._state[experiment, time_index + self.horizon_days], dtype=np.float32)
        )
        geometry = np.asarray(self._static[experiment], dtype=np.float32).copy()
        geometry[0] = (geometry[0] - self.wind_mean) / self.wind_scale
        geometry[0, ~self.wet] = 0.0
        features = np.ascontiguousarray(np.concatenate((present, geometry), axis=0), dtype=np.float32)
        return torch.from_numpy(features), torch.from_numpy(future - present), torch.from_numpy(future)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_group"] = state["_state"] = state["_static"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open()


class ChunkAwareBatchSampler:
    """Shuffle contiguous compressed-Zarr batches while preserving local reads."""

    def __init__(self, dataset: ModelAPairDataset, batch_size: int, seed: int) -> None:
        self.seed, self.epoch = int(seed), 0
        batches: list[tuple[int, ...]] = []
        current: list[int] = []
        previous: int | None = None
        for index, (experiment, _) in enumerate(dataset.records):
            if previous is not None and experiment != previous and current:
                batches.append(tuple(current))
                current = []
            current.append(index)
            if len(current) == batch_size:
                batches.append(tuple(current))
                current = []
            previous = experiment
        if current:
            batches.append(tuple(current))
        self.batches = tuple(batches)

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        order = list(range(len(self.batches)))
        random.Random(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        yield from (list(self.batches[index]) for index in order)


def masked_relative_l2(prediction: Any, target: Any, wet_mask: Any, *, epsilon: float = 1.0e-12) -> Any:
    """Mean per-sample masked relative L2 state loss for the modern model ladder."""

    if prediction.shape != target.shape:
        raise ValueError(f"Model A prediction/target mismatch: {prediction.shape} != {target.shape}")
    if wet_mask.shape != (1, 1, *prediction.shape[-2:]):
        raise ValueError("Model A wet mask must be shaped (1,1,Y,X)")
    numerator = ((prediction - target).square() * wet_mask).sum(dim=(1, 2, 3))
    denominator = (target.square() * wet_mask).sum(dim=(1, 2, 3)).clamp_min(epsilon)
    return torch.sqrt(numerator / denominator).mean()


def _epoch(model: Any, loader: Any, *, device: Any, wet: Any, optimizer: Any | None) -> float:
    training = optimizer is not None
    model.train(training)
    total, samples = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, _, future in loader:
            features = features.to(device=device, dtype=torch.float32, non_blocking=True)
            future = future.to(device=device, dtype=torch.float32, non_blocking=True)
            prediction = features[:, :STATE_CHANNEL_COUNT] + model(features)
            loss = masked_relative_l2(prediction, future, wet)
            if not bool(torch.isfinite(loss).item()):
                raise ModelATrainingError("Model A encountered a non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            size = int(features.shape[0])
            total += float(loss.detach().cpu()) * size
            samples += size
    if not samples:
        raise ModelATrainingError("Model A data loader produced no samples")
    return total / samples


def _persistence_loss(loader: Any, *, device: Any, wet: Any) -> float:
    """Score the identity-state baseline with Model A's exact relative-L2 loss."""

    total, samples = 0.0, 0
    with torch.no_grad():
        for features, _, future in loader:
            features = features.to(device=device, dtype=torch.float32, non_blocking=True)
            future = future.to(device=device, dtype=torch.float32, non_blocking=True)
            loss = masked_relative_l2(features[:, :STATE_CHANNEL_COUNT], future, wet)
            size = int(features.shape[0])
            total += float(loss.detach().cpu()) * size
            samples += size
    if not samples:
        raise ModelATrainingError("Model A persistence loader produced no samples")
    return total / samples


def _checkpoint_payload(
    architecture: ModelAArchitecture,
    config: ModelAOverfitConfig,
    dataset: ModelAPairDataset,
    records: Sequence[tuple[int, int]],
    model: Any,
    history: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model_class": "ModelAResidualFNO",
        "model_config": architecture.to_dict(),
        "model_a_overfit_config": asdict(config),
        "dataset": str(dataset.dataset_path),
        "records": [list(record) for record in records],
        "normalization": {
            "state_mean": dataset.mean.tolist(),
            "state_scale": dataset.scale.tolist(),
            "wind_mean": dataset.wind_mean,
            "wind_scale": dataset.wind_scale,
        },
        "state_target": "normalized_future_minus_normalized_present",
        "model_state_dict": _checkpoint_state_dict(model),
        "history": list(history),
    }


def _checkpoint_state_dict(model: Any) -> dict[str, Any]:
    """Copy only loadable parameters from NeuralOperator's augmented state dict."""

    # neuraloperator 2.0's FNO includes a top-level ``_metadata`` value in its
    # state dictionary.  PyTorch's load_state_dict treats it as an unexpected
    # parameter key, so it must not enter a portable experiment checkpoint.
    return copy.deepcopy({key: value for key, value in model.state_dict().items() if key != "_metadata"})


def run_overfit(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    config: ModelAOverfitConfig = ModelAOverfitConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run the Model A small-sample residual-learning and reload acceptance gate."""

    require_model_a_runtime()
    seed_everything(config.seed)
    dataset_path, output = Path(dataset_path).resolve(), Path(output_dir).resolve()
    checkpoint_path, report_path = output / "model_a_overfit_best.pt", output / "model_a_overfit_report.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite or reuse Model A overfit directory: {output}")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Model A CUDA run requested but no CUDA device is visible")
    device = torch.device(device_name)

    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    records = stratified_training_records(np.asarray(group["pair_split"][:], dtype=np.uint8), config.sample_count, config.seed)
    dataset = ModelAPairDataset(dataset_path, records, horizon_days=config.horizon_days)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
    )
    evaluation_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    wet = torch.from_numpy(dataset.wet.astype(np.float32))[None, None].to(device)
    architecture = model_a_architecture()
    attempts: list[dict[str, Any]] = []
    started = time.monotonic()
    for attempt, learning_rate in enumerate(config.learning_rates):
        seed_everything(config.seed + attempt)
        model = build_model_a(architecture).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, betas=config.adam_betas, weight_decay=config.weight_decay
        )
        initial_loss = _epoch(model, evaluation_loader, device=device, wet=wet, optimizer=None)
        history: list[dict[str, Any]] = []
        best_loss, best_state = float("inf"), None
        for epoch in range(1, config.epochs + 1):
            train_loss = _epoch(model, train_loader, device=device, wet=wet, optimizer=optimizer)
            evaluation_loss = _epoch(model, evaluation_loader, device=device, wet=wet, optimizer=None)
            history.append({"epoch": epoch, "learning_rate": learning_rate, "train_loss": train_loss, "evaluation_loss": evaluation_loss})
            if evaluation_loss < best_loss:
                best_loss, best_state = evaluation_loss, _checkpoint_state_dict(model)
        accepted = bool(best_state is not None and best_loss < initial_loss)
        attempts.append({"learning_rate": learning_rate, "initial_loss": initial_loss, "best_loss": best_loss, "final_loss": history[-1]["evaluation_loss"], "accepted": accepted})
        if not accepted:
            continue
        model.load_state_dict(best_state)
        model.eval()
        features, _, _ = dataset[0]
        features = features[None].to(device)
        with torch.no_grad():
            reference = model(features).detach().cpu()
        output.mkdir(parents=True, exist_ok=False)
        torch.save(_checkpoint_payload(architecture, config, dataset, records, model, history), checkpoint_path)
        restored = build_model_a(architecture).to(device)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        restored.load_state_dict(payload["model_state_dict"])
        restored.eval()
        with torch.no_grad():
            reloaded = restored(features).detach().cpu()
        bitwise_exact = bool(torch.equal(reference, reloaded))
        if not bitwise_exact:
            raise ModelATrainingError("Model A checkpoint reload changed deterministic inference")
        report = {
            "status": "accepted",
            "dataset": str(dataset_path),
            "device": str(device),
            "neuraloperator_version": metadata.version("neuraloperator"),
            "architecture": architecture.to_dict(),
            "config": asdict(config),
            "records": [list(record) for record in records],
            "attempts": attempts,
            "selected_learning_rate": learning_rate,
            "save_reload_bitwise_exact": bitwise_exact,
            "elapsed_seconds": time.monotonic() - started,
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    raise ModelATrainingError("Model A did not lower its held-sample state loss: " + json.dumps(attempts, sort_keys=True))


def run_development(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    config: ModelADevelopmentConfig = ModelADevelopmentConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train Model A on sealed chronological pairs and evaluate the held validation split."""

    require_model_a_runtime()
    seed_everything(config.seed)
    dataset_path, output = Path(dataset_path).resolve(), Path(output_dir).resolve()
    checkpoint_path = output / "model_a_development_best.pt"
    report_path = output / "model_a_development_report.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite or reuse Model A development directory: {output}")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Model A CUDA run requested but no CUDA device is visible")
    device = torch.device(device_name)

    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    train_records = records_for_pair_split(pair_codes, 1)
    validation_records = records_for_pair_split(pair_codes, 2)
    train_dataset = ModelAPairDataset(dataset_path, train_records)
    validation_dataset = ModelAPairDataset(dataset_path, validation_records)
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
    wet = torch.from_numpy(train_dataset.wet.astype(np.float32))[None, None].to(device)
    architecture = model_a_architecture()
    model = build_model_a(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=config.adam_betas, weight_decay=config.weight_decay
    )
    persistence = _persistence_loss(validation_loader, device=device, wet=wet)
    history: list[dict[str, Any]] = []
    best_validation, best_state = float("inf"), None
    started = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        train_loss = _epoch(model, train_loader, device=device, wet=wet, optimizer=optimizer)
        validation_loss = _epoch(model, validation_loader, device=device, wet=wet, optimizer=None)
        history.append({"epoch": epoch, "learning_rate": config.learning_rate, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_validation:
            best_validation, best_state = validation_loss, _checkpoint_state_dict(model)
    if best_state is None:  # pragma: no cover - defensive after finite training
        raise ModelATrainingError("Model A development run never produced a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    features, _, _ = validation_dataset[0]
    features = features[None].to(device)
    with torch.no_grad():
        reference = model(features).detach().cpu()
    output.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "model_class": "ModelAResidualFNO",
            "model_config": architecture.to_dict(),
            "model_a_development_config": asdict(config),
            "dataset": str(dataset_path),
            "split_contract": {"train_pair_code": 1, "validation_pair_code": 2},
            "pair_counts": {"train": len(train_records), "validation": len(validation_records)},
            "normalization": {
                "state_mean": train_dataset.mean.tolist(),
                "state_scale": train_dataset.scale.tolist(),
                "wind_mean": train_dataset.wind_mean,
                "wind_scale": train_dataset.wind_scale,
            },
            "state_target": "normalized_future_minus_normalized_present",
            "model_state_dict": _checkpoint_state_dict(model),
            "history": history,
        },
        checkpoint_path,
    )
    restored = build_model_a(architecture).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored.load_state_dict(payload["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        reloaded = restored(features).detach().cpu()
    bitwise_exact = bool(torch.equal(reference, reloaded))
    if not bitwise_exact:
        raise ModelATrainingError("Model A development checkpoint reload changed deterministic inference")
    report = {
        "status": "accepted" if best_validation < persistence else "completed_without_persistence_skill",
        "dataset": str(dataset_path),
        "device": str(device),
        "neuraloperator_version": metadata.version("neuraloperator"),
        "architecture": architecture.to_dict(),
        "config": asdict(config),
        "pair_counts": {"train": len(train_records), "validation": len(validation_records)},
        "persistence_validation_loss": persistence,
        "best_validation_loss": best_validation,
        "final_validation_loss": history[-1]["validation_loss"],
        "best_epoch": next(entry["epoch"] for entry in history if entry["validation_loss"] == best_validation),
        "save_reload_bitwise_exact": bitwise_exact,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def run_final(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    config: ModelAFinalConfig = ModelAFinalConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train the one frozen Model A realization and score validation/inference pairs."""

    require_model_a_runtime()
    seed_everything(config.seed)
    dataset_path, output = Path(dataset_path).resolve(), Path(output_dir).resolve()
    checkpoint_path = output / "model_a_final.pt"
    report_path = output / "model_a_final_report.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite or reuse frozen Model A directory: {output}")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Model A CUDA run requested but no CUDA device is visible")
    device = torch.device(device_name)

    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    train_records = records_for_pair_split(pair_codes, 1)
    validation_records = records_for_pair_split(pair_codes, 2)
    inference_records = records_for_pair_split(pair_codes, 3)
    train_dataset = ModelAPairDataset(dataset_path, train_records)
    validation_dataset = ModelAPairDataset(dataset_path, validation_records)
    inference_dataset = ModelAPairDataset(dataset_path, inference_records)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=ChunkAwareBatchSampler(train_dataset, config.batch_size, config.seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    evaluation_loaders = {
        "validation": DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0),
        "inference": DataLoader(inference_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0),
    }
    wet = torch.from_numpy(train_dataset.wet.astype(np.float32))[None, None].to(device)
    architecture = model_a_architecture()
    model = build_model_a(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=config.adam_betas, weight_decay=config.weight_decay
    )
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        train_loss = _epoch(model, train_loader, device=device, wet=wet, optimizer=optimizer)
        history.append({"epoch": epoch, "learning_rate": config.learning_rate, "train_loss": train_loss})
    evaluation = {
        name: {
            "model_a_loss": _epoch(model, loader, device=device, wet=wet, optimizer=None),
            "persistence_loss": _persistence_loss(loader, device=device, wet=wet),
        }
        for name, loader in evaluation_loaders.items()
    }
    model.eval()
    features, _, _ = inference_dataset[0]
    features = features[None].to(device)
    with torch.no_grad():
        reference = model(features).detach().cpu()
    output.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "model_class": "ModelAResidualFNO",
            "model_config": architecture.to_dict(),
            "model_a_final_config": asdict(config),
            "dataset": str(dataset_path),
            "split_contract": {"train_pair_code": 1, "validation_pair_code": 2, "inference_pair_code": 3},
            "pair_counts": {"train": len(train_records), "validation": len(validation_records), "inference": len(inference_records)},
            "normalization": {
                "state_mean": train_dataset.mean.tolist(),
                "state_scale": train_dataset.scale.tolist(),
                "wind_mean": train_dataset.wind_mean,
                "wind_scale": train_dataset.wind_scale,
            },
            "state_target": "normalized_future_minus_normalized_present",
            "model_state_dict": _checkpoint_state_dict(model),
            "history": history,
        },
        checkpoint_path,
    )
    restored = build_model_a(architecture).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored.load_state_dict(payload["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        reloaded = restored(features).detach().cpu()
    bitwise_exact = bool(torch.equal(reference, reloaded))
    if not bitwise_exact:
        raise ModelATrainingError("frozen Model A checkpoint reload changed deterministic inference")
    report = {
        "status": "frozen",
        "dataset": str(dataset_path),
        "device": str(device),
        "neuraloperator_version": metadata.version("neuraloperator"),
        "architecture": architecture.to_dict(),
        "config": asdict(config),
        "pair_counts": {"train": len(train_records), "validation": len(validation_records), "inference": len(inference_records)},
        "evaluation": evaluation,
        "save_reload_bitwise_exact": bitwise_exact,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the modern state-only Model A FNO")
    commands = parser.add_subparsers(dest="command", required=True)
    overfit = commands.add_parser("overfit", help="run the declared 20--100 sample residual-learning gate")
    overfit.add_argument("--dataset", type=Path, required=True)
    overfit.add_argument("--output-dir", type=Path, required=True)
    overfit.add_argument("--samples", type=int, default=96)
    overfit.add_argument("--epochs", type=int, default=160)
    overfit.add_argument("--batch-size", type=int, default=4)
    overfit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    develop = commands.add_parser("develop", help="train sealed chronological pairs and assess held validation")
    develop.add_argument("--dataset", type=Path, required=True)
    develop.add_argument("--output-dir", type=Path, required=True)
    develop.add_argument("--epochs", type=int, default=12)
    develop.add_argument("--batch-size", type=int, default=8)
    develop.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    final = commands.add_parser("final", help="run the one frozen Model A realization")
    final.add_argument("--dataset", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, required=True)
    final.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "overfit":
        result = run_overfit(
            args.dataset,
            args.output_dir,
            config=ModelAOverfitConfig(sample_count=args.samples, epochs=args.epochs, batch_size=args.batch_size),
            device_name=args.device,
        )
    elif args.command == "develop":
        result = run_development(
            args.dataset,
            args.output_dir,
            config=ModelADevelopmentConfig(epochs=args.epochs, batch_size=args.batch_size),
            device_name=args.device,
        )
    elif args.command == "final":
        result = run_final(args.dataset, args.output_dir, device_name=args.device)
    else:  # pragma: no cover - argparse enforces the command
        raise ValueError(f"unsupported Model A command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
