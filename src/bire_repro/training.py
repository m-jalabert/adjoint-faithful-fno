"""Deterministic pretraining and two-step fine-tuning for the paper FNO."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .fno import (
    DEPENDENCY_MESSAGE,
    PaperFNO2d,
    PaperFNOConfig,
    build_paper_fno,
    require_fno_dependencies,
)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except (ImportError, OSError):  # pragma: no cover - depends on login-node env
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]


class TrainingFailure(RuntimeError):
    """Raised after every locked learning-rate attempt fails acceptance."""


class NonFiniteTraining(RuntimeError):
    """Raised as soon as loss or gradients become non-finite."""


def _require_training_dependencies() -> None:
    require_fno_dependencies()
    if torch is None or DataLoader is None:  # pragma: no cover
        raise RuntimeError(DEPENDENCY_MESSAGE)


def seed_everything(seed: int = 1024) -> None:
    """Seed all stochastic libraries and request deterministic kernels."""

    _require_training_dependencies()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True)
class DataConfig:
    """Location and exact temporal split of the consolidated daily data."""

    store: Path
    stats: Path | None = None
    variable: str = "state"
    experiment_ids: tuple[int, ...] = (1, 3, 5)
    train_start: int = 0
    train_stop: int = 6000
    validation_start: int = 6000
    validation_stop: int = 7200
    state_channels: int = 10
    wind_channel: int = 10
    epsilon: float = 1.0e-5

    def __post_init__(self) -> None:
        object.__setattr__(self, "store", Path(self.store))
        if self.stats is not None:
            object.__setattr__(self, "stats", Path(self.stats))
        object.__setattr__(
            self, "experiment_ids", tuple(int(item) for item in self.experiment_ids)
        )
        if not self.experiment_ids:
            raise ValueError("experiment_ids cannot be empty")
        if not (0 <= self.train_start < self.train_stop):
            raise ValueError("invalid training interval")
        if not (0 <= self.validation_start < self.validation_stop):
            raise ValueError("invalid validation interval")
        if self.train_stop > self.validation_start:
            raise ValueError("training and validation intervals overlap")
        if self.state_channels <= 0 or self.wind_channel < self.state_channels:
            raise ValueError("state/wind channel indices are inconsistent")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    """Locked optimizer protocol from the accepted reproduction plan."""

    batch_size: int = 8
    seed: int = 1024
    learning_rates: tuple[float, ...] = (0.01, 0.001, 0.0005)
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    l1_weight: float = 0.01
    scheduler_t_max: int = 3
    scheduler_eta_min: float = 1.0e-5
    pretrain_max_epochs: int = 30
    pretrain_patience: int = 5
    finetune_max_epochs: int = 15
    finetune_patience: int = 3
    num_workers: int = 0
    pin_memory: bool = True
    chunk_aware_batches: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "learning_rates", tuple(float(item) for item in self.learning_rates)
        )
        object.__setattr__(self, "adam_betas", tuple(self.adam_betas))
        if self.batch_size <= 0 or self.seed < 0:
            raise ValueError("batch_size must be positive and seed non-negative")
        if not self.learning_rates or any(rate <= 0 for rate in self.learning_rates):
            raise ValueError("learning_rates must contain positive values")
        if len(self.adam_betas) != 2 or any(not 0 <= beta < 1 for beta in self.adam_betas):
            raise ValueError("adam_betas must contain two values in [0, 1)")
        if self.weight_decay != 0:
            raise ValueError("the paper-faithful protocol requires zero weight decay")
        if self.l1_weight < 0 or self.scheduler_t_max <= 0:
            raise ValueError("invalid loss/scheduler settings")
        for name in (
            "pretrain_max_epochs",
            "pretrain_patience",
            "finetune_max_epochs",
            "finetune_patience",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "TrainingConfig":
        if values is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        kwargs = {key: value for key, value in values.items() if key in allowed}
        for name in ("learning_rates", "adam_betas"):
            if name in kwargs:
                kwargs[name] = tuple(kwargs[name])
        return cls(**kwargs)


@dataclass(frozen=True)
class PointwiseNormalizer:
    """Pointwise channel statistics computed from the 18,000 training states."""

    mean: np.ndarray
    std: np.ndarray
    epsilon: float = 1.0e-5

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if mean.shape != std.shape or mean.ndim != 3:
            raise ValueError(
                "pointwise mean/std must have identical (channel, y, x) shapes"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("normalization statistics contain NaN or Inf")
        if np.any(std < 0):
            raise ValueError("normalization standard deviations cannot be negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def load(
        cls, stats_path: str | Path, *, epsilon: float = 1.0e-5
    ) -> "PointwiseNormalizer":
        path = Path(stats_path)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                mean = _first_present(payload, ("mean", "means", "timemeans"))
                std = _first_present(
                    payload, ("raw_std", "std", "stdevs", "timestdevs")
                )
            return cls(mean, std, epsilon)

        try:
            import zarr
        except (ImportError, OSError) as exc:  # pragma: no cover
            raise RuntimeError(
                "Reading pointwise statistics from Zarr requires the locked zarr package"
            ) from exc
        group = zarr.open_group(str(path), mode="r")
        mean_name = _first_group_key(group, ("mean", "means", "timemeans"))
        std_name = _first_group_key(
            group, ("raw_std", "std", "stdevs", "timestdevs")
        )
        return cls(
            np.asarray(group[mean_name][:]),
            np.asarray(group[std_name][:]),
            epsilon,
        )

    def normalize(self, values: np.ndarray, *, channels: slice | None = None) -> np.ndarray:
        selection = channels if channels is not None else slice(None)
        return np.asarray(
            (values - self.mean[selection]) / (self.std[selection] + self.epsilon),
            dtype=np.float32,
        )

    def denormalize(
        self, values: np.ndarray, *, channels: slice | None = None
    ) -> np.ndarray:
        selection = channels if channels is not None else slice(None)
        return np.asarray(
            values * (self.std[selection] + self.epsilon) + self.mean[selection],
            dtype=np.float32,
        )

    def for_stride(self, stride: int) -> "PointwiseNormalizer":
        if stride <= 0:
            raise ValueError("stride must be positive")
        return PointwiseNormalizer(
            self.mean[..., ::stride, ::stride],
            self.std[..., ::stride, ::stride],
            self.epsilon,
        )


def _first_present(payload: Any, names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name in payload:
            return np.asarray(payload[name])
    raise KeyError(f"none of {tuple(names)} was found")


def _first_group_key(group: Any, names: Sequence[str]) -> str:
    for name in names:
        if name in group:
            return name
    raise KeyError(f"none of {tuple(names)} was found in the Zarr group")


class SequenceSource:
    """Read-only adapter for ``(experiment,time,channel,y,x)`` data."""

    def __init__(self, store: str | Path, variable: str = "state") -> None:
        self.store = Path(store)
        self.requested_variable = variable
        self._array: Any = None
        self._group: Any = None
        self._experiment_values: tuple[int, ...] | None = None
        self.variable = variable
        self._open()

    def _open(self) -> None:
        if self.store.suffix == ".npy":
            self._array = np.load(self.store, mmap_mode="r", allow_pickle=False)
        elif self.store.suffix == ".npz":
            self._group = np.load(self.store, mmap_mode="r", allow_pickle=False)
            candidates = (self.requested_variable, "state", "data", "fields")
            self.variable = next(
                (name for name in candidates if name in self._group), ""
            )
            if not self.variable:
                raise KeyError(f"no state array found in {self.store}")
            self._array = self._group[self.variable]
            if "experiment" in self._group:
                self._experiment_values = tuple(
                    int(item) for item in self._group["experiment"]
                )
        else:
            try:
                import zarr
            except (ImportError, OSError) as exc:  # pragma: no cover
                raise RuntimeError(
                    "Reading the consolidated dataset requires the locked zarr package"
                ) from exc
            self._group = zarr.open_group(str(self.store), mode="r")
            candidates = (self.requested_variable, "state", "data", "fields")
            self.variable = next((name for name in candidates if name in self._group), "")
            if not self.variable:
                raise KeyError(f"no state array found in {self.store}")
            self._array = self._group[self.variable]
            if "experiment" in self._group:
                self._experiment_values = tuple(
                    int(item) for item in np.asarray(self._group["experiment"][:])
                )

        if getattr(self._array, "ndim", None) != 5:
            raise ValueError(
                f"{self.store}:{self.variable} must have shape "
                "(experiment,time,channel,y,x)"
            )

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return tuple(int(item) for item in self._array.shape)  # type: ignore[return-value]

    def experiment_index(self, experiment_id: int) -> int:
        if self._experiment_values is not None:
            try:
                return self._experiment_values.index(int(experiment_id))
            except ValueError as exc:
                raise KeyError(f"experiment {experiment_id} is absent from the store") from exc
        if 1 <= experiment_id <= self.shape[0]:
            return experiment_id - 1
        if 0 <= experiment_id < self.shape[0]:
            return experiment_id
        raise KeyError(f"experiment {experiment_id} cannot map to axis length {self.shape[0]}")

    def read(self, experiment_index: int, time_index: int) -> np.ndarray:
        values = np.asarray(self._array[experiment_index, time_index], dtype=np.float32)
        if values.ndim != 3:
            raise ValueError("a source state must have (channel,y,x) shape")
        return values

    def read_many(
        self, experiment_index: int, time_indices: Sequence[int]
    ) -> np.ndarray:
        """Read one experiment's times in one chunk-aware storage request."""

        indices = np.asarray(time_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("time_indices must be a non-empty one-dimensional sequence")
        try:
            if hasattr(self._array, "oindex"):
                values = self._array.oindex[experiment_index, indices, :, :, :]
            else:
                values = self._array[experiment_index, indices, :, :, :]
            result = np.asarray(values, dtype=np.float32)
        except (IndexError, TypeError, ValueError):
            result = np.stack(
                [self.read(experiment_index, int(index)) for index in indices], axis=0
            )
        if result.shape != (len(indices), *self.shape[2:]):
            raise ValueError(f"multi-state read returned unexpected shape {result.shape}")
        return result

    def close(self) -> None:
        close = getattr(self._group, "close", None)
        if callable(close):
            close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_array"] = None
        state["_group"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open()


class AutoregressivePairDataset(Dataset):
    """Lazy one- or two-step samples without crossing split boundaries."""

    def __init__(
        self,
        source: SequenceSource,
        normalizer: PointwiseNormalizer,
        *,
        experiment_ids: Sequence[int],
        start: int,
        stop: int,
        lag_days: int,
        steps: int,
        state_channels: int = 10,
        wind_channel: int = 10,
    ) -> None:
        _require_training_dependencies()
        if lag_days <= 0 or steps not in (1, 2):
            raise ValueError("lag_days must be positive and steps must be 1 or 2")
        if stop > source.shape[1]:
            raise ValueError(f"split stop {stop} exceeds {source.shape[1]} source times")
        final_start = stop - steps * lag_days
        if final_start <= start:
            raise ValueError("split is too short for the requested autoregressive horizon")
        required_channels = max(state_channels, wind_channel + 1)
        if source.shape[2] < required_channels:
            raise ValueError(
                f"source has {source.shape[2]} channels; {required_channels} are required"
            )
        if normalizer.mean.shape != source.shape[2:]:
            raise ValueError(
                f"stats shape {normalizer.mean.shape} != source field shape {source.shape[2:]}"
            )
        self.source = source
        self.normalizer = normalizer
        self.lag_days = int(lag_days)
        self.steps = int(steps)
        self.state_channels = int(state_channels)
        self.wind_channel = int(wind_channel)
        self.samples = tuple(
            (source.experiment_index(int(experiment_id)), time_index)
            for experiment_id in experiment_ids
            for time_index in range(start, final_start)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        experiment_index, time_index = self.samples[index]
        raw_input = self.source.read(experiment_index, time_index)
        raw_targets = [
            self.source.read(experiment_index, time_index + step * self.lag_days)
            for step in range(1, self.steps + 1)
        ]
        return self._format_sample(raw_input, raw_targets)

    def _format_sample(
        self, raw_input: np.ndarray, raw_targets: Sequence[np.ndarray]
    ) -> tuple[Any, Any]:
        input_channels = np.concatenate(
            (
                raw_input[: self.state_channels],
                raw_input[self.wind_channel : self.wind_channel + 1],
            ),
            axis=0,
        )
        # Current production layout is ten state fields followed by wind. Keep
        # this explicit so an accidental channel reorder fails loudly.
        if self.wind_channel != self.state_channels:
            mean = np.concatenate(
                (
                    self.normalizer.mean[: self.state_channels],
                    self.normalizer.mean[self.wind_channel : self.wind_channel + 1],
                ),
                axis=0,
            )
            std = np.concatenate(
                (
                    self.normalizer.std[: self.state_channels],
                    self.normalizer.std[self.wind_channel : self.wind_channel + 1],
                ),
                axis=0,
            )
            input_values = (input_channels - mean) / (std + self.normalizer.epsilon)
        else:
            input_values = self.normalizer.normalize(
                input_channels, channels=slice(0, self.state_channels + 1)
            )

        targets = []
        for raw_target_all_channels in raw_targets:
            raw_target = raw_target_all_channels[: self.state_channels]
            targets.append(
                self.normalizer.normalize(
                    raw_target, channels=slice(0, self.state_channels)
                )
            )

        x = torch.from_numpy(np.ascontiguousarray(input_values, dtype=np.float32))
        y = tuple(
            torch.from_numpy(np.ascontiguousarray(item, dtype=np.float32))
            for item in targets
        )
        return x, y[0] if self.steps == 1 else y

    def __getitems__(self, indices: Sequence[int]) -> list[tuple[Any, Any]]:
        """Vectorize a DataLoader batch so each Zarr chunk is decompressed once."""

        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_position, dataset_index in enumerate(indices):
            experiment_index, time_index = self.samples[int(dataset_index)]
            grouped.setdefault(experiment_index, []).append((output_position, time_index))
        output: list[tuple[Any, Any] | None] = [None] * len(indices)
        for experiment_index, requests in grouped.items():
            times = [time for _, time in requests]
            raw_inputs = self.source.read_many(experiment_index, times)
            raw_targets_by_step = [
                self.source.read_many(
                    experiment_index,
                    [time + step * self.lag_days for time in times],
                )
                for step in range(1, self.steps + 1)
            ]
            for local_index, (output_position, _) in enumerate(requests):
                output[output_position] = self._format_sample(
                    raw_inputs[local_index],
                    [values[local_index] for values in raw_targets_by_step],
                )
        if any(item is None for item in output):  # pragma: no cover - defensive
            raise RuntimeError("failed to assemble a vectorized data batch")
        return output  # type: ignore[return-value]


class ChunkAwareBatchSampler:
    """Shuffle contiguous time blocks while retaining efficient Zarr access."""

    def __init__(self, dataset: AutoregressivePairDataset, batch_size: int, seed: int):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        batches: list[list[int]] = []
        current: list[int] = []
        previous_experiment: int | None = None
        for index, (experiment, _) in enumerate(dataset.samples):
            if previous_experiment is not None and experiment != previous_experiment:
                if current:
                    batches.append(current)
                current = []
            current.append(index)
            if len(current) == self.batch_size:
                batches.append(current)
                current = []
            previous_experiment = experiment
        if current:
            batches.append(current)
        self.batches = tuple(tuple(batch) for batch in batches)

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        order = list(range(len(self.batches)))
        generator.shuffle(order)
        self.epoch += 1
        for batch_index in order:
            batch = list(self.batches[batch_index])
            generator.shuffle(batch)
            yield batch


def forecast_loss(
    prediction: Any, target: Any, *, l1_weight: float = 0.01
) -> Any:
    """Elementwise MSE + 0.01 MAE, averaged over batch/channel/grid."""

    _require_training_dependencies()
    return torch.mean((prediction - target) ** 2) + l1_weight * torch.mean(
        torch.abs(prediction - target)
    )


def autoregressive_two_step(
    model: PaperFNO2d, x: Any, *, state_channels: int = 10
) -> tuple[Any, Any]:
    """Predict two steps while retaining the static wind input channel."""

    first = model(x)
    forcing = x[:, state_channels : state_channels + 1]
    second = model(torch.cat((first, forcing), dim=1))
    return first, second


def two_step_loss(
    first: Any,
    second: Any,
    target_first: Any,
    target_second: Any,
    *,
    l1_weight: float = 0.01,
) -> Any:
    return forecast_loss(first, target_first, l1_weight=l1_weight) + forecast_loss(
        second, target_second, l1_weight=l1_weight
    )


def persistence_validation_loss(
    data_loader: Any, *, state_channels: int = 10, l1_weight: float = 0.01
) -> float:
    """Evaluate the exact one-step loss for the held-state baseline."""

    _require_training_dependencies()
    total = 0.0
    samples = 0
    for x, target in data_loader:
        batch = int(x.shape[0])
        value = forecast_loss(
            x[:, :state_channels], target, l1_weight=l1_weight
        ).item()
        total += value * batch
        samples += batch
    if samples == 0:
        raise ValueError("validation loader is empty")
    return total / samples


def _worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_data_loader(
    dataset: Any,
    config: TrainingConfig,
    *,
    shuffle: bool,
    seed_offset: int = 0,
) -> Any:
    _require_training_dependencies()
    generator = torch.Generator()
    generator.manual_seed(config.seed + seed_offset)
    common = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory and torch.cuda.is_available(),
        "worker_init_fn": _worker_seed if config.num_workers else None,
        "generator": generator,
        "persistent_workers": config.num_workers > 0,
    }
    if (
        shuffle
        and config.chunk_aware_batches
        and isinstance(dataset, AutoregressivePairDataset)
    ):
        return DataLoader(
            dataset,
            batch_sampler=ChunkAwareBatchSampler(
                dataset, config.batch_size, config.seed + seed_offset
            ),
            **common,
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=False,
        **common,
    )


def _run_epoch(
    model: PaperFNO2d,
    data_loader: Any,
    *,
    device: Any,
    optimizer: Any | None,
    two_step: bool,
    state_channels: int,
    l1_weight: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, target in data_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            if two_step:
                target_first, target_second = (
                    item.to(device=device, dtype=torch.float32, non_blocking=True)
                    for item in target
                )
                first, second = autoregressive_two_step(
                    model, x, state_channels=state_channels
                )
                loss = two_step_loss(
                    first,
                    second,
                    target_first,
                    target_second,
                    l1_weight=l1_weight,
                )
            else:
                target = target.to(device=device, dtype=torch.float32, non_blocking=True)
                loss = forecast_loss(model(x), target, l1_weight=l1_weight)

            if not bool(torch.isfinite(loss).item()):
                raise NonFiniteTraining("encountered non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    parameter.grad is not None
                    and not bool(torch.all(torch.isfinite(parameter.grad)).item())
                    for parameter in model.parameters()
                ):
                    raise NonFiniteTraining("encountered non-finite gradient")
                optimizer.step()
            batch = int(x.shape[0])
            total += float(loss.detach().item()) * batch
            samples += batch
    if samples == 0:
        raise ValueError("data loader is empty")
    return total / samples


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(destination)


def _atomic_json_dump(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def _checkpoint_payload(
    *,
    model: PaperFNO2d,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    stage: str,
    learning_rate: float,
    best_validation_loss: float,
    persistence_loss: float,
    lag_days: int,
    history: Sequence[Mapping[str, Any]],
    training_config: TrainingConfig,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_class": "PaperFNO2d",
        "model_config": model.architecture_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "learning_rate": learning_rate,
        "best_validation_loss": best_validation_loss,
        "persistence_validation_loss": persistence_loss,
        "lag_days": lag_days,
        "history": list(history),
        "training_config": asdict(training_config),
        "provenance": dict(provenance),
    }


def _train_stage(
    model: PaperFNO2d,
    train_loader: Any,
    validation_loader: Any,
    *,
    device: Any,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    two_step: bool,
    stage: str,
    checkpoint_path: Path,
    lag_days: int,
    state_channels: int,
    persistence_loss: float,
    config: TrainingConfig,
    provenance: Mapping[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    # A rerun must never accept a stale best checkpoint after failing before
    # its first finite epoch.
    checkpoint_path.unlink(missing_ok=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=config.adam_betas,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.scheduler_t_max,
        eta_min=config.scheduler_eta_min,
    )
    best = math.inf
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            two_step=two_step,
            state_channels=state_channels,
            l1_weight=config.l1_weight,
        )
        validation_loss = _run_epoch(
            model,
            validation_loader,
            device=device,
            optimizer=None,
            two_step=two_step,
            state_channels=state_channels,
            l1_weight=config.l1_weight,
        )
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(entry)
        if not (math.isfinite(train_loss) and math.isfinite(validation_loss)):
            raise NonFiniteTraining(f"non-finite {stage} epoch metrics")

        if validation_loss < best:
            best = validation_loss
            stale_epochs = 0
            _atomic_torch_save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    stage=stage,
                    learning_rate=learning_rate,
                    best_validation_loss=best,
                    persistence_loss=persistence_loss,
                    lag_days=lag_days,
                    history=history,
                    training_config=config,
                    provenance=provenance,
                ),
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= patience:
            break

    if not checkpoint_path.exists():
        raise NonFiniteTraining(f"{stage} did not produce a finite checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return best, history


def train_model_with_retries(
    *,
    architecture: PaperFNOConfig,
    pretrain_train: Any,
    pretrain_validation: Any,
    finetune_train: Any,
    finetune_validation: Any,
    lag_days: int,
    output_dir: str | Path,
    config: TrainingConfig | None = None,
    state_channels: int = 10,
    device: str | Any | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the locked LR retry protocol and return a machine-readable summary."""

    _require_training_dependencies()
    config = config or TrainingConfig()
    if lag_days not in (5, 10, 30):
        raise ValueError("production lag_days must be one of 5, 10, or 30")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    run_provenance = dict(provenance or {})
    run_provenance.update(
        {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "torch_version": torch.__version__,
            "device": str(resolved_device),
            "device_name": (
                torch.cuda.get_device_name(resolved_device)
                if resolved_device.type == "cuda"
                else "cpu"
            ),
        }
    )

    baseline_loader = make_data_loader(pretrain_validation, config, shuffle=False)
    persistence = persistence_validation_loss(
        baseline_loader,
        state_channels=state_channels,
        l1_weight=config.l1_weight,
    )
    attempts: list[dict[str, Any]] = []

    for attempt_index, learning_rate in enumerate(config.learning_rates):
        seed_everything(config.seed)
        attempt_dir = destination / f"attempt_{attempt_index + 1}_lr_{learning_rate:g}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt: dict[str, Any] = {
            "learning_rate": learning_rate,
            "status": "running",
            "deviation_from_published_lr": learning_rate != config.learning_rates[0],
        }
        attempts.append(attempt)
        try:
            # Do not pass a real dtype here: SpectralConv owns complex64 weights.
            model = build_paper_fno(architecture).to(resolved_device)
            pretrain_train_loader = make_data_loader(
                pretrain_train, config, shuffle=True, seed_offset=attempt_index
            )
            pretrain_validation_loader = make_data_loader(
                pretrain_validation, config, shuffle=False
            )
            pretrain_path = attempt_dir / "pretrain_best.pt"
            pretrain_best, pretrain_history = _train_stage(
                model,
                pretrain_train_loader,
                pretrain_validation_loader,
                device=resolved_device,
                learning_rate=learning_rate,
                max_epochs=config.pretrain_max_epochs,
                patience=config.pretrain_patience,
                two_step=False,
                stage="pretrain",
                checkpoint_path=pretrain_path,
                lag_days=lag_days,
                state_channels=state_channels,
                persistence_loss=persistence,
                config=config,
                provenance=run_provenance,
            )
            attempt["pretrain_best_validation_loss"] = pretrain_best
            attempt["persistence_validation_loss"] = persistence
            history_path = attempt_dir / "history.json"
            _atomic_json_dump({"pretrain": pretrain_history}, history_path)
            if not pretrain_best < persistence:
                attempt["status"] = "rejected_not_better_than_persistence"
                _atomic_json_dump(attempt, attempt_dir / "attempt.json")
                continue

            # _train_stage has already restored the best one-step weights.
            finetune_train_loader = make_data_loader(
                finetune_train, config, shuffle=True, seed_offset=100 + attempt_index
            )
            finetune_validation_loader = make_data_loader(
                finetune_validation, config, shuffle=False
            )
            finetune_path = attempt_dir / "finetune_best.pt"
            finetune_best, finetune_history = _train_stage(
                model,
                finetune_train_loader,
                finetune_validation_loader,
                device=resolved_device,
                learning_rate=learning_rate,
                max_epochs=config.finetune_max_epochs,
                patience=config.finetune_patience,
                two_step=True,
                stage="finetune",
                checkpoint_path=finetune_path,
                lag_days=lag_days,
                state_channels=state_channels,
                persistence_loss=persistence,
                config=config,
                provenance=run_provenance,
            )
            finetune_one_step = _run_epoch(
                model,
                make_data_loader(pretrain_validation, config, shuffle=False),
                device=resolved_device,
                optimizer=None,
                two_step=False,
                state_channels=state_channels,
                l1_weight=config.l1_weight,
            )
            if not finetune_one_step < persistence:
                attempt.update(
                    status="rejected_finetune_not_better_than_persistence",
                    finetune_one_step_validation_loss=finetune_one_step,
                )
                _atomic_json_dump(
                    {"pretrain": pretrain_history, "finetune": finetune_history},
                    history_path,
                )
                _atomic_json_dump(attempt, attempt_dir / "attempt.json")
                continue
            attempt.update(
                status="accepted",
                finetune_best_validation_loss=finetune_best,
                finetune_one_step_validation_loss=finetune_one_step,
                pretrain_epochs=len(pretrain_history),
                finetune_epochs=len(finetune_history),
            )
            _atomic_json_dump(
                {"pretrain": pretrain_history, "finetune": finetune_history},
                history_path,
            )
            _atomic_json_dump(attempt, attempt_dir / "attempt.json")
            shutil.copy2(pretrain_path, destination / "pretrain_best.pt")
            shutil.copy2(finetune_path, destination / "finetune_best.pt")
            shutil.copy2(history_path, destination / "training_history.json")
            summary = {
                "status": "accepted",
                "lag_days": lag_days,
                "accepted_learning_rate": learning_rate,
                "published_learning_rate_used": attempt_index == 0,
                "persistence_validation_loss": persistence,
                "pretrain_best_validation_loss": pretrain_best,
                "finetune_best_validation_loss": finetune_best,
                "finetune_one_step_validation_loss": finetune_one_step,
                "architecture": architecture.to_dict(),
                "training": asdict(config),
                "attempts": attempts,
                "checkpoint": str((destination / "finetune_best.pt").resolve()),
                "provenance": run_provenance,
            }
            _atomic_json_dump(summary, destination / "training_summary.json")
            return summary
        except NonFiniteTraining as exc:
            attempt.update(status="rejected_non_finite", reason=str(exc))
            _atomic_json_dump(attempt, attempt_dir / "attempt.json")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "status": "failed",
        "lag_days": lag_days,
        "persistence_validation_loss": persistence,
        "attempts": attempts,
        "provenance": run_provenance,
    }
    _atomic_json_dump(summary, destination / "training_summary.json")
    raise TrainingFailure(
        "all learning-rate attempts were non-finite or failed to beat validation persistence"
    )


def load_model_checkpoint(
    checkpoint_path: str | Path, *, device: str | Any = "cpu"
) -> tuple[PaperFNO2d, dict[str, Any]]:
    """Restore a trusted checkpoint produced by this module."""

    _require_training_dependencies()
    payload = torch.load(
        Path(checkpoint_path), map_location=torch.device(device), weights_only=False
    )
    if payload.get("model_class") != "PaperFNO2d":
        raise ValueError("checkpoint is not a PaperFNO2d checkpoint")
    model = build_paper_fno(PaperFNOConfig.from_mapping(payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    # Preserve SpectralConv's complex64 parameters while moving devices.
    model.to(torch.device(device))
    return model, payload


def checkpoint_sha256(checkpoint_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(checkpoint_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_configuration(
    configuration: str | Path | Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    if isinstance(configuration, Mapping):
        payload = dict(configuration)
        config_path = payload.get("_config_path")
        return payload, Path(config_path).resolve().parent if config_path else Path.cwd()
    path = Path(configuration).resolve()
    if path.suffix.lower() == ".toml":
        from .config import load_config

        payload = load_config(path)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("reading a YAML reproduction manifest requires PyYAML") from exc
        payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    return dict(payload), path.parent


def _path_from_config(
    config: Mapping[str, Any],
    base: Path,
    *,
    names: Sequence[tuple[str, ...]],
    required: bool,
) -> Path | None:
    for route in names:
        value: Any = config
        for key in route:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            result = Path(value)
            return result if result.is_absolute() else (base / result).resolve()
    if required:
        routes = [".".join(route) for route in names]
        raise KeyError(f"configuration requires one of {routes}")
    return None


def _data_and_normalizer_from_config(
    config: Mapping[str, Any], base: Path
) -> tuple[DataConfig, SequenceSource, PointwiseNormalizer]:
    data_values = config.get("data", {})
    if not isinstance(data_values, Mapping):
        raise ValueError("data configuration must be a mapping")
    store = _path_from_config(
        config,
        base,
        names=(
            ("data", "store"),
            ("data", "zarr"),
            ("paths", "reduced_zarr"),
            ("paths", "reduced"),
        ),
        required=True,
    )
    if store is None:  # pragma: no cover - required=True already raises
        raise KeyError("missing consolidated data store")
    # The canonical manifest stores a reduced-data directory, not duplicated
    # product filenames.
    if store.name == "reduced" or (store.suffix == "" and not store.name.endswith(".zarr")):
        store = store / "mitgcm_state.zarr"
    stats = _path_from_config(
        config,
        base,
        names=(("data", "stats"), ("paths", "normalization_stats")),
        required=False,
    )
    if stats is None and isinstance(config.get("paths"), Mapping):
        reduced = config["paths"].get("reduced")
        if reduced is not None:
            reduced_path = Path(reduced)
            if not reduced_path.is_absolute():
                reduced_path = (base / reduced_path).resolve()
            stats = reduced_path / "normalization.zarr"
    data_config = DataConfig(
        store=store,
        stats=stats,
        variable=str(data_values.get("variable", "state")),
        experiment_ids=tuple(data_values.get("training_experiments", (1, 3, 5))),
        train_start=int(data_values.get("training_start", data_values.get("train_start", 0))),
        train_stop=int(data_values.get("training_stop", data_values.get("train_stop", 6000))),
        validation_start=int(data_values.get("validation_start", 6000)),
        validation_stop=int(data_values.get("validation_stop", 7200)),
        state_channels=int(data_values.get("state_channels", 10)),
        wind_channel=int(data_values.get("wind_channel", 10)),
        epsilon=float(
            data_values.get("normalization_epsilon", data_values.get("epsilon", 1.0e-5))
        ),
    )
    source = SequenceSource(data_config.store, data_config.variable)
    normalizer = PointwiseNormalizer.load(
        data_config.stats or data_config.store, epsilon=data_config.epsilon
    )
    return data_config, source, normalizer


def _architecture_from_config(config: Mapping[str, Any]) -> PaperFNOConfig:
    values = config.get("fno", {})
    if not isinstance(values, Mapping):
        raise ValueError("fno configuration must be a mapping")
    nested = values.get("architecture")
    if isinstance(nested, Mapping):
        return PaperFNOConfig.from_mapping(nested)
    translated = {
        "in_channels": values.get("input_channels", 11),
        "out_channels": values.get("output_channels", 10),
        "lifting_channels": values.get("lift_channels", 256),
        "hidden_channels": values.get("latent_channels", 128),
        "projection_channels": values.get("projection_channels", 256),
        "channel_mlp_channels": values.get("hidden_channels", 512),
        "n_layers": values.get("blocks", 3),
        "n_modes": (
            values.get("modes_y", 64),
            values.get("modes_x", 64),
        ),
        "norm_eps": values.get("norm_eps", 1.0e-6),
    }
    return PaperFNOConfig.from_mapping(translated)


def _training_from_config(config: Mapping[str, Any]) -> TrainingConfig:
    fno = config.get("fno", {})
    if not isinstance(fno, Mapping):
        raise ValueError("fno configuration must be a mapping")
    nested = fno.get("training", config.get("training"))
    if isinstance(nested, Mapping):
        return TrainingConfig.from_mapping(nested)
    primary_lr = float(fno.get("learning_rate", 0.01))
    fallbacks = tuple(float(item) for item in fno.get("fallback_learning_rates", (0.001, 0.0005)))
    translated = {
        "batch_size": fno.get("batch_size", 8),
        "seed": config.get("seed", 1024),
        "learning_rates": (primary_lr, *fallbacks),
        "adam_betas": (fno.get("adam_beta1", 0.9), fno.get("adam_beta2", 0.95)),
        "weight_decay": fno.get("weight_decay", 0.0),
        "l1_weight": fno.get("mae_weight", 0.01),
        "scheduler_t_max": fno.get("cosine_t_max", 3),
        "scheduler_eta_min": fno.get("cosine_eta_min", 1.0e-5),
        "pretrain_max_epochs": fno.get("pretrain_epochs", 30),
        "pretrain_patience": fno.get("pretrain_patience", 5),
        "finetune_max_epochs": fno.get("finetune_epochs", 15),
        "finetune_patience": fno.get("finetune_patience", 3),
        "num_workers": fno.get("num_workers", 0),
        "pin_memory": fno.get("pin_memory", True),
    }
    return TrainingConfig.from_mapping(translated)


def train_from_config(
    configuration: str | Path | Mapping[str, Any],
    *,
    lag_days: int,
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Manifest-facing entry point used by ``repro fno train``."""

    config, base = _load_configuration(configuration)
    data_config, source, normalizer = _data_and_normalizer_from_config(config, base)
    architecture = _architecture_from_config(config)
    training_config = _training_from_config(config)

    pretrain_train = AutoregressivePairDataset(
        source,
        normalizer,
        experiment_ids=data_config.experiment_ids,
        start=data_config.train_start,
        stop=data_config.train_stop,
        lag_days=lag_days,
        steps=1,
        state_channels=data_config.state_channels,
        wind_channel=data_config.wind_channel,
    )
    pretrain_validation = AutoregressivePairDataset(
        source,
        normalizer,
        experiment_ids=data_config.experiment_ids,
        start=data_config.validation_start,
        stop=data_config.validation_stop,
        lag_days=lag_days,
        steps=1,
        state_channels=data_config.state_channels,
        wind_channel=data_config.wind_channel,
    )
    finetune_train = AutoregressivePairDataset(
        source,
        normalizer,
        experiment_ids=data_config.experiment_ids,
        start=data_config.train_start,
        stop=data_config.train_stop,
        lag_days=lag_days,
        steps=2,
        state_channels=data_config.state_channels,
        wind_channel=data_config.wind_channel,
    )
    finetune_validation = AutoregressivePairDataset(
        source,
        normalizer,
        experiment_ids=data_config.experiment_ids,
        start=data_config.validation_start,
        stop=data_config.validation_stop,
        lag_days=lag_days,
        steps=2,
        state_channels=data_config.state_channels,
        wind_channel=data_config.wind_channel,
    )
    if output_dir is None:
        root = _path_from_config(
            config,
            base,
            names=(
                ("paths", "fno_checkpoints"),
                ("paths", "checkpoints"),
                ("fno", "output_dir"),
            ),
            required=False,
        ) or (base / "outputs" / "fno")
        destination = root / f"lag_{lag_days:02d}d"
    else:
        destination = Path(output_dir)
    from .config import config_sha256

    upstream = config.get("upstream", {})
    provenance = {
        "config_sha256": config_sha256(config),
        "data_store": str(data_config.store.resolve()),
        "normalization_stats": (
            str(data_config.stats.resolve()) if data_config.stats is not None else None
        ),
        "mitgcm_sha": upstream.get("mitgcm_sha") if isinstance(upstream, Mapping) else None,
        "paper_code_sha": (
            upstream.get("paper_code_sha") if isinstance(upstream, Mapping) else None
        ),
        "neuraloperator_version_locked": (
            upstream.get("neuraloperator_version")
            if isinstance(upstream, Mapping)
            else None
        ),
    }
    return train_model_with_retries(
        architecture=architecture,
        pretrain_train=pretrain_train,
        pretrain_validation=pretrain_validation,
        finetune_train=finetune_train,
        finetune_validation=finetune_validation,
        lag_days=lag_days,
        output_dir=destination,
        config=training_config,
        state_channels=data_config.state_channels,
        device=device,
        provenance=provenance,
    )


def train_all_lags(
    configuration: str | Path | Mapping[str, Any],
    *,
    lags: Iterable[int] = (5, 10, 30),
    output_root: str | Path | None = None,
    device: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Sequential convenience wrapper; Slurm normally runs each lag separately."""

    results = {}
    for lag_days in lags:
        destination = (
            None
            if output_root is None
            else Path(output_root) / f"lag_{int(lag_days):02d}d"
        )
        results[int(lag_days)] = train_from_config(
            configuration,
            lag_days=int(lag_days),
            output_dir=destination,
            device=device,
        )
    return results


__all__ = [
    "AutoregressivePairDataset",
    "ChunkAwareBatchSampler",
    "DataConfig",
    "NonFiniteTraining",
    "PointwiseNormalizer",
    "SequenceSource",
    "TrainingConfig",
    "TrainingFailure",
    "autoregressive_two_step",
    "checkpoint_sha256",
    "forecast_loss",
    "load_model_checkpoint",
    "make_data_loader",
    "persistence_validation_loss",
    "seed_everything",
    "train_all_lags",
    "train_from_config",
    "train_model_with_retries",
    "two_step_loss",
]
