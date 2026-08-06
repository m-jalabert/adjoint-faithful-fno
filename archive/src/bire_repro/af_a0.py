"""Adapted Bire-architecture A0 overfit gate for the shared AF--FNO dataset.

A0 deliberately retains the recovered one-step FNO architecture and loss while making
only benchmark-forced changes: a 46-channel Markov state, one wind-stress input channel,
and resolution-equivalent 16 Fourier modes on the 62 by 62 tutorial grid.  This module
is intentionally separate from the retired paper reconstruction workflow.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .af_data import STATIC_FEATURES, STATE_CHANNELS
from .fno import PaperFNOConfig, build_paper_fno, require_fno_dependencies

try:  # Keep data-only commands importable when the ML runtime is unavailable.
    import torch
    from torch.utils.data import DataLoader, Dataset
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]


STATE_CHANNEL_COUNT = len(STATE_CHANNELS)
WIND_FEATURE_INDEX = STATIC_FEATURES.index("wind_stress_x")


class A0TrainingError(RuntimeError):
    """Raised when the mandatory A0 overfit acceptance gate is not met."""


@dataclass(frozen=True)
class A0OverfitConfig:
    """Frozen small-sample acceptance settings for the adapted Bire baseline."""

    sample_count: int = 96
    horizon_days: int = 10
    batch_size: int = 4
    epochs: int = 160
    learning_rates: tuple[float, ...] = (0.01, 0.001, 0.0005)
    adam_betas: tuple[float, float] = (0.9, 0.95)
    l1_weight: float = 0.01
    cosine_t_max: int = 3
    cosine_eta_min: float = 1.0e-5
    seed: int = 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "learning_rates", tuple(float(rate) for rate in self.learning_rates))
        if not 20 <= self.sample_count <= 100:
            raise ValueError("A0 overfit sample_count must be in the declared 20--100 range")
        if self.horizon_days != 10:
            raise ValueError("A0 is locked to the shared 10-day forecast interval")
        if self.batch_size <= 0 or self.epochs <= 0 or self.seed < 0:
            raise ValueError("batch_size, epochs, and seed must be positive")
        if not self.learning_rates or any(rate <= 0 for rate in self.learning_rates):
            raise ValueError("learning_rates must be positive")
        if len(self.adam_betas) != 2 or any(not 0 <= value < 1 for value in self.adam_betas):
            raise ValueError("adam_betas must be two values in [0, 1)")
        if self.l1_weight < 0 or self.cosine_t_max <= 0 or self.cosine_eta_min <= 0:
            raise ValueError("invalid paper-style loss or scheduler settings")


@dataclass(frozen=True)
class A0DevelopmentConfig:
    """One controlled 10-day A0 development run before the frozen baseline."""

    batch_size: int = 8
    epochs: int = 12
    learning_rate: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.95)
    l1_weight: float = 0.01
    cosine_t_max: int = 3
    cosine_eta_min: float = 1.0e-5
    seed: int = 1024

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0 or self.learning_rate <= 0 or self.seed < 0:
            raise ValueError("batch_size, epochs, learning_rate, and seed must be positive")
        if len(self.adam_betas) != 2 or any(not 0 <= value < 1 for value in self.adam_betas):
            raise ValueError("adam_betas must be two values in [0, 1)")
        if self.l1_weight < 0 or self.cosine_t_max <= 0 or self.cosine_eta_min <= 0:
            raise ValueError("invalid paper-style loss or scheduler settings")


@dataclass(frozen=True)
class A0FinalConfig:
    """Frozen A0 settings selected once from the development validation result."""

    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.95)
    l1_weight: float = 0.01
    cosine_t_max: int = 3
    cosine_eta_min: float = 1.0e-5
    seed: int = 1024

    def __post_init__(self) -> None:
        if self.batch_size != 8 or self.epochs != 10 or self.learning_rate != 0.01:
            raise ValueError("the frozen A0 final configuration is batch=8, epochs=10, learning_rate=0.01")
        if self.adam_betas != (0.9, 0.95) or self.l1_weight != 0.01:
            raise ValueError("the frozen A0 optimizer/loss settings must match the development gate")
        if self.cosine_t_max != 3 or self.cosine_eta_min != 1.0e-5 or self.seed != 1024:
            raise ValueError("the frozen A0 scheduler/seed settings must match the development gate")


def a0_architecture() -> PaperFNOConfig:
    """Return the declared architecture-transfer configuration, not a modern model."""

    return PaperFNOConfig(
        in_channels=STATE_CHANNEL_COUNT + 1,
        out_channels=STATE_CHANNEL_COUNT,
        n_modes=(16, 16),
    )


def _require_training_runtime() -> None:
    require_fno_dependencies()
    if torch is None or DataLoader is None:  # pragma: no cover - environment dependent
        raise RuntimeError("A0 training requires the project PyTorch environment")


def seed_everything(seed: int) -> None:
    _require_training_runtime()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def stratified_training_records(pair_codes: Sequence[int], sample_count: int, seed: int) -> tuple[tuple[int, int], ...]:
    """Choose deterministic, regime-balanced training pairs from code 1 only."""

    codes = np.asarray(pair_codes, dtype=np.uint8)
    candidates = np.flatnonzero(codes == 1)
    if candidates.size == 0:
        raise ValueError("the dataset has no declared training pairs")
    counts = [sample_count // 3 + int(index < sample_count % 3) for index in range(3)]
    if any(count > candidates.size for count in counts):
        raise ValueError("requested A0 overfit sample count exceeds available training pairs")
    records: list[tuple[int, int]] = []
    for experiment_index, count in enumerate(counts):
        generator = np.random.default_rng(seed + experiment_index)
        selected = np.sort(generator.choice(candidates, size=count, replace=False))
        records.extend((experiment_index, int(time_index)) for time_index in selected)
    return tuple(records)


def records_for_pair_split(pair_codes: Sequence[int], split_code: int) -> tuple[tuple[int, int], ...]:
    """Return all declared split pairs in regime-major chronological order."""

    if split_code not in {1, 2, 3}:
        raise ValueError("split_code must be one of train=1, validation=2, inference=3")
    times = np.flatnonzero(np.asarray(pair_codes, dtype=np.uint8) == split_code)
    if times.size == 0:
        raise ValueError(f"the dataset has no pair records for split code {split_code}")
    return tuple((experiment_index, int(time_index)) for experiment_index in range(3) for time_index in times)


def _normalization_from_group(group: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    channels = tuple(group.attrs.get("state_channels", ()))
    if channels != STATE_CHANNELS:
        raise ValueError("dataset state channel order is not the declared 46-channel AF contract")
    if tuple(group.attrs.get("static_features", ())) != STATIC_FEATURES:
        raise ValueError("dataset static-feature order is not the declared AF contract")
    mean = np.asarray(group["state_mean"][:], dtype=np.float32)
    scale = np.asarray(group["state_scale"][:], dtype=np.float32)
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    if mean.shape != (STATE_CHANNEL_COUNT,) or scale.shape != mean.shape or np.any(scale <= 0.0):
        raise ValueError("invalid state normalizers")
    wind = np.asarray(group["static_features"][:, WIND_FEATURE_INDEX], dtype=np.float32)
    wind_wet = wind[:, wet]
    wind_mean = float(wind_wet.mean())
    wind_scale = float(wind_wet.std())
    if not np.isfinite(wind_mean) or not np.isfinite(wind_scale) or wind_scale <= 0.0:
        raise ValueError("invalid wind-stress normalization")
    return mean, scale, wet, wind_mean, wind_scale


class A0PairDataset(Dataset):
    """Lazy normalized 10-day state pairs with one normalized wind input channel."""

    def __init__(
        self,
        dataset_path: str | Path,
        records: Sequence[tuple[int, int]],
        *,
        horizon_days: int = 10,
    ) -> None:
        _require_training_runtime()
        self.dataset_path = Path(dataset_path).resolve()
        self.records = tuple((int(experiment), int(time_index)) for experiment, time_index in records)
        self.horizon_days = int(horizon_days)
        if not self.records:
            raise ValueError("A0 pair dataset requires at least one record")
        self._group: Any | None = None
        self._state: Any | None = None
        self._static: Any | None = None
        self._open()

    def _open(self) -> None:
        import zarr

        self._group = zarr.open_consolidated(str(self.dataset_path), mode="r")
        self._state = self._group["state"]
        self._static = self._group["static_features"]
        self.mean, self.scale, self.wet, self.wind_mean, self.wind_scale = _normalization_from_group(self._group)
        if self._state.shape[2] != STATE_CHANNEL_COUNT:
            raise ValueError("dataset state channel count does not match A0")
        if any(time_index + self.horizon_days >= self._state.shape[1] for _, time_index in self.records):
            raise ValueError("an A0 pair exceeds the available trajectory length")

    def __len__(self) -> int:
        return len(self.records)

    def _normalize_state(self, raw: np.ndarray) -> np.ndarray:
        normalized = (raw - self.mean[:, None, None]) / self.scale[:, None, None]
        normalized[:, ~self.wet] = 0.0
        return np.ascontiguousarray(normalized, dtype=np.float32)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        experiment_index, time_index = self.records[index]
        raw_input = np.asarray(self._state[experiment_index, time_index], dtype=np.float32)
        raw_target = np.asarray(self._state[experiment_index, time_index + self.horizon_days], dtype=np.float32)
        wind = np.asarray(self._static[experiment_index, WIND_FEATURE_INDEX], dtype=np.float32)
        normalized_wind = (wind - self.wind_mean) / self.wind_scale
        normalized_wind[~self.wet] = 0.0
        features = np.concatenate((self._normalize_state(raw_input), normalized_wind[None]), axis=0)
        return torch.from_numpy(features), torch.from_numpy(self._normalize_state(raw_target))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_group"] = None
        state["_state"] = None
        state["_static"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open()


class A0ChunkAwareBatchSampler:
    """Shuffle contiguous time batches so each compressed Zarr chunk is reused."""

    def __init__(self, dataset: A0PairDataset, batch_size: int, seed: int) -> None:
        self.seed = int(seed)
        self.epoch = 0
        batches: list[tuple[int, ...]] = []
        current: list[int] = []
        previous_experiment: int | None = None
        for index, (experiment_index, _) in enumerate(dataset.records):
            if previous_experiment is not None and experiment_index != previous_experiment and current:
                batches.append(tuple(current))
                current = []
            current.append(index)
            if len(current) == batch_size:
                batches.append(tuple(current))
                current = []
            previous_experiment = experiment_index
        if current:
            batches.append(tuple(current))
        self.batches = tuple(batches)

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        order = list(range(len(self.batches)))
        generator.shuffle(order)
        self.epoch += 1
        yield from (list(self.batches[index]) for index in order)


def masked_paper_loss(prediction: Any, target: Any, wet_mask: Any, *, l1_weight: float = 0.01) -> Any:
    """Paper MSE + 0.01 MAE, restricted to physical ocean cells."""

    _require_training_runtime()
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} != {target.shape}")
    if wet_mask.ndim != 4 or wet_mask.shape[1] != 1:
        raise ValueError("wet_mask must have shape (1,1,y,x)")
    masked = (prediction - target) * wet_mask
    denominator = prediction.shape[0] * prediction.shape[1] * wet_mask.sum()
    return masked.square().sum() / denominator + l1_weight * masked.abs().sum() / denominator


def _epoch(
    model: Any,
    loader: Any,
    *,
    device: Any,
    wet_mask: Any,
    optimizer: Any | None,
    l1_weight: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, target in loader:
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            prediction = model(inputs)
            loss = masked_paper_loss(prediction, target, wet_mask, l1_weight=l1_weight)
            if not bool(torch.isfinite(loss).item()):
                raise A0TrainingError("A0 encountered a non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = int(inputs.shape[0])
            total += float(loss.detach().cpu()) * batch_size
            samples += batch_size
    if samples == 0:
        raise A0TrainingError("A0 loader produced no samples")
    return total / samples


def _persistence_loss(loader: Any, *, device: Any, wet_mask: Any, l1_weight: float) -> float:
    _require_training_runtime()
    total = 0.0
    samples = 0
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            value = masked_paper_loss(inputs[:, :STATE_CHANNEL_COUNT], target, wet_mask, l1_weight=l1_weight)
            batch_size = int(inputs.shape[0])
            total += float(value.detach().cpu()) * batch_size
            samples += batch_size
    if samples == 0:
        raise A0TrainingError("persistence baseline loader produced no samples")
    return total / samples


def _checkpoint_payload(
    architecture: PaperFNOConfig,
    config: A0OverfitConfig,
    dataset: A0PairDataset,
    records: Sequence[tuple[int, int]],
    model: Any,
    history: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model_class": "PaperFNO2d",
        "model_config": architecture.to_dict(),
        "a0_overfit_config": asdict(config),
        "dataset": str(dataset.dataset_path),
        "records": [list(record) for record in records],
        "normalization": {
            "state_mean": dataset.mean.tolist(),
            "state_scale": dataset.scale.tolist(),
            "wind_mean": dataset.wind_mean,
            "wind_scale": dataset.wind_scale,
        },
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "history": list(history),
    }


def run_overfit(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    config: A0OverfitConfig = A0OverfitConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run the mandatory A0 overfit/save-reload acceptance gate."""

    _require_training_runtime()
    seed_everything(config.seed)
    dataset_path = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    checkpoint_path = output / "a0_overfit_best.pt"
    report_path = output / "a0_overfit_report.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite or reuse existing A0 overfit directory: {output}")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A0 CUDA run requested but no CUDA device is visible")
    device = torch.device(device_name)

    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    records = stratified_training_records(pair_codes, config.sample_count, config.seed)
    dataset = A0PairDataset(dataset_path, records, horizon_days=config.horizon_days)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
    )
    evaluation_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    wet = torch.from_numpy(dataset.wet.astype(np.float32))[None, None].to(device)
    architecture = a0_architecture()
    attempts: list[dict[str, Any]] = []
    started = time.monotonic()

    for attempt_index, learning_rate in enumerate(config.learning_rates):
        seed_everything(config.seed + attempt_index)
        model = build_paper_fno(architecture).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=config.adam_betas, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.cosine_t_max, eta_min=config.cosine_eta_min
        )
        initial_loss = _epoch(
            model, evaluation_loader, device=device, wet_mask=wet, optimizer=None, l1_weight=config.l1_weight
        )
        history: list[dict[str, Any]] = []
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        for epoch in range(1, config.epochs + 1):
            train_loss = _epoch(
                model, loader, device=device, wet_mask=wet, optimizer=optimizer, l1_weight=config.l1_weight
            )
            evaluation_loss = _epoch(
                model, evaluation_loader, device=device, wet_mask=wet, optimizer=None, l1_weight=config.l1_weight
            )
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train_loss": train_loss,
                    "evaluation_loss": evaluation_loss,
                }
            )
            if evaluation_loss < best_loss:
                best_loss = evaluation_loss
                best_state = copy.deepcopy(model.state_dict())
            scheduler.step()
        final_loss = history[-1]["evaluation_loss"]
        attempts.append(
            {
                "learning_rate": learning_rate,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "best_loss": best_loss,
                "accepted": bool(best_loss < initial_loss and final_loss < initial_loss),
            }
        )
        if best_state is None or not (best_loss < initial_loss and final_loss < initial_loss):
            continue

        model.load_state_dict(best_state)
        model.eval()
        first_input, _ = dataset[0]
        first_input = first_input[None].to(device)
        with torch.no_grad():
            reference_output = model(first_input).detach().cpu()
        output.mkdir(parents=True, exist_ok=False)
        torch.save(
            _checkpoint_payload(architecture, config, dataset, records, model, history),
            checkpoint_path,
        )
        restored = build_paper_fno(architecture).to(device)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        restored.load_state_dict(payload["model_state_dict"])
        restored.eval()
        with torch.no_grad():
            restored_output = restored(first_input).detach().cpu()
        reload_exact = bool(torch.equal(reference_output, restored_output))
        if not reload_exact:
            raise A0TrainingError("A0 checkpoint reload changed deterministic inference")
        report = {
            "status": "accepted",
            "dataset": str(dataset_path),
            "device": str(device),
            "architecture": architecture.to_dict(),
            "config": asdict(config),
            "records": [list(record) for record in records],
            "attempts": attempts,
            "selected_learning_rate": learning_rate,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "best_loss": best_loss,
            "save_reload_bitwise_exact": reload_exact,
            "elapsed_seconds": time.monotonic() - started,
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    raise A0TrainingError(
        "A0 did not reduce the held-sample loss for any declared learning-rate attempt: "
        + json.dumps(attempts, sort_keys=True)
    )


def run_development(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    config: A0DevelopmentConfig = A0DevelopmentConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train A0 on the sealed train pairs and assess it on sealed validation pairs."""

    _require_training_runtime()
    seed_everything(config.seed)
    dataset_path = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    checkpoint_path = output / "a0_development_best.pt"
    report_path = output / "a0_development_report.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite or reuse existing A0 development directory: {output}")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A0 CUDA run requested but no CUDA device is visible")
    device = torch.device(device_name)

    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    train_records = records_for_pair_split(pair_codes, 1)
    validation_records = records_for_pair_split(pair_codes, 2)
    train_dataset = A0PairDataset(dataset_path, train_records)
    validation_dataset = A0PairDataset(dataset_path, validation_records)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=A0ChunkAwareBatchSampler(train_dataset, config.batch_size, config.seed),
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
    architecture = a0_architecture()
    model = build_paper_fno(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.adam_betas, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.cosine_t_max, eta_min=config.cosine_eta_min
    )
    persistence = _persistence_loss(
        validation_loader, device=device, wet_mask=wet, l1_weight=config.l1_weight
    )
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_state: dict[str, Any] | None = None
    started = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        train_loss = _epoch(
            model, train_loader, device=device, wet_mask=wet, optimizer=optimizer, l1_weight=config.l1_weight
        )
        validation_loss = _epoch(
            model, validation_loader, device=device, wet_mask=wet, optimizer=None, l1_weight=config.l1_weight
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step()
    if best_state is None:  # pragma: no cover - defensive after finite training
        raise A0TrainingError("A0 development run never produced a checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    first_input, _ = validation_dataset[0]
    first_input = first_input[None].to(device)
    with torch.no_grad():
        reference_output = model(first_input).detach().cpu()
    output.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "model_class": "PaperFNO2d",
            "model_config": architecture.to_dict(),
            "a0_development_config": asdict(config),
            "dataset": str(dataset_path),
            "split_contract": {"train_pair_code": 1, "validation_pair_code": 2},
            "pair_counts": {"train": len(train_records), "validation": len(validation_records)},
            "normalization": {
                "state_mean": train_dataset.mean.tolist(),
                "state_scale": train_dataset.scale.tolist(),
                "wind_mean": train_dataset.wind_mean,
                "wind_scale": train_dataset.wind_scale,
            },
            "model_state_dict": model.state_dict(),
            "history": history,
        },
        checkpoint_path,
    )
    restored = build_paper_fno(architecture).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored.load_state_dict(payload["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        restored_output = restored(first_input).detach().cpu()
    reload_exact = bool(torch.equal(reference_output, restored_output))
    if not reload_exact:
        raise A0TrainingError("A0 development checkpoint reload changed deterministic inference")
    report = {
        "status": "accepted" if best_validation < persistence else "completed_without_persistence_skill",
        "dataset": str(dataset_path),
        "device": str(device),
        "architecture": architecture.to_dict(),
        "config": asdict(config),
        "pair_counts": {"train": len(train_records), "validation": len(validation_records)},
        "persistence_validation_loss": persistence,
        "best_validation_loss": best_validation,
        "final_validation_loss": history[-1]["validation_loss"],
        "save_reload_bitwise_exact": reload_exact,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def run_final(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    config: A0FinalConfig = A0FinalConfig(),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train the one frozen A0 realization and score held validation/inference pairs."""

    _require_training_runtime()
    seed_everything(config.seed)
    dataset_path = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    checkpoint_path = output / "a0_final.pt"
    report_path = output / "a0_final_report.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite or reuse existing frozen A0 directory: {output}")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A0 CUDA run requested but no CUDA device is visible")
    device = torch.device(device_name)

    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    train_records = records_for_pair_split(pair_codes, 1)
    validation_records = records_for_pair_split(pair_codes, 2)
    inference_records = records_for_pair_split(pair_codes, 3)
    train_dataset = A0PairDataset(dataset_path, train_records)
    validation_dataset = A0PairDataset(dataset_path, validation_records)
    inference_dataset = A0PairDataset(dataset_path, inference_records)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=A0ChunkAwareBatchSampler(train_dataset, config.batch_size, config.seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    evaluation_loaders = {
        "validation": DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0),
        "inference": DataLoader(inference_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0),
    }
    wet = torch.from_numpy(train_dataset.wet.astype(np.float32))[None, None].to(device)
    architecture = a0_architecture()
    model = build_paper_fno(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.adam_betas, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.cosine_t_max, eta_min=config.cosine_eta_min
    )
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        train_loss = _epoch(
            model, train_loader, device=device, wet_mask=wet, optimizer=optimizer, l1_weight=config.l1_weight
        )
        history.append(
            {"epoch": epoch, "learning_rate": float(optimizer.param_groups[0]["lr"]), "train_loss": train_loss}
        )
        scheduler.step()
    evaluation = {
        name: {
            "a0_loss": _epoch(model, loader, device=device, wet_mask=wet, optimizer=None, l1_weight=config.l1_weight),
            "persistence_loss": _persistence_loss(loader, device=device, wet_mask=wet, l1_weight=config.l1_weight),
        }
        for name, loader in evaluation_loaders.items()
    }
    model.eval()
    first_input, _ = inference_dataset[0]
    first_input = first_input[None].to(device)
    with torch.no_grad():
        reference_output = model(first_input).detach().cpu()
    output.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "model_class": "PaperFNO2d",
            "model_config": architecture.to_dict(),
            "a0_final_config": asdict(config),
            "dataset": str(dataset_path),
            "split_contract": {"train_pair_code": 1, "validation_pair_code": 2, "inference_pair_code": 3},
            "pair_counts": {"train": len(train_records), "validation": len(validation_records), "inference": len(inference_records)},
            "normalization": {
                "state_mean": train_dataset.mean.tolist(),
                "state_scale": train_dataset.scale.tolist(),
                "wind_mean": train_dataset.wind_mean,
                "wind_scale": train_dataset.wind_scale,
            },
            "model_state_dict": model.state_dict(),
            "history": history,
        },
        checkpoint_path,
    )
    restored = build_paper_fno(architecture).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored.load_state_dict(payload["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        restored_output = restored(first_input).detach().cpu()
    reload_exact = bool(torch.equal(reference_output, restored_output))
    if not reload_exact:
        raise A0TrainingError("frozen A0 checkpoint reload changed deterministic inference")
    report = {
        "status": "frozen",
        "dataset": str(dataset_path),
        "device": str(device),
        "architecture": architecture.to_dict(),
        "config": asdict(config),
        "pair_counts": {"train": len(train_records), "validation": len(validation_records), "inference": len(inference_records)},
        "evaluation": evaluation,
        "save_reload_bitwise_exact": reload_exact,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the adapted Bire A0 overfit gate")
    commands = parser.add_subparsers(dest="command", required=True)
    overfit = commands.add_parser("overfit", help="train 20--100 samples and verify save/reload")
    overfit.add_argument("--dataset", type=Path, required=True)
    overfit.add_argument("--output-dir", type=Path, required=True)
    overfit.add_argument("--samples", type=int, default=96)
    overfit.add_argument("--epochs", type=int, default=160)
    overfit.add_argument("--batch-size", type=int, default=4)
    overfit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    development = commands.add_parser("develop", help="train on chronological pairs and score validation")
    development.add_argument("--dataset", type=Path, required=True)
    development.add_argument("--output-dir", type=Path, required=True)
    development.add_argument("--epochs", type=int, default=12)
    development.add_argument("--batch-size", type=int, default=8)
    development.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    final = commands.add_parser("final", help="run the frozen A0 realization and score held pairs")
    final.add_argument("--dataset", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, required=True)
    final.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "overfit":
        config = A0OverfitConfig(sample_count=args.samples, epochs=args.epochs, batch_size=args.batch_size)
        result = run_overfit(args.dataset, args.output_dir, config=config, device_name=args.device)
    elif args.command == "develop":
        config = A0DevelopmentConfig(epochs=args.epochs, batch_size=args.batch_size)
        result = run_development(args.dataset, args.output_dir, config=config, device_name=args.device)
    elif args.command == "final":
        result = run_final(args.dataset, args.output_dir, device_name=args.device)
    else:  # pragma: no cover - argparse enforces this
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
