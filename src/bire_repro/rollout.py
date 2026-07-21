"""Autoregressive full- and stride-8 rollout for trained paper FNOs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .training import (
    PointwiseNormalizer,
    SequenceSource,
    _data_and_normalizer_from_config,
    _load_configuration,
    _path_from_config,
    checkpoint_sha256,
    load_model_checkpoint,
)

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - login-node environment
    torch = None  # type: ignore[assignment]


class NonFiniteRollout(RuntimeError):
    """Raised when an autoregressive trajectory develops NaN or Inf."""


RECOVERED_INITIAL_INDICES = (
    387,
    1860,
    701,
    1056,
    1358,
    1159,
    49,
    463,
    676,
    833,
    1531,
    189,
    1972,
    2030,
    35,
)


def rollout_days(horizon_days: int, lag_days: int) -> np.ndarray:
    """Return model times not exceeding the requested physical horizon."""

    if horizon_days < 0 or lag_days <= 0:
        raise ValueError("horizon_days must be non-negative and lag_days positive")
    return np.arange(horizon_days // lag_days + 1, dtype=np.int32) * lag_days


def stride_for_resolution(resolution: str | int) -> int:
    if isinstance(resolution, int):
        if resolution <= 0:
            raise ValueError("resolution stride must be positive")
        return resolution
    normalized = resolution.lower().replace("_", "-")
    if normalized in {"full", "0.25deg", "quarter-degree", "native"}:
        return 1
    if normalized in {"low", "coarse", "2deg", "stride-8", "stride8"}:
        return 8
    raise ValueError(f"unknown rollout resolution {resolution!r}")


def spatial_subsample(values: Any, stride: int) -> Any:
    """Match the archived ``[..., ::8, ::8]`` coarsening operation."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    return values[..., ::stride, ::stride]


def iter_autoregressive(
    model: Any,
    initial_state: Any,
    forcing: Any,
    *,
    n_steps: int,
) -> Iterator[tuple[int, Any]]:
    """Yield normalized states while retaining the static forcing channel."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("FNO rollout requires the project's PyTorch environment")
    if n_steps < 0:
        raise ValueError("n_steps cannot be negative")
    if initial_state.ndim != 4 or forcing.ndim != 4:
        raise ValueError("initial_state and forcing must be NCHW tensors")
    if initial_state.shape[0] != forcing.shape[0] or initial_state.shape[-2:] != forcing.shape[-2:]:
        raise ValueError("initial state and forcing batch/spatial shapes differ")
    if forcing.shape[1] != 1:
        raise ValueError("forcing must have exactly one channel")

    current = initial_state
    yield 0, current
    with torch.inference_mode():
        for step in range(1, n_steps + 1):
            current = model(torch.cat((current, forcing), dim=1))
            if not bool(torch.all(torch.isfinite(current)).item()):
                raise NonFiniteRollout(f"non-finite prediction at autoregressive step {step}")
            yield step, current


def rollout_autoregressive(
    model: Any,
    initial_state: Any,
    forcing: Any,
    *,
    n_steps: int,
) -> Any:
    """Small-run convenience API; production streaming uses the iterator."""

    states = [state.detach().cpu() for _, state in iter_autoregressive(
        model, initial_state, forcing, n_steps=n_steps
    )]
    return torch.stack(states, dim=1)


class _RolloutWriter:
    def write_prediction(self, member: int, day_index: int, values: np.ndarray) -> None:
        raise NotImplementedError

    def write_truth(self, member: int, day_index: int, values: np.ndarray) -> None:
        raise NotImplementedError

    def finalize(self, attrs: Mapping[str, Any]) -> None:
        raise NotImplementedError


class _ZarrRolloutWriter(_RolloutWriter):
    def __init__(
        self,
        destination: Path,
        *,
        n_members: int,
        days: np.ndarray,
        initial_indices: Sequence[int],
        field_shape: tuple[int, int, int],
        attrs: Mapping[str, Any],
    ) -> None:
        try:
            import zarr
        except (ImportError, OSError) as exc:  # pragma: no cover
            raise RuntimeError(
                "Zarr rollout output requires the locked zarr dependency; "
                "use .npz only for small tests"
            ) from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.group = zarr.open_group(str(destination), mode="w")
        shape = (n_members, len(days), *field_shape)
        chunks = (1, 1, *field_shape)
        self.prediction = self.group.create_dataset(
            "prediction",
            shape=shape,
            chunks=chunks,
            dtype="f4",
            fill_value=np.nan,
            overwrite=True,
        )
        self.truth = self.group.create_dataset(
            "truth",
            shape=shape,
            chunks=chunks,
            dtype="f4",
            fill_value=np.nan,
            overwrite=True,
        )
        self.initial_index = self.group.create_dataset(
            "initial_index",
            shape=(n_members,),
            chunks=(n_members,),
            dtype="i8",
            overwrite=True,
        )
        self.day = self.group.create_dataset(
            "day",
            shape=(len(days),),
            chunks=(len(days),),
            dtype="i4",
            overwrite=True,
        )
        self.initial_index[:] = np.asarray(initial_indices, dtype=np.int64)
        self.day[:] = days
        self.group.attrs.update(dict(attrs))

    def write_prediction(self, member: int, day_index: int, values: np.ndarray) -> None:
        self.prediction[member, day_index] = values

    def write_truth(self, member: int, day_index: int, values: np.ndarray) -> None:
        self.truth[member, day_index] = values

    def finalize(self, attrs: Mapping[str, Any]) -> None:
        self.group.attrs.update(dict(attrs))


class _NpzRolloutWriter(_RolloutWriter):
    """In-memory fallback intended for reduced tests, not 2,000-day production."""

    def __init__(
        self,
        destination: Path,
        *,
        n_members: int,
        days: np.ndarray,
        initial_indices: Sequence[int],
        field_shape: tuple[int, int, int],
        attrs: Mapping[str, Any],
    ) -> None:
        self.destination = destination
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        shape = (n_members, len(days), *field_shape)
        self.prediction = np.full(shape, np.nan, dtype=np.float32)
        self.truth = np.full(shape, np.nan, dtype=np.float32)
        self.initial_index = np.asarray(initial_indices, dtype=np.int64)
        self.day = days
        self.attrs = dict(attrs)

    def write_prediction(self, member: int, day_index: int, values: np.ndarray) -> None:
        self.prediction[member, day_index] = values

    def write_truth(self, member: int, day_index: int, values: np.ndarray) -> None:
        self.truth[member, day_index] = values

    def finalize(self, attrs: Mapping[str, Any]) -> None:
        self.attrs.update(dict(attrs))
        temporary = self.destination.with_suffix(self.destination.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary,
            prediction=self.prediction,
            truth=self.truth,
            initial_index=self.initial_index,
            day=self.day,
            attrs_json=np.asarray(json.dumps(self.attrs, sort_keys=True)),
        )
        temporary.replace(self.destination)


def _writer(
    destination: Path,
    *,
    n_members: int,
    days: np.ndarray,
    initial_indices: Sequence[int],
    field_shape: tuple[int, int, int],
    attrs: Mapping[str, Any],
) -> _RolloutWriter:
    cls = _NpzRolloutWriter if destination.suffix == ".npz" else _ZarrRolloutWriter
    return cls(
        destination,
        n_members=n_members,
        days=days,
        initial_indices=initial_indices,
        field_shape=field_shape,
        attrs=attrs,
    )


def _configuration_hash(configuration: Mapping[str, Any]) -> str:
    filtered = {
        key: value for key, value in configuration.items() if not str(key).startswith("_")
    }
    serialized = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_rollout(
    *,
    model: Any,
    checkpoint_path: str | Path,
    source: SequenceSource,
    normalizer: PointwiseNormalizer,
    experiment_id: int,
    initial_indices: Sequence[int],
    lag_days: int,
    horizon_days: int,
    output_path: str | Path,
    resolution: str | int = "full",
    state_channels: int = 10,
    wind_channel: int = 10,
    device: str | Any = "cpu",
    config_hash: str = "unknown",
    training_stage: str = "unknown",
) -> dict[str, Any]:
    """Stream a standardized prediction/truth rollout group to disk."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("FNO rollout requires the project's PyTorch environment")
    if not initial_indices:
        raise ValueError("initial_indices cannot be empty")
    if getattr(model, "out_channels", state_channels) != state_channels:
        raise ValueError("checkpoint output channels do not match state_channels")
    if getattr(model, "in_channels", state_channels + 1) != state_channels + 1:
        raise ValueError("checkpoint input channels must be state fields plus wind")
    stride = stride_for_resolution(resolution)
    days = rollout_days(horizon_days, lag_days)
    experiment_index = source.experiment_index(int(experiment_id))
    spatial_shape = (
        math.ceil(source.shape[-2] / stride),
        math.ceil(source.shape[-1] / stride),
    )
    field_shape = (state_channels, *spatial_shape)
    destination = Path(output_path)
    checkpoint = Path(checkpoint_path).resolve()
    attrs: dict[str, Any] = {
        "format_version": 1,
        "experiment_id": int(experiment_id),
        "lag_days": int(lag_days),
        "horizon_days_requested": int(horizon_days),
        "horizon_days_realized": int(days[-1]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "config_sha256": config_hash,
        "training_stage": training_stage,
        "resolution": "full" if stride == 1 else "low",
        "spatial_stride": stride,
        "truth_missing_value": "NaN",
        "complete": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    writer = _writer(
        destination,
        n_members=len(initial_indices),
        days=days,
        initial_indices=initial_indices,
        field_shape=field_shape,
        attrs=attrs,
    )
    device = torch.device(device)
    # A real dtype conversion would discard SpectralConv's imaginary weights.
    model.to(device=device)
    model.eval()
    stats = normalizer.for_stride(stride) if stride != 1 else normalizer
    missing_truth = 0

    for member, initial_index in enumerate(initial_indices):
        if not 0 <= int(initial_index) < source.shape[1]:
            raise IndexError(f"initial index {initial_index} is outside the source run")
        initial_raw = spatial_subsample(
            source.read(experiment_index, int(initial_index)), stride
        )
        state_raw = initial_raw[:state_channels]
        forcing_raw = initial_raw[wind_channel : wind_channel + 1]
        state_normalized = stats.normalize(
            state_raw, channels=slice(0, state_channels)
        )
        if wind_channel == state_channels:
            forcing_normalized = stats.normalize(
                forcing_raw, channels=slice(wind_channel, wind_channel + 1)
            )
        else:
            forcing_normalized = (
                forcing_raw - stats.mean[wind_channel : wind_channel + 1]
            ) / (
                stats.std[wind_channel : wind_channel + 1] + stats.epsilon
            )
        state_tensor = torch.from_numpy(np.ascontiguousarray(state_normalized))[None].to(
            device=device, dtype=torch.float32
        )
        forcing_tensor = torch.from_numpy(
            np.ascontiguousarray(forcing_normalized, dtype=np.float32)
        )[None].to(device=device, dtype=torch.float32)

        for day_index, (_, normalized_prediction) in enumerate(
            iter_autoregressive(
                model,
                state_tensor,
                forcing_tensor,
                n_steps=len(days) - 1,
            )
        ):
            prediction = stats.denormalize(
                normalized_prediction[0].detach().cpu().numpy(),
                channels=slice(0, state_channels),
            )
            if not np.all(np.isfinite(prediction)):
                raise NonFiniteRollout(
                    f"non-finite denormalized prediction for member {member}, day {days[day_index]}"
                )
            writer.write_prediction(member, day_index, prediction)
            truth_time = int(initial_index) + int(days[day_index])
            if truth_time < source.shape[1]:
                truth = spatial_subsample(
                    source.read(experiment_index, truth_time)[:state_channels], stride
                )
                writer.write_truth(member, day_index, truth)
            else:
                missing_truth += 1

    attrs["missing_truth_member_times"] = missing_truth
    attrs["complete_truth"] = missing_truth == 0
    attrs["complete"] = True
    writer.finalize(attrs)
    return {
        "output": str(destination.resolve()),
        "experiment_id": int(experiment_id),
        "lag_days": int(lag_days),
        "resolution": attrs["resolution"],
        "members": len(initial_indices),
        "day_indices": len(days),
        "last_day": int(days[-1]),
        "missing_truth_member_times": missing_truth,
    }


def rollout_from_config(
    configuration: str | Path | Mapping[str, Any],
    *,
    lag_days: int,
    experiment_id: int = 3,
    checkpoint_path: str | Path | None = None,
    stage: str = "finetune",
    initial_indices: Sequence[int] | None = None,
    horizon_days: int | None = None,
    resolution: str | int = "full",
    output_path: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Manifest-facing entry point used by ``repro fno rollout``."""

    configuration_values, base = _load_configuration(configuration)
    data_config, source, normalizer = _data_and_normalizer_from_config(
        configuration_values, base
    )
    fno_values = configuration_values.get("fno", {})
    evaluation_values = configuration_values.get("evaluation", {})
    rollout_values = (
        fno_values.get("rollout", {}) if isinstance(fno_values, Mapping) else {}
    )
    if not isinstance(rollout_values, Mapping):
        raise ValueError("fno.rollout configuration must be a mapping")
    stage = stage.lower()
    if stage not in {"pretrain", "finetune"}:
        raise ValueError("stage must be 'pretrain' or 'finetune'")
    if checkpoint_path is None:
        checkpoint_root = _path_from_config(
            configuration_values,
            base,
            names=(
                ("paths", "fno_checkpoints"),
                ("paths", "checkpoints"),
                ("fno", "output_dir"),
            ),
            required=False,
        ) or (base / "outputs" / "fno")
        checkpoint = checkpoint_root / f"lag_{lag_days:02d}d" / f"{stage}_best.pt"
    else:
        checkpoint = Path(checkpoint_path)
    resolved_device = device or (
        "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    )
    model, checkpoint_payload = load_model_checkpoint(checkpoint, device=resolved_device)
    checkpoint_lag = checkpoint_payload.get("lag_days")
    if checkpoint_lag is not None and int(checkpoint_lag) != int(lag_days):
        raise ValueError(
            f"checkpoint lag {checkpoint_lag} does not match requested lag {lag_days}"
        )
    selected_indices = tuple(
        int(item)
        for item in (
            initial_indices
            if initial_indices is not None
            else rollout_values.get(
                "initial_indices",
                evaluation_values.get("initial_condition_indices", RECOVERED_INITIAL_INDICES)
                if isinstance(evaluation_values, Mapping)
                else RECOVERED_INITIAL_INDICES,
            )
        )
    )
    horizon = int(
        horizon_days
        if horizon_days is not None
        else rollout_values.get(
            "horizon_days",
            evaluation_values.get("long_horizon_days", 2000)
            if isinstance(evaluation_values, Mapping)
            else 2000,
        )
    )
    stride = stride_for_resolution(resolution)
    resolution_name = "full" if stride == 1 else "low"
    if output_path is None:
        rollout_root = _path_from_config(
            configuration_values,
            base,
            names=(("paths", "fno_rollouts"), ("paths", "rollouts")),
            required=False,
        ) or (base / "outputs" / "rollouts")
        destination = (
            rollout_root
            / (
                f"experiment_{int(experiment_id)}_lag_{lag_days:02d}d_"
                f"{stage}_{resolution_name}.zarr"
            )
        )
    else:
        destination = Path(output_path)
    return write_rollout(
        model=model,
        checkpoint_path=checkpoint,
        source=source,
        normalizer=normalizer,
        experiment_id=experiment_id,
        initial_indices=selected_indices,
        lag_days=lag_days,
        horizon_days=horizon,
        output_path=destination,
        resolution=stride,
        state_channels=data_config.state_channels,
        wind_channel=data_config.wind_channel,
        device=resolved_device,
        config_hash=_configuration_hash(configuration_values),
        training_stage=stage,
    )


__all__ = [
    "NonFiniteRollout",
    "RECOVERED_INITIAL_INDICES",
    "iter_autoregressive",
    "rollout_autoregressive",
    "rollout_days",
    "rollout_from_config",
    "spatial_subsample",
    "stride_for_resolution",
    "write_rollout",
]
