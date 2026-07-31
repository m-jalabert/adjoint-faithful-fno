"""Reduced ten-channel state used by the Model C Arm-R causal control.

The transform follows the recovered Bire-facing output set while retaining
the active one-degree MITgcm truth:

* surface and mid-depth (k=7) tracer-centred U, V, and temperature;
* MITgcm-consistent PHIHYD at k=0, 7, and 14;
* barotropic transport streamfunction.

The derived state is cached once from ``trajectories_v2``.  During an
autoregressive rollout the model receives and predicts only these ten dynamic
channels, so no omitted full-state channel is reconstructed or fed back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zarr
from numcodecs import Blosc

from .af_data import STATIC_FEATURES, STATE_CHANNELS
from .af_model_a import require_model_a_runtime
from .af_model_c import ModelCLossConfig
from .af_pressure import DRF_M, phihyd_from_theta_eta
from .af_tutorial_analysis import EARTH_RADIUS_M

try:  # Keep documentation imports usable without the optional ML stack.
    import torch
    from neuralop.models import FNO
    from torch import nn
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    FNO = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


VERSION = "model_c_reduced_bire_channels_v1"
QUALITY_NAME = "model_c_reduced_bire_channels_v1.quality.json"
FULL_CHANNEL_COUNT = len(STATE_CHANNELS)
REDUCED_CHANNELS = (
    "U_surface_k0",
    "U_mid_k7",
    "V_surface_k0",
    "V_mid_k7",
    "Theta_surface_k0",
    "Theta_mid_k7",
    "PHIHYD_surface_k0",
    "PHIHYD_mid_k7",
    "PHIHYD_bottom_k14",
    "barotropic_streamfunction",
)
REDUCED_UNITS = (
    "m s-1",
    "m s-1",
    "m s-1",
    "m s-1",
    "degC",
    "degC",
    "m2 s-2",
    "m2 s-2",
    "m2 s-2",
    "Sv",
)
REDUCED_CHANNEL_COUNT = len(REDUCED_CHANNELS)
REDUCED_INPUT_COUNT = REDUCED_CHANNEL_COUNT + len(STATIC_FEATURES)
REDUCED_GROUP_SLICES = {
    "u": slice(0, 2),
    "v": slice(2, 4),
    "temperature": slice(4, 6),
    "phihyd": slice(6, 9),
    "streamfunction": slice(9, 10),
}
REDUCED_AUDIT_TERMS = (
    "total",
    "state",
    "increment",
    "rollout",
    "spectral",
    "boundary",
    *(f"state_{name}" for name in REDUCED_GROUP_SLICES),
    *(f"increment_{name}" for name in REDUCED_GROUP_SLICES),
)


class ModelCReducedChannelError(RuntimeError):
    """Raised when the Arm-R reduced-state contract is violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def barotropic_streamfunction(
    full_states: np.ndarray,
    wet: np.ndarray,
) -> np.ndarray:
    """Return meridionally integrated full-depth U transport in Sv."""

    values = np.asarray(full_states)
    mask = np.asarray(wet, dtype=bool)
    if (
        values.ndim != 4
        or values.shape[1] != FULL_CHANNEL_COUNT
        or values.shape[-2:] != mask.shape
    ):
        raise ValueError("full states must have shape (batch,46,y,x)")
    dy_m = EARTH_RADIUS_M * np.deg2rad(1.0)
    transport = np.sum(
        values[:, :15].astype(np.float64)
        * DRF_M[None, :, None, None],
        axis=1,
    )
    result = np.cumsum(-transport * dy_m, axis=1) / 1.0e6
    result[:, ~mask] = 0.0
    return np.ascontiguousarray(result, dtype=np.float32)


def reduce_full_state(
    full_states: np.ndarray,
    wet: np.ndarray,
) -> np.ndarray:
    """Map one or more 46-channel physical states to the ten Arm-R channels."""

    values = np.asarray(full_states)
    squeeze = values.ndim == 3
    if squeeze:
        values = values[None]
    mask = np.asarray(wet, dtype=bool)
    if (
        values.ndim != 4
        or values.shape[1] != FULL_CHANNEL_COUNT
        or values.shape[-2:] != mask.shape
    ):
        raise ValueError("full states must have shape (...,46,y,x)")
    pressure = phihyd_from_theta_eta(
        values[:, 30:45],
        values[:, 45],
        mask,
    )
    streamfunction = barotropic_streamfunction(values, mask)
    result = np.stack(
        (
            values[:, 0],
            values[:, 7],
            values[:, 15],
            values[:, 22],
            values[:, 30],
            values[:, 37],
            pressure[:, 0],
            pressure[:, 7],
            pressure[:, 14],
            streamfunction,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    result[:, :, ~mask] = 0.0
    if not np.all(np.isfinite(result)):
        raise ModelCReducedChannelError("derived reduced state is non-finite")
    contiguous = np.ascontiguousarray(result, dtype=np.float32)
    return contiguous[0] if squeeze else contiguous


def reduced_fields(states: np.ndarray, wet: np.ndarray) -> dict[str, np.ndarray]:
    """Return the fields required by the training and Figure 3--8 gates."""

    values = np.asarray(states)
    mask = np.asarray(wet, dtype=bool)
    if (
        values.ndim != 4
        or values.shape[1] != REDUCED_CHANNEL_COUNT
        or values.shape[-2:] != mask.shape
    ):
        raise ValueError("reduced states must have shape (batch,10,y,x)")
    fields = {
        "surface_u": np.asarray(values[:, 0], dtype=np.float32),
        "surface_v": np.asarray(values[:, 2], dtype=np.float32),
        "sst": np.asarray(values[:, 4], dtype=np.float32),
        "phihyd_surface": np.asarray(values[:, 6], dtype=np.float32),
        "streamfunction": np.asarray(values[:, 9], dtype=np.float32),
    }
    fields["surface_speed"] = np.sqrt(
        np.square(fields["surface_u"], dtype=np.float64)
        + np.square(fields["surface_v"], dtype=np.float64)
    ).astype(np.float32)
    for value in fields.values():
        value[:, ~mask] = 0.0
    return fields


def _copy_array(source: Any, destination: Any, name: str) -> None:
    values = np.asarray(source[name][:])
    destination.create_dataset(
        name,
        data=values,
        chunks=getattr(source[name], "chunks", None),
        compressor=getattr(source[name], "compressor", None),
        overwrite=False,
    )


def prepare_reduced_dataset(
    source_path: str | Path,
    output_path: str | Path,
    quality_path: str | Path,
    *,
    block_days: int = 32,
) -> dict[str, Any]:
    """Create the non-overwriting ten-channel cache from trajectories-v2."""

    started = time.monotonic()
    source_resolved = Path(source_path).resolve()
    output = Path(output_path).resolve()
    quality = Path(quality_path).resolve()
    temporary = output.with_name(output.name + ".tmp")
    quality_temporary = quality.with_suffix(quality.suffix + ".tmp")
    if (
        output.exists()
        or temporary.exists()
        or quality.exists()
        or quality_temporary.exists()
    ):
        raise FileExistsError("refusing to overwrite a reduced-channel cache")
    if block_days <= 0 or not (source_resolved / ".zmetadata").is_file():
        raise ValueError("a consolidated source and positive block size are required")
    source = zarr.open_consolidated(str(source_resolved), mode="r")
    if tuple(source.attrs["state_channels"]) != STATE_CHANNELS:
        raise ModelCReducedChannelError("source state channels changed")
    full = source["state"]
    wet = np.asarray(source["wet_mask"][:], dtype=bool)
    if full.shape != (3, 7200, 46, 62, 62) or wet.shape != (62, 62):
        raise ModelCReducedChannelError("source dataset dimensions changed")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    logical_digest = hashlib.sha256()
    try:
        destination = zarr.open_group(str(temporary), mode="w")
        compressor = Blosc(
            cname="zstd",
            clevel=3,
            shuffle=Blosc.BITSHUFFLE,
        )
        reduced = destination.create_dataset(
            "state",
            shape=(3, 7200, REDUCED_CHANNEL_COUNT, 62, 62),
            chunks=(1, 8, REDUCED_CHANNEL_COUNT, 62, 62),
            dtype="f4",
            compressor=compressor,
            fill_value=0.0,
        )
        for experiment in range(full.shape[0]):
            for start in range(0, full.shape[1], block_days):
                stop = min(start + block_days, full.shape[1])
                block = reduce_full_state(
                    np.asarray(full[experiment, start:stop], dtype=np.float32),
                    wet,
                )
                logical_digest.update(block.tobytes(order="C"))
                reduced[experiment, start:stop] = block
        for name in (
            "static_features",
            "wet_mask",
            "longitude_deg",
            "latitude_deg",
            "pair_split",
            "snapshot_split",
        ):
            _copy_array(source, destination, name)
        destination.attrs.update(
            {
                "version": VERSION,
                "state_channels": list(REDUCED_CHANNELS),
                "state_units": list(REDUCED_UNITS),
                "source_dataset": str(source_resolved),
                "source_metadata_sha256": file_sha256(
                    source_resolved / ".zmetadata"
                ),
                "transform": (
                    "U/V/Theta k0,k7; MITgcm-consistent PHIHYD k0,k7,k14; "
                    "meridionally integrated full-depth U transport streamfunction"
                ),
                "autoregressive_semantics": (
                    "only_the_ten_reduced_channels_are_available_after_initialization"
                ),
            }
        )
        zarr.consolidate_metadata(str(temporary))
        metadata_sha = file_sha256(temporary / ".zmetadata")
        report = {
            "status": "valid",
            "version": VERSION,
            "source": str(source_resolved),
            "source_metadata_sha256": file_sha256(
                source_resolved / ".zmetadata"
            ),
            "output": str(output),
            "output_metadata_sha256": metadata_sha,
            "logical_state_sha256": logical_digest.hexdigest(),
            "shape": list(reduced.shape),
            "chunks": list(reduced.chunks),
            "dtype": str(reduced.dtype),
            "channels": list(REDUCED_CHANNELS),
            "channel_count": REDUCED_CHANNEL_COUNT,
            "wet_cells": int(wet.sum()),
            "elapsed_seconds": time.monotonic() - started,
        }
        report["report_content_sha256"] = json_sha256(report)
        quality_temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, output)
        os.replace(quality_temporary, quality)
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        quality_temporary.unlink(missing_ok=True)
        raise
    return report


@dataclass(frozen=True)
class ReducedChannelArchitecture:
    """Selected Model C architecture with only the dynamic channel count changed."""

    in_channels: int = REDUCED_INPUT_COUNT
    out_channels: int = REDUCED_CHANNEL_COUNT
    n_modes: tuple[int, int] = (24, 16)
    hidden_channels: int = 128
    n_layers: int = 4
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    channel_mlp_expansion: float = 4.0
    domain_padding: float = 0.1
    positional_embedding: str = "grid"
    use_channel_mlp: bool = True
    local_kernel_size: int = 3
    fno_block_precision: str = "full"
    factorization: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_modes",
            tuple(int(value) for value in self.n_modes),
        )
        expected = {
            "in_channels": REDUCED_INPUT_COUNT,
            "out_channels": REDUCED_CHANNEL_COUNT,
            "n_modes": (24, 16),
            "hidden_channels": 128,
            "n_layers": 4,
            "lifting_channel_ratio": 2,
            "projection_channel_ratio": 2,
            "channel_mlp_expansion": 4.0,
            "domain_padding": 0.1,
            "positional_embedding": "grid",
            "use_channel_mlp": True,
            "local_kernel_size": 3,
            "fno_block_precision": "full",
            "factorization": None,
        }
        observed = asdict(self)
        observed["n_modes"] = self.n_modes
        if observed != expected:
            raise ValueError("Arm R changes only the dynamic channel contract")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_modes"] = list(self.n_modes)
        return result


if nn is not None:

    class ReducedChannelFNO(nn.Module):
        """Direct-state FNO with the selected Model C trunk and local branch."""

        def __init__(self, architecture: ReducedChannelArchitecture) -> None:
            super().__init__()
            require_model_a_runtime()
            self.architecture = architecture
            self.fno = FNO(
                n_modes=architecture.n_modes,
                in_channels=architecture.in_channels,
                out_channels=architecture.out_channels,
                hidden_channels=architecture.hidden_channels,
                n_layers=architecture.n_layers,
                lifting_channel_ratio=architecture.lifting_channel_ratio,
                projection_channel_ratio=architecture.projection_channel_ratio,
                positional_embedding=architecture.positional_embedding,
                use_channel_mlp=architecture.use_channel_mlp,
                channel_mlp_expansion=architecture.channel_mlp_expansion,
                domain_padding=architecture.domain_padding,
                fno_block_precision=architecture.fno_block_precision,
                factorization=architecture.factorization,
            )
            self.local = nn.Conv2d(
                architecture.in_channels,
                architecture.out_channels,
                kernel_size=architecture.local_kernel_size,
                padding=architecture.local_kernel_size // 2,
            )

        def forward(self, features: Any) -> Any:
            if (
                features.ndim != 4
                or features.shape[1] != self.architecture.in_channels
            ):
                raise ValueError(
                    "Arm-R FNO expects "
                    f"N,{self.architecture.in_channels},Y,X features"
                )
            return self.fno(features) + self.local(features)

else:  # pragma: no cover - optional dependency unavailable
    ReducedChannelFNO = None  # type: ignore[assignment,misc]


def build_reduced_model(
    architecture: ReducedChannelArchitecture,
) -> Any:
    require_model_a_runtime()
    if ReducedChannelFNO is None:  # pragma: no cover
        raise RuntimeError("Arm R requires the project FNO environment")
    return ReducedChannelFNO(architecture)


def reduced_loss_contract(config: ModelCLossConfig) -> dict[str, Any]:
    """Machine-readable adaptation of loss-v1 to five reduced state groups."""

    return {
        "version": "arm_r_reduced_loss_v1",
        "base_model_c_loss_version": "v1",
        "base_config": config.to_dict(),
        "groups": {
            name: list(range(channels.start, channels.stop))
            for name, channels in REDUCED_GROUP_SLICES.items()
        },
        "state": "equal_mean_five_group_masked_relative_l2_at_10_days",
        "increment": (
            "equal_mean_five_group_rmse_of_10_day_increment_error_scaled_by_"
            "training_only_per_channel_increment_rms"
        ),
        "rollout": "same_state_form_at_20_and_30_days",
        "spectral": (
            "equal_mean_five_group_12_bin_amplitude_relative_l2_of_each_"
            "10_day_increment_after_wet_rectangle_hann_taper"
        ),
        "boundary": (
            "equal_mean_five_group_relative_l2_at_10_20_30_days_on_first_"
            "4_wet_cells_east_of_western_wall"
        ),
        "total": (
            "state + 0.001*increment + 0.15*rollout + "
            "0.00001*spectral + 0.065*boundary"
        ),
        "unavoidable_adaptation": (
            "46_channel_U_V_temperature_SSH_groups_are_replaced_by_the_five_"
            "semantic_groups_present_in_the_declared_ten_channel_state"
        ),
    }


def reduced_loss_contract_sha256(config: ModelCLossConfig) -> str:
    return json_sha256(reduced_loss_contract(config))


def _validate_pair(prediction: Any, target: Any, mask: Any) -> None:
    if (
        prediction.shape != target.shape
        or prediction.ndim != 4
        or prediction.shape[1] != REDUCED_CHANNEL_COUNT
    ):
        raise ValueError("Arm-R state tensors must share N,10,Y,X shape")
    if mask.shape != (1, 1, *prediction.shape[-2:]):
        raise ValueError("Arm-R masks must have shape 1,1,Y,X")


def _group_relative_l2(
    prediction: Any,
    target: Any,
    mask: Any,
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    _validate_pair(prediction, target, mask)
    terms = {}
    for name, channels in REDUCED_GROUP_SLICES.items():
        error = (prediction[:, channels] - target[:, channels]).square() * mask
        reference = target[:, channels].square() * mask
        numerator = error.sum(dim=(1, 2, 3))
        denominator = reference.sum(dim=(1, 2, 3)).clamp_min(epsilon)
        terms[name] = torch.sqrt(numerator / denominator).mean()
    terms["mean"] = torch.stack(list(terms.values())).mean()
    return terms


def _group_increment_nrmse(
    prediction: Any,
    target: Any,
    mask: Any,
    increment_scale: Any,
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    _validate_pair(prediction, target, mask)
    if increment_scale.shape not in (
        (REDUCED_CHANNEL_COUNT,),
        (1, REDUCED_CHANNEL_COUNT, 1, 1),
    ):
        raise ValueError("Arm-R increment scale must contain ten values")
    scale = increment_scale.reshape(
        1,
        REDUCED_CHANNEL_COUNT,
        1,
        1,
    ).clamp_min(epsilon)
    standardized = (prediction - target) / scale
    wet_count = mask.sum().clamp_min(1.0)
    terms = {}
    for name, channels in REDUCED_GROUP_SLICES.items():
        channel_count = channels.stop - channels.start
        squared = standardized[:, channels].square() * mask
        terms[name] = torch.sqrt(
            squared.sum(dim=(1, 2, 3)) / (wet_count * channel_count)
        ).mean()
    terms["mean"] = torch.stack(list(terms.values())).mean()
    return terms


def _wet_bounds(mask: Any) -> tuple[int, int, int, int]:
    wet = mask[0, 0] > 0
    rows, columns = torch.where(wet)
    if rows.numel() == 0:
        raise ValueError("Arm-R wet mask is empty")
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    if not bool(wet[y0:y1, x0:x1].all().item()):
        raise ValueError("Arm-R spectrum requires a rectangular wet basin")
    return y0, y1, x0, x1


def _spectral_loss(
    prediction: Any,
    target: Any,
    mask: Any,
    *,
    bins: int,
    epsilon: float = 1.0e-12,
) -> Any:
    _validate_pair(prediction, target, mask)
    y0, y1, x0, x1 = _wet_bounds(mask)

    def amplitudes(value: Any, channels: slice) -> Any:
        field = value[:, channels, y0:y1, x0:x1]
        field = field - field.mean(dim=(-2, -1), keepdim=True)
        window_y = torch.hann_window(
            field.shape[-2],
            periodic=False,
            dtype=field.dtype,
            device=field.device,
        )
        window_x = torch.hann_window(
            field.shape[-1],
            periodic=False,
            dtype=field.dtype,
            device=field.device,
        )
        field = field * window_y[:, None] * window_x[None, :]
        power = torch.fft.rfft2(field, norm="ortho").abs().square()
        fy = torch.fft.fftfreq(field.shape[-2], device=field.device)
        fx = torch.fft.rfftfreq(field.shape[-1], device=field.device)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        valid = radius > 0
        indices = (
            torch.floor(radius[valid] / radius.max() * bins)
            .to(torch.long)
            .clamp_max(bins - 1)
        )
        flattened = power[..., valid]
        spectrum = torch.zeros(
            (*power.shape[:2], bins),
            device=field.device,
            dtype=field.dtype,
        )
        spectrum.scatter_add_(
            2,
            indices[None, None].expand_as(flattened),
            flattened,
        )
        counts = torch.bincount(
            indices,
            minlength=bins,
        ).to(field.dtype).clamp_min(1.0)
        return torch.sqrt(spectrum / counts[None, None] + epsilon)

    losses = []
    for channels in REDUCED_GROUP_SLICES.values():
        predicted = amplitudes(prediction, channels)
        expected = amplitudes(target, channels)
        numerator = (predicted - expected).square().sum(dim=(1, 2))
        denominator = expected.square().sum(dim=(1, 2)).clamp_min(epsilon)
        losses.append(torch.sqrt(numerator / denominator).mean())
    return torch.stack(losses).mean()


def reduced_loss_terms(
    predictions: Any,
    targets: Any,
    present: Any,
    wet: Any,
    boundary: Any,
    increment_scale: Any,
    config: ModelCLossConfig,
) -> dict[str, Any]:
    """Evaluate the unchanged three-step loss form on the reduced state."""

    if (
        predictions.shape != targets.shape
        or predictions.ndim != 5
        or predictions.shape[1] != config.rollout_steps
        or present.shape != predictions[:, 0].shape
    ):
        raise ValueError("Arm-R rollout tensors are inconsistent")
    state = _group_relative_l2(predictions[:, 0], targets[:, 0], wet)
    increment = _group_increment_nrmse(
        predictions[:, 0] - present,
        targets[:, 0] - present,
        wet,
        increment_scale,
    )
    rollout = torch.stack(
        [
            _group_relative_l2(
                predictions[:, step],
                targets[:, step],
                wet,
            )["mean"]
            for step in range(1, config.rollout_steps)
        ]
    ).mean()
    spectral_values = []
    previous_prediction = present
    previous_target = present
    for step in range(config.rollout_steps):
        spectral_values.append(
            _spectral_loss(
                predictions[:, step] - previous_prediction,
                targets[:, step] - previous_target,
                wet,
                bins=config.spectral_bins,
            )
        )
        previous_prediction = predictions[:, step]
        previous_target = targets[:, step]
    spectral = torch.stack(spectral_values).mean()
    boundary_value = torch.stack(
        [
            _group_relative_l2(
                predictions[:, step],
                targets[:, step],
                boundary,
            )["mean"]
            for step in range(config.rollout_steps)
        ]
    ).mean()
    total = (
        state["mean"]
        + config.increment_weight * increment["mean"]
        + config.rollout_weight * rollout
        + config.spectral_weight * spectral
        + config.boundary_weight * boundary_value
    )
    result = {
        "total": total,
        "state": state["mean"],
        "increment": increment["mean"],
        "rollout": rollout,
        "spectral": spectral,
        "boundary": boundary_value,
    }
    result.update(
        {
            f"state_{name}": state[name]
            for name in REDUCED_GROUP_SLICES
        }
    )
    result.update(
        {
            f"increment_{name}": increment[name]
            for name in REDUCED_GROUP_SLICES
        }
    )
    return result


def direct_unroll(
    model: Any,
    features: Any,
    wet: Any,
    steps: int,
) -> Any:
    if steps <= 0:
        raise ValueError("Arm-R rollout needs at least one step")
    current = features[:, :REDUCED_CHANNEL_COUNT]
    static = features[:, REDUCED_CHANNEL_COUNT:]
    predictions = []
    for _ in range(steps):
        current = model(torch.cat((current, static), dim=1)) * wet
        predictions.append(current)
    return torch.stack(predictions, dim=1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--block-days", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = prepare_reduced_dataset(
        args.source,
        args.output,
        args.quality_report,
        block_days=args.block_days,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
