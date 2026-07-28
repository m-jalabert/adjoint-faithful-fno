"""Forward-optimized FNO components for Model C.

Model C is not allowed to use response, inference, intermediate-wind, or
adjoint data.  It keeps the modern residual-FNO family used by Models A and B,
but corrects two observed Model B weaknesses before any bounded architecture
search: the 15/15/15/1 channel multiplicity is replaced by equal physical-group
weighting, and spectra are compared on tapered ten-day increments instead of
masked full states.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .af_data import STATIC_FEATURES, STATE_CHANNELS
from .af_model_a import ModelAResidualFNO, require_model_a_runtime

try:  # Keep documentation-only imports usable without the optional ML stack.
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


STATE_CHANNEL_COUNT = len(STATE_CHANNELS)
MODEL_C_INPUT_CHANNELS = STATE_CHANNEL_COUNT + len(STATIC_FEATURES)
GROUP_SLICES = {
    "u": slice(0, 15),
    "v": slice(15, 30),
    "temperature": slice(30, 45),
    "ssh": slice(45, 46),
}
MODE_CANDIDATES = ((12, 12), (16, 16), (16, 24), (24, 16), (24, 24))
MODEL_C_LOSS_CALIBRATION_SHA256 = (
    "3406c9104dd63702f0c10c2d968ae0eb3f1dfc8d3b84db967e069b3df592e469"
)
MODEL_C_LOSS_V1_CONTRACT_SHA256 = (
    "19000a1426ea928db7799c82a73ce071a874911eb7e1df50bd276582ec30b5f9"
)
MODEL_C_LATE_AUDIT_SHA256 = {
    "width32": "2c9c2dc9d31ce171c3c5e51eea7a92adb8ff03f489bfc60f205019edac8ec098",
    "width64": "15a191fe808ff22b0db78e73491584fad3618b172123bab9eae145d5a06b433e",
}


@dataclass(frozen=True)
class ModelCArchitecture:
    """Bounded dense-FNO search space declared before validation is opened."""

    in_channels: int = MODEL_C_INPUT_CHANNELS
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
        if self.in_channels != MODEL_C_INPUT_CHANNELS or self.out_channels != STATE_CHANNEL_COUNT:
            raise ValueError("Model C channel contract is 46 dynamic + 5 static -> 46 residuals")
        if self.n_modes not in MODE_CANDIDATES:
            raise ValueError(f"Model C modes must be one of {MODE_CANDIDATES}")
        if self.hidden_channels not in (32, 64):
            raise ValueError("Model C hidden channels are restricted to 32 or 64")
        if self.n_layers not in (4, 6):
            raise ValueError("Model C FNO layers are restricted to 4 or 6")
        if self.domain_padding not in (0.10, 0.20):
            raise ValueError("Model C domain padding is restricted to 10% or 20%")
        if self.positional_embedding != "grid" or not self.use_channel_mlp:
            raise ValueError("Model C requires grid embedding and the Channel MLP")
        if self.local_kernel_size != 3:
            raise ValueError("Model C retains the 3-by-3 local correction")
        if self.precision != "full" or self.factorization is not None:
            raise ValueError("Model C candidates are dense float32 FNOs")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_modes"] = list(self.n_modes)
        return result


@dataclass(frozen=True)
class ModelCLossConfig:
    """Balanced objective frozen from the training-only C1a gradient audit."""

    rollout_steps: int = 3
    increment_weight: float = 0.001
    rollout_weight: float = 0.15
    spectral_weight: float = 0.00001
    boundary_weight: float = 0.065
    spectral_bins: int = 12
    western_boundary_width: int = 4

    def __post_init__(self) -> None:
        weights = (
            self.increment_weight,
            self.rollout_weight,
            self.spectral_weight,
            self.boundary_weight,
        )
        if self.rollout_steps != 3:
            raise ValueError("Model C uses the predeclared three-step rollout")
        if any(value <= 0 for value in weights):
            raise ValueError("every frozen Model C loss term must have positive weight")
        if self.spectral_bins <= 1 or self.western_boundary_width <= 0:
            raise ValueError("Model C needs spectral bins and a positive boundary width")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelCLossV2Config(ModelCLossConfig):
    """Training-only priority revision after the two late-checkpoint audits."""

    increment_weight: float = 0.0025

    def __post_init__(self) -> None:
        super().__post_init__()
        expected = {
            "rollout_steps": 3,
            "increment_weight": 0.0025,
            "rollout_weight": 0.15,
            "spectral_weight": 0.00001,
            "boundary_weight": 0.065,
            "spectral_bins": 12,
            "western_boundary_width": 4,
        }
        if self.to_dict() != expected:
            raise ValueError("Model C loss v2 changes only increment weight to 0.0025")


def model_c_loss_config(version: str = "v1") -> ModelCLossConfig:
    """Return one explicit, immutable Model C loss version."""

    if version == "v1":
        return ModelCLossConfig()
    if version == "v2":
        return ModelCLossV2Config()
    raise ValueError("Model C loss version must be 'v1' or 'v2'")


def model_c_architecture(
    *,
    n_modes: tuple[int, int] = (16, 16),
    hidden_channels: int = 32,
    n_layers: int = 4,
    domain_padding: float = 0.10,
) -> ModelCArchitecture:
    """Construct one candidate inside the predeclared search space."""

    return ModelCArchitecture(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        domain_padding=domain_padding,
    )


def build_model_c(architecture: ModelCArchitecture = ModelCArchitecture()) -> Any:
    """Build the dense residual FNO while retaining the A/B local branch."""

    require_model_a_runtime()
    if ModelAResidualFNO is None:  # pragma: no cover - guarded by runtime check
        raise RuntimeError("Model C requires the project FNO environment")
    return ModelAResidualFNO(architecture)  # type: ignore[arg-type]


def loss_contract(config: ModelCLossConfig) -> dict[str, Any]:
    """Machine-readable semantics saved with every Model C experiment."""

    result = {
        "state": "equal_mean_U_V_temperature_SSH_masked_relative_l2_at_10_days",
        "increment": (
            "equal_mean_group_rmse_of_10_day_increment_error_scaled_by_"
            "training_only_per_channel_increment_rms"
        ),
        "rollout": (
            "equal_mean_U_V_temperature_SSH_masked_relative_l2_at_20_and_30_days"
        ),
        "spectral": (
            "equal_mean_group_12_bin_amplitude_relative_l2_of_10_day_increments_"
            "on_exact_wet_rectangle_after_hann_taper"
        ),
        "boundary": (
            "equal_mean_U_V_temperature_SSH_relative_l2_at_10_20_30_days_"
            "on_first_4_wet_cells_east_of_western_wall"
        ),
        "total": (
            "state + increment_weight*increment + rollout_weight*rollout + "
            "spectral_weight*spectral + boundary_weight*boundary"
        ),
        "config": config.to_dict(),
    }
    if isinstance(config, ModelCLossV2Config):
        result.update(
            {
                "version": "v2",
                "status": "frozen_from_training_only_late_checkpoint_audit_2026-07-26",
                "calibration_report_sha256": MODEL_C_LOSS_CALIBRATION_SHA256,
                "late_checkpoint_audit_sha256": MODEL_C_LATE_AUDIT_SHA256,
                "supersedes_loss_contract_sha256": MODEL_C_LOSS_V1_CONTRACT_SHA256,
                "weight_basis": (
                    "increment_weight_raised_2.5x_to_target_late_weighted_increment_"
                    "gradient_norm_near_1.0_relative_to_state_on_width32_and_width64"
                ),
                "scope": (
                    "training_only_bounded_memorization_test_before_validation"
                ),
            }
        )
    else:
        result.update(
            {
                "status": "frozen_from_training_only_calibration_2026-07-23",
                "calibration_report_sha256": MODEL_C_LOSS_CALIBRATION_SHA256,
                "weight_basis": (
                    "rounded_unclipped_coefficients_targeting_gradient_norms_relative_to_"
                    "state_increment_0.5_rollout_0.5_spectral_0.25_boundary_0.25"
                ),
            }
        )
    return result


def loss_contract_sha256(config: ModelCLossConfig) -> str:
    encoded = json.dumps(loss_contract(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def loss_config_from_contract(contract: dict[str, Any]) -> ModelCLossConfig:
    """Reconstruct the explicit loss version stored in a checkpoint/report."""

    config = contract.get("config")
    if not isinstance(config, dict):
        raise ValueError("Model C loss contract has no configuration")
    version = contract.get("version", "v1")
    if version == "v2":
        return ModelCLossV2Config(**config)
    if version == "v1":
        return ModelCLossConfig(**config)
    raise ValueError(f"unsupported Model C loss contract version: {version!r}")


def _validate_state_pair(prediction: Any, target: Any, mask: Any) -> None:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Model C prediction and target must share N,C,Y,X shape")
    if prediction.shape[1] != STATE_CHANNEL_COUNT:
        raise ValueError("Model C loss requires the declared 46 state channels")
    if mask.shape != (1, 1, *prediction.shape[-2:]):
        raise ValueError("Model C masks must be shaped 1,1,Y,X")


def group_relative_l2_terms(
    prediction: Any,
    target: Any,
    mask: Any,
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """Return equal-status physical-group state errors."""

    _validate_state_pair(prediction, target, mask)
    terms: dict[str, Any] = {}
    for name, channels in GROUP_SLICES.items():
        error = (prediction[:, channels] - target[:, channels]).square() * mask
        reference = target[:, channels].square() * mask
        numerator = error.sum(dim=(1, 2, 3))
        denominator = reference.sum(dim=(1, 2, 3)).clamp_min(epsilon)
        terms[name] = torch.sqrt(numerator / denominator).mean()
    terms["mean"] = torch.stack(list(terms.values())).mean()
    return terms


def group_increment_nrmse_terms(
    prediction_increment: Any,
    target_increment: Any,
    mask: Any,
    increment_scale: Any,
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """Return per-group RMSE after training-only per-channel increment scaling."""

    _validate_state_pair(prediction_increment, target_increment, mask)
    if increment_scale.shape not in ((STATE_CHANNEL_COUNT,), (1, STATE_CHANNEL_COUNT, 1, 1)):
        raise ValueError("Model C increment scale must contain exactly 46 channel values")
    scale = increment_scale.reshape(1, STATE_CHANNEL_COUNT, 1, 1).clamp_min(epsilon)
    standardized = (prediction_increment - target_increment) / scale
    terms: dict[str, Any] = {}
    wet_count = mask.sum().clamp_min(1.0)
    for name, channels in GROUP_SLICES.items():
        channel_count = channels.stop - channels.start
        squared = standardized[:, channels].square() * mask
        terms[name] = torch.sqrt(
            squared.sum(dim=(1, 2, 3)) / (wet_count * channel_count)
        ).mean()
    terms["mean"] = torch.stack(list(terms.values())).mean()
    return terms


def wet_rectangle_bounds(mask: Any) -> tuple[int, int, int, int]:
    """Return the exact rectangular wet basin or reject unsupported interior holes."""

    if mask.ndim != 4 or mask.shape[:2] != (1, 1):
        raise ValueError("Model C wet mask must be shaped 1,1,Y,X")
    wet = mask[0, 0] > 0
    rows, columns = torch.where(wet)
    if rows.numel() == 0:
        raise ValueError("Model C wet mask is empty")
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    if not bool(wet[y0:y1, x0:x1].all().item()):
        raise ValueError("tapered Model C spectrum requires an exact rectangular wet basin")
    if bool(wet[:y0].any() or wet[y1:].any() or wet[:, :x0].any() or wet[:, x1:].any()):
        raise ValueError("wet cells exist outside the inferred Model C spectral rectangle")
    return y0, y1, x0, x1


def tapered_group_spectral_loss(
    prediction: Any,
    target: Any,
    wet_mask: Any,
    *,
    bins: int = 12,
    epsilon: float = 1.0e-12,
) -> Any:
    """Compare group-balanced radial amplitudes after cropping and Hann tapering."""

    _validate_state_pair(prediction, target, wet_mask)
    if bins <= 1:
        raise ValueError("Model C spectral loss needs at least two bins")
    y0, y1, x0, x1 = wet_rectangle_bounds(wet_mask)

    def amplitudes(value: Any, channels: slice) -> Any:
        field = value[:, channels, y0:y1, x0:x1]
        field = field - field.mean(dim=(-2, -1), keepdim=True)
        window_y = torch.hann_window(
            field.shape[-2], periodic=False, dtype=field.dtype, device=field.device
        )
        window_x = torch.hann_window(
            field.shape[-1], periodic=False, dtype=field.dtype, device=field.device
        )
        field = field * window_y[:, None] * window_x[None, :]
        power = torch.fft.rfft2(field, norm="ortho").abs().square()
        fy = torch.fft.fftfreq(field.shape[-2], device=field.device)
        fx = torch.fft.rfftfreq(field.shape[-1], device=field.device)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        valid = radius > 0
        indices = (
            torch.floor(radius[valid] / radius.max() * bins).to(torch.long).clamp_max(bins - 1)
        )
        flattened = power[..., valid]
        spectrum = torch.zeros(
            (*power.shape[:2], bins), device=field.device, dtype=field.dtype
        )
        spectrum.scatter_add_(2, indices[None, None].expand_as(flattened), flattened)
        counts = torch.bincount(indices, minlength=bins).to(field.dtype).clamp_min(1.0)
        return torch.sqrt(spectrum / counts[None, None] + epsilon)

    losses = []
    for channels in GROUP_SLICES.values():
        predicted = amplitudes(prediction, channels)
        expected = amplitudes(target, channels)
        numerator = (predicted - expected).square().sum(dim=(1, 2))
        denominator = expected.square().sum(dim=(1, 2)).clamp_min(epsilon)
        losses.append(torch.sqrt(numerator / denominator).mean())
    return torch.stack(losses).mean()


def model_c_loss_terms(
    predictions: Any,
    targets: Any,
    present: Any,
    wet: Any,
    western_boundary: Any,
    increment_scale: Any,
    config: ModelCLossConfig,
) -> dict[str, Any]:
    """Compute the complete balanced Model C objective and audit components."""

    if predictions.shape != targets.shape or predictions.ndim != 5:
        raise ValueError("Model C rollouts must share N,S,C,Y,X shape")
    if predictions.shape[1] != config.rollout_steps or present.shape != predictions[:, 0].shape:
        raise ValueError("Model C loss received an inconsistent rollout or present state")

    state_groups = group_relative_l2_terms(predictions[:, 0], targets[:, 0], wet)
    increment_groups = group_increment_nrmse_terms(
        predictions[:, 0] - present,
        targets[:, 0] - present,
        wet,
        increment_scale,
    )
    rollout = torch.stack(
        [
            group_relative_l2_terms(predictions[:, step], targets[:, step], wet)["mean"]
            for step in range(1, config.rollout_steps)
        ]
    ).mean()

    spectral_values = []
    previous_prediction = present
    previous_target = present
    for step in range(config.rollout_steps):
        spectral_values.append(
            tapered_group_spectral_loss(
                predictions[:, step] - previous_prediction,
                targets[:, step] - previous_target,
                wet,
                bins=config.spectral_bins,
            )
        )
        previous_prediction = predictions[:, step]
        previous_target = targets[:, step]
    spectral = torch.stack(spectral_values).mean()
    boundary = torch.stack(
        [
            group_relative_l2_terms(
                predictions[:, step], targets[:, step], western_boundary
            )["mean"]
            for step in range(config.rollout_steps)
        ]
    ).mean()
    total = (
        state_groups["mean"]
        + config.increment_weight * increment_groups["mean"]
        + config.rollout_weight * rollout
        + config.spectral_weight * spectral
        + config.boundary_weight * boundary
    )
    result = {
        "total": total,
        "state": state_groups["mean"],
        "increment": increment_groups["mean"],
        "rollout": rollout,
        "spectral": spectral,
        "boundary": boundary,
    }
    result.update({f"state_{name}": state_groups[name] for name in GROUP_SLICES})
    result.update({f"increment_{name}": increment_groups[name] for name in GROUP_SLICES})
    return result
