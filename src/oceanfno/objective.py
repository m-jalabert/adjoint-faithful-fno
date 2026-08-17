"""The production six-step objective and its machine-readable contract.

One configuration, active in full from optimizer step one::

    L = L_state
      + 0.001  * L_increment
      + 0.50   * L_rollout
      + 1e-5   * L_spectral
      + 0.065  * L_boundary
      + 0.05   * L_pressure_gradient
      + 0.05   * L_continuity
      + 0.05   * L_barotropic_transport

The first five terms are defined here; the three physics terms live in
:mod:`oceanfno.pressure_gradient`, :mod:`oceanfno.continuity` and
:mod:`oceanfno.barotropic_transport`, and all are joined in
:func:`production_loss_terms`.

This is the v1 objective, unchanged. The spectral-normalization arm carries no
stability *penalty*: long-horizon growth is constrained by the operator's own
reparameterization in :mod:`oceanfno.spectral_norm`, so the objective is left
exactly as it was and the two effects cannot be confounded. Every term is dimensionless and every one gives
the four physical groups --- U, V, temperature, SSH --- equal status, so no
group's dimensional scale decides its own optimizer weight.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping
from dataclasses import asdict, dataclass

from .runtime import GROUP_SLICES, STATE_CHANNEL_COUNT, torch

ROLLOUT_STEPS = 6

INCREMENT_WEIGHT = 0.001

ROLLOUT_WEIGHT = 0.50

SPECTRAL_WEIGHT = 1.0e-5

BOUNDARY_WEIGHT = 0.065

PRESSURE_GRADIENT_WEIGHT = 0.05

CONTINUITY_WEIGHT = 0.05

BAROTROPIC_TRANSPORT_WEIGHT = 0.05

SPECTRAL_BINS = 12

WESTERN_BOUNDARY_WIDTH = 4


@dataclass(frozen=True)
class ProductionLossConfig:
    """The one immutable objective configuration this tree trains against."""

    rollout_steps: int = ROLLOUT_STEPS
    increment_weight: float = INCREMENT_WEIGHT
    rollout_weight: float = ROLLOUT_WEIGHT
    spectral_weight: float = SPECTRAL_WEIGHT
    boundary_weight: float = BOUNDARY_WEIGHT
    pressure_gradient_weight: float = PRESSURE_GRADIENT_WEIGHT
    continuity_weight: float = CONTINUITY_WEIGHT
    barotropic_transport_weight: float = BAROTROPIC_TRANSPORT_WEIGHT
    spectral_bins: int = SPECTRAL_BINS
    western_boundary_width: int = WESTERN_BOUNDARY_WIDTH

    def __post_init__(self) -> None:
        if self.to_dict() != _FROZEN_CONFIG:
            raise ValueError(
                "the production objective is frozen; every coefficient is fixed"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FROZEN_CONFIG = {
    "rollout_steps": ROLLOUT_STEPS,
    "increment_weight": INCREMENT_WEIGHT,
    "rollout_weight": ROLLOUT_WEIGHT,
    "spectral_weight": SPECTRAL_WEIGHT,
    "boundary_weight": BOUNDARY_WEIGHT,
    "pressure_gradient_weight": PRESSURE_GRADIENT_WEIGHT,
    "continuity_weight": CONTINUITY_WEIGHT,
    "barotropic_transport_weight": BAROTROPIC_TRANSPORT_WEIGHT,
    "spectral_bins": SPECTRAL_BINS,
    "western_boundary_width": WESTERN_BOUNDARY_WIDTH,
}


def production_loss_config() -> ProductionLossConfig:
    """Return the single frozen configuration."""

    return ProductionLossConfig()


def loss_contract(config: ProductionLossConfig) -> dict[str, Any]:
    """Machine-readable semantics of the objective, hashed into every checkpoint."""

    if not isinstance(config, ProductionLossConfig):
        raise ValueError("the loss contract describes only the production objective")
    return {
        "version": "production_v1_unchanged",
        "state": "equal_mean_U_V_temperature_SSH_masked_relative_l2_at_10_days",
        "increment": (
            "equal_mean_group_rmse_of_10_day_increment_error_scaled_by_"
            "training_only_per_channel_increment_rms"
        ),
        "rollout": (
            "equal_mean_U_V_temperature_SSH_masked_relative_l2_at_"
            "20_30_40_50_and_60_days"
        ),
        "spectral": (
            "equal_mean_group_12_bin_amplitude_relative_l2_of_10_day_increments_"
            "on_exact_wet_rectangle_after_hann_taper_over_all_six_calls"
        ),
        "boundary": (
            "equal_mean_U_V_temperature_SSH_relative_l2_at_10_20_30_40_50_60_days_"
            "on_first_4_wet_cells_east_of_western_wall"
        ),
        "pressure_gradient": (
            "relative_l2_of_the_horizontal_gradient_of_PHIHYD_reconstructed_from_"
            "THETA_and_ETAN_equal_weight_x_y_15_levels_and_6_rollout_calls"
        ),
        "continuity": (
            "truth_referenced_relative_squared_l2_of_the_free_surface_residual_"
            "d_eta_dt_plus_divergence_of_the_time_centered_depth_integrated_"
            "transport_chained_like_the_rollout_over_6_calls"
        ),
        "barotropic_transport": (
            "truth_referenced_relative_squared_l2_of_the_ten_day_tendency_of_the_"
            "depth_integrated_transport_equal_weight_zonal_and_meridional_over_6_calls"
        ),
        "total": (
            "state + increment_weight*increment + rollout_weight*rollout + "
            "spectral_weight*spectral + boundary_weight*boundary + "
            "pressure_gradient_weight*pressure_gradient + "
            "continuity_weight*continuity + "
            "barotropic_transport_weight*barotropic_transport"
        ),
        "activation": "every_term_active_from_optimizer_step_one_no_staged_fine_tuning",
        "physics_terms_add_no_output_channels": True,
        "groups": {name: 0.25 for name in GROUP_SLICES},
        "config": config.to_dict(),
    }


def loss_contract_sha256(config: ProductionLossConfig) -> str:
    encoded = json.dumps(
        loss_contract(config), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


LOSS_CONTRACT_SHA256 = loss_contract_sha256(production_loss_config())


def _validate_state_pair(prediction: Any, target: Any, mask: Any) -> None:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must share N,C,Y,X shape")
    if prediction.shape[1] != STATE_CHANNEL_COUNT:
        raise ValueError("the objective requires the declared 46 state channels")
    if mask.shape != (1, 1, *prediction.shape[-2:]):
        raise ValueError("masks must be shaped 1,1,Y,X")


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
    if increment_scale.shape not in (
        (STATE_CHANNEL_COUNT,),
        (1, STATE_CHANNEL_COUNT, 1, 1),
    ):
        raise ValueError("the increment scale must contain exactly 46 channel values")
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
        raise ValueError("the wet mask must be shaped 1,1,Y,X")
    wet = mask[0, 0] > 0
    rows, columns = torch.where(wet)
    if rows.numel() == 0:
        raise ValueError("the wet mask is empty")
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    if not bool(wet[y0:y1, x0:x1].all().item()):
        raise ValueError("the tapered spectrum requires an exact rectangular wet basin")
    if bool(wet[:y0].any() or wet[y1:].any() or wet[:, :x0].any() or wet[:, x1:].any()):
        raise ValueError("wet cells exist outside the inferred spectral rectangle")
    return y0, y1, x0, x1


def tapered_group_spectral_loss(
    prediction: Any,
    target: Any,
    wet_mask: Any,
    *,
    bins: int = SPECTRAL_BINS,
    epsilon: float = 1.0e-12,
) -> Any:
    """Compare group-balanced radial amplitudes after cropping and Hann tapering."""

    _validate_state_pair(prediction, target, wet_mask)
    if bins <= 1:
        raise ValueError("the spectral loss needs at least two bins")
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
            torch.floor(radius[valid] / radius.max() * bins)
            .to(torch.long)
            .clamp_max(bins - 1)
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


def base_loss_terms(
    predictions: Any,
    targets: Any,
    present: Any,
    wet: Any,
    western_boundary: Any,
    increment_scale: Any,
    config: ProductionLossConfig,
) -> dict[str, Any]:
    """The five state-space terms and their per-group audit components."""

    if predictions.shape != targets.shape or predictions.ndim != 5:
        raise ValueError("rollouts must share N,S,C,Y,X shape")
    if (
        predictions.shape[1] != config.rollout_steps
        or present.shape != predictions[:, 0].shape
    ):
        raise ValueError("the objective received an inconsistent rollout or present state")

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

    result = {
        "state": state_groups["mean"],
        "increment": increment_groups["mean"],
        "rollout": rollout,
        "spectral": spectral,
        "boundary": boundary,
    }
    result.update({f"state_{name}": state_groups[name] for name in GROUP_SLICES})
    result.update({f"increment_{name}": increment_groups[name] for name in GROUP_SLICES})
    return result


def production_loss_terms(
    predictions: Any,
    targets: Any,
    present: Any,
    wet: Any,
    western_boundary: Any,
    increment_scale: Any,
    config: ProductionLossConfig,
    auxiliary: Mapping[str, Any],
) -> dict[str, Any]:
    """Join the five state terms with the three physics terms into one total.

    ``auxiliary`` carries the already-evaluated ``pressure_gradient``,
    ``continuity`` and ``barotropic_transport`` tensors, passed in rather than
    computed here so this module stays free of the MITgcm grid metrics those
    terms need, while every weight stays in one place.
    """

    required = ("pressure_gradient", "continuity", "barotropic_transport")
    missing = [name for name in required if name not in auxiliary]
    if missing:
        raise ValueError(f"the objective is missing auxiliary terms: {missing}")
    result = base_loss_terms(
        predictions, targets, present, wet, western_boundary, increment_scale, config
    )
    result.update({name: auxiliary[name] for name in required})
    result["total"] = (
        result["state"]
        + config.increment_weight * result["increment"]
        + config.rollout_weight * result["rollout"]
        + config.spectral_weight * result["spectral"]
        + config.boundary_weight * result["boundary"]
        + config.pressure_gradient_weight * result["pressure_gradient"]
        + config.continuity_weight * result["continuity"]
        + config.barotropic_transport_weight * result["barotropic_transport"]
    )
    return result
