"""MITgcm-consistent differentiable pressure-gradient loss.

Reconstructs total PHIHYD (pressure anomaly divided by rhoConst) from the
predicted THETA and ETAN channels using the same finite-difference hydrostatic
integration already validated in archive/src/bire_repro/af_pressure.py.  The
loss compares the *horizontal gradient* of PHIHYD, because that is the quantity
that enters the horizontal momentum equations; a spatially uniform pressure
offset is dynamically irrelevant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .runtime import torch

GRAVITY_M_S2 = 9.81
THERMAL_EXPANSION_PER_C = 2.0e-4
EARTH_RADIUS_M = 6.370e6
#: Degrees per grid cell on the 0.25-degree turbulent grid.
DEGREES_PER_CELL = 0.25

#: Metres per grid cell in the meridional direction.  The zonal counterpart is
#: read per column from MITgcm's own DXF, but the meridional spacing is uniform,
#: so it is a constant --- and it is a *cell* spacing, not metres per degree.
#: On the 1-degree grid the two coincided, which is why the original constant
#: could stand in for both; at 0.25 degrees they differ by exactly four.
MERIDIONAL_CELL_M = EARTH_RADIUS_M * np.deg2rad(DEGREES_PER_CELL)

# Exact tutorial_baroclinic_gyre vertical grid and reference-temperature profile
# used by the already validated archived PHIHYD reconstruction.
DRF_M = (
    50.0, 60.0, 70.0, 80.0, 90.0,
    100.0, 110.0, 120.0, 130.0, 140.0,
    150.0, 160.0, 170.0, 180.0, 190.0,
)
T_REF_C = (
    30.0, 27.0, 24.0, 21.0, 18.0,
    15.0, 13.0, 11.0, 9.0, 7.0,
    6.0, 5.0, 4.0, 3.0, 2.0,
)
THETA_SLICE = slice(30, 45)
ETA_INDEX = 45


def _vertical_distances() -> tuple[np.ndarray, np.ndarray]:
    interfaces = -np.concatenate(([0.0], np.cumsum(np.asarray(DRF_M))))
    centers = 0.5 * (interfaces[:-1] + interfaces[1:])
    center_spacing = np.diff(centers)
    above = np.empty(len(DRF_M), dtype=np.float64)
    below = np.empty(len(DRF_M), dtype=np.float64)
    above[0] = interfaces[0] - centers[0]
    above[1:] = -0.5 * center_spacing
    below[:-1] = -0.5 * center_spacing
    below[-1] = centers[-1] - interfaces[-1]
    return above, below


_ABOVE_M, _BELOW_M = _vertical_distances()


@dataclass(frozen=True)
class PressureGradientContext:
    """Fixed physical coordinates required by the auxiliary loss."""

    pointwise_mean: np.ndarray  # (46, Y, X), physical units
    pointwise_scale: np.ndarray  # (46, Y, X), physical units
    zonal_spacing_m: np.ndarray  # (Y, X), MITgcm DXF
    wet_mask: np.ndarray  # (Y, X)

    def __post_init__(self) -> None:
        mean = np.asarray(self.pointwise_mean)
        scale = np.asarray(self.pointwise_scale)
        dx = np.asarray(self.zonal_spacing_m)
        wet = np.asarray(self.wet_mask)
        if mean.shape != scale.shape or mean.ndim != 3 or mean.shape[0] != 46:
            raise ValueError("pressure-gradient normalizers must have shape (46,Y,X)")
        if dx.shape != mean.shape[-2:] or wet.shape != dx.shape:
            raise ValueError("pressure-gradient grid fields must match the state grid")
        if np.any(dx[wet] <= 0.0) or not np.all(np.isfinite(dx[wet])):
            raise ValueError("MITgcm zonal spacing must be positive and finite on wet cells")


def phihyd_from_normalized_state(state: Any, context: PressureGradientContext) -> Any:
    """Return differentiable total PHIHYD for normalized 46-channel states.

    ``state`` may have any leading dimensions before ``(46,Y,X)``.  Output has
    the same leading dimensions followed by ``(15,Y,X)`` and units m2 s-2.
    """

    if state.shape[-3] != 46:
        raise ValueError("pressure-gradient loss requires the 46 state channels")
    device, dtype = state.device, state.dtype
    mean = torch.as_tensor(context.pointwise_mean, device=device, dtype=dtype)
    scale = torch.as_tensor(context.pointwise_scale, device=device, dtype=dtype)
    physical = state * scale + mean
    theta = physical[..., THETA_SLICE, :, :]
    eta = physical[..., ETA_INDEX, :, :]

    reference = torch.as_tensor(T_REF_C, device=device, dtype=dtype)
    reference = reference.reshape((1,) * (theta.ndim - 3) + (15, 1, 1))
    density_anomaly_over_reference = -THERMAL_EXPANSION_PER_C * (theta - reference)
    above = torch.as_tensor(_ABOVE_M, device=device, dtype=dtype)
    below = torch.as_tensor(_BELOW_M, device=device, dtype=dtype)

    interface = torch.zeros_like(eta)
    levels = []
    for level in range(15):
        layer = density_anomaly_over_reference[..., level, :, :]
        center = interface + above[level] * GRAVITY_M_S2 * layer
        levels.append(center + GRAVITY_M_S2 * eta)
        interface = center + below[level] * GRAVITY_M_S2 * layer
    return torch.stack(levels, dim=-3)


def pressure_gradient_components(phihyd: Any, context: PressureGradientContext) -> tuple[Any, Any, Any, Any]:
    """Return face gradients and pairwise wet masks in x and y.

    Differences are taken between neighboring tracer points, which is the
    natural location of the pressure force on the C-grid velocity faces.
    ``DXF`` supplies the physical zonal metric; meridional spacing is one degree
    on this regular spherical-polar tutorial grid.
    """

    device, dtype = phihyd.device, phihyd.dtype
    dx = torch.as_tensor(context.zonal_spacing_m, device=device, dtype=dtype)
    wet = torch.as_tensor(context.wet_mask, device=device, dtype=torch.bool)

    dx_face = 0.5 * (dx[:, 1:] + dx[:, :-1])
    gx = (phihyd[..., :, 1:] - phihyd[..., :, :-1]) / dx_face
    gy = (phihyd[..., 1:, :] - phihyd[..., :-1, :]) / float(MERIDIONAL_CELL_M)
    mask_x = wet[:, 1:] & wet[:, :-1]
    mask_y = wet[1:, :] & wet[:-1, :]
    return gx, gy, mask_x, mask_y


def pressure_gradient_relative_l2(
    predictions: Any,
    targets: Any,
    context: PressureGradientContext,
    *,
    epsilon: float = 1.0e-20,
) -> Any:
    """Six-step, level-resolved relative L2 error of the pressure force.

    The x and y components receive equal status.  The denominator is the truth
    gradient energy, making the term dimensionless and preventing the large
    dimensional scale of pressure from dictating its optimizer weight.
    """

    if predictions.shape != targets.shape or predictions.ndim != 5:
        raise ValueError("pressure-gradient rollouts must share N,S,C,Y,X shape")
    pred_phi = phihyd_from_normalized_state(predictions, context)
    true_phi = phihyd_from_normalized_state(targets, context)
    pred_x, pred_y, mask_x, mask_y = pressure_gradient_components(pred_phi, context)
    true_x, true_y, _, _ = pressure_gradient_components(true_phi, context)

    def relative(pred: Any, truth: Any, mask: Any) -> Any:
        # Broadcast horizontal face mask across batch, rollout and depth.
        masked_error = (pred - truth).square() * mask
        masked_truth = truth.square() * mask
        numerator = masked_error.sum(dim=(-2, -1))
        denominator = masked_truth.sum(dim=(-2, -1)).clamp_min(epsilon)
        ratio = numerator / denominator
        # sqrt has an infinite slope at zero, so a level the prediction matches
        # exactly backpropagates NaN instead of the finite subgradient 0.  Root
        # a substitute value there and discard it, keeping an exact zero loss.
        matched = ratio > 0.0
        safe = torch.where(matched, ratio, torch.ones_like(ratio))
        rooted = torch.where(matched, torch.sqrt(safe), torch.zeros_like(ratio))
        # Equal status to all 15 levels and all six rollout calls.
        return rooted.mean()

    return 0.5 * (relative(pred_x, true_x, mask_x) + relative(pred_y, true_y, mask_y))
