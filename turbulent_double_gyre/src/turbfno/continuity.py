"""MITgcm-consistent differentiable free-surface transport-continuity loss.

Depth-integrates the predicted velocity channels into a barotropic transport,
forms the free-surface continuity residual against the ten-day surface-height
tendency, and compares that residual to the same residual evaluated on truth.

The loss is *truth-referenced* rather than a penalty on a vanishing residual:
ten-day sampling and the centered U/V representation only approximate MITgcm's
native discrete continuity operator, so the exactly-balanced residual of the
model is not zero.  Asking the prediction to reproduce the truth residual keeps
the term honest about that discretization gap while still penalising a
transport field that is inconsistent with its own surface-height tendency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .pressure_gradient import DRF_M, MERIDIONAL_CELL_M
from .runtime import torch

SECONDS_PER_DAY = 86400.0
STEP_DAYS = 10.0
STEP_SECONDS = STEP_DAYS * SECONDS_PER_DAY

U_SLICE = slice(0, 15)
V_SLICE = slice(15, 30)
ETA_INDEX = 45


@dataclass(frozen=True)
class ContinuityContext:
    """Fixed physical coordinates required by the continuity loss."""

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
            raise ValueError("continuity normalizers must have shape (46,Y,X)")
        if dx.shape != mean.shape[-2:] or wet.shape != dx.shape:
            raise ValueError("continuity grid fields must match the state grid")
        if np.any(dx[wet] <= 0.0) or not np.all(np.isfinite(dx[wet])):
            raise ValueError("MITgcm zonal spacing must be positive and finite on wet cells")


def _physical(state: Any, context: ContinuityContext) -> Any:
    if state.shape[-3] != 46:
        raise ValueError("continuity loss requires the 46 state channels")
    device, dtype = state.device, state.dtype
    mean = torch.as_tensor(context.pointwise_mean, device=device, dtype=dtype)
    scale = torch.as_tensor(context.pointwise_scale, device=device, dtype=dtype)
    return state * scale + mean


def depth_integrated_transport(state: Any, context: ContinuityContext) -> tuple[Any, Any]:
    """Return the barotropic transport ``(Qx, Qy)`` in m2 s-1.

    ``state`` may carry any leading dimensions before ``(46,Y,X)``; each output
    has those leading dimensions followed by ``(Y,X)``.  The tutorial's exact 15
    ``delR`` thicknesses supply the vertical metric, so this is the same
    discrete depth integral the hydrostatic pressure reconstruction uses.
    """

    physical = _physical(state, context)
    thickness = torch.as_tensor(DRF_M, device=state.device, dtype=state.dtype)
    thickness = thickness.reshape((1,) * (state.ndim - 3) + (15, 1, 1))
    transport_x = (physical[..., U_SLICE, :, :] * thickness).sum(dim=-3)
    transport_y = (physical[..., V_SLICE, :, :] * thickness).sum(dim=-3)
    return transport_x, transport_y


def transport_divergence(transport_x: Any, transport_y: Any, context: ContinuityContext) -> Any:
    """Return the centered horizontal divergence at interior tracer points.

    ``DXF`` supplies the physical zonal metric; meridional spacing is one degree
    on this regular spherical-polar tutorial grid.  Centered differences put the
    divergence at the tracer point, which is where the surface height lives.
    """

    device, dtype = transport_x.device, transport_x.dtype
    dx = torch.as_tensor(context.zonal_spacing_m, device=device, dtype=dtype)
    zonal = (transport_x[..., 1:-1, 2:] - transport_x[..., 1:-1, :-2]) / (2.0 * dx[1:-1, 1:-1])
    meridional = (transport_y[..., 2:, 1:-1] - transport_y[..., :-2, 1:-1]) / (2.0 * float(MERIDIONAL_CELL_M))
    return zonal + meridional


def interior_wet_mask(context: ContinuityContext) -> Any:
    """Return the interior points whose five-point divergence stencil is wet."""

    wet = torch.as_tensor(context.wet_mask, dtype=torch.bool)
    return (
        wet[1:-1, 1:-1]
        & wet[1:-1, 2:]
        & wet[1:-1, :-2]
        & wet[2:, 1:-1]
        & wet[:-2, 1:-1]
    )


def continuity_residual(present: Any, future: Any, context: ContinuityContext) -> Any:
    """Return ``d(eta)/dt + div(Q)`` over one ten-day step, at interior points.

    The transport is time-centered between the two states, matching the
    trapezoidal average that a ten-day increment of the surface height implies.
    """

    physical_present = _physical(present, context)
    physical_future = _physical(future, context)
    tendency = (
        physical_future[..., ETA_INDEX, :, :] - physical_present[..., ETA_INDEX, :, :]
    ) / STEP_SECONDS

    present_x, present_y = depth_integrated_transport(present, context)
    future_x, future_y = depth_integrated_transport(future, context)
    divergence = transport_divergence(
        0.5 * (present_x + future_x), 0.5 * (present_y + future_y), context
    )
    return tendency[..., 1:-1, 1:-1] + divergence


def continuity_relative_l2(
    predictions: Any,
    targets: Any,
    present: Any,
    context: ContinuityContext,
    *,
    epsilon: float = 1.0e-20,
) -> Any:
    """Six-step truth-referenced relative-L2 error of the continuity residual.

    The residual is chained exactly like the rollout itself: the first step is
    referenced to the observed present state and every later step to the state
    the model produced before it, so a transport that drifts away from its own
    surface-height tendency is penalised at the step where it drifts.
    """

    if predictions.shape != targets.shape or predictions.ndim != 5:
        raise ValueError("continuity rollouts must share N,S,C,Y,X shape")
    if present.shape != predictions[:, 0].shape:
        raise ValueError("continuity loss received an inconsistent present state")

    mask = interior_wet_mask(context).to(device=predictions.device)
    values = []
    previous_prediction = present
    previous_target = present
    for step in range(predictions.shape[1]):
        predicted = continuity_residual(previous_prediction, predictions[:, step], context)
        truth = continuity_residual(previous_target, targets[:, step], context)
        masked_error = (predicted - truth).square() * mask
        masked_truth = truth.square() * mask
        # A ratio of squared norms, so the term is dimensionless and its
        # gradient stays finite where the prediction matches truth exactly.
        numerator = masked_error.sum(dim=(-2, -1))
        denominator = masked_truth.sum(dim=(-2, -1)) + epsilon
        values.append((numerator / denominator).mean())
        previous_prediction = predictions[:, step]
        previous_target = targets[:, step]
    # Equal status to all six rollout calls.
    return torch.stack(values).mean()
