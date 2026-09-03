"""Depth-integrated barotropic transport-tendency loss.

The streamfunction reconstruction audit established two things about the
long-horizon error.  Most of the visible day-2000 striping is produced by the
one-way cumulative integral used to reconstruct ``psi``, not by the operator:
integrating ``-U_BT`` along ``y`` stripes the error in columns, integrating
``+V_BT`` along ``x`` stripes it in rows, and a path-independent least-squares
reconstruction removes both.  But a real error remains underneath it --- the
depth-integrated zonal transport is 10.5 % wrong at day 2000, concentrated near
the western boundary, and it is flat until roughly day 200 before drifting.

This term supervises the quantity the audit isolated, and nothing derived from
it.  A streamfunction loss was considered and rejected: because
``d(psi)_j = -sum_{m<=j} d(T_x)_m dy``, one local transport error would enter
the objective at every downstream latitude, asking the operator to compensate
for the amplification mechanism of the diagnostic as well as for the physical
error.

It is the *tendency* that is supervised rather than the absolute transport.
The model reproduces the large mean gyre transport well; what drifts is the
ten-day increment, and a small systematic bias in that increment is exactly
what accumulates into a large error after two hundred autoregressive calls.

The ``ContinuityContext`` and the depth integral itself are imported rather
than redefined: both terms read the same normalizers and the same wet mask, and
the transport integral must stay defined in exactly one place.
"""
from __future__ import annotations

from typing import Any

from .continuity import ContinuityContext, depth_integrated_transport
from .runtime import torch


def barotropic_transport_relative_l2(
    predictions: Any,
    targets: Any,
    present: Any,
    context: ContinuityContext,
    *,
    epsilon: float = 1.0e-20,
) -> Any:
    """Six-step truth-referenced relative-L2 error of the transport tendency.

    For the first call the increment is measured from the observed present
    state; from the second onward the model side is predicted-to-predicted and
    the reference side truth-to-truth, so the term sees the same chain the
    rollout does.  The two components are given deliberately equal status: the
    zonal error is the larger one in the S0 evaluation, but weighting them by
    that would tune the objective to one realization of the failure.
    """

    if predictions.shape != targets.shape or predictions.ndim != 5:
        raise ValueError("barotropic-transport rollouts must share N,S,C,Y,X shape")
    if present.shape != predictions[:, 0].shape:
        raise ValueError("barotropic-transport loss received an inconsistent present state")

    wet = torch.as_tensor(context.wet_mask, dtype=torch.bool, device=predictions.device)
    steps = predictions.shape[1]
    # Each state's transport is integrated once and reused by both increments.
    present_transport = depth_integrated_transport(present, context)
    predicted = [depth_integrated_transport(predictions[:, s], context) for s in range(steps)]
    reference = [depth_integrated_transport(targets[:, s], context) for s in range(steps)]

    def relative(predicted_increment: Any, truth_increment: Any) -> Any:
        masked_error = (predicted_increment - truth_increment).square() * wet
        masked_truth = truth_increment.square() * wet
        # A ratio of squared norms: dimensionless, and finite-gradient where the
        # prediction matches truth exactly.
        numerator = masked_error.sum(dim=(-2, -1))
        denominator = masked_truth.sum(dim=(-2, -1)) + epsilon
        return numerator / denominator

    values = []
    for step in range(steps):
        previous_predicted = present_transport if step == 0 else predicted[step - 1]
        previous_reference = present_transport if step == 0 else reference[step - 1]
        components = [
            relative(
                predicted[step][axis] - previous_predicted[axis],
                reference[step][axis] - previous_reference[axis],
            )
            for axis in (0, 1)
        ]
        # Equal status to the zonal and meridional transport.
        values.append((0.5 * (components[0] + components[1])).mean())
    # Equal status to all six rollout calls.
    return torch.stack(values).mean()
