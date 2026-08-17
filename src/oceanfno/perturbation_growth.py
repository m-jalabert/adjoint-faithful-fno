"""Perturbation-growth diagnostics for the autoregressive operator.

Two measurements, both **diagnostic only in this arm** --- neither contributes a
gradient. The spectral-normalization arm constrains the operator directly, so
these exist to say whether that worked.

``g`` --- *dominant-direction single-step amplification*::

    g = || F(x_t + eps r, S) - F(x_t, S) || / || eps r ||   ~   || J_F r || / || r ||

with ``r`` a persistent unit direction advanced by ``r <- J_F r / ||J_F r||``,
one power-iteration step per measurement. Because the iteration is on ``J``
itself it converges to ``J``'s dominant eigendirection, so ``g`` estimates the
spectral radius ``|lambda_1(J_F)|``, **not** ``sigma_max(J)``, which would need
``J^H J``. A fresh random direction instead measures the rms amplification over
all directions --- 0.512-0.517 against a dominant 1.037-1.046 on the v1
checkpoint --- which is why the direction is persistent.

``lambda_hat`` --- the composite growth rate over a 200-call rollout, fitted on
calls 51-200 of ``log ||x_twin - x||`` from a 1e-2-relative perturbation. This is
the quantity that decides whether a 2,000-day inference stays bounded, and it is
a hard condition on checkpoint selection.

Measured history of ``lambda_hat`` at the selected checkpoint: v1 1.0168 (no
constraint), v2 1.01126 (squared contraction hinge at 0.01), v3 1.01361 (log
hinge at 0.1, best checkpoint 0.99922 at step 1,920).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .runtime import torch

#: Perturbation size as a fraction of the state norm. Large enough to sit well
#: clear of the float32 quantisation floor in normalized units, small enough
#: that the response stays linear: measured g varies by under 1e-3 across seeds.
EPSILON_RELATIVE = 1.0e-3

#: Perturbation size for the *diagnostic* rollout, which is a finite-amplitude
#: twin experiment rather than a derivative estimate.
DIAGNOSTIC_EPSILON_RELATIVE = 1.0e-2

#: Calls of the diagnostic rollout, and the window the growth rate is fitted on.
#: The first 50 calls are discarded: the random initial direction has not yet
#: aligned with the dominant mode, so including them biases the fit low.
DIAGNOSTIC_CALLS = 200

DIAGNOSTIC_FIT_FROM = 50

#: A checkpoint is eligible for selection only if its measured growth rate is at
#: or below this. One means "a perturbation neither grows nor is damped", which
#: is what the true S0 system does --- twin perturbations there do not grow.
GROWTH_RATE_CEILING = 1.0

def initial_direction(
    shape: tuple[int, ...],
    wet: Any,
    device: Any,
    *,
    seed: int,
) -> Any:
    """A unit-norm, wet-masked random direction to start power iteration from."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    value = torch.randn(shape, generator=generator).to(device=device, dtype=torch.float32)
    value = value * wet
    norm = value.norm().clamp_min(1.0e-12)
    return (value / norm).detach()


def single_call_amplification(
    model: Any,
    state: Any,
    static: Any,
    wet: Any,
    baseline: Any,
    direction: Any,
    *,
    epsilon_relative: float = EPSILON_RELATIVE,
) -> tuple[Any, Any]:
    """Return ``(mean g, advanced direction)`` for one present state.

    Diagnostic only: the whole computation runs under ``no_grad``, so it adds one
    forward pass and no backward. ``baseline`` is ``F(x_t, S) * wet``, which the
    rollout has already computed.
    """

    if state.ndim != 4 or baseline.shape != state.shape:
        raise ValueError("the contraction probe expects N,C,Y,X state and baseline")
    if direction.shape[0] != 1 or direction.shape[1:] != state.shape[1:]:
        raise ValueError("the power-iteration direction must be shaped 1,C,Y,X")

    batch = state.shape[0]
    with torch.no_grad():
        # One epsilon per sample, proportional to that sample's own norm.
        magnitude = state.reshape(batch, -1).norm(dim=1).clamp_min(1.0e-12)
        epsilon = (float(epsilon_relative) * magnitude).reshape(batch, 1, 1, 1)
        perturbed = state + epsilon * direction
        response = model(torch.cat((perturbed, static), dim=1)) * wet
        # (F(x + eps r) - F(x)) / eps, a finite-difference J_F r. ||r|| is one,
        # so the norm of this *is* the amplification ratio.
        difference = (response - baseline.detach()) / epsilon
        growth = difference.reshape(batch, -1).norm(dim=1)
        # Advance the power iteration on the batch-mean response direction.
        advanced = difference.mean(dim=0, keepdim=True) * wet
        norm = advanced.norm()
        advanced = direction if float(norm) <= 0.0 else advanced / norm
    return growth.mean(), advanced


def measure_growth_rate(
    model: Any,
    state: Any,
    static: Any,
    wet: Any,
    *,
    calls: int = DIAGNOSTIC_CALLS,
    fit_from: int = DIAGNOSTIC_FIT_FROM,
    epsilon_relative: float = DIAGNOSTIC_EPSILON_RELATIVE,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure the per-call growth rate of a finite twin perturbation.

    This is the diagnostic the v1 post-mortem used, reproduced exactly: roll a
    state and a slightly perturbed copy ``calls`` times and fit the exponential
    growth of their separation. Unlike the training term it is a *composite*
    measurement over the whole rollout, which is the quantity that actually
    decides whether a 2,000-day inference stays bounded.
    """

    if calls <= fit_from + 1:
        raise ValueError("the growth-rate fit needs calls beyond the discarded window")
    direction = initial_direction(state.shape, wet, state.device, seed=seed)
    perturbation = float(epsilon_relative) * float(state.norm()) * direction
    current, twin = state, state + perturbation
    separations = []
    with torch.no_grad():
        for _ in range(int(calls)):
            current = model(torch.cat((current, static), dim=1)) * wet
            twin = model(torch.cat((twin, static), dim=1)) * wet
            separations.append(float((twin - current).norm()))
    values = np.asarray(separations, dtype=np.float64)
    index = np.arange(1, values.size + 1, dtype=np.float64)
    window = index > int(fit_from)
    if not np.all(np.isfinite(values)) or np.any(values[window] <= 0.0):
        return {
            "growth_rate_per_call": None,
            "finite": False,
            "calls": int(calls),
            "fit_from_call": int(fit_from),
        }
    slope = float(np.polyfit(index[window], np.log(values[window]), 1)[0])
    rate = float(np.exp(slope))
    return {
        "growth_rate_per_call": rate,
        "finite": True,
        "calls": int(calls),
        "fit_from_call": int(fit_from),
        "epsilon_relative": float(epsilon_relative),
        "initial_separation": float(values[0]),
        "final_separation": float(values[-1]),
        "total_amplification": float(values[-1] / values[0]),
        # None, not infinity: a non-growing mode never doubles, and the report
        # this lands in must stay JSON-representable.
        "doubling_days": (
            float(np.log(2.0) / slope * 10.0) if slope > 0.0 else None
        ),
    }


def growth_rate_summary(
    model: Any,
    states: Any,
    static: Any,
    wet: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Average the growth-rate diagnostic over several independent starts."""

    records = [
        measure_growth_rate(model, state, static, wet, seed=index, **kwargs)
        for index, state in enumerate(states)
    ]
    rates = [r["growth_rate_per_call"] for r in records]
    # A None rate means the fit failed on that start; the checkpoint is then
    # treated as not meeting the ceiling rather than silently averaged away.
    failed = any(rate is None for rate in rates)
    worst = None if failed else max(rates)
    return {
        "growth_rate_per_call": None if failed else float(np.mean(rates)),
        "worst_growth_rate_per_call": None if failed else float(worst),
        "measurement_failed_on_a_start": failed,
        "per_start": records,
        "starts": len(records),
        "ceiling": GROWTH_RATE_CEILING,
        "at_or_below_ceiling": bool(not failed and worst <= GROWTH_RATE_CEILING),
        "definition": (
            "exp(slope of log||x_twin - x|| against call index) over calls "
            f"{DIAGNOSTIC_FIT_FROM + 1}-{DIAGNOSTIC_CALLS}, from a "
            f"{DIAGNOSTIC_EPSILON_RELATIVE:g}-relative random initial perturbation"
        ),
    }
