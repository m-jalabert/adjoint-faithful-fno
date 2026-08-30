"""The group-balanced, sign-oriented response loss (plan section 14.2).

For one response direction q with input family h(q), lead k:

    r_M[s] = (raw_truth_diff[s] / sigma) / (s * alpha)     truth
    r_F[s] = (F_k(x + s*alpha*v) - F_k(x)) / (s * alpha)   model, both branches
                                                            from the model's
                                                            own nominal rollout
    l_{q,k} = (1/8) * sum_{s in {-1,+1}} sum_{g in groups}
              mean_{c in g, wet}[(r_F[s]-r_M[s])^2] / d_{h,g,k}^2

using the frozen, training-only, floored scales ``d_{h,g,k}`` from
``response_scales_v1.json`` (built in step 10 -- section 14.2's own text:
"Compute, from response-training data only, an RMS scale... Floor it at ten
times the corresponding differentiated combined rounding/repeat/CG-disagreement
scale from section 10"). Short samples use ``l_{q,10}``; long samples use the
unweighted mean of ``l_{q,k}`` over ``k in {10,...,60}`` (section 14.2).

Section 14.3 ("no ordinary state loss on perturbed trajectories"): this
module never compares a perturbed branch's absolute state to any truth
state, only response differences to response differences.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .response_dataset import GROUP_SLICES, GROUPS, DEFAULT_OUTPUT_ROOT
from .runtime import torch

DEFAULT_SCALES_PATH = DEFAULT_OUTPUT_ROOT / "response_scales_v1.json"


class ResponseObjectiveError(RuntimeError):
    """Raised when the response loss cannot be legitimately computed."""


def load_response_scale_entries(path: Path = DEFAULT_SCALES_PATH) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    """Returns ``scales[h][g][k] -> {"d":..., "n_diff":..., ...}`` -- the raw
    frozen entries, for callers that need more than just the training-loss
    scale ``d`` (e.g. ``response_validation.py`` needs ``n_diff``, section
    16.2's own floor term, distinct from section 14.2's loss scale)."""

    payload = json.loads(Path(path).read_text())
    raw = payload["scales"]
    return {h: {g: {int(k): dict(entry) for k, entry in raw[h][g].items()} for g in GROUPS} for h in GROUPS}


def load_response_scales(path: Path = DEFAULT_SCALES_PATH) -> dict[str, dict[str, dict[int, float]]]:
    """Returns ``scales[h][g][k] -> d_{h,g,k}`` (already floored), float leads --
    section 14.2's training-loss scale specifically."""

    entries = load_response_scale_entries(path)
    return {h: {g: {k: float(entry["d"]) for k, entry in entries[h][g].items()} for g in GROUPS} for h in GROUPS}


def response_scale_tensor(scales: Mapping[str, Mapping[str, Mapping[int, float]]], input_family: str, lead: int, *, device: Any, dtype: Any) -> Any:
    """``d_{h,g,k}`` for every output group ``g``, shape ``(4,)`` in group order."""

    values = [scales[input_family][g][lead] for g in GROUPS]
    return torch.tensor(values, device=device, dtype=dtype)


def oriented_response(raw_diff: Any, sigma: Any, sign: int, alpha: float) -> Any:
    """``r^s = (raw_diff/sigma)/(s*alpha)`` -- shared by truth and model,
    section 10.2/14.2's oriented, standardized response."""

    return (raw_diff / sigma) / (float(sign) * float(alpha))


def response_term(
    r_f_minus: Any,
    r_f_plus: Any,
    r_m_minus: Any,
    r_m_plus: Any,
    wet: Any,
    scale_by_group: Any,
) -> Any:
    """``l_{q,k}``: mean over sign and output group of the group-balanced
    squared error, scaled by ``d_{h,g,k}^2``. All four response tensors are
    ``(46,62,62)`` (or batched ``(N,46,62,62)``); ``wet`` broadcasts as
    ``(62,62)`` or ``(1,62,62)``; ``scale_by_group`` is ``(4,)`` in
    ``GROUPS`` order.
    """

    total = 0.0
    for index, group in enumerate(GROUPS):
        sl = GROUP_SLICES[group]
        diff_minus = (r_f_minus[..., sl, :, :] - r_m_minus[..., sl, :, :]) ** 2
        diff_plus = (r_f_plus[..., sl, :, :] - r_m_plus[..., sl, :, :]) ** 2
        mean_minus = diff_minus[..., wet].mean()
        mean_plus = diff_plus[..., wet].mean()
        total = total + (mean_minus + mean_plus) / (scale_by_group[index] ** 2)
    return total / 8.0


def short_response_loss(*args: Any, **kwargs: Any) -> Any:
    """Section 14.2: ``L_response^short = l_{q,10}`` -- literally ``response_term``
    evaluated at lead 10, kept as a distinct name for call-site clarity."""

    return response_term(*args, **kwargs)


def long_response_loss(per_lead_terms: list[Any]) -> Any:
    """Section 14.2: ``L_response^long`` is the unweighted mean of
    ``l_{q,k}`` over ``k in {10,20,...,60}`` -- exactly six terms for the
    train/validation long-lead set."""

    if len(per_lead_terms) != 6:
        raise ResponseObjectiveError(f"expected exactly 6 lead terms for a long direction, got {len(per_lead_terms)}")
    return sum(per_lead_terms) / 6.0
