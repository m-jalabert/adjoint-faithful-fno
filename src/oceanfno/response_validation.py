"""Held-out forward-response validation, View 2 of plan section 16.2.

Computes the development composite ``S_resp_10:60`` (and the underlying
per-cell relative-L2 breakdown) against the 216 response-validation
directions (anchors 6010/6050/6080, disjoint from response training and
from pilot/blind), for one model checkpoint. Never reads blind or adjoint
data -- only the ``validation`` role of the same curated store
``response_dataset.py`` reads from.

    E_{q,k,g,s}(all wet) = sqrt(
        sum_{c in g, wet} (r_F - r_M)^2
        / max(sum_{c in g, wet} r_M^2, N_{g} * n_diff_{h,g,k}^2)
    )

    S_resp_10:60 = (1/(4*4*5)) * sum_{h,g,R} [
        (19/24) * Ebar_{h,g,R,10} + (1/24) * sum_{k in 20..60} Ebar_{h,g,R,k}
    ]

where ``Ebar_{h,g,R,k}`` is the arithmetic mean of ``E`` over sign, case
(every validation direction sharing that (h,R) with a lead-k target), and
regime. ``n_diff_{h,g,k}`` is the same training-only combined numerical
floor already frozen in ``response_scales_v1.json`` (its ``n_diff`` field,
not ``d`` -- ``d`` is section 14.2's *training-loss* scale; this is section
16.2's own, separately-declared floor term in the relative-L2 denominator).

**Units.** The curated store's ``anchors/state_p32`` and
``{short,long}/input_state_p32`` arrays are *physical*-unit P32 projections
(``build_forward_response_inventory.pickup_to_trajectory_p32``'s output,
step 10); the model, like every other production code path, predicts
*normalized* state ``x_hat=(x-mean)/scale``. Initial states are normalized
here before the first forward call, exactly as ``ProductionStepper``/
``RolloutDataset`` do; the model's own output at every later call is fed
back autoregressively unconverted (no denormalize-renormalize round trip),
matching ``state_unroll``. The truth response arrays, by contrast, are
already stored as *raw physical* differences
(``P64[perturbed]-P64[nominal]``, section 13), so truth still divides by
``sigma`` via ``oriented_response``. The model side does not: dividing a
normalized-state output difference by ``sigma`` again would double-count
it. Both are mathematically the same quantity --
``(y_physical_plus - y_physical_nominal)/sigma`` telescopes to
``normalized_output_plus - normalized_output_nominal`` exactly, since
``y_physical = normalized_output*scale + mean`` and the additive ``mean``
term cancels between the two branches -- so the model side just needs
``(output_diff)/(sign*alpha)``, computed by ``_model_oriented_response``
below rather than reusing ``oriented_response`` (which would apply the
extra, wrong, division).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .response_dataset import GROUP_SLICES, GROUPS, REGIMES, ResponseStore, load_direction_table
from .response_objective import load_response_scale_entries, oriented_response
from .runtime import torch

REGIONS = ("WBC", "interior", "eastern", "northern", "southern")
LEAD_WEIGHTS = {10: 19.0 / 24.0, 20: 1.0 / 24.0, 30: 1.0 / 24.0, 40: 1.0 / 24.0, 50: 1.0 / 24.0, 60: 1.0 / 24.0}
GROUP_CHANNEL_COUNTS = {"U": 15, "V": 15, "Theta": 15, "SSH": 1}


class ResponseValidationError(RuntimeError):
    """Raised when the held-out response validation cannot be computed."""


def _model_oriented_response(output_diff: Any, sign: int, alpha: float) -> Any:
    """``(F(x+s*alpha*v) - F(x)) / (s*alpha)`` from the model's own
    *normalized*-state output difference -- no sigma division (see module
    docstring's units note)."""

    return output_diff / (float(sign) * float(alpha))


def _model_response(
    model: Any,
    nominal_p32: np.ndarray,
    minus_p32: np.ndarray,
    plus_p32: np.ndarray,
    point_mean: np.ndarray,
    point_scale: np.ndarray,
    statics: np.ndarray,
    wet_bool: np.ndarray,
    wet_float: Any,
    leads: tuple[int, ...],
    device: Any,
) -> dict[int, tuple[Any, Any]]:
    """Autoregressively rolls the (nominal, minus, plus) triplet in one
    batched forward per lead (section 15.2's convention), returning
    ``{lead: (normalized_minus_out - normalized_nominal_out, normalized_plus_out - normalized_nominal_out)}``."""

    physical = np.stack([nominal_p32, minus_p32, plus_p32], axis=0)
    normalized = (physical - point_mean[None]) / point_scale[None]
    normalized[:, :, ~wet_bool] = 0.0
    state = torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32)).to(device)
    static = torch.from_numpy(np.broadcast_to(statics, (3, *statics.shape)).copy()).to(device=device, dtype=torch.float32)
    results: dict[int, tuple[Any, Any]] = {}
    max_lead = max(leads)
    calls = max_lead // 10
    with torch.no_grad():
        for call in range(1, calls + 1):
            features = torch.cat([state, static], dim=1)
            state = model(features) * wet_float
            lead = call * 10
            if lead in leads:
                nominal_out, minus_out, plus_out = state[0], state[1], state[2]
                results[lead] = (minus_out - nominal_out, plus_out - nominal_out)
    return results


def evaluate_response_validation(
    model: Any,
    device: Any,
    point_mean: np.ndarray,
    point_scale: np.ndarray,
    wet_array: np.ndarray,
    statics: np.ndarray,
    *,
    scales_path=None,
    dataset_path=None,
    max_lead: int = 60,
) -> dict[str, Any]:
    """Full View-2 pass: every validation direction, every available lead
    up to ``max_lead``, both signs. Returns the per-cell breakdown and the
    ``S_resp_10:60`` composite. ``statics`` is the ``(3,5,62,62)``
    per-regime static block already computed for nominal training/validation
    (shared, not recomputed here); ``point_mean``/``point_scale`` are the
    same ``(46,62,62)`` recomputed-normalizer arrays nominal training uses."""

    directions = load_direction_table("validation")
    store = ResponseStore("validation", dataset_path=dataset_path) if dataset_path else ResponseStore("validation")
    scale_entries = load_response_scale_entries(scales_path) if scales_path else load_response_scale_entries()
    sigma_t = torch.from_numpy(np.asarray(point_scale, dtype=np.float32)).to(device)
    wet_bool = wet_array.astype(bool)
    wet_bool_t = torch.from_numpy(wet_bool).to(device)
    wet_float_t = torch.from_numpy(wet_bool.astype(np.float32))[None, None].to(device)

    # cell_values[(h,g,R,k)] -> list of per-case E values (pooled over sign/case/regime)
    cell_values: dict[tuple[str, str, str, int], list[float]] = {
        (h, g, R, k): [] for h in GROUPS for g in GROUPS for R in REGIONS for k in LEAD_WEIGHTS
    }

    model.eval()
    for direction in directions:
        leads = tuple(k for k in LEAD_WEIGHTS if k <= (60 if direction.long else 10) and k <= max_lead)
        nominal = store.anchor_state_p32(direction.anchor_row)
        minus_in, plus_in = store.branch_inputs_p32(direction)
        raw_minus, raw_plus = store.raw_response(direction)  # (L,46,62,62) each, L matches leads count
        regime_index = REGIMES.index(direction.regime)
        model_out = _model_response(
            model, nominal, minus_in, plus_in, point_mean, point_scale,
            statics[regime_index], wet_bool, wet_float_t, leads, device,
        )
        for lead_index, lead in enumerate(leads):
            model_diff_minus, model_diff_plus = model_out[lead]
            r_f_minus = _model_oriented_response(model_diff_minus, -1, direction.alpha)
            r_f_plus = _model_oriented_response(model_diff_plus, 1, direction.alpha)
            truth_minus = torch.from_numpy(np.ascontiguousarray(raw_minus[lead_index], dtype=np.float32)).to(device)
            truth_plus = torch.from_numpy(np.ascontiguousarray(raw_plus[lead_index], dtype=np.float32)).to(device)
            r_m_minus = oriented_response(truth_minus, sigma_t, -1, direction.alpha)
            r_m_plus = oriented_response(truth_plus, sigma_t, 1, direction.alpha)
            for group in GROUPS:
                sl = GROUP_SLICES[group]
                n_scored = GROUP_CHANNEL_COUNTS[group] * int(wet_bool.sum())
                floor = scale_entries[direction.input_family][group][lead]["n_diff"]
                for r_f, r_m in ((r_f_minus, r_m_minus), (r_f_plus, r_m_plus)):
                    diff_sq = ((r_f[sl] - r_m[sl])[..., wet_bool_t] ** 2).sum()
                    truth_sq = ((r_m[sl])[..., wet_bool_t] ** 2).sum()
                    denom = torch.clamp(truth_sq, min=n_scored * (floor**2))
                    e_value = float(torch.sqrt(diff_sq / denom).item())
                    cell_values[(direction.input_family, group, direction.region, lead)].append(e_value)

    missing = [key for key, values in cell_values.items() if not values]
    if missing:
        raise ResponseValidationError(f"{len(missing)} (family,group,region,lead) cells have no scored case: {missing[:5]}...")

    cell_means = {key: float(np.mean(values)) for key, values in cell_values.items()}
    composite = 0.0
    per_hgr: dict[tuple[str, str, str], float] = {}
    for h in GROUPS:
        for g in GROUPS:
            for R in REGIONS:
                cell_composite = sum(LEAD_WEIGHTS[k] * cell_means[(h, g, R, k)] for k in LEAD_WEIGHTS)
                per_hgr[(h, g, R)] = cell_composite
                composite += cell_composite
    composite /= 4 * 4 * 5

    return {
        "S_resp_10_60": composite,
        "per_family_group_region": {f"{h}|{g}|{R}": v for (h, g, R), v in per_hgr.items()},
        "cell_means": {f"{h}|{g}|{R}|{k}": v for (h, g, R, k), v in cell_means.items()},
        "n_directions": len(directions),
    }
