"""Blind forward-response scoring through day 90 (plan section 17).

Section 16.2's development scorer stops at lead 60, because no development
score may read a model response beyond day 60 (section 15.3). The blind test
is the one package that does: "Report every section-16.2 diagnostic for all
216 directions at day 10 and for the predeclared 36-direction long subset at
days 20,30,...,90", and it defines a second composite over the long cases,

    S_resp^90 = (1/(4*4*5)) * sum_{h,g,R} Ebar_{h,g,R,90}.

This is a separate module rather than a parameter on ``response_validation``
for the same reason ``figures_response``/``anomaly_response`` are separate:
``src/oceanfno/response_validation.py`` is pinned by
``model_c_adjoint_faithful_response_v1.json``'s
``study_contract.new_runner_source_hashes`` and re-verified on every contract
load, so editing it would retire arm B and arm C's ability to re-verify
themselves. Every numerical helper is imported from it and executed unchanged
-- the autoregressive triplet roll, the oriented-response conventions, the
relative-L2 definition -- so a blind case is scored by exactly the code that
scored the development cases, extended in lead only.

**The lead-90 numerical floor.** Section 16.2's denominator floors the truth
energy at ``N_{g,Omega} * n_{h,g,k}^2``, where ``n`` is the *training-only*
combined numerical floor frozen in ``response_scales_v1.json``. That file
declares ``train_lead_days = [10..60]``: no lead-90 value exists, and none may
be computed now, because computing one from the blind responses would let
blind data into its own scoring rule. This module therefore carries the
frozen lead-60 floor forward to lead 90, which is training-only by
construction. The choice is conservative in the model's disfavour: ``n``
decreases with lead over 10-60, so the lead-60 value is the smallest
available, making the denominator least likely to be raised and ``E``
correspondingly no smaller. It is recorded in the result as
``lead_90_floor_source`` so the substitution is never silent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .response_dataset import GROUP_SLICES, GROUPS, REGIMES, ResponseStore, load_direction_table
from .response_objective import load_response_scale_entries, oriented_response
from .response_validation import (
    GROUP_CHANNEL_COUNTS,
    REGIONS,
    ResponseValidationError,
    _model_oriented_response,
    _model_response,
)
from .runtime import torch

#: Section 16.2's development weights, unchanged: 19/24 at day 10 and 1/24 at
#: each of days 20-60. Day 90 is deliberately absent -- it enters only through
#: S_resp^90, never through S_resp^10:60, so the two composites stay exactly
#: the quantities sections 16.2 and 17 define.
LEAD_WEIGHTS_10_60 = {10: 19.0 / 24.0, 20: 1.0 / 24.0, 30: 1.0 / 24.0, 40: 1.0 / 24.0, 50: 1.0 / 24.0, 60: 1.0 / 24.0}
BLIND_LEADS = (10, 20, 30, 40, 50, 60, 70, 80, 90)
LONG_LEAD = 90
FLOOR_FALLBACK_LEAD = 60


def evaluate_blind_response(
    model: Any,
    device: Any,
    point_mean: np.ndarray,
    point_scale: np.ndarray,
    wet_array: np.ndarray,
    statics: np.ndarray,
    *,
    role: str = "blind_test",
    scales_path=None,
    dataset_path=None,
    output_root=None,
) -> dict[str, Any]:
    """Score one checkpoint on the blind directions at every available lead.

    Returns both composites plus the full per-cell breakdown, so section 17's
    per-family, per-region and day-10 aggregate conditions can all be applied
    from one pass.
    """

    # The blind role's direction table lives under its own evaluator-only
    # output root, not the development one.
    directions = load_direction_table(role, output_root) if output_root else load_direction_table(role)
    store = ResponseStore(role, dataset_path=dataset_path) if dataset_path else ResponseStore(role)
    scale_entries = load_response_scale_entries(scales_path) if scales_path else load_response_scale_entries()
    sigma_t = torch.from_numpy(np.asarray(point_scale, dtype=np.float32)).to(device)
    wet_bool = wet_array.astype(bool)
    wet_bool_t = torch.from_numpy(wet_bool).to(device)
    wet_float_t = torch.from_numpy(wet_bool.astype(np.float32))[None, None].to(device)

    store_leads = tuple(store.lead_days)
    cell_values: dict[tuple[str, str, str, int], list[float]] = {
        (h, g, R, k): [] for h in GROUPS for g in GROUPS for R in REGIONS for k in store_leads
    }

    model.eval()
    for direction in directions:
        available = store_leads if direction.long else (10,)
        leads = tuple(k for k in available if k in store_leads)
        nominal = store.anchor_state_p32(direction.anchor_row)
        minus_in, plus_in = store.branch_inputs_p32(direction)
        raw_minus, raw_plus = store.raw_response(direction)
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
            floor_lead = lead if lead in scale_entries[direction.input_family][GROUPS[0]] else FLOOR_FALLBACK_LEAD
            for group in GROUPS:
                sl = GROUP_SLICES[group]
                n_scored = GROUP_CHANNEL_COUNTS[group] * int(wet_bool.sum())
                floor = scale_entries[direction.input_family][group][floor_lead]["n_diff"]
                for r_f, r_m in ((r_f_minus, r_m_minus), (r_f_plus, r_m_plus)):
                    diff_sq = ((r_f[sl] - r_m[sl])[..., wet_bool_t] ** 2).sum()
                    truth_sq = ((r_m[sl])[..., wet_bool_t] ** 2).sum()
                    denom = torch.clamp(truth_sq, min=n_scored * (floor**2))
                    cell_values[(direction.input_family, group, direction.region, lead)].append(
                        float(torch.sqrt(diff_sq / denom).item())
                    )

    cell_means = {key: float(np.mean(v)) for key, v in cell_values.items() if v}
    missing_10_60 = [k for k in cell_values if k[3] in LEAD_WEIGHTS_10_60 and not cell_values[k]]
    if missing_10_60:
        raise ResponseValidationError(
            f"{len(missing_10_60)} (family,group,region,lead) cells have no scored case in 10-60: {missing_10_60[:5]}..."
        )
    missing_90 = [k for k in cell_values if k[3] == LONG_LEAD and not cell_values[k]]
    if LONG_LEAD in store_leads and missing_90:
        raise ResponseValidationError(f"{len(missing_90)} cells have no scored case at lead 90")

    per_hgr_10_60: dict[tuple[str, str, str], float] = {}
    per_hgr_90: dict[tuple[str, str, str], float] = {}
    for h in GROUPS:
        for g in GROUPS:
            for R in REGIONS:
                per_hgr_10_60[(h, g, R)] = sum(w * cell_means[(h, g, R, k)] for k, w in LEAD_WEIGHTS_10_60.items())
                if LONG_LEAD in store_leads:
                    per_hgr_90[(h, g, R)] = cell_means[(h, g, R, LONG_LEAD)]

    n_cells = len(GROUPS) * len(GROUPS) * len(REGIONS)
    result: dict[str, Any] = {
        "role": role,
        "S_resp_10_60": sum(per_hgr_10_60.values()) / n_cells,
        "per_family_group_region": {f"{h}|{g}|{R}": v for (h, g, R), v in per_hgr_10_60.items()},
        "cell_means": {f"{h}|{g}|{R}|{k}": v for (h, g, R, k), v in cell_means.items()},
        "n_directions": len(directions),
        "store_lead_days": list(store_leads),
        "lead_90_floor_source": (
            f"section 16.2's training-only n_diff is frozen for leads 10-60 only; the lead-"
            f"{FLOOR_FALLBACK_LEAD} value is carried forward to any longer lead, since computing "
            "one from the blind responses would let blind data into its own scoring rule"
        ),
    }
    if LONG_LEAD in store_leads:
        result["S_resp_90"] = sum(per_hgr_90.values()) / n_cells
        result["per_family_group_region_90"] = {f"{h}|{g}|{R}": v for (h, g, R), v in per_hgr_90.items()}
    return result


def day10_family_region(cell_means: dict[str, float]) -> dict[str, float]:
    """Section 17's "input-family/region day-10 aggregate": for each (input
    family, input-centre region), the mean over output groups at lead 10.
    Same definition Gate M1 applied, reused so the two are comparable."""

    return {
        f"{h}|{R}": float(np.mean([cell_means[f"{h}|{g}|{R}|10"] for g in GROUPS]))
        for h in GROUPS
        for R in REGIONS
    }
