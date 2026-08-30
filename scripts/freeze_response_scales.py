"""Execution step 10 (continued) of docs/Adjoint_faithful_response_training_plan.md.

Freezes plan section 14.2's response-loss normalization scales ``d_{h,g,k}``
(input family ``h``, output group ``g``, lead ``k``), floored at ten times
section 10.3's combined differentiated numerical floor ``n^diff_{h,g,k}``.

Both halves need output that did not exist before
``scripts/extract_forward_response_dataset.py`` ran:

- ``d_{h,g,k}`` is defined "from response-training data only" (section 14.2)
  -- it needs the just-extracted ``train`` role's real response arrays.
- The floor generalizes Gate D2's already-frozen combined floor
  (``amplitude_pilot_final_selection_v1.json``'s ``combined_floor_gb_by_lead``)
  from one GB-pooled scalar per lead to one value per (family, output group,
  lead): section 14.2 needs the un-pooled per-group term, which
  ``analyze_amplitude_pilot_controls.py`` never computed because Gate D2's own
  gate (Q_SNR) only ever needed the pooled aggregate. Recomputed here from the
  same section-10.3 duplicate/tight-CG pilot control reports, not re-derived
  from the pooled number.

One real wrinkle handled explicitly: SSH does not have a single frozen alpha
in the extracted train set. ``config/forward_response_amplitude_pilot_ssh_v2.json``
overrides 12 of ~222 train-role SSH directions to alpha=0.03 (see the plan's
2026-08-26 amendment); the rest stay at the family default, 0.05. Section
10.3's raw noise floor ``n^raw`` is solver/precision noise (nominal-repeat
disagreement, CG-tolerance disagreement) that does not depend on perturbation
size, so converting it to the differentiated floor ``n^diff = n^raw/alpha``
needs *one* alpha per family. Using the larger, majority alpha (0.05) would
under-floor the smaller-alpha-normalized responses. This module reads the
actual minimum alpha used per family directly from the extracted train
direction table (0.10/0.10/0.005/0.03 for U/V/Theta/SSH, the last from data,
not hardcoded) and uses that -- the conservative choice that safely bounds
every direction in that family, matching this study's established
smaller-is-safer convention for edge cases (e.g. the SSH peak-cap repair
preferring the default alpha before ever falling back to the override).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "archive" / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import analyze_amplitude_pilot as pilot_analysis  # noqa: E402
import build_amplitude_pilot as pilot  # noqa: E402
from build_forward_response_inventory import load_json_strict  # noqa: E402
from extract_forward_response_dataset import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUT_ROOT,
    GROUP_SLICES,
    GROUPS,
    _gb_norm_by_group,
    _reject_symlinked_output_path,
)

OUTPUT_PATH = DEFAULT_OUTPUT_ROOT / "response_scales_v1.json"
ALL_LEADS = tuple(range(10, 91, 10))
TRAIN_LEADS = tuple(range(10, 61, 10))
FLOOR_MULTIPLE = 10.0
EPS64 = float(np.finfo(np.float64).eps)


class ResponseScaleError(RuntimeError):
    """Raised when the response scale/floor cannot be legitimately frozen."""


def _by_group_squared(z: np.ndarray, wet: np.ndarray) -> dict[str, float]:
    return {group: float(np.mean(z[sl][:, wet] ** 2)) for group, sl in GROUP_SLICES.items()}


# ---------------------------------------------------------------------------
# Section 10.3 combined floor, per (family, output group, lead).


def _nominal_floor_by_group(wet: np.ndarray, sigma: np.ndarray) -> dict[int, dict[str, float]]:
    """Section 10.2's duplicate-nominal/rounding floor, per output group
    instead of GB-pooled: max over the six pilot anchors."""

    directions = pilot._load_geometry()
    anchors = sorted({(d["regime"], d["anchor_day"]) for d in directions})
    reports = pilot_analysis._load_reports()
    floor: dict[int, dict[str, float]] = {lead: {g: 0.0 for g in GROUPS} for lead in ALL_LEADS}
    for regime, day in anchors:
        nominal_report = reports[pilot_analysis._nominal_key(regime, day)]
        dup_report = reports[pilot_analysis._nominal_key(regime, day, duplicate=True)]
        for lead in ALL_LEADS:
            state_1 = pilot_analysis._state_at_lead(nominal_report, lead, wet) / sigma
            state_2 = pilot_analysis._state_at_lead(dup_report, lead, wet) / sigma
            for group, sl in GROUP_SLICES.items():
                repeat = float(np.sqrt(np.mean((state_1[sl] - state_2[sl])[:, wet] ** 2)))
                bound = 8.0 * EPS64 * max(
                    1.0,
                    float(np.sqrt(np.mean(state_1[sl][:, wet] ** 2))),
                    float(np.sqrt(np.mean(state_2[sl][:, wet] ** 2))),
                )
                floor[lead][group] = max(floor[lead][group], repeat, bound)
    return floor


def _raw_floor_by_family_group(
    wet: np.ndarray, sigma: np.ndarray, provisional_alpha: dict[str, float]
) -> dict[str, dict[int, dict[str, float]]]:
    """Section 10.3's frozen ``n^raw_{h,A,k}``, generalized from the GB-pooled
    scalar ``analyze_amplitude_pilot_controls.py`` computed to one value per
    output group -- the max, over the 12 controlled long pilot directions'
    regime/sign, of the duplicate-nominal floor, the perturbed-repeat
    disagreement, and the tight-CG disagreement."""

    reports = pilot_analysis._load_reports()
    nominal_floor = _nominal_floor_by_group(wet, sigma)
    directions = [row for row in pilot._load_geometry() if row["long"]]
    if len(directions) != 12:
        raise ResponseScaleError(f"expected 12 controlled long pilot directions, found {len(directions)}")

    worst: dict[str, dict[int, dict[str, float]]] = {
        family: {lead: {g: nominal_floor[lead][g] for g in GROUPS} for lead in ALL_LEADS}
        for family in GROUPS
    }
    for direction in directions:
        family = direction["family"]
        alpha = provisional_alpha[family]
        regime, day = direction["regime"], direction["anchor_day"]
        nominal_report = reports[pilot_analysis._nominal_key(regime, day)]
        dup_nominal_report = reports[pilot_analysis._nominal_key(regime, day, duplicate=True)]
        tight_nominal_report = reports[f"{regime}_d{day:04d}_nominal_tight"]
        for sign, sign_token in ((1, "plus"), (-1, "minus")):
            prod_key = pilot_analysis._signed_key(regime, day, family, alpha, sign)
            prod_report = reports[prod_key]
            dup_report = reports[f"{prod_key}_duplicate"]
            tight_report = reports[f"{prod_key}_tight"]
            for lead in ALL_LEADS:
                nominal = pilot_analysis._state_at_lead(nominal_report, lead, wet) / sigma
                nominal_dup = pilot_analysis._state_at_lead(dup_nominal_report, lead, wet) / sigma
                nominal_tight = pilot_analysis._state_at_lead(tight_nominal_report, lead, wet) / sigma
                prod_state = pilot_analysis._state_at_lead(prod_report, lead, wet) / sigma
                dup_state = pilot_analysis._state_at_lead(dup_report, lead, wet) / sigma
                tight_state = pilot_analysis._state_at_lead(tight_report, lead, wet) / sigma
                r_prod = (prod_state - nominal) / float(sign)
                r_dup = (dup_state - nominal_dup) / float(sign)
                r_tight = (tight_state - nominal_tight) / float(sign)
                repeat_by_group = _gb_norm_diff_by_group(r_prod, r_dup, wet)
                cg_by_group = _gb_norm_diff_by_group(r_prod, r_tight, wet)
                for group in GROUPS:
                    worst[family][lead][group] = max(
                        worst[family][lead][group], repeat_by_group[group], cg_by_group[group]
                    )
    return worst


def _gb_norm_diff_by_group(a: np.ndarray, b: np.ndarray, wet: np.ndarray) -> dict[str, float]:
    diff = a - b
    return {group: float(np.sqrt(np.mean(diff[sl][:, wet] ** 2))) for group, sl in GROUP_SLICES.items()}


# ---------------------------------------------------------------------------
# Section 14.2 scale, from the extracted train response arrays.


def _load_train_direction_table(output_root: Path) -> list[dict[str, Any]]:
    path = output_root / "train_direction_table.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_scales(
    dataset_path: Path, output_root: Path, sigma: np.ndarray, wet: np.ndarray
) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, float]]:
    rows = _load_train_direction_table(output_root)
    store = zarr.open_consolidated(str(dataset_path), mode="r")
    train = store["train"]
    short_response = np.asarray(train["short"]["response_p64"])
    long_response = np.asarray(train["long"]["response_p64"])
    lead_days = [int(v) for v in np.asarray(train["lead_days"])]
    if tuple(lead_days) != TRAIN_LEADS:
        raise ResponseScaleError(f"unexpected train lead_days {lead_days}")

    sum_sq = {h: {g: {k: 0.0 for k in TRAIN_LEADS} for g in GROUPS} for h in GROUPS}
    count = {h: {g: {k: 0 for k in TRAIN_LEADS} for g in GROUPS} for h in GROUPS}
    alpha_used: dict[str, set[float]] = {family: set() for family in GROUPS}

    for row in rows:
        family = row["input_family"]
        alpha = float(row["alpha"])
        alpha_used[family].add(alpha)
        array = short_response if row["array_group"] == "short" else long_response
        leads = (10,) if row["array_group"] == "short" else TRAIN_LEADS
        response = array[row["array_row"]]  # (2, L_or_1, 46, 62, 62); index0=minus, index1=plus
        for sign_index, sign in ((0, -1), (1, 1)):
            for lead_index, lead in enumerate(leads):
                raw_diff = response[sign_index, lead_index]
                r_m = (raw_diff / sigma) / (float(sign) * alpha)
                by_group = _by_group_squared(r_m, wet)
                for group in GROUPS:
                    sum_sq[family][group][lead] += by_group[group]
                    count[family][group][lead] += 1

    d_values: dict[str, dict[str, dict[int, float]]] = {
        h: {g: {} for g in GROUPS} for h in GROUPS
    }
    for h in GROUPS:
        for g in GROUPS:
            for k in TRAIN_LEADS:
                n = count[h][g][k]
                if n == 0:
                    raise ResponseScaleError(f"no train response data for family={h} group={g} lead={k}")
                d_values[h][g][k] = math_sqrt(sum_sq[h][g][k] / n)

    alpha_for_floor = {family: min(values) for family, values in alpha_used.items() if values}
    missing = set(GROUPS) - set(alpha_for_floor)
    if missing:
        raise ResponseScaleError(f"no train directions found for families {missing}")
    return d_values, alpha_for_floor


def math_sqrt(value: float) -> float:
    if value < 0.0:
        raise ResponseScaleError(f"negative mean-square value {value}")
    return float(np.sqrt(value))


# ---------------------------------------------------------------------------
# Orchestration.


def freeze(
    dataset_path: Path = DEFAULT_DATASET_PATH, output_root: Path = DEFAULT_OUTPUT_ROOT
) -> dict[str, Any]:
    contract, _roots, grid = pilot._load_sources(pilot.DEFAULT_DATASET_CONTRACT)
    pilot_contract = load_json_strict(pilot.DEFAULT_PILOT_CONTRACT)
    sigma = pilot._load_normalizer(pilot_contract)
    wet = grid.wet

    final_amplitudes_path = output_root / "amplitude_pilot_final_selection_v1.json"
    theta_v2 = load_json_strict(output_root / "amplitude_pilot_theta_v2_selection.json")
    provisional_for_controls = {
        "U": load_json_strict(final_amplitudes_path)["provisional_alpha_per_family"]["U"],
        "V": load_json_strict(final_amplitudes_path)["provisional_alpha_per_family"]["V"],
        "SSH": load_json_strict(final_amplitudes_path)["provisional_alpha_per_family"]["SSH"],
        "Theta": theta_v2["provisional_alpha_theta"],
    }

    n_raw = _raw_floor_by_family_group(wet, sigma, provisional_for_controls)
    d_values, alpha_for_floor = compute_scales(dataset_path, output_root, sigma, wet)

    frozen: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for h in GROUPS:
        frozen[h] = {}
        for g in GROUPS:
            frozen[h][g] = {}
            for k in TRAIN_LEADS:
                n_diff = n_raw[h][k][g] / alpha_for_floor[h]
                floor_value = FLOOR_MULTIPLE * n_diff
                raw_d = d_values[h][g][k]
                frozen[h][g][k] = {
                    "d_unfloored": raw_d,
                    "n_raw": n_raw[h][k][g],
                    "n_diff": n_diff,
                    "floor": floor_value,
                    "d": max(raw_d, floor_value),
                    "floor_active": floor_value > raw_d,
                }

    payload = {
        "version": "response_scales_v1",
        "definition": "plan section 14.2 d_{h,g,k}, floored at 10x section 10.3's n_diff_{h,g,k}",
        "alpha_used_for_floor_conversion": alpha_for_floor,
        "train_lead_days": list(TRAIN_LEADS),
        "scales": frozen,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_PATH
    _reject_symlinked_output_path(output_path)
    if output_path.exists():
        raise ResponseScaleError(f"refusing to overwrite an already-frozen scales file: {output_path}")
    payload_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    import os

    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload_bytes)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)
    return payload


def main() -> int:
    payload = freeze()
    active = [
        (h, g, k)
        for h in GROUPS
        for g in GROUPS
        for k in TRAIN_LEADS
        if payload["scales"][h][g][k]["floor_active"]
    ]
    print(json.dumps({"alpha_used_for_floor_conversion": payload["alpha_used_for_floor_conversion"]}, indent=2))
    print(f"floor active for {len(active)} of {len(GROUPS)**2 * len(TRAIN_LEADS)} (family,group,lead) cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
