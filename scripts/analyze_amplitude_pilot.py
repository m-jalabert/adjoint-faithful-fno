"""Execution step 8, provisional stage (plan sections 10.2/10.4).

Extracts oriented raw responses from the 154 completed amplitude-pilot runs
(step 7), computes the section-10.2 diagnostics (Q_lin, Q_SNR against the
duplicate-nominal noise floor, P32 realization/antisymmetry, adjacent-alpha
centred-JVP convergence), and applies the frozen provisional-stage selection
rule from ``config/forward_response_amplitude_pilot_v1.json`` to choose the
largest candidate alpha passing every gate, per input family.

This step reads only pilot report/pickup files already on disk -- it submits
no new MITgcm compute. Section 10.3's duplicate/tight-CG confirmatory runs
(new compute, at the provisional alpha only) are a separate, later step.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "archive" / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import build_amplitude_pilot as pilot  # noqa: E402
from build_forward_response_inventory import pickup_to_trajectory_p32  # noqa: E402


REPORT_ROOT = pilot.DEFAULT_REPORT_ROOT
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "af_fno"
    / "response"
    / "forward_response_v1"
    / "amplitude_pilot_provisional_selection_v1.json"
)
EPS64 = float(np.finfo(np.float64).eps)
GROUP_SLICES = {"U": slice(0, 15), "V": slice(15, 30), "Theta": slice(30, 45), "SSH": slice(45, 46)}


class AnalysisError(RuntimeError):
    """Raised when the pilot analysis cannot proceed or draw a conclusion."""


# ---------------------------------------------------------------------------
# Loading


def _load_reports() -> dict[str, dict[str, Any]]:
    reports = {}
    for path in REPORT_ROOT.glob("*.json"):
        reports[path.stem] = json.loads(path.read_text())
    return reports


def _nominal_key(regime: str, day: int, duplicate: bool = False) -> str:
    return f"{regime}_d{day:04d}_nominal" + ("_dup" if duplicate else "")


def _signed_key(regime: str, day: int, family: str, alpha: float, sign: int) -> str:
    return f"{regime}_d{day:04d}_{family}_a{pilot._alpha_token(alpha)}_{'plus' if sign == 1 else 'minus'}"


def _state_at_lead(report: dict[str, Any], lead: int, wet: np.ndarray) -> np.ndarray:
    manifest = report["manifest"]
    run_dir = Path(manifest["run_dir"])
    entry = next(
        c for c in manifest["archived_checkpoints"] if c["day"] - manifest["start_day"] == lead
    )
    meta_path = run_dir / f"pickup.{entry['iteration']:010d}.meta"
    return pickup_to_trajectory_p32(meta_path, wet)


def _initial_state(meta_path: str, wet: np.ndarray) -> np.ndarray:
    return pickup_to_trajectory_p32(meta_path, wet)


# ---------------------------------------------------------------------------
# Diagnostics (section 10.2)


def _gb_norm(z: np.ndarray, wet: np.ndarray) -> float:
    total = 0.0
    for group_slice in GROUP_SLICES.values():
        values = z[group_slice][:, wet]
        total += float(np.mean(values**2))
    return math.sqrt(total / 4.0)


def _raw_nominal_floor(
    nominal_states: dict[tuple[str, int], dict[int, np.ndarray]],
    duplicate_states: dict[tuple[str, int], dict[int, np.ndarray]],
    scale: np.ndarray,
    wet: np.ndarray,
    leads: list[int],
) -> dict[int, float]:
    """section 10.2: max over the six pilot anchors of the duplicate-repeat
    disagreement (or the float64 rounding bound, whichever is larger), per lead."""

    floor: dict[int, float] = {}
    for lead in leads:
        worst = 0.0
        for anchor in nominal_states:
            state_1 = nominal_states[anchor][lead] / scale
            state_2 = duplicate_states[anchor][lead] / scale
            repeat_disagreement = _gb_norm(state_1 - state_2, wet)
            rounding_bound = 8.0 * EPS64 * max(1.0, _gb_norm(state_1, wet), _gb_norm(state_2, wet))
            worst = max(worst, repeat_disagreement, rounding_bound)
        floor[lead] = worst
    return floor


def _p32_checks(
    direction: dict[str, Any],
    alpha: float,
    wet: np.ndarray,
    reports: dict[str, dict[str, Any]],
    sigma: np.ndarray,
) -> tuple[float, float]:
    """Realized-magnitude relative error and antisymmetry error (section 10.2).

    Compares the *realized* P32-cast, centred perturbation (initial pickup,
    before any integration), standardized by the same per-cell sigma used to
    build the direction, against the *intended* standardized RMS target
    (== alpha, since the direction vector was built with unit RMS). Uses the
    same centred support ``direction_vector`` used to solve the RMS scale.
    """

    plus = reports[
        _signed_key(direction["regime"], direction["anchor_day"], direction["family"], alpha, 1)
    ]
    minus = reports[
        _signed_key(direction["regime"], direction["anchor_day"], direction["family"], alpha, -1)
    ]
    nominal = reports[_nominal_key(direction["regime"], direction["anchor_day"])]

    initial_nominal = _initial_state(_resolve_nominal_initial(nominal), wet)
    initial_plus = _initial_state(plus["manifest"]["pickup_meta_path"], wet)
    initial_minus = _initial_state(minus["manifest"]["pickup_meta_path"], wet)

    channel = pilot.channel_index(direction["family"], direction["levels"])
    delta_plus = (initial_plus[channel] - initial_nominal[channel]) / sigma[channel]
    delta_minus = (initial_minus[channel] - initial_nominal[channel]) / sigma[channel]

    native_kernel = pilot._native_kernel(direction)
    support = (pilot._centred_projection(direction["family"], native_kernel) != 0.0) & wet
    magnitude_plus = float(np.sqrt(np.mean(delta_plus[support] ** 2)))
    magnitude_minus = float(np.sqrt(np.mean(delta_minus[support] ** 2)))
    relative_error = max(abs(magnitude_plus - alpha) / alpha, abs(magnitude_minus - alpha) / alpha)

    norm_plus = float(np.sqrt(np.sum(delta_plus**2)))
    norm_minus = float(np.sqrt(np.sum(delta_minus**2)))
    antisymmetry = float(np.sqrt(np.sum((delta_plus + delta_minus) ** 2))) / (
        0.5 * (norm_plus + norm_minus)
    )
    return relative_error, antisymmetry


def _resolve_nominal_initial(nominal_report: dict[str, Any]) -> str:
    # Nominal branches start from the untouched annual pickup, symlinked into
    # their own run_dir under the source's own name.
    run_dir = Path(nominal_report["manifest"]["run_dir"])
    start_iteration = nominal_report["manifest"]["start_iteration"]
    starts = sorted(run_dir.glob(f"pickup.{start_iteration:010d}.meta"))
    if len(starts) != 1:
        raise AnalysisError(f"cannot resolve nominal initial pickup in {run_dir}")
    return str(starts[0])


def main() -> int:
    directions = pilot._load_geometry()
    pilot_contract = pilot.load_json_strict(pilot.DEFAULT_PILOT_CONTRACT)
    alphas = sorted(pilot_contract["directions"]["candidate_alphas"], reverse=True)
    rule = pilot_contract["selection"]

    _contract, _roots, grid = pilot._load_sources(pilot.DEFAULT_DATASET_CONTRACT)
    sigma = pilot._load_normalizer(pilot_contract)
    wet = grid.wet
    reports = _load_reports()

    anchors = sorted({(d["regime"], d["anchor_day"]) for d in directions})
    all_leads = list(range(10, 91, 10))
    nominal_states: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    duplicate_states: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    for regime, day in anchors:
        nominal_report = reports[_nominal_key(regime, day)]
        dup_report = reports[_nominal_key(regime, day, duplicate=True)]
        nominal_states[(regime, day)] = {
            lead: _state_at_lead(nominal_report, lead, wet) for lead in all_leads
        }
        duplicate_states[(regime, day)] = {
            lead: _state_at_lead(dup_report, lead, wet) for lead in all_leads
        }
    floor = _raw_nominal_floor(nominal_states, duplicate_states, sigma, wet, all_leads)

    per_family: dict[str, dict[str, Any]] = {
        family: {"directions": []} for family in ("U", "V", "Theta", "SSH")
    }
    for direction in directions:
        leads = [10] if not direction["long"] else all_leads
        family = direction["family"]
        anchor = (direction["regime"], direction["anchor_day"])
        entry = {
            "regime": direction["regime"],
            "day": direction["anchor_day"],
            "family": family,
            "long": direction["long"],
            "by_alpha": {},
        }
        jhat: dict[float, dict[int, np.ndarray]] = {}
        for alpha in alphas:
            plus_key = _signed_key(direction["regime"], direction["anchor_day"], family, alpha, 1)
            minus_key = _signed_key(direction["regime"], direction["anchor_day"], family, alpha, -1)
            if "manifest" not in reports.get(plus_key, {}) or "manifest" not in reports.get(
                minus_key, {}
            ):
                entry["by_alpha"][alpha] = {"status": "ssh_cap_failure"}
                continue
            plus_report, minus_report = reports[plus_key], reports[minus_key]
            per_lead = {}
            jhat[alpha] = {}
            for lead in leads:
                nominal = nominal_states[anchor][lead] / sigma
                plus_state = _state_at_lead(plus_report, lead, wet) / sigma
                minus_state = _state_at_lead(minus_report, lead, wet) / sigma
                r_plus = plus_state - nominal
                r_minus = nominal - minus_state
                q_lin = _gb_norm(r_plus - r_minus, wet) / (
                    0.5 * (_gb_norm(r_plus, wet) + _gb_norm(r_minus, wet))
                )
                q_snr = 0.5 * (_gb_norm(r_plus, wet) + _gb_norm(r_minus, wet)) / floor[lead]
                per_lead[lead] = {"q_lin": q_lin, "q_snr": q_snr}
                jhat[alpha][lead] = (r_plus + r_minus) / (2.0 * alpha)
            magnitude_error, antisymmetry_error = _p32_checks(direction, alpha, wet, reports, sigma)
            entry["by_alpha"][alpha] = {
                "status": "ran",
                "per_lead": per_lead,
                "p32_magnitude_relative_error": magnitude_error,
                "p32_antisymmetry_error": antisymmetry_error,
                "day10_q_lin_pass": per_lead[10]["q_lin"] <= rule["day_10_q_lin_max"],
                "long_q_lin_pass": all(
                    per_lead[lead]["q_lin"] <= rule["long_every_lead_q_lin_max"] for lead in leads
                ),
                "q_snr_pass": all(per_lead[lead]["q_snr"] >= rule["q_snr_min"] for lead in leads),
                "p32_pass": (
                    magnitude_error
                    <= pilot_contract["precision"]["p32_realized_magnitude_relative_tolerance"]
                    and antisymmetry_error
                    <= pilot_contract["precision"]["p32_antisymmetry_relative_tolerance"]
                ),
            }

        # Adjacent-alpha centred-JVP convergence: each candidate is compared
        # against the next *smaller* candidate present (none for the
        # smallest alpha itself, per the frozen contract).
        for index, alpha in enumerate(alphas):
            if entry["by_alpha"][alpha]["status"] != "ran":
                continue
            smaller = alphas[index + 1] if index + 1 < len(alphas) else None
            if smaller is None or entry["by_alpha"][smaller]["status"] != "ran":
                entry["by_alpha"][alpha]["adjacent_alpha_pass"] = True
                entry["by_alpha"][alpha]["adjacent_alpha_reference"] = None
                continue
            worst_score = 0.0
            for lead in leads:
                diff = _gb_norm(jhat[alpha][lead] - jhat[smaller][lead], wet)
                denom = max(_gb_norm(jhat[smaller][lead], wet), floor[lead] / smaller)
                worst_score = max(worst_score, diff / denom)
            entry["by_alpha"][alpha]["adjacent_alpha_score"] = worst_score
            entry["by_alpha"][alpha]["adjacent_alpha_pass"] = (
                worst_score <= rule["adjacent_alpha_centred_jvp_relative_max"]
            )
            entry["by_alpha"][alpha]["adjacent_alpha_reference"] = smaller
        per_family[family]["directions"].append(entry)

    provisional: dict[str, Any] = {}
    for family, data in per_family.items():
        chosen = None
        for alpha in alphas:
            all_pass = True
            for entry in data["directions"]:
                info = entry["by_alpha"][alpha]
                if info["status"] != "ran" or not (
                    info["day10_q_lin_pass"]
                    and info["long_q_lin_pass"]
                    and info["q_snr_pass"]
                    and info["p32_pass"]
                    and info["adjacent_alpha_pass"]
                ):
                    all_pass = False
                    break
            if all_pass:
                chosen = alpha
                break
        provisional[family] = chosen
        data["provisional_alpha"] = chosen

    output = {
        "version": "amplitude_pilot_provisional_selection_v1",
        "candidate_alphas_largest_first": alphas,
        "raw_nominal_floor_gb_by_lead": floor,
        "provisional_alpha_per_family": provisional,
        "detail": per_family,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n")
    print(
        json.dumps(
            {"provisional_alpha_per_family": provisional, "raw_nominal_floor_gb_by_lead": floor},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
