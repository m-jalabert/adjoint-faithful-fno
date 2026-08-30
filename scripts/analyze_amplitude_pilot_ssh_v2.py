"""Provisional-stage analysis for the SSH-only follow-up pilot (v2).

Reuses every diagnostic function from ``analyze_amplitude_pilot.py``
unchanged (group-balanced norm, duplicate-nominal floor, P32 checks,
adjacent-alpha JVP, and the SSH-cap-failure ``status`` handling -- kept
here, unlike the Theta v2 script, because a smaller alpha could in
principle still fail the cap at an unusually large-response pilot
location, and the frozen contract's own selection loop already knows how
to treat a missing report as a hard fail for that alpha rather than
crashing on it) -- only the alpha list and family are different, per
``config/forward_response_amplitude_pilot_ssh_v2.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "archive" / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import analyze_amplitude_pilot as base  # noqa: E402
import build_amplitude_pilot as pilot  # noqa: E402

SSH_V2_CONTRACT = PROJECT_ROOT / "config" / "forward_response_amplitude_pilot_ssh_v2.json"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "af_fno"
    / "response"
    / "forward_response_v1"
    / "amplitude_pilot_ssh_v2_selection.json"
)


def main() -> int:
    ssh_contract = pilot.load_json_strict(SSH_V2_CONTRACT)
    alphas = sorted(ssh_contract["candidate_alphas"], reverse=True)
    pilot_contract = pilot.load_json_strict(pilot.DEFAULT_PILOT_CONTRACT)
    rule = pilot_contract["selection"]

    _contract, _roots, grid = pilot._load_sources(pilot.DEFAULT_DATASET_CONTRACT)
    sigma = pilot._load_normalizer(pilot_contract)
    wet = grid.wet
    reports = base._load_reports()

    directions = [row for row in pilot._load_geometry() if row["family"] == "SSH"]
    anchors = sorted({(d["regime"], d["anchor_day"]) for d in directions})
    all_leads = list(range(10, 91, 10))
    nominal_states = {}
    duplicate_states = {}
    for regime, day in anchors:
        nominal_report = reports[base._nominal_key(regime, day)]
        dup_report = reports[base._nominal_key(regime, day, duplicate=True)]
        nominal_states[(regime, day)] = {
            lead: base._state_at_lead(nominal_report, lead, wet) for lead in all_leads
        }
        duplicate_states[(regime, day)] = {
            lead: base._state_at_lead(dup_report, lead, wet) for lead in all_leads
        }
    floor = base._raw_nominal_floor(nominal_states, duplicate_states, sigma, wet, all_leads)

    detail = []
    for direction in directions:
        leads = [10] if not direction["long"] else all_leads
        anchor = (direction["regime"], direction["anchor_day"])
        entry = {
            "regime": direction["regime"],
            "day": direction["anchor_day"],
            "kernel": direction["kernel"],
            "long": direction["long"],
            "by_alpha": {},
        }
        jhat = {}
        for alpha in alphas:
            plus_key = base._signed_key(direction["regime"], direction["anchor_day"], "SSH", alpha, 1)
            minus_key = base._signed_key(
                direction["regime"], direction["anchor_day"], "SSH", alpha, -1
            )
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
                plus_state = base._state_at_lead(plus_report, lead, wet) / sigma
                minus_state = base._state_at_lead(minus_report, lead, wet) / sigma
                r_plus = plus_state - nominal
                r_minus = nominal - minus_state
                q_lin = base._gb_norm(r_plus - r_minus, wet) / (
                    0.5 * (base._gb_norm(r_plus, wet) + base._gb_norm(r_minus, wet))
                )
                q_snr = (
                    0.5 * (base._gb_norm(r_plus, wet) + base._gb_norm(r_minus, wet)) / floor[lead]
                )
                per_lead[lead] = {"q_lin": q_lin, "q_snr": q_snr}
                jhat[alpha][lead] = (r_plus + r_minus) / (2.0 * alpha)
            magnitude_error, antisymmetry_error = base._p32_checks(
                direction, alpha, wet, reports, sigma
            )
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
        for index, alpha in enumerate(alphas):
            if entry["by_alpha"][alpha]["status"] != "ran":
                continue
            smaller = alphas[index + 1] if index + 1 < len(alphas) else None
            if smaller is None or entry["by_alpha"][smaller]["status"] != "ran":
                entry["by_alpha"][alpha]["adjacent_alpha_pass"] = True
                continue
            worst = max(
                base._gb_norm(jhat[alpha][lead] - jhat[smaller][lead], wet)
                / max(base._gb_norm(jhat[smaller][lead], wet), floor[lead] / smaller)
                for lead in leads
            )
            entry["by_alpha"][alpha]["adjacent_alpha_score"] = worst
            entry["by_alpha"][alpha]["adjacent_alpha_pass"] = (
                worst <= rule["adjacent_alpha_centred_jvp_relative_max"]
            )
        detail.append(entry)

    chosen = None
    for alpha in alphas:
        all_pass = True
        for entry in detail:
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

    output = {
        "version": "amplitude_pilot_ssh_v2_selection",
        "candidate_alphas_largest_first": alphas,
        "provisional_alpha_ssh": chosen,
        "detail": detail,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"provisional_alpha_ssh": chosen}, indent=2))
    for entry in detail:
        print(f"{entry['regime']} d{entry['day']} kernel={entry['kernel']} long={entry['long']}")
        for alpha in alphas:
            info = entry["by_alpha"][alpha]
            if info["status"] != "ran":
                print(f"  alpha={alpha} status={info['status']}")
                continue
            print(
                f"  alpha={alpha} day10_q_lin={info['per_lead'][10]['q_lin']:.4f} "
                f"day10_pass={info['day10_q_lin_pass']} long_pass={info['long_q_lin_pass']} "
                f"snr_pass={info['q_snr_pass']} p32_pass={info['p32_pass']} adj_pass={info['adjacent_alpha_pass']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
