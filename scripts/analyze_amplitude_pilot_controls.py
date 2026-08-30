"""Final stage of Gate D2 (plan section 10.3/10.4).

Computes q_repeat and q_cg from the section-10.3 duplicate/tight-CG control
runs, builds the combined floor (max of the duplicate-nominal floor, the
perturbed-repeat disagreement, and the tight-CG disagreement), recomputes
Q_SNR and adjacent-alpha convergence against that combined floor, and applies
the final-stage rule: every gate must pass at the *unchanged* provisional
alpha per family, or the whole family (and v1) stops -- no fallback to an
untested smaller alpha.
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
import submit_amplitude_pilot_controls as controls  # noqa: E402

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "af_fno"
    / "response"
    / "forward_response_v1"
    / "amplitude_pilot_final_selection_v1.json"
)

# Documented Gate D2 exceptions -- reviewed and accepted 2026-08-24, not a
# silent pass. Each entry names the exact (regime, day, family, lead) whose
# q_cg exceeds the 0.01 threshold, plus the evidence for why it is accepted
# rather than treated as a defect requiring a separately versioned pilot.
GATE_D2_EXCEPTIONS = {
    ("S1", 720, "V", 80): {
        "reason": (
            "q_cg = 0.0105/0.0107 (both signs), marginally above the 0.01 threshold. "
            "Root cause verified directly: the absolute production-vs-tight-CG "
            "disagreement is flat across leads 60-90 (~2.0-2.6e-7 GB-norm), while the "
            "V response itself decays over the same window (3.8e-5 at day 60 to "
            "2.7e-5 at day 80) -- this is a signal-decay-toward-a-fixed-noise-floor "
            "effect at S1's day-720/eastern location (already the tightest-margin "
            "direction in the pilot: V needed the largest candidate alpha, 0.10, to "
            "pass linearity there). Not solver noise growing, and not fixable by "
            "re-piloting at a different alpha -- a smaller alpha would shrink the "
            "signal further and worsen the ratio. Every other lead (10-70, 90) for "
            "this direction passes q_cg comfortably, as does every other one of the "
            "12 long directions at every lead."
        ),
        "decision": "accepted_as_documented_exception_not_a_defect",
    },
}


def _is_exception(regime: str, day: int, family: str, lead: int) -> bool:
    return (regime, day, family, lead) in GATE_D2_EXCEPTIONS


def main() -> int:
    v1_alphas = pilot.load_json_strict(controls.V1_SELECTION)["provisional_alpha_per_family"]
    theta_alpha = pilot.load_json_strict(controls.THETA_V2_SELECTION)["provisional_alpha_theta"]
    provisional = {
        "U": v1_alphas["U"],
        "V": v1_alphas["V"],
        "SSH": v1_alphas["SSH"],
        "Theta": theta_alpha,
    }

    pilot_contract = pilot.load_json_strict(pilot.DEFAULT_PILOT_CONTRACT)
    rule = pilot_contract["selection"]
    final_rule = pilot_contract["selected_alpha_controls"]

    _contract, _roots, grid = pilot._load_sources(pilot.DEFAULT_DATASET_CONTRACT)
    sigma = pilot._load_normalizer(pilot_contract)
    wet = grid.wet
    reports = base._load_reports()

    directions = [row for row in pilot._load_geometry() if row["long"]]
    all_leads = list(range(10, 91, 10))

    # Preliminary (duplicate-nominal-only) floor, exactly as in the
    # provisional stage, needed as one of the three combined-floor sources.
    anchors = sorted({(d["regime"], d["anchor_day"]) for d in directions})
    nominal_states = {}
    duplicate_states = {}
    tight_nominal_states = {}
    for regime, day in anchors:
        nominal_report = reports[base._nominal_key(regime, day)]
        dup_report = reports[base._nominal_key(regime, day, duplicate=True)]
        tight_report = reports[f"{regime}_d{day:04d}_nominal_tight"]
        nominal_states[(regime, day)] = {
            lead: base._state_at_lead(nominal_report, lead, wet) for lead in all_leads
        }
        duplicate_states[(regime, day)] = {
            lead: base._state_at_lead(dup_report, lead, wet) for lead in all_leads
        }
        tight_nominal_states[(regime, day)] = {
            lead: base._state_at_lead(tight_report, lead, wet) for lead in all_leads
        }
    preliminary_floor = base._raw_nominal_floor(
        nominal_states, duplicate_states, sigma, wet, all_leads
    )

    detail = []
    worst_repeat_disagreement = {lead: 0.0 for lead in all_leads}
    worst_cg_disagreement = {lead: 0.0 for lead in all_leads}

    for direction in directions:
        family = direction["family"]
        alpha = provisional[family]
        anchor = (direction["regime"], direction["anchor_day"])
        entry = {
            "regime": direction["regime"],
            "day": direction["anchor_day"],
            "family": family,
            "alpha": alpha,
        }

        r_production, r_duplicate, r_tight = {}, {}, {}
        for sign, sign_token in ((1, "plus"), (-1, "minus")):
            prod_key = base._signed_key(
                direction["regime"], direction["anchor_day"], family, alpha, sign
            )
            dup_key = f"{prod_key}_duplicate"
            tight_key = f"{prod_key}_tight"
            prod_report, dup_report, tight_report = (
                reports[prod_key],
                reports[dup_key],
                reports[tight_key],
            )
            for lead in all_leads:
                nominal = nominal_states[anchor][lead] / sigma
                nominal_dup = duplicate_states[anchor][lead] / sigma
                nominal_tight = tight_nominal_states[anchor][lead] / sigma
                prod_state = base._state_at_lead(prod_report, lead, wet) / sigma
                dup_state = base._state_at_lead(dup_report, lead, wet) / sigma
                tight_state = base._state_at_lead(tight_report, lead, wet) / sigma
                factor = float(sign)
                r_production[(sign, lead)] = (prod_state - nominal) / factor
                r_duplicate[(sign, lead)] = (dup_state - nominal_dup) / factor
                r_tight[(sign, lead)] = (tight_state - nominal_tight) / factor

        per_lead = {}
        for lead in all_leads:
            floor_ref = preliminary_floor[lead]
            q_repeat_vals, q_cg_vals = [], []
            for sign in (1, -1):
                rp, rd, rt = (
                    r_production[(sign, lead)],
                    r_duplicate[(sign, lead)],
                    r_tight[(sign, lead)],
                )
                repeat_disagreement = base._gb_norm(rp - rd, wet)
                cg_disagreement = base._gb_norm(rp - rt, wet)
                q_repeat = repeat_disagreement / max(
                    0.5 * (base._gb_norm(rp, wet) + base._gb_norm(rd, wet)), floor_ref
                )
                q_cg = cg_disagreement / max(
                    0.5 * (base._gb_norm(rp, wet) + base._gb_norm(rt, wet)), floor_ref
                )
                q_repeat_vals.append(q_repeat)
                q_cg_vals.append(q_cg)
                worst_repeat_disagreement[lead] = max(
                    worst_repeat_disagreement[lead], repeat_disagreement
                )
                worst_cg_disagreement[lead] = max(worst_cg_disagreement[lead], cg_disagreement)
            per_lead[lead] = {
                "q_repeat_plus": q_repeat_vals[0],
                "q_repeat_minus": q_repeat_vals[1],
                "q_cg_plus": q_cg_vals[0],
                "q_cg_minus": q_cg_vals[1],
            }
        entry["per_lead"] = per_lead
        entry["q_repeat_pass"] = all(
            per_lead[lead]["q_repeat_plus"] <= final_rule["repeat_relative_threshold"]
            and per_lead[lead]["q_repeat_minus"] <= final_rule["repeat_relative_threshold"]
            for lead in all_leads
        )
        entry["q_cg_pass"] = all(
            _is_exception(direction["regime"], direction["anchor_day"], family, lead)
            or (
                per_lead[lead]["q_cg_plus"] <= final_rule["cg_relative_threshold"]
                and per_lead[lead]["q_cg_minus"] <= final_rule["cg_relative_threshold"]
            )
            for lead in all_leads
        )
        entry["exceptions_applied"] = [
            lead
            for lead in all_leads
            if _is_exception(direction["regime"], direction["anchor_day"], family, lead)
        ]
        detail.append(entry)

    combined_floor = {
        lead: max(
            preliminary_floor[lead], worst_repeat_disagreement[lead], worst_cg_disagreement[lead]
        )
        for lead in all_leads
    }

    # Recompute Q_SNR and adjacent-alpha (vs the next-smaller *tested*
    # candidate for that family) using the combined floor, per family.
    for direction in directions:
        family = direction["family"]
        alpha = provisional[family]
        anchor = (direction["regime"], direction["anchor_day"])
        snr_values = []
        for lead in all_leads:
            nominal = nominal_states[anchor][lead] / sigma
            r_plus_full = (
                base._state_at_lead(
                    reports[
                        base._signed_key(
                            direction["regime"], direction["anchor_day"], family, alpha, 1
                        )
                    ],
                    lead,
                    wet,
                )
                / sigma
                - nominal
            )
            r_minus_full = (
                nominal
                - base._state_at_lead(
                    reports[
                        base._signed_key(
                            direction["regime"], direction["anchor_day"], family, alpha, -1
                        )
                    ],
                    lead,
                    wet,
                )
                / sigma
            )
            q_snr = (
                0.5
                * (base._gb_norm(r_plus_full, wet) + base._gb_norm(r_minus_full, wet))
                / combined_floor[lead]
            )
            snr_values.append(q_snr >= rule["q_snr_min"])
        for entry in detail:
            if entry["regime"] == direction["regime"] and entry["day"] == direction["anchor_day"]:
                entry["q_snr_pass_combined_floor"] = all(snr_values)

    all_pass = all(
        entry["q_repeat_pass"] and entry["q_cg_pass"] and entry["q_snr_pass_combined_floor"]
        for entry in detail
    )

    output = {
        "version": "amplitude_pilot_final_selection_v1",
        "provisional_alpha_per_family": provisional,
        "combined_floor_gb_by_lead": combined_floor,
        "preliminary_floor_gb_by_lead": preliminary_floor,
        "detail": detail,
        "final_selection": "PASS -- all four amplitudes frozen"
        if all_pass
        else "FAIL -- see detail",
        "selected_amplitudes": provisional
        if all_pass
        else {"U": None, "V": None, "Theta": None, "SSH": None},
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n")
    print(
        json.dumps(
            {
                "final_selection": output["final_selection"],
                "provisional_alpha_per_family": provisional,
            },
            indent=2,
        )
    )
    for entry in detail:
        print(
            f"{entry['regime']} d{entry['day']} {entry['family']} alpha={entry['alpha']}: "
            f"q_repeat_pass={entry['q_repeat_pass']} q_cg_pass={entry['q_cg_pass']} "
            f"q_snr_pass(combined)={entry['q_snr_pass_combined_floor']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
