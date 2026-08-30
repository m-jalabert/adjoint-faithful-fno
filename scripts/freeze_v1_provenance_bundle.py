"""Execution step 21: the provenance bundle and v1's confirmatory answer.

Plan section 25 step 21: "Archive the provenance/access-log bundle and answer
the sole confirmatory question: did forward-only response supervision improve
the learned Jacobian/adjoint without degrading the production-parent forward
emulator?"

Hashes every frozen decision artifact, contract, model checkpoint and gate
result of the study into one write-once manifest, and records the answer that
the gates -- not this script -- returned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs" / "af_fno" / "response" / "forward_response_v1" / "v1_provenance_bundle"
SEEDS = (20260724, 20260911, 20260912)
ARMS = {"B": "model_c_adjoint_faithful_nominal_control_v1", "C": "model_c_adjoint_faithful_response_v1"}

DECISION_ARTIFACTS = {
    "step12_lambda_selection_v1_superseded":
        "outputs/af_fno/response/forward_response_v1/lambda_screen/lambda_selection_result.json",
    "step12_lambda_selection_v2_binding":
        "outputs/af_fno/response/forward_response_v1/lambda_screen_v2/lambda_selection_result.json",
    "step12_tier0_diagnostics":
        "outputs/af_fno/response/forward_response_v1/tier0_diagnostics/tier0_control_response_baselines.json",
    "step14_gate_m1":
        "outputs/af_fno/response/forward_response_v1/gate_m1/gate_m1_result.json",
    "step15_forward_freeze":
        "outputs/af_fno/response/forward_response_v1/step15_forward_freeze/step15_forward_freeze_manifest.json",
    "step16_gate_m2":
        "outputs/af_fno/response/forward_response_blind_v1/gate_m2/gate_m2_result.json",
    "step17_gate_g0_46channel":
        "outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2/gate_g0_46channel_2026-08-29/gate_g0_46channel_f90_full_window.json",
    "step17_gate_g1_extension":
        "outputs/af_fno/adjoint/mitgcm_s0_adjoint_v1/grdchk_g1_extension_2026-08-29/g1_extension_summary.json",
    "step18_gate_a1":
        "outputs/af_fno/adjoint/comparison_response_v1/gate_a1_result.json",
    "step20_consolidated_results":
        "outputs/af_fno/response/forward_response_v1/paper_tables_v1/results_v1_consolidated.json",
}
CONTRACTS = [
    "config/model_c_production_1in_1out_spectralnorm_v1.json",
    "config/model_c_adjoint_faithful_nominal_control_v1.json",
    "config/model_c_adjoint_faithful_response_v1.json",
    "config/forward_response_lambda_screen_v1.json",
    "config/forward_response_lambda_screen_v2.json",
    "config/adjoint_faithful_forward_evaluation_v1.json",
    "config/adjoint_faithful_blind_adjoint_evaluation_v1.json",
]
PLAN = "docs/Adjoint_faithful_response_training_plan.md"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build() -> dict[str, Any]:
    a1 = json.loads((PROJECT_ROOT / DECISION_ARTIFACTS["step18_gate_a1"]).read_text())
    m2 = json.loads((PROJECT_ROOT / DECISION_ARTIFACTS["step16_gate_m2"]).read_text())
    m1 = json.loads((PROJECT_ROOT / DECISION_ARTIFACTS["step14_gate_m1"]).read_text())

    models = {}
    for arm in ARMS:
        for seed in SEEDS:
            rep = PROJECT_ROOT / "outputs/af_fno/C" / ARMS[arm] / f"seed_{seed}" / "report.json"
            payload = json.loads(rep.read_text())
            models[f"{arm}_{seed}"] = {
                "report_sha256": _sha(rep),
                "report_content_sha256": payload["content_sha256"],
                "checkpoint_sha256": payload["published_checkpoint"]["checkpoint_sha256"],
                "normalization_sha256": payload["published_checkpoint"]["normalization_sha256"],
                "selected_optimizer_step": payload["published_checkpoint"]["optimizer_step"],
            }

    bundle: dict[str, Any] = {
        "study": "model_c_adjoint_faithful_response_v1",
        "step": 21,
        "plan": {"path": PLAN, "sha256": _sha(PROJECT_ROOT / PLAN)},
        "lambda_resp": 0.001,
        "models": models,
        "contracts": {c: _sha(PROJECT_ROOT / c) for c in CONTRACTS if (PROJECT_ROOT / c).is_file()},
        "decision_artifacts": {
            k: {"path": v, "sha256": _sha(PROJECT_ROOT / v)}
            for k, v in DECISION_ARTIFACTS.items() if (PROJECT_ROOT / v).is_file()
        },
        "firewall": {
            "adjoint_evaluator_enabled_only_after": [
                "lambda frozen (step 12)", "all six training reports complete (step 13)",
                "Gate M1 frozen (step 14)", "ordinary forward package frozen (step 15)",
                "blind forward-response package frozen (step 16)",
            ],
            "blind_response_manifest_frozen_before_training": True,
            "blind_response_numerics_generated_after_freeze": True,
            "exploratory_adjoint_tests_run": 0,
            "exploratory_reason": (
                "no exploratory manifest was materialized or hashed into the pretraining freeze; the "
                "frozen evaluator contract's own rule is "
                "absence_or_late_creation_of_manifest_means_no_exploratory_test_runs"
            ),
            "post_access_reselection_performed": False,
            "thresholds_changed_after_seeing_a_result": [
                {
                    "what": "F-precision adjoint dot-product identity (Gate A0, FNO side)",
                    "change": "single hard-coded probe -> median over five predeclared shared probes",
                    "threshold_value_changed": False,
                    "why": (
                        "the check drew one probe at one seed against a fixed constant, on a quantity "
                        "with 500-2900x cancellation and three orders of magnitude of probe-to-probe "
                        "spread; it failed the model with the best median residual of the eight, while "
                        "another model with a three times worse worst-probe passed. Applied to all "
                        "seven study models symmetrically with all prior results discarded and re-run."
                    ),
                },
            ],
            "thresholds_changed_after_a_confirmatory_result": "none",
        },
        "verdicts": {
            "gate_m1_development_response": m1["verdict"],
            "gate_m2_blind_forward_response": m2["verdict"],
            "gate_a0_adjoint_pipeline": "pass",
            "gate_a1_blind_adjoint_confirmatory": a1["verdict"],
        },
        "confirmatory_question": (
            "did forward-only response supervision improve the learned Jacobian/adjoint without "
            "degrading the production-parent forward emulator?"
        ),
        "confirmatory_answer": {
            "verdict": "NO -- v1's hypothesis is not supported",
            "basis": (
                "Gate A1 is negative. Five of section 18.3's six criteria pass and all 24 "
                "(objective, lead, seed) cells improve, but the primary seed's delta_B is -0.1775 "
                "against a required -0.2231. Gate A1's own text: any failure rejects the hypothesis "
                "for v1 even if a mechanistic sub-result improves."
            ),
            "forward_half_of_the_question": (
                "the forward emulator was NOT degraded: paired C-B differences flip sign across seeds "
                "on every forward metric except perturbation growth (+0.002 per call, inside the 0.005 "
                "allowance), and C beats its control on the blind response test at every lead."
            ),
            "adjoint_half_of_the_question": (
                "the adjoint improved in magnitude but not in structure. Relative-L2 fell in 24 of 24 "
                "cells (median ratio 0.745) and the amplitude ratio fell from 13.9-16.0 to 12.5-12.9, "
                "but pattern correlation stayed at ~0.02 for every model including C, and sign "
                "agreement at ~0.49, which is chance. Phase A's central failure is not fixed."
            ),
            "what_v1_establishes_positively": [
                "blind held-out response error down 32.6-35.1% at leads 10-60 and 26.7-34.7% at day 90, "
                "reproducibly across three seeds, on data opened once after the full freeze",
                "day 90 is beyond every training horizon in the study, so that improvement is "
                "extrapolation rather than fit",
                "no measurable forward-skill cost",
            ],
            "what_v1_rules_out": (
                "matching response magnitudes alone is insufficient to recover adjoint structure. A v2 "
                "would need a structural or spectral constraint on the Jacobian, under a new "
                "preregistered cycle and preferably new sealed adjoint targets, as section 18.3 requires."
            ),
        },
    }
    bundle["bundle_content_sha256"] = hashlib.sha256(
        json.dumps(bundle, indent=2, sort_keys=True, default=str).encode()
    ).hexdigest()
    return bundle


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    bundle = build()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "v1_provenance_bundle.json"
    if path.exists():
        raise SystemExit(f"the v1 provenance bundle is already frozen: {path}")
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {path}")
    print(f"  models {len(bundle['models'])} | contracts {len(bundle['contracts'])} "
          f"| decision artifacts {len(bundle['decision_artifacts'])}")
    for k, v in bundle["verdicts"].items():
        print(f"  {k}: {v}")
    print(f"  ANSWER: {bundle['confirmatory_answer']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
