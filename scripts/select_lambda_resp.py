"""Execution step 12 (continued): applies plan section 14.4's frozen
reject/select rule to the four completed lambda-screen candidates and
freezes the chosen ``lambda_resp`` before full three-seed C training
(execution step 13).

Rule, applied to each candidate against arm B's own already-published
checkpoint at the contract's matched step (the "matched lambda-zero
control", loaded from each
candidate's own recorded copy of it -- all four must agree, since it is the
same fixed comparator):

1. reject if any 10-90-day primary-field AUC is > 1.05x B's matched value;
2. reject if the worst 90-360-day AUC/climatology ratio is > 1.05x B's
   (section 14.4 criterion 2b, amended 2026-08-28: section 16.3's existing
   forward-preservation criterion applied at the screen rather than only
   after training, so the screen cannot select a weight that is already
   known to fail step 14, where no reselection is permitted);
3. reject if growth is > B's matched growth + 0.005 per call, or any
   rollout is non-finite;
4. among the remaining (forward-feasible) candidates, minimize
   ``S_resp_10:60``;
5. candidates within 2% of the minimum tie in favour of the smaller lambda.

If no candidate is forward-feasible, this stops v1 rather than choosing one
-- section 14.4 is explicit that there is no fallback.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCREEN_CONTRACT_PATH = PROJECT_ROOT / "config" / "forward_response_lambda_screen_v2.json"
RESPONSE_CONTRACT_PATH = PROJECT_ROOT / "config" / "model_c_adjoint_faithful_response_v1.json"
RESPONSE_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "af_fno" / "response" / "forward_response_v1"
B_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "af_fno" / "C" / "model_c_adjoint_faithful_nominal_control_v1"
    / "seed_20260724" / "report.json"
)
PRIMARY_FIELDS = ("surface_speed", "sst", "phihyd_surface")
AUC_MAX_RATIO = 1.05
GROWTH_MAX_ADDITIVE_WORSENING = 0.005
TIE_FRACTION = 0.02
SELECTION_DATE = "2026-08-28"


class LambdaSelectionError(RuntimeError):
    """Raised when the lambda screen cannot be legitimately resolved."""


def _candidate_root(screen: dict[str, Any]) -> Path:
    return RESPONSE_OUTPUT_ROOT / screen.get("output_subdirectory", "lambda_screen")


def _load_candidates(screen: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for lambda_resp in screen["candidate_lambda_resp"]:
        path = _candidate_root(screen) / f"candidate_lambda_{lambda_resp}.json"
        if not path.is_file():
            raise LambdaSelectionError(f"candidate result missing: {path}")
        candidates.append(json.loads(path.read_text()))
    return candidates


def evaluate(contract_path: Path = DEFAULT_SCREEN_CONTRACT_PATH) -> dict[str, Any]:
    contract_path = Path(contract_path)
    screen = json.loads(contract_path.read_text())
    candidates = _load_candidates(screen)

    controls = [c["matched_lambda_zero_control"] for c in candidates]
    if len({json.dumps(ctrl, sort_keys=True) for ctrl in controls}) != 1:
        raise LambdaSelectionError("candidates disagree on the matched lambda-zero control -- B's report changed?")
    control = controls[0]

    long_criterion = screen["forward_feasibility"].get(
        "candidate_to_lambda_zero_long_ratio_to_climatology_max_ratio"
    )
    control_worst_long = None
    if long_criterion is not None:
        b_report = json.loads(B_REPORT_PATH.read_text())
        matched_step = int(control["matched_optimizer_step"])
        b_summary = next(s for s in b_report["validation_summaries"] if s["optimizer_step"] == matched_step)
        control_worst_long = max(b_summary["long_ratio_to_climatology"].values())

    report: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    for candidate in candidates:
        lambda_resp = candidate["lambda_resp"]
        auc = candidate["nominal_validation"]["short_auc_10_90"]
        growth = candidate["nominal_validation"]["perturbation_growth"]["worst_growth_rate_per_call"]
        s_resp = candidate["response_validation"]["S_resp_10_60"]

        auc_ratios = {field: auc[field] / control["short_auc_10_90"][field] for field in PRIMARY_FIELDS}
        auc_ok = all(r <= AUC_MAX_RATIO for r in auc_ratios.values())

        # Criterion 2b. The control's own long-horizon value is not carried in
        # `matched_lambda_zero_control` (v1 had no use for it), so it is read
        # from arm B's published report at the contract's matched step -- the
        # same checkpoint the rest of the comparison already uses.
        long_ratio_ok: bool | None = None
        long_ratio_to_control: float | None = None
        if long_criterion is not None:
            worst = max(candidate["nominal_validation"]["long_ratio_to_climatology"].values())
            long_ratio_to_control = worst / control_worst_long
            long_ratio_ok = long_ratio_to_control <= long_criterion
        growth_ok = (growth is not None) and math.isfinite(growth) and (growth <= control["growth"] + GROWTH_MAX_ADDITIVE_WORSENING)
        finite_ok = math.isfinite(s_resp) and all(math.isfinite(v) for v in auc.values())
        forward_feasible = auc_ok and growth_ok and finite_ok and (long_ratio_ok is not False)

        entry = {
            "lambda_resp": lambda_resp,
            "auc_ratios_to_control": auc_ratios,
            "auc_ok": auc_ok,
            "long_ratio_to_control": long_ratio_to_control,
            "long_ratio_ok": long_ratio_ok,
            "growth": growth,
            "growth_ok": growth_ok,
            "finite_ok": finite_ok,
            "forward_feasible": forward_feasible,
            "S_resp_10_60": s_resp,
        }
        report.append(entry)
        if forward_feasible:
            feasible.append(entry)

    if not feasible:
        result = {
            "status": "no_forward_feasible_candidate",
            "action": "stop_v1",
            "control": control,
            "candidates": report,
        }
        _freeze(contract_path, screen, result, selected=None)
        raise LambdaSelectionError("no lambda candidate is forward-feasible -- v1 stops per section 14.4")

    minimum = min(e["S_resp_10_60"] for e in feasible)
    tied = [e for e in feasible if e["S_resp_10_60"] <= minimum * (1.0 + TIE_FRACTION)]
    selected = min(tied, key=lambda e: e["lambda_resp"])

    result = {
        "status": "selected",
        "control": control,
        "candidates": report,
        "minimum_S_resp_10_60": minimum,
        "tied_within_2_percent": [e["lambda_resp"] for e in tied],
        "selected_lambda_resp": selected["lambda_resp"],
    }
    _freeze(contract_path, screen, result, selected=selected["lambda_resp"])
    return result


def _freeze(
    contract_path: Path, screen: dict[str, Any], result: dict[str, Any], *, selected: float | None
) -> None:
    report_path = _candidate_root(screen) / "lambda_selection_result.json"
    if report_path.is_file():
        raise LambdaSelectionError(f"selection result already frozen: {report_path}")
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if selected is None:
        return

    # The status string carries the contract version (v1 "pending_forward_only_screen",
    # v2 "pending_forward_only_screen_v2"), so match the contract's own value
    # rather than a hard-coded one.
    pending_status = screen["selection_status"]
    screen_text = contract_path.read_text()
    if '"selected_lambda_resp": null' not in screen_text or f'"selection_status": "{pending_status}"' not in screen_text:
        raise LambdaSelectionError("screen contract is not in its expected pre-selection state -- refusing to overwrite")
    screen_text = screen_text.replace('"selected_lambda_resp": null', f'"selected_lambda_resp": {selected}')
    screen_text = screen_text.replace(
        f'"selection_status": "{pending_status}"',
        f'"selection_status": "selected_{SELECTION_DATE}"',
    )
    contract_path.write_text(screen_text)

    response_text = RESPONSE_CONTRACT_PATH.read_text()
    if '"lambda_resp": null' not in response_text:
        raise LambdaSelectionError("response contract's lambda_resp is not null -- refusing to overwrite")
    response_text = response_text.replace('"lambda_resp": null', f'"lambda_resp": {selected}', 1)
    response_text = response_text.replace(
        '"lambda_selection_status": "pending_forward_only_screen"',
        f'"lambda_selection_status": "selected_{SELECTION_DATE}"',
    )
    # Point the study contract at the screen that actually chose lambda.
    response_text = response_text.replace(
        '"lambda_contract": "config/forward_response_lambda_screen_v1.json"',
        f'"lambda_contract": "{contract_path.relative_to(PROJECT_ROOT).as_posix()}"',
    )
    RESPONSE_CONTRACT_PATH.write_text(response_text)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_SCREEN_CONTRACT_PATH)
    args = parser.parse_args()
    result = evaluate(args.contract)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
