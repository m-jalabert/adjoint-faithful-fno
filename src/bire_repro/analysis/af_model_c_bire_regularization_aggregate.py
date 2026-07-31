"""Source-locked cross-arm aggregate for the Model C Bire controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


VERSION = "model_c_anomaly_direct_bire_regularization_aggregate_v1"
ARM_IDS = ("layernorm", "dropout", "layernorm_dropout")
REPORT_NAME = "bire_regularization_aggregate_report.json"
TABLE_NAME = "bire_regularization_aggregate.csv"
FIGURE_NAME = "model_c_bire_regularization_cross_arm.png"
README_NAME = "README.md"
MANIFEST_NAME = "manifest.json"


class BireRegularizationAggregateError(RuntimeError):
    """Raised when an aggregate source or frozen decision changes."""


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _selected_summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    step = int(report["selection_decision"]["selected_optimizer_step"])
    matches = [
        value
        for value in report["evaluation_summaries"]
        if int(value["optimizer_step"]) == step
    ]
    if len(matches) != 1:
        raise BireRegularizationAggregateError(
            "arm report does not contain exactly one selected summary"
        )
    return matches[0]


def select_cross_arm(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen pass-first cross-arm selection without later data."""

    if tuple(report["arm"]["arm_id"] for report in reports) != ARM_IDS:
        raise ValueError("cross-arm reports must follow the frozen arm order")
    records = []
    for report in reports:
        if report.get("status") != "complete":
            raise BireRegularizationAggregateError("arm report is incomplete")
        summary = _selected_summary(report)
        arm_passed = bool(
            report["selection_decision"]["arm_training_gate_passed"]
        )
        if arm_passed != bool(summary["gate"]["pass"]):
            raise BireRegularizationAggregateError(
                "arm report and selected checkpoint gate disagree"
            )
        records.append(
            {
                "arm_id": report["arm"]["arm_id"],
                "optimizer_step": int(summary["optimizer_step"]),
                "arm_training_gate_passed": arm_passed,
                "worst_mid_bottom_modewise_ratio_all_leads": float(
                    summary["worst_mid_bottom_modewise_ratio_all_leads"]
                ),
                "worst_primary_relative_to_source": float(
                    summary["worst_primary_relative_to_source"]
                ),
            }
        )
    passing = [record for record in records if record["arm_training_gate_passed"]]
    pool = passing if passing else records
    best = min(
        pool,
        key=lambda value: (
            value["worst_mid_bottom_modewise_ratio_all_leads"],
            value["worst_primary_relative_to_source"],
            value["optimizer_step"],
            ARM_IDS.index(value["arm_id"]),
        ),
    )
    if passing:
        return {
            "status": "regularization_arm_selected",
            "at_least_one_arm_passed": True,
            "selected_arm": best["arm_id"],
            "selected_optimizer_step": best["optimizer_step"],
            "best_diagnostic_arm": best["arm_id"],
            "best_diagnostic_optimizer_step": best["optimizer_step"],
            "retain_original_model": False,
            "next_action": (
                "freeze_selected_arm_replication_before_any_later_archive_read"
            ),
        }
    return {
        "status": "no_regularization_arm_passed",
        "at_least_one_arm_passed": False,
        "selected_arm": None,
        "selected_optimizer_step": None,
        "best_diagnostic_arm": best["arm_id"],
        "best_diagnostic_optimizer_step": best["optimizer_step"],
        "retain_original_model": True,
        "next_action": "freeze_three_layer_no_padding_training_only_control",
    }


def _load_contract(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if (
        contract.get("version") != VERSION
        or tuple(contract.get("arm_order", ())) != ARM_IDS
        or contract.get("read_contract", {}).get("training_state") is not True
        or any(
            contract.get("read_contract", {}).get(name) is not False
            for name in (
                "validation_state",
                "inference_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
                "long_term_state",
            )
        )
    ):
        raise ValueError("Bire regularization aggregate contract changed")
    root = resolved.parents[1]
    for relative, expected in contract.get("source_hashes", {}).items():
        source = root / relative
        if not source.is_file() or _file_sha256(source) != expected:
            raise BireRegularizationAggregateError(
                f"aggregate implementation source changed: {source}"
            )
    return contract, resolved, _file_sha256(resolved)


def _load_reports(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    reports = []
    original_contract = contract["factorial_contract"]
    original_path = Path(original_contract["path"]).resolve()
    if (
        not original_path.is_file()
        or _file_sha256(original_path) != original_contract["sha256"]
    ):
        raise BireRegularizationAggregateError("factorial contract changed")
    for expected_arm, source in zip(ARM_IDS, contract["arm_sources"], strict=True):
        if source["arm_id"] != expected_arm:
            raise BireRegularizationAggregateError("arm source order changed")
        loaded: dict[str, Any] = {}
        for name in ("report", "arrays", "manifest"):
            record = source[name]
            path = Path(record["path"]).resolve()
            if not path.is_file() or _file_sha256(path) != record["sha256"]:
                raise BireRegularizationAggregateError(
                    f"{expected_arm} {name} source changed"
                )
            if name in ("report", "manifest"):
                loaded[name] = json.loads(path.read_text())
        report = loaded["report"]
        manifest = loaded["manifest"]
        if (
            report["arm"]["arm_id"] != expected_arm
            or report["contract_sha256"] != original_contract["sha256"]
            or report["content_sha256"] != source["report"]["content_sha256"]
            or manifest["arm"] != expected_arm
            or manifest["content_sha256"]
            != source["manifest"]["content_sha256"]
            or manifest["artifacts"]["bire_regularization_control_report.json"]
            != source["report"]["sha256"]
            or manifest["artifacts"]["bire_regularization_control_arrays.npz"]
            != source["arrays"]["sha256"]
            or any(
                report.get(name) is not False
                for name in (
                    "validation_state_opened",
                    "inference_state_opened",
                    "response_or_adjoint_state_opened",
                    "long_term_state_opened",
                )
            )
        ):
            raise BireRegularizationAggregateError(
                f"{expected_arm} report provenance changed"
            )
        reports.append(report)
    return reports


def _cross_arm_rows(reports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        summary = _selected_summary(report)
        rows.append(
            {
                "arm_id": report["arm"]["arm_id"],
                "optimizer_step": int(summary["optimizer_step"]),
                "gate_passed": bool(summary["gate"]["pass"]),
                "day360_mid_modewise_ratio": float(
                    summary["day360_mid_modewise_ratio"]
                ),
                "day360_bottom_modewise_ratio": float(
                    summary["day360_bottom_modewise_ratio"]
                ),
                "worst_mid_bottom_modewise_ratio_all_leads": float(
                    summary["worst_mid_bottom_modewise_ratio_all_leads"]
                ),
                "worst_primary_10_to_90_ratio": float(
                    summary["worst_primary_10_to_90_ratio"]
                ),
                "worst_primary_relative_to_source": float(
                    summary["worst_primary_relative_to_source"]
                ),
                "first_mid_factor_four_failure_day": (
                    summary["first_mid_factor_four_failure_day"]
                ),
                "first_bottom_factor_four_failure_day": (
                    summary["first_bottom_factor_four_failure_day"]
                ),
            }
        )
    return rows


def _plot(
    path: Path,
    reports: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    source = reports[0]["source_summary"]
    labels = ("Source", "LayerNorm", "Dropout", "Both")
    x = np.arange(len(labels))
    mid = [source["day360_mid_modewise_ratio"]] + [
        row["day360_mid_modewise_ratio"] for row in rows
    ]
    bottom = [source["day360_bottom_modewise_ratio"]] + [
        row["day360_bottom_modewise_ratio"] for row in rows
    ]
    relative = [1.0] + [row["worst_primary_relative_to_source"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, mid, width, label="PHIHYD mid")
    axes[0].bar(x + width / 2, bottom, width, label="PHIHYD bottom")
    axes[0].axhline(4.0, color="black", linestyle="--", label="factor-four gate")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Day-360 modewise RMSE ratio")
    axes[0].legend()
    axes[1].bar(x, relative, width=0.62)
    axes[1].axhline(1.1, color="black", linestyle="--", label="10% limit")
    axes[1].set_ylabel("Worst primary ratio relative to source")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Model C Bire regularization controls: frozen cross-arm result")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(contract_path: str | Path) -> dict[str, Any]:
    """Validate all three arms and publish one immutable aggregate."""

    contract, resolved_contract, contract_sha = _load_contract(contract_path)
    reports = _load_reports(contract)
    decision = select_cross_arm(reports)
    rows = _cross_arm_rows(reports)
    output = Path(contract["output"]["directory"]).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError("refusing to overwrite regularization aggregate")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()

    report = {
        "status": "complete",
        "version": VERSION,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "factorial_contract_sha256": contract["factorial_contract"]["sha256"],
        "scheduler_job": contract["scheduler_job"],
        "arm_order": list(ARM_IDS),
        "source_summary": reports[0]["source_summary"],
        "arm_summaries": rows,
        "selection_decision": decision,
        "read_contract": contract["read_contract"],
        "validation_state_opened": False,
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
        "long_term_state_opened": False,
    }
    report["content_sha256"] = _json_sha256(report)
    report_path = temporary / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (temporary / TABLE_NAME).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(temporary / FIGURE_NAME, reports, rows)
    (temporary / README_NAME).write_text(
        "# Model C Bire regularization cross-arm aggregate\n\n"
        f"Decision: `{decision['status']}`. "
        "All comparisons use only the frozen split-1 training-state audit; "
        "later archives remained sealed.\n"
    )
    artifacts = {
        name: _file_sha256(temporary / name)
        for name in (REPORT_NAME, TABLE_NAME, FIGURE_NAME, README_NAME)
    }
    manifest = {
        "status": "complete",
        "version": VERSION,
        "contract_sha256": contract_sha,
        "artifacts": artifacts,
        "content_sha256": _json_sha256(artifacts),
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
        "long_term_state_opened": False,
    }
    (temporary / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, output)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(run(args.contract), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
