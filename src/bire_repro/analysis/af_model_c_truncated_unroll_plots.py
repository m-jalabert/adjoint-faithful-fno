"""Publish the immutable Model C truncated-unroll training-only result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..af_model_c_overfit import _file_sha256
from ..af_model_c_successor_validation import LEAD_DAYS
from ..af_model_c_truncated_unroll_objective import (
    ARRAYS_NAME,
    REPORT_NAME,
    TRUNCATED_VERSION,
)


PLOT_VERSION = "model_c_truncated_unroll_plots_v1"
FIGURE_NAME = "model_c_truncated_unroll_sst_phihyd_rmse.png"
CSV_NAME = "model_c_truncated_unroll_sst_phihyd_rmse.csv"
SUMMARY_NAME = "truncated_unroll_summary.json"
MANIFEST_NAME = "figure_manifest.json"
README_NAME = "README.md"
FIELDS = ("sst", "phihyd_surface")
FIELD_LABELS = {
    "sst": r"SST RMSE ($^\circ$C)",
    "phihyd_surface": r"Surface $P/\rho$ RMSE (m$^2$ s$^{-2}$)",
}


class TruncatedUnrollPlotError(RuntimeError):
    """Raised when publication provenance is invalid."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_evidence(
    report_path: str | Path,
    arrays_path: str | Path,
) -> tuple[dict[str, Any], Mapping[str, np.ndarray]]:
    """Load and verify one complete sealed truncated-unroll result."""

    report_path = Path(report_path).resolve()
    arrays_path = Path(arrays_path).resolve()
    report = json.loads(report_path.read_text())
    if (
        report_path.name != REPORT_NAME
        or arrays_path.name != ARRAYS_NAME
        or report.get("status") != "complete"
        or report.get("version") != TRUNCATED_VERSION
        or report.get("validation_state_opened") is not False
        or report.get("inference_opened") is not False
        or report.get("intermediate_wind_opened") is not False
        or report.get("response_or_adjoint_opened") is not False
        or report.get("save_reload_nine_step_bitwise_exact") is not True
    ):
        raise TruncatedUnrollPlotError(
            "expected a complete sealed reload-exact truncated result"
        )
    content = dict(report)
    expected_hash = content.pop("report_content_sha256", None)
    if expected_hash != _json_sha256(content):
        raise TruncatedUnrollPlotError("report content hash changed")
    if _file_sha256(arrays_path) != report.get("arrays_sha256"):
        raise TruncatedUnrollPlotError("truncated arrays changed")
    arrays = np.load(arrays_path)
    if (
        tuple(int(value) for value in arrays["lead_days"]) != LEAD_DAYS
        or arrays["records"].shape != (540, 2)
    ):
        raise TruncatedUnrollPlotError("evaluation records changed")
    return report, arrays


def mean_rmse_curve(
    arrays: Mapping[str, np.ndarray],
    method: str,
    field: str,
) -> np.ndarray:
    """Return the member-mean RMSE curve for one method."""

    if method in ("persistence", "climatology"):
        prefix = f"source_{method}"
    elif method == "duration_5760":
        prefix = "source_duration_5760"
    else:
        prefix = f"truncated_{int(method)}"
    key = f"{prefix}__rmse__{field}"
    values = np.asarray(arrays[key], dtype=np.float64)
    if values.shape != (540, len(LEAD_DAYS)):
        raise ValueError(f"invalid truncated curve {key}")
    return values.mean(axis=0)


def publication_rows(
    arrays: Mapping[str, np.ndarray],
    steps: Sequence[int],
) -> list[dict[str, Any]]:
    """Return every plotted value as portable rows."""

    rows = []
    methods = (
        "persistence",
        "climatology",
        "duration_5760",
        *(str(step) for step in steps),
    )
    for field in FIELDS:
        persistence = mean_rmse_curve(arrays, "persistence", field)
        climatology = mean_rmse_curve(arrays, "climatology", field)
        for method in methods:
            curve = mean_rmse_curve(arrays, method, field)
            published = (
                method
                if not method.isdigit()
                else f"truncated_{method}"
            )
            for index, lead in enumerate(LEAD_DAYS):
                rows.append(
                    {
                        "field": field,
                        "method": published,
                        "lead_days": lead,
                        "mean_rmse": float(curve[index]),
                        "ratio_to_persistence": float(
                            curve[index] / persistence[index]
                        ),
                        "ratio_to_climatology": float(
                            curve[index] / climatology[index]
                        ),
                    }
                )
    return rows


def _plot(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    steps: Sequence[int],
    selected_step: int,
) -> None:
    leads = np.asarray(LEAD_DAYS)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(steps)))
    figure, axes = plt.subplots(
        2, 2, figsize=(11.8, 7.4), sharex=True, constrained_layout=True
    )
    for row, field in enumerate(FIELDS):
        raw_axis, ratio_axis = axes[row]
        persistence = mean_rmse_curve(arrays, "persistence", field)
        climatology = mean_rmse_curve(arrays, "climatology", field)
        duration = mean_rmse_curve(arrays, "duration_5760", field)
        raw_axis.plot(
            leads,
            persistence,
            color="#202020",
            linestyle="--",
            linewidth=2.0,
            label="Persistence",
        )
        raw_axis.plot(
            leads,
            climatology,
            color="#B7791F",
            linestyle=":",
            linewidth=2.0,
            label="Training climatology",
        )
        raw_axis.plot(
            leads,
            duration,
            color="#777777",
            linewidth=1.7,
            label="Duration source step 5,760",
        )
        ratio_axis.plot(
            leads, duration / persistence, color="#777777", linewidth=1.4
        )
        ratio_axis.plot(
            leads,
            duration / climatology,
            color="#777777",
            linewidth=1.1,
            linestyle=":",
        )
        for color, step in zip(colors, steps):
            curve = mean_rmse_curve(arrays, str(step), field)
            width = 2.8 if step == selected_step else 1.1
            label = (
                f"Truncated {step} (diagnostic best)"
                if step == selected_step
                else f"Truncated {step}"
            )
            raw_axis.plot(
                leads,
                curve,
                color=color,
                linewidth=width,
                label=label,
            )
            ratio_axis.plot(
                leads,
                curve / persistence,
                color=color,
                linewidth=width,
            )
            ratio_axis.plot(
                leads,
                curve / climatology,
                color=color,
                linewidth=max(0.8, width - 0.7),
                linestyle=":",
            )
        ratio_axis.axhline(
            1.0, color="black", linestyle="--", linewidth=1.0
        )
        raw_axis.set_title(f"{FIELD_LABELS[field]}: absolute curves")
        raw_axis.set_ylabel(FIELD_LABELS[field])
        ratio_axis.set_title(
            "RMSE ratio (solid / persistence; dotted / climatology)"
        )
        ratio_axis.set_ylabel("Model RMSE / baseline")
        ratio_axis.set_yscale("log")
        for axis in (raw_axis, ratio_axis):
            axis.set_xlabel("Lead (days)")
            axis.set_xticks(LEAD_DAYS)
            axis.grid(alpha=0.25, which="both")
    axes[0, 0].legend(ncol=2, fontsize=6.7, loc="upper left")
    figure.suptitle(
        "Model C three-call truncated-unroll diagnosis "
        "(fixed training chronology)"
    )
    figure.savefig(output / FIGURE_NAME, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    checkpoints = []
    for item in report["checkpoint_summary"]:
        gate = item["checkpoint_gate"]
        checkpoints.append(
            {
                "total_fine_tune_step": int(item["fine_tune_step"]),
                "truncated_fine_tune_step": int(
                    item["truncated_fine_tune_step"]
                ),
                "passed": bool(gate["passed"]),
                "worst_primary_rmse_auc_ratio": float(
                    gate["worst_primary_rmse_auc_ratio"]
                ),
                "worst_slow_field_lead_ratio": float(
                    gate["worst_slow_field_lead_ratio"]
                ),
                "ten_day_worst_regime_group_ratio": float(
                    item["ten_day_diagnostic"][
                        "worst_per_regime_group_ratio"
                    ]
                ),
                "day90_ratios": gate["day90_ratios"],
            }
        )
    decision = report["truncated_decision"]
    return {
        "status": "complete",
        "classification": decision["classification"],
        "passed": bool(decision["passed"]),
        "selected_total_fine_tune_step": int(
            decision["selected_total_fine_tune_step"]
        ),
        "selected_truncated_fine_tune_step": int(
            decision["selected_truncated_fine_tune_step"]
        ),
        "selected_for": decision["selected_for"],
        "save_reload_nine_step_bitwise_exact": True,
        "checkpoints": checkpoints,
        "validation_state_opened": False,
        "inference_opened": False,
    }


def generate_outputs(
    report_path: str | Path,
    arrays_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate immutable lightweight outputs from the scratch result."""

    report_path = Path(report_path).resolve()
    arrays_path = Path(arrays_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite truncated outputs: {output}"
        )
    report, arrays = load_evidence(report_path, arrays_path)
    steps = [
        int(item["truncated_fine_tune_step"])
        for item in report["checkpoint_summary"]
    ]
    summary = _summary(report)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    try:
        _plot(
            temporary,
            arrays,
            steps,
            int(summary["selected_truncated_fine_tune_step"]),
        )
        rows = publication_rows(arrays, steps)
        columns = (
            "field",
            "method",
            "lead_days",
            "mean_rmse",
            "ratio_to_persistence",
            "ratio_to_climatology",
        )
        with (temporary / CSV_NAME).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        summary["report_sha256"] = _file_sha256(report_path)
        summary["arrays_sha256"] = _file_sha256(arrays_path)
        summary["summary_content_sha256"] = _json_sha256(summary)
        summary_path = temporary / SUMMARY_NAME
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "version": PLOT_VERSION,
            "status": "complete",
            "source_report": str(report_path),
            "source_report_sha256": _file_sha256(report_path),
            "source_report_content_sha256": report[
                "report_content_sha256"
            ],
            "source_arrays": str(arrays_path),
            "source_arrays_sha256": _file_sha256(arrays_path),
            "lead_days": list(LEAD_DAYS),
            "truncated_fine_tune_steps": steps,
            "figure": {
                "path": str(output / FIGURE_NAME),
                "sha256": _file_sha256(temporary / FIGURE_NAME),
            },
            "csv": {
                "path": str(output / CSV_NAME),
                "sha256": _file_sha256(temporary / CSV_NAME),
            },
            "summary": {
                "path": str(output / SUMMARY_NAME),
                "sha256": _file_sha256(summary_path),
            },
            "validation_state_opened": False,
            "inference_opened": False,
        }
        manifest["manifest_content_sha256"] = _json_sha256(manifest)
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (temporary / README_NAME).write_text(
            "# Model C truncated-unroll diagnosis\n\n"
            f"Classification: `{summary['classification']}`.\n\n"
            "The figure and CSV compare duration step 5,760 and all four "
            "three-call truncated-unroll checkpoints with persistence and "
            "training climatology. Solid ratio curves use persistence; "
            "dotted curves use climatology; values below one win.\n\n"
            "This split-1 experiment reloaded its nine-step rollout "
            "bitwise exactly. It does not authorize replication or "
            "validation.\n"
        )
        os.replace(temporary, output)
    except Exception:
        for path in temporary.glob("*"):
            path.unlink()
        temporary.rmdir()
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = generate_outputs(args.report, args.arrays, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
