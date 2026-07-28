"""Project-facing outputs for the immutable Model C pushforward experiment."""

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
from ..af_model_c_pushforward_objective import (
    ARRAYS_NAME,
    OBJECTIVE_VERSION,
    REPORT_NAME,
)
from ..af_model_c_successor_validation import LEAD_DAYS


PLOT_VERSION = "model_c_pushforward_objective_plots_v1"
FIGURE_NAME = "model_c_pushforward_sst_phihyd_rmse.png"
CSV_NAME = "model_c_pushforward_sst_phihyd_rmse.csv"
SUMMARY_NAME = "pushforward_objective_summary.json"
MANIFEST_NAME = "figure_manifest.json"
README_NAME = "README.md"
FIELDS = ("sst", "phihyd_surface")
BASELINES = ("persistence", "climatology")
FIELD_LABELS = {
    "sst": r"SST RMSE ($^\circ$C)",
    "phihyd_surface": r"Surface $P/\rho$ RMSE (m$^2$ s$^{-2}$)",
}


class ModelCPushforwardPlotError(RuntimeError):
    """Raised when pushforward publication provenance is invalid."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_pushforward_evidence(
    report_path: str | Path,
    arrays_path: str | Path,
) -> tuple[dict[str, Any], Mapping[str, np.ndarray]]:
    """Load and verify one complete, sealed pushforward result."""

    report_path = Path(report_path).resolve()
    arrays_path = Path(arrays_path).resolve()
    report = json.loads(report_path.read_text())
    if (
        report_path.name != REPORT_NAME
        or arrays_path.name != ARRAYS_NAME
        or report.get("status") != "complete"
        or report.get("version") != OBJECTIVE_VERSION
        or report.get("validation_state_opened") is not False
        or report.get("inference_opened") is not False
        or report.get("intermediate_wind_opened") is not False
        or report.get("response_or_adjoint_opened") is not False
    ):
        raise ModelCPushforwardPlotError(
            "expected a complete, sealed pushforward experiment"
        )
    content = dict(report)
    expected_hash = content.pop("report_content_sha256", None)
    if expected_hash != _json_sha256(content):
        raise ModelCPushforwardPlotError(
            "pushforward report content hash changed"
        )
    if _file_sha256(arrays_path) != report.get("arrays_sha256"):
        raise ModelCPushforwardPlotError("pushforward arrays changed")
    arrays = np.load(arrays_path)
    if (
        tuple(int(value) for value in arrays["lead_days"]) != LEAD_DAYS
        or arrays["records"].shape != (540, 2)
    ):
        raise ModelCPushforwardPlotError(
            "pushforward evaluation records changed"
        )
    return report, arrays


def mean_rmse_curve(
    arrays: Mapping[str, np.ndarray],
    method: str,
    field: str,
) -> np.ndarray:
    """Return the member-mean RMSE curve for one published method."""

    if method in BASELINES:
        prefix = f"source_{method}"
    elif method == "source":
        prefix = "source_step_14400"
    else:
        prefix = f"fine_tune_{int(method)}"
    key = f"{prefix}__rmse__{field}"
    values = np.asarray(arrays[key], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LEAD_DAYS):
        raise ValueError(f"invalid pushforward curve {key}")
    return values.mean(axis=0)


def publication_rows(
    arrays: Mapping[str, np.ndarray],
    steps: Sequence[int],
) -> list[dict[str, Any]]:
    """Return every plotted field, method, and lead as portable rows."""

    rows = []
    for field in FIELDS:
        persistence = mean_rmse_curve(
            arrays, "persistence", field
        )
        climatology = mean_rmse_curve(
            arrays, "climatology", field
        )
        for method in (*BASELINES, "source", *(str(step) for step in steps)):
            curve = mean_rmse_curve(arrays, method, field)
            for index, lead in enumerate(LEAD_DAYS):
                rows.append(
                    {
                        "field": field,
                        "method": (
                            method
                            if method in BASELINES
                            else (
                                "source_step_14400"
                                if method == "source"
                                else f"fine_tune_{method}"
                            )
                        ),
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
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(steps)))
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 7.2),
        sharex=True,
        constrained_layout=True,
    )
    for row, field in enumerate(FIELDS):
        raw_axis, ratio_axis = axes[row]
        raw_axis.plot(
            leads,
            mean_rmse_curve(arrays, "persistence", field),
            color="#202020",
            linestyle="--",
            linewidth=2.0,
            label="Persistence",
        )
        raw_axis.plot(
            leads,
            mean_rmse_curve(arrays, "climatology", field),
            color="#B7791F",
            linestyle=":",
            linewidth=2.0,
            label="Training climatology",
        )
        source = mean_rmse_curve(arrays, "source", field)
        raw_axis.plot(
            leads,
            source,
            color="#7A7A7A",
            linewidth=1.5,
            label="Source step 14,400",
        )
        for color, step in zip(colors, steps):
            curve = mean_rmse_curve(arrays, str(step), field)
            width = 2.6 if step == selected_step else 1.2
            label = (
                f"Fine-tune {step} (best)"
                if step == selected_step
                else f"Fine-tune {step}"
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
                curve
                / mean_rmse_curve(arrays, "persistence", field),
                color=color,
                linewidth=width,
            )
            ratio_axis.plot(
                leads,
                curve
                / mean_rmse_curve(arrays, "climatology", field),
                color=color,
                linewidth=max(0.9, width - 0.6),
                linestyle=":",
            )
        ratio_axis.plot(
            leads,
            source / mean_rmse_curve(arrays, "persistence", field),
            color="#7A7A7A",
            linewidth=1.2,
        )
        ratio_axis.plot(
            leads,
            source / mean_rmse_curve(arrays, "climatology", field),
            color="#7A7A7A",
            linewidth=1.0,
            linestyle=":",
        )
        ratio_axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
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
    axes[0, 0].legend(ncol=2, fontsize=7, loc="upper left")
    figure.suptitle(
        "Model C detached pushforward correction on fixed training chronology"
    )
    figure.savefig(output / FIGURE_NAME, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    checkpoints = []
    for item in report["checkpoint_summary"]:
        gate = item["checkpoint_gate"]
        checkpoints.append(
            {
                "fine_tune_step": int(item["fine_tune_step"]),
                "passed": bool(gate["passed"]),
                "ten_day_worst_regime_group_ratio": float(
                    item["ten_day_diagnostic"][
                        "worst_per_regime_group_ratio"
                    ]
                ),
                "worst_primary_rmse_auc_ratio": float(
                    gate["worst_primary_rmse_auc_ratio"]
                ),
                "worst_slow_field_lead_ratio": float(
                    gate["worst_slow_field_lead_ratio"]
                ),
                "day90_ratios": gate["day90_ratios"],
                "training_window": {
                    "total": float(item["training_window"]["total"]),
                    "pushforward_sst": float(
                        item["training_window"]["pushforward_sst"]
                    ),
                    "pushforward_phihyd_surface": float(
                        item["training_window"][
                            "pushforward_phihyd_surface"
                        ]
                    ),
                },
            }
        )
    return {
        "status": "complete",
        "classification": report["objective_decision"]["classification"],
        "passed": bool(report["objective_decision"]["passed"]),
        "selected_fine_tune_step": int(
            report["objective_decision"]["selected_fine_tune_step"]
        ),
        "selected_for": report["objective_decision"]["selected_for"],
        "save_reload_nine_step_bitwise_exact": bool(
            report["save_reload_nine_step_bitwise_exact"]
        ),
        "checkpoints": checkpoints,
        "validation_state_opened": False,
        "inference_opened": False,
    }


def generate_pushforward_outputs(
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
            f"refusing to overwrite pushforward outputs: {output}"
        )
    report, arrays = load_pushforward_evidence(report_path, arrays_path)
    steps = [
        int(item["fine_tune_step"])
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
            int(summary["selected_fine_tune_step"]),
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
            "fine_tune_steps": steps,
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
            "# Model C pushforward objective\n\n"
            f"Classification: `{summary['classification']}`.\n\n"
            "The figure and CSV compare the source and all four fine-tune "
            "checkpoints with persistence and training climatology. Solid "
            "ratio curves use persistence and dotted curves use climatology; "
            "values below one win.\n\n"
            "This experiment used split 1 only and did not authorize "
            "replication or validation.\n"
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
    result = generate_pushforward_outputs(
        args.report,
        args.arrays,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
