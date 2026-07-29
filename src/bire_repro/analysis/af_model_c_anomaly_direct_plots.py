"""Project-facing plots for the pointwise-anomaly direct-state Model C."""

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

from ..af_model_c_anomaly_direct import (
    ARRAYS_NAME,
    REPORT_NAME,
    VERSION,
)
from ..af_model_c_bire_figures import (
    FIGURE_3_NAME,
    FIGURE_4_NAME,
    FIELDS as BIRE_FIELDS,
    FIELD_LABELS as BIRE_FIELD_LABELS,
    LEAD_DAYS as BIRE_LEAD_DAYS,
    METHODS as BIRE_METHODS,
    METHOD_COLORS,
    METHOD_LABELS,
    _plot_figure3,
    _plot_figure4,
    _style,
    percentile_curve,
)
from ..af_model_c_overfit import _file_sha256
from ..af_model_c_successor_validation import LEAD_DAYS


PLOT_VERSION = "model_c_anomaly_direct_plots_v1"
TRAINING_FIGURE = "model_c_anomaly_direct_training_rmse_ratios.png"
CHECKPOINT_FIGURE = "model_c_anomaly_direct_checkpoint_selection.png"
NORMALIZATION_FIGURE = "model_c_anomaly_direct_normalization_scales.png"
FULL_RANGE_FIGURE = "model_c_anomaly_direct_rmse_0_200_days_full_scale.png"
CSV_NAME = "model_c_anomaly_direct_rmse_curves.csv"
SUMMARY_NAME = "anomaly_direct_summary.json"
MANIFEST_NAME = "figure_manifest.json"
TRAINING_FIELDS = (
    "surface_speed",
    "sst",
    "phihyd_surface",
    "streamfunction",
)
FIELD_LABELS = {
    **BIRE_FIELD_LABELS,
    "streamfunction": "Barotropic streamfunction",
}


class ModelCAnomalyDirectPlotError(RuntimeError):
    """Raised when anomaly-direct plotting evidence fails provenance checks."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_anomaly_direct_evidence(
    report_path: str | Path,
    arrays_path: str | Path,
) -> tuple[dict[str, Any], Mapping[str, np.ndarray]]:
    """Load and verify the complete anomaly-direct evidence package."""

    report_path = Path(report_path).resolve()
    arrays_path = Path(arrays_path).resolve()
    report = json.loads(report_path.read_text())
    if (
        report_path.name != REPORT_NAME
        or arrays_path.name != ARRAYS_NAME
        or report.get("status") != "complete"
        or report.get("version") != VERSION
        or report.get("save_reload_nine_step_bitwise_exact") is not True
        or report.get("validation_state_opened") is not True
        or report.get("inference_state_opened") is not False
        or report.get("response_or_adjoint_state_opened") is not False
    ):
        raise ModelCAnomalyDirectPlotError(
            "expected complete, reload-exact anomaly-direct evidence"
        )
    content = dict(report)
    expected_content = content.pop("report_content_sha256", None)
    if expected_content != _json_sha256(content):
        raise ModelCAnomalyDirectPlotError(
            "anomaly-direct report content hash changed"
        )
    if _file_sha256(arrays_path) != report.get("arrays_sha256"):
        raise ModelCAnomalyDirectPlotError("anomaly-direct arrays hash changed")
    arrays = np.load(arrays_path)
    if (
        tuple(int(value) for value in arrays["training_lead_days"])
        != LEAD_DAYS
        or tuple(int(value) for value in arrays["lead_days"])
        != BIRE_LEAD_DAYS
        or arrays["training_records"].shape != (540, 2)
        or arrays["start_draw_order"].shape != (15,)
    ):
        raise ModelCAnomalyDirectPlotError(
            "anomaly-direct evaluation arrays changed"
        )
    normalization = report["normalization"]
    hashes = {
        "pointwise_mean": "pointwise_mean_sha256",
        "pointwise_raw_scale": "pointwise_raw_scale_sha256",
        "pointwise_scale": "pointwise_scale_sha256",
        "channel_scale_floor": "channel_floor_sha256",
    }
    for array_name, report_name in hashes.items():
        observed = hashlib.sha256(
            np.ascontiguousarray(
                arrays[array_name],
                dtype=np.float32,
            ).tobytes(order="C")
        ).hexdigest()
        if observed != normalization[report_name]:
            raise ModelCAnomalyDirectPlotError(
                f"anomaly-direct normalizer changed: {array_name}"
            )
    return report, arrays


def _mean_training_curve(
    arrays: Mapping[str, np.ndarray],
    method: str,
    field: str,
) -> np.ndarray:
    key = f"training_{method}__rmse__{field}"
    values = np.asarray(arrays[key], dtype=np.float64)
    if values.shape != (540, len(LEAD_DAYS)):
        raise ValueError(f"unexpected anomaly-direct curve shape: {key}")
    return values.mean(axis=0)


def _plot_training_curves(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    selected_step: int,
) -> None:
    leads = np.asarray(LEAD_DAYS)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.2, 7.4),
        constrained_layout=True,
    )
    for axis, field in zip(axes.flat, TRAINING_FIELDS):
        persistence = _mean_training_curve(arrays, "persistence", field)
        climatology = _mean_training_curve(arrays, "climatology", field)
        model = _mean_training_curve(
            arrays,
            f"step_{selected_step}",
            field,
        )
        axis.plot(
            leads,
            model / persistence,
            color="#7C3AED",
            linewidth=2.3,
            label="Anomaly-direct / persistence",
        )
        axis.plot(
            leads,
            model / climatology,
            color="#0891B2",
            linewidth=1.8,
            label="Anomaly-direct / climatology",
        )
        axis.axhline(1.0, color="black", linestyle=":", linewidth=1)
        axis.set_title(FIELD_LABELS[field])
        axis.set_xlabel("Lead (days)")
        axis.set_ylabel("RMSE ratio")
        axis.set_xticks(LEAD_DAYS)
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "Pointwise-anomaly direct-state Model C on the frozen training gate"
    )
    figure.savefig(output / TRAINING_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_checkpoint_selection(
    output: Path,
    report: Mapping[str, Any],
    selected_step: int,
) -> None:
    summaries = sorted(
        report["checkpoint_summary"],
        key=lambda item: int(item["optimizer_step"]),
    )
    steps = np.asarray([int(item["optimizer_step"]) for item in summaries])
    primary = np.asarray(
        [
            float(item["checkpoint_gate"]["worst_primary_rmse_auc_ratio"])
            for item in summaries
        ]
    )
    slow = np.asarray(
        [
            float(item["checkpoint_gate"]["worst_slow_field_lead_ratio"])
            for item in summaries
        ]
    )
    one_step = np.asarray(
        [
            float(item["ten_day_diagnostic"]["worst_per_regime_group_ratio"])
            for item in summaries
        ]
    )
    figure, axis = plt.subplots(figsize=(8.6, 4.9), constrained_layout=True)
    axis.plot(steps, primary, marker="o", label="Worst primary RMSE-AUC ratio")
    axis.plot(steps, slow, marker="s", label="Worst slow-field lead ratio")
    axis.plot(steps, one_step, marker="^", label="Worst 10-day regime/group ratio")
    axis.axhline(1.0, color="black", linestyle=":", linewidth=1)
    axis.axvline(selected_step, color="#7C3AED", linestyle="--", linewidth=1.4)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Error ratio (below 1 passes)")
    axis.set_yscale("log")
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8)
    axis.set_title(
        f"Training-only checkpoint selection; selected step {selected_step:,}"
    )
    figure.savefig(output / CHECKPOINT_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_normalization(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    wet = np.asarray(arrays["wet_mask"], dtype=bool)
    raw = np.asarray(arrays["pointwise_raw_scale"], dtype=np.float64)
    floor = np.asarray(arrays["channel_scale_floor"], dtype=np.float64)
    fraction = np.asarray(
        arrays["fraction_wet_cells_floored"],
        dtype=np.float64,
    )
    median = np.median(raw[:, wet], axis=1)
    channels = np.arange(46)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(channels, median, marker="o", markersize=3, label="Median raw std")
    axes[0].plot(
        channels,
        floor,
        marker="s",
        markersize=3,
        label="5th-percentile floor",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Physical-unit scale")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend()
    axes[1].bar(channels, 100.0 * fraction, color="#7C3AED")
    axes[1].axhline(5.0, color="black", linestyle=":", linewidth=1)
    axes[1].set_ylabel("Wet cells floored (%)")
    axes[1].set_xlabel("Dynamic channel (U 0–14, V 15–29, T 30–44, SSH 45)")
    axes[1].set_ylim(0.0, max(6.0, 105.0 * float(np.max(fraction))))
    axes[1].grid(alpha=0.2, axis="y")
    for boundary in (14.5, 29.5, 44.5):
        for axis in axes:
            axis.axvline(boundary, color="0.65", linewidth=0.8)
    figure.suptitle("Training-only pointwise temporal scaling and robust floor")
    figure.savefig(output / NORMALIZATION_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_full_range(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    leads = np.asarray(BIRE_LEAD_DAYS)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(6.0, 8.8),
        sharex=True,
        constrained_layout=True,
    )
    for axis, field in zip(axes, BIRE_FIELDS):
        for method in BIRE_METHODS:
            summary = percentile_curve(arrays[f"{method}__rmse__{field}"])
            axis.plot(
                leads,
                summary["mean"],
                color=METHOD_COLORS[method],
                linewidth=1.7,
                label=METHOD_LABELS[method],
            )
            axis.fill_between(
                leads,
                summary["p10"],
                summary["p90"],
                color=METHOD_COLORS[method],
                alpha=0.15,
                linewidth=0,
            )
        axis.set_ylabel(BIRE_FIELD_LABELS[field])
        axis.grid(alpha=0.25)
        axis.set_ylim(bottom=0.0)
    axes[0].set_title("Pointwise-anomaly direct-state Model C (full range)")
    axes[-1].set_xlabel("Time (days)")
    axes[-1].legend(loc="best")
    figure.savefig(output / FULL_RANGE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _publication_rows(
    arrays: Mapping[str, np.ndarray],
    selected_step: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in TRAINING_FIELDS:
        persistence = _mean_training_curve(arrays, "persistence", field)
        climatology = _mean_training_curve(arrays, "climatology", field)
        model = _mean_training_curve(arrays, f"step_{selected_step}", field)
        for index, lead in enumerate(LEAD_DAYS):
            rows.append(
                {
                    "scope": "training_gate",
                    "field": field,
                    "method": "model",
                    "lead_days": lead,
                    "mean_rmse": float(model[index]),
                    "ratio_to_persistence": float(model[index] / persistence[index]),
                    "ratio_to_climatology": float(model[index] / climatology[index]),
                }
            )
    for field in BIRE_FIELDS:
        persistence = np.asarray(
            arrays[f"persistence__rmse__{field}"],
            dtype=np.float64,
        ).mean(axis=0)
        climatology = np.asarray(
            arrays[f"climatology__rmse__{field}"],
            dtype=np.float64,
        ).mean(axis=0)
        for method in BIRE_METHODS:
            values = np.asarray(
                arrays[f"{method}__rmse__{field}"],
                dtype=np.float64,
            ).mean(axis=0)
            for index, lead in enumerate(BIRE_LEAD_DAYS):
                rows.append(
                    {
                        "scope": "fixed_S2_validation_figure",
                        "field": field,
                        "method": method,
                        "lead_days": lead,
                        "mean_rmse": float(values[index]),
                        "ratio_to_persistence": (
                            float(values[index] / persistence[index])
                            if lead > 0
                            else float("nan")
                        ),
                        "ratio_to_climatology": (
                            float(values[index] / climatology[index])
                            if climatology[index] > 0
                            else float("nan")
                        ),
                    }
                )
    return rows


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report["selection_decision"]
    selected = report["selected_training_summary"]
    figure_metrics = report["validation_figure"]["metrics"]
    return {
        "status": "complete",
        "classification": decision["classification"],
        "training_gate_passed": bool(decision["passed"]),
        "selected_optimizer_step": int(decision["selected_fine_tune_step"]),
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "worst_primary_rmse_auc_ratio": float(
            selected["checkpoint_gate"]["worst_primary_rmse_auc_ratio"]
        ),
        "worst_slow_field_lead_ratio": float(
            selected["checkpoint_gate"]["worst_slow_field_lead_ratio"]
        ),
        "ten_day_ratio_to_persistence": selected["ten_day_diagnostic"][
            "aggregate"
        ]["ratio_to_persistence"],
        "day200_fixed_S2_validation": {
            field: {
                method: float(figure_metrics[field][method]["day200_mean"])
                for method in BIRE_METHODS
            }
            for field in BIRE_FIELDS
        },
        "normalization": report["normalization"],
        "save_reload_nine_step_bitwise_exact": True,
        "validation_state_opened": True,
        "inference_state_opened": False,
    }


def generate_anomaly_direct_outputs(
    report_path: str | Path,
    arrays_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate all requested hash-bound anomaly-direct output products."""

    report_path = Path(report_path).resolve()
    arrays_path = Path(arrays_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite anomaly-direct outputs: {output}")
    report, arrays = load_anomaly_direct_evidence(report_path, arrays_path)
    selected_step = int(report["selection_decision"]["selected_fine_tune_step"])
    summary = _summary(report)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    try:
        _style()
        _plot_training_curves(temporary, arrays, selected_step)
        _plot_checkpoint_selection(temporary, report, selected_step)
        _plot_normalization(temporary, arrays)
        _plot_figure3(
            temporary,
            arrays,
            np.asarray(arrays["longitude_deg"], dtype=np.float64),
            np.asarray(arrays["latitude_deg"], dtype=np.float64),
            np.asarray(arrays["wet_mask"], dtype=bool),
        )
        _plot_figure4(temporary, arrays)
        _plot_full_range(temporary, arrays)
        rows = _publication_rows(arrays, selected_step)
        with (temporary / CSV_NAME).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary["report_sha256"] = _file_sha256(report_path)
        summary["arrays_sha256"] = _file_sha256(arrays_path)
        summary["summary_content_sha256"] = _json_sha256(summary)
        summary_path = temporary / SUMMARY_NAME
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        figure_names = (
            TRAINING_FIGURE,
            CHECKPOINT_FIGURE,
            NORMALIZATION_FIGURE,
            FIGURE_3_NAME,
            FIGURE_4_NAME,
            FULL_RANGE_FIGURE,
        )
        manifest = {
            "version": PLOT_VERSION,
            "status": "complete",
            "source_report": str(report_path),
            "source_report_sha256": _file_sha256(report_path),
            "source_report_content_sha256": report["report_content_sha256"],
            "source_arrays": str(arrays_path),
            "source_arrays_sha256": _file_sha256(arrays_path),
            "selected_optimizer_step": selected_step,
            "figures": {
                name: {
                    "path": str(output / name),
                    "sha256": _file_sha256(temporary / name),
                }
                for name in figure_names
            },
            "csv": {
                "path": str(output / CSV_NAME),
                "sha256": _file_sha256(temporary / CSV_NAME),
            },
            "summary": {
                "path": str(output / SUMMARY_NAME),
                "sha256": _file_sha256(summary_path),
            },
            "validation_state_opened": True,
            "inference_state_opened": False,
        }
        manifest["manifest_content_sha256"] = _json_sha256(manifest)
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "README.md").write_text(
            "# Model C pointwise-anomaly direct-state experiment\n\n"
            "This directory contains the training-only selection diagnostics, "
            "pointwise-normalization evidence, the native-grid streamfunction "
            "comparison, and both the requested Bire-axis and full-range "
            "0--200-day RMSE figures for the selected checkpoint. The fixed "
            "15-member S2 validation ensemble was opened only after the "
            "training-only checkpoint rule selected the model. Inference, "
            "response, and adjoint states remained sealed.\n"
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
    result = generate_anomaly_direct_outputs(
        args.report,
        args.arrays,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
