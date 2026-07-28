"""Project-facing plots for the immutable Model C rollout-conditioned loss-v3 run."""

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
from ..af_model_c_rollout_conditioned_loss_v3 import ARRAYS_NAME, REPORT_NAME, VERSION
from ..af_model_c_successor_validation import LEAD_DAYS


PLOT_VERSION = "model_c_rollout_conditioned_loss_v3_plots_v1"
CURVE_FIGURE = "model_c_loss_v3_rmse_ratios.png"
CHECKPOINT_FIGURE = "model_c_loss_v3_checkpoint_selection.png"
CSV_NAME = "model_c_loss_v3_rmse_ratios.csv"
SUMMARY_NAME = "rollout_conditioned_loss_v3_summary.json"
MANIFEST_NAME = "figure_manifest.json"
FIELDS = ("sst", "phihyd_surface", "surface_speed", "streamfunction")
FIELD_LABELS = {
    "sst": "SST",
    "phihyd_surface": r"Surface $P/\rho$",
    "surface_speed": "Surface speed",
    "streamfunction": "Barotropic streamfunction",
}


class LossV3PlotError(RuntimeError):
    """Raised when the immutable loss-v3 evidence fails provenance checks."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_loss_v3_evidence(
    report_path: str | Path,
    arrays_path: str | Path,
) -> tuple[dict[str, Any], Mapping[str, np.ndarray]]:
    """Load and verify the complete, sealed loss-v3 evidence."""

    report_path = Path(report_path).resolve()
    arrays_path = Path(arrays_path).resolve()
    report = json.loads(report_path.read_text())
    if (
        report_path.name != REPORT_NAME
        or arrays_path.name != ARRAYS_NAME
        or report.get("status") != "complete"
        or report.get("version") != VERSION
        or report.get("validation_state_opened") is not False
        or report.get("inference_opened") is not False
        or report.get("intermediate_wind_opened") is not False
        or report.get("response_or_adjoint_opened") is not False
        or report.get("save_reload_nine_step_bitwise_exact") is not True
    ):
        raise LossV3PlotError("expected a complete, sealed, reload-exact loss-v3 run")
    content = dict(report)
    expected_content_hash = content.pop("report_content_sha256", None)
    if expected_content_hash != _json_sha256(content):
        raise LossV3PlotError("loss-v3 report content hash changed")
    if _file_sha256(arrays_path) != report.get("arrays_sha256"):
        raise LossV3PlotError("loss-v3 arrays hash changed")
    arrays = np.load(arrays_path)
    if (
        tuple(int(value) for value in arrays["lead_days"]) != LEAD_DAYS
        or arrays["records"].shape != (540, 2)
    ):
        raise LossV3PlotError("loss-v3 evaluation record contract changed")
    return report, arrays


def mean_rmse_curve(
    arrays: Mapping[str, np.ndarray],
    method: str,
    field: str,
) -> np.ndarray:
    """Return the member-mean RMSE curve for one stored method."""

    key = f"{method}__rmse__{field}"
    values = np.asarray(arrays[key], dtype=np.float64)
    if values.shape != (540, len(LEAD_DAYS)):
        raise ValueError(f"invalid curve {key}: {values.shape}")
    return values.mean(axis=0)


def publication_rows(
    arrays: Mapping[str, np.ndarray],
    checkpoint_steps: Sequence[int],
) -> list[dict[str, Any]]:
    """Return portable rows for every plotted field and method."""

    methods = (
        "source_persistence",
        "source_climatology",
        "source_duration_5760",
        *(f"loss_v3_{step}" for step in checkpoint_steps),
    )
    rows = []
    for field in FIELDS:
        persistence = mean_rmse_curve(arrays, "source_persistence", field)
        climatology = mean_rmse_curve(arrays, "source_climatology", field)
        for method in methods:
            curve = mean_rmse_curve(arrays, method, field)
            for index, lead in enumerate(LEAD_DAYS):
                rows.append(
                    {
                        "field": field,
                        "method": method,
                        "lead_days": lead,
                        "mean_rmse": float(curve[index]),
                        "ratio_to_persistence": float(curve[index] / persistence[index]),
                        "ratio_to_climatology": float(curve[index] / climatology[index]),
                    }
                )
    return rows


def _plot_curves(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    checkpoint_steps: Sequence[int],
    selected_step: int,
) -> None:
    leads = np.asarray(LEAD_DAYS)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(checkpoint_steps)))
    for axis, field in zip(axes.flat, FIELDS):
        persistence = mean_rmse_curve(arrays, "source_persistence", field)
        source = mean_rmse_curve(arrays, "source_duration_5760", field)
        axis.plot(
            leads,
            source / persistence,
            color="#777777",
            linewidth=2,
            linestyle="--",
            label="Duration source step 5,760",
        )
        for color, step in zip(colors, checkpoint_steps):
            curve = mean_rmse_curve(arrays, f"loss_v3_{step}", field)
            selected = step == selected_step
            axis.plot(
                leads,
                curve / persistence,
                color=color,
                linewidth=2.8 if selected else 0.9,
                alpha=1.0 if selected else 0.55,
                label=f"Loss v3 step {step:,} (selected)" if selected else None,
            )
        axis.axhline(1.0, color="black", linewidth=1, linestyle=":")
        axis.set_title(FIELD_LABELS[field])
        axis.set_xlabel("Lead (days)")
        axis.set_ylabel("RMSE / persistence RMSE")
        axis.set_xticks(LEAD_DAYS)
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    figure.suptitle(
        "Model C rollout-conditioned loss v3 on the frozen 540-record training gate"
    )
    figure.savefig(output / CURVE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_checkpoint_selection(
    output: Path,
    report: Mapping[str, Any],
    selected_step: int,
) -> None:
    summaries = sorted(report["checkpoint_summary"], key=lambda item: item["fine_tune_step"])
    steps = np.asarray([int(item["fine_tune_step"]) for item in summaries])
    primary = np.asarray(
        [float(item["checkpoint_gate"]["worst_primary_rmse_auc_ratio"]) for item in summaries]
    )
    slow = np.asarray(
        [float(item["checkpoint_gate"]["worst_slow_field_lead_ratio"]) for item in summaries]
    )
    ten_day = np.asarray(
        [
            float(item["ten_day_diagnostic"]["worst_per_regime_group_ratio"])
            for item in summaries
        ]
    )
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    axis.plot(steps, primary, marker="o", label="Worst primary RMSE-AUC ratio")
    axis.plot(steps, slow, marker="s", label="Worst slow-field lead ratio")
    axis.plot(steps, ten_day, marker="^", label="Worst 10-day regime/group ratio")
    axis.axhline(1.0, color="black", linewidth=1, linestyle=":")
    axis.axvline(selected_step, color="#7C3AED", linewidth=1.5, linestyle="--")
    axis.annotate(
        f"diagnostic selection: {selected_step:,}",
        xy=(selected_step, primary[np.where(steps == selected_step)[0][0]]),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=8,
    )
    axis.set_xlabel("Loss-v3 fine-tune step")
    axis.set_ylabel("Error ratio (below 1 passes)")
    axis.set_yscale("log")
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8)
    figure.suptitle("Loss-v3 checkpoint gate: material improvement, no long-gate pass")
    figure.savefig(output / CHECKPOINT_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report["loss_v3_decision"]
    selected_step = int(decision["selected_fine_tune_step"])
    selected = next(
        item for item in report["checkpoint_summary"] if int(item["fine_tune_step"]) == selected_step
    )
    gate = selected["checkpoint_gate"]
    return {
        "status": "complete",
        "classification": decision["classification"],
        "passed": bool(decision["passed"]),
        "selected_fine_tune_step": selected_step,
        "selected_for": decision["selected_for"],
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "worst_primary_rmse_auc_ratio": float(gate["worst_primary_rmse_auc_ratio"]),
        "worst_slow_field_lead_ratio": float(gate["worst_slow_field_lead_ratio"]),
        "day90_ratios": gate["day90_ratios"],
        "ten_day_ratio_to_persistence": selected["ten_day_diagnostic"]["aggregate"][
            "ratio_to_persistence"
        ],
        "projection_exactness": report["projection_exactness"],
        "save_reload_nine_step_bitwise_exact": True,
        "validation_state_opened": False,
        "inference_opened": False,
    }


def generate_loss_v3_outputs(
    report_path: str | Path,
    arrays_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate hash-bound local plots, CSV, and summary without reading held states."""

    report_path = Path(report_path).resolve()
    arrays_path = Path(arrays_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite loss-v3 outputs: {output}")
    report, arrays = load_loss_v3_evidence(report_path, arrays_path)
    checkpoint_steps = [
        int(item["fine_tune_step"]) for item in report["checkpoint_summary"]
    ]
    summary = _summary(report)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    try:
        _plot_curves(
            temporary,
            arrays,
            checkpoint_steps,
            int(summary["selected_fine_tune_step"]),
        )
        _plot_checkpoint_selection(
            temporary,
            report,
            int(summary["selected_fine_tune_step"]),
        )
        rows = publication_rows(arrays, checkpoint_steps)
        with (temporary / CSV_NAME).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary["report_sha256"] = _file_sha256(report_path)
        summary["arrays_sha256"] = _file_sha256(arrays_path)
        summary["summary_content_sha256"] = _json_sha256(summary)
        summary_path = temporary / SUMMARY_NAME
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        figures = {}
        for name in (CURVE_FIGURE, CHECKPOINT_FIGURE):
            figures[name] = {
                "path": str(output / name),
                "sha256": _file_sha256(temporary / name),
            }
        manifest = {
            "version": PLOT_VERSION,
            "status": "complete",
            "source_report": str(report_path),
            "source_report_sha256": _file_sha256(report_path),
            "source_report_content_sha256": report["report_content_sha256"],
            "source_arrays": str(arrays_path),
            "source_arrays_sha256": _file_sha256(arrays_path),
            "lead_days": list(LEAD_DAYS),
            "checkpoint_steps": checkpoint_steps,
            "figures": figures,
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
        (temporary / "README.md").write_text(
            "# Model C rollout-conditioned loss v3\n\n"
            f"Classification: `{summary['classification']}`.\n\n"
            f"The selected diagnostic checkpoint is step "
            f"{summary['selected_fine_tune_step']:,}. The long-rollout gate did not "
            "pass, although the primary and slow-field curves improve materially "
            "relative to the duration source. The run used split 1 only and opened "
            "no validation, inference, response, or adjoint state.\n"
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
    result = generate_loss_v3_outputs(args.report, args.arrays, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
