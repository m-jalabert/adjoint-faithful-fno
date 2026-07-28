"""Project-facing outputs for the immutable Model C checkpoint replay audit."""

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

from ..af_model_c_checkpoint_replay_audit import (
    ARRAYS_NAME,
    AUDIT_VERSION,
    REPORT_NAME,
    SLOW_PRIMARY_FIELDS,
)
from ..af_model_c_overfit import _file_sha256
from ..af_model_c_successor_validation import LEAD_DAYS


PLOT_VERSION = "model_c_checkpoint_replay_audit_plots_v1"
FIGURE_NAME = "model_c_checkpoint_replay_sst_phihyd_rmse.png"
CSV_NAME = "model_c_checkpoint_replay_sst_phihyd_rmse.csv"
SUMMARY_NAME = "checkpoint_replay_summary.json"
MANIFEST_NAME = "figure_manifest.json"
README_NAME = "README.md"
BASELINES = ("persistence", "climatology")
FIELD_LABELS = {
    "sst": r"SST RMSE ($^\circ$C)",
    "phihyd_surface": r"Surface $P/\rho$ RMSE (m$^2$ s$^{-2}$)",
}
BASELINE_LABELS = {
    "persistence": "Persistence",
    "climatology": "Training climatology",
}
BASELINE_STYLES = {
    "persistence": ("#202020", "--"),
    "climatology": ("#B7791F", ":"),
}


class ModelCCheckpointReplayPlotError(RuntimeError):
    """Raised when audit publication provenance is invalid."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_checkpoint_audit_evidence(
    report_path: str | Path,
    arrays_path: str | Path,
) -> tuple[dict[str, Any], Mapping[str, np.ndarray]]:
    """Load and verify a complete, sealed checkpoint audit."""

    report_path = Path(report_path).resolve()
    arrays_path = Path(arrays_path).resolve()
    report = json.loads(report_path.read_text())
    if (
        report_path.name != REPORT_NAME
        or arrays_path.name != ARRAYS_NAME
        or report.get("status") != "complete"
        or report.get("version") != AUDIT_VERSION
        or report.get("validation_state_opened") is not False
        or report.get("inference_opened") is not False
        or report.get("intermediate_wind_opened") is not False
        or report.get("response_or_adjoint_opened") is not False
    ):
        raise ModelCCheckpointReplayPlotError(
            "expected a complete, sealed checkpoint replay audit"
        )
    content = dict(report)
    expected_content_hash = content.pop("report_content_sha256", None)
    if expected_content_hash != _json_sha256(content):
        raise ModelCCheckpointReplayPlotError(
            "checkpoint replay report content hash changed"
        )
    if _file_sha256(arrays_path) != report.get("arrays_sha256"):
        raise ModelCCheckpointReplayPlotError(
            "checkpoint replay arrays changed"
        )
    arrays = np.load(arrays_path)
    if (
        tuple(int(value) for value in arrays["lead_days"]) != LEAD_DAYS
        or arrays["records"].shape != (540, 2)
    ):
        raise ModelCCheckpointReplayPlotError(
            "checkpoint replay record contract changed"
        )
    return report, arrays


def mean_rmse_curve(
    arrays: Mapping[str, np.ndarray],
    method: str,
    field: str,
) -> np.ndarray:
    """Return one member-mean RMSE curve."""

    if method in BASELINES:
        key = f"{method}__rmse__{field}"
    else:
        key = f"step_{int(method)}__rmse__{field}"
    values = np.asarray(arrays[key], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LEAD_DAYS):
        raise ValueError(f"invalid audit curve {key}")
    return values.mean(axis=0)


def ratio_curve(
    arrays: Mapping[str, np.ndarray],
    step: int,
    field: str,
    baseline: str,
) -> np.ndarray:
    """Return a checkpoint's mean RMSE divided by one baseline."""

    return mean_rmse_curve(arrays, str(step), field) / mean_rmse_curve(
        arrays,
        baseline,
        field,
    )


def audit_csv_rows(
    arrays: Mapping[str, np.ndarray],
    checkpoint_steps: Sequence[int],
) -> list[dict[str, Any]]:
    """Build a portable row for every plotted field, method, and lead."""

    rows = []
    for field in SLOW_PRIMARY_FIELDS:
        persistence = mean_rmse_curve(
            arrays,
            "persistence",
            field,
        )
        climatology = mean_rmse_curve(
            arrays,
            "climatology",
            field,
        )
        for method in (*BASELINES, *(str(step) for step in checkpoint_steps)):
            curve = mean_rmse_curve(arrays, method, field)
            label = method if method in BASELINES else f"step_{method}"
            for index, lead in enumerate(LEAD_DAYS):
                rows.append(
                    {
                        "field": field,
                        "method": label,
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


def _plot_audit(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    checkpoint_steps: Sequence[int],
    diagnostic_best: int | None,
) -> None:
    leads = np.asarray(LEAD_DAYS)
    colors = plt.cm.viridis(
        np.linspace(0.08, 0.92, len(checkpoint_steps))
    )
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 7.2),
        sharex=True,
        constrained_layout=True,
    )
    for row, field in enumerate(SLOW_PRIMARY_FIELDS):
        raw_axis, ratio_axis = axes[row]
        for baseline in BASELINES:
            color, linestyle = BASELINE_STYLES[baseline]
            raw_axis.plot(
                leads,
                mean_rmse_curve(arrays, baseline, field),
                color=color,
                linestyle=linestyle,
                linewidth=2.1,
                label=BASELINE_LABELS[baseline],
            )
        for color, step in zip(colors, checkpoint_steps):
            width = 2.4 if step == diagnostic_best else 1.2
            label = (
                f"Step {step} (diagnostic best)"
                if step == diagnostic_best
                else f"Step {step}"
            )
            raw_axis.plot(
                leads,
                mean_rmse_curve(arrays, str(step), field),
                color=color,
                linewidth=width,
                alpha=0.95,
                label=label,
            )
            ratio_axis.plot(
                leads,
                ratio_curve(arrays, step, field, "persistence"),
                color=color,
                linewidth=width,
                alpha=0.95,
                label=f"Step {step} / persistence",
            )
            ratio_axis.plot(
                leads,
                ratio_curve(arrays, step, field, "climatology"),
                color=color,
                linewidth=max(0.9, width - 0.5),
                linestyle=":",
                alpha=0.8,
            )
        raw_axis.set_title(f"{FIELD_LABELS[field]}: absolute curves")
        raw_axis.set_ylabel(FIELD_LABELS[field])
        raw_axis.grid(alpha=0.25)
        ratio_axis.axhline(
            1.0,
            color="black",
            linewidth=1.0,
            linestyle="--",
        )
        ratio_axis.set_title(
            "RMSE ratio (solid / persistence; dotted / climatology)"
        )
        ratio_axis.set_ylabel("Checkpoint RMSE / baseline")
        ratio_axis.set_yscale("log")
        ratio_axis.grid(alpha=0.25, which="both")
        for axis in (raw_axis, ratio_axis):
            axis.set_xlabel("Lead (days)")
            axis.set_xticks(LEAD_DAYS)
    axes[0, 0].legend(ncol=2, loc="upper left")
    figure.suptitle(
        "Model C exact-replay late-checkpoint audit on fixed training chronology"
    )
    figure.savefig(output / FIGURE_NAME, bbox_inches="tight")
    plt.close(figure)


def _write_csv(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        "field",
        "method",
        "lead_days",
        "mean_rmse",
        "ratio_to_persistence",
        "ratio_to_climatology",
    )
    with (output / CSV_NAME).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _lightweight_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report["audit_decision"]
    checkpoints = []
    for item in report["checkpoint_summary"]:
        gate = item["checkpoint_gate"]
        checkpoints.append(
            {
                "optimizer_step": int(item["optimizer_step"]),
                "passed": bool(gate["passed"]),
                "worst_primary_rmse_auc_ratio": float(
                    gate["worst_primary_rmse_auc_ratio"]
                ),
                "worst_slow_field_lead_ratio": float(
                    gate["worst_slow_field_lead_ratio"]
                ),
                "day90_ratios": gate["day90_ratios"],
            }
        )
    return {
        "status": "complete",
        "classification": decision["classification"],
        "selected_optimizer_step": decision.get("selected_optimizer_step"),
        "diagnostic_best_optimizer_step": decision.get(
            "diagnostic_best_optimizer_step"
        ),
        "exact_replay_passed": bool(
            report["exact_replay_verification"]["passed"]
        ),
        "checkpoints": checkpoints,
        "validation_state_opened": False,
        "inference_opened": False,
    }


def _readme(summary: Mapping[str, Any], manifest_hash: str) -> str:
    rows = "\n".join(
        "| {optimizer_step} | {passed} | {worst_primary_rmse_auc_ratio:.3f} | "
        "{worst_slow_field_lead_ratio:.3f} |".format(**item)
        for item in summary["checkpoints"]
    )
    return f"""# Model C exact-replay checkpoint audit

Classification: `{summary["classification"]}`.

Exact replay passed: `{summary["exact_replay_passed"]}`.

| Optimizer step | Full gate | Worst primary RMSE-AUC ratio | Worst SST/P-rho lead ratio |
| ---: | :---: | ---: | ---: |
{rows}

The figure shows mean SST and surface P/rho RMSE for all six replayed late
checkpoints against persistence and training climatology. Solid ratio curves
use persistence; dotted ratio curves use climatology. Values below one win.

The CSV contains every plotted mean and both ratios. The summary and manifest
bind these project-facing files to the immutable scratch report and arrays.
Manifest content SHA-256: `{manifest_hash}`.
"""


def generate_checkpoint_audit_outputs(
    report_path: str | Path,
    arrays_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate immutable project-facing checkpoint-audit outputs."""

    report_path = Path(report_path).resolve()
    arrays_path = Path(arrays_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite checkpoint audit outputs: {output}"
        )
    report, arrays = load_checkpoint_audit_evidence(
        report_path,
        arrays_path,
    )
    checkpoint_steps = [
        int(item["optimizer_step"]) for item in report["checkpoint_summary"]
    ]
    summary = _lightweight_summary(report)
    diagnostic_best = summary.get("diagnostic_best_optimizer_step")
    if diagnostic_best is None:
        diagnostic_best = summary.get("selected_optimizer_step")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    try:
        _plot_audit(
            temporary,
            arrays,
            checkpoint_steps,
            diagnostic_best,
        )
        rows = audit_csv_rows(arrays, checkpoint_steps)
        _write_csv(temporary, rows)
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
            "audit_report": str(report_path),
            "audit_report_sha256": _file_sha256(report_path),
            "audit_report_content_sha256": report[
                "report_content_sha256"
            ],
            "audit_arrays": str(arrays_path),
            "audit_arrays_sha256": _file_sha256(arrays_path),
            "lead_days": list(LEAD_DAYS),
            "checkpoint_steps": checkpoint_steps,
            "summary": {
                "path": str(output / SUMMARY_NAME),
                "sha256": _file_sha256(summary_path),
            },
            "csv": {
                "path": str(output / CSV_NAME),
                "sha256": _file_sha256(temporary / CSV_NAME),
            },
            "figure": {
                "path": str(output / FIGURE_NAME),
                "sha256": _file_sha256(temporary / FIGURE_NAME),
            },
            "validation_state_opened": False,
            "inference_opened": False,
        }
        manifest["manifest_content_sha256"] = _json_sha256(manifest)
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (temporary / README_NAME).write_text(
            _readme(summary, manifest["manifest_content_sha256"])
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
    result = generate_checkpoint_audit_outputs(
        args.report,
        args.arrays,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
