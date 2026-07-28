"""Project-facing figures for the immutable Model C rollout diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..af_forward_complete import FIELD_LABELS
from ..af_model_c_overfit import _file_sha256
from ..af_model_c_rollout_diagnosis import (
    ARRAYS_NAME,
    REPORT_NAME,
    ROLLOUT_DIAGNOSIS_VERSION,
)
from ..af_model_c_successor_validation import (
    EVALUATION_FIELDS,
    LEAD_DAYS,
    curve_auc,
)
from .af_model_c_successor_validation_plots import load_validation_evidence


PLOT_VERSION = "model_c_rollout_diagnosis_plots_v1"
SEEDS = (20260723, 20260724, 20260725)
PRIMARY_FIELDS = ("surface_speed", "sst", "phihyd_surface")
SLOW_FIELDS = ("sst", "phihyd_surface", "ssh")
BASELINES = ("persistence", "climatology")
BASELINE_LABELS = {
    "persistence": "Persistence",
    "climatology": "Training climatology",
}
BASELINE_COLORS = {
    "persistence": "#333333",
    "climatology": "#B7791F",
}
TRAINING_COLOR = "#147D92"
VALIDATION_COLOR = "#B13A3A"
FIGURE_NAMES = (
    "model_c_training_primary_rmse_ratio_vs_persistence.png",
    "model_c_training_vs_validation_slow_field_drift.png",
    "model_c_training_all_field_rmse_auc_ratios.png",
)
SUMMARY_NAME = "diagnosis_summary.json"
MANIFEST_NAME = "figure_manifest.json"
README_NAME = "README.md"


class ModelCRolloutDiagnosisPlotError(RuntimeError):
    """Raised when plotting would violate diagnosis provenance."""


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.grid": False,
        }
    )


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_diagnosis_evidence(
    report_path: str | Path,
    arrays_path: str | Path,
) -> tuple[dict[str, Any], Mapping[str, np.ndarray]]:
    """Load and verify the immutable training-only diagnosis."""

    report_path = Path(report_path).resolve()
    arrays_path = Path(arrays_path).resolve()
    report = json.loads(report_path.read_text())
    interpretation = report.get("diagnostic_interpretation", {})
    if (
        report_path.name != REPORT_NAME
        or arrays_path.name != ARRAYS_NAME
        or report.get("status") != "complete"
        or report.get("version") != ROLLOUT_DIAGNOSIS_VERSION
        or interpretation.get("classification")
        != "training_objective_or_checkpoint_gate_mismatch"
        or report.get("inference_opened") is not False
        or report.get("intermediate_wind_opened") is not False
        or report.get("response_or_adjoint_opened") is not False
    ):
        raise ModelCRolloutDiagnosisPlotError(
            "expected a complete, sealed objective-mismatch diagnosis"
        )
    content = dict(report)
    expected_content_hash = content.pop("report_content_sha256", None)
    if expected_content_hash != _json_sha256(content):
        raise ModelCRolloutDiagnosisPlotError(
            "rollout diagnosis report content hash changed"
        )
    if _file_sha256(arrays_path) != report.get("arrays_sha256"):
        raise ModelCRolloutDiagnosisPlotError(
            "rollout diagnosis arrays changed"
        )
    arrays = np.load(arrays_path)
    if (
        tuple(int(value) for value in arrays["lead_days"]) != LEAD_DAYS
        or arrays["records"].shape != (540, 2)
    ):
        raise ModelCRolloutDiagnosisPlotError(
            "rollout diagnosis record contract changed"
        )
    return report, arrays


def model_curve(
    arrays: Mapping[str, np.ndarray],
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the mean, minimum, and maximum seed-mean RMSE curves."""

    curves = np.stack(
        [
            np.asarray(
                arrays[f"seed_{seed}__rmse__{field}"],
                dtype=np.float64,
            ).mean(axis=0)
            for seed in SEEDS
        ]
    )
    return curves.mean(axis=0), curves.min(axis=0), curves.max(axis=0)


def baseline_curve(
    arrays: Mapping[str, np.ndarray],
    baseline: str,
    field: str,
) -> np.ndarray:
    """Return a member-mean baseline RMSE curve."""

    return np.asarray(
        arrays[f"{baseline}__rmse__{field}"],
        dtype=np.float64,
    ).mean(axis=0)


def field_auc_ratios(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Return mean-seed RMSE-AUC ratios for every field and baseline."""

    result: dict[str, dict[str, float]] = {}
    for field in EVALUATION_FIELDS:
        model_auc = float(
            np.mean(
                [
                    curve_auc(
                        np.asarray(
                            arrays[f"seed_{seed}__rmse__{field}"],
                            dtype=np.float64,
                        )
                    ).mean()
                    for seed in SEEDS
                ]
            )
        )
        result[field] = {}
        for baseline in BASELINES:
            reference_auc = float(
                curve_auc(
                    np.asarray(
                        arrays[f"{baseline}__rmse__{field}"],
                        dtype=np.float64,
                    )
                ).mean()
            )
            result[field][baseline] = model_auc / reference_auc
    return result


def ratio_curve(
    arrays: Mapping[str, np.ndarray],
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return seed-mean model RMSE divided by persistence."""

    model_mean, model_low, model_high = model_curve(arrays, field)
    persistence = baseline_curve(arrays, "persistence", field)
    return (
        model_mean / persistence,
        model_low / persistence,
        model_high / persistence,
    )


def _plot_training_primary(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    leads = np.asarray(LEAD_DAYS)
    figure, axes = plt.subplots(
        1,
        len(PRIMARY_FIELDS),
        figsize=(12.5, 3.7),
        sharex=True,
        constrained_layout=True,
    )
    for axis, field in zip(axes, PRIMARY_FIELDS):
        mean, low, high = ratio_curve(arrays, field)
        axis.plot(leads, mean, color=TRAINING_COLOR, linewidth=2.2)
        axis.fill_between(
            leads,
            low,
            high,
            color=TRAINING_COLOR,
            alpha=0.18,
        )
        axis.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
        axis.set_yscale("log")
        axis.set_ylim(0.05, 8.0)
        axis.set_title(FIELD_LABELS[field])
        axis.set_xlabel("Lead (days)")
        axis.set_ylabel("Model C RMSE / persistence")
        axis.grid(alpha=0.25, which="both")
    figure.suptitle(
        "Training chronology: one-step skill and slow-field rollout drift"
    )
    figure.savefig(
        output / FIGURE_NAMES[0],
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_training_validation_comparison(
    output: Path,
    training_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
) -> None:
    leads = np.asarray(LEAD_DAYS)
    figure, axes = plt.subplots(
        1,
        len(SLOW_FIELDS),
        figsize=(12.5, 3.7),
        sharex=True,
        constrained_layout=True,
    )
    for axis, field in zip(axes, SLOW_FIELDS):
        for arrays, color, label in (
            (training_arrays, TRAINING_COLOR, "Training chronology"),
            (validation_arrays, VALIDATION_COLOR, "Fresh validation"),
        ):
            mean, low, high = ratio_curve(arrays, field)
            axis.plot(leads, mean, color=color, linewidth=2.1, label=label)
            axis.fill_between(
                leads,
                low,
                high,
                color=color,
                alpha=0.14,
            )
        axis.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
        axis.set_yscale("log")
        axis.set_ylim(0.2, 8.0)
        axis.set_title(FIELD_LABELS[field])
        axis.set_xlabel("Lead (days)")
        axis.set_ylabel("Model C RMSE / persistence")
        axis.grid(alpha=0.25, which="both")
    axes[0].legend(loc="best")
    figure.suptitle(
        "Slow-field drift is reproduced on training and validation chronology"
    )
    figure.savefig(
        output / FIGURE_NAMES[1],
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_all_field_auc(
    output: Path,
    ratios: Mapping[str, Mapping[str, float]],
) -> None:
    fields = list(EVALUATION_FIELDS)
    positions = np.arange(len(fields))
    width = 0.34
    figure, axis = plt.subplots(figsize=(13.5, 4.8), constrained_layout=True)
    for index, baseline in enumerate(BASELINES):
        axis.bar(
            positions + (index - 0.5) * width,
            [ratios[field][baseline] for field in fields],
            width,
            label=f"/ {BASELINE_LABELS[baseline]}",
            color=BASELINE_COLORS[baseline],
            alpha=0.9,
        )
    axis.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
    axis.set_yscale("log")
    axis.set_ylim(0.04, 10.0)
    axis.set_ylabel("Model C RMSE-AUC / baseline")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [FIELD_LABELS[field].replace(" ", "\n", 1) for field in fields]
    )
    axis.grid(alpha=0.25, axis="y", which="both")
    axis.legend(ncol=2)
    axis.set_title("Training chronology 10–90-day skill across all fields")
    figure.savefig(
        output / FIGURE_NAMES[2],
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    interpretation = report["diagnostic_interpretation"]
    result: dict[str, Any] = {
        "status": "complete",
        "classification": interpretation["classification"],
        "every_seed_reproduces_drift": interpretation[
            "every_seed_reproduces_good_one_step_then_failed_slow_rollout"
        ],
        "slow_field_mean_ratios_to_persistence": {},
        "inference_opened": False,
    }
    for field in SLOW_FIELDS:
        result["slow_field_mean_ratios_to_persistence"][field] = {
            "day10": float(
                np.mean(
                    interpretation["one_step_ratio_to_persistence"][field]
                )
            ),
            "rmse_auc_10_90_day": float(
                np.mean(
                    interpretation["rmse_auc_ratio_to_persistence"][field]
                )
            ),
            "day90": float(
                np.mean(
                    interpretation["day90_ratio_to_persistence"][field]
                )
            ),
        }
    return result


def _readme(summary: Mapping[str, Any], manifest_hash: str) -> str:
    ratios = summary["slow_field_mean_ratios_to_persistence"]
    return f"""# Model C training-only rollout diagnosis

Status: operationally complete. The predeclared classification is
`training_objective_or_checkpoint_gate_mismatch`.

Every seed reproduces the fresh-validation pattern on fixed split-1 chronology:
the slow fields beat persistence at day 10, but lose over the 10--90-day curve
and at day 90.

| Field | Day 10 | 10--90-day RMSE-AUC | Day 90 |
| --- | ---: | ---: | ---: |
| SST | {ratios["sst"]["day10"]:.3f} | {ratios["sst"]["rmse_auc_10_90_day"]:.3f} | {ratios["sst"]["day90"]:.3f} |
| Surface PHIHYD | {ratios["phihyd_surface"]["day10"]:.3f} | {ratios["phihyd_surface"]["rmse_auc_10_90_day"]:.3f} | {ratios["phihyd_surface"]["day90"]:.3f} |
| SSH | {ratios["ssh"]["day10"]:.3f} | {ratios["ssh"]["rmse_auc_10_90_day"]:.3f} | {ratios["ssh"]["day90"]:.3f} |

Values are Model C RMSE divided by persistence; lower than one is better.
The result supports revising long-rollout supervision/checkpoint selection
before attributing the failure to inadequate data or opening inference.

`diagnosis_summary.json` is the lightweight numerical evidence. Three figures
show training curves, training-versus-validation slow-field drift, and all-field
RMSE-AUC ratios. `figure_manifest.json` binds them to the immutable report and
array hashes; its content SHA-256 is `{manifest_hash}`.
"""


def generate_rollout_diagnosis_figures(
    report_path: str | Path,
    arrays_path: str | Path,
    validation_report_path: str | Path,
    validation_arrays_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate immutable diagnosis summaries and figures."""

    report_path = Path(report_path).resolve()
    arrays_path = Path(arrays_path).resolve()
    validation_report_path = Path(validation_report_path).resolve()
    validation_arrays_path = Path(validation_arrays_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite rollout diagnosis figures: {output}"
        )
    report, arrays = load_diagnosis_evidence(report_path, arrays_path)
    validation_report, validation_arrays = load_validation_evidence(
        validation_report_path,
        validation_arrays_path,
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    try:
        _style()
        _plot_training_primary(temporary, arrays)
        _plot_training_validation_comparison(
            temporary,
            arrays,
            validation_arrays,
        )
        ratios = field_auc_ratios(arrays)
        _plot_all_field_auc(temporary, ratios)
        summary = _summary(report)
        summary["report_sha256"] = _file_sha256(report_path)
        summary["arrays_sha256"] = _file_sha256(arrays_path)
        summary["all_field_rmse_auc_ratios"] = ratios
        summary["summary_content_sha256"] = _json_sha256(summary)
        summary_path = temporary / SUMMARY_NAME
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "version": PLOT_VERSION,
            "status": "complete",
            "diagnosis_report": str(report_path),
            "diagnosis_report_sha256": _file_sha256(report_path),
            "diagnosis_report_content_sha256": report[
                "report_content_sha256"
            ],
            "diagnosis_arrays": str(arrays_path),
            "diagnosis_arrays_sha256": _file_sha256(arrays_path),
            "validation_report": str(validation_report_path),
            "validation_report_sha256": _file_sha256(
                validation_report_path
            ),
            "validation_report_content_sha256": validation_report[
                "report_content_sha256"
            ],
            "validation_arrays": str(validation_arrays_path),
            "validation_arrays_sha256": _file_sha256(
                validation_arrays_path
            ),
            "record_count": int(arrays["records"].shape[0]),
            "lead_days": [int(value) for value in arrays["lead_days"]],
            "seeds": list(SEEDS),
            "summary": str(output / SUMMARY_NAME),
            "summary_sha256": _file_sha256(summary_path),
            "figures": {
                name: {
                    "path": str(output / name),
                    "sha256": _file_sha256(temporary / name),
                }
                for name in FIGURE_NAMES
            },
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
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--validation-arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = generate_rollout_diagnosis_figures(
        args.report,
        args.arrays,
        args.validation_report,
        args.validation_arrays,
        args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
