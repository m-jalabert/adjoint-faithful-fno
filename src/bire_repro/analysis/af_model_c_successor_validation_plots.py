"""Project-facing figures for the immutable Model C successor validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..af_forward_complete import FIELD_LABELS, FIELD_UNITS
from ..af_model_c_overfit import _file_sha256
from ..af_model_c_successor_validation import (
    EVALUATION_FIELDS,
    LEAD_DAYS,
    PRIMARY_FIELDS,
    VALIDATION_ARRAYS_NAME,
    curve_auc,
)


PLOT_VERSION = "model_c_successor_validation_plots_v1"
SEEDS = (20260723, 20260724, 20260725)
BASELINES = ("persistence", "climatology", "a0")
BASELINE_LABELS = {
    "persistence": "Persistence",
    "climatology": "Training climatology",
    "a0": "Frozen A0",
}
BASELINE_COLORS = {
    "persistence": "#222222",
    "climatology": "#A86600",
    "a0": "#6A51A3",
}
MODEL_COLOR = "#147D92"
REGIME_COLORS = ("#3876B6", "#D17B0F", "#25855A")
FIGURE_NAMES = (
    "model_c_primary_rmse_vs_lead.png",
    "model_c_primary_acc_vs_lead.png",
    "model_c_primary_rmse_ratio_vs_persistence.png",
    "model_c_primary_regime_rmse_ratio.png",
    "model_c_all_field_rmse_auc_ratios.png",
)
MANIFEST_NAME = "figure_manifest.json"


class ModelCValidationPlotError(RuntimeError):
    """Raised when plotting would violate immutable validation provenance."""


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


def load_validation_evidence(
    report_path: str | Path,
    arrays_path: str | Path,
) -> tuple[dict[str, Any], Mapping[str, np.ndarray]]:
    """Load and verify the immutable rejected validation package."""

    report_path = Path(report_path).resolve()
    arrays_path = Path(arrays_path).resolve()
    report = json.loads(report_path.read_text())
    if (
        report.get("status") != "complete"
        or report.get("validation_gate", {}).get("status")
        != "scientifically_rejected_fresh_v2_validation"
        or report.get("inference_opened") is not False
    ):
        raise ModelCValidationPlotError(
            "expected a complete, sealed validation rejection"
        )
    content_digest = report.get("report_content_sha256")
    content = dict(report)
    content.pop("report_content_sha256", None)
    if content_digest != _json_sha256(content):
        raise ModelCValidationPlotError("validation report content hash changed")
    if (
        arrays_path.name != VALIDATION_ARRAYS_NAME
        or _file_sha256(arrays_path) != report.get("arrays_sha256")
    ):
        raise ModelCValidationPlotError("validation arrays changed")
    arrays = np.load(arrays_path)
    if (
        tuple(int(value) for value in arrays["lead_days"]) != LEAD_DAYS
        or arrays["records"].shape != (540, 2)
    ):
        raise ModelCValidationPlotError("validation record contract changed")
    return report, arrays


def mean_model_curve(
    arrays: Mapping[str, np.ndarray],
    *,
    metric: str,
    field: str,
    selected: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three-seed mean, minimum, and maximum member-mean curves."""

    curves = []
    for seed in SEEDS:
        values = np.asarray(
            arrays[f"seed_{seed}__{metric}__{field}"],
            dtype=np.float64,
        )
        if selected is not None:
            values = values[selected]
        curves.append(values.mean(axis=0))
    stacked = np.stack(curves)
    return stacked.mean(axis=0), stacked.min(axis=0), stacked.max(axis=0)


def baseline_curve(
    arrays: Mapping[str, np.ndarray],
    *,
    baseline: str,
    metric: str,
    field: str,
    selected: np.ndarray | None = None,
) -> np.ndarray:
    """Return one baseline's member-mean lead curve."""

    values = np.asarray(
        arrays[f"{baseline}__{metric}__{field}"],
        dtype=np.float64,
    )
    if selected is not None:
        values = values[selected]
    return values.mean(axis=0)


def all_field_auc_ratios(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Return three-seed-mean RMSE-AUC ratios for every reported field."""

    result: dict[str, dict[str, float]] = {}
    for field in EVALUATION_FIELDS:
        model = np.mean(
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
        result[field] = {}
        for baseline in BASELINES:
            reference = curve_auc(
                np.asarray(
                    arrays[f"{baseline}__rmse__{field}"],
                    dtype=np.float64,
                )
            ).mean()
            result[field][baseline] = float(model / reference)
    return result


def _plot_primary_absolute(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    metric: str,
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
        mean, low, high = mean_model_curve(
            arrays,
            metric=metric,
            field=field,
        )
        axis.plot(leads, mean, color=MODEL_COLOR, linewidth=2.2, label="Model C")
        axis.fill_between(
            leads,
            low,
            high,
            color=MODEL_COLOR,
            alpha=0.18,
            label="seed range",
        )
        for baseline in BASELINES:
            axis.plot(
                leads,
                baseline_curve(
                    arrays,
                    baseline=baseline,
                    metric=metric,
                    field=field,
                ),
                color=BASELINE_COLORS[baseline],
                linestyle={"persistence": "--", "climatology": ":", "a0": "-."}[
                    baseline
                ],
                label=BASELINE_LABELS[baseline],
            )
        axis.set_title(FIELD_LABELS[field])
        axis.set_xlabel("Lead (days)")
        axis.grid(alpha=0.25)
        if metric == "rmse":
            axis.set_ylabel(f"RMSE ({FIELD_UNITS[field]})")
        else:
            axis.set_ylabel("Anomaly correlation")
            axis.set_ylim(-0.1, 1.02)
            axis.axhline(0.0, color="0.6", linewidth=0.7)
    axes[0].legend(loc="best")
    figure.suptitle(
        "Model C fresh validation: "
        + ("primary-field error" if metric == "rmse" else "primary-field ACC")
    )
    target = output / (
        "model_c_primary_rmse_vs_lead.png"
        if metric == "rmse"
        else "model_c_primary_acc_vs_lead.png"
    )
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_primary_persistence_ratio(
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
        persistence = baseline_curve(
            arrays,
            baseline="persistence",
            metric="rmse",
            field=field,
        )
        mean, low, high = mean_model_curve(
            arrays,
            metric="rmse",
            field=field,
        )
        axis.plot(leads, mean / persistence, color=MODEL_COLOR, linewidth=2.2)
        axis.fill_between(
            leads,
            low / persistence,
            high / persistence,
            color=MODEL_COLOR,
            alpha=0.18,
        )
        axis.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
        axis.set_yscale("log")
        axis.set_ylim(0.08, 8.0)
        axis.set_title(FIELD_LABELS[field])
        axis.set_xlabel("Lead (days)")
        axis.set_ylabel("Model C RMSE / persistence")
        axis.grid(alpha=0.25, which="both")
    figure.suptitle(
        "Fresh validation: one-step skill and slow-field rollout drift"
    )
    figure.savefig(
        output / "model_c_primary_rmse_ratio_vs_persistence.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_primary_regime_ratio(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    leads = np.asarray(LEAD_DAYS)
    records = np.asarray(arrays["records"], dtype=np.int64)
    figure, axes = plt.subplots(
        len(PRIMARY_FIELDS),
        3,
        figsize=(12.5, 9.0),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    for row, field in enumerate(PRIMARY_FIELDS):
        for experiment in range(3):
            axis = axes[row, experiment]
            selected = records[:, 0] == experiment
            persistence = baseline_curve(
                arrays,
                baseline="persistence",
                metric="rmse",
                field=field,
                selected=selected,
            )
            mean, low, high = mean_model_curve(
                arrays,
                metric="rmse",
                field=field,
                selected=selected,
            )
            axis.plot(
                leads,
                mean / persistence,
                color=REGIME_COLORS[experiment],
                linewidth=2.0,
            )
            axis.fill_between(
                leads,
                low / persistence,
                high / persistence,
                color=REGIME_COLORS[experiment],
                alpha=0.18,
            )
            axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            axis.set_yscale("log")
            axis.set_ylim(0.08, 8.0)
            axis.grid(alpha=0.25, which="both")
            if row == 0:
                axis.set_title(f"S{experiment}")
            if experiment == 0:
                axis.set_ylabel(
                    f"{FIELD_LABELS[field]}\nRMSE / persistence"
                )
            if row == len(PRIMARY_FIELDS) - 1:
                axis.set_xlabel("Lead (days)")
    figure.suptitle("Fresh validation primary skill by wind regime")
    figure.savefig(
        output / "model_c_primary_regime_rmse_ratio.png",
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
    width = 0.24
    figure, axis = plt.subplots(figsize=(13.5, 4.8), constrained_layout=True)
    for index, baseline in enumerate(BASELINES):
        axis.bar(
            positions + (index - 1) * width,
            [ratios[field][baseline] for field in fields],
            width,
            label=f"/ {BASELINE_LABELS[baseline]}",
            color=BASELINE_COLORS[baseline],
            alpha=0.88,
        )
    axis.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
    axis.set_yscale("log")
    axis.set_ylim(0.05, 20.0)
    axis.set_ylabel("Model C RMSE-AUC / baseline")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [FIELD_LABELS[field].replace(" ", "\n", 1) for field in fields],
        rotation=0,
    )
    axis.grid(alpha=0.25, axis="y", which="both")
    axis.legend(ncol=3)
    axis.set_title("Fresh validation 10–90-day skill across all fields")
    figure.savefig(
        output / "model_c_all_field_rmse_auc_ratios.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def generate_validation_figures(
    report_path: str | Path,
    arrays_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate all validation figures and a hash-addressed manifest."""

    report_path = Path(report_path).resolve()
    arrays_path = Path(arrays_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / name for name in (*FIGURE_NAMES, MANIFEST_NAME)]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite validation figures: "
            + ", ".join(str(path) for path in existing)
        )
    report, arrays = load_validation_evidence(report_path, arrays_path)
    _style()
    _plot_primary_absolute(output, arrays, metric="rmse")
    _plot_primary_absolute(output, arrays, metric="acc")
    _plot_primary_persistence_ratio(output, arrays)
    _plot_primary_regime_ratio(output, arrays)
    ratios = all_field_auc_ratios(arrays)
    _plot_all_field_auc(output, ratios)
    manifest = {
        "version": PLOT_VERSION,
        "status": "complete",
        "validation_report": str(report_path),
        "validation_report_sha256": _file_sha256(report_path),
        "validation_report_content_sha256": report[
            "report_content_sha256"
        ],
        "validation_arrays": str(arrays_path),
        "validation_arrays_sha256": _file_sha256(arrays_path),
        "record_count": int(arrays["records"].shape[0]),
        "lead_days": [int(value) for value in arrays["lead_days"]],
        "seeds": list(SEEDS),
        "all_field_rmse_auc_ratios": ratios,
        "figures": {
            name: {
                "path": str(output / name),
                "sha256": _file_sha256(output / name),
            }
            for name in FIGURE_NAMES
        },
        "inference_opened": False,
    }
    manifest["manifest_content_sha256"] = _json_sha256(manifest)
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = generate_validation_figures(
        args.report,
        args.arrays,
        args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
