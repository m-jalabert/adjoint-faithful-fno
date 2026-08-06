"""Baseline-visible SST and surface-pressure companion to Model C Figure 4.

The original Bire-style figure is immutable and intentionally retains the
paper-matched axes.  This module reads only its hash-pinned numerical archive
and produces a versioned full-range/baseline-zoom companion plus a CSV table.
It never evaluates a model or reads a trajectory dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .af_model_c_bire_figures import percentile_curve
from .af_model_c_overfit import _file_sha256


VERSION = "model_c_bire_baseline_zoom_v1"
FIGURE_NAME = "model_c_bire_figure4_dt10_sst_phihyd_full_and_zoom.png"
CSV_NAME = "model_c_bire_figure4_dt10_sst_phihyd_rmse.csv"
SUMMARY_NAME = "baseline_zoom_summary.json"
MANIFEST_NAME = "figure_manifest.json"
README_NAME = "README.md"
FIELDS = ("sst", "phihyd_surface")
METHODS = ("model", "climatology", "persistence")
FIELD_LABELS = {
    "sst": r"SST RMSE ($^\circ$C)",
    "phihyd_surface": r"Surface $P/\rho$ RMSE (m$^2$ s$^{-2}$)",
}
METHOD_LABELS = {
    "model": "Prediction",
    "climatology": "Climatology",
    "persistence": "Persistence",
}
METHOD_COLORS = {
    "model": "red",
    "climatology": "black",
    "persistence": "blue",
}


class BaselineZoomError(RuntimeError):
    """Raised when the immutable companion-figure contract is violated."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load and verify the frozen descriptive-addendum contract."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status")
        != "frozen_before_versioned_baseline_zoom_generation"
    ):
        raise ValueError("baseline-zoom contract is not frozen")
    figure = contract.get("figure", {})
    if (
        tuple(figure.get("fields", ())) != FIELDS
        or tuple(figure.get("methods", ())) != METHODS
        or figure.get("lead_days") != list(range(0, 201, 10))
        or figure.get("zoom_limits")
        != {"sst": [0.0, 0.08], "phihyd_surface": [0.0, 0.10]}
    ):
        raise ValueError("baseline-zoom figure contract changed")
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(f"baseline-zoom source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def first_worse_lead(
    model: np.ndarray,
    baseline: np.ndarray,
    lead_days: np.ndarray,
) -> int | None:
    """Return the first positive lead where mean model RMSE exceeds baseline."""

    model_values = np.asarray(model, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    leads = np.asarray(lead_days, dtype=np.int64)
    if (
        model_values.shape != baseline_values.shape
        or model_values.ndim != 2
        or leads.shape != (model_values.shape[1],)
    ):
        raise ValueError("model, baseline, and lead arrays are inconsistent")
    model_mean = model_values.mean(axis=0)
    baseline_mean = baseline_values.mean(axis=0)
    candidates = leads[(leads > 0) & (model_mean > baseline_mean)]
    return int(candidates[0]) if candidates.size else None


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    leads = np.asarray(arrays["lead_days"], dtype=np.int64)
    if not np.array_equal(leads, np.arange(0, 201, 10)):
        raise BaselineZoomError("source lead-day array changed")
    for field in FIELDS:
        for method in METHODS:
            values = np.asarray(arrays[f"{method}__rmse__{field}"])
            if values.shape != (15, leads.size) or not np.isfinite(values).all():
                raise BaselineZoomError(
                    f"invalid source curve {method}__rmse__{field}"
                )
    return leads


def curve_summary(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Return crossing days and selected numerical values for the companion."""

    leads = _validate_arrays(arrays)
    lead_indices = {int(lead): index for index, lead in enumerate(leads)}
    result: dict[str, Any] = {}
    for field in FIELDS:
        field_result: dict[str, Any] = {"crossings": {}, "selected_leads": {}}
        model = np.asarray(arrays[f"model__rmse__{field}"])
        for baseline in ("persistence", "climatology"):
            field_result["crossings"][baseline] = first_worse_lead(
                model,
                np.asarray(arrays[f"{baseline}__rmse__{field}"]),
                leads,
            )
        for lead in (10, 20, 30, 60, 70, 90, 100, 160, 200):
            index = lead_indices[lead]
            field_result["selected_leads"][str(lead)] = {
                method: float(
                    np.asarray(arrays[f"{method}__rmse__{field}"])[
                        :, index
                    ].mean()
                )
                for method in METHODS
            }
        result[field] = field_result
    return result


def plot_companion(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    zoom_limits: Mapping[str, list[float]],
) -> None:
    """Plot full-range and baseline-scale views without changing source data."""

    leads = _validate_arrays(arrays)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 6.8),
        sharex=True,
        constrained_layout=True,
    )
    for row, field in enumerate(FIELDS):
        summaries = {
            method: percentile_curve(arrays[f"{method}__rmse__{field}"])
            for method in METHODS
        }
        for column, axis in enumerate(axes[row]):
            for method in METHODS:
                summary = summaries[method]
                color = METHOD_COLORS[method]
                axis.plot(
                    leads,
                    summary["mean"],
                    color=color,
                    linewidth=1.7,
                    label=METHOD_LABELS[method],
                )
                axis.fill_between(
                    leads,
                    summary["p10"],
                    summary["p90"],
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
            axis.set_xlim(0, 200)
            axis.set_xticks((0, 50, 100, 150, 200))
            axis.grid(color="0.78", linewidth=0.6)
            axis.set_ylabel(FIELD_LABELS[field])
            if column == 0:
                maximum = max(
                    float(np.max(summaries[method]["p90"]))
                    for method in METHODS
                )
                axis.set_ylim(0.0, 1.04 * maximum)
                axis.set_title("Full range")
            else:
                axis.set_ylim(*zoom_limits[field])
                axis.set_title("Baseline-scale zoom")
                for baseline, linestyle in (
                    ("persistence", "--"),
                    ("climatology", ":"),
                ):
                    crossing = first_worse_lead(
                        arrays[f"model__rmse__{field}"],
                        arrays[f"{baseline}__rmse__{field}"],
                        leads,
                    )
                    if crossing is not None:
                        axis.axvline(
                            crossing,
                            color=METHOD_COLORS[baseline],
                            linestyle=linestyle,
                            linewidth=0.9,
                            alpha=0.75,
                        )
        axes[row, 0].set_xlabel("Lead (days)")
        axes[row, 1].set_xlabel("Lead (days)")
    axes[0, 0].legend(loc="upper left")
    figure.suptitle(
        "Rejected Model C: SST and surface pressure RMSE against baselines\n"
        r"15 fixed S2 starts, $\Delta t=10$ days; bands show 10th--90th percentiles"
    )
    figure.savefig(output / FIGURE_NAME, bbox_inches="tight")
    plt.close(figure)


def write_csv(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write every plotted mean and percentile to a portable table."""

    leads = _validate_arrays(arrays)
    with (output / CSV_NAME).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("field", "method", "lead_days", "mean", "p10", "p90"))
        for field in FIELDS:
            for method in METHODS:
                summary = percentile_curve(arrays[f"{method}__rmse__{field}"])
                for index, lead in enumerate(leads):
                    writer.writerow(
                        (
                            field,
                            method,
                            int(lead),
                            f"{float(summary['mean'][index]):.10g}",
                            f"{float(summary['p10'][index]):.10g}",
                            f"{float(summary['p90'][index]):.10g}",
                        )
                    )


def _readme(summary: Mapping[str, Any]) -> str:
    crossings = summary["curve_summary"]
    return f"""# Model C SST and surface-pressure baseline companion

This immutable descriptive package makes the climatology and persistence
comparisons from the original fixed-axis Bire-style Figure 4 easier to inspect.
It does not rerun or select a model and does not open any new dataset split.

The left column of `{FIGURE_NAME}` shows the complete finite Model C RMSE range.
The right column zooms to the baseline scale. Vertical lines mark the first lead
where Model C mean RMSE becomes worse than persistence or climatology.

SST first loses to persistence/climatology at
{crossings["sst"]["crossings"]["persistence"]}/
{crossings["sst"]["crossings"]["climatology"]} days. Surface PHIHYD first loses
at {crossings["phihyd_surface"]["crossings"]["persistence"]}/
{crossings["phihyd_surface"]["crossings"]["climatology"]} days.

`{CSV_NAME}` contains all member-mean and 10th--90th percentile values.
Manifest content SHA-256: `{summary["manifest_content_sha256"]}`.
"""


def generate(
    *,
    arrays_path: Path,
    source_report_path: Path,
    contract_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate the non-overwriting project-facing companion package."""

    contract, resolved_contract, contract_sha = load_contract(contract_path)
    source = contract["source_artifacts"]
    if (
        _file_sha256(arrays_path) != source["arrays_sha256"]
        or _file_sha256(source_report_path) != source["report_sha256"]
    ):
        raise BaselineZoomError("immutable Bire-style source artifact changed")
    source_report = json.loads(source_report_path.read_text())
    if (
        source_report.get("status") != "complete"
        or source_report.get("arrays_sha256") != source["arrays_sha256"]
        or source_report.get("tuning_authorized") is not False
        or source_report.get("inference_opened") is not False
    ):
        raise BaselineZoomError("source report is not the sealed descriptive result")

    output = output_dir.resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"baseline-zoom output already exists: {output}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        with np.load(arrays_path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        summary_values = curve_summary(arrays)
        plot_companion(
            temporary,
            arrays,
            contract["figure"]["zoom_limits"],
        )
        write_csv(temporary, arrays)
        artifact_hashes = {
            FIGURE_NAME: _file_sha256(temporary / FIGURE_NAME),
            CSV_NAME: _file_sha256(temporary / CSV_NAME),
        }
        manifest = {
            "version": VERSION,
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "source_arrays": str(arrays_path.resolve()),
            "source_arrays_sha256": source["arrays_sha256"],
            "artifacts": artifact_hashes,
        }
        manifest["manifest_content_sha256"] = _json_sha256(manifest)
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        summary = {
            "version": VERSION,
            "status": "complete",
            "descriptive_only": True,
            "tuning_authorized": False,
            "inference_opened": False,
            "curve_summary": summary_values,
            "source_arrays_sha256": source["arrays_sha256"],
            "source_report_sha256": source["report_sha256"],
            "contract_sha256": contract_sha,
            "artifacts": artifact_hashes,
            "manifest_content_sha256": manifest["manifest_content_sha256"],
        }
        summary["summary_content_sha256"] = _json_sha256(summary)
        (temporary / SUMMARY_NAME).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        (temporary / README_NAME).write_text(_readme(summary))
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Model C SST/PHIHYD baseline-visible Figure 4 addendum"
    )
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = generate(
        arrays_path=args.arrays.resolve(),
        source_report_path=args.source_report.resolve(),
        contract_path=args.contract.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
