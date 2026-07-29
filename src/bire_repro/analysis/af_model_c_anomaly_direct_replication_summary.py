"""Summarize the frozen three-seed anomaly-direct replication decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..af_model_c_anomaly_direct_replication import (
    EXPECTED_DECLARED_SEEDS,
    load_replication_contract,
)

VERSION = "model_c_anomaly_direct_replication_summary_v1"
FIELDS = ("surface_speed", "sst", "phihyd_surface")
BASELINES = ("persistence", "climatology")
EXPECTED_LEADS = tuple(range(10, 91, 10))
REPORT_NAME = "model_c_anomaly_direct_v1_report.json"
CSV_NAME = "model_c_anomaly_direct_rmse_curves.csv"
OUTPUT_NAMES = (
    "replication_summary.json",
    "replication_metrics.csv",
    "model_c_anomaly_direct_replication_summary.png",
    "replication_manifest.json",
    "README.md",
)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def curve_auc(values: Mapping[int, float]) -> float:
    """Return the lead-normalized trapezoidal AUC over the frozen leads."""

    if tuple(sorted(values)) != EXPECTED_LEADS:
        raise ValueError("replication AUC leads changed")
    leads = np.asarray(EXPECTED_LEADS, dtype=np.float64)
    curve = np.asarray([values[int(lead)] for lead in leads], dtype=np.float64)
    if not np.all(np.isfinite(curve)):
        raise ValueError("replication AUC curve is non-finite")
    return float(np.trapezoid(curve, leads) / (leads[-1] - leads[0]))


def _csv_auc(path: Path) -> dict[str, dict[str, float]]:
    values: dict[tuple[str, str], dict[int, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            lead = int(row["lead_days"])
            if (
                row["scope"] != "fixed_S2_validation_figure"
                or lead not in EXPECTED_LEADS
            ):
                continue
            key = (row["field"], row["method"])
            values.setdefault(key, {})[lead] = float(row["mean_rmse"])
    result: dict[str, dict[str, float]] = {}
    for field in FIELDS:
        result[field] = {
            method: curve_auc(values[(field, method)])
            for method in ("model", *BASELINES)
        }
    return result


def _paths(
    contract: Mapping[str, Any],
    resolved_contract: Path,
) -> dict[int, dict[str, Path]]:
    project_root = Path(contract["output_contract"]["project_root"])
    scratch_root = Path(contract["output_contract"]["scratch_root"])
    parent_path = (
        resolved_contract.parents[1] / contract["parent_contract"]["path"]
    )
    parent = json.loads(parent_path.read_text())
    return {
        20260723: {
            "report": Path(contract["reference_result"]["report"]),
            "csv": Path(parent["output_contract"]["project_output"]) / CSV_NAME,
            "manifest": Path(parent["output_contract"]["project_output"])
            / "figure_manifest.json",
        },
        **{
            seed: {
                "report": scratch_root
                / "seeds"
                / f"seed_{seed}"
                / REPORT_NAME,
                "csv": project_root / f"seed_{seed}" / CSV_NAME,
                "manifest": project_root
                / f"seed_{seed}"
                / "figure_manifest.json",
            }
            for seed in contract["seed_replication"]["new_seeds"]
        },
    }


def _verify_artifact_bundle(
    seed: int,
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = paths["report"]
    manifest_path = paths["manifest"]
    if not report_path.is_file() or not paths["csv"].is_file():
        raise FileNotFoundError(f"replication seed {seed} is incomplete")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"replication seed {seed} manifest is missing")
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (
        int(report.get("seed", -1)) != seed
        or manifest.get("status") != "complete"
        or manifest.get("source_report_sha256") != _file_sha256(report_path)
        or manifest.get("source_report_content_sha256")
        != report.get("report_content_sha256")
        or manifest.get("source_arrays_sha256") != report.get("arrays_sha256")
        or manifest.get("csv", {}).get("sha256") != _file_sha256(paths["csv"])
        or manifest.get("inference_state_opened") is not False
    ):
        raise ValueError(f"replication seed {seed} artifact bundle changed")
    return report, manifest


def seed_summary(
    seed: int,
    report: Mapping[str, Any],
    auc: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Apply every frozen per-seed replication requirement."""

    ratios: dict[str, dict[str, float]] = {}
    auc_checks: dict[str, dict[str, bool]] = {}
    for field in FIELDS:
        model_auc = float(auc[field]["model"])
        ratios[field] = {
            baseline: model_auc / float(auc[field][baseline])
            for baseline in BASELINES
        }
        auc_checks[field] = {
            baseline: ratios[field][baseline] < 1.0
            for baseline in BASELINES
        }

    metrics = report["validation_figure"]["metrics"]
    bounded_checks = {
        field: bool(
            np.isfinite(metrics[field]["model"]["maximum_mean"])
            and np.isfinite(metrics[field]["model"]["maximum_p90"])
            and not metrics[field]["model"]["mean_curve_exceeds_requested_axis"]
            and not metrics[field]["model"]["p90_curve_exceeds_requested_axis"]
        )
        for field in FIELDS
    }
    climatology_checks = {
        field: float(metrics[field]["model"]["day200_mean"])
        < float(metrics[field]["climatology"]["day200_mean"])
        for field in FIELDS
    }
    persistence_checks = {
        field: float(metrics[field]["model"]["day200_mean"])
        < float(metrics[field]["persistence"]["day200_mean"])
        for field in ("surface_speed", "sst")
    }
    training_gate = report["selected_training_summary"]["checkpoint_gate"]
    checks = {
        "training_gate": bool(report["selection_decision"]["passed"]),
        "exact_reload": bool(report["save_reload_nine_step_bitwise_exact"]),
        "zero_land": float(training_gate["normalized_land_max_abs"]) == 0.0,
        "all_primary_auc": all(
            value
            for field in auc_checks.values()
            for value in field.values()
        ),
        "all_curves_bounded": all(bounded_checks.values()),
        "all_day200_below_climatology": all(climatology_checks.values()),
        "speed_and_sst_day200_below_persistence": all(
            persistence_checks.values()
        ),
        "inference_sealed": report["inference_state_opened"] is False,
    }
    six_ratios = [
        ratios[field][baseline] for field in FIELDS for baseline in BASELINES
    ]
    return {
        "seed": seed,
        "selected_optimizer_step": int(
            report["selection_decision"]["selected_fine_tune_step"]
        ),
        "selected_checkpoint": report["selected_checkpoint"],
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "report_content_sha256": report["report_content_sha256"],
        "arrays_sha256": report["arrays_sha256"],
        "training_worst_primary_auc_ratio": float(
            training_gate["worst_primary_rmse_auc_ratio"]
        ),
        "training_worst_slow_field_lead_ratio": float(
            training_gate["worst_slow_field_lead_ratio"]
        ),
        "fixed_S2_auc_ratio": ratios,
        "fixed_S2_day200_rmse": {
            field: {
                method: float(metrics[field][method]["day200_mean"])
                for method in ("model", *BASELINES)
            }
            for field in FIELDS
        },
        "fixed_S2_auc_checks": auc_checks,
        "bounded_checks": bounded_checks,
        "day200_climatology_checks": climatology_checks,
        "day200_persistence_checks": persistence_checks,
        "checks": checks,
        "passed": all(checks.values()),
        "median_ranking_score": float(np.mean(six_ratios)),
    }


def _write_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "seed",
                "selected_optimizer_step",
                "field",
                "auc_ratio_to_persistence",
                "auc_ratio_to_climatology",
                "day200_model_rmse",
                "day200_ratio_to_persistence",
                "day200_ratio_to_climatology",
                "seed_passed",
                "median_ranking_score",
            )
        )
        for summary in summaries:
            for field in FIELDS:
                day200 = summary["fixed_S2_day200_rmse"][field]
                writer.writerow(
                    (
                        summary["seed"],
                        summary["selected_optimizer_step"],
                        field,
                        summary["fixed_S2_auc_ratio"][field]["persistence"],
                        summary["fixed_S2_auc_ratio"][field]["climatology"],
                        day200["model"],
                        day200["model"] / day200["persistence"],
                        day200["model"] / day200["climatology"],
                        summary["passed"],
                        summary["median_ranking_score"],
                    )
                )


def _plot(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    seeds = [int(value["seed"]) for value in summaries]
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13, 7.8),
        sharex=True,
        constrained_layout=True,
    )
    x = np.arange(len(seeds), dtype=float)
    width = 0.34
    for column, field in enumerate(FIELDS):
        axis = axes[0, column]
        for offset, baseline, color in (
            (-width / 2, "persistence", "#2F75B5"),
            (width / 2, "climatology", "#A86600"),
        ):
            values = [
                summary["fixed_S2_auc_ratio"][field][baseline]
                for summary in summaries
            ]
            axis.bar(
                x + offset,
                values,
                width,
                label=baseline.capitalize(),
                color=color,
            )
        axis.axhline(1.0, color="black", linestyle=":", linewidth=1)
        axis.set_title(field.replace("_", " ").title())
        axis.set_ylabel("10–90-day RMSE-AUC ratio")
        axis.grid(axis="y", alpha=0.25)

        axis = axes[1, column]
        for offset, baseline, color in (
            (-width / 2, "persistence", "#2F75B5"),
            (width / 2, "climatology", "#A86600"),
        ):
            values = []
            for summary in summaries:
                day200 = summary["fixed_S2_day200_rmse"][field]
                values.append(day200["model"] / day200[baseline])
            axis.bar(
                x + offset,
                values,
                width,
                label=baseline.capitalize(),
                color=color,
            )
        axis.axhline(1.0, color="black", linestyle=":", linewidth=1)
        axis.set_ylabel("Day-200 RMSE ratio")
        axis.set_xticks(x, [str(seed) for seed in seeds], rotation=20)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    figure.suptitle(
        "Pointwise-anomaly direct-state Model C: frozen three-seed replication"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize(
    contract_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Apply the frozen gate and publish the three-seed decision package."""

    contract, resolved, digest = load_replication_contract(contract_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        if (output / name).exists():
            raise FileExistsError(f"replication summary output exists: {name}")

    summaries = []
    source_artifacts = {}
    for seed, paths in sorted(_paths(contract, resolved).items()):
        report, manifest = _verify_artifact_bundle(seed, paths)
        summary = seed_summary(seed, report, _csv_auc(paths["csv"]))
        summaries.append(summary)
        source_artifacts[str(seed)] = {
            "report": str(paths["report"]),
            "report_sha256": _file_sha256(paths["report"]),
            "report_content_sha256": report["report_content_sha256"],
            "arrays_sha256": report["arrays_sha256"],
            "csv": str(paths["csv"]),
            "csv_sha256": _file_sha256(paths["csv"]),
            "figure_manifest": str(paths["manifest"]),
            "figure_manifest_sha256": _file_sha256(paths["manifest"]),
            "figure_manifest_content_sha256": manifest[
                "manifest_content_sha256"
            ],
        }
    if tuple(int(value["seed"]) for value in summaries) != EXPECTED_DECLARED_SEEDS:
        raise ValueError("declared seed result set changed")

    all_pass = all(bool(value["passed"]) for value in summaries)
    ranking = sorted(
        (
            float(value["median_ranking_score"]),
            int(value["seed"]),
        )
        for value in summaries
    )
    selected_seed = ranking[len(ranking) // 2][1] if all_pass else None
    selected = next(
        (value for value in summaries if value["seed"] == selected_seed),
        None,
    )
    result = {
        "status": "complete",
        "version": VERSION,
        "classification": (
            "three_seed_replication_gate_passed"
            if all_pass
            else "three_seed_replication_gate_failed"
        ),
        "replication_contract": str(resolved),
        "replication_contract_sha256": digest,
        "source_artifacts": source_artifacts,
        "per_seed": summaries,
        "all_three_seeds_passed": all_pass,
        "median_ranking": [
            {"seed": seed, "score": score} for score, seed in ranking
        ],
        "selected_median_seed": selected_seed,
        "selected_optimizer_step": (
            selected["selected_optimizer_step"] if selected else None
        ),
        "selected_checkpoint": (
            selected["selected_checkpoint"] if selected else None
        ),
        "selected_checkpoint_sha256": (
            selected["selected_checkpoint_sha256"] if selected else None
        ),
        "inference_state_opened": False,
        "intermediate_wind_state_opened": False,
        "response_or_adjoint_state_opened": False,
        "next_decision": contract["next_decision"][
            "all_three_seeds_pass"
            if all_pass
            else "training_gate_not_reproduced"
        ],
    }
    result["summary_content_sha256"] = _json_sha256(result)
    summary_path = output / "replication_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    csv_path = output / "replication_metrics.csv"
    _write_csv(csv_path, summaries)
    figure_path = output / "model_c_anomaly_direct_replication_summary.png"
    _plot(figure_path, summaries)
    readme_path = output / "README.md"
    readme_path.write_text(
        "# Model C anomaly-direct replication\n\n"
        "This directory contains the frozen three-seed decision summary, "
        "machine-readable metrics, a comparison plot, and the two per-seed "
        "Figure-4 packages. Checkpoints were selected using split 1 before "
        "the unchanged fixed-S2 characterization. Later archives remained sealed.\n"
    )
    source_path = Path(__file__).resolve()
    manifest = {
        "status": "complete",
        "version": VERSION,
        "summary": {
            "path": str(summary_path),
            "sha256": _file_sha256(summary_path),
            "content_sha256": result["summary_content_sha256"],
        },
        "csv": {
            "path": str(csv_path),
            "sha256": _file_sha256(csv_path),
        },
        "figure": {
            "path": str(figure_path),
            "sha256": _file_sha256(figure_path),
        },
        "readme": {
            "path": str(readme_path),
            "sha256": _file_sha256(readme_path),
        },
        "source": {
            "path": str(source_path),
            "sha256": _file_sha256(source_path),
        },
        "inference_state_opened": False,
    }
    manifest["manifest_content_sha256"] = _json_sha256(manifest)
    (output / "replication_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = summarize(args.contract, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
