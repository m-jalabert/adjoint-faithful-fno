"""Post-hoc long-rollout stability instrumentation for selected Model C.

This module does not run the emulator or select a checkpoint.  It reads the
immutable 15-member S0 day-2000 arrays from job 304736, quantifies sustained
multiplicative RMSE growth, records baseline-crossing dates, and redraws the
long-horizon curves on logarithmic axes.  The fitted RMSE gain is explicitly
not labeled a Jacobian eigenvalue or spectral radius.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VERSION = "model_c_s0_stability_instrument_v1"
CONTRACT_STATUS = "posthoc_instrument_correction_after_job304736"
FIELDS = ("surface_speed", "phihyd_surface", "sst")
BASELINES = ("climatology", "persistence")
WINDOWS = ((30, 330), (200, 500), (300, 600), (400, 700), (700, 1000), (1700, 2000))
BOOTSTRAP_SEED = 20260729
BOOTSTRAP_REPLICATES = 10_000
FIGURES = (
    "model_c_s0_rmse_log_growth.png",
    "model_c_s0_windowed_rmse_gain.png",
    "model_c_s0_normalized_amplitude_log_growth.png",
)
REPORT = "model_c_s0_stability_instrument_report.json"
CSV = "model_c_s0_stability_instrument_curves.csv"
README = "README.md"
MANIFEST = "manifest.json"

FIELD_LABELS = {
    "surface_speed": r"Surface speed (m s$^{-1}$)",
    "phihyd_surface": r"Surface $P/\rho$ (m$^2$ s$^{-2}$)",
    "sst": r"SST ($^\circ$C)",
}


class StabilityInstrumentError(RuntimeError):
    """Raised when immutable inputs or the declared analysis change."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def fit_log_gain(
    lead_days: np.ndarray,
    curve: np.ndarray,
    window: Sequence[int],
) -> tuple[float, float]:
    """Return multiplicative gain per ten days and its e-folding time."""

    leads = np.asarray(lead_days, dtype=np.float64)
    values = np.asarray(curve, dtype=np.float64)
    lower, upper = (int(window[0]), int(window[1]))
    selected = (
        (leads >= lower)
        & (leads <= upper)
        & np.isfinite(values)
        & (values > 0)
    )
    if np.count_nonzero(selected) < 3:
        raise ValueError("log-gain fit needs at least three positive samples")
    slope = np.polyfit(leads[selected] / 10.0, np.log(values[selected]), 1)[0]
    gain = float(np.exp(slope))
    e_folding_days = float(10.0 / slope) if slope > 0 else float("inf")
    return gain, e_folding_days


def bootstrap_gain_interval(
    lead_days: np.ndarray,
    member_curves: np.ndarray,
    window: Sequence[int],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap a 95% interval by resampling the 15 ensemble members."""

    values = np.asarray(member_curves, dtype=np.float64)
    leads = np.asarray(lead_days, dtype=np.float64)
    lower, upper = (int(window[0]), int(window[1]))
    selected = (leads >= lower) & (leads <= upper)
    if values.ndim != 2 or values.shape[0] != 15 or np.count_nonzero(selected) < 3:
        raise ValueError("bootstrap expects 15 member curves and a valid window")
    x = leads[selected] / 10.0
    centered = x - np.mean(x)
    denominator = float(np.sum(np.square(centered)))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.shape[0], size=(replicates, values.shape[0]))
    sampled = values[draws][:, :, selected].mean(axis=1)
    if np.any(sampled <= 0) or not np.all(np.isfinite(sampled)):
        raise ValueError("bootstrap log curves must be finite and positive")
    slopes = np.sum(
        centered[None] * (np.log(sampled) - np.log(sampled).mean(axis=1, keepdims=True)),
        axis=1,
    ) / denominator
    gains = np.exp(slopes)
    lower_gain, upper_gain = np.percentile(gains, (2.5, 97.5))
    return float(lower_gain), float(upper_gain)


def first_and_sustained_crossing(
    lead_days: np.ndarray,
    model: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, int | None]:
    """Return first and first permanently-above-baseline lead."""

    leads = np.asarray(lead_days, dtype=np.int64)
    above = np.asarray(model) > np.asarray(baseline)
    indices = np.flatnonzero(above & (leads > 0))
    first = int(leads[indices[0]]) if indices.size else None
    sustained = next(
        (
            int(leads[index])
            for index in range(1, leads.size)
            if bool(np.all(above[index:]))
        ),
        None,
    )
    return {"first_day": first, "sustained_day": sustained}


def load_contract(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or tuple(map(tuple, contract["analysis"]["fit_windows_days"])) != WINDOWS
        or int(contract["analysis"]["bootstrap_seed"]) != BOOTSTRAP_SEED
        or int(contract["analysis"]["bootstrap_replicates"]) != BOOTSTRAP_REPLICATES
        or tuple(contract["analysis"]["fields"]) != FIELDS
    ):
        raise StabilityInstrumentError("stability-instrument contract changed")
    for label, specification in contract["inputs"].items():
        source = Path(specification["path"]).resolve()
        if not source.is_file() or file_sha256(source) != specification["sha256"]:
            raise StabilityInstrumentError(f"immutable input changed: {label}")
    root = resolved.parents[1]
    for relative, expected in contract["source_hashes"].items():
        source = root / relative
        if not source.is_file() or file_sha256(source) != expected:
            raise StabilityInstrumentError(f"analysis source changed: {relative}")
    output = Path(contract["output"]["project"]).resolve()
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise FileExistsError("refusing to overwrite stability-instrument output")
    return contract, resolved, file_sha256(resolved)


def _summary(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    leads = np.asarray(arrays["lead_days"], dtype=np.int64)
    result: dict[str, Any] = {
        "classification": (
            "sustained_supercritical_rmse_and_normalized_amplitude_growth;"
            "descriptive_gain_not_jacobian_spectral_radius"
        ),
        "fields": {},
        "normalized_max_abs": {},
    }
    for field_index, field in enumerate(FIELDS):
        members = np.asarray(arrays[f"rmse__model__{field}"], dtype=np.float64)
        mean = members.mean(axis=0)
        field_result: dict[str, Any] = {
            "windowed_gain_per_10_days": {},
            "baseline_crossing": {},
        }
        for window_index, window in enumerate(WINDOWS):
            gain, e_folding = fit_log_gain(leads, mean, window)
            ci = bootstrap_gain_interval(
                leads,
                members,
                window,
                seed=BOOTSTRAP_SEED + 100 * field_index + window_index,
            )
            field_result["windowed_gain_per_10_days"][
                f"{window[0]}_{window[1]}"
            ] = {
                "gain": gain,
                "bootstrap_95_percent_interval": list(ci),
                "e_folding_days_if_gain_gt_1": e_folding,
            }
        for baseline in BASELINES:
            baseline_mean = np.asarray(
                arrays[f"rmse__{baseline}__{field}"],
                dtype=np.float64,
            ).mean(axis=0)
            field_result["baseline_crossing"][baseline] = (
                first_and_sustained_crossing(leads, mean, baseline_mean)
            )
        result["fields"][field] = field_result
    normalized = np.asarray(arrays["normalized_max_abs"], dtype=np.float64)
    normalized_mean = normalized.mean(axis=0)
    for day in (0, 200, 300, 2000):
        index = int(np.flatnonzero(leads == day)[0])
        result["normalized_max_abs"][f"day{day}_ensemble_mean"] = float(
            normalized_mean[index]
        )
    for window in ((300, 600), (700, 1000), (1700, 2000)):
        gain, e_folding = fit_log_gain(leads, normalized_mean, window)
        result["normalized_max_abs"][f"gain_{window[0]}_{window[1]}"] = {
            "gain": gain,
            "e_folding_days_if_gain_gt_1": e_folding,
        }
    return result


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )


def _plot_log_rmse(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    leads = np.asarray(arrays["lead_days"])
    colors = {"model": "#D62728", "climatology": "#111111", "persistence": "#1F77B4"}
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.6), sharex=True, constrained_layout=True)
    for axis, field in zip(axes, FIELDS):
        for method in ("model", *BASELINES):
            values = np.asarray(arrays[f"rmse__{method}__{field}"], dtype=np.float64)
            mean = values.mean(axis=0)
            p10, p90 = np.percentile(values, (10, 90), axis=0)
            positive = leads > 0
            axis.plot(leads[positive], mean[positive], color=colors[method], label=method.capitalize())
            axis.fill_between(
                leads[positive],
                np.maximum(p10[positive], np.finfo(float).tiny),
                p90[positive],
                color=colors[method],
                alpha=0.13,
                linewidth=0,
            )
        axis.axvspan(300, 600, color="#F4A261", alpha=0.12)
        axis.set_yscale("log")
        axis.set_ylabel(FIELD_LABELS[field])
        axis.grid(which="both", color="0.84", linewidth=0.55)
    axes[0].set_title("S0 Model C error growth: logarithmic view of the same 15-member rollout")
    axes[-1].set_xlabel("Lead (days)")
    axes[-1].set_xlim(0, 2000)
    axes[-1].legend(loc="best", ncol=3)
    figure.savefig(output / FIGURES[0], bbox_inches="tight")
    plt.close(figure)


def _plot_windowed_gain(
    output: Path,
    summary: Mapping[str, Any],
) -> None:
    labels = [f"{lower}–{upper}" for lower, upper in WINDOWS]
    x = np.arange(len(WINDOWS), dtype=np.float64)
    offsets = (-0.22, 0.0, 0.22)
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axis = plt.subplots(figsize=(8.2, 4.3), constrained_layout=True)
    for offset, color, field in zip(offsets, colors, FIELDS):
        records = summary["fields"][field]["windowed_gain_per_10_days"]
        gain = np.asarray([records[f"{a}_{b}"]["gain"] for a, b in WINDOWS])
        ci = np.asarray(
            [records[f"{a}_{b}"]["bootstrap_95_percent_interval"] for a, b in WINDOWS]
        )
        axis.errorbar(
            x + offset,
            gain,
            yerr=np.vstack((gain - ci[:, 0], ci[:, 1] - gain)),
            fmt="o-",
            capsize=3,
            linewidth=1.2,
            color=color,
            label=FIELD_LABELS[field],
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Fitted RMSE gain per 10-day call")
    axis.set_xlabel("Fit window (days)")
    axis.set_title("Positive gain persists far beyond the ~90-day decorrelation horizon")
    axis.grid(color="0.86", linewidth=0.6)
    axis.legend(loc="best")
    figure.savefig(output / FIGURES[1], bbox_inches="tight")
    plt.close(figure)


def _plot_normalized_amplitude(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    leads = np.asarray(arrays["lead_days"])
    values = np.asarray(arrays["normalized_max_abs"], dtype=np.float64)
    mean = values.mean(axis=0)
    p10, p90 = np.percentile(values, (10, 90), axis=0)
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    axis.plot(leads, mean, color="#7B2CBF", linewidth=1.8, label="Ensemble mean")
    axis.fill_between(leads, p10, p90, color="#7B2CBF", alpha=0.18, label="10th–90th percentile")
    for threshold in (20, 40, 100, 200):
        axis.axhline(threshold, color="0.55", linestyle=":", linewidth=0.7)
    axis.axvline(200, color="#D62728", linestyle="--", linewidth=1.0, label="Former day-200 bound")
    axis.set_yscale("log")
    axis.set_xlim(0, 2000)
    axis.set_xlabel("Lead (days)")
    axis.set_ylabel("Maximum |normalized state|")
    axis.set_title("The day-200 amplitude check ends before sustained runaway is visible")
    axis.grid(which="both", color="0.84", linewidth=0.55)
    axis.legend(loc="best")
    figure.savefig(output / FIGURES[2], bbox_inches="tight")
    plt.close(figure)


def _write_csv(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    leads = np.asarray(arrays["lead_days"])
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "method", "field", "lead_days", "mean", "p10", "p90"))
        for field in FIELDS:
            for method in ("model", *BASELINES):
                values = np.asarray(arrays[f"rmse__{method}__{field}"], dtype=np.float64)
                mean = values.mean(axis=0)
                p10, p90 = np.percentile(values, (10, 90), axis=0)
                for lead, center, lower, upper in zip(leads, mean, p10, p90):
                    writer.writerow(("rmse", method, field, int(lead), center, lower, upper))
        values = np.asarray(arrays["normalized_max_abs"], dtype=np.float64)
        mean = values.mean(axis=0)
        p10, p90 = np.percentile(values, (10, 90), axis=0)
        for lead, center, lower, upper in zip(leads, mean, p10, p90):
            writer.writerow(("normalized_max_abs", "model", "all_channels", int(lead), center, lower, upper))


def run(contract_path: str | Path) -> dict[str, Any]:
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    arrays_path = Path(contract["inputs"]["job304736_arrays"]["path"]).resolve()
    with np.load(arrays_path) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if (
        arrays["lead_days"].shape != (201,)
        or arrays["normalized_max_abs"].shape != (15, 201)
        or not np.array_equal(arrays["lead_days"], np.arange(0, 2001, 10))
    ):
        raise StabilityInstrumentError("job-304736 array shape changed")
    summary = _summary(arrays)
    output = Path(contract["output"]["project"]).resolve()
    temporary = output.with_name(output.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    try:
        _style()
        _plot_log_rmse(temporary, arrays)
        _plot_windowed_gain(temporary, summary)
        _plot_normalized_amplitude(temporary, arrays)
        _write_csv(temporary / CSV, arrays)
        report = {
            "version": VERSION,
            "status": "complete",
            "scope": "posthoc_instrument_correction_no_rollout_no_training_no_selection",
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "inputs": contract["inputs"],
            "analysis": contract["analysis"],
            "summary": summary,
            "figures": list(FIGURES),
        }
        report["report_content_sha256"] = json_sha256(report)
        (temporary / REPORT).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (temporary / README).write_text(
            "# Model C S0 stability-instrument correction\n\n"
            "This post-hoc package re-expresses immutable job-304736 curves. "
            "It performs no rollout, training, or checkpoint selection. "
            "Fitted RMSE gain is descriptive and is not a Jacobian spectral radius.\n\n"
            f"Report content SHA-256: `{report['report_content_sha256']}`.\n"
        )
        manifest = {
            "version": VERSION,
            "contract_sha256": contract_sha,
            "report_content_sha256": report["report_content_sha256"],
            "artifacts": {
                path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                for path in sorted(temporary.iterdir())
                if path.is_file()
            },
        }
        manifest["manifest_content_sha256"] = json_sha256(manifest)
        (temporary / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.contract), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
