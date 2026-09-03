"""The frozen S0 figure definitions: filenames, lead grids, styling and reductions.

Six plates --- streamfunction structure, RMSE 0--200, single-member RMSE, ACC
0--200, streamfunction at day 60 and day 2,000, and RMSE 0--2,000 --- plus the
ensemble summary and the CSV of every published curve. All 15 members are
reduced the same way: mean with the 10th and 90th percentile band.

This module imports nothing from the package. It knows about one model, because
the production arm has one; there is no comparator series anywhere in the suite.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np


LEAD_DAYS = tuple(range(0, 2001, 10))

SHORT_LEAD_DAYS = tuple(range(0, 201, 10))

FIGURE_3_LEADS = (0, 10, 20, 30, 40)

FIGURE_7_LEADS = (60, 2000)

RMSE_FIELDS = ("surface_speed", "phihyd_surface", "sst")

ACC_FIELDS = ("surface_u", "surface_v", "phihyd_surface", "sst")

METHODS = ("model", "climatology", "persistence")

FIGURE_NAMES = (
    "turb_figure3_streamfunction_025deg_s0turb_dt10.png",
    "turb_figure4_dt10_rmse_0_200_days_s0turb.png",
    "turb_figure5_dt10_single_member_rmse_s0turb.png",
    "turb_figure6_dt10_acc_0_200_days_s0turb.png",
    "turb_figure7_dt10_streamfunction_day060_day2000_s0turb.png",
    "turb_figure8_dt10_rmse_0_2000_days_s0turb.png",
)

REPORT_NAME = "turb_s0turb_figures_report.json"

ARRAYS_NAME = "turb_s0turb_figures_arrays.npz"

SUMMARY_NAME = "turb_s0turb_figures_summary.json"

CSV_NAME = "turb_s0turb_ensemble_curves.csv"

MANIFEST_NAME = "manifest.json"

README_NAME = "README.md"

FIELD_LABELS = {
    "surface_speed": r"Surface speed (m s$^{-1}$)",
    "phihyd_surface": r"Surface $P/\rho$ (m$^2$ s$^{-2}$)",
    "sst": r"SST ($^\circ$C)",
    "surface_u": r"Surface U (m s$^{-1}$)",
    "surface_v": r"Surface V (m s$^{-1}$)",
}

MODEL_LABEL = "Production FNO (1-in / 1-out)"

METHOD_LABELS = {
    "model": MODEL_LABEL,
    "climatology": "Climatology",
    "persistence": "Persistence",
}

METHOD_COLORS = {
    "model": "red",
    "climatology": "black",
    "persistence": "blue",
}

class FigureArtifactError(RuntimeError):
    """Raised when a pinned figure input is missing or its bytes changed."""

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def percentile_curve(values: np.ndarray) -> dict[str, np.ndarray]:
    """Return ensemble mean and the paper's 10th/90th percentiles."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != 15:
        raise ValueError("ensemble curves must have shape (15, leads)")
    return {
        "mean": np.mean(array, axis=0),
        "p10": np.percentile(array, 10.0, axis=0),
        "p90": np.percentile(array, 90.0, axis=0),
    }

#: Fixed colour limit, in Sverdrups, for every panel that shows a barotropic
#: streamfunction or streamfunction anomaly. Declared rather than derived from
#: the data so that panels are comparable across leads, across figures and
#: across runs: an auto-scaled difference panel makes a 2 Sv day-60 error look
#: exactly as alarming as a 50 Sv day-2,000 one.
STREAMFUNCTION_BOUND_SV = 50.0


def _finite_bound(values: Sequence[np.ndarray]) -> float:
    pieces = [
        np.abs(np.asarray(value, dtype=np.float64)).ravel()
        for value in values
    ]
    finite = np.concatenate(pieces)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 1.0
    return max(float(np.max(finite)), np.finfo(float).eps)

def _masked(value: np.ndarray, wet: np.ndarray) -> np.ma.MaskedArray:
    array = np.asarray(value)
    return np.ma.masked_where((~wet) | (~np.isfinite(array)), array)

def _verify_file(specification: Mapping[str, Any], label: str) -> Path:
    path = Path(str(specification["path"])).resolve()
    if not path.is_file():
        raise FigureArtifactError(f"{label} is missing: {path}")
    actual = file_sha256(path)
    if actual != specification["sha256"]:
        raise FigureArtifactError(
            f"{label} hash changed: expected {specification['sha256']}, got {actual}"
        )
    return path

def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )

def _plot_streamfunction_grid(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
) -> None:
    truth = np.asarray(arrays["figure3_truth_streamfunction"])
    model = np.asarray(arrays["figure3_model_streamfunction"])
    difference = truth - model
    bound = STREAMFUNCTION_BOUND_SV
    # Its own colorbar, its own quantity: the truth-minus-model row stays
    # data-scaled, or a few-Sv error at day 10 renders as blank white against a
    # bound built for the field itself.
    difference_bound = _finite_bound((difference,))
    figure, axes = plt.subplots(
        3,
        len(FIGURE_3_LEADS),
        figsize=(11.0, 6.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    state_image = difference_image = None
    for column, lead in enumerate(FIGURE_3_LEADS):
        state_image = axes[0, column].pcolormesh(
            longitude,
            latitude,
            _masked(truth[column], wet),
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
            shading="auto",
        )
        axes[1, column].pcolormesh(
            longitude,
            latitude,
            _masked(model[column], wet),
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
            shading="auto",
        )
        difference_image = axes[2, column].pcolormesh(
            longitude,
            latitude,
            _masked(difference[column], wet),
            cmap="RdBu_r",
            vmin=-difference_bound,
            vmax=difference_bound,
            shading="auto",
        )
        axes[0, column].set_title(f"Day {lead}")
        axes[2, column].set_xlabel("Longitude (°)")
    axes[0, 0].set_ylabel("MITgcm\nLatitude (°)")
    axes[1, 0].set_ylabel("Emulator\nLatitude (°)")
    axes[2, 0].set_ylabel("Truth − model\nLatitude (°)")
    for axis in axes.flat:
        axis.set_aspect("equal")
        axis.set_facecolor("0.86")
    figure.colorbar(
        state_image,
        ax=axes[:2].ravel().tolist(),
        label="Barotropic streamfunction (Sv)",
        shrink=0.82,
    )
    figure.colorbar(
        difference_image,
        ax=axes[2].ravel().tolist(),
        label="Truth − model (Sv)",
        shrink=0.82,
    )
    figure.suptitle(
        r"Control wind $\tau_0=0.1$ N m$^{-2}$; "
        r"$\Delta t=10$ days; native $0.25^\circ$ grid"
    )
    figure.savefig(output / FIGURE_NAMES[0], bbox_inches="tight")
    plt.close(figure)

def _plot_rmse(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    long: bool,
) -> None:
    leads = np.asarray(LEAD_DAYS if long else SHORT_LEAD_DAYS)
    limit = len(leads)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(5.4, 8.8),
        sharex=True,
        constrained_layout=True,
    )
    for axis, field in zip(axes, RMSE_FIELDS):
        for method in METHODS:
            summary = percentile_curve(
                arrays[f"rmse__{method}__{field}"][:, :limit]
            )
            color = METHOD_COLORS[method]
            axis.plot(
                leads,
                summary["mean"],
                color=color,
                linewidth=1.6,
                label=METHOD_LABELS[method],
            )
            axis.fill_between(
                leads,
                summary["p10"],
                summary["p90"],
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        axis.set_ylabel(FIELD_LABELS[field])
        axis.grid(color="0.80", linewidth=0.6)
    axes[0].set_title(
        r"$\tau_0=0.1$ N m$^{-2}$; $\Delta t=10$ days; "
        "15 inference initial conditions"
    )
    axes[-1].set_xlabel("Time (days)")
    axes[-1].legend(loc="best")
    axes[-1].set_xlim(0, 2000 if long else 200)
    figure.savefig(
        output / (FIGURE_NAMES[5] if long else FIGURE_NAMES[1]),
        bbox_inches="tight",
    )
    plt.close(figure)

def _plot_single_member(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    leads = np.asarray(SHORT_LEAD_DAYS)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(5.4, 5.9),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(
        leads,
        arrays["single_rmse__streamfunction"],
        color="red",
        linewidth=1.7,
    )
    axes[1].plot(
        leads,
        arrays["single_rmse__sst"],
        color="red",
        linewidth=1.7,
    )
    axes[0].set_ylabel("Streamfunction RMSE (Sv)")
    axes[1].set_ylabel(r"SST RMSE ($^\circ$C)")
    axes[1].set_xlabel("Time (days)")
    for axis in axes:
        axis.grid(color="0.80", linewidth=0.6)
        axis.set_xlim(0, 200)
    axes[0].set_title(
        r"One S0 inference member; $\tau_0=0.1$ N m$^{-2}$; "
        r"$\Delta t=10$ days"
    )
    figure.savefig(output / FIGURE_NAMES[2], bbox_inches="tight")
    plt.close(figure)

def _plot_acc(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    leads = np.asarray(SHORT_LEAD_DAYS)
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(5.4, 10.2),
        sharex=True,
        constrained_layout=True,
    )
    for axis, field in zip(axes, ACC_FIELDS):
        curve = percentile_curve(arrays[f"acc__model__{field}"])
        axis.plot(
            leads,
            curve["mean"],
            color="red",
            linewidth=1.6,
            label=METHOD_LABELS["model"],
        )
        axis.fill_between(
            leads,
            curve["p10"],
            curve["p90"],
            color="red",
            alpha=0.17,
            linewidth=0,
        )
        axis.axhline(0.0, color="0.65", linewidth=0.6)
        axis.set_ylim(-1.0, 1.02)
        axis.set_ylabel(f"{FIELD_LABELS[field]}\nACC")
        axis.grid(color="0.82", linewidth=0.6)
    axes[0].set_title(
        r"S0 anomaly correlation; 15 inference members; $\Delta t=10$ days"
    )
    axes[-1].set_xlabel("Time (days)")
    axes[-1].set_xlim(0, 200)
    axes[-1].legend(loc="best")
    figure.savefig(output / FIGURE_NAMES[3], bbox_inches="tight")
    plt.close(figure)

def _plot_day60_day2000(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
) -> None:
    truth = np.asarray(arrays["figure7_truth_streamfunction"])
    model = np.asarray(arrays["figure7_model_streamfunction"])
    bound = STREAMFUNCTION_BOUND_SV
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.0, 6.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for row, lead in enumerate(FIGURE_7_LEADS):
        for column, value in enumerate((truth[row], model[row])):
            image = axes[row, column].pcolormesh(
                longitude,
                latitude,
                _masked(value, wet),
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
                shading="auto",
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_facecolor("0.86")
            axes[row, column].set_xlabel("Longitude (°)")
        axes[row, 0].set_ylabel(f"Day {lead}\nLatitude (°)")
    axes[0, 0].set_title("MITgcm ground truth")
    axes[0, 1].set_title(MODEL_LABEL)
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Barotropic streamfunction (Sv)",
        shrink=0.84,
    )
    figure.suptitle(
        r"Control wind $\tau_0=0.1$ N m$^{-2}$; "
        r"$\Delta t=10$ days"
    )
    figure.savefig(output / FIGURE_NAMES[4], bbox_inches="tight")
    plt.close(figure)

def _summary(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "complete",
        "member_count": 15,
        "lead_days": {"short": 200, "long": 2000, "interval": 10},
        "all_selected_states_finite": bool(np.all(arrays["finite"])),
        "maximum_selected_normalized_abs": _finite_bound(
            (arrays["normalized_max_abs"],)
        ),
        "rmse": {},
        "acc": {},
    }
    for field in RMSE_FIELDS:
        result["rmse"][field] = {}
        for method in METHODS:
            curve = percentile_curve(arrays[f"rmse__{method}__{field}"])
            result["rmse"][field][method] = {
                "day200_mean": float(curve["mean"][20]),
                "day2000_mean": float(curve["mean"][-1]),
                "day2000_p10": float(curve["p10"][-1]),
                "day2000_p90": float(curve["p90"][-1]),
            }
    for field in ACC_FIELDS:
        curve = percentile_curve(arrays[f"acc__model__{field}"])
        result["acc"][field] = {
            "day200_mean": float(curve["mean"][-1]),
            "day200_p10": float(curve["p10"][-1]),
            "day200_p90": float(curve["p90"][-1]),
            "minimum_mean_0_200": float(np.min(curve["mean"])),
        }
    return result

def _write_csv(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "metric",
                "model_or_method",
                "field",
                "lead_days",
                "mean",
                "p10",
                "p90",
            )
        )
        for field in RMSE_FIELDS:
            for method in METHODS:
                curve = percentile_curve(
                    arrays[f"rmse__{method}__{field}"]
                )
                for index, lead in enumerate(LEAD_DAYS):
                    writer.writerow(
                        (
                            "rmse",
                            method,
                            field,
                            lead,
                            curve["mean"][index],
                            curve["p10"][index],
                            curve["p90"][index],
                        )
                    )
        for field in ACC_FIELDS:
            curve = percentile_curve(arrays[f"acc__model__{field}"])
            for index, lead in enumerate(SHORT_LEAD_DAYS):
                writer.writerow(
                    (
                        "acc",
                        "model",
                        field,
                        lead,
                        curve["mean"][index],
                        curve["p10"][index],
                        curve["p90"][index],
                    )
                )

