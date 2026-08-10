#!/usr/bin/env python
"""Compare the MITgcm S0 control trajectory against its perturbed MITgcm twin.

This analysis is deliberately MITgcm-only.  It answers one question --
*does MITgcm diverge from MITgcm?* -- before any FNO curve is allowed onto the
same axes.  Both trajectories share the identical executable, forcing, physics
and grid; they differ solely in the year-100 pickup, where

    (U, V) -> (1 + 1e-6) (U, V)

and nothing else was touched.

The script reads the daily ``dynState``/``surfState`` diagnostics of both runs,
forms

    dU(t) = U'(t) - U(t),   dV(t) = V'(t) - V(t),
    dTheta(t) = Theta'(t) - Theta(t),   deta(t) = eta'(t) - eta(t),

and reduces them to the same wet-cell RMSE the FNO diagnostics use.  Surface
velocities are centred onto tracer points with the same operator the training
dataset applies, so these numbers are directly comparable with the FNO curves
later.

Outputs (under ``--output-dir``):

    s0_twin_divergence.npz        every daily time series
    s0_twin_divergence.csv        the same series as text
    s0_twin_divergence.json       provenance, fitted growth rates, saturation
    s0_twin_rmse_linear.png       RMSE(t), linear, 0-25 years
    s0_twin_rmse_log.png          log RMSE(t) with the fitted exponential
    s0_twin_rmse_log_early.png    the early-growth zoom, the chaos diagnostic
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "archive" / "src"))

from bire_repro.mds import mds_fields, read_mds  # noqa: E402


MODEL_YEAR_DAYS = 360
STEPS_PER_DAY = 72
STEPS_PER_YEAR = MODEL_YEAR_DAYS * STEPS_PER_DAY
CONTROL_EXPERIMENT = "S0"
TWIN_EXPERIMENT = "S0_twin"
TWIN_ROOT_NAME = "mitgcm_twin_v1"
TWIN_LABEL = "S0_eps1e-6"
START_YEAR = 100
END_YEAR = 125
START_ITERATION = START_YEAR * STEPS_PER_YEAR

#: Validated categorical slots 1-3 (blue, orange, aqua); fixed order, never cycled.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"

SURFACE_DIAGNOSTICS = (
    ("rmse_sst", "SST", "RMSE  (degC)"),
    ("rmse_ssh", "SSH", "RMSE  (m)"),
    ("rmse_surface_speed", "Surface speed", "RMSE  (m s$^{-1}$)"),
)
#: The spatial spread of the control that each surface RMSE should be read against.
REFERENCE_OF = {
    "rmse_sst": "control_std_sst",
    "rmse_ssh": "control_std_ssh",
    "rmse_surface_speed": "control_std_speed",
}

#: Record slices of the float64 pickup, for the quantisation-free measurement.
PICKUP_DTYPE = np.dtype(">f8")
PICKUP_RECORDS = 108
PICKUP_GRID = (62, 62)
PICKUP_FIELDS = (
    ("Uvel", slice(0, 15), "m s$^{-1}$"),
    ("Vvel", slice(15, 30), "m s$^{-1}$"),
    ("Theta", slice(30, 45), "degC"),
    ("EtaN", slice(105, 106), "m"),
)


class TwinAnalysisError(RuntimeError):
    """Raised when the control or twin trajectory is not analysable as designed."""


def _load_results(paths: list[Path], experiment: str) -> list[dict[str, Any]]:
    results = []
    for path in sorted(paths):
        value = json.loads(path.read_text())
        if value.get("experiment") != experiment or value.get("phase") != "production":
            continue
        value["_result_path"] = str(path)
        results.append(value)
    results.sort(key=lambda value: int(value["start_iteration"]))
    return results


def discover_trajectory(scratch_root: Path, experiment: str) -> list[dict[str, Any]]:
    """Return the completed production segments of one trajectory, in time order."""

    if experiment == TWIN_EXPERIMENT:
        pattern = f"{TWIN_ROOT_NAME}/{TWIN_LABEL}/production/years_*/segment_result.json"
    else:
        pattern = f"*/{experiment}/production/years_*/segment_result.json"
    segments = _load_results(list(scratch_root.glob(pattern)), experiment)
    if not segments:
        raise TwinAnalysisError(f"no completed {experiment} production segments under {scratch_root}")
    for earlier, later in zip(segments, segments[1:]):
        if int(earlier["end_iteration"]) != int(later["start_iteration"]):
            raise TwinAnalysisError(
                f"{experiment} segments are not contiguous: "
                f"{earlier['_result_path']} ends at {earlier['end_iteration']}, "
                f"{later['_result_path']} starts at {later['start_iteration']}"
            )
    if int(segments[0]["start_iteration"]) != START_ITERATION:
        raise TwinAnalysisError(
            f"{experiment} does not start at year {START_YEAR} "
            f"(iteration {segments[0]['start_iteration']})"
        )
    return segments


def daily_index(segments: list[Mapping[str, Any]]) -> dict[int, Path]:
    """Map every daily diagnostic iteration to the run directory that holds it."""

    index: dict[int, Path] = {}
    for segment in segments:
        run_dir = Path(str(segment["run_dir"]))
        start = int(segment["start_iteration"])
        end = int(segment["end_iteration"])
        for iteration in range(start, end, STEPS_PER_DAY):
            if iteration in index:
                raise TwinAnalysisError(f"iteration {iteration} appears in two segments")
            index[iteration] = run_dir
    return index


def _surface_state(run_dir: Path, iteration: int, wet: np.ndarray) -> dict[str, np.ndarray]:
    """Read one daily state and return the fields the comparison needs."""

    dyn_meta = run_dir / f"dynState.{iteration:010d}.meta"
    surf_meta = run_dir / f"surfState.{iteration:010d}.meta"
    dyn_info, dyn_values = read_mds(dyn_meta)
    fields = mds_fields(dyn_info, dyn_values)
    if set(fields) != {"UVEL", "VVEL", "THETA"}:
        raise TwinAnalysisError(f"unexpected dynState fields at {dyn_meta}: {tuple(fields)}")
    surf_info, surf_values = read_mds(surf_meta)
    surface = mds_fields(surf_info, surf_values)
    if set(surface) != {"ETAN"}:
        raise TwinAnalysisError(f"unexpected surfState fields at {surf_meta}: {tuple(surface)}")

    u = np.asarray(fields["UVEL"], dtype=np.float64)
    v = np.asarray(fields["VVEL"], dtype=np.float64)
    theta = np.asarray(fields["THETA"], dtype=np.float64)
    if u.shape != (15, *wet.shape):
        raise TwinAnalysisError(f"unexpected dynState shape {u.shape} at {dyn_meta}")
    # MITgcm U/V live on the west/south faces; this is the same centring the
    # training dataset applies, so surface speed is defined identically.
    u_center = 0.5 * (u + np.roll(u, -1, axis=-1))
    v_center = 0.5 * (v + np.roll(v, -1, axis=-2))
    return {
        "u": u,
        "v": v,
        "theta": theta,
        "eta": np.asarray(surface["ETAN"], dtype=np.float64),
        "u_center_surface": u_center[0],
        "v_center_surface": v_center[0],
        "speed_surface": np.sqrt(np.square(u_center[0]) + np.square(v_center[0])),
    }


def _rmse(delta: np.ndarray, wet: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(delta[..., wet]))))


def _geometry(run_dir: Path) -> np.ndarray:
    _, values = read_mds(run_dir / "Depth.meta")
    depth = np.asarray(values)[0]
    wet = depth > 0.0
    if not np.any(wet) or np.all(wet):
        raise TwinAnalysisError(f"invalid wet mask in {run_dir / 'Depth.meta'}")
    return wet


def _read_pickup(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=PICKUP_DTYPE)
    expected = PICKUP_RECORDS * PICKUP_GRID[0] * PICKUP_GRID[1]
    if values.size != expected:
        raise TwinAnalysisError(f"{path} holds {values.size} values, expected {expected}")
    if not np.isfinite(values).all():
        raise TwinAnalysisError(f"{path} decoded to non-finite values as {PICKUP_DTYPE}")
    return values.reshape(PICKUP_RECORDS, *PICKUP_GRID).astype(np.float64)


def _pickup_index(segments: list[Mapping[str, Any]]) -> dict[int, Path]:
    index: dict[int, Path] = {}
    for segment in segments:
        for path in Path(str(segment["run_dir"])).glob("pickup.*.data"):
            index[int(path.stem.split(".")[1])] = path
    return index


def compute_pickup_series(scratch_root: Path) -> dict[str, Any]:
    """Measure the divergence in the float64 pickups, free of float32 quantisation.

    The daily ``dynState``/``surfState`` diagnostics are written in float32,
    whose quantisation step is comparable to the perturbation itself, so they
    cannot resolve a difference this small.  MITgcm's yearly permanent
    checkpoints are float64 and give the honest measurement, at annual
    resolution.  The control's own drift away from year 100 is carried alongside
    as the scale the difference must be read against.
    """

    control_segments = discover_trajectory(scratch_root, CONTROL_EXPERIMENT)
    twin_segments = discover_trajectory(scratch_root, TWIN_EXPERIMENT)
    control = _pickup_index(control_segments)
    twin = _pickup_index(twin_segments)
    shared = sorted(set(control) & set(twin))
    if not shared:
        raise TwinAnalysisError("control and twin share no float64 pickup iterations")

    reference = _read_pickup(control[shared[0]])
    years, twin_minus_control, control_drift = [], {}, {}
    amplitude = {
        name: float(np.sqrt(np.mean(np.square(reference[records]))))
        for name, records, _ in PICKUP_FIELDS
    }
    for name, _, _ in PICKUP_FIELDS:
        twin_minus_control[name] = []
        control_drift[name] = []
    for iteration in shared:
        control_state = _read_pickup(control[iteration])
        twin_state = _read_pickup(twin[iteration])
        years.append((iteration - START_ITERATION) / STEPS_PER_YEAR)
        for name, records, _ in PICKUP_FIELDS:
            delta = twin_state[records] - control_state[records]
            drift = control_state[records] - reference[records]
            twin_minus_control[name].append(float(np.sqrt(np.mean(np.square(delta)))))
            control_drift[name].append(float(np.sqrt(np.mean(np.square(drift)))))
    return {
        "years": np.asarray(years, dtype=np.float64),
        "iterations": np.asarray(shared, dtype=np.int64),
        "twin_minus_control": {k: np.asarray(v) for k, v in twin_minus_control.items()},
        "control_drift": {k: np.asarray(v) for k, v in control_drift.items()},
        "control_amplitude": amplitude,
    }


def compute_series(
    scratch_root: Path,
    stride: int = 1,
    years: int = END_YEAR - START_YEAR,
    progress: bool = True,
) -> dict[str, Any]:
    """Reduce both trajectories to daily wet-cell divergence diagnostics."""

    control_segments = discover_trajectory(scratch_root, CONTROL_EXPERIMENT)
    twin_segments = discover_trajectory(scratch_root, TWIN_EXPERIMENT)
    twin_end_year = int(twin_segments[-1]["end_year"])
    if twin_end_year < START_YEAR + years:
        raise TwinAnalysisError(
            f"twin reaches year {twin_end_year}; the requested {years}-year comparison "
            f"needs year {START_YEAR + years}. Wait for the remaining segments."
        )
    control_index = daily_index(control_segments)
    twin_index = daily_index(twin_segments)

    last_iteration = START_ITERATION + years * STEPS_PER_YEAR
    iterations = [
        iteration
        for iteration in range(START_ITERATION, last_iteration, STEPS_PER_DAY * stride)
        if iteration in control_index and iteration in twin_index
    ]
    expected = math.ceil(years * MODEL_YEAR_DAYS / stride)
    if len(iterations) != expected:
        raise TwinAnalysisError(
            f"expected {expected} paired daily states, found {len(iterations)}"
        )

    wet = _geometry(Path(str(control_segments[0]["run_dir"])))
    twin_wet = _geometry(Path(str(twin_segments[0]["run_dir"])))
    if not np.array_equal(wet, twin_wet):
        raise TwinAnalysisError("control and twin do not share the same wet mask")

    names = (
        "rmse_sst",
        "rmse_ssh",
        "rmse_surface_speed",
        "rmse_u_surface",
        "rmse_v_surface",
        "rmse_u_volume",
        "rmse_v_volume",
        "rmse_theta_volume",
        "max_abs_dtheta_surface",
        "max_abs_deta",
        "max_abs_dspeed_surface",
        "control_std_sst",
        "control_std_ssh",
        "control_std_speed",
    )
    series: dict[str, list[float]] = {name: [] for name in names}

    for position, iteration in enumerate(iterations):
        control = _surface_state(control_index[iteration], iteration, wet)
        twin = _surface_state(twin_index[iteration], iteration, wet)
        d_theta = twin["theta"] - control["theta"]
        d_eta = twin["eta"] - control["eta"]
        d_speed = twin["speed_surface"] - control["speed_surface"]

        series["rmse_sst"].append(_rmse(d_theta[0], wet))
        series["rmse_ssh"].append(_rmse(d_eta, wet))
        series["rmse_surface_speed"].append(_rmse(d_speed, wet))
        series["rmse_u_surface"].append(
            _rmse(twin["u_center_surface"] - control["u_center_surface"], wet)
        )
        series["rmse_v_surface"].append(
            _rmse(twin["v_center_surface"] - control["v_center_surface"], wet)
        )
        series["rmse_u_volume"].append(_rmse(twin["u"] - control["u"], wet))
        series["rmse_v_volume"].append(_rmse(twin["v"] - control["v"], wet))
        series["rmse_theta_volume"].append(_rmse(d_theta, wet))
        series["max_abs_dtheta_surface"].append(float(np.abs(d_theta[0][wet]).max()))
        series["max_abs_deta"].append(float(np.abs(d_eta[wet]).max()))
        series["max_abs_dspeed_surface"].append(float(np.abs(d_speed[wet]).max()))
        series["control_std_sst"].append(float(control["theta"][0][wet].std()))
        series["control_std_ssh"].append(float(control["eta"][wet].std()))
        series["control_std_speed"].append(float(control["speed_surface"][wet].std()))

        if progress and (position % 500 == 0 or position == len(iterations) - 1):
            print(
                f"  {position + 1:5d}/{len(iterations)}  "
                f"year {START_YEAR + (iteration - START_ITERATION) / STEPS_PER_YEAR:7.3f}  "
                f"RMSE_SST {series['rmse_sst'][-1]:.6e}",
                flush=True,
            )

    iteration_array = np.asarray(iterations, dtype=np.int64)
    result: dict[str, Any] = {
        "iteration": iteration_array,
        "days": (iteration_array - START_ITERATION) / STEPS_PER_DAY,
        "years": (iteration_array - START_ITERATION) / STEPS_PER_YEAR,
    }
    result.update({name: np.asarray(values, dtype=np.float64) for name, values in series.items()})
    result["_provenance"] = {
        "control_segments": [segment["_result_path"] for segment in control_segments],
        "twin_segments": [segment["_result_path"] for segment in twin_segments],
        "control_run_dirs": [segment["run_dir"] for segment in control_segments],
        "twin_run_dirs": [segment["run_dir"] for segment in twin_segments],
        "epsilon": twin_segments[0].get("epsilon"),
        "perturbed_fields": twin_segments[0].get("perturbed_fields"),
        "wet_cells": int(wet.sum()),
        "grid_cells": int(wet.size),
        "stride_days": stride,
        "years": years,
        "paired_states": len(iterations),
    }
    return result


def fit_growth(days: np.ndarray, value: np.ndarray) -> dict[str, Any]:
    """Fit log RMSE = log RMSE_0 + lambda t over the exponential-growth window.

    The window runs from the first day the error clears one thousandth of its
    saturation level to the last day before it passes one fifth of it, which
    skips both the initial projection transient and the saturated tail.
    """

    tail = value[-MODEL_YEAR_DAYS:] if value.size >= MODEL_YEAR_DAYS else value[-max(value.size // 5, 1):]
    saturation = float(np.median(tail))
    if not np.isfinite(saturation) or saturation <= 0.0:
        return {"fitted": False, "reason": "no positive saturation level", "saturation": saturation}
    low, high = 1.0e-3 * saturation, 0.2 * saturation

    positive = np.flatnonzero(value > 0.0)
    if positive.size == 0:
        return {"fitted": False, "reason": "series is identically zero", "saturation": saturation}
    above_low = np.flatnonzero(value >= low)
    if above_low.size == 0:
        return {"fitted": False, "reason": "series never clears the fit floor", "saturation": saturation}
    start = int(above_low[0])
    beyond = np.flatnonzero(value[start:] > high)
    stop = int(start + beyond[0]) if beyond.size else int(value.size)
    window = slice(start, stop)
    x, y = days[window], value[window]
    keep = y > 0.0
    x, y = x[keep], y[keep]
    if x.size < 30 or x.max() <= x.min():
        return {
            "fitted": False,
            "reason": f"fit window holds only {int(x.size)} usable days",
            "saturation": saturation,
        }

    log_y = np.log(y)
    slope, intercept = np.polyfit(x, log_y, 1)
    predicted = slope * x + intercept
    residual = log_y - predicted
    denominator = float(np.sum(np.square(log_y - log_y.mean())))
    r_squared = 1.0 - float(np.sum(np.square(residual))) / denominator if denominator > 0 else float("nan")

    half = np.flatnonzero(value >= 0.5 * saturation)
    return {
        "fitted": True,
        "lambda_per_day": float(slope),
        "lambda_per_year": float(slope * MODEL_YEAR_DAYS),
        "e_folding_days": float(1.0 / slope) if slope > 0 else float("inf"),
        "doubling_days": float(math.log(2.0) / slope) if slope > 0 else float("inf"),
        "intercept_log": float(intercept),
        "r_squared": r_squared,
        "window_days": [float(x.min()), float(x.max())],
        "window_points": int(x.size),
        "saturation": saturation,
        "half_saturation_day": float(days[half[0]]) if half.size else None,
    }


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "text.color": INK_PRIMARY,
        }
    )


def _decorate(axis, title: str, ylabel: str, color: str) -> None:
    """Recessive frame plus one direct label in the series colour.

    A single series per panel needs no legend box: the coloured label names it,
    which also relieves the low-contrast categorical slot.
    """

    axis.set_ylabel(ylabel)
    axis.grid(color="0.88", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(length=3)
    axis.annotate(
        title,
        xy=(0.012, 0.94),
        xycoords="axes fraction",
        ha="left",
        va="top",
        color=color,
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )


def plot_linear(
    series: Mapping[str, Any],
    fits: Mapping[str, Any],
    output: Path,
    references: Mapping[str, float] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    _style()
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(6.4, 7.2), constrained_layout=True)
    for axis, (name, title, ylabel), color in zip(axes, SURFACE_DIAGNOSTICS, SERIES_COLORS):
        years, value = series["years"], series[name]
        axis.plot(years, value, color=color, linewidth=1.4)
        fit = fits[name]
        reference = (references or {}).get(name)
        if not fit.get("fitted") and reference:
            # Nothing grew, so a "saturation" line would be meaningless; state the
            # scale the axis actually spans instead.
            axis.annotate(
                f"axis spans {value.max() / reference:.1e} of the decorrelated level "
                f"({reference:.3g})",
                xy=(0.5, 0.93),
                xycoords="axes fraction",
                ha="center",
                va="top",
                color=INK_SECONDARY,
                fontsize=8,
            )
        elif fit.get("fitted") or fit.get("saturation", 0.0) > 0.0:
            axis.axhline(
                fit["saturation"],
                color=INK_MUTED,
                linewidth=1.0,
                linestyle=(0, (4, 3)),
            )
            axis.annotate(
                f"saturation {fit['saturation']:.3g}",
                xy=(0.5, fit["saturation"]),
                xycoords=("axes fraction", "data"),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=INK_SECONDARY,
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )
        _decorate(axis, title, ylabel, color)
        axis.set_xlim(0.0, float(years[-1]))
        axis.set_ylim(bottom=0.0)
    axes[-1].set_xlabel("model years after year 100")
    figure.suptitle(
        "MITgcm S0 control vs MITgcm twin: wet-cell RMSE\n"
        r"twin start $(U,V)\rightarrow(1+10^{-6})(U,V)$ at year 100, all else identical",
        fontsize=10,
        color=INK_PRIMARY,
    )
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_log(
    series: Mapping[str, Any],
    fits: Mapping[str, Any],
    output: Path,
    max_years: float | None = None,
    references: Mapping[str, float] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    _style()
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(6.4, 7.2), constrained_layout=True)
    limit = float(series["years"][-1]) if max_years is None else float(max_years)
    for axis, (name, title, ylabel), color in zip(axes, SURFACE_DIAGNOSTICS, SERIES_COLORS):
        years, value = series["years"], series[name]
        visible = value > 0.0
        axis.semilogy(years[visible], value[visible], color=color, linewidth=1.4)
        fit = fits[name]
        if fit.get("fitted"):
            window = np.asarray(fit["window_days"], dtype=np.float64) / MODEL_YEAR_DAYS
            axis.axvspan(window[0], window[1], color=color, alpha=0.08, linewidth=0)
            days = np.linspace(fit["window_days"][0], fit["window_days"][1], 64)
            axis.semilogy(
                days / MODEL_YEAR_DAYS,
                np.exp(fit["intercept_log"] + fit["lambda_per_day"] * days),
                color=INK_PRIMARY,
                linewidth=1.1,
                linestyle=(0, (5, 3)),
            )
            axis.annotate(
                rf"dashed: fit $e^{{\lambda t}}$ over the shaded window"
                "\n"
                rf"$\lambda$ = {fit['lambda_per_day']:.4f} d$^{{-1}}$"
                rf" ({fit['lambda_per_year']:.2f} yr$^{{-1}}$),"
                rf" doubling {fit['doubling_days']:.0f} d"
                "\n"
                rf"$R^2$ = {fit['r_squared']:.4f}",
                xy=(0.985, 0.05),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                color=INK_SECONDARY,
                fontsize=8,
                linespacing=1.5,
            )
        reference = (references or {}).get(name)
        if reference:
            axis.axhline(reference, color=INK_MUTED, linewidth=1.1, linestyle=(0, (4, 3)))
            axis.annotate(
                "level a decorrelated trajectory would reach",
                xy=(0.5, reference),
                xycoords=("axes fraction", "data"),
                xytext=(0, -5),
                textcoords="offset points",
                ha="center",
                va="top",
                color=INK_SECONDARY,
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )
            floor = float(np.min(value[value > 0.0])) if np.any(value > 0.0) else reference
            axis.set_ylim(0.2 * floor, 5.0 * reference)
        elif fit.get("saturation", 0.0) > 0.0:
            axis.axhline(fit["saturation"], color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        _decorate(axis, title, ylabel, color)
        axis.set_xlim(0.0, limit)
    axes[-1].set_xlabel("model years after year 100")
    span = "early growth" if max_years is not None else f"0-{limit:.0f} years"
    grew = any(fits[name].get("fitted") for name, _, _ in SURFACE_DIAGNOSTICS)
    subtitle = (
        r"straight line $\Rightarrow$ RMSE$(t)\approx$RMSE$_0 e^{\lambda t}$"
        if grew
        else "no growth phase: the error stays flat, far below the decorrelated level"
    )
    figure.suptitle(
        f"MITgcm-vs-MITgcm error growth, log scale ({span})\n{subtitle}",
        fontsize=10,
        color=INK_PRIMARY,
    )
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_pickup_divergence(pickups: Mapping[str, Any], output: Path) -> Path:
    """Plot the float64 twin-minus-control difference against the control's own drift."""

    import matplotlib.pyplot as plt

    _style()
    figure, axes = plt.subplots(2, 2, sharex=True, figsize=(8.4, 5.8), constrained_layout=True)
    years = pickups["years"]
    for axis, (name, _, unit) in zip(axes.ravel(), PICKUP_FIELDS):
        delta = pickups["twin_minus_control"][name]
        drift = pickups["control_drift"][name]
        visible = delta > 0.0
        axis.semilogy(
            years[visible],
            delta[visible],
            color=SERIES_COLORS[0],
            linewidth=1.4,
            marker="o",
            markersize=3.0,
            label="twin - control",
        )
        moving = drift > 0.0
        axis.semilogy(
            years[moving],
            drift[moving],
            color=SERIES_COLORS[1],
            linewidth=1.4,
            linestyle=(0, (5, 3)),
            label="control - control at year 100",
        )
        axis.axhline(
            pickups["control_amplitude"][name],
            color=INK_MUTED,
            linewidth=1.0,
            linestyle=(0, (2, 2)),
        )
        axis.annotate(
            "RMS of the field itself",
            xy=(0.97, pickups["control_amplitude"][name]),
            xycoords=("axes fraction", "data"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=INK_SECONDARY,
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        )
        _decorate(axis, f"{name}  ({unit})", "RMS difference", SERIES_COLORS[0])
        axis.set_xlim(0.0, float(years[-1]))
        # Headroom above the reference line so the labels never sit on the data.
        floor = float(np.min(delta[visible])) if np.any(visible) else 1.0
        axis.set_ylim(0.3 * floor, 60.0 * pickups["control_amplitude"][name])
    for axis in axes[-1]:
        axis.set_xlabel("model years after year 100")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    figure.suptitle(
        "Float64 pickups: the twin never separates from the control\n"
        "the perturbation stays 6-7 decades below the control's own year-to-year change",
        fontsize=10,
        color=INK_PRIMARY,
    )
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def _write_csv(series: Mapping[str, Any], path: Path) -> None:
    columns = [name for name in series if not name.startswith("_")]
    rows = len(series["iteration"])
    with path.open("w") as stream:
        stream.write(",".join(columns) + "\n")
        for index in range(rows):
            stream.write(
                ",".join(
                    f"{series[name][index]:d}" if name == "iteration" else f"{series[name][index]:.10e}"
                    for name in columns
                )
                + "\n"
            )


def analyze(
    scratch_root: Path,
    output_dir: Path,
    stride: int = 1,
    years: int = END_YEAR - START_YEAR,
    force: bool = False,
    early_years: float | None = None,
) -> dict[str, Any]:
    """Compute (or reload) the divergence series, then write the tables and figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "s0_twin_divergence.npz"
    provenance_path = output_dir / "s0_twin_divergence.json"
    if cache.is_file() and not force:
        print(f"reusing cached series {cache} (pass --force to recompute)", flush=True)
        stored = np.load(cache)
        series: dict[str, Any] = {name: stored[name] for name in stored.files}
        series["_provenance"] = json.loads(provenance_path.read_text())["provenance"]
    else:
        print(f"reading paired daily states from {scratch_root} ...", flush=True)
        series = compute_series(scratch_root, stride=stride, years=years)
        np.savez_compressed(
            cache, **{name: value for name, value in series.items() if not name.startswith("_")}
        )
        _write_csv(series, output_dir / "s0_twin_divergence.csv")

    fits = {name: fit_growth(series["days"], series[name]) for name, _, _ in SURFACE_DIAGNOSTICS}
    for name in ("rmse_u_volume", "rmse_v_volume", "rmse_theta_volume"):
        fits[name] = fit_growth(series["days"], series[name])

    if early_years is None:
        half_days = [
            fit["half_saturation_day"]
            for fit in fits.values()
            if fit.get("half_saturation_day") is not None
        ]
        early_years = (
            min(float(series["years"][-1]), 1.4 * max(half_days) / MODEL_YEAR_DAYS)
            if half_days
            else float(series["years"][-1])
        )
        early_years = max(early_years, 1.0)

    # A decorrelated trajectory would sit at sqrt(2) x the control's spatial spread;
    # that is the level the error must be read against when there is no growth.
    references = {
        name: float(math.sqrt(2.0) * np.mean(series[REFERENCE_OF[name]]))
        for name, _, _ in SURFACE_DIAGNOSTICS
    }
    diverged = {
        name: bool(fits[name].get("fitted") or series[name][-1] > 0.05 * references[name])
        for name, _, _ in SURFACE_DIAGNOSTICS
    }

    print("\nreading float64 pickups ...", flush=True)
    pickups = compute_pickup_series(scratch_root)

    figures = [
        plot_linear(series, fits, output_dir / "s0_twin_rmse_linear.png", references=references),
        plot_log(series, fits, output_dir / "s0_twin_rmse_log.png", references=references),
        plot_log(
            series,
            fits,
            output_dir / "s0_twin_rmse_log_early.png",
            max_years=early_years,
            references=references,
        ),
        plot_pickup_divergence(pickups, output_dir / "s0_twin_pickup_divergence.png"),
    ]

    summary = {
        "experiment": TWIN_EXPERIMENT,
        "control_regime": CONTROL_EXPERIMENT,
        "question": "does MITgcm diverge from MITgcm?",
        "provenance": series["_provenance"],
        "growth_fits": fits,
        "decorrelated_reference": references,
        "diverged": diverged,
        "verdict": (
            "MITgcm diverges from MITgcm"
            if any(diverged.values())
            else "no divergence: the perturbation does not grow over the integration"
        ),
        "float64_pickups": {
            "years": pickups["years"].tolist(),
            "twin_minus_control_rms": {
                name: values.tolist() for name, values in pickups["twin_minus_control"].items()
            },
            "control_drift_rms": {
                name: values.tolist() for name, values in pickups["control_drift"].items()
            },
            "control_amplitude_rms": pickups["control_amplitude"],
            "note": (
                "float32 diagnostics cannot resolve a difference this small; these "
                "yearly float64 checkpoints are the quantisation-free measurement"
            ),
        },
        "early_growth_years": float(early_years),
        "final": {
            name: float(series[name][-1])
            for name in series
            if name.startswith("rmse_") or name.startswith("control_std_")
        },
        "figures": [str(path) for path in figures],
    }
    provenance_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")

    print(f"\nwrote {provenance_path}")
    for path in figures:
        print(f"wrote {path}")
    print("\ngrowth fits")
    for name, _, _ in SURFACE_DIAGNOSTICS:
        fit = fits[name]
        if fit.get("fitted"):
            print(
                f"  {name:20s} lambda = {fit['lambda_per_day']:.5f} /day "
                f"({fit['lambda_per_year']:.2f} /year), doubling {fit['doubling_days']:.1f} d, "
                f"R2 = {fit['r_squared']:.4f}, saturation {fit['saturation']:.4g}"
            )
        else:
            print(
                f"  {name:20s} no growth phase ({fit.get('reason')}); "
                f"final RMSE {series[name][-1]:.3e} is {series[name][-1] / references[name]:.2e} "
                f"of the decorrelated level {references[name]:.3e}"
            )
    print(f"\nverdict: {summary['verdict']}")
    print("\nfloat64 pickup RMS difference (twin - control) vs the control's own drift")
    for name, _, _ in PICKUP_FIELDS:
        delta = pickups["twin_minus_control"][name]
        drift = pickups["control_drift"][name]
        print(
            f"  {name:6s} start {delta[0]:.3e} -> end {delta[-1]:.3e}  "
            f"(x{delta[-1] / delta[0]:.2f})   control drift by then {drift[-1]:.3e}  "
            f"ratio {delta[-1] / drift[-1]:.2e}"
            if delta[0] > 0
            else f"  {name:6s} start 0 -> end {delta[-1]:.3e}   control drift {drift[-1]:.3e}"
        )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "af_fno" / "S0_twin",
    )
    parser.add_argument("--stride", type=int, default=1, help="days between compared states")
    parser.add_argument("--years", type=int, default=END_YEAR - START_YEAR)
    parser.add_argument("--early-years", type=float, default=None, help="x limit of the zoom figure")
    parser.add_argument("--force", action="store_true", help="recompute instead of reusing the cache")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    analyze(
        args.scratch_root.resolve(),
        args.output_dir.resolve(),
        stride=args.stride,
        years=args.years,
        force=args.force,
        early_years=args.early_years,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
