"""Training-only consistency audit for barotropic streamfunction diagnostics.

The audit uses raw MITgcm C-grid face velocities from one prospectively fixed
training year in each S0--S2 continuation.  It constructs transport
streamfunctions independently from U and V, measures path mismatch and
transport divergence for daily and time-averaged states, and verifies that the
project's centered-U operator is the deterministic counterpart of the
tutorial's U-derived diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from .af_data_v2 import inventory_extension
from .af_forward_complete import _barotropic_streamfunction
from .af_s0 import _sha256
from .af_trajectory_expansion import EXPERIMENTS, load_expansion_contract
from .af_tutorial_analysis import DRF_M, EARTH_RADIUS_M
from .mds import mds_fields, read_mds


VERSION = "streamfunction_consistency_v1"
REPORT_NAME = "streamfunction_consistency_report.json"
ARRAYS_NAME = "streamfunction_consistency_arrays.npz"
SUMMARY_FIGURE_NAME = "streamfunction_consistency_vs_averaging.png"
MAP_FIGURE_NAME = "streamfunction_daily_and_annual_uv.png"
MANIFEST_NAME = "figure_manifest.json"
README_NAME = "README.md"
SECONDS_PER_DAY = 86_400.0
SV_SCALE = 1.0e6


class StreamfunctionConsistencyError(RuntimeError):
    """Raised when the frozen streamfunction audit contract is violated."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_streamfunction_consistency_contract(
    path: str | Path,
    *,
    verify_source_files: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the audit contract frozen before daily transport metrics."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != VERSION:
        raise ValueError(f"expected streamfunction contract {VERSION}")
    if (
        contract.get("contract_status")
        != "frozen_before_raw_daily_uv_streamfunction_consistency_metrics"
    ):
        raise ValueError("streamfunction consistency contract was not frozen")
    audit = contract.get("audit", {})
    if (
        tuple(audit.get("regimes", ())) != EXPERIMENTS
        or int(audit.get("global_start_index", -1)) != 3690
        or int(audit.get("extension_local_start_index", -1)) != 90
        or int(audit.get("days", -1)) != 360
        or tuple(audit.get("averaging_windows_days", ()))
        != (1, 5, 30, 90, 360)
        or audit.get("velocity_source") != "raw_mitgcm_c_grid_faces"
        or audit.get("u_path") != "south_to_north"
        or audit.get("v_path") != "west_to_east"
        or audit.get("gauge_alignment")
        != "wet_cell_mean_difference_removed_per_state"
    ):
        raise ValueError("streamfunction audit design changed")
    thresholds = contract.get("interpretation_thresholds", {})
    if (
        float(thresholds.get("project_operator_metric_rmse_fraction", -1.0))
        != 0.005
        or float(thresholds.get("annual_uv_rmse_fraction", -1.0)) != 0.05
        or float(thresholds.get("annual_uv_minimum_correlation", -1.0))
        != 0.99
        or tuple(thresholds.get("annual_uv_amplitude_ratio_interval", ()))
        != (0.95, 1.05)
        or thresholds.get("averaging_improvement_required") is not True
    ):
        raise ValueError("streamfunction interpretation thresholds changed")
    read = contract.get("read_contract", {})
    if (
        read.get("training_snapshot_code") != 1
        or read.get("raw_training_state_read") is not True
        or any(
            read.get(name) is not False
            for name in (
                "validation_state_read",
                "inference_read",
                "intermediate_wind_read",
                "response_read",
                "adjoint_read",
            )
        )
    ):
        raise ValueError("streamfunction read contract changed")
    if verify_source_files:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _sha256(source) != expected:
                raise ValueError(
                    f"streamfunction consistency source changed: {source}"
                )
    return contract, resolved, _sha256(resolved)


def _scalar_mds(path: Path) -> np.ndarray:
    _, values = read_mds(path)
    value = np.asarray(values, dtype=np.float64).squeeze()
    if value.shape != (62, 62):
        raise ValueError(f"unexpected grid field shape in {path}: {value.shape}")
    return value


def depth_integrated_velocity(
    velocity: np.ndarray,
) -> np.ndarray:
    """Depth integrate a 15-level face velocity in m2 s-1."""

    values = np.asarray(velocity, dtype=np.float64)
    if values.shape != (15, 62, 62):
        raise ValueError("velocity must have shape (15, 62, 62)")
    return np.sum(values * DRF_M[:, None, None], axis=0)


def u_derived_streamfunction(
    u_bt: np.ndarray,
    dyg: np.ndarray,
    *,
    center_x: bool = True,
) -> np.ndarray:
    """Integrate U transport northward and return Sv."""

    if u_bt.shape != (62, 62) or dyg.shape != (62, 62):
        raise ValueError("U transport and DYG must be 62 x 62")
    face = np.cumsum(-u_bt * dyg, axis=0) / SV_SCALE
    if center_x:
        face = 0.5 * (face + np.roll(face, -1, axis=1))
    return face


def v_derived_streamfunction(
    v_bt: np.ndarray,
    dxg: np.ndarray,
    *,
    center_y: bool = True,
) -> np.ndarray:
    """Integrate V transport eastward and return Sv."""

    if v_bt.shape != (62, 62) or dxg.shape != (62, 62):
        raise ValueError("V transport and DXG must be 62 x 62")
    face = np.cumsum(v_bt * dxg, axis=1) / SV_SCALE
    if center_y:
        face = 0.5 * (face + np.roll(face, -1, axis=0))
    return face


def transport_divergence_volume(
    u_bt: np.ndarray,
    v_bt: np.ndarray,
    dyg: np.ndarray,
    dxg: np.ndarray,
) -> np.ndarray:
    """Return C-grid horizontal transport divergence in m3 s-1 per cell."""

    if any(value.shape != (62, 62) for value in (u_bt, v_bt, dyg, dxg)):
        raise ValueError("transport divergence arrays must be 62 x 62")
    u_volume = u_bt * dyg
    v_volume = v_bt * dxg
    return (
        np.roll(u_volume, -1, axis=1)
        - u_volume
        + np.roll(v_volume, -1, axis=0)
        - v_volume
    )


def uv_streamfunction_metrics(
    psi_u: np.ndarray,
    psi_v: np.ndarray,
    wet: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Gauge-align two path integrals and return mismatch diagnostics."""

    if (
        psi_u.shape != (62, 62)
        or psi_v.shape != psi_u.shape
        or wet.shape != psi_u.shape
        or not np.any(wet)
    ):
        raise ValueError("streamfunction metric arrays are inconsistent")
    offset = float(np.mean((psi_u - psi_v)[wet]))
    aligned_v = psi_v + offset
    difference = psi_u - aligned_v
    u_values = psi_u[wet]
    v_values = aligned_v[wet]
    rmse = float(np.sqrt(np.mean(np.square(difference[wet]))))
    u_rms = float(np.sqrt(np.mean(np.square(u_values))))
    correlation = float(np.corrcoef(u_values, v_values)[0, 1])
    u_amplitude = float(np.max(np.abs(u_values)))
    v_amplitude = float(np.max(np.abs(v_values)))
    return {
        "gauge_offset_sv": offset,
        "rmse_sv": rmse,
        "relative_rmse": rmse / max(u_rms, np.finfo(float).eps),
        "correlation": correlation,
        "amplitude_ratio_v_to_u": v_amplitude
        / max(u_amplitude, np.finfo(float).eps),
        "aligned_v": aligned_v,
        "difference": difference,
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("summary received no finite values")
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "maximum": float(np.max(finite)),
    }


def _window_metrics(
    u_bt: np.ndarray,
    v_bt: np.ndarray,
    dyg: np.ndarray,
    dxg: np.ndarray,
    rac: np.ndarray,
    wet: np.ndarray,
    window: int,
) -> dict[str, np.ndarray]:
    if u_bt.shape != v_bt.shape or u_bt.shape[0] % window:
        raise ValueError("averaging window must partition the selected year")
    result = {
        "relative_rmse": [],
        "correlation": [],
        "amplitude_ratio": [],
        "gauge_offset_sv": [],
        "divergence_rms_m_per_day": [],
    }
    for start in range(0, u_bt.shape[0], window):
        stop = start + window
        mean_u = np.mean(u_bt[start:stop], axis=0)
        mean_v = np.mean(v_bt[start:stop], axis=0)
        psi_u = u_derived_streamfunction(mean_u, dyg)
        psi_v = v_derived_streamfunction(mean_v, dxg)
        metrics = uv_streamfunction_metrics(psi_u, psi_v, wet)
        divergence = transport_divergence_volume(
            mean_u,
            mean_v,
            dyg,
            dxg,
        )
        divergence_m_per_day = (
            divergence / rac * SECONDS_PER_DAY
        )
        result["relative_rmse"].append(metrics["relative_rmse"])
        result["correlation"].append(metrics["correlation"])
        result["amplitude_ratio"].append(
            metrics["amplitude_ratio_v_to_u"]
        )
        result["gauge_offset_sv"].append(metrics["gauge_offset_sv"])
        result["divergence_rms_m_per_day"].append(
            float(
                np.sqrt(
                    np.mean(np.square(divergence_m_per_day[wet]))
                )
            )
        )
    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in result.items()
    }


def _read_velocity_eta(
    dyn_path: Path,
    surf_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dyn_meta, dyn_values = read_mds(dyn_path)
    fields = mds_fields(dyn_meta, dyn_values)
    if set(fields) != {"UVEL", "VVEL", "THETA"}:
        raise StreamfunctionConsistencyError(
            f"unexpected dynamic fields in {dyn_path}"
        )
    surf_meta, surf_values = read_mds(surf_path)
    surface = mds_fields(surf_meta, surf_values)
    if set(surface) != {"ETAN"}:
        raise StreamfunctionConsistencyError(
            f"unexpected surface fields in {surf_path}"
        )
    raw_u = np.asarray(fields["UVEL"], dtype=np.float32)
    raw_v = np.asarray(fields["VVEL"], dtype=np.float32)
    eta = np.asarray(surface["ETAN"], dtype=np.float64)
    u_center = 0.5 * (raw_u + np.roll(raw_u, -1, axis=-1))
    v_center = 0.5 * (raw_v + np.roll(raw_v, -1, axis=-2))
    return (
        depth_integrated_velocity(raw_u),
        depth_integrated_velocity(raw_v),
        eta,
        u_center,
        v_center,
    )


def _plot_summary(
    output: Path,
    window_results: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    project_metric_error: np.ndarray,
) -> None:
    windows = np.asarray((1, 5, 30, 90, 360))
    colors = ("#2F75B5", "#2A9D8F", "#C45A30")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.8, 7.2),
        constrained_layout=True,
    )
    fields = (
        ("relative_rmse", "U/V path RMSE / U RMS", "log"),
        ("correlation", "U/V spatial correlation", "linear"),
        (
            "divergence_rms_m_per_day",
            "Transport-divergence equivalent (m day$^{-1}$)",
            "log",
        ),
    )
    for axis, (field, label, scale) in zip(axes.flat[:3], fields):
        for color, regime in zip(colors, EXPERIMENTS):
            medians = []
            lower = []
            upper = []
            for window in windows:
                values = window_results[regime][int(window)][field]
                median = float(np.median(values))
                medians.append(median)
                lower.append(median - float(np.percentile(values, 10)))
                upper.append(float(np.percentile(values, 90)) - median)
            axis.errorbar(
                windows,
                medians,
                yerr=np.asarray((lower, upper)),
                color=color,
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=regime,
            )
        axis.set_xscale("log")
        axis.set_yscale(scale)
        axis.set_xticks(windows)
        axis.set_xticklabels([str(value) for value in windows])
        axis.set_xlabel("Averaging window (days)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25, which="both")
    axes[0, 0].axhline(0.05, color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].axhline(0.99, color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].legend()

    axis = axes[1, 1]
    x = np.arange(len(EXPERIMENTS))
    axis.bar(
        x,
        100.0 * project_metric_error,
        color=colors,
    )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    axis.set_xticks(x, EXPERIMENTS)
    axis.set_ylabel("Project U operator vs metric U (%)")
    axis.set_title("Centered-U implementation agreement")
    axis.grid(alpha=0.25, axis="y")
    figure.suptitle(
        "Raw MITgcm U/V transport consistency: daily to annual mean"
    )
    figure.savefig(
        output / SUMMARY_FIGURE_NAME,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_maps(
    output: Path,
    daily_u: np.ndarray,
    daily_v: np.ndarray,
    annual_u: np.ndarray,
    annual_v: np.ndarray,
    wet: np.ndarray,
) -> None:
    daily = uv_streamfunction_metrics(daily_u, daily_v, wet)
    annual = uv_streamfunction_metrics(annual_u, annual_v, wet)
    rows = (
        (
            daily_u,
            np.asarray(daily["aligned_v"]),
            np.asarray(daily["difference"]),
            "Instantaneous day",
            daily,
        ),
        (
            annual_u,
            np.asarray(annual["aligned_v"]),
            np.asarray(annual["difference"]),
            "360-day mean",
            annual,
        ),
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(11.2, 7.2),
        constrained_layout=True,
    )
    for row_index, (psi_u, psi_v, difference, label, metrics) in enumerate(rows):
        state_bound = max(
            float(np.max(np.abs(psi_u[wet]))),
            float(np.max(np.abs(psi_v[wet]))),
        )
        difference_bound = max(
            float(np.max(np.abs(difference[wet]))),
            np.finfo(float).eps,
        )
        plotted = (
            (psi_u, state_bound, "U-derived"),
            (psi_v, state_bound, "V-derived, gauge aligned"),
            (difference, difference_bound, "U minus V"),
        )
        for column, (values, bound, title) in enumerate(plotted):
            shown = np.where(wet, values, np.nan)
            image = axes[row_index, column].imshow(
                shown,
                origin="lower",
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
            )
            axes[row_index, column].set_title(f"{label}: {title}")
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            figure.colorbar(
                image,
                ax=axes[row_index, column],
                label="Sv",
                shrink=0.78,
            )
        axes[row_index, 2].set_xlabel(
            "relative RMSE="
            f"{float(metrics['relative_rmse']):.3%}; "
            f"r={float(metrics['correlation']):.4f}"
        )
    figure.suptitle(
        "S0 streamfunction path consistency on the fixed training year"
    )
    figure.savefig(
        output / MAP_FIGURE_NAME,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _preflight_sources(
    contract: Mapping[str, Any],
    dataset: Path,
) -> tuple[dict[str, Any], str, list[Any]]:
    sources = contract["source_artifacts"]
    if _sha256(dataset / ".zmetadata") != sources[
        "dataset_metadata_sha256"
    ]:
        raise StreamfunctionConsistencyError(
            "trajectory-v2 metadata changed"
        )
    expansion, _, expansion_sha = load_expansion_contract(
        sources["expansion_contract"]
    )
    if expansion_sha != sources["expansion_contract_sha256"]:
        raise StreamfunctionConsistencyError(
            "trajectory-v2 expansion contract changed"
        )
    inventories = [
        inventory_extension(
            sources["scratch_root"],
            expansion,
            expansion_sha,
            regime,
        )
        for regime in EXPERIMENTS
    ]
    group = zarr.open_consolidated(str(dataset), mode="r")
    start = int(contract["audit"]["global_start_index"])
    days = int(contract["audit"]["days"])
    codes = np.asarray(group["snapshot_split"][start : start + days])
    if codes.shape != (days,) or not np.all(codes == 1):
        raise StreamfunctionConsistencyError(
            "selected streamfunction year is not wholly training-only"
        )
    return expansion, expansion_sha, inventories


def run_streamfunction_consistency(
    dataset_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the frozen raw-face U/V consistency audit."""

    started = time.monotonic()
    contract, resolved_contract, contract_sha = (
        load_streamfunction_consistency_contract(contract_path)
    )
    dataset = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite streamfunction audit output: {output}"
        )
    _, expansion_sha, inventories = _preflight_sources(contract, dataset)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    global_start = int(contract["audit"]["global_start_index"])
    local_start = int(contract["audit"]["extension_local_start_index"])
    days = int(contract["audit"]["days"])
    windows = tuple(
        int(value)
        for value in contract["audit"]["averaging_windows_days"]
    )

    first_root = inventories[0].run_dir
    dxg = _scalar_mds(first_root / "DXG.meta")
    dyg = _scalar_mds(first_root / "DYG.meta")
    rac = _scalar_mds(first_root / "RAC.meta")
    depth = _scalar_mds(first_root / "Depth.meta")
    if (
        not np.array_equal(wet, depth > 0.0)
        or not np.all(rac[wet] > 0.0)
        or not np.all(dxg[wet] > 0.0)
        or not np.all(dyg[wet] > 0.0)
    ):
        raise StreamfunctionConsistencyError(
            "MITgcm grid metrics or wet mask changed"
        )
    for inventory in inventories[1:]:
        if any(
            not np.array_equal(
                reference,
                _scalar_mds(inventory.run_dir / name),
            )
            for reference, name in (
                (dxg, "DXG.meta"),
                (dyg, "DYG.meta"),
                (rac, "RAC.meta"),
                (depth, "Depth.meta"),
            )
        ):
            raise StreamfunctionConsistencyError(
                "streamfunction grid differs across regimes"
            )

    arrays: dict[str, np.ndarray] = {
        "global_indices": np.arange(
            global_start,
            global_start + days,
            dtype=np.int32,
        ),
        "extension_local_indices": np.arange(
            local_start,
            local_start + days,
            dtype=np.int32,
        ),
        "wet_mask": wet.astype(np.uint8),
        "dxg_m": dxg.astype(np.float32),
        "dyg_m": dyg.astype(np.float32),
        "rac_m2": rac.astype(np.float32),
    }
    window_results: dict[
        str,
        dict[int, dict[str, np.ndarray]],
    ] = {}
    regime_reports: dict[str, Any] = {}
    project_metric_errors = np.empty(len(EXPERIMENTS), dtype=np.float64)
    example_maps: dict[str, np.ndarray] = {}

    for regime_index, (regime, inventory) in enumerate(
        zip(EXPERIMENTS, inventories)
    ):
        u_bt = np.empty((days, 62, 62), dtype=np.float64)
        v_bt = np.empty_like(u_bt)
        eta = np.empty_like(u_bt)
        project_operator_difference = np.empty(days, dtype=np.float64)
        project_metric_relative = np.empty(days, dtype=np.float64)
        centered_state_exact = np.ones(days, dtype=bool)
        iterations = np.empty(days, dtype=np.int64)
        for destination, local_index in enumerate(
            range(local_start, local_start + days)
        ):
            dyn_path = inventory.dyn_meta[local_index]
            surf_path = inventory.surf_meta[local_index]
            (
                u_bt[destination],
                v_bt[destination],
                eta[destination],
                raw_u_center,
                raw_v_center,
            ) = _read_velocity_eta(dyn_path, surf_path)
            iterations[destination] = inventory.iterations[local_index]
            stored = np.asarray(
                state[
                    regime_index,
                    global_start + destination,
                    :30,
                ],
                dtype=np.float32,
            )
            centered_state_exact[destination] = bool(
                np.array_equal(stored[:15], raw_u_center)
                and np.array_equal(stored[15:30], raw_v_center)
            )
            stored_full = np.asarray(
                state[
                    regime_index,
                    global_start + destination,
                ],
                dtype=np.float32,
            )[None]
            project = _barotropic_streamfunction(stored_full, wet)[0]
            dy_constant = EARTH_RADIUS_M * np.deg2rad(1.0)
            raw_project = (
                np.cumsum(
                    -depth_integrated_velocity(raw_u_center) * dy_constant,
                    axis=0,
                )
                / SV_SCALE
            ).astype(np.float32)
            raw_project[~wet] = 0.0
            project_operator_difference[destination] = float(
                np.max(np.abs(project - raw_project))
            )
            metric_u = u_derived_streamfunction(
                u_bt[destination],
                dyg,
            )
            difference = project.astype(np.float64) - metric_u
            project_metric_relative[destination] = float(
                np.sqrt(np.mean(np.square(difference[wet])))
                / max(
                    np.sqrt(np.mean(np.square(metric_u[wet]))),
                    np.finfo(float).eps,
                )
            )

        arrays[f"{regime}__iterations"] = iterations
        arrays[f"{regime}__centered_state_exact"] = centered_state_exact
        arrays[
            f"{regime}__project_operator_max_abs_difference_sv"
        ] = project_operator_difference
        arrays[
            f"{regime}__project_metric_relative_rmse"
        ] = project_metric_relative
        project_metric_errors[regime_index] = float(
            np.max(project_metric_relative)
        )

        window_results[regime] = {}
        window_summary: dict[str, Any] = {}
        for window in windows:
            metrics = _window_metrics(
                u_bt,
                v_bt,
                dyg,
                dxg,
                rac,
                wet,
                window,
            )
            window_results[regime][window] = metrics
            window_summary[str(window)] = {
                name: _stats(values)
                for name, values in metrics.items()
            }
            for name, values in metrics.items():
                arrays[f"{regime}__window_{window:03d}__{name}"] = values

        divergence = np.stack(
            [
                transport_divergence_volume(
                    u_bt[index],
                    v_bt[index],
                    dyg,
                    dxg,
                )
                / rac
                * SECONDS_PER_DAY
                for index in range(days)
            ]
        )
        eta_change = eta[1:] - eta[:-1]
        divergence_mid = 0.5 * (divergence[1:] + divergence[:-1])
        continuity_residual = eta_change + divergence_mid
        eta_values = eta_change[:, wet].ravel()
        convergence_values = (-divergence_mid[:, wet]).ravel()
        residual_values = continuity_residual[:, wet].ravel()
        continuity = {
            "eta_change_rms_m_per_day": float(
                np.sqrt(np.mean(np.square(eta_values)))
            ),
            "transport_convergence_rms_m_per_day": float(
                np.sqrt(np.mean(np.square(convergence_values)))
            ),
            "daily_snapshot_residual_rms_m_per_day": float(
                np.sqrt(np.mean(np.square(residual_values)))
            ),
            "eta_change_transport_convergence_correlation": float(
                np.corrcoef(eta_values, convergence_values)[0, 1]
            ),
            "interpretation": (
                "daily endpoint SSH change versus trapezoidal instantaneous "
                "transport; not a sub-daily exact continuity closure"
            ),
        }
        arrays[f"{regime}__eta_change_m_per_day"] = eta_change.astype(
            np.float32
        )
        arrays[
            f"{regime}__continuity_residual_m_per_day"
        ] = continuity_residual.astype(np.float32)
        regime_reports[regime] = {
            "selected_global_indices": [
                global_start,
                global_start + days - 1,
            ],
            "selected_extension_local_indices": [
                local_start,
                local_start + days - 1,
            ],
            "selected_iterations": [
                int(iterations[0]),
                int(iterations[-1]),
            ],
            "centered_velocity_matches_zarr_bitwise": bool(
                np.all(centered_state_exact)
            ),
            "project_operator_max_abs_difference_sv": float(
                np.max(project_operator_difference)
            ),
            "project_vs_metric_u_relative_rmse_maximum": float(
                np.max(project_metric_relative)
            ),
            "window_summary": window_summary,
            "daily_endpoint_continuity": continuity,
        }

        if regime == "S0":
            daily_u = u_derived_streamfunction(u_bt[0], dyg)
            daily_v = v_derived_streamfunction(v_bt[0], dxg)
            annual_u = u_derived_streamfunction(np.mean(u_bt, axis=0), dyg)
            annual_v = v_derived_streamfunction(np.mean(v_bt, axis=0), dxg)
            daily_metrics = uv_streamfunction_metrics(daily_u, daily_v, wet)
            annual_metrics = uv_streamfunction_metrics(
                annual_u,
                annual_v,
                wet,
            )
            example_maps = {
                "S0_daily_u_sv": daily_u.astype(np.float32),
                "S0_daily_v_aligned_sv": np.asarray(
                    daily_metrics["aligned_v"],
                    dtype=np.float32,
                ),
                "S0_daily_difference_sv": np.asarray(
                    daily_metrics["difference"],
                    dtype=np.float32,
                ),
                "S0_annual_u_sv": annual_u.astype(np.float32),
                "S0_annual_v_aligned_sv": np.asarray(
                    annual_metrics["aligned_v"],
                    dtype=np.float32,
                ),
                "S0_annual_difference_sv": np.asarray(
                    annual_metrics["difference"],
                    dtype=np.float32,
                ),
            }
            arrays.update(example_maps)

    thresholds = contract["interpretation_thresholds"]
    project_checks = {
        regime: (
            regime_reports[regime][
                "centered_velocity_matches_zarr_bitwise"
            ]
            and regime_reports[regime][
                "project_operator_max_abs_difference_sv"
            ]
            <= float(thresholds["project_operator_absolute_tolerance_sv"])
            and regime_reports[regime][
                "project_vs_metric_u_relative_rmse_maximum"
            ]
            <= float(
                thresholds["project_operator_metric_rmse_fraction"]
            )
        )
        for regime in EXPERIMENTS
    }
    annual_checks = {}
    averaging_checks = {}
    daily_checks = {}
    for regime in EXPERIMENTS:
        annual = window_results[regime][360]
        daily = window_results[regime][1]
        ninety = window_results[regime][90]
        amplitude = float(annual["amplitude_ratio"][0])
        lower, upper = thresholds[
            "annual_uv_amplitude_ratio_interval"
        ]
        annual_checks[regime] = bool(
            annual["relative_rmse"][0]
            <= float(thresholds["annual_uv_rmse_fraction"])
            and annual["correlation"][0]
            >= float(thresholds["annual_uv_minimum_correlation"])
            and float(lower) <= amplitude <= float(upper)
        )
        averaging_checks[regime] = bool(
            annual["relative_rmse"][0]
            < float(np.median(daily["relative_rmse"]))
            and annual["relative_rmse"][0]
            < float(np.median(ninety["relative_rmse"]))
            and annual["divergence_rms_m_per_day"][0]
            < float(np.median(daily["divergence_rms_m_per_day"]))
        )
        daily_checks[regime] = bool(
            np.percentile(daily["relative_rmse"], 90)
            <= float(thresholds["daily_uv_rmse_fraction"])
            and np.percentile(daily["correlation"], 10)
            >= float(thresholds["daily_uv_minimum_correlation"])
        )
    time_mean_supported = bool(
        all(project_checks.values())
        and all(annual_checks.values())
        and all(averaging_checks.values())
    )
    daily_unique_supported = bool(all(daily_checks.values()))
    decision = {
        "project_u_operator_verified": bool(all(project_checks.values())),
        "project_operator_checks_by_regime": project_checks,
        "annual_uv_consistency_passed": bool(all(annual_checks.values())),
        "annual_checks_by_regime": annual_checks,
        "averaging_improvement_passed": bool(
            all(averaging_checks.values())
        ),
        "averaging_checks_by_regime": averaging_checks,
        "instantaneous_daily_uniqueness_supported": daily_unique_supported,
        "daily_checks_by_regime": daily_checks,
        "time_mean_transport_streamfunction_supported": time_mean_supported,
        "classification": (
            "time_mean_streamfunction_supported"
            if time_mean_supported
            else "retain_u_derived_transport_diagnostic_without_unique_streamfunction_claim"
        ),
        "instantaneous_label": (
            "instantaneous_uv_paths_meet_declared_consistency"
            if daily_unique_supported
            else "instantaneous_field_remains_u_derived_not_unique"
        ),
    }

    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    arrays_path = temporary / ARRAYS_NAME
    np.savez_compressed(arrays_path, **arrays)
    _plot_summary(
        temporary,
        window_results,
        project_metric_errors,
    )
    _plot_maps(
        temporary,
        arrays["S0_daily_u_sv"],
        arrays["S0_daily_v_aligned_sv"],
        arrays["S0_annual_u_sv"],
        arrays["S0_annual_v_aligned_sv"],
        wet,
    )
    report = {
        "status": "complete",
        "version": VERSION,
        "purpose": "training_only_raw_face_uv_streamfunction_consistency",
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset),
        "dataset_metadata_sha256": _sha256(dataset / ".zmetadata"),
        "expansion_contract_sha256": expansion_sha,
        "audit": contract["audit"],
        "interpretation_thresholds": thresholds,
        "grid": {
            "wet_cells": int(np.sum(wet)),
            "depth_m": _stats(depth[wet]),
            "dxg_m": _stats(dxg[wet]),
            "dyg_m": _stats(dyg[wet]),
            "rac_m2": _stats(rac[wet]),
            "drf_m": DRF_M.tolist(),
        },
        "regimes": regime_reports,
        "decision": decision,
        "arrays": str(output / ARRAYS_NAME),
        "arrays_sha256": _sha256(arrays_path),
        "figures": {
            SUMMARY_FIGURE_NAME: {
                "path": str(output / SUMMARY_FIGURE_NAME),
                "sha256": _sha256(temporary / SUMMARY_FIGURE_NAME),
            },
            MAP_FIGURE_NAME: {
                "path": str(output / MAP_FIGURE_NAME),
                "sha256": _sha256(temporary / MAP_FIGURE_NAME),
            },
        },
        "read_contract": contract["read_contract"],
        "validation_state_opened": False,
        "inference_opened": False,
        "intermediate_wind_opened": False,
        "response_or_adjoint_opened": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    report["report_content_sha256"] = _json_sha256(report)
    report_path = temporary / REPORT_NAME
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "version": VERSION,
        "status": "complete",
        "source_report": str(output / REPORT_NAME),
        "source_report_sha256": _sha256(report_path),
        "source_report_content_sha256": report["report_content_sha256"],
        "source_arrays": str(output / ARRAYS_NAME),
        "source_arrays_sha256": report["arrays_sha256"],
        "figures": report["figures"],
        "validation_state_opened": False,
        "inference_opened": False,
    }
    manifest["manifest_content_sha256"] = _json_sha256(manifest)
    (temporary / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (temporary / README_NAME).write_text(
        "# Streamfunction consistency audit\n\n"
        "This training-only side study compares raw-face U- and V-derived "
        "barotropic transport streamfunctions over one fixed 360-day year "
        "in each S0--S2 regime. It also verifies the project centered-U "
        "operator and measures how path mismatch and transport divergence "
        "change under 1/5/30/90/360-day averaging.\n\n"
        f"Classification: `{decision['classification']}`.\n\n"
        "The U/V comparison removes one wet-cell mean gauge offset per "
        "state. Daily SSH continuity residuals compare daily endpoint SSH "
        "change with trapezoidal instantaneous transports and are explicitly "
        "not treated as a sub-daily exact closure test.\n"
    )
    os.replace(temporary, output)
    return report


def preflight_streamfunction_consistency(
    dataset_path: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Verify immutable sources and the training-only chronology."""

    contract, resolved, digest = load_streamfunction_consistency_contract(
        contract_path
    )
    dataset = Path(dataset_path).resolve()
    _, expansion_sha, inventories = _preflight_sources(contract, dataset)
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "expansion_contract_sha256": expansion_sha,
        "regimes": list(EXPERIMENTS),
        "selected_extension_iterations": {
            regime: [
                int(inventory.iterations[90]),
                int(inventory.iterations[449]),
            ]
            for regime, inventory in zip(EXPERIMENTS, inventories)
        },
        "read_contract": contract["read_contract"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--dataset", type=Path, required=True)
        child.add_argument("--contract", type=Path, required=True)
        if command == "run":
            child.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight_streamfunction_consistency(
            args.dataset,
            args.contract,
        )
    else:
        result = run_streamfunction_consistency(
            args.dataset,
            args.contract,
            args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
