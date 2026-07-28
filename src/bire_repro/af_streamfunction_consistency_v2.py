"""Collocated C-grid audit of the barotropic transport streamfunction.

Version 1 deliberately compared the two independently accumulated arrays
after the same centering operations used by the project plotting diagnostic.
Those operations place the U path on north faces and the V path on east
faces.  This follow-up keeps the frozen year and thresholds but compares both
prefix integrals at the same C-grid corner indices.  The v1 result remains an
immutable diagnostic of the uncollocated implementation.
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
from .af_s0 import _sha256
from .af_streamfunction_consistency import (
    ARRAYS_NAME,
    MAP_FIGURE_NAME,
    README_NAME,
    REPORT_NAME,
    SECONDS_PER_DAY,
    SUMMARY_FIGURE_NAME,
    SV_SCALE,
    StreamfunctionConsistencyError,
    _read_velocity_eta,
    _scalar_mds,
    _stats,
    transport_divergence_volume,
)
from .af_trajectory_expansion import EXPERIMENTS, load_expansion_contract
from .af_tutorial_analysis import DRF_M


VERSION = "streamfunction_consistency_v2"
MANIFEST_NAME = "figure_manifest.json"


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_contract(
    path: str | Path,
    *,
    verify_source_files: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the v2 contract frozen before collocated daily metrics."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != VERSION:
        raise ValueError(f"expected streamfunction contract {VERSION}")
    if (
        contract.get("contract_status")
        != "frozen_before_collocated_c_grid_daily_metrics"
    ):
        raise ValueError("collocated streamfunction contract was not frozen")
    audit = contract.get("audit", {})
    if (
        tuple(audit.get("regimes", ())) != EXPERIMENTS
        or int(audit.get("global_start_index", -1)) != 3690
        or int(audit.get("extension_local_start_index", -1)) != 90
        or int(audit.get("days", -1)) != 360
        or tuple(audit.get("averaging_windows_days", ()))
        != (1, 5, 30, 90, 360)
        or audit.get("velocity_source") != "raw_mitgcm_c_grid_faces"
        or audit.get("comparison_grid")
        != "common_unshifted_c_grid_corner_prefix_locations"
        or audit.get("comparison_mask")
        != "indices_of_the_60_by_60_wet_tracer_interior"
    ):
        raise ValueError("collocated streamfunction audit design changed")
    thresholds = contract.get("interpretation_thresholds", {})
    if (
        float(thresholds.get("daily_uv_rmse_fraction", -1.0)) != 0.05
        or float(thresholds.get("daily_uv_minimum_correlation", -1.0))
        != 0.99
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
                    f"collocated streamfunction source changed: {source}"
                )
        prior = Path(contract["source_artifacts"]["v1_report"]).resolve()
        if _sha256(prior) != contract["source_artifacts"]["v1_report_sha256"]:
            raise ValueError("v1 streamfunction report changed")
    return contract, resolved, _sha256(resolved)


def u_corner_streamfunction(
    u_bt: np.ndarray,
    dyg: np.ndarray,
) -> np.ndarray:
    """Accumulate U transport to the unshifted southwest C-grid corners."""

    if u_bt.shape != (62, 62) or dyg.shape != (62, 62):
        raise ValueError("U transport and DYG must be 62 x 62")
    prefix = np.vstack(
        (
            np.zeros((1, 62), dtype=np.float64),
            np.cumsum(-np.asarray(u_bt) * dyg, axis=0),
        )
    )
    return prefix[:-1] / SV_SCALE


def v_corner_streamfunction(
    v_bt: np.ndarray,
    dxg: np.ndarray,
) -> np.ndarray:
    """Accumulate V transport to the unshifted southwest C-grid corners."""

    if v_bt.shape != (62, 62) or dxg.shape != (62, 62):
        raise ValueError("V transport and DXG must be 62 x 62")
    prefix = np.hstack(
        (
            np.zeros((62, 1), dtype=np.float64),
            np.cumsum(np.asarray(v_bt) * dxg, axis=1),
        )
    )
    return prefix[:, :-1] / SV_SCALE


def collocated_metrics(
    psi_u: np.ndarray,
    psi_v: np.ndarray,
    comparison_mask: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Gauge-align collocated corner fields and measure their mismatch."""

    if (
        psi_u.shape != (62, 62)
        or psi_v.shape != psi_u.shape
        or comparison_mask.shape != psi_u.shape
        or not np.any(comparison_mask)
    ):
        raise ValueError("collocated metric arrays are inconsistent")
    offset = float(np.mean((psi_u - psi_v)[comparison_mask]))
    aligned_v = psi_v + offset
    difference = psi_u - aligned_v
    u_values = psi_u[comparison_mask]
    v_values = aligned_v[comparison_mask]
    rmse = float(
        np.sqrt(np.mean(np.square(difference[comparison_mask])))
    )
    u_rms = float(np.sqrt(np.mean(np.square(u_values))))
    return {
        "gauge_offset_sv": offset,
        "rmse_sv": rmse,
        "relative_rmse": rmse / max(u_rms, np.finfo(float).eps),
        "correlation": float(np.corrcoef(u_values, v_values)[0, 1]),
        "amplitude_ratio_v_to_u": float(np.max(np.abs(v_values)))
        / max(float(np.max(np.abs(u_values))), np.finfo(float).eps),
        "aligned_v": aligned_v,
        "difference": difference,
    }


def _window_metrics(
    u_bt: np.ndarray,
    v_bt: np.ndarray,
    dyg: np.ndarray,
    dxg: np.ndarray,
    rac: np.ndarray,
    comparison_mask: np.ndarray,
    window: int,
) -> dict[str, np.ndarray]:
    if u_bt.shape != v_bt.shape or u_bt.shape[0] % window:
        raise ValueError("averaging window must partition the selected year")
    result: dict[str, list[float]] = {
        "relative_rmse": [],
        "correlation": [],
        "amplitude_ratio": [],
        "gauge_offset_sv": [],
        "rmse_sv": [],
        "divergence_rms_m_per_day": [],
    }
    for start in range(0, u_bt.shape[0], window):
        mean_u = np.mean(u_bt[start : start + window], axis=0)
        mean_v = np.mean(v_bt[start : start + window], axis=0)
        metrics = collocated_metrics(
            u_corner_streamfunction(mean_u, dyg),
            v_corner_streamfunction(mean_v, dxg),
            comparison_mask,
        )
        divergence = (
            transport_divergence_volume(mean_u, mean_v, dyg, dxg)
            / rac
            * SECONDS_PER_DAY
        )
        for name in (
            "relative_rmse",
            "correlation",
            "gauge_offset_sv",
            "rmse_sv",
        ):
            result[name].append(float(metrics[name]))
        result["amplitude_ratio"].append(
            float(metrics["amplitude_ratio_v_to_u"])
        )
        result["divergence_rms_m_per_day"].append(
            float(
                np.sqrt(
                    np.mean(
                        np.square(divergence[comparison_mask])
                    )
                )
            )
        )
    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in result.items()
    }


def _preflight(
    contract: Mapping[str, Any],
    dataset: Path,
) -> tuple[str, list[Any]]:
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
    return expansion_sha, inventories


def _plot_summary(
    output: Path,
    results: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
) -> None:
    windows = np.asarray((1, 5, 30, 90, 360))
    colors = ("#2F75B5", "#2A9D8F", "#C45A30")
    figure, axes = plt.subplots(
        2, 2, figsize=(10.8, 7.2), constrained_layout=True
    )
    fields = (
        ("relative_rmse", "Collocated U/V RMSE / U RMS", "log"),
        ("correlation", "Collocated U/V correlation", "linear"),
        ("rmse_sv", "Collocated U/V RMSE (Sv)", "log"),
        (
            "divergence_rms_m_per_day",
            "Transport-divergence equivalent (m day$^{-1}$)",
            "log",
        ),
    )
    for axis, (field, label, scale) in zip(axes.flat, fields):
        for color, regime in zip(colors, EXPERIMENTS):
            medians = []
            lower = []
            upper = []
            for window in windows:
                values = results[regime][int(window)][field]
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
    axes[0, 0].axhline(
        0.05, color="black", linestyle="--", linewidth=1.0
    )
    axes[0, 1].axhline(
        0.99, color="black", linestyle="--", linewidth=1.0
    )
    axes[0, 0].legend()
    figure.suptitle(
        "Collocated raw MITgcm U/V transport consistency"
    )
    figure.savefig(
        output / SUMMARY_FIGURE_NAME, dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def _plot_maps(
    output: Path,
    daily_u: np.ndarray,
    daily_v: np.ndarray,
    annual_u: np.ndarray,
    annual_v: np.ndarray,
    comparison_mask: np.ndarray,
) -> None:
    daily = collocated_metrics(daily_u, daily_v, comparison_mask)
    annual = collocated_metrics(annual_u, annual_v, comparison_mask)
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
        2, 3, figsize=(11.2, 7.2), constrained_layout=True
    )
    for row_index, (psi_u, psi_v, difference, label, metrics) in enumerate(
        rows
    ):
        state_bound = max(
            float(np.max(np.abs(psi_u[comparison_mask]))),
            float(np.max(np.abs(psi_v[comparison_mask]))),
        )
        difference_bound = max(
            float(np.max(np.abs(difference[comparison_mask]))),
            np.finfo(float).eps,
        )
        plotted = (
            (psi_u, state_bound, "U-derived"),
            (psi_v, state_bound, "V-derived, gauge aligned"),
            (difference, difference_bound, "U minus V"),
        )
        for column, (values, bound, title) in enumerate(plotted):
            shown = np.where(comparison_mask, values, np.nan)
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
                image, ax=axes[row_index, column], label="Sv", shrink=0.78
            )
        axes[row_index, 2].set_xlabel(
            f"RMSE={float(metrics['rmse_sv']):.3e} Sv; "
            f"relative={float(metrics['relative_rmse']):.3%}; "
            f"r={float(metrics['correlation']):.8f}"
        )
    figure.suptitle(
        "S0 collocated C-grid streamfunction consistency"
    )
    figure.savefig(
        output / MAP_FIGURE_NAME, dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def run(
    dataset_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the frozen collocated C-grid consistency audit."""

    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    dataset = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    expansion_sha, inventories = _preflight(contract, dataset)
    group = zarr.open_consolidated(str(dataset), mode="r")
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    if int(np.sum(wet)) != 3600:
        raise StreamfunctionConsistencyError("wet interior changed")
    comparison_mask = wet.copy()
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
    if not np.array_equal(wet, depth > 0.0):
        raise StreamfunctionConsistencyError("wet mask changed")

    arrays: dict[str, np.ndarray] = {
        "comparison_mask": comparison_mask.astype(np.uint8),
        "dxg_m": dxg.astype(np.float32),
        "dyg_m": dyg.astype(np.float32),
        "rac_m2": rac.astype(np.float32),
    }
    all_results: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    regime_reports: dict[str, Any] = {}
    example_maps: dict[str, np.ndarray] = {}

    for regime, inventory in zip(EXPERIMENTS, inventories):
        u_bt = np.empty((days, 62, 62), dtype=np.float64)
        v_bt = np.empty_like(u_bt)
        eta = np.empty_like(u_bt)
        iterations = np.empty(days, dtype=np.int64)
        boundary_normal_max = np.empty(days, dtype=np.float64)
        for destination, local_index in enumerate(
            range(local_start, local_start + days)
        ):
            u, v, eta[destination], _, _ = _read_velocity_eta(
                inventory.dyn_meta[local_index],
                inventory.surf_meta[local_index],
            )
            u_bt[destination] = u
            v_bt[destination] = v
            iterations[destination] = inventory.iterations[local_index]
            u_volume = u * dyg
            v_volume = v * dxg
            boundary_normal_max[destination] = max(
                float(np.max(np.abs(u_volume[:, (0, -1)]))),
                float(np.max(np.abs(v_volume[(0, -1), :]))),
            )

        all_results[regime] = {}
        summaries: dict[str, Any] = {}
        for window in windows:
            metrics = _window_metrics(
                u_bt,
                v_bt,
                dyg,
                dxg,
                rac,
                comparison_mask,
                window,
            )
            all_results[regime][window] = metrics
            summaries[str(window)] = {
                name: _stats(values)
                for name, values in metrics.items()
            }
            for name, values in metrics.items():
                arrays[f"{regime}__window_{window:03d}__{name}"] = values

        divergence = np.stack(
            [
                transport_divergence_volume(
                    u_bt[index], v_bt[index], dyg, dxg
                )
                / rac
                * SECONDS_PER_DAY
                for index in range(days)
            ]
        )
        eta_change = eta[1:] - eta[:-1]
        divergence_mid = 0.5 * (divergence[1:] + divergence[:-1])
        residual = eta_change + divergence_mid
        continuity = {
            "eta_change_rms_m_per_day": float(
                np.sqrt(np.mean(np.square(eta_change[:, wet])))
            ),
            "transport_convergence_rms_m_per_day": float(
                np.sqrt(np.mean(np.square(divergence_mid[:, wet])))
            ),
            "daily_snapshot_residual_rms_m_per_day": float(
                np.sqrt(np.mean(np.square(residual[:, wet])))
            ),
            "eta_change_transport_convergence_correlation": float(
                np.corrcoef(
                    eta_change[:, wet].ravel(),
                    (-divergence_mid[:, wet]).ravel(),
                )[0, 1]
            ),
            "interpretation": (
                "daily endpoint SSH change versus trapezoidal instantaneous "
                "transport; not a sub-daily exact continuity closure"
            ),
        }
        arrays[f"{regime}__iterations"] = iterations
        arrays[f"{regime}__boundary_normal_max_m3_s"] = boundary_normal_max
        regime_reports[regime] = {
            "selected_iterations": [
                int(iterations[0]),
                int(iterations[-1]),
            ],
            "boundary_normal_transport_maximum_m3_s": float(
                np.max(boundary_normal_max)
            ),
            "window_summary": summaries,
            "daily_endpoint_continuity": continuity,
        }

        if regime == "S0":
            daily_u = u_corner_streamfunction(u_bt[0], dyg)
            daily_v = v_corner_streamfunction(v_bt[0], dxg)
            annual_u = u_corner_streamfunction(
                np.mean(u_bt, axis=0), dyg
            )
            annual_v = v_corner_streamfunction(
                np.mean(v_bt, axis=0), dxg
            )
            example_maps = {
                "S0_daily_u_sv": daily_u.astype(np.float32),
                "S0_daily_v_sv": daily_v.astype(np.float32),
                "S0_annual_u_sv": annual_u.astype(np.float32),
                "S0_annual_v_sv": annual_v.astype(np.float32),
            }
            arrays.update(example_maps)

    thresholds = contract["interpretation_thresholds"]
    daily_checks: dict[str, bool] = {}
    annual_checks: dict[str, bool] = {}
    averaging_checks: dict[str, bool] = {}
    boundary_checks: dict[str, bool] = {}
    for regime in EXPERIMENTS:
        daily = all_results[regime][1]
        annual = all_results[regime][360]
        ninety = all_results[regime][90]
        lower, upper = thresholds["annual_uv_amplitude_ratio_interval"]
        daily_checks[regime] = bool(
            np.percentile(daily["relative_rmse"], 90)
            <= float(thresholds["daily_uv_rmse_fraction"])
            and np.percentile(daily["correlation"], 10)
            >= float(thresholds["daily_uv_minimum_correlation"])
        )
        annual_checks[regime] = bool(
            annual["relative_rmse"][0]
            <= float(thresholds["annual_uv_rmse_fraction"])
            and annual["correlation"][0]
            >= float(thresholds["annual_uv_minimum_correlation"])
            and float(lower)
            <= annual["amplitude_ratio"][0]
            <= float(upper)
        )
        averaging_checks[regime] = bool(
            annual["relative_rmse"][0]
            < float(np.median(daily["relative_rmse"]))
            and annual["relative_rmse"][0]
            < float(np.median(ninety["relative_rmse"]))
            and annual["divergence_rms_m_per_day"][0]
            < float(np.median(daily["divergence_rms_m_per_day"]))
        )
        boundary_checks[regime] = bool(
            regime_reports[regime][
                "boundary_normal_transport_maximum_m3_s"
            ]
            <= float(thresholds["boundary_normal_transport_tolerance_m3_s"])
        )
    daily_supported = bool(all(daily_checks.values()))
    time_mean_supported = bool(
        all(annual_checks.values())
        and all(averaging_checks.values())
        and all(boundary_checks.values())
    )
    decision = {
        "daily_collocated_path_consistency_supported": daily_supported,
        "daily_checks_by_regime": daily_checks,
        "annual_collocated_path_consistency_supported": bool(
            all(annual_checks.values())
        ),
        "annual_checks_by_regime": annual_checks,
        "averaging_improvement_passed": bool(
            all(averaging_checks.values())
        ),
        "averaging_checks_by_regime": averaging_checks,
        "impermeable_boundary_check_passed": bool(
            all(boundary_checks.values())
        ),
        "boundary_checks_by_regime": boundary_checks,
        "time_mean_transport_streamfunction_supported": time_mean_supported,
        "classification": (
            "collocated_daily_and_time_mean_transport_streamfunction_supported"
            if daily_supported and time_mean_supported
            else "retain_u_derived_transport_diagnostic_with_failed_path_check"
        ),
        "mathematical_caveat": (
            "finite free-surface tendency permits nonzero instantaneous "
            "depth-integrated divergence; passing is numerical path "
            "consistency, not an algebraic exact-uniqueness claim"
        ),
    }

    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    arrays_path = temporary / ARRAYS_NAME
    np.savez_compressed(arrays_path, **arrays)
    _plot_summary(temporary, all_results)
    _plot_maps(
        temporary,
        example_maps["S0_daily_u_sv"],
        example_maps["S0_daily_v_sv"],
        example_maps["S0_annual_u_sv"],
        example_maps["S0_annual_v_sv"],
        comparison_mask,
    )
    report = {
        "status": "complete",
        "version": VERSION,
        "purpose": "training_only_collocated_raw_face_uv_streamfunction_consistency",
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset),
        "dataset_metadata_sha256": _sha256(dataset / ".zmetadata"),
        "expansion_contract_sha256": expansion_sha,
        "v1_report": contract["source_artifacts"]["v1_report"],
        "v1_report_sha256": contract["source_artifacts"]["v1_report_sha256"],
        "v1_diagnosis": (
            "v1 centered U and V paths occupied north and east C-grid faces; "
            "its 10.9 percent result is retained as an uncollocated warning, "
            "not used as the v2 collocated path metric"
        ),
        "audit": contract["audit"],
        "interpretation_thresholds": thresholds,
        "grid": {
            "comparison_points": int(np.sum(comparison_mask)),
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
        "# Collocated streamfunction consistency audit\n\n"
        "This training-only v2 study corrects the C-grid collocation issue "
        "exposed by v1. Raw-face U and V prefix integrals are compared at "
        "the same unshifted corner indices for one fixed 360-day year in "
        "S0--S2 and under 1/5/30/90/360-day averaging.\n\n"
        f"Classification: `{decision['classification']}`.\n\n"
        "A pass establishes numerical U/V path consistency for this "
        "configuration. It does not remove the mathematical linear-free-"
        "surface caveat: instantaneous depth-integrated transport may have "
        "small divergence when SSH changes.\n"
    )
    os.replace(temporary, output)
    return report


def preflight(
    dataset_path: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Verify immutable sources and the training-only chronology."""

    contract, resolved, digest = load_contract(contract_path)
    dataset = Path(dataset_path).resolve()
    expansion_sha, inventories = _preflight(contract, dataset)
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "expansion_contract_sha256": expansion_sha,
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
        result = preflight(args.dataset, args.contract)
    else:
        result = run(args.dataset, args.contract, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
