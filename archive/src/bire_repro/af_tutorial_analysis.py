"""Reproduce and validate the MITgcm baroclinic-gyre tutorial figures from AF--FNO S0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .af_s0 import MODEL_YEAR_DAYS
from .mds import mds_fields, read_mds


NX = 62
NY = 62
NR = 15
MONTHS_PER_YEAR = 12
RHO_CONST = 999.8
HEAT_CAPACITY_CP = 3994.0
THETA_RELAX_SECONDS = 2_592_000.0
EARTH_RADIUS_M = 6_371_000.0
DRF_M = np.asarray(
    [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    dtype=float,
)
REFERENCE_LEVELS = (0, 4, 14)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _grid() -> dict[str, np.ndarray]:
    lon = -0.5 + np.arange(NX, dtype=float)
    lat = 14.5 + np.arange(NY, dtype=float)
    lon_edges = -1.0 + np.arange(NX + 1, dtype=float)
    lat_edges = 14.0 + np.arange(NY + 1, dtype=float)
    wet = np.zeros((NY, NX), dtype=bool)
    wet[1:-1, 1:-1] = True
    area = np.cos(np.deg2rad(lat))[:, None] * wet
    dy_m = EARTH_RADIUS_M * np.deg2rad(1.0)
    rc = -np.cumsum(DRF_M) + 0.5 * DRF_M
    return {
        "lon": lon,
        "lat": lat,
        "lon_edges": lon_edges,
        "lat_edges": lat_edges,
        "wet": wet,
        "area": area,
        "dy_m": np.asarray(dy_m),
        "rc": rc,
    }


def _weighted_mean(field: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(field * weights) / np.sum(weights))


def _weighted_std(field: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(field, weights)
    return float(np.sqrt(np.sum((field - mean) ** 2 * weights) / np.sum(weights)))


def reconstruct_trelax(theta_surface: np.ndarray, target: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Reconstruct the core-model ``TRELAX`` diagnostic in W m-2.

    This is the algebra in ``model/src/forcing_surf_relax.F`` for z coordinates.
    Because the restoring operator is linear, it is valid for monthly/annual means.
    """
    if theta_surface.shape != (NY, NX) or target.shape != (NY, NX):
        raise ValueError("surface temperature and restoring target must be 62 x 62")
    scale = DRF_M[0] * RHO_CONST * HEAT_CAPACITY_CP / THETA_RELAX_SECONDS
    return np.where(wet, (target - theta_surface) * scale, np.nan)


def barotropic_streamfunction(u: np.ndarray, dy_m: float) -> np.ndarray:
    """Compute the tutorial's meridionally integrated transport streamfunction in Sv."""
    if u.shape != (NR, NY, NX):
        raise ValueError("UVEL must have shape (15, 62, 62)")
    depth_integrated_u = np.sum(u * DRF_M[:, None, None], axis=0)
    return np.cumsum(-depth_integrated_u * dy_m, axis=0) / 1.0e6


def _center_velocity(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u_center = 0.5 * (u + np.roll(u, -1, axis=-1))
    v_center = 0.5 * (v + np.roll(v, -1, axis=-2))
    return u_center, v_center


def _trend_change(values: np.ndarray, samples: int = 60) -> tuple[float, float]:
    selected = np.asarray(values[-samples:], dtype=float)
    years = np.arange(selected.size, dtype=float) / MONTHS_PER_YEAR
    slope = float(np.polyfit(years, selected, 1)[0])
    change = slope * (samples / MONTHS_PER_YEAR)
    relative = abs(change) / max(abs(float(np.mean(selected))), np.finfo(float).eps)
    return change, relative


def _diagnostic_files(s0_root: Path, phase: str, prefix: str) -> list[Path]:
    paths = list((s0_root / phase).glob(f"years_*/{prefix}.*.meta"))
    return sorted(paths, key=lambda path: int(path.name.split(".")[1]))


def _load_fields(metadata_path: Path) -> dict[str, np.ndarray]:
    meta, values = read_mds(metadata_path)
    return mds_fields(meta, values)


def _spinup_series(
    paths: Iterable[Path], grid: Mapping[str, np.ndarray], target: np.ndarray
) -> dict[str, np.ndarray]:
    series: dict[str, list[float]] = {
        "heat_flux": [],
        "theta_mean_k0": [],
        "theta_mean_k4": [],
        "theta_mean_k14": [],
        "theta_std_k0": [],
        "theta_std_k4": [],
        "theta_std_k14": [],
        "surface_kinetic_energy": [],
        "psi_max": [],
        "psi_min": [],
        "volume_mean_theta": [],
    }
    wet = grid["wet"]
    area = grid["area"]
    volume_weights = DRF_M[:, None, None] * area[None, :, :]
    for path in paths:
        fields = _load_fields(path)
        theta = np.asarray(fields["THETA"], dtype=float)
        u = np.asarray(fields["UVEL"], dtype=float)
        v = np.asarray(fields["VVEL"], dtype=float)
        if theta.shape != (NR, NY, NX) or not np.all(np.isfinite(theta)):
            raise ValueError(f"invalid THETA diagnostic in {path}")
        trelax = reconstruct_trelax(theta[0], target, wet)
        series["heat_flux"].append(_weighted_mean(np.nan_to_num(trelax), area))
        for level in REFERENCE_LEVELS:
            series[f"theta_mean_k{level}"].append(_weighted_mean(theta[level], area))
            series[f"theta_std_k{level}"].append(_weighted_std(theta[level], area))
        u_center, v_center = _center_velocity(u, v)
        surface_ke = 0.5 * RHO_CONST * (u_center[0] ** 2 + v_center[0] ** 2)
        series["surface_kinetic_energy"].append(_weighted_mean(surface_ke, area))
        psi = barotropic_streamfunction(u, float(grid["dy_m"]))
        series["psi_max"].append(float(np.max(psi[wet])))
        series["psi_min"].append(float(np.min(psi[wet])))
        series["volume_mean_theta"].append(
            float(np.sum(theta * volume_weights) / np.sum(volume_weights))
        )
    return {name: np.asarray(values) for name, values in series.items()}


def _annual_production_mean(
    dyn_paths: list[Path], surf_paths: list[Path], days: int = MODEL_YEAR_DAYS
) -> dict[str, np.ndarray]:
    if len(dyn_paths) < days or len(surf_paths) < days:
        raise ValueError("S0 production does not contain a complete first model year")
    accumulators: dict[str, np.ndarray] = {}
    for dyn_path, surf_path in zip(dyn_paths[:days], surf_paths[:days], strict=True):
        fields = {**_load_fields(dyn_path), **_load_fields(surf_path)}
        for name in ("UVEL", "VVEL", "THETA", "ETAN"):
            value = np.asarray(fields[name], dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite {name} in {dyn_path.parent}")
            if name not in accumulators:
                accumulators[name] = np.zeros_like(value, dtype=float)
            accumulators[name] += value
    return {name: values / days for name, values in accumulators.items()}


def _plot_timeseries(series: Mapping[str, np.ndarray], output: Path) -> None:
    time = (np.arange(len(series["heat_flux"])) + 0.5) / MONTHS_PER_YEAR
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    axes = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)]
    axes[0].plot(time, series["heat_flux"], color="#3658a7", linewidth=2)
    axes[0].set(title="a) Net heat flux into ocean", ylabel="W m$^{-2}$", ylim=(-400, 5))
    labels = ("surface", "305 m", "1705 m")
    colors = ("#58c3d3", "#58b43f", "#e62222")
    for level, label, color in zip(REFERENCE_LEVELS, labels, colors, strict=True):
        axes[1].plot(time, series[f"theta_mean_k{level}"], label=label, color=color, linewidth=2)
        axes[2].plot(time, series[f"theta_std_k{level}"], label=label, color=color, linewidth=2)
    axes[1].set(title="b) Mean potential temperature by level", ylabel="°C", ylim=(0, 30))
    axes[2].set(title="c) Standard deviation by level", ylabel="°C", ylim=(0, 8.5))
    for axis in axes:
        axis.set(xlabel="Time (360-day years)", xlim=(0, 100))
        axis.grid(alpha=0.25)
    axes[1].legend()
    axes[2].legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_surface(
    annual: Mapping[str, np.ndarray], trelax: np.ndarray, grid: Mapping[str, np.ndarray], output: Path
) -> None:
    eta = np.asarray(annual["ETAN"])
    if eta.ndim == 3:
        eta = eta[0]
    wet = grid["wet"]
    fig, axis = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    image = axis.pcolormesh(
        grid["lon_edges"], grid["lat_edges"], trelax, cmap="RdBu_r", vmin=-250, vmax=250
    )
    contours = axis.contour(
        grid["lon"], grid["lat"], np.where(wet, eta, np.nan), levels=np.arange(-0.6, 0.7, 0.1),
        colors="black", linewidths=0.7
    )
    axis.clabel(contours, fmt="%.1f", fontsize=6)
    axis.set(xlim=(0, 60), ylim=(15, 75), xlabel="Longitude", ylabel="Latitude")
    axis.set_title("Free surface height (contours) and TRELAX (shading)\nfirst S0 production year")
    fig.colorbar(image, ax=axis, label="W m$^{-2}$")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_streamfunction(psi: np.ndarray, grid: Mapping[str, np.ndarray], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    levels = np.arange(-35, 40, 5)
    image = axis.contourf(grid["lon"], grid["lat"], psi, levels=levels, cmap="RdYlBu_r", extend="both")
    contours = axis.contour(grid["lon"], grid["lat"], psi, levels=levels, colors="black", linewidths=0.7)
    axis.clabel(contours, fmt="%.0f", fontsize=7)
    axis.set(xlim=(0, 60), ylim=(15, 75), xlabel="Longitude", ylabel="Latitude")
    axis.set_title("Barotropic streamfunction (Sv), first S0 production year")
    fig.colorbar(image, ax=axis, label="Sv")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_temperature(
    theta: np.ndarray, grid: Mapping[str, np.ndarray], output: Path
) -> None:
    wet = grid["wet"]
    level = 3
    latitude_index = 14
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    plan = np.where(wet, theta[level], np.nan)
    image = axes[0].pcolormesh(
        grid["lon_edges"], grid["lat_edges"], plan, cmap="coolwarm", vmin=0, vmax=30
    )
    axes[0].contour(grid["lon"], grid["lat"], plan, levels=np.arange(0, 31, 2), colors="black", linewidths=0.6)
    axes[0].set(xlim=(0, 60), ylim=(15, 75), xlabel="Longitude", ylabel="Latitude")
    axes[0].set_title("a) THETA at 220 m")
    section = np.where(wet[latitude_index][None, :], theta[:, latitude_index], np.nan)
    section_image = axes[1].contourf(
        grid["lon"], grid["rc"], section, levels=np.arange(0, 30.2, 0.2), cmap="coolwarm"
    )
    axes[1].contour(
        grid["lon"], grid["rc"], section, levels=np.arange(0, 32, 2), colors="black", linewidths=0.6
    )
    axes[1].set(xlim=(0, 60), ylim=(-1800, 0), xlabel="Longitude", ylabel="Depth (m)")
    axes[1].set_title("b) THETA at 28.5°N")
    fig.colorbar(image, ax=axes[0], label="°C")
    fig.colorbar(section_image, ax=axes[1], label="°C")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze(project_root: Path, scratch_root: Path, output_dir: Path) -> dict[str, Any]:
    """Analyze S0, write four tutorial figures, and return a machine-readable gate."""
    s0_root = scratch_root / "mitgcm" / "S0"
    spin_paths = _diagnostic_files(s0_root, "spinup", "dynSpin")
    dyn_paths = _diagnostic_files(s0_root, "production", "dynState")
    surf_paths = _diagnostic_files(s0_root, "production", "surfState")
    expected = {"spinup_months": 1200, "production_days": 3600}
    counts = {
        "spinup_months": len(spin_paths),
        "dynstate_days": len(dyn_paths),
        "surfstate_days": len(surf_paths),
    }
    if counts != {"spinup_months": 1200, "dynstate_days": 3600, "surfstate_days": 3600}:
        raise ValueError(f"unexpected S0 diagnostic inventory: {counts}")

    tutorial_input = project_root / "external/MITgcm/verification/tutorial_baroclinic_gyre/input"
    target = np.fromfile(tutorial_input / "SST_relax.bin", dtype=">f4").reshape(NY, NX).astype(float)
    grid = _grid()
    series = _spinup_series(spin_paths, grid, target)
    annual = _annual_production_mean(dyn_paths, surf_paths)
    trelax = reconstruct_trelax(np.asarray(annual["THETA"])[0], target, grid["wet"])
    psi = barotropic_streamfunction(np.asarray(annual["UVEL"]), float(grid["dy_m"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "timeseries": output_dir / "tutorial_figure_4_6_timeseries.png",
        "surface": output_dir / "tutorial_figure_4_7_surface.png",
        "streamfunction": output_dir / "tutorial_figure_4_8_streamfunction.png",
        "temperature": output_dir / "tutorial_figure_4_9_temperature.png",
    }
    _plot_timeseries(series, figures["timeseries"])
    _plot_surface(annual, trelax, grid, figures["surface"])
    _plot_streamfunction(psi, grid, figures["streamfunction"])
    _plot_temperature(np.asarray(annual["THETA"]), grid, figures["temperature"])

    final = {name: float(values[-1]) for name, values in series.items()}
    five_year_trends = {}
    for name in (
        "theta_mean_k0",
        "theta_mean_k4",
        "theta_mean_k14",
        "surface_kinetic_energy",
        "psi_max",
        "psi_min",
        "volume_mean_theta",
    ):
        change, relative = _trend_change(series[name])
        five_year_trends[name] = {"change": change, "relative_change": relative}

    eta = np.asarray(annual["ETAN"])
    if eta.ndim == 3:
        eta = eta[0]
    wet = grid["wet"]
    checks = {
        "forcing_files_identical_to_tutorial": all(
            _sha256(tutorial_input / name)
            == _sha256(s0_root / "spinup/years_000_010" / name)
            for name in ("bathy.bin", "windx_cosy.bin", "SST_relax.bin")
        ),
        "complete_diagnostic_inventory": counts
        == {"spinup_months": 1200, "dynstate_days": 3600, "surfstate_days": 3600},
        "tutorial_temperature_means": (
            16.5 <= final["theta_mean_k0"] <= 19.0
            and 7.0 <= final["theta_mean_k4"] <= 10.5
            and 1.5 <= final["theta_mean_k14"] <= 3.2
        ),
        "tutorial_temperature_spreads": (
            7.0 <= final["theta_std_k0"] <= 8.5
            and 2.5 <= final["theta_std_k4"] <= 4.5
            and 0.01 <= final["theta_std_k14"] <= 0.6
        ),
        "tutorial_double_gyre_transport": (
            20.0 <= float(np.max(psi[wet])) <= 45.0
            and -45.0 <= float(np.min(psi[wet])) <= -20.0
        ),
        "tutorial_ssh_relief": float(np.min(eta[wet])) < -0.3 and float(np.max(eta[wet])) > 0.3,
        "upper_ocean_five_year_stationarity": (
            abs(five_year_trends["theta_mean_k0"]["change"]) < 0.1
            and five_year_trends["surface_kinetic_energy"]["relative_change"] < 0.1
            and five_year_trends["psi_max"]["relative_change"] < 0.1
            and five_year_trends["psi_min"]["relative_change"] < 0.1
        ),
    }
    report: dict[str, Any] = {
        "experiment": "S0",
        "passed": all(checks.values()),
        "checks": checks,
        "diagnostic_counts": counts,
        "expected_counts": expected,
        "final_month_metrics": final,
        "last_five_year_trends": five_year_trends,
        "first_production_year_metrics": {
            "trelax_area_mean_w_m2": _weighted_mean(np.nan_to_num(trelax), grid["area"]),
            "ssh_min_m": float(np.min(eta[wet])),
            "ssh_max_m": float(np.max(eta[wet])),
            "psi_min_sv": float(np.min(psi[wet])),
            "psi_max_sv": float(np.max(psi[wet])),
            "theta_220m_min_c": float(np.min(np.asarray(annual["THETA"])[3][wet])),
            "theta_220m_max_c": float(np.max(np.asarray(annual["THETA"])[3][wet])),
        },
        "figures": {name: str(path) for name, path in figures.items()},
        "notes": [
            "The official tutorial reference figures use year-100 annual means.",
            "S0 spin-up provides the exact 100-year monthly series; spatial panels use daily snapshots averaged over the first production year (model years 100-101).",
            "The deep ocean is expected to continue adjusting after year 100 and is reported but not included in the upper-ocean stationarity gate.",
        ],
    }
    _atomic_json(output_dir / "tutorial_validation.json", report)
    np.savez_compressed(output_dir / "tutorial_timeseries.npz", **series)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = analyze(args.project_root.resolve(), args.scratch_root.resolve(), args.output_dir.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
