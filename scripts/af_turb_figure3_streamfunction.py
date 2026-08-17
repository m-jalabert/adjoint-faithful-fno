"""Ground-truth-only streamfunction panel, styled like ``model_c_bire_figure3``.

``src/oceanfno/plots.py:_plot_streamfunction_grid`` draws a 3-row grid --
MITgcm truth, Model C, and their difference -- at lead days (0, 10, 20, 30, 40)
from one S0 inference initial condition.  There is no FNO model for the
turbulent 0.25-degree configuration yet, so this script reproduces only the
truth row: the same pcolormesh/RdBu_r styling, the same land mask, the same
per-column day titles and shared colorbar, applied to the actual S0_turb
ground-truth barotropic streamfunction at day (0, 10, 20, 30, 40) of the
production record, i.e. the same five snapshots the model row would have been
scored against.

Day 0 is the exact model-year-100 state (the first production dump, written at
the segment's own iteration zero), matching the FIGURE_3_LEADS convention where
lead day 0 is the inference initial condition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "archive" / "src"))

from bire_repro.mds import mds_fields, read_mds  # noqa: E402

NR = 15
NX = NY = 248
SPACING_DEG = 0.25
RIM_CELLS = 4
XG_ORIGIN = -1.0
YG_ORIGIN = 14.0
EARTH_RADIUS_M = 6_371_000.0
DRF_M = np.asarray(
    [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    dtype=float,
)
LEAD_DAYS = (0, 10, 20, 30, 40)
STEPS_PER_DAY = 288
OUTPUT_NAME = "s0_turb_figure3_streamfunction_truth_dt10.png"


def _grid() -> dict[str, np.ndarray]:
    lon = XG_ORIGIN + np.arange(NX + 1, dtype=float) * SPACING_DEG
    lat = YG_ORIGIN + np.arange(NY + 1, dtype=float) * SPACING_DEG
    wet = np.zeros((NY, NX), dtype=bool)
    wet[RIM_CELLS:-RIM_CELLS, RIM_CELLS:-RIM_CELLS] = True
    dy_m = EARTH_RADIUS_M * np.deg2rad(SPACING_DEG)
    return {"lon_centers": lon[:-1] + SPACING_DEG / 2, "lat_centers": lat[:-1] + SPACING_DEG / 2,
            "wet": wet, "dy_m": dy_m}


def _masked(value: np.ndarray, wet: np.ndarray) -> np.ma.MaskedArray:
    array = np.asarray(value)
    return np.ma.masked_where((~wet) | (~np.isfinite(array)), array)


def barotropic_streamfunction(u: np.ndarray, dy_m: float) -> np.ndarray:
    if u.shape != (NR, NY, NX):
        raise ValueError(f"UVEL must have shape ({NR}, {NY}, {NX})")
    depth_integrated_u = np.sum(u * DRF_M[:, None, None], axis=0)
    return np.cumsum(-depth_integrated_u * dy_m, axis=0) / 1.0e6


def _dyn_path(regime_root: Path, start_iteration: int, day: int) -> Path:
    iteration = start_iteration + day * STEPS_PER_DAY
    path = regime_root / "production" / "years_100_110" / f"dynState.{iteration:010d}.meta"
    if not path.is_file():
        raise FileNotFoundError(f"missing day-{day} dynState diagnostic: {path}")
    return path


def load_truth_streamfunctions(
    scratch_root: Path, regime: str, dy_m: float
) -> dict[int, np.ndarray]:
    regime_root = scratch_root / "mitgcm_turb_v1" / regime
    start_iteration = 100 * 103680  # model year 100, deltaT = 300 s
    psi_by_day: dict[int, np.ndarray] = {}
    for day in LEAD_DAYS:
        path = _dyn_path(regime_root, start_iteration, day)
        meta, values = read_mds(path)
        fields = mds_fields(meta, values)
        u = np.asarray(fields["UVEL"], dtype=float)
        psi_by_day[day] = barotropic_streamfunction(u, dy_m)
    return psi_by_day


def plot(
    psi_by_day: Mapping[int, np.ndarray], grid: Mapping[str, np.ndarray], regime: str, output: Path
) -> None:
    plt.rcParams.update(
        {"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "figure.dpi": 120, "savefig.dpi": 180}
    )
    wet = grid["wet"]
    bound = 45.0

    figure, axes = plt.subplots(1, len(LEAD_DAYS), figsize=(11.0, 2.6), sharex=True, sharey=True,
                                 constrained_layout=True)
    image = None
    for column, day in enumerate(LEAD_DAYS):
        image = axes[column].pcolormesh(
            grid["lon_centers"], grid["lat_centers"], _masked(psi_by_day[day], wet),
            cmap="RdBu_r", vmin=-bound, vmax=bound, shading="auto",
        )
        axes[column].set_title(f"Day {day}")
        axes[column].set_xlabel("Longitude (°)")
    axes[0].set_ylabel(f"{regime}\nLatitude (°)")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_facecolor("0.86")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Barotropic streamfunction (Sv)", shrink=0.82)
    figure.suptitle(
        rf"{regime} ground truth; production year 100; native $0.25^\circ$ grid (no FNO model yet)"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--regime", default="S0_turb", choices=("S0_turb", "S1_turb", "S2_turb"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    grid = _grid()
    psi_by_day = load_truth_streamfunctions(args.scratch_root.resolve(), args.regime, float(grid["dy_m"]))
    output = args.output_dir.resolve() / OUTPUT_NAME
    plot(psi_by_day, grid, args.regime, output)
    summary = {
        "regime": args.regime,
        "lead_days": list(LEAD_DAYS),
        "psi_min_sv": {day: float(np.min(psi[grid["wet"]])) for day, psi in psi_by_day.items()},
        "psi_max_sv": {day: float(np.max(psi[grid["wet"]])) for day, psi in psi_by_day.items()},
        "figure": str(output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
