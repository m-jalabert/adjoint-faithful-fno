"""Freeze the MITgcm adjoint target cell p* and the wet-area constant.

Implements section 3 of docs/mitgcm_adjoint_ground_truth_plan.md.

    p* = argmax over the western search region of the time-mean surface speed,
         taken on the S0 training block only (days 0-5999), with the C-grid
         velocities averaged to cell centres first.

The result is written to config/mitgcm_adjoint_s0_target_v1.json and is
immutable once produced: every later stage reads p* and A_wet from that file,
and tests/test_mitgcm_adjoint.py re-derives them and fails on any mismatch.

Nothing here reads or writes the FNO tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

# --- frozen conventions (plan section 3.1) ---------------------------------

DATASET_VERSION = "trajectories_v3"
CONTRACT_VERSION = "mitgcm_adjoint_s0_target_v1"
REGIME = "S0"
REGIME_INDEX = 0

#: Selection uses the training block only.  The target must not be chosen with
#: data the FNO never saw.
SELECTION_DAYS = (0, 6000)

#: Surface velocity channels of the 46-channel state: U is 0..14, V is 15..29.
SURFACE_U_CHANNEL = 0
SURFACE_V_CHANNEL = 15

#: Western search region, 1-based global i.  The one-cell land rim is i=1, so
#: i=2 is the first wet column.
SEARCH_I_MIN = 2
SEARCH_I_MAX = 20

#: The plan's first draft excluded the wet column adjacent to the western wall,
#: a heuristic imported from configurations where the boundary current spans
#: several cells.  It does not hold here: with viscAh=5000 the Munk layer is
#: (A_h/beta)^(1/3) ~ 63 km against ~79 km zonal grid spacing at 45N, so the
#: current is one cell wide and lives entirely in i=2 (0.84 m/s at j=17,
#: against 0.17 m/s at i=3).  Excluding i=2 removes the jet rather than the
#: wall artifact, and leaves a flat field where the argmax is arbitrary.
#: i=2 is also the first cell of the "western 4 wet cells" band the FNO's
#: boundary loss and day-2000 anomaly diagnostics already use.
EXCLUDE_FIRST_WET_COLUMN = False

#: Rows to drop from each meridional end of the basin, so p* cannot land in a
#: corner where the sidewall rather than the jet sets the sensitivity.
EXCLUDE_ROWS_PER_END = 2

#: MITgcm S0 production starts at year 100.
BASE_ITERATION = 2_592_000
STEPS_PER_DAY = 72


class TargetSelectionError(RuntimeError):
    """Raised when the frozen selection rules cannot be satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centred_surface_speed(u_face: np.ndarray, v_face: np.ndarray) -> np.ndarray:
    """Average C-grid face velocities to cell centres, then take the speed.

    MITgcm holds UVEL on the western face of its cell and VVEL on the southern
    face, so the centred value needs the neighbour to the east / north.  Taking
    the speed from the face values directly would bias p* half a cell west.
    The last column / row has no neighbour and is left at the face value; both
    lie in the land rim, so they never enter the search region.
    """

    u_centre = u_face.copy()
    u_centre[..., :, :-1] = 0.5 * (u_face[..., :, :-1] + u_face[..., :, 1:])
    v_centre = v_face.copy()
    v_centre[..., :-1, :] = 0.5 * (v_face[..., :-1, :] + v_face[..., 1:, :])
    return np.sqrt(u_centre * u_centre + v_centre * v_centre)


def mean_surface_speed(store: Any, day_block: int = 500) -> np.ndarray:
    """Time-mean centred surface speed over the S0 training block."""

    start, stop = SELECTION_DAYS
    state = store["state"]
    total = np.zeros(state.shape[-2:], dtype=np.float64)
    counted = 0
    for begin in range(start, stop, day_block):
        end = min(begin + day_block, stop)
        u_face = np.asarray(state[REGIME_INDEX, begin:end, SURFACE_U_CHANNEL])
        v_face = np.asarray(state[REGIME_INDEX, begin:end, SURFACE_V_CHANNEL])
        total += centred_surface_speed(u_face, v_face).sum(axis=0, dtype=np.float64)
        counted += end - begin
    if counted != stop - start:
        raise TargetSelectionError(f"expected {stop - start} days, accumulated {counted}")
    return total / counted


def first_wet_column_mask(wet: np.ndarray) -> np.ndarray:
    """The first wet cell east of each row's western wall.

    Reuses the convention of ``oceanfno.dataset.western_boundary_mask`` at
    width 1, so the cell excluded here is the same one the FNO's boundary loss
    calls the start of its western band.
    """

    result = np.zeros_like(wet, dtype=bool)
    for row in range(wet.shape[0]):
        columns = np.flatnonzero(wet[row])
        if columns.size:
            result[row, columns[0]] = True
    return result


def search_region(wet: np.ndarray) -> np.ndarray:
    """The frozen search region W as a boolean mask."""

    region = wet.astype(bool).copy()

    # western band, 1-based global i in [SEARCH_I_MIN, SEARCH_I_MAX]
    band = np.zeros_like(region)
    band[:, SEARCH_I_MIN - 1 : SEARCH_I_MAX] = True
    region &= band

    # optionally drop the cell adjacent to the western wall; see the comment on
    # EXCLUDE_FIRST_WET_COLUMN for why this is off for S0
    if EXCLUDE_FIRST_WET_COLUMN:
        region &= ~first_wet_column_mask(wet.astype(bool))

    # drop the two northernmost and two southernmost wet rows of the basin
    wet_rows = np.flatnonzero(wet.astype(bool).any(axis=1))
    if wet_rows.size <= 2 * EXCLUDE_ROWS_PER_END:
        raise TargetSelectionError("basin has too few wet rows for the exclusion rule")
    keep_rows = wet_rows[EXCLUDE_ROWS_PER_END : wet_rows.size - EXCLUDE_ROWS_PER_END]
    row_mask = np.zeros(wet.shape[0], dtype=bool)
    row_mask[keep_rows] = True
    region &= row_mask[:, None]

    if not region.any():
        raise TargetSelectionError("the frozen search region is empty")
    return region


def pick_target(speed: np.ndarray, region: np.ndarray) -> tuple[int, int]:
    """argmax of the mean speed over the region; ties to lowest j then lowest i.

    ``np.argmax`` on a C-ordered array already resolves ties by lowest row then
    lowest column, which is the frozen rule.
    """

    masked = np.where(region, speed, -np.inf)
    flat = int(np.argmax(masked))
    j, i = np.unravel_index(flat, speed.shape)
    return int(j), int(i)


def read_mds_2d(meta_path: Path) -> np.ndarray:
    """Read a 62x62 MITgcm MDS record, honouring the precision in its .meta."""

    text = meta_path.read_text()
    if "float32" in text:
        dtype = ">f4"
    elif "float64" in text:
        dtype = ">f8"
    else:
        raise TargetSelectionError(f"cannot determine dataprec from {meta_path}")
    dims = [int(v) for v in text.split("dimList = [", 1)[1].split("]", 1)[0].replace("\n", "").split(",") if v.strip()]
    nx, ny = dims[0], dims[3]
    data = np.fromfile(meta_path.with_suffix(".data"), dtype=dtype)
    if data.size != nx * ny:
        raise TargetSelectionError(f"{meta_path.with_suffix('.data')} has {data.size} values, expected {nx * ny}")
    return data.reshape(ny, nx).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/trajectories_v3.zarr",
    )
    parser.add_argument(
        "--grid-dir",
        default="/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_long_truth_v1/S0/production/years_120_126",
        help="an S0 run directory holding RAC.data / RAC.meta",
    )
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_path = Path(arguments.out) if arguments.out else project_root / "config" / f"{CONTRACT_VERSION}.json"

    store = zarr.open(arguments.dataset, mode="r")
    wet = np.asarray(store["wet_mask"]).astype(bool)

    speed = mean_surface_speed(store)
    region = search_region(wet)
    j0, i0 = pick_target(speed, region)
    j_global, i_global = j0 + 1, i0 + 1

    rac_meta = Path(arguments.grid_dir) / "RAC.meta"
    rac = read_mds_2d(rac_meta)
    wet_area = float((rac * wet).sum())

    payload = {
        "version": CONTRACT_VERSION,
        "dataset": DATASET_VERSION,
        "regime": REGIME,
        "regime_index": REGIME_INDEX,
        "selection_days": list(SELECTION_DAYS),
        "search_region": {
            "i_min": SEARCH_I_MIN,
            "i_max": SEARCH_I_MAX,
            "exclude_first_wet_column": EXCLUDE_FIRST_WET_COLUMN,
            "exclude_rows_from_each_meridional_end": EXCLUDE_ROWS_PER_END,
            "candidate_cells": int(region.sum()),
        },
        "grid": {
            "nx": int(wet.shape[1]),
            "ny": int(wet.shape[0]),
            "wet_cell_count": int(wet.sum()),
            "rac_source": str(rac_meta),
            "rac_sha256": _sha256(rac_meta.with_suffix(".data")),
        },
        "i_global": i_global,
        "j_global": j_global,
        "i_index0": i0,
        "j_index0": j0,
        "mean_surface_speed_m_s": float(speed[j0, i0]),
        "mean_surface_speed_region_max_rank": 1,
        "wet_area_m2": wet_area,
        "mean_wet_cell_area_m2": wet_area / float(wet.sum()),
        "iteration_of_day": {
            "base_iteration": BASE_ITERATION,
            "steps_per_day": STEPS_PER_DAY,
            "day_7200": BASE_ITERATION + STEPS_PER_DAY * 7200,
            "day_7210": BASE_ITERATION + STEPS_PER_DAY * 7210,
            "day_7220": BASE_ITERATION + STEPS_PER_DAY * 7220,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"wrote {out_path}")
    print(f"  p*            = (i={i_global}, j={j_global})  [1-based global, Fortran]")
    print(f"  mean speed    = {speed[j0, i0]:.6f} m/s   over {region.sum()} candidate cells")
    print(f"  wet cells     = {int(wet.sum())}")
    print(f"  A_wet         = {wet_area:.10e} m^2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
