"""Build the MITgcm adjoint cost weight field.

Implements section 4.4 of docs/mitgcm_adjoint_ground_truth_plan.md.  The
adjoint cost is the general linear functional

    J = sum_ij w_ij * etaN_ij(T)

so the quantity of interest is a runtime input file rather than compiled-in
code.  One TAF build therefore serves every linear-in-eta QoI in the study.

For the SSH anomaly at the frozen target cell p*,

    w = delta_{p*} - rA * maskC / A_wet

which is what ``--qoi ssh_anomaly`` (the default) writes.  ``--qoi mean_only``
writes just the second term: the adjoint of that functional is analytically
constant in time, because the implicit free surface with exactConserv conserves
the area integral of eta exactly, so the returned map must equal w itself at
every dump time.  That is acceptance gate G3.

``--qoi ssh_anomaly_kernel`` (docs/Adjoint_study_Phase_A.md section 1.2)
replaces the delta by a normalized Gaussian stencil, so the objective is a
small smooth average around p* rather than a single cell:

    w = g - rA * maskC / A_wet,      sum_ij g_ij = 1

The default stencil is **meridional**, sigma = 1 cell, radius 2 -- five cells
in j at the target's own i.  That anisotropy is a decision, not an oversight:

* the Munk layer here is one grid cell wide, (A_h/beta)^(1/3) ~ 63 km against
  ~79-96 km zonal spacing, so any zonal smoothing mixes the western boundary
  current (0.84 m/s at i=2) with the interior (0.17 m/s at i=3) and changes
  what is being measured;
* p* sits in the first wet column, so a symmetric 5x5 stencil reaches the land
  rim and off the grid.  Only 15 of its 25 cells are usable, and renormalizing
  over those displaces the effective centroid 0.504 cells east, off the jet;
* the emulator's spectral path cuts at |k| <= 16 of 37 on its 74x74
  domain-padded grid -- a 4.63-cell resolution floor -- so a delta functional
  puts weight on modes it structurally cannot carry.  sigma = 1 attenuates the
  first unrepresented mode to 0.40.  Partial by design: the zonal delta is left
  intact.

Because the whole functional is a runtime input file, sigma, radius and axis
are all knobs rather than commitments -- a different choice costs one MITgcm
run and one backward pass, with no rebuild and no TAF submission.

Output is 62x62 big-endian float32 -- the same convention as bathy.bin,
windx_cosy.bin and SST_relax.bin, because MITgcm reads it through
READ_REC_XY_RL, which uses readBinaryPrec, and readBinaryPrec must stay 32 for
those tutorial inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

from select_adjoint_target import CONTRACT_VERSION, read_mds_2d

#: MITgcm reads this through READ_REC_XY_RL at readBinaryPrec, which is 32 for
#: this configuration because bathy.bin / windx_cosy.bin / SST_relax.bin are
#: float32.  Big-endian is the MDS convention.
MDS_DTYPE = ">f4"

QOI_CHOICES = (
    "ssh_anomaly",
    "mean_only",
    "point_only",
    "ssh_anomaly_kernel",
    "kernel_only",
)

#: Stencils that replace the delta at p* by a normalized smooth kernel.
KERNEL_QOI = ("ssh_anomaly_kernel", "kernel_only")

#: The Phase A defaults, frozen in config/adjoint_phase_a_v1.json.
DEFAULT_KERNEL_AXIS = "meridional"
DEFAULT_KERNEL_SIGMA = 1.0
DEFAULT_KERNEL_RADIUS = 2

KERNEL_AXES = ("meridional", "zonal", "isotropic")


class CostWeightError(RuntimeError):
    """Raised when the weight field fails its own consistency checks."""


def build_kernel(
    wet: np.ndarray,
    j_index0: int,
    i_index0: int,
    *,
    axis: str = DEFAULT_KERNEL_AXIS,
    sigma: float = DEFAULT_KERNEL_SIGMA,
    radius: int = DEFAULT_KERNEL_RADIUS,
    allow_land: bool = False,
) -> tuple[np.ndarray, dict]:
    """A normalized Gaussian stencil centred on p*, summing to exactly one.

    ``allow_land`` is refused by default on purpose.  A stencil that overlaps
    land has to be renormalized over the wet cells that remain, and that moves
    the effective centroid away from the target: for a symmetric 5x5 at this
    p* it lands 0.504 cells east, in water five times slower.  The kernel then
    no longer measures SSH in the boundary current, which is the entire reason
    p* was frozen where it was.  Refusing is better than silently measuring
    something else.
    """

    if axis not in KERNEL_AXES:
        raise CostWeightError(f"unknown kernel axis {axis!r}, expected one of {KERNEL_AXES}")
    if sigma <= 0.0:
        raise CostWeightError(f"kernel sigma must be positive, got {sigma!r}")
    if radius < 1:
        raise CostWeightError(f"kernel radius must be at least 1, got {radius!r}")

    offsets = range(-int(radius), int(radius) + 1)
    if axis == "meridional":
        cells = [(j_index0 + d, i_index0) for d in offsets]
    elif axis == "zonal":
        cells = [(j_index0, i_index0 + d) for d in offsets]
    else:
        cells = [(j_index0 + dj, i_index0 + di) for dj in offsets for di in offsets]

    kernel = np.zeros(wet.shape, dtype=np.float64)
    off_grid, on_land = [], []
    for j, i in cells:
        if not (0 <= j < wet.shape[0] and 0 <= i < wet.shape[1]):
            off_grid.append((int(j), int(i)))
            continue
        if not wet[j, i]:
            on_land.append((int(j), int(i)))
            continue
        kernel[j, i] = np.exp(-0.5 * ((j - j_index0) ** 2 + (i - i_index0) ** 2) / sigma**2)

    if (off_grid or on_land) and not allow_land:
        raise CostWeightError(
            f"the {axis} sigma={sigma} radius={radius} stencil at "
            f"(i={i_index0 + 1}, j={j_index0 + 1}) leaves the basin: "
            f"{len(off_grid)} cells off the grid, {len(on_land)} on land. "
            "Renormalizing over the rest would displace the effective centroid "
            "off the target -- pass --allow-land only if that is intended."
        )

    total = float(kernel.sum())
    if total <= 0.0:
        raise CostWeightError("the kernel stencil has no wet cells")
    kernel /= total

    grid_j, grid_i = np.mgrid[0 : wet.shape[0], 0 : wet.shape[1]]
    provenance = {
        "axis": axis,
        "sigma_cells": float(sigma),
        "radius_cells": int(radius),
        "stencil_cells": len(cells),
        "wet_cells_used": int((kernel > 0.0).sum()),
        "cells_off_grid": off_grid,
        "cells_on_land": on_land,
        "centroid_j": float((kernel * grid_j).sum()),
        "centroid_i": float((kernel * grid_i).sum()),
        "centroid_displacement_cells": float(
            np.hypot((kernel * grid_j).sum() - j_index0, (kernel * grid_i).sum() - i_index0)
        ),
        "peak_weight": float(kernel.max()),
        "profile": [float(v) for v in kernel[kernel > 0.0]],
    }
    return kernel, provenance


def build_weight(
    qoi: str,
    wet: np.ndarray,
    rac: np.ndarray,
    wet_area: float,
    j_index0: int,
    i_index0: int,
    *,
    kernel_axis: str = DEFAULT_KERNEL_AXIS,
    kernel_sigma: float = DEFAULT_KERNEL_SIGMA,
    kernel_radius: int = DEFAULT_KERNEL_RADIUS,
    allow_land: bool = False,
) -> tuple[np.ndarray, dict]:
    """Assemble the 62x62 weight field for one quantity of interest."""

    if qoi not in QOI_CHOICES:
        raise CostWeightError(f"unknown qoi {qoi!r}, expected one of {QOI_CHOICES}")
    if not wet[j_index0, i_index0]:
        raise CostWeightError(f"target cell (i={i_index0 + 1}, j={j_index0 + 1}) is not wet")

    weight = np.zeros(wet.shape, dtype=np.float64)
    provenance: dict = {"qoi": qoi}
    if qoi in ("ssh_anomaly", "mean_only", "ssh_anomaly_kernel"):
        weight -= rac * wet / wet_area
    if qoi in ("ssh_anomaly", "point_only"):
        weight[j_index0, i_index0] += 1.0
    if qoi in KERNEL_QOI:
        kernel, kernel_provenance = build_kernel(
            wet,
            j_index0,
            i_index0,
            axis=kernel_axis,
            sigma=kernel_sigma,
            radius=kernel_radius,
            allow_land=allow_land,
        )
        weight += kernel
        provenance["kernel"] = kernel_provenance
    # dry cells never influence the cost: etaN there is not a degree of freedom
    weight *= wet
    return weight, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qoi", default="ssh_anomaly", choices=QOI_CHOICES)
    parser.add_argument("--out", default=None, help="output .bin path")
    parser.add_argument("--kernel-axis", default=DEFAULT_KERNEL_AXIS, choices=KERNEL_AXES)
    parser.add_argument("--kernel-sigma", type=float, default=DEFAULT_KERNEL_SIGMA)
    parser.add_argument("--kernel-radius", type=int, default=DEFAULT_KERNEL_RADIUS)
    parser.add_argument(
        "--allow-land",
        action="store_true",
        help="permit a stencil that overlaps land or the grid edge, renormalizing over "
        "the wet cells that remain.  This displaces the effective centroid off the "
        "target and is refused by default.",
    )
    parser.add_argument(
        "--dataset",
        default="/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/trajectories_v3.zarr",
    )
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    contract = json.loads((project_root / "config" / f"{CONTRACT_VERSION}.json").read_text())

    store = zarr.open(arguments.dataset, mode="r")
    wet = np.asarray(store["wet_mask"]).astype(bool)
    rac = read_mds_2d(Path(contract["grid"]["rac_source"]))
    wet_area = float(contract["wet_area_m2"])
    j0, i0 = int(contract["j_index0"]), int(contract["i_index0"])

    # the contract's A_wet must be the one implied by these grid files
    recomputed = float((rac * wet).sum())
    if abs(recomputed - wet_area) > 1e-6 * wet_area:
        raise CostWeightError(f"A_wet mismatch: contract {wet_area!r}, grid files {recomputed!r}")

    weight, provenance = build_weight(
        arguments.qoi,
        wet,
        rac,
        wet_area,
        j0,
        i0,
        kernel_axis=arguments.kernel_axis,
        kernel_sigma=arguments.kernel_sigma,
        kernel_radius=arguments.kernel_radius,
        allow_land=arguments.allow_land,
    )

    out_path = Path(arguments.out) if arguments.out else project_root / "work" / f"costWeight_{arguments.qoi}.bin"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weight.astype(MDS_DTYPE).tofile(out_path)

    # read back exactly what MITgcm will read, and report on that, not on the
    # float64 array we started from
    stored = np.fromfile(out_path, dtype=MDS_DTYPE).reshape(wet.shape).astype(np.float64)
    if stored.size != wet.size:
        raise CostWeightError(f"{out_path} has {stored.size} values, expected {wet.size}")
    if np.any(stored[~wet] != 0.0):
        raise CostWeightError("weight field is non-zero on land")

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()

    print(f"wrote {out_path}  ({out_path.stat().st_size} bytes, {MDS_DTYPE})")
    print(f"  qoi              = {arguments.qoi}")
    print(f"  p*               = (i={i0 + 1}, j={j0 + 1})")
    print(f"  w[p*]            = {stored[j0, i0]:+.10e}")
    print(f"  w mean-term cell = {-rac[j0, i0] / wet_area:+.10e}   (-rA/A_wet at p*)")
    if "kernel" in provenance:
        k = provenance["kernel"]
        print(f"  kernel           = {k['axis']}, sigma={k['sigma_cells']}, radius={k['radius_cells']}")
        print(f"    stencil        = {k['wet_cells_used']} wet of {k['stencil_cells']} requested")
        print(f"    profile        = {[round(v, 6) for v in k['profile']]}")
        print(f"    sums to        = {sum(k['profile']):.12f}")
        print(f"    centroid       = (i0={k['centroid_i']:.6f}, j0={k['centroid_j']:.6f})"
              f"   displacement {k['centroid_displacement_cells']:.6f} cells")
    print(f"  sum over wet     = {stored[wet].sum():+.6e}   (0 for ssh_anomaly, up to float32 round-off)")
    print(f"  min / max        = {stored.min():+.6e} / {stored.max():+.6e}")
    print(f"  non-zero cells   = {int((stored != 0).sum())}  of {wet.size}")
    print(f"  sha256           = {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
