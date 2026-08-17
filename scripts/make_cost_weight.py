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

QOI_CHOICES = ("ssh_anomaly", "mean_only", "point_only")


class CostWeightError(RuntimeError):
    """Raised when the weight field fails its own consistency checks."""


def build_weight(
    qoi: str,
    wet: np.ndarray,
    rac: np.ndarray,
    wet_area: float,
    j_index0: int,
    i_index0: int,
) -> np.ndarray:
    """Assemble the 62x62 weight field for one quantity of interest."""

    if qoi not in QOI_CHOICES:
        raise CostWeightError(f"unknown qoi {qoi!r}, expected one of {QOI_CHOICES}")
    if not wet[j_index0, i_index0]:
        raise CostWeightError(f"target cell (i={i_index0 + 1}, j={j_index0 + 1}) is not wet")

    weight = np.zeros(wet.shape, dtype=np.float64)
    if qoi in ("ssh_anomaly", "mean_only"):
        weight -= rac * wet / wet_area
    if qoi in ("ssh_anomaly", "point_only"):
        weight[j_index0, i_index0] += 1.0
    # dry cells never influence the cost: etaN there is not a degree of freedom
    weight *= wet
    return weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qoi", default="ssh_anomaly", choices=QOI_CHOICES)
    parser.add_argument("--out", default=None, help="output .bin path")
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

    weight = build_weight(arguments.qoi, wet, rac, wet_area, j0, i0)

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
    print(f"  sum over wet     = {stored[wet].sum():+.6e}   (0 for ssh_anomaly, up to float32 round-off)")
    print(f"  min / max        = {stored.min():+.6e} / {stored.max():+.6e}")
    print(f"  non-zero cells   = {int((stored != 0).sum())}  of {wet.size}")
    print(f"  sha256           = {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
