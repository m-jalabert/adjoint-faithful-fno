"""Cost-weight fields for the post-hoc interior/eastern adjoint diagnostic.

**Not part of the frozen adjoint-faithful contract.** Gate A1 already closed
v1 as a negative result on 2026-08-29
(outputs/af_fno/adjoint/comparison_response_v1/gate_a1_result.json), and the
interior/eastern exploratory extension in
config/adjoint_faithful_blind_adjoint_evaluation_v1.json was never unlocked
before B/C training (no pretraining_manifest was ever written), so per that
config's own rule ("absence_or_late_creation_of_manifest_means_no_exploratory_
test_runs") nothing generated here can count as confirmatory or exploratory
evidence for/against the response-training hypothesis. This is purely a
post-hoc "why did it fail" diagnostic, run after knowing which seed lost.

Targets are NOT chosen by looking at where B or C happen to do well: they are
the "interior" and "eastern" points already named in the frozen
outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2/README.md G1-90 grdchk table
(written 2026-08-19, before B/C training completed and long before the
2026-08-29 Gate A1 verdict), reusing the row (j=17, the pstar/WBC target's own
latitude) so the only thing that varies across the three points is distance
from the western wall:

    pstar (western, frozen v1/v2 target) = (i=2,  j=17)
    interior                             = (i=31, j=17)
    eastern                              = (i=61, j=17)

The gradient check at both new points already passed in that same frozen
table (best FD/adjoint agreement 2.8e-08 interior, 4.2e-07 eastern) so no new
grdchk run is needed here -- only new sensitivity maps.

Writes four new files under work/, alongside but never overwriting the frozen
western ones (costWeight_ssh_anomaly.bin, costWeight_ssh_anomaly_kernel.bin,
costWeight_mean_only.bin, all sha256-pinned in
config/adjoint_faithful_blind_adjoint_evaluation_v1.json):

    work/costWeight_interior_ssh_anomaly.bin
    work/costWeight_interior_ssh_anomaly_kernel.bin
    work/costWeight_eastern_ssh_anomaly.bin
    work/costWeight_eastern_ssh_anomaly_kernel.bin

mean_only is the basin-mean conservation probe, not tied to any (i,j), so the
existing work/costWeight_mean_only.bin is reused as-is and nothing new is
written for it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

from make_cost_weight import DEFAULT_KERNEL_AXIS, DEFAULT_KERNEL_RADIUS, DEFAULT_KERNEL_SIGMA, MDS_DTYPE, build_weight
from select_adjoint_target import CONTRACT_VERSION, read_mds_2d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = "/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/trajectories_v3.zarr"

#: (i_global, j_global), 1-based -- see module docstring for provenance.
TARGETS = {
    "interior": (31, 17),
    "eastern": (61, 17),
}


def main() -> int:
    contract = json.loads((PROJECT_ROOT / "config" / f"{CONTRACT_VERSION}.json").read_text())
    store = zarr.open(DATASET, mode="r")
    wet = np.asarray(store["wet_mask"]).astype(bool)
    rac = read_mds_2d(Path(contract["grid"]["rac_source"]))
    wet_area = float(contract["wet_area_m2"])

    recomputed = float((rac * wet).sum())
    if abs(recomputed - wet_area) > 1e-6 * wet_area:
        raise SystemExit(f"A_wet mismatch: contract {wet_area!r}, grid files {recomputed!r}")

    out_dir = PROJECT_ROOT / "work"
    manifest: dict[str, dict] = {}

    for name, (i_global, j_global) in TARGETS.items():
        i0, j0 = i_global - 1, j_global - 1
        if not wet[j0, i0]:
            raise SystemExit(f"{name} target (i={i_global}, j={j_global}) is not wet")

        for qoi, suffix in (("ssh_anomaly", ""), ("ssh_anomaly_kernel", "_kernel")):
            weight, provenance = build_weight(
                qoi,
                wet,
                rac,
                wet_area,
                j0,
                i0,
                kernel_axis=DEFAULT_KERNEL_AXIS,
                kernel_sigma=DEFAULT_KERNEL_SIGMA,
                kernel_radius=DEFAULT_KERNEL_RADIUS,
            )
            out_path = out_dir / f"costWeight_{name}_{qoi}.bin"
            weight.astype(MDS_DTYPE).tofile(out_path)

            stored = np.fromfile(out_path, dtype=MDS_DTYPE).reshape(wet.shape).astype(np.float64)
            if np.any(stored[~wet] != 0.0):
                raise SystemExit(f"{out_path} is non-zero on land")

            digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
            manifest[out_path.name] = {
                "target_name": name,
                "qoi": qoi,
                "i_global": i_global,
                "j_global": j_global,
                "sha256": digest,
                "provenance": provenance,
            }
            print(f"wrote {out_path}  sha256={digest}")

    manifest_path = out_dir / "costWeight_posthoc_v1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
