"""Cost-weight fields for the raw-SSH (no mean subtraction) diagnostic.

Same non-confirmatory status as scripts/make_cost_weight_posthoc_v1.py: Gate
A1 already closed v1 negative on 2026-08-29, so nothing built from these
weights is confirmatory or exploratory evidence under
config/adjoint_faithful_blind_adjoint_evaluation_v1.json.

The frozen ``ssh_anomaly`` objective is J = eta(target, T) - mean(eta(.,T)):
a delta at the target minus the (exactly conserved) basin-mean term, so that
the trivial barotropic mean mode does not contaminate the target signal (see
make_cost_weight.py). This script builds the objective WITHOUT that
subtraction -- J = eta(target, T), i.e. the raw local SSH -- using
make_cost_weight.py's own ``point_only`` qoi, at all three targets discussed
so far:

    western (pstar, frozen v1/v2 target) = (i=2,  j=17)
    interior                             = (i=31, j=17)
    eastern                              = (i=61, j=17)

Writes work/costWeight_{western,interior,eastern}_point_only.bin, alongside
but never overwriting the frozen work/costWeight_ssh_anomaly.bin or the
interior/eastern work/costWeight_{interior,eastern}_ssh_anomaly.bin from the
first post-hoc round.

Because the underlying adjoint is exactly linear in the cost weight (J is a
linear functional of eta(T) for any fixed source state, so its gradient is
linear in the functional's coefficients), this predicts an exact identity
that scripts/extract_mitgcm_adjoint_raw_ssh_v1.py checks once the new MITgcm
runs land:

    S_point_only = S_ssh_anomaly - w_mean_only

using the already gate-G3-verified w_mean_only (worst 3.57e-8 over 91 dumps
in the frozen v2 record) -- an independent check that this new weight family
is wired correctly, not just an assertion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

from make_cost_weight import MDS_DTYPE, build_weight
from select_adjoint_target import CONTRACT_VERSION, read_mds_2d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = "/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/trajectories_v3.zarr"

#: (i_global, j_global), 1-based.
TARGETS = {
    "western": (2, 17),
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

        weight, provenance = build_weight("point_only", wet, rac, wet_area, j0, i0)
        out_path = out_dir / f"costWeight_{name}_point_only.bin"
        weight.astype(MDS_DTYPE).tofile(out_path)

        stored = np.fromfile(out_path, dtype=MDS_DTYPE).reshape(wet.shape).astype(np.float64)
        if np.any(stored[~wet] != 0.0):
            raise SystemExit(f"{out_path} is non-zero on land")
        if stored[j0, i0] != 1.0:
            raise SystemExit(f"{out_path}: point_only weight at target is {stored[j0, i0]!r}, expected 1.0")

        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        manifest[out_path.name] = {
            "target_name": name,
            "qoi": "point_only",
            "i_global": i_global,
            "j_global": j_global,
            "sha256": digest,
            "provenance": provenance,
        }
        print(f"wrote {out_path}  sha256={digest}")

    manifest_path = out_dir / "costWeight_raw_ssh_v1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
