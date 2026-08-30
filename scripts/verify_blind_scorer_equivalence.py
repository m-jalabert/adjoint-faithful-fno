"""Prove `response_validation_blind` reproduces the development scorer at leads 10-60.

Execution step 16 needs `S_resp^90`, which section 16.2's development scorer
does not compute (section 15.3 forbids any development score reading beyond
day 60). `src/oceanfno/response_validation_blind.py` extends the scoring in
lead only, importing every numerical helper from the pinned development
module rather than reimplementing it.

This script is the evidence for "in lead only": it scores one published
checkpoint on the *validation* role through both scorers and requires the
composites and every per-cell mean to agree exactly. Run before the blind
package is opened, so the blind responses are scored by code already shown
to be numerically identical to what produced the development numbers.

Reads the validation role only. No blind data is touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import zarr  # noqa: E402

from oceanfno import train as parent_train  # noqa: E402
from oceanfno.runtime import _device, require_runtime, torch  # noqa: E402
from oceanfno.dataset import assert_store_is_v3, store_wind_normalization  # noqa: E402
from oceanfno.model import ProductionArchitecture, build_model  # noqa: E402
from oceanfno.response_validation import evaluate_response_validation  # noqa: E402
from oceanfno.response_validation_blind import evaluate_blind_response  # noqa: E402

REPORT = (
    PROJECT_ROOT / "outputs/af_fno/C/model_c_adjoint_faithful_nominal_control_v1"
    / "seed_20260724/report.json"
)
DATASET = "/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/trajectories_v3.zarr"
TOLERANCE = 1e-12


def main() -> int:
    require_runtime()
    device = _device("cuda" if torch.cuda.is_available() else "cpu")
    report = json.loads(REPORT.read_text())
    group = zarr.open_consolidated(DATASET, mode="r")
    assert_store_is_v3(group)
    wet, _, _ = store_wind_normalization(group)
    wet = np.asarray(wet, dtype=bool)
    stored = np.load(report["published_checkpoint"]["normalization"])
    point_mean = stored["pointwise_mean"].astype(np.float32)
    point_scale = stored["pointwise_scale"].astype(np.float32)
    production = json.loads((PROJECT_ROOT / "config/model_c_production_1in_1out_spectralnorm_v1.json").read_text())
    statics, _ = parent_train.physical_static_block(production["sources"], group, point_mean, point_scale)
    model = build_model(ProductionArchitecture(**production["architecture"])).to(device)
    payload = torch.load(report["published_checkpoint"]["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    development = evaluate_response_validation(model, device, point_mean, point_scale, wet, statics)
    extended = evaluate_blind_response(model, device, point_mean, point_scale, wet, statics, role="validation")

    d, n = development["S_resp_10_60"], extended["S_resp_10_60"]
    shared = [k for k in development["cell_means"] if k in extended["cell_means"]]
    worst = max(abs(development["cell_means"][k] - extended["cell_means"][k]) for k in shared)
    identical = abs(d - n) < TOLERANCE and worst < TOLERANCE and "S_resp_90" not in extended

    print(f"development scorer  S_resp_10:60 = {d:.12f}")
    print(f"blind scorer, role=validation     = {n:.12f}")
    print(f"composite difference              = {abs(d - n):.3e}")
    print(f"cells compared {len(shared)}, worst per-cell difference = {worst:.3e}")
    print(f"S_resp_90 emitted on a 10-60 store: {'S_resp_90' in extended} (must be False)")
    print(f"VERDICT: {'IDENTICAL' if identical else 'DIFFERS -- do not open the blind package'}")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
