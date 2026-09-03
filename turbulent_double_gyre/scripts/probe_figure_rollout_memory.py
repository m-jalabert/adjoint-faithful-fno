"""Measure the peak GPU memory of one 15-member inference step at 248 x 248.

The figure package rolls all fifteen members as a single batch. At 62 x 62 that
was unremarkable; the validation stage learned the hard way that a batch which
was free on the coarse grid can exhaust a 32 GB card on this one. Fifteen is far
short of the 102 that did it, but the point is to measure rather than assume.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "src"))

from turbfno import figures as F  # noqa: E402
from turbfno.model import ProductionArchitecture, ProductionStepper, build_model  # noqa: E402

CHECKPOINT = Path(
    "/bigscratch/mjalabert314/bire_james25_repro/af_fno/models/turb"
) / "turb_forward_control_v1" / "selected.pt"


def main() -> int:
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(ProductionArchitecture()).to(device)
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    del payload
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated(device) / 2**30

    grid = ProductionArchitecture().grid_shape
    wet = np.ones(grid, dtype=bool)
    wet[:4] = wet[-4:] = wet[:, :4] = wet[:, -4:] = False
    stepper = ProductionStepper(
        model=model,
        device=device,
        wet=wet,
        mean=np.zeros((46, *grid), dtype=np.float32),
        scale=np.ones((46, *grid), dtype=np.float32),
        statics=np.zeros((1, 5, *grid), dtype=np.float32),
    )
    members = F.MEMBER_COUNT
    current = stepper.normalized_state(
        np.random.default_rng(0).standard_normal((members, 46, *grid)).astype(np.float32)
    )
    forcing = stepper.normalized_static(None, np.zeros(members, dtype=np.int64))

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(3):
            current = stepper.step(current, forcing)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    result = {
        "members": members,
        "grid": list(grid),
        "resident_model_gib": round(resident, 2),
        "peak_gib": round(peak, 2),
        "card_gib": round(total, 2),
        "headroom_gib": round(total - peak, 2),
        "verdict": "fits" if peak < 0.75 * total else "TOO TIGHT -- chunk the rollout",
    }
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "fits" else 1


if __name__ == "__main__":
    raise SystemExit(main())
