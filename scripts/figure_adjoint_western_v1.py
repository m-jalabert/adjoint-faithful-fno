"""2x3 MITgcm/TAF vs emulator(S_forced) figures at the frozen western target.

Unlike figure_adjoint_posthoc_v1.py, this is **not** a post-hoc diagnostic:
the western (pstar) target is the actual Gate A1 confirmatory target, and the
arrays read here are the same frozen ones behind
outputs/af_fno/adjoint/comparison_response_v1/gate_a1_result.json (verdict
negative, 2026-08-29) --

    outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2/mitgcm_s0_adjoint_v2.npz
    outputs/af_fno/adjoint/fno_{b,c}_seed_20260911_s0_adjoint_v1/fno_ft90_s0_adjoint_arrays.npz

No new run of any kind. This script only re-renders already-frozen numbers in
the same 2-row (B/C) x 3-panel (MITgcm/TAF, emulator S_forced, difference)
layout as the interior/eastern post-hoc figures (adjoint_comparison_figure),
so all three targets are directly comparable side by side. wet_mask/
target_ij/weight fields are verified equal across the two source files
before plotting (the same guarantee gate F6 gives the frozen result).

    python scripts/figure_adjoint_western_v1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from adjoint_comparison_figure import ROWS, render_2row_grid  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEAD_DAYS = (10, 30, 90)
ROW_SOURCES = (
    ("outputs/af_fno/adjoint/fno_b_seed_20260911_s0_adjoint_v1/fno_ft90_s0_adjoint_arrays.npz", "fno_b_seed_20260911"),
    ("outputs/af_fno/adjoint/fno_c_seed_20260911_s0_adjoint_v1/fno_ft90_s0_adjoint_arrays.npz", "fno_c_seed_20260911"),
)
MITGCM_NPZ = PROJECT_ROOT / "outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2/mitgcm_s0_adjoint_v2.npz"
OUTPUT_DIR = PROJECT_ROOT / "outputs/af_fno/adjoint/comparison_posthoc_v1/western/ssh_anomaly"


def figure_for(lead: int) -> Path:
    mit_npz = np.load(MITGCM_NPZ)
    wet = mit_npz["wet_mask"].astype(bool)
    target_ij = tuple(int(v) for v in mit_npz["target_ij"])
    lead_index = list(int(v) for v in mit_npz["lead_days"]).index(lead)
    reference = np.asarray(mit_npz["S_ssh_anomaly"][lead_index], dtype=np.float64)

    rows = []
    for npz_relative, identity_label in ROW_SOURCES:
        fno_npz = np.load(PROJECT_ROOT / npz_relative)
        if not np.array_equal(fno_npz["wet_mask"].astype(bool), wet) or tuple(
            int(v) for v in fno_npz["target_ij"]
        ) != target_ij:
            raise RuntimeError(f"{npz_relative} disagrees with {MITGCM_NPZ} about the basin or target")
        fno_lead_index = list(int(v) for v in fno_npz["lead_days"]).index(lead)
        emulator = np.asarray(fno_npz["S_ssh_anomaly_forced"][fno_lead_index], dtype=np.float64)
        rows.append((identity_label, reference, emulator))

    return render_2row_grid(
        OUTPUT_DIR / f"comparison_lead_{lead:03d}.png",
        f"frozen Gate A1 confirmatory target -- western boundary (pstar, i={target_ij[1] + 1}, j={target_ij[0] + 1}), "
        f"lead {lead} d, point SSH anomaly objective -- columns 1-2 share a colour scale, column 3 its own per row",
        wet, target_ij, rows,
    )


def main() -> int:
    for lead in LEAD_DAYS:
        path = figure_for(lead)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
