"""2x3 MITgcm/TAF vs emulator(S_forced) figures, five-point kernel SSH-anomaly
objective -- zero new computation, all three targets.

Non-confirmatory for interior/eastern (see figure_adjoint_posthoc_v1.py);
frozen Gate A1 data for western (see figure_adjoint_western_v1.py). The
underlying arrays were already produced -- S_ssh_anomaly_kernel /
S_ssh_anomaly_kernel_forced sat unused in the frozen and round-1 post-hoc
npz stores (flagged as a gap in comparison_posthoc_v1/README.md's first
version) -- so this script is pure plotting, no new MITgcm or emulator run.

    python scripts/figure_adjoint_kernel_v1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from adjoint_comparison_figure import render_2row_grid  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEAD_DAYS = (10, 30, 90)
OUTPUT_ROOT = PROJECT_ROOT / "outputs/af_fno/adjoint/comparison_posthoc_v1"

WESTERN_ROWS = (
    ("outputs/af_fno/adjoint/fno_b_seed_20260911_s0_adjoint_v1/fno_ft90_s0_adjoint_arrays.npz", "fno_b_seed_20260911"),
    ("outputs/af_fno/adjoint/fno_c_seed_20260911_s0_adjoint_v1/fno_ft90_s0_adjoint_arrays.npz", "fno_c_seed_20260911"),
)
POSTHOC_ROWS = (("B_20260911", "fno_b_seed_20260911"), ("C_20260911", "fno_c_seed_20260911"))


def figure_western(lead: int) -> Path:
    mit_npz = np.load(PROJECT_ROOT / "outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2/mitgcm_s0_adjoint_v2.npz")
    wet = mit_npz["wet_mask"].astype(bool)
    target_ij = tuple(int(v) for v in mit_npz["target_ij"])
    lead_index = list(int(v) for v in mit_npz["lead_days"]).index(lead)
    reference = np.asarray(mit_npz["S_ssh_anomaly_kernel"][lead_index], dtype=np.float64)

    rows = []
    for npz_relative, identity_label in WESTERN_ROWS:
        fno_npz = np.load(PROJECT_ROOT / npz_relative)
        fno_lead_index = list(int(v) for v in fno_npz["lead_days"]).index(lead)
        emulator = np.asarray(fno_npz["S_ssh_anomaly_kernel_forced"][fno_lead_index], dtype=np.float64)
        rows.append((identity_label, reference, emulator))

    return render_2row_grid(
        OUTPUT_ROOT / "western" / "ssh_anomaly_kernel" / f"comparison_lead_{lead:03d}.png",
        f"frozen Gate A1 confirmatory target -- western boundary (pstar, i={target_ij[1] + 1}, j={target_ij[0] + 1}), "
        f"lead {lead} d, five-point meridional kernel SSH anomaly objective -- columns 1-2 share a colour scale",
        wet, target_ij, rows,
    )


def figure_posthoc(target: str, lead: int) -> Path:
    mit_npz = np.load(
        PROJECT_ROOT
        / f"outputs/af_fno/adjoint/mitgcm_s0_adjoint_posthoc_v1/{target}/mitgcm_s0_adjoint_posthoc_{target}_v1.npz"
    )
    wet = mit_npz["wet_mask"].astype(bool)
    target_ij = tuple(int(v) for v in mit_npz["target_ij"])
    lead_index = list(int(v) for v in mit_npz["lead_days"]).index(lead)
    reference = np.asarray(mit_npz["S_ssh_anomaly_kernel"][lead_index], dtype=np.float64)

    rows = []
    for identity_key, identity_label in POSTHOC_ROWS:
        forced_npz = np.load(
            PROJECT_ROOT / f"outputs/af_fno/adjoint/fno_adjoint_posthoc_v1/{identity_key}/{target}/s_forced.npz"
        )
        forced_lead_index = list(int(v) for v in forced_npz["lead_days"]).index(lead)
        emulator = np.asarray(forced_npz["S_forced_ssh_anomaly_kernel"][forced_lead_index], dtype=np.float64)
        rows.append((identity_label, reference, emulator))

    return render_2row_grid(
        OUTPUT_ROOT / target / "ssh_anomaly_kernel" / f"comparison_lead_{lead:03d}.png",
        f"post-hoc diagnostic, NOT a v1 result -- target {target} (i={target_ij[1] + 1}, j={target_ij[0] + 1}), "
        f"lead {lead} d, five-point meridional kernel SSH anomaly objective -- columns 1-2 share a colour scale",
        wet, target_ij, rows,
    )


def main() -> int:
    for lead in LEAD_DAYS:
        print(f"wrote {figure_western(lead)}")
    for target in ("interior", "eastern"):
        for lead in LEAD_DAYS:
            print(f"wrote {figure_posthoc(target, lead)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
