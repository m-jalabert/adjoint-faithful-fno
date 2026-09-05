"""2x3 emulator-vs-MITgcm figures for the basin-mean conservation functional.

Not target-specific (this objective is not tied to any grid cell), so there
is one set of 3 figures, not one per target -- see
outputs/af_fno/adjoint/comparison_posthoc_v1/mean_conservation/.

J = mean_ij(eta_ij(T)), the exact area-weighted basin mean. This
configuration's implicit free surface with exactConserv conserves that
quantity exactly, so MITgcm's own adjoint of it is provably constant in time
and equal to the weight field itself -- Gate G3 in mitgcm_s0_adjoint_v2
measured that to a worst 3.57e-8 over 91 daily dumps. The "MITgcm / TAF"
panel here is therefore just w_mean_only, unchanged across the three leads;
what varies is only the emulator panel (S_mean_only_forced, already computed
for the frozen western study). A large or lead-varying mismatch means the
emulator does NOT conserve the basin mean the way MITgcm does -- "improved
mean-mode preservation" is a named secondary endpoint in the study
(config/adjoint_faithful_blind_adjoint_evaluation_v1.json).

No new MITgcm or emulator run: both sides were already computed for the
frozen western comparison and are reused as-is.

    python scripts/figure_adjoint_mean_conservation_v1.py
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
ROW_SOURCES = (
    ("outputs/af_fno/adjoint/fno_b_seed_20260911_s0_adjoint_v1/fno_ft90_s0_adjoint_arrays.npz", "fno_b_seed_20260911"),
    ("outputs/af_fno/adjoint/fno_c_seed_20260911_s0_adjoint_v1/fno_ft90_s0_adjoint_arrays.npz", "fno_c_seed_20260911"),
)
MITGCM_NPZ = PROJECT_ROOT / "outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2/mitgcm_s0_adjoint_v2.npz"
OUTPUT_DIR = PROJECT_ROOT / "outputs/af_fno/adjoint/comparison_posthoc_v1/mean_conservation"


def figure_for(lead: int) -> Path:
    with np.load(MITGCM_NPZ) as mit_npz:
        wet = mit_npz["wet_mask"].astype(bool)
        reference = np.asarray(mit_npz["w_mean_only"], dtype=np.float64)  # Gate G3: == S_mean_only at every lead

    rows = []
    for npz_relative, identity_label in ROW_SOURCES:
        fno_npz = np.load(PROJECT_ROOT / npz_relative)
        fno_lead_index = list(int(v) for v in fno_npz["lead_days"]).index(lead)
        emulator = np.asarray(fno_npz["S_mean_only_forced"][fno_lead_index], dtype=np.float64)
        rows.append((identity_label, reference, emulator))

    return render_2row_grid(
        OUTPUT_DIR / f"comparison_lead_{lead:03d}.png",
        f"basin-mean conservation check, NOT target-specific -- lead {lead} d -- "
        "MITgcm panel is the weight field itself (Gate G3: exact, lead-independent); "
        "columns 1-2 share a colour scale",
        wet, None, rows,
        reference_label="MITgcm / TAF (= weight field, Gate G3)",
    )


def main() -> int:
    for lead in LEAD_DAYS:
        path = figure_for(lead)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
