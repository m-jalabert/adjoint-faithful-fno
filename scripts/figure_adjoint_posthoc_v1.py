"""2x3 MITgcm/TAF vs emulator(S_forced) figures, point SSH-anomaly objective.

**Not part of the frozen adjoint-faithful contract.** Gate A1 already closed
v1 negative on 2026-08-29
(outputs/af_fno/adjoint/comparison_response_v1/gate_a1_result.json), and the
interior/eastern extension was never unlocked before B/C training (see
scripts/make_cost_weight_posthoc_v1.py). These figures are a post-hoc
diagnostic -- not evidence for or against the response-training hypothesis,
and not a v1 result.

Layout (adjoint_comparison_figure.render_2row_grid), one PNG per (target,
lead), under comparison_posthoc_v1/<target>/ssh_anomaly/:

    row 1  fno_b_seed_20260911   [MITgcm/TAF] [emulator S_forced] [emulator - MITgcm]
    row 2  fno_c_seed_20260911   [MITgcm/TAF] [emulator S_forced] [emulator - MITgcm]

    python scripts/figure_adjoint_posthoc_v1.py
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
TARGETS = ("interior", "eastern")
LEAD_DAYS = (10, 30, 90)
OUTPUT_ROOT = PROJECT_ROOT / "outputs/af_fno/adjoint/comparison_posthoc_v1"


def figure_for(target: str, lead: int) -> Path:
    mit_npz = np.load(
        PROJECT_ROOT
        / f"outputs/af_fno/adjoint/mitgcm_s0_adjoint_posthoc_v1/{target}/mitgcm_s0_adjoint_posthoc_{target}_v1.npz"
    )
    wet = mit_npz["wet_mask"].astype(bool)
    target_ij = tuple(int(v) for v in mit_npz["target_ij"])
    lead_index = list(int(v) for v in mit_npz["lead_days"]).index(lead)
    reference = np.asarray(mit_npz["S_ssh_anomaly"][lead_index], dtype=np.float64)

    rows = []
    for identity_key, identity_label in ROWS:
        forced_npz = np.load(
            PROJECT_ROOT / f"outputs/af_fno/adjoint/fno_adjoint_posthoc_v1/{identity_key}/{target}/s_forced.npz"
        )
        assert tuple(int(v) for v in forced_npz["target_ij"]) == target_ij
        forced_lead_index = list(int(v) for v in forced_npz["lead_days"]).index(lead)
        emulator = np.asarray(forced_npz["S_forced_ssh_anomaly"][forced_lead_index], dtype=np.float64)
        rows.append((identity_label, reference, emulator))

    return render_2row_grid(
        OUTPUT_ROOT / target / "ssh_anomaly" / f"comparison_lead_{lead:03d}.png",
        f"post-hoc diagnostic, NOT a v1 result -- target {target} (i={target_ij[1] + 1}, j={target_ij[0] + 1}), "
        f"lead {lead} d, point SSH anomaly objective -- columns 1-2 share a colour scale, column 3 its own per row",
        wet, target_ij, rows,
    )


def main() -> int:
    for target in TARGETS:
        for lead in LEAD_DAYS:
            path = figure_for(target, lead)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
