"""2x3 MITgcm/TAF vs emulator(S_forced) figures, raw SSH (point_only) objective.

Non-confirmatory, same status as figure_adjoint_posthoc_v1.py -- including
for western: point_only was never part of the frozen Gate A1 confirmatory
suite (that used ssh_anomaly/ssh_anomaly_kernel only), so all three targets'
raw-SSH figures here are equally post-hoc, unlike the ssh_anomaly/ and
ssh_anomaly_kernel/ western figures which reuse frozen confirmatory data.

J = eta(target, T) here, with no basin-mean subtraction -- see
scripts/make_cost_weight_raw_ssh_v1.py for why the frozen ssh_anomaly
objective subtracts the mean, and extract_mitgcm_adjoint_raw_ssh_v1.py for
the linearity check (S_point_only == S_ssh_anomaly - w_mean_only, confirmed
to ~4e-8 relative L2, matching Gate G3) that verifies this new weight family
before it reaches a figure.

    python scripts/figure_adjoint_raw_ssh_v1.py
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
TARGETS = ("western", "interior", "eastern")
LEAD_DAYS = (10, 30, 90)
OUTPUT_ROOT = PROJECT_ROOT / "outputs/af_fno/adjoint/comparison_posthoc_v1"


def figure_for(target: str, lead: int) -> Path:
    mit_npz = np.load(
        PROJECT_ROOT
        / f"outputs/af_fno/adjoint/mitgcm_s0_adjoint_posthoc_v1/{target}/mitgcm_s0_adjoint_raw_ssh_{target}_v1.npz"
    )
    wet = mit_npz["wet_mask"].astype(bool)
    target_ij = tuple(int(v) for v in mit_npz["target_ij"])
    lead_index = list(int(v) for v in mit_npz["lead_days"]).index(lead)
    reference = np.asarray(mit_npz["S_point_only"][lead_index], dtype=np.float64)

    rows = []
    for identity_key, identity_label in ROWS:
        forced_npz = np.load(
            PROJECT_ROOT / f"outputs/af_fno/adjoint/fno_adjoint_posthoc_v1/{identity_key}/{target}/raw_ssh.npz"
        )
        assert tuple(int(v) for v in forced_npz["target_ij"]) == target_ij
        forced_lead_index = list(int(v) for v in forced_npz["lead_days"]).index(lead)
        emulator = np.asarray(forced_npz["S_forced_point_only"][forced_lead_index], dtype=np.float64)
        rows.append((identity_label, reference, emulator))

    label = "frozen Gate A1 target, new objective" if target == "western" else "post-hoc diagnostic"
    return render_2row_grid(
        OUTPUT_ROOT / target / "raw_ssh" / f"comparison_lead_{lead:03d}.png",
        f"{label}, NOT a v1 result -- target {target} (i={target_ij[1] + 1}, j={target_ij[0] + 1}), "
        f"lead {lead} d, RAW SSH objective (no basin-mean subtraction) -- columns 1-2 share a colour scale",
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
