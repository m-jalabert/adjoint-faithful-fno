"""Submit section 10.3's duplicate + tight-CG confirmatory controls.

For each of the 12 preassigned long directions (one per input group and
regime), at that family's provisional alpha (read from the two provisional-
selection reports, not hardcoded):

- duplicate both signs at production cg2dTargetResidual=1.E-7 (24 runs);
- rerun both signs at cg2dTargetResidual=1.E-10 (24 runs);
- rerun the corresponding 6 nominal anchors at cg2dTargetResidual=1.E-10 (6 runs).

All reuse the already-staged edited pickups from the provisional-alpha
run-signed calls; nothing is re-edited. Idempotent -- already-completed
segments (e.g. from a manual canary check) return their cached result.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "archive" / "src"))

import build_amplitude_pilot as pilot  # noqa: E402

V1_SELECTION = (
    PROJECT_ROOT
    / "outputs"
    / "af_fno"
    / "response"
    / "forward_response_v1"
    / "amplitude_pilot_provisional_selection_v1.json"
)
THETA_V2_SELECTION = (
    PROJECT_ROOT
    / "outputs"
    / "af_fno"
    / "response"
    / "forward_response_v1"
    / "amplitude_pilot_theta_v2_selection.json"
)


def _sbatch(env: dict[str, str], job_name: str) -> str:
    export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        job_name,
        "--export",
        export,
        "slurm/mitgcm/af_amplitude_pilot_segment.sbatch",
    ]
    return subprocess.run(
        command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    v1_alphas = pilot.load_json_strict(V1_SELECTION)["provisional_alpha_per_family"]
    theta_alpha = pilot.load_json_strict(THETA_V2_SELECTION)["provisional_alpha_theta"]
    provisional = {
        "U": v1_alphas["U"],
        "V": v1_alphas["V"],
        "SSH": v1_alphas["SSH"],
        "Theta": theta_alpha,
    }
    if any(value is None for value in provisional.values()):
        raise SystemExit(f"not every family has a provisional alpha: {provisional}")
    print("provisional alphas:", provisional)

    directions = [row for row in pilot._load_geometry() if row["long"]]
    if len(directions) != 12:
        raise SystemExit(f"expected 12 long directions, found {len(directions)}")

    submitted = 0
    for direction in directions:
        alpha = provisional[direction["family"]]
        for sign in (1, -1):
            for condition in ("duplicate", "tight"):
                label = f"{direction['regime']}-d{direction['anchor_day']}-{direction['family']}-{condition}-s{sign}"
                env = {
                    "AF_PILOT_KIND": "control",
                    "AF_REGIME": direction["regime"],
                    "AF_DAY": str(direction["anchor_day"]),
                    "AF_FAMILY": direction["family"],
                    "AF_ALPHA": str(alpha),
                    "AF_SIGN": str(sign),
                    "AF_CONDITION": condition,
                }
                job_id = _sbatch(env, f"af-pilot-ctrl-{label}")
                print(f"SUBMIT {label} (alpha={alpha}): job {job_id}")
                submitted += 1

    anchors = sorted({(d["regime"], d["anchor_day"]) for d in directions})
    for regime, day in anchors:
        env = {
            "AF_PILOT_KIND": "nominal",
            "AF_REGIME": regime,
            "AF_DAY": str(day),
            "AF_PILOT_TIGHT": "1",
        }
        label = f"{regime}-d{day}-nominal-tight"
        job_id = _sbatch(env, f"af-pilot-ctrl-{label}")
        print(f"SUBMIT {label}: job {job_id}")
        submitted += 1

    print(f"\n{submitted} control jobs submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
