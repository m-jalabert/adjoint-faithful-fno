"""Submit the SSH-only follow-up pilot (config/forward_response_amplitude_pilot_ssh_v2.json).

Reuses the frozen v1 pilot geometry (same 6 SSH centres -- 3 point, 3 smooth
-- same per-direction horizon) and the same sbatch template as the main
pilot -- only the candidate alphas differ (smaller, per the v2 contract's
diagnosis). Both kernels are resubmitted, not just the smooth one that
produced the production-scale cap violations, so the frozen result still
satisfies section 8.5's "one common alpha_SSH ... for both point and smooth"
rule. Nominal branches are reused unchanged from v1; this submits only the
6 directions x 3 new alphas x 2 signs = 36 signed branches.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "archive" / "src"))

import build_amplitude_pilot as pilot  # noqa: E402

SSH_V2_CONTRACT = PROJECT_ROOT / "config" / "forward_response_amplitude_pilot_ssh_v2.json"


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
    contract = pilot.load_json_strict(SSH_V2_CONTRACT)
    alphas = contract["candidate_alphas"]
    directions = [row for row in pilot._load_geometry() if row["family"] == "SSH"]
    if len(directions) != 6:
        raise SystemExit(f"expected 6 SSH directions, found {len(directions)}")

    submitted = 0
    for direction in directions:
        for alpha in alphas:
            for sign in (1, -1):
                label = f"{direction['regime']}-d{direction['anchor_day']}-SSH-a{alpha}-s{sign}-v2"
                env = {
                    "AF_PILOT_KIND": "signed",
                    "AF_REGIME": direction["regime"],
                    "AF_DAY": str(direction["anchor_day"]),
                    "AF_FAMILY": "SSH",
                    "AF_ALPHA": str(alpha),
                    "AF_SIGN": str(sign),
                }
                job_id = _sbatch(env, f"af-pilot-{label}")
                print(f"SUBMIT {label}: job {job_id}")
                submitted += 1
    print(f"\n{submitted} jobs submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
