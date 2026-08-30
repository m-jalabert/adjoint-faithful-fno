"""Submit every branch of the amplitude pilot (plan step 7) via sbatch.

Reads the frozen pilot geometry (``build_amplitude_pilot.py
materialize-geometry`` must have already run) and the candidate alphas from
the pilot contract, then submits:

- one job per (direction, alpha, sign) = up to 144 signed branches, skipping
  any combination that fails the section-8.5 SSH physical cap pre-flight
  check (recorded as a static failure report instead of spending real
  compute on a run whose outcome is already analytically determined);
- one job per (regime, anchor day) nominal branch (6) plus its duplicate (6).

All jobs are independent -- no ``afterok`` chaining.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "archive" / "src"))

import build_amplitude_pilot as pilot  # noqa: E402


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
    directions = pilot._load_geometry()
    pilot_contract = pilot.load_json_strict(pilot.DEFAULT_PILOT_CONTRACT)
    alphas = pilot_contract["directions"]["candidate_alphas"]
    contract, roots, grid = pilot._load_sources(pilot.DEFAULT_DATASET_CONTRACT)
    sigma = pilot._load_normalizer(pilot_contract)

    submitted = skipped = 0
    for direction in directions:
        v_q = pilot.direction_vector(direction, grid.wet, sigma)
        for alpha in alphas:
            for sign in (1, -1):
                label = (
                    f"{direction['regime']}-d{direction['anchor_day']}-{direction['family']}"
                    f"-a{alpha}-s{sign}"
                )
                if direction["family"] == "SSH":
                    _field, _edits, peak = pilot.pickup_edits_for(direction, v_q, alpha, sign)
                    if peak > pilot.SSH_PEAK_METERS_MAX:
                        report_root = pilot.DEFAULT_REPORT_ROOT
                        report_root.mkdir(parents=True, exist_ok=True)
                        run_label = (
                            f"{direction['regime']}_d{direction['anchor_day']:04d}_SSH_"
                            f"a{pilot._alpha_token(alpha)}_{'plus' if sign == 1 else 'minus'}"
                        )
                        (report_root / f"{run_label}.json").write_text(
                            json.dumps(
                                {
                                    "kind": "signed",
                                    "run_label": run_label,
                                    "regime": direction["regime"],
                                    "day": direction["anchor_day"],
                                    "family": "SSH",
                                    "alpha": alpha,
                                    "sign": sign,
                                    "status": "failed_ssh_peak_cap",
                                    "ssh_peak_m": peak,
                                    "ssh_peak_m_max": pilot.SSH_PEAK_METERS_MAX,
                                    "note": (
                                        "section 10.1: a cap-triggered direction is recorded as "
                                        "a failure of this alpha, not silently clipped or run"
                                    ),
                                },
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        print(
                            f"SKIP  {label}: SSH peak {peak:.5f} m exceeds {pilot.SSH_PEAK_METERS_MAX} m cap"
                        )
                        skipped += 1
                        continue
                env = {
                    "AF_PILOT_KIND": "signed",
                    "AF_REGIME": direction["regime"],
                    "AF_DAY": str(direction["anchor_day"]),
                    "AF_FAMILY": direction["family"],
                    "AF_ALPHA": str(alpha),
                    "AF_SIGN": str(sign),
                }
                job_id = _sbatch(env, f"af-pilot-{label}")
                print(f"SUBMIT {label}: job {job_id}")
                submitted += 1

    seen_anchors = {(row["regime"], row["anchor_day"]) for row in directions}
    for regime, day in sorted(seen_anchors):
        for duplicate in (False, True):
            env = {"AF_PILOT_KIND": "nominal", "AF_REGIME": regime, "AF_DAY": str(day)}
            if duplicate:
                env["AF_PILOT_DUPLICATE"] = "1"
            label = f"{regime}-d{day}-nominal" + ("-dup" if duplicate else "")
            job_id = _sbatch(env, f"af-pilot-{label}")
            print(f"SUBMIT {label}: job {job_id}")
            submitted += 1

    print(f"\n{submitted} jobs submitted, {skipped} skipped (recorded as static cap failures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
