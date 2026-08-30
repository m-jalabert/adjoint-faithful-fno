"""Submit every step-9 production response branch (plan section 25 step 9) via sbatch.

Reads ``stage_forward_response_run.list_work()`` and submits:

- one job per signed production direction (both signs run inside a single
  ``run-signed`` call, matching how ``stage_forward_response_run.run_signed``
  is written) -- up to 888 jobs;
- one job per new shared nominal group (excludes the 6 pilot-overlap
  anchors, which reuse the amplitude pilot's own existing 90-day runs and
  need no new job) -- up to 45 jobs.

All jobs are independent -- no ``afterok`` chaining, matching
``submit_amplitude_pilot.py``'s convention.

``--dry-run`` prints what would be submitted without calling ``sbatch``.
``--limit N`` submits only the first ``N`` signed directions and first ``N``
nominal groups -- use this for a canary before submitting the full run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "archive" / "src"))

import stage_forward_response_run as staging  # noqa: E402


def _sbatch(env: dict[str, str], job_name: str) -> str:
    export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        job_name,
        "--export",
        export,
        "slurm/mitgcm/af_forward_response_array.sbatch",
    ]
    return subprocess.run(
        command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print without calling sbatch")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="submit only the first N signed directions and first N nominal groups (canary)",
    )
    arguments = parser.parse_args(argv)

    work = staging.list_work()
    signed = work["signed_directions"]
    nominal = work["new_nominal_groups"]
    if arguments.limit is not None:
        signed = signed[: arguments.limit]
        nominal = nominal[: arguments.limit]

    print(
        f"Plan: {len(signed)} signed directions ({sum(1 for s in signed if not s['long'])} short, "
        f"{sum(1 for s in signed if s['long'])} long), {len(nominal)} new nominal groups, "
        f"{len(work['reused_pilot_nominal_groups'])} pilot-overlap groups reused (no new job).",
        file=sys.stderr,
    )

    submitted = 0
    for item in signed:
        label = f"{item['regime']}-d{item['anchor_day']:04d}-{item['family']}-q{item['direction_slot']}"
        if arguments.dry_run:
            print(f"DRY-RUN signed {label} direction_id={item['direction_id']}")
        else:
            env = {
                "AF_RESPONSE_KIND": "signed",
                "AF_REGIME": item["regime"],
                "AF_DAY": str(item["anchor_day"]),
                "AF_FAMILY": item["family"],
                "AF_SLOT": str(item["direction_slot"]),
            }
            job_id = _sbatch(env, f"af-resp-{label}")
            print(f"SUBMIT signed {label}: job {job_id}")
        submitted += 1

    for group in nominal:
        label = f"{group['role']}-{group['regime']}-d{group['anchor_day']:04d}-nominal"
        if arguments.dry_run:
            print(f"DRY-RUN nominal {label} duration_days={group['duration_days']}")
        else:
            env = {
                "AF_RESPONSE_KIND": "nominal",
                "AF_ROLE": group["role"],
                "AF_REGIME": group["regime"],
                "AF_DAY": str(group["anchor_day"]),
            }
            job_id = _sbatch(env, f"af-resp-{label}")
            print(f"SUBMIT nominal {label}: job {job_id}")
        submitted += 1

    verb = "would be submitted" if arguments.dry_run else "submitted"
    print(f"\n{submitted} jobs {verb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
