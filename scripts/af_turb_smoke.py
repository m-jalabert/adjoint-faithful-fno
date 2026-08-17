"""Short cold-start smoke test for the 0.25-degree turbulent configuration.

Runs a handful of model days from the tutorial initial condition with the exact
segment physics, then reports the elapsed time per step, the cg2d iteration
count, and the monitor extrema.  Its purpose is to catch a blown-up time step or
an unconverged elliptic solver in minutes rather than after a chain of
eight-hour jobs has been queued, and to measure the per-step cost that sizes the
real segments.

The scratch directory it writes is disposable and lives outside the campaign
root, so it can never be mistaken for a production segment.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from af_turb_trajectories import (
    DELTA_T_SECONDS,
    MPI_RANKS,
    STEPS_PER_DAY,
    render_data,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    steps = args.days * STEPS_PER_DAY
    (run_dir / "data").write_text(render_data(0, steps))
    input_root = project_root / "af_fno" / "mitgcm" / "input_turb"
    for name in ("data.pkg", "eedata", "bathy.bin", "windx_cosy.bin", "SST_relax.bin"):
        shutil.copy2(input_root / name, run_dir / name)
    shutil.copy2(input_root / "data.diagnostics.production", run_dir / "data.diagnostics")
    (run_dir / "mitgcmuv").symlink_to(args.executable.resolve())

    command = ["srun", "--mpi=pmix", "-n", str(MPI_RANKS), str(run_dir / "mitgcmuv")]
    started = time.monotonic()
    with (run_dir / "run.log").open("w") as stdout:
        completed = subprocess.run(
            command, cwd=run_dir, stdout=stdout, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.monotonic() - started

    log = (run_dir / "run.log").read_text(errors="replace")
    monitors: dict[str, float] = {}
    for key in (
        "ke_max",
        "ke_mean",
        "theta_max",
        "theta_min",
        "eta_max",
        "eta_min",
        "advcfl_uvel_max",
        "advcfl_vvel_max",
        "advcfl_wvel_max",
    ):
        found = re.findall(rf"{key}\s*=\s*([-\dEe+.]+)", log)
        if found:
            monitors[key] = float(found[-1])
    cg2d = [int(value) for value in re.findall(r"cg2d_iters\s*=\s*(\d+)", log)]
    report = {
        "returncode": completed.returncode,
        "days": args.days,
        "steps": steps,
        "delta_t_seconds": DELTA_T_SECONDS,
        "mpi_ranks": MPI_RANKS,
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_step": round(elapsed / steps, 5),
        "projected_seconds_per_model_year": round(elapsed / args.days * 360, 1),
        "projected_hours_per_125_years": round(elapsed / args.days * 360 * 125 / 3600, 2),
        "cg2d_iters_max": max(cg2d) if cg2d else None,
        "monitors": monitors,
        "daily_dumps": len(list(run_dir.glob("dynState.*.meta"))),
        "run_dir": str(run_dir),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if completed.returncode:
        print(log[-4000:])
    return completed.returncode


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
