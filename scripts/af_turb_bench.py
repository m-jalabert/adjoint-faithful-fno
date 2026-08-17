"""Instrumented micro-benchmark of the turbulent configuration.

Runs a small number of steps with a monitor dump at the end, then reports the
wall clock per step, the cg2d iteration count, and the MITgcm internal timers.
Its job is to tell apart the two candidate explanations for a slow segment --
an elliptic solver that never converges, and a solver whose global reductions
do not scale -- so the fix can be aimed at the right one.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from af_turb_trajectories import DELTA_T_SECONDS, render_data


def _patch(text: str, **overrides: str) -> str:
    for key, value in overrides.items():
        text, count = re.subn(rf"(?m)^ {key}=[^,]*,", f" {key}={value},", text)
        if count != 1:
            raise SystemExit(f"could not patch {key} ({count} matches)")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--steps", type=int, default=288)
    parser.add_argument("--target-residual", default="1.E-7")
    parser.add_argument("--max-iters", default="1000")
    parser.add_argument("--srcg", action="store_true", help="use the single-reduction CG solver")
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    deck = _patch(
        render_data(0, args.steps),
        cg2dTargetResidual=args.target_residual,
        cg2dMaxIters=args.max_iters,
        monitorFreq=f"{args.steps * DELTA_T_SECONDS}.",
    )
    if args.srcg:
        deck = deck.replace(" cg2dMaxIters=", " useSRCGSolver=.TRUE.,\n cg2dMaxIters=")
    (run_dir / "data").write_text(deck)

    input_root = project_root / "af_fno" / "mitgcm" / "input_turb"
    for name in ("data.pkg", "eedata", "bathy.bin", "windx_cosy.bin", "SST_relax.bin"):
        shutil.copy2(input_root / name, run_dir / name)
    # No daily diagnostics: this measures the time stepping, not the writer.
    (run_dir / "data.diagnostics").write_text(
        " &DIAGNOSTICS_LIST\n dumpAtLast=.FALSE.,\n diag_mnc=.FALSE.,\n &\n"
        " &DIAG_STATIS_PARMS\n &\n"
    )
    (run_dir / "mitgcmuv").symlink_to(args.executable.resolve())

    command = ["srun", "--mpi=pmix", "-n", str(args.ranks), str(run_dir / "mitgcmuv")]
    started = time.monotonic()
    with (run_dir / "run.log").open("w") as stdout:
        completed = subprocess.run(
            command, cwd=run_dir, stdout=stdout, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.monotonic() - started

    log = (run_dir / "STDOUT.0000").read_text(errors="replace")
    monitors = {}
    for key in ("cg2d_iters", "ke_max", "ke_mean", "advcfl_uvel_max", "advcfl_wvel_max"):
        found = re.findall(rf"%MON {key}\s*=\s*([-\dEe+.]+)", log)
        if found:
            monitors[key] = float(found[-1])
    timers = dict(
        re.findall(r"\(PID\.TID 0000\.0001\)\s+(\S[^:]*?)\s+\[SECONDS\]:\s+([\d.E+-]+)", log)
    )
    interesting = {
        name.strip(): float(value)
        for name, value in timers.items()
        if any(k in name.upper() for k in ("SOLVE_FOR_PRESSURE", "DYNAMICS", "THERMO", "ALL"))
    }
    report = {
        "returncode": completed.returncode,
        "ranks": args.ranks,
        "steps": args.steps,
        "srcg": args.srcg,
        "target_residual": args.target_residual,
        "max_iters": args.max_iters,
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_step": round(elapsed / args.steps, 4),
        "projected_hours_per_125_years": round(
            elapsed / args.steps * 103680 * 125 / 3600, 1
        ),
        "monitors": monitors,
        "timers": interesting,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if completed.returncode:
        print((run_dir / "run.log").read_text(errors="replace")[-3000:])
    return completed.returncode


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
