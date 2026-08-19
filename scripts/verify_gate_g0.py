"""Acceptance gate G0: the adjoint linearises about the trajectory the FNO sees.

Implements section 7 / gate G0 of docs/mitgcm_adjoint_ground_truth_plan.md.

The 'pickup' mode run restarts the forward model from the archived day-7200
pickup and integrates to day 7220, writing daily dynState/surfState snapshots.
Those snapshots must reproduce trajectories_v3.zarr bit for bit.  If they do
not, the adjoint is being taken about a different trajectory than the FNO is
evaluated on, and every sensitivity map downstream is meaningless.

Bit-for-bit is the right standard here, not "close": the same executable, the
same 2x2 decomposition and the same node type produced the archived data, so
any difference at all signals a changed input, a changed build, or a changed
restart, none of which should be tolerated silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

from select_adjoint_target import BASE_ITERATION, STEPS_PER_DAY

#: Channel layout of the 46-channel state: U 0..14, V 15..29, theta 30..44, eta 45.
DYNSTATE_CHANNELS = 45  # UVEL, VVEL, THETA -> 15 levels each
SURFSTATE_CHANNEL = 45  # ETAN
NR = 15


def read_mds_record(run_dir: Path, prefix: str, iteration: int, nz: int) -> np.ndarray:
    """Read one MITgcm MDS output record written with useSingleCpuIO=.TRUE."""

    meta = run_dir / f"{prefix}.{iteration:010d}.meta"
    data = run_dir / f"{prefix}.{iteration:010d}.data"
    if not data.is_file():
        raise FileNotFoundError(f"missing {data}")
    text = meta.read_text()
    dtype = ">f4" if "float32" in text else ">f8"
    values = np.fromfile(data, dtype=dtype)
    expected = 62 * 62 * nz
    if values.size != expected:
        raise ValueError(f"{data} holds {values.size} values, expected {expected}")
    return values.reshape(nz, 62, 62).astype(np.float32)


def center_velocities(dyn: np.ndarray) -> np.ndarray:
    """Put UVEL/VVEL on cell centres, exactly as the dataset builder does.

    The raw MDS `dynState` holds U on west faces and V on south faces, but
    trajectories_v3.zarr stores them at cell centres: af_data.py:178-179 applies

        u_center = 0.5 * (u + np.roll(u, -1, axis=-1))
        v_center = 0.5 * (v + np.roll(v, -1, axis=-2))

    and records the convention in its manifest as "u_centering" /
    "v_centering".  THETA and ETAN are already centred and pass through.

    Comparing raw MDS against the archive without this step makes every wet
    velocity differ by O(1e-1) while THETA and ETAN match bit-for-bit -- which
    is a statement about the C grid, not about the restart.
    """

    u = dyn[0:NR]
    v = dyn[NR : 2 * NR]
    theta = dyn[2 * NR : 3 * NR]
    u_center = 0.5 * (u + np.roll(u, -1, axis=-1))
    v_center = 0.5 * (v + np.roll(v, -1, axis=-2))
    return np.concatenate([u_center, v_center, theta], axis=0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v1/pickup",
    )
    parser.add_argument(
        "--dataset",
        default="/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/trajectories_v3.zarr",
    )
    parser.add_argument("--first-day", type=int, default=7200)
    parser.add_argument("--last-day", type=int, default=7220)
    parser.add_argument("--regime-index", type=int, default=0)
    parser.add_argument("--report", default=None)
    arguments = parser.parse_args()

    run_dir = Path(arguments.run_dir)
    store = zarr.open(arguments.dataset, mode="r")
    state = store["state"]

    rows: list[dict] = []
    worst = 0.0
    failures = 0

    for day in range(arguments.first_day, arguments.last_day + 1):
        iteration = BASE_ITERATION + STEPS_PER_DAY * day
        try:
            dyn = read_mds_record(run_dir, "dynState", iteration, DYNSTATE_CHANNELS)
            surf = read_mds_record(run_dir, "surfState", iteration, 1)
        except FileNotFoundError as error:
            rows.append({"day": day, "iteration": iteration, "status": "missing", "detail": str(error)})
            failures += 1
            continue

        rerun = np.concatenate([center_velocities(dyn), surf], axis=0)
        truth = np.asarray(state[arguments.regime_index, day]).astype(np.float32)

        identical = bool(np.array_equal(rerun, truth))
        difference = float(np.abs(rerun.astype(np.float64) - truth.astype(np.float64)).max())
        worst = max(worst, difference)
        if not identical:
            failures += 1

        rows.append(
            {
                "day": day,
                "iteration": iteration,
                "status": "bit-identical" if identical else "DIFFERS",
                "max_abs_difference": difference,
                "differing_values": int((rerun != truth).sum()),
            }
        )

    print(f"Gate G0  --  {run_dir}")
    print(f"{'day':>6} {'iteration':>11}  {'status':<14} {'max|diff|':>12} {'n differ':>9}")
    for row in rows:
        if row["status"] == "missing":
            print(f"{row['day']:>6} {row['iteration']:>11}  {'MISSING':<14}")
            continue
        print(
            f"{row['day']:>6} {row['iteration']:>11}  {row['status']:<14}"
            f" {row['max_abs_difference']:>12.6e} {row['differing_values']:>9d}"
        )

    verdict = failures == 0
    print()
    print(f"  days checked   {len(rows)}")
    print(f"  worst max|diff| {worst:.6e}")
    print(f"  GATE G0: {'PASS' if verdict else 'FAIL'}")

    if arguments.report:
        Path(arguments.report).write_text(
            json.dumps(
                {
                    "gate": "G0",
                    "run_dir": str(run_dir),
                    "dataset": arguments.dataset,
                    "days": [arguments.first_day, arguments.last_day],
                    "criterion": "rerun snapshots bit-identical to trajectories_v3",
                    "worst_max_abs_difference": worst,
                    "pass": verdict,
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"  wrote {arguments.report}")

    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
