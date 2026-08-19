"""Stage 9: extract the MITgcm adjoint ground truth and run gates G2-G5.

Implements sections 9-11 of docs/mitgcm_adjoint_ground_truth_plan.md.

Reads the MDS products of Run A (10-day map) and Run B (20-day map plus the
lead sweep) and writes one .npz plus a report.  Loads no FNO weights and reads
nothing under outputs/af_fno/C/**; this is the reference half of the comparison
only.

Conventions, stated here because the FNO side must match all four:

  Sign   S > 0 means raising eta at (i,j) raises the target anomaly.
  Units  dimensionless -- 0.2 means 1 cm at the source gives 2 mm at the target.
  Grid   cell centres, same (j,i) index order as the zarr's spatial axes.
  Land   exactly 0, not NaN.

Gates run here:

  G2  Run B's ADJetan at day 7210 equals Run A's adxx_etan.  The adjoint state
      at time t inside a run whose cost sits at T is dJ/deta(.,t) regardless of
      when the run started, so this is a free end-to-end consistency check.
  G3  With w = -rA*maskC/A_wet alone, the returned map must equal w itself at
      every dump time, because the area integral of eta is exactly conserved by
      this configuration and the adjoint of a conserved functional is constant
      in time.  Needs a separate run staged with costWeight_mean_only.bin.
  G4  S is exactly 0 on all land cells and finite everywhere.
  G5  fc from the run log equals sum(w * eta_7220) computed independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from select_adjoint_target import BASE_ITERATION, CONTRACT_VERSION, STEPS_PER_DAY

#: cost_final.F prints this once per run to the model's STDOUT.0000.
GLOBAL_FC = re.compile(r"global fc\s*=\s*(\S+)")

GATE_G2_TOLERANCE = 1.0e-6
GATE_G3_TOLERANCE = 1.0e-5
GATE_G5_TOLERANCE = 1.0e-10


class ExtractionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mds(prefix: Path, nx: int = 62, ny: int = 62) -> np.ndarray:
    """Read one 2-D MDS record, taking the precision from the .meta.

    The plan says to check the .meta rather than assume, and it matters:
    adxx_etan comes back at ctrlprec (float64) while ADJetan follows
    writeBinaryPrec, which is 32 unless the run set it otherwise.
    """

    # Not with_suffix(): the iteration stamp in "adxx_etan.0000000000" looks
    # like a suffix to pathlib and would be replaced rather than extended.
    data = prefix.parent / f"{prefix.name}.data"
    meta = prefix.parent / f"{prefix.name}.meta"
    if not data.is_file():
        raise ExtractionError(f"missing {data}")
    dtype = ">f8"
    if meta.is_file():
        dtype = ">f4" if "float32" in meta.read_text() else ">f8"
    values = np.fromfile(data, dtype=dtype)
    if values.size % (nx * ny):
        raise ExtractionError(
            f"{data} holds {values.size} values, not a multiple of {nx * ny}"
            f" (read as {dtype}; check the .meta)"
        )
    # Vertically resolved fields (hFacC) come back with Nr records; the surface
    # level is the first one and is all this study needs.
    return values.reshape(-1, ny, nx)[0].astype(np.float64)


def global_fc(run_dir: Path) -> float | None:
    stdout = run_dir / "STDOUT.0000"
    if not stdout.is_file():
        return None
    matches = GLOBAL_FC.findall(stdout.read_text(errors="replace"))
    if not matches:
        return None
    return float(matches[-1].replace("D", "E"))


def adjetan_series(run_dir: Path, start_day: int, end_day: int) -> tuple[np.ndarray, np.ndarray]:
    """ADJetan at each dump time, ordered by day ascending.

    Day d carries dJ/deta(.,d), so lead = end_day - d.
    """

    maps, days = [], []
    for day in range(start_day, end_day + 1):
        iteration = BASE_ITERATION + STEPS_PER_DAY * day
        prefix = run_dir / f"ADJetan.{iteration:010d}"
        if not (run_dir / f"{prefix.name}.data").is_file():
            continue
        maps.append(read_mds(prefix))
        days.append(day)
    if not maps:
        raise ExtractionError(f"no ADJetan dumps found in {run_dir}")
    return np.stack(maps), np.asarray(days, dtype=np.int32)


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(b))
    if denominator == 0.0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denominator)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/home/mjalabert314/bire_james25_repro")
    parser.add_argument(
        "--scratch-root",
        default="/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v1",
    )
    parser.add_argument(
        "--dataset",
        default="/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/trajectories_v3.zarr",
    )
    parser.add_argument("--regime-index", type=int, default=0)
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()

    project_root = Path(arguments.project_root)
    scratch_root = Path(arguments.scratch_root)
    output_dir = Path(
        arguments.output
        or project_root / "outputs" / "af_fno" / "adjoint" / "mitgcm_s0_adjoint_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = json.loads((project_root / "config" / f"{CONTRACT_VERSION}.json").read_text())
    target_ij = np.asarray([contract["i_global"], contract["j_global"]], dtype=np.int32)

    run_a = scratch_root / "runA"
    run_b = scratch_root / "runB"

    s10 = read_mds(run_a / "adxx_etan.0000000000")
    s20 = read_mds(run_b / "adxx_etan.0000000000")
    s_lead, lead_source_days = adjetan_series(run_b, 7200, 7220)
    lead_days = (7220 - lead_source_days).astype(np.int32)

    # rA and the surface mask come from the same run that produced the maps, so
    # the geometry is the run's own rather than an assumption about it.
    rac = read_mds(run_a / "RAC")
    wet_mask = (read_mds(run_a / "hFacC") > 0.0).astype(np.int8) if (
        run_a / "hFacC.data"
    ).is_file() else (s10 != 0.0).astype(np.int8)

    gates: dict[str, dict] = {}

    # --- G2: Run B's day-7210 slice must equal Run A's map --------------
    day_7210 = np.where(lead_source_days == 7210)[0]
    if day_7210.size:
        residual = relative_l2(s_lead[int(day_7210[0])], s10)
        gates["G2"] = {
            "description": "Run B ADJetan at day 7210 vs Run A adxx_etan, relative L2",
            "residual": residual,
            "tolerance": GATE_G2_TOLERANCE,
            "pass": bool(residual < GATE_G2_TOLERANCE),
        }
    else:
        gates["G2"] = {"description": "no day-7210 ADJetan dump in Run B", "pass": False}

    # --- G4: land exactly zero, everything finite -----------------------
    land = wet_mask == 0
    g4_pass = bool(
        np.all(s10[land] == 0.0)
        and np.all(s20[land] == 0.0)
        and np.isfinite(s10).all()
        and np.isfinite(s20).all()
        and np.isfinite(s_lead).all()
    )
    gates["G4"] = {
        "description": "S exactly 0 on every land cell, finite everywhere",
        "land_cells": int(land.sum()),
        "nonzero_on_land": int((s10[land] != 0.0).sum() + (s20[land] != 0.0).sum()),
        "pass": g4_pass,
    }

    # --- G5: fc equals sum(w * eta) computed independently ---------------
    fc_a, fc_b = global_fc(run_a), global_fc(run_b)
    weight_path = run_a / "costWeight.bin"
    g5: dict = {
        "description": "fc from the run vs sum(w * eta_7220) in numpy",
        "fc_runA": fc_a,
        "fc_runB": fc_b,
    }
    if weight_path.is_file() and fc_a is not None:
        weight = np.fromfile(weight_path, dtype=">f4").reshape(62, 62).astype(np.float64)
        try:
            import zarr

            state = zarr.open(arguments.dataset, mode="r")["state"]
            eta = np.asarray(state[arguments.regime_index, 7220, 45]).astype(np.float64)
            predicted = float((weight * eta).sum())
            error = abs(predicted - fc_a) / abs(fc_a) if fc_a else abs(predicted)
            # The plan's flat 1e-10 assumed both sides were exact.  They are
            # not: the archived eta is the float32 diagnostic snapshot, while
            # fc is accumulated from the model's float64 state.  The reachable
            # floor is therefore the float32 half-ulp bound on this particular
            # weighted sum, which is computed here rather than assumed -- the
            # same discipline the s0-twin-float32-floor finding forced on the
            # daily diagnostics.  A residual well under that bound is a pass;
            # one above it means the cost is genuinely not what we think.
            floor = float((np.abs(weight) * np.abs(eta)).sum() * 2.0**-24 / abs(fc_a))
            g5.update(
                {
                    "predicted": predicted,
                    "relative_error": error,
                    "float32_eta_floor": floor,
                    "error_over_floor": error / floor if floor else None,
                    "tolerance": max(GATE_G5_TOLERANCE, floor),
                    "pass": bool(error < max(GATE_G5_TOLERANCE, floor)),
                }
            )
        except Exception as error:  # zarr missing, dataset moved, etc.
            g5.update({"pass": None, "detail": f"not evaluated: {error}"})
    else:
        g5.update({"pass": None, "detail": "costWeight.bin or fc unavailable"})
    gates["G5"] = g5

    # Run A and Run B evaluate the same J at the same day, so their fc must
    # agree exactly.  Cheap, and it catches a mis-staged weight file.
    if fc_a is not None and fc_b is not None:
        gates["fc_consistency"] = {
            "description": "Run A and Run B evaluate the same J at day 7220",
            "difference": abs(fc_a - fc_b),
            "pass": bool(abs(fc_a - fc_b) <= 1e-12 * max(1.0, abs(fc_a))),
        }

    # --- G3: the mean-only run, if it was staged ------------------------
    run_g3 = scratch_root / "runG3"
    if (run_g3 / "adxx_etan.0000000000.data").is_file():
        weight = np.fromfile(run_g3 / "costWeight.bin", dtype=">f4").reshape(62, 62)
        series, _ = adjetan_series(run_g3, 7210, 7220)
        residuals = [relative_l2(m, weight.astype(np.float64)) for m in series]
        worst = max(residuals) if residuals else float("nan")
        gates["G3"] = {
            "description": "mean-only weight: map must equal w at every dump time",
            "worst_relative_l2": worst,
            "per_dump": residuals,
            "tolerance": GATE_G3_TOLERANCE,
            "pass": bool(worst < GATE_G3_TOLERANCE),
        }
    else:
        gates["G3"] = {
            "description": "not run; stage runG3 with work/costWeight_mean_only.bin",
            "pass": None,
        }

    npz_path = output_dir / "mitgcm_s0_adjoint_v1.npz"
    np.savez_compressed(
        npz_path,
        S10=s10,
        S20=s20,
        S_lead=s_lead,
        lead_days=lead_days,
        lead_source_days=lead_source_days,
        wet_mask=wet_mask,
        rA=rac,
        target_ij=target_ij,
    )

    executable = run_a / "mitgcmuv_ad"
    report = {
        "version": "mitgcm_s0_adjoint_v1",
        "conventions": {
            "sign": "S > 0: raising eta at (i,j) raises the target anomaly",
            "units": "dimensionless (metres of J per metre of eta)",
            "grid": "cell centres, (j,i) order matching the zarr spatial axes",
            "land": "exactly 0, not NaN",
        },
        "mitgcm_commit": "f03a2f5e214bc57b8393f6201a6a1266dd1f53d6",
        "taf_version": "6.8.11",
        "executable_sha256": _sha256(executable.resolve()) if executable.exists() else None,
        "target_contract": contract,
        "runs": {
            "runA": json.loads((run_a / "run_manifest.json").read_text())
            if (run_a / "run_manifest.json").is_file()
            else None,
            "runB": json.loads((run_b / "run_manifest.json").read_text())
            if (run_b / "run_manifest.json").is_file()
            else None,
        },
        "gates": gates,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"wrote {npz_path}")
    print(f"  S10        {s10.shape}  min {s10.min():.4e}  max {s10.max():.4e}")
    print(f"  S20        {s20.shape}  min {s20.min():.4e}  max {s20.max():.4e}")
    print(f"  S_lead     {s_lead.shape}  leads {lead_days.min()}..{lead_days.max()} days")
    print(f"  fc         runA {fc_a}  runB {fc_b}")
    print()
    failed = 0
    for name, gate in sorted(gates.items()):
        verdict = gate.get("pass")
        mark = "PASS" if verdict else ("SKIP" if verdict is None else "FAIL")
        if verdict is False:
            failed += 1
        detail = gate.get("residual", gate.get("worst_relative_l2", gate.get("relative_error")))
        suffix = f"  {detail:.3e}" if isinstance(detail, float) else ""
        print(f"  {name:<15} {mark}{suffix}   {gate['description']}")
    print(f"\n  wrote {output_dir / 'report.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
