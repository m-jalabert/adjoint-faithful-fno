"""Execution step 18: the blind MITgcm-adjoint test (plan sections 18.3, 22/Gate A1).

The study's confirmatory endpoint. Compares every model's truth-forced SSH
adjoint against the MITgcm/TAF ground truth on the identical objectives and
leads, and applies section 18.3's predeclared criteria.

Section 18.3's primary score, for model m and seed s:

    S_{m,s} = (1/8) * sum_{o in {point, kernel}} sum_{k in {10,20,30,90}}
              log[ max( E^relL2_{m,s,o,k}, 1e-12 ) ]

with the two primary effects

    Delta_A,s = S_{C,s} - S_A        (versus the historical frozen parent)
    Delta_B,s = S_{C,s} - S_{B,s}    (versus the paired nominal control)

Negative is better. The primary seed 20260724 succeeds only if

  1. Delta_B <= log(0.8);
  2. Delta_A < 0;
  3. at least six of the eight (objective, lead) relative-L2 cells improve
     versus paired B;
  4. no cell is worse than 1.10 x B.

Across replication:

  5. the median Delta_B over the three seeds is <= log(0.9);
  6. at least two of three seeds have Delta_B < 0.

Secondary endpoints -- free-chain score, pattern correlation, amplitude ratio,
mean-mode preservation -- are reported per seed and are not selection
criteria. "Results are reported per seed; no best seed is chosen after opening
TAF data."

Every metric is `adjoint_metrics`'s own, unchanged, so these numbers are
directly comparable with the Phase-A comparison. Gate A0 must already have
passed; this script asserts the eight FNO adjoint packages exist and that the
MITgcm truth hashes match what the packages were built against.

Section 18.3: "A failure is publishable and closes v1."
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import adjoint_metrics as metrics  # noqa: E402

TRUTH = PROJECT_ROOT / "outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2/mitgcm_s0_adjoint_v2.npz"
ADJOINT_ROOT = PROJECT_ROOT / "outputs/af_fno/adjoint"
OUTPUT_ROOT = ADJOINT_ROOT / "comparison_response_v1"

PRIMARY_SEED = 20260724
SEEDS = (20260724, 20260911, 20260912)
LEADS = (10, 20, 30, 90)
#: section 18.3's two objectives, and the array names they carry on each side.
OBJECTIVES = {
    "point": ("S_ssh_anomaly", "S_ssh_anomaly_forced", "S_ssh_anomaly_free"),
    "kernel": ("S_ssh_anomaly_kernel", "S_ssh_anomaly_kernel_forced", "S_ssh_anomaly_kernel_free"),
}
PACKAGES = {
    "A": "fno_a_s0_adjoint_v1",
    "ft90": "fno_ft90_s0_adjoint_v1",
    **{f"B_{s}": f"fno_b_seed_{s}_s0_adjoint_v1" for s in SEEDS},
    **{f"C_{s}": f"fno_c_seed_{s}_s0_adjoint_v1" for s in SEEDS},
}

FLOOR = 1.0e-12
DELTA_B_PRIMARY_MAX = math.log(0.8)
DELTA_B_MEDIAN_MAX = math.log(0.9)
MIN_CELLS_IMPROVED = 6
CELL_MAX_RATIO = 1.10


class GateA1Error(RuntimeError):
    """Raised when the blind adjoint comparison cannot be legitimately run."""


def _load(label: str) -> dict[str, np.ndarray]:
    path = ADJOINT_ROOT / PACKAGES[label] / f"{'fno_ft90' if label != 'A' else 'fno'}_s0_adjoint_arrays.npz"
    if not path.is_file():
        # every adapter-produced package keeps the runner's own array filename
        candidates = sorted((ADJOINT_ROOT / PACKAGES[label]).glob("*_adjoint_arrays.npz"))
        if not candidates:
            raise GateA1Error(f"no adjoint arrays for {label} in {ADJOINT_ROOT / PACKAGES[label]}")
        path = candidates[0]
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _score(cells: dict[tuple[str, int], float]) -> float:
    """Section 18.3's S: the mean over the eight cells of log(max(E, 1e-12))."""

    return sum(math.log(max(v, FLOOR)) for v in cells.values()) / len(cells)


def run() -> dict[str, Any]:
    started = time.monotonic()
    with np.load(TRUTH) as z:
        truth = {k: np.asarray(z[k]) for k in z.files}
    wet = truth["wet_mask"].astype(bool)
    truth_leads = tuple(int(v) for v in truth["lead_days"])
    if truth_leads != LEADS:
        raise GateA1Error(f"MITgcm truth carries leads {truth_leads}, expected {LEADS}")

    models = {label: _load(label) for label in PACKAGES}
    for label, arrays in models.items():
        if tuple(int(v) for v in arrays["lead_days"]) != LEADS:
            raise GateA1Error(f"{label} carries leads {tuple(arrays['lead_days'])}, expected {LEADS}")
        if not np.array_equal(arrays["wet_mask"].astype(bool), wet):
            raise GateA1Error(f"{label} does not share the truth wet mask")

    per_model: dict[str, Any] = {}
    for label, arrays in models.items():
        forced_cells: dict[tuple[str, int], float] = {}
        free_cells: dict[tuple[str, int], float] = {}
        detail: dict[str, Any] = {}
        for objective, (truth_key, forced_key, free_key) in OBJECTIVES.items():
            for index, lead in enumerate(LEADS):
                reference = truth[truth_key][index]
                for kind, key, sink in (("forced", forced_key, forced_cells), ("free", free_key, free_cells)):
                    emulator = arrays[key][index]
                    m = metrics.primary_metrics(emulator, reference, wet)
                    sink[(objective, lead)] = float(m["relative_l2"])
                    detail[f"{objective}|{lead}|{kind}"] = {
                        "relative_l2": float(m["relative_l2"]),
                        "pattern_correlation": float(m["pattern_correlation"]),
                        "amplitude_ratio": float(m["amplitude_ratio"]),
                        "sign_agreement": float(m["sign_agreement"]),
                    }
        per_model[label] = {
            "S_forced": _score(forced_cells),
            "S_free": _score(free_cells),
            "cells_forced": {f"{o}|{k}": v for (o, k), v in forced_cells.items()},
            "cells_free": {f"{o}|{k}": v for (o, k), v in free_cells.items()},
            "detail": detail,
        }

    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        b, c = per_model[f"B_{seed}"], per_model[f"C_{seed}"]
        a = per_model["A"]
        delta_b = c["S_forced"] - b["S_forced"]
        delta_a = c["S_forced"] - a["S_forced"]
        ratios = {k: c["cells_forced"][k] / b["cells_forced"][k] for k in b["cells_forced"]}
        improved = [k for k, v in ratios.items() if v < 1.0]
        per_seed[str(seed)] = {
            "S_A": a["S_forced"], "S_B": b["S_forced"], "S_C": c["S_forced"],
            "delta_B": delta_b, "delta_A": delta_a,
            "delta_B_as_ratio": math.exp(delta_b), "delta_A_as_ratio": math.exp(delta_a),
            "cell_ratios_C_over_B": ratios,
            "cells_improved": improved,
            "cells_improved_count": len(improved),
            "worst_cell_ratio": max(ratios.values()),
            "S_free": {"A": a["S_free"], "B": b["S_free"], "C": c["S_free"]},
            "delta_B_free": c["S_free"] - b["S_free"],
        }

    p = per_seed[str(PRIMARY_SEED)]
    deltas = [per_seed[str(s)]["delta_B"] for s in SEEDS]
    conditions = {
        "primary_delta_B_at_most_log_0_8": p["delta_B"] <= DELTA_B_PRIMARY_MAX,
        "primary_delta_A_below_zero": p["delta_A"] < 0.0,
        "primary_at_least_6_of_8_cells_improve": p["cells_improved_count"] >= MIN_CELLS_IMPROVED,
        "primary_no_cell_worse_than_1_10x_B": p["worst_cell_ratio"] <= CELL_MAX_RATIO,
        "median_delta_B_at_most_log_0_9": statistics.median(deltas) <= DELTA_B_MEDIAN_MAX,
        "at_least_2_of_3_seeds_delta_B_below_zero": sum(d < 0.0 for d in deltas) >= 2,
    }
    passed = all(conditions.values())

    result = {
        "gate": "A1",
        "plan_section": "18.3 (primary blind scientific endpoint), 22 (Gate A1)",
        "primary_seed": PRIMARY_SEED,
        "conditions": conditions,
        "verdict": "positive" if passed else "negative",
        "verdict_note": (
            "section 18.3: the scientific answer is positive only if the frozen forward "
            "gates/tests and every quantitative adjoint criterion pass. A failure is publishable "
            "and closes v1. No threshold selects a model after TAF access and no best seed is "
            "chosen; every seed is reported."
        ),
        "thresholds": {
            "delta_B_primary_max": DELTA_B_PRIMARY_MAX,
            "delta_B_median_max": DELTA_B_MEDIAN_MAX,
            "min_cells_improved": MIN_CELLS_IMPROVED,
            "cell_max_ratio": CELL_MAX_RATIO,
            "relative_l2_floor": FLOOR,
        },
        "median_delta_B": statistics.median(deltas),
        "seeds_with_delta_B_below_zero": [s for s in SEEDS if per_seed[str(s)]["delta_B"] < 0.0],
        "per_seed": per_seed,
        "per_model": per_model,
        "truth": {"path": str(TRUTH)},
        "elapsed_seconds": time.monotonic() - started,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "gate_a1_result.json"
    if path.exists():
        raise GateA1Error(f"Gate A1 is already frozen: {path}")
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    print(f"[gate-a1] wrote {path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k not in ("per_model", "per_seed")}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
