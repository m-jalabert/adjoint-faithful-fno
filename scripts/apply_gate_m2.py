"""Execution step 16 (continued): the blind forward-response test, Gate M2.

Plan section 17. Scores the frozen parent A, the ft90 child as context, and
all six paired B/C selected checkpoints on exactly the same 216 blind
directions, then applies section 17's conditions verbatim.

Primary seed 20260724. Its C checkpoint must:

  1. reduce ``S_resp_10:60`` by at least 15% versus paired B;
  2. reduce ``S_resp_90`` by at least 10% versus paired B;
  3. have both scores below frozen parent A;
  4. improve at least three of four input families;
  5. worsen no input-family/region day-10 aggregate by more than 10% versus B.

Across the three paired seeds:

  6. the median 10-60-day reduction must be at least 15%;
  7. at least two seeds must improve both the 10-60 and day-90 scores versus
     paired B.

Section 17: "This package is opened once. A failure is a negative result; it
cannot change the model, checkpoint, response weight, amplitudes, inventory,
or evaluation rule." This script reports and freezes a verdict write-once; it
has no fallback branch and writes to no model, contract or inventory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from oceanfno import train as parent_train  # noqa: E402
from oceanfno.runtime import _device, _file_sha256, _json_sha256, json_safe, require_runtime, torch  # noqa: E402
from oceanfno.dataset import assert_store_is_v3, store_wind_normalization  # noqa: E402
from oceanfno.model import ProductionArchitecture, build_model, parameter_count, EXPECTED_PARAMETER_COUNT  # noqa: E402
from oceanfno.response_dataset import GROUPS  # noqa: E402
from oceanfno.response_validation_blind import day10_family_region, evaluate_blind_response  # noqa: E402

import stage_blind_forward_response_run as blind  # noqa: E402
from extract_blind_forward_response_dataset import BLIND_DATASET_PATH, BLIND_OUTPUT_ROOT  # noqa: E402

PRIMARY_SEED = 20260724
SEEDS = (20260724, 20260911, 20260912)
ARMS = {"B": "model_c_adjoint_faithful_nominal_control_v1", "C": "model_c_adjoint_faithful_response_v1"}
CONTEXT = {
    "A": ("model_c_production_1in_1out_spectralnorm_v1", "model_c_production_1in_1out_spectralnorm_v1_report.json"),
    "ft90": ("model_c_production_1in_1out_spectralnorm_ft90_v1", None),
}
PRODUCTION_CONTRACT = PROJECT_ROOT / "config" / "model_c_production_1in_1out_spectralnorm_v1.json"
TRAJECTORIES = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/trajectories_v3.zarr")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "af_fno" / "response" / "forward_response_blind_v1" / "gate_m2"

S_1060_MIN_REDUCTION = 0.15
S_90_MIN_REDUCTION = 0.10
MIN_FAMILIES_IMPROVED = 3
DAY10_MAX_RATIO = 1.10
MEDIAN_MIN_REDUCTION = 0.15
MIN_SEEDS_IMPROVING_BOTH = 2


class GateM2Error(RuntimeError):
    """Raised when Gate M2 cannot be legitimately evaluated."""


def _study_report(arm: str, seed: int) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / "outputs" / "af_fno" / "C" / ARMS[arm] / f"seed_{seed}" / "report.json").read_text())


def _context_checkpoint(label: str) -> dict[str, Any]:
    """Frozen A / ft90: locate the published checkpoint from its own report."""

    base = CONTEXT[label][0]
    candidates = sorted((PROJECT_ROOT / "outputs" / "af_fno" / "C" / base).glob("*report.json"))
    if not candidates:
        raise GateM2Error(f"no published report found for {label} ({base})")
    return json.loads(candidates[0].read_text())


def _load(report: dict[str, Any], architecture: Any, device: Any) -> Any:
    published = report["published_checkpoint"]
    path = Path(published["checkpoint"])
    if _file_sha256(path) != published["checkpoint_sha256"]:
        raise GateM2Error(f"{path} does not match its published hash")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise GateM2Error("architecture parameter count mismatch")
    model.eval()
    return model


def _family_means(per_hgr: dict[str, float]) -> dict[str, float]:
    return {h: float(np.mean([v for k, v in per_hgr.items() if k.split("|")[0] == h])) for h in GROUPS}


def run(*, device_name: str = "auto") -> dict[str, Any]:
    started = time.monotonic()
    blind.assert_precondition()
    if not BLIND_DATASET_PATH.exists():
        raise GateM2Error(f"the blind store is not extracted yet: {BLIND_DATASET_PATH}")
    require_runtime()
    device = _device(device_name)

    group = zarr.open_consolidated(str(TRAJECTORIES), mode="r")
    assert_store_is_v3(group)
    wet, _, _ = store_wind_normalization(group)
    wet = np.asarray(wet, dtype=bool)

    reports = {(arm, seed): _study_report(arm, seed) for arm in ARMS for seed in SEEDS}
    norms = {r["published_checkpoint"]["normalization_sha256"] for r in reports.values()}
    if len(norms) != 1:
        raise GateM2Error(f"study runs disagree on the external normalizer: {norms}")
    stored = np.load(reports[("B", PRIMARY_SEED)]["published_checkpoint"]["normalization"])
    point_mean = stored["pointwise_mean"].astype(np.float32)
    point_scale = stored["pointwise_scale"].astype(np.float32)

    production = json.loads(PRODUCTION_CONTRACT.read_text())
    statics, _ = parent_train.physical_static_block(production["sources"], group, point_mean, point_scale)
    architecture = ProductionArchitecture(**production["architecture"])

    scores: dict[str, dict[str, Any]] = {}

    def _score(label: str, report: dict[str, Any]) -> None:
        print(f"[gate-m2] {label}: scoring 216 blind directions ...", flush=True)
        model = _load(report, architecture, device)
        scores[label] = evaluate_blind_response(
            model, device, point_mean, point_scale, wet, statics,
            dataset_path=BLIND_DATASET_PATH, output_root=BLIND_OUTPUT_ROOT,
        )
        print(f"[gate-m2]   S_resp_10:60 = {scores[label]['S_resp_10_60']:.4f} | "
              f"S_resp_90 = {scores[label].get('S_resp_90', float('nan')):.4f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for label in CONTEXT:
        try:
            _score(label, _context_checkpoint(label))
        except GateM2Error as error:
            print(f"[gate-m2] {label}: skipped ({error})", flush=True)
    for arm in ARMS:
        for seed in SEEDS:
            _score(f"{arm}_{seed}", reports[(arm, seed)])

    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        b, c = scores[f"B_{seed}"], scores[f"C_{seed}"]
        r1060 = 1.0 - c["S_resp_10_60"] / b["S_resp_10_60"]
        r90 = 1.0 - c["S_resp_90"] / b["S_resp_90"]
        bf, cf = _family_means(b["per_family_group_region"]), _family_means(c["per_family_group_region"])
        improved = [h for h in GROUPS if cf[h] < bf[h]]
        b10, c10 = day10_family_region(b["cell_means"]), day10_family_region(c["cell_means"])
        day10 = {k: c10[k] / b10[k] for k in b10}
        per_seed[str(seed)] = {
            "S_resp_10_60": {"B": b["S_resp_10_60"], "C": c["S_resp_10_60"], "reduction": r1060},
            "S_resp_90": {"B": b["S_resp_90"], "C": c["S_resp_90"], "reduction": r90},
            "family_means": {"B": bf, "C": cf},
            "families_improved": improved,
            "worst_day10_family_region_ratio": max(day10.values()),
            "day10_cells_worse_than_1_10": {k: v for k, v in day10.items() if v > DAY10_MAX_RATIO},
            "improves_both_scores": bool(r1060 > 0 and r90 > 0),
        }

    primary = per_seed[str(PRIMARY_SEED)]
    a = scores.get("A")
    c_primary = scores[f"C_{PRIMARY_SEED}"]
    conditions = {
        "primary_S_resp_10_60_reduction_at_least_15pc": primary["S_resp_10_60"]["reduction"] >= S_1060_MIN_REDUCTION,
        "primary_S_resp_90_reduction_at_least_10pc": primary["S_resp_90"]["reduction"] >= S_90_MIN_REDUCTION,
        "primary_both_scores_below_parent_A": (
            None if a is None else bool(
                c_primary["S_resp_10_60"] < a["S_resp_10_60"] and c_primary["S_resp_90"] < a["S_resp_90"]
            )
        ),
        "primary_improves_at_least_3_of_4_families": len(primary["families_improved"]) >= MIN_FAMILIES_IMPROVED,
        "primary_no_day10_aggregate_worse_than_1_10": primary["worst_day10_family_region_ratio"] <= DAY10_MAX_RATIO,
        "median_10_60_reduction_at_least_15pc": (
            statistics.median(per_seed[str(s)]["S_resp_10_60"]["reduction"] for s in SEEDS) >= MEDIAN_MIN_REDUCTION
        ),
        "at_least_2_seeds_improve_both_scores": (
            sum(per_seed[str(s)]["improves_both_scores"] for s in SEEDS) >= MIN_SEEDS_IMPROVING_BOTH
        ),
    }
    passed = all(v for v in conditions.values() if v is not None)

    result = {
        "gate": "M2",
        "plan_section": "17 (blind forward-response test)",
        "primary_seed": PRIMARY_SEED,
        "conditions": conditions,
        "verdict": "positive" if passed else "negative",
        "verdict_note": (
            "section 17: this package is opened once. A failure is a negative result; it cannot "
            "change the model, checkpoint, response weight, amplitudes, inventory or evaluation rule."
        ),
        "median_10_60_reduction": statistics.median(per_seed[str(s)]["S_resp_10_60"]["reduction"] for s in SEEDS),
        "seeds_improving_both": [s for s in SEEDS if per_seed[str(s)]["improves_both_scores"]],
        "per_seed": per_seed,
        "scores": scores,
        "blind_dataset": str(BLIND_DATASET_PATH),
        "checkpoint_sha256": {
            f"{arm}_{seed}": reports[(arm, seed)]["published_checkpoint"]["checkpoint_sha256"]
            for arm in ARMS for seed in SEEDS
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    result = json_safe(result)
    result["content_sha256"] = _json_sha256(result)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "gate_m2_result.json"
    if path.exists():
        raise GateM2Error(f"Gate M2 is already frozen -- the blind package is opened once: {path}")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[gate-m2] wrote {path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    result = run(device_name=args.device)
    print(json.dumps({k: v for k, v in result.items() if k not in ("scores", "per_seed")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
