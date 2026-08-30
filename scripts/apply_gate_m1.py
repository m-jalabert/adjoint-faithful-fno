"""Execution step 14 of docs/Adjoint_faithful_response_training_plan.md:
apply Gate M1 (plan section 22) to the three paired B/C runs.

The production selector has already run inside each training job and
published exactly one checkpoint per run, on nominal validation only. This
script does the part that must happen strictly *after* that selection:
score each selected checkpoint on held-out forward-response validation
(section 16.2's ``S_resp 10:60``), then apply section 16.3's two criterion
sets against paired B.

Section 16.3, forward-preservation -- C is forward-preserving only if,
relative to paired selected B:

  * each primary 10-90-day AUC ratio is at most 1.05;
  * the worst 90-360-day AUC/climatology ratio is at most 1.05 times B's;
  * perturbation growth is no more than 0.005 per call worse;
  * maximum normalized amplitude is at most 1.05 times B's;
  * all rollouts are finite.

Section 16.3, response effect -- C demonstrates the intended effect only if:

  * ``S_resp 10:60`` is at least 20% lower overall than paired B;
  * at least 10% lower within each input family;
  * no day-10 input-family/region aggregate is more than 1.10 times B's.

Gate M1's own text: "Failure labels the development result negative. It does
not authorize another lambda, seed, checkpoint, continuation, curriculum, or
data edit." This script therefore reports and freezes a verdict; it never
reselects, and it has no fallback branch.

Firewall: reads the ``validation`` role of the curated forward-response store
and the six published checkpoints. No MITgcm adjoint, TAF output, FNO adjoint
map, blind response case, or nested final-inference start is opened. Every
number here is produced after checkpoint selection and can therefore not have
influenced it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oceanfno import train as parent_train  # noqa: E402
from oceanfno.runtime import (  # noqa: E402
    _device, _file_sha256, _json_sha256, json_safe, require_runtime, torch,
)
from oceanfno.dataset import assert_store_is_v3, store_wind_normalization  # noqa: E402
from oceanfno.model import ProductionArchitecture, build_model, parameter_count, EXPECTED_PARAMETER_COUNT  # noqa: E402
from oceanfno.response_dataset import GROUPS  # noqa: E402
from oceanfno.response_validation import REGIONS, evaluate_response_validation  # noqa: E402

ARMS = {
    "B": "model_c_adjoint_faithful_nominal_control_v1",
    "C": "model_c_adjoint_faithful_response_v1",
}
SEEDS = (20260724, 20260911, 20260912)
PRIMARY_FIELDS = ("surface_speed", "sst", "phihyd_surface")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "af_fno" / "response" / "forward_response_v1" / "gate_m1"
PRODUCTION_CONTRACT_PATH = PROJECT_ROOT / "config" / "model_c_production_1in_1out_spectralnorm_v1.json"
DATASET_PATH = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/trajectories_v3.zarr")

AUC_MAX_RATIO = 1.05
LONG_MAX_RATIO = 1.05
AMPLITUDE_MAX_RATIO = 1.05
GROWTH_MAX_ADDITIVE = 0.005
S_RESP_OVERALL_MIN_REDUCTION = 0.20
S_RESP_FAMILY_MIN_REDUCTION = 0.10
DAY10_MAX_RATIO = 1.10


class GateM1Error(RuntimeError):
    """Raised when Gate M1 cannot be legitimately evaluated."""


def _report(arm: str, seed: int) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / "outputs" / "af_fno" / "C" / ARMS[arm] / f"seed_{seed}" / "report.json").read_text())


def _selected_summary(report: dict[str, Any]) -> dict[str, Any]:
    step = int(report["selection_decision"]["selected_optimizer_step"])
    return next(s for s in report["validation_summaries"] if s["optimizer_step"] == step)


def _load_selected_model(report: dict[str, Any], architecture: Any, device: Any) -> Any:
    published = report["published_checkpoint"]
    path = Path(published["checkpoint"])
    actual = _file_sha256(path)
    if actual != published["checkpoint_sha256"]:
        raise GateM1Error(f"{path} hash {actual} != published {published['checkpoint_sha256']}")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise GateM1Error("architecture parameter count mismatch")
    model.eval()
    return model


def _family_means(per_hgr: dict[str, float]) -> dict[str, float]:
    return {h: float(np.mean([v for k, v in per_hgr.items() if k.split("|")[0] == h])) for h in GROUPS}


def _day10_family_region(cell_means: dict[str, float]) -> dict[str, float]:
    """Section 16.3's "day-10 input-family/region aggregate": for each
    (input family, input-centre region), the mean over output groups of the
    lead-10 cell means."""

    out: dict[str, float] = {}
    for h in GROUPS:
        for region in REGIONS:
            values = [cell_means[f"{h}|{g}|{region}|10"] for g in GROUPS]
            out[f"{h}|{region}"] = float(np.mean(values))
    return out


def run(*, device_name: str = "auto") -> dict[str, Any]:
    started = time.monotonic()
    require_runtime()
    device = _device(device_name)

    group = zarr.open_consolidated(str(DATASET_PATH), mode="r")
    assert_store_is_v3(group)
    wet_array, _, _ = store_wind_normalization(group)
    wet_array = np.asarray(wet_array, dtype=bool)

    production_contract = json.loads(PRODUCTION_CONTRACT_PATH.read_text())
    architecture = ProductionArchitecture(**production_contract["architecture"])

    reports = {(arm, seed): _report(arm, seed) for arm in ARMS for seed in SEEDS}

    # Section 16.2: "All models are scored in the exact parent external
    # normalization." Every run recomputes it and must land on the same hash.
    norm_hashes = {r["published_checkpoint"]["normalization_sha256"] for r in reports.values()}
    if len(norm_hashes) != 1:
        raise GateM1Error(f"runs disagree on the external normalizer: {norm_hashes}")
    normalizer = np.load(reports[("B", SEEDS[0])]["published_checkpoint"]["normalization"])
    point_mean = normalizer["pointwise_mean"].astype(np.float32)
    point_scale = normalizer["pointwise_scale"].astype(np.float32)
    statics, _ = parent_train.physical_static_block(production_contract["sources"], group, point_mean, point_scale)

    response: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in ARMS:
        for seed in SEEDS:
            print(f"[gate-m1] {arm} seed {seed}: S_resp 10:60 ...", flush=True)
            model = _load_selected_model(reports[(arm, seed)], architecture, device)
            response[(arm, seed)] = evaluate_response_validation(
                model, device, point_mean, point_scale, wet_array, statics
            )
            print(f"[gate-m1]   S_resp = {response[(arm, seed)]['S_resp_10_60']:.4f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        b_sum, c_sum = _selected_summary(reports[("B", seed)]), _selected_summary(reports[("C", seed)])
        b_resp, c_resp = response[("B", seed)], response[("C", seed)]

        auc_ratios = {f: c_sum["short_auc_10_90"][f] / b_sum["short_auc_10_90"][f] for f in PRIMARY_FIELDS}
        long_ratio = max(c_sum["long_ratio_to_climatology"].values()) / max(b_sum["long_ratio_to_climatology"].values())
        amplitude_ratio = c_sum["maximum_normalized_amplitude"] / b_sum["maximum_normalized_amplitude"]
        growth_delta = (
            c_sum["perturbation_growth"]["worst_growth_rate_per_call"]
            - b_sum["perturbation_growth"]["worst_growth_rate_per_call"]
        )
        finite = not (
            c_sum["perturbation_growth"]["measurement_failed_on_a_start"]
            or b_sum["perturbation_growth"]["measurement_failed_on_a_start"]
        )
        forward = {
            "auc_ratios": auc_ratios,
            "auc_ok": all(v <= AUC_MAX_RATIO for v in auc_ratios.values()),
            "long_ratio_to_b": long_ratio,
            "long_ok": long_ratio <= LONG_MAX_RATIO,
            "amplitude_ratio_to_b": amplitude_ratio,
            "amplitude_ok": amplitude_ratio <= AMPLITUDE_MAX_RATIO,
            "growth_delta": growth_delta,
            "growth_ok": growth_delta <= GROWTH_MAX_ADDITIVE,
            "finite_ok": finite,
        }
        forward["forward_preserving"] = all(
            forward[k] for k in ("auc_ok", "long_ok", "amplitude_ok", "growth_ok", "finite_ok")
        )

        b_fam, c_fam = _family_means(b_resp["per_family_group_region"]), _family_means(c_resp["per_family_group_region"])
        family_reduction = {h: 1.0 - c_fam[h] / b_fam[h] for h in GROUPS}
        overall_reduction = 1.0 - c_resp["S_resp_10_60"] / b_resp["S_resp_10_60"]
        b_day10, c_day10 = _day10_family_region(b_resp["cell_means"]), _day10_family_region(c_resp["cell_means"])
        day10_ratios = {k: c_day10[k] / b_day10[k] for k in b_day10}
        effect = {
            "S_resp_b": b_resp["S_resp_10_60"],
            "S_resp_c": c_resp["S_resp_10_60"],
            "overall_reduction": overall_reduction,
            "overall_ok": overall_reduction >= S_RESP_OVERALL_MIN_REDUCTION,
            "family_means_b": b_fam,
            "family_means_c": c_fam,
            "family_reduction": family_reduction,
            "family_ok": all(v >= S_RESP_FAMILY_MIN_REDUCTION for v in family_reduction.values()),
            "families_below_threshold": [h for h, v in family_reduction.items() if v < S_RESP_FAMILY_MIN_REDUCTION],
            "worst_day10_family_region_ratio": max(day10_ratios.values()),
            "day10_ok": all(v <= DAY10_MAX_RATIO for v in day10_ratios.values()),
            "day10_cells_above_threshold": {k: v for k, v in day10_ratios.items() if v > DAY10_MAX_RATIO},
        }
        effect["demonstrates_response_effect"] = all(
            effect[k] for k in ("overall_ok", "family_ok", "day10_ok")
        )

        per_seed[str(seed)] = {
            "seed": seed,
            "selected_optimizer_step": {
                "B": int(reports[("B", seed)]["selection_decision"]["selected_optimizer_step"]),
                "C": int(reports[("C", seed)]["selection_decision"]["selected_optimizer_step"]),
            },
            "checkpoint_sha256": {
                arm: reports[(arm, seed)]["published_checkpoint"]["checkpoint_sha256"] for arm in ARMS
            },
            "report_sha256": {arm: reports[(arm, seed)]["content_sha256"] for arm in ARMS},
            "forward_preservation": forward,
            "response_effect": effect,
            "gate_m1_pass": forward["forward_preserving"] and effect["demonstrates_response_effect"],
        }

    passing = [s for s, v in per_seed.items() if v["gate_m1_pass"]]
    primary_pass = per_seed[str(SEEDS[0])]["gate_m1_pass"]
    result = {
        "gate": "M1",
        "plan_section": "22 (Gate M1), criteria from 16.3",
        "primary_seed": SEEDS[0],
        "primary_seed_pass": primary_pass,
        "seeds_passing": passing,
        "verdict": "positive" if primary_pass and len(passing) == len(SEEDS) else "negative",
        "verdict_note": (
            "Gate M1 is a stop/go gate: failure labels the development result negative and "
            "authorizes no new lambda, seed, checkpoint, continuation, curriculum or data edit. "
            "The preregistered blind forward-response and adjoint evaluations still run."
        ),
        "lambda_resp": json.loads(
            (PROJECT_ROOT / "config" / "model_c_adjoint_faithful_response_v1.json").read_text()
        )["response"]["lambda_resp"],
        "per_seed": per_seed,
        "response_validation_detail": {
            f"{arm}_{seed}": response[(arm, seed)] for arm in ARMS for seed in SEEDS
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    result = json_safe(result)
    result["content_sha256"] = _json_sha256(result)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / "gate_m1_result.json"
    if out_path.exists():
        raise GateM1Error(f"Gate M1 is already frozen: {out_path}")
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[gate-m1] wrote {out_path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    result = run(device_name=args.device)
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_seed", "response_validation_detail")}, indent=2))
    for seed, v in result["per_seed"].items():
        f, e = v["forward_preservation"], v["response_effect"]
        print(f"  seed {seed}: forward_preserving={f['forward_preserving']} "
              f"response_effect={e['demonstrates_response_effect']} -> pass={v['gate_m1_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
