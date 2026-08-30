"""Tier-0 response diagnostics: the measurements step 12 never took.

Step 12 reported `S_resp_10:60` for four lambda candidates and for no
control, and a *mean* response training loss with no reference scale. Both
numbers are uninterpretable on their own. This script supplies the two
missing baselines, by inference only -- it trains nothing, writes no
decision artifact, and never touches
`lambda_selection_result.json`.

Three products, all for arm B (`model_c_adjoint_faithful_nominal_control_v1`,
primary seed 20260724), the frozen lambda-zero control:

1. **`S_resp_10:60` for the control** at every saved step (1,920, 3,840,
   5,760, 7,680), computed with the unchanged
   `response_validation.evaluate_response_validation` the screen itself
   called. Step 1,920 is the number that makes the candidates'
   12.13/13.96/15.01/14.82 mean something: it is the matched lambda-zero
   value at the matched step.

2. **The reference scale of the training loss `l_{q,k}`.** Section 14.2's
   `d_{h,g,k}` is, by construction in `freeze_response_scales.compute_scales`,
   the RMS of the oriented truth response `r_M` over the *same* train
   directions the auxiliary stream draws from, in the same oriented/sigma
   convention. So a model predicting **zero response** scores
   `l = (1/8) * sum_{s,g} mean(r_M^2)/d^2 ~= 1.0` in the training mean --
   8 terms each of expectation one. This script measures that zero-response
   value directly on the exact 480-direction schedule prefix the screen
   used, and alongside it the control's own `l` at the same directions.
   `mean_response_loss = 10.73` at lambda=0.03 is then readable: above 1.0
   means the model's response is worse than having no response at all.

3. The per-direction breakdown behind both, so the aggregate can be
   attributed to input family / output group / lead rather than taken on
   faith.

Firewall: reads only the `train` and `validation` roles of the curated
forward-response store and arm B's own published checkpoints/normalizer.
No MITgcm adjoint, TAF output, FNO adjoint map, blind response case, or
nested final-inference start is opened. This is a diagnostic, not a
selection input: nothing here may be used to choose lambda, a checkpoint,
an amplitude, or an architecture.
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
    _device, _file_sha256, _json_sha256, json_safe, require_runtime, seed_everything, torch,
)
from oceanfno.dataset import assert_store_is_v3, store_wind_normalization  # noqa: E402
from oceanfno.model import ProductionArchitecture, build_model, parameter_count, EXPECTED_PARAMETER_COUNT  # noqa: E402
from oceanfno.response_dataset import (  # noqa: E402
    GROUPS, REGIMES, ResponseStore, build_auxiliary_schedule, load_direction_table,
)
from oceanfno.response_objective import load_response_scales, oriented_response, response_term  # noqa: E402
from oceanfno.response_validation import evaluate_response_validation  # noqa: E402

B_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "af_fno" / "C" / "model_c_adjoint_faithful_nominal_control_v1"
    / "seed_20260724" / "report.json"
)
SCREEN_CONTRACT_PATH = PROJECT_ROOT / "config" / "forward_response_lambda_screen_v1.json"
PRODUCTION_CONTRACT_PATH = PROJECT_ROOT / "config" / "model_c_production_1in_1out_spectralnorm_v1.json"
DATASET_PATH = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/trajectories_v3.zarr")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "af_fno" / "response" / "forward_response_v1" / "tier0_diagnostics"


class Tier0Error(RuntimeError):
    """Raised when a Tier-0 diagnostic cannot be legitimately computed."""


def _load_control_model(checkpoint_path: Path, expected_sha: str, architecture: Any, device: Any) -> Any:
    """Arm B checkpoints are written from ``materialized_state_dict``, so they
    load strictly into a plain ``ProductionFNO`` with no reparameterization
    attached -- the inference operator is already the normalized one."""

    actual = _file_sha256(checkpoint_path)
    if actual != expected_sha:
        raise Tier0Error(f"{checkpoint_path.name} hash {actual} != published {expected_sha}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise Tier0Error("architecture parameter count mismatch")
    model.eval()
    return model


def _training_loss_on_schedule(
    model: Any | None,
    schedule: list[Any],
    store: ResponseStore,
    scales: Any,
    point_mean: np.ndarray,
    point_scale: np.ndarray,
    sigma_t: Any,
    statics: np.ndarray,
    wet_bool: np.ndarray,
    wet_bool_t: Any,
    wet_float: Any,
    device: Any,
) -> dict[str, Any]:
    """Section 14.2's ``l_q`` on the exact auxiliary-schedule prefix the screen
    consumed, under ``no_grad``.

    ``model=None`` evaluates the **zero-response predictor** (``r_F = 0``),
    whose value is the reference scale: ~1.0 by construction of ``d``.
    Mirrors ``train_response.auxiliary_update``'s forward exactly -- same
    batched (nominal, minus, plus) triplet, same oriented-response
    conventions, same short/long reduction -- minus the backward pass and
    the spectral snapshot context (nothing here mutates a buffer: the
    materialized checkpoint carries no power vectors).
    """

    per_direction: list[dict[str, Any]] = []
    for direction in schedule:
        leads = (10,) if direction.array_group == "short" else (10, 20, 30, 40, 50, 60)
        raw_minus, raw_plus = store.raw_response(direction)

        if model is not None:
            nominal = store.anchor_state_p32(direction.anchor_row)
            minus_in, plus_in = store.branch_inputs_p32(direction)
            physical = np.stack([nominal, minus_in, plus_in], axis=0)
            normalized = (physical - point_mean[None]) / point_scale[None]
            normalized[:, :, ~wet_bool] = 0.0
            state = torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32)).to(device)
            regime_static = statics[REGIMES.index(direction.regime)]
            static = torch.from_numpy(
                np.broadcast_to(regime_static, (3, *regime_static.shape)).copy()
            ).to(device=device, dtype=torch.float32)

        lead_values: dict[int, float] = {}
        with torch.no_grad():
            model_out: dict[int, tuple[Any, Any]] = {}
            if model is not None:
                for call in range(1, max(leads) // 10 + 1):
                    features = torch.cat([state, static], dim=1)
                    state = model(features) * wet_float
                    lead = call * 10
                    if lead in leads:
                        nominal_out, minus_out, plus_out = state[0], state[1], state[2]
                        model_out[lead] = (minus_out - nominal_out, plus_out - nominal_out)

            for lead_index, lead in enumerate(leads):
                truth_minus = torch.from_numpy(np.ascontiguousarray(raw_minus[lead_index], dtype=np.float32)).to(device)
                truth_plus = torch.from_numpy(np.ascontiguousarray(raw_plus[lead_index], dtype=np.float32)).to(device)
                r_m_minus = oriented_response(truth_minus, sigma_t, -1, direction.alpha)
                r_m_plus = oriented_response(truth_plus, sigma_t, 1, direction.alpha)
                if model is None:
                    r_f_minus = torch.zeros_like(r_m_minus)
                    r_f_plus = torch.zeros_like(r_m_plus)
                else:
                    diff_minus, diff_plus = model_out[lead]
                    r_f_minus = diff_minus / (-1.0 * float(direction.alpha))
                    r_f_plus = diff_plus / (1.0 * float(direction.alpha))
                scale_vector = torch.tensor(
                    [scales[direction.input_family][g][lead] for g in GROUPS],
                    device=device, dtype=torch.float32,
                )
                lead_values[lead] = float(
                    response_term(r_f_minus, r_f_plus, r_m_minus, r_m_plus, wet_bool_t, scale_vector).cpu()
                )

        loss = lead_values[10] if direction.array_group == "short" else float(np.mean([lead_values[k] for k in leads]))
        per_direction.append({
            "direction_id": direction.direction_id,
            "input_family": direction.input_family,
            "regime": direction.regime,
            "array_group": direction.array_group,
            "response_loss": loss,
            "lead_terms": lead_values,
        })

    values = np.array([r["response_loss"] for r in per_direction], dtype=np.float64)
    by_family = {
        h: float(np.mean([r["response_loss"] for r in per_direction if r["input_family"] == h]))
        for h in GROUPS
    }
    by_group_arr = {
        "short": [r["response_loss"] for r in per_direction if r["array_group"] == "short"],
        "long": [r["response_loss"] for r in per_direction if r["array_group"] == "long"],
    }
    lead10 = [r["lead_terms"][10] for r in per_direction]
    return {
        "n_directions": len(per_direction),
        "mean_response_loss": float(values.mean()),
        "median_response_loss": float(np.median(values)),
        "min_response_loss": float(values.min()),
        "max_response_loss": float(values.max()),
        "mean_by_input_family": by_family,
        "mean_by_array_group": {k: float(np.mean(v)) for k, v in by_group_arr.items() if v},
        "mean_lead_10_term": float(np.mean(lead10)),
        "per_direction": per_direction,
    }


def run(*, device_name: str = "auto", steps: tuple[int, ...] = (1920, 3840, 5760, 7680)) -> dict[str, Any]:
    started = time.monotonic()
    report = json.loads(B_REPORT_PATH.read_text())
    screen = json.loads(SCREEN_CONTRACT_PATH.read_text())
    seed = int(screen["primary_seed"])
    n_updates = int(screen["screen"]["response_updates"])

    require_runtime()
    device = _device(device_name)
    seed_everything(seed)

    group = zarr.open_consolidated(str(DATASET_PATH), mode="r")
    assert_store_is_v3(group)
    wet_array, _, _ = store_wind_normalization(group)
    wet_array = np.asarray(wet_array, dtype=bool)

    normalization_path = Path(report["normalization"]["artifact"])
    if _file_sha256(normalization_path) != report["normalization"]["artifact_sha256"]:
        raise Tier0Error("arm B's published normalizer changed on disk")
    normalizer_npz = np.load(normalization_path)
    point_mean = normalizer_npz["pointwise_mean"].astype(np.float32)
    point_scale = normalizer_npz["pointwise_scale"].astype(np.float32)

    production_contract = json.loads(PRODUCTION_CONTRACT_PATH.read_text())
    statics, _provenance = parent_train.physical_static_block(
        production_contract["sources"], group, point_mean, point_scale
    )
    architecture = ProductionArchitecture(**production_contract["architecture"])

    sigma_t = torch.from_numpy(point_scale).to(device)
    wet_bool_t = torch.from_numpy(wet_array).to(device)
    wet_float = torch.from_numpy(wet_array.astype(np.float32))[None, None].to(device)

    # The exact 480-direction prefix every screen candidate consumed.
    train_directions = load_direction_table("train")
    schedule = build_auxiliary_schedule(seed, train_directions)[:n_updates]
    if len(schedule) != n_updates:
        raise Tier0Error(f"schedule prefix is {len(schedule)}, expected {n_updates}")
    train_store = ResponseStore("train")
    scales = load_response_scales()

    checkpoint_dir = Path(report["published_checkpoint"]["checkpoint"]).parent / "training_checkpoints"
    by_step = {int(c["optimizer_step"]): c for c in report["checkpoints"]}

    # Product 2 baseline first: no model, no GPU work, pure reference scale.
    print("[tier0] zero-response predictor on the 480-direction schedule prefix ...", flush=True)
    zero_response = _training_loss_on_schedule(
        None, schedule, train_store, scales, point_mean, point_scale, sigma_t,
        statics, wet_array, wet_bool_t, wet_float, device,
    )
    print(f"[tier0]   l_zero mean = {zero_response['mean_response_loss']:.4f}", flush=True)

    per_step: dict[str, Any] = {}
    for step in steps:
        if step not in by_step:
            raise Tier0Error(f"arm B has no published checkpoint at step {step}")
        record = by_step[step]
        print(f"[tier0] control step {step}: loading {record['checkpoint']} ...", flush=True)
        model = _load_control_model(
            checkpoint_dir / record["checkpoint"], record["checkpoint_sha256"], architecture, device
        )

        print(f"[tier0] control step {step}: S_resp_10:60 (216 validation directions) ...", flush=True)
        response_validation = evaluate_response_validation(
            model, device, point_mean, point_scale, wet_array, statics
        )
        print(f"[tier0]   S_resp_10:60 = {response_validation['S_resp_10_60']:.4f}", flush=True)

        print(f"[tier0] control step {step}: training-loss l on the schedule prefix ...", flush=True)
        training_loss = _training_loss_on_schedule(
            model, schedule, train_store, scales, point_mean, point_scale, sigma_t,
            statics, wet_array, wet_bool_t, wet_float, device,
        )
        print(f"[tier0]   l mean = {training_loss['mean_response_loss']:.4f}", flush=True)

        per_step[str(step)] = {
            "optimizer_step": step,
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "response_validation": response_validation,
            "response_training_loss": training_loss,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "status": "complete",
        "purpose": "tier_0_missing_baselines_for_step_12_diagnostic_only_not_a_selection_input",
        "control_model": "model_c_adjoint_faithful_nominal_control_v1",
        "seed": seed,
        "auxiliary_schedule_prefix": n_updates,
        "zero_response_predictor": zero_response,
        "control_by_step": per_step,
        "elapsed_seconds": time.monotonic() - started,
    }
    result = json_safe(result)
    result["content_sha256"] = _json_sha256(result)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / "tier0_control_response_baselines.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[tier0] wrote {out_path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--steps", default="1920,3840,5760,7680",
        help="Comma-separated arm B checkpoint steps to score.",
    )
    args = parser.parse_args(argv)
    steps = tuple(int(s) for s in args.steps.split(","))
    result = run(device_name=args.device, steps=steps)
    summary = {
        "l_zero_response": result["zero_response_predictor"]["mean_response_loss"],
        "control": {
            step: {
                "S_resp_10_60": v["response_validation"]["S_resp_10_60"],
                "l_mean": v["response_training_loss"]["mean_response_loss"],
            }
            for step, v in result["control_by_step"].items()
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
