"""Execution step 12 of docs/Adjoint_faithful_response_training_plan.md:
the four-lambda primary-seed forward-only screen (plan section 14.4).

Contract-driven: ``--contract`` selects the frozen screen contract, which
supplies the candidate grid, the optimizer-step budget, and the arm B step
the candidate is matched against.

- ``config/forward_response_lambda_screen_v1.json`` -- the original
  {0.03, 0.10, 0.30, 1.00} grid screened at step 1,920. Ran to completion
  2026-08-27; every candidate failed forward feasibility. Kept reachable so
  that result stays reproducible.
- ``config/forward_response_lambda_screen_v2.json`` -- the 2026-08-27
  amendment: the {3e-4, 1e-3, 3e-3, 1e-2} grid screened at the full 7,680
  steps, against arm B's step-7,680 checkpoint. See the plan's
  "Implementation status and amendments (2026-08-27, step 12 re-screen)"
  section for why both the grid and the measurement step moved.

Deliberately a standalone script, not a short invocation of
``train_response.run()``, so the screen's own reporting and disposition
rules stay separate from the production runner. Reuses
``train_response.auxiliary_update`` (the same section-15.2 mechanism the
full C run will use) and every other shared production function
(``training_pointwise_normalizers`` is *not* recomputed -- it loads arm B's
already-published, hash-verified normalizer directly, since "identical
initialization/batch order" already pins it to the same recipe and result).

For each candidate, from the *same* random initialization and nominal batch
order as arm B (primary seed 20260724):

1. train the contract's step budget, with one auxiliary response update
   every 4th optimizer step, drawn as a prefix of the frozen full-run
   schedule (at 7,680 steps that prefix is the entire 1,920-entry schedule);
2. follow section 15.1's learning-rate schedule exactly -- ``initial_learning_rate``
   through ``round(maximum_steps * decay_fraction)``, then ``* decay_factor``.
   A screen that stops at 1,920 never reaches the decay, but a full-length
   one must or it is not matched to arm B at the step it is compared against;
3. evaluate the unchanged ``validate_checkpoint`` (nominal View 1) and
   ``evaluate_response_validation`` (View 2, ``S_resp_10:60``) at the final step;
4. apply plan section 14.4's reject/select rule against arm B's own
   already-published checkpoint at the contract's matched step (the "matched
   lambda-zero control" -- loaded from its report, not retrained).

No MITgcm adjoint, TAF output, FNO adjoint map, blind response case, or
test metric is read at any point (each contract's own
``forbidden_decision_inputs``).
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
from oceanfno import train_response as tr  # noqa: E402
from oceanfno.runtime import (  # noqa: E402
    AUDIT_TERMS, ChunkAwareBatchSampler, DataLoader, STATE_CHANNEL_COUNT,
    _device, _file_sha256, _json_sha256, json_safe, require_runtime, seed_everything, torch,
)
from oceanfno.dataset import (  # noqa: E402
    TRAIN_CODE, assert_store_is_v3, records_for_rollout_split, store_codes,
    store_wind_normalization, validation_records, validation_starts,
    western_boundary_mask, STATIC_FEATURES, EXPERIMENTS, RolloutDataset,
)
from oceanfno.objective import production_loss_config  # noqa: E402
from oceanfno.model import ProductionArchitecture, build_model, parameter_count, EXPECTED_PARAMETER_COUNT  # noqa: E402
from oceanfno.spectral_norm import apply_mode_spectral_norm, materialized_state_dict, mode_sigma_summary  # noqa: E402
from oceanfno.perturbation_growth import initial_direction  # noqa: E402
from oceanfno.validation import validate_checkpoint  # noqa: E402
from oceanfno.response_dataset import REGIMES, ResponseStore, build_auxiliary_schedule, load_direction_table, schedule_sha256  # noqa: E402
from oceanfno.response_objective import load_response_scales  # noqa: E402
from oceanfno.response_validation import evaluate_response_validation  # noqa: E402

DEFAULT_SCREEN_CONTRACT_PATH = PROJECT_ROOT / "config" / "forward_response_lambda_screen_v2.json"
B_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "af_fno" / "C" / "model_c_adjoint_faithful_nominal_control_v1"
    / "seed_20260724" / "report.json"
)
RESPONSE_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "af_fno" / "response" / "forward_response_v1"
DATASET_PATH = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno/datasets/trajectories_v3.zarr")
PRODUCTION_CONTRACT_PATH = PROJECT_ROOT / "config" / "model_c_production_1in_1out_spectralnorm_v1.json"
DEFAULT_SCRATCH_CHECKPOINT_ROOT = (
    "/bigscratch/mjalabert314/bire_james25_repro/af_fno/models/C/lambda_screen"
)


class LambdaScreenError(RuntimeError):
    """Raised when one candidate of the lambda screen cannot be legitimately run."""


def _response_loss_summary(log: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary statistics for the auxiliary stream's own loss.

    The first version of this recorded only ``mean_response_loss``, which
    turned out to be the one number that cannot answer the question the
    screen exists to ask: whether the auxiliary term makes progress on its
    own objective. A flat mean is consistent both with a loss that never
    moves and with one that halves halfway through. The quartile means below
    give the trend, and ``_write_response_loss_log`` keeps the full
    per-update series in a sidecar so the trajectory is recoverable without
    re-running 1,920 steps.
    """

    values = np.array([r["response_loss"] for r in log], dtype=np.float64)
    # np.array_split rather than a fixed stride: a short smoke run with fewer
    # than four updates would otherwise index an empty slice and write NaN
    # into the artifact.
    quarters = [chunk for chunk in np.array_split(values, 4) if chunk.size]
    quarter = max(1, len(values) // 4)
    return {
        "n_updates": len(log),
        "mean_response_loss": float(values.mean()),
        "median_response_loss": float(np.median(values)),
        "min_response_loss": float(values.min()),
        "max_response_loss": float(values.max()),
        "mean_first_quarter": float(values[:quarter].mean()),
        "mean_last_quarter": float(values[-quarter:].mean()),
        "quarter_means": [float(chunk.mean()) for chunk in quarters],
        "mean_by_array_group": {
            name: float(np.mean([r["response_loss"] for r in log if r["array_group"] == name]))
            for name in ("short", "long")
            if any(r["array_group"] == name for r in log)
        },
        "mean_by_input_family": {
            family: float(np.mean([r["response_loss"] for r in log if r["input_family"] == family]))
            for family in ("U", "V", "Theta", "SSH")
            if any(r["input_family"] == family for r in log)
        },
    }


def _write_response_loss_log(output_root: Path, lambda_resp: float, log: list[dict[str, Any]]) -> Path:
    """The full per-update series, one JSON object per auxiliary update.

    A sidecar rather than a field on the candidate result: the decision
    artifact stays small and hash-stable in shape, while the series it
    summarizes remains inspectable.
    """

    path = output_root / f"candidate_lambda_{lambda_resp}_response_loss_log.jsonl"
    if path.exists():
        raise LambdaScreenError(f"refusing to overwrite an existing response-loss log: {path}")
    with path.open("w", encoding="utf-8") as stream:
        for record in log:
            stream.write(json.dumps(json_safe(record), sort_keys=True) + "\n")
    return path


def _load_b_control(matched_step: int) -> dict[str, Any]:
    report = json.loads(B_REPORT_PATH.read_text())
    summary = next(s for s in report["validation_summaries"] if s["optimizer_step"] == matched_step)
    return {
        "matched_optimizer_step": matched_step,
        "short_auc_10_90": summary["short_auc_10_90"],
        "growth": summary["perturbation_growth"]["worst_growth_rate_per_call"],
        "normalization_artifact": report["normalization"]["artifact"],
        "normalization_artifact_sha256": report["normalization"]["artifact_sha256"],
    }


def run_candidate(
    lambda_resp: float,
    *,
    device_name: str = "auto",
    steps_override: int | None = None,
    contract_path: Path = DEFAULT_SCREEN_CONTRACT_PATH,
) -> dict[str, Any]:
    started = time.monotonic()
    contract_path = Path(contract_path)
    screen = json.loads(contract_path.read_text())
    output_root = RESPONSE_OUTPUT_ROOT / screen.get("output_subdirectory", "lambda_screen")
    if lambda_resp not in screen["candidate_lambda_resp"]:
        raise LambdaScreenError(f"{lambda_resp} is not one of the frozen candidates {screen['candidate_lambda_resp']}")
    seed = int(screen["primary_seed"])
    steps = int(screen["screen"]["optimizer_steps"]) if steps_override is None else int(steps_override)
    joint_every = int(screen["screen"]["joint_update_every"])
    matched_step = int(screen["screen"]["matched_lambda_zero_control"]["optimizer_step"])
    b_control = _load_b_control(matched_step)
    if steps_override is None and steps != matched_step:
        raise LambdaScreenError(
            f"screen trains {steps} steps but is matched against arm B at step {matched_step}"
        )

    require_runtime()
    device = _device(device_name)
    seed_everything(seed)

    group = zarr.open_consolidated(str(DATASET_PATH), mode="r")
    state = group["state"]
    store_static = group["static_features"]
    assert_store_is_v3(group)
    snapshot_split, pair_split = store_codes()
    wet_array, _, _ = store_wind_normalization(group)
    wet_array = np.asarray(wet_array, dtype=bool)

    normalization_path = Path(b_control["normalization_artifact"])
    if _file_sha256(normalization_path) != b_control["normalization_artifact_sha256"]:
        raise LambdaScreenError("arm B's published normalizer changed on disk")
    normalizer_npz = np.load(normalization_path)
    point_mean = normalizer_npz["pointwise_mean"].astype(np.float32)
    point_scale = normalizer_npz["pointwise_scale"].astype(np.float32)
    increment_values = normalizer_npz["increment_scale"].astype(np.float32)

    production_contract = json.loads(PRODUCTION_CONTRACT_PATH.read_text())
    statics, _provenance = parent_train.physical_static_block(
        production_contract["sources"], group, point_mean, point_scale
    )
    pressure_context, continuity_context = parent_train.physics_contexts(
        group, point_mean, point_scale, production_contract["sources"]["mitgcm_zonal_spacing"]["path"]
    )
    climatology_state, climatology_derived, _days = parent_train.train_only_climatology(state, wet_array)

    loss_config = production_loss_config()
    training_records = records_for_rollout_split(pair_split, TRAIN_CODE, rollout_steps=loss_config.rollout_steps)
    training_dataset = RolloutDataset(
        str(DATASET_PATH), training_records, point_mean, point_scale, statics, rollout_steps=loss_config.rollout_steps
    )
    loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(training_dataset, 4, seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    architecture = ProductionArchitecture(**production_contract["architecture"])
    model = build_model(architecture).to(device)
    spectral_provenance = apply_mode_spectral_norm(model)
    if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise LambdaScreenError("architecture parameter count mismatch")
    training = production_contract["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["initial_learning_rate"]),
        betas=tuple(float(b) for b in training["adam_betas"]),
        weight_decay=float(training["weight_decay"]),
    )
    # Section 15.1: 5e-4 through step 5,760, then 1e-4. train_response.run
    # derives the same boundary the same way; a 1,920-step screen never
    # reaches it, a 7,680-step one must.
    decay_step = int(round(int(training["maximum_steps"]) * float(training["decay_fraction"])))
    decay_factor = float(training["decay_factor"])

    wet_float = torch.from_numpy(wet_array.astype(np.float32))[None, None].to(device)
    wet_bool_t = torch.from_numpy(wet_array).to(device)
    boundary = torch.from_numpy(
        western_boundary_mask(wet_array, loss_config.western_boundary_width).astype(np.float32)
    )[None, None].to(device)
    increment_scale_t = torch.from_numpy(increment_values).to(device)
    sigma_t = torch.from_numpy(point_scale).to(device)

    direction_vec = initial_direction((1, STATE_CHANNEL_COUNT, *wet_array.shape), wet_float, device, seed=seed)

    train_directions = load_direction_table("train")
    full_schedule = build_auxiliary_schedule(seed, train_directions)
    schedule = full_schedule[: steps // joint_every]
    expected_updates = int(screen["screen"]["response_updates"]) if steps_override is None else steps // joint_every
    if steps_override is None and expected_updates > len(full_schedule):
        raise LambdaScreenError(
            f"contract asks for {expected_updates} response updates, frozen schedule holds {len(full_schedule)}"
        )
    if len(schedule) != expected_updates:
        raise LambdaScreenError(f"screen schedule has {len(schedule)} entries, expected {expected_updates}")
    response_store = ResponseStore("train")
    response_scales = load_response_scales()

    checkpoint_steps = set(int(v) for v in training["checkpoint_steps"] if int(v) <= steps)
    checkpoint_directory = (
        Path(screen["screen"].get("scratch_checkpoint_root", DEFAULT_SCRATCH_CHECKPOINT_ROOT))
        / f"lambda_{lambda_resp}" / f"seed_{seed}"
    )

    iterator = iter(loader)
    totals = {name: 0.0 for name in AUDIT_TERMS}
    growth_total = 0.0
    samples = 0
    response_loss_log: list[dict[str, Any]] = []
    auxiliary_index = 0

    for step in range(1, steps + 1):
        if step == decay_step + 1:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= decay_factor
        optimizer.zero_grad(set_to_none=True)
        model.train()
        step_samples = 0
        for _micro in range(2):
            try:
                raw_features, futures = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_features, futures = next(iterator)
            features = raw_features.to(device=device, dtype=torch.float32, non_blocking=True)
            futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
            predictions = parent_train.state_unroll(model, features, wet_float, loss_config.rollout_steps)
            terms, growth, direction_vec = parent_train.evaluate_loss(
                predictions, futures, features[:, :STATE_CHANNEL_COUNT], wet_float, boundary,
                increment_scale_t, loss_config, pressure_context, continuity_context,
                model=model, static=features[:, STATE_CHANNEL_COUNT:], direction=direction_vec,
            )
            if not all(bool(torch.isfinite(terms[n]).item()) for n in AUDIT_TERMS):
                raise LambdaScreenError(f"lambda={lambda_resp}: nominal objective non-finite at step {step}")
            (terms["total"] / 2).backward()
            batch = int(features.shape[0])
            for name in AUDIT_TERMS:
                totals[name] += float(terms[name].detach().cpu()) * batch
            growth_total += float(growth.cpu()) * batch
            step_samples += batch

        if step % joint_every == 0:
            direction_row = schedule[auxiliary_index]
            auxiliary_index += 1
            result = tr.auxiliary_update(
                model, direction_row, response_store, response_scales, point_mean, point_scale,
                sigma_t, statics[REGIMES.index(direction_row.regime)], wet_array, wet_bool_t, wet_float,
                device, lambda_resp,
            )
            if not np.isfinite(result["response_loss"]):
                raise LambdaScreenError(f"lambda={lambda_resp}: response loss non-finite at step {step} ({direction_row.direction_id})")
            response_loss_log.append({
                "step": step,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "input_family": direction_row.input_family,
                "regime": direction_row.regime,
                "array_group": direction_row.array_group,
                **result,
            })

        if not all(bool(torch.isfinite(p.grad).all().item()) for p in model.parameters() if p.grad is not None):
            raise LambdaScreenError(f"lambda={lambda_resp}: gradients non-finite at step {step}")
        optimizer.step()
        samples += step_samples

        if step in checkpoint_steps:
            # Section 14.4 discards screen state once lambda is frozen, and
            # these are written to scratch, never published. They exist because
            # the v1 screen kept nothing: with no candidate checkpoint there
            # was no way to evaluate a candidate's response loss as a snapshot
            # against arm B's, only its running mean against B's endpoint,
            # which is not a like-for-like comparison.
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": materialized_state_dict(model), "optimizer_step": step,
                 "lambda_resp": lambda_resp, "seed": seed},
                checkpoint_directory / f"step_{step:05d}.pt",
            )

    if auxiliary_index != len(schedule):
        raise LambdaScreenError(f"used {auxiliary_index} of {len(schedule)} scheduled auxiliary updates")

    window = {name: totals[name] / samples for name in AUDIT_TERMS}

    records = validation_records()
    growth_starts = [int(validation_starts()[0]), int(validation_starts()[-1])]
    growth_states = []
    for start in growth_starts:
        raw = np.asarray(state[0, start], dtype=np.float32)
        normalized = (raw - point_mean) / point_scale
        normalized[:, ~wet_array] = 0.0
        growth_states.append(torch.from_numpy(np.ascontiguousarray(normalized))[None].to(device))
    growth_static = torch.from_numpy(np.ascontiguousarray(statics[0]))[None].to(device)

    from oceanfno.model import ProductionStepper
    from oceanfno.perturbation_growth import growth_rate_summary

    model.eval()
    stepper = ProductionStepper(model=model, device=device, wet=wet_array, mean=point_mean, scale=point_scale, statics=statics)
    nominal_summary = validate_checkpoint(stepper, state, store_static, records, climatology_state, climatology_derived, wet_array)
    nominal_summary.pop("arrays", None)
    nominal_summary["perturbation_growth"] = growth_rate_summary(model, growth_states, growth_static, wet_float)

    response_summary = evaluate_response_validation(model, device, point_mean, point_scale, wet_array, statics)

    result = {
        "status": "complete",
        "lambda_resp": lambda_resp,
        "seed": seed,
        "steps": steps,
        "training_window": window,
        "spectral_norm_final": mode_sigma_summary(model),
        "nominal_validation": nominal_summary,
        "response_validation": response_summary,
        "response_loss_log_summary": _response_loss_summary(response_loss_log),
        "matched_lambda_zero_control": b_control,
        "screen_contract": str(contract_path),
        "screen_contract_sha256": _file_sha256(contract_path),
        "learning_rate_schedule": {
            "initial": float(training["initial_learning_rate"]),
            "decay_step": decay_step,
            "decay_factor": decay_factor,
            "final": float(optimizer.param_groups[0]["lr"]),
        },
        "scratch_checkpoints": sorted(str(p) for p in checkpoint_directory.glob("step_*.pt")),
        "elapsed_seconds": time.monotonic() - started,
    }
    result = json_safe(result)
    result["content_sha256"] = _json_sha256(result)

    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / f"candidate_lambda_{lambda_resp}.json"
    if out_path.exists():
        raise LambdaScreenError(f"refusing to overwrite an existing candidate result: {out_path}")
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_response_loss_log(output_root, lambda_resp, response_loss_log)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-resp", type=float, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--contract", type=Path, default=DEFAULT_SCREEN_CONTRACT_PATH,
        help="Frozen screen contract; supplies the candidate grid, step budget, and matched control step.",
    )
    parser.add_argument(
        "--steps-override", type=int, default=None,
        help="Smoke-test only: run fewer than the frozen 1,920 steps. Never use for a real screen candidate.",
    )
    args = parser.parse_args(argv)
    result = run_candidate(
        args.lambda_resp, device_name=args.device, steps_override=args.steps_override,
        contract_path=args.contract,
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("nominal_validation", "response_validation")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
