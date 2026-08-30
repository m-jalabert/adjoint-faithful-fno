"""Execution steps 11/13 of docs/Adjoint_faithful_response_training_plan.md.

The one common parameterized runner for arm B (response disabled) and arm C
(response enabled): the frozen ``src/oceanfno/train.py`` "cannot be
parameterized in place" (it hardcodes the parent's version string, primary
seed, output paths, and pinned source hashes as module constants used inside
its own contract validation), so this is a *new* module rather than an edit
to that one -- section 23.3 keeps ``train.py`` byte-unchanged as "the
immutable primary-seed equivalence reference, not the B/C study runner".

Reuses, unchanged, every piece of ``train.py`` that is not intrinsically
parent-specific: the model/objective/dataset/spectral-norm/validation
modules (imported the same way ``train.py`` imports them), and several of
``train.py``'s own helper functions directly (``physical_static_block``,
``physics_contexts``, ``evaluate_loss``, ``split_summary``,
``acceptance_gate``, ``_verify_file``, ``_verify_dataset``) -- these are
generic given a contract dict, not closed over the parent's specific
version/seed, so importing them verbatim is strictly safer than a parallel
reimplementation that could silently drift. Only the pieces that really are
parent-specific -- contract validation, output naming, and the top-level
``preflight``/``run`` orchestration -- are new code here, structured to
mirror ``train.py``'s own control flow as closely as possible so that the
step-11 equivalence harness (``tests/test_response_training.py``) has a
narrow, well-defined difference to check rather than two independent
implementations that happen to agree.

Response mixing itself (arm C, plan section 15.2) is out of scope for this
module's first version: ``response.enabled`` must be ``false`` in the loaded
contract until ``response_dataset.py``/``response_objective.py``/
``response_spectral_context.py`` exist and are wired in (execution step 13).
Attempting to enable it before then raises ``ResponseTrainingContractError``
rather than silently ignoring the flag.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from . import train as parent_train
from .runtime import (
    AUDIT_TERMS,
    ChunkAwareBatchSampler,
    DataLoader,
    STATE_CHANNEL_COUNT,
    _device,
    _file_sha256,
    _json_sha256,
    json_safe,
    require_runtime,
    seed_everything,
    torch,
)
from .dataset import (
    DATASET_VERSION,
    EXPERIMENTS,
    RolloutDataset,
    STATIC_FEATURES,
    TRAIN_CODE,
    assert_store_is_v3,
    records_for_rollout_split,
    store_codes,
    store_wind_normalization,
    training_increment_scale,
    training_pointwise_normalizers,
    validation_records,
    validation_starts,
    western_boundary_mask,
)
from .objective import LOSS_CONTRACT_SHA256, loss_contract, production_loss_config
from .model import (
    CHECKPOINT_DIRECTORY,
    DivergenceError,
    EXPECTED_PARAMETER_COUNT,
    MANIFEST_NAME,
    README_NAME,
    ProductionArchitecture,
    ProductionStepper,
    build_model,
    parameter_count,
    state_unroll,
)
from .perturbation_growth import growth_rate_summary, initial_direction
from .spectral_norm import apply_mode_spectral_norm, materialized_state_dict, mode_sigma_summary
from .validation import PRIMARY_FIELDS, _plot, select_by_validation
from .response_dataset import (
    REGIMES,
    ResponseDirection,
    ResponseStore,
    build_auxiliary_schedule,
    load_direction_table,
    schedule_sha256,
)
from .response_objective import load_response_scales, oriented_response, response_term
from .response_spectral_context import auxiliary_chain

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Reused verbatim from train.py -- generic given a contract dict, not
#: closed over the parent's specific version/seed (see module docstring).
_verify_file = parent_train._verify_file
_verify_dataset = parent_train._verify_dataset
physical_static_block = parent_train.physical_static_block
physics_contexts = parent_train.physics_contexts
evaluate_loss = parent_train.evaluate_loss
acceptance_gate = parent_train.acceptance_gate

#: Every one of these is frozen identical between the parent and this study
#: by plan section 5.1 ("trajectory store... six-call/60-day autonomous
#: unroll... Adam, learning-rate schedule, update count, checkpoints...").
#: Importing the parent's own constants, rather than re-declaring equal
#: values here, makes that equality a Python identity rather than a claim
#: that could quietly go stale.
ROLLOUT_STEPS = parent_train.ROLLOUT_STEPS
BATCH_SIZE = parent_train.BATCH_SIZE
MICROBATCH_SIZE = parent_train.MICROBATCH_SIZE
GRADIENT_ACCUMULATION_STEPS = parent_train.GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = parent_train.LEARNING_RATE
ADAM_BETAS = parent_train.ADAM_BETAS
WEIGHT_DECAY = parent_train.WEIGHT_DECAY
MAXIMUM_STEPS = parent_train.MAXIMUM_STEPS
CHECKPOINT_STEPS = parent_train.CHECKPOINT_STEPS
DECAY_FRACTION = parent_train.DECAY_FRACTION
DECAY_FACTOR = parent_train.DECAY_FACTOR
TRAINING_STARTS_PER_REGIME = parent_train.TRAINING_STARTS_PER_REGIME
TRAINING_RECORDS = parent_train.TRAINING_RECORDS
STATE_TRANSITIONS = parent_train.STATE_TRANSITIONS
SHORT_AUC_TOLERANCE = parent_train.SHORT_AUC_TOLERANCE
WORST_LONG_RATIO_CEILING = parent_train.WORST_LONG_RATIO_CEILING
REQUIRED_MITGCM_SOURCES = parent_train.REQUIRED_MITGCM_SOURCES
GROWTH_RATE_CEILING = parent_train.GROWTH_RATE_CEILING
REFERENCE_VERSION = parent_train.REFERENCE_VERSION
REFERENCE_GROWTH_RATE = parent_train.REFERENCE_GROWTH_RATE
PENALTY_ARM_GROWTH_RATES = parent_train.PENALTY_ARM_GROWTH_RATES

#: JSON-path prefixes the study contract is permitted to differ from the
#: parent contract on (the "explicit infrastructure whitelist" of plan
#: section 23.1). Anything outside these must be byte-identical -- checked
#: by ``_diff_outside_whitelist`` below, not by field-by-field duplication
#: of train.py's own load_contract (duplicating that logic risks the two
#: silently drifting apart; a whitelist diff against the real parent
#: contract on disk cannot drift, because it is not a copy).
WHITELIST_PREFIXES = (
    ("version",),
    ("contract_status",),
    ("study_contract",),
    ("output",),
    ("training", "seed"),
    ("response",),
    # Arm C reads the curated forward-response store; the parent does not.
    # Already declared in the contract's own paired_causal_whitelist_json_paths
    # as an intended B/C difference, but this tuple -- not that field -- is what
    # load_contract enforces, and the omission blocked every arm C preflight.
    # Deliberately the single leaf, not ("read_contract",): adjoint_state,
    # blind_response_state, inference_state and intermediate_wind_state stay
    # outside the whitelist and stay pinned to the parent's false.
    ("read_contract", "response_state"),
)

#: For arm B (response disabled), only train_response.py's own hash is on
#: the active code path; the response_*.py modules do not need to exist yet.
#: Arm C additionally requires all four.
RESPONSE_RUNNER_SOURCES = (
    "src/oceanfno/response_dataset.py",
    "src/oceanfno/response_objective.py",
    "src/oceanfno/response_spectral_context.py",
    "src/oceanfno/response_validation.py",
)
RUNNER_SOURCE = "src/oceanfno/train_response.py"
RESPONSE_LOG_NAME = "response_loss_log.jsonl"


class ResponseTrainingContractError(RuntimeError):
    """Raised when the B/C study contract cannot be legitimately used to train."""


def _model_oriented_response(output_diff: Any, sign: int, alpha: float) -> Any:
    """``(F(x+s*alpha*v)-F(x))/(s*alpha)`` from the model's own *normalized*-
    state output difference. No sigma division: see
    ``response_validation``'s module docstring for why the model side and
    the (physical-unit, sigma-divided) truth side use different formulas
    for the same underlying quantity."""

    return output_diff / (float(sign) * float(alpha))



def _response_loss_summary(log: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Trend statistics for the auxiliary stream's own loss.

    Deliberately the same shape as ``run_lambda_screen._response_loss_summary``
    so a C run's response trajectory can be read against the screen candidate
    that chose its lambda without reformatting either.
    """

    values = np.array([r["response_loss"] for r in log], dtype=np.float64)
    quarters = [chunk for chunk in np.array_split(values, 4) if chunk.size]
    return {
        "n_updates": len(log),
        "mean_response_loss": float(values.mean()),
        "median_response_loss": float(np.median(values)),
        "min_response_loss": float(values.min()),
        "max_response_loss": float(values.max()),
        "mean_first_quarter": float(quarters[0].mean()),
        "mean_last_quarter": float(quarters[-1].mean()),
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


def auxiliary_update(
    model: Any,
    direction: ResponseDirection,
    store: ResponseStore,
    scales: Mapping[str, Mapping[str, Mapping[int, float]]],
    point_mean: Any,
    point_scale: Any,
    sigma_t: Any,
    statics: Any,
    wet_bool_np: Any,
    wet_bool_t: Any,
    wet_float: Any,
    device: Any,
    lambda_resp: float,
) -> dict[str, Any]:
    """One section-15.2 auxiliary chain: batched (nominal, minus, plus)
    forward per lead, WITH gradient, inside a spectral-buffer
    snapshot/restore context, backpropagating
    ``lambda_resp * response_loss`` immediately (a standalone addition to
    whatever gradients are already accumulated on ``model.parameters()``
    from the nominal microbatch loop -- not divided by
    ``gradient_accumulation_steps``, since the auxiliary update happens once
    per *optimizer step*, not once per microbatch).

    Returns ``{"response_loss": float, "lead_terms": {lead: float}}`` for
    logging; raises nothing on non-finite loss -- the caller checks
    finiteness the same way it already checks the nominal gradients, so one
    non-finite guard covers both paths.
    """

    leads = (10,) if direction.array_group == "short" else (10, 20, 30, 40, 50, 60)
    nominal = store.anchor_state_p32(direction.anchor_row)
    minus_in, plus_in = store.branch_inputs_p32(direction)
    raw_minus, raw_plus = store.raw_response(direction)

    physical = np.stack([nominal, minus_in, plus_in], axis=0)
    normalized = (physical - point_mean[None]) / point_scale[None]
    normalized[:, :, ~wet_bool_np] = 0.0
    state = torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32)).to(device)
    static = torch.from_numpy(np.broadcast_to(statics, (3, *statics.shape)).copy()).to(device=device, dtype=torch.float32)

    lead_terms: dict[int, Any] = {}
    lead_values: dict[int, float] = {}
    with auxiliary_chain(model):
        for call in range(1, max(leads) // 10 + 1):
            features = torch.cat([state, static], dim=1)
            state = model(features) * wet_float
            lead = call * 10
            if lead not in leads:
                continue
            nominal_out, minus_out, plus_out = state[0], state[1], state[2]
            r_f_minus = _model_oriented_response(minus_out - nominal_out, -1, direction.alpha)
            r_f_plus = _model_oriented_response(plus_out - nominal_out, 1, direction.alpha)
            lead_index = leads.index(lead)
            truth_minus = torch.from_numpy(np.ascontiguousarray(raw_minus[lead_index], dtype=np.float32)).to(device)
            truth_plus = torch.from_numpy(np.ascontiguousarray(raw_plus[lead_index], dtype=np.float32)).to(device)
            r_m_minus = oriented_response(truth_minus, sigma_t, -1, direction.alpha)
            r_m_plus = oriented_response(truth_plus, sigma_t, 1, direction.alpha)
            scale_vector = torch.tensor(
                [scales[direction.input_family][g][lead] for g in ("U", "V", "Theta", "SSH")],
                device=device, dtype=torch.float32,
            )
            term = response_term(r_f_minus, r_f_plus, r_m_minus, r_m_plus, wet_bool_t, scale_vector)
            lead_terms[lead] = term
            lead_values[lead] = float(term.detach().cpu())

        if direction.array_group == "short":
            response_loss = lead_terms[10]
        else:
            response_loss = sum(lead_terms[k] for k in (10, 20, 30, 40, 50, 60)) / 6.0

        if not bool(torch.isfinite(response_loss).item()):
            return {"response_loss": float("nan"), "lead_terms": lead_values, "direction_id": direction.direction_id}
        (float(lambda_resp) * response_loss).backward()

    return {
        "response_loss": float(response_loss.detach().cpu()),
        "lead_terms": lead_values,
        "direction_id": direction.direction_id,
    }


def _whitelisted(path: tuple[str, ...]) -> bool:
    return any(path[: len(prefix)] == prefix for prefix in WHITELIST_PREFIXES)


def _diff_outside_whitelist(
    contract: Any, parent: Any, path: tuple[str, ...] = ()
) -> list[str]:
    """Every JSON path where ``contract`` and ``parent`` disagree, excluding
    whitelisted subtrees -- proves the study contract is the parent contract
    plus only the declared infrastructure delta, not a fork."""

    if _whitelisted(path):
        return []
    if isinstance(contract, dict) and isinstance(parent, dict):
        diffs = []
        for key in sorted(set(contract) | set(parent)):
            child_path = path + (key,)
            if _whitelisted(child_path):
                continue
            if key not in contract:
                diffs.append(".".join(child_path) + ": missing in study contract")
            elif key not in parent:
                diffs.append(".".join(child_path) + ": absent from parent contract")
            else:
                diffs.extend(_diff_outside_whitelist(contract[key], parent[key], child_path))
        return diffs
    if contract != parent:
        return [f"{'.'.join(path) or '$'}: {contract!r} != {parent!r}"]
    return []


def load_contract(
    contract_path: str | Path,
    *,
    seed_override: int | None = None,
    lambda_override: float | None = None,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load a B/C study contract and prove it is the parent contract plus
    only the declared infrastructure delta (plan section 23.1's equality
    checker), then apply an optional seed override for a secondary paired
    replication (plan section 5.1's three seeds)."""

    resolved = Path(contract_path).resolve()
    contract = json.loads(resolved.read_text())
    study = contract.get("study_contract")
    if not isinstance(study, dict):
        raise ResponseTrainingContractError("contract has no study_contract block")

    parent_path = (PROJECT_ROOT / study["parent_config"]).resolve()
    if not parent_path.is_file():
        raise ResponseTrainingContractError(f"parent config missing: {parent_path}")
    if _file_sha256(parent_path) != study["parent_config_sha256"]:
        raise ResponseTrainingContractError("parent config changed on disk since the study was frozen")
    parent = json.loads(parent_path.read_text())

    diffs = _diff_outside_whitelist(contract, parent)
    if diffs:
        raise ResponseTrainingContractError(
            "study contract diverges from the parent contract outside the "
            f"declared whitelist: {diffs}"
        )

    if contract.get("contract_status") not in (
        "frozen_scientific_contract_runner_hashes_pending",
        "frozen_scientific_contract_and_runner_hashes",
    ):
        raise ResponseTrainingContractError(
            f"unexpected contract_status: {contract.get('contract_status')!r}"
        )
    if study.get("equality_reference_source") != "src/oceanfno/train.py":
        raise ResponseTrainingContractError("study contract does not name train.py as its reference")

    seed = int(contract["training"]["seed"]) if seed_override is None else int(seed_override)
    paired_seeds = tuple(int(s) for s in study.get("paired_seeds", ()))
    if seed not in paired_seeds:
        raise ResponseTrainingContractError(f"seed {seed} is not one of the frozen paired seeds {paired_seeds}")
    contract = json.loads(json.dumps(contract))  # deep copy before mutating
    contract["training"]["seed"] = seed

    response = contract.get("response", {})
    if lambda_override is not None:
        if not bool(response.get("enabled")):
            raise ResponseTrainingContractError("lambda_override was given but response.enabled is false")
        contract["response"]["lambda_resp"] = float(lambda_override)

    if verify_sources:
        _verify_dataset(contract)
        for key in REQUIRED_MITGCM_SOURCES:
            _verify_file(contract["sources"][key], key)
        for relative, expected in contract["source_hashes"].items():
            source = PROJECT_ROOT / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ResponseTrainingContractError(f"a pinned shared source changed: {relative}")
        runner_hashes = study.get("new_runner_source_hashes", {})
        required_runner_sources = (RUNNER_SOURCE,) if not response.get("enabled") else (
            RUNNER_SOURCE,
            *RESPONSE_RUNNER_SOURCES,
        )
        for relative in required_runner_sources:
            expected = runner_hashes.get(relative)
            source = PROJECT_ROOT / relative
            if not source.is_file():
                raise ResponseTrainingContractError(f"runner source missing: {relative}")
            actual = _file_sha256(source)
            if expected is not None and actual != expected:
                raise ResponseTrainingContractError(
                    f"runner source {relative} does not match its pinned hash "
                    "(re-freeze new_runner_source_hashes after an intentional change)"
                )

    return contract, resolved, _file_sha256(resolved)


def _resolve_output_paths(contract: Mapping[str, Any], seed: int) -> tuple[Path, Path]:
    output = contract["output"]
    scratch = Path(output["scratch_root_template"].format(seed=seed)).resolve()
    project = Path(output["project_root_template"].format(seed=seed)).resolve()
    return scratch, project


def split_summary() -> dict[str, Any]:
    """Reuses ``train.split_summary`` -- it closes over TRAINING_STARTS_PER_REGIME
    and TRAINING_RECORDS, both study-invariant (imported above from the same
    source), so calling the parent's own function is exact, not approximate."""

    return parent_train.split_summary()


def preflight(contract_path: str | Path, *, seed_override: int | None = None) -> dict[str, Any]:
    """Verify the contract, the sources, the record counts and the architecture.

    Mirrors ``train.preflight`` exactly in structure (see that function's own
    docstring); the only differences are contract loading (study-aware) and
    the ``version``/``arm``/``seed`` fields in the returned summary."""

    contract, resolved, digest = load_contract(contract_path, seed_override=seed_override)
    seed = int(contract["training"]["seed"])
    version = str(contract["version"])
    dataset = _verify_dataset(contract)
    group = zarr.open_consolidated(str(dataset), mode="r")
    assert_store_is_v3(group)
    _, pair_split = store_codes()
    records = records_for_rollout_split(pair_split, TRAIN_CODE, rollout_steps=ROLLOUT_STEPS)
    if len(records) != TRAINING_RECORDS:
        raise ResponseTrainingContractError(
            f"the training set is {len(records)} records, not {TRAINING_RECORDS}"
        )
    architecture = ProductionArchitecture(**contract["architecture"])
    result: dict[str, Any] = {
        "status": "ready",
        "version": version,
        "arm": contract["study_contract"]["arm"],
        "seed": seed,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "dataset_version": str(group.attrs["version"]),
        "split": split_summary(),
        "loss_contract_sha256": LOSS_CONTRACT_SHA256,
        "from_scratch": True,
        "parent_checkpoint": None,
        "rollout_steps": ROLLOUT_STEPS,
        "static_channels": list(STATIC_FEATURES),
        "external_input_channels": architecture.in_channels,
        "lifting_input_channels": architecture.lifting_in_channels,
        "training_rollout_records": len(records),
        "training_starts_per_regime": len(records) // len(EXPERIMENTS),
    }
    if torch is not None:
        seed_everything(seed)
        model = build_model(architecture)
        count = parameter_count(model)
        if count != EXPECTED_PARAMETER_COUNT:
            raise ResponseTrainingContractError(
                f"the architecture builds {count:,} parameters, not {EXPECTED_PARAMETER_COUNT:,}"
            )
        if float(model.local.weight.abs().max()) != 0.0:
            raise ResponseTrainingContractError("the local 3x3 branch is not zero-initialized")
        result["parameter_count"] = count
        result["local_branch_zero_initialized"] = True
        spectral = apply_mode_spectral_norm(model)
        conv = model.fno.fno_blocks.convs[0]
        with torch.no_grad():
            free = conv.weight.original.detach()
            capped = conv.weight.tensor.detach()

            def _exact(tensor: Any) -> Any:
                matrices = tensor.reshape(tensor.shape[0], tensor.shape[1], -1).permute(2, 0, 1)
                return torch.linalg.matrix_norm(matrices, ord=2)

            before, after = _exact(free), _exact(capped)
        if float(after.max()) > 1.0 + 1.0e-3:
            raise ResponseTrainingContractError(
                f"per-mode spectral normalization did not cap sigma_max at one: {float(after.max()):.6f}"
            )
        if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
            raise ResponseTrainingContractError("spectral normalization changed the parameter count")
        result["spectral_normalization"] = spectral
        result["spectral_norm_check"] = {
            "block_0_sigma_max_before": float(before.max()),
            "block_0_sigma_max_after": float(after.max()),
        }

    response = contract.get("response", {})
    if bool(response.get("enabled")):
        if response.get("lambda_resp") is None:
            raise ResponseTrainingContractError(
                "response.enabled is true but lambda_resp is null -- run the section-14.4 screen first"
            )
        schedule = build_auxiliary_schedule(seed, load_direction_table("train"))
        n_short = sum(1 for d in schedule if d.array_group == "short")
        declared = (int(response["short_updates"]), int(response["long_updates"]))
        if (n_short, len(schedule) - n_short) != declared:
            raise ResponseTrainingContractError(
                f"auxiliary schedule is {n_short} short / {len(schedule) - n_short} long, "
                f"contract declares {declared[0]} / {declared[1]}"
            )
        result["response_training"] = {
            "enabled": True,
            "lambda_resp": float(response["lambda_resp"]),
            "lambda_contract": response.get("lambda_contract"),
            "joint_update_every": int(response["joint_update_every"]),
            "auxiliary_updates": len(schedule),
            "short_updates": n_short,
            "long_updates": len(schedule) - n_short,
            "schedule_sha256": schedule_sha256(schedule),
            "dataset": response["dataset"],
            "dataset_role": response["dataset_role"],
        }
    else:
        result["response_training"] = {"enabled": False}
    return result


def _readme(report: Mapping[str, Any]) -> str:
    decision = report["selection_decision"]
    gate = report["acceptance_gate"]
    rows = "\n".join(
        "| {step:,} | {short} | {long} | {growth} |".format(
            step=int(summary["optimizer_step"]),
            short=" / ".join(f"{summary['short_auc_10_90'][field]:.3f}" for field in PRIMARY_FIELDS),
            long=" / ".join(f"{summary['long_ratio_to_climatology'][field]:.3f}" for field in PRIMARY_FIELDS),
            growth=f"{summary['perturbation_growth']['worst_growth_rate_per_call']:.4f}",
        )
        for summary in report["validation_summaries"]
    )
    decision_growth = float(decision.get("selected_growth_rate_per_call", float("nan")))
    return f"""# {report['version']} (seed {report['seed']})

Plan document arm `{report['arm']}`: `{report['study_contract']['role']}`.
Exact parent-protocol replay of `model_c_production_1in_1out_spectralnorm_v1`
through the common response-study runner, response disabled. See
`docs/Adjoint_faithful_response_training_plan.md` section 5.1/25 step 11.

Selected step {int(decision['selected_optimizer_step']):,} via
`{decision['branch']}`, growth rate {decision_growth:.5f} per call
(ceiling {GROWTH_RATE_CEILING}). Validation gate:
**{'pass' if gate['validation_conditions_pass'] else 'fail'}**.

| step | short AUC 10-90 (speed / SST / pressure) | long / climatology | growth rate |
| --- | --- | --- | --- |
{rows}

Report content SHA-256: `{report['content_sha256']}`.
"""


def run(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
    seed_override: int | None = None,
    steps_override: int | None = None,
    output_suffix: str | None = None,
) -> dict[str, Any]:
    """Train from random initialization, validate every checkpoint, publish one.

    Mirrors ``train.run`` step-for-step: same seeding point relative to model
    construction, same normalizer/dataset/loader construction, same
    spectral-norm-before-optimizer ordering, same accumulation-then-step
    loop, same checkpoint/validation/selection/report sequence.

    With ``response.enabled`` false this function's body is exactly arm B's
    path, unchanged: the auxiliary block below is skipped entirely, the
    schedule is never built, and no response artifact is opened. With it
    true (arm C) the section-15.2 auxiliary stream is mixed in on every
    ``joint_update_every``-th optimizer step. The schedule is keyed by an
    isolated counter/hash stream (``response_dataset._hash_key``), never by
    the parent RNG, so building it cannot perturb initialization or nominal
    batch order -- which is what lets B and C share an initialization and a
    batch sequence exactly.
    """

    if torch is None or DataLoader is None:  # pragma: no cover - environment dependent
        raise RuntimeError("training the response-aware emulator requires PyTorch")
    require_runtime()
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path, seed_override=seed_override)
    seed = int(contract["training"]["seed"])
    version = str(contract["version"])
    study = contract["study_contract"]
    dataset = _verify_dataset(contract)
    summary_of_split = split_summary()
    if (steps_override is None) != (output_suffix is None):
        raise ResponseTrainingContractError(
            "steps_override and output_suffix are smoke-test-only and must be given together, "
            "so an abbreviated run can never write to a publishable output path"
        )
    scratch, project = _resolve_output_paths(contract, seed)
    if output_suffix is not None:
        scratch = scratch.with_name(scratch.name + output_suffix)
        project = project.with_name(project.name + output_suffix)
    scratch_tmp = scratch.with_name(scratch.name + ".tmp")
    project_tmp = project.with_name(project.name + ".tmp")
    if any(p.exists() for p in (scratch, project, scratch_tmp, project_tmp)):
        raise FileExistsError(f"refusing to overwrite existing training output: {scratch} / {project}")

    training = contract["training"]
    seed_everything(seed)
    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    store_static = group["static_features"]
    assert_store_is_v3(group)
    snapshot_split, pair_split = store_codes()
    wet_array, _, _ = store_wind_normalization(group)
    wet_array = np.asarray(wet_array, dtype=bool)

    normalizers = training_pointwise_normalizers(group, snapshot_split)
    point_mean = normalizers["mean"]
    point_scale = normalizers["scale"]
    increment_values = training_increment_scale(group, pair_split, point_scale)
    statics, static_provenance = physical_static_block(contract["sources"], group, point_mean, point_scale)
    pressure_context, continuity_context = physics_contexts(
        group, point_mean, point_scale, contract["sources"]["mitgcm_zonal_spacing"]["path"]
    )
    climatology_state, climatology_derived, climatology_days = parent_train.train_only_climatology(
        state, wet_array
    )

    loss_config = production_loss_config()
    training_records = records_for_rollout_split(pair_split, TRAIN_CODE, rollout_steps=loss_config.rollout_steps)
    if len(training_records) != TRAINING_RECORDS:
        raise ResponseTrainingContractError("the training record count changed")
    training_dataset = RolloutDataset(
        dataset, training_records, point_mean, point_scale, statics, rollout_steps=loss_config.rollout_steps
    )
    microbatch = int(training["microbatch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(training_dataset, microbatch, seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    architecture = ProductionArchitecture(**contract["architecture"])
    model = build_model(architecture).to(device)
    spectral_provenance = apply_mode_spectral_norm(model)
    spectral_provenance["sigma_at_initialization"] = mode_sigma_summary(model)
    count = parameter_count(model)
    if count != EXPECTED_PARAMETER_COUNT:
        raise ResponseTrainingContractError(
            f"the architecture builds {count:,} parameters, not {EXPECTED_PARAMETER_COUNT:,}"
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["initial_learning_rate"]),
        betas=tuple(float(v) for v in training["adam_betas"]),
        weight_decay=float(training["weight_decay"]),
    )
    wet = torch.from_numpy(wet_array.astype(np.float32))[None, None].to(device)
    boundary = torch.from_numpy(
        western_boundary_mask(wet_array, loss_config.western_boundary_width).astype(np.float32)
    )[None, None].to(device)
    increment_scale = torch.from_numpy(increment_values).to(device)
    maximum_steps = int(training["maximum_steps"]) if steps_override is None else int(steps_override)
    decay_step = int(round(maximum_steps * float(training["decay_fraction"])))
    checkpoint_steps = tuple(int(v) for v in training["checkpoint_steps"])
    if steps_override is not None:
        checkpoint_steps = tuple(v for v in checkpoint_steps if v <= maximum_steps) or (maximum_steps,)

    response_contract = contract.get("response", {})
    response_enabled = bool(response_contract.get("enabled"))
    response_state: dict[str, Any] | None = None
    if response_enabled:
        lambda_resp = response_contract.get("lambda_resp")
        if lambda_resp is None:
            raise ResponseTrainingContractError(
                "response.enabled is true but lambda_resp is null -- run the section-14.4 screen first"
            )
        joint_every = int(response_contract["joint_update_every"])
        expected_joint = int(response_contract["joint_optimizer_steps"])
        if steps_override is not None:
            expected_joint = maximum_steps // joint_every
        if maximum_steps % joint_every != 0 or maximum_steps // joint_every != expected_joint:
            raise ResponseTrainingContractError(
                f"{maximum_steps} steps every {joint_every} gives "
                f"{maximum_steps // joint_every} auxiliary updates, contract declares {expected_joint}"
            )
        schedule = build_auxiliary_schedule(seed, load_direction_table("train"))
        if steps_override is not None:
            schedule = schedule[:expected_joint]
        if len(schedule) != expected_joint:
            raise ResponseTrainingContractError(
                f"auxiliary schedule has {len(schedule)} entries, expected {expected_joint}"
            )
        n_short = sum(1 for d in schedule if d.array_group == "short")
        n_long = len(schedule) - n_short
        if steps_override is None and (
            n_short != int(response_contract["short_updates"])
            or n_long != int(response_contract["long_updates"])
        ):
            raise ResponseTrainingContractError(
                f"auxiliary schedule is {n_short} short / {n_long} long, contract declares "
                f"{response_contract['short_updates']} / {response_contract['long_updates']}"
            )
        response_state = {
            "lambda_resp": float(lambda_resp),
            "joint_every": joint_every,
            "schedule": schedule,
            "schedule_sha256": schedule_sha256(schedule),
            "store": ResponseStore(response_contract["dataset_role"], dataset_path=Path(response_contract["dataset"])),
            "scales": load_response_scales(),
            "sigma_t": torch.from_numpy(point_scale).to(device),
            "wet_bool_t": torch.from_numpy(wet_array).to(device),
            "log": [],
            "index": 0,
        }

    scratch_tmp.parent.mkdir(parents=True, exist_ok=True)
    project_tmp.parent.mkdir(parents=True, exist_ok=True)
    scratch_tmp.mkdir()
    project_tmp.mkdir()
    checkpoint_directory = scratch_tmp / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir()
    normalization_path = scratch_tmp / "normalization.npz"
    np.savez_compressed(
        normalization_path,
        pointwise_mean=point_mean,
        pointwise_raw_scale=normalizers["raw_scale"],
        pointwise_scale=point_scale,
        channel_scale_floor=normalizers["floor"],
        increment_scale=increment_values,
    )

    direction = initial_direction((1, STATE_CHANNEL_COUNT, *wet_array.shape), wet, device, seed=seed)
    iterator = iter(loader)
    totals = {name: 0.0 for name in AUDIT_TERMS}
    growth_total = 0.0
    samples = 0
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    def _diverged(step: int, reason: str) -> None:
        (project_tmp / "divergence.json").write_text(
            json.dumps(
                {
                    "status": "diverged",
                    "version": version,
                    "seed": seed,
                    "reason": reason,
                    "optimizer_step": int(step),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        shutil.rmtree(scratch_tmp, ignore_errors=True)
        os.replace(project_tmp, project)
        raise DivergenceError(f"{reason} at optimizer step {step}")

    for step in range(1, maximum_steps + 1):
        if step == decay_step + 1:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= float(training["decay_factor"])
        optimizer.zero_grad(set_to_none=True)
        model.train()
        step_samples = 0
        for _micro in range(accumulation):
            try:
                raw_features, futures = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_features, futures = next(iterator)
            features = raw_features.to(device=device, dtype=torch.float32, non_blocking=True)
            futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
            predictions = state_unroll(model, features, wet, loss_config.rollout_steps)
            terms, growth, direction = evaluate_loss(
                predictions,
                futures,
                features[:, :STATE_CHANNEL_COUNT],
                wet,
                boundary,
                increment_scale,
                loss_config,
                pressure_context,
                continuity_context,
                model=model,
                static=features[:, STATE_CHANNEL_COUNT:],
                direction=direction,
            )
            if not all(bool(torch.isfinite(terms[n]).item()) for n in AUDIT_TERMS):
                _diverged(step, "training objective became non-finite")
            (terms["total"] / accumulation).backward()
            batch = int(features.shape[0])
            for name in AUDIT_TERMS:
                totals[name] += float(terms[name].detach().cpu()) * batch
            growth_total += float(growth.cpu()) * batch
            step_samples += batch

        if response_state is not None and step % response_state["joint_every"] == 0:
            row = response_state["schedule"][response_state["index"]]
            response_state["index"] += 1
            outcome = auxiliary_update(
                model,
                row,
                response_state["store"],
                response_state["scales"],
                point_mean,
                point_scale,
                response_state["sigma_t"],
                statics[REGIMES.index(row.regime)],
                wet_array,
                response_state["wet_bool_t"],
                wet,
                device,
                response_state["lambda_resp"],
            )
            if not np.isfinite(outcome["response_loss"]):
                _diverged(step, "response objective became non-finite")
            response_state["log"].append({
                "step": step,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "input_family": row.input_family,
                "regime": row.regime,
                "array_group": row.array_group,
                **outcome,
            })

        if not all(
            bool(torch.isfinite(p.grad).all().item()) for p in model.parameters() if p.grad is not None
        ):
            _diverged(step, "training gradients became non-finite")
        optimizer.step()
        samples += step_samples

        if step not in checkpoint_steps:
            continue
        window = {name: totals[name] / samples for name in AUDIT_TERMS}
        history_record = {
            "optimizer_step": step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_window": window,
            "mean_single_call_amplification": growth_total / samples,
            "spectral_norm": mode_sigma_summary(model),
        }
        history.append(history_record)
        path = checkpoint_directory / f"step_{step:05d}.pt"
        torch.save(
            {
                "version": version,
                "arm": study["arm"],
                "seed": seed,
                "optimizer_step": step,
                "architecture": architecture.to_dict(),
                "contract": str(resolved_contract),
                "contract_sha256": contract_sha,
                "dataset_version": DATASET_VERSION,
                "loss_contract": loss_contract(loss_config),
                "loss_contract_sha256": LOSS_CONTRACT_SHA256,
                "rollout_steps": loss_config.rollout_steps,
                "input_states": 1,
                "static_channels": list(STATIC_FEATURES),
                "from_scratch": True,
                "spectral_normalization": "per_mode_sigma_max_capped_at_one",
                "training_history_record": history_record,
                "model_state_dict": materialized_state_dict(model),
            },
            path,
        )
        checkpoints.append(
            {"optimizer_step": step, "checkpoint": path.name, "checkpoint_sha256": _file_sha256(path)}
        )
        totals = {name: 0.0 for name in AUDIT_TERMS}
        growth_total = 0.0
        samples = 0

    if len(checkpoints) != len(checkpoint_steps):
        raise ResponseTrainingContractError("not every declared checkpoint was written")
    if response_state is not None and response_state["index"] != len(response_state["schedule"]):
        raise ResponseTrainingContractError(
            f"consumed {response_state['index']} of {len(response_state['schedule'])} auxiliary directions"
        )

    records = validation_records()
    growth_starts = [int(validation_starts()[0]), int(validation_starts()[-1])]
    growth_states = []
    for start in growth_starts:
        raw = np.asarray(state[0, start], dtype=np.float32)
        normalized = (raw - point_mean) / point_scale
        normalized[:, ~wet_array] = 0.0
        growth_states.append(torch.from_numpy(np.ascontiguousarray(normalized))[None].to(device))
    growth_static = torch.from_numpy(np.ascontiguousarray(statics[0]))[None].to(device)

    summaries = []
    evaluated_arrays = []
    for record in checkpoints:
        payload = torch.load(checkpoint_directory / record["checkpoint"], map_location=device, weights_only=False)
        probe = build_model(architecture).to(device)
        probe.load_state_dict(payload["model_state_dict"], strict=True)
        probe.eval()
        stepper = ProductionStepper(
            model=probe, device=device, wet=wet_array, mean=point_mean, scale=point_scale, statics=statics
        )
        value = parent_train.validate_checkpoint(
            stepper, state, store_static, records, climatology_state, climatology_derived, wet_array
        )
        value["perturbation_growth"] = growth_rate_summary(probe, growth_states, growth_static, wet)
        evaluated_arrays.append(value.pop("arrays"))
        value["optimizer_step"] = int(record["optimizer_step"])
        summaries.append(value)
        del probe, stepper
        if device.type == "cuda":
            torch.cuda.empty_cache()

    decision = select_by_validation(summaries)
    selected_step = int(decision["selected_optimizer_step"])
    selected_name = next(r["checkpoint"] for r in checkpoints if r["optimizer_step"] == selected_step)
    selected_summary = next(s for s in summaries if int(s["optimizer_step"]) == selected_step)
    shutil.copy2(checkpoint_directory / selected_name, scratch_tmp / "selected.pt")
    published = {
        "optimizer_step": selected_step,
        "checkpoint": str(scratch / "selected.pt"),
        "checkpoint_sha256": _file_sha256(scratch_tmp / "selected.pt"),
        "normalization": str(scratch / "normalization.npz"),
        "normalization_sha256": _file_sha256(normalization_path),
    }

    arrays_path = scratch_tmp / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        optimizer_steps=np.asarray([s["optimizer_step"] for s in summaries], dtype=np.int32),
        lead_days=np.asarray(summaries[0]["lead_days"], dtype=np.int16),
        validation_records=records.astype(np.int32),
        snapshot_codes=snapshot_split,
        pair_codes=pair_split,
        **{
            f"validation_rmse__{method}__{field}": np.stack([a[method][field] for a in evaluated_arrays]).astype(
                np.float32
            )
            for method in ("model", "persistence", "climatology")
            for field in PRIMARY_FIELDS
        },
    )
    report = {
        "status": "complete",
        "version": version,
        "arm": study["arm"],
        "seed": seed,
        "study_contract": study,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": {
            "path": str(dataset),
            "version": DATASET_VERSION,
            "metadata_sha256": _file_sha256(dataset / ".zmetadata"),
        },
        "split": summary_of_split,
        "architecture": architecture.to_dict(),
        "parameter_count": count,
        "initialization": {
            "from_scratch": True,
            "parent_checkpoint": None,
            "load_model_state": False,
            "load_optimizer_state": False,
            "normalization_reused": False,
            "fno_weights": "neuraloperator_default_random_initialization",
            "local_branch_initialization": "zeros",
            "local_branch_bias": False,
            "layer_norm_scale": 1.0,
            "layer_norm_bias": 0.0,
            "seed": seed,
        },
        "normalization": {
            "recomputed_from": "train_only_days_0_5999_of_S0_S1_S2",
            "reused_from_a_previous_run": False,
            "summary": normalizers["summary"],
            "artifact": str(scratch / "normalization.npz"),
            "artifact_sha256": _file_sha256(normalization_path),
        },
        "climatology": {
            "source": "per_regime_pointwise_mean_over_train_only_0_5999",
            "days_per_regime": climatology_days,
        },
        "increment_scale": increment_values.tolist(),
        "loss": contract["loss"],
        "loss_contract": loss_contract(loss_config),
        "loss_contract_sha256": LOSS_CONTRACT_SHA256,
        "response": contract["response"],
        "response_training": (
            {
                "enabled": True,
                "lambda_resp": response_state["lambda_resp"],
                "joint_update_every": response_state["joint_every"],
                "auxiliary_updates": response_state["index"],
                "schedule_sha256": response_state["schedule_sha256"],
                "loss_summary": _response_loss_summary(response_state["log"]),
                "per_update_log": "response_loss_log.jsonl",
                "spectral_buffer_isolation": (
                    "section 15.2: every auxiliary chain snapshots and restores the per-mode "
                    "power vectors bit-for-bit, so the auxiliary path leaves no persistent "
                    "spectral-normalization state behind"
                ),
            }
            if response_state is not None
            else {"enabled": False}
        ),
        "spectral_normalization": spectral_provenance,
        "contraction_penalty": False,
        "temporal_context": {
            "input_states": 1,
            "map": "x_t -> x_t_plus_10",
            "prediction": "direct_state_not_residual",
            "autoregression": "no_teacher_forcing_after_the_initial_state",
        },
        "static_channels": {
            "channels": list(STATIC_FEATURES),
            "external_input_channels": architecture.in_channels,
            "lifting_input_channels": architecture.lifting_in_channels,
            "provenance": static_provenance,
        },
        "optimizer": {
            "name": "adam",
            "initial_learning_rate": float(training["initial_learning_rate"]),
            "adam_betas": [float(v) for v in training["adam_betas"]],
            "weight_decay": float(training["weight_decay"]),
            "batch_size": int(training["batch_size"]),
            "microbatch_size": microbatch,
            "gradient_accumulation_steps": accumulation,
            "gradient_clipping": False,
            "decay_step": decay_step,
            "decay_factor": float(training["decay_factor"]),
            "state_transitions": STATE_TRANSITIONS,
        },
        "counts": {
            "training_rollout_records": len(training_records),
            "training_starts_per_regime": len(training_records) // len(EXPERIMENTS),
            "earliest_training_start": int(min(t for _, t in training_records)),
            "latest_training_start": int(max(t for _, t in training_records)),
            "validation_records": int(records.shape[0]),
            "validation_starts_per_regime": int(validation_starts().size),
        },
        "training_history": history,
        "checkpoints": checkpoints,
        "validation_summaries": summaries,
        "selection_decision": decision,
        "acceptance_gate": acceptance_gate(selected_summary, decision["best_short_auc_10_90"]),
        "published_checkpoint": published,
        "arrays": str(scratch / "arrays.npz"),
        "arrays_sha256": _file_sha256(arrays_path),
        "read_contract": contract["read_contract"],
        "elapsed_seconds": time.monotonic() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if response_state is not None:
        # Written before the report so its hash covers a file already on disk.
        # The report keeps only the summary; the full 1,920-entry series lives
        # here, so a C run's response trajectory stays inspectable without
        # re-training it.
        log_lines = "".join(
            json.dumps(json_safe(record), sort_keys=True) + "\n" for record in response_state["log"]
        )
        (scratch_tmp / RESPONSE_LOG_NAME).write_text(log_lines)
        (project_tmp / RESPONSE_LOG_NAME).write_text(log_lines)

    report = json_safe(report)
    report["content_sha256"] = _json_sha256(report)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (scratch_tmp / "report.json").write_text(rendered)
    (project_tmp / "report.json").write_text(rendered)
    shutil.copy2(arrays_path, project_tmp / "arrays.npz")
    _plot(project_tmp / "selection.png", summaries, selected_step)
    (project_tmp / README_NAME).write_text(_readme(report))
    artifact_names = ["report.json", "arrays.npz", "selection.png", README_NAME]
    if response_state is not None:
        artifact_names.append(RESPONSE_LOG_NAME)
    artifacts = {name: _file_sha256(project_tmp / name) for name in artifact_names}
    (project_tmp / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "status": "complete",
                "version": version,
                "arm": study["arm"],
                "seed": seed,
                "contract_sha256": contract_sha,
                "artifacts": artifacts,
                "content_sha256": _json_sha256(artifacts),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(scratch_tmp, scratch)
    os.replace(project_tmp, project)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--contract", required=True)
    preflight_parser.add_argument("--seed", type=int, default=None)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--contract", required=True)
    run_parser.add_argument(
        "--steps-override", type=int, default=None,
        help="Smoke-test only, and only together with --output-suffix: run fewer than the "
             "contract's optimizer steps. Never use for a published run.",
    )
    run_parser.add_argument(
        "--output-suffix", default=None,
        help="Smoke-test only: append this to both output directory names so an abbreviated "
             "run cannot occupy the publishable path.",
    )
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.contract, seed_override=args.seed)
    else:
        result = run(
            args.contract,
            device_name=args.device,
            seed_override=args.seed,
            steps_override=args.steps_override,
            output_suffix=args.output_suffix,
        )
    print(json.dumps(json_safe(result), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
