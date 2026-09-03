"""S0 Figures 3--8 for the production emulator, plus the 2,000-day acceptance gate.

Held evaluation only: no training, no checkpoint selection, no promotion. The
selected checkpoint is autoregressed from 15 fixed inference starts to day 2,000
at ten-day steps and reduced by the frozen plot definitions in
:mod:`turbfno.plots`:

    figure 3  streamfunction structure at days 0--40
    figure 4  RMSE 0--200 days
    figure 5  single-member RMSE
    figure 6  ACC 0--200 days
    figure 7  streamfunction at day 60 and day 2,000
    figure 8  RMSE 0--2,000 days

There is one model in every plate. The production arm has no predecessor, so the
suite carries no comparator series.

Entry points::

    python -m turbfno.figures finalize  --contract config/...json
    python -m turbfno.figures preflight --contract config/...json
    python -m turbfno.figures run       --contract config/...json [--device cuda]
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

from . import plots
from .runtime import _device, _file_sha256, _json_sha256, json_safe, torch
from .dataset import (
    DATASET_VERSION,
    INFERENCE_RANGE,
    INFERENCE_START_RANGE,
    MAXIMUM_INFERENCE_ROLLOUT_DAYS,
    STATIC_FEATURES,
    TRAIN_RANGE,
    VALIDATION_RANGE,
    assert_model_visible,
    assert_truth_available,
    inference_starts,
)
from .diagnostics import _member_acc, _member_rmse, derived_fields
from .model import ProductionArchitecture, ProductionStepper, build_model
from .validation import _gather, train_only_climatology
from .train import (
    CHECKPOINT_STEPS,
    LOSS_CONTRACT_SHA256,
    NORMALIZATION_NAME as SELECTED_NORMALIZATION_NAME,
    REPORT_NAME as TRAINING_REPORT_NAME,
    ROLLOUT_STEPS,
    VERSION as TRAINING_VERSION,
    physical_static_block,
)

VERSION = "turb_forward_control_v1_s0_turb_figures_v1"

CONTRACT_STATUS = (
    "frozen_after_the_production_training_and_validation_and_before_any_"
    "inference_metric"
)

MEMBER_COUNT = 15

START_SEED = 20260802

REGIMES = ("S0_turb",)

REGIME_INDEX = {"S0_turb": 0}

TAU0_N_M2 = {"S0_turb": 0.1}

LEAD_DAYS = plots.LEAD_DAYS

SHORT_LEAD_DAYS = plots.SHORT_LEAD_DAYS

RMSE_FIELDS = plots.RMSE_FIELDS

ACC_FIELDS = plots.ACC_FIELDS

PENDING = "PENDING_AFTER_TRAINING"

PENDING_PATHS: tuple[tuple[str, ...], ...] = (
    ("selected_model", "optimizer_step"),
    ("artifacts", "selected_checkpoint", "sha256"),
    ("artifacts", "selected_normalization", "sha256"),
    ("artifacts", "selected_report", "sha256"),
)

MAXIMUM_NORMALIZED_MAGNITUDE = 8.0

MINIMUM_STREAMFUNCTION_SV = -33.0

DAY2000_STD_RATIO_RANGE = (0.80, 1.25)

GATE_NAME = f"{TRAINING_VERSION}_acceptance_gate.json"

_EXPECTED_OUTPUTS = (
    *plots.FIGURE_NAMES,
    plots.REPORT_NAME,
    plots.ARRAYS_NAME,
    plots.SUMMARY_NAME,
    plots.CSV_NAME,
    plots.MANIFEST_NAME,
    plots.README_NAME,
)

_EXPECTED_BASELINES = {
    "climatology": "pointwise_S0_turb_mean_over_the_training_block_0_5999_only",
    "ensemble_summary": "mean_and_10th_90th_percentiles_across_15_member_metrics",
    "persistence": "each_members_initial_physical_field_held_fixed_at_every_lead",
    "rmse": "square_root_of_mean_squared_error_over_wet_cells_for_each_member",
}

_EXPECTED_TRUTH = {
    "continuation_required": False,
    "lead_matched_to_day": 2000,
    "source": "trajectories_turb_v1_store",
}

_REQUIRED_ARTIFACTS = (
    "dataset_metadata",
    "selected_checkpoint",
    "selected_normalization",
    "selected_report",
    "mitgcm_zonal_spacing",
    "mitgcm_sst_relaxation",
    "mitgcm_declaration",
)

_REQUIRED_SOURCE_HASHES = frozenset(
    {
        "src/turbfno/dataset.py",
        "src/turbfno/diagnostics.py",
        "src/turbfno/figures.py",
        "src/turbfno/model.py",
        "src/turbfno/plots.py",
        "src/turbfno/runtime.py",
        "src/turbfno/train.py",
        "src/turbfno/perturbation_growth.py",
        "src/turbfno/spectral_norm.py",
        "src/turbfno/validation.py",
    }
)


class FigureContractError(RuntimeError):
    """Raised when the held-evaluation contract is violated."""


def _integer(value: Any, fallback: int = -1) -> int:
    try:
        if isinstance(value, bool):
            return fallback
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _read(contract: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = contract
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def declared_inference_starts() -> np.ndarray:
    """15 members from the inference set, fixed by the declared seed."""

    starts = inference_starts(MEMBER_COUNT, START_SEED)
    # The starts are what the model is handed; truth may run past the record.
    assert_model_visible(starts, "inference starts")
    assert_truth_available(
        starts + MAXIMUM_INFERENCE_ROLLOUT_DAYS, "day-2,000 lead-matched truth"
    )
    return starts


def unfilled_fields(contract: Mapping[str, Any]) -> list[str]:
    """Declared-pending contract fields that training has not yet supplied."""

    return [
        ".".join(path) for path in PENDING_PATHS if _read(contract, path) in (None, PENDING)
    ]


def _training_provenance(contract: Mapping[str, Any]) -> None:
    """Bind the figure declaration to the completed production training report."""

    selected = contract["selected_model"]
    artifacts = contract["artifacts"]
    report_path = Path(str(artifacts["selected_report"].get("path", ""))).resolve()
    if report_path.name != TRAINING_REPORT_NAME or not report_path.is_file():
        raise FigureContractError(
            "the selected report is not the completed production training report"
        )
    if _file_sha256(report_path) != artifacts["selected_report"].get("sha256"):
        raise FigureContractError("the selected training report hash changed")
    report = json.loads(report_path.read_text())
    published = report.get("published_checkpoint", {})
    checkpoint = artifacts["selected_checkpoint"]
    normalization = artifacts["selected_normalization"]
    if (
        report.get("status") != "complete"
        or report.get("version") != TRAINING_VERSION
        or report.get("architecture") != selected.get("architecture")
        or report.get("loss_contract_sha256") != LOSS_CONTRACT_SHA256
        or report.get("initialization", {}).get("from_scratch") is not True
        or _integer(published.get("optimizer_step"))
        != _integer(selected.get("optimizer_step"))
        or published.get("checkpoint") != checkpoint.get("path")
        or published.get("checkpoint_sha256") != checkpoint.get("sha256")
        or published.get("normalization") != normalization.get("path")
        or published.get("normalization_sha256") != normalization.get("sha256")
        or Path(str(normalization.get("path", ""))).name != SELECTED_NORMALIZATION_NAME
    ):
        raise FigureContractError(
            "the selected model disagrees with its production training report"
        )
    training_path = Path(str(selected.get("training_contract", ""))).resolve()
    if not training_path.is_file():
        raise FigureContractError("the selected training contract is absent")
    training = json.loads(training_path.read_text())
    if (
        training.get("version") != TRAINING_VERSION
        or training.get("architecture") != selected.get("architecture")
        or training.get("initialization", {}).get("from_scratch") is not True
    ):
        raise FigureContractError("the training provenance changed")


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load and strictly audit the S0 held-evaluation declaration."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    pending = unfilled_fields(contract)
    if pending:
        raise FigureContractError(
            "the figure contract still carries post-training fields: "
            + ", ".join(pending)
            + " -- run `python -m turbfno.figures finalize` first"
        )

    protocol = contract.get("protocol", {})
    selected = contract.get("selected_model", {})
    output = contract.get("output", {})
    dataset = contract.get("dataset", {})
    expected_starts = tuple(int(value) for value in declared_inference_starts())
    protocol_ok = (
        _integer(protocol.get("member_count")) == MEMBER_COUNT
        and _integer(protocol.get("start_seed")) == START_SEED
        and tuple(protocol.get("start_draw_order", ())) == expected_starts
        and tuple(protocol.get("regimes", ())) == REGIMES
        and protocol.get("primary_regime") == "S0_turb"
        and tuple(protocol.get("figure_names", ())) == tuple(plots.FIGURE_NAMES)
        and tuple(protocol.get("figure3_lead_days", ())) == tuple(plots.FIGURE_3_LEADS)
        and tuple(protocol.get("figure7_lead_days", ())) == tuple(plots.FIGURE_7_LEADS)
        and tuple(protocol.get("rmse_fields", ())) == tuple(plots.RMSE_FIELDS)
        and tuple(protocol.get("acc_fields", ())) == tuple(plots.ACC_FIELDS)
        and tuple(protocol.get("inference_set", ())) == INFERENCE_RANGE
        and tuple(protocol.get("start_window", ())) == INFERENCE_START_RANGE
        and _integer(protocol.get("maximum_lead_days")) == MAXIMUM_INFERENCE_ROLLOUT_DAYS
        and _integer(protocol.get("prediction_interval_days")) == 10
        and protocol.get("short_lead_days") == "0_to_200_inclusive_by_10"
        and protocol.get("long_lead_days") == "0_to_2000_inclusive_by_10"
        and protocol.get("comparator_model") is None
        and protocol.get("nesting")
        == "nested_validation_inference_protocol_no_independent_third_test_split"
        and tuple(protocol.get("static_channels", ())) == STATIC_FEATURES
    )
    models_ok = (
        selected.get("version") == TRAINING_VERSION
        and _integer(selected.get("optimizer_step")) in CHECKPOINT_STEPS
        and _integer(selected.get("rollout_steps")) == ROLLOUT_STEPS
        and selected.get("loss_contract_sha256") == LOSS_CONTRACT_SHA256
        and selected.get("architecture") == ProductionArchitecture().to_dict()
        and selected.get("from_scratch") is True
    )
    output_ok = (
        str(output.get("project_root", "")).endswith(VERSION)
        and str(output.get("scratch_root", "")).endswith(VERSION)
        and output.get("overwrite") is False
        and output.get("one_folder_per_regime") is True
        and tuple(output.get("required", ())) == _EXPECTED_OUTPUTS
    )
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or "comparator_model" in contract
        or "figure6" in contract
        or dataset.get("version") != DATASET_VERSION
        or tuple(dataset.get("train", ())) != TRAIN_RANGE
        or tuple(dataset.get("validation", ())) != VALIDATION_RANGE
        or tuple(dataset.get("inference", ())) != INFERENCE_RANGE
        or dataset.get("tau0_n_m2") != TAU0_N_M2
        or contract.get("baselines") != _EXPECTED_BASELINES
        or contract.get("truth") != _EXPECTED_TRUTH
        or not set(_REQUIRED_ARTIFACTS).issubset(contract.get("artifacts", {}))
        or not protocol_ok
        or not models_ok
        or not output_ok
    ):
        raise FigureContractError("the production S0 figure contract changed")
    try:
        ProductionArchitecture(**selected["architecture"])
        _training_provenance(contract)
    except FigureContractError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise FigureContractError("the selected figure provenance changed") from error
    if verify_sources:
        hashes = contract.get("source_hashes", {})
        if not _REQUIRED_SOURCE_HASHES.issubset(hashes):
            raise FigureContractError("the figure source declaration is incomplete")
        for label, specification in contract.get("artifacts", {}).items():
            plots._verify_file(specification, label)
        root = resolved.parents[1]
        for relative, expected in hashes.items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise FigureContractError(f"a pinned figure source changed: {relative}")
    return contract, resolved, _file_sha256(resolved)


def finalize(contract_path: str | Path) -> dict[str, Any]:
    """Fill the declared-pending fields from the training run's own report.

    Idempotent: a field already equal to what the report says is left alone, and
    a field that disagrees is refused rather than overwritten, so this can be run
    before every figure job without becoming a way to edit a frozen contract.
    """

    resolved = Path(contract_path).resolve()
    contract = json.loads(resolved.read_text())
    report_path = Path(contract["artifacts"]["selected_report"]["path"])
    if not report_path.is_file():
        raise FigureContractError(f"the training report is not on disk yet: {report_path}")
    if report_path.name != TRAINING_REPORT_NAME:
        raise FigureContractError(f"the declared report is not {TRAINING_REPORT_NAME}")
    report = json.loads(report_path.read_text())
    if report.get("version") != TRAINING_VERSION or report.get("status") != "complete":
        raise FigureContractError("the report is not this arm's completed training report")
    published = report["published_checkpoint"]
    resolutions = {
        ("selected_model", "optimizer_step"): int(published["optimizer_step"]),
        ("artifacts", "selected_checkpoint", "sha256"): str(
            published["checkpoint_sha256"]
        ),
        ("artifacts", "selected_normalization", "sha256"): str(
            published["normalization_sha256"]
        ),
        ("artifacts", "selected_report", "sha256"): _file_sha256(report_path),
    }
    applied: dict[str, Any] = {}
    for path, value in resolutions.items():
        current = _read(contract, path)
        if current not in (None, PENDING) and current != value:
            raise FigureContractError(
                f"{'.'.join(path)} is already {current!r}, not {value!r}; "
                "refusing to overwrite a filled contract field"
            )
        node = contract
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        if current != value:
            applied[".".join(path)] = value
    for key, declared in (
        ("selected_checkpoint", published["checkpoint"]),
        ("selected_normalization", published["normalization"]),
    ):
        if contract["artifacts"][key]["path"] != declared:
            raise FigureContractError(
                f"{key} path disagrees with the training report: "
                f"{contract['artifacts'][key]['path']} vs {declared}"
            )
    if applied:
        resolved.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return {
        "status": "filled" if applied else "already_complete",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": _file_sha256(resolved),
        "applied": applied,
        "selected_optimizer_step": int(published["optimizer_step"]),
    }


def _stepper(
    contract: Mapping[str, Any],
    device: Any,
    wet: np.ndarray,
    statics: np.ndarray,
) -> ProductionStepper:
    """Build the selected model after checking its recorded identity."""

    if torch is None:  # pragma: no cover - environment dependent
        raise RuntimeError("figure evaluation requires PyTorch")
    declared = contract["selected_model"]
    architecture = ProductionArchitecture(**declared["architecture"])
    payload = torch.load(
        Path(contract["artifacts"]["selected_checkpoint"]["path"]),
        map_location=device,
        weights_only=False,
    )
    if (
        payload.get("version") != TRAINING_VERSION
        or payload.get("architecture") != declared["architecture"]
        or _integer(payload.get("optimizer_step")) != _integer(declared.get("optimizer_step"))
        or payload.get("dataset_version") != DATASET_VERSION
        or payload.get("loss_contract_sha256") != declared["loss_contract_sha256"]
        or _integer(payload.get("rollout_steps")) != _integer(declared.get("rollout_steps"))
        or payload.get("from_scratch") is not True
    ):
        raise FigureContractError(
            "the selected checkpoint's identity, architecture, dataset or objective changed"
        )
    try:
        model = build_model(architecture).to(device)
        incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise FigureContractError("the selected state dictionary changed") from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise FigureContractError("the selected checkpoint did not load strictly")
    model.eval()
    with np.load(
        Path(contract["artifacts"]["selected_normalization"]["path"])
    ) as artifact:
        mean = np.asarray(artifact["pointwise_mean"], dtype=np.float32)
        scale = np.asarray(artifact["pointwise_scale"], dtype=np.float32)
    return ProductionStepper(
        model=model, device=device, wet=wet, mean=mean, scale=scale, statics=statics
    )


def _fields(states: np.ndarray, wet: np.ndarray) -> dict[str, np.ndarray]:
    result = derived_fields(states, wet)
    result["surface_u"] = np.asarray(states[:, 0], dtype=np.float32)
    result["surface_v"] = np.asarray(states[:, 15], dtype=np.float32)
    return result


def evaluate_regime(
    stepper: ProductionStepper,
    state: Any,
    static: Any,
    regime_index: int,
    starts: np.ndarray,
    climatology_state: np.ndarray,
    climatology_derived: Mapping[str, np.ndarray],
    wet: np.ndarray,
) -> dict[str, np.ndarray]:
    """Roll the selected checkpoint to day 2,000 and reduce as the frozen suite."""

    records = np.stack(
        [np.full(starts.shape, regime_index, dtype=np.int64), starts], axis=1
    )
    initial = _gather(state, records, 0)
    experiments = records[:, 0]
    current = stepper.normalized_state(initial)
    forcing = stepper.normalized_static(static, experiments)
    initial_fields = _fields(initial, wet)
    climate = np.repeat(climatology_state[regime_index][None], starts.size, axis=0)
    climate_fields = _fields(climate, wet)
    for name, value in climatology_derived.items():
        climate_fields[name] = np.repeat(value[regime_index][None], starts.size, axis=0)

    arrays: dict[str, np.ndarray] = {
        "lead_days": np.asarray(LEAD_DAYS, dtype=np.int16),
        "short_lead_days": np.asarray(SHORT_LEAD_DAYS, dtype=np.int16),
        "start_draw_order": starts.astype(np.int32),
        "finite": np.empty((starts.size, len(LEAD_DAYS)), dtype=np.uint8),
        "normalized_max_abs": np.empty((starts.size, len(LEAD_DAYS)), dtype=np.float32),
    }
    for method in plots.METHODS:
        for field in RMSE_FIELDS:
            arrays[f"rmse__{method}__{field}"] = np.empty(
                (starts.size, len(LEAD_DAYS)), dtype=np.float32
            )
    for field in ACC_FIELDS:
        arrays[f"acc__model__{field}"] = np.empty(
            (starts.size, len(SHORT_LEAD_DAYS)), dtype=np.float32
        )
    arrays["single_rmse__streamfunction"] = np.empty(
        len(SHORT_LEAD_DAYS), dtype=np.float32
    )
    arrays["single_rmse__sst"] = np.empty_like(arrays["single_rmse__streamfunction"])
    for tag, leads in (("figure3", plots.FIGURE_3_LEADS), ("figure7", plots.FIGURE_7_LEADS)):
        arrays[f"{tag}_truth_streamfunction"] = np.empty(
            (len(leads), *wet.shape), dtype=np.float32
        )
        arrays[f"{tag}_model_streamfunction"] = np.empty_like(
            arrays[f"{tag}_truth_streamfunction"]
        )
    figure3 = {lead: index for index, lead in enumerate(plots.FIGURE_3_LEADS)}
    figure7 = {lead: index for index, lead in enumerate(plots.FIGURE_7_LEADS)}
    wet_tensor = torch.from_numpy(wet).to(stepper.device)

    with torch.no_grad():
        for lead_index, lead in enumerate(LEAD_DAYS):
            if lead:
                current = stepper.step(current, forcing)
                prediction = stepper.physical(current)
            else:
                prediction = initial.copy()
            truth = _gather(state, records, lead)
            truth_fields = _fields(truth, wet)
            predicted_fields = _fields(prediction, wet)
            for field in RMSE_FIELDS:
                arrays[f"rmse__model__{field}"][:, lead_index] = _member_rmse(
                    predicted_fields[field], truth_fields[field], wet
                )
                arrays[f"rmse__persistence__{field}"][:, lead_index] = _member_rmse(
                    initial_fields[field], truth_fields[field], wet
                )
                arrays[f"rmse__climatology__{field}"][:, lead_index] = _member_rmse(
                    climate_fields[field], truth_fields[field], wet
                )
            arrays["finite"][:, lead_index] = np.isfinite(prediction).all(axis=(1, 2, 3))
            arrays["normalized_max_abs"][:, lead_index] = (
                torch.amax(torch.abs(current[:, :, wet_tensor]), dim=(1, 2))
                .detach()
                .cpu()
                .numpy()
            )
            if lead <= 200:
                short_index = lead // 10
                for field in ACC_FIELDS:
                    arrays[f"acc__model__{field}"][:, short_index] = _member_acc(
                        predicted_fields[field],
                        truth_fields[field],
                        climate_fields[field],
                        wet,
                    )
                arrays["single_rmse__streamfunction"][short_index] = _member_rmse(
                    predicted_fields["streamfunction"], truth_fields["streamfunction"], wet
                )[0]
                arrays["single_rmse__sst"][short_index] = _member_rmse(
                    predicted_fields["sst"], truth_fields["sst"], wet
                )[0]
            if lead in figure3:
                arrays["figure3_truth_streamfunction"][figure3[lead]] = truth_fields[
                    "streamfunction"
                ][0]
                arrays["figure3_model_streamfunction"][figure3[lead]] = predicted_fields[
                    "streamfunction"
                ][0]
            if lead in figure7:
                arrays["figure7_truth_streamfunction"][figure7[lead]] = truth_fields[
                    "streamfunction"
                ][0]
                arrays["figure7_model_streamfunction"][figure7[lead]] = predicted_fields[
                    "streamfunction"
                ][0]
    return arrays


def long_rollout_gate(
    arrays: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """The 2,000-day half of the acceptance gate, from the published arrays."""

    wet = np.asarray(arrays["wet_mask"], dtype=bool)
    day2000 = plots.FIGURE_7_LEADS.index(2000)
    model = np.asarray(arrays["figure7_model_streamfunction"], dtype=np.float64)[day2000][wet]
    truth = np.asarray(arrays["figure7_truth_streamfunction"], dtype=np.float64)[day2000][wet]
    finite = bool(summary["all_selected_states_finite"])
    magnitude = float(summary["maximum_selected_normalized_abs"])
    minimum = float(model.min())
    truth_std = float(truth.std())
    ratio = float(model.std() / truth_std) if truth_std > 0.0 else float("inf")
    low, high = DAY2000_STD_RATIO_RANGE
    conditions = {
        "all_values_finite": finite,
        "maximum_normalized_magnitude_at_most_8": magnitude <= MAXIMUM_NORMALIZED_MAGNITUDE,
        "streamfunction_minimum_at_least_minus_33_sv": minimum >= MINIMUM_STREAMFUNCTION_SV,
        "day2000_spatial_std_ratio_in_range": low <= ratio <= high,
    }
    collapse = {
        field: float(summary["rmse"][field]["model"]["day2000_mean"])
        / float(summary["rmse"][field]["climatology"]["day2000_mean"])
        for field in plots.RMSE_FIELDS
    }
    return {
        "measured": {
            "all_values_finite": finite,
            "maximum_normalized_magnitude": magnitude,
            "day2000_streamfunction_minimum_sv": minimum,
            "day2000_streamfunction_maximum_sv": float(model.max()),
            "day2000_truth_streamfunction_minimum_sv": float(truth.min()),
            "day2000_spatial_std_ratio_to_truth": ratio,
        },
        "thresholds": {
            "maximum_normalized_magnitude": MAXIMUM_NORMALIZED_MAGNITUDE,
            "streamfunction_minimum_sv": MINIMUM_STREAMFUNCTION_SV,
            "day2000_spatial_std_ratio": list(DAY2000_STD_RATIO_RANGE),
        },
        "conditions": conditions,
        "long_rollout_conditions_pass": all(conditions.values()),
        "advisory_day2000_rmse_ratio_to_climatology": collapse,
        "advisory_note": (
            "a ratio at or near 1.0 means the day-2,000 field is indistinguishable "
            "from climatology; the training gate's long window integrates 90-360 "
            "days only, so this number is reported here and is deliberately not "
            "gated in either half"
        ),
        "streamfunction_basis": (
            "member 0's day-2,000 barotropic streamfunction, the field figure 7 "
            "publishes and the visual criterion inspects"
        ),
        "visual_criterion": "by_inspection_of_figures_3_and_7",
    }


def acceptance_gate(contract: Mapping[str, Any], regime: str = "S0_turb") -> dict[str, Any]:
    """Join the training arm's validation half with this package's 2,000-day half."""

    output = Path(contract["output"]["project_root"]).resolve() / regime
    with np.load(output / plots.ARRAYS_NAME) as stored:
        arrays = {name: stored[name] for name in stored.files}
    summary = json.loads((output / plots.SUMMARY_NAME).read_text())
    training = json.loads(Path(contract["artifacts"]["selected_report"]["path"]).read_text())
    validation = training["acceptance_gate"]
    long_half = long_rollout_gate(arrays, summary)
    gate = {
        "version": VERSION,
        "regime": regime,
        "selected_optimizer_step": int(contract["selected_model"]["optimizer_step"]),
        "validation_half": validation,
        "long_rollout_half": long_half,
        "measurable_conditions_pass": bool(
            validation["validation_conditions_pass"]
            and long_half["long_rollout_conditions_pass"]
        ),
        "remaining_by_inspection": (
            "western boundary sharp, gyre structure identifiable, no basin-wide "
            "smoothing -- figures 3 and 7"
        ),
        "decision_note": (
            "these measurable diagnostics and visual inspection describe the "
            "production model's held performance; this package promotes nothing"
        ),
    }
    gate = json_safe(gate)
    gate["content_sha256"] = _json_sha256(gate)
    return gate


def _readme(regime: str, report: Mapping[str, Any]) -> str:
    starts = declared_inference_starts()
    selected = int(report["selected_optimizer_step"])
    return f"""# Production emulator, {regime}: Figures 3--8

This held package evaluates the selected step-{selected:,} checkpoint of
`{TRAINING_VERSION}` on the 15-member {regime} inference protocol.

The model is the one-input production operator: `F(x_t, S) -> x_(t+10)` with
`x_t` the 46-channel state at one time level and `S` the five physical static
channels `[tau_x, wet, f, dx, theta_clim]`; 32 x 32 Fourier modes, width 128,
three blocks, a bias-free local 3 x 3 path and the deterministic sine/cosine
position encoder. It was trained once from random initialization, so there is no
predecessor to compare against and every plate carries a single model curve
against persistence and climatology.

Starts span {int(starts.min())}--{int(starts.max())} and are drawn from the
inference block {list(INFERENCE_RANGE)} nested inside validation
{list(VALIDATION_RANGE)}; there is no independent third test split. Every member
has lead-matched MITgcm truth through day 2,000.

The measurable gate is written beside the {regime} folder as `{GATE_NAME}`. This
package performs no training, selection, or checkpoint promotion.

Report content SHA-256: `{report['report_content_sha256']}`.
"""


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    """Publish the six figures for the S0 inference protocol."""

    if torch is None:  # pragma: no cover - environment dependent
        raise RuntimeError("figure evaluation requires PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    device = _device(device_name)
    dataset = Path(contract["dataset"]["path"]).resolve()
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    longitude = np.asarray(group["longitude_deg"][:], dtype=np.float32)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float32)
    starts = declared_inference_starts()
    climatology_state, climatology_derived, climatology_days = train_only_climatology(
        state, wet
    )
    if climatology_days != TRAIN_RANGE[1] - TRAIN_RANGE[0]:
        raise FigureContractError("the train-only climatology did not cover 0--5999")

    # Three of the five statics are derived from the simulation's own inputs
    # rather than the store, so the block is built once from the published
    # normalizers and shared.
    with np.load(
        Path(contract["artifacts"]["selected_normalization"]["path"])
    ) as stored:
        point_mean = np.asarray(stored["pointwise_mean"], dtype=np.float32)
        point_scale = np.asarray(stored["pointwise_scale"], dtype=np.float32)
    statics, static_provenance = physical_static_block(
        contract["artifacts"], group, point_mean, point_scale
    )
    stepper = _stepper(contract, device, wet, statics)

    published: dict[str, Any] = {}
    plots._style()
    for regime in REGIMES:
        regime_index = REGIME_INDEX[regime]
        output = Path(contract["output"]["project_root"]).resolve() / regime
        scratch = Path(contract["output"]["scratch_root"]).resolve() / regime
        for path in (output, scratch):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
        arrays = evaluate_regime(
            stepper,
            state,
            static,
            regime_index,
            starts,
            climatology_state,
            climatology_derived,
            wet,
        )
        arrays["longitude_deg"] = longitude
        arrays["latitude_deg"] = latitude
        arrays["wet_mask"] = wet.astype(np.uint8)
        summary = plots._summary(arrays)
        output_tmp = output.with_name(output.name + ".tmp")
        scratch_tmp = scratch.with_name(scratch.name + ".tmp")
        output_tmp.parent.mkdir(parents=True, exist_ok=True)
        scratch_tmp.parent.mkdir(parents=True, exist_ok=True)
        output_tmp.mkdir()
        scratch_tmp.mkdir()
        try:
            scratch_arrays = scratch_tmp / plots.ARRAYS_NAME
            np.savez_compressed(scratch_arrays, **arrays)
            plots._plot_streamfunction_grid(output_tmp, arrays, longitude, latitude, wet)
            plots._plot_rmse(output_tmp, arrays, long=False)
            plots._plot_single_member(output_tmp, arrays)
            plots._plot_acc(output_tmp, arrays)
            plots._plot_day60_day2000(output_tmp, arrays, longitude, latitude, wet)
            plots._plot_rmse(output_tmp, arrays, long=True)
            plots._write_csv(output_tmp / plots.CSV_NAME, arrays)
            (output_tmp / plots.SUMMARY_NAME).write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
            report = {
                "version": VERSION,
                "status": "complete",
                "regime": regime,
                "tau0_n_m2": contract["dataset"]["tau0_n_m2"][regime],
                "role": "primary",
                "contract": str(resolved_contract),
                "contract_sha256": contract_sha,
                "dataset": str(dataset),
                "dataset_version": DATASET_VERSION,
                "selected_optimizer_step": int(
                    contract["selected_model"]["optimizer_step"]
                ),
                "comparator_model": None,
                "start_draw_order": starts.astype(int).tolist(),
                "inference_start_range": list(INFERENCE_START_RANGE),
                "summary": summary,
                "arrays": str(scratch / plots.ARRAYS_NAME),
                "arrays_sha256": _file_sha256(scratch_arrays),
                "figures": list(plots.FIGURE_NAMES),
                "static_channels": list(STATIC_FEATURES),
                "static_channel_provenance": static_provenance,
                "elapsed_seconds": time.monotonic() - started,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
            report = json_safe(report)
            report["report_content_sha256"] = _json_sha256(report)
            (scratch_tmp / plots.REPORT_NAME).write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
            shutil.copy2(scratch_tmp / plots.REPORT_NAME, output_tmp / plots.REPORT_NAME)
            shutil.copy2(scratch_arrays, output_tmp / plots.ARRAYS_NAME)
            (output_tmp / plots.README_NAME).write_text(_readme(regime, report))
            manifest = {
                "version": VERSION,
                "regime": regime,
                "contract_sha256": contract_sha,
                "report_content_sha256": report["report_content_sha256"],
                "artifacts": {
                    path.name: {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}
                    for path in sorted(output_tmp.iterdir())
                    if path.is_file()
                },
            }
            manifest["manifest_content_sha256"] = _json_sha256(manifest)
            (output_tmp / plots.MANIFEST_NAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            scratch_tmp.replace(scratch)
            output_tmp.replace(output)
        except Exception:
            shutil.rmtree(scratch_tmp, ignore_errors=True)
            shutil.rmtree(output_tmp, ignore_errors=True)
            raise
        published[regime] = report
    return published


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify sources, starts and the checkpoint identity without plotting."""

    contract, resolved, digest = load_contract(contract_path)
    starts = declared_inference_starts()
    return {
        "status": "pass",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset_version": DATASET_VERSION,
        "regimes": list(REGIMES),
        "primary_regime": "S0_turb",
        "member_count": MEMBER_COUNT,
        "start_draw_order": starts.astype(int).tolist(),
        "inference_start_range": list(INFERENCE_START_RANGE),
        "maximum_lead_days": max(LEAD_DAYS),
        "selected_optimizer_step": int(contract["selected_model"]["optimizer_step"]),
        "comparator_model": None,
        "static_channels": list(STATIC_FEATURES),
        "selected_rollout_steps": ROLLOUT_STEPS,
        "selected_loss_contract_sha256": LOSS_CONTRACT_SHA256,
        "acceptance_gate_artifact": str(
            Path(contract["output"]["project_root"]).resolve() / GATE_NAME
        ),
        "continuation_required": False,
    }


def publish(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    """Publish the six figures, then evaluate the 2,000-day acceptance gate."""

    published = dict(run(contract_path, device_name=device_name))
    contract, _, _ = load_contract(contract_path, verify_sources=False)
    root = Path(contract["output"]["project_root"]).resolve()
    gate = acceptance_gate(contract, "S0_turb")
    # Beside the regime folder, never inside it: that package's manifest is
    # already sealed over its own file list.
    (root / GATE_NAME).write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    published["acceptance_gate"] = gate
    return published


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("finalize", "preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        if command == "run":
            child.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "finalize":
        result: Any = finalize(args.contract)
    elif args.command == "preflight":
        result = preflight(args.contract)
    else:
        result = publish(args.contract, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
