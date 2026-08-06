"""S0 figure suite and acceptance gate for the six-step rollout fine-tune.

Publishes Bire figures 3-8 on the S0 inference set for the fine-tuned
checkpoint, then evaluates the 2,000-day half of the final acceptance gate.

Figure 6 is a literal pre-train / fine-tune pair: black is the step-15,360
checkpoint the arm started from, red is the selected fine-tuned one. The two
carry different objective hashes -- v1 over three steps against the six-step
contract -- so :func:`_stepper` verifies each against the hash its own contract
block declares, rather than against one shared constant.

Held-evaluation only: no training, no checkpoint selection, no promotion.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .train import REPORT_NAME as TRAINING_REPORT_NAME

import numpy as np
import zarr

from .runtime import torch
from . import plots
from .runtime import _device, _file_sha256, _json_sha256
from .dataset import DATASET_VERSION, INFERENCE_START_RANGE, MAXIMUM_INFERENCE_ROLLOUT_DAYS, TRAIN_RANGE, _normalizers, assert_model_visible, assert_truth_available, inference_starts
from .diagnostics import _member_acc, _member_rmse, derived_fields
from .objective import MODEL_C_LOSS_V1_CONTRACT_SHA256
from .model import BireAlignedArchitecture, BireAlignedStepper, MANIFEST_NAME, README_NAME, build_bire_aligned_model
from .validation import _gather, _plot, train_only_climatology
from .plots import ARRAYS_NAME, CSV_NAME, FIGURE_3_LEADS, FIGURE_7_LEADS, FIGURE_NAMES, METHODS, METHOD_LABELS, REPORT_NAME, SUMMARY_NAME, _plot_acc, _plot_day60_day2000, _plot_rmse, _plot_single_member, _plot_streamfunction_grid, _style, _summary, _verify_file, _write_csv
from .train import BASELINE_OPTIMIZER_STEP, CHECKPOINT_STEPS, FINE_TUNE_LOSS_CONTRACT_SHA256, ROLLOUT_STEPS

MEMBER_COUNT = 15

START_SEED = 20260802

LEAD_DAYS = plots.LEAD_DAYS

SHORT_LEAD_DAYS = plots.SHORT_LEAD_DAYS

RMSE_FIELDS = plots.RMSE_FIELDS

ACC_FIELDS = plots.ACC_FIELDS

class BireProtocolFigureError(RuntimeError):
    """Raised when the Bire-protocol figure contract is violated."""

REGIMES = ("S0",)

REGIME_INDEX = {"S0": 0}

def declared_inference_starts() -> np.ndarray:
    """15 members from the inference set, fixed by the declared seed."""

    starts = inference_starts(MEMBER_COUNT, START_SEED)
    # The starts are what the model is handed; truth may run past the record.
    assert_model_visible(starts, "inference starts")
    assert_truth_available(
        starts + MAXIMUM_INFERENCE_ROLLOUT_DAYS, "day-2,000 lead-matched truth"
    )
    return starts

REGIME_WIND_LABEL = {"S0": "Control wind", "S1": "Low wind", "S2": "High wind"}

def _fields(states: np.ndarray, wet: np.ndarray) -> dict[str, np.ndarray]:
    result = derived_fields(states, wet)
    result["surface_u"] = np.asarray(states[:, 0], dtype=np.float32)
    result["surface_v"] = np.asarray(states[:, 15], dtype=np.float32)
    return result

def evaluate_regime(
    selected: BireAlignedStepper,
    comparator: BireAlignedStepper,
    state: Any,
    static: Any,
    regime_index: int,
    starts: np.ndarray,
    climatology_state: np.ndarray,
    climatology_derived: Mapping[str, np.ndarray],
    wet: np.ndarray,
) -> dict[str, np.ndarray]:
    """Roll both checkpoints to day 2,000 and reduce exactly as the frozen suite."""

    records = np.stack(
        [np.full(starts.shape, regime_index, dtype=np.int64), starts], axis=1
    )
    initial = _gather(state, records, 0)
    experiments = records[:, 0]
    selected_current = selected.normalized_state(initial)
    selected_static = selected.normalized_static(static, experiments)
    comparator_current = comparator.normalized_state(initial)
    comparator_static = comparator.normalized_static(static, experiments)
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
    for name in ("selected", "prior"):
        for field in ACC_FIELDS:
            arrays[f"acc__{name}__{field}"] = np.empty(
                (starts.size, len(SHORT_LEAD_DAYS)), dtype=np.float32
            )
    arrays["single_rmse__streamfunction"] = np.empty(len(SHORT_LEAD_DAYS), dtype=np.float32)
    arrays["single_rmse__sst"] = np.empty_like(arrays["single_rmse__streamfunction"])
    for tag, leads in (("figure3", plots.FIGURE_3_LEADS), ("figure7", plots.FIGURE_7_LEADS)):
        arrays[f"{tag}_truth_streamfunction"] = np.empty((len(leads), *wet.shape), dtype=np.float32)
        arrays[f"{tag}_model_streamfunction"] = np.empty_like(arrays[f"{tag}_truth_streamfunction"])
    figure3 = {lead: index for index, lead in enumerate(plots.FIGURE_3_LEADS)}
    figure7 = {lead: index for index, lead in enumerate(plots.FIGURE_7_LEADS)}
    wet_tensor = torch.from_numpy(wet).to(selected.device)

    with torch.no_grad():
        for lead_index, lead in enumerate(LEAD_DAYS):
            if lead:
                selected_current = selected.step(selected_current, selected_static)
                selected_prediction = selected.physical(selected_current)
                if lead <= 200:
                    comparator_current = comparator.step(comparator_current, comparator_static)
                    comparator_prediction = comparator.physical(comparator_current)
            else:
                selected_prediction = initial.copy()
                comparator_prediction = initial.copy()
            truth = _gather(state, records, lead)
            truth_fields = _fields(truth, wet)
            selected_fields = _fields(selected_prediction, wet)
            for field in RMSE_FIELDS:
                arrays[f"rmse__model__{field}"][:, lead_index] = _member_rmse(
                    selected_fields[field], truth_fields[field], wet)
                arrays[f"rmse__persistence__{field}"][:, lead_index] = _member_rmse(
                    initial_fields[field], truth_fields[field], wet)
                arrays[f"rmse__climatology__{field}"][:, lead_index] = _member_rmse(
                    climate_fields[field], truth_fields[field], wet)
            arrays["finite"][:, lead_index] = np.isfinite(selected_prediction).all(axis=(1, 2, 3))
            arrays["normalized_max_abs"][:, lead_index] = (
                torch.amax(torch.abs(selected_current[:, :, wet_tensor]), dim=(1, 2))
                .detach().cpu().numpy()
            )
            if lead <= 200:
                short_index = lead // 10
                comparator_fields = _fields(comparator_prediction, wet)
                for field in ACC_FIELDS:
                    arrays[f"acc__selected__{field}"][:, short_index] = _member_acc(
                        selected_fields[field], truth_fields[field], climate_fields[field], wet)
                    arrays[f"acc__prior__{field}"][:, short_index] = _member_acc(
                        comparator_fields[field], truth_fields[field], climate_fields[field], wet)
                arrays["single_rmse__streamfunction"][short_index] = _member_rmse(
                    selected_fields["streamfunction"], truth_fields["streamfunction"], wet)[0]
                arrays["single_rmse__sst"][short_index] = _member_rmse(
                    selected_fields["sst"], truth_fields["sst"], wet)[0]
            if lead in figure3:
                arrays["figure3_truth_streamfunction"][figure3[lead]] = truth_fields["streamfunction"][0]
                arrays["figure3_model_streamfunction"][figure3[lead]] = selected_fields["streamfunction"][0]
            if lead in figure7:
                arrays["figure7_truth_streamfunction"][figure7[lead]] = truth_fields["streamfunction"][0]
                arrays["figure7_model_streamfunction"][figure7[lead]] = selected_fields["streamfunction"][0]
    return arrays

VERSION = "model_c_bire_protocol_rollout_ft_s0_figures_v2"

CONTRACT_STATUS = (
    "frozen_after_the_bire_protocol_rollout_ft_training_and_validation"
    "_and_before_any_inference_metric"
)

COMPARATOR_STEP = BASELINE_OPTIMIZER_STEP

MODEL_LABEL = f"Bire-protocol Model C ({ROLLOUT_STEPS}-step fine-tune)"

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

GATE_NAME = "bire_protocol_rollout_ft_acceptance_gate.json"

#: The arm is the same arm before and after the code was consolidated, so a
#: report published by the v1 tree finalizes a v2 contract. The code that
#: produced it is byte-identical; only the module layout moved.
ACCEPTED_TRAINING_REPORT_VERSIONS = (
    "model_c_bire_protocol_rollout_ft_v1",
    "model_c_bire_protocol_rollout_ft_v2",
)

class BireProtocolRolloutFineTuneFigureError(RuntimeError):
    """Raised when the rollout fine-tune figure contract is violated."""

def _read(contract: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = contract
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value

def unfilled_fields(contract: Mapping[str, Any]) -> list[str]:
    """Declared-pending contract fields that training has not yet supplied."""

    return [
        ".".join(path)
        for path in PENDING_PATHS
        if _read(contract, path) in (None, PENDING)
    ]

def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the figure contract frozen after training and before any inference metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    protocol = contract.get("protocol", {})
    figure6 = contract.get("figure6", {})
    selected = contract.get("selected_model", {})
    pending = unfilled_fields(contract)
    if pending:
        raise BireProtocolRolloutFineTuneFigureError(
            "the figure contract still carries post-training fields: "
            + ", ".join(pending)
            + " -- run `finalize` against the training report first"
        )
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or contract.get("dataset", {}).get("version") != DATASET_VERSION
        or int(protocol.get("member_count", -1)) != MEMBER_COUNT
        or tuple(protocol.get("start_draw_order", ()))
        != tuple(int(v) for v in declared_inference_starts())
        or tuple(protocol.get("regimes", ())) != REGIMES
        or protocol.get("primary_regime") != "S0"
        or int(figure6.get("comparator_optimizer_step", -1)) != COMPARATOR_STEP
        or figure6.get("literal_pretrain_finetune_pair") is not True
        or int(selected.get("rollout_steps", -1)) != ROLLOUT_STEPS
        or selected.get("base_loss_contract_sha256") != FINE_TUNE_LOSS_CONTRACT_SHA256
        or contract.get("comparator_model", {}).get("base_loss_contract_sha256")
        != MODEL_C_LOSS_V1_CONTRACT_SHA256
        or int(selected.get("optimizer_step", -1)) not in CHECKPOINT_STEPS
    ):
        raise BireProtocolRolloutFineTuneFigureError("rollout fine-tune figure contract changed")
    if verify_sources:
        for label, specification in contract["artifacts"].items():
            plots._verify_file(specification, label)
        root = resolved.parents[1]
        for relative, expected in contract["source_hashes"].items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise BireProtocolRolloutFineTuneFigureError(f"source changed: {relative}")
    return contract, resolved, _file_sha256(resolved)

def _stepper(
    contract: Mapping[str, Any],
    key: str,
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> BireAlignedStepper:
    """Build one checkpoint's stepper against the objective declared for *it*.

    The suite's own ``_stepper`` checks every checkpoint against a single loss
    hash.  That is right when both curves come from one run; here the comparator
    is a three-step v1 model and the selected one is its six-step fine-tune, so
    each is verified against the hash its own contract block declares.
    """

    record = contract["artifacts"][key]
    payload = torch.load(Path(record["path"]), map_location=device, weights_only=False)
    if key == "selected_checkpoint":
        declared = contract["selected_model"]
        expected_step = int(declared["optimizer_step"])
    else:
        declared = contract["comparator_model"]
        expected_step = COMPARATOR_STEP
    architecture_dict = contract["selected_model"]["architecture"]
    if (
        payload.get("architecture") != architecture_dict
        or int(payload.get("optimizer_step", -1)) != expected_step
        or payload.get("dataset_version") != DATASET_VERSION
        or payload.get("base_loss_contract_sha256") != declared["base_loss_contract_sha256"]
        or int(payload.get("rollout_steps", -1)) != int(declared["rollout_steps"])
    ):
        raise BireProtocolRolloutFineTuneFigureError(
            f"{key} identity, dataset, or objective changed"
        )
    model = build_bire_aligned_model(BireAlignedArchitecture(**architecture_dict)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with np.load(Path(contract["artifacts"]["selected_normalization"]["path"])) as artifact:
        mean = np.asarray(artifact["pointwise_mean"], dtype=np.float32)
        scale = np.asarray(artifact["pointwise_scale"], dtype=np.float32)
    return BireAlignedStepper(
        model=model, device=device, wet=wet, mean=mean, scale=scale,
        wind_mean=wind_mean, wind_scale=wind_scale,
    )

class _S0Captions:
    """Rewrite the frozen plotters' hard-coded S0 control-wind captions.

    The frozen figure functions embed ``$\\tau_0=0.1$``, ``One S0 inference
    member``, ``S0 architecture-direction comparison``, and the v2 legend names
    directly in their title strings.  Reusing those functions verbatim --- which
    is what keeps the figure definitions identical across every package --- would
    therefore label the S1 and S2 packages with S0's wind and with a comparison
    this run does not perform.

    Copying the plotters instead would let the definitions drift, so the captions
    are rewritten in place for the duration of one regime's plotting.  The
    substitutions are explicit and asserted by tests; nothing about the data,
    axes, or reductions changes.
    """

    def __init__(self, regime: str, tau0: float, selected_step: int) -> None:
        self.regime = regime
        tau = f"{tau0:g}"
        wind = REGIME_WIND_LABEL[regime]
        self.rules: tuple[tuple[str, str], ...] = (
            # Longest first: the bare tau rule would otherwise consume this one.
            (
                r"Control wind $\tau_0=0.1$ N m$^{-2}$",
                rf"{wind} $\tau_0={tau}$ N m$^{{-2}}$",
            ),
            (r"$\tau_0=0.1$ N m$^{-2}$", rf"$\tau_0={tau}$ N m$^{{-2}}$"),
            # "inference" is Bire's own word for this block, so unlike the v3
            # chronological arm -- which called it the test block -- the captions
            # keep it; only the regime name is substituted.
            ("One S0 inference member", f"One {regime} inference member"),
            (
                "S0 architecture-direction comparison",
                f"{regime} training-progress comparison",
            ),
            ("Prior residual Model C", f"Step {COMPARATOR_STEP:,} checkpoint"),
            (
                "Selected anomaly-direct Model C",
                f"Selected step {selected_step:,} checkpoint",
            ),
        )

    def rewrite(self, text: Any) -> Any:
        if not isinstance(text, str):
            return text
        for old, new in self.rules:
            text = text.replace(old, new)
        return text

    def __enter__(self) -> "_S0Captions":
        import matplotlib.axes
        import matplotlib.figure

        self._axes = matplotlib.axes.Axes
        self._figure = matplotlib.figure.Figure
        self._set_title = matplotlib.axes.Axes.set_title
        self._suptitle = matplotlib.figure.Figure.suptitle
        self._plot = matplotlib.axes.Axes.plot
        self._method_labels = plots.METHOD_LABELS
        rewrite, set_title, suptitle, plot = (
            self.rewrite,
            self._set_title,
            self._suptitle,
            self._plot,
        )

        def patched_set_title(axis, label, *args, **kwargs):
            return set_title(axis, rewrite(label), *args, **kwargs)

        def patched_suptitle(figure, text, *args, **kwargs):
            return suptitle(figure, rewrite(text), *args, **kwargs)

        def patched_plot(axis, *args, **kwargs):
            if "label" in kwargs:
                kwargs["label"] = rewrite(kwargs["label"])
            return plot(axis, *args, **kwargs)

        matplotlib.axes.Axes.set_title = patched_set_title
        matplotlib.figure.Figure.suptitle = patched_suptitle
        matplotlib.axes.Axes.plot = patched_plot
        plots.METHOD_LABELS = {
            **self._method_labels,
            "model": MODEL_LABEL,
        }
        return self

    def __exit__(self, *exc: Any) -> None:
        self._axes.set_title = self._set_title
        self._figure.suptitle = self._suptitle
        self._axes.plot = self._plot
        plots.METHOD_LABELS = self._method_labels

class FineTuneLabels(_S0Captions):
    """The frozen captions, rewritten for a literal pre-train / fine-tune pair.

    The suite rewrites ``S0 architecture-direction comparison`` into a
    *training-progress* comparison, which is what an earlier arm's own-run
    comparator was.  Here the two curves really are the model before and after
    fine-tuning, so the caption says so.
    """

    def __init__(self, regime: str, tau0: float, selected_step: int) -> None:
        super().__init__(regime, tau0, selected_step)
        self.rules = (
            (
                "S0 architecture-direction comparison",
                f"{regime} three-step model vs six-step fine-tune",
            ),
            (
                "Prior residual Model C",
                f"Before fine-tuning (step {COMPARATOR_STEP:,})",
            ),
            (
                "Selected anomaly-direct Model C",
                f"After {ROLLOUT_STEPS}-step fine-tune (step {selected_step:,})",
            ),
            *(
                rule
                for rule in self.rules
                if rule[0]
                not in (
                    "S0 architecture-direction comparison",
                    "Prior residual Model C",
                    "Selected anomaly-direct Model C",
                )
            ),
        )

def long_rollout_gate(
    arrays: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """The 2,000-day half of the final acceptance gate, from the published arrays."""

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
            "from climatology; it is reported, not gated, because the arm "
            "declaration does not require day-2,000 skill over persistence"
        ),
        "streamfunction_basis": (
            "member 0's day-2,000 barotropic streamfunction, the field figure 7 "
            "publishes and the visual criterion inspects"
        ),
        "visual_criterion": "by_inspection_of_figures_3_and_7",
    }

def acceptance_gate(contract: Mapping[str, Any], regime: str = "S0") -> dict[str, Any]:
    """Join the training arm's validation half with this package's 2,000-day half."""

    output = Path(contract["output"]["project_root"]).resolve() / regime
    with np.load(output / plots.ARRAYS_NAME) as stored:
        arrays = {name: stored[name] for name in stored.files}
    summary = json.loads((output / plots.SUMMARY_NAME).read_text())
    training = json.loads(
        Path(contract["artifacts"]["selected_report"]["path"]).read_text()
    )
    validation = training["acceptance_gate"]
    long_half = long_rollout_gate(arrays, summary)
    gate = {
        "version": VERSION,
        "regime": regime,
        "selected_optimizer_step": int(contract["selected_model"]["optimizer_step"]),
        "comparator_optimizer_step": COMPARATOR_STEP,
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
            "the arm declaration freezes this checkpoint and opens the adjoint "
            "study after the evaluation regardless of whether this gate passes"
        ),
    }
    gate["content_sha256"] = _json_sha256(gate)
    return gate

def _readme(regime: str, report: Mapping[str, Any]) -> str:
    starts = declared_inference_starts()
    selected = int(report["selected_optimizer_step"])
    return f"""# Six-step rollout fine-tune, {regime}: Figures 3--8

This package evaluates the step-{selected:,} checkpoint of the six-step rollout
fine-tune on the **{regime}** inference set (indices 6200--7199), tau0 =
{report['tau0_n_m2']} N m-2.

The fine-tune started from the step-{COMPARATOR_STEP:,} checkpoint of
`model_c_bire_protocol_duration_v1` and continued for 3,840 steps at 2e-5 with
the autoregressive rollout deepened from three ten-day calls to six and the
rollout weight raised from 0.15 to 0.50. Architecture, normalization, split,
Fourier modes, static inputs, positional encoding and the 46-channel output are
unchanged, so this package and
`outputs/af_fno/C/bire_protocol_duration_s0_figures_v1/S0/` differ only in the
checkpoint: same 15 members, same seed, same lead grid, same truth window, same
climatology and persistence baselines.

The starts are drawn from 6200--6999, this draw spanning {int(starts.min())}--{int(starts.max())},
so every member has lead-matched MITgcm truth to day 2,000
({int(starts.max())} + 2000 = {int(starts.max()) + 2000} < 9000) from days 7200--8999, which the
model never saw in any capacity.

**Figure 6 is a literal pre-train / fine-tune pair.** The black curve is the
step-{COMPARATOR_STEP:,} model the fine-tune started from; the red curve is the
selected step-{selected:,} fine-tuned model. Both were trained on the same data in
the same normalized coordinates by the same code path, so the gap between them
is what deepening the rollout bought and nothing else. Every earlier package in
this project could only offer a within-run training-progress comparison.

The two checkpoints carry different objective hashes -- v1 over three steps for
the comparator, the six-step contract for the selected model -- and each is
verified against its own.

Climatology is the pointwise {regime} mean over the Bire training block
(0--5999) only. Persistence holds each member's initial physical field fixed.
RMSE is computed over wet cells per member; lines and bands are the mean and
10th/90th percentiles across the 15 members.

The 2,000-day half of the final acceptance gate is written beside this folder as
`{GATE_NAME}`.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `{report['report_content_sha256']}`.
"""

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
        raise BireProtocolRolloutFineTuneFigureError(
            f"the training report is not on disk yet: {report_path}"
        )
    if report_path.name != TRAINING_REPORT_NAME:
        raise BireProtocolRolloutFineTuneFigureError(
            f"the declared report is not {TRAINING_REPORT_NAME}"
        )
    report = json.loads(report_path.read_text())
    if report.get("version") not in ACCEPTED_TRAINING_REPORT_VERSIONS:
        raise BireProtocolRolloutFineTuneFigureError("the report is not this arm's")
    published = report["published_checkpoint"]
    resolutions = {
        ("selected_model", "optimizer_step"): int(published["optimizer_step"]),
        ("artifacts", "selected_checkpoint", "sha256"): str(published["checkpoint_sha256"]),
        ("artifacts", "selected_normalization", "sha256"): str(published["normalization_sha256"]),
        ("artifacts", "selected_report", "sha256"): _file_sha256(report_path),
    }
    applied: dict[str, Any] = {}
    for path, value in resolutions.items():
        current = _read(contract, path)
        if current not in (None, PENDING) and current != value:
            raise BireProtocolRolloutFineTuneFigureError(
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
            raise BireProtocolRolloutFineTuneFigureError(
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


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    """Publish the six figures for every regime on the v3 test block."""

    if torch is None:
        raise RuntimeError("v3 figure evaluation requires PyTorch")
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
    _, _, _, _, wind_mean, wind_scale = _normalizers(group)
    starts = declared_inference_starts()
    climatology_state, climatology_derived, climatology_days = train_only_climatology(state, wet)
    if climatology_days != TRAIN_RANGE[1] - TRAIN_RANGE[0]:
        raise BireProtocolFigureError("the v3 train-only climatology did not cover 0--5039")

    selected = _stepper(contract, "selected_checkpoint", device, wet, wind_mean, wind_scale)
    comparator = _stepper(contract, "comparator_checkpoint", device, wet, wind_mean, wind_scale)

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
            selected, comparator, state, static, regime_index, starts,
            climatology_state, climatology_derived, wet,
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
            with _S0Captions(
                regime,
                float(contract["dataset"]["tau0_n_m2"][regime]),
                int(contract["selected_model"]["optimizer_step"]),
            ):
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
                "role": "primary" if regime == "S0" else "robustness",
                "contract": str(resolved_contract),
                "contract_sha256": contract_sha,
                "dataset": str(dataset),
                "dataset_version": DATASET_VERSION,
                "selected_optimizer_step": int(contract["selected_model"]["optimizer_step"]),
                "comparator_optimizer_step": COMPARATOR_STEP,
                "start_draw_order": starts.astype(int).tolist(),
                "inference_start_range": list(INFERENCE_START_RANGE),
                "summary": summary,
                "arrays": str(scratch / plots.ARRAYS_NAME),
                "arrays_sha256": _file_sha256(scratch_arrays),
                "figures": list(plots.FIGURE_NAMES),
                "figure6": contract["figure6"],
                "regime_captions": "frozen S0 control-wind captions rewritten for this regime",
                "elapsed_seconds": time.monotonic() - started,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
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
                    for path in sorted(output_tmp.iterdir()) if path.is_file()
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
    """Verify sources, starts and both checkpoint identities without plotting."""

    contract, resolved, digest = load_contract(contract_path)
    starts = declared_inference_starts()
    return {
        "status": "pass",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset_version": DATASET_VERSION,
        "regimes": list(REGIMES),
        "primary_regime": "S0",
        "member_count": MEMBER_COUNT,
        "start_draw_order": starts.astype(int).tolist(),
        "inference_start_range": list(INFERENCE_START_RANGE),
        "maximum_lead_days": max(LEAD_DAYS),
        "selected_optimizer_step": int(contract["selected_model"]["optimizer_step"]),
        "comparator_optimizer_step": COMPARATOR_STEP,
        "literal_pretrain_finetune_pair": True,
        "selected_rollout_steps": ROLLOUT_STEPS,
        "selected_base_loss_contract_sha256": FINE_TUNE_LOSS_CONTRACT_SHA256,
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
    gate = acceptance_gate(contract, "S0")
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
