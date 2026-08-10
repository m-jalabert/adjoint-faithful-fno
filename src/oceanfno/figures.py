"""S0 Figures 3--8 for the meridional-32 successor to local24.

Publishes the frozen Bire figure suite for the selected 32x24 checkpoint and
evaluates the 2,000-day half of the final acceptance gate. Figure 6 is the
literal local24-parent / Y32-fine-tune pair. Each checkpoint is verified and
built against its own architecture declaration.

Held-evaluation only: no training, no checkpoint selection, no promotion.
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

from .runtime import torch
from . import plots
from .runtime import _device, _file_sha256, _json_sha256
from .dataset import DATASET_VERSION, INFERENCE_START_RANGE, MAXIMUM_INFERENCE_ROLLOUT_DAYS, TRAIN_RANGE, _normalizers, assert_model_visible, assert_truth_available, inference_starts
from .diagnostics import _member_acc, _member_rmse, derived_fields
from .model import BireAlignedStepper, BireLocal24Architecture, BireY32Architecture, build_bire_local24_model, build_bire_y32_model
from .validation import _gather, train_only_climatology
from .train import BASELINE_OPTIMIZER_STEP, CHECKPOINT_STEPS, FINE_TUNE_LOSS_CONTRACT_SHA256, NORMALIZATION_NAME as Y32_NORMALIZATION_NAME, PARENT_VERSION as LOCAL24_TRAINING_VERSION, REPORT_NAME as TRAINING_REPORT_NAME, ROLLOUT_STEPS, VERSION as TRAINING_VERSION

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

VERSION = "model_c_bire_protocol_rollout_ft_local24_y32_s0_figures_v1"

CONTRACT_STATUS = (
    "frozen_after_the_bire_protocol_rollout_ft_local24_y32_training_and_"
    "validation_and_before_any_inference_metric"
)

COMPARATOR_STEP = BASELINE_OPTIMIZER_STEP

MODEL_LABEL = "Bire-protocol Model C (local 3x3, 32x24 modes)"

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

GATE_NAME = "bire_protocol_rollout_ft_local24_y32_acceptance_gate.json"

ACCEPTED_TRAINING_REPORT_VERSIONS = (TRAINING_VERSION,)

class BireProtocolRolloutFineTuneFigureError(RuntimeError):
    """Compatibility base for held-evaluation contract failures."""


class BireY32FigureError(BireProtocolRolloutFineTuneFigureError):
    """Raised when the Y32 held-evaluation contract is violated."""


_REPOSITORY = Path(__file__).resolve().parents[2]
_EXPECTED_PROJECT_ROOT = str(
    _REPOSITORY
    / "outputs/af_fno/C/bire_protocol_rollout_ft_local24_y32_s0_figures_v1"
)
_EXPECTED_SCRATCH_ROOT = (
    "/bigscratch/mjalabert314/bire_james25_repro/af_fno/models/C/"
    "bire_protocol_rollout_ft_local24_y32_s0_figures_v1"
)
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
    "climatology": "pointwise_S0_mean_over_the_bire_training_block_0_5999_only",
    "ensemble_summary": "mean_and_10th_90th_percentiles_across_15_member_metrics",
    "persistence": "each_members_initial_physical_field_held_fixed_at_every_lead",
    "rmse": "square_root_of_mean_squared_error_over_wet_cells_for_each_member",
}
_EXPECTED_TRUTH = {
    "continuation_required": False,
    "lead_matched_to_day": 2000,
    "source": "trajectory_v3_store",
}
_REQUIRED_SOURCE_HASHES = frozenset(
    {
        "src/oceanfno/dataset.py",
        "src/oceanfno/diagnostics.py",
        "src/oceanfno/figures.py",
        "src/oceanfno/model.py",
        "src/oceanfno/plots.py",
        "src/oceanfno/runtime.py",
        "src/oceanfno/train.py",
        "src/oceanfno/validation.py",
    }
)


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

def unfilled_fields(contract: Mapping[str, Any]) -> list[str]:
    """Declared-pending contract fields that training has not yet supplied."""

    return [
        ".".join(path)
        for path in PENDING_PATHS
        if _read(contract, path) in (None, PENDING)
    ]


def _training_provenance(contract: Mapping[str, Any]) -> None:
    """Bind the figure declaration to the completed Y32 training report."""

    selected = contract["selected_model"]
    comparator = contract["comparator_model"]
    artifacts = contract["artifacts"]
    report_path = Path(str(artifacts["selected_report"].get("path", ""))).resolve()
    if report_path.name != TRAINING_REPORT_NAME or not report_path.is_file():
        raise BireY32FigureError("the selected report is not the completed Y32 report")
    if _file_sha256(report_path) != artifacts["selected_report"].get("sha256"):
        raise BireY32FigureError("the selected Y32 report hash changed")
    report = json.loads(report_path.read_text())
    published = report.get("published_checkpoint", {})
    selected_checkpoint = artifacts["selected_checkpoint"]
    normalization = artifacts["selected_normalization"]
    if (
        report.get("status") != "complete"
        or report.get("version") != TRAINING_VERSION
        or report.get("architecture") != selected.get("architecture")
        or report.get("base_loss_contract_sha256")
        != FINE_TUNE_LOSS_CONTRACT_SHA256
        or _integer(published.get("optimizer_step"))
        != _integer(selected.get("optimizer_step"))
        or published.get("checkpoint") != selected_checkpoint.get("path")
        or published.get("checkpoint_sha256") != selected_checkpoint.get("sha256")
        or published.get("normalization") != normalization.get("path")
        or published.get("normalization_sha256") != normalization.get("sha256")
    ):
        raise BireY32FigureError("the selected model disagrees with its Y32 report")

    training_path = Path(str(selected.get("training_contract", ""))).resolve()
    comparator_path = Path(str(comparator.get("training_contract", ""))).resolve()
    if not training_path.is_file() or not comparator_path.is_file():
        raise BireY32FigureError("a selected or comparator training contract is absent")
    training = json.loads(training_path.read_text())
    comparator_training = json.loads(comparator_path.read_text())
    parent_checkpoint = training.get("sources", {}).get(
        "initialization_checkpoint", {}
    )
    parent_normalization = training.get("sources", {}).get(
        "parent_normalization", {}
    )
    if (
        training.get("version") != TRAINING_VERSION
        or training.get("architecture") != selected.get("architecture")
        or comparator_training.get("version") != LOCAL24_TRAINING_VERSION
        or comparator_training.get("architecture") != comparator.get("architecture")
        or parent_checkpoint.get("path")
        != artifacts["comparator_checkpoint"].get("path")
        or parent_checkpoint.get("sha256")
        != artifacts["comparator_checkpoint"].get("sha256")
        or parent_normalization.get("sha256") != normalization.get("sha256")
        or Path(str(normalization.get("path", ""))).name
        != Y32_NORMALIZATION_NAME
    ):
        raise BireY32FigureError("the local24-to-Y32 training provenance changed")


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load and strictly audit the S0 Y32 held-evaluation declaration."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    pending = unfilled_fields(contract)
    if pending:
        raise BireY32FigureError(
            "the Y32 figure contract still carries post-training fields: "
            + ", ".join(pending)
            + " -- run `finalize` against the Y32 report first"
        )

    protocol = contract.get("protocol", {})
    selected = contract.get("selected_model", {})
    comparator = contract.get("comparator_model", {})
    figure6 = contract.get("figure6", {})
    output = contract.get("output", {})
    dataset = contract.get("dataset", {})
    expected_starts = tuple(int(value) for value in declared_inference_starts())
    protocol_ok = (
        _integer(protocol.get("member_count")) == MEMBER_COUNT
        and _integer(protocol.get("start_seed")) == START_SEED
        and tuple(protocol.get("start_draw_order", ())) == expected_starts
        and tuple(protocol.get("regimes", ())) == REGIMES
        and protocol.get("primary_regime") == "S0"
        and tuple(protocol.get("figure_names", ())) == tuple(plots.FIGURE_NAMES)
        and tuple(protocol.get("figure3_lead_days", ()))
        == tuple(plots.FIGURE_3_LEADS)
        and tuple(protocol.get("figure7_lead_days", ()))
        == tuple(plots.FIGURE_7_LEADS)
        and tuple(protocol.get("rmse_fields", ())) == tuple(plots.RMSE_FIELDS)
        and tuple(protocol.get("acc_fields", ())) == tuple(plots.ACC_FIELDS)
        and tuple(protocol.get("inference_set", ())) == (6200, 7200)
        and tuple(protocol.get("start_window", ())) == (6200, 7000)
        and _integer(protocol.get("maximum_lead_days")) == 2000
        and _integer(protocol.get("prediction_interval_days")) == 10
        and protocol.get("short_lead_days") == "0_to_200_inclusive_by_10"
        and protocol.get("long_lead_days") == "0_to_2000_inclusive_by_10"
    )
    models_ok = (
        selected.get("version") == TRAINING_VERSION
        and comparator.get("version") == LOCAL24_TRAINING_VERSION
        and _integer(selected.get("optimizer_step")) in CHECKPOINT_STEPS
        and _integer(comparator.get("optimizer_step")) == COMPARATOR_STEP
        and _integer(selected.get("rollout_steps")) == ROLLOUT_STEPS
        and _integer(comparator.get("rollout_steps")) == ROLLOUT_STEPS
        and selected.get("base_loss_contract_sha256")
        == FINE_TUNE_LOSS_CONTRACT_SHA256
        and comparator.get("base_loss_contract_sha256")
        == FINE_TUNE_LOSS_CONTRACT_SHA256
        and selected.get("architecture") == BireY32Architecture().to_dict()
        and comparator.get("architecture") == BireLocal24Architecture().to_dict()
    )
    output_ok = (
        output.get("project_root") == _EXPECTED_PROJECT_ROOT
        and output.get("scratch_root") == _EXPECTED_SCRATCH_ROOT
        and output.get("overwrite") is False
        and output.get("one_folder_per_regime") is True
        and tuple(output.get("required", ())) == _EXPECTED_OUTPUTS
    )
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or dataset.get("version") != DATASET_VERSION
        or tuple(dataset.get("train", ())) != (0, 6000)
        or tuple(dataset.get("validation", ())) != (6000, 7200)
        or tuple(dataset.get("inference", ())) != (6200, 7200)
        or dataset.get("tau0_n_m2") != {"S0": 0.1}
        or contract.get("baselines") != _EXPECTED_BASELINES
        or contract.get("truth") != _EXPECTED_TRUTH
        or not protocol_ok
        or not models_ok
        or not output_ok
        or _integer(figure6.get("comparator_optimizer_step")) != COMPARATOR_STEP
        or figure6.get("literal_pretrain_finetune_pair") is not True
    ):
        raise BireY32FigureError("the Y32 S0 figure contract changed")
    try:
        BireY32Architecture(**selected["architecture"])
        BireLocal24Architecture(**comparator["architecture"])
        _training_provenance(contract)
    except BireY32FigureError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise BireY32FigureError(
            "the selected or comparator Y32 figure provenance changed"
        ) from error
    if verify_sources:
        hashes = contract.get("source_hashes", {})
        if not _REQUIRED_SOURCE_HASHES.issubset(hashes):
            raise BireY32FigureError("the Y32 figure source declaration is incomplete")
        for label, specification in contract.get("artifacts", {}).items():
            plots._verify_file(specification, label)
        root = resolved.parents[1]
        for relative, expected in hashes.items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise BireY32FigureError(f"Y32 figure source changed: {relative}")
    return contract, resolved, _file_sha256(resolved)

def _stepper(
    contract: Mapping[str, Any],
    key: str,
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> BireAlignedStepper:
    """Build either Y32 or its local24 parent after identity checks."""

    if torch is None:  # pragma: no cover - environment dependent
        raise RuntimeError("Y32 figure evaluation requires PyTorch")
    if key == "selected_checkpoint":
        declared = contract["selected_model"]
        expected_version = TRAINING_VERSION
        architecture = BireY32Architecture(**declared["architecture"])
        builder = build_bire_y32_model
    elif key == "comparator_checkpoint":
        declared = contract["comparator_model"]
        expected_version = LOCAL24_TRAINING_VERSION
        architecture = BireLocal24Architecture(**declared["architecture"])
        builder = build_bire_local24_model
    else:
        raise BireY32FigureError(f"unknown checkpoint key: {key}")
    payload = torch.load(
        Path(contract["artifacts"][key]["path"]),
        map_location=device,
        weights_only=False,
    )
    if (
        payload.get("version") != expected_version
        or payload.get("architecture") != declared["architecture"]
        or _integer(payload.get("optimizer_step"))
        != _integer(declared.get("optimizer_step"))
        or payload.get("dataset_version") != DATASET_VERSION
        or payload.get("base_loss_contract_sha256")
        != declared["base_loss_contract_sha256"]
        or _integer(payload.get("rollout_steps"))
        != _integer(declared.get("rollout_steps"))
    ):
        raise BireY32FigureError(
            f"{key} identity, architecture, dataset, or objective changed"
        )
    try:
        model = builder(architecture).to(device)
        incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise BireY32FigureError(f"{key} state dictionary changed") from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise BireY32FigureError(f"{key} did not load strictly")
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
    """Label Figure 6 as the literal local24 / Y32 comparison."""

    def __init__(self, regime: str, tau0: float, selected_step: int) -> None:
        super().__init__(regime, tau0, selected_step)
        replaced = {
            "S0 architecture-direction comparison",
            "Prior residual Model C",
            "Selected anomaly-direct Model C",
        }
        self.rules = (
            (
                "S0 architecture-direction comparison",
                f"{regime} local24 parent vs meridional-32 fine-tune",
            ),
            (
                "Prior residual Model C",
                f"Local 3x3 + 24x24 modes (step {COMPARATOR_STEP:,})",
            ),
            (
                "Selected anomaly-direct Model C",
                f"Local 3x3 + 32x24 modes (step {selected_step:,})",
            ),
            *(rule for rule in self.rules if rule[0] not in replaced),
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
            "from climatology; it is not gated, but is reported alongside "
            "persistence and the anomaly diagnostics for the Y32 "
            "keep-or-revert decision"
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
            "these measurable diagnostics and visual inspection support the "
            "keep-or-revert decision; this package promotes no checkpoint"
        ),
    }
    gate["content_sha256"] = _json_sha256(gate)
    return gate

def _readme(regime: str, report: Mapping[str, Any]) -> str:
    starts = declared_inference_starts()
    selected = int(report["selected_optimizer_step"])
    return f"""# Meridional-32 local Model C, {regime}: Figures 3--8

This held package evaluates the selected step-{selected:,} checkpoint of
`{TRAINING_VERSION}` on the exact 15-member S0 inference protocol used for
local24. The black Figure 6 curve is its literal step-{COMPARATOR_STEP:,}
local24 parent; the red curve is the 32x24 fine-tune. Both retain the trained
bias-free 49-to-46 local 3x3 path and the same six-step objective. Only four
new meridional modes on each side were opened during fine-tuning; zonal modes
remain fixed at 24.

Starts span {int(starts.min())}--{int(starts.max())}; every member has
lead-matched truth through day 2,000. Climatology remains the pointwise
{regime} training-block mean and persistence holds the initial physical state.
The numerical reductions, plot functions, filenames and lead grid are reused
unchanged from the preceding local24 package.

The measurable gate is written beside the S0 folder as `{GATE_NAME}`. This
package performs no training, selection, or checkpoint promotion.

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
        raise BireY32FigureError(
            f"the training report is not on disk yet: {report_path}"
        )
    if report_path.name != TRAINING_REPORT_NAME:
        raise BireY32FigureError(
            f"the declared report is not {TRAINING_REPORT_NAME}"
        )
    report = json.loads(report_path.read_text())
    if report.get("version") not in ACCEPTED_TRAINING_REPORT_VERSIONS:
        raise BireY32FigureError("the report is not this arm's")
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
            raise BireY32FigureError(
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
            raise BireY32FigureError(
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
            with FineTuneLabels(
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
