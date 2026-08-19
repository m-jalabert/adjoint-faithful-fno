"""S0 Figures 3--8 for the ninety-day fine-tune, plus its 2,000-day gate.

The same held evaluation the production arm runs, pointed at the fine-tuned
checkpoint instead: 15 fixed inference starts, autoregressed to day 2,000 at
ten-day steps, reduced by the frozen plot definitions in :mod:`oceanfno.plots`.

**This module adds a lineage, not a method.** Every numerical routine is
imported from :mod:`oceanfno.figures` and used unchanged --- the rollout
(:func:`~oceanfno.figures.evaluate_regime`), the 2,000-day conditions
(:func:`~oceanfno.figures.long_rollout_gate`), the member draw, the plots and
the summary. What is reimplemented here is only the part that identifies *which
training arm produced the checkpoint*: the parent arm asserts
``from_scratch is True`` and binds to ``train.VERSION``, and the fine-tune is by
construction neither of those things.

Doing it this way keeps ``figures.py`` byte-identical to the hash its own frozen
contract pins, so the parent's figure package stays reproducible --- which
matters here more than usual, because the parent's day-2,000 numbers are the
comparison this arm exists to move.

What the comparison is. At day 2,000 the parent reaches a maximum normalized
magnitude of 10.33 against a declared ceiling of 8, and its day-2,000 RMSE
divided by climatology's is 0.48 / 1.43 / 1.76 in pressure / SST / speed --- two
of the three fields worse than simply predicting the training-mean state. A
model on the right attractor should sit near or below one there. Those parent
numbers are read from its sealed figure summary, pinned by digest, and reported
beside this arm's own so the two are never compared from memory.

The figure plates themselves carry a single model curve, exactly as the parent's
do: adding a comparator series would mean changing ``plots.py``, which the
parent's contract also pins. The comparison lives in the gate and the report.

Entry points::

    python -m oceanfno.figures_ft90 finalize  --contract config/...json
    python -m oceanfno.figures_ft90 preflight --contract config/...json
    python -m oceanfno.figures_ft90 run       --contract config/...json [--device cuda]
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
    STATIC_FEATURES,
    TRAIN_RANGE,
    VALIDATION_RANGE,
)
from .diagnostics import _member_acc
from .model import ProductionArchitecture, ProductionStepper, build_model
from .validation import _gather, train_only_climatology
from .train import physical_static_block
from .finetune import (
    CHECKPOINT_STEPS,
    FINETUNE_LOSS_CONTRACT_SHA256,
    NORMALIZATION_NAME as SELECTED_NORMALIZATION_NAME,
    PARENT_OPTIMIZER_STEP,
    PARENT_VERSION,
    REPORT_NAME as TRAINING_REPORT_NAME,
    ROLLOUT_STEPS,
    VERSION as TRAINING_VERSION,
)

# Everything below is the production figure package's own machinery, reused
# rather than reimplemented. If a number moves there it moves here too.
from .figures import (
    ACC_FIELDS,
    FigureContractError,
    LEAD_DAYS,
    MAXIMUM_NORMALIZED_MAGNITUDE,
    MEMBER_COUNT,
    PENDING,
    REGIME_INDEX,
    REGIMES,
    RMSE_FIELDS,
    START_SEED,
    TAU0_N_M2,
    _EXPECTED_BASELINES,
    _EXPECTED_OUTPUTS,
    _EXPECTED_TRUTH,
    _REQUIRED_ARTIFACTS,
    _integer,
    _read,
    _fields,
    declared_inference_starts,
    evaluate_regime,
    long_rollout_gate,
    unfilled_fields,
)

VERSION = "model_c_production_1in_1out_spectralnorm_ft90_v1_s0_figures_v1"

CONTRACT_STATUS = (
    "frozen_after_the_ninety_day_fine_tuning_and_validation_and_before_any_"
    "inference_metric"
)

GATE_NAME = f"{TRAINING_VERSION}_acceptance_gate.json"

#: The parent's sealed figure package, read for the day-2,000 comparison. It is
#: an *artifact* of this contract, pinned by digest like every other input, so
#: the comparison can never drift against a re-run parent.
PARENT_FIGURES_VERSION = "model_c_production_1in_1out_spectralnorm_v1_s0_figures_v1"

PARENT_SUMMARY_ARTIFACT = "parent_figure_summary"

#: Seed for the no-information ACC reference. Fifteen states are drawn from the
#: inference block independently of the member starts; each lands at a random
#: phase of S0's cycle, which is exactly what "knows nothing about the ocean
#: state" means.
ACC_REFERENCE_SEED = 20260818

_REQUIRED_ARTIFACTS_FT90 = (*_REQUIRED_ARTIFACTS, PARENT_SUMMARY_ARTIFACT)

_REQUIRED_SOURCE_HASHES = frozenset(
    {
        "src/oceanfno/dataset.py",
        "src/oceanfno/diagnostics.py",
        "src/oceanfno/figures.py",
        "src/oceanfno/figures_ft90.py",
        "src/oceanfno/finetune.py",
        "src/oceanfno/model.py",
        "src/oceanfno/perturbation_growth.py",
        "src/oceanfno/plots.py",
        "src/oceanfno/runtime.py",
        "src/oceanfno/spectral_norm.py",
        "src/oceanfno/train.py",
        "src/oceanfno/validation.py",
    }
)


# ---------------------------------------------------------------------------
# provenance: this checkpoint came from the fine-tune, which came from the parent
# ---------------------------------------------------------------------------


def _training_provenance(contract: Mapping[str, Any]) -> None:
    """Bind the figure declaration to the completed fine-tuning report.

    The production arm's equivalent asserts ``from_scratch is True``. Here the
    whole point is the opposite, so the chain is checked one link further: this
    package's checkpoint must come from the fine-tuning report, and that report
    must name the published parent it continued.
    """

    selected = contract["selected_model"]
    artifacts = contract["artifacts"]
    report_path = Path(str(artifacts["selected_report"].get("path", ""))).resolve()
    if report_path.name != TRAINING_REPORT_NAME or not report_path.is_file():
        raise FigureContractError(
            "the selected report is not the completed fine-tuning report"
        )
    if _file_sha256(report_path) != artifacts["selected_report"].get("sha256"):
        raise FigureContractError("the selected fine-tuning report hash changed")
    report = json.loads(report_path.read_text())
    published = report.get("published_checkpoint", {})
    checkpoint = artifacts["selected_checkpoint"]
    normalization = artifacts["selected_normalization"]
    initialization = report.get("initialization", {})
    if (
        report.get("status") != "complete"
        or report.get("version") != TRAINING_VERSION
        or report.get("architecture") != selected.get("architecture")
        or report.get("loss_contract_sha256") != FINETUNE_LOSS_CONTRACT_SHA256
        or initialization.get("from_scratch") is not False
        or initialization.get("parent_version") != PARENT_VERSION
        or _integer(initialization.get("parent_optimizer_step")) != PARENT_OPTIMIZER_STEP
        or initialization.get("load_optimizer_state") is not False
        or initialization.get("normalization_reused") is not True
        or _integer(published.get("optimizer_step"))
        != _integer(selected.get("optimizer_step"))
        or published.get("checkpoint") != checkpoint.get("path")
        or published.get("checkpoint_sha256") != checkpoint.get("sha256")
        or published.get("normalization") != normalization.get("path")
        or published.get("normalization_sha256") != normalization.get("sha256")
        or Path(str(normalization.get("path", ""))).name != SELECTED_NORMALIZATION_NAME
    ):
        raise FigureContractError(
            "the selected model disagrees with its fine-tuning report"
        )
    training_path = Path(str(selected.get("training_contract", ""))).resolve()
    if not training_path.is_file():
        raise FigureContractError("the fine-tuning contract is absent")
    training = json.loads(training_path.read_text())
    if (
        training.get("version") != TRAINING_VERSION
        or training.get("architecture") != selected.get("architecture")
        or training.get("initialization", {}).get("from_scratch") is not False
        or training.get("initialization", {}).get("parent_version") != PARENT_VERSION
    ):
        raise FigureContractError("the fine-tuning provenance changed")


def parent_day2000(contract: Mapping[str, Any]) -> dict[str, Any]:
    """The parent's published day-2,000 numbers, for the comparison.

    Read from the parent's sealed figure summary rather than restated, and the
    file is verified by digest first, so this arm cannot quietly compare itself
    against something other than what the parent actually published.
    """

    specification = contract["artifacts"][PARENT_SUMMARY_ARTIFACT]
    path = Path(str(specification["path"])).resolve()
    if not path.is_file() or _file_sha256(path) != specification.get("sha256"):
        raise FigureContractError("the parent figure summary is missing or changed")
    summary = json.loads(path.read_text())
    return {
        "version": PARENT_FIGURES_VERSION,
        "summary": str(path),
        "summary_sha256": _file_sha256(path),
        "maximum_normalized_magnitude": float(
            summary["maximum_selected_normalized_abs"]
        ),
        "all_values_finite": bool(summary["all_selected_states_finite"]),
        "day2000_rmse_ratio_to_climatology": {
            field: float(summary["rmse"][field]["model"]["day2000_mean"])
            / float(summary["rmse"][field]["climatology"]["day2000_mean"])
            for field in RMSE_FIELDS
        },
        "day200_rmse": {
            field: float(summary["rmse"][field]["model"]["day200_mean"])
            for field in RMSE_FIELDS
        },
    }


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load and strictly audit the fine-tune's S0 held-evaluation declaration."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    pending = unfilled_fields(contract)
    if pending:
        raise FigureContractError(
            "the figure contract still carries post-training fields: "
            + ", ".join(pending)
            + " -- run `python -m oceanfno.figures_ft90 finalize` first"
        )

    protocol = contract.get("protocol", {})
    selected = contract.get("selected_model", {})
    output = contract.get("output", {})
    dataset = contract.get("dataset", {})
    expected_starts = tuple(int(value) for value in declared_inference_starts())
    # The evaluation protocol is the parent's, to the member: a fine-tune scored
    # on different starts would not be comparable to the thing it fine-tuned.
    protocol_ok = (
        _integer(protocol.get("member_count")) == MEMBER_COUNT
        and _integer(protocol.get("start_seed")) == START_SEED
        and tuple(protocol.get("start_draw_order", ())) == expected_starts
        and tuple(protocol.get("regimes", ())) == REGIMES
        and protocol.get("primary_regime") == "S0"
        and tuple(protocol.get("figure_names", ())) == tuple(plots.FIGURE_NAMES)
        and tuple(protocol.get("figure3_lead_days", ())) == tuple(plots.FIGURE_3_LEADS)
        and tuple(protocol.get("figure7_lead_days", ())) == tuple(plots.FIGURE_7_LEADS)
        and tuple(protocol.get("rmse_fields", ())) == tuple(plots.RMSE_FIELDS)
        and tuple(protocol.get("acc_fields", ())) == tuple(plots.ACC_FIELDS)
        and tuple(protocol.get("inference_set", ())) == INFERENCE_RANGE
        and tuple(protocol.get("start_window", ())) == INFERENCE_START_RANGE
        and _integer(protocol.get("maximum_lead_days")) == max(LEAD_DAYS)
        and _integer(protocol.get("prediction_interval_days")) == 10
        and _integer(protocol.get("acc_reference_seed")) == ACC_REFERENCE_SEED
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
        and selected.get("loss_contract_sha256") == FINETUNE_LOSS_CONTRACT_SHA256
        and selected.get("architecture") == ProductionArchitecture().to_dict()
        and selected.get("from_scratch") is False
        and selected.get("parent_version") == PARENT_VERSION
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
        or dataset.get("version") != DATASET_VERSION
        or tuple(dataset.get("train", ())) != TRAIN_RANGE
        or tuple(dataset.get("validation", ())) != VALIDATION_RANGE
        or tuple(dataset.get("inference", ())) != INFERENCE_RANGE
        or dataset.get("tau0_n_m2") != TAU0_N_M2
        or contract.get("baselines") != _EXPECTED_BASELINES
        or contract.get("truth") != _EXPECTED_TRUTH
        or not set(_REQUIRED_ARTIFACTS_FT90).issubset(contract.get("artifacts", {}))
        or not protocol_ok
        or not models_ok
        or not output_ok
    ):
        raise FigureContractError("the fine-tuned S0 figure contract changed")
    try:
        ProductionArchitecture(**selected["architecture"])
        _training_provenance(contract)
        parent_day2000(contract)
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
    """Fill the declared-pending fields from the fine-tuning report.

    Idempotent, and refuses to overwrite a field that already disagrees, exactly
    as the production package's ``finalize`` does.
    """

    resolved = Path(contract_path).resolve()
    contract = json.loads(resolved.read_text())
    report_path = Path(contract["artifacts"]["selected_report"]["path"])
    if not report_path.is_file():
        raise FigureContractError(
            f"the fine-tuning report is not on disk yet: {report_path}"
        )
    if report_path.name != TRAINING_REPORT_NAME:
        raise FigureContractError(f"the declared report is not {TRAINING_REPORT_NAME}")
    report = json.loads(report_path.read_text())
    if report.get("version") != TRAINING_VERSION or report.get("status") != "complete":
        raise FigureContractError(
            "the report is not this arm's completed fine-tuning report"
        )
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
                f"{key} path disagrees with the fine-tuning report: "
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


def acc_reference_curves(
    state: Any,
    regime_index: int,
    starts: np.ndarray,
    climatology_state: np.ndarray,
    climatology_derived: Mapping[str, np.ndarray],
    wet: np.ndarray,
    *,
    seed: int = ACC_REFERENCE_SEED,
) -> dict[str, np.ndarray]:
    """The two references an ACC curve is meaningless without.

    ACC is a correlation about a *time-mean* climatology, so its no-skill value
    is not zero in general --- it is whatever two unrelated states of the same
    climate happen to correlate at. For surface pressure in S0 that is nearly
    0.9, because roughly ninety percent of the pressure anomaly about the
    training mean is a stationary spatial pattern rather than a transient. A
    curve plotted without that floor invites reading 0.995 as "99.5 % of the
    signal", which it is not.

    Two references are computed here, from truth alone --- no model is involved:

    ``persistence``
        each member's own initial field held fixed, the usual trivial forecast;
    ``independent``
        fifteen states drawn from the inference block at random phase, i.e. a
        forecast that knows the climate but nothing whatever about the state.

    The gap between the model curve and ``independent`` is the skill that is
    actually being demonstrated.
    """

    records = np.stack(
        [np.full(starts.shape, regime_index, dtype=np.int64), starts], axis=1
    )
    initial_fields = _fields(_gather(state, records, 0), wet)
    climate = np.repeat(climatology_state[regime_index][None], starts.size, axis=0)
    climate_fields = _fields(climate, wet)
    for name, value in climatology_derived.items():
        climate_fields[name] = np.repeat(value[regime_index][None], starts.size, axis=0)

    low, high = INFERENCE_RANGE
    draw = np.sort(
        np.random.default_rng(int(seed)).choice(
            np.arange(low, high, dtype=np.int64), size=starts.size, replace=False
        )
    )
    independent_records = np.stack(
        [np.full(draw.shape, regime_index, dtype=np.int64), draw], axis=1
    )
    independent_fields = _fields(_gather(state, independent_records, 0), wet)

    leads = plots.SHORT_LEAD_DAYS
    curves = {
        f"acc__{tag}__{field}": np.empty((starts.size, len(leads)), dtype=np.float32)
        for tag in ("persistence", "independent")
        for field in ACC_FIELDS
    }
    for index, lead in enumerate(leads):
        truth_fields = _fields(_gather(state, records, lead), wet)
        for field in ACC_FIELDS:
            curves[f"acc__persistence__{field}"][:, index] = _member_acc(
                initial_fields[field], truth_fields[field], climate_fields[field], wet
            )
            curves[f"acc__independent__{field}"][:, index] = _member_acc(
                independent_fields[field],
                truth_fields[field],
                climate_fields[field],
                wet,
            )
    curves["acc_reference_start_draw_order"] = draw.astype(np.int32)
    return curves


def acc_day200_summary(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Day-200 ACC for the model and both references, per field.

    Reported together because the model number alone is not interpretable: an
    ACC of 0.995 against a no-information floor of 0.886 and one against a floor
    of 0.0 are very different claims.
    """

    index = list(plots.SHORT_LEAD_DAYS).index(200)
    record: dict[str, Any] = {}
    for field in ACC_FIELDS:
        model = float(np.mean(arrays[f"acc__model__{field}"][:, index]))
        floor = float(np.mean(arrays[f"acc__independent__{field}"][:, index]))
        persistence = float(
            np.mean(arrays[f"acc__persistence__{field}"][:, index])
        )
        record[field] = {
            "model": model,
            "persistence": persistence,
            "no_information_floor": floor,
            # How much of the reachable range above the floor was captured.
            "skill_above_floor": (
                (model - floor) / (1.0 - floor) if floor < 1.0 else float("nan")
            ),
        }
    record["definition"] = (
        "uncentered pattern correlation over all wet surface cells about the "
        "train-only pointwise time-mean, unweighted by cell area; the floor is "
        "the same statistic for fifteen independent states of the same climate"
    )
    return record


def _plot_acc_with_reference(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Figure 6, with the no-skill floor drawn on it.

    Same file, same four panels and same model series as the production
    package's figure 6. What is added is the persistence curve and the
    no-information floor, so the model curve is read against the range it can
    actually occupy rather than against zero.
    """

    import matplotlib.pyplot as plt

    leads = np.asarray(plots.SHORT_LEAD_DAYS)
    figure, axes = plt.subplots(
        4, 1, figsize=(5.4, 10.2), sharex=True, constrained_layout=True
    )
    for axis, field in zip(axes, ACC_FIELDS):
        floor = plots.percentile_curve(arrays[f"acc__independent__{field}"])
        axis.fill_between(
            leads,
            -1.0,
            floor["mean"],
            color="0.85",
            linewidth=0,
            zorder=0,
        )
        axis.plot(
            leads,
            floor["mean"],
            color="0.45",
            linewidth=1.1,
            linestyle=":",
            label="No-information floor",
            zorder=2,
        )
        persistence = plots.percentile_curve(arrays[f"acc__persistence__{field}"])
        axis.plot(
            leads,
            persistence["mean"],
            color="blue",
            linewidth=1.2,
            linestyle="--",
            label=plots.METHOD_LABELS["persistence"],
            zorder=3,
        )
        curve = plots.percentile_curve(arrays[f"acc__model__{field}"])
        axis.plot(
            leads,
            curve["mean"],
            color="red",
            linewidth=1.6,
            label=plots.METHOD_LABELS["model"],
            zorder=4,
        )
        axis.fill_between(
            leads,
            curve["p10"],
            curve["p90"],
            color="red",
            alpha=0.17,
            linewidth=0,
            zorder=1,
        )
        axis.axhline(0.0, color="0.65", linewidth=0.6)
        axis.set_ylim(-1.0, 1.02)
        axis.set_ylabel(f"{plots.FIELD_LABELS[field]}\nACC")
        axis.grid(color="0.82", linewidth=0.6)
    axes[0].set_title(
        r"S0 anomaly correlation; 15 inference members; $\Delta t=10$ days"
    )
    axes[-1].set_xlabel("Time (days)")
    axes[-1].set_xlim(0, 200)
    axes[0].legend(loc="lower left", fontsize=7, framealpha=0.9)
    figure.text(
        0.5,
        -0.012,
        "Shading below the dotted line is unreachable skill: two independent states of the\n"
        "same climate already correlate there. S0 is near-periodic (the SST anomaly pattern\n"
        "recurs at 0.99 after 171 days), so these curves do not decay the way a chaotic\n"
        "double gyre's would.",
        ha="center",
        va="top",
        fontsize=6.6,
        color="0.3",
    )
    figure.savefig(output / plots.FIGURE_NAMES[3], bbox_inches="tight")
    plt.close(figure)


def _stepper(
    contract: Mapping[str, Any],
    device: Any,
    wet: np.ndarray,
    statics: np.ndarray,
) -> ProductionStepper:
    """Build the fine-tuned model after checking its recorded identity."""

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
        or _integer(payload.get("optimizer_step"))
        != _integer(declared.get("optimizer_step"))
        or payload.get("dataset_version") != DATASET_VERSION
        or payload.get("loss_contract_sha256") != declared["loss_contract_sha256"]
        or _integer(payload.get("rollout_steps")) != _integer(declared.get("rollout_steps"))
        or payload.get("from_scratch") is not False
        or payload.get("parent_version") != PARENT_VERSION
    ):
        raise FigureContractError(
            "the fine-tuned checkpoint's identity, lineage, architecture, dataset "
            "or objective changed"
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


def acceptance_gate(contract: Mapping[str, Any], regime: str = "S0") -> dict[str, Any]:
    """The fine-tune's validation half, its 2,000-day half, and the comparison.

    The two halves are computed exactly as the production package computes them.
    What is added is the third block: the same day-2,000 quantities as the
    parent published them, and the change between the two. That change is the
    result of the experiment, so it is recorded rather than left to be read off
    two reports side by side.
    """

    output = Path(contract["output"]["project_root"]).resolve() / regime
    with np.load(output / plots.ARRAYS_NAME) as stored:
        arrays = {name: stored[name] for name in stored.files}
    summary = json.loads((output / plots.SUMMARY_NAME).read_text())
    training = json.loads(
        Path(contract["artifacts"]["selected_report"]["path"]).read_text()
    )
    validation = training["acceptance_gate"]
    long_half = long_rollout_gate(arrays, summary)
    parent = parent_day2000(contract)
    child_ratio = long_half["advisory_day2000_rmse_ratio_to_climatology"]
    parent_ratio = parent["day2000_rmse_ratio_to_climatology"]
    comparison = {
        "parent": parent,
        "maximum_normalized_magnitude": {
            "parent": parent["maximum_normalized_magnitude"],
            "fine_tuned": long_half["measured"]["maximum_normalized_magnitude"],
            "ceiling": MAXIMUM_NORMALIZED_MAGNITUDE,
        },
        "day2000_rmse_ratio_to_climatology": {
            field: {
                "parent": parent_ratio[field],
                "fine_tuned": child_ratio[field],
                "change": child_ratio[field] - parent_ratio[field],
            }
            for field in RMSE_FIELDS
        },
        "interpretation": (
            "a day-2000 ratio near or below one means the emulator's error has "
            "settled to about the error between two independent states of the "
            "same climate, which is the correct long-horizon behaviour past the "
            "roughly 90-day decorrelation time. A ratio well above one means it "
            "is doing worse than predicting the training mean, which is what the "
            "parent does in SST and surface speed"
        ),
    }
    gate = {
        "version": VERSION,
        "regime": regime,
        "selected_optimizer_step": int(contract["selected_model"]["optimizer_step"]),
        "parent_version": PARENT_VERSION,
        "validation_half": validation,
        "long_rollout_half": long_half,
        "comparison_to_the_parent": comparison,
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
            "fine-tuned model's held performance; this package promotes nothing"
        ),
    }
    gate = json_safe(gate)
    gate["content_sha256"] = _json_sha256(gate)
    return gate


def _readme(regime: str, report: Mapping[str, Any]) -> str:
    starts = declared_inference_starts()
    selected = int(report["selected_optimizer_step"])
    return f"""# Ninety-day fine-tune, {regime}: Figures 3--8

This held package evaluates the selected step-{selected:,} checkpoint of
`{TRAINING_VERSION}` on the 15-member {regime} inference protocol --- the same
protocol, the same 15 starts and the same plot definitions the parent package
used, so the two are directly comparable.

The model is the production operator, architecturally unchanged:
`F(x_t, S) -> x_(t+10)` with `x_t` the 46-channel state at one time level and
`S` the five physical static channels `[tau_x, wet, f, dx, theta_clim]`;
32 x 32 Fourier modes, width 128, three blocks, a bias-free local 3 x 3 path,
per-mode spectral normalization and the deterministic sine/cosine position
encoder. It differs from `{PARENT_VERSION}` step {PARENT_OPTIMIZER_STEP:,} only
in having been continued for {ROLLOUT_STEPS * 10} autonomous days per training
sequence instead of 60, at a constant 5e-5 for 1,920 further steps.

Each plate carries one model curve against persistence and climatology, exactly
as the parent's does. The parent-versus-fine-tune comparison is not plotted ---
that would mean changing the frozen plot definitions --- it is recorded in
`{GATE_NAME}` beside this folder, which reports the parent's published
day-2,000 magnitude and RMSE-to-climatology ratios next to this arm's and the
change between them.

Starts span {int(starts.min())}--{int(starts.max())} and are drawn from the
inference block {list(INFERENCE_RANGE)} nested inside validation
{list(VALIDATION_RANGE)}; there is no independent third test split. Every member
has lead-matched MITgcm truth through day 2,000.

This package performs no training, selection, or checkpoint promotion.

Report content SHA-256: `{report['report_content_sha256']}`.
"""


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    """Publish the six figures for the fine-tune's S0 inference protocol."""

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
        # Truth-only references, so the ACC panel can be read against the range
        # it can actually occupy. No model is involved in either curve.
        arrays.update(
            acc_reference_curves(
                state,
                regime_index,
                starts,
                climatology_state,
                climatology_derived,
                wet,
            )
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
            _plot_acc_with_reference(output_tmp, arrays)
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
                "training_version": TRAINING_VERSION,
                "parent_version": PARENT_VERSION,
                "parent_optimizer_step": PARENT_OPTIMIZER_STEP,
                "parent_day2000": parent_day2000(contract),
                "acc_day200": acc_day200_summary(arrays),
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
    """Verify sources, starts, lineage and checkpoint identity without plotting."""

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
        "training_version": TRAINING_VERSION,
        "parent_version": PARENT_VERSION,
        "parent_day2000": parent_day2000(contract),
        "comparator_model": None,
        "static_channels": list(STATIC_FEATURES),
        "selected_rollout_steps": ROLLOUT_STEPS,
        "selected_loss_contract_sha256": FINETUNE_LOSS_CONTRACT_SHA256,
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
    print(json.dumps(json_safe(result), indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
