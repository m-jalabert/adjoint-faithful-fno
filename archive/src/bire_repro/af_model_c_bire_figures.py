"""Bire-style Figure 3/4 characterization for the rejected Model C successor.

The evaluator uses one frozen Model C checkpoint and 15 prospectively selected
fresh-validation starts from S2.  It produces a 1-degree barotropic
streamfunction comparison and 0--200-day RMSE curves at the ten-day prediction
interval.  It is descriptive only and never reads inference or later archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.ticker import ScalarFormatter

from .af_a0_evaluate import _normalizers
from .af_data import STATE_CHANNELS
from .af_forward_complete import (
    _member_rmse,
    _state_fields,
    _training_climatology,
)
from .af_model_c_overfit import _file_sha256
from .af_model_c_successor_validation import (
    _climatology_batch_fields,
    _load_successor_stepper,
    load_validation_contract,
)

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


VERSION = "model_c_bire_figures_v1"
REPORT_NAME = "model_c_bire_figures_report.json"
ARRAYS_NAME = "model_c_bire_figures_arrays.npz"
SUMMARY_NAME = "bire_figure_summary.json"
MANIFEST_NAME = "figure_manifest.json"
README_NAME = "README.md"
FIGURE_3_NAME = "model_c_bire_figure3_streamfunction_1deg.png"
FIGURE_4_NAME = "model_c_bire_figure4_dt10_rmse_0_200_days.png"
FIGURE_NAMES = (FIGURE_3_NAME, FIGURE_4_NAME)
LEAD_DAYS = tuple(range(0, 201, 10))
FIGURE_3_LEADS = (0, 10, 20, 30, 40)
FIELDS = ("surface_speed", "sst", "phihyd_surface")
FIELD_LABELS = {
    "surface_speed": r"Speed (m s$^{-1}$)",
    "sst": r"SST ($^\circ$C)",
    "phihyd_surface": r"$P/\rho$ (m$^2$ s$^{-2}$)",
}
AXIS_LIMITS = {
    "surface_speed": (0.0, 0.12),
    "sst": (0.0, 0.30),
    "phihyd_surface": (0.0, 1.0),
}
METHODS = ("model", "climatology", "persistence")
METHOD_LABELS = {
    "model": "Prediction",
    "climatology": "Climatology",
    "persistence": "Persistence",
}
METHOD_COLORS = {
    "model": "red",
    "climatology": "black",
    "persistence": "blue",
}


class ModelCBireFigureError(RuntimeError):
    """Raised when the frozen descriptive-figure contract is violated."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def complete_figure_starts(
    pair_codes: np.ndarray,
    snapshot_codes: np.ndarray,
    *,
    pair_code: int = 2,
    maximum_lead_days: int = 200,
) -> np.ndarray:
    """Return starts with complete ten-day pairs and truth through maximum lead."""

    pairs = np.asarray(pair_codes, dtype=np.uint8)
    snapshots = np.asarray(snapshot_codes, dtype=np.uint8)
    if (
        pairs.ndim != 1
        or snapshots.ndim != 1
        or maximum_lead_days <= 0
        or maximum_lead_days % 10
    ):
        raise ValueError("invalid split arrays or maximum lead")
    offsets = np.arange(maximum_lead_days // 10 + 1, dtype=np.int64) * 10
    selected = []
    for start in range(pairs.size):
        pair_indices = start + offsets[:-1]
        snapshot_indices = start + offsets
        if (
            pair_indices[-1] < pairs.size
            and snapshot_indices[-1] < snapshots.size
            and np.all(pairs[pair_indices] == pair_code)
            and np.all(snapshots[snapshot_indices] == pair_code)
        ):
            selected.append(start)
    return np.asarray(selected, dtype=np.int64)


def select_ensemble_starts(
    candidates: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    """Select starts without replacement in deterministic RNG draw order."""

    values = np.asarray(candidates, dtype=np.int64)
    if values.ndim != 1 or count <= 0 or values.size < count:
        raise ValueError("not enough candidate starts")
    rng = np.random.default_rng(seed)
    return np.asarray(
        rng.choice(values, size=count, replace=False),
        dtype=np.int64,
    )


def percentile_curve(values: np.ndarray) -> dict[str, np.ndarray]:
    """Return the paper's ensemble mean and 10th/90th percentiles."""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("curve values must have member and lead dimensions")
    return {
        "mean": np.mean(data, axis=0),
        "p10": np.percentile(data, 10.0, axis=0),
        "p90": np.percentile(data, 90.0, axis=0),
    }


def load_figure_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the contract frozen before 100--200-day validation metrics."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status")
        != "frozen_before_bire_style_100_to_200_day_validation_metrics"
    ):
        raise ValueError("Bire-style figure contract is not frozen")
    ensemble = contract.get("ensemble", {})
    if (
        int(ensemble.get("regime_index", -1)) != 2
        or int(ensemble.get("validation_pair_code", -1)) != 2
        or int(ensemble.get("member_count", -1)) != 15
        or tuple(ensemble.get("lead_days", ())) != LEAD_DAYS
        or tuple(ensemble.get("start_draw_order", ()))
        != (
            6335,
            6353,
            6330,
            6361,
            6358,
            6308,
            6313,
            6346,
            6324,
            6323,
            6319,
            6325,
            6355,
            6366,
            6351,
        )
        or int(ensemble.get("figure3_start", -1)) != 6335
    ):
        raise ValueError("Bire-style ensemble contract changed")
    figure4 = contract.get("figure4", {})
    if (
        tuple(figure4.get("fields", ())) != FIELDS
        or figure4.get("summary") != "mean_p10_p90_over_15_members"
        or figure4.get("axis_limits") != {
            "surface_speed": [0.0, 0.12],
            "sst": [0.0, 0.3],
            "phihyd_surface": [0.0, 1.0],
        }
    ):
        raise ValueError("Bire-style Figure 4 contract changed")
    read = contract.get("read_contract", {})
    if (
        read.get("training_state_for_climatology") is not True
        or read.get("fresh_validation_state") is not True
        or any(
            read.get(name) is not False
            for name in (
                "inference_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
            )
        )
    ):
        raise ValueError("Bire-style read contract changed")
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(f"Bire-style figure source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def _verify_external_sources(
    contract: Mapping[str, Any],
    dataset: Path,
    quality_path: Path,
    validation_report_path: Path,
) -> dict[str, Any]:
    source = contract["source_artifacts"]
    if (
        _file_sha256(dataset / ".zmetadata")
        != source["dataset_metadata_sha256"]
        or _file_sha256(quality_path) != source["quality_report_sha256"]
        or _file_sha256(validation_report_path)
        != source["fresh_validation_report_sha256"]
        or _file_sha256(Path(source["checkpoint"]))
        != source["checkpoint_sha256"]
    ):
        raise ModelCBireFigureError("Bire-style source artifact changed")
    quality = json.loads(quality_path.read_text())
    validation_report = json.loads(validation_report_path.read_text())
    validation_content = dict(validation_report)
    expected_content_hash = validation_content.pop(
        "report_content_sha256",
        None,
    )
    if (
        quality.get("status") != "valid"
        or quality.get("inference_state_metrics_read") is not False
        or validation_report.get("validation_gate", {}).get("status")
        != "scientifically_rejected_fresh_v2_validation"
        or validation_report.get("inference_opened") is not False
        or expected_content_hash != _json_sha256(validation_content)
    ):
        raise ModelCBireFigureError("source evidence is not sealed and valid")
    return validation_report


def _evaluate_curves(
    stepper: Any,
    state: Any,
    static: Any,
    starts: np.ndarray,
    climatology_state: np.ndarray,
    climatology_derived: Mapping[str, np.ndarray],
    wet: np.ndarray,
    figure3_start: int,
) -> dict[str, np.ndarray]:
    experiments = np.full(starts.size, 2, dtype=np.int64)
    initial = np.stack(
        [
            np.asarray(state[2, int(start)], dtype=np.float32)
            for start in starts
        ]
    )
    current = stepper.normalized_state(initial)
    forcing = stepper.normalized_static(static, experiments)
    initial_fields = _state_fields(initial, wet)
    climate_fields = _climatology_batch_fields(
        experiments,
        climatology_state,
        climatology_derived,
        wet,
    )
    result: dict[str, np.ndarray] = {
        "lead_days": np.asarray(LEAD_DAYS, dtype=np.int16),
        "start_draw_order": starts.astype(np.int32),
        "finite": np.empty((starts.size, len(LEAD_DAYS)), dtype=np.uint8),
        "normalized_max_abs": np.empty(
            (starts.size, len(LEAD_DAYS)),
            dtype=np.float32,
        ),
    }
    for method in METHODS:
        for field in FIELDS:
            result[f"{method}__rmse__{field}"] = np.empty(
                (starts.size, len(LEAD_DAYS)),
                dtype=np.float32,
            )
    figure3_member = int(np.flatnonzero(starts == figure3_start)[0])
    figure3_indices = {lead: FIGURE_3_LEADS.index(lead) for lead in FIGURE_3_LEADS}
    result["figure3_truth_streamfunction"] = np.empty(
        (len(FIGURE_3_LEADS), *wet.shape),
        dtype=np.float32,
    )
    result["figure3_prediction_streamfunction"] = np.empty_like(
        result["figure3_truth_streamfunction"]
    )
    wet_t = torch.from_numpy(wet).to(stepper.device)
    with torch.no_grad():
        for lead_index, lead in enumerate(LEAD_DAYS):
            if lead:
                current = stepper.step(current, forcing)
                prediction = stepper.physical(current)
            else:
                prediction = initial.copy()
            truth = np.stack(
                [
                    np.asarray(
                        state[2, int(start) + lead],
                        dtype=np.float32,
                    )
                    for start in starts
                ]
            )
            predicted_fields = _state_fields(prediction, wet)
            truth_fields = _state_fields(truth, wet)
            for field in FIELDS:
                result[f"model__rmse__{field}"][:, lead_index] = (
                    _member_rmse(
                        predicted_fields[field],
                        truth_fields[field],
                        wet,
                    )
                )
                result[f"persistence__rmse__{field}"][:, lead_index] = (
                    _member_rmse(
                        initial_fields[field],
                        truth_fields[field],
                        wet,
                    )
                )
                result[f"climatology__rmse__{field}"][:, lead_index] = (
                    _member_rmse(
                        climate_fields[field],
                        truth_fields[field],
                        wet,
                    )
                )
            result["finite"][:, lead_index] = np.isfinite(prediction).all(
                axis=(1, 2, 3)
            )
            result["normalized_max_abs"][:, lead_index] = (
                torch.amax(torch.abs(current[:, :, wet_t]), dim=(1, 2))
                .detach()
                .cpu()
                .numpy()
            )
            if lead in figure3_indices:
                destination = figure3_indices[lead]
                result["figure3_truth_streamfunction"][destination] = (
                    truth_fields["streamfunction"][figure3_member, 0]
                )
                result["figure3_prediction_streamfunction"][destination] = (
                    predicted_fields["streamfunction"][figure3_member, 0]
                )
    return result


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )


def _plot_figure3(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
) -> None:
    truth = np.asarray(
        arrays["figure3_truth_streamfunction"],
        dtype=np.float64,
    )
    prediction = np.asarray(
        arrays["figure3_prediction_streamfunction"],
        dtype=np.float64,
    )
    difference = truth - prediction
    bound = float(np.max(np.abs(np.concatenate((truth, prediction)))))
    bound = max(bound, np.finfo(float).eps)
    figure, axes = plt.subplots(
        3,
        5,
        figsize=(10.5, 6.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    title_letters = (
        ("a", "b", "c", "d", "e"),
        ("", "f", "g", "h", "i"),
        ("", "j", "k", "l", "m"),
    )
    image = None
    for column, lead in enumerate(FIGURE_3_LEADS):
        image = axes[0, column].pcolormesh(
            longitude,
            latitude,
            np.ma.masked_where(~wet, truth[column]),
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
            shading="auto",
        )
        axes[0, column].set_title(
            f"({title_letters[0][column]}) GT (day={lead})"
        )
        if column == 0:
            axes[1, column].axis("off")
            axes[2, column].axis("off")
            continue
        axes[1, column].pcolormesh(
            longitude,
            latitude,
            np.ma.masked_where(~wet, prediction[column]),
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
            shading="auto",
        )
        axes[1, column].set_title(f"({title_letters[1][column]})")
        axes[2, column].pcolormesh(
            longitude,
            latitude,
            np.ma.masked_where(~wet, difference[column]),
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
            shading="auto",
        )
        axes[2, column].set_title(f"({title_letters[2][column]})")
    axes[0, 0].set_ylabel("Lat (°)")
    axes[1, 1].set_ylabel("Pred")
    axes[2, 1].set_ylabel("GT − Pred")
    for column in range(1, 5):
        axes[2, column].set_xlabel("Lon (°)")
    for axis in axes.flat:
        if axis.axison:
            axis.set_aspect("equal")
            axis.set_facecolor("0.86")
    if image is None:  # pragma: no cover - construction invariant
        raise RuntimeError("Figure 3 has no image")
    figure.colorbar(
        image,
        ax=axes[:, 1:].ravel().tolist(),
        label="Barotropic streamfunction (Sv)",
        shrink=0.82,
    )
    figure.suptitle("Model C barotropic streamfunction at 1° (Δt = 10 days)")
    figure.savefig(output / FIGURE_3_NAME, bbox_inches="tight")
    plt.close(figure)


def _scientific_minus_one(axis: Any) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, -1))
    axis.yaxis.set_major_formatter(formatter)


def _plot_figure4(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    leads = np.asarray(LEAD_DAYS)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(5.1, 8.6),
        sharex=True,
        constrained_layout=True,
    )
    for axis, field in zip(axes, FIELDS):
        for method in METHODS:
            summary = percentile_curve(
                arrays[f"{method}__rmse__{field}"]
            )
            color = METHOD_COLORS[method]
            axis.plot(
                leads,
                summary["mean"],
                color=color,
                linewidth=1.6,
                label=METHOD_LABELS[method],
            )
            axis.fill_between(
                leads,
                summary["p10"],
                summary["p90"],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        axis.set_ylim(*AXIS_LIMITS[field])
        axis.set_xlim(0, 200)
        axis.set_ylabel(FIELD_LABELS[field])
        axis.set_xticks((0, 50, 100, 150, 200))
        axis.grid(color="0.75", linewidth=0.6)
        if field in ("surface_speed", "sst"):
            _scientific_minus_one(axis)
    axes[0].set_title(r"$\Delta t = 10$ days")
    axes[-1].set_xlabel("Time (days)")
    axes[-1].legend(loc="lower right")
    figure.savefig(output / FIGURE_4_NAME, bbox_inches="tight")
    plt.close(figure)


def _metric_summary(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in FIELDS:
        result[field] = {}
        upper = AXIS_LIMITS[field][1]
        for method in METHODS:
            values = np.asarray(
                arrays[f"{method}__rmse__{field}"],
                dtype=np.float64,
            )
            curves = percentile_curve(values)
            result[field][method] = {
                "day200_mean": float(curves["mean"][-1]),
                "day200_p10": float(curves["p10"][-1]),
                "day200_p90": float(curves["p90"][-1]),
                "maximum_mean": float(np.max(curves["mean"])),
                "maximum_p90": float(np.max(curves["p90"])),
                "mean_curve_exceeds_requested_axis": bool(
                    np.any(curves["mean"] > upper)
                ),
                "p90_curve_exceeds_requested_axis": bool(
                    np.any(curves["p90"] > upper)
                ),
            }
    return result


def _project_readme(report: Mapping[str, Any], manifest_hash: str) -> str:
    return f"""# Model C Bire-style Figures 3 and 4

Status: complete descriptive characterization of the scientifically rejected
Model C successor. This package cannot authorize tuning or inference.

`{FIGURE_3_NAME}` follows Bire Figure 3 at the project's native 1-degree
resolution: MITgcm truth, Model C prediction, and truth-minus-prediction
barotropic streamfunction at 0--40 days. The prospectively selected member is
S2 validation start {report["ensemble"]["figure3_start"]}.

`{FIGURE_4_NAME}` follows the middle column of Bire Figure 4 for
delta_t = 10 days. It uses 15 prospectively selected S2 fresh-validation
initial conditions and reports the member mean with 10th--90th percentile
shading from 0 to 200 days.

Persistence repeats each member's initial condition at every lead. Climatology
is the S2 pointwise time mean over all split-1 training snapshots; nonlinear
derived fields are time-averaged after derivation. This is the paper's
mathematical definition with a stricter training-only, leakage-free source.

The complete report and arrays remain immutable in scratch. Figure-manifest
content SHA-256: `{manifest_hash}`.
"""


def evaluate_bire_figures(
    dataset_path: str | Path,
    quality_report_path: str | Path,
    validation_report_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    figure_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate the frozen 15-member characterization and save both figures."""

    if torch is None:
        raise RuntimeError("Bire-style figures require PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_figure_contract(
        contract_path
    )
    dataset = Path(dataset_path).resolve()
    quality_path = Path(quality_report_path).resolve()
    validation_report_path = Path(validation_report_path).resolve()
    output = Path(output_dir).resolve()
    figures = Path(figure_dir).resolve()
    expected_output = contract["output_contract"]
    if (
        output != Path(expected_output["scratch_output"]).resolve()
        or figures != Path(expected_output["project_output"]).resolve()
    ):
        raise ModelCBireFigureError("Bire-style output path changed")
    temporary_output = output.with_name(output.name + ".tmp")
    temporary_figures = figures.with_name(figures.name + ".tmp")
    if any(
        path.exists()
        for path in (output, figures, temporary_output, temporary_figures)
    ):
        raise FileExistsError("refusing to overwrite Bire-style figure output")
    validation_report = _verify_external_sources(
        contract,
        dataset,
        quality_path,
        validation_report_path,
    )
    validation_contract, _, validation_contract_sha = (
        load_validation_contract(
            contract["source_artifacts"]["successor_validation_contract"]
        )
    )
    if (
        validation_contract_sha
        != contract["source_artifacts"][
            "successor_validation_contract_sha256"
        ]
    ):
        raise ModelCBireFigureError("successor validation contract changed")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA characterization requested without a GPU")
    device = torch.device(device_name)

    group = zarr.open_consolidated(str(dataset), mode="r")
    if tuple(group.attrs["state_channels"]) != STATE_CHANNELS:
        raise ModelCBireFigureError("trajectory-v2 channels changed")
    state = group["state"]
    static = group["static_features"]
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(
        group["snapshot_split"][:],
        dtype=np.uint8,
    )
    candidates = complete_figure_starts(pair_codes, snapshot_codes)
    ensemble = contract["ensemble"]
    starts = select_ensemble_starts(
        candidates,
        count=int(ensemble["member_count"]),
        seed=int(ensemble["selection_seed"]),
    )
    if (
        candidates.size != int(ensemble["candidate_count"])
        or int(candidates[0]) != int(ensemble["candidate_bounds"][0])
        or int(candidates[-1]) != int(ensemble["candidate_bounds"][1])
        or _array_sha256(candidates) != ensemble["candidate_times_sha256"]
        or tuple(int(value) for value in starts)
        != tuple(ensemble["start_draw_order"])
        or _array_sha256(starts) != ensemble["start_draw_order_sha256"]
    ):
        raise ModelCBireFigureError("validation start selection changed")

    mean, scale, wet, _, wind_mean, wind_scale = _normalizers(group)
    climatology_state, climatology_derived, training_days = (
        _training_climatology(state, snapshot_codes, wet)
    )
    if training_days != int(contract["baselines"]["training_snapshots"]):
        raise ModelCBireFigureError("training climatology count changed")
    source = contract["source_artifacts"]
    stepper, payload = _load_successor_stepper(
        Path(source["checkpoint"]),
        device,
        wet,
        mean,
        scale,
        wind_mean,
        wind_scale,
        validation_contract["architecture"],
    )
    if int(payload.get("seed", -1)) != int(source["checkpoint_seed"]):
        raise ModelCBireFigureError("checkpoint seed changed")
    arrays = _evaluate_curves(
        stepper,
        state,
        static,
        starts,
        climatology_state,
        climatology_derived,
        wet,
        int(ensemble["figure3_start"]),
    )
    arrays["longitude_deg"] = np.asarray(
        group["longitude_deg"][:],
        dtype=np.float32,
    )
    arrays["latitude_deg"] = np.asarray(
        group["latitude_deg"][:],
        dtype=np.float32,
    )
    arrays["wet_mask"] = wet.astype(np.uint8)

    temporary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_figures.parent.mkdir(parents=True, exist_ok=True)
    temporary_output.mkdir(exist_ok=False)
    temporary_figures.mkdir(exist_ok=False)
    try:
        arrays_path = temporary_output / ARRAYS_NAME
        np.savez_compressed(arrays_path, **arrays)
        _style()
        _plot_figure3(
            temporary_figures,
            arrays,
            arrays["longitude_deg"],
            arrays["latitude_deg"],
            wet,
        )
        _plot_figure4(temporary_figures, arrays)
        metrics = _metric_summary(arrays)
        report = {
            "version": VERSION,
            "status": "complete",
            "purpose": "descriptive_bire_style_rejected_model_characterization",
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "dataset": str(dataset),
            "dataset_metadata_sha256": _file_sha256(
                dataset / ".zmetadata"
            ),
            "fresh_validation_report": str(validation_report_path),
            "fresh_validation_report_sha256": _file_sha256(
                validation_report_path
            ),
            "fresh_validation_report_content_sha256": validation_report[
                "report_content_sha256"
            ],
            "checkpoint": source["checkpoint"],
            "checkpoint_sha256": source["checkpoint_sha256"],
            "checkpoint_seed": int(source["checkpoint_seed"]),
            "device": str(device),
            "elapsed_seconds": time.monotonic() - started,
            "ensemble": {
                "regime": "S2",
                "member_count": int(starts.size),
                "lead_days": list(LEAD_DAYS),
                "start_draw_order": [int(value) for value in starts],
                "figure3_start": int(ensemble["figure3_start"]),
            },
            "baselines": {
                "persistence": (
                    "each_member_initial_condition_repeated_at_every_lead"
                ),
                "climatology": (
                    "regime_specific_pointwise_mean_of_split1_training_truth"
                ),
                "training_snapshots": training_days,
                "leakage_free": True,
            },
            "metric_summary": metrics,
            "all_predictions_finite": bool(
                np.all(np.asarray(arrays["finite"], dtype=bool))
            ),
            "maximum_normalized_abs": float(
                np.max(arrays["normalized_max_abs"])
            ),
            "arrays": str(output / ARRAYS_NAME),
            "arrays_sha256": _file_sha256(arrays_path),
            "figures": {
                name: {
                    "path": str(figures / name),
                    "sha256": _file_sha256(temporary_figures / name),
                }
                for name in FIGURE_NAMES
            },
            "read_contract": {
                "training_state_for_climatology": True,
                "fresh_validation_state": True,
                "inference_state": False,
                "intermediate_wind_state": False,
                "response_state": False,
                "adjoint_state": False,
            },
            "inference_opened": False,
            "tuning_authorized": False,
        }
        report["report_content_sha256"] = _json_sha256(report)
        report_path = temporary_output / REPORT_NAME
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        summary = {
            "version": VERSION,
            "status": "complete",
            "report": str(output / REPORT_NAME),
            "report_sha256": _file_sha256(report_path),
            "report_content_sha256": report["report_content_sha256"],
            "metric_summary": metrics,
            "all_predictions_finite": report["all_predictions_finite"],
            "maximum_normalized_abs": report["maximum_normalized_abs"],
            "inference_opened": False,
            "tuning_authorized": False,
        }
        summary["summary_content_sha256"] = _json_sha256(summary)
        summary_path = temporary_figures / SUMMARY_NAME
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "version": VERSION,
            "status": "complete",
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "report": str(output / REPORT_NAME),
            "report_sha256": _file_sha256(report_path),
            "report_content_sha256": report["report_content_sha256"],
            "arrays": str(output / ARRAYS_NAME),
            "arrays_sha256": report["arrays_sha256"],
            "summary": str(figures / SUMMARY_NAME),
            "summary_sha256": _file_sha256(summary_path),
            "figures": report["figures"],
            "inference_opened": False,
            "tuning_authorized": False,
        }
        manifest["manifest_content_sha256"] = _json_sha256(manifest)
        (temporary_figures / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (temporary_figures / README_NAME).write_text(
            _project_readme(report, manifest["manifest_content_sha256"])
        )
        os.replace(temporary_output, output)
        os.replace(temporary_figures, figures)
    except Exception:
        shutil.rmtree(temporary_output, ignore_errors=True)
        shutil.rmtree(temporary_figures, ignore_errors=True)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_bire_figures(
        args.dataset,
        args.quality_report,
        args.validation_report,
        args.contract,
        args.output_dir,
        args.figure_dir,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
