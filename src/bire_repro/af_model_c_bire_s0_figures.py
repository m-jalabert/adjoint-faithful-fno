"""Six control-wind Bire-style figures for the selected ten-day Model C.

The evaluator combines the immutable ``trajectories_v2`` S0 record with the
evaluation-only year-120--126 continuation.  It rolls the prospectively
selected 15-member inference ensemble through day 2000 and refuses to
overwrite either its numerical archive or project output package.
"""

from __future__ import annotations

import argparse
import csv
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

from .af_a0_evaluate import _normalizers
from .af_data import (
    MODEL_DAY_STEPS,
    STATE_CHANNELS,
    DatasetSpec,
    _read_field_pair,
)
from .af_forward_complete import (
    _member_acc,
    _member_rmse,
    derived_fields,
)
from .af_model_c_anomaly_direct import PointwiseDirectStepper
from .af_model_c_bire_s0_long_truth import EXPECTED_STARTS
from .af_model_c_rollout_conditioned_loss_v3 import ProjectedIncrementModel
from .af_model_c_slow_field_bias_projection import wet_area_weights
from .af_model_c_successor import ModelCSuccessorArchitecture, build_successor
from .af_model_c_successor_validation import ValidationStepper

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


VERSION = "model_c_bire_s0_figures_v1"
CONTRACT_STATUS = "frozen_after_long_truth_and_before_figure_metrics"
LEAD_DAYS = tuple(range(0, 2001, 10))
SHORT_LEAD_DAYS = tuple(range(0, 201, 10))
FIGURE_3_LEADS = (0, 10, 20, 30, 40)
FIGURE_7_LEADS = (60, 2000)
RMSE_FIELDS = ("surface_speed", "phihyd_surface", "sst")
ACC_FIELDS = ("surface_u", "surface_v", "phihyd_surface", "sst")
METHODS = ("model", "climatology", "persistence")
FIGURE_NAMES = (
    "model_c_bire_figure3_streamfunction_1deg_s0_dt10.png",
    "model_c_bire_figure4_dt10_rmse_0_200_days_s0.png",
    "model_c_bire_figure5_dt10_single_member_rmse_s0.png",
    "model_c_bire_figure6_dt10_acc_0_200_days_s0.png",
    "model_c_bire_figure7_dt10_streamfunction_day060_day2000_s0.png",
    "model_c_bire_figure8_dt10_rmse_0_2000_days_s0.png",
)
REPORT_NAME = "model_c_bire_s0_figures_report.json"
ARRAYS_NAME = "model_c_bire_s0_figures_arrays.npz"
SUMMARY_NAME = "model_c_bire_s0_figures_summary.json"
CSV_NAME = "model_c_bire_s0_ensemble_curves.csv"
MANIFEST_NAME = "manifest.json"
README_NAME = "README.md"

FIELD_LABELS = {
    "surface_speed": r"Surface speed (m s$^{-1}$)",
    "phihyd_surface": r"Surface $P/\rho$ (m$^2$ s$^{-2}$)",
    "sst": r"SST ($^\circ$C)",
    "surface_u": r"Surface U (m s$^{-1}$)",
    "surface_v": r"Surface V (m s$^{-1}$)",
}
METHOD_LABELS = {
    "model": "Selected Model C",
    "climatology": "Climatology",
    "persistence": "Persistence",
}
METHOD_COLORS = {
    "model": "red",
    "climatology": "black",
    "persistence": "blue",
}


class ModelCBireS0FigureError(RuntimeError):
    """Raised when the frozen control-wind figure contract is violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def percentile_curve(values: np.ndarray) -> dict[str, np.ndarray]:
    """Return ensemble mean and the paper's 10th/90th percentiles."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != 15:
        raise ValueError("ensemble curves must have shape (15, leads)")
    return {
        "mean": np.mean(array, axis=0),
        "p10": np.percentile(array, 10.0, axis=0),
        "p90": np.percentile(array, 90.0, axis=0),
    }


def _finite_bound(values: Sequence[np.ndarray]) -> float:
    pieces = [
        np.abs(np.asarray(value, dtype=np.float64)).ravel()
        for value in values
    ]
    finite = np.concatenate(pieces)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 1.0
    return max(float(np.max(finite)), np.finfo(float).eps)


def _masked(value: np.ndarray, wet: np.ndarray) -> np.ma.MaskedArray:
    array = np.asarray(value)
    return np.ma.masked_where((~wet) | (~np.isfinite(array)), array)


def _verify_file(specification: Mapping[str, Any], label: str) -> Path:
    path = Path(str(specification["path"])).resolve()
    if not path.is_file():
        raise ModelCBireS0FigureError(f"{label} is missing: {path}")
    actual = file_sha256(path)
    if actual != specification["sha256"]:
        raise ModelCBireS0FigureError(
            f"{label} hash changed: expected {specification['sha256']}, got {actual}"
        )
    return path


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the figure contract frozen before any new model metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
    ):
        raise ModelCBireS0FigureError("S0 figure contract is not frozen")
    protocol = contract["protocol"]
    if (
        float(protocol.get("tau0_n_m2", -1.0)) != 0.1
        or int(protocol.get("prediction_interval_days", -1)) != 10
        or int(protocol.get("maximum_lead_days", -1)) != 2000
        or int(protocol.get("member_count", -1)) != 15
        or tuple(protocol.get("start_draw_order", ())) != EXPECTED_STARTS
        or int(protocol.get("single_member_start", -1)) != EXPECTED_STARTS[0]
        or tuple(protocol.get("figure_names", ())) != FIGURE_NAMES
        or tuple(protocol.get("rmse_fields", ())) != RMSE_FIELDS
        or tuple(protocol.get("acc_fields", ())) != ACC_FIELDS
    ):
        raise ModelCBireS0FigureError("S0 figure protocol changed")
    if contract["figure6"].get("literal_pretrain_finetune_pair") is not False:
        raise ModelCBireS0FigureError("Figure 6 comparison is mislabeled")
    if verify_sources:
        for label, specification in contract["artifacts"].items():
            _verify_file(specification, label)
        root = resolved.parents[1]
        for relative, expected in contract["source_hashes"].items():
            source = root / relative
            if not source.is_file() or file_sha256(source) != expected:
                raise ModelCBireS0FigureError(f"source changed: {relative}")
    return contract, resolved, file_sha256(resolved)


class ContinuousS0Truth:
    """Read S0 states across the Zarr/continuation boundary by global index."""

    def __init__(
        self,
        state: Any,
        run_dir: Path,
        wet: np.ndarray,
        *,
        base_records: int = 7200,
        first_extension_iteration: int = 3110400,
        extension_records: int = 2160,
    ) -> None:
        self.state = state
        self.run_dir = run_dir.resolve()
        self.wet = np.asarray(wet, dtype=bool)
        self.base_records = int(base_records)
        self.first_extension_iteration = int(first_extension_iteration)
        self.extension_records = int(extension_records)
        self.spec = DatasetSpec()
        if (
            state.shape[0] != 3
            or state.shape[1] != self.base_records
            or state.shape[2:] != (46, 62, 62)
            or self.wet.shape != (62, 62)
        ):
            raise ValueError("continuous truth received an incompatible base dataset")

    @property
    def total_records(self) -> int:
        return self.base_records + self.extension_records

    def extension_iteration(self, index: int) -> int:
        if not self.base_records <= index < self.total_records:
            raise IndexError("index is outside the evaluation-only continuation")
        return self.first_extension_iteration + (
            index - self.base_records
        ) * MODEL_DAY_STEPS

    def read(self, index: int) -> np.ndarray:
        if not 0 <= index < self.total_records:
            raise IndexError(f"truth index {index} is unavailable")
        if index < self.base_records:
            return np.asarray(self.state[0, index], dtype=np.float32)
        iteration = self.extension_iteration(index)
        value = _read_field_pair(
            self.run_dir / f"dynState.{iteration:010d}.meta",
            self.run_dir / f"surfState.{iteration:010d}.meta",
            self.spec,
        )
        value[:, ~self.wet] = 0.0
        if not np.all(np.isfinite(value)):
            raise ModelCBireS0FigureError(
                f"non-finite long truth at index {index}"
            )
        return np.ascontiguousarray(value, dtype=np.float32)

    def batch(self, indices: Sequence[int]) -> np.ndarray:
        return np.stack([self.read(int(index)) for index in indices])


def _s0_training_climatology(
    state: Any,
    snapshot_codes: np.ndarray,
    wet: np.ndarray,
    *,
    chunk_days: int = 60,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Pointwise S0 means from split-1 training snapshots only."""

    selected = np.flatnonzero(np.asarray(snapshot_codes) == 1)
    if not selected.size or chunk_days <= 0:
        raise ValueError("S0 climatology needs split-1 snapshots")
    cuts = np.flatnonzero(np.diff(selected) != 1) + 1
    blocks = np.split(selected, cuts)
    state_sum = np.zeros((46, *wet.shape), dtype=np.float64)
    derived_sum = {
        name: np.zeros(wet.shape, dtype=np.float64)
        for name in (
            "surface_speed",
            "phihyd_surface",
            "sst",
            "streamfunction",
        )
    }
    count = 0
    for block in blocks:
        start_block, stop_block = int(block[0]), int(block[-1]) + 1
        for start in range(start_block, stop_block, chunk_days):
            stop = min(start + chunk_days, stop_block)
            raw = np.asarray(state[0, start:stop], dtype=np.float32)
            state_sum += raw.sum(axis=0, dtype=np.float64)
            fields = derived_fields(raw, wet)
            for name in derived_sum:
                derived_sum[name] += fields[name].sum(
                    axis=0,
                    dtype=np.float64,
                )
            count += raw.shape[0]
    if count != selected.size:
        raise ModelCBireS0FigureError("S0 climatology count changed")
    state_mean = (state_sum / count).astype(np.float32)
    state_mean[:, ~wet] = 0.0
    derived_mean = {
        name: (value / count).astype(np.float32)
        for name, value in derived_sum.items()
    }
    for value in derived_mean.values():
        value[~wet] = 0.0
    return state_mean, derived_mean, int(count)


def _selected_stepper(
    contract: Mapping[str, Any],
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> PointwiseDirectStepper:
    checkpoint = Path(contract["artifacts"]["selected_checkpoint"]["path"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture_dict = contract["selected_model"]["architecture"]
    if (
        payload.get("architecture") != architecture_dict
        or int(payload.get("optimizer_step", -1)) != 13440
    ):
        raise ModelCBireS0FigureError("selected checkpoint identity changed")
    architecture = ModelCSuccessorArchitecture(**architecture_dict)
    model = build_successor(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    normalization_path = Path(
        contract["artifacts"]["selected_normalization"]["path"]
    )
    with np.load(normalization_path) as artifact:
        mean = np.asarray(artifact["pointwise_mean"], dtype=np.float32)
        scale = np.asarray(artifact["pointwise_scale"], dtype=np.float32)
    return PointwiseDirectStepper(
        model=model,
        device=device,
        wet=wet,
        mean=mean,
        scale=scale,
        wind_mean=wind_mean,
        wind_scale=wind_scale,
    )


def _prior_stepper(
    contract: Mapping[str, Any],
    device: Any,
    wet: np.ndarray,
    latitude: np.ndarray,
    static: Any,
    mean: np.ndarray,
    scale: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> ValidationStepper:
    checkpoint = Path(contract["artifacts"]["prior_checkpoint"]["path"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture_dict = contract["selected_model"]["architecture"]
    if payload.get("architecture") != architecture_dict:
        raise ModelCBireS0FigureError("prior checkpoint architecture changed")
    architecture = ModelCSuccessorArchitecture(**architecture_dict)
    raw_model = build_successor(architecture).to(device)
    raw_model.load_state_dict(payload["model_state_dict"])
    raw_model.eval()
    area_array = wet_area_weights(latitude, wet)
    area = torch.from_numpy(area_array)[None, None].to(
        device=device,
        dtype=torch.float32,
    )
    wet_tensor = torch.from_numpy(wet.astype(np.float32))[None, None].to(
        device=device,
        dtype=torch.float32,
    )
    target = np.asarray(
        payload["temperature_target_normalized"],
        dtype=np.float32,
    )
    wind = np.asarray(static[:, 0], dtype=np.float32)
    wind_signatures = (wind - wind_mean) / wind_scale
    wind_signatures[:, ~wet] = 0.0
    projected = ProjectedIncrementModel(
        raw_model,
        area,
        wet_tensor,
        torch.from_numpy(target).to(device=device, dtype=torch.float32),
        torch.from_numpy(wind_signatures).to(
            device=device,
            dtype=torch.float32,
        ),
    )
    projected.eval()
    return ValidationStepper(
        kind="successor",
        model=projected,
        device=device,
        wet=wet,
        mean=mean,
        scale=scale,
        wind_mean=wind_mean,
        wind_scale=wind_scale,
    )


def _evaluation_fields(states: np.ndarray, wet: np.ndarray) -> dict[str, np.ndarray]:
    fields = derived_fields(states, wet)
    fields["surface_u"] = np.asarray(states[:, 0], dtype=np.float32)
    fields["surface_v"] = np.asarray(states[:, 15], dtype=np.float32)
    return fields


def _evaluate(
    selected: PointwiseDirectStepper,
    prior: ValidationStepper,
    truth_source: ContinuousS0Truth,
    static: Any,
    starts: np.ndarray,
    climatology_state: np.ndarray,
    climatology_derived: Mapping[str, np.ndarray],
    wet: np.ndarray,
) -> dict[str, np.ndarray]:
    member_count = starts.size
    initial = truth_source.batch(starts)
    experiments = np.zeros(member_count, dtype=np.int64)
    selected_current = selected.normalized_state(initial)
    selected_static = selected.normalized_static(static, experiments)
    prior_current = prior.normalized_state(initial)
    prior_static = prior.normalized_static(static, experiments)
    initial_fields = _evaluation_fields(initial, wet)
    climate_fields = _evaluation_fields(
        np.repeat(climatology_state[None], member_count, axis=0),
        wet,
    )
    for name, value in climatology_derived.items():
        climate_fields[name] = np.repeat(
            value[None],
            member_count,
            axis=0,
        )

    arrays: dict[str, np.ndarray] = {
        "lead_days": np.asarray(LEAD_DAYS, dtype=np.int16),
        "short_lead_days": np.asarray(SHORT_LEAD_DAYS, dtype=np.int16),
        "start_draw_order": starts.astype(np.int32),
        "finite": np.empty((member_count, len(LEAD_DAYS)), dtype=np.uint8),
        "normalized_max_abs": np.empty(
            (member_count, len(LEAD_DAYS)),
            dtype=np.float32,
        ),
    }
    for method in METHODS:
        for field in RMSE_FIELDS:
            arrays[f"rmse__{method}__{field}"] = np.empty(
                (member_count, len(LEAD_DAYS)),
                dtype=np.float32,
            )
    for model_name in ("selected", "prior"):
        for field in ACC_FIELDS:
            arrays[f"acc__{model_name}__{field}"] = np.empty(
                (member_count, len(SHORT_LEAD_DAYS)),
                dtype=np.float32,
            )
    arrays["single_rmse__streamfunction"] = np.empty(
        len(SHORT_LEAD_DAYS),
        dtype=np.float32,
    )
    arrays["single_rmse__sst"] = np.empty_like(
        arrays["single_rmse__streamfunction"]
    )
    arrays["figure3_truth_streamfunction"] = np.empty(
        (len(FIGURE_3_LEADS), *wet.shape),
        dtype=np.float32,
    )
    arrays["figure3_model_streamfunction"] = np.empty_like(
        arrays["figure3_truth_streamfunction"]
    )
    arrays["figure7_truth_streamfunction"] = np.empty(
        (len(FIGURE_7_LEADS), *wet.shape),
        dtype=np.float32,
    )
    arrays["figure7_model_streamfunction"] = np.empty_like(
        arrays["figure7_truth_streamfunction"]
    )
    figure3_lookup = {
        lead: index for index, lead in enumerate(FIGURE_3_LEADS)
    }
    figure7_lookup = {
        lead: index for index, lead in enumerate(FIGURE_7_LEADS)
    }
    wet_tensor = torch.from_numpy(wet).to(selected.device)

    with torch.no_grad():
        for lead_index, lead in enumerate(LEAD_DAYS):
            if lead:
                selected_current = selected.step(
                    selected_current,
                    selected_static,
                )
                selected_prediction = selected.physical(selected_current)
                if lead <= 200:
                    prior_current = prior.step(prior_current, prior_static)
                    prior_prediction = prior.physical(prior_current)
            else:
                selected_prediction = initial.copy()
                prior_prediction = initial.copy()
            truth = truth_source.batch(starts + lead)
            truth_fields = _evaluation_fields(truth, wet)
            selected_fields = _evaluation_fields(selected_prediction, wet)

            for field in RMSE_FIELDS:
                arrays[f"rmse__model__{field}"][:, lead_index] = (
                    _member_rmse(
                        selected_fields[field],
                        truth_fields[field],
                        wet,
                    )
                )
                arrays[f"rmse__persistence__{field}"][:, lead_index] = (
                    _member_rmse(
                        initial_fields[field],
                        truth_fields[field],
                        wet,
                    )
                )
                arrays[f"rmse__climatology__{field}"][:, lead_index] = (
                    _member_rmse(
                        climate_fields[field],
                        truth_fields[field],
                        wet,
                    )
                )
            arrays["finite"][:, lead_index] = np.isfinite(
                selected_prediction
            ).all(axis=(1, 2, 3))
            arrays["normalized_max_abs"][:, lead_index] = (
                torch.amax(
                    torch.abs(selected_current[:, :, wet_tensor]),
                    dim=(1, 2),
                )
                .detach()
                .cpu()
                .numpy()
            )

            if lead <= 200:
                short_index = lead // 10
                prior_fields = _evaluation_fields(prior_prediction, wet)
                for field in ACC_FIELDS:
                    arrays[f"acc__selected__{field}"][:, short_index] = (
                        _member_acc(
                            selected_fields[field],
                            truth_fields[field],
                            climate_fields[field],
                            wet,
                        )
                    )
                    arrays[f"acc__prior__{field}"][:, short_index] = (
                        _member_acc(
                            prior_fields[field],
                            truth_fields[field],
                            climate_fields[field],
                            wet,
                        )
                    )
                arrays["single_rmse__streamfunction"][short_index] = (
                    _member_rmse(
                        selected_fields["streamfunction"],
                        truth_fields["streamfunction"],
                        wet,
                    )[0]
                )
                arrays["single_rmse__sst"][short_index] = _member_rmse(
                    selected_fields["sst"],
                    truth_fields["sst"],
                    wet,
                )[0]
            if lead in figure3_lookup:
                destination = figure3_lookup[lead]
                arrays["figure3_truth_streamfunction"][destination] = (
                    truth_fields["streamfunction"][0]
                )
                arrays["figure3_model_streamfunction"][destination] = (
                    selected_fields["streamfunction"][0]
                )
            if lead in figure7_lookup:
                destination = figure7_lookup[lead]
                arrays["figure7_truth_streamfunction"][destination] = (
                    truth_fields["streamfunction"][0]
                )
                arrays["figure7_model_streamfunction"][destination] = (
                    selected_fields["streamfunction"][0]
                )
    return arrays


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


def _plot_streamfunction_grid(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
) -> None:
    truth = np.asarray(arrays["figure3_truth_streamfunction"])
    model = np.asarray(arrays["figure3_model_streamfunction"])
    difference = truth - model
    bound = _finite_bound((truth, model))
    difference_bound = _finite_bound((difference,))
    figure, axes = plt.subplots(
        3,
        len(FIGURE_3_LEADS),
        figsize=(11.0, 6.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    state_image = difference_image = None
    for column, lead in enumerate(FIGURE_3_LEADS):
        state_image = axes[0, column].pcolormesh(
            longitude,
            latitude,
            _masked(truth[column], wet),
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
            shading="auto",
        )
        axes[1, column].pcolormesh(
            longitude,
            latitude,
            _masked(model[column], wet),
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
            shading="auto",
        )
        difference_image = axes[2, column].pcolormesh(
            longitude,
            latitude,
            _masked(difference[column], wet),
            cmap="RdBu_r",
            vmin=-difference_bound,
            vmax=difference_bound,
            shading="auto",
        )
        axes[0, column].set_title(f"Day {lead}")
        axes[2, column].set_xlabel("Longitude (°)")
    axes[0, 0].set_ylabel("MITgcm\nLatitude (°)")
    axes[1, 0].set_ylabel("Model C\nLatitude (°)")
    axes[2, 0].set_ylabel("Truth − model\nLatitude (°)")
    for axis in axes.flat:
        axis.set_aspect("equal")
        axis.set_facecolor("0.86")
    figure.colorbar(
        state_image,
        ax=axes[:2].ravel().tolist(),
        label="Barotropic streamfunction (Sv)",
        shrink=0.82,
    )
    figure.colorbar(
        difference_image,
        ax=axes[2].ravel().tolist(),
        label="Truth − model (Sv)",
        shrink=0.82,
    )
    figure.suptitle(
        r"Control wind $\tau_0=0.1$ N m$^{-2}$; "
        r"Model C $\Delta t=10$ days; native $1^\circ$ grid"
    )
    figure.savefig(output / FIGURE_NAMES[0], bbox_inches="tight")
    plt.close(figure)


def _plot_rmse(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    long: bool,
) -> None:
    leads = np.asarray(LEAD_DAYS if long else SHORT_LEAD_DAYS)
    limit = len(leads)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(5.4, 8.8),
        sharex=True,
        constrained_layout=True,
    )
    for axis, field in zip(axes, RMSE_FIELDS):
        for method in METHODS:
            summary = percentile_curve(
                arrays[f"rmse__{method}__{field}"][:, :limit]
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
                alpha=0.15,
                linewidth=0,
            )
        axis.set_ylabel(FIELD_LABELS[field])
        axis.grid(color="0.80", linewidth=0.6)
    axes[0].set_title(
        r"$\tau_0=0.1$ N m$^{-2}$; $\Delta t=10$ days; "
        "15 inference initial conditions"
    )
    axes[-1].set_xlabel("Time (days)")
    axes[-1].legend(loc="best")
    axes[-1].set_xlim(0, 2000 if long else 200)
    figure.savefig(
        output / (FIGURE_NAMES[5] if long else FIGURE_NAMES[1]),
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_single_member(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    leads = np.asarray(SHORT_LEAD_DAYS)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(5.4, 5.9),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(
        leads,
        arrays["single_rmse__streamfunction"],
        color="red",
        linewidth=1.7,
    )
    axes[1].plot(
        leads,
        arrays["single_rmse__sst"],
        color="red",
        linewidth=1.7,
    )
    axes[0].set_ylabel("Streamfunction RMSE (Sv)")
    axes[1].set_ylabel(r"SST RMSE ($^\circ$C)")
    axes[1].set_xlabel("Time (days)")
    for axis in axes:
        axis.grid(color="0.80", linewidth=0.6)
        axis.set_xlim(0, 200)
    axes[0].set_title(
        r"One S0 inference member; $\tau_0=0.1$ N m$^{-2}$; "
        r"$\Delta t=10$ days"
    )
    figure.savefig(output / FIGURE_NAMES[2], bbox_inches="tight")
    plt.close(figure)


def _plot_acc(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    leads = np.asarray(SHORT_LEAD_DAYS)
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(5.4, 10.2),
        sharex=True,
        constrained_layout=True,
    )
    for axis, field in zip(axes, ACC_FIELDS):
        selected = percentile_curve(arrays[f"acc__selected__{field}"])
        prior = percentile_curve(arrays[f"acc__prior__{field}"])
        axis.plot(
            leads,
            prior["mean"],
            color="black",
            linewidth=1.5,
            label="Prior residual Model C",
        )
        axis.plot(
            leads,
            prior["p10"],
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        axis.plot(
            leads,
            prior["p90"],
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        axis.plot(
            leads,
            selected["mean"],
            color="red",
            linewidth=1.6,
            label="Selected anomaly-direct Model C",
        )
        axis.fill_between(
            leads,
            selected["p10"],
            selected["p90"],
            color="red",
            alpha=0.17,
            linewidth=0,
        )
        axis.axhline(0.0, color="0.65", linewidth=0.6)
        axis.set_ylim(-1.0, 1.02)
        axis.set_ylabel(f"{FIELD_LABELS[field]}\nACC")
        axis.grid(color="0.82", linewidth=0.6)
    axes[0].set_title(
        r"S0 architecture-direction comparison; $\Delta t=10$ days"
    )
    axes[-1].set_xlabel("Time (days)")
    axes[-1].set_xlim(0, 200)
    axes[-1].legend(loc="best")
    figure.savefig(output / FIGURE_NAMES[3], bbox_inches="tight")
    plt.close(figure)


def _plot_day60_day2000(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
) -> None:
    truth = np.asarray(arrays["figure7_truth_streamfunction"])
    model = np.asarray(arrays["figure7_model_streamfunction"])
    bound = _finite_bound((truth, model))
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.0, 6.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for row, lead in enumerate(FIGURE_7_LEADS):
        for column, value in enumerate((truth[row], model[row])):
            image = axes[row, column].pcolormesh(
                longitude,
                latitude,
                _masked(value, wet),
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
                shading="auto",
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_facecolor("0.86")
            axes[row, column].set_xlabel("Longitude (°)")
        axes[row, 0].set_ylabel(f"Day {lead}\nLatitude (°)")
    axes[0, 0].set_title("MITgcm ground truth")
    axes[0, 1].set_title("Selected Model C")
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Barotropic streamfunction (Sv)",
        shrink=0.84,
    )
    figure.suptitle(
        r"Control wind $\tau_0=0.1$ N m$^{-2}$; "
        r"$\Delta t=10$ days"
    )
    figure.savefig(output / FIGURE_NAMES[4], bbox_inches="tight")
    plt.close(figure)


def _summary(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "complete",
        "member_count": 15,
        "lead_days": {"short": 200, "long": 2000, "interval": 10},
        "all_selected_states_finite": bool(np.all(arrays["finite"])),
        "maximum_selected_normalized_abs": _finite_bound(
            (arrays["normalized_max_abs"],)
        ),
        "rmse": {},
        "acc": {},
    }
    for field in RMSE_FIELDS:
        result["rmse"][field] = {}
        for method in METHODS:
            curve = percentile_curve(arrays[f"rmse__{method}__{field}"])
            result["rmse"][field][method] = {
                "day200_mean": float(curve["mean"][20]),
                "day2000_mean": float(curve["mean"][-1]),
                "day2000_p10": float(curve["p10"][-1]),
                "day2000_p90": float(curve["p90"][-1]),
            }
    for field in ACC_FIELDS:
        result["acc"][field] = {}
        for model_name in ("selected", "prior"):
            curve = percentile_curve(arrays[f"acc__{model_name}__{field}"])
            result["acc"][field][model_name] = {
                "day200_mean": float(curve["mean"][-1]),
                "minimum_mean_0_200": float(np.min(curve["mean"])),
            }
    return result


def _write_csv(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "metric",
                "model_or_method",
                "field",
                "lead_days",
                "mean",
                "p10",
                "p90",
            )
        )
        for field in RMSE_FIELDS:
            for method in METHODS:
                curve = percentile_curve(
                    arrays[f"rmse__{method}__{field}"]
                )
                for index, lead in enumerate(LEAD_DAYS):
                    writer.writerow(
                        (
                            "rmse",
                            method,
                            field,
                            lead,
                            curve["mean"][index],
                            curve["p10"][index],
                            curve["p90"][index],
                        )
                    )
        for field in ACC_FIELDS:
            for model_name in ("selected", "prior"):
                curve = percentile_curve(
                    arrays[f"acc__{model_name}__{field}"]
                )
                for index, lead in enumerate(SHORT_LEAD_DAYS):
                    writer.writerow(
                        (
                            "acc",
                            model_name,
                            field,
                            lead,
                            curve["mean"][index],
                            curve["p10"][index],
                            curve["p90"][index],
                        )
                    )


def _readme(report: Mapping[str, Any]) -> str:
    return f"""# Selected Model C: S0 Bire-style Figures 3--8

This package evaluates the frozen seed-20260724, step-13440 pointwise-anomaly
direct-state Model C under the control wind (tau0 = 0.1 N m-2). All forecasts
use a ten-day prediction interval.

The 15 initial conditions were randomly drawn without replacement from the
fresh late S0 inference block before model metrics. MITgcm truth through day
2000 combines the immutable trajectories-v2 prefix with evaluation-only job
304735; none of the continuation is training data.

Climatology is the pointwise temporal mean over S0 split-1 training snapshots.
Persistence holds each member's initial physical field fixed. RMSE is first
computed over wet cells for each member; lines/bands are the mean and
10th/90th percentiles across the 15 members.

Figure 6 compares the prior residual rollout-conditioned Model C with the
selected anomaly-direct Model C. It is an architecture-direction comparison,
not a literal pretrained/fine-tuned pairing.

Report content SHA-256: `{report["report_content_sha256"]}`.
"""


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Validate immutable sources and the complete long-truth chronology."""

    contract, path, digest = load_contract(contract_path)
    result = json.loads(
        Path(contract["artifacts"]["long_truth_result"]["path"]).read_text()
    )
    if (
        result.get("slurm_job_id") != "304735"
        or result.get("returncode") != 0
        or result.get("diagnostics")
        != {"dynState": 2160, "surfState": 2160}
        or int(result.get("start_iteration", -1)) != 3110400
        or int(result.get("end_iteration", -1)) != 3265920
        or float(result.get("tau0_n_m2", -1.0)) != 0.1
    ):
        raise ModelCBireS0FigureError("long-truth result is invalid")
    run_dir = Path(result["run_dir"])
    dyn = sorted(run_dir.glob("dynState.*.meta"))
    surf = sorted(run_dir.glob("surfState.*.meta"))
    if (
        len(dyn) != 2160
        or len(surf) != 2160
        or dyn[0].name != "dynState.0003110400.meta"
        or dyn[-1].name != "dynState.0003265848.meta"
        or max(EXPECTED_STARTS) + 2000 >= 7200 + len(dyn)
    ):
        raise ModelCBireS0FigureError("long-truth chronology is incomplete")
    return {
        "status": "pass",
        "version": VERSION,
        "contract": str(path),
        "contract_sha256": digest,
        "long_truth_job": 304735,
        "long_truth_records": 2160,
        "member_count": 15,
        "maximum_truth_index": max(EXPECTED_STARTS) + 2000,
        "available_maximum_index": 7200 + len(dyn) - 1,
        "figure_count": len(FIGURE_NAMES),
    }


def evaluate(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run the frozen ensemble, save numerical results, and make six figures."""

    if torch is None:
        raise RuntimeError("S0 figure evaluation requires PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    preflight(resolved_contract)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested without an available GPU")
    device = torch.device(device_name)

    scratch_output = Path(contract["output"]["scratch"]).resolve()
    project_output = Path(contract["output"]["project"]).resolve()
    scratch_temporary = scratch_output.with_name(scratch_output.name + ".tmp")
    project_temporary = project_output.with_name(project_output.name + ".tmp")
    if any(
        path.exists()
        for path in (
            scratch_output,
            project_output,
            scratch_temporary,
            project_temporary,
        )
    ):
        raise FileExistsError("refusing to overwrite S0 figure outputs")

    dataset_path = Path(contract["artifacts"]["dataset_metadata"]["path"]).parent
    group = zarr.open_consolidated(str(dataset_path), mode="r")
    if tuple(group.attrs["state_channels"]) != STATE_CHANNELS:
        raise ModelCBireS0FigureError("trajectory-v2 channels changed")
    state = group["state"]
    static = group["static_features"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    longitude = np.asarray(group["longitude_deg"][:], dtype=np.float32)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float32)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    mean, scale, _, _, wind_mean, wind_scale = _normalizers(group)

    result_path = Path(contract["artifacts"]["long_truth_result"]["path"])
    run_dir = Path(json.loads(result_path.read_text())["run_dir"])
    truth = ContinuousS0Truth(state, run_dir, wet)
    starts = np.asarray(EXPECTED_STARTS, dtype=np.int64)
    if max(starts) + 2000 >= truth.total_records:
        raise ModelCBireS0FigureError("day-2000 truth coverage changed")
    climatology_state, climatology_derived, climatology_days = (
        _s0_training_climatology(state, snapshot_codes, wet)
    )
    if climatology_days != int(contract["baselines"]["training_snapshots"]):
        raise ModelCBireS0FigureError("S0 training climatology count changed")
    selected = _selected_stepper(
        contract,
        device,
        wet,
        wind_mean,
        wind_scale,
    )
    prior = _prior_stepper(
        contract,
        device,
        wet,
        latitude,
        static,
        mean,
        scale,
        wind_mean,
        wind_scale,
    )
    arrays = _evaluate(
        selected,
        prior,
        truth,
        static,
        starts,
        climatology_state,
        climatology_derived,
        wet,
    )
    arrays["longitude_deg"] = longitude
    arrays["latitude_deg"] = latitude
    arrays["wet_mask"] = wet.astype(np.uint8)
    summary = _summary(arrays)
    scratch_temporary.parent.mkdir(parents=True, exist_ok=True)
    project_temporary.parent.mkdir(parents=True, exist_ok=True)
    scratch_temporary.mkdir(exist_ok=False)
    project_temporary.mkdir(exist_ok=False)
    try:
        scratch_arrays = scratch_temporary / ARRAYS_NAME
        np.savez_compressed(scratch_arrays, **arrays)
        _style()
        _plot_streamfunction_grid(
            project_temporary,
            arrays,
            longitude,
            latitude,
            wet,
        )
        _plot_rmse(project_temporary, arrays, long=False)
        _plot_single_member(project_temporary, arrays)
        _plot_acc(project_temporary, arrays)
        _plot_day60_day2000(
            project_temporary,
            arrays,
            longitude,
            latitude,
            wet,
        )
        _plot_rmse(project_temporary, arrays, long=True)
        _write_csv(project_temporary / CSV_NAME, arrays)
        (project_temporary / SUMMARY_NAME).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

        report = {
            "version": VERSION,
            "status": "complete",
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "long_truth_job": "304735",
            "long_truth_result": str(result_path),
            "long_truth_result_sha256": file_sha256(result_path),
            "selected_checkpoint": contract["artifacts"][
                "selected_checkpoint"
            ]["path"],
            "selected_checkpoint_sha256": contract["artifacts"][
                "selected_checkpoint"
            ]["sha256"],
            "prior_checkpoint": contract["artifacts"]["prior_checkpoint"][
                "path"
            ],
            "prior_checkpoint_sha256": contract["artifacts"][
                "prior_checkpoint"
            ]["sha256"],
            "dataset": str(dataset_path),
            "dataset_metadata_sha256": contract["artifacts"][
                "dataset_metadata"
            ]["sha256"],
            "device": str(device),
            "ensemble": contract["protocol"],
            "baselines": contract["baselines"],
            "figure6": contract["figure6"],
            "summary": summary,
            "arrays": str(scratch_output / ARRAYS_NAME),
            "arrays_sha256": file_sha256(scratch_arrays),
            "figures": list(FIGURE_NAMES),
            "elapsed_seconds": time.monotonic() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        report["report_content_sha256"] = json_sha256(report)
        (scratch_temporary / REPORT_NAME).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        shutil.copy2(
            scratch_temporary / REPORT_NAME,
            project_temporary / REPORT_NAME,
        )
        shutil.copy2(scratch_arrays, project_temporary / ARRAYS_NAME)
        (project_temporary / README_NAME).write_text(_readme(report))
        manifest = {
            "version": VERSION,
            "contract_sha256": contract_sha,
            "report_content_sha256": report["report_content_sha256"],
            "artifacts": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in sorted(project_temporary.iterdir())
                if path.is_file()
            },
        }
        manifest["manifest_content_sha256"] = json_sha256(manifest)
        (project_temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        scratch_temporary.replace(scratch_output)
        project_temporary.replace(project_output)
    except Exception:
        shutil.rmtree(scratch_temporary, ignore_errors=True)
        shutil.rmtree(project_temporary, ignore_errors=True)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--contract", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.contract)
    else:
        result = evaluate(args.contract, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
