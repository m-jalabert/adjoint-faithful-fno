"""Zero-retraining S0 boundary-stability diagnosis across Model C checkpoints.

The diagnostic was motivated by the boundary-localized day-2000 streamfunction
error in the frozen Bire-style Figure 7.  It does not select or modify weights.
It applies every stored seed-20260724 anomaly-direct checkpoint to the same 15
fresh S0 inference initial conditions and evaluation-only truth used by job
304736.
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
from .af_data import STATE_CHANNELS
from .af_model_c_anomaly_direct import PointwiseDirectStepper
from .af_model_c_bire_s0_figures import ContinuousS0Truth, EXPECTED_STARTS
from .af_model_c_successor import (
    ModelCSuccessorArchitecture,
    build_successor,
)
from .af_tutorial_analysis import DRF_M, EARTH_RADIUS_M

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


VERSION = "model_c_bire_s0_boundary_checkpoint_v1"
CHECKPOINT_STEPS = (3840, 7680, 11520, 13440, 14400, 14880, 15360)
SELECTED_STEP = 13440
LEAD_DAYS = tuple(range(0, 2001, 10))
SNAPSHOT_LEADS = (200, 500, 1000, 1500, 2000)
REGION_NAMES = (
    "wet",
    "boundary",
    "west",
    "east",
    "south",
    "north",
    "interior",
)
NORMAL_REGIONS = ("west", "east", "south", "north")
TRANSPORTS = ("qx", "qy")
FIELDS = ("qx", "qy", "streamfunction")
THRESHOLDS = (20.0, 40.0, 100.0, 200.0)
GROWTH_FACTORS = (2.0, 5.0, 10.0)

ARRAYS_NAME = "model_c_s0_boundary_checkpoint_arrays.npz"
REPORT_NAME = "model_c_s0_boundary_checkpoint_report.json"
SUMMARY_NAME = "model_c_s0_boundary_checkpoint_summary.json"
CSV_NAME = "model_c_s0_boundary_checkpoint_curves.csv"
README_NAME = "README.md"
MANIFEST_NAME = "manifest.json"
FIGURE_NAMES = (
    "model_c_s0_boundary_transport_rmse_checkpoints.png",
    "model_c_s0_boundary_fraction_checkpoints.png",
    "model_c_s0_checkpoint_stability_timing.png",
    "model_c_s0_selected_boundary_spatial_consistency.png",
    "model_c_s0_selected_boundary_growth.png",
)

U = slice(0, 15)
V = slice(15, 30)


class ModelCBoundaryCheckpointError(RuntimeError):
    """Raised when the frozen diagnostic contract is violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def region_masks(wet: np.ndarray, band_cells: int = 4) -> dict[str, np.ndarray]:
    """Return fixed boundary strips and the disjoint boundary/interior masks."""

    wet = np.asarray(wet, dtype=bool)
    rows, columns = np.where(wet)
    if (
        wet.ndim != 2
        or not rows.size
        or band_cells <= 0
        or 2 * band_cells >= min(np.ptp(rows) + 1, np.ptp(columns) + 1)
    ):
        raise ValueError("invalid wet mask or boundary-band width")
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    yy, xx = np.indices(wet.shape)
    west = wet & (xx < x0 + band_cells)
    east = wet & (xx >= x1 - band_cells)
    south = wet & (yy < y0 + band_cells)
    north = wet & (yy >= y1 - band_cells)
    boundary = west | east | south | north
    interior = wet & ~boundary
    result = {
        "wet": wet,
        "boundary": boundary,
        "west": west,
        "east": east,
        "south": south,
        "north": north,
        "interior": interior,
    }
    if not all(np.any(result[name]) for name in REGION_NAMES):
        raise ValueError("one or more boundary regions are empty")
    return result


def barotropic_transports(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return depth-integrated zonal/meridional transport in m2 s-1."""

    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (46, 62, 62):
        raise ValueError("barotropic transport received an invalid state")
    thickness = np.asarray(DRF_M, dtype=np.float64)[None, :, None, None]
    qx = np.sum(values[:, U] * thickness, axis=1)
    qy = np.sum(values[:, V] * thickness, axis=1)
    return qx, qy


def streamfunction_from_qx(
    qx: np.ndarray,
    wet: np.ndarray,
) -> np.ndarray:
    """Return the south-to-north transport streamfunction in Sv."""

    value = np.asarray(qx, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != wet.shape:
        raise ValueError("streamfunction received an invalid transport")
    dy_m = EARTH_RADIUS_M * np.deg2rad(1.0)
    result = np.cumsum(-value * dy_m, axis=1) / 1.0e6
    result[:, ~wet] = 0.0
    return result


def member_rmse(
    error: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Wet-cell RMSE for each member from a precomputed error field."""

    selected = np.asarray(error, dtype=np.float64)[:, mask]
    with np.errstate(over="ignore", invalid="ignore"):
        return np.sqrt(np.mean(np.square(selected), axis=1))


def member_boundary_fraction(
    error: np.ndarray,
    boundary: np.ndarray,
    wet: np.ndarray,
) -> np.ndarray:
    """Fraction of memberwise squared error inside the boundary union."""

    values = np.asarray(error, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        numerator = np.sum(np.square(values[:, boundary]), axis=1)
        denominator = np.sum(np.square(values[:, wet]), axis=1)
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )


def first_crossing(
    curve: Sequence[float],
    leads: Sequence[int],
    threshold: float,
    *,
    start_index: int = 0,
) -> int | None:
    """First lead at or after start_index with finite value above threshold."""

    values = np.asarray(curve, dtype=np.float64)
    lead_array = np.asarray(leads, dtype=np.int64)
    if values.shape != lead_array.shape:
        raise ValueError("crossing curve and leads differ")
    candidates = np.flatnonzero(
        np.isfinite(values[start_index:])
        & (values[start_index:] > threshold)
    )
    if not candidates.size:
        return None
    return int(lead_array[start_index + int(candidates[0])])


def growth_crossing(
    curve: Sequence[float],
    leads: Sequence[int],
    factor: float,
    *,
    baseline_day: int = 200,
) -> int | None:
    """First post-baseline lead exceeding factor times the baseline value."""

    lead_array = np.asarray(leads, dtype=np.int64)
    matches = np.flatnonzero(lead_array == baseline_day)
    if matches.size != 1 or factor <= 1.0:
        raise ValueError("invalid growth-crossing definition")
    index = int(matches[0])
    values = np.asarray(curve, dtype=np.float64)
    baseline = float(values[index])
    if not np.isfinite(baseline):
        return None
    return first_crossing(
        values,
        lead_array,
        max(baseline * factor, np.finfo(float).eps),
        start_index=index + 1,
    )


def percentile_curve(values: np.ndarray) -> dict[str, np.ndarray]:
    """Mean and 10th/90th percentiles over members."""

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": np.nanmean(array, axis=0),
        "p10": np.nanpercentile(array, 10, axis=0),
        "p90": np.nanpercentile(array, 90, axis=0),
    }


def _verify_file(specification: Mapping[str, str], label: str) -> None:
    path = Path(specification["path"])
    if not path.is_file() or file_sha256(path) != specification["sha256"]:
        raise ModelCBoundaryCheckpointError(f"artifact changed: {label}")


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load and validate the frozen, source-locked diagnostic contract."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    protocol = contract.get("protocol", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status")
        != "frozen_after_single_member_figure7_before_ensemble_checkpoint_metrics"
        or tuple(protocol.get("checkpoint_steps", ())) != CHECKPOINT_STEPS
        or int(protocol.get("selected_step", -1)) != SELECTED_STEP
        or protocol.get("lead_days")
        != "0_to_2000_inclusive_by_10"
        or tuple(protocol.get("start_draw_order", ())) != EXPECTED_STARTS
        or tuple(protocol.get("snapshot_leads", ())) != SNAPSHOT_LEADS
        or tuple(protocol.get("regions", ())) != REGION_NAMES
        or int(protocol.get("boundary_band_cells", -1)) != 4
        or protocol.get("retraining") is not False
        or protocol.get("checkpoint_selection") is not False
    ):
        raise ModelCBoundaryCheckpointError("boundary contract changed")
    checkpoints = contract.get("checkpoints", [])
    if tuple(int(item["optimizer_step"]) for item in checkpoints) != CHECKPOINT_STEPS:
        raise ModelCBoundaryCheckpointError("checkpoint order changed")
    if verify_sources:
        for index, specification in enumerate(checkpoints):
            _verify_file(specification, f"checkpoint_{index}")
        for label, specification in contract["artifacts"].items():
            _verify_file(specification, label)
        root = resolved.parents[1]
        for relative, expected in contract["source_hashes"].items():
            source = root / relative
            if not source.is_file() or file_sha256(source) != expected:
                raise ModelCBoundaryCheckpointError(
                    f"source changed: {relative}"
                )
    return contract, resolved, file_sha256(resolved)


def _load_stepper(
    contract: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> PointwiseDirectStepper:
    payload = torch.load(
        checkpoint["path"],
        map_location=device,
        weights_only=False,
    )
    architecture_dict = contract["model"]["architecture"]
    if (
        payload.get("architecture") != architecture_dict
        or int(payload.get("optimizer_step", -1))
        != int(checkpoint["optimizer_step"])
    ):
        raise ModelCBoundaryCheckpointError(
            "checkpoint payload identity changed"
        )
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


def _truth_cache(
    truth_source: ContinuousS0Truth,
    starts: np.ndarray,
    wet: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read long truth once and retain only transport-facing fields."""

    initial = truth_source.batch(starts)
    shape = (len(LEAD_DAYS), starts.size, *wet.shape)
    cache = {
        field: np.empty(shape, dtype=np.float32)
        for field in FIELDS
    }
    for lead_index, lead in enumerate(LEAD_DAYS):
        truth = truth_source.batch(starts + lead)
        qx, qy = barotropic_transports(truth)
        cache["qx"][lead_index] = qx
        cache["qy"][lead_index] = qy
        cache["streamfunction"][lead_index] = streamfunction_from_qx(
            qx,
            wet,
        )
    return initial, cache


def _empty_arrays(
    wet: np.ndarray,
    starts: np.ndarray,
) -> dict[str, np.ndarray]:
    checkpoints = len(CHECKPOINT_STEPS)
    members = starts.size
    leads = len(LEAD_DAYS)
    regions = len(REGION_NAMES)
    arrays: dict[str, np.ndarray] = {
        "checkpoint_steps": np.asarray(CHECKPOINT_STEPS, dtype=np.int32),
        "lead_days": np.asarray(LEAD_DAYS, dtype=np.int16),
        "start_draw_order": starts.astype(np.int32),
        "finite": np.empty((checkpoints, members, leads), dtype=np.uint8),
        "normalized_max_abs": np.empty(
            (checkpoints, members, leads),
            dtype=np.float32,
        ),
        "region_names": np.asarray(REGION_NAMES),
        "snapshot_leads": np.asarray(SNAPSHOT_LEADS, dtype=np.int16),
        "wet_mask": wet.astype(np.uint8),
    }
    for field in FIELDS:
        arrays[f"rmse__{field}"] = np.empty(
            (checkpoints, members, leads, regions),
            dtype=np.float64,
        )
        arrays[f"boundary_fraction__{field}"] = np.empty(
            (checkpoints, members, leads),
            dtype=np.float64,
        )
    snapshot_shape = (
        len(SNAPSHOT_LEADS),
        members,
        *wet.shape,
    )
    for field in FIELDS:
        arrays[f"selected_snapshot_error__{field}"] = np.empty(
            snapshot_shape,
            dtype=np.float32,
        )
    return arrays


def _evaluate_checkpoint(
    checkpoint_index: int,
    checkpoint: Mapping[str, Any],
    contract: Mapping[str, Any],
    arrays: dict[str, np.ndarray],
    initial: np.ndarray,
    truth_cache: Mapping[str, np.ndarray],
    static: Any,
    starts: np.ndarray,
    wet: np.ndarray,
    masks: Mapping[str, np.ndarray],
    device: Any,
    wind_mean: float,
    wind_scale: float,
) -> None:
    stepper = _load_stepper(
        contract,
        checkpoint,
        device,
        wet,
        wind_mean,
        wind_scale,
    )
    experiments = np.zeros(starts.size, dtype=np.int64)
    current = stepper.normalized_state(initial)
    normalized_static = stepper.normalized_static(static, experiments)
    wet_tensor = torch.from_numpy(wet).to(device=device)
    snapshot_lookup = {
        lead: index for index, lead in enumerate(SNAPSHOT_LEADS)
    }

    with torch.no_grad():
        for lead_index, lead in enumerate(LEAD_DAYS):
            if lead:
                current = stepper.step(current, normalized_static)
                prediction = stepper.physical(current)
            else:
                prediction = initial.copy()
            arrays["finite"][checkpoint_index, :, lead_index] = np.isfinite(
                prediction
            ).all(axis=(1, 2, 3))
            arrays["normalized_max_abs"][
                checkpoint_index,
                :,
                lead_index,
            ] = (
                torch.amax(
                    torch.abs(current[:, :, wet_tensor]),
                    dim=(1, 2),
                )
                .detach()
                .cpu()
                .numpy()
            )
            qx, qy = barotropic_transports(prediction)
            fields = {
                "qx": qx,
                "qy": qy,
                "streamfunction": streamfunction_from_qx(qx, wet),
            }
            for field in FIELDS:
                error = fields[field] - truth_cache[field][lead_index]
                for region_index, region in enumerate(REGION_NAMES):
                    arrays[f"rmse__{field}"][
                        checkpoint_index,
                        :,
                        lead_index,
                        region_index,
                    ] = member_rmse(error, masks[region])
                arrays[f"boundary_fraction__{field}"][
                    checkpoint_index,
                    :,
                    lead_index,
                ] = member_boundary_fraction(
                    error,
                    masks["boundary"],
                    wet,
                )
                if (
                    int(checkpoint["optimizer_step"]) == SELECTED_STEP
                    and lead in snapshot_lookup
                ):
                    arrays[f"selected_snapshot_error__{field}"][
                        snapshot_lookup[lead]
                    ] = error.astype(np.float32)

    del stepper.model
    del stepper
    del current
    del normalized_static
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _spatial_cosines(
    values: np.ndarray,
    wet: np.ndarray,
) -> np.ndarray:
    reference = np.nanmean(values, axis=0)[wet]
    result = np.empty(values.shape[0], dtype=np.float64)
    reference_norm = np.linalg.norm(reference)
    for member, value in enumerate(values):
        candidate = np.asarray(value, dtype=np.float64)[wet]
        denominator = np.linalg.norm(candidate) * reference_norm
        result[member] = (
            float(np.dot(candidate, reference) / denominator)
            if denominator > 0.0
            else np.nan
        )
    return result


def _selected_spatial_summary(
    arrays: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    day_index = SNAPSHOT_LEADS.index(2000)
    wet = masks["wet"]
    result: dict[str, Any] = {}
    for field in FIELDS:
        values = np.asarray(
            arrays[f"selected_snapshot_error__{field}"][day_index],
            dtype=np.float64,
        )
        ensemble_mean = np.nanmean(values, axis=0)
        positive = np.mean(values >= 0.0, axis=0)
        agreement = np.maximum(positive, 1.0 - positive)
        hotspot = wet & (
            np.abs(ensemble_mean)
            >= 0.25 * np.nanmax(np.abs(ensemble_mean[wet]))
        )
        cosines = _spatial_cosines(values, wet)
        result[field] = {
            "member_to_ensemble_mean_spatial_cosine": {
                "mean": float(np.nanmean(cosines)),
                "p10": float(np.nanpercentile(cosines, 10)),
                "p90": float(np.nanpercentile(cosines, 90)),
            },
            "mean_sign_agreement_wet": float(np.mean(agreement[wet])),
            "mean_sign_agreement_boundary": float(
                np.mean(agreement[masks["boundary"]])
            ),
            "mean_sign_agreement_interior": float(
                np.mean(agreement[masks["interior"]])
            ),
            "hotspot_cell_count": int(hotspot.sum()),
            "hotspot_mean_sign_agreement": float(
                np.mean(agreement[hotspot])
            ),
        }
    return result


def _summary(
    arrays: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    leads = np.asarray(arrays["lead_days"], dtype=int)
    region_index = {
        name: index for index, name in enumerate(REGION_NAMES)
    }
    result: dict[str, Any] = {
        "status": "complete",
        "classification_scope": (
            "zero_retraining_training_duration_and_boundary_localization_"
            "diagnosis_only"
        ),
        "checkpoint_count": len(CHECKPOINT_STEPS),
        "member_count": len(EXPECTED_STARTS),
        "selected_step": SELECTED_STEP,
        "regions": {
            name: int(mask.sum()) for name, mask in masks.items()
        },
        "checkpoints": {},
    }
    for checkpoint_index, step in enumerate(CHECKPOINT_STEPS):
        normalized = percentile_curve(
            arrays["normalized_max_abs"][checkpoint_index]
        )["mean"]
        checkpoint_summary: dict[str, Any] = {
            "all_states_finite": bool(
                np.all(arrays["finite"][checkpoint_index])
            ),
            "normalized_max_abs_day2000": {
                key: float(value[-1])
                for key, value in percentile_curve(
                    arrays["normalized_max_abs"][checkpoint_index]
                ).items()
            },
            "first_mean_normalized_max_crossing_days": {
                str(int(threshold)): first_crossing(
                    normalized,
                    leads,
                    threshold,
                )
                for threshold in THRESHOLDS
            },
            "fields": {},
        }
        for field in FIELDS:
            rmse = np.asarray(
                arrays[f"rmse__{field}"][checkpoint_index],
                dtype=np.float64,
            )
            fractions = percentile_curve(
                arrays[f"boundary_fraction__{field}"][checkpoint_index]
            )
            boundary_curve = np.nanmean(
                rmse[:, :, region_index["boundary"]],
                axis=0,
            )
            interior_curve = np.nanmean(
                rmse[:, :, region_index["interior"]],
                axis=0,
            )
            checkpoint_summary["fields"][field] = {
                "day2000_rmse_mean": {
                    region: float(
                        np.nanmean(rmse[:, -1, region_index[region]])
                    )
                    for region in REGION_NAMES
                },
                "day2000_boundary_squared_error_fraction": {
                    key: float(value[-1])
                    for key, value in fractions.items()
                },
                "post_day200_growth_crossing_days": {
                    "boundary": {
                        str(int(factor)): growth_crossing(
                            boundary_curve,
                            leads,
                            factor,
                        )
                        for factor in GROWTH_FACTORS
                    },
                    "interior": {
                        str(int(factor)): growth_crossing(
                            interior_curve,
                            leads,
                            factor,
                        )
                        for factor in GROWTH_FACTORS
                    },
                },
            }
        result["checkpoints"][str(step)] = checkpoint_summary
    result["selected_spatial_consistency_day2000"] = (
        _selected_spatial_summary(arrays, masks)
    )
    selected = result["checkpoints"][str(SELECTED_STEP)]
    earlier = [
        result["checkpoints"][str(step)]
        for step in CHECKPOINT_STEPS
        if step < SELECTED_STEP
    ]
    selected_crossing = selected[
        "first_mean_normalized_max_crossing_days"
    ]["20"]
    earlier_crossings = [
        item["first_mean_normalized_max_crossing_days"]["20"]
        for item in earlier
    ]
    result["training_duration_diagnosis"] = {
        "selected_first_normalized_20_day": selected_crossing,
        "earlier_first_normalized_20_days": earlier_crossings,
        "any_earlier_checkpoint_avoids_normalized_20_through_day2000": any(
            value is None for value in earlier_crossings
        ),
        "descriptive_only_not_checkpoint_reselection": True,
    }
    return result


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )


def _checkpoint_colors() -> dict[int, Any]:
    color_values = plt.cm.viridis(np.linspace(0.08, 0.92, len(CHECKPOINT_STEPS)))
    result = {
        step: color for step, color in zip(CHECKPOINT_STEPS, color_values)
    }
    result[SELECTED_STEP] = "#D62728"
    return result


def _plot_checkpoint_curves(
    axis: Any,
    arrays: Mapping[str, np.ndarray],
    key: str,
    *,
    region: str | None = None,
    ylabel: str,
    log_y: bool = False,
) -> None:
    colors = _checkpoint_colors()
    leads = np.asarray(arrays["lead_days"])
    region_index = (
        REGION_NAMES.index(region) if region is not None else None
    )
    for checkpoint_index, step in enumerate(CHECKPOINT_STEPS):
        values = np.asarray(arrays[key][checkpoint_index])
        if region_index is not None:
            values = values[:, :, region_index]
        curve = percentile_curve(values)
        axis.plot(
            leads,
            curve["mean"],
            color=colors[step],
            linewidth=2.3 if step == SELECTED_STEP else 1.0,
            alpha=1.0 if step == SELECTED_STEP else 0.78,
            label=f"{step:,}" + (" selected" if step == SELECTED_STEP else ""),
        )
    axis.set_xlim(0, 2000)
    axis.set_ylabel(ylabel)
    axis.grid(color="0.85", linewidth=0.6)
    if log_y:
        axis.set_yscale("log")


def _plot_transport_rmse(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    for axis, region in zip(
        axes.flat,
        ("west", "north", "east", "interior"),
    ):
        _plot_checkpoint_curves(
            axis,
            arrays,
            "rmse__qx",
            region=region,
            ylabel=r"$Q_x$ RMSE (m$^2$ s$^{-1}$)",
            log_y=True,
        )
        axis.set_title(region.capitalize())
    axes[1, 0].set_xlabel("Lead (days)")
    axes[1, 1].set_xlabel("Lead (days)")
    axes[0, 1].legend(ncol=2, loc="best")
    figure.suptitle(
        "Depth-integrated zonal-transport error by checkpoint and region"
    )
    figure.savefig(output / FIGURE_NAMES[0], bbox_inches="tight")
    plt.close(figure)


def _plot_boundary_fractions(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    labels = {
        "qx": r"$Q_x$",
        "qy": r"$Q_y$",
        "streamfunction": r"$\psi$",
    }
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9.2, 8.1),
        sharex=True,
        constrained_layout=True,
    )
    for axis, field in zip(axes, FIELDS):
        _plot_checkpoint_curves(
            axis,
            arrays,
            f"boundary_fraction__{field}",
            ylabel=f"{labels[field]} boundary fraction",
        )
        axis.axhline(896.0 / 3600.0, color="0.3", linestyle=":", linewidth=0.9)
        axis.set_ylim(0.0, 1.02)
    axes[0].legend(ncol=4, loc="best")
    axes[-1].set_xlabel("Lead (days)")
    figure.suptitle(
        "Fraction of total squared error in the four-cell wet-boundary band"
    )
    figure.savefig(output / FIGURE_NAMES[1], bbox_inches="tight")
    plt.close(figure)


def _plot_checkpoint_timing(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.3, 7.0),
        constrained_layout=True,
    )
    _plot_checkpoint_curves(
        axes[0],
        arrays,
        "normalized_max_abs",
        ylabel="Mean maximum |normalized state|",
        log_y=True,
    )
    for threshold in THRESHOLDS:
        axes[0].axhline(
            threshold,
            color="0.45",
            linestyle=":",
            linewidth=0.7,
        )
    axes[0].legend(ncol=4, loc="best")
    first_20 = []
    first_100 = []
    for step in CHECKPOINT_STEPS:
        values = summary["checkpoints"][str(step)][
            "first_mean_normalized_max_crossing_days"
        ]
        first_20.append(values["20"] if values["20"] is not None else 2050)
        first_100.append(
            values["100"] if values["100"] is not None else 2050
        )
    x = np.arange(len(CHECKPOINT_STEPS))
    axes[1].plot(x, first_20, "o-", label="first mean max > 20")
    axes[1].plot(x, first_100, "s-", label="first mean max > 100")
    axes[1].axhline(2000, color="0.4", linestyle=":", linewidth=0.8)
    axes[1].set_xticks(
        x,
        [f"{step:,}" for step in CHECKPOINT_STEPS],
        rotation=25,
    )
    axes[1].set_ylim(0, 2100)
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("First crossing lead (days)")
    axes[1].grid(color="0.85", linewidth=0.6)
    axes[1].legend(loc="best")
    figure.suptitle("Does checkpoint age control long-term runaway?")
    figure.savefig(output / FIGURE_NAMES[2], bbox_inches="tight")
    plt.close(figure)


def _plot_spatial_consistency(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
) -> None:
    index = SNAPSHOT_LEADS.index(2000)
    psi = np.asarray(
        arrays["selected_snapshot_error__streamfunction"][index],
        dtype=np.float64,
    )
    qx = np.asarray(
        arrays["selected_snapshot_error__qx"][index],
        dtype=np.float64,
    )
    mean_psi = np.nanmean(psi, axis=0)
    p90_abs_psi = np.nanpercentile(np.abs(psi), 90, axis=0)
    positive = np.mean(psi >= 0.0, axis=0)
    agreement = np.maximum(positive, 1.0 - positive)
    mean_qx = np.nanmean(qx, axis=0)
    fields = (mean_psi, p90_abs_psi, agreement, mean_qx)
    titles = (
        r"Mean $\psi$ error (Sv)",
        r"90th-percentile $|\psi|$ error (Sv)",
        r"$\psi$ sign agreement",
        r"Mean $Q_x$ error (m$^2$ s$^{-1}$)",
    )
    cmaps = ("RdBu_r", "magma", "viridis", "RdBu_r")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 7.5),
        constrained_layout=True,
    )
    for axis, value, title, cmap in zip(axes.flat, fields, titles, cmaps):
        masked = np.ma.array(value, mask=~wet)
        if title == r"$\psi$ sign agreement":
            image = axis.pcolormesh(
                longitude,
                latitude,
                masked,
                cmap=cmap,
                vmin=0.5,
                vmax=1.0,
                shading="auto",
            )
        elif cmap == "RdBu_r":
            bound = np.nanmax(np.abs(value[wet]))
            image = axis.pcolormesh(
                longitude,
                latitude,
                masked,
                cmap=cmap,
                vmin=-bound,
                vmax=bound,
                shading="auto",
            )
        else:
            image = axis.pcolormesh(
                longitude,
                latitude,
                masked,
                cmap=cmap,
                shading="auto",
            )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlabel("Longitude (°)")
        axis.set_ylabel("Latitude (°)")
        figure.colorbar(image, ax=axis, shrink=0.78)
    figure.suptitle(
        "Selected step 13,440: day-2,000 spatial consistency across 15 members"
    )
    figure.savefig(output / FIGURE_NAMES[3], bbox_inches="tight")
    plt.close(figure)


def _plot_selected_growth(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    checkpoint_index = CHECKPOINT_STEPS.index(SELECTED_STEP)
    leads = np.asarray(arrays["lead_days"])
    labels = {
        "qx": r"$Q_x$ (m$^2$ s$^{-1}$)",
        "qy": r"$Q_y$ (m$^2$ s$^{-1}$)",
        "streamfunction": r"$\psi$ (Sv)",
    }
    colors = {"boundary": "#D62728", "interior": "#1F77B4"}
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9.2, 8.2),
        sharex=True,
        constrained_layout=True,
    )
    for axis, field in zip(axes, FIELDS):
        for region in ("boundary", "interior"):
            values = arrays[f"rmse__{field}"][
                checkpoint_index,
                :,
                :,
                REGION_NAMES.index(region),
            ]
            curve = percentile_curve(values)
            axis.plot(
                leads,
                curve["mean"],
                color=colors[region],
                linewidth=1.8,
                label=region.capitalize(),
            )
            axis.fill_between(
                leads,
                curve["p10"],
                curve["p90"],
                color=colors[region],
                alpha=0.15,
                linewidth=0,
            )
        axis.set_yscale("log")
        axis.set_ylabel(labels[field] + " RMSE")
        axis.grid(color="0.85", linewidth=0.6)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Lead (days)")
    axes[-1].set_xlim(0, 2000)
    figure.suptitle(
        "Selected Model C: boundary versus interior error growth"
    )
    figure.savefig(output / FIGURE_NAMES[4], bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "optimizer_step",
                "lead_days",
                "metric",
                "field",
                "region",
                "mean",
                "p10",
                "p90",
            )
        )
        for checkpoint_index, step in enumerate(CHECKPOINT_STEPS):
            for field in FIELDS:
                for region_index, region in enumerate(REGION_NAMES):
                    curve = percentile_curve(
                        arrays[f"rmse__{field}"][
                            checkpoint_index,
                            :,
                            :,
                            region_index,
                        ]
                    )
                    for lead_index, lead in enumerate(LEAD_DAYS):
                        writer.writerow(
                            (
                                step,
                                lead,
                                "rmse",
                                field,
                                region,
                                curve["mean"][lead_index],
                                curve["p10"][lead_index],
                                curve["p90"][lead_index],
                            )
                        )
                curve = percentile_curve(
                    arrays[f"boundary_fraction__{field}"][checkpoint_index]
                )
                for lead_index, lead in enumerate(LEAD_DAYS):
                    writer.writerow(
                        (
                            step,
                            lead,
                            "boundary_squared_error_fraction",
                            field,
                            "boundary",
                            curve["mean"][lead_index],
                            curve["p10"][lead_index],
                            curve["p90"][lead_index],
                        )
                    )
            curve = percentile_curve(
                arrays["normalized_max_abs"][checkpoint_index]
            )
            for lead_index, lead in enumerate(LEAD_DAYS):
                writer.writerow(
                    (
                        step,
                        lead,
                        "normalized_max_abs",
                        "state",
                        "wet",
                        curve["mean"][lead_index],
                        curve["p10"][lead_index],
                        curve["p90"][lead_index],
                    )
                )


def _readme(report: Mapping[str, Any]) -> str:
    return f"""# Model C S0 boundary/checkpoint stability diagnosis

This package performs zero retraining. It evaluates all seven stored
seed-20260724 anomaly-direct checkpoints on the same 15 fresh S0 inference
initializations and evaluation-only day-2000 truth used by job 304736.

The four-cell boundary band is defined inside the 60-by-60 wet rectangle.
Qx and Qy are depth-integrated zonal and meridional transports. Streamfunction
is reconstructed by cumulatively integrating Qx from south to north and is not
a neural-network output.

The package diagnoses whether earlier checkpoints avoid the runaway and
whether boundary transport error precedes interior error. It is descriptive
and cannot reselect a checkpoint.

Report content SHA-256: `{report["report_content_sha256"]}`.
"""


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify sources, checkpoints, prior evidence, and truth coverage."""

    contract, resolved, digest = load_contract(contract_path)
    prior_arrays_path = Path(
        contract["artifacts"]["job304736_arrays"]["path"]
    )
    with np.load(prior_arrays_path) as arrays:
        if (
            tuple(int(value) for value in arrays["start_draw_order"])
            != EXPECTED_STARTS
            or tuple(int(value) for value in arrays["lead_days"])
            != LEAD_DAYS
            or arrays["figure7_model_streamfunction"].shape
            != (2, 62, 62)
        ):
            raise ModelCBoundaryCheckpointError(
                "job-304736 evidence changed"
            )
    result = json.loads(
        Path(contract["artifacts"]["long_truth_result"]["path"]).read_text()
    )
    run_dir = Path(result["run_dir"])
    if (
        result.get("slurm_job_id") != "304735"
        or result.get("returncode") != 0
        or len(list(run_dir.glob("dynState.*.meta"))) != 2160
        or max(EXPECTED_STARTS) + 2000 >= 9360
    ):
        raise ModelCBoundaryCheckpointError("long truth is incomplete")
    return {
        "status": "pass",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "member_count": len(EXPECTED_STARTS),
        "lead_count": len(LEAD_DAYS),
        "retraining": False,
        "checkpoint_selection": False,
    }


def evaluate(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Execute the frozen zero-retraining checkpoint/boundary diagnosis."""

    if torch is None:
        raise RuntimeError("boundary checkpoint evaluation requires PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    preflight(resolved_contract)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested without an available GPU")
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
        raise FileExistsError("refusing to overwrite boundary diagnosis")

    dataset_path = Path(
        contract["artifacts"]["dataset_metadata"]["path"]
    ).parent
    group = zarr.open_consolidated(str(dataset_path), mode="r")
    if tuple(group.attrs["state_channels"]) != STATE_CHANNELS:
        raise ModelCBoundaryCheckpointError("dataset channels changed")
    state = group["state"]
    static = group["static_features"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    longitude = np.asarray(group["longitude_deg"][:], dtype=np.float32)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float32)
    _, _, _, _, wind_mean, wind_scale = _normalizers(group)
    masks = region_masks(wet, int(contract["protocol"]["boundary_band_cells"]))

    result_path = Path(contract["artifacts"]["long_truth_result"]["path"])
    run_dir = Path(json.loads(result_path.read_text())["run_dir"])
    truth_source = ContinuousS0Truth(state, run_dir, wet)
    starts = np.asarray(EXPECTED_STARTS, dtype=np.int64)
    initial, truth_cache = _truth_cache(truth_source, starts, wet)
    arrays = _empty_arrays(wet, starts)
    arrays["longitude_deg"] = longitude
    arrays["latitude_deg"] = latitude

    for checkpoint_index, checkpoint in enumerate(contract["checkpoints"]):
        _evaluate_checkpoint(
            checkpoint_index,
            checkpoint,
            contract,
            arrays,
            initial,
            truth_cache,
            static,
            starts,
            wet,
            masks,
            device,
            wind_mean,
            wind_scale,
        )

    summary = _summary(arrays, masks)
    scratch_temporary.parent.mkdir(parents=True, exist_ok=True)
    project_temporary.parent.mkdir(parents=True, exist_ok=True)
    scratch_temporary.mkdir(exist_ok=False)
    project_temporary.mkdir(exist_ok=False)
    try:
        scratch_arrays = scratch_temporary / ARRAYS_NAME
        np.savez_compressed(scratch_arrays, **arrays)
        _style()
        _plot_transport_rmse(project_temporary, arrays)
        _plot_boundary_fractions(project_temporary, arrays)
        _plot_checkpoint_timing(project_temporary, arrays, summary)
        _plot_spatial_consistency(
            project_temporary,
            arrays,
            longitude,
            latitude,
            wet,
        )
        _plot_selected_growth(project_temporary, arrays)
        _write_csv(project_temporary / CSV_NAME, arrays)
        (project_temporary / SUMMARY_NAME).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        report = {
            "version": VERSION,
            "status": "complete",
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "purpose": contract["purpose"],
            "classification_scope": summary["classification_scope"],
            "retraining": False,
            "checkpoint_selection": False,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "selected_step": SELECTED_STEP,
            "dataset": str(dataset_path),
            "long_truth_result": str(result_path),
            "job304736_evidence": contract["artifacts"][
                "job304736_report"
            ]["path"],
            "device": str(device),
            "protocol": contract["protocol"],
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
