"""Non-finite-safe evaluation-only recovery of Model C S0 stability comparison.

Job 304750 completed its model rollouts but failed while reducing explosive
float32 diagnostics.  This recovery changes no model, state, initialization,
lead, or tangent protocol.  It uses float64 physical reductions, records the
first non-finite model-state lead per member, and censors rather than fits a
requested RMSE window after a member becomes non-finite.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from .af_a0_evaluate import _normalizers
from .af_data import STATE_CHANNELS
from .af_model_c_bire_s0_figures import (
    ContinuousS0Truth,
    _prior_stepper,
    _s0_training_climatology,
)
from .af_model_c_bire_s0_long_truth import EXPECTED_STARTS
from .af_model_c_s0_stability_tangent_comparison import (
    COLORS,
    DIRECT_METHODS,
    FIELDS,
    FIELD_LABELS,
    FIT_WINDOWS,
    LABELS,
    LEAD_DAYS,
    MANIFEST,
    METHODS,
    README,
    SPECTRUM_BANDS,
    SPECTRUM_LEADS,
    STAT_FIELDS,
    TANGENT_BANDS,
    TANGENT_TIMES,
    _band_projector,
    _pointwise_stepper,
    dominant_singular_and_tangent_gain,
    file_sha256,
    json_sha256,
)
from .af_pressure import (
    DRF_M,
    GRAVITY_M_S2,
    PHIHYD_LEVELS,
    THERMAL_EXPANSION_PER_C,
    T_REF_C,
    _vertical_distances,
)
from .af_tutorial_analysis import EARTH_RADIUS_M

try:
    import torch
except (ImportError, OSError):  # pragma: no cover
    torch = None  # type: ignore[assignment]


VERSION = "model_c_s0_stability_tangent_recovery_v2"
CONTRACT_STATUS = "frozen_evaluation_only_recovery_after_job304750_overflow"
FAILED_JOB_ID = "304750"
FIGURES = (
    "model_c_s0_stability_models_log_rmse.png",
    "model_c_s0_stability_normalized_envelope.png",
    "model_c_s0_stability_day2000_spectra.png",
    "model_c_s0_tangent_gain_comparison.png",
    "model_c_s0_first_nonfinite_lead.png",
)
RECOVERY_ARRAYS = "model_c_s0_stability_tangent_recovery_arrays.npz"
REPORT = "model_c_s0_stability_tangent_recovery_report.json"
SUMMARY = "model_c_s0_stability_tangent_recovery_summary.json"
RECOVERY_CSV = "model_c_s0_stability_tangent_recovery_curves.csv"


class StabilityRecoveryError(RuntimeError):
    """Raised when the frozen recovery protocol changes."""


def _verify_file(record: Mapping[str, Any], label: str) -> Path:
    path = Path(record["path"]).resolve()
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise StabilityRecoveryError(f"immutable artifact changed: {label}")
    return path


def load_contract(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    protocol = contract.get("protocol", {})
    recovery = contract.get("recovery_changes", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or contract.get("failed_job_id") != FAILED_JOB_ID
        or tuple(protocol.get("methods", ())) != METHODS
        or tuple(protocol.get("fields", ())) != FIELDS
        or tuple(protocol.get("statistical_fields", ())) != STAT_FIELDS
        or tuple(protocol.get("start_indices", ())) != EXPECTED_STARTS
        or tuple(protocol.get("lead_days", ())) != (0, 2000)
        or int(protocol.get("dt_days", -1)) != 10
        or int(protocol.get("calls", -1)) != 200
        or tuple(protocol.get("spectrum_leads", ())) != SPECTRUM_LEADS
        or protocol.get("spectrum_bands")
        != {name: list(band) for name, band in SPECTRUM_BANDS.items()}
        or tuple(map(tuple, protocol.get("fit_windows", ()))) != FIT_WINDOWS
        or tuple(protocol.get("tangent_methods", ())) != DIRECT_METHODS
        or tuple(protocol.get("tangent_times", ())) != TANGENT_TIMES
        or recovery
        != {
            "physical_reduction_dtype": "float64",
            "model_state_dtype": "unchanged_float32",
            "nonfinite_policy": (
                "record_first_lead_per_member_then_censor_later_metrics"
            ),
            "fit_policy": (
                "fit_only_complete_all_member_positive_prefix_within_window"
            ),
            "model_or_weight_change": False,
            "retraining_steps": 0,
            "checkpoint_selection": False,
        }
    ):
        raise StabilityRecoveryError("stability recovery contract changed")
    for label, record in contract["artifacts"].items():
        _verify_file(record, label)
    root = resolved.parents[1]
    for relative, expected in contract["source_hashes"].items():
        source = root / relative
        if not source.is_file() or file_sha256(source) != expected:
            raise StabilityRecoveryError(f"source changed: {relative}")
    for key in ("scratch", "project"):
        output = Path(contract["output"][key]).resolve()
        if output.exists() or output.with_name(output.name + ".tmp").exists():
            raise FileExistsError(f"refusing to overwrite output: {output}")
    return contract, resolved, file_sha256(resolved)


def _physical64(stepper: Any, normalized: Any) -> np.ndarray:
    value = normalized.detach().cpu().numpy().astype(np.float64)
    mean = np.asarray(stepper.mean, dtype=np.float64)
    scale = np.asarray(stepper.scale, dtype=np.float64)
    if mean.ndim == 3:
        value = value * scale[None] + mean[None]
    else:
        value = value * scale[None, :, None, None]
        value += mean[None, :, None, None]
    value[:, :, ~stepper.wet] = 0.0
    return value


def _phihyd64(theta_c: np.ndarray, eta_m: np.ndarray, wet: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta_c, dtype=np.float64)
    eta = np.asarray(eta_m, dtype=np.float64)
    density = -THERMAL_EXPANSION_PER_C * (
        theta - T_REF_C.reshape((1, T_REF_C.size, 1, 1))
    )
    above, below = _vertical_distances()
    interface = np.zeros(theta.shape[:1] + theta.shape[-2:], dtype=np.float64)
    baroclinic = np.empty_like(theta)
    for level in range(DRF_M.size):
        layer = density[:, level]
        center = interface + above[level] * GRAVITY_M_S2 * layer
        baroclinic[:, level] = center
        interface = center + below[level] * GRAVITY_M_S2 * layer
    result = baroclinic + GRAVITY_M_S2 * eta[:, None]
    result[:, :, ~wet] = 0.0
    return result


def derived_fields64(states: np.ndarray, wet: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(states, dtype=np.float64)
    pressure = _phihyd64(values[:, 30:45], values[:, 45], wet)
    dy_m = EARTH_RADIUS_M * np.deg2rad(1.0)
    depth_integrated_u = np.sum(
        values[:, :15] * DRF_M[None, :, None, None],
        axis=1,
    )
    streamfunction = np.cumsum(-depth_integrated_u * dy_m, axis=1) / 1.0e6
    result = {
        "surface_speed": np.hypot(values[:, 0], values[:, 15]),
        "sst": values[:, 30].copy(),
        "phihyd_surface": pressure[:, PHIHYD_LEVELS["phihyd_surface"]].copy(),
        "streamfunction": streamfunction,
    }
    for field in result.values():
        field[:, ~wet] = 0.0
    return result


def member_rmse64(
    prediction: np.ndarray,
    truth: np.ndarray,
    wet: np.ndarray,
) -> np.ndarray:
    error = (
        np.asarray(prediction, dtype=np.float64)
        - np.asarray(truth, dtype=np.float64)
    )[..., wet].reshape(prediction.shape[0], -1)
    return np.sqrt(np.mean(np.square(error), axis=1, dtype=np.float64))


def radial_spectrum64(
    fields: np.ndarray,
    wet: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.where(wet)
    y0, y1 = rows.min(), rows.max() + 1
    x0, x1 = columns.min(), columns.max() + 1
    cropped = np.asarray(fields[:, y0:y1, x0:x1], dtype=np.float64)
    cropped -= np.mean(cropped, axis=(1, 2), keepdims=True)
    window = (
        np.hanning(cropped.shape[1])[:, None]
        * np.hanning(cropped.shape[2])[None, :]
    )
    transformed = np.fft.fft2(cropped * window[None], axes=(-2, -1))
    power = np.square(np.abs(transformed))
    ky = np.fft.fftfreq(cropped.shape[1]) * cropped.shape[1]
    kx = np.fft.fftfreq(cropped.shape[2]) * cropped.shape[2]
    radius = np.sqrt(np.square(ky[:, None]) + np.square(kx[None, :]))
    shell = np.floor(radius).astype(int)
    modes = np.arange(1, min(cropped.shape[1], cropped.shape[2]) // 2 + 1)
    spectra = np.empty((cropped.shape[0], modes.size), dtype=np.float64)
    for index, mode in enumerate(modes):
        spectra[:, index] = np.mean(power[:, shell == mode], axis=1)
    return modes.astype(np.float64), spectra


def _spatial_stats64(
    fields: Mapping[str, np.ndarray],
    wet: np.ndarray,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for field in STAT_FIELDS:
        selected = np.asarray(fields[field], dtype=np.float64)[:, wet]
        result[f"mean__{field}"] = np.mean(selected, axis=1)
        result[f"std__{field}"] = np.std(selected, axis=1)
        if field == "streamfunction":
            result[f"minimum__{field}"] = np.min(selected, axis=1)
            result[f"maximum__{field}"] = np.max(selected, axis=1)
    return result


def _allocate(member_count: int, spectrum_modes: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "lead_days": np.asarray(LEAD_DAYS, dtype=np.int16),
        "spectrum_leads": np.asarray(SPECTRUM_LEADS, dtype=np.int16),
        "start_draw_order": np.asarray(EXPECTED_STARTS, dtype=np.int32),
        "finite": np.zeros(
            (len(METHODS), member_count, len(LEAD_DAYS)),
            dtype=np.uint8,
        ),
        "normalized_max_abs": np.full(
            (len(METHODS), member_count, len(LEAD_DAYS)),
            np.nan,
            dtype=np.float64,
        ),
        "truth_normalized_max_abs": np.full(
            (len(METHODS), member_count, len(LEAD_DAYS)),
            np.nan,
            dtype=np.float64,
        ),
        "spectrum_modes": np.arange(1, spectrum_modes + 1, dtype=np.float64),
    }
    for method in METHODS:
        for field in FIELDS:
            arrays[f"rmse__{method}__{field}"] = np.full(
                (member_count, len(LEAD_DAYS)),
                np.nan,
                dtype=np.float64,
            )
        for field in STAT_FIELDS:
            statistics = (
                ("mean", "std", "minimum", "maximum")
                if field == "streamfunction"
                else ("mean", "std")
            )
            for statistic in statistics:
                arrays[f"{statistic}__{method}__{field}"] = np.full(
                    (member_count, len(LEAD_DAYS)),
                    np.nan,
                    dtype=np.float64,
                )
            arrays[f"spectrum__{method}__{field}"] = np.full(
                (member_count, len(SPECTRUM_LEADS), spectrum_modes),
                np.nan,
                dtype=np.float64,
            )
    for field in FIELDS:
        arrays[f"rmse__climatology__{field}"] = np.full(
            (member_count, len(LEAD_DAYS)),
            np.nan,
            dtype=np.float64,
        )
    for field in STAT_FIELDS:
        statistics = (
            ("mean", "std", "minimum", "maximum")
            if field == "streamfunction"
            else ("mean", "std")
        )
        for statistic in statistics:
            arrays[f"{statistic}__truth__{field}"] = np.full(
                (member_count, len(LEAD_DAYS)),
                np.nan,
                dtype=np.float64,
            )
        arrays[f"spectrum__truth__{field}"] = np.full(
            (member_count, len(SPECTRUM_LEADS), spectrum_modes),
            np.nan,
            dtype=np.float64,
        )
    return arrays


def _long_rollout(
    steppers: Mapping[str, Any],
    truth: ContinuousS0Truth,
    static: Any,
    starts: np.ndarray,
    climatology_derived: Mapping[str, np.ndarray],
    wet: np.ndarray,
) -> dict[str, np.ndarray]:
    member_count = starts.size
    experiments = np.zeros(member_count, dtype=np.int64)
    initial = truth.batch(starts)
    currents = {
        name: stepper.normalized_state(initial)
        for name, stepper in steppers.items()
    }
    statics = {
        name: stepper.normalized_static(static, experiments)
        for name, stepper in steppers.items()
    }
    _, trial_spectrum = radial_spectrum64(
        derived_fields64(initial[:1], wet)["sst"],
        wet,
    )
    arrays = _allocate(member_count, trial_spectrum.shape[-1])
    wet_tensor = torch.from_numpy(wet).to(
        device=next(iter(steppers.values())).device
    )
    spectrum_lookup = {lead: index for index, lead in enumerate(SPECTRUM_LEADS)}
    climate = {
        field: np.repeat(
            np.asarray(climatology_derived[field], dtype=np.float64)[None],
            member_count,
            axis=0,
        )
        for field in FIELDS
    }
    active = {
        name: np.ones(member_count, dtype=bool)
        for name in steppers
    }

    with torch.no_grad():
        for lead_index, lead in enumerate(LEAD_DAYS):
            normalized_max: dict[str, np.ndarray] = {}
            if lead:
                for name, stepper in steppers.items():
                    previously_active = active[name].copy()
                    candidate = stepper.step(currents[name], statics[name])
                    member_finite = (
                        torch.isfinite(candidate)
                        .all(dim=(1, 2, 3))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    maximum = (
                        torch.amax(
                            torch.abs(candidate[:, :, wet_tensor]),
                            dim=(1, 2),
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    maximum[~previously_active] = np.nan
                    active[name] &= member_finite
                    active_tensor = torch.from_numpy(active[name]).to(
                        device=candidate.device
                    )
                    currents[name] = torch.where(
                        active_tensor[:, None, None, None],
                        candidate,
                        torch.zeros_like(candidate),
                    )
                    normalized_max[name] = maximum
            else:
                for name in steppers:
                    normalized_max[name] = (
                        torch.amax(
                            torch.abs(currents[name][:, :, wet_tensor]),
                            dim=(1, 2),
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )

            physical: dict[str, np.ndarray] = {}
            method_fields: dict[str, dict[str, np.ndarray]] = {}
            for name, stepper in steppers.items():
                physical[name] = _physical64(stepper, currents[name])
                physical[name][~active[name]] = np.nan
                method_fields[name] = derived_fields64(physical[name], wet)

            truth_state = np.asarray(
                truth.batch(starts + lead),
                dtype=np.float64,
            )
            truth_fields = derived_fields64(truth_state, wet)
            for key, values in _spatial_stats64(truth_fields, wet).items():
                statistic, field = key.split("__")
                arrays[f"{statistic}__truth__{field}"][:, lead_index] = values
            for field in FIELDS:
                arrays[f"rmse__climatology__{field}"][:, lead_index] = (
                    member_rmse64(climate[field], truth_fields[field], wet)
                )

            for method_index, (name, stepper) in enumerate(steppers.items()):
                fields = method_fields[name]
                for field in FIELDS:
                    arrays[f"rmse__{name}__{field}"][:, lead_index] = (
                        member_rmse64(fields[field], truth_fields[field], wet)
                    )
                for key, values in _spatial_stats64(fields, wet).items():
                    statistic, field = key.split("__")
                    arrays[f"{statistic}__{name}__{field}"][:, lead_index] = values
                arrays["finite"][method_index, :, lead_index] = active[name]
                arrays["normalized_max_abs"][
                    method_index, :, lead_index
                ] = normalized_max[name]
                normalized_truth = stepper.normalized_state(
                    truth_state.astype(np.float32)
                )
                arrays["truth_normalized_max_abs"][
                    method_index, :, lead_index
                ] = (
                    torch.amax(
                        torch.abs(normalized_truth[:, :, wet_tensor]),
                        dim=(1, 2),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

            if lead in spectrum_lookup:
                spectrum_index = spectrum_lookup[lead]
                for field in STAT_FIELDS:
                    modes, spectrum = radial_spectrum64(truth_fields[field], wet)
                    if not np.array_equal(modes, arrays["spectrum_modes"]):
                        raise StabilityRecoveryError("spectrum modes changed")
                    arrays[f"spectrum__truth__{field}"][
                        :, spectrum_index
                    ] = spectrum
                    for name in steppers:
                        _, spectrum = radial_spectrum64(
                            method_fields[name][field],
                            wet,
                        )
                        arrays[f"spectrum__{name}__{field}"][
                            :, spectrum_index
                        ] = spectrum
    return arrays


def safe_log_gain(
    leads: np.ndarray,
    members: np.ndarray,
    window: tuple[int, int],
) -> dict[str, Any]:
    values = np.asarray(members, dtype=np.float64)
    requested = (leads >= window[0]) & (leads <= window[1])
    complete = np.all(np.isfinite(values) & (values > 0.0), axis=0)
    selected = requested & complete
    requested_indices = np.flatnonzero(requested)
    selected_indices = np.flatnonzero(selected)
    if selected_indices.size:
        first_gap = np.flatnonzero(~complete[requested_indices])
        if first_gap.size:
            selected_indices = requested_indices[: first_gap[0]]
    if selected_indices.size < 3:
        return {
            "status": "censored_insufficient_complete_samples",
            "requested_window_days": list(window),
            "complete_sample_count": int(selected_indices.size),
            "gain": None,
            "e_folding_days_if_gain_gt_1": None,
            "bootstrap_95_percent_interval": None,
        }
    x = leads[selected_indices].astype(np.float64) / 10.0
    y = np.log(np.mean(values[:, selected_indices], axis=0))
    slope = float(np.polyfit(x, y, 1)[0])
    gain = float(np.exp(slope))
    complete_window = selected_indices.size == requested_indices.size
    interval: list[float] | None = None
    if complete_window:
        generator = np.random.default_rng(20260729)
        bootstrap = np.empty(10000, dtype=np.float64)
        for index in range(bootstrap.size):
            chosen = generator.integers(0, values.shape[0], size=values.shape[0])
            curve = np.mean(values[chosen][:, selected_indices], axis=0)
            bootstrap[index] = np.exp(np.polyfit(x, np.log(curve), 1)[0])
        interval = [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ]
    return {
        "status": "complete" if complete_window else "censored_at_first_nonfinite",
        "requested_window_days": list(window),
        "fit_lead_days": [
            int(leads[selected_indices[0]]),
            int(leads[selected_indices[-1]]),
        ],
        "complete_sample_count": int(selected_indices.size),
        "gain": gain,
        "e_folding_days_if_gain_gt_1": (
            float(10.0 / np.log(gain)) if gain > 1.0 else None
        ),
        "bootstrap_95_percent_interval": interval,
    }


def _first_nonfinite_leads(
    finite: np.ndarray,
    leads: np.ndarray,
) -> list[int | None]:
    result: list[int | None] = []
    for member in finite:
        failed = np.flatnonzero(member == 0)
        result.append(int(leads[failed[0]]) if failed.size else None)
    return result


def _finite_float(value: Any) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def _safe_nanmean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def _summarize(
    arrays: Mapping[str, np.ndarray],
    tangent_report: Mapping[str, Any],
) -> dict[str, Any]:
    leads = np.asarray(arrays["lead_days"], dtype=np.int64)
    modes = np.asarray(arrays["spectrum_modes"], dtype=np.float64)
    post500 = leads >= 500
    summary: dict[str, Any] = {
        "classification_scope": (
            "zero_retraining_recovery_no_checkpoint_selection_or_promotion"
        ),
        "failed_job": {
            "job_id": FAILED_JOB_ID,
            "failure": "float32_diagnostic_overflow_then_log_fit_rejection",
        },
        "methods": {},
        "tangent": tangent_report,
    }
    for method_index, method in enumerate(METHODS):
        finite = np.asarray(arrays["finite"][method_index], dtype=np.uint8)
        first = _first_nonfinite_leads(finite, leads)
        present = [value for value in first if value is not None]
        record: dict[str, Any] = {
            "all_states_finite_through_day2000": not present,
            "first_nonfinite_state_lead_by_member": first,
            "earliest_nonfinite_state_lead": min(present) if present else None,
            "latest_nonfinite_state_lead": max(present) if present else None,
            "rmse_gain": {},
            "day2000_rmse_to_climatology": {},
            "normalized_amplitude_ratio_to_truth": {},
            "post_day500_statistics": {},
            "spectrum_power_ratio_to_truth": {},
        }
        normalized = np.asarray(
            arrays["normalized_max_abs"][method_index],
            dtype=np.float64,
        )
        truth_normalized = np.asarray(
            arrays["truth_normalized_max_abs"][method_index],
            dtype=np.float64,
        )
        record["normalized_amplitude_ratio_to_truth"]["day2000_mean"] = (
            _finite_float(
                np.mean(normalized[:, -1]) / np.mean(truth_normalized[:, -1])
            )
            if np.all(np.isfinite(normalized[:, -1]))
            else None
        )
        valid_envelope = np.isfinite(normalized[:, post500])
        record["normalized_amplitude_ratio_to_truth"]["post_day500_finite_mean"] = (
            _safe_nanmean(
                normalized[:, post500][valid_envelope]
                / truth_normalized[:, post500][valid_envelope]
            )
        )
        for field in FIELDS:
            members = np.asarray(arrays[f"rmse__{method}__{field}"])
            record["rmse_gain"][field] = {
                f"{window[0]}_{window[1]}": safe_log_gain(
                    leads,
                    members,
                    window,
                )
                for window in FIT_WINDOWS
            }
            climate = np.asarray(arrays[f"rmse__climatology__{field}"])
            record["day2000_rmse_to_climatology"][field] = (
                _finite_float(
                    np.mean(members[:, -1]) / np.mean(climate[:, -1])
                )
                if np.all(np.isfinite(members[:, -1]))
                else None
            )
        for field in STAT_FIELDS:
            model_std = np.asarray(arrays[f"std__{method}__{field}"])[:, post500]
            truth_std = np.asarray(arrays[f"std__truth__{field}"])[:, post500]
            model_mean = np.asarray(arrays[f"mean__{method}__{field}"])[:, post500]
            truth_mean = np.asarray(arrays[f"mean__truth__{field}"])[:, post500]
            valid = np.isfinite(model_std) & np.isfinite(model_mean)
            statistics = {
                "finite_sample_count": int(valid.sum()),
                "spatial_std_ratio_to_truth": (
                    _finite_float(
                        np.mean(model_std[valid]) / np.mean(truth_std[valid])
                    )
                    if np.any(valid)
                    else None
                ),
                "mean_bias_in_truth_temporal_sd": (
                    _finite_float(
                        np.mean((model_mean - truth_mean)[valid])
                        / max(float(np.std(truth_mean[valid])), 1.0e-12)
                    )
                    if np.any(valid)
                    else None
                ),
            }
            if field == "streamfunction":
                model_min = np.asarray(
                    arrays[f"minimum__{method}__{field}"]
                )[:, post500]
                model_max = np.asarray(
                    arrays[f"maximum__{method}__{field}"]
                )[:, post500]
                truth_min = np.asarray(
                    arrays[f"minimum__truth__{field}"]
                )[:, post500]
                truth_max = np.asarray(
                    arrays[f"maximum__truth__{field}"]
                )[:, post500]
                statistics.update(
                    {
                        "minimum": _safe_nanmean(model_min),
                        "maximum": _safe_nanmean(model_max),
                        "mean_spatial_range_ratio_to_truth": (
                            _finite_float(
                                np.mean((model_max - model_min)[valid])
                                / np.mean((truth_max - truth_min)[valid])
                            )
                            if np.any(valid)
                            else None
                        ),
                    }
                )
            record["post_day500_statistics"][field] = statistics
            record["spectrum_power_ratio_to_truth"][field] = {}
            model_spectrum = np.asarray(arrays[f"spectrum__{method}__{field}"])
            truth_spectrum = np.asarray(arrays[f"spectrum__truth__{field}"])
            for lead_index, lead in enumerate(SPECTRUM_LEADS):
                record["spectrum_power_ratio_to_truth"][field][str(lead)] = {}
                for band_name, (lower, upper) in SPECTRUM_BANDS.items():
                    band = (modes >= lower) & (modes <= upper)
                    numerator = np.sum(
                        model_spectrum[:, lead_index, band],
                        axis=1,
                    )
                    denominator = np.sum(
                        truth_spectrum[:, lead_index, band],
                        axis=1,
                    )
                    ratio = numerator / np.maximum(denominator, 1.0e-300)
                    finite_ratio = ratio[np.isfinite(ratio)]
                    record["spectrum_power_ratio_to_truth"][field][str(lead)][
                        band_name
                    ] = {
                        "finite_member_count": int(finite_ratio.size),
                        "mean": (
                            float(finite_ratio.mean())
                            if finite_ratio.size
                            else None
                        ),
                        "p10": (
                            float(np.percentile(finite_ratio, 10))
                            if finite_ratio.size
                            else None
                        ),
                        "p90": (
                            float(np.percentile(finite_ratio, 90))
                            if finite_ratio.size
                            else None
                        ),
                    }
        summary["methods"][method] = record
    return summary


def _tangent_audit_safe(
    direct: Mapping[str, Any],
    state: Any,
    static: Any,
    wet: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.full(
        (len(DIRECT_METHODS), len(TANGENT_TIMES), len(TANGENT_BANDS), 3),
        np.nan,
        dtype=np.float64,
    )
    report: dict[str, Any] = {}
    for model_index, name in enumerate(DIRECT_METHODS):
        stepper = direct[name]
        report[name] = {}
        for time_index, time_value in enumerate(TANGENT_TIMES):
            physical = np.asarray(
                state[0, time_value : time_value + 1],
                dtype=np.float32,
            )
            current = stepper.normalized_state(physical)
            fixed_static = stepper.normalized_static(
                static,
                np.asarray([0], dtype=np.int64),
            )
            report[name][str(time_value)] = {}
            for band_index, (band_name, band) in enumerate(TANGENT_BANDS.items()):
                try:
                    result = dominant_singular_and_tangent_gain(
                        stepper,
                        current,
                        fixed_static,
                        _band_projector(wet, band),
                        seed=(
                            20260729
                            + 1000 * model_index
                            + 100 * time_index
                            + band_index
                        ),
                    )
                    numeric = np.asarray(
                        [
                            result["dominant_one_step_singular_gain"],
                            result["ten_call_tangent_geometric_gain_per_call"],
                            result["ten_call_tangent_total_gain"],
                        ],
                        dtype=np.float64,
                    )
                    status = "complete" if np.all(np.isfinite(numeric)) else "nonfinite"
                    if status == "complete":
                        values[model_index, time_index, band_index] = numeric
                    report[name][str(time_value)][band_name] = {
                        "status": status,
                        **{
                            key: _finite_float(value)
                            for key, value in result.items()
                        },
                    }
                except (RuntimeError, ValueError) as error:
                    report[name][str(time_value)][band_name] = {
                        "status": "failed_censored",
                        "error_type": type(error).__name__,
                    }
    return values, report


def _plot_log_rmse(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    leads = np.asarray(arrays["lead_days"])
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 8.6),
        sharex=True,
        constrained_layout=True,
    )
    for axis, field in zip(axes, FIELDS):
        for method in (*METHODS, "climatology"):
            values = np.asarray(arrays[f"rmse__{method}__{field}"])
            complete = np.all(np.isfinite(values), axis=0)
            mean = np.full(leads.shape, np.nan, dtype=np.float64)
            mean[complete] = np.mean(values[:, complete], axis=0)
            positive = (leads > 0) & complete
            axis.plot(
                leads[positive],
                mean[positive],
                color=COLORS[method],
                linewidth=1.5,
                label=LABELS[method],
            )
        axis.set_yscale("log")
        axis.set_ylabel(FIELD_LABELS[field])
        axis.grid(which="both", color="0.84", linewidth=0.55)
    axes[0].set_title("S0 recovery: curves end at first non-finite ensemble member")
    axes[-1].set_xlabel("Lead (days)")
    axes[-1].set_xlim(0, 2000)
    axes[-1].legend(loc="best", ncol=2)
    figure.savefig(output / FIGURES[0], bbox_inches="tight")
    plt.close(figure)


def _plot_normalized(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    leads = np.asarray(arrays["lead_days"])
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for method_index, method in enumerate(METHODS):
        model = np.asarray(arrays["normalized_max_abs"][method_index])
        truth = np.asarray(arrays["truth_normalized_max_abs"][method_index])
        complete = np.all(np.isfinite(model), axis=0)
        ratio = np.full(leads.shape, np.nan, dtype=np.float64)
        ratio[complete] = (
            np.mean(model[:, complete], axis=0)
            / np.mean(truth[:, complete], axis=0)
        )
        axis.plot(leads, ratio, color=COLORS[method], label=LABELS[method])
    axis.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    axis.set_yscale("log")
    axis.set_xlabel("Lead (days)")
    axis.set_ylabel("Mean maximum normalized amplitude / truth")
    axis.set_title("Method-native envelope; censored after first non-finite member")
    axis.grid(which="both", color="0.84", linewidth=0.55)
    axis.legend(loc="best")
    figure.savefig(output / FIGURES[1], bbox_inches="tight")
    plt.close(figure)


def _plot_spectra(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    modes = np.asarray(arrays["spectrum_modes"])
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), constrained_layout=True)
    for axis, field in zip(axes.flat, STAT_FIELDS):
        truth = np.asarray(arrays[f"spectrum__truth__{field}"])[:, -1]
        truth_mean = np.mean(truth, axis=0)
        for method in METHODS:
            model = np.asarray(arrays[f"spectrum__{method}__{field}"])[:, -1]
            if np.all(np.isfinite(model)):
                ratio = np.mean(model, axis=0) / np.maximum(truth_mean, 1.0e-300)
                axis.plot(
                    modes,
                    ratio,
                    color=COLORS[method],
                    label=LABELS[method],
                )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_yscale("log")
        axis.set_title(field.replace("_", " "))
        axis.set_xlabel("Radial wavenumber")
        axis.set_ylabel("Day-2000 power / truth")
        axis.grid(which="both", color="0.85", linewidth=0.5)
    axes[-1, -1].legend(loc="best")
    figure.savefig(output / FIGURES[2], bbox_inches="tight")
    plt.close(figure)


def _plot_tangent(output: Path, tangent: np.ndarray) -> None:
    bands = list(TANGENT_BANDS)
    x = np.arange(len(bands))
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
    for model_index, method in enumerate(DIRECT_METHODS):
        for axis, metric_index in zip(axes, (0, 1)):
            metric = tangent[model_index, :, :, metric_index]
            axis.errorbar(
                x + (-0.08 if model_index == 0 else 0.08),
                np.nanmean(metric, axis=0),
                yerr=np.nanstd(metric, axis=0),
                marker="o",
                capsize=3,
                color=COLORS[method],
                label=LABELS[method],
            )
    for axis in axes:
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
        axis.set_xticks(x, bands, rotation=15)
        axis.grid(color="0.85", linewidth=0.55)
        axis.legend(loc="best")
    axes[0].set_title("Estimated dominant one-step singular gain")
    axes[1].set_title("Ten-call tangent geometric gain per call")
    axes[0].set_ylabel("Gain")
    figure.savefig(output / FIGURES[3], bbox_inches="tight")
    plt.close(figure)


def _plot_nonfinite(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    leads = np.asarray(arrays["lead_days"], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    for method_index, method in enumerate(METHODS):
        first = _first_nonfinite_leads(arrays["finite"][method_index], leads)
        display = np.asarray(
            [value if value is not None else 2050 for value in first],
            dtype=np.float64,
        )
        axis.scatter(
            np.full(display.shape, method_index),
            display,
            color=COLORS[method],
            alpha=0.75,
        )
    axis.axhline(2000, color="black", linestyle="--", linewidth=0.9)
    axis.set_xticks(np.arange(len(METHODS)), [LABELS[name] for name in METHODS])
    axis.set_ylim(0, 2100)
    axis.set_ylabel("First non-finite lead (days)")
    axis.set_title("Each point is one of the 15 fixed S0 members; 2050 means none")
    axis.grid(axis="y", color="0.85", linewidth=0.55)
    figure.savefig(output / FIGURES[4], bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    leads = np.asarray(arrays["lead_days"])
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "metric",
                "method",
                "field",
                "lead_days",
                "finite_members",
                "mean",
                "p10",
                "p90",
            )
        )
        for method in (*METHODS, "climatology"):
            for field in FIELDS:
                values = np.asarray(arrays[f"rmse__{method}__{field}"])
                for lead_index, lead in enumerate(leads):
                    finite = values[:, lead_index]
                    finite = finite[np.isfinite(finite)]
                    writer.writerow(
                        (
                            "rmse",
                            method,
                            field,
                            int(lead),
                            int(finite.size),
                            float(finite.mean()) if finite.size else "",
                            float(np.percentile(finite, 10)) if finite.size else "",
                            float(np.percentile(finite, 90)) if finite.size else "",
                        )
                    )


def preflight(contract_path: str | Path) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("stability recovery requires PyTorch")
    contract, resolved, digest = load_contract(contract_path)
    failed_log = Path(contract["artifacts"]["failed_job_log"]["path"]).read_text()
    required_log_text = (
        "ValueError: log-gain fit needs at least three positive samples",
        "RuntimeWarning: overflow encountered",
    )
    if not all(text in failed_log for text in required_log_text):
        raise StabilityRecoveryError("job 304750 failure signature changed")
    original_contract_path = Path(
        contract["artifacts"]["original_contract"]["path"]
    )
    from .af_model_c_s0_stability_tangent_comparison import preflight as v1_preflight

    original = v1_preflight(original_contract_path)
    if original["contract_sha256"] != contract["original_contract_sha256"]:
        raise StabilityRecoveryError("original contract digest changed")
    dataset = Path(contract["artifacts"]["dataset_metadata"]["path"]).parent
    group = zarr.open_consolidated(str(dataset), mode="r")
    if tuple(group.attrs["state_channels"]) != STATE_CHANNELS:
        raise StabilityRecoveryError("trajectory channels changed")
    return {
        "status": "pass",
        "contract": str(resolved),
        "contract_sha256": digest,
        "failed_job_id": FAILED_JOB_ID,
        "original_contract_sha256": original["contract_sha256"],
        "member_count": len(EXPECTED_STARTS),
        "long_calls": 200,
        "retraining_steps": 0,
        "checkpoint_selection": False,
        "recovery_changes": contract["recovery_changes"],
        "one_step_smoke": original["one_step_smoke"],
    }


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("stability recovery requires PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    preflight(resolved_contract)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested without GPU")
    device = torch.device(device_name)
    dataset = Path(contract["artifacts"]["dataset_metadata"]["path"]).parent
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float32)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    mean, scale, _, _, wind_mean, wind_scale = _normalizers(group)
    normalization = Path(contract["artifacts"]["pointwise_normalization"]["path"])
    architecture = contract["architecture"]
    selected = _pointwise_stepper(
        Path(contract["artifacts"]["selected_checkpoint"]["path"]),
        normalization,
        architecture,
        device,
        wet,
        wind_mean,
        wind_scale,
        layernorm=False,
    )
    layernorm = _pointwise_stepper(
        Path(contract["artifacts"]["layernorm_checkpoint"]["path"]),
        normalization,
        architecture,
        device,
        wet,
        wind_mean,
        wind_scale,
        layernorm=True,
    )
    figure_contract = json.loads(
        Path(contract["artifacts"]["job304736_contract"]["path"]).read_text()
    )
    prior = _prior_stepper(
        figure_contract,
        device,
        wet,
        latitude,
        static,
        mean,
        scale,
        wind_mean,
        wind_scale,
    )
    steppers = {
        "selected": selected,
        "layernorm": layernorm,
        "prior_residual": prior,
    }
    long_result = json.loads(
        Path(contract["artifacts"]["long_truth_result"]["path"]).read_text()
    )
    truth = ContinuousS0Truth(state, Path(long_result["run_dir"]), wet)
    _, climatology_derived, count = _s0_training_climatology(
        state,
        snapshot_codes,
        wet,
    )
    if count != 5040:
        raise StabilityRecoveryError("training climatology count changed")
    arrays = _long_rollout(
        steppers,
        truth,
        static,
        np.asarray(EXPECTED_STARTS, dtype=np.int64),
        climatology_derived,
        wet,
    )
    tangent_values, tangent_report = _tangent_audit_safe(
        {"selected": selected, "layernorm": layernorm},
        state,
        static,
        wet,
    )
    arrays["tangent_values"] = tangent_values
    arrays["tangent_training_times"] = np.asarray(TANGENT_TIMES, dtype=np.int32)
    arrays["tangent_band_names"] = np.asarray(list(TANGENT_BANDS))
    summary = _summarize(arrays, tangent_report)

    scratch = Path(contract["output"]["scratch"]).resolve()
    project = Path(contract["output"]["project"]).resolve()
    scratch_tmp = scratch.with_name(scratch.name + ".tmp")
    project_tmp = project.with_name(project.name + ".tmp")
    scratch_tmp.parent.mkdir(parents=True, exist_ok=True)
    project_tmp.parent.mkdir(parents=True, exist_ok=True)
    scratch_tmp.mkdir(exist_ok=False)
    project_tmp.mkdir(exist_ok=False)
    try:
        np.savez_compressed(scratch_tmp / RECOVERY_ARRAYS, **arrays)
        plt.rcParams.update({"font.size": 9, "figure.dpi": 160})
        _plot_log_rmse(project_tmp, arrays)
        _plot_normalized(project_tmp, arrays)
        _plot_spectra(project_tmp, arrays)
        _plot_tangent(project_tmp, tangent_values)
        _plot_nonfinite(project_tmp, arrays)
        _write_csv(project_tmp / RECOVERY_CSV, arrays)
        (project_tmp / SUMMARY).write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        report = {
            "version": VERSION,
            "status": "complete",
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "failed_job_id": FAILED_JOB_ID,
            "original_contract_sha256": contract["original_contract_sha256"],
            "classification_scope": (
                "zero_retraining_recovery_no_checkpoint_selection_or_promotion"
            ),
            "recovery_changes": contract["recovery_changes"],
            "summary": summary,
            "arrays": str(scratch / RECOVERY_ARRAYS),
            "arrays_sha256": file_sha256(scratch_tmp / RECOVERY_ARRAYS),
            "figures": list(FIGURES),
            "elapsed_seconds": time.monotonic() - started,
            "device": str(device),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        report["report_content_sha256"] = json_sha256(report)
        (scratch_tmp / REPORT).write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        shutil.copy2(scratch_tmp / REPORT, project_tmp / REPORT)
        shutil.copy2(scratch_tmp / RECOVERY_ARRAYS, project_tmp / RECOVERY_ARRAYS)
        (project_tmp / README).write_text(
            "# Model C S0 stability/tangent evaluation recovery\n\n"
            "Job 304750 failed during float32 diagnostic reduction after model "
            "rollout. This zero-retraining recovery uses float64 physical "
            "reductions and explicitly censors metrics after a member's first "
            "non-finite model state. It changes no model or checkpoint.\n\n"
            f"Report content SHA-256: `{report['report_content_sha256']}`.\n"
        )
        manifest = {
            "version": VERSION,
            "contract_sha256": contract_sha,
            "report_content_sha256": report["report_content_sha256"],
            "artifacts": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in sorted(project_tmp.iterdir())
                if path.is_file()
            },
        }
        manifest["manifest_content_sha256"] = json_sha256(manifest)
        (project_tmp / MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        scratch_tmp.replace(scratch)
        project_tmp.replace(project)
    except Exception:
        shutil.rmtree(scratch_tmp, ignore_errors=True)
        shutil.rmtree(project_tmp, ignore_errors=True)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("preflight")
    check.add_argument("--contract", type=Path, required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--contract", type=Path, required=True)
    execute.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    result = (
        preflight(args.contract)
        if args.command == "preflight"
        else run(args.contract, device_name=args.device)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
