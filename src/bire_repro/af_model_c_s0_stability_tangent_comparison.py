"""Zero-retraining S0 stability and tangent-gain comparison for Model C.

The job compares three already-trained maps:

* selected pointwise-anomaly/direct-state Model C;
* the diagnostic-best LayerNorm-only control;
* the prior contracting residual Model C.

All three are rolled on the already-open 15-member S0 day-2000 protocol.
One-step dominant singular gain and ten-call tangent growth are evaluated only
for the two direct-state maps, in their shared pointwise-normalized coordinates,
at fixed training-only S0 states.  No quantity is called a spectral radius.
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
from typing import Any, Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from .af_a0_evaluate import _normalizers
from .af_data import STATE_CHANNELS
from .af_forward_complete import _member_rmse, derived_fields, radial_spectrum
from .af_model_c_anomaly_direct import PointwiseDirectStepper
from .af_model_c_anomaly_direct_bire_regularization_controls import (
    RegularizationArm,
    build_regularized_model,
)
from .af_model_c_bire_s0_figures import (
    ContinuousS0Truth,
    _prior_stepper,
    _s0_training_climatology,
)
from .af_model_c_bire_s0_long_truth import EXPECTED_STARTS
from .af_model_c_s0_stability_instrument import (
    bootstrap_gain_interval,
    fit_log_gain,
)
from .af_model_c_successor import ModelCSuccessorArchitecture, build_successor

try:
    import torch
except (ImportError, OSError):  # pragma: no cover
    torch = None  # type: ignore[assignment]


VERSION = "model_c_s0_stability_tangent_comparison_v1"
CONTRACT_STATUS = "frozen_zero_retraining_after_stability_instrument_correction"
METHODS = ("selected", "layernorm", "prior_residual")
DIRECT_METHODS = ("selected", "layernorm")
FIELDS = ("surface_speed", "phihyd_surface", "sst")
STAT_FIELDS = (*FIELDS, "streamfunction")
LEAD_DAYS = tuple(range(0, 2001, 10))
SPECTRUM_LEADS = (200, 500, 1000, 1500, 2000)
SPECTRUM_BANDS = {
    "low_k1_3": (1, 3),
    "mid_k4_9": (4, 9),
    "high_k10_30": (10, 30),
}
FIT_WINDOWS = ((300, 600), (700, 1000), (1700, 2000))
TANGENT_TIMES = (431, 2159, 4985)
TANGENT_BANDS = {
    "full": None,
    "low_k1_3": (1, 3),
    "mid_k4_9": (4, 9),
    "high_k10_30": (10, 30),
}
POWER_ITERATIONS = 5
TANGENT_CALLS = 10
FIGURES = (
    "model_c_s0_stability_models_log_rmse.png",
    "model_c_s0_stability_normalized_envelope.png",
    "model_c_s0_stability_day2000_spectra.png",
    "model_c_s0_tangent_gain_comparison.png",
)
ARRAYS = "model_c_s0_stability_tangent_arrays.npz"
REPORT = "model_c_s0_stability_tangent_report.json"
SUMMARY = "model_c_s0_stability_tangent_summary.json"
CSV = "model_c_s0_stability_tangent_curves.csv"
README = "README.md"
MANIFEST = "manifest.json"

COLORS = {
    "selected": "#D62728",
    "layernorm": "#9467BD",
    "prior_residual": "#2CA02C",
    "climatology": "#111111",
}
LABELS = {
    "selected": "Selected anomaly-direct",
    "layernorm": "LayerNorm diagnostic",
    "prior_residual": "Prior residual",
    "climatology": "Climatology",
}
FIELD_LABELS = {
    "surface_speed": r"Surface speed (m s$^{-1}$)",
    "phihyd_surface": r"Surface $P/\rho$ (m$^2$ s$^{-2}$)",
    "sst": r"SST ($^\circ$C)",
}


class StabilityTangentError(RuntimeError):
    """Raised when the frozen diagnostic changes."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_file(record: Mapping[str, Any], label: str) -> Path:
    path = Path(record["path"]).resolve()
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise StabilityTangentError(f"immutable artifact changed: {label}")
    return path


def load_contract(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    protocol = contract.get("protocol", {})
    tangent = contract.get("tangent", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or protocol.get("wind_regime") != "S0"
        or float(protocol.get("wind_stress_n_m2", -1.0)) != 0.1
        or int(protocol.get("dt_days", -1)) != 10
        or tuple(protocol.get("lead_days", ())) != (0, 2000)
        or int(protocol.get("calls", -1)) != 200
        or tuple(protocol.get("start_indices", ())) != EXPECTED_STARTS
        or tuple(protocol.get("methods", ())) != METHODS
        or tuple(protocol.get("fields", ())) != FIELDS
        or tuple(protocol.get("statistical_fields", ())) != STAT_FIELDS
        or tuple(protocol.get("spectrum_leads", ())) != SPECTRUM_LEADS
        or protocol.get("spectrum_bands")
        != {name: list(band) for name, band in SPECTRUM_BANDS.items()}
        or tuple(map(tuple, protocol.get("fit_windows", ()))) != FIT_WINDOWS
        or tuple(tangent.get("methods", ())) != DIRECT_METHODS
        or tuple(tangent.get("training_times", ())) != TANGENT_TIMES
        or tangent.get("bands")
        != {
            name: None if band is None else list(band)
            for name, band in TANGENT_BANDS.items()
        }
        or int(tangent.get("power_iterations", -1)) != POWER_ITERATIONS
        or int(tangent.get("finite_time_calls", -1)) != TANGENT_CALLS
        or tangent.get("coordinate_system")
        != "shared_pointwise_normalized_state_selected_and_layernorm_only"
    ):
        raise StabilityTangentError("stability/tangent contract changed")
    for label, record in contract["artifacts"].items():
        _verify_file(record, label)
    root = resolved.parents[1]
    for relative, expected in contract["source_hashes"].items():
        source = root / relative
        if not source.is_file() or file_sha256(source) != expected:
            raise StabilityTangentError(f"source changed: {relative}")
    for key in ("scratch", "project"):
        output = Path(contract["output"][key]).resolve()
        if output.exists() or output.with_name(output.name + ".tmp").exists():
            raise FileExistsError(f"refusing to overwrite output: {output}")
    return contract, resolved, file_sha256(resolved)


def _pointwise_stepper(
    checkpoint: Path,
    normalization: Path,
    architecture_record: Mapping[str, Any],
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
    *,
    layernorm: bool,
) -> PointwiseDirectStepper:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != architecture_record:
        raise StabilityTangentError("direct checkpoint architecture changed")
    architecture = ModelCSuccessorArchitecture(**architecture_record)
    if layernorm:
        model = build_regularized_model(
            architecture,
            RegularizationArm("layernorm", True, 0.0),
        ).to(device)
    else:
        model = build_successor(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with np.load(normalization) as artifact:
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


def _allocate(member_count: int, spectrum_modes: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "lead_days": np.asarray(LEAD_DAYS, dtype=np.int16),
        "spectrum_leads": np.asarray(SPECTRUM_LEADS, dtype=np.int16),
        "start_draw_order": np.asarray(EXPECTED_STARTS, dtype=np.int32),
        "finite": np.empty((len(METHODS), member_count, len(LEAD_DAYS)), dtype=np.uint8),
        "normalized_max_abs": np.empty(
            (len(METHODS), member_count, len(LEAD_DAYS)),
            dtype=np.float32,
        ),
        "truth_normalized_max_abs": np.empty(
            (len(METHODS), member_count, len(LEAD_DAYS)),
            dtype=np.float32,
        ),
        "spectrum_modes": np.arange(1, spectrum_modes + 1, dtype=np.float32),
    }
    for method in METHODS:
        for field in FIELDS:
            arrays[f"rmse__{method}__{field}"] = np.empty(
                (member_count, len(LEAD_DAYS)),
                dtype=np.float32,
            )
        for field in STAT_FIELDS:
            statistics = (
                ("mean", "std", "minimum", "maximum")
                if field == "streamfunction"
                else ("mean", "std")
            )
            for statistic in statistics:
                arrays[f"{statistic}__{method}__{field}"] = np.empty(
                    (member_count, len(LEAD_DAYS)),
                    dtype=np.float32,
                )
            arrays[f"spectrum__{method}__{field}"] = np.empty(
                (member_count, len(SPECTRUM_LEADS), spectrum_modes),
                dtype=np.float32,
            )
    for field in FIELDS:
        arrays[f"rmse__climatology__{field}"] = np.empty(
            (member_count, len(LEAD_DAYS)),
            dtype=np.float32,
        )
    for field in STAT_FIELDS:
        statistics = (
            ("mean", "std", "minimum", "maximum")
            if field == "streamfunction"
            else ("mean", "std")
        )
        for statistic in statistics:
            arrays[f"{statistic}__truth__{field}"] = np.empty(
                (member_count, len(LEAD_DAYS)),
                dtype=np.float32,
            )
        arrays[f"spectrum__truth__{field}"] = np.empty(
            (member_count, len(SPECTRUM_LEADS), spectrum_modes),
            dtype=np.float32,
        )
    return arrays


def _spatial_stats(fields: Mapping[str, np.ndarray], wet: np.ndarray) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for field in STAT_FIELDS:
        selected = np.asarray(fields[field])[:, wet]
        values[f"mean__{field}"] = selected.mean(axis=1)
        values[f"std__{field}"] = selected.std(axis=1)
        if field == "streamfunction":
            values[f"minimum__{field}"] = selected.min(axis=1)
            values[f"maximum__{field}"] = selected.max(axis=1)
    return values


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
    _, trial_spectrum = radial_spectrum(
        derived_fields(initial[:1], wet)["sst"],
        wet,
    )
    arrays = _allocate(member_count, trial_spectrum.shape[-1])
    wet_tensor = torch.from_numpy(wet).to(device=next(iter(steppers.values())).device)
    spectrum_lookup = {lead: index for index, lead in enumerate(SPECTRUM_LEADS)}
    climate = {
        field: np.repeat(climatology_derived[field][None], member_count, axis=0)
        for field in FIELDS
    }

    with torch.no_grad():
        for lead_index, lead in enumerate(LEAD_DAYS):
            if lead:
                for name, stepper in steppers.items():
                    currents[name] = stepper.step(currents[name], statics[name])
            physical = {
                name: stepper.physical(currents[name])
                for name, stepper in steppers.items()
            }
            truth_state = truth.batch(starts + lead)
            truth_fields = derived_fields(truth_state, wet)
            truth_stats = _spatial_stats(truth_fields, wet)
            for key, values in truth_stats.items():
                statistic, field = key.split("__")
                arrays[f"{statistic}__truth__{field}"][:, lead_index] = values
            for field in FIELDS:
                arrays[f"rmse__climatology__{field}"][:, lead_index] = _member_rmse(
                    climate[field],
                    truth_fields[field],
                    wet,
                )
            method_fields = {
                name: derived_fields(physical[name], wet)
                for name in steppers
            }
            for method_index, (name, stepper) in enumerate(steppers.items()):
                fields = method_fields[name]
                for field in FIELDS:
                    arrays[f"rmse__{name}__{field}"][:, lead_index] = _member_rmse(
                        fields[field],
                        truth_fields[field],
                        wet,
                    )
                stats = _spatial_stats(fields, wet)
                for key, values in stats.items():
                    statistic, field = key.split("__")
                    arrays[f"{statistic}__{name}__{field}"][:, lead_index] = values
                arrays["finite"][method_index, :, lead_index] = np.isfinite(
                    physical[name]
                ).all(axis=(1, 2, 3))
                arrays["normalized_max_abs"][method_index, :, lead_index] = (
                    torch.amax(
                        torch.abs(currents[name][:, :, wet_tensor]),
                        dim=(1, 2),
                    )
                    .cpu()
                    .numpy()
                )
                normalized_truth = stepper.normalized_state(truth_state)
                arrays["truth_normalized_max_abs"][method_index, :, lead_index] = (
                    torch.amax(
                        torch.abs(normalized_truth[:, :, wet_tensor]),
                        dim=(1, 2),
                    )
                    .cpu()
                    .numpy()
                )
            if lead in spectrum_lookup:
                spectrum_index = spectrum_lookup[lead]
                for field in STAT_FIELDS:
                    modes, values = radial_spectrum(truth_fields[field], wet)
                    if not np.array_equal(modes, arrays["spectrum_modes"]):
                        raise StabilityTangentError("radial spectrum modes changed")
                    arrays[f"spectrum__truth__{field}"][:, spectrum_index] = values
                    for name in steppers:
                        _, values = radial_spectrum(method_fields[name][field], wet)
                        arrays[f"spectrum__{name}__{field}"][:, spectrum_index] = values
    return arrays


def _band_projector(
    wet: np.ndarray,
    band: tuple[int, int] | None,
) -> Callable[[Any], Any]:
    rows, columns = np.where(wet)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    height, width = y1 - y0, x1 - x0
    ky = torch.fft.fftfreq(height) * height
    kx = torch.fft.rfftfreq(width) * width
    radius = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    if band is None:
        spectral_mask = torch.ones_like(radius, dtype=torch.bool)
    else:
        spectral_mask = (radius >= band[0]) & (radius <= band[1])

    def project(value: Any) -> Any:
        cropped = value[..., y0:y1, x0:x1]
        transformed = torch.fft.rfft2(cropped)
        mask = spectral_mask.to(device=value.device)
        filtered = torch.fft.irfft2(
            transformed * mask[None, None],
            s=(height, width),
        )
        result = torch.zeros_like(value)
        result[..., y0:y1, x0:x1] = filtered
        return result

    return project


def _norm(value: Any) -> Any:
    return torch.linalg.vector_norm(value)


def dominant_singular_and_tangent_gain(
    stepper: PointwiseDirectStepper,
    current: Any,
    static: Any,
    projector: Callable[[Any], Any],
    *,
    seed: int,
) -> dict[str, float]:
    """Estimate one-step singular gain and ten-call tangent growth."""

    generator = torch.Generator(device=current.device)
    generator.manual_seed(seed)
    vector = torch.randn(
        current.shape,
        generator=generator,
        device=current.device,
        dtype=current.dtype,
    )
    vector = projector(vector)
    vector = vector / torch.clamp(_norm(vector), min=1.0e-12)

    def mapping(value: Any) -> Any:
        return stepper.step(value, static)

    base = current.detach()
    for _ in range(POWER_ITERATIONS):
        _, jv = torch.autograd.functional.jvp(
            mapping,
            base,
            vector,
            create_graph=False,
            strict=False,
        )
        base_for_vjp = base.detach().requires_grad_(True)
        output = mapping(base_for_vjp)
        jt_j_v = torch.autograd.grad(
            output,
            base_for_vjp,
            grad_outputs=jv,
            retain_graph=False,
            create_graph=False,
        )[0]
        vector = projector(jt_j_v.detach())
        vector = vector / torch.clamp(_norm(vector), min=1.0e-12)
    _, jv = torch.autograd.functional.jvp(
        mapping,
        base,
        vector,
        create_graph=False,
        strict=False,
    )
    singular_gain = float((_norm(jv) / _norm(vector)).detach().cpu())

    tangent = vector
    trajectory = base
    log_gain = 0.0
    for _ in range(TANGENT_CALLS):
        trajectory, tangent_next = torch.autograd.functional.jvp(
            mapping,
            trajectory,
            tangent,
            create_graph=False,
            strict=False,
        )
        gain = torch.clamp(_norm(tangent_next), min=1.0e-12)
        log_gain += float(torch.log(gain).detach().cpu())
        tangent = tangent_next / gain
        trajectory = trajectory.detach()
    return {
        "dominant_one_step_singular_gain": singular_gain,
        "ten_call_tangent_geometric_gain_per_call": float(
            np.exp(log_gain / TANGENT_CALLS)
        ),
        "ten_call_tangent_total_gain": float(np.exp(log_gain)),
    }


def _tangent_audit(
    direct: Mapping[str, PointwiseDirectStepper],
    state: Any,
    static: Any,
    wet: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.empty(
        (
            len(DIRECT_METHODS),
            len(TANGENT_TIMES),
            len(TANGENT_BANDS),
            3,
        ),
        dtype=np.float64,
    )
    report: dict[str, Any] = {}
    for model_index, name in enumerate(DIRECT_METHODS):
        stepper = direct[name]
        report[name] = {}
        for time_index, time_value in enumerate(TANGENT_TIMES):
            physical = np.asarray(state[0, time_value : time_value + 1], dtype=np.float32)
            current = stepper.normalized_state(physical)
            fixed_static = stepper.normalized_static(
                static,
                np.asarray([0], dtype=np.int64),
            )
            report[name][str(time_value)] = {}
            for band_index, (band_name, band) in enumerate(TANGENT_BANDS.items()):
                result = dominant_singular_and_tangent_gain(
                    stepper,
                    current,
                    fixed_static,
                    _band_projector(wet, band),
                    seed=20260729 + 1000 * model_index + 100 * time_index + band_index,
                )
                values[model_index, time_index, band_index] = (
                    result["dominant_one_step_singular_gain"],
                    result["ten_call_tangent_geometric_gain_per_call"],
                    result["ten_call_tangent_total_gain"],
                )
                report[name][str(time_value)][band_name] = result
    return values, report


def _summarize(
    arrays: Mapping[str, np.ndarray],
    tangent_report: Mapping[str, Any],
) -> dict[str, Any]:
    leads = np.asarray(arrays["lead_days"], dtype=np.int64)
    summary: dict[str, Any] = {
        "classification_scope": (
            "zero_retraining_causal_comparison_only_no_checkpoint_promotion"
        ),
        "methods": {},
        "tangent": tangent_report,
    }
    for method_index, method in enumerate(METHODS):
        record: dict[str, Any] = {
            "all_states_finite": bool(np.all(arrays["finite"][method_index])),
            "rmse_gain": {},
            "day2000_rmse_to_climatology": {},
            "normalized_amplitude_ratio_to_truth": {},
            "post_day500_statistics": {},
            "spectrum_power_ratio_to_truth": {},
        }
        normalized = arrays["normalized_max_abs"][method_index]
        truth_normalized = arrays["truth_normalized_max_abs"][method_index]
        record["normalized_amplitude_ratio_to_truth"]["day2000_mean"] = float(
            normalized[:, -1].mean() / truth_normalized[:, -1].mean()
        )
        mask = leads >= 500
        record["normalized_amplitude_ratio_to_truth"]["post_day500_mean"] = float(
            normalized[:, mask].mean() / truth_normalized[:, mask].mean()
        )
        for field_index, field in enumerate(FIELDS):
            members = np.asarray(arrays[f"rmse__{method}__{field}"])
            record["rmse_gain"][field] = {}
            for window_index, window in enumerate(FIT_WINDOWS):
                gain, e_folding = fit_log_gain(leads, members.mean(axis=0), window)
                interval = bootstrap_gain_interval(
                    leads,
                    members,
                    window,
                    seed=20260729
                    + 1000 * method_index
                    + 100 * field_index
                    + window_index,
                )
                record["rmse_gain"][field][f"{window[0]}_{window[1]}"] = {
                    "gain": gain,
                    "bootstrap_95_percent_interval": list(interval),
                    "e_folding_days_if_gain_gt_1": e_folding,
                }
            climate = np.asarray(arrays[f"rmse__climatology__{field}"])
            record["day2000_rmse_to_climatology"][field] = float(
                members[:, -1].mean() / climate[:, -1].mean()
            )
        for field in STAT_FIELDS:
            model_std = np.asarray(arrays[f"std__{method}__{field}"])[:, mask]
            truth_std = np.asarray(arrays[f"std__truth__{field}"])[:, mask]
            model_mean = np.asarray(arrays[f"mean__{method}__{field}"])[:, mask]
            truth_mean = np.asarray(arrays[f"mean__truth__{field}"])[:, mask]
            mean_scale = max(float(np.std(truth_mean)), 1.0e-12)
            record["post_day500_statistics"][field] = {
                "spatial_std_ratio_to_truth": float(
                    model_std.mean() / truth_std.mean()
                ),
                "mean_bias_in_truth_temporal_sd": float(
                    np.mean(model_mean - truth_mean) / mean_scale
                ),
            }
            if field == "streamfunction":
                model_minimum = np.asarray(
                    arrays[f"minimum__{method}__{field}"]
                )[:, mask]
                model_maximum = np.asarray(
                    arrays[f"maximum__{method}__{field}"]
                )[:, mask]
                truth_minimum = np.asarray(arrays[f"minimum__truth__{field}"])[
                    :, mask
                ]
                truth_maximum = np.asarray(arrays[f"maximum__truth__{field}"])[
                    :, mask
                ]
                model_range = model_maximum - model_minimum
                truth_range = truth_maximum - truth_minimum
                record["post_day500_statistics"][field].update(
                    {
                        "minimum": float(model_minimum.min()),
                        "maximum": float(model_maximum.max()),
                        "truth_minimum": float(truth_minimum.min()),
                        "truth_maximum": float(truth_maximum.max()),
                        "mean_spatial_range_ratio_to_truth": float(
                            model_range.mean() / truth_range.mean()
                        ),
                    }
                )
            modes = np.asarray(arrays["spectrum_modes"])
            record["spectrum_power_ratio_to_truth"][field] = {}
            truth_spectrum = np.asarray(arrays[f"spectrum__truth__{field}"])
            model_spectrum = np.asarray(arrays[f"spectrum__{method}__{field}"])
            for lead_index, lead in enumerate(SPECTRUM_LEADS):
                record["spectrum_power_ratio_to_truth"][field][str(lead)] = {}
                for band_name, (lower, upper) in SPECTRUM_BANDS.items():
                    band = (modes >= lower) & (modes <= upper)
                    ratio = (
                        model_spectrum[:, lead_index, band].sum(axis=1)
                        / np.maximum(
                            truth_spectrum[:, lead_index, band].sum(axis=1),
                            1.0e-30,
                        )
                    )
                    record["spectrum_power_ratio_to_truth"][field][str(lead)][
                        band_name
                    ] = {
                        "mean": float(ratio.mean()),
                        "p10": float(np.percentile(ratio, 10)),
                        "p90": float(np.percentile(ratio, 90)),
                    }
        summary["methods"][method] = record
    return summary


def _plot_log_rmse(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    leads = np.asarray(arrays["lead_days"])
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.6), sharex=True, constrained_layout=True)
    for axis, field in zip(axes, FIELDS):
        for method in (*METHODS, "climatology"):
            values = np.asarray(arrays[f"rmse__{method}__{field}"]).mean(axis=0)
            positive = leads > 0
            axis.plot(
                leads[positive],
                values[positive],
                color=COLORS[method],
                linewidth=1.5,
                label=LABELS[method],
            )
        axis.set_yscale("log")
        axis.set_ylabel(FIELD_LABELS[field])
        axis.grid(which="both", color="0.84", linewidth=0.55)
    axes[0].set_title("S0 day-2000 stability comparison; no retraining or reselection")
    axes[-1].set_xlabel("Lead (days)")
    axes[-1].set_xlim(0, 2000)
    axes[-1].legend(loc="best", ncol=2)
    figure.savefig(output / FIGURES[0], bbox_inches="tight")
    plt.close(figure)


def _plot_normalized(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    leads = np.asarray(arrays["lead_days"])
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for method_index, method in enumerate(METHODS):
        ratio = (
            arrays["normalized_max_abs"][method_index].mean(axis=0)
            / arrays["truth_normalized_max_abs"][method_index].mean(axis=0)
        )
        axis.plot(leads, ratio, color=COLORS[method], label=LABELS[method])
    axis.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    axis.set_yscale("log")
    axis.set_xlabel("Lead (days)")
    axis.set_ylabel("Mean maximum normalized amplitude / truth")
    axis.set_title("Method-native normalized amplitude relative to same-time truth")
    axis.grid(which="both", color="0.84", linewidth=0.55)
    axis.legend(loc="best")
    figure.savefig(output / FIGURES[1], bbox_inches="tight")
    plt.close(figure)


def _plot_spectra(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    modes = np.asarray(arrays["spectrum_modes"])
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), constrained_layout=True)
    for axis, field in zip(axes.flat, STAT_FIELDS):
        truth = arrays[f"spectrum__truth__{field}"][:, -1].mean(axis=0)
        for method in METHODS:
            model = arrays[f"spectrum__{method}__{field}"][:, -1].mean(axis=0)
            axis.plot(
                modes,
                model / np.maximum(truth, 1.0e-30),
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
        one_step = tangent[model_index, :, :, 0]
        finite = tangent[model_index, :, :, 1]
        axes[0].errorbar(
            x + (-0.08 if model_index == 0 else 0.08),
            one_step.mean(axis=0),
            yerr=one_step.std(axis=0),
            marker="o",
            capsize=3,
            color=COLORS[method],
            label=LABELS[method],
        )
        axes[1].errorbar(
            x + (-0.08 if model_index == 0 else 0.08),
            finite.mean(axis=0),
            yerr=finite.std(axis=0),
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


def _write_csv(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    leads = np.asarray(arrays["lead_days"])
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "method", "field", "lead_days", "mean", "p10", "p90"))
        for method in (*METHODS, "climatology"):
            for field in FIELDS:
                values = np.asarray(arrays[f"rmse__{method}__{field}"])
                mean = values.mean(axis=0)
                p10, p90 = np.percentile(values, (10, 90), axis=0)
                for lead, center, lower, upper in zip(leads, mean, p10, p90):
                    writer.writerow(("rmse", method, field, int(lead), center, lower, upper))


def preflight(contract_path: str | Path) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("stability/tangent comparison requires PyTorch")
    contract, resolved, digest = load_contract(contract_path)
    dataset = Path(contract["artifacts"]["dataset_metadata"]["path"]).parent
    group = zarr.open_consolidated(str(dataset), mode="r")
    if tuple(group.attrs["state_channels"]) != STATE_CHANNELS:
        raise StabilityTangentError("trajectory channels changed")
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    if any(int(split[index]) != 1 for index in TANGENT_TIMES):
        raise StabilityTangentError("tangent state is not training-only")
    result = json.loads(
        Path(contract["artifacts"]["long_truth_result"]["path"]).read_text()
    )
    if (
        result.get("slurm_job_id") != "304735"
        or result.get("returncode") != 0
        or max(EXPECTED_STARTS) + 2000 >= 9360
    ):
        raise StabilityTangentError("long truth is incomplete")
    device = torch.device("cpu")
    state = group["state"]
    static = group["static_features"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float32)
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
    physical = np.asarray(
        state[0, TANGENT_TIMES[0] : TANGENT_TIMES[0] + 1],
        dtype=np.float32,
    )
    experiments = np.asarray([0], dtype=np.int64)
    for name, stepper in {
        "selected": selected,
        "layernorm": layernorm,
        "prior_residual": prior,
    }.items():
        current = stepper.normalized_state(physical)
        fixed_static = stepper.normalized_static(static, experiments)
        with torch.no_grad():
            prediction = stepper.physical(stepper.step(current, fixed_static))
        if not np.isfinite(prediction).all():
            raise StabilityTangentError(f"{name} failed finite one-step smoke test")
        if np.any(prediction[:, :, ~wet] != 0):
            raise StabilityTangentError(f"{name} failed land-zero one-step smoke test")
    return {
        "status": "pass",
        "contract": str(resolved),
        "contract_sha256": digest,
        "methods": list(METHODS),
        "member_count": len(EXPECTED_STARTS),
        "long_calls": 200,
        "tangent_training_times": list(TANGENT_TIMES),
        "tangent_bands": list(TANGENT_BANDS),
        "retraining": False,
        "checkpoint_selection": False,
        "one_step_smoke": {
            "methods": list(METHODS),
            "finite": True,
            "land_zero": True,
        },
    }


def run(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("stability/tangent comparison requires PyTorch")
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
    climatology_state, climatology_derived, count = _s0_training_climatology(
        state,
        snapshot_codes,
        wet,
    )
    if count != 5040:
        raise StabilityTangentError("training climatology count changed")
    del climatology_state
    arrays = _long_rollout(
        steppers,
        truth,
        static,
        np.asarray(EXPECTED_STARTS, dtype=np.int64),
        climatology_derived,
        wet,
    )
    tangent_values, tangent_report = _tangent_audit(
        {"selected": selected, "layernorm": layernorm},
        state,
        static,
        wet,
    )
    arrays["tangent_values"] = tangent_values.astype(np.float32)
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
        np.savez_compressed(scratch_tmp / ARRAYS, **arrays)
        plt.rcParams.update({"font.size": 9, "figure.dpi": 160})
        _plot_log_rmse(project_tmp, arrays)
        _plot_normalized(project_tmp, arrays)
        _plot_spectra(project_tmp, arrays)
        _plot_tangent(project_tmp, tangent_values)
        _write_csv(project_tmp / CSV, arrays)
        (project_tmp / SUMMARY).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        report = {
            "version": VERSION,
            "status": "complete",
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "classification_scope": (
                "zero_retraining_causal_comparison_no_checkpoint_promotion"
            ),
            "methods": list(METHODS),
            "summary": summary,
            "arrays": str(scratch / ARRAYS),
            "arrays_sha256": file_sha256(scratch_tmp / ARRAYS),
            "figures": list(FIGURES),
            "elapsed_seconds": time.monotonic() - started,
            "device": str(device),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        report["report_content_sha256"] = json_sha256(report)
        (scratch_tmp / REPORT).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        shutil.copy2(scratch_tmp / REPORT, project_tmp / REPORT)
        shutil.copy2(scratch_tmp / ARRAYS, project_tmp / ARRAYS)
        (project_tmp / README).write_text(
            "# Model C S0 stability and tangent comparison\n\n"
            "Zero retraining and no checkpoint selection. The fitted RMSE gain "
            "is not a Jacobian spectral radius. Tangent metrics compare only "
            "the selected and LayerNorm direct-state maps in shared coordinates.\n\n"
            f"Report content SHA-256: `{report['report_content_sha256']}`.\n"
        )
        manifest = {
            "version": VERSION,
            "contract_sha256": contract_sha,
            "report_content_sha256": report["report_content_sha256"],
            "artifacts": {
                path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                for path in sorted(project_tmp.iterdir())
                if path.is_file()
            },
        }
        manifest["manifest_content_sha256"] = json_sha256(manifest)
        (project_tmp / MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
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
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
