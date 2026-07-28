"""Complete frozen forward-emulator diagnostics for A0 and Models A--C.

The protocol follows the diagnostic *forms* and qualitative trends in Bire et
al. (2025), while keeping the present 1-degree tutorial data, variables, and
ten-day map explicit.  It never retrains a model and refuses to overwrite an
existing output package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .af_a0 import a0_architecture
from .af_a0_evaluate import _normalizers, _normalise
from .af_data import STATE_CHANNELS
from .af_model_a import build_model_a, model_a_architecture
from .af_model_b import (
    ModelBLossConfig,
    build_model_b,
    loss_contract_sha256,
    model_b_architecture,
    western_boundary_mask,
)
from .af_pressure import PHIHYD_LEVELS, pressure_diagnostics
from .af_tutorial_analysis import DRF_M, EARTH_RADIUS_M
from .fno import build_paper_fno

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


HORIZON_DAYS = 10
ROLLOUT_STEPS = 36
STARTS_PER_REGIME = 15
MAP_LEADS = (10, 60, 180, 360)
SPECTRAL_LEADS = (10, 90, 180, 360)
U = slice(0, 15)
V = slice(15, 30)
THETA = slice(30, 45)
ETA = slice(45, 46)
GROUPS = {"u": U, "v": V, "temperature": THETA, "ssh": ETA}
PRESSURE_FIELDS = tuple(PHIHYD_LEVELS)
BIRE_FIELDS = (
    "surface_speed",
    "sst",
    "ssh",
    *PRESSURE_FIELDS,
    "streamfunction",
)
SCALAR_FIELDS = ("surface_speed", "sst", "ssh", "streamfunction")
# SSH belongs to both the full-state groups and the Bire-facing diagnostics.
# Preserve the requested order while evaluating it only once per lead.
ALL_FIELDS = tuple(dict.fromkeys((*GROUPS, *BIRE_FIELDS)))
FIELD_LABELS = {
    "u": "U (all depths)",
    "v": "V (all depths)",
    "temperature": "Temperature (all depths)",
    "ssh": "SSH",
    "surface_speed": "Surface speed",
    "sst": "SST",
    "phihyd_surface": "PHIHYD surface (k=0)",
    "phihyd_mid": "PHIHYD mid-depth (k=7)",
    "phihyd_bottom": "PHIHYD bottom (k=14)",
    "streamfunction": "Barotropic streamfunction",
}
FIELD_UNITS = {
    "u": "m s$^{-1}$",
    "v": "m s$^{-1}$",
    "temperature": "$^\\circ$C",
    "ssh": "m",
    "surface_speed": "m s$^{-1}$",
    "sst": "$^\\circ$C",
    "phihyd_surface": "m$^2$ s$^{-2}$",
    "phihyd_mid": "m$^2$ s$^{-2}$",
    "phihyd_bottom": "m$^2$ s$^{-2}$",
    "streamfunction": "Sv",
}
METHODS = ("model", "persistence", "climatology")
METHOD_LABELS = {
    "model": "Emulator",
    "persistence": "Persistence",
    "climatology": "Training climatology",
}
METHOD_COLORS = {"model": "#2F75B5", "persistence": "#222222", "climatology": "#A86600"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"mean": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
    }


def _finite_bound(values: Sequence[np.ndarray], *, percentile: float = 100.0) -> float:
    """Return a positive plotting bound even when a rollout has diverged."""

    pieces = [np.abs(np.asarray(value, dtype=float)) for value in values]
    finite = np.concatenate([value[np.isfinite(value)] for value in pieces])
    if not finite.size:
        return 1.0
    return max(float(np.percentile(finite, percentile)), np.finfo(float).eps)


def _barotropic_streamfunction(states: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Return meridionally integrated transport streamfunction in Sv."""

    dy_m = EARTH_RADIUS_M * np.deg2rad(1.0)
    depth_integrated_u = np.sum(states[:, U] * DRF_M[None, :, None, None], axis=1)
    value = np.cumsum(-depth_integrated_u * dy_m, axis=1) / 1.0e6
    value[:, ~wet] = 0.0
    return value.astype(np.float32)


def derived_fields(states: np.ndarray, wet: np.ndarray) -> dict[str, np.ndarray]:
    """Return the seven Bire-facing two-dimensional diagnostics."""

    fields = {
        "surface_speed": np.sqrt(np.square(states[:, 0]) + np.square(states[:, 15])),
        "sst": states[:, 30].copy(),
        "ssh": states[:, 45].copy(),
        **pressure_diagnostics(states, wet),
        "streamfunction": _barotropic_streamfunction(states, wet),
    }
    for value in fields.values():
        value[:, ~wet] = 0.0
    return fields


def radial_spectrum(fields: np.ndarray, wet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed radial power spectrum over the rectangular wet basin."""

    rows, columns = np.where(wet)
    y0, y1, x0, x1 = rows.min(), rows.max() + 1, columns.min(), columns.max() + 1
    cropped = np.asarray(fields[:, y0:y1, x0:x1], dtype=np.float64)
    cropped -= cropped.mean(axis=(1, 2), keepdims=True)
    window = np.hanning(cropped.shape[1])[:, None] * np.hanning(cropped.shape[2])[None, :]
    power = np.square(np.abs(np.fft.fft2(cropped * window[None], axes=(-2, -1))))
    ky = np.fft.fftfreq(cropped.shape[1]) * cropped.shape[1]
    kx = np.fft.fftfreq(cropped.shape[2]) * cropped.shape[2]
    radius = np.sqrt(np.square(ky[:, None]) + np.square(kx[None, :]))
    shell = np.floor(radius).astype(int)
    modes = np.arange(1, min(cropped.shape[1], cropped.shape[2]) // 2 + 1)
    spectra = np.empty((cropped.shape[0], modes.size), dtype=np.float64)
    for index, mode in enumerate(modes):
        selected = shell == mode
        spectra[:, index] = power[:, selected].mean(axis=1)
    return modes.astype(np.float32), spectra.astype(np.float32)


def _member_rmse(prediction: np.ndarray, truth: np.ndarray, wet: np.ndarray) -> np.ndarray:
    error = (prediction - truth)[..., wet].reshape(prediction.shape[0], -1)
    return np.sqrt(np.mean(np.square(error), axis=1))


def _member_acc(
    prediction: np.ndarray, truth: np.ndarray, climatology: np.ndarray, wet: np.ndarray
) -> np.ndarray:
    left = (prediction - climatology)[..., wet].reshape(prediction.shape[0], -1).astype(np.float64)
    right = (truth - climatology)[..., wet].reshape(truth.shape[0], -1).astype(np.float64)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(np.square(left), axis=1) * np.sum(np.square(right), axis=1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def _as_channels(value: np.ndarray) -> np.ndarray:
    return value[:, None] if value.ndim == 3 else value


def _state_fields(states: np.ndarray, wet: np.ndarray) -> dict[str, np.ndarray]:
    result = {name: states[:, fields] for name, fields in GROUPS.items()}
    result.update({name: value[:, None] for name, value in derived_fields(states, wet).items()})
    return result


def _training_climatology(
    state: Any, snapshot_codes: np.ndarray, wet: np.ndarray, *, chunk: int = 60
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Regime-specific pointwise training means, including nonlinear diagnostics."""

    selected = np.flatnonzero(snapshot_codes == 1)
    if not selected.size or chunk <= 0:
        raise ValueError("training climatology needs training snapshots and a positive chunk")
    cuts = np.flatnonzero(np.diff(selected) != 1) + 1
    training_blocks = tuple(np.split(selected, cuts))
    state_sum = np.zeros(
        (state.shape[0], state.shape[2], state.shape[3], state.shape[4]), dtype=np.float64
    )
    derived_sum = {
        name: np.zeros((state.shape[0], state.shape[3], state.shape[4]), dtype=np.float64)
        for name in BIRE_FIELDS
    }
    for experiment in range(state.shape[0]):
        for training_block in training_blocks:
            block_start = int(training_block[0])
            block_stop = int(training_block[-1]) + 1
            for start in range(block_start, block_stop, chunk):
                stop = min(start + chunk, block_stop)
                block = np.asarray(state[experiment, start:stop], dtype=np.float32)
                state_sum[experiment] += block.sum(axis=0, dtype=np.float64)
                for name, value in derived_fields(block, wet).items():
                    derived_sum[name][experiment] += value.sum(axis=0, dtype=np.float64)
    state_mean = (state_sum / selected.size).astype(np.float32)
    state_mean[:, :, ~wet] = 0.0
    derived_mean = {
        name: (value / selected.size).astype(np.float32) for name, value in derived_sum.items()
    }
    for value in derived_mean.values():
        value[:, ~wet] = 0.0
    return state_mean, derived_mean, int(selected.size)


@dataclass
class FrozenStepper:
    kind: str
    model: Any
    device: Any
    wet: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def step(self, current: Any, static: Any) -> Any:
        features = torch.cat((current, static[:, :1] if self.kind == "a0" else static), dim=1)
        predicted = self.model(features)
        if self.kind != "a0":
            predicted = current + predicted
        predicted[:, :, ~torch.from_numpy(self.wet).to(self.device)] = 0.0
        return predicted

    def physical(self, normalized: Any) -> np.ndarray:
        raw = normalized.detach().cpu().numpy() * self.scale[None, :, None, None]
        raw += self.mean[None, :, None, None]
        raw[:, :, ~self.wet] = 0.0
        return raw.astype(np.float32)


def _normalized_static(
    static: Any, experiments: Sequence[int], wet: np.ndarray, wind_mean: float, wind_scale: float
) -> np.ndarray:
    value = np.stack(
        [np.asarray(static[int(experiment)], dtype=np.float32) for experiment in experiments]
    )
    value[:, 0] = (value[:, 0] - wind_mean) / wind_scale
    value[:, 0, ~wet] = 0.0
    return np.ascontiguousarray(value)


def _load_stepper(
    kind: str,
    checkpoint: Path,
    device: Any,
    wet: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[FrozenStepper, Mapping[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if kind == "a0":
        architecture = a0_architecture()
        model = build_paper_fno(architecture)
    elif kind == "a":
        architecture = model_a_architecture()
        model = build_model_a(architecture)
    else:
        architecture = model_b_architecture()
        model = build_model_b(architecture)
        expected = loss_contract_sha256(ModelBLossConfig())
        if payload.get("model_b_loss_contract_sha256") != expected:
            raise ValueError("Model B checkpoint does not match the frozen complete loss contract")
    if payload.get("model_config") != architecture.to_dict():
        raise ValueError(f"checkpoint does not match the frozen {kind.upper()} architecture")
    model.to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return FrozenStepper(kind, model, device, wet, mean, scale), payload


def _choose_starts(
    pair_codes: np.ndarray, record_count: int
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    times = np.arange(pair_codes.size)
    valid = np.flatnonzero(
        (pair_codes == 3) & (times + HORIZON_DAYS * ROLLOUT_STEPS < record_count)
    )
    selected = np.unique(np.linspace(valid[0], valid[-1], STARTS_PER_REGIME, dtype=int))
    if selected.size != STARTS_PER_REGIME:
        raise ValueError("could not select 15 unique one-year inference starts")
    starts = tuple((experiment, int(time)) for experiment in range(3) for time in selected)
    return selected, starts


def _scalar_diagnostics(fields: Mapping[str, np.ndarray], wet: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "surface_speed_mean": fields["surface_speed"][:, wet].mean(axis=1),
        "surface_kinetic_energy": 0.5 * np.mean(np.square(fields["surface_speed"][:, wet]), axis=1),
        "sst_mean": fields["sst"][:, wet].mean(axis=1),
        "ssh_rms": np.sqrt(np.mean(np.square(fields["ssh"][:, wet]), axis=1)),
        "streamfunction_max_abs": np.max(np.abs(fields["streamfunction"][:, wet]), axis=1),
    }


def _append_curve(
    curves: dict[str, Any],
    raw_arrays: dict[str, np.ndarray],
    lead_index: int,
    name: str,
    method: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    climatology: np.ndarray,
    wet: np.ndarray,
) -> None:
    rmse = _member_rmse(prediction, truth, wet)
    acc = _member_acc(prediction, truth, climatology, wet)
    curves[name][method]["rmse"].append(_summary(rmse))
    curves[name][method]["acc"].append(_summary(acc))
    raw_arrays[f"rmse_{name}_{method}_{lead_index:02d}"] = rmse.astype(np.float32)
    raw_arrays[f"acc_{name}_{method}_{lead_index:02d}"] = acc.astype(np.float32)


def _rollout(
    stepper: FrozenStepper,
    state: Any,
    static: Any,
    starts: Sequence[tuple[int, int]],
    climatology_state: np.ndarray,
    climatology_derived: Mapping[str, np.ndarray],
    wind_mean: float,
    wind_scale: float,
    boundary: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    experiments = np.asarray([record[0] for record in starts], dtype=int)
    initial = np.stack([np.asarray(state[e, t], dtype=np.float32) for e, t in starts])
    current = torch.from_numpy(_normalise(initial, stepper.mean, stepper.scale, stepper.wet)).to(
        stepper.device
    )
    static_batch = torch.from_numpy(
        _normalized_static(static, experiments, stepper.wet, wind_mean, wind_scale)
    ).to(stepper.device)
    climate_state = climatology_state[experiments]
    climate_derived = {name: value[experiments] for name, value in climatology_derived.items()}
    initial_fields = _state_fields(initial, stepper.wet)
    climate_fields = {name: climate_state[:, fields] for name, fields in GROUPS.items()}
    climate_fields.update({name: value[:, None] for name, value in climate_derived.items()})
    curves = {name: {method: {"rmse": [], "acc": []} for method in METHODS} for name in ALL_FIELDS}
    boundary_curves = {
        name: {method: [] for method in ("model", "persistence")} for name in ALL_FIELDS
    }
    arrays: dict[str, np.ndarray] = {
        "lead_days": np.arange(1, ROLLOUT_STEPS + 1, dtype=np.int32) * HORIZON_DAYS,
        "ensemble_starts": np.asarray(starts, dtype=np.int32),
        "wind_experiment_index": experiments.astype(np.int16),
        "wet_mask": stepper.wet.astype(np.uint8),
        "western_boundary_mask": boundary.astype(np.uint8),
    }
    scalar_names = (
        "surface_speed_mean",
        "surface_kinetic_energy",
        "sst_mean",
        "ssh_rms",
        "streamfunction_max_abs",
    )
    scalar = {method: {name: [] for name in scalar_names} for method in ("model", "truth")}
    normalized_max: list[float] = []
    finite_state: list[bool] = []
    land_abs_max: list[float] = []
    group_bias_z = {name: [] for name in GROUPS}
    vertical_bias_z = {name: [] for name in ("u", "v", "temperature")}
    vertical_std_ratio = {name: [] for name in ("u", "v", "temperature")}
    fixed_index = STARTS_PER_REGIME // 2  # middle control member; starts are regime-major
    hov_truth, hov_model = [], []
    y_hov = stepper.wet.shape[0] // 2
    x_hov = np.flatnonzero(stepper.wet[y_hov])

    with torch.no_grad():
        for step in range(1, ROLLOUT_STEPS + 1):
            current = stepper.step(current, static_batch)
            prediction = stepper.physical(current)
            truth = np.stack(
                [np.asarray(state[e, t + step * HORIZON_DAYS], dtype=np.float32) for e, t in starts]
            )
            truth[:, :, ~stepper.wet] = 0.0
            predicted_fields = _state_fields(prediction, stepper.wet)
            truth_fields = _state_fields(truth, stepper.wet)
            lead = step * HORIZON_DAYS
            for name in ALL_FIELDS:
                for method, values in (
                    ("model", predicted_fields[name]),
                    ("persistence", initial_fields[name]),
                    ("climatology", climate_fields[name]),
                ):
                    _append_curve(
                        curves,
                        arrays,
                        step - 1,
                        name,
                        method,
                        values,
                        truth_fields[name],
                        climate_fields[name],
                        stepper.wet,
                    )
            for name in ALL_FIELDS:
                boundary_curves[name]["model"].append(
                    _summary(_member_rmse(predicted_fields[name], truth_fields[name], boundary))
                )
                boundary_curves[name]["persistence"].append(
                    _summary(_member_rmse(initial_fields[name], truth_fields[name], boundary))
                )
            for method, fields in (("model", predicted_fields), ("truth", truth_fields)):
                for name, value in _scalar_diagnostics(
                    {key: fields[key][:, 0] for key in SCALAR_FIELDS}, stepper.wet
                ).items():
                    scalar[method][name].append(value.astype(np.float32))
            normalized_max.append(float(torch.max(torch.abs(current)).cpu()))
            finite_state.append(bool(torch.all(torch.isfinite(current)).cpu()))
            land = ~torch.from_numpy(stepper.wet).to(stepper.device)
            land_abs_max.append(float(torch.max(torch.abs(current[:, :, land])).cpu()))
            for name, fields in GROUPS.items():
                error_z = (prediction[:, fields] - truth[:, fields]) / stepper.scale[fields][
                    None, :, None, None
                ]
                group_bias_z[name].append(
                    np.mean(error_z[:, :, stepper.wet], axis=(1, 2)).astype(np.float32)
                )
                if name in vertical_bias_z:
                    vertical_bias_z[name].append(
                        np.mean(error_z[:, :, stepper.wet], axis=2).astype(np.float32)
                    )
                    model_z = (
                        prediction[:, fields] - stepper.mean[fields][None, :, None, None]
                    ) / stepper.scale[fields][None, :, None, None]
                    truth_z = (
                        truth[:, fields] - stepper.mean[fields][None, :, None, None]
                    ) / stepper.scale[fields][None, :, None, None]
                    model_std = np.std(model_z[:, :, stepper.wet], axis=2)
                    truth_std = np.std(truth_z[:, :, stepper.wet], axis=2)
                    vertical_std_ratio[name].append(
                        np.divide(
                            model_std,
                            truth_std,
                            out=np.full_like(model_std, np.nan),
                            where=truth_std > 0,
                        ).astype(np.float32)
                    )
            hov_truth.append(truth[fixed_index, 45, y_hov, x_hov])
            hov_model.append(prediction[fixed_index, 45, y_hov, x_hov])

            if lead in MAP_LEADS:
                for name in BIRE_FIELDS:
                    for method, values in (
                        ("model", predicted_fields[name][:, 0]),
                        ("persistence", initial_fields[name][:, 0]),
                        ("climatology", climate_fields[name][:, 0]),
                    ):
                        arrays[f"spatial_rmse_{name}_{method}_day{lead:03d}"] = np.sqrt(
                            np.mean(np.square(values - truth_fields[name][:, 0]), axis=0)
                        ).astype(np.float32)
                    arrays[f"snapshot_{name}_truth_day{lead:03d}"] = truth_fields[name][
                        fixed_index, 0
                    ]
                    arrays[f"snapshot_{name}_model_day{lead:03d}"] = predicted_fields[name][
                        fixed_index, 0
                    ]
                    arrays[f"snapshot_{name}_persistence_day{lead:03d}"] = initial_fields[name][
                        fixed_index, 0
                    ]
                for name, fields in (("u", U), ("v", V), ("temperature", THETA)):
                    arrays[f"vertical_rmse_{name}_day{lead:03d}"] = np.sqrt(
                        np.mean(
                            np.square(
                                (prediction[:, fields] - truth[:, fields])[:, :, stepper.wet]
                            ),
                            axis=(0, 2),
                        )
                    ).astype(np.float32)
            if lead in SPECTRAL_LEADS:
                for name in BIRE_FIELDS:
                    modes, truth_spectrum = radial_spectrum(truth_fields[name][:, 0], stepper.wet)
                    _, model_spectrum = radial_spectrum(predicted_fields[name][:, 0], stepper.wet)
                    arrays["spectral_modes"] = modes
                    arrays[f"spectrum_{name}_truth_day{lead:03d}"] = truth_spectrum
                    arrays[f"spectrum_{name}_model_day{lead:03d}"] = model_spectrum

    arrays["hovmoller_longitude_index"] = x_hov.astype(np.int16)
    arrays["hovmoller_ssh_truth"] = np.asarray(hov_truth, dtype=np.float32)
    arrays["hovmoller_ssh_model"] = np.asarray(hov_model, dtype=np.float32)
    arrays["normalized_state_max"] = np.asarray(normalized_max, dtype=np.float32)
    arrays["finite_state"] = np.asarray(finite_state, dtype=np.uint8)
    arrays["land_abs_max_normalized"] = np.asarray(land_abs_max, dtype=np.float32)
    for method in ("model", "truth"):
        for name in scalar_names:
            arrays[f"scalar_{name}_{method}"] = np.stack(scalar[method][name])
    for name in GROUPS:
        arrays[f"group_mean_bias_z_{name}"] = np.stack(group_bias_z[name])
    for name in vertical_bias_z:
        arrays[f"vertical_mean_bias_z_{name}"] = np.stack(vertical_bias_z[name])
        arrays[f"vertical_std_ratio_{name}"] = np.stack(vertical_std_ratio[name])
    return {"curves": curves, "western_boundary": boundary_curves}, arrays


def _forcing_switch(
    stepper: FrozenStepper,
    state: Any,
    static: Any,
    selected_times: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Start from identical S0 states and branch only the prescribed static forcing."""

    base_times = selected_times[::3][:5]
    initial = np.stack([np.asarray(state[0, int(time)], dtype=np.float32) for time in base_times])
    branched = np.repeat(initial, 3, axis=0)
    forcing_experiments = np.tile(np.arange(3), base_times.size)
    current = torch.from_numpy(_normalise(branched, stepper.mean, stepper.scale, stepper.wet)).to(
        stepper.device
    )
    forcing = torch.from_numpy(
        _normalized_static(static, forcing_experiments, stepper.wet, wind_mean, wind_scale)
    ).to(stepper.device)
    psi = []
    with torch.no_grad():
        for _ in range(ROLLOUT_STEPS):
            current = stepper.step(current, forcing)
            fields = derived_fields(stepper.physical(current), stepper.wet)
            psi.append(np.max(np.abs(fields["streamfunction"][:, stepper.wet]), axis=1))
    values = np.asarray(psi, dtype=np.float32).reshape(ROLLOUT_STEPS, base_times.size, 3)
    low, control, high = 1, 0, 2
    final = values[-1].mean(axis=0)
    return {
        "definition": "five fixed control-regime starts, branched with only S1/S0/S2 static forcing",
        "mitgcm_truth_available": False,
        "final_mean_streamfunction_max_abs_sv": {
            "low": float(final[low]),
            "control": float(final[control]),
            "high": float(final[high]),
        },
        "expected_sign_pass": bool(final[high] > final[low]),
        "provisional": True,
    }, {
        "forcing_switch_psi": values,
        "forcing_switch_times": base_times.astype(np.int32),
        "forcing_switch_experiment_order": np.asarray((0, 1, 2), dtype=np.int16),
    }


def _spectral_gate(arrays: Mapping[str, np.ndarray]) -> tuple[dict[str, float], bool]:
    ratios: dict[str, float] = {}
    passed = True
    for name in BIRE_FIELDS:
        truth = np.asarray(arrays[f"spectrum_{name}_truth_day360"], dtype=float).mean(axis=0)
        model = np.asarray(arrays[f"spectrum_{name}_model_day360"], dtype=float).mean(axis=0)
        valid = truth > max(float(np.max(truth)) * 1.0e-8, 1.0e-20)
        ratio = float(np.median(model[valid] / truth[valid])) if np.any(valid) else float("nan")
        ratios[name] = ratio
        passed = passed and bool(np.isfinite(ratio) and 0.25 <= ratio <= 4.0)
    return ratios, passed


def _wind_gate(
    arrays: Mapping[str, np.ndarray], wind_stress: np.ndarray
) -> tuple[dict[str, float], bool]:
    experiments = np.asarray(arrays["wind_experiment_index"], dtype=int)
    values = {}
    for method in ("model", "truth"):
        final = np.asarray(arrays[f"scalar_streamfunction_max_abs_{method}"])[-1]
        regime_means = np.asarray([final[experiments == index].mean() for index in range(3)])
        values[method] = float(np.polyfit(wind_stress, regime_means, 1)[0])
    ratio = values["model"] / values["truth"] if values["truth"] else float("nan")
    values["slope_ratio"] = float(ratio)
    passed = bool(np.sign(values["model"]) == np.sign(values["truth"]) and 0.5 <= abs(ratio) <= 2.0)
    return values, passed


def _gate(
    diagnostics: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    wind_stress: np.ndarray,
    switch: Mapping[str, Any],
    model_kind: str,
    a0_reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    surface = {}
    skill_pass = True
    acc_pass = True
    for name in BIRE_FIELDS:
        curve = diagnostics["curves"][name]
        model_rmse = curve["model"]["rmse"][0]["mean"]
        baselines = [curve[method]["rmse"][0]["mean"] for method in ("persistence", "climatology")]
        surface[name] = {
            "model_rmse": model_rmse,
            "baseline_rmse": baselines,
            "pass": model_rmse < min(baselines),
        }
        skill_pass = skill_pass and surface[name]["pass"]
        model_acc20 = curve["model"]["acc"][1]["mean"]
        baseline_acc20 = max(
            curve[method]["acc"][1]["mean"] for method in ("persistence", "climatology")
        )
        acc_pass = acc_pass and bool(model_acc20 > baseline_acc20 and model_acc20 > 0.0)
    normalized_max = float(np.nanmax(arrays["normalized_state_max"]))
    all_finite = bool(np.all(arrays["finite_state"]))
    land_abs_max = float(np.max(arrays["land_abs_max_normalized"]))
    final_bias = {
        name: float(abs(np.mean(arrays[f"group_mean_bias_z_{name}"][-1]))) for name in GROUPS
    }
    stability_pass = bool(
        all_finite
        and np.isfinite(normalized_max)
        and normalized_max < 20.0
        and land_abs_max == 0.0
        and max(final_bias.values()) < 2.0
    )
    spectral_ratio, spectral_pass = _spectral_gate(arrays)
    wind, wind_pass = _wind_gate(arrays, wind_stress)
    criteria: dict[str, bool] = {
        "ten_day_beats_both_baselines_all_bire_diagnostics": skill_pass,
        "twenty_day_acc_beats_both_baselines_all_bire_diagnostics": acc_pass,
        "one_year_bounded_and_low_group_mean_drift": stability_pass,
        "one_year_resolved_spectral_energy_within_factor_four": spectral_pass,
        "one_year_streamfunction_wind_slope_sign_and_factor_two": wind_pass,
    }
    cross_model: dict[str, Any] = {"applicable": model_kind != "a0"}
    if model_kind != "a0" and a0_reference is not None:
        a0_curves = a0_reference["rollout"]["curves"]
        details = {
            name: {
                "model_rmse": diagnostics["curves"][name]["model"]["rmse"][0]["mean"],
                "a0_rmse": a0_curves[name]["model"]["rmse"][0]["mean"],
            }
            for name in BIRE_FIELDS
        }
        cross_pass = all(value["model_rmse"] < value["a0_rmse"] for value in details.values())
        criteria["ten_day_improves_a0_all_bire_diagnostics"] = cross_pass
        cross_model.update({"available": True, "pass": cross_pass, "details": details})
    elif model_kind != "a0":
        criteria["ten_day_improves_a0_all_bire_diagnostics"] = False
        cross_model.update({"available": False, "pass": False})
    else:
        cross_model.update({"available": False, "pass": None})
    within_model_pass = bool(
        skill_pass and acc_pass and stability_pass and spectral_pass and wind_pass
    )
    complete_pass = bool(
        model_kind != "a0"
        and within_model_pass
        and cross_model.get("available")
        and cross_model.get("pass")
    )
    criteria["forcing_switch_expected_sign_provisional"] = bool(switch["expected_sign_pass"])
    return {
        "status": "pass" if complete_pass else "fail",
        "within_model_criteria_pass": within_model_pass,
        "complete_adjoint_campaign_may_open": complete_pass,
        "criteria": criteria,
        "cross_model_a0_comparison": cross_model,
        "ten_day_bire_diagnostic_details": surface,
        "normalized_state_max_over_one_year": normalized_max,
        "all_rollout_states_finite": all_finite,
        "normalized_land_abs_max": land_abs_max,
        "final_group_mean_bias_training_sigma": final_bias,
        "spectral_energy_ratio_model_over_truth": spectral_ratio,
        "streamfunction_wind_slope": wind,
        "note": "forcing-switch criterion remains provisional until matching MITgcm truth exists",
    }


def _plot_curves(output: Path, diagnostics: Mapping[str, Any], metric: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, constrained_layout=True)
    leads = np.arange(1, ROLLOUT_STEPS + 1) * HORIZON_DAYS
    for axis, name in zip(axes.flat, BIRE_FIELDS):
        for method in METHODS:
            entries = diagnostics["curves"][name][method][metric]
            mean = np.asarray([entry["mean"] for entry in entries])
            low = np.asarray([entry["p10"] for entry in entries])
            high = np.asarray([entry["p90"] for entry in entries])
            axis.plot(leads, mean, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            axis.fill_between(leads, low, high, color=METHOD_COLORS[method], alpha=0.12)
        axis.set_title(FIELD_LABELS[name])
        axis.grid(alpha=0.3)
        if metric == "acc":
            axis.axhline(0.0, color="0.6", linewidth=0.7)
            axis.set_ylim(-1.0, 1.02)
        else:
            axis.set_ylabel(f"RMSE ({FIELD_UNITS[name]})")
    for axis in axes.flat[len(BIRE_FIELDS) :]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Lead (model days)")
    if metric == "acc":
        for axis in axes[:, 0]:
            axis.set_ylabel("Anomaly correlation")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Frozen one-year ensemble skill: 15 starts per wind regime")
    figure.savefig(output / f"forward_{metric}_vs_lead.png", dpi=180)
    plt.close(figure)


def _plot_group_ratios(output: Path, diagnostics: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    leads = np.arange(1, ROLLOUT_STEPS + 1) * HORIZON_DAYS
    figure, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, constrained_layout=True)
    for axis, name in zip(axes.flat, BIRE_FIELDS):
        model = np.asarray(
            [entry["mean"] for entry in diagnostics["curves"][name]["model"]["rmse"]]
        )
        for method in ("persistence", "climatology"):
            baseline = np.asarray(
                [entry["mean"] for entry in diagnostics["curves"][name][method]["rmse"]]
            )
            axis.plot(
                leads,
                model / baseline,
                label=f"/ {METHOD_LABELS[method]}",
                color=METHOD_COLORS[method],
            )
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_title(FIELD_LABELS[name])
        axis.set_ylabel("Emulator RMSE / baseline")
        axis.grid(alpha=0.3)
    for axis in axes.flat[len(BIRE_FIELDS) :]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Lead (model days)")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Bire-facing diagnostic error ratios")
    figure.savefig(output / "forward_group_rmse_ratios.png", dpi=180)
    plt.close(figure)


def _plot_spatial(output: Path, arrays: Mapping[str, np.ndarray], wet: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(BIRE_FIELDS), 8, figsize=(18, 2.3 * len(BIRE_FIELDS)), constrained_layout=True
    )
    for row, name in enumerate(BIRE_FIELDS):
        values = [
            arrays[f"spatial_rmse_{name}_{method}_day{lead:03d}"]
            for lead in MAP_LEADS
            for method in ("model", "persistence")
        ]
        limit = _finite_bound([value[wet] for value in values], percentile=99)
        for lead_index, lead in enumerate(MAP_LEADS):
            for offset, method in enumerate(("model", "persistence")):
                axis = axes[row, 2 * lead_index + offset]
                image = axis.imshow(
                    np.ma.masked_where(~wet, arrays[f"spatial_rmse_{name}_{method}_day{lead:03d}"]),
                    origin="lower",
                    cmap="magma",
                    vmin=0,
                    vmax=limit,
                )
                axis.set_title(
                    f"{FIELD_LABELS[name]}\n{METHOD_LABELS[method]}, day {lead}", fontsize=8
                )
                axis.set_xticks([])
                axis.set_yticks([])
                if offset == 1:
                    figure.colorbar(image, ax=axis, shrink=0.72, label=FIELD_UNITS[name])
    figure.suptitle("Spatial RMSE: common scale within each field")
    figure.savefig(output / "forward_spatial_rmse_maps.png", dpi=180)
    plt.close(figure)


def _plot_snapshots(output: Path, arrays: Mapping[str, np.ndarray], wet: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    for lead in MAP_LEADS:
        figure, axes = plt.subplots(
            len(BIRE_FIELDS), 4, figsize=(14, 2.8 * len(BIRE_FIELDS)), constrained_layout=True
        )
        for row, name in enumerate(BIRE_FIELDS):
            truth = arrays[f"snapshot_{name}_truth_day{lead:03d}"]
            model = arrays[f"snapshot_{name}_model_day{lead:03d}"]
            persistence = arrays[f"snapshot_{name}_persistence_day{lead:03d}"]
            fields = [truth, model, model - truth, persistence]
            if name in ("ssh", "streamfunction", *PRESSURE_FIELDS):
                bound = _finite_bound([value[wet] for value in (truth, model, persistence)])
                vmin, vmax, cmap = -bound, bound, "coolwarm"
            else:
                finite = np.concatenate(
                    [value[wet][np.isfinite(value[wet])] for value in (truth, model, persistence)]
                )
                vmin, vmax = (
                    (float(np.min(finite)), float(np.max(finite))) if finite.size else (-1.0, 1.0)
                )
                cmap = "viridis" if name == "surface_speed" else "coolwarm"
            for column, (method, value) in enumerate(
                zip(("truth", "model", "model error", "persistence"), fields)
            ):
                panel_vmin, panel_vmax, panel_cmap = vmin, vmax, cmap
                if method == "model error":
                    error_bound = _finite_bound([value[wet]])
                    panel_vmin, panel_vmax, panel_cmap = -error_bound, error_bound, "coolwarm"
                image = axes[row, column].imshow(
                    np.ma.masked_where(~wet, value),
                    origin="lower",
                    cmap=panel_cmap,
                    vmin=panel_vmin,
                    vmax=panel_vmax,
                )
                axes[row, column].set_title(f"{FIELD_LABELS[name]}: {method}")
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                figure.colorbar(image, ax=axes[row, column], shrink=0.72, label=FIELD_UNITS[name])
        figure.suptitle(f"Fixed middle control member at day {lead}")
        figure.savefig(output / f"forward_snapshots_day{lead:03d}.png", dpi=180)
        plt.close(figure)


def _plot_spectra(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    modes = arrays["spectral_modes"]
    figure, axes = plt.subplots(
        len(BIRE_FIELDS), 4, figsize=(13, 2.7 * len(BIRE_FIELDS)), sharex=True,
        constrained_layout=True
    )
    for row, name in enumerate(BIRE_FIELDS):
        for column, lead in enumerate(SPECTRAL_LEADS):
            axis = axes[row, column]
            for method in ("truth", "model"):
                values = arrays[f"spectrum_{name}_{method}_day{lead:03d}"].mean(axis=0)
                axis.loglog(
                    modes,
                    values,
                    label=method.capitalize(),
                    color="#222222" if method == "truth" else "#2F75B5",
                )
            axis.set_title(f"{FIELD_LABELS[name]}, day {lead}", fontsize=9)
            axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Radial mode")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Resolved anomaly spectra (ensemble mean)")
    figure.savefig(output / "forward_spectra.png", dpi=180)
    plt.close(figure)


def _plot_stability_and_response(
    output: Path, arrays: Mapping[str, np.ndarray], wind_stress: np.ndarray
) -> None:
    import matplotlib.pyplot as plt

    leads = arrays["lead_days"]
    experiments = arrays["wind_experiment_index"].astype(int)
    regime_labels = ("control", "low", "high")
    scalar_names = (
        "surface_kinetic_energy",
        "sst_mean",
        "ssh_rms",
        "streamfunction_max_abs",
    )
    scalar_labels = (
        "Surface kinetic-energy proxy",
        "Mean SST",
        "SSH RMS",
        "Max |streamfunction|",
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, constrained_layout=True)
    for axis, name, label in zip(axes.flat, scalar_names, scalar_labels):
        for experiment, regime in enumerate(regime_labels):
            for method, style in (("truth", "--"), ("model", "-")):
                value = arrays[f"scalar_{name}_{method}"][:, experiments == experiment].mean(axis=1)
                axis.plot(leads, value, linestyle=style, label=f"{regime} {method}")
        axis.set_title(label)
        axis.grid(alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("Lead (model days)")
    axes[0, 0].legend(fontsize=6, ncol=2)
    figure.suptitle("One-year physical stability by forcing regime")
    figure.savefig(output / "forward_stability_by_regime.png", dpi=180)
    plt.close(figure)

    selected_leads = (10, 180, 360)
    order = np.argsort(wind_stress)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    response_names = ("surface_speed_mean", "sst_mean", "ssh_rms", "streamfunction_max_abs")
    response_labels = ("Mean surface speed", "Mean SST", "SSH RMS", "Max |streamfunction|")
    for axis, name, label in zip(axes.flat, response_names, response_labels):
        for lead in selected_leads:
            index = lead // HORIZON_DAYS - 1
            for method, style in (("truth", "--"), ("model", "-")):
                values = arrays[f"scalar_{name}_{method}"][index]
                means = np.asarray(
                    [values[experiments == experiment].mean() for experiment in range(3)]
                )
                axis.plot(
                    wind_stress[order],
                    means[order],
                    marker="o",
                    linestyle=style,
                    label=f"{method}, day {lead}",
                )
        axis.set_title(label)
        axis.set_xlabel("Wind stress (N m$^{-2}$)")
        axis.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6, ncol=2)
    figure.suptitle("Low/control/high physical response")
    figure.savefig(output / "forward_wind_response.png", dpi=180)
    plt.close(figure)


def _plot_hov_boundary_vertical(
    output: Path, diagnostics: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    import matplotlib.pyplot as plt

    truth, model = arrays["hovmoller_ssh_truth"], arrays["hovmoller_ssh_model"]
    bound = _finite_bound((truth, model))
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for axis, title, value in zip(
        axes, ("MITgcm", "Emulator", "Error"), (truth, model, model - truth)
    ):
        image = axis.imshow(
            value, aspect="auto", origin="lower", cmap="coolwarm", vmin=-bound, vmax=bound
        )
        axis.set_title(title)
        axis.set_xlabel("Zonal wet-cell index")
        axis.set_ylabel("10-day step")
        figure.colorbar(image, ax=axis, shrink=0.8, label="SSH (m)")
    figure.suptitle("Fixed control-member longitude--time diagram")
    figure.savefig(output / "forward_ssh_hovmoller.png", dpi=180)
    plt.close(figure)

    leads = arrays["lead_days"]
    figure, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, constrained_layout=True)
    for axis, name in zip(axes.flat, BIRE_FIELDS):
        for method in ("model", "persistence"):
            values = [entry["mean"] for entry in diagnostics["western_boundary"][name][method]]
            axis.plot(leads, values, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        axis.set_title(FIELD_LABELS[name])
        axis.set_ylabel(f"Boundary RMSE ({FIELD_UNITS[name]})")
        axis.grid(alpha=0.3)
    for axis in axes.flat[len(BIRE_FIELDS) :]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Lead (model days)")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Bire-facing diagnostics in the first four wet cells from the western wall")
    figure.savefig(output / "forward_western_boundary_rmse.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), constrained_layout=True)
    for axis, name in zip(axes, ("u", "v", "temperature")):
        for lead in MAP_LEADS:
            axis.plot(
                arrays[f"vertical_rmse_{name}_day{lead:03d}"], np.arange(15), label=f"day {lead}"
            )
        axis.invert_yaxis()
        axis.set_title(FIELD_LABELS[name])
        axis.set_xlabel(f"RMSE ({FIELD_UNITS[name]})")
        axis.set_ylabel("Vertical level")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    figure.suptitle("Full-domain vertical error profiles")
    figure.savefig(output / "forward_vertical_rmse_profiles.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(11.5, 7), sharey=True, constrained_layout=True)
    for column, name in enumerate(("u", "v", "temperature")):
        bias = arrays[f"vertical_mean_bias_z_{name}"][-1].mean(axis=0)
        ratio = arrays[f"vertical_std_ratio_{name}"][-1].mean(axis=0)
        axes[0, column].plot(bias, np.arange(15), color="#B13A3A")
        axes[0, column].axvline(0.0, color="black", linewidth=0.7)
        axes[0, column].set_title(FIELD_LABELS[name])
        axes[0, column].set_xlabel("Mean bias / training sigma")
        axes[1, column].plot(ratio, np.arange(15), color="#2F75B5")
        axes[1, column].axvline(1.0, color="black", linewidth=0.7)
        axes[1, column].set_xlabel("Std(model) / std(MITgcm)")
        for axis in axes[:, column]:
            axis.grid(alpha=0.3)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylabel("Vertical level")
    axes[1, 0].set_ylabel("Vertical level")
    figure.suptitle("Day-360 normalized vertical mean and variance drift")
    figure.savefig(output / "forward_vertical_drift.png", dpi=180)
    plt.close(figure)


def _plot_switch_and_gate(
    output: Path, arrays: Mapping[str, np.ndarray], gate: Mapping[str, Any]
) -> None:
    import matplotlib.pyplot as plt

    values = arrays["forcing_switch_psi"]
    leads = np.arange(1, ROLLOUT_STEPS + 1) * HORIZON_DAYS
    figure, axis = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    for experiment, label, color in (
        (1, "low wind", "#5B8FF9"),
        (0, "control", "#222222"),
        (2, "high wind", "#B13A3A"),
    ):
        mean = values[:, :, experiment].mean(axis=1)
        axis.plot(leads, np.where(np.isfinite(mean), mean, np.nan), label=label, color=color)
    axis.set_xlabel("Lead (model days)")
    axis.set_ylabel("Max |streamfunction| (Sv)")
    axis.set_title("Forcing-switch response from identical control states (provisional)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(output / "forward_forcing_switch_response.png", dpi=180)
    plt.close(figure)

    criteria = gate["criteria"]
    figure, axis = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    names = list(criteria)
    colors = ["#2E7D32" if criteria[name] else "#B13A3A" for name in names]
    axis.barh(np.arange(len(names)), np.ones(len(names)), color=colors)
    axis.set_yticks(np.arange(len(names)), [name.replace("_", " ") for name in names], fontsize=8)
    axis.set_xticks([])
    axis.set_xlim(0, 1)
    axis.invert_yaxis()
    axis.set_title(f"Predeclared forward gate: {gate['status'].upper()} (last item provisional)")
    figure.savefig(output / "forward_gate_scorecard.png", dpi=180)
    plt.close(figure)


def evaluate(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    model_kind: str,
    device_name: str = "auto",
    a0_reference_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate one complete, non-overwriting output package for a frozen model."""

    if torch is None:
        raise RuntimeError("complete forward evaluation requires the project PyTorch environment")
    if model_kind not in {"a0", "a", "b"}:
        raise ValueError("model_kind must be a0, a, or b")
    dataset_path = Path(dataset_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite complete forward outputs: {output}")
    temporary_output = output.with_name(output.name + ".tmp")
    if temporary_output.exists():
        raise FileExistsError(
            f"incomplete temporary forward output already exists: {temporary_output}"
        )
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but no CUDA device is visible")
    device = torch.device(device_name)

    import zarr

    group = zarr.open_consolidated(str(dataset_path), mode="r")
    if tuple(group.attrs["state_channels"]) != STATE_CHANNELS:
        raise ValueError("unexpected 46-channel state contract")
    state, static = group["state"], group["static_features"]
    mean, scale, wet, _, wind_mean, wind_scale = _normalizers(group)
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    selected, starts = _choose_starts(pair_codes, state.shape[1])
    climatology_state, climatology_derived, training_days = _training_climatology(
        state, snapshot_codes, wet
    )
    stepper, _ = _load_stepper(model_kind, checkpoint_path, device, wet, mean, scale)
    boundary = western_boundary_mask(wet, 4)
    diagnostics, arrays = _rollout(
        stepper,
        state,
        static,
        starts,
        climatology_state,
        climatology_derived,
        wind_mean,
        wind_scale,
        boundary,
    )
    switch, switch_arrays = _forcing_switch(stepper, state, static, selected, wind_mean, wind_scale)
    arrays.update(switch_arrays)
    wind_stress = np.asarray(group.attrs["wind_stress_n_m2"], dtype=np.float64)
    if a0_reference_path is not None:
        a0_reference_file = Path(a0_reference_path).resolve()
        a0_reference = json.loads(a0_reference_file.read_text())
        if a0_reference.get("model") != "a0":
            raise ValueError("cross-model reference is not a complete A0 package")
        if a0_reference["protocol"].get("evaluation_contract_version") != 2:
            raise ValueError("A0 reference predates the pressure-complete evaluation contract v2")
        if a0_reference["protocol"]["ensemble_starts"] != [list(record) for record in starts]:
            raise ValueError("A0 reference does not use the identical frozen ensemble starts")
    else:
        a0_reference_file = None
        a0_reference = None
    gate = _gate(diagnostics, arrays, wind_stress, switch, model_kind, a0_reference)
    metadata_path = dataset_path / ".zmetadata"
    metrics = {
        "status": "complete",
        "model": model_kind,
        "purpose": "complete Bire-trend forward diagnostic package on the distinct 1-degree tutorial",
        "paper_reference": {
            "citation": "Bire et al., JAMES 17, e2023MS004137 (2025)",
            "comparison": "qualitative trends and diagnostic forms only; no numerical reproduction claim",
            "figures_discussed": [3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        "dataset": str(dataset_path),
        "dataset_metadata_sha256": _sha256(metadata_path) if metadata_path.is_file() else None,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_load_verified_in_this_evaluation": True,
        "checkpoint_reload_note": "bitwise save/reload was verified when the frozen checkpoint was created",
        "a0_complete_metrics_reference": str(a0_reference_file) if a0_reference_file else None,
        "device": str(device),
        "protocol": {
            "evaluation_contract_version": 2,
            "horizon_days": HORIZON_DAYS,
            "rollout_days": HORIZON_DAYS * ROLLOUT_STEPS,
            "starts_per_regime": STARTS_PER_REGIME,
            "ensemble_starts": [list(record) for record in starts],
            "climatology": "regime-specific pointwise mean over training snapshots; no seasonal cycle is imposed",
            "climatology_training_days_per_regime": training_days,
            "acc": "pointwise training-climatology anomaly correlation over wet cells",
            "western_boundary": "first four wet cells east of each row's western wall",
            "spectrum": "Hann-windowed radial FFT power over the rectangular wet basin",
            "bire_diagnostics": list(BIRE_FIELDS),
            "pressure": {
                "name": "MITgcm PHIHYD",
                "definition": (
                    "hydrostatic pressure anomaly divided by rhoConst; reconstructed from "
                    "Theta and Eta with the configured linear EOS and MITgcm FD integration"
                ),
                "levels_zero_based": PHIHYD_LEVELS,
                "units": "m2 s-2",
                "pressure_pa_conversion": "multiply by rhoConst = 999.8 kg m-3",
                "independent_learned_channels": False,
                "validation_report": (
                    "outputs/af_fno/pressure_validation_v1/pressure_validation.json"
                ),
            },
            "wind_stress_n_m2_in_dataset_order": wind_stress.tolist(),
        },
        "rollout": diagnostics,
        "forcing_switch": switch,
        "forward_gate": gate,
    }
    temporary_output.mkdir(parents=True)
    (temporary_output / "forward_complete_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(temporary_output / "forward_complete_arrays.npz", **arrays)
    _plot_curves(temporary_output, diagnostics, "rmse")
    _plot_curves(temporary_output, diagnostics, "acc")
    _plot_group_ratios(temporary_output, diagnostics)
    _plot_spatial(temporary_output, arrays, wet)
    _plot_snapshots(temporary_output, arrays, wet)
    _plot_spectra(temporary_output, arrays)
    _plot_stability_and_response(temporary_output, arrays, wind_stress)
    _plot_hov_boundary_vertical(temporary_output, diagnostics, arrays)
    _plot_switch_and_gate(temporary_output, arrays, gate)
    temporary_output.rename(output)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate complete frozen forward diagnostics")
    parser.add_argument("--model", choices=("a0", "a", "b"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a0-reference", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    result = evaluate(
        args.dataset,
        args.checkpoint,
        args.output_dir,
        model_kind=args.model,
        device_name=args.device,
        a0_reference_path=args.a0_reference,
    )
    print(json.dumps(result["forward_gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
