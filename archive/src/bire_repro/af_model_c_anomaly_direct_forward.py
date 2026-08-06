"""Frozen held-inference forward and adjoint-readiness test for Model C.

The experiment opens only the prospectively declared fresh inference block in
``trajectories_v2``.  It evaluates the frozen median pointwise-anomaly,
direct-state replicate for 10--360 days, keeps the 10/30-day
adjoint-readiness decision separate, and refuses to overwrite an output
package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from . import af_forward_complete as complete
from .af_a0_evaluate import _normalise, _normalizers
from .af_data import STATE_CHANNELS
from .af_model_b import western_boundary_mask
from .af_model_c_anomaly_direct import PointwiseDirectStepper
from .af_model_c_successor import ModelCSuccessorArchitecture, build_successor

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


VERSION = "model_c_anomaly_direct_forward_v1"
HORIZON_DAYS = 10
ROLLOUT_STEPS = 36
STARTS_PER_REGIME = 15
PRIMARY_FIELDS = ("surface_speed", "sst", "phihyd_surface")
METHODS = ("model", "persistence", "climatology", "damped_persistence")
ADJOINT_LEADS = (10, 30)
AUC_LEADS = tuple(range(10, 91, 10))


class ForwardContractError(RuntimeError):
    """Raised when a frozen forward-evaluation contract is violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_file(specification: Mapping[str, Any], label: str) -> Path:
    path = Path(specification["path"]).resolve()
    if not path.is_file():
        raise ForwardContractError(f"{label} is missing: {path}")
    actual = file_sha256(path)
    if actual != specification["sha256"]:
        raise ForwardContractError(
            f"{label} hash changed: expected {specification['sha256']}, got {actual}"
        )
    return path


def load_contract(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != VERSION:
        raise ForwardContractError("unexpected forward contract version")
    if contract.get("inference_opened_before_freeze") is not False:
        raise ForwardContractError("contract does not preserve the inference seal")
    if (
        contract["protocol"]["horizon_days"] != HORIZON_DAYS
        or contract["protocol"]["rollout_steps"] != ROLLOUT_STEPS
        or contract["protocol"]["starts_per_regime"] != STARTS_PER_REGIME
        or tuple(contract["gates"]["primary_fields"]) != PRIMARY_FIELDS
        or tuple(contract["gates"]["adjoint_readiness_leads_days"])
        != ADJOINT_LEADS
        or tuple(contract["gates"]["primary_auc_leads_days"]) != AUC_LEADS
    ):
        raise ForwardContractError("forward protocol constants changed")
    expected_times = tuple(contract["protocol"]["start_times"])
    starts = tuple(
        (experiment, int(time))
        for experiment in range(3)
        for time in expected_times
    )
    if len(expected_times) != STARTS_PER_REGIME or len(set(expected_times)) != len(
        expected_times
    ):
        raise ForwardContractError("inference start contract is not 15 unique times")
    if json_sha256([list(record) for record in starts]) != contract["protocol"][
        "ensemble_starts_json_sha256"
    ]:
        raise ForwardContractError("ensemble-start hash changed")
    for label, specification in contract["artifacts"].items():
        if "metadata_sha256" in specification:
            metadata = Path(specification["path"]).resolve() / ".zmetadata"
            if file_sha256(metadata) != specification["metadata_sha256"]:
                raise ForwardContractError(f"{label} metadata changed")
        else:
            _verify_file(specification, label)
    for label, specification in contract["sources"].items():
        _verify_file(specification, f"source {label}")
    return contract, resolved, file_sha256(resolved)


def curve_auc(entries: Sequence[Mapping[str, float]], leads: Sequence[int]) -> float:
    x = np.asarray(leads, dtype=np.float64)
    y = np.asarray([entry["mean"] for entry in entries], dtype=np.float64)
    if x.size < 2 or x.size != y.size:
        raise ValueError("AUC inputs changed")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def _load_stepper(
    contract: Mapping[str, Any],
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> tuple[PointwiseDirectStepper, Mapping[str, Any]]:
    checkpoint = Path(contract["artifacts"]["checkpoint"]["path"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture = ModelCSuccessorArchitecture(**contract["architecture"])
    if payload.get("architecture") != architecture.to_dict():
        raise ForwardContractError("checkpoint architecture changed")
    if payload.get("optimizer_step") != contract["selection"]["optimizer_step"]:
        raise ForwardContractError("selected optimizer step changed")
    if (
        payload.get("loss_contract_sha256")
        != contract["selection"]["loss_contract_sha256"]
    ):
        raise ForwardContractError("checkpoint loss contract changed")
    model = build_successor(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    normalization_path = Path(contract["artifacts"]["normalization"]["path"])
    with np.load(normalization_path) as artifact:
        pointwise_mean = np.asarray(artifact["pointwise_mean"], dtype=np.float32)
        pointwise_scale = np.asarray(artifact["pointwise_scale"], dtype=np.float32)
    expected = contract["normalization"]
    if (
        complete._sha256(normalization_path)
        != contract["artifacts"]["normalization"]["sha256"]
        or hashlib.sha256(pointwise_mean.tobytes()).hexdigest()
        != expected["pointwise_mean_sha256"]
        or hashlib.sha256(pointwise_scale.tobytes()).hexdigest()
        != expected["pointwise_scale_sha256"]
    ):
        raise ForwardContractError("pointwise normalization arrays changed")
    return (
        PointwiseDirectStepper(
            model=model,
            device=device,
            wet=wet,
            mean=pointwise_mean,
            scale=pointwise_scale,
            wind_mean=wind_mean,
            wind_scale=wind_scale,
        ),
        payload,
    )


def _declared_starts(
    contract: Mapping[str, Any],
    pair_codes: np.ndarray,
    snapshot_codes: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    times = tuple(int(value) for value in contract["protocol"]["start_times"])
    starts = tuple((experiment, time) for experiment in range(3) for time in times)
    for _, time in starts:
        target = time + HORIZON_DAYS * ROLLOUT_STEPS
        if (
            pair_codes[time] != 3
            or np.any(snapshot_codes[time : target + 1] != 3)
        ):
            raise ForwardContractError(
                "declared trajectory leaves the fresh inference block"
            )
    return starts


def _damped_state(
    initial: np.ndarray,
    climatology: np.ndarray,
    alpha: np.ndarray,
    experiments: np.ndarray,
    step: int,
    wet: np.ndarray,
) -> np.ndarray:
    coefficient = np.power(alpha[experiments], step)[:, :, None, None]
    value = climatology + coefficient * (initial - climatology)
    value[:, :, ~wet] = 0.0
    return value.astype(np.float32)


def _rollout(
    stepper: PointwiseDirectStepper,
    state: Any,
    static: Any,
    starts: Sequence[tuple[int, int]],
    climatology_state: np.ndarray,
    climatology_derived: Mapping[str, np.ndarray],
    damped_alpha: np.ndarray,
    global_mean: np.ndarray,
    global_scale: np.ndarray,
    boundary: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    experiments = np.asarray([record[0] for record in starts], dtype=int)
    initial = np.stack(
        [np.asarray(state[experiment, time], dtype=np.float32) for experiment, time in starts]
    )
    current = stepper.normalized_state(initial)
    static_batch = stepper.normalized_static(static, experiments)
    climate_state = climatology_state[experiments]
    climate_derived = {
        name: value[experiments] for name, value in climatology_derived.items()
    }
    initial_fields = complete._state_fields(initial, stepper.wet)
    climate_fields = {
        name: climate_state[:, fields] for name, fields in complete.GROUPS.items()
    }
    climate_fields.update(
        {name: value[:, None] for name, value in climate_derived.items()}
    )
    curves = {
        name: {
            method: {"rmse": [], "acc": []}
            for method in METHODS
        }
        for name in complete.ALL_FIELDS
    }
    boundary_curves = {
        name: {method: [] for method in ("model", "persistence")}
        for name in complete.ALL_FIELDS
    }
    arrays: dict[str, np.ndarray] = {
        "lead_days": np.arange(1, ROLLOUT_STEPS + 1, dtype=np.int32)
        * HORIZON_DAYS,
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
    scalar = {
        method: {name: [] for name in scalar_names}
        for method in ("model", "truth")
    }
    normalized_max: list[float] = []
    finite_state: list[bool] = []
    land_abs_max: list[float] = []
    group_bias_z = {name: [] for name in complete.GROUPS}
    vertical_bias_z = {name: [] for name in ("u", "v", "temperature")}
    vertical_std_ratio = {
        name: [] for name in ("u", "v", "temperature")
    }
    fixed_index = STARTS_PER_REGIME // 2
    y_hov = stepper.wet.shape[0] // 2
    x_hov = np.flatnonzero(stepper.wet[y_hov])
    hov_truth: list[np.ndarray] = []
    hov_model: list[np.ndarray] = []
    land = ~torch.from_numpy(stepper.wet).to(stepper.device)

    with torch.no_grad():
        for step in range(1, ROLLOUT_STEPS + 1):
            current = stepper.step(current, static_batch)
            prediction = stepper.physical(current)
            truth = np.stack(
                [
                    np.asarray(
                        state[experiment, time + step * HORIZON_DAYS],
                        dtype=np.float32,
                    )
                    for experiment, time in starts
                ]
            )
            truth[:, :, ~stepper.wet] = 0.0
            damped = _damped_state(
                initial,
                climate_state,
                damped_alpha,
                experiments,
                step,
                stepper.wet,
            )
            predicted_fields = complete._state_fields(prediction, stepper.wet)
            truth_fields = complete._state_fields(truth, stepper.wet)
            damped_fields = complete._state_fields(damped, stepper.wet)
            lead = step * HORIZON_DAYS
            for name in complete.ALL_FIELDS:
                for method, values in (
                    ("model", predicted_fields[name]),
                    ("persistence", initial_fields[name]),
                    ("climatology", climate_fields[name]),
                    ("damped_persistence", damped_fields[name]),
                ):
                    complete._append_curve(
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
                for method, values in (
                    ("model", predicted_fields[name]),
                    ("persistence", initial_fields[name]),
                ):
                    boundary_curves[name][method].append(
                        complete._summary(
                            complete._member_rmse(
                                values, truth_fields[name], boundary
                            )
                        )
                    )
            for method, fields in (
                ("model", predicted_fields),
                ("truth", truth_fields),
            ):
                two_dimensional = {
                    key: fields[key][:, 0] for key in complete.SCALAR_FIELDS
                }
                for name, value in complete._scalar_diagnostics(
                    two_dimensional, stepper.wet
                ).items():
                    scalar[method][name].append(value.astype(np.float32))
            normalized_max.append(float(torch.max(torch.abs(current)).cpu()))
            finite_state.append(bool(torch.all(torch.isfinite(current)).cpu()))
            land_abs_max.append(
                float(torch.max(torch.abs(current[:, :, land])).cpu())
            )
            for name, fields in complete.GROUPS.items():
                error_z = (
                    prediction[:, fields] - truth[:, fields]
                ) / global_scale[fields][None, :, None, None]
                group_bias_z[name].append(
                    np.mean(error_z[:, :, stepper.wet], axis=(1, 2)).astype(
                        np.float32
                    )
                )
                if name in vertical_bias_z:
                    vertical_bias_z[name].append(
                        np.mean(error_z[:, :, stepper.wet], axis=2).astype(
                            np.float32
                        )
                    )
                    model_z = (
                        prediction[:, fields]
                        - global_mean[fields][None, :, None, None]
                    ) / global_scale[fields][None, :, None, None]
                    truth_z = (
                        truth[:, fields]
                        - global_mean[fields][None, :, None, None]
                    ) / global_scale[fields][None, :, None, None]
                    model_std = np.std(
                        model_z[:, :, stepper.wet], axis=2
                    )
                    truth_std = np.std(
                        truth_z[:, :, stepper.wet], axis=2
                    )
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

            if lead in complete.MAP_LEADS:
                for name in complete.BIRE_FIELDS:
                    for method, values in (
                        ("model", predicted_fields[name][:, 0]),
                        ("persistence", initial_fields[name][:, 0]),
                        ("climatology", climate_fields[name][:, 0]),
                    ):
                        arrays[
                            f"spatial_rmse_{name}_{method}_day{lead:03d}"
                        ] = np.sqrt(
                            np.mean(
                                np.square(values - truth_fields[name][:, 0]),
                                axis=0,
                            )
                        ).astype(np.float32)
                    for method, values in (
                        ("truth", truth_fields[name]),
                        ("model", predicted_fields[name]),
                        ("persistence", initial_fields[name]),
                    ):
                        arrays[
                            f"snapshot_{name}_{method}_day{lead:03d}"
                        ] = values[fixed_index, 0]
                for name, fields in (
                    ("u", complete.U),
                    ("v", complete.V),
                    ("temperature", complete.THETA),
                ):
                    arrays[f"vertical_rmse_{name}_day{lead:03d}"] = np.sqrt(
                        np.mean(
                            np.square(
                                (
                                    prediction[:, fields]
                                    - truth[:, fields]
                                )[:, :, stepper.wet]
                            ),
                            axis=(0, 2),
                        )
                    ).astype(np.float32)
            if lead in complete.SPECTRAL_LEADS:
                for name in complete.BIRE_FIELDS:
                    modes, truth_spectrum = complete.radial_spectrum(
                        truth_fields[name][:, 0], stepper.wet
                    )
                    _, model_spectrum = complete.radial_spectrum(
                        predicted_fields[name][:, 0], stepper.wet
                    )
                    arrays["spectral_modes"] = modes
                    arrays[
                        f"spectrum_{name}_truth_day{lead:03d}"
                    ] = truth_spectrum
                    arrays[
                        f"spectrum_{name}_model_day{lead:03d}"
                    ] = model_spectrum

    arrays["hovmoller_longitude_index"] = x_hov.astype(np.int16)
    arrays["hovmoller_ssh_truth"] = np.asarray(hov_truth, dtype=np.float32)
    arrays["hovmoller_ssh_model"] = np.asarray(hov_model, dtype=np.float32)
    arrays["normalized_state_max"] = np.asarray(
        normalized_max, dtype=np.float32
    )
    arrays["finite_state"] = np.asarray(finite_state, dtype=np.uint8)
    arrays["land_abs_max_normalized"] = np.asarray(
        land_abs_max, dtype=np.float32
    )
    for method in ("model", "truth"):
        for name in scalar_names:
            arrays[f"scalar_{name}_{method}"] = np.stack(
                scalar[method][name]
            )
    for name in complete.GROUPS:
        arrays[f"group_mean_bias_z_{name}"] = np.stack(group_bias_z[name])
    for name in vertical_bias_z:
        arrays[f"vertical_mean_bias_z_{name}"] = np.stack(
            vertical_bias_z[name]
        )
        arrays[f"vertical_std_ratio_{name}"] = np.stack(
            vertical_std_ratio[name]
        )
    return {
        "curves": curves,
        "western_boundary": boundary_curves,
    }, arrays


def _forcing_switch(
    stepper: PointwiseDirectStepper,
    state: Any,
    static: Any,
    selected_times: Sequence[int],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    base_times = np.asarray(selected_times[::3][:5], dtype=int)
    initial = np.stack(
        [np.asarray(state[0, int(time)], dtype=np.float32) for time in base_times]
    )
    branched = np.repeat(initial, 3, axis=0)
    forcing_experiments = np.tile(np.arange(3), base_times.size)
    current = stepper.normalized_state(branched)
    forcing = stepper.normalized_static(static, forcing_experiments)
    streamfunction = []
    with torch.no_grad():
        for _ in range(ROLLOUT_STEPS):
            current = stepper.step(current, forcing)
            fields = complete.derived_fields(
                stepper.physical(current), stepper.wet
            )
            streamfunction.append(
                np.max(
                    np.abs(fields["streamfunction"][:, stepper.wet]), axis=1
                )
            )
    values = np.asarray(streamfunction, dtype=np.float32).reshape(
        ROLLOUT_STEPS, base_times.size, 3
    )
    final = values[-1].mean(axis=0)
    return {
        "definition": (
            "five fixed fresh-inference control states branched with only "
            "S1/S0/S2 static forcing"
        ),
        "mitgcm_truth_available": False,
        "final_mean_streamfunction_max_abs_sv": {
            "control": float(final[0]),
            "low": float(final[1]),
            "high": float(final[2]),
        },
        "expected_sign_pass": bool(final[2] > final[1]),
        "provisional": True,
    }, {
        "forcing_switch_psi": values,
        "forcing_switch_times": base_times.astype(np.int32),
        "forcing_switch_experiment_order": np.asarray(
            (0, 1, 2), dtype=np.int16
        ),
    }


def _a0_primary_curves(
    contract: Mapping[str, Any],
    state: Any,
    static: Any,
    starts: Sequence[tuple[int, int]],
    wet: np.ndarray,
    device: Any,
) -> dict[str, list[dict[str, float]]]:
    v1_path = Path(contract["artifacts"]["a0_dataset"]["path"])
    v1 = zarr.open_consolidated(str(v1_path), mode="r")
    mean, scale, old_wet, _, wind_mean, wind_scale = _normalizers(v1)
    if not np.array_equal(old_wet, wet):
        raise ForwardContractError("A0 wet mask differs from trajectories_v2")
    stepper, _ = complete._load_stepper(
        "a0",
        Path(contract["artifacts"]["a0_checkpoint"]["path"]),
        device,
        wet,
        mean,
        scale,
    )
    experiments = np.asarray([record[0] for record in starts], dtype=int)
    initial = np.stack(
        [np.asarray(state[experiment, time], dtype=np.float32) for experiment, time in starts]
    )
    current = torch.from_numpy(
        _normalise(initial, mean, scale, wet)
    ).to(device)
    static_batch = torch.from_numpy(
        complete._normalized_static(
            static, experiments, wet, wind_mean, wind_scale
        )
    ).to(device)
    curves = {name: [] for name in PRIMARY_FIELDS}
    with torch.no_grad():
        for step in range(1, len(AUC_LEADS) + 1):
            current = stepper.step(current, static_batch)
            prediction = complete._state_fields(
                stepper.physical(current), wet
            )
            truth = np.stack(
                [
                    np.asarray(
                        state[experiment, time + step * HORIZON_DAYS],
                        dtype=np.float32,
                    )
                    for experiment, time in starts
                ]
            )
            truth_fields = complete._state_fields(truth, wet)
            for name in PRIMARY_FIELDS:
                curves[name].append(
                    complete._summary(
                        complete._member_rmse(
                            prediction[name], truth_fields[name], wet
                        )
                    )
                )
    return curves


def _primary_auc(
    diagnostics: Mapping[str, Any],
    a0_curves: Mapping[str, Sequence[Mapping[str, float]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    count = len(AUC_LEADS)
    for field in PRIMARY_FIELDS:
        model_auc = curve_auc(
            diagnostics["curves"][field]["model"]["rmse"][:count],
            AUC_LEADS,
        )
        baselines = {
            method: curve_auc(
                diagnostics["curves"][field][method]["rmse"][:count],
                AUC_LEADS,
            )
            for method in (
                "persistence",
                "climatology",
                "damped_persistence",
            )
        }
        baselines["a0"] = curve_auc(a0_curves[field], AUC_LEADS)
        ratios = {
            name: model_auc / value for name, value in baselines.items()
        }
        result[field] = {
            "model_auc": model_auc,
            "baseline_auc": baselines,
            "ratio": ratios,
            "beats_every_baseline": all(value < 1.0 for value in ratios.values()),
        }
    return result


def _gate(
    contract: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    wind_stress: np.ndarray,
    switch: Mapping[str, Any],
    a0_curves: Mapping[str, Sequence[Mapping[str, float]]],
) -> dict[str, Any]:
    primary_auc = _primary_auc(diagnostics, a0_curves)
    auc_pass = all(
        value["beats_every_baseline"] for value in primary_auc.values()
    )
    lead_details: dict[str, Any] = {}
    readiness_pass = True
    for lead in ADJOINT_LEADS:
        index = lead // HORIZON_DAYS - 1
        lead_fields = {}
        for field in PRIMARY_FIELDS:
            model_rmse = diagnostics["curves"][field]["model"]["rmse"][index][
                "mean"
            ]
            baselines = {
                method: diagnostics["curves"][field][method]["rmse"][index][
                    "mean"
                ]
                for method in (
                    "persistence",
                    "climatology",
                    "damped_persistence",
                )
            }
            baselines["a0"] = a0_curves[field][index]["mean"]
            model_acc = diagnostics["curves"][field]["model"]["acc"][index][
                "mean"
            ]
            passed = bool(
                all(model_rmse < value for value in baselines.values())
                and model_acc > 0.0
            )
            lead_fields[field] = {
                "model_rmse": model_rmse,
                "baseline_rmse": baselines,
                "model_acc": model_acc,
                "passed": passed,
            }
            readiness_pass = readiness_pass and passed
        lead_details[str(lead)] = lead_fields
    readiness_steps = max(ADJOINT_LEADS) // HORIZON_DAYS
    short_stability = bool(
        np.all(arrays["finite_state"][:readiness_steps])
        and np.max(arrays["normalized_state_max"][:readiness_steps])
        < contract["gates"]["normalized_state_max_abs"]
        and np.max(arrays["land_abs_max_normalized"][:readiness_steps]) == 0.0
    )
    readiness_pass = readiness_pass and short_stability

    normalized_max = float(np.nanmax(arrays["normalized_state_max"]))
    final_bias = {
        name: float(
            abs(np.mean(arrays[f"group_mean_bias_z_{name}"][-1]))
        )
        for name in complete.GROUPS
    }
    stability_pass = bool(
        np.all(arrays["finite_state"])
        and np.isfinite(normalized_max)
        and normalized_max < contract["gates"]["normalized_state_max_abs"]
        and float(np.max(arrays["land_abs_max_normalized"])) == 0.0
        and max(final_bias.values())
        < contract["gates"]["final_group_mean_bias_training_sigma_abs"]
    )
    spectral_ratio, spectral_pass = complete._spectral_gate(arrays)
    wind, wind_pass = complete._wind_gate(arrays, wind_stress)
    long_forward_pass = bool(
        auc_pass and stability_pass and spectral_pass and wind_pass
    )
    complete_pass = bool(readiness_pass and long_forward_pass)
    return {
        "status": "pass" if complete_pass else "fail",
        "complete_adjoint_campaign_may_open": complete_pass,
        "criteria": {
            "ten_and_thirty_day_adjoint_readiness": readiness_pass,
            "primary_10_to_90_day_auc_beats_all_baselines": auc_pass,
            "one_year_bounded_and_low_group_mean_drift": stability_pass,
            "one_year_resolved_spectral_energy_within_factor_four": spectral_pass,
            "one_year_streamfunction_wind_slope_sign_and_factor_two": wind_pass,
            "forcing_switch_expected_sign_provisional": bool(
                switch["expected_sign_pass"]
            ),
        },
        "adjoint_readiness": {
            "status": "pass" if readiness_pass else "fail",
            "horizon_details": lead_details,
            "short_rollout_finite_bounded_zero_land": short_stability,
        },
        "long_forward": {
            "status": "pass" if long_forward_pass else "fail",
            "primary_10_to_90_day_auc": primary_auc,
            "all_primary_auc_beat_every_baseline": auc_pass,
            "one_year_bounded_and_low_group_mean_drift": stability_pass,
            "one_year_resolved_spectral_energy_within_factor_four": spectral_pass,
            "one_year_streamfunction_wind_slope_sign_and_factor_two": wind_pass,
        },
        "normalized_state_max_over_one_year": normalized_max,
        "all_rollout_states_finite": bool(np.all(arrays["finite_state"])),
        "normalized_land_abs_max": float(
            np.max(arrays["land_abs_max_normalized"])
        ),
        "final_group_mean_bias_training_sigma": final_bias,
        "spectral_energy_ratio_model_over_truth": spectral_ratio,
        "streamfunction_wind_slope": wind,
        "forcing_switch_expected_sign_provisional": bool(
            switch["expected_sign_pass"]
        ),
        "note": (
            "forcing-switch response is reported but remains non-vetoing until "
            "matched MITgcm switch truth exists"
        ),
    }


def _plot_adjoint_readiness(
    output: Path,
    gate: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(PRIMARY_FIELDS),
        len(ADJOINT_LEADS),
        figsize=(10, 8),
        constrained_layout=True,
    )
    for column, lead in enumerate(ADJOINT_LEADS):
        for row, field in enumerate(PRIMARY_FIELDS):
            detail = gate["adjoint_readiness"]["horizon_details"][str(lead)][
                field
            ]
            labels = ("Model C", "Persistence", "Climatology", "Damped", "A0")
            values = (
                detail["model_rmse"],
                detail["baseline_rmse"]["persistence"],
                detail["baseline_rmse"]["climatology"],
                detail["baseline_rmse"]["damped_persistence"],
                detail["baseline_rmse"]["a0"],
            )
            axis = axes[row, column]
            axis.bar(
                np.arange(len(values)),
                values,
                color=("#2F75B5", "#222222", "#A86600", "#777777", "#8E4A9B"),
            )
            axis.set_xticks(np.arange(len(values)), labels, rotation=35, ha="right")
            axis.set_title(
                f"{complete.FIELD_LABELS[field]}, day {lead}: "
                f"{'PASS' if detail['passed'] else 'FAIL'}"
            )
            axis.set_ylabel(f"RMSE ({complete.FIELD_UNITS[field]})")
            axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Frozen horizon-matched Model C adjoint-readiness comparison"
    )
    figure.savefig(
        output / "model_c_anomaly_direct_adjoint_readiness.png", dpi=180
    )
    plt.close(figure)


def evaluate(
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("forward evaluation requires PyTorch")
    contract, resolved_contract, contract_sha256 = load_contract(contract_path)
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite forward output or temporary package: {output}"
        )
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible")
    device = torch.device(device_name)

    dataset_path = Path(contract["artifacts"]["dataset"]["path"])
    group = zarr.open_consolidated(str(dataset_path), mode="r")
    if tuple(group.attrs["state_channels"]) != STATE_CHANNELS:
        raise ForwardContractError("dataset state-channel order changed")
    state, static = group["state"], group["static_features"]
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    starts = _declared_starts(contract, pair_codes, snapshot_codes)
    global_mean, global_scale, wet, _, wind_mean, wind_scale = _normalizers(
        group
    )
    climatology_state, climatology_derived, training_days = (
        complete._training_climatology(state, snapshot_codes, wet)
    )
    stepper, _ = _load_stepper(
        contract, device, wet, wind_mean, wind_scale
    )
    predictability_report = json.loads(
        Path(
            contract["artifacts"]["damped_persistence_report"]["path"]
        ).read_text()
    )
    damped_alpha = np.asarray(
        predictability_report["predictability"][
            "ten_day_alpha_by_regime_and_channel"
        ],
        dtype=np.float64,
    )
    if damped_alpha.shape != (3, 46) or np.any(
        (damped_alpha < 0.0) | (damped_alpha > 1.0)
    ):
        raise ForwardContractError("damped-persistence coefficients changed")
    diagnostics, arrays = _rollout(
        stepper,
        state,
        static,
        starts,
        climatology_state,
        climatology_derived,
        damped_alpha,
        global_mean,
        global_scale,
        western_boundary_mask(wet, 4),
    )
    switch, switch_arrays = _forcing_switch(
        stepper, state, static, contract["protocol"]["start_times"]
    )
    arrays.update(switch_arrays)
    a0_curves = _a0_primary_curves(
        contract,
        state,
        static,
        starts,
        wet,
        device,
    )
    wind_stress = np.asarray(
        group.attrs["wind_stress_n_m2"], dtype=np.float64
    )
    gate = _gate(
        contract,
        diagnostics,
        arrays,
        wind_stress,
        switch,
        a0_curves,
    )
    metrics = {
        "status": "complete",
        "version": VERSION,
        "model": "c_pointwise_anomaly_direct_state",
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha256,
        "dataset": str(dataset_path),
        "dataset_metadata_sha256": contract["artifacts"]["dataset"][
            "metadata_sha256"
        ],
        "checkpoint": contract["artifacts"]["checkpoint"]["path"],
        "checkpoint_sha256": contract["artifacts"]["checkpoint"]["sha256"],
        "normalization": contract["artifacts"]["normalization"]["path"],
        "normalization_sha256": contract["artifacts"]["normalization"]["sha256"],
        "device": str(device),
        "inference_state_opened": True,
        "response_or_adjoint_state_opened": False,
        "protocol": {
            "evaluation_contract_version": 1,
            "horizon_days": HORIZON_DAYS,
            "rollout_days": HORIZON_DAYS * ROLLOUT_STEPS,
            "starts_per_regime": STARTS_PER_REGIME,
            "ensemble_starts": [list(record) for record in starts],
            "fresh_inference_block": contract["protocol"][
                "fresh_inference_block"
            ],
            "climatology": (
                "regime-specific pointwise mean over trajectories_v2 split-1 "
                "snapshots"
            ),
            "climatology_training_days_per_regime": training_days,
            "damped_persistence": (
                "training-only regime/channel scalar AR(1) anomaly coefficient "
                "toward regime-specific pointwise climatology"
            ),
            "a0_comparison": (
                "frozen A0 with its original trajectories_v1 normalization, "
                "scored on the identical fresh trajectories_v2 inference starts"
            ),
            "adjoint_readiness_leads_days": list(ADJOINT_LEADS),
            "primary_auc_leads_days": list(AUC_LEADS),
            "wind_stress_n_m2_in_dataset_order": wind_stress.tolist(),
        },
        "rollout": diagnostics,
        "a0_primary_10_to_90_day_curves": a0_curves,
        "forcing_switch": switch,
        "forward_gate": gate,
    }
    temporary.mkdir(parents=True)
    metrics_path = temporary / "model_c_anomaly_direct_forward_metrics.json"
    arrays_path = temporary / "model_c_anomaly_direct_forward_arrays.npz"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(arrays_path, **arrays)
    complete._plot_curves(temporary, diagnostics, "rmse")
    complete._plot_curves(temporary, diagnostics, "acc")
    complete._plot_group_ratios(temporary, diagnostics)
    complete._plot_spatial(temporary, arrays, wet)
    complete._plot_snapshots(temporary, arrays, wet)
    complete._plot_spectra(temporary, arrays)
    complete._plot_stability_and_response(temporary, arrays, wind_stress)
    complete._plot_hov_boundary_vertical(temporary, diagnostics, arrays)
    complete._plot_switch_and_gate(temporary, arrays, gate)
    _plot_adjoint_readiness(temporary, gate)
    (temporary / "README.md").write_text(
        "# Model C anomaly-direct held-inference forward evaluation\n\n"
        "Frozen 10–360-day forward and 10/30-day adjoint-readiness evaluation "
        "of the median three-seed replicate on 45 fresh inference starts. "
        "The package includes exact metrics, compressed arrays, and plots.\n"
    )
    manifest = {
        "version": VERSION,
        "contract_sha256": contract_sha256,
        "metrics_sha256": file_sha256(metrics_path),
        "arrays_sha256": file_sha256(arrays_path),
        "files": {
            path.name: file_sha256(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        },
    }
    manifest["manifest_content_sha256"] = json_sha256(manifest)
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    temporary.rename(output)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen Model C held-inference forward evaluation"
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    args = parser.parse_args(argv)
    result = evaluate(
        args.contract, args.output_dir, device_name=args.device
    )
    print(json.dumps(result["forward_gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
