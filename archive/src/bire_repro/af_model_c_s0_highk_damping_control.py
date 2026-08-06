"""Zero-retraining S0 high-wavenumber damping control for selected Model C.

A weak reflected-boundary binomial smoother is applied to the predicted
pointwise-normalized anomaly after each ten-day call.  Its strength is chosen
only on fixed training-only S0 trajectories.  The fixed S0 day-2000 inference
test is opened only when a nonzero strength passes the prospective
training-only short-skill and long-rollout improvement gates.
"""

from __future__ import annotations

import argparse
import csv
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
from .af_model_c_bire_s0_figures import (
    ContinuousS0Truth,
    _s0_training_climatology,
)
from .af_model_c_bire_s0_long_truth import EXPECTED_STARTS
from .af_model_c_s0_stability_tangent_comparison import (
    FIELDS,
    FIELD_LABELS,
    STAT_FIELDS,
    _pointwise_stepper,
    file_sha256,
    json_sha256,
)
from .af_model_c_s0_stability_tangent_recovery import (
    _physical64,
    derived_fields64,
    member_rmse64,
    radial_spectrum64,
    safe_log_gain,
)

try:
    import torch
    import torch.nn.functional as torch_functional
except (ImportError, OSError):  # pragma: no cover
    torch = None  # type: ignore[assignment]
    torch_functional = None  # type: ignore[assignment]


VERSION = "model_c_s0_highk_damping_control_v1"
CONTRACT_STATUS = "frozen_zero_retraining_after_job304751"
ALPHAS = (0.0, 0.02, 0.05, 0.1, 0.2)
TRAINING_TIMES = (0, 379, 759, 1139, 1519, 3690, 4069, 4449, 4829, 5209)
TRAINING_LEADS = tuple(range(0, 1001, 10))
TRAINING_SPECTRUM_LEADS = (500, 1000)
SHORT_LEADS = (10, 30, 90)
INFERENCE_LEADS = tuple(range(0, 2001, 10))
INFERENCE_SPECTRUM_LEADS = (200, 500, 1000, 1500, 2000)
HIGH_BAND = (10, 30)
TRAINING_GATE = {
    "maximum_short_candidate_to_source_rmse": 1.05,
    "maximum_day1000_worst_rmse_climate_relative_to_source": 0.8,
    "maximum_day1000_normalized_amplitude_relative_to_source": 0.8,
    "maximum_day1000_worst_highk_power_relative_to_source": 0.5,
}
INFERENCE_CHARACTERIZATION = {
    "maximum_short_candidate_to_source_rmse": 1.05,
    "maximum_day2000_rmse_to_climatology": 2.0,
    "maximum_day2000_normalized_amplitude_to_truth": 2.0,
    "maximum_day2000_highk_power_to_truth": 4.0,
    "maximum_late_gain_per_call": 1.005,
}
FIGURES = (
    "model_c_s0_highk_training_selection.png",
    "model_c_s0_highk_inference_log_rmse.png",
    "model_c_s0_highk_inference_normalized_envelope.png",
    "model_c_s0_highk_inference_day2000_spectra.png",
)
ARRAYS = "model_c_s0_highk_damping_arrays.npz"
REPORT = "model_c_s0_highk_damping_report.json"
SUMMARY = "model_c_s0_highk_damping_summary.json"
CSV = "model_c_s0_highk_damping_curves.csv"
README = "README.md"
MANIFEST = "manifest.json"


class HighKDampingError(RuntimeError):
    """Raised when the frozen high-k control changes."""


def _verify_file(record: Mapping[str, Any], label: str) -> Path:
    path = Path(record["path"]).resolve()
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise HighKDampingError(f"immutable artifact changed: {label}")
    return path


def load_contract(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    protocol = contract.get("protocol", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or tuple(protocol.get("alphas", ())) != ALPHAS
        or tuple(protocol.get("training_times", ())) != TRAINING_TIMES
        or tuple(protocol.get("training_leads", ())) != (0, 1000)
        or tuple(protocol.get("training_spectrum_leads", ()))
        != TRAINING_SPECTRUM_LEADS
        or tuple(protocol.get("short_leads", ())) != SHORT_LEADS
        or tuple(protocol.get("inference_leads", ())) != (0, 2000)
        or tuple(protocol.get("inference_spectrum_leads", ()))
        != INFERENCE_SPECTRUM_LEADS
        or tuple(protocol.get("fields", ())) != FIELDS
        or tuple(protocol.get("statistical_fields", ())) != STAT_FIELDS
        or protocol.get("filter")
        != "reflected_3x3_binomial_channelwise_on_normalized_anomaly"
        or protocol.get("selection") != "smallest_nonzero_alpha_passing_all_gates"
        or protocol.get("training_gate") != TRAINING_GATE
        or protocol.get("inference_characterization") != INFERENCE_CHARACTERIZATION
        or bool(protocol.get("retraining", True))
    ):
        raise HighKDampingError("high-k damping contract changed")
    for label, record in contract["artifacts"].items():
        _verify_file(record, label)
    root = resolved.parents[1]
    for relative, expected in contract["source_hashes"].items():
        source = root / relative
        if not source.is_file() or file_sha256(source) != expected:
            raise HighKDampingError(f"source changed: {relative}")
    for key in ("scratch", "project"):
        output = Path(contract["output"][key]).resolve()
        if output.exists() or output.with_name(output.name + ".tmp").exists():
            raise FileExistsError(f"refusing to overwrite output: {output}")
    return contract, resolved, file_sha256(resolved)


def reflected_binomial_damping(
    value: Any,
    wet: np.ndarray,
    alpha_by_member: Any,
) -> Any:
    """Apply a channelwise reflected 3x3 binomial smoother."""

    rows, columns = np.where(wet)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    cropped = value[..., y0:y1, x0:x1]
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=value.device,
        dtype=value.dtype,
    )
    kernel = (kernel / 16.0)[None, None].repeat(value.shape[1], 1, 1, 1)
    smoothed = torch_functional.conv2d(
        torch_functional.pad(cropped, (1, 1, 1, 1), mode="reflect"),
        kernel,
        groups=value.shape[1],
    )
    alpha = alpha_by_member.to(device=value.device, dtype=value.dtype)
    filtered = cropped + alpha[:, None, None, None] * (smoothed - cropped)
    result = value.clone()
    result[..., y0:y1, x0:x1] = filtered
    wet_tensor = torch.from_numpy(wet.astype(np.float32))[None, None].to(
        device=value.device,
        dtype=value.dtype,
    )
    return result * wet_tensor


def training_records(split: np.ndarray) -> np.ndarray:
    records = np.asarray([(0, time) for time in TRAINING_TIMES], dtype=np.int32)
    for _, time_value in records:
        window = split[time_value : time_value + 1001]
        if window.size != 1001 or np.any(window != 1):
            raise HighKDampingError("training-only 1000-day window changed")
    return records


def _gather_state(state: Any, times: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.asarray(state[0, int(time_value)], dtype=np.float32) for time_value in times]
    )


def _rollout(
    stepper: Any,
    initial: np.ndarray,
    normalized_static: Any,
    alphas: tuple[float, ...],
    truth_loader: Callable[[int], np.ndarray],
    climatology: Mapping[str, np.ndarray],
    leads: tuple[int, ...],
    spectrum_leads: tuple[int, ...],
    wet: np.ndarray,
) -> dict[str, np.ndarray]:
    arm_count = len(alphas)
    member_count = initial.shape[0]
    alpha_tensor = torch.repeat_interleave(
        torch.tensor(alphas, device=stepper.device, dtype=torch.float32),
        member_count,
    )
    current = stepper.normalized_state(
        np.repeat(initial[None], arm_count, axis=0).reshape(
            arm_count * member_count,
            *initial.shape[1:],
        )
    )
    fixed_static = normalized_static.repeat(arm_count, 1, 1, 1)
    wet_tensor = torch.from_numpy(wet).to(device=stepper.device)
    spectrum_lookup = {lead: index for index, lead in enumerate(spectrum_leads)}
    modes, _ = radial_spectrum64(
        derived_fields64(initial[:1], wet)["sst"],
        wet,
    )
    arrays: dict[str, np.ndarray] = {
        "lead_days": np.asarray(leads, dtype=np.int16),
        "spectrum_leads": np.asarray(spectrum_leads, dtype=np.int16),
        "spectrum_modes": modes,
        "finite": np.zeros(
            (arm_count, member_count, len(leads)),
            dtype=np.uint8,
        ),
        "normalized_max_abs": np.empty(
            (arm_count, member_count, len(leads)),
            dtype=np.float64,
        ),
        "truth_normalized_max_abs": np.empty(
            (member_count, len(leads)),
            dtype=np.float64,
        ),
    }
    for field in FIELDS:
        arrays[f"rmse__{field}"] = np.empty(
            (arm_count, member_count, len(leads)),
            dtype=np.float64,
        )
        arrays[f"rmse__climatology__{field}"] = np.empty(
            (member_count, len(leads)),
            dtype=np.float64,
        )
    for field in STAT_FIELDS:
        arrays[f"spectrum__{field}"] = np.empty(
            (arm_count, member_count, len(spectrum_leads), modes.size),
            dtype=np.float64,
        )
        arrays[f"spectrum__truth__{field}"] = np.empty(
            (member_count, len(spectrum_leads), modes.size),
            dtype=np.float64,
        )

    with torch.no_grad():
        for lead_index, lead in enumerate(leads):
            if lead:
                current = reflected_binomial_damping(
                    stepper.step(current, fixed_static),
                    wet,
                    alpha_tensor,
                )
            physical = _physical64(stepper, current)
            truth = np.asarray(truth_loader(lead), dtype=np.float64)
            truth_fields = derived_fields64(truth, wet)
            prediction_fields = derived_fields64(physical, wet)
            tiled_truth = {
                field: np.repeat(truth_fields[field][None], arm_count, axis=0).reshape(
                    arm_count * member_count,
                    *truth_fields[field].shape[1:],
                )
                for field in STAT_FIELDS
            }
            for field in FIELDS:
                arrays[f"rmse__{field}"][:, :, lead_index] = member_rmse64(
                    prediction_fields[field],
                    tiled_truth[field],
                    wet,
                ).reshape(arm_count, member_count)
                climate = np.repeat(
                    np.asarray(climatology[field], dtype=np.float64)[None],
                    member_count,
                    axis=0,
                )
                arrays[f"rmse__climatology__{field}"][:, lead_index] = (
                    member_rmse64(climate, truth_fields[field], wet)
                )
            finite = (
                torch.isfinite(current)
                .all(dim=(1, 2, 3))
                .detach()
                .cpu()
                .numpy()
                .reshape(arm_count, member_count)
            )
            arrays["finite"][:, :, lead_index] = finite
            arrays["normalized_max_abs"][:, :, lead_index] = (
                torch.amax(torch.abs(current[:, :, wet_tensor]), dim=(1, 2))
                .detach()
                .cpu()
                .numpy()
                .reshape(arm_count, member_count)
            )
            normalized_truth = stepper.normalized_state(truth.astype(np.float32))
            arrays["truth_normalized_max_abs"][:, lead_index] = (
                torch.amax(
                    torch.abs(normalized_truth[:, :, wet_tensor]),
                    dim=(1, 2),
                )
                .detach()
                .cpu()
                .numpy()
            )
            if lead in spectrum_lookup:
                spectrum_index = spectrum_lookup[lead]
                for field in STAT_FIELDS:
                    _, truth_spectrum = radial_spectrum64(truth_fields[field], wet)
                    arrays[f"spectrum__truth__{field}"][
                        :, spectrum_index
                    ] = truth_spectrum
                    _, prediction_spectrum = radial_spectrum64(
                        prediction_fields[field],
                        wet,
                    )
                    arrays[f"spectrum__{field}"][
                        :, :, spectrum_index
                    ] = prediction_spectrum.reshape(
                        arm_count,
                        member_count,
                        modes.size,
                    )
    return arrays


def _lead_index(arrays: Mapping[str, np.ndarray], lead: int) -> int:
    selected = np.flatnonzero(np.asarray(arrays["lead_days"]) == lead)
    if selected.size != 1:
        raise HighKDampingError(f"missing lead {lead}")
    return int(selected[0])


def _spectrum_index(arrays: Mapping[str, np.ndarray], lead: int) -> int:
    selected = np.flatnonzero(np.asarray(arrays["spectrum_leads"]) == lead)
    if selected.size != 1:
        raise HighKDampingError(f"missing spectrum lead {lead}")
    return int(selected[0])


def _highk_ratio(
    arrays: Mapping[str, np.ndarray],
    arm_index: int,
    field: str,
    lead: int,
) -> float:
    spectrum_index = _spectrum_index(arrays, lead)
    modes = np.asarray(arrays["spectrum_modes"])
    selected = (modes >= HIGH_BAND[0]) & (modes <= HIGH_BAND[1])
    prediction = np.asarray(arrays[f"spectrum__{field}"])[
        arm_index, :, spectrum_index
    ][:, selected].sum(axis=1)
    truth = np.asarray(arrays[f"spectrum__truth__{field}"])[
        :, spectrum_index
    ][:, selected].sum(axis=1)
    return float(np.mean(prediction / np.maximum(truth, 1.0e-300)))


def select_alpha(arrays: Mapping[str, np.ndarray]) -> tuple[int | None, dict[str, Any]]:
    source_index = 0
    day1000 = _lead_index(arrays, 1000)
    truth_amplitude = np.asarray(arrays["truth_normalized_max_abs"])[:, day1000].mean()
    source_amplitude = (
        np.asarray(arrays["normalized_max_abs"])[source_index, :, day1000].mean()
        / truth_amplitude
    )

    def worst_rmse_climate(arm_index: int) -> float:
        return max(
            float(
                np.asarray(arrays[f"rmse__{field}"])[arm_index, :, day1000].mean()
                / np.asarray(arrays[f"rmse__climatology__{field}"])[:, day1000].mean()
            )
            for field in FIELDS
        )

    def worst_highk(arm_index: int) -> float:
        return max(
            _highk_ratio(arrays, arm_index, field, 1000)
            for field in STAT_FIELDS
        )

    source_rmse = worst_rmse_climate(source_index)
    source_highk = worst_highk(source_index)
    records: dict[str, Any] = {}
    selected_index: int | None = None
    for arm_index, alpha in enumerate(ALPHAS):
        short_ratios = []
        for lead in SHORT_LEADS:
            lead_index = _lead_index(arrays, lead)
            for field in FIELDS:
                candidate = np.asarray(arrays[f"rmse__{field}"])[
                    arm_index, :, lead_index
                ].mean()
                source = np.asarray(arrays[f"rmse__{field}"])[
                    source_index, :, lead_index
                ].mean()
                short_ratios.append(float(candidate / source))
        amplitude = (
            np.asarray(arrays["normalized_max_abs"])[arm_index, :, day1000].mean()
            / truth_amplitude
        )
        rmse = worst_rmse_climate(arm_index)
        highk = worst_highk(arm_index)
        passes = bool(
            alpha > 0.0
            and np.all(np.asarray(arrays["finite"])[arm_index])
            and max(short_ratios)
            <= TRAINING_GATE["maximum_short_candidate_to_source_rmse"]
            and rmse / source_rmse
            <= TRAINING_GATE[
                "maximum_day1000_worst_rmse_climate_relative_to_source"
            ]
            and amplitude / source_amplitude
            <= TRAINING_GATE[
                "maximum_day1000_normalized_amplitude_relative_to_source"
            ]
            and highk / source_highk
            <= TRAINING_GATE[
                "maximum_day1000_worst_highk_power_relative_to_source"
            ]
        )
        records[str(alpha)] = {
            "maximum_short_candidate_to_source_rmse": max(short_ratios),
            "day1000_worst_rmse_to_climatology": rmse,
            "day1000_worst_rmse_climate_relative_to_source": rmse / source_rmse,
            "day1000_normalized_amplitude_to_truth": amplitude,
            "day1000_normalized_amplitude_relative_to_source": (
                amplitude / source_amplitude
            ),
            "day1000_worst_highk_power_to_truth": highk,
            "day1000_worst_highk_power_relative_to_source": highk / source_highk,
            "all_states_finite": bool(
                np.all(np.asarray(arrays["finite"])[arm_index])
            ),
            "passes": passes,
        }
        if selected_index is None and passes:
            selected_index = arm_index
    return selected_index, records


def _inference_summary(
    arrays: Mapping[str, np.ndarray],
    selected_alpha: float,
) -> dict[str, Any]:
    leads = np.asarray(arrays["lead_days"], dtype=np.int64)
    day2000 = _lead_index(arrays, 2000)
    source_index, filtered_index = 0, 1
    short = []
    for lead in SHORT_LEADS:
        lead_index = _lead_index(arrays, lead)
        for field in FIELDS:
            short.append(
                float(
                    np.asarray(arrays[f"rmse__{field}"])[
                        filtered_index, :, lead_index
                    ].mean()
                    / np.asarray(arrays[f"rmse__{field}"])[
                        source_index, :, lead_index
                    ].mean()
                )
            )
    rmse_to_climate = {
        field: float(
            np.asarray(arrays[f"rmse__{field}"])[
                filtered_index, :, day2000
            ].mean()
            / np.asarray(arrays[f"rmse__climatology__{field}"])[:, day2000].mean()
        )
        for field in FIELDS
    }
    truth_amplitude = np.asarray(arrays["truth_normalized_max_abs"])[:, day2000].mean()
    normalized_amplitude = float(
        np.asarray(arrays["normalized_max_abs"])[
            filtered_index, :, day2000
        ].mean()
        / truth_amplitude
    )
    highk = {
        field: _highk_ratio(arrays, filtered_index, field, 2000)
        for field in STAT_FIELDS
    }
    late_gain = {
        field: safe_log_gain(
            leads,
            np.asarray(arrays[f"rmse__{field}"])[filtered_index],
            (1700, 2000),
        )
        for field in FIELDS
    }
    passes = bool(
        max(short)
        <= INFERENCE_CHARACTERIZATION["maximum_short_candidate_to_source_rmse"]
        and max(rmse_to_climate.values())
        <= INFERENCE_CHARACTERIZATION["maximum_day2000_rmse_to_climatology"]
        and normalized_amplitude
        <= INFERENCE_CHARACTERIZATION[
            "maximum_day2000_normalized_amplitude_to_truth"
        ]
        and max(highk.values())
        <= INFERENCE_CHARACTERIZATION["maximum_day2000_highk_power_to_truth"]
        and max(
            record["gain"]
            for record in late_gain.values()
            if record["gain"] is not None
        )
        <= INFERENCE_CHARACTERIZATION["maximum_late_gain_per_call"]
    )
    return {
        "selected_alpha": selected_alpha,
        "maximum_short_filtered_to_source_rmse": max(short),
        "day2000_rmse_to_climatology": rmse_to_climate,
        "day2000_normalized_amplitude_to_truth": normalized_amplitude,
        "day2000_highk_power_to_truth": highk,
        "late_gain": late_gain,
        "passes_statistical_characterization": passes,
        "interpretation": (
            "causal_highk_control_passes_characterization"
            if passes
            else "highk_filter_alone_is_insufficient"
        ),
        "checkpoint_promotion": False,
    }


def _plot_training(
    output: Path,
    records: Mapping[str, Any],
    selected_index: int | None,
) -> None:
    alphas = np.asarray(ALPHAS)
    metrics = (
        (
            "maximum_short_candidate_to_source_rmse",
            "Worst short RMSE / source",
            1.0,
        ),
        (
            "day1000_worst_rmse_climate_relative_to_source",
            "Day-1000 worst RMSE/climate / source",
            1.0,
        ),
        (
            "day1000_normalized_amplitude_relative_to_source",
            "Day-1000 amplitude / source",
            1.0,
        ),
        (
            "day1000_worst_highk_power_relative_to_source",
            "Day-1000 worst high-k power / source",
            1.0,
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), constrained_layout=True)
    for axis, (key, title, reference) in zip(axes.flat, metrics):
        values = [records[str(alpha)][key] for alpha in ALPHAS]
        axis.plot(alphas, values, marker="o", color="#D62728")
        axis.axhline(reference, color="black", linestyle="--", linewidth=0.8)
        if selected_index is not None:
            axis.axvline(
                ALPHAS[selected_index],
                color="#2CA02C",
                linestyle=":",
                linewidth=1.2,
            )
        axis.set_title(title)
        axis.set_xlabel(r"Smoothing strength $\alpha$")
        axis.grid(color="0.85", linewidth=0.5)
    figure.suptitle("Training-only S0 selection; green line is prospective choice")
    figure.savefig(output / FIGURES[0], bbox_inches="tight")
    plt.close(figure)


def _plot_inference_rmse(
    output: Path,
    arrays: Mapping[str, np.ndarray] | None,
    selected_alpha: float | None,
) -> None:
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 8.5),
        sharex=True,
        constrained_layout=True,
    )
    if arrays is None:
        for axis in axes:
            axis.text(0.5, 0.5, "Training gate did not open inference", ha="center")
            axis.set_axis_off()
    else:
        leads = np.asarray(arrays["lead_days"])
        for axis, field in zip(axes, FIELDS):
            for arm_index, (label, color) in enumerate(
                (("Source selected", "#D62728"), (f"Filtered α={selected_alpha}", "#1F77B4"))
            ):
                axis.plot(
                    leads[1:],
                    np.asarray(arrays[f"rmse__{field}"])[arm_index, :, 1:].mean(axis=0),
                    color=color,
                    label=label,
                )
            axis.plot(
                leads[1:],
                np.asarray(arrays[f"rmse__climatology__{field}"])[:, 1:].mean(axis=0),
                color="black",
                label="Climatology",
            )
            axis.set_yscale("log")
            axis.set_ylabel(FIELD_LABELS[field])
            axis.grid(which="both", color="0.85", linewidth=0.5)
        axes[-1].set_xlabel("Lead (days)")
        axes[-1].legend(loc="best")
    figure.savefig(output / FIGURES[1], bbox_inches="tight")
    plt.close(figure)


def _plot_inference_envelope(
    output: Path,
    arrays: Mapping[str, np.ndarray] | None,
    selected_alpha: float | None,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    if arrays is None:
        axis.text(0.5, 0.5, "Training gate did not open inference", ha="center")
        axis.set_axis_off()
    else:
        leads = np.asarray(arrays["lead_days"])
        truth = np.asarray(arrays["truth_normalized_max_abs"]).mean(axis=0)
        for arm_index, (label, color) in enumerate(
            (("Source selected", "#D62728"), (f"Filtered α={selected_alpha}", "#1F77B4"))
        ):
            ratio = (
                np.asarray(arrays["normalized_max_abs"])[arm_index].mean(axis=0)
                / truth
            )
            axis.plot(leads, ratio, color=color, label=label)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_yscale("log")
        axis.set_xlabel("Lead (days)")
        axis.set_ylabel("Normalized maximum amplitude / truth")
        axis.grid(which="both", color="0.85", linewidth=0.5)
        axis.legend(loc="best")
    figure.savefig(output / FIGURES[2], bbox_inches="tight")
    plt.close(figure)


def _plot_inference_spectra(
    output: Path,
    arrays: Mapping[str, np.ndarray] | None,
    selected_alpha: float | None,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), constrained_layout=True)
    if arrays is None:
        for axis in axes.flat:
            axis.text(0.5, 0.5, "Training gate did not open inference", ha="center")
            axis.set_axis_off()
    else:
        modes = np.asarray(arrays["spectrum_modes"])
        spectrum_index = _spectrum_index(arrays, 2000)
        for axis, field in zip(axes.flat, STAT_FIELDS):
            truth = np.asarray(arrays[f"spectrum__truth__{field}"])[
                :, spectrum_index
            ].mean(axis=0)
            for arm_index, (label, color) in enumerate(
                (("Source selected", "#D62728"), (f"Filtered α={selected_alpha}", "#1F77B4"))
            ):
                prediction = np.asarray(arrays[f"spectrum__{field}"])[
                    arm_index, :, spectrum_index
                ].mean(axis=0)
                axis.plot(
                    modes,
                    prediction / np.maximum(truth, 1.0e-300),
                    color=color,
                    label=label,
                )
            axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
            axis.set_yscale("log")
            axis.set_title(field.replace("_", " "))
            axis.set_xlabel("Radial wavenumber")
            axis.set_ylabel("Day-2000 power / truth")
            axis.grid(which="both", color="0.85", linewidth=0.5)
        axes[-1, -1].legend(loc="best")
    figure.savefig(output / FIGURES[3], bbox_inches="tight")
    plt.close(figure)


def _write_csv(
    path: Path,
    training: Mapping[str, np.ndarray],
    inference: Mapping[str, np.ndarray] | None,
) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("phase", "arm", "field", "lead_days", "mean_rmse", "p10", "p90")
        )
        for phase, arrays, arms in (
            ("training", training, [str(alpha) for alpha in ALPHAS]),
            (
                "inference",
                inference,
                ["source", "filtered"] if inference is not None else [],
            ),
        ):
            if arrays is None:
                continue
            leads = np.asarray(arrays["lead_days"])
            for arm_index, arm in enumerate(arms):
                for field in FIELDS:
                    values = np.asarray(arrays[f"rmse__{field}"])[arm_index]
                    for lead_index, lead in enumerate(leads):
                        selected = values[:, lead_index]
                        writer.writerow(
                            (
                                phase,
                                arm,
                                field,
                                int(lead),
                                float(selected.mean()),
                                float(np.percentile(selected, 10)),
                                float(np.percentile(selected, 90)),
                            )
                        )


def preflight(contract_path: str | Path) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("high-k damping control requires PyTorch")
    contract, resolved, digest = load_contract(contract_path)
    dataset = Path(contract["artifacts"]["dataset_metadata"]["path"]).parent
    group = zarr.open_consolidated(str(dataset), mode="r")
    if tuple(group.attrs["state_channels"]) != STATE_CHANNELS:
        raise HighKDampingError("trajectory channels changed")
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    records = training_records(split)
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    sample = torch.randn(3, 4, *wet.shape)
    alpha = torch.zeros(3)
    if not torch.equal(sample * torch.from_numpy(wet)[None, None], reflected_binomial_damping(sample, wet, alpha)):
        raise HighKDampingError("zero-strength filter is not exact")
    _, _, _, _, wind_mean, wind_scale = _normalizers(group)
    stepper = _pointwise_stepper(
        Path(contract["artifacts"]["selected_checkpoint"]["path"]),
        Path(contract["artifacts"]["pointwise_normalization"]["path"]),
        contract["architecture"],
        torch.device("cpu"),
        wet,
        wind_mean,
        wind_scale,
        layernorm=False,
    )
    physical = np.asarray(
        group["state"][0, TRAINING_TIMES[0] : TRAINING_TIMES[0] + 1],
        dtype=np.float32,
    )
    current = stepper.normalized_state(physical)
    fixed_static = stepper.normalized_static(
        group["static_features"],
        np.asarray([0], dtype=np.int64),
    )
    with torch.no_grad():
        raw = stepper.step(current, fixed_static)
        filtered = reflected_binomial_damping(
            raw,
            wet,
            torch.tensor([0.02], dtype=torch.float32),
        )
    prediction = _physical64(stepper, filtered)
    if not np.isfinite(prediction).all() or np.any(prediction[:, :, ~wet] != 0):
        raise HighKDampingError("native-grid filtered one-step smoke failed")
    result = json.loads(
        Path(contract["artifacts"]["long_truth_result"]["path"]).read_text()
    )
    if result.get("slurm_job_id") != "304735" or result.get("returncode") != 0:
        raise HighKDampingError("long inference truth changed")
    return {
        "status": "pass",
        "contract": str(resolved),
        "contract_sha256": digest,
        "training_records": records.tolist(),
        "training_records_count": int(records.shape[0]),
        "training_only": True,
        "inference_open_condition": "nonzero_alpha_passes_all_training_gates",
        "retraining_steps": 0,
        "checkpoint_selection": False,
        "zero_alpha_exact": True,
        "native_grid_filtered_one_step": {
            "finite": True,
            "land_zero": True,
        },
    }


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("high-k damping control requires PyTorch")
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
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    _, _, _, _, wind_mean, wind_scale = _normalizers(group)
    stepper = _pointwise_stepper(
        Path(contract["artifacts"]["selected_checkpoint"]["path"]),
        Path(contract["artifacts"]["pointwise_normalization"]["path"]),
        contract["architecture"],
        device,
        wet,
        wind_mean,
        wind_scale,
        layernorm=False,
    )
    records = training_records(split)
    training_initial = _gather_state(state, records[:, 1])
    normalized_static = stepper.normalized_static(
        static,
        np.zeros(records.shape[0], dtype=np.int64),
    )
    _, training_climatology, count = _s0_training_climatology(state, split, wet)
    if count != 5040:
        raise HighKDampingError("training climatology changed")
    training = _rollout(
        stepper,
        training_initial,
        normalized_static,
        ALPHAS,
        lambda lead: _gather_state(state, records[:, 1] + lead),
        training_climatology,
        TRAINING_LEADS,
        TRAINING_SPECTRUM_LEADS,
        wet,
    )
    selected_index, selection_records = select_alpha(training)
    selected_alpha = ALPHAS[selected_index] if selected_index is not None else None

    inference: dict[str, np.ndarray] | None = None
    inference_summary: dict[str, Any] | None = None
    if selected_alpha is not None:
        long_result = json.loads(
            Path(contract["artifacts"]["long_truth_result"]["path"]).read_text()
        )
        truth_source = ContinuousS0Truth(state, Path(long_result["run_dir"]), wet)
        inference_starts = np.asarray(EXPECTED_STARTS, dtype=np.int64)
        inference_initial = truth_source.batch(inference_starts)
        inference_static = stepper.normalized_static(
            static,
            np.zeros(inference_starts.size, dtype=np.int64),
        )
        inference = _rollout(
            stepper,
            inference_initial,
            inference_static,
            (0.0, selected_alpha),
            lambda lead: truth_source.batch(inference_starts + lead),
            training_climatology,
            INFERENCE_LEADS,
            INFERENCE_SPECTRUM_LEADS,
            wet,
        )
        inference_summary = _inference_summary(inference, selected_alpha)

    summary = {
        "classification_scope": (
            "zero_retraining_causal_filter_control_no_checkpoint_promotion"
        ),
        "training_gate": TRAINING_GATE,
        "selection_records": selection_records,
        "selected_alpha": selected_alpha,
        "inference_opened": inference is not None,
        "inference_characterization": inference_summary,
        "decision": (
            "training_gate_failed_no_inference_opened"
            if selected_alpha is None
            else inference_summary["interpretation"]
        ),
        "checkpoint_promotion": False,
    }

    scratch = Path(contract["output"]["scratch"]).resolve()
    project = Path(contract["output"]["project"]).resolve()
    scratch_tmp = scratch.with_name(scratch.name + ".tmp")
    project_tmp = project.with_name(project.name + ".tmp")
    scratch_tmp.parent.mkdir(parents=True, exist_ok=True)
    project_tmp.parent.mkdir(parents=True, exist_ok=True)
    scratch_tmp.mkdir(exist_ok=False)
    project_tmp.mkdir(exist_ok=False)
    try:
        packed = {f"training__{key}": value for key, value in training.items()}
        if inference is not None:
            packed.update(
                {f"inference__{key}": value for key, value in inference.items()}
            )
        np.savez_compressed(scratch_tmp / ARRAYS, **packed)
        plt.rcParams.update({"font.size": 9, "figure.dpi": 160})
        _plot_training(project_tmp, selection_records, selected_index)
        _plot_inference_rmse(project_tmp, inference, selected_alpha)
        _plot_inference_envelope(project_tmp, inference, selected_alpha)
        _plot_inference_spectra(project_tmp, inference, selected_alpha)
        _write_csv(project_tmp / CSV, training, inference)
        (project_tmp / SUMMARY).write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        report = {
            "version": VERSION,
            "status": "complete",
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "classification_scope": summary["classification_scope"],
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
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        shutil.copy2(scratch_tmp / REPORT, project_tmp / REPORT)
        shutil.copy2(scratch_tmp / ARRAYS, project_tmp / ARRAYS)
        (project_tmp / README).write_text(
            "# Model C S0 high-k damping control\n\n"
            "Zero retraining. A filter strength is selected only on fixed "
            "training-only S0 trajectories. Fixed day-2000 inference opens "
            "only after the prospective training gate passes. No checkpoint "
            "can be promoted by this control.\n\n"
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
