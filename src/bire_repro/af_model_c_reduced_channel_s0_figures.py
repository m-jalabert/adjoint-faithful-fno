"""Bire Figure 3--8 analogues for Model C Arm R under the S0 control wind.

The evaluator runs only after split-1 training and checkpoint selection have
completed.  It uses the same 15 prospectively fixed S0 inference starts,
continuous day-2000 MITgcm truth, persistence, S0 training climatology, metric
reductions, and filenames as ``anomaly_direct_bire_s0_inference_v1``.
Figure 6 is the causal comparison between the retained 46-channel source map
and the new ten-channel Arm-R map.
"""

from __future__ import annotations

import argparse
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

from . import af_model_c_bire_s0_figures as source_figures
from .af_a0_evaluate import _normalizers
from .af_data import STATE_CHANNELS
from .af_forward_complete import _member_acc, _member_rmse
from .af_model_c_anomaly_direct import PointwiseDirectStepper
from .af_model_c_bire_s0_figures import (
    ACC_FIELDS,
    ARRAYS_NAME,
    CSV_NAME,
    FIGURE_3_LEADS,
    FIGURE_7_LEADS,
    FIGURE_NAMES,
    LEAD_DAYS,
    MANIFEST_NAME,
    METHODS,
    README_NAME,
    REPORT_NAME,
    RMSE_FIELDS,
    SHORT_LEAD_DAYS,
    SUMMARY_NAME,
    ContinuousS0Truth,
    _finite_bound,
    _masked,
    _plot_rmse,
    _plot_single_member,
    _style,
    _summary,
    _write_csv,
    percentile_curve,
)
from .af_model_c_bire_s0_long_truth import EXPECTED_STARTS
from .af_model_c_reduced_channel_control import (
    CHECKPOINT_NAME,
    NORMALIZATION_NAME,
    REPORT_NAME as TRAINING_REPORT_NAME,
    ReducedDirectStepper,
    load_contract,
)
from .af_model_c_reduced_channels import (
    REDUCED_CHANNELS,
    ReducedChannelArchitecture,
    build_reduced_model,
    file_sha256,
    json_sha256,
    reduce_full_state,
    reduced_fields,
)
from .af_model_c_successor import (
    ModelCSuccessorArchitecture,
    build_successor,
)
from .af_model_c_successor_validation import curve_auc

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


VERSION = "model_c_reduced_channel_s0_figures_v1"
CONTRACT_STATUS = "frozen_before_arm_r_training_or_held_evaluation"


class ModelCReducedS0FigureError(RuntimeError):
    """Raised when the Arm-R S0 evaluation contract is violated."""


def _verify_artifact(specification: Mapping[str, Any], label: str) -> Path:
    path = Path(str(specification["path"])).resolve()
    if (
        not path.is_file()
        or file_sha256(path) != specification["sha256"]
    ):
        raise ModelCReducedS0FigureError(
            f"fixed S0 artifact changed: {label}"
        )
    return path


def _long_truth_preflight(contract: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = contract["s0_evaluation"]["artifacts"]
    result_path = _verify_artifact(
        artifacts["long_truth_result"],
        "long_truth_result",
    )
    _verify_artifact(
        artifacts["long_truth_manifest"],
        "long_truth_manifest",
    )
    result = json.loads(result_path.read_text())
    if (
        result.get("slurm_job_id") != "304735"
        or result.get("returncode") != 0
        or result.get("diagnostics")
        != {"dynState": 2160, "surfState": 2160}
        or int(result.get("start_iteration", -1)) != 3110400
        or int(result.get("end_iteration", -1)) != 3265920
    ):
        raise ModelCReducedS0FigureError("long S0 truth result is invalid")
    run_dir = Path(result["run_dir"])
    dyn = sorted(run_dir.glob("dynState.*.meta"))
    surf = sorted(run_dir.glob("surfState.*.meta"))
    if (
        len(dyn) != 2160
        or len(surf) != 2160
        or max(EXPECTED_STARTS) + 2000 >= 7200 + len(dyn)
    ):
        raise ModelCReducedS0FigureError("long S0 chronology is incomplete")
    return {
        "long_truth_job": 304735,
        "long_truth_records": len(dyn),
        "run_dir": str(run_dir),
    }


def _training_artifacts(
    contract: Mapping[str, Any],
    contract_sha: str,
) -> tuple[dict[str, Any], Path, Path]:
    training_output = Path(
        contract["output"]["training_scratch"]
    ).resolve()
    report_path = training_output / TRAINING_REPORT_NAME
    checkpoint_path = training_output / CHECKPOINT_NAME
    normalization_path = training_output / NORMALIZATION_NAME
    if not all(
        path.is_file()
        for path in (
            report_path,
            checkpoint_path,
            normalization_path,
        )
    ):
        raise FileNotFoundError("Arm-R training artifacts are not complete")
    report = json.loads(report_path.read_text())
    if (
        report.get("status") != "complete"
        or report.get("contract_sha256") != contract_sha
        or tuple(report.get("reduced_channels", ())) != REDUCED_CHANNELS
        or report.get("selected_checkpoint_sha256")
        != file_sha256(checkpoint_path)
        or report.get("normalization_artifact_sha256")
        != file_sha256(normalization_path)
        or report.get("save_reload_nine_step_bitwise_exact") is not True
        or report.get("held_s0_state_opened") is not False
    ):
        raise ModelCReducedS0FigureError(
            "Arm-R training provenance is invalid"
        )
    return report, checkpoint_path, normalization_path


def preflight(
    contract_path: str | Path,
    *,
    require_training: bool = True,
) -> dict[str, Any]:
    """Verify all fixed sources and optionally the completed training phase."""

    contract, resolved, digest = load_contract(contract_path)
    protocol = contract["s0_evaluation"]["protocol"]
    if (
        tuple(protocol.get("start_draw_order", ())) != EXPECTED_STARTS
        or int(protocol.get("maximum_lead_days", -1)) != 2000
        or int(protocol.get("prediction_interval_days", -1)) != 10
        or int(protocol.get("member_count", -1)) != 15
        or tuple(protocol.get("figure_names", ())) != FIGURE_NAMES
        or tuple(protocol.get("rmse_fields", ())) != RMSE_FIELDS
        or tuple(protocol.get("acc_fields", ())) != ACC_FIELDS
    ):
        raise ModelCReducedS0FigureError("Arm-R figure protocol changed")
    truth = _long_truth_preflight(contract)
    artifacts = contract["s0_evaluation"]["artifacts"]
    for label in (
        "full_dataset_metadata",
        "source_checkpoint",
        "source_normalization",
    ):
        _verify_artifact(artifacts[label], label)
    training_ready = False
    if require_training:
        _training_artifacts(contract, digest)
        training_ready = True
    else:
        try:
            _training_artifacts(contract, digest)
            training_ready = True
        except FileNotFoundError:
            training_ready = False
    return {
        "status": "ready" if training_ready else "awaiting_training",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "training_ready": training_ready,
        "member_count": 15,
        "figure_count": len(FIGURE_NAMES),
        **truth,
    }


def _arm_r_stepper(
    contract: Mapping[str, Any],
    contract_sha: str,
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> tuple[ReducedDirectStepper, dict[str, Any]]:
    report, checkpoint, normalization = _training_artifacts(
        contract,
        contract_sha,
    )
    payload = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )
    architecture_dict = contract["architecture"]
    if (
        payload.get("architecture") != architecture_dict
        or payload.get("contract_sha256") != contract_sha
        or tuple(payload.get("reduced_channels", ())) != REDUCED_CHANNELS
    ):
        raise ModelCReducedS0FigureError("Arm-R checkpoint identity changed")
    architecture = ReducedChannelArchitecture(**architecture_dict)
    model = build_reduced_model(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with np.load(normalization) as artifact:
        mean = np.asarray(artifact["pointwise_mean"], dtype=np.float32)
        scale = np.asarray(artifact["pointwise_scale"], dtype=np.float32)
    if mean.shape != (10, 62, 62) or scale.shape != mean.shape:
        raise ModelCReducedS0FigureError(
            "Arm-R normalization dimensions changed"
        )
    return (
        ReducedDirectStepper(
            model=model,
            device=device,
            wet=wet,
            mean=mean,
            scale=scale,
            wind_mean=wind_mean,
            wind_scale=wind_scale,
        ),
        report,
    )


def _source_stepper(
    contract: Mapping[str, Any],
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> PointwiseDirectStepper:
    artifacts = contract["s0_evaluation"]["artifacts"]
    checkpoint = Path(artifacts["source_checkpoint"]["path"])
    normalization = Path(artifacts["source_normalization"]["path"])
    payload = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )
    architecture_dict = contract["s0_evaluation"][
        "source_model_architecture"
    ]
    if (
        payload.get("architecture") != architecture_dict
        or int(payload.get("optimizer_step", -1)) != 13440
    ):
        raise ModelCReducedS0FigureError("source checkpoint changed")
    architecture = ModelCSuccessorArchitecture(**architecture_dict)
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


def _s0_climatology(
    state: Any,
    snapshot_codes: np.ndarray,
    wet: np.ndarray,
    *,
    chunk_days: int = 64,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    selected = np.flatnonzero(
        np.asarray(snapshot_codes, dtype=np.uint8) == 1
    )
    cuts = np.flatnonzero(np.diff(selected) != 1) + 1
    blocks = np.split(selected, cuts)
    state_sum = np.zeros((10, *wet.shape), dtype=np.float64)
    field_sum = {
        name: np.zeros(wet.shape, dtype=np.float64)
        for name in (
            "surface_speed",
            "surface_u",
            "surface_v",
            "sst",
            "phihyd_surface",
            "streamfunction",
        )
    }
    count = 0
    for block in blocks:
        run_start, run_stop = int(block[0]), int(block[-1]) + 1
        for start in range(run_start, run_stop, chunk_days):
            stop = min(start + chunk_days, run_stop)
            raw = np.asarray(state[0, start:stop], dtype=np.float32)
            state_sum += raw.sum(axis=0, dtype=np.float64)
            fields = reduced_fields(raw, wet)
            for name in field_sum:
                field_sum[name] += fields[name].sum(
                    axis=0,
                    dtype=np.float64,
                )
            count += int(raw.shape[0])
    if count != selected.size:
        raise ModelCReducedS0FigureError("S0 climatology count changed")
    state_mean = (state_sum / count).astype(np.float32)
    state_mean[:, ~wet] = 0.0
    field_mean = {
        name: (value / count).astype(np.float32)
        for name, value in field_sum.items()
    }
    for value in field_mean.values():
        value[~wet] = 0.0
    return state_mean, field_mean, int(count)


def _full_fields(states: np.ndarray, wet: np.ndarray) -> dict[str, np.ndarray]:
    reduced = reduce_full_state(states, wet)
    return reduced_fields(reduced, wet)


def _evaluate(
    arm_r: ReducedDirectStepper,
    source: PointwiseDirectStepper,
    truth_source: ContinuousS0Truth,
    static: Any,
    starts: np.ndarray,
    climate_state: np.ndarray,
    climate_fields: Mapping[str, np.ndarray],
    wet: np.ndarray,
) -> dict[str, np.ndarray]:
    member_count = int(starts.size)
    initial_full = truth_source.batch(starts)
    initial = reduce_full_state(initial_full, wet)
    experiments = np.zeros(member_count, dtype=np.int64)
    arm_current = arm_r.normalized_state(initial)
    arm_static = arm_r.normalized_static(static, experiments)
    source_current = source.normalized_state(initial_full)
    source_static = source.normalized_static(static, experiments)
    initial_fields = reduced_fields(initial, wet)
    climate = reduced_fields(
        np.repeat(climate_state[None], member_count, axis=0),
        wet,
    )
    for name, value in climate_fields.items():
        climate[name] = np.repeat(
            value[None],
            member_count,
            axis=0,
        )
    arrays: dict[str, np.ndarray] = {
        "lead_days": np.asarray(LEAD_DAYS, dtype=np.int16),
        "short_lead_days": np.asarray(
            SHORT_LEAD_DAYS,
            dtype=np.int16,
        ),
        "start_draw_order": starts.astype(np.int32),
        "finite": np.empty(
            (member_count, len(LEAD_DAYS)),
            dtype=np.uint8,
        ),
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
    arrays["single_rmse__sst"] = np.empty(
        len(SHORT_LEAD_DAYS),
        dtype=np.float32,
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
    wet_tensor = torch.from_numpy(wet).to(arm_r.device)
    with torch.no_grad():
        for lead_index, lead in enumerate(LEAD_DAYS):
            if lead:
                arm_current = arm_r.step(arm_current, arm_static)
                arm_prediction = arm_r.physical(arm_current)
                if lead <= 200:
                    source_current = source.step(
                        source_current,
                        source_static,
                    )
                    source_prediction = source.physical(source_current)
            else:
                arm_prediction = initial.copy()
                source_prediction = initial_full.copy()
            truth_full = truth_source.batch(starts + lead)
            truth = reduce_full_state(truth_full, wet)
            truth_fields = reduced_fields(truth, wet)
            arm_fields = reduced_fields(arm_prediction, wet)
            for field in RMSE_FIELDS:
                arrays[f"rmse__model__{field}"][:, lead_index] = (
                    _member_rmse(
                        arm_fields[field],
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
                        climate[field],
                        truth_fields[field],
                        wet,
                    )
                )
            arrays["finite"][:, lead_index] = np.isfinite(
                arm_prediction
            ).all(axis=(1, 2, 3))
            arrays["normalized_max_abs"][:, lead_index] = (
                torch.amax(
                    torch.abs(arm_current[:, :, wet_tensor]),
                    dim=(1, 2),
                )
                .detach()
                .cpu()
                .numpy()
            )
            if lead <= 200:
                short_index = lead // 10
                source_fields = _full_fields(source_prediction, wet)
                for field in ACC_FIELDS:
                    arrays[f"acc__selected__{field}"][
                        :,
                        short_index,
                    ] = _member_acc(
                        arm_fields[field],
                        truth_fields[field],
                        climate[field],
                        wet,
                    )
                    arrays[f"acc__prior__{field}"][
                        :,
                        short_index,
                    ] = _member_acc(
                        source_fields[field],
                        truth_fields[field],
                        climate[field],
                        wet,
                    )
                arrays["single_rmse__streamfunction"][
                    short_index
                ] = _member_rmse(
                    arm_fields["streamfunction"],
                    truth_fields["streamfunction"],
                    wet,
                )[0]
                arrays["single_rmse__sst"][short_index] = _member_rmse(
                    arm_fields["sst"],
                    truth_fields["sst"],
                    wet,
                )[0]
            if lead in figure3_lookup:
                destination = figure3_lookup[lead]
                arrays["figure3_truth_streamfunction"][
                    destination
                ] = truth_fields["streamfunction"][0]
                arrays["figure3_model_streamfunction"][
                    destination
                ] = arm_fields["streamfunction"][0]
            if lead in figure7_lookup:
                destination = figure7_lookup[lead]
                arrays["figure7_truth_streamfunction"][
                    destination
                ] = truth_fields["streamfunction"][0]
                arrays["figure7_model_streamfunction"][
                    destination
                ] = arm_fields["streamfunction"][0]
    return arrays


def _plot_streamfunction_grid(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    longitude: np.ndarray,
    latitude: np.ndarray,
    wet: np.ndarray,
) -> None:
    truth = np.asarray(arrays["figure3_truth_streamfunction"])
    model = np.asarray(arrays["figure3_model_streamfunction"])
    bound = _finite_bound((truth, model))
    figure, axes = plt.subplots(
        len(FIGURE_3_LEADS),
        2,
        figsize=(7.0, 12.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for row, lead in enumerate(FIGURE_3_LEADS):
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
        axes[row, 0].set_ylabel(f"Day {lead}\nLatitude (°)")
    axes[0, 0].set_title("MITgcm ground truth")
    axes[0, 1].set_title("Arm R: ten-channel Model C")
    axes[-1, 0].set_xlabel("Longitude (°)")
    axes[-1, 1].set_xlabel("Longitude (°)")
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
    figure.savefig(output / FIGURE_NAMES[0], bbox_inches="tight")
    plt.close(figure)


def _plot_acc(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    labels = source_figures.FIELD_LABELS
    leads = np.asarray(SHORT_LEAD_DAYS)
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(5.4, 10.2),
        sharex=True,
        constrained_layout=True,
    )
    for axis, field in zip(axes, ACC_FIELDS):
        arm = percentile_curve(arrays[f"acc__selected__{field}"])
        source = percentile_curve(arrays[f"acc__prior__{field}"])
        axis.plot(
            leads,
            source["mean"],
            color="black",
            linewidth=1.5,
            label="46-channel source Model C",
        )
        axis.plot(
            leads,
            source["p10"],
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        axis.plot(
            leads,
            source["p90"],
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        axis.plot(
            leads,
            arm["mean"],
            color="red",
            linewidth=1.6,
            label="Arm R: ten-channel Model C",
        )
        axis.fill_between(
            leads,
            arm["p10"],
            arm["p90"],
            color="red",
            alpha=0.17,
            linewidth=0,
        )
        axis.axhline(0.0, color="0.65", linewidth=0.6)
        axis.set_ylim(-1.0, 1.02)
        axis.set_ylabel(f"{labels[field]}\nACC")
        axis.grid(color="0.82", linewidth=0.6)
    axes[0].set_title(
        r"S0 channel-count causal comparison; $\Delta t=10$ days"
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
    axes[0, 1].set_title("Arm R: ten-channel Model C")
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


def _deterministic_gate(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    selected = slice(1, 10)
    fields = {}
    passed = True
    for field in RMSE_FIELDS:
        model = np.asarray(
            arrays[f"rmse__model__{field}"][:, selected],
            dtype=np.float64,
        )
        comparisons = {}
        for baseline in ("persistence", "climatology"):
            reference = np.asarray(
                arrays[f"rmse__{baseline}__{field}"][:, selected],
                dtype=np.float64,
            )
            ratio = float(
                curve_auc(model).mean()
                / curve_auc(reference).mean()
            )
            comparisons[baseline] = {
                "rmse_auc_ratio_10_90": ratio,
                "passed": ratio < 1.0,
            }
            passed = passed and ratio < 1.0
        fields[field] = comparisons
    return {
        "passed": bool(passed),
        "requirement": (
            "each_primary_field_10_90_day_rmse_auc_below_persistence_"
            "and_S0_training_climatology"
        ),
        "fields": fields,
    }


def _readme(report: Mapping[str, Any]) -> str:
    return f"""# Model C Arm R: S0 Bire-style Figures 3--8

This package evaluates the ten-output reduced-channel causal control under
the S0 control wind (tau0 = 0.1 N m-2) at a ten-day prediction interval.

Arm R retains selected Model C's pointwise anomaly normalization, direct-state
map, width 128, four FNO blocks, 24x16 modes, 10% padding, optimizer, and
three-step objective form. Its autoregressive state contains only surface/mid
U, V, and temperature; surface/mid/bottom PHIHYD; and barotropic
streamfunction.

The 15 starts, continuous day-2000 MITgcm truth, persistence, S0 split-1
climatology, metric reductions, and six filenames match the source
`anomaly_direct_bire_s0_inference_v1` evaluation. Figure 6 compares the
retained 46-channel source map (black) with Arm R (red); it is the direct
channel-count causal comparison.

Training-only checkpoint gate passed: {
str(report["training_selection_passed"]).lower()
}.
S0 10--90-day deterministic gate passed: {
str(report["deterministic_gate"]["passed"]).lower()
}.
Report content SHA-256: `{report["report_content_sha256"]}`.
"""


def evaluate(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run the fixed S0 ensemble and publish the six matching figures."""

    if torch is None:
        raise RuntimeError("Arm-R S0 figures require PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    preflight(resolved_contract, require_training=True)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested without a GPU")
    device = torch.device(device_name)
    scratch_output = Path(
        contract["output"]["figure_scratch"]
    ).resolve()
    project_output = Path(
        contract["output"]["figure_project"]
    ).resolve()
    scratch_temporary = scratch_output.with_name(
        scratch_output.name + ".tmp"
    )
    project_temporary = project_output.with_name(
        project_output.name + ".tmp"
    )
    if any(
        path.exists()
        for path in (
            scratch_output,
            project_output,
            scratch_temporary,
            project_temporary,
        )
    ):
        raise FileExistsError("refusing to overwrite Arm-R figure outputs")
    artifacts = contract["s0_evaluation"]["artifacts"]
    full_dataset = Path(
        artifacts["full_dataset_metadata"]["path"]
    ).parent
    reduced_dataset = Path(
        contract["source_artifacts"]["reduced_dataset"]["path"]
    )
    full_group = zarr.open_consolidated(
        str(full_dataset),
        mode="r",
    )
    reduced_group = zarr.open_consolidated(
        str(reduced_dataset),
        mode="r",
    )
    if (
        tuple(full_group.attrs["state_channels"]) != STATE_CHANNELS
        or tuple(reduced_group.attrs["state_channels"])
        != REDUCED_CHANNELS
    ):
        raise ModelCReducedS0FigureError("dataset channels changed")
    state = full_group["state"]
    reduced_state = reduced_group["state"]
    static = full_group["static_features"]
    wet = np.asarray(full_group["wet_mask"][:], dtype=bool)
    longitude = np.asarray(
        full_group["longitude_deg"][:],
        dtype=np.float32,
    )
    latitude = np.asarray(
        full_group["latitude_deg"][:],
        dtype=np.float32,
    )
    snapshot_codes = np.asarray(
        full_group["snapshot_split"][:],
        dtype=np.uint8,
    )
    _, _, _, _, wind_mean, wind_scale = _normalizers(full_group)
    long_result = json.loads(
        Path(artifacts["long_truth_result"]["path"]).read_text()
    )
    truth = ContinuousS0Truth(
        state,
        Path(long_result["run_dir"]),
        wet,
    )
    starts = np.asarray(EXPECTED_STARTS, dtype=np.int64)
    climate_state, climate_fields, climate_days = _s0_climatology(
        reduced_state,
        snapshot_codes,
        wet,
    )
    if climate_days != 5040:
        raise ModelCReducedS0FigureError(
            "S0 training climatology coverage changed"
        )
    arm_r, training_report = _arm_r_stepper(
        contract,
        contract_sha,
        device,
        wet,
        wind_mean,
        wind_scale,
    )
    source = _source_stepper(
        contract,
        device,
        wet,
        wind_mean,
        wind_scale,
    )
    arrays = _evaluate(
        arm_r,
        source,
        truth,
        static,
        starts,
        climate_state,
        climate_fields,
        wet,
    )
    arrays["longitude_deg"] = longitude
    arrays["latitude_deg"] = latitude
    arrays["wet_mask"] = wet.astype(np.uint8)
    summary = _summary(arrays)
    deterministic_gate = _deterministic_gate(arrays)
    scratch_temporary.parent.mkdir(parents=True, exist_ok=True)
    project_temporary.parent.mkdir(parents=True, exist_ok=True)
    scratch_temporary.mkdir(exist_ok=False)
    project_temporary.mkdir(exist_ok=False)
    try:
        scratch_arrays = scratch_temporary / ARRAYS_NAME
        np.savez_compressed(scratch_arrays, **arrays)
        source_figures.METHOD_LABELS["model"] = (
            "Arm R: ten-channel Model C"
        )
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
            "training_report": str(
                Path(contract["output"]["training_scratch"])
                / TRAINING_REPORT_NAME
            ),
            "training_report_content_sha256": training_report[
                "report_content_sha256"
            ],
            "training_selection_passed": bool(
                training_report["selection_decision"]["passed"]
            ),
            "selected_optimizer_step": int(
                training_report["selection_decision"][
                    "selected_optimizer_step"
                ]
            ),
            "selected_checkpoint": training_report[
                "selected_checkpoint"
            ],
            "selected_checkpoint_sha256": training_report[
                "selected_checkpoint_sha256"
            ],
            "source_checkpoint": artifacts["source_checkpoint"]["path"],
            "source_checkpoint_sha256": artifacts[
                "source_checkpoint"
            ]["sha256"],
            "long_truth_job": "304735",
            "dataset": str(full_dataset),
            "reduced_dataset": str(reduced_dataset),
            "device": str(device),
            "ensemble": contract["s0_evaluation"]["protocol"],
            "baselines": contract["s0_evaluation"]["baselines"],
            "figure6": contract["s0_evaluation"]["figure6"],
            "summary": summary,
            "deterministic_gate": deterministic_gate,
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
    check.add_argument(
        "--allow-missing-training",
        action="store_true",
    )
    run = commands.add_parser("run")
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(
            args.contract,
            require_training=not args.allow_missing_training,
        )
    else:
        result = evaluate(args.contract, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
