"""One bounded deep-pressure spectral fine-tune for anomaly-direct Model C.

The corrected three-seed training attribution showed that the day-360
high-wavenumber PHIHYD tail is reproducible while total pressure energy and
10--90-day primary skill remain good.  This module implements the single
prospectively frozen follow-up:

* start from the accepted median seed/checkpoint;
* retain pointwise anomaly coordinates, direct-state prediction, architecture,
  and the complete loss-v1 objective;
* add one differentiable physical-PHIHYD high-mode error term on levels 7--14
  over the existing 10/20/30-day differentiable rollout;
* select only on fixed split-1 trajectories through day 360.

No validation, held-inference, response, or adjoint state is opened.
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

import numpy as np
import zarr

from .af_a0_evaluate import _normalizers
from .af_model_a import (
    ChunkAwareBatchSampler,
    _checkpoint_state_dict,
    require_model_a_runtime,
    seed_everything,
)
from .af_model_b import records_for_rollout_split, western_boundary_mask
from .af_model_c import (
    MODEL_C_LOSS_V1_CONTRACT_SHA256,
    loss_contract,
    loss_contract_sha256,
    model_c_loss_config,
    model_c_loss_terms,
    wet_rectangle_bounds,
)
from .af_model_c_anomaly_direct import (
    ModelCAnomalyRolloutDataset,
    direct_state_unroll,
    pointwise_increment_scale,
)
from . import (
    af_model_c_anomaly_direct_training_spectral_attribution_v2 as attribution,
)
from .af_model_c_overfit import AUDIT_TERMS, _device, _file_sha256
from .af_model_c_successor import (
    STATE_CHANNEL_COUNT,
    ModelCSuccessorArchitecture,
    build_successor,
)
from .af_pressure import (
    DRF_M,
    GRAVITY_M_S2,
    T_REF_C,
    THERMAL_EXPANSION_PER_C,
)

try:
    import torch
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


VERSION = "model_c_anomaly_direct_deep_pressure_spectral_regularization_v1"
REPORT_NAME = "deep_pressure_spectral_regularization_report.json"
ARRAYS_NAME = "deep_pressure_spectral_regularization_arrays.npz"
CHECKPOINT_NAME = "model_c_anomaly_direct_deep_pressure_spectral_best.pt"
CHECKPOINT_DIRECTORY = "fine_tune_checkpoints"
SELECTION_FIGURE = "model_c_deep_pressure_spectral_checkpoint_selection.png"
SPECTRAL_FIGURE = "model_c_deep_pressure_spectral_rollout.png"
README_NAME = "README.md"
MANIFEST_NAME = "manifest.json"
SOURCE_OPTIMIZER_STEP = 13440
TAIL_START_MODE = 10
TAIL_END_MODE = 30
DEEP_START_LEVEL = 7
TRAINING_TERMS = (
    "total",
    "base_total",
    "deep_pressure_tail",
    *(name for name in AUDIT_TERMS if name != "total"),
)


class DeepPressureSpectralRegularizationError(RuntimeError):
    """Raised when the frozen fine-tune contract or evidence changes."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _vertical_distances() -> tuple[np.ndarray, np.ndarray]:
    interfaces = -np.concatenate(([0.0], np.cumsum(DRF_M)))
    centers = 0.5 * (interfaces[:-1] + interfaces[1:])
    center_spacing = np.diff(centers)
    above = np.empty(DRF_M.size, dtype=np.float32)
    below = np.empty(DRF_M.size, dtype=np.float32)
    above[0] = interfaces[0] - centers[0]
    above[1:] = -0.5 * center_spacing
    below[:-1] = -0.5 * center_spacing
    below[-1] = centers[-1] - interfaces[-1]
    return above, below


def torch_phihyd_from_normalized_state(
    normalized_state: Any,
    pointwise_mean: Any,
    pointwise_scale: Any,
) -> Any:
    """Differentiably reconstruct all 15 MITgcm PHIHYD levels."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("PHIHYD reconstruction requires PyTorch")
    if normalized_state.ndim not in (4, 5):
        raise ValueError("normalized state must have N,C,Y,X or N,S,C,Y,X shape")
    channel_axis = normalized_state.ndim - 3
    if normalized_state.shape[channel_axis] != STATE_CHANNEL_COUNT:
        raise ValueError("normalized state must contain all 46 dynamic channels")
    expected = (STATE_CHANNEL_COUNT, *normalized_state.shape[-2:])
    if tuple(pointwise_mean.shape[-3:]) != expected or tuple(
        pointwise_scale.shape[-3:]
    ) != expected:
        raise ValueError("pointwise PHIHYD normalizers have changed shape")

    leading = (1,) * (normalized_state.ndim - 3)
    mean = pointwise_mean.reshape(*leading, *expected)
    scale = pointwise_scale.reshape(*leading, *expected)
    physical = normalized_state * scale + mean
    theta = physical[..., 30:45, :, :]
    eta = physical[..., 45, :, :]
    reference = torch.as_tensor(
        T_REF_C,
        dtype=physical.dtype,
        device=physical.device,
    ).reshape(*leading, 15, 1, 1)
    density = -THERMAL_EXPANSION_PER_C * (theta - reference)
    above_values, below_values = _vertical_distances()
    above = torch.as_tensor(
        above_values,
        dtype=physical.dtype,
        device=physical.device,
    )
    below = torch.as_tensor(
        below_values,
        dtype=physical.dtype,
        device=physical.device,
    )
    interface = torch.zeros_like(eta)
    levels = []
    for level in range(15):
        center = (
            interface
            + above[level] * GRAVITY_M_S2 * density[..., level, :, :]
        )
        levels.append(center)
        interface = (
            center
            + below[level] * GRAVITY_M_S2 * density[..., level, :, :]
        )
    return torch.stack(levels, dim=-3) + GRAVITY_M_S2 * eta.unsqueeze(-3)


def deep_pressure_high_mode_loss(
    predictions: Any,
    targets: Any,
    pointwise_mean: Any,
    pointwise_scale: Any,
    wet: Any,
    *,
    deep_start_level: int = DEEP_START_LEVEL,
    tail_start_mode: int = TAIL_START_MODE,
    tail_end_mode: int = TAIL_END_MODE,
    truth_floor_fraction: float = 1.0e-8,
) -> Any:
    """High-mode complex-Fourier RMSE relative to truth tail energy."""

    if predictions.shape != targets.shape or predictions.ndim != 5:
        raise ValueError("pressure-tail loss needs matching N,S,C,Y,X rollouts")
    if not 0 <= deep_start_level < 15:
        raise ValueError("deep pressure start level is outside PHIHYD")
    if tail_start_mode < 1 or tail_end_mode < tail_start_mode:
        raise ValueError("invalid pressure-tail radial-mode band")
    if truth_floor_fraction <= 0.0:
        raise ValueError("pressure-tail truth floor must be positive")
    predicted_pressure = torch_phihyd_from_normalized_state(
        predictions,
        pointwise_mean,
        pointwise_scale,
    )[..., deep_start_level:, :, :]
    target_pressure = torch_phihyd_from_normalized_state(
        targets,
        pointwise_mean,
        pointwise_scale,
    )[..., deep_start_level:, :, :]
    y0, y1, x0, x1 = wet_rectangle_bounds(wet)

    def transform(value: Any) -> Any:
        cropped = value[..., y0:y1, x0:x1]
        cropped = cropped - cropped.mean(dim=(-2, -1), keepdim=True)
        window_y = torch.hann_window(
            cropped.shape[-2],
            periodic=False,
            dtype=cropped.dtype,
            device=cropped.device,
        )
        window_x = torch.hann_window(
            cropped.shape[-1],
            periodic=False,
            dtype=cropped.dtype,
            device=cropped.device,
        )
        return torch.fft.fft2(
            cropped * window_y[:, None] * window_x[None, :],
            dim=(-2, -1),
        )

    predicted_fft = transform(predicted_pressure)
    target_fft = transform(target_pressure)
    height, width = predicted_fft.shape[-2:]
    ky = torch.fft.fftfreq(height, device=predicted_fft.device) * height
    kx = torch.fft.fftfreq(width, device=predicted_fft.device) * width
    shell = torch.floor(torch.sqrt(ky[:, None].square() + kx[None, :].square()))
    tail = (shell >= tail_start_mode) & (shell <= tail_end_mode)
    if not bool(tail.any().item()):
        raise ValueError("pressure-tail mode band is empty")
    error_energy = (
        predicted_fft[..., tail] - target_fft[..., tail]
    ).abs().square().sum(dim=-1)
    truth_tail_energy = target_fft[..., tail].abs().square().sum(dim=-1)
    truth_total_energy = target_fft.abs().square().sum(dim=(-2, -1))
    denominator = torch.maximum(
        truth_tail_energy,
        truth_total_energy * truth_floor_fraction,
    ).clamp_min(torch.finfo(predictions.dtype).tiny)
    return torch.sqrt(error_energy / denominator).mean()


def summarize_evaluation(
    evaluated: Mapping[str, Any],
    *,
    source_primary_ratios: Mapping[str, float] | None = None,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize one fixed split-1 360-day evaluation."""

    fields = ("surface_speed", "sst", "phihyd_surface")
    model_rmse = np.asarray(evaluated["model_rmse"], dtype=np.float64)
    persistence_rmse = np.asarray(
        evaluated["persistence_rmse"],
        dtype=np.float64,
    )
    ratios = {
        field: float(
            np.mean(model_rmse[:9, index])
            / np.mean(persistence_rmse[:9, index])
        )
        for index, field in enumerate(fields)
    }
    spectral = np.asarray(evaluated["ratio"], dtype=np.float64)
    integrated = np.asarray(evaluated["integrated"], dtype=np.float64)
    mid_bottom = spectral[:, [7, 14]]
    result: dict[str, Any] = {
        "optimizer_step": int(evaluated["optimizer_step"]),
        "primary_10_to_90_rmse_ratio_to_persistence": ratios,
        "worst_primary_10_to_90_ratio": max(ratios.values()),
        "worst_mid_bottom_modewise_ratio_all_leads": float(
            np.max(mid_bottom)
        ),
        "day360_mid_modewise_ratio": float(spectral[-1, 7]),
        "day360_bottom_modewise_ratio": float(spectral[-1, 14]),
        "day360_mid_integrated_energy_ratio": float(integrated[-1, 7]),
        "day360_bottom_integrated_energy_ratio": float(integrated[-1, 14]),
        "first_mid_factor_four_failure_day": _first_failure_day(
            spectral[:, 7]
        ),
        "first_bottom_factor_four_failure_day": _first_failure_day(
            spectral[:, 14]
        ),
    }
    if source_primary_ratios is None or selection is None:
        return result
    relative = {
        field: ratios[field] / float(source_primary_ratios[field])
        for field in fields
    }
    energy_min, energy_max = (
        float(value) for value in selection["integrated_energy_bounds"]
    )
    result["primary_ratio_relative_to_source"] = relative
    result["worst_primary_relative_to_source"] = max(relative.values())
    result["gate"] = {
        "primary_below_persistence": bool(
            result["worst_primary_10_to_90_ratio"]
            < float(selection["primary_ratio_limit"])
        ),
        "primary_degradation_bounded": bool(
            result["worst_primary_relative_to_source"]
            <= float(selection["maximum_primary_relative_to_source"])
        ),
        "mid_bottom_factor_four_all_leads": bool(
            result["worst_mid_bottom_modewise_ratio_all_leads"]
            <= float(selection["spectral_ratio_limit"])
        ),
        "day360_integrated_energy_bounded": bool(
            energy_min
            <= result["day360_mid_integrated_energy_ratio"]
            <= energy_max
            and energy_min
            <= result["day360_bottom_integrated_energy_ratio"]
            <= energy_max
        ),
    }
    result["gate"]["pass"] = all(result["gate"].values())
    return result


def _first_failure_day(values: np.ndarray) -> int | None:
    failed = np.flatnonzero(np.asarray(values) > 4.0)
    return int((failed[0] + 1) * 10) if failed.size else None


def select_candidate(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select a passing fine-tune or retain immutable source step zero."""

    if not summaries or int(summaries[0]["fine_tune_step"]) != 0:
        raise ValueError("candidate selection requires source step zero first")
    passing = [
        value
        for value in summaries[1:]
        if bool(value.get("gate", {}).get("pass", False))
    ]
    if not passing:
        return {
            "status": "no_fine_tune_checkpoint_passed",
            "selected_fine_tune_step": 0,
            "selected_optimizer_step": int(summaries[0]["optimizer_step"]),
            "candidate_training_gate_passed": False,
            "next_action": (
                "retain_original_anomaly_direct_model_and_do_not_open_new_"
                "held_or_physics_archives"
            ),
        }
    selected = min(
        passing,
        key=lambda value: (
            float(value["worst_mid_bottom_modewise_ratio_all_leads"]),
            float(value["worst_primary_relative_to_source"]),
            int(value["fine_tune_step"]),
        ),
    )
    return {
        "status": "candidate_training_gate_passed",
        "selected_fine_tune_step": int(selected["fine_tune_step"]),
        "selected_optimizer_step": int(selected["optimizer_step"]),
        "candidate_training_gate_passed": True,
        "next_action": (
            "freeze_identical_two_seed_training_only_replication_before_any_"
            "new_held_inference_or_learned_physics_read"
        ),
    }


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the single-candidate source-locked fine-tune contract."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    objective = contract.get("objective", {})
    fine_tune = contract.get("fine_tune", {})
    read = contract.get("read_contract", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status")
        != "frozen_after_corrected_three_seed_attribution_before_fine_tune_metrics"
        or objective.get("base_loss_version") != "v1"
        or objective.get("pressure_levels") != list(range(7, 15))
        or objective.get("radial_modes") != [TAIL_START_MODE, TAIL_END_MODE]
        or float(objective.get("weight", -1.0)) != 0.001
        or int(fine_tune.get("source_optimizer_step", -1))
        != SOURCE_OPTIMIZER_STEP
        or int(fine_tune.get("maximum_steps", -1)) != 1920
        or tuple(fine_tune.get("checkpoint_steps", ()))
        != (240, 480, 960, 1440, 1920)
        or int(fine_tune.get("batch_size", -1)) != 4
        or float(fine_tune.get("learning_rate", -1.0)) != 2.0e-5
        or read.get("training_state") is not True
        or any(
            read.get(name) is not False
            for name in (
                "validation_state",
                "inference_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
            )
        )
    ):
        raise ValueError("deep-pressure spectral regularization contract changed")
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(f"fine-tune source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def _verify_artifacts(
    contract: Mapping[str, Any],
    dataset: Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    sources = contract["sources"]
    if (
        dataset != Path(sources["dataset"]["path"]).resolve()
        or _file_sha256(dataset / ".zmetadata")
        != sources["dataset"]["metadata_sha256"]
    ):
        raise DeepPressureSpectralRegularizationError("dataset source changed")
    for name in (
        "source_checkpoint",
        "source_normalization",
        "replication_summary",
        "attribution_report",
        "attribution_manifest",
    ):
        record = sources[name]
        artifact = Path(record["path"]).resolve()
        if not artifact.is_file() or _file_sha256(artifact) != record["sha256"]:
            raise DeepPressureSpectralRegularizationError(
                f"fine-tune source artifact changed: {name}"
            )
    attribution_contract, _, attribution_sha = attribution.load_contract(
        sources["attribution_contract"]["path"]
    )
    if attribution_sha != sources["attribution_contract"]["sha256"]:
        raise DeepPressureSpectralRegularizationError(
            "corrected attribution contract changed"
        )
    attribution._verify_sources(attribution_contract, dataset)
    report = json.loads(Path(sources["attribution_report"]["path"]).read_text())
    manifest = json.loads(
        Path(sources["attribution_manifest"]["path"]).read_text()
    )
    if (
        report.get("classification") != "seed_consistent_training_split_tail"
        or report.get("next_decision")
        != (
            "freeze_one_single_candidate_training_only_direct_state_deep_"
            "pressure_spectral_regularization_fine_tune_with_split1_selection_"
            "and_no_inference_read"
        )
        or report.get("content_sha256")
        != sources["attribution_report"]["content_sha256"]
        or manifest.get("content_sha256")
        != sources["attribution_manifest"]["content_sha256"]
    ):
        raise DeepPressureSpectralRegularizationError(
            "corrected attribution does not authorize the fine-tune"
        )
    source_checkpoint = Path(sources["source_checkpoint"]["path"]).resolve()
    source_normalization = Path(
        sources["source_normalization"]["path"]
    ).resolve()
    if torch is None:  # pragma: no cover
        raise RuntimeError("fine-tune source verification requires PyTorch")
    payload = torch.load(
        source_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if (
        payload.get("version") != "model_c_anomaly_direct_v1"
        or int(payload.get("optimizer_step", -1)) != SOURCE_OPTIMIZER_STEP
        or payload.get("loss_contract_sha256")
        != MODEL_C_LOSS_V1_CONTRACT_SHA256
    ):
        raise DeepPressureSpectralRegularizationError(
            "source anomaly-direct checkpoint changed"
        )
    return source_checkpoint, source_normalization, payload, attribution_contract


def _checkpoint_path(directory: Path, fine_tune_step: int) -> Path:
    return directory / (
        "model_c_anomaly_direct_deep_pressure_spectral_"
        f"step_{fine_tune_step:04d}.pt"
    )


def _evaluate_checkpoint(
    checkpoint: Path,
    optimizer_step: int,
    normalization: Path,
    *,
    device: Any,
    initial: np.ndarray,
    raw_static: np.ndarray,
    experiments: np.ndarray,
    state: Any,
    records: np.ndarray,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
    modes: np.ndarray,
) -> dict[str, Any]:
    record = {
        "seed": 20260724,
        "optimizer_step": int(optimizer_step),
        "checkpoint": {"path": str(checkpoint)},
        "normalization": {"path": str(normalization)},
    }
    return attribution._evaluate_seed(
        record,
        device=device,
        initial=initial,
        raw_static=raw_static,
        experiments=experiments,
        state=state,
        records=records,
        wet=wet,
        wind_mean=wind_mean,
        wind_scale=wind_scale,
        modes=modes,
    )


def _plot_selection(
    path: Path,
    summaries: Sequence[Mapping[str, Any]],
    selected_step: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([value["fine_tune_step"] for value in summaries])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(
        steps,
        [value["day360_mid_modewise_ratio"] for value in summaries],
        "o-",
        label="PHIHYD mid",
    )
    axes[0].plot(
        steps,
        [value["day360_bottom_modewise_ratio"] for value in summaries],
        "s-",
        label="PHIHYD bottom",
    )
    axes[0].axhline(4.0, color="black", linestyle="--", label="factor-four gate")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Fine-tune step (0 = source)")
    axes[0].set_ylabel("Day-360 median modewise energy ratio")
    axes[0].legend()

    for field in ("surface_speed", "sst", "phihyd_surface"):
        axes[1].plot(
            steps,
            [
                value["primary_10_to_90_rmse_ratio_to_persistence"][field]
                for value in summaries
            ],
            "o-",
            label=field.replace("_", " "),
        )
    axes[1].axhline(1.0, color="black", linestyle="--", label="persistence")
    axes[1].set_xlabel("Fine-tune step (0 = source)")
    axes[1].set_ylabel("10--90-day mean RMSE ratio")
    axes[1].legend()
    for axis in axes:
        axis.axvline(selected_step, color="#d95f02", alpha=0.7)
        axis.grid(alpha=0.25)
    figure.suptitle("Training-only deep-pressure spectral fine-tune selection")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_rollout(
    path: Path,
    lead_days: np.ndarray,
    ratios: np.ndarray,
    steps: np.ndarray,
    selected_step: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_index = int(np.flatnonzero(steps == selected_step)[0])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, level, label in zip(
        axes,
        (7, 14),
        ("PHIHYD mid (k=7)", "PHIHYD bottom (k=14)"),
        strict=True,
    ):
        axis.plot(
            lead_days,
            ratios[0, :, level],
            color="black",
            linestyle="--",
            label="source",
        )
        axis.plot(
            lead_days,
            ratios[selected_index, :, level],
            color="#d95f02",
            label=f"selected step {selected_step}",
        )
        axis.axhline(4.0, color="gray", linestyle=":", label="factor-four gate")
        axis.set_yscale("log")
        axis.set_xlabel("Lead day")
        axis.set_ylabel("Median model/truth modewise energy ratio")
        axis.set_title(label)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Correctly aligned split-1 pressure spectra through day 360")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def preflight(
    contract_path: str | Path,
) -> dict[str, Any]:
    """Verify the frozen contract and sources without training."""

    contract, resolved, digest = load_contract(contract_path)
    dataset = Path(contract["sources"]["dataset"]["path"]).resolve()
    _, _, _, attribution_contract = _verify_artifacts(contract, dataset)
    group = zarr.open_consolidated(str(dataset), mode="r")
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    records = attribution.training_records(attribution_contract, split)
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "selection_records": int(records.shape[0]),
        "source_optimizer_step": SOURCE_OPTIMIZER_STEP,
        "validation_state_opened": False,
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }


def run(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Fine-tune, select on split 1, and publish immutable evidence."""

    if torch is None or DataLoader is None:  # pragma: no cover
        raise RuntimeError("deep-pressure spectral fine-tune requires PyTorch")
    require_model_a_runtime()
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    dataset = Path(contract["sources"]["dataset"]["path"]).resolve()
    scratch = Path(contract["output"]["scratch_directory"]).resolve()
    project = Path(contract["output"]["project_directory"]).resolve()
    scratch_temporary = scratch.with_name(scratch.name + ".tmp")
    project_temporary = project.with_name(project.name + ".tmp")
    if any(
        value.exists()
        for value in (scratch, project, scratch_temporary, project_temporary)
    ):
        raise FileExistsError("refusing to overwrite fine-tune output")
    (
        source_checkpoint,
        source_normalization,
        source_payload,
        attribution_contract,
    ) = _verify_artifacts(contract, dataset)

    fine_tune = contract["fine_tune"]
    seed_everything(int(fine_tune["batch_order_seed"]))
    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    _, _, wet_array, _, wind_mean, wind_scale = _normalizers(group)
    wet_array = np.asarray(wet_array, dtype=bool)
    with np.load(source_normalization, allow_pickle=False) as values:
        point_mean = np.asarray(values["pointwise_mean"], dtype=np.float32)
        point_scale = np.asarray(values["pointwise_scale"], dtype=np.float32)
    training_records = records_for_rollout_split(
        pair_codes,
        1,
        rollout_steps=3,
    )
    training_dataset = ModelCAnomalyRolloutDataset(
        dataset,
        training_records,
        point_mean,
        point_scale,
        rollout_steps=3,
    )
    loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            training_dataset,
            int(fine_tune["batch_size"]),
            int(fine_tune["batch_order_seed"]),
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    loss_config = model_c_loss_config("v1")
    if loss_contract_sha256(loss_config) != MODEL_C_LOSS_V1_CONTRACT_SHA256:
        raise DeepPressureSpectralRegularizationError("loss-v1 changed")
    architecture = ModelCSuccessorArchitecture(**source_payload["architecture"])
    model = build_successor(architecture).to(device)
    model.load_state_dict(source_payload["model_state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(fine_tune["learning_rate"]),
        betas=tuple(float(value) for value in fine_tune["adam_betas"]),
        weight_decay=float(fine_tune["weight_decay"]),
    )
    wet = torch.from_numpy(wet_array.astype(np.float32))[None, None].to(device)
    boundary_array = western_boundary_mask(
        wet_array,
        loss_config.western_boundary_width,
    )
    boundary = torch.from_numpy(boundary_array.astype(np.float32))[
        None, None
    ].to(device)
    increment_scale = torch.from_numpy(
        pointwise_increment_scale(group, pair_codes, point_scale)
    ).to(device)
    point_mean_tensor = torch.from_numpy(point_mean).to(device)
    point_scale_tensor = torch.from_numpy(point_scale).to(device)
    maximum_steps = int(fine_tune["maximum_steps"])
    checkpoint_steps = tuple(
        int(value) for value in fine_tune["checkpoint_steps"]
    )
    pressure_weight = float(contract["objective"]["weight"])

    scratch_temporary.parent.mkdir(parents=True, exist_ok=True)
    project_temporary.parent.mkdir(parents=True, exist_ok=True)
    scratch_temporary.mkdir()
    project_temporary.mkdir()
    checkpoint_directory = scratch_temporary / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir()
    iterator = iter(loader)
    totals = {name: 0.0 for name in TRAINING_TERMS}
    samples = 0
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for fine_tune_step in range(1, maximum_steps + 1):
        try:
            features, futures = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            features, futures = next(iterator)
        features = features.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        futures = futures.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        model.train()
        predictions = direct_state_unroll(model, features, wet, 3)
        base_terms = model_c_loss_terms(
            predictions,
            futures,
            features[:, :STATE_CHANNEL_COUNT],
            wet,
            boundary,
            increment_scale,
            loss_config,
        )
        pressure_tail = deep_pressure_high_mode_loss(
            predictions,
            futures,
            point_mean_tensor,
            point_scale_tensor,
            wet,
            truth_floor_fraction=float(
                contract["objective"]["truth_tail_floor_fraction_of_total"]
            ),
        )
        total = base_terms["total"] + pressure_weight * pressure_tail
        values = {
            **base_terms,
            "total": total,
            "base_total": base_terms["total"],
            "deep_pressure_tail": pressure_tail,
        }
        if not all(
            bool(torch.isfinite(values[name]).item())
            for name in TRAINING_TERMS
        ):
            raise DeepPressureSpectralRegularizationError(
                "fine-tune objective became non-finite"
            )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        batch = int(features.shape[0])
        for name in TRAINING_TERMS:
            totals[name] += float(values[name].detach().cpu()) * batch
        samples += batch
        if fine_tune_step not in checkpoint_steps:
            continue
        training_window = {
            name: totals[name] / samples for name in TRAINING_TERMS
        }
        optimizer_step = SOURCE_OPTIMIZER_STEP + fine_tune_step
        history_record = {
            "fine_tune_step": fine_tune_step,
            "optimizer_step": optimizer_step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_window": training_window,
        }
        history.append(history_record)
        checkpoint_path = _checkpoint_path(
            checkpoint_directory,
            fine_tune_step,
        )
        payload = {
            "version": VERSION,
            "purpose": "deep_pressure_spectral_regularized_anomaly_direct_checkpoint",
            "source_optimizer_step": SOURCE_OPTIMIZER_STEP,
            "fine_tune_step": fine_tune_step,
            "optimizer_step": optimizer_step,
            "architecture": architecture.to_dict(),
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "base_loss_contract": loss_contract(loss_config),
            "base_loss_contract_sha256": loss_contract_sha256(loss_config),
            "spectral_regularization_objective": contract["objective"],
            "fine_tune_contract": fine_tune,
            "training_history_record": history_record,
            "model_state_dict": _checkpoint_state_dict(model),
        }
        torch.save(payload, checkpoint_path)
        checkpoints.append(
            {
                "fine_tune_step": fine_tune_step,
                "optimizer_step": optimizer_step,
                "checkpoint": str(
                    scratch
                    / CHECKPOINT_DIRECTORY
                    / checkpoint_path.name
                ),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
            }
        )
        totals = {name: 0.0 for name in TRAINING_TERMS}
        samples = 0
    if len(checkpoints) != len(checkpoint_steps):
        raise DeepPressureSpectralRegularizationError(
            "not every contracted fine-tune checkpoint was saved"
        )
    del model, optimizer, loader, training_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    records = attribution.training_records(attribution_contract, split)
    initial = attribution.base._gather_states(state, records, 0)
    raw_static = attribution.base._gather_static(static, records)
    experiments = records[:, 0]
    modes = np.arange(1, 31, dtype=np.float32)
    evaluated = [
        _evaluate_checkpoint(
            source_checkpoint,
            SOURCE_OPTIMIZER_STEP,
            source_normalization,
            device=device,
            initial=initial,
            raw_static=raw_static,
            experiments=experiments,
            state=state,
            records=records,
            wet=wet_array,
            wind_mean=float(wind_mean),
            wind_scale=float(wind_scale),
            modes=modes,
        )
    ]
    for record in checkpoints:
        evaluated.append(
            _evaluate_checkpoint(
                scratch_temporary
                / CHECKPOINT_DIRECTORY
                / Path(record["checkpoint"]).name,
                int(record["optimizer_step"]),
                source_normalization,
                device=device,
                initial=initial,
                raw_static=raw_static,
                experiments=experiments,
                state=state,
                records=records,
                wet=wet_array,
                wind_mean=float(wind_mean),
                wind_scale=float(wind_scale),
                modes=modes,
            )
        )
    source_summary = summarize_evaluation(evaluated[0])
    source_primary = source_summary[
        "primary_10_to_90_rmse_ratio_to_persistence"
    ]
    summaries: list[dict[str, Any]] = []
    for index, value in enumerate(evaluated):
        summary = summarize_evaluation(
            value,
            source_primary_ratios=source_primary,
            selection=contract["selection"],
        )
        summary["fine_tune_step"] = (
            0 if index == 0 else checkpoints[index - 1]["fine_tune_step"]
        )
        summaries.append(summary)
    decision = select_candidate(summaries)
    selected_step = int(decision["selected_fine_tune_step"])
    selected_path = scratch_temporary / CHECKPOINT_NAME
    if selected_step == 0:
        shutil.copy2(source_checkpoint, selected_path)
    else:
        shutil.copy2(
            _checkpoint_path(checkpoint_directory, selected_step),
            selected_path,
        )

    steps = np.asarray(
        [value["fine_tune_step"] for value in summaries],
        dtype=np.int32,
    )
    lead_days = np.arange(10, 361, 10, dtype=np.int16)
    ratios = np.stack([value["ratio"] for value in evaluated]).astype(
        np.float32
    )
    integrated = np.stack(
        [value["integrated"] for value in evaluated]
    ).astype(np.float32)
    tail_model = np.stack(
        [value["tail_model"] for value in evaluated]
    ).astype(np.float32)
    tail_truth = np.stack(
        [value["tail_truth"] for value in evaluated]
    ).astype(np.float32)
    model_rmse = np.stack(
        [value["model_rmse"] for value in evaluated]
    ).astype(np.float32)
    persistence_rmse = np.asarray(
        evaluated[0]["persistence_rmse"],
        dtype=np.float32,
    )
    arrays_path = scratch_temporary / ARRAYS_NAME
    np.savez_compressed(
        arrays_path,
        fine_tune_steps=steps,
        optimizer_steps=steps + SOURCE_OPTIMIZER_STEP,
        lead_days=lead_days,
        pressure_levels=np.arange(15, dtype=np.int16),
        spectral_modes=modes,
        frozen_median_modewise_ratio=ratios,
        integrated_energy_ratio=integrated,
        tail_model_fraction=tail_model,
        tail_truth_fraction=tail_truth,
        primary_model_rmse=model_rmse,
        primary_persistence_rmse=persistence_rmse,
        selection_records=records.astype(np.int32),
    )

    report = {
        "status": "complete",
        "version": VERSION,
        "purpose": "single_candidate_training_only_deep_pressure_spectral_regularization",
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset),
        "dataset_metadata_sha256": _file_sha256(dataset / ".zmetadata"),
        "device": str(device),
        "source": {
            "seed": 20260724,
            "optimizer_step": SOURCE_OPTIMIZER_STEP,
            "checkpoint": str(source_checkpoint),
            "checkpoint_sha256": _file_sha256(source_checkpoint),
            "normalization": str(source_normalization),
            "normalization_sha256": _file_sha256(source_normalization),
        },
        "architecture": architecture.to_dict(),
        "base_loss_contract_sha256": MODEL_C_LOSS_V1_CONTRACT_SHA256,
        "objective": contract["objective"],
        "fine_tune": fine_tune,
        "training_history": history,
        "checkpoints": checkpoints,
        "evaluation_summaries": summaries,
        "selection_decision": decision,
        "selected_checkpoint": str(scratch / CHECKPOINT_NAME),
        "selected_checkpoint_sha256": _file_sha256(selected_path),
        "arrays": str(scratch / ARRAYS_NAME),
        "arrays_sha256": _file_sha256(arrays_path),
        "selection_records_sha256": attribution.base._array_sha256(records),
        "read_contract": contract["read_contract"],
        "validation_state_opened": False,
        "inference_state_opened": False,
        "intermediate_wind_state_opened": False,
        "response_or_adjoint_state_opened": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    report["content_sha256"] = _json_sha256(report)
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (scratch_temporary / REPORT_NAME).write_text(report_text)
    (project_temporary / REPORT_NAME).write_text(report_text)
    shutil.copy2(arrays_path, project_temporary / ARRAYS_NAME)
    _plot_selection(
        project_temporary / SELECTION_FIGURE,
        summaries,
        selected_step,
    )
    _plot_rollout(
        project_temporary / SPECTRAL_FIGURE,
        lead_days,
        ratios,
        steps,
        selected_step,
    )
    readme = (
        "# Model C deep-pressure spectral regularization\n\n"
        f"Training-only decision: `{decision['status']}`. "
        f"Selected fine-tune step: {selected_step}.  No validation, held-"
        "inference, response, or adjoint state was opened.\n"
    )
    (project_temporary / README_NAME).write_text(readme)
    artifacts = {
        name: _file_sha256(project_temporary / name)
        for name in (
            REPORT_NAME,
            ARRAYS_NAME,
            SELECTION_FIGURE,
            SPECTRAL_FIGURE,
            README_NAME,
        )
    }
    manifest = {
        "status": "complete",
        "version": VERSION,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "artifacts": artifacts,
        "content_sha256": _json_sha256(artifacts),
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }
    (project_temporary / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(scratch_temporary, scratch)
    os.replace(project_temporary, project)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--contract", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--contract", type=Path, required=True)
    run_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.contract)
    else:
        result = run(args.contract, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
