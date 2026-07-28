"""Bounded slow-field pushforward correction for the trajectory-v2 Model C.

This training-only experiment starts from the exact-replay checkpoint at step
14,400.  It preserves the width-128 architecture and every loss-v1 term, then
adds one detached pushforward endpoint loss for SST and surface PHIHYD.  The
endpoint alternates deterministically between days 60 and 90.  Validation,
inference, response, and adjoint states remain sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from .af_a0 import records_for_pair_split
from .af_a0_evaluate import _normalizers
from .af_forward_complete import _training_climatology
from .af_model_a import (
    ChunkAwareBatchSampler,
    _checkpoint_state_dict,
    _normalization_from_group,
    require_model_a_runtime,
    seed_everything,
)
from .af_model_b import (
    _unroll,
    records_for_rollout_split,
    western_boundary_mask,
)
from .af_model_c import (
    MODEL_C_LOSS_V1_CONTRACT_SHA256,
    loss_contract,
    loss_contract_sha256,
    model_c_loss_config,
    model_c_loss_terms,
)
from .af_model_c_checkpoint_replay_audit import (
    _audit_records,
    _array_sha256,
    checkpoint_gate_summary,
    load_checkpoint_replay_contract,
)
from .af_model_c_overfit import AUDIT_TERMS, _device, _file_sha256
from .af_model_c_rollout_diagnosis import lead_curve_summary
from .af_model_c_successor import (
    REFERENCE_DIAGNOSTIC_SEED,
    STATE_CHANNEL_COUNT,
    _one_step_diagnostics,
    _sample_records_by_regime,
    architecture_from_candidate,
    build_successor,
    load_successor_contract,
    training_increment_scale,
)
from .af_model_c_successor_validation import (
    ValidationStepper,
    _evaluate_stepper,
    _method_auc_summary,
)
from .af_pressure import GRAVITY_M_S2, THERMAL_EXPANSION_PER_C

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    Dataset = object  # type: ignore[assignment,misc]


OBJECTIVE_VERSION = "model_c_pushforward_objective_v1"
REPORT_NAME = "model_c_pushforward_objective_report.json"
ARRAYS_NAME = "model_c_pushforward_objective_arrays.npz"
CHECKPOINT_DIRECTORY = "fine_tune_checkpoints"
SELECTED_CHECKPOINT_NAME = "model_c_pushforward_objective_best.pt"
REFERENCE_CANDIDATE = "v2_bireprop_w128_mlp4"
HORIZON_DAYS = 10
ROLLOUT_STEPS = 9
SURFACE_THETA_CHANNEL = 30
ETA_CHANNEL = 45
SURFACE_CENTER_DEPTH_M = 25.0
TRAINING_TERMS = (*AUDIT_TERMS, "pushforward_sst", "pushforward_phihyd_surface")


class ModelCPushforwardObjectiveError(RuntimeError):
    """Raised when the bounded objective experiment violates its contract."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ModelCPushforwardDataset(Dataset):
    """Lazy Model C features plus nine normalized ten-day targets."""

    def __init__(
        self,
        dataset_path: str | Path,
        records: Sequence[tuple[int, int]],
        *,
        horizon_days: int = HORIZON_DAYS,
        rollout_steps: int = ROLLOUT_STEPS,
    ) -> None:
        require_model_a_runtime()
        self.dataset_path = Path(dataset_path).resolve()
        self.records = tuple(
            (int(experiment), int(time_index))
            for experiment, time_index in records
        )
        self.horizon_days = int(horizon_days)
        self.rollout_steps = int(rollout_steps)
        if (
            not self.records
            or self.horizon_days != HORIZON_DAYS
            or self.rollout_steps != ROLLOUT_STEPS
        ):
            raise ValueError(
                "pushforward dataset requires complete ten-day targets through day 90"
            )
        self._group: Any | None = None
        self._state: Any | None = None
        self._static: Any | None = None
        self._open()

    def _open(self) -> None:
        self._group = zarr.open_consolidated(str(self.dataset_path), mode="r")
        self._state = self._group["state"]
        self._static = self._group["static_features"]
        self.mean, self.scale, self.wet, self.wind_mean, self.wind_scale = (
            _normalization_from_group(self._group)
        )
        if (
            self._state.shape[2] != STATE_CHANNEL_COUNT
            or self._static.shape[1] != 5
        ):
            raise ValueError("pushforward dataset channel count changed")
        if any(
            time_index + self.horizon_days * self.rollout_steps
            >= self._state.shape[1]
            for _, time_index in self.records
        ):
            raise ValueError("a pushforward rollout exceeds the trajectory")

    def __len__(self) -> int:
        return len(self.records)

    def _normalise_state(self, value: np.ndarray) -> np.ndarray:
        result = (
            value - self.mean[:, None, None]
        ) / self.scale[:, None, None]
        result[:, ~self.wet] = 0.0
        return np.ascontiguousarray(result, dtype=np.float32)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        experiment, time_index = self.records[index]
        present = self._normalise_state(
            np.asarray(
                self._state[experiment, time_index],
                dtype=np.float32,
            )
        )
        futures = np.stack(
            [
                self._normalise_state(
                    np.asarray(
                        self._state[
                            experiment,
                            time_index + step * self.horizon_days,
                        ],
                        dtype=np.float32,
                    )
                )
                for step in range(1, self.rollout_steps + 1)
            ]
        )
        geometry = np.asarray(
            self._static[experiment],
            dtype=np.float32,
        ).copy()
        geometry[0] = (
            geometry[0] - self.wind_mean
        ) / self.wind_scale
        geometry[0, ~self.wet] = 0.0
        features = np.ascontiguousarray(
            np.concatenate((present, geometry), axis=0),
            dtype=np.float32,
        )
        return torch.from_numpy(features), torch.from_numpy(futures)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_group"] = state["_state"] = state["_static"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open()


def surface_phihyd_error(
    normalized_prediction: Any,
    normalized_target: Any,
    state_scale: Any,
) -> Any:
    """Return differentiable surface-PHIHYD error in m2 s-2.

    At the top tracer center, the MITgcm linear-EOS finite-difference
    reconstruction is ``g*eta - 25*g*alpha*(theta_0 - Tref_0)``.  Tref
    cancels in an error, so only the normalized theta and eta differences are
    needed here.
    """

    if (
        normalized_prediction.shape != normalized_target.shape
        or normalized_prediction.ndim != 4
        or state_scale.shape != (STATE_CHANNEL_COUNT,)
    ):
        raise ValueError("surface PHIHYD error received inconsistent tensors")
    difference = normalized_prediction - normalized_target
    theta_error = (
        difference[:, SURFACE_THETA_CHANNEL]
        * state_scale[SURFACE_THETA_CHANNEL]
    )
    eta_error = difference[:, ETA_CHANNEL] * state_scale[ETA_CHANNEL]
    return (
        GRAVITY_M_S2 * eta_error
        - SURFACE_CENTER_DEPTH_M
        * GRAVITY_M_S2
        * THERMAL_EXPANSION_PER_C
        * theta_error
    )


def slow_field_pushforward_loss(
    prediction: Any,
    target: Any,
    wet: Any,
    state_scale: Any,
    climatology_rmse_scale: Mapping[str, float],
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """Equal SST/PHIHYD masked RMSE normalized by frozen climatology RMSE."""

    if (
        prediction.shape != target.shape
        or prediction.ndim != 4
        or wet.shape != (1, 1, *prediction.shape[-2:])
    ):
        raise ValueError("pushforward loss tensors or wet mask are inconsistent")
    scales = {
        name: float(climatology_rmse_scale[name])
        for name in ("sst", "phihyd_surface")
    }
    if any(not np.isfinite(value) or value <= 0 for value in scales.values()):
        raise ValueError("pushforward climatology scales must be positive")
    mask = wet[:, 0]
    wet_count = mask.sum(dim=(-2, -1)).clamp_min(1.0)
    sst_error = (
        prediction[:, SURFACE_THETA_CHANNEL]
        - target[:, SURFACE_THETA_CHANNEL]
    ) * state_scale[SURFACE_THETA_CHANNEL]
    phihyd_error = surface_phihyd_error(
        prediction,
        target,
        state_scale,
    )

    def normalized_rmse(error: Any, scale: float) -> Any:
        member_mse = (
            error.square() * mask
        ).sum(dim=(-2, -1)) / wet_count
        return torch.sqrt(member_mse + epsilon).mean() / scale

    sst = normalized_rmse(sst_error, scales["sst"])
    phihyd = normalized_rmse(
        phihyd_error,
        scales["phihyd_surface"],
    )
    return {
        "mean": 0.5 * (sst + phihyd),
        "sst": sst,
        "phihyd_surface": phihyd,
    }


def detached_pushforward_endpoint(
    model: Any,
    features: Any,
    wet: Any,
    base_predictions: Any,
    *,
    endpoint_step: int,
) -> Any:
    """Predict one terminal state with detached intermediate model states."""

    if (
        endpoint_step not in (6, 9)
        or base_predictions.ndim != 5
        or base_predictions.shape[1] != 3
    ):
        raise ValueError("pushforward endpoint must be day 60 or day 90")
    geometry = features[:, STATE_CHANNEL_COUNT:]
    current = base_predictions[:, -1].detach()
    with torch.no_grad():
        for _ in range(3, endpoint_step - 1):
            current = (
                current
                + model(torch.cat((current, geometry), dim=1))
            ) * wet
    current = current.detach()
    return (
        current + model(torch.cat((current, geometry), dim=1))
    ) * wet


def select_pushforward_checkpoint(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select a passing checkpoint or retain a diagnostic-only best."""

    ranking = []
    for summary in summaries:
        gate = summary["checkpoint_gate"]
        ranking.append(
            {
                "fine_tune_step": int(summary["fine_tune_step"]),
                "passed": bool(gate["passed"]),
                "selection_key": [
                    float(gate["worst_slow_field_lead_ratio"]),
                    float(gate["worst_primary_rmse_auc_ratio"]),
                    float(
                        summary["ten_day_diagnostic"][
                            "worst_per_regime_group_ratio"
                        ]
                    ),
                    int(summary["fine_tune_step"]),
                ],
            }
        )
    ranking.sort(key=lambda value: tuple(value["selection_key"]))
    if not ranking:
        raise ValueError("pushforward selection needs checkpoint summaries")
    eligible = [value for value in ranking if value["passed"]]
    selected = eligible[0] if eligible else ranking[0]
    return {
        "classification": (
            "training_only_pushforward_gate_passed"
            if eligible
            else "training_only_pushforward_gate_not_yet_passed"
        ),
        "passed": bool(eligible),
        "selected_fine_tune_step": int(selected["fine_tune_step"]),
        "selected_for": (
            "replication_before_fresh_validation"
            if eligible
            else "diagnostic_only_no_validation_authorized"
        ),
        "ranking": ranking,
    }


def load_pushforward_contract(
    path: str | Path,
    *,
    verify_source_files: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the objective contract frozen before pushforward training."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != OBJECTIVE_VERSION:
        raise ValueError(f"expected pushforward contract {OBJECTIVE_VERSION}")
    if (
        contract.get("contract_status")
        != "frozen_after_exact_replay_objective_decision_before_pushforward_metrics"
    ):
        raise ValueError("pushforward objective contract was not frozen")
    objective = contract.get("objective", {})
    if (
        objective.get("base_loss_version") != "v1"
        or tuple(objective.get("pushforward_endpoint_days", ()))
        != (60, 90)
        or objective.get("endpoint_schedule")
        != "alternate_60_and_90_days_by_optimizer_step"
        or float(objective.get("pushforward_weight", -1.0)) != 0.0025
        or tuple(objective.get("slow_fields", ()))
        != ("sst", "phihyd_surface")
        or objective.get("intermediate_gradient")
        != "detach_every_model_state_before_terminal_step"
    ):
        raise ValueError("bounded pushforward objective changed")
    fine_tune = contract.get("fine_tune", {})
    if (
        int(fine_tune.get("source_optimizer_step", -1)) != 14400
        or int(fine_tune.get("maximum_steps", -1)) != 1920
        or tuple(fine_tune.get("checkpoint_steps", ()))
        != (480, 960, 1440, 1920)
        or int(fine_tune.get("batch_size", -1)) != 4
        or float(fine_tune.get("initial_learning_rate", -1.0)) != 0.0001
        or float(fine_tune.get("decay_fraction", -1.0)) != 0.75
        or float(fine_tune.get("decay_factor", -1.0)) != 0.2
        or tuple(fine_tune.get("adam_betas", ())) != (0.9, 0.95)
        or float(fine_tune.get("weight_decay", -1.0)) != 0.00001
    ):
        raise ValueError("bounded fine-tune schedule changed")
    read = contract.get("read_contract", {})
    if (
        read.get("training_pair_code") != 1
        or read.get("training_state_read") is not True
        or any(
            read.get(name) is not False
            for name in (
                "validation_state_read",
                "inference_read",
                "intermediate_wind_read",
                "response_read",
                "adjoint_read",
            )
        )
    ):
        raise ValueError("pushforward read contract changed")
    if verify_source_files:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(f"pushforward source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def _checkpoint_path(directory: Path, step: int) -> Path:
    return directory / f"model_c_pushforward_step_{step:04d}.pt"


def _verify_artifacts(
    contract: Mapping[str, Any],
    dataset: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    Path,
]:
    sources = contract["source_artifacts"]
    if _file_sha256(dataset / ".zmetadata") != sources[
        "dataset_metadata_sha256"
    ]:
        raise ModelCPushforwardObjectiveError("trajectory-v2 metadata changed")
    audit_contract, _, audit_contract_sha = load_checkpoint_replay_contract(
        sources["checkpoint_replay_contract"]
    )
    if audit_contract_sha != sources["checkpoint_replay_contract_sha256"]:
        raise ModelCPushforwardObjectiveError(
            "checkpoint replay contract changed"
        )
    audit_report_path = Path(sources["checkpoint_replay_report"]).resolve()
    audit_arrays_path = Path(sources["checkpoint_replay_arrays"]).resolve()
    source_checkpoint_path = Path(sources["source_checkpoint"]).resolve()
    expected = {
        audit_report_path: sources["checkpoint_replay_report_sha256"],
        audit_arrays_path: sources["checkpoint_replay_arrays_sha256"],
        source_checkpoint_path: sources["source_checkpoint_sha256"],
    }
    for artifact, digest in expected.items():
        if not artifact.is_file() or _file_sha256(artifact) != digest:
            raise ModelCPushforwardObjectiveError(
                f"pushforward source artifact changed: {artifact}"
            )
    audit_report = json.loads(audit_report_path.read_text())
    if (
        audit_report.get("status") != "complete"
        or audit_report.get("audit_decision", {}).get("classification")
        != "objective_correction_required"
        or audit_report.get("audit_decision", {}).get(
            "diagnostic_best_optimizer_step"
        )
        != 14400
        or audit_report.get("exact_replay_verification", {}).get("passed")
        is not True
        or audit_report.get("report_content_sha256")
        != sources["checkpoint_replay_report_content_sha256"]
    ):
        raise ModelCPushforwardObjectiveError(
            "source audit does not authorize objective correction"
        )
    successor_contract, _, _ = load_successor_contract(
        sources["successor_training_contract"]
    )
    return (
        audit_contract,
        successor_contract,
        audit_report_path,
        audit_arrays_path,
        source_checkpoint_path,
    )


def run_pushforward_objective(
    dataset_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run the frozen training-only objective correction and its gate."""

    if torch is None or DataLoader is None:
        raise RuntimeError("pushforward objective requires PyTorch")
    require_model_a_runtime()
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_pushforward_contract(
        contract_path
    )
    dataset = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite pushforward output: {output}"
        )
    (
        audit_contract,
        successor_contract,
        audit_report_path,
        audit_arrays_path,
        source_checkpoint_path,
    ) = _verify_artifacts(contract, dataset)

    seed = int(contract["fine_tune"]["batch_order_seed"])
    seed_everything(seed)
    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    audit_records, audit_times, block_bounds = _audit_records(
        pair_codes,
        snapshot_codes,
        audit_contract,
    )
    training_records = records_for_rollout_split(
        pair_codes,
        1,
        rollout_steps=ROLLOUT_STEPS,
    )
    one_step_records = records_for_pair_split(pair_codes, 1)
    diagnostic_records = _sample_records_by_regime(
        one_step_records,
        count_per_regime=int(
            successor_contract["diagnostics"][
                "checkpoint_records_per_regime"
            ]
        ),
        seed=REFERENCE_DIAGNOSTIC_SEED,
    )

    loss_config = model_c_loss_config("v1")
    if loss_contract_sha256(loss_config) != MODEL_C_LOSS_V1_CONTRACT_SHA256:
        raise ModelCPushforwardObjectiveError("loss-v1 contract changed")
    architecture = architecture_from_candidate(
        successor_contract,
        REFERENCE_CANDIDATE,
    )
    training_dataset = ModelCPushforwardDataset(
        dataset,
        training_records,
    )
    fine_tune = contract["fine_tune"]
    batch_size = int(fine_tune["batch_size"])
    loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            training_dataset,
            batch_size,
            seed,
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    wet = torch.from_numpy(training_dataset.wet.astype(np.float32))[
        None, None
    ].to(device)
    boundary_array = western_boundary_mask(
        training_dataset.wet,
        loss_config.western_boundary_width,
    )
    boundary = torch.from_numpy(boundary_array.astype(np.float32))[
        None, None
    ].to(device)
    increment_values = training_increment_scale(group, pair_codes)
    increment_scale = torch.from_numpy(increment_values).to(device)
    physical_scale = torch.from_numpy(
        training_dataset.scale.astype(np.float32)
    ).to(device)

    source_payload = torch.load(
        source_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if (
        source_payload.get("optimizer_step") != 14400
        or source_payload.get("architecture") != architecture.to_dict()
        or source_payload.get("loss_contract_sha256")
        != MODEL_C_LOSS_V1_CONTRACT_SHA256
    ):
        raise ModelCPushforwardObjectiveError(
            "source checkpoint contract changed"
        )
    model = build_successor(architecture).to(device)
    model.load_state_dict(source_payload["model_state_dict"])
    parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(fine_tune["initial_learning_rate"]),
        betas=tuple(float(value) for value in fine_tune["adam_betas"]),
        weight_decay=float(fine_tune["weight_decay"]),
    )
    maximum_steps = int(fine_tune["maximum_steps"])
    decay_step = int(
        round(maximum_steps * float(fine_tune["decay_fraction"]))
    )
    checkpoint_steps = tuple(
        int(value) for value in fine_tune["checkpoint_steps"]
    )
    endpoint_steps = tuple(
        int(value // HORIZON_DAYS)
        for value in contract["objective"]["pushforward_endpoint_days"]
    )
    pushforward_weight = float(
        contract["objective"]["pushforward_weight"]
    )
    climatology_scales = contract["objective"][
        "climatology_rmse_scales"
    ]

    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    checkpoint_directory = temporary / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir()
    iterator = iter(loader)
    window_totals = {name: 0.0 for name in TRAINING_TERMS}
    window_samples = 0
    history = []
    saved = []

    for step in range(1, maximum_steps + 1):
        if step == decay_step + 1:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= float(
                    fine_tune["decay_factor"]
                )
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
        endpoint_step = endpoint_steps[(step - 1) % len(endpoint_steps)]
        model.train()
        base_predictions = _unroll(
            model,
            features,
            wet,
            loss_config.rollout_steps,
        )
        terms = model_c_loss_terms(
            base_predictions,
            futures[:, : loss_config.rollout_steps],
            features[:, :STATE_CHANNEL_COUNT],
            wet,
            boundary,
            increment_scale,
            loss_config,
        )
        endpoint = detached_pushforward_endpoint(
            model,
            features,
            wet,
            base_predictions,
            endpoint_step=endpoint_step,
        )
        slow = slow_field_pushforward_loss(
            endpoint,
            futures[:, endpoint_step - 1],
            wet,
            physical_scale,
            climatology_scales[str(endpoint_step * HORIZON_DAYS)],
        )
        training_values = {
            **terms,
            "pushforward_sst": slow["sst"],
            "pushforward_phihyd_surface": slow["phihyd_surface"],
        }
        training_values["total"] = (
            terms["total"] + pushforward_weight * slow["mean"]
        )
        if not all(
            bool(torch.isfinite(training_values[name]).item())
            for name in TRAINING_TERMS
        ):
            raise ModelCPushforwardObjectiveError(
                "pushforward training loss became non-finite"
            )
        optimizer.zero_grad(set_to_none=True)
        training_values["total"].backward()
        optimizer.step()
        size = int(features.shape[0])
        for name in TRAINING_TERMS:
            window_totals[name] += (
                float(training_values[name].detach().cpu()) * size
            )
        window_samples += size
        if step not in checkpoint_steps:
            continue
        training_window = {
            name: window_totals[name] / window_samples
            for name in TRAINING_TERMS
        }
        history_record = {
            "fine_tune_step": step,
            "optimizer_learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "training_window": training_window,
        }
        history.append(history_record)
        checkpoint_path = _checkpoint_path(checkpoint_directory, step)
        payload = {
            "version": OBJECTIVE_VERSION,
            "purpose": "bounded_slow_field_pushforward_checkpoint",
            "fine_tune_step": step,
            "source_optimizer_step": 14400,
            "candidate_id": REFERENCE_CANDIDATE,
            "architecture": architecture.to_dict(),
            "parameter_count": parameter_count,
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "base_loss_contract": loss_contract(loss_config),
            "base_loss_contract_sha256": loss_contract_sha256(loss_config),
            "pushforward_objective": contract["objective"],
            "fine_tune_contract": fine_tune,
            "training_history_record": history_record,
            "model_state_dict": _checkpoint_state_dict(model),
        }
        torch.save(payload, checkpoint_path)
        saved.append(
            {
                "fine_tune_step": step,
                "checkpoint": str(
                    output / CHECKPOINT_DIRECTORY / checkpoint_path.name
                ),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
            }
        )
        window_totals = {name: 0.0 for name in TRAINING_TERMS}
        window_samples = 0

    if len(saved) != len(checkpoint_steps):
        raise ModelCPushforwardObjectiveError(
            "fine-tune did not save every checkpoint"
        )

    mean, scale, wet_array, _, wind_mean, wind_scale = _normalizers(group)
    climatology_state, climatology_derived, training_days = (
        _training_climatology(state, snapshot_codes, wet_array)
    )
    source_arrays = np.load(audit_arrays_path)
    if not np.array_equal(source_arrays["records"], audit_records):
        raise ModelCPushforwardObjectiveError(
            "source audit records changed"
        )
    arrays: dict[str, np.ndarray] = {
        "records": audit_records.astype(np.int32),
        "training_times": audit_times.astype(np.int32),
        "lead_days": np.asarray(range(10, 91, 10), dtype=np.int16),
    }
    for prefix in ("persistence", "climatology", "step_14400"):
        for name in source_arrays.files:
            if name.startswith(prefix + "__"):
                arrays[f"source_{prefix}__{name.split('__', 1)[1]}"] = (
                    np.asarray(source_arrays[name])
                )
    baselines = {
        method: {
            name.split("__", 1)[1]: np.asarray(source_arrays[name])
            for name in source_arrays.files
            if name.startswith(method + "__")
        }
        for method in ("persistence", "climatology")
    }

    summaries = []
    for saved_record, history_record in zip(saved, history):
        checkpoint_path = _checkpoint_path(
            checkpoint_directory,
            int(saved_record["fine_tune_step"]),
        )
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        evaluation_model = build_successor(architecture).to(device)
        evaluation_model.load_state_dict(payload["model_state_dict"])
        evaluation_model.eval()
        stepper = ValidationStepper(
            kind="successor",
            model=evaluation_model,
            device=device,
            wet=wet_array,
            mean=mean,
            scale=scale,
            wind_mean=wind_mean,
            wind_scale=wind_scale,
        )
        metrics = _evaluate_stepper(
            stepper,
            state,
            static,
            audit_records,
            climatology_state,
            climatology_derived,
            mean,
            scale,
            batch_size=int(contract["evaluation"]["batch_size"]),
        )
        step = int(saved_record["fine_tune_step"])
        for name, value in metrics.items():
            arrays[f"fine_tune_{step}__{name}"] = np.asarray(value)
        diagnostic = _one_step_diagnostics(
            evaluation_model,
            dataset,
            diagnostic_records,
            batch_size=16,
            device=device,
        )
        curves = lead_curve_summary(
            metrics,
            baselines,
            audit_records,
        )
        gate = checkpoint_gate_summary(
            curves,
            diagnostic,
            metrics,
            audit_contract["checkpoint_gate"],
        )
        summaries.append(
            {
                **saved_record,
                "training_window": history_record["training_window"],
                "ten_day_diagnostic": diagnostic,
                "all_field_auc": _method_auc_summary(metrics),
                "lead_curves": curves,
                "checkpoint_gate": gate,
            }
        )
        del evaluation_model, stepper, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()

    decision = select_pushforward_checkpoint(summaries)
    selected_step = int(decision["selected_fine_tune_step"])
    selected_source = _checkpoint_path(
        checkpoint_directory,
        selected_step,
    )
    selected_payload = torch.load(
        selected_source,
        map_location="cpu",
        weights_only=False,
    )
    selected_path = temporary / SELECTED_CHECKPOINT_NAME
    torch.save(selected_payload, selected_path)
    restored = build_successor(architecture).to(device)
    restored.load_state_dict(selected_payload["model_state_dict"])
    reload_features = training_dataset[0][0][None].to(
        device=device,
        dtype=torch.float32,
    )
    model_for_reference = build_successor(architecture).to(device)
    model_for_reference.load_state_dict(selected_payload["model_state_dict"])
    model_for_reference.eval()
    restored.eval()
    with torch.no_grad():
        reference = _unroll(
            model_for_reference,
            reload_features,
            wet,
            9,
        ).detach().cpu()
        reloaded = _unroll(
            restored,
            reload_features,
            wet,
            9,
        ).detach().cpu()
    reload_exact = bool(torch.equal(reference, reloaded))
    if not reload_exact:
        raise ModelCPushforwardObjectiveError(
            "selected nine-step save/reload was not bitwise exact"
        )

    arrays_path = temporary / ARRAYS_NAME
    np.savez_compressed(arrays_path, **arrays)
    report = {
        "status": "complete",
        "purpose": "training_only_bounded_slow_field_pushforward_correction",
        "version": OBJECTIVE_VERSION,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset),
        "dataset_metadata_sha256": _file_sha256(dataset / ".zmetadata"),
        "device": str(device),
        "candidate_id": REFERENCE_CANDIDATE,
        "architecture": architecture.to_dict(),
        "parameter_count": parameter_count,
        "base_loss_contract": loss_contract(loss_config),
        "base_loss_contract_sha256": loss_contract_sha256(loss_config),
        "pushforward_objective": contract["objective"],
        "fine_tune_contract": fine_tune,
        "source_artifacts": {
            "checkpoint_replay_report": str(audit_report_path),
            "checkpoint_replay_report_sha256": _file_sha256(
                audit_report_path
            ),
            "checkpoint_replay_arrays": str(audit_arrays_path),
            "checkpoint_replay_arrays_sha256": _file_sha256(
                audit_arrays_path
            ),
            "source_checkpoint": str(source_checkpoint_path),
            "source_checkpoint_sha256": _file_sha256(
                source_checkpoint_path
            ),
        },
        "read_contract": contract["read_contract"],
        "counts": {
            "training_nine_step_rollouts": len(training_records),
            "training_one_step_pairs": len(one_step_records),
            "checkpoint_diagnostic_pairs": len(diagnostic_records),
            "evaluation_rollouts": int(audit_records.shape[0]),
            "training_climatology_snapshots_per_regime": training_days,
        },
        "record_contract": {
            "complete_training_blocks": [
                list(value) for value in block_bounds
            ],
            "training_times_sha256": _array_sha256(audit_times),
            "records_sha256": _array_sha256(audit_records),
        },
        "increment_scale": increment_values.tolist(),
        "history": history,
        "checkpoint_summary": summaries,
        "objective_decision": decision,
        "selected_checkpoint": str(output / SELECTED_CHECKPOINT_NAME),
        "selected_checkpoint_sha256": _file_sha256(selected_path),
        "save_reload_nine_step_bitwise_exact": reload_exact,
        "arrays": str(output / ARRAYS_NAME),
        "arrays_sha256": _file_sha256(arrays_path),
        "elapsed_seconds": time.monotonic() - started,
        "validation_state_opened": False,
        "inference_opened": False,
        "intermediate_wind_opened": False,
        "response_or_adjoint_opened": False,
    }
    report["report_content_sha256"] = _json_sha256(report)
    (temporary / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, output)
    return report


def preflight_pushforward_objective(
    dataset_path: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Verify the immutable training-only sources without loading states."""

    contract, resolved, digest = load_pushforward_contract(contract_path)
    dataset = Path(dataset_path).resolve()
    _verify_artifacts(contract, dataset)
    return {
        "status": "ready",
        "version": OBJECTIVE_VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset": str(dataset),
        "source_optimizer_step": contract["fine_tune"][
            "source_optimizer_step"
        ],
        "fine_tune_steps": contract["fine_tune"]["maximum_steps"],
        "pushforward_endpoint_days": contract["objective"][
            "pushforward_endpoint_days"
        ],
        "read_contract": contract["read_contract"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--dataset", type=Path, required=True)
        child.add_argument("--contract", type=Path, required=True)
        if command == "run":
            child.add_argument("--output-dir", type=Path, required=True)
            child.add_argument(
                "--device",
                choices=("auto", "cpu", "cuda"),
                default="auto",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight_pushforward_objective(
            args.dataset,
            args.contract,
        )
    else:
        result = run_pushforward_objective(
            args.dataset,
            args.contract,
            args.output_dir,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
