"""Training-only Model C successor diagnosis on trajectory dataset version 2.

The fresh validation and inference chronologies remain sealed.  Phase 1 compares
the selected version-1 architecture with an isolated Channel-MLP expansion.
Phase 2 restores the Bire latent proportions only after phase 1 is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from .af_a0 import records_for_pair_split
from .af_data import STATIC_FEATURES, STATE_CHANNELS
from .af_model_a import (
    ChunkAwareBatchSampler,
    ModelAPairDataset,
    _checkpoint_state_dict,
    require_model_a_runtime,
    seed_everything,
)
from .af_model_b import (
    ModelBRolloutDataset,
    _unroll,
    records_for_rollout_split,
    rollout_start_indices,
    western_boundary_mask,
)
from .af_model_c import (
    GROUP_SLICES,
    MODEL_C_LOSS_V1_CONTRACT_SHA256,
    loss_contract,
    loss_contract_sha256,
    model_c_loss_config,
    model_c_loss_terms,
)
from .af_model_c_overfit import AUDIT_TERMS, _device, _file_sha256
from .af_model_c_validation import one_step_metrics

try:  # Keep documentation-only imports usable without the optional ML stack.
    import torch
    from neuralop.models import FNO
    from torch import nn
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    FNO = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


SUCCESSOR_VERSION = "model_c_successor_training_v1"
REPORT_NAME = "model_c_successor_training_report.json"
CHECKPOINT_NAME = "model_c_successor_training_best.pt"
STATE_CHANNEL_COUNT = len(STATE_CHANNELS)
INPUT_CHANNEL_COUNT = STATE_CHANNEL_COUNT + len(STATIC_FEATURES)
HORIZON_DAYS = 10
LONG_ROLLOUT_STEPS = 18
LONG_LEADS = (30, 90, 180)
REFERENCE_DIAGNOSTIC_SEED = 20260723


class ModelCSuccessorError(RuntimeError):
    """Raised when a successor run violates its frozen training-only contract."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class ModelCSuccessorArchitecture:
    """One dense FNO in the bounded trajectory-v2 successor study."""

    in_channels: int = INPUT_CHANNEL_COUNT
    out_channels: int = STATE_CHANNEL_COUNT
    n_modes: tuple[int, int] = (24, 16)
    hidden_channels: int = 64
    n_layers: int = 4
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    channel_mlp_expansion: float = 0.5
    domain_padding: float = 0.1
    positional_embedding: str = "grid"
    use_channel_mlp: bool = True
    local_kernel_size: int = 3
    fno_block_precision: str = "full"
    factorization: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "n_modes", tuple(int(value) for value in self.n_modes)
        )
        if self.in_channels != INPUT_CHANNEL_COUNT or self.out_channels != 46:
            raise ValueError("successor channel contract is 46 state + 5 static -> 46")
        if self.n_modes != (24, 16) or self.n_layers != 4:
            raise ValueError("successor training fixes modes=(24,16) and four layers")
        if self.hidden_channels not in (64, 128):
            raise ValueError("successor width must be 64 or 128")
        if (
            self.lifting_channel_ratio != 2
            or self.projection_channel_ratio != 2
            or self.channel_mlp_expansion not in (0.5, 4.0)
        ):
            raise ValueError("successor lifting/projection/Channel-MLP ratios changed")
        if self.hidden_channels == 128 and self.channel_mlp_expansion != 4.0:
            raise ValueError("the width-128 successor is the complete Bire-proportion test")
        if (
            self.domain_padding != 0.1
            or self.positional_embedding != "grid"
            or not self.use_channel_mlp
            or self.local_kernel_size != 3
        ):
            raise ValueError("successor padding, embedding, and local branch are fixed")
        if self.fno_block_precision != "full" or self.factorization is not None:
            raise ValueError("successor candidates are dense float32 FNOs")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_modes"] = list(self.n_modes)
        return result


if nn is not None:

    class ModelCSuccessorResidualFNO(nn.Module):
        """Residual FNO with explicit lifting, projection, and Channel-MLP widths."""

        def __init__(self, architecture: ModelCSuccessorArchitecture) -> None:
            super().__init__()
            require_model_a_runtime()
            self.architecture = architecture
            self.fno = FNO(
                n_modes=architecture.n_modes,
                in_channels=architecture.in_channels,
                out_channels=architecture.out_channels,
                hidden_channels=architecture.hidden_channels,
                n_layers=architecture.n_layers,
                lifting_channel_ratio=architecture.lifting_channel_ratio,
                projection_channel_ratio=architecture.projection_channel_ratio,
                positional_embedding=architecture.positional_embedding,
                use_channel_mlp=architecture.use_channel_mlp,
                channel_mlp_expansion=architecture.channel_mlp_expansion,
                domain_padding=architecture.domain_padding,
                fno_block_precision=architecture.fno_block_precision,
                factorization=architecture.factorization,
            )
            self.local = nn.Conv2d(
                architecture.in_channels,
                architecture.out_channels,
                kernel_size=architecture.local_kernel_size,
                padding=architecture.local_kernel_size // 2,
            )

        def forward(self, features: Any) -> Any:
            if (
                features.ndim != 4
                or features.shape[1] != self.architecture.in_channels
            ):
                raise ValueError(
                    "successor expects "
                    f"N,{self.architecture.in_channels},Y,X features"
                )
            return self.fno(features) + self.local(features)

else:  # pragma: no cover - only relevant without the optional ML environment
    ModelCSuccessorResidualFNO = None  # type: ignore[assignment]


def build_successor(architecture: ModelCSuccessorArchitecture) -> Any:
    """Build one explicitly proportioned Model C successor."""

    require_model_a_runtime()
    if ModelCSuccessorResidualFNO is None:  # pragma: no cover
        raise RuntimeError("Model C successor requires the project ML environment")
    return ModelCSuccessorResidualFNO(architecture)


def load_successor_contract(
    path: str | Path,
) -> tuple[dict[str, Any], Path, str]:
    """Load the contract frozen before any trajectory-v2 training metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != SUCCESSOR_VERSION:
        raise ValueError(f"expected successor contract {SUCCESSOR_VERSION}")
    if (
        contract.get("contract_status")
        != "frozen_after_v2_coverage_before_successor_training_metrics"
    ):
        raise ValueError("Model C successor contract was not frozen")
    read = contract.get("read_contract", {})
    if (
        read.get("training_pair_code") != 1
        or any(
            read.get(name) is not False
            for name in (
                "validation_read",
                "inference_read",
                "intermediate_wind_read",
                "response_read",
                "adjoint_read",
            )
        )
    ):
        raise ValueError("Model C successor does not preserve sealed data")
    if contract.get("loss") != {
        "contract_sha256": MODEL_C_LOSS_V1_CONTRACT_SHA256,
        "version": "v1",
    }:
        raise ValueError("Model C successor must preserve loss v1")
    expected_ids = (
        "v2_control_w64_mlp05",
        "v2_mix_w64_mlp4",
        "v2_bireprop_w128_mlp4",
    )
    candidates = contract.get("candidates", [])
    if tuple(candidate.get("candidate_id") for candidate in candidates) != expected_ids:
        raise ValueError("Model C successor candidate set changed")
    for candidate in candidates:
        architecture_from_candidate(contract, candidate["candidate_id"])
    return contract, resolved, _file_sha256(resolved)


def architecture_from_candidate(
    contract: Mapping[str, Any], candidate_id: str
) -> ModelCSuccessorArchitecture:
    """Construct one architecture from the frozen common and varying fields."""

    matches = [
        candidate
        for candidate in contract.get("candidates", [])
        if candidate.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown successor candidate {candidate_id!r}")
    common = dict(contract["architecture_constants"])
    candidate = matches[0]
    common.update(
        {
            "hidden_channels": candidate["hidden_channels"],
            "lifting_channel_ratio": candidate["lifting_channel_ratio"],
            "projection_channel_ratio": candidate["projection_channel_ratio"],
            "channel_mlp_expansion": candidate["channel_mlp_expansion"],
        }
    )
    return ModelCSuccessorArchitecture(**common)


def resolve_candidate_id(
    contract_path: str | Path, *, phase: int, array_index: int
) -> str:
    """Resolve one phase-local scheduler index without reading scientific data."""

    contract, _, _ = load_successor_contract(contract_path)
    candidates = [
        candidate
        for candidate in contract["candidates"]
        if int(candidate["phase"]) == phase
    ]
    if not 0 <= array_index < len(candidates):
        raise ValueError(
            f"array index {array_index} is outside phase {phase}'s "
            f"{len(candidates)} candidates"
        )
    return str(candidates[array_index]["candidate_id"])


def _verify_sources(
    contract: Mapping[str, Any],
    dataset_path: Path,
    quality_path: Path,
    coverage_path: Path,
    v1_decision_path: Path,
) -> None:
    hashes = contract["source_hashes"]
    paths = {
        "dataset_metadata_sha256": dataset_path / ".zmetadata",
        "dataset_quality_report_sha256": quality_path,
        "coverage_report_sha256": coverage_path,
        "model_c_v1_rejection_decision_sha256": v1_decision_path,
    }
    for name, path in paths.items():
        if not path.is_file() or _file_sha256(path) != hashes[name]:
            raise ModelCSuccessorError(f"successor source changed: {path}")
    quality = json.loads(quality_path.read_text())
    coverage = json.loads(coverage_path.read_text())
    if (
        quality.get("status") != "valid"
        or quality.get("inference_state_metrics_read") is not False
        or coverage.get("assessment", {}).get("target_met") is not True
        or coverage.get("read_contract", {}).get("validation_read") is not False
        or coverage.get("read_contract", {}).get("inference_read") is not False
    ):
        raise ModelCSuccessorError("successor source reports do not pass their gates")


def _contiguous_runs(indices: np.ndarray) -> tuple[tuple[int, int], ...]:
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("training indices must be a nonempty vector")
    cuts = np.flatnonzero(np.diff(indices) != 1) + 1
    pieces = np.split(indices, cuts)
    return tuple((int(piece[0]), int(piece[-1]) + 1) for piece in pieces)


def training_increment_scale(
    group: Any, pair_codes: np.ndarray, *, chunk_days: int = 32
) -> np.ndarray:
    """Compute the loss-v1 per-channel scale from split-1 pairs only."""

    if chunk_days <= 0:
        raise ValueError("increment-scale chunk size must be positive")
    state = group["state"]
    scale = np.asarray(group["state_scale"][:], dtype=np.float64)
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    runs = _contiguous_runs(np.flatnonzero(pair_codes == 1))
    squares = np.zeros(STATE_CHANNEL_COUNT, dtype=np.float64)
    count = 0
    for experiment in range(state.shape[0]):
        for run_start, run_stop in runs:
            for start in range(run_start, run_stop, chunk_days):
                stop = min(start + chunk_days, run_stop)
                present = np.asarray(
                    state[experiment, start:stop], dtype=np.float32
                )
                future = np.asarray(
                    state[
                        experiment,
                        start + HORIZON_DAYS : stop + HORIZON_DAYS,
                    ],
                    dtype=np.float32,
                )
                increment = (future - present) / scale[None, :, None, None]
                squares += np.square(
                    increment[:, :, wet].astype(np.float64)
                ).sum(axis=(0, 2))
                count += increment.shape[0] * int(wet.sum())
    result = np.sqrt(squares / count).astype(np.float32)
    if (
        count <= 0
        or result.shape != (STATE_CHANNEL_COUNT,)
        or not np.all(np.isfinite(result))
        or np.any(result <= 0)
    ):
        raise ModelCSuccessorError("invalid trajectory-v2 increment scale")
    return result


def _sample_records_by_regime(
    records: Sequence[tuple[int, int]], *, count_per_regime: int, seed: int
) -> tuple[tuple[int, int], ...]:
    selected: list[tuple[int, int]] = []
    for experiment in range(3):
        candidates = np.asarray(
            [time_index for regime, time_index in records if regime == experiment],
            dtype=np.int64,
        )
        if candidates.size < count_per_regime:
            raise ValueError("not enough training-only diagnostic records")
        indices = np.sort(
            np.random.default_rng(seed + experiment).choice(
                candidates.size, size=count_per_regime, replace=False
            )
        )
        selected.extend((experiment, int(candidates[index])) for index in indices)
    return tuple(selected)


def _complete_long_training_records(
    pair_codes: np.ndarray,
    snapshot_codes: np.ndarray,
    *,
    count_per_regime: int,
    seed: int,
) -> tuple[tuple[int, int], ...]:
    starts = rollout_start_indices(
        pair_codes, 1, rollout_steps=LONG_ROLLOUT_STEPS
    )
    valid = [
        start
        for start in starts
        if all(
            snapshot_codes[start + step * HORIZON_DAYS] == 1
            for step in range(LONG_ROLLOUT_STEPS + 1)
        )
    ]
    records = tuple(
        (experiment, start) for experiment in range(3) for start in valid
    )
    return _sample_records_by_regime(
        records, count_per_regime=count_per_regime, seed=seed
    )


def _one_step_diagnostics(
    model: Any,
    dataset_path: Path,
    records: Sequence[tuple[int, int]],
    *,
    batch_size: int,
    device: Any,
) -> dict[str, Any]:
    """Score aggregate and per-regime training-only ten-day skill."""

    def score(selected: Sequence[tuple[int, int]]) -> dict[str, Any]:
        dataset = ModelAPairDataset(dataset_path, selected)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        mask = torch.from_numpy(dataset.wet.astype(np.float32))[None, None].to(
            device
        )
        return one_step_metrics(
            model, loader, dataset, device=device, metric_mask=mask
        )

    by_regime = {
        f"S{experiment}": score(
            [record for record in records if record[0] == experiment]
        )
        for experiment in range(3)
    }
    aggregate = score(records)
    worst = max(
        float(metrics["worst_group_ratio"]) for metrics in by_regime.values()
    )
    all_pass = all(
        bool(metrics["all_groups_beat_persistence"])
        for metrics in by_regime.values()
    )
    return {
        "aggregate": aggregate,
        "by_regime": by_regime,
        "worst_per_regime_group_ratio": worst,
        "every_regime_and_group_beats_persistence": all_pass,
    }


def _normalise_raw(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray, wet: np.ndarray
) -> np.ndarray:
    result = (values - mean[None, :, None, None]) / scale[
        None, :, None, None
    ]
    result[:, :, ~wet] = 0.0
    return np.ascontiguousarray(result, dtype=np.float32)


def training_rollout_stability(
    model: Any,
    dataset_path: Path,
    records: Sequence[tuple[int, int]],
    *,
    batch_size: int,
    device: Any,
) -> dict[str, Any]:
    """Measure decorrelation-aware amplitude stability on split-1 starts."""

    dataset = ModelAPairDataset(dataset_path, records)
    group = zarr.open_consolidated(str(dataset_path), mode="r")
    state = group["state"]
    wet_array = dataset.wet
    wet = torch.from_numpy(wet_array.astype(np.float32))[None, None].to(device)
    leads = {
        lead: {
            name: {
                "prediction_squares": 0.0,
                "target_squares": 0.0,
                "values": 0,
                "max_abs_prediction": 0.0,
            }
            for name in GROUP_SLICES
        }
        for lead in LONG_LEADS
    }
    finite = True
    divergence_step: int | None = None
    model.eval()
    with torch.no_grad():
        for start_index in range(0, len(records), batch_size):
            batch_records = records[start_index : start_index + batch_size]
            examples = [
                dataset[index]
                for index in range(
                    start_index, min(start_index + batch_size, len(records))
                )
            ]
            features = torch.stack([example[0] for example in examples]).to(
                device=device, dtype=torch.float32
            )
            current = features[:, :STATE_CHANNEL_COUNT]
            geometry = features[:, STATE_CHANNEL_COUNT:]
            for step in range(1, LONG_ROLLOUT_STEPS + 1):
                current = (
                    current + model(torch.cat((current, geometry), dim=1))
                ) * wet
                if not bool(torch.isfinite(current).all().item()):
                    finite = False
                    divergence_step = (
                        step
                        if divergence_step is None
                        else min(divergence_step, step)
                    )
                    break
                lead = step * HORIZON_DAYS
                if lead not in leads:
                    continue
                raw_target = np.stack(
                    [
                        np.asarray(
                            state[experiment, time_index + lead],
                            dtype=np.float32,
                        )
                        for experiment, time_index in batch_records
                    ]
                )
                target = torch.from_numpy(
                    _normalise_raw(
                        raw_target,
                        dataset.mean,
                        dataset.scale,
                        wet_array,
                    )
                ).to(device)
                for name, channels in GROUP_SLICES.items():
                    predicted_group = current[:, channels] * wet
                    target_group = target[:, channels] * wet
                    values = leads[lead][name]
                    values["prediction_squares"] += float(
                        predicted_group.square().sum().cpu()
                    )
                    values["target_squares"] += float(
                        target_group.square().sum().cpu()
                    )
                    values["values"] += (
                        len(batch_records)
                        * (channels.stop - channels.start)
                        * int(wet_array.sum())
                    )
                    values["max_abs_prediction"] = max(
                        float(values["max_abs_prediction"]),
                        float(predicted_group.abs().max().cpu()),
                    )

    result: dict[str, Any] = {}
    for lead in LONG_LEADS:
        result[str(lead)] = {}
        for name, raw in leads[lead].items():
            count = int(raw["values"])
            if (
                count == 0
                or raw["target_squares"] <= 0
                or not finite
            ):
                prediction_rms = math.inf
                target_rms = math.nan
                amplitude_ratio = math.inf
            else:
                prediction_rms = math.sqrt(raw["prediction_squares"] / count)
                target_rms = math.sqrt(raw["target_squares"] / count)
                amplitude_ratio = prediction_rms / target_rms
            result[str(lead)][name] = {
                "prediction_normalized_rms": prediction_rms,
                "target_normalized_rms": target_rms,
                "amplitude_ratio": amplitude_ratio,
                "max_abs_normalized_prediction": float(
                    raw["max_abs_prediction"]
                ),
            }
    return {
        "record_count": len(records),
        "finite": finite,
        "divergence_step": divergence_step,
        "lead_metrics": result,
    }


def apply_training_gate(
    one_step: Mapping[str, Any],
    stability: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    reload_exact: bool,
) -> dict[str, Any]:
    """Apply the prospective training-only advancement gate."""

    lower, upper = (
        float(value) for value in gate["long_rollout_amplitude_ratio_bounds"]
    )
    amplitude_checks: dict[str, dict[str, bool]] = {}
    for lead in gate["long_rollout_gate_leads_days"]:
        metrics = stability["lead_metrics"][str(lead)]
        amplitude_checks[str(lead)] = {
            name: lower <= float(values["amplitude_ratio"]) <= upper
            for name, values in metrics.items()
        }
    one_step_pass = bool(
        one_step["every_regime_and_group_beats_persistence"]
    )
    finite_pass = bool(stability["finite"])
    amplitude_pass = all(
        passed
        for by_group in amplitude_checks.values()
        for passed in by_group.values()
    )
    passed = one_step_pass and finite_pass and amplitude_pass and reload_exact
    return {
        "status": (
            "eligible_for_fresh_v2_validation"
            if passed
            else "training_only_gate_rejected"
        ),
        "passed": passed,
        "ten_day_every_regime_and_group_passed": one_step_pass,
        "long_rollout_finite": finite_pass,
        "long_rollout_amplitude_checks": amplitude_checks,
        "long_rollout_amplitude_passed": amplitude_pass,
        "three_step_reload_bitwise_exact": reload_exact,
    }


def _candidate_spec(
    contract: Mapping[str, Any], candidate_id: str
) -> Mapping[str, Any]:
    return next(
        candidate
        for candidate in contract["candidates"]
        if candidate["candidate_id"] == candidate_id
    )


def run_successor_candidate(
    dataset_path: str | Path,
    quality_report_path: str | Path,
    coverage_report_path: str | Path,
    v1_decision_path: str | Path,
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    candidate_id: str,
    seed: int = REFERENCE_DIAGNOSTIC_SEED,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train and diagnose one immutable successor candidate using split 1 only."""

    require_model_a_runtime()
    contract, resolved_contract, contract_sha = load_successor_contract(
        contract_path
    )
    architecture = architecture_from_candidate(contract, candidate_id)
    candidate = _candidate_spec(contract, candidate_id)
    dataset_path = Path(dataset_path).resolve()
    quality_path = Path(quality_report_path).resolve()
    coverage_path = Path(coverage_report_path).resolve()
    v1_decision_path = Path(v1_decision_path).resolve()
    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite successor output: {output}")
    _verify_sources(
        contract,
        dataset_path,
        quality_path,
        coverage_path,
        v1_decision_path,
    )
    loss_config = model_c_loss_config("v1")
    if loss_contract_sha256(loss_config) != MODEL_C_LOSS_V1_CONTRACT_SHA256:
        raise ModelCSuccessorError("loss-v1 code contract changed")

    seed = int(seed)
    if seed < 0:
        raise ValueError("successor training seed must be nonnegative")
    seed_everything(seed)
    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset_path), mode="r")
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    training_records = records_for_rollout_split(pair_codes, 1)
    one_step_records = records_for_pair_split(pair_codes, 1)
    diagnostic_records = _sample_records_by_regime(
        one_step_records,
        count_per_regime=int(
            contract["diagnostics"]["checkpoint_records_per_regime"]
        ),
        seed=REFERENCE_DIAGNOSTIC_SEED,
    )
    long_records = _complete_long_training_records(
        pair_codes,
        snapshot_codes,
        count_per_regime=int(
            contract["diagnostics"]["long_rollout_records_per_regime"]
        ),
        seed=REFERENCE_DIAGNOSTIC_SEED + 100,
    )
    increment_values = training_increment_scale(group, pair_codes)

    training_dataset = ModelBRolloutDataset(dataset_path, training_records)
    optimizer_contract = contract["optimizer"]
    batch_size = int(optimizer_contract["batch_size"])
    train_loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            training_dataset, batch_size, seed
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    wet = torch.from_numpy(training_dataset.wet.astype(np.float32))[
        None, None
    ].to(device)
    boundary_array = western_boundary_mask(
        training_dataset.wet, loss_config.western_boundary_width
    )
    boundary = torch.from_numpy(boundary_array.astype(np.float32))[
        None, None
    ].to(device)
    increment_scale = torch.from_numpy(increment_values).to(device)
    model = build_successor(architecture).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_contract["initial_learning_rate"]),
        betas=tuple(float(value) for value in optimizer_contract["adam_betas"]),
        weight_decay=float(optimizer_contract["weight_decay"]),
    )
    maximum_steps = int(optimizer_contract["maximum_steps"])
    decay_step = int(
        round(maximum_steps * float(optimizer_contract["decay_fraction"]))
    )
    checkpoint_steps = tuple(
        sorted(
            {
                int(round(maximum_steps * float(fraction)))
                for fraction in contract["checkpoint_fractions"]
            }
        )
    )
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, int] | None = None
    best_state: dict[str, Any] | None = None
    best_record: dict[str, Any] | None = None
    window_totals = {name: 0.0 for name in AUDIT_TERMS}
    window_samples = 0
    iterator = iter(train_loader)
    started = time.monotonic()

    for step in range(1, maximum_steps + 1):
        if step == decay_step + 1:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= float(
                    optimizer_contract["decay_factor"]
                )
        try:
            features, futures = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            features, futures = next(iterator)
        features = features.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        futures = futures.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        model.train()
        predictions = _unroll(model, features, wet, loss_config.rollout_steps)
        terms = model_c_loss_terms(
            predictions,
            futures,
            features[:, :STATE_CHANNEL_COUNT],
            wet,
            boundary,
            increment_scale,
            loss_config,
        )
        if not all(
            bool(torch.isfinite(terms[name]).item()) for name in AUDIT_TERMS
        ):
            raise ModelCSuccessorError("successor training loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        optimizer.step()
        size = int(features.shape[0])
        for name in AUDIT_TERMS:
            window_totals[name] += float(terms[name].detach().cpu()) * size
        window_samples += size
        if step not in checkpoint_steps:
            continue
        training_window = {
            name: window_totals[name] / window_samples for name in AUDIT_TERMS
        }
        diagnostic = _one_step_diagnostics(
            model,
            dataset_path,
            diagnostic_records,
            batch_size=16,
            device=device,
        )
        key = (
            float(diagnostic["worst_per_regime_group_ratio"]),
            float(training_window["total"]),
            step,
        )
        record = {
            "optimizer_step": step,
            "optimizer_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_window": training_window,
            "training_only_ten_day_diagnostic": diagnostic,
            "selection_key": list(key),
        }
        history.append(record)
        if best_key is None or key < best_key:
            best_key = key
            best_state = _checkpoint_state_dict(model)
            best_record = record
        window_totals = {name: 0.0 for name in AUDIT_TERMS}
        window_samples = 0

    if best_state is None or best_record is None:
        raise ModelCSuccessorError("successor selected no checkpoint")
    model.load_state_dict(best_state)
    full_training = _one_step_diagnostics(
        model,
        dataset_path,
        one_step_records,
        batch_size=16,
        device=device,
    )
    stability = training_rollout_stability(
        model,
        dataset_path,
        long_records,
        batch_size=4,
        device=device,
    )

    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    checkpoint_path = temporary / CHECKPOINT_NAME
    reload_dataset = ModelAPairDataset(dataset_path, diagnostic_records[:1])
    reload_features = reload_dataset[0][0][None].to(
        device=device, dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        reference = _unroll(model, reload_features, wet, 3).detach().cpu()
    payload = {
        "model_class": "ModelCSuccessorResidualFNO",
        "version": SUCCESSOR_VERSION,
        "candidate_id": candidate_id,
        "candidate": candidate,
        "architecture": architecture.to_dict(),
        "parameter_count": parameter_count,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset_path),
        "dataset_metadata_sha256": _file_sha256(dataset_path / ".zmetadata"),
        "loss_contract": loss_contract(loss_config),
        "loss_contract_sha256": loss_contract_sha256(loss_config),
        "optimizer_contract": optimizer_contract,
        "seed": seed,
        "runtime_source_sha256": _file_sha256(Path(__file__).resolve()),
        "selected_checkpoint": best_record,
        "increment_scale": increment_values.tolist(),
        "increment_scale_sha256": _array_sha256(increment_values),
        "model_state_dict": _checkpoint_state_dict(model),
    }
    torch.save(payload, checkpoint_path)
    restored = build_successor(architecture).to(device)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored.load_state_dict(loaded["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        reloaded = _unroll(restored, reload_features, wet, 3).detach().cpu()
    reload_exact = bool(torch.equal(reference, reloaded))
    gate = apply_training_gate(
        full_training,
        stability,
        contract["training_gate"],
        reload_exact=reload_exact,
    )
    checkpoint_sha = _file_sha256(checkpoint_path)
    report = {
        "status": "complete",
        "purpose": "training_only_model_c_trajectory_v2_successor_diagnosis",
        "version": SUCCESSOR_VERSION,
        "candidate_id": candidate_id,
        "candidate": candidate,
        "architecture": architecture.to_dict(),
        "parameter_count": parameter_count,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "dataset": str(dataset_path),
        "dataset_metadata_sha256": _file_sha256(dataset_path / ".zmetadata"),
        "quality_report": str(quality_path),
        "quality_report_sha256": _file_sha256(quality_path),
        "coverage_report": str(coverage_path),
        "coverage_report_sha256": _file_sha256(coverage_path),
        "v1_rejection_decision": str(v1_decision_path),
        "v1_rejection_decision_sha256": _file_sha256(v1_decision_path),
        "read_contract": {
            "training_pair_code": 1,
            "validation_read": False,
            "inference_read": False,
            "intermediate_wind_read": False,
            "response_read": False,
            "adjoint_read": False,
        },
        "loss_contract": loss_contract(loss_config),
        "loss_contract_sha256": loss_contract_sha256(loss_config),
        "optimizer_contract": optimizer_contract,
        "optimizer_decay_step": decay_step,
        "seed": seed,
        "runtime_source_sha256": _file_sha256(Path(__file__).resolve()),
        "counts": {
            "training_rollouts": len(training_records),
            "training_one_step_pairs": len(one_step_records),
            "checkpoint_diagnostic_pairs": len(diagnostic_records),
            "long_rollout_starts": len(long_records),
        },
        "increment_scale": increment_values.tolist(),
        "increment_scale_sha256": _array_sha256(increment_values),
        "diagnostic_records_sha256": _json_sha256(diagnostic_records),
        "long_rollout_records_sha256": _json_sha256(long_records),
        "selected_checkpoint": best_record,
        "history": history,
        "full_training_ten_day": full_training,
        "training_rollout_stability": stability,
        "training_gate": gate,
        "save_reload_three_step_bitwise_exact": reload_exact,
        "checkpoint": str(output / CHECKPOINT_NAME),
        "checkpoint_sha256": checkpoint_sha,
        "elapsed_seconds": time.monotonic() - started,
    }
    report["report_content_sha256"] = _json_sha256(report)
    (temporary / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, output)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser("resolve")
    resolve.add_argument("--contract", type=Path, required=True)
    resolve.add_argument("--phase", type=int, required=True)
    resolve.add_argument("--array-index", type=int, required=True)
    run = commands.add_parser("run")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--quality-report", type=Path, required=True)
    run.add_argument("--coverage-report", type=Path, required=True)
    run.add_argument("--v1-decision", type=Path, required=True)
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--candidate-id", required=True)
    run.add_argument("--seed", type=int, default=REFERENCE_DIAGNOSTIC_SEED)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "resolve":
        print(
            resolve_candidate_id(
                args.contract,
                phase=args.phase,
                array_index=args.array_index,
            )
        )
        return 0
    result = run_successor_candidate(
        args.dataset,
        args.quality_report,
        args.coverage_report,
        args.v1_decision,
        args.contract,
        args.output_dir,
        candidate_id=args.candidate_id,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
