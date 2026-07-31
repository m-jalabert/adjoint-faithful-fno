"""Evaluation-only recovery for the completed three-layer/no-padding training."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import zarr

from .af_a0_evaluate import _normalizers
from . import (
    af_model_c_anomaly_direct_bire_regularization_controls as regularization,
)
from . import af_model_c_anomaly_direct_three_layer_no_padding as control
from . import (
    af_model_c_anomaly_direct_training_spectral_attribution_v2 as attribution,
)
from .af_model_c_anomaly_direct_deep_pressure_spectral_regularization import (
    summarize_evaluation,
)
from .af_model_c_overfit import _device, _file_sha256

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


VERSION = "model_c_anomaly_direct_three_layer_no_padding_recovery_v2"
REPORT_NAME = control.REPORT_NAME
ARRAYS_NAME = control.ARRAYS_NAME
FIGURE_NAME = control.FIGURE_NAME
README_NAME = control.README_NAME
MANIFEST_NAME = control.MANIFEST_NAME
BEST_NAME = control.BEST_NAME
CONTROL_ID = control.CONTROL_ID


class ThreeLayerNoPaddingRecoveryError(RuntimeError):
    """Raised when failed-job recovery provenance changes."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    read = contract.get("read_contract", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status")
        != "frozen_after_complete_training_before_any_control_metric"
        or contract.get("failed_job", {}).get("job_id") != 304526
        or contract.get("recovery", {}).get("retraining_steps") != 0
        or tuple(contract.get("checkpoint_steps", ()))
        != control.CHECKPOINT_STEPS
        or read.get("training_state") is not True
        or any(
            read.get(name) is not False
            for name in (
                "validation_state",
                "inference_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
                "long_term_state",
            )
        )
    ):
        raise ValueError("three-layer/no-padding recovery contract changed")
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ThreeLayerNoPaddingRecoveryError(
                    f"recovery implementation source changed: {source}"
                )
    return contract, resolved, _file_sha256(resolved)


def _validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    step: int,
    source_contract: Mapping[str, Any],
    source_contract_path: Path,
    source_contract_sha: str,
) -> None:
    if (
        payload.get("version") != control.VERSION
        or payload.get("optimizer_step") != step
        or payload.get("fine_tune_step") != step
        or payload.get("architecture") != source_contract["architecture"]
        or payload.get("contract") != str(source_contract_path)
        or payload.get("contract_sha256") != source_contract_sha
        or payload.get("base_loss_contract_sha256")
        != source_contract["training"]["base_loss_contract_sha256"]
        or payload.get("arm")
        != {
            "arm_id": CONTROL_ID,
            "pointwise_layer_norm": False,
            "channel_mlp_dropout": 0.0,
        }
        or payload.get("training_history_record", {}).get("optimizer_step")
        != step
        or "model_state_dict" not in payload
    ):
        raise ThreeLayerNoPaddingRecoveryError(
            f"checkpoint payload changed at step {step}"
        )


def _verify_sources(
    contract: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    Path,
    str,
    Path,
    dict[str, Any],
    list[dict[str, Any]],
]:
    source_record = contract["source_contract"]
    source_contract, source_path, source_sha = control._load_contract(
        source_record["path"]
    )
    if source_sha != source_record["sha256"]:
        raise ThreeLayerNoPaddingRecoveryError("source contract changed")
    log = contract["failed_job"]["log"]
    log_path = Path(log["path"]).resolve()
    if not log_path.is_file() or _file_sha256(log_path) != log["sha256"]:
        raise ThreeLayerNoPaddingRecoveryError("failed-job log changed")
    log_text = log_path.read_text()
    if (
        "ValueError: successor training fixes modes=(24,16) and four layers"
        not in log_text
        or "model_c_three_layer_no_padding_step_15360.pt" in log_text
    ):
        raise ThreeLayerNoPaddingRecoveryError(
            "failed-job terminal signature changed"
        )

    dataset = Path(source_contract["sources"]["dataset"]["path"]).resolve()
    normalization, attribution_contract = control._verify_artifacts(
        source_contract,
        dataset,
    )
    scratch_tmp = Path(contract["output"]["scratch_temporary"]).resolve()
    scratch_final = Path(contract["output"]["scratch_final"]).resolve()
    project_tmp = Path(contract["output"]["project_temporary"]).resolve()
    project_final = Path(contract["output"]["project_final"]).resolve()
    if (
        not scratch_tmp.is_dir()
        or scratch_final.exists()
        or not project_tmp.is_dir()
        or any(project_tmp.iterdir())
        or project_final.exists()
    ):
        raise ThreeLayerNoPaddingRecoveryError(
            "failed-job temporary output state changed"
        )

    checkpoints = []
    for record in contract["checkpoints"]:
        step = int(record["optimizer_step"])
        path = Path(record["path"]).resolve()
        if (
            step not in control.CHECKPOINT_STEPS
            or not path.is_file()
            or _file_sha256(path) != record["sha256"]
        ):
            raise ThreeLayerNoPaddingRecoveryError(
                f"checkpoint source changed at step {step}"
            )
        if torch is None:  # pragma: no cover
            raise RuntimeError("checkpoint recovery requires PyTorch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _validate_checkpoint_payload(
            payload,
            step=step,
            source_contract=source_contract,
            source_contract_path=source_path,
            source_contract_sha=source_sha,
        )
        checkpoints.append(
            {
                **record,
                "path": path,
                "payload": payload,
            }
        )
    if tuple(value["optimizer_step"] for value in checkpoints) != (
        control.CHECKPOINT_STEPS
    ):
        raise ThreeLayerNoPaddingRecoveryError("checkpoint order changed")
    return (
        source_contract,
        source_path,
        source_sha,
        normalization,
        attribution_contract,
        checkpoints,
    )


@contextlib.contextmanager
def _patched_attribution_architecture() -> Iterator[None]:
    original = attribution.ModelCSuccessorArchitecture
    attribution.ModelCSuccessorArchitecture = (
        control.ThreeLayerNoPaddingArchitecture
    )
    try:
        yield
    finally:
        attribution.ModelCSuccessorArchitecture = original


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify the seven completed checkpoints and sealed read contract."""

    contract, resolved, digest = load_contract(contract_path)
    (
        source_contract,
        _,
        source_sha,
        _,
        attribution_contract,
        checkpoints,
    ) = _verify_sources(contract)
    dataset = Path(source_contract["sources"]["dataset"]["path"]).resolve()
    group = zarr.open_consolidated(str(dataset), mode="r")
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    records = attribution.training_records(attribution_contract, split)
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "source_contract_sha256": source_sha,
        "checkpoint_steps": [
            int(value["optimizer_step"]) for value in checkpoints
        ],
        "selection_records": int(records.shape[0]),
        "retraining_steps": 0,
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
        "long_term_state_opened": False,
    }


def run(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate completed checkpoints and atomically publish the v1 package."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("checkpoint recovery requires PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    (
        source_contract,
        source_contract_path,
        source_contract_sha,
        normalization,
        attribution_contract,
        checkpoints,
    ) = _verify_sources(contract)
    dataset = Path(source_contract["sources"]["dataset"]["path"]).resolve()
    scratch_tmp = Path(contract["output"]["scratch_temporary"]).resolve()
    scratch_final = Path(contract["output"]["scratch_final"]).resolve()
    project_tmp = Path(contract["output"]["project_temporary"]).resolve()
    project_final = Path(contract["output"]["project_final"]).resolve()

    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    _, _, wet, _, wind_mean, wind_scale = _normalizers(group)
    wet = np.asarray(wet, dtype=bool)
    records = attribution.training_records(attribution_contract, split)
    initial = attribution.base._gather_states(state, records, 0)
    raw_static = attribution.base._gather_static(static, records)
    common = {
        "device": device,
        "initial": initial,
        "raw_static": raw_static,
        "experiments": records[:, 0],
        "state": state,
        "records": records,
        "wet": wet,
        "wind_mean": float(wind_mean),
        "wind_scale": float(wind_scale),
        "modes": np.arange(1, 31, dtype=np.float32),
    }
    source = regularization._source_evaluation(source_contract["sources"])
    source_summary = summarize_evaluation(source)
    source_primary = source_summary[
        "primary_10_to_90_rmse_ratio_to_persistence"
    ]
    evaluated = []
    summaries = []
    arm = control.ArchitectureControl()
    with _patched_attribution_architecture():
        for record in checkpoints:
            value = regularization._evaluate_checkpoint(
                record["path"],
                int(record["optimizer_step"]),
                normalization,
                arm,
                **common,
            )
            evaluated.append(value)
            summary = summarize_evaluation(
                value,
                source_primary_ratios=source_primary,
                selection=source_contract["selection"],
            )
            summary["fine_tune_step"] = int(record["optimizer_step"])
            summaries.append(summary)
    decision = control.select_control_checkpoint(summaries)
    selected_step = int(decision["selected_optimizer_step"])
    selected = next(
        value for value in checkpoints if value["optimizer_step"] == selected_step
    )
    shutil.copy2(selected["path"], scratch_tmp / BEST_NAME)

    arrays_path = scratch_tmp / ARRAYS_NAME
    np.savez_compressed(
        arrays_path,
        optimizer_steps=np.asarray(
            [value["optimizer_step"] for value in summaries],
            dtype=np.int32,
        ),
        lead_days=np.arange(10, 361, 10, dtype=np.int16),
        frozen_median_modewise_ratio=np.stack(
            [value["ratio"] for value in evaluated]
        ).astype(np.float32),
        integrated_energy_ratio=np.stack(
            [value["integrated"] for value in evaluated]
        ).astype(np.float32),
        primary_model_rmse=np.stack(
            [value["model_rmse"] for value in evaluated]
        ).astype(np.float32),
        primary_persistence_rmse=np.asarray(
            source["persistence_rmse"],
            dtype=np.float32,
        ),
        source_frozen_median_modewise_ratio=np.asarray(
            source["ratio"],
            dtype=np.float32,
        ),
        selection_records=records.astype(np.int32),
    )
    checkpoint_records = [
        {
            "optimizer_step": int(value["optimizer_step"]),
            "fine_tune_step": int(value["optimizer_step"]),
            "checkpoint": str(
                scratch_final
                / control.CHECKPOINT_DIRECTORY
                / value["path"].name
            ),
            "checkpoint_sha256": value["sha256"],
        }
        for value in checkpoints
    ]
    report = {
        "status": "complete",
        "version": control.VERSION,
        "recovery_version": VERSION,
        "recovery_contract": str(resolved_contract),
        "recovery_contract_sha256": contract_sha,
        "failed_job": contract["failed_job"],
        "retraining_steps": 0,
        "arm": {
            "arm_id": CONTROL_ID,
            "pointwise_layer_norm": False,
            "channel_mlp_dropout": 0.0,
        },
        "contract": str(source_contract_path),
        "contract_sha256": source_contract_sha,
        "architecture": source_contract["architecture"],
        "parameter_count": int(
            sum(
                value.numel()
                for value in selected["payload"]["model_state_dict"].values()
            )
        ),
        "training_history": [
            value["payload"]["training_history_record"] for value in checkpoints
        ],
        "checkpoints": checkpoint_records,
        "source_summary": source_summary,
        "evaluation_summaries": summaries,
        "selection_decision": decision,
        "best_diagnostic_checkpoint": str(scratch_final / BEST_NAME),
        "best_diagnostic_checkpoint_sha256": selected["sha256"],
        "arrays": str(scratch_final / ARRAYS_NAME),
        "arrays_sha256": _file_sha256(arrays_path),
        "read_contract": contract["read_contract"],
        "validation_state_opened": False,
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
        "long_term_state_opened": False,
        "recovery_elapsed_seconds": time.monotonic() - started,
    }
    report["content_sha256"] = _json_sha256(report)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (scratch_tmp / REPORT_NAME).write_text(rendered)
    (project_tmp / REPORT_NAME).write_text(rendered)
    shutil.copy2(arrays_path, project_tmp / ARRAYS_NAME)
    regularization._plot(
        project_tmp / FIGURE_NAME,
        [{**source_summary, "fine_tune_step": 0}, *summaries],
        selected_step,
        arm,
    )
    (project_tmp / README_NAME).write_text(
        "# Model C three-layer/no-padding architecture control\n\n"
        f"Decision: `{decision['status']}`. Evaluation-only recovery from "
        "completed job-304526 checkpoints used zero retraining steps and only "
        "the frozen split-1 training-state audit; later archives stayed sealed.\n"
    )
    artifacts = {
        name: _file_sha256(project_tmp / name)
        for name in (REPORT_NAME, ARRAYS_NAME, FIGURE_NAME, README_NAME)
    }
    manifest = {
        "status": "complete",
        "version": control.VERSION,
        "recovery_version": VERSION,
        "contract_sha256": source_contract_sha,
        "recovery_contract_sha256": contract_sha,
        "artifacts": artifacts,
        "content_sha256": _json_sha256(artifacts),
        "retraining_steps": 0,
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
        "long_term_state_opened": False,
    }
    (project_tmp / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(scratch_tmp, scratch_final)
    os.replace(project_tmp, project_final)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        if command == "run":
            child.add_argument(
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
