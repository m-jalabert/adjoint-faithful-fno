"""Bire-style Figure 4 for the projected rollout-conditioned Model C checkpoint."""

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

from ..af_a0_evaluate import _normalizers
from ..af_data import STATE_CHANNELS
from ..af_forward_complete import _training_climatology
from ..af_model_c_bire_figures import (
    FIGURE_4_NAME,
    FIELDS,
    LEAD_DAYS,
    _array_sha256,
    _evaluate_curves,
    _metric_summary,
    _plot_figure4,
    _style,
    complete_figure_starts,
    select_ensemble_starts,
)
from ..af_model_c_overfit import _file_sha256
from ..af_model_c_rollout_conditioned_loss_v3 import (
    ProjectedIncrementModel,
    _forcing_signatures,
)
from ..af_model_c_slow_field_bias_projection import wet_area_weights
from ..af_model_c_successor_validation import (
    _load_successor_stepper,
    load_validation_contract,
)

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


VERSION = "model_c_rollout_conditioned_loss_v3_bire_figure4_v1"
REPORT_NAME = "model_c_rollout_conditioned_loss_v3_bire_figure4_report.json"
ARRAYS_NAME = "model_c_rollout_conditioned_loss_v3_bire_figure4_arrays.npz"
MANIFEST_NAME = "model_c_bire_figure4_dt10_rmse_0_200_days.manifest.json"
SELECTED_STEP = 960
EXPECTED_STARTS = (
    6335,
    6353,
    6330,
    6361,
    6358,
    6308,
    6313,
    6346,
    6324,
    6323,
    6319,
    6325,
    6355,
    6366,
    6351,
)


class CurrentModelBireFigure4Error(RuntimeError):
    """Raised when the current-model Figure-4 contract is violated."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_current_figure4_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the source-locked current-model Figure-4 contract."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    protocol = contract.get("protocol", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status")
        != "frozen_before_current_model_100_to_200_day_validation_metrics"
        or tuple(protocol.get("lead_days", ())) != LEAD_DAYS
        or tuple(protocol.get("fields", ())) != FIELDS
        or int(protocol.get("member_count", -1)) != len(EXPECTED_STARTS)
        or tuple(protocol.get("start_draw_order", ())) != EXPECTED_STARTS
        or int(protocol.get("selected_loss_v3_step", -1)) != SELECTED_STEP
        or protocol.get("apply_loss_v3_projection_every_call") is not True
    ):
        raise ValueError("current-model Figure-4 protocol changed")
    read = contract.get("read_contract", {})
    if (
        read.get("training_state_for_climatology") is not True
        or read.get("fresh_validation_state") is not True
        or any(
            read.get(name) is not False
            for name in (
                "inference_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
            )
        )
    ):
        raise ValueError("current-model Figure-4 read contract changed")
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(f"current-model Figure-4 source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def _verified_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or _file_sha256(path) != expected_sha256:
        raise CurrentModelBireFigure4Error(f"source artifact changed: {path}")
    return json.loads(path.read_text())


def _verify_external_sources(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    source = contract["source_artifacts"]
    dataset = Path(source["dataset"]).resolve()
    if (
        not dataset.is_dir()
        or _file_sha256(dataset / ".zmetadata")
        != source["dataset_metadata_sha256"]
    ):
        raise CurrentModelBireFigure4Error("trajectory-v2 dataset changed")
    quality = _verified_json(
        Path(source["quality_report"]).resolve(),
        source["quality_report_sha256"],
    )
    validation_report = _verified_json(
        Path(source["fresh_validation_report"]).resolve(),
        source["fresh_validation_report_sha256"],
    )
    loss_v3_report = _verified_json(
        Path(source["loss_v3_report"]).resolve(),
        source["loss_v3_report_sha256"],
    )
    for report, expected_content in (
        (validation_report, source["fresh_validation_report_content_sha256"]),
        (loss_v3_report, source["loss_v3_report_content_sha256"]),
    ):
        content = dict(report)
        observed_content = content.pop("report_content_sha256", None)
        if observed_content != expected_content or observed_content != _json_sha256(content):
            raise CurrentModelBireFigure4Error("source report content hash changed")
    checkpoint = Path(source["checkpoint"]).resolve()
    if (
        quality.get("status") != "valid"
        or validation_report.get("validation_gate", {}).get("status")
        != "scientifically_rejected_fresh_v2_validation"
        or loss_v3_report.get("status") != "complete"
        or loss_v3_report.get("validation_state_opened") is not False
        or loss_v3_report.get("selected_checkpoint_sha256")
        != source["checkpoint_sha256"]
        or not checkpoint.is_file()
        or _file_sha256(checkpoint) != source["checkpoint_sha256"]
    ):
        raise CurrentModelBireFigure4Error("source evidence is not valid and sealed")
    validation_contract, _, validation_contract_sha = load_validation_contract(
        source["successor_validation_contract"]
    )
    if validation_contract_sha != source["successor_validation_contract_sha256"]:
        raise CurrentModelBireFigure4Error("successor validation contract changed")
    return validation_report, loss_v3_report, validation_contract


def _output_paths(contract: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    output_contract = contract["output_contract"]
    scratch = Path(output_contract["scratch_output"]).resolve()
    project = Path(output_contract["project_output"]).resolve()
    return scratch, project / FIGURE_4_NAME, project / MANIFEST_NAME


def preflight_current_figure4(
    contract_path: str | Path,
) -> dict[str, Any]:
    """Verify every source and output without reading state arrays."""

    if torch is None:
        raise RuntimeError("current-model Figure 4 requires PyTorch")
    contract, resolved, contract_sha = load_current_figure4_contract(contract_path)
    _, loss_v3_report, validation_contract = _verify_external_sources(contract)
    source = contract["source_artifacts"]
    payload = torch.load(
        Path(source["checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    if (
        payload.get("architecture") != validation_contract["architecture"]
        or int(payload.get("loss_v3_fine_tune_step", -1)) != SELECTED_STEP
        or payload.get("contract_sha256") != loss_v3_report["contract_sha256"]
        or np.asarray(payload.get("temperature_target_normalized")).shape != (3, 15)
    ):
        raise CurrentModelBireFigure4Error("loss-v3 checkpoint payload changed")
    scratch, figure, manifest = _output_paths(contract)
    temporary = scratch.with_name(scratch.name + ".tmp")
    project_temporary = (
        figure.with_name(figure.name + ".tmp"),
        manifest.with_name(manifest.name + ".tmp"),
    )
    if any(
        path.exists()
        for path in (scratch, temporary, figure, manifest, *project_temporary)
    ):
        raise FileExistsError("current-model Figure-4 output already exists")
    return {
        "status": "preflight_passed",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": contract_sha,
        "checkpoint": source["checkpoint"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "selected_loss_v3_step": SELECTED_STEP,
        "start_draw_order": list(EXPECTED_STARTS),
        "lead_days": list(LEAD_DAYS),
        "loss_v3_projection_every_call": True,
        "fresh_validation_state_opened": False,
        "inference_state_opened": False,
    }


def evaluate_current_figure4(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate the current projected map and publish the requested Figure 4."""

    if torch is None:
        raise RuntimeError("current-model Figure 4 requires PyTorch")
    started = time.monotonic()
    preflight = preflight_current_figure4(contract_path)
    contract, resolved_contract, contract_sha = load_current_figure4_contract(
        contract_path
    )
    validation_report, loss_v3_report, validation_contract = _verify_external_sources(
        contract
    )
    source = contract["source_artifacts"]
    dataset = Path(source["dataset"]).resolve()
    scratch, project_figure, project_manifest = _output_paths(contract)
    temporary = scratch.with_name(scratch.name + ".tmp")
    temporary_project_figure = project_figure.with_name(project_figure.name + ".tmp")
    temporary_project_manifest = project_manifest.with_name(
        project_manifest.name + ".tmp"
    )
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA Figure-4 evaluation requested without a GPU")
    device = torch.device(device_name)

    group = zarr.open_consolidated(str(dataset), mode="r")
    if tuple(group.attrs["state_channels"]) != STATE_CHANNELS:
        raise CurrentModelBireFigure4Error("trajectory-v2 channels changed")
    state = group["state"]
    static = group["static_features"]
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    snapshot_codes = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    candidates = complete_figure_starts(pair_codes, snapshot_codes)
    starts = select_ensemble_starts(candidates, count=15, seed=20260727)
    protocol = contract["protocol"]
    if (
        candidates.size != int(protocol["candidate_count"])
        or int(candidates[0]) != int(protocol["candidate_bounds"][0])
        or int(candidates[-1]) != int(protocol["candidate_bounds"][1])
        or _array_sha256(candidates) != protocol["candidate_times_sha256"]
        or tuple(int(value) for value in starts) != EXPECTED_STARTS
        or _array_sha256(starts) != protocol["start_draw_order_sha256"]
    ):
        raise CurrentModelBireFigure4Error("validation start selection changed")

    mean, scale, wet, _, wind_mean, wind_scale = _normalizers(group)
    climatology_state, climatology_derived, training_days = _training_climatology(
        state,
        snapshot_codes,
        wet,
    )
    if training_days != int(protocol["training_climatology_snapshots"]):
        raise CurrentModelBireFigure4Error("training climatology count changed")
    stepper, payload = _load_successor_stepper(
        Path(source["checkpoint"]),
        device,
        wet,
        mean,
        scale,
        wind_mean,
        wind_scale,
        validation_contract["architecture"],
    )
    if int(payload.get("loss_v3_fine_tune_step", -1)) != SELECTED_STEP:
        raise CurrentModelBireFigure4Error("selected loss-v3 step changed")
    area_weights_array = wet_area_weights(
        np.asarray(group["latitude_deg"][:], dtype=np.float64),
        wet,
    )
    area_weights = torch.from_numpy(
        area_weights_array[None, None].astype(np.float32)
    ).to(device)
    wet_tensor = torch.from_numpy(wet[None, None].astype(np.float32)).to(device)
    temperature_target = torch.as_tensor(
        payload["temperature_target_normalized"],
        dtype=torch.float32,
        device=device,
    )
    wind_signatures = torch.from_numpy(
        _forcing_signatures(
            static,
            wind_mean=wind_mean,
            wind_scale=wind_scale,
            wet=wet,
        )
    ).to(device)
    stepper.model = ProjectedIncrementModel(
        stepper.model,
        area_weights,
        wet_tensor,
        temperature_target,
        wind_signatures,
    )
    stepper.model.eval()
    arrays = _evaluate_curves(
        stepper,
        state,
        static,
        starts,
        climatology_state,
        climatology_derived,
        wet,
        EXPECTED_STARTS[0],
    )
    metrics = _metric_summary(arrays)

    temporary.parent.mkdir(parents=True, exist_ok=True)
    project_figure.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    try:
        arrays_path = temporary / ARRAYS_NAME
        np.savez_compressed(arrays_path, **arrays)
        _style()
        _plot_figure4(temporary, arrays)
        figure_path = temporary / FIGURE_4_NAME
        report = {
            "version": VERSION,
            "status": "complete",
            "purpose": "descriptive_bire_figure4_for_projected_loss_v3_step_960",
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "dataset": str(dataset),
            "dataset_metadata_sha256": source["dataset_metadata_sha256"],
            "fresh_validation_report": source["fresh_validation_report"],
            "fresh_validation_report_sha256": source[
                "fresh_validation_report_sha256"
            ],
            "fresh_validation_report_content_sha256": validation_report[
                "report_content_sha256"
            ],
            "loss_v3_report": source["loss_v3_report"],
            "loss_v3_report_sha256": source["loss_v3_report_sha256"],
            "loss_v3_report_content_sha256": loss_v3_report[
                "report_content_sha256"
            ],
            "checkpoint": source["checkpoint"],
            "checkpoint_sha256": source["checkpoint_sha256"],
            "selected_loss_v3_step": SELECTED_STEP,
            "projection_application": "every_model_call",
            "protocol": {
                "regime": "S2",
                "start_draw_order": list(EXPECTED_STARTS),
                "lead_days": list(LEAD_DAYS),
                "fields": list(FIELDS),
                "member_summary": "mean_p10_p90_over_15_members",
                "persistence": "repeat_each_member_initial_condition",
                "climatology": "S2_pointwise_mean_over_split1_training_snapshots",
            },
            "metrics": metrics,
            "arrays": str(scratch / ARRAYS_NAME),
            "arrays_sha256": _file_sha256(arrays_path),
            "figure": str(project_figure),
            "figure_sha256": _file_sha256(figure_path),
            "elapsed_seconds": time.monotonic() - started,
            "validation_state_opened": True,
            "inference_state_opened": False,
            "intermediate_wind_state_opened": False,
            "response_or_adjoint_state_opened": False,
        }
        report["report_content_sha256"] = _json_sha256(report)
        report_path = temporary / REPORT_NAME
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest = {
            "version": VERSION,
            "status": "complete",
            "figure": str(project_figure),
            "figure_sha256": report["figure_sha256"],
            "scratch_report": str(scratch / REPORT_NAME),
            "scratch_report_sha256": _file_sha256(report_path),
            "scratch_report_content_sha256": report["report_content_sha256"],
            "scratch_arrays": str(scratch / ARRAYS_NAME),
            "scratch_arrays_sha256": report["arrays_sha256"],
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "checkpoint_sha256": source["checkpoint_sha256"],
            "validation_state_opened": True,
            "inference_state_opened": False,
        }
        manifest["manifest_content_sha256"] = _json_sha256(manifest)
        manifest_path = temporary / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        shutil.copy2(figure_path, temporary_project_figure)
        shutil.copy2(manifest_path, temporary_project_manifest)
        os.replace(temporary, scratch)
        os.replace(temporary_project_figure, project_figure)
        os.replace(temporary_project_manifest, project_manifest)
    except Exception:
        for path in (temporary_project_figure, temporary_project_manifest):
            if path.exists():
                path.unlink()
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        **preflight,
        "status": "complete",
        "scratch_output": str(scratch),
        "project_figure": str(project_figure),
        "figure_sha256": manifest["figure_sha256"],
        "manifest": str(project_manifest),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "validation_state_opened": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        if command == "run":
            child.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight_current_figure4(args.contract)
    else:
        result = evaluate_current_figure4(args.contract, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
