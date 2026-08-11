"""Fine-tune the selected 2-in/1-out physical-static Model C with PHIHYD-gradient loss.

This is deliberately a *loss-only* successor.  It reuses the canonical training
engine in :mod:`oceanfno.train`, strict-loads the selected
``model_c_2in_1out_new_channels_v1`` weights, resets Adam, and adds one
MITgcm-consistent auxiliary term.  The parent training module is not modified,
so the retained experiment remains byte-for-byte reproducible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from .runtime import torch
from .dataset import read_mitgcm_2d
from .model import BireTwoInNewChannelsArchitecture
from .pressure_gradient import PressureGradientContext, pressure_gradient_relative_l2
from . import train as base

VERSION = "model_c_2in_1out_new_channels_pressure_gradient_v1"
PARENT_VERSION = "model_c_2in_1out_new_channels_v1"
SLUG = "model_c_2in_1out_new_channels_pressure_gradient"
PRESSURE_TERM = "pressure_gradient"
DEFAULT_PRESSURE_WEIGHT = 0.05

NORMALIZATION_NAME = f"{SLUG}_train_only_normalization.npz"
DIVERGENCE_NAME = f"{SLUG}_divergence.json"
CHECKPOINT_STEM = f"{SLUG}_step"
REPORT_NAME = f"{SLUG}_report.json"
ARRAYS_NAME = f"{SLUG}_arrays.npz"
FIGURE_NAME = f"{SLUG}_selection.png"

_CONTEXT: PressureGradientContext | None = None
_ACTIVE_CONTRACT: Mapping[str, Any] | None = None
_ORIGINAL_LOSS = base.model_c_loss_terms
_ORIGINAL_PERTURBATION_FEATURES = base._perturbation_features
_ORIGINAL_BUILD_PARENT = base.build_bire_two_in_one_out_model
_ORIGINAL_PARENT_ARCH = base.BireTwoInOneOutArchitecture
_ORIGINAL_README = base._readme


class PressureGradientTrainingError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return base._file_sha256(path)


def load_contract(path: str | Path, *, verify_sources: bool = True) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != VERSION:
        raise PressureGradientTrainingError("wrong pressure-gradient training contract version")
    architecture = BireTwoInNewChannelsArchitecture(**contract["architecture"])
    if architecture.to_dict() != contract["architecture"]:
        raise PressureGradientTrainingError("pressure-gradient architecture is not canonical")
    initialization = contract["initialization"]
    if (
        initialization.get("version") != PARENT_VERSION
        or initialization.get("load_only") != "model_state_dict"
        or initialization.get("strict_same_shape_load") is not True
        or initialization.get("function_preserving") is not True
        or initialization.get("optimizer_state_loaded") is not False
    ):
        raise PressureGradientTrainingError("initialization must be a strict function-preserving parent load")
    training = contract["training"]
    expected_training = {
        "batch_size": base.BATCH_SIZE,
        "maximum_steps": base.MAXIMUM_STEPS,
        "rollout_steps": base.ROLLOUT_STEPS,
        "seed": base.SEED,
    }
    for key, expected in expected_training.items():
        if int(training.get(key, -1)) != int(expected):
            raise PressureGradientTrainingError(f"training field {key} moved from the parent")
    weight = float(contract["loss"].get("pressure_gradient_weight", -1.0))
    if not np.isfinite(weight) or weight <= 0.0:
        raise PressureGradientTrainingError("pressure-gradient weight must be positive")
    if verify_sources:
        dataset = Path(contract["sources"]["dataset"]["path"])
        if not dataset.is_dir() or _sha(dataset / ".zmetadata") != contract["sources"]["dataset"]["metadata_sha256"]:
            raise PressureGradientTrainingError("trajectory-v3 source changed")
        for key in (
            "parent_checkpoint", "parent_normalization", "parent_report",
            "mitgcm_zonal_spacing", "mitgcm_sst_relaxation", "mitgcm_declaration",
        ):
            record = contract["sources"][key]
            source = Path(record["path"])
            if not source.is_file() or _sha(source) != record["sha256"]:
                raise PressureGradientTrainingError(f"pinned source changed: {key}")
    return contract, resolved, _sha(resolved)


def _strict_initial_state(contract: Mapping[str, Any], device: Any, target_model: Any) -> dict[str, Any]:
    record = contract["sources"]["parent_checkpoint"]
    path = Path(record["path"]).resolve()
    if _sha(path) != record["sha256"]:
        raise PressureGradientTrainingError("selected parent checkpoint changed")
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict")
    if state is None:
        raise PressureGradientTrainingError("parent checkpoint has no model_state_dict")
    # Let PyTorch prove every tensor name and shape agrees before training starts.
    target_model.load_state_dict(state, strict=True)
    return {
        "state_dict": state,
        "provenance": {
            "path": str(path),
            "sha256": record["sha256"],
            "version": PARENT_VERSION,
            "optimizer_step": int(contract["initialization"]["optimizer_step"]),
            "loaded": "model_state_dict_strict",
            "optimizer_state_loaded": False,
            "function_preserving": True,
            "scientific_change_at_initialization": "none_loss_only_fine_tune",
        },
    }


def _loss_with_pressure_gradient(
    predictions: Any,
    targets: Any,
    present: Any,
    wet: Any,
    western_boundary: Any,
    increment_scale: Any,
    config: Any,
) -> dict[str, Any]:
    if _CONTEXT is None or _ACTIVE_CONTRACT is None:
        raise PressureGradientTrainingError("pressure-gradient context was not initialized")
    result = _ORIGINAL_LOSS(
        predictions, targets, present, wet, western_boundary, increment_scale, config
    )
    pressure = pressure_gradient_relative_l2(predictions, targets, _CONTEXT)
    weight = float(_ACTIVE_CONTRACT["loss"]["pressure_gradient_weight"])
    result[PRESSURE_TERM] = pressure
    result["total"] = result["total"] + weight * pressure
    return result


def _same_layout_perturbation_features(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    # The parent and child have identical input layouts; take the existing helper's
    # new-channel tensor for both sides so the recorded initialization delta must
    # be exactly zero up to floating-point replay.
    _, current = _ORIGINAL_PERTURBATION_FEATURES(*args, **kwargs)
    return current, current


def _parent_architecture_proxy() -> BireTwoInNewChannelsArchitecture:
    if _ACTIVE_CONTRACT is None:
        raise PressureGradientTrainingError("pressure-gradient contract is not active")
    return BireTwoInNewChannelsArchitecture(**_ACTIVE_CONTRACT["architecture"])


def _readme(report: Mapping[str, Any]) -> str:
    weight = float(report["loss"]["pressure_gradient_weight"])
    selected = int(report["selection_decision"]["selected_optimizer_step"])
    return f"""# 2-in / 1-out physical-static Model C + pressure-gradient consistency

Loss-only fine-tune of `{PARENT_VERSION}`. Architecture, inputs, 32 x 32 modes,
local 3 x 3 branch, six-step autoregression, dataset, normalizers, optimizer
schedule, validation records and checkpoint selection are unchanged.

The sole scientific addition is

    L = L_parent + {weight:g} * L_pressure_gradient

where `L_pressure_gradient` reconstructs total MITgcm PHIHYD from predicted
THETA and ETAN, forms neighboring-tracer horizontal gradients on C-grid velocity
faces, and compares them to truth with a dimensionless relative-L2 metric. The
hydrostatic integration uses the tutorial's exact 15 `delR` values and reference
temperature profile already validated against MITgcm PHIHYD in the archived
pressure diagnostic.

Selected optimizer step: {selected:,}.
Parent initialization is strict same-shape and function-preserving; Adam state is
reset.
"""


def _install_patches(contract: Mapping[str, Any]) -> None:
    global _ACTIVE_CONTRACT
    _ACTIVE_CONTRACT = contract
    base.VERSION = VERSION
    base.PARENT_VERSION = PARENT_VERSION
    base.SLUG = SLUG
    base.NORMALIZATION_NAME = NORMALIZATION_NAME
    base.DIVERGENCE_NAME = DIVERGENCE_NAME
    base.CHECKPOINT_STEM = CHECKPOINT_STEM
    base.REPORT_NAME = REPORT_NAME
    base.ARRAYS_NAME = ARRAYS_NAME
    base.FIGURE_NAME = FIGURE_NAME
    base.AUDIT_TERMS = tuple(dict.fromkeys(tuple(base.AUDIT_TERMS) + (PRESSURE_TERM,)))
    base.load_contract = load_contract
    base.load_initial_state_dict = _strict_initial_state
    base.model_c_loss_terms = _loss_with_pressure_gradient
    base._perturbation_features = _same_layout_perturbation_features
    base.BireTwoInOneOutArchitecture = _parent_architecture_proxy
    base.build_bire_two_in_one_out_model = base.build_bire_two_in_new_channels_model
    base._readme = _readme


def _build_context(contract: Mapping[str, Any]) -> PressureGradientContext:
    dataset = Path(contract["sources"]["dataset"]["path"])
    group = zarr.open_consolidated(str(dataset), mode="r")
    normalization = base.reused_normalization(contract)
    dx = read_mitgcm_2d(contract["sources"]["mitgcm_zonal_spacing"]["path"])
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    return PressureGradientContext(
        pointwise_mean=np.asarray(normalization["mean"], dtype=np.float32),
        pointwise_scale=np.asarray(normalization["scale"], dtype=np.float32),
        zonal_spacing_m=np.asarray(dx, dtype=np.float32),
        wet_mask=wet,
    )


def preflight(contract_path: str | Path) -> dict[str, Any]:
    global _CONTEXT
    contract, resolved, digest = load_contract(contract_path)
    _ACTIVE = contract
    _CONTEXT = _build_context(contract)
    if torch is not None:
        # Mathematical sanity: truth against itself must have zero pressure loss.
        dummy = torch.zeros((1, 1, 46, 62, 62), dtype=torch.float32, requires_grad=True)
        zero = pressure_gradient_relative_l2(dummy, dummy, _CONTEXT)
        if float(zero.detach()) != 0.0:
            raise PressureGradientTrainingError("pressure-gradient identity loss is not zero")
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "parent_version": PARENT_VERSION,
        "parent_checkpoint_sha256": contract["sources"]["parent_checkpoint"]["sha256"],
        "pressure_gradient_weight": float(contract["loss"]["pressure_gradient_weight"]),
        "architecture_change": "none",
        "optimizer_state_loaded": False,
        "loss_change_only": True,
    }


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    global _CONTEXT
    contract, _, _ = load_contract(contract_path)
    _CONTEXT = _build_context(contract)
    _install_patches(contract)
    return base.run(contract_path, device_name=device_name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        if command == "run":
            child.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = preflight(args.contract) if args.command == "preflight" else run(
        args.contract, device_name=args.device
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
