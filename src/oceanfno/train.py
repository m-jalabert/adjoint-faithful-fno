"""Canonical training entrypoint for the retained 2-in/1-out Model C.

This is deliberately a *loss-only* successor. It reuses the proven training
engine preserved in :mod:`oceanfno._train_core`, strict-loads the selected
``model_c_2in_1out_new_channels_pressure_gradient_continuity_v1`` weights,
resets Adam, keeps that arm's pressure-gradient and continuity terms, and adds
one further auxiliary term: a depth-integrated barotropic transport-tendency
constraint, chosen because the streamfunction reconstruction audit isolated the
residual long-horizon error there. Architecture, data, rollout, optimizer
schedule, normalization, and checkpoint selection remain unchanged.
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
from .barotropic_transport import barotropic_transport_relative_l2
from .continuity import ContinuityContext, continuity_relative_l2
from .pressure_gradient import PressureGradientContext, pressure_gradient_relative_l2
from . import _train_core as base
from ._train_core import *  # noqa: F401,F403

VERSION = "model_c_2in_1out_new_channels_p_cont_BT_loss_v1"
PARENT_VERSION = "model_c_2in_1out_new_channels_pressure_gradient_continuity_v1"
SLUG = "model_c_2in_1out_new_channels_p_cont_BT_loss"
PRESSURE_TERM = "pressure_gradient"
CONTINUITY_TERM = "continuity"
BAROTROPIC_TERM = "barotropic_transport"
DEFAULT_PRESSURE_WEIGHT = 0.05
DEFAULT_CONTINUITY_WEIGHT = 0.05
DEFAULT_BAROTROPIC_WEIGHT = 0.05

NORMALIZATION_NAME = f"{SLUG}_train_only_normalization.npz"
DIVERGENCE_NAME = f"{SLUG}_divergence.json"
CHECKPOINT_STEM = f"{SLUG}_step"
REPORT_NAME = f"{SLUG}_report.json"
ARRAYS_NAME = f"{SLUG}_arrays.npz"
FIGURE_NAME = f"{SLUG}_selection.png"

_CONTEXT: PressureGradientContext | None = None
_CONTINUITY_CONTEXT: ContinuityContext | None = None
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
    continuity_weight = float(contract["loss"].get("continuity_weight", -1.0))
    if not np.isfinite(continuity_weight) or continuity_weight <= 0.0:
        raise PressureGradientTrainingError("continuity weight must be positive")
    barotropic_weight = float(contract["loss"].get("barotropic_transport_weight", -1.0))
    if not np.isfinite(barotropic_weight) or barotropic_weight <= 0.0:
        raise PressureGradientTrainingError("barotropic-transport weight must be positive")
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


def _loss_with_physics_terms(
    predictions: Any,
    targets: Any,
    present: Any,
    wet: Any,
    western_boundary: Any,
    increment_scale: Any,
    config: Any,
) -> dict[str, Any]:
    if _CONTEXT is None or _CONTINUITY_CONTEXT is None or _ACTIVE_CONTRACT is None:
        raise PressureGradientTrainingError("physics loss context was not initialized")
    result = _ORIGINAL_LOSS(
        predictions, targets, present, wet, western_boundary, increment_scale, config
    )
    pressure = pressure_gradient_relative_l2(predictions, targets, _CONTEXT)
    continuity = continuity_relative_l2(predictions, targets, present, _CONTINUITY_CONTEXT)
    barotropic = barotropic_transport_relative_l2(
        predictions, targets, present, _CONTINUITY_CONTEXT
    )
    weight = float(_ACTIVE_CONTRACT["loss"]["pressure_gradient_weight"])
    continuity_weight = float(_ACTIVE_CONTRACT["loss"]["continuity_weight"])
    barotropic_weight = float(_ACTIVE_CONTRACT["loss"]["barotropic_transport_weight"])
    result[PRESSURE_TERM] = pressure
    result[CONTINUITY_TERM] = continuity
    result[BAROTROPIC_TERM] = barotropic
    result["total"] = (
        result["total"]
        + weight * pressure
        + continuity_weight * continuity
        + barotropic_weight * barotropic
    )
    return result


def _same_layout_perturbation_features(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    _, current = _ORIGINAL_PERTURBATION_FEATURES(*args, **kwargs)
    return current, current


def _parent_architecture_proxy() -> BireTwoInNewChannelsArchitecture:
    if _ACTIVE_CONTRACT is None:
        raise PressureGradientTrainingError("pressure-gradient contract is not active")
    return BireTwoInNewChannelsArchitecture(**_ACTIVE_CONTRACT["architecture"])


def _readme(report: Mapping[str, Any]) -> str:
    weight = float(report["loss"]["pressure_gradient_weight"])
    continuity_weight = float(report["loss"]["continuity_weight"])
    barotropic_weight = float(report["loss"]["barotropic_transport_weight"])
    selected = int(report["selection_decision"]["selected_optimizer_step"])
    return f"""# 2-in / 1-out physical-static Model C + pressure gradient + continuity + transport

Loss-only fine-tune of `{PARENT_VERSION}`. Architecture, inputs, 32 x 32 modes,
local 3 x 3 branch, six-step autoregression, dataset, normalizers, optimizer
schedule, validation records and checkpoint selection are unchanged.

The retained objective is

    L = L_parent
        + {weight:g} * L_pressure_gradient
        + {continuity_weight:g} * L_continuity
        + {barotropic_weight:g} * L_barotropic_transport

`L_pressure_gradient` and `L_continuity` are inherited unchanged from the
parent: the first reconstructs total MITgcm PHIHYD from predicted THETA and
ETAN and compares its horizontal gradient to truth; the second forms the
free-surface residual `d(eta)/dt + div(Q)` and scores it against the truth
residual.

The sole scientific addition is `L_barotropic_transport`. For each ten-day step
it depth-integrates the predicted velocities with the tutorial's exact 15
`delR` thicknesses into `T_x` and `T_y`, and scores the *tendency*

    dT_pred = T(x_hat_s) - T(x_hat_s-1)   against   dT_true = T(x_s) - T(x_s-1)

as `0.5 * [ ||dT_x_pred - dT_x_true||^2 / (||dT_x_true||^2 + eps)
          + ||dT_y_pred - dT_y_true||^2 / (||dT_y_true||^2 + eps) ]`,
with the first call measured from the observed present state. All six calls
carry equal status, and the zonal and meridional components are weighted
equally on purpose.

The target was chosen by the streamfunction reconstruction audit, which showed
that most of the visible day-2000 streamfunction striping is manufactured by
the one-way cumulative integral used to reconstruct `psi`, while a real 10.5 %
error remains in the depth-integrated zonal transport. A streamfunction loss
was rejected for that reason: it would have asked the operator to compensate
for the diagnostic's amplification as well as for the physical error.

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
    base.AUDIT_TERMS = tuple(
        dict.fromkeys(
            tuple(base.AUDIT_TERMS) + (PRESSURE_TERM, CONTINUITY_TERM, BAROTROPIC_TERM)
        )
    )
    base.load_contract = load_contract
    base.load_initial_state_dict = _strict_initial_state
    base.model_c_loss_terms = _loss_with_physics_terms
    base._perturbation_features = _same_layout_perturbation_features
    base.BireTwoInOneOutArchitecture = _parent_architecture_proxy
    base.build_bire_two_in_one_out_model = base.build_bire_two_in_new_channels_model
    base._readme = _readme


def _build_context(
    contract: Mapping[str, Any],
) -> tuple[PressureGradientContext, ContinuityContext]:
    dataset = Path(contract["sources"]["dataset"]["path"])
    group = zarr.open_consolidated(str(dataset), mode="r")
    normalization = base.reused_normalization(contract)
    dx = read_mitgcm_2d(contract["sources"]["mitgcm_zonal_spacing"]["path"])
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    # Both auxiliary terms read the same normalizers and the same MITgcm metric.
    fields = {
        "pointwise_mean": np.asarray(normalization["mean"], dtype=np.float32),
        "pointwise_scale": np.asarray(normalization["scale"], dtype=np.float32),
        "zonal_spacing_m": np.asarray(dx, dtype=np.float32),
        "wet_mask": wet,
    }
    return PressureGradientContext(**fields), ContinuityContext(**fields)


def preflight(contract_path: str | Path) -> dict[str, Any]:
    global _CONTEXT, _CONTINUITY_CONTEXT
    contract, resolved, digest = load_contract(contract_path)
    _CONTEXT, _CONTINUITY_CONTEXT = _build_context(contract)
    if torch is not None:
        # Mathematical sanity: truth against itself must have zero physics loss.
        dummy = torch.zeros((1, 1, 46, 62, 62), dtype=torch.float32, requires_grad=True)
        zero = pressure_gradient_relative_l2(dummy, dummy, _CONTEXT)
        if float(zero.detach()) != 0.0:
            raise PressureGradientTrainingError("pressure-gradient identity loss is not zero")
        zero_continuity = continuity_relative_l2(
            dummy, dummy, dummy[:, 0], _CONTINUITY_CONTEXT
        )
        if float(zero_continuity.detach()) != 0.0:
            raise PressureGradientTrainingError("continuity identity loss is not zero")
        zero_barotropic = barotropic_transport_relative_l2(
            dummy, dummy, dummy[:, 0], _CONTINUITY_CONTEXT
        )
        if float(zero_barotropic.detach()) != 0.0:
            raise PressureGradientTrainingError("barotropic-transport identity loss is not zero")
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "parent_version": PARENT_VERSION,
        "parent_checkpoint_sha256": contract["sources"]["parent_checkpoint"]["sha256"],
        "pressure_gradient_weight": float(contract["loss"]["pressure_gradient_weight"]),
        "continuity_weight": float(contract["loss"]["continuity_weight"]),
        "barotropic_transport_weight": float(contract["loss"]["barotropic_transport_weight"]),
        "architecture_change": "none",
        "optimizer_state_loaded": False,
        "loss_change_only": True,
    }


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    global _CONTEXT, _CONTINUITY_CONTEXT
    contract, _, _ = load_contract(contract_path)
    _CONTEXT, _CONTINUITY_CONTEXT = _build_context(contract)
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
