"""Canonical S0 Figures 3--8 for the retained pressure-gradient-loss model.

Evaluation is intentionally unchanged from the proven figure engine preserved
in :mod:`oceanfno._figures_core`. Both selected and comparator use the
physical-static 2-in/1-out architecture: black is the retained parent and red is
its loss-only pressure-gradient fine-tune.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .runtime import torch
from .model import (
    BireTwoInNewChannelsArchitecture,
    BireTwoInNewChannelsStepper,
    build_bire_two_in_new_channels_model,
)
from . import _figures_core as base
from ._figures_core import *  # noqa: F401,F403

VERSION = "model_c_2in_1out_new_channels_pressure_gradient_s0_figures_v1"
TRAINING_VERSION = "model_c_2in_1out_new_channels_pressure_gradient_v1"
COMPARATOR_VERSION = "model_c_2in_1out_new_channels_v1"
COMPARATOR_STEP = 3840
PENDING = "PENDING_AFTER_TRAINING"
GATE_NAME = "model_c_2in_1out_new_channels_pressure_gradient_acceptance_gate.json"


class PressureGradientFigureError(RuntimeError):
    pass


def _read(contract: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = contract
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def load_contract(path: str | Path, *, verify_sources: bool = True) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    pending = [
        "selected_model.optimizer_step",
        "artifacts.selected_checkpoint.sha256",
        "artifacts.selected_normalization.sha256",
        "artifacts.selected_report.sha256",
    ]
    if any(_read(contract, tuple(name.split("."))) in (None, PENDING) for name in pending):
        raise PressureGradientFigureError("run `python -m oceanfno.figures finalize` after training")
    if (
        contract.get("version") != VERSION
        or contract.get("selected_model", {}).get("version") != TRAINING_VERSION
        or contract.get("comparator_model", {}).get("version") != COMPARATOR_VERSION
        or contract.get("selected_model", {}).get("architecture")
        != contract.get("comparator_model", {}).get("architecture")
        or int(contract.get("comparator_model", {}).get("optimizer_step", -1)) != COMPARATOR_STEP
    ):
        raise PressureGradientFigureError("pressure-gradient figure identity changed")
    starts = base.declared_inference_starts()
    if int(contract.get("protocol", {}).get("member_count", -1)) != len(starts):
        raise PressureGradientFigureError("figure member count changed")
    if verify_sources:
        for key in (
            "dataset_metadata", "selected_checkpoint", "selected_normalization",
            "selected_report", "comparator_checkpoint", "mitgcm_zonal_spacing",
            "mitgcm_sst_relaxation", "mitgcm_declaration",
        ):
            record = contract["artifacts"][key]
            source = Path(record["path"])
            if not source.is_file() or base._file_sha256(source) != record["sha256"]:
                raise PressureGradientFigureError(f"pinned figure source changed: {key}")
    return contract, resolved, base._file_sha256(resolved)


def finalize(contract_path: str | Path) -> dict[str, Any]:
    resolved = Path(contract_path).resolve()
    contract = json.loads(resolved.read_text())
    report_path = Path(contract["artifacts"]["selected_report"]["path"])
    if not report_path.is_file():
        raise PressureGradientFigureError(f"training report is not on disk: {report_path}")
    report = json.loads(report_path.read_text())
    if report.get("status") != "complete" or report.get("version") != TRAINING_VERSION:
        raise PressureGradientFigureError("training report is not the pressure-gradient arm")
    published = report["published_checkpoint"]
    resolutions = {
        ("selected_model", "optimizer_step"): int(published["optimizer_step"]),
        ("artifacts", "selected_checkpoint", "sha256"): str(published["checkpoint_sha256"]),
        ("artifacts", "selected_normalization", "sha256"): str(published["normalization_sha256"]),
        ("artifacts", "selected_report", "sha256"): base._file_sha256(report_path),
    }
    applied: dict[str, Any] = {}
    for keys, value in resolutions.items():
        current = _read(contract, keys)
        if current not in (None, PENDING) and current != value:
            raise PressureGradientFigureError(f"{'.'.join(keys)} already disagrees with report")
        node = contract
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
        if current != value:
            applied[".".join(keys)] = value
    if contract["artifacts"]["selected_checkpoint"]["path"] != published["checkpoint"]:
        raise PressureGradientFigureError("selected checkpoint path disagrees with report")
    if contract["artifacts"]["selected_normalization"]["path"] != published["normalization"]:
        raise PressureGradientFigureError("selected normalization path disagrees with report")
    if applied:
        resolved.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return {
        "status": "filled" if applied else "already_complete",
        "version": VERSION,
        "contract": str(resolved),
        "selected_optimizer_step": int(published["optimizer_step"]),
        "applied": applied,
    }


def _stepper(
    contract: Mapping[str, Any],
    key: str,
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
    static_block: np.ndarray,
) -> Any:
    if torch is None:
        raise RuntimeError("pressure-gradient figures require PyTorch")
    if key == "selected_checkpoint":
        declared = contract["selected_model"]
        expected_version = TRAINING_VERSION
    elif key == "comparator_checkpoint":
        declared = contract["comparator_model"]
        expected_version = COMPARATOR_VERSION
    else:
        raise PressureGradientFigureError(f"unknown checkpoint key: {key}")
    architecture = BireTwoInNewChannelsArchitecture(**declared["architecture"])
    payload = torch.load(Path(contract["artifacts"][key]["path"]), map_location=device, weights_only=False)
    if (
        payload.get("version") != expected_version
        or payload.get("architecture") != declared["architecture"]
        or int(payload.get("optimizer_step", -1)) != int(declared["optimizer_step"])
    ):
        raise PressureGradientFigureError(f"{key} identity changed")
    model = build_bire_two_in_new_channels_model(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    with np.load(Path(contract["artifacts"]["selected_normalization"]["path"])) as stored:
        mean = np.asarray(stored["pointwise_mean"], dtype=np.float32)
        scale = np.asarray(stored["pointwise_scale"], dtype=np.float32)
    return BireTwoInNewChannelsStepper(
        model=model,
        device=device,
        wet=wet,
        mean=mean,
        scale=scale,
        wind_mean=wind_mean,
        wind_scale=wind_scale,
        static_block=static_block,
    )


class PressureGradientLabels(base._S0Captions):
    def __init__(self, regime: str, tau0: float, selected_step: int) -> None:
        super().__init__(regime, tau0, selected_step)
        replaced = {
            "S0 architecture-direction comparison",
            "Prior residual Model C",
            "Selected anomaly-direct Model C",
        }
        self.rules = (
            ("S0 architecture-direction comparison", f"{regime} current best vs pressure-gradient loss"),
            ("Prior residual Model C", "2-in/1-out physical statics (current best)"),
            ("Selected anomaly-direct Model C", f"+ pressure-gradient loss (step {selected_step:,})"),
            *(rule for rule in self.rules if rule[0] not in replaced),
        )


def _install() -> None:
    base.VERSION = VERSION
    base.TRAINING_VERSION = TRAINING_VERSION
    base.COMPARATOR_TRAINING_VERSION = COMPARATOR_VERSION
    base.COMPARATOR_STEP = COMPARATOR_STEP
    base.GATE_NAME = GATE_NAME
    base.ACCEPTED_TRAINING_REPORT_VERSIONS = (TRAINING_VERSION,)
    base.MODEL_LABEL = "Model C 2-in / 1-out + pressure-gradient loss"
    base.load_contract = load_contract
    base._stepper = _stepper
    base.FineTuneLabels = PressureGradientLabels


def preflight(contract_path: str | Path) -> dict[str, Any]:
    _install()
    contract, resolved, digest = load_contract(contract_path)
    return {
        "status": "pass",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "selected_optimizer_step": int(contract["selected_model"]["optimizer_step"]),
        "comparator_optimizer_step": COMPARATOR_STEP,
        "member_count": len(base.declared_inference_starts()),
        "maximum_lead_days": 2000,
        "evaluation_change": "none_only_selected_checkpoint_and_comparator_identity",
    }


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    _install()
    return base.publish(contract_path, device_name=device_name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("finalize", "preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        if command == "run":
            child.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "finalize":
        result = finalize(args.contract)
    elif args.command == "preflight":
        result = preflight(args.contract)
    else:
        result = run(args.contract, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
