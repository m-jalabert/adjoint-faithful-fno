"""Canonical anomaly diagnostics for the retained pressure-gradient-loss model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import _anomaly_core as base
from ._anomaly_core import *  # noqa: F401,F403
from . import figures as pgfig
from . import plots

VERSION = "model_c_2in_1out_new_channels_pressure_gradient_s0_anomaly_v1"
FIGURE_VERSION = pgfig.VERSION
PENDING = "PENDING_AFTER_FIGURES"


class PressureGradientAnomalyError(RuntimeError):
    pass


def _digest(path: Path) -> str:
    return base._file_sha256(path)


def _sealed(contract: Mapping[str, Any]) -> dict[str, str]:
    artifacts = contract["artifacts"]
    paths = {
        key: Path(artifacts[key]["path"]).resolve()
        for key in (
            "figure_package_contract", "figure_package_report",
            "figure_package_arrays", "figure_package_manifest",
        )
    }
    if not all(path.is_file() for path in paths.values()):
        raise PressureGradientAnomalyError("pressure-gradient figure package is incomplete")
    digests = {key: _digest(path) for key, path in paths.items()}
    figure_contract = json.loads(paths["figure_package_contract"].read_text())
    report = json.loads(paths["figure_package_report"].read_text())
    manifest = json.loads(paths["figure_package_manifest"].read_text())
    if (
        figure_contract.get("version") != FIGURE_VERSION
        or report.get("status") != "complete"
        or report.get("version") != FIGURE_VERSION
        or report.get("regime") != "S0"
        or report.get("contract_sha256") != digests["figure_package_contract"]
        or report.get("arrays_sha256") != digests["figure_package_arrays"]
        or manifest.get("version") != FIGURE_VERSION
        or manifest.get("contract_sha256") != digests["figure_package_contract"]
        or manifest.get("artifacts", {}).get(plots.ARRAYS_NAME, {}).get("sha256")
        != digests["figure_package_arrays"]
        or manifest.get("artifacts", {}).get(plots.REPORT_NAME, {}).get("sha256")
        != digests["figure_package_report"]
    ):
        raise PressureGradientAnomalyError("sealed pressure-gradient figure provenance changed")
    with np.load(paths["figure_package_arrays"]) as stored:
        required = {
            "figure3_truth_streamfunction", "figure3_model_streamfunction",
            "figure7_truth_streamfunction", "figure7_model_streamfunction",
            "longitude_deg", "latitude_deg", "wet_mask", "start_draw_order",
        }
        if not required.issubset(stored.files):
            raise PressureGradientAnomalyError("figure arrays are incomplete")
    return digests


def finalize(contract_path: str | Path) -> dict[str, Any]:
    resolved = Path(contract_path).resolve()
    contract = json.loads(resolved.read_text())
    digests = _sealed(contract)
    applied: dict[str, str] = {}
    for key, digest in digests.items():
        current = contract["artifacts"][key].get("sha256")
        if current not in (None, PENDING) and current != digest:
            raise PressureGradientAnomalyError(f"{key} is already pinned to different bytes")
        if current != digest:
            contract["artifacts"][key]["sha256"] = digest
            applied[f"artifacts.{key}.sha256"] = digest
    if applied:
        resolved.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return {
        "status": "filled" if applied else "already_complete",
        "version": VERSION,
        "contract": str(resolved),
        "applied": applied,
    }


def load_contract(path: str | Path, *, verify_sources: bool = True) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != VERSION:
        raise PressureGradientAnomalyError("wrong anomaly contract version")
    if any(
        contract["artifacts"][key].get("sha256") in (None, PENDING)
        for key in (
            "figure_package_contract", "figure_package_report",
            "figure_package_arrays", "figure_package_manifest",
        )
    ):
        raise PressureGradientAnomalyError("run `python -m oceanfno.anomaly finalize` after figures")
    digests = _sealed(contract)
    for key, digest in digests.items():
        if contract["artifacts"][key]["sha256"] != digest:
            raise PressureGradientAnomalyError(f"{key} changed after finalization")
    dataset = contract["artifacts"]["dataset_metadata"]
    target = Path(dataset["path"])
    if verify_sources and (not target.is_file() or _digest(target) != dataset["sha256"]):
        raise PressureGradientAnomalyError("dataset metadata changed")
    return contract, resolved, _digest(resolved)


def _install(contract: Mapping[str, Any]) -> None:
    base.VERSION = VERSION
    base.CONTRACT_STATUS = contract["contract_status"]
    base.FIGURE_PACKAGE_VERSION = FIGURE_VERSION
    base.figures = pgfig
    base.ModelCTwoInAnomalyError = PressureGradientAnomalyError
    base.ModelCNewChannelsAnomalyError = PressureGradientAnomalyError
    base._EXPECTED_PROJECT_ROOT = contract["output"]["project_root"]
    base._EXPECTED_SCRATCH_ROOT = contract["output"]["scratch_root"]
    base.load_contract = load_contract
    base.finalize = finalize


def preflight(contract_path: str | Path) -> dict[str, Any]:
    contract = json.loads(Path(contract_path).read_text())
    _install(contract)
    loaded, resolved, digest = load_contract(contract_path)
    return {
        "status": "pass",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "reads_model_weights": False,
        "rolls_nothing_out": True,
        "diagnostic_change": "none",
        "figure_package_arrays": loaded["artifacts"]["figure_package_arrays"]["path"],
    }


def run(contract_path: str | Path) -> dict[str, Any]:
    contract = json.loads(Path(contract_path).read_text())
    _install(contract)
    return base.run(contract_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("finalize", "preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "finalize":
        result = finalize(args.contract)
    elif args.command == "preflight":
        result = preflight(args.contract)
    else:
        result = run(args.contract)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
