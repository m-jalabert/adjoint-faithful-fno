"""Immutable ten-year S0--S2 extensions for trajectory dataset version 2."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .af_s0 import (
    DELTA_T_SECONDS,
    MITGCM_COMMIT,
    MODEL_YEAR_DAYS,
    STEPS_PER_YEAR,
    _atomic_json,
    _git_revision,
    _link_parent_pickups,
    _sha256,
    render_data,
)
from .af_wind_trajectories import run_segment


EXPANSION_VERSION = "trajectories_v2_expansion_v1"
EXPERIMENTS = ("S0", "S1", "S2")


class TrajectoryExpansionError(RuntimeError):
    """Raised when a version-2 trajectory extension violates its contract."""


def dataset_pair_counts(contract: Mapping[str, Any]) -> dict[str, int]:
    """Recompute pair counts from the version-2 split blocks."""

    design = contract["dataset_v2_design"]
    horizon = int(design["horizon_days"])
    records = int(design["raw_records_per_regime"])
    blocks = design["snapshot_blocks"]
    occupied = np.zeros(records, dtype=np.uint8)
    labels = {"training": 1, "validation": 2, "inference": 3}
    counts: dict[str, int] = {}
    for name, code in labels.items():
        count = 0
        for start, stop in blocks[name]:
            start, stop = int(start), int(stop)
            if not 0 <= start < stop <= records or np.any(occupied[start:stop]):
                raise ValueError("trajectory-v2 split blocks overlap or exceed the dataset")
            occupied[start:stop] = code
            count += max(0, stop - start - horizon)
        counts[name] = count
    for start, stop in blocks["excluded"]:
        start, stop = int(start), int(stop)
        if not 0 <= start < stop <= records or np.any(occupied[start:stop]):
            raise ValueError("trajectory-v2 excluded blocks overlap a declared split")
        occupied[start:stop] = 255
    if np.any(occupied == 0):
        raise ValueError("trajectory-v2 split design leaves undeclared snapshots")
    return counts


def load_expansion_contract(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    """Load and validate the simulation contract frozen before submission."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if contract.get("version") != EXPANSION_VERSION:
        raise ValueError(f"expected {EXPANSION_VERSION}")
    if (
        contract.get("contract_status")
        != "frozen_after_data_adequacy_authorization_and_before_v2_simulation"
    ):
        raise ValueError("trajectory-v2 expansion contract was not frozen")
    if tuple(contract.get("experiments", {})) != EXPERIMENTS:
        raise ValueError("trajectory-v2 expansion must contain ordered S0--S2 contracts")
    expected_counts = {
        key: int(value)
        for key, value in contract["dataset_v2_design"][
            "pair_counts_per_regime"
        ].items()
    }
    if dataset_pair_counts(contract) != expected_counts:
        raise ValueError("trajectory-v2 pair counts do not reproduce from split blocks")
    if expected_counts["training"] * len(EXPERIMENTS) != int(
        contract["dataset_v2_design"]["training_pairs_total"]
    ):
        raise ValueError("trajectory-v2 total training-pair count is inconsistent")
    return contract, resolved, _sha256(resolved)


def resolve_experiment(array_index: int) -> str:
    """Resolve one Slurm-array index without allowing an implicit experiment."""

    if not 0 <= array_index < len(EXPERIMENTS):
        raise ValueError("trajectory-v2 array index must be 0, 1, or 2")
    return EXPERIMENTS[array_index]


def _validate_decision(contract: Mapping[str, Any]) -> dict[str, Any]:
    declared = contract["data_adequacy_decision"]
    path = Path(str(declared["path"])).resolve()
    if not path.is_file() or _sha256(path) != declared["sha256"]:
        raise TrajectoryExpansionError("data-adequacy decision is missing or changed")
    report = json.loads(path.read_text())
    if (
        report.get("decision", {}).get("status") != declared["status"]
        or report.get("decision", {}).get("expansion_authorized") is not True
    ):
        raise TrajectoryExpansionError("data-adequacy report does not authorize expansion")
    return report


def prepare_expansion(
    project_root: Path,
    scratch_root: Path,
    executable: Path,
    contract_path: Path,
    experiment: str,
) -> dict[str, Any]:
    """Prepare one exact continuous extension from its immutable v1 pickup."""

    contract, resolved_contract, contract_sha = load_expansion_contract(contract_path)
    _validate_decision(contract)
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}")
    spec = contract["experiments"][experiment]
    parent_path = Path(str(spec["parent_result"])).resolve()
    if not parent_path.is_file() or _sha256(parent_path) != spec["parent_result_sha256"]:
        raise TrajectoryExpansionError(f"{experiment} parent result is missing or changed")
    parent = json.loads(parent_path.read_text())
    parent["_result_path"] = str(parent_path)

    local_start = int(spec["local_start_year"])
    local_end = int(spec["local_end_year"])
    absolute_start = int(spec["absolute_start_year"])
    absolute_end = int(spec["absolute_end_year"])
    years = int(spec["years"])
    if (
        local_end - local_start != years
        or absolute_end - absolute_start != years
        or int(parent["end_iteration"]) != absolute_start * STEPS_PER_YEAR
    ):
        raise TrajectoryExpansionError(f"{experiment} extension years do not meet its parent")

    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"AF--FNO executable is missing: {executable}")
    run_dir = (
        scratch_root.resolve()
        / contract["simulation_design"]["extension_root"]
        / experiment
        / "production"
        / f"years_{local_start:03d}_{local_end:03d}"
    )
    identity = {
        "version": EXPANSION_VERSION,
        "experiment": experiment,
        "phase": "production",
        "start_year": local_start,
        "years": years,
        "absolute_start_year": absolute_start,
        "absolute_end_year": absolute_end,
        "tau0_n_m2": float(spec["tau0_n_m2"]),
        "expansion_contract_sha256": contract_sha,
        "executable_sha256": _sha256(executable),
    }
    manifest_path = run_dir / "segment_manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if any(prior.get(key) != value for key, value in identity.items()):
            raise TrajectoryExpansionError(
                f"{experiment} trajectory-v2 segment identity changed"
            )
        return prior

    run_dir.mkdir(parents=True, exist_ok=False)
    start_iteration = absolute_start * STEPS_PER_YEAR
    end_iteration = absolute_end * STEPS_PER_YEAR
    rendered = render_data(start_iteration, years * STEPS_PER_YEAR).replace(
        "AF--FNO S0", f"AF--FNO {experiment} trajectories-v2 extension", 1
    )
    (run_dir / "data").write_text(rendered)
    input_root = project_root.resolve() / "af_fno/mitgcm/input"
    for name in ("data.pkg", "eedata"):
        shutil.copy2(input_root / name, run_dir / name)
    shutil.copy2(
        input_root / "data.diagnostics.production",
        run_dir / "data.diagnostics",
    )
    parent_root = Path(str(parent["run_dir"])).resolve()
    for name in ("bathy.bin", "windx_cosy.bin", "SST_relax.bin"):
        shutil.copy2(parent_root / name, run_dir / name)
    (run_dir / "mitgcmuv").symlink_to(executable)
    parent_pickups = _link_parent_pickups(parent, run_dir, start_iteration)

    revision = _git_revision(project_root.resolve() / "external/MITgcm")
    if revision != MITGCM_COMMIT:
        raise TrajectoryExpansionError(f"expected MITgcm {MITGCM_COMMIT}, found {revision}")
    plan = project_root.resolve() / "docs/AF_FNO_Project_Plan.tex"
    manifest = {
        **identity,
        "local_end_year": local_end,
        "end_iteration": end_iteration,
        "start_iteration": start_iteration,
        "n_time_steps": years * STEPS_PER_YEAR,
        "delta_t_seconds": DELTA_T_SECONDS,
        "expected_daily_records": years * MODEL_YEAR_DAYS,
        "run_dir": str(run_dir),
        "parent_result": str(parent_path),
        "parent_result_sha256": spec["parent_result_sha256"],
        "parent_pickups": parent_pickups,
        "mitgcm_commit": revision,
        "expansion_contract": str(resolved_contract),
        "data_adequacy_decision": contract["data_adequacy_decision"],
        "project_plan": {"path": str(plan), "sha256": _sha256(plan)},
        "configuration_sha256": {
            name: _sha256(run_dir / name)
            for name in ("data", "data.pkg", "eedata", "data.diagnostics")
        },
        "forcing_sha256": {
            name: _sha256(run_dir / name)
            for name in ("bathy.bin", "windx_cosy.bin", "SST_relax.bin")
        },
        "created_by": {
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def run_expansion(
    project_root: Path,
    scratch_root: Path,
    executable: Path,
    contract_path: Path,
    experiment: str,
) -> dict[str, Any]:
    """Prepare, run, and validate one trajectories-v2 extension."""

    manifest = prepare_expansion(
        project_root,
        scratch_root,
        executable,
        contract_path,
        experiment,
    )
    result = run_segment(manifest)
    if result.get("diagnostics") != {
        "dynState": int(manifest["expected_daily_records"]),
        "surfState": int(manifest["expected_daily_records"]),
    }:
        raise TrajectoryExpansionError(
            f"{experiment} extension did not produce its complete daily inventory"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser("resolve-experiment")
    resolve.add_argument("--array-index", type=int, required=True)
    run = commands.add_parser("run")
    run.add_argument("--project-root", type=Path, required=True)
    run.add_argument("--scratch-root", type=Path, required=True)
    run.add_argument("--executable", type=Path, required=True)
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "resolve-experiment":
        print(resolve_experiment(args.array_index))
        return 0
    result = run_expansion(
        args.project_root.resolve(),
        args.scratch_root.resolve(),
        args.executable.resolve(),
        args.contract.resolve(),
        args.experiment,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
