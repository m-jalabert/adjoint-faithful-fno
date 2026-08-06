"""Evaluation-only S0 continuation for Bire-style day-2000 Model C figures.

The continuation begins at the immutable end of ``trajectories_v2`` and is
never appended to a training dataset.  It exists only to provide continuous,
previously unseen MITgcm truth for 15 prospectively selected initial states
from the late S0 inference block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any, Mapping

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


VERSION = "model_c_bire_s0_long_truth_v1"
CONTRACT_STATUS = "frozen_before_evaluation_only_s0_continuation"
EXPECTED_STARTS = (
    6913,
    6689,
    7174,
    6962,
    6781,
    7155,
    6676,
    7019,
    7199,
    6733,
    6969,
    6662,
    6986,
    7068,
    6885,
)


class LongTruthContractError(RuntimeError):
    """Raised when the frozen S0 long-truth contract is violated."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_file(specification: Mapping[str, Any], label: str) -> Path:
    path = Path(str(specification["path"])).resolve()
    if not path.is_file():
        raise LongTruthContractError(f"{label} is missing: {path}")
    actual = _sha256(path)
    if actual != specification["sha256"]:
        raise LongTruthContractError(
            f"{label} hash changed: expected {specification['sha256']}, got {actual}"
        )
    return path


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load and validate the immutable continuation and ensemble contract."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
    ):
        raise LongTruthContractError("S0 long-truth contract is not frozen")

    simulation = contract["simulation"]
    if (
        simulation.get("experiment") != "S0"
        or float(simulation.get("tau0_n_m2", -1.0)) != 0.1
        or int(simulation.get("start_year", -1)) != 120
        or int(simulation.get("end_year", -1)) != 126
        or int(simulation.get("years", -1)) != 6
        or int(simulation.get("start_iteration", -1))
        != 120 * STEPS_PER_YEAR
        or int(simulation.get("end_iteration", -1))
        != 126 * STEPS_PER_YEAR
        or int(simulation.get("expected_daily_records", -1))
        != 6 * MODEL_YEAR_DAYS
        or simulation.get("append_to_training_dataset") is not False
    ):
        raise LongTruthContractError("S0 continuation design changed")

    ensemble = contract["ensemble"]
    starts = tuple(int(value) for value in ensemble["start_draw_order"])
    if (
        int(ensemble.get("member_count", -1)) != 15
        or int(ensemble.get("rng_seed", -1)) != 20260729
        or tuple(ensemble.get("eligible_inference_block", ())) != (6660, 7199)
        or int(ensemble.get("maximum_lead_days", -1)) != 2000
        or int(ensemble.get("prediction_interval_days", -1)) != 10
        or starts != EXPECTED_STARTS
        or len(set(starts)) != len(starts)
        or any(start < 6660 or start > 7199 for start in starts)
        or _json_sha256(list(starts))
        != ensemble["start_draw_order_json_sha256"]
    ):
        raise LongTruthContractError("S0 15-member ensemble changed")
    if max(starts) + 2000 > 7199 + int(
        simulation["expected_daily_records"]
    ):
        raise LongTruthContractError("continuation is too short for day-2000 truth")

    if verify_sources:
        for label, specification in contract["artifacts"].items():
            _verify_file(specification, label)
        project_root = resolved.parents[1]
        for relative, expected in contract["source_hashes"].items():
            source = project_root / relative
            if not source.is_file() or _sha256(source) != expected:
                raise LongTruthContractError(
                    f"source changed: {relative}"
                )
    return contract, resolved, _sha256(resolved)


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Validate every immutable source without creating a run directory."""

    contract, path, digest = load_contract(contract_path)
    parent_path = Path(contract["artifacts"]["parent_result"]["path"])
    parent = json.loads(parent_path.read_text())
    simulation = contract["simulation"]
    if (
        parent.get("experiment") != "S0"
        or float(parent.get("tau0_n_m2", -1.0)) != 0.1
        or int(parent.get("end_iteration", -1))
        != int(simulation["start_iteration"])
        or parent.get("pickup_sha256")
        != {
            "meta": contract["artifacts"]["parent_pickup_meta"]["sha256"],
            "data": contract["artifacts"]["parent_pickup_data"]["sha256"],
        }
    ):
        raise LongTruthContractError("parent S0 result does not match continuation")
    project_root = path.parents[1]
    revision = _git_revision(project_root / "external/MITgcm")
    if revision != MITGCM_COMMIT:
        raise LongTruthContractError(
            f"expected MITgcm {MITGCM_COMMIT}, found {revision}"
        )
    return {
        "status": "pass",
        "version": VERSION,
        "contract": str(path),
        "contract_sha256": digest,
        "experiment": "S0",
        "tau0_n_m2": 0.1,
        "member_count": len(EXPECTED_STARTS),
        "maximum_lead_days": 2000,
        "expected_daily_records": int(simulation["expected_daily_records"]),
        "mitgcm_commit": revision,
    }


def prepare_continuation(
    project_root: Path,
    scratch_root: Path,
    contract_path: Path,
) -> dict[str, Any]:
    """Prepare one source-locked continuous S0 production segment."""

    contract, resolved_contract, contract_sha = load_contract(contract_path)
    preflight(resolved_contract)
    simulation = contract["simulation"]
    executable = Path(contract["artifacts"]["executable"]["path"]).resolve()
    parent_path = Path(
        contract["artifacts"]["parent_result"]["path"]
    ).resolve()
    parent = json.loads(parent_path.read_text())
    parent["_result_path"] = str(parent_path)

    run_dir = (
        scratch_root.resolve()
        / str(simulation["output_root"])
        / "S0"
        / "production"
        / "years_120_126"
    )
    identity = {
        "version": VERSION,
        "experiment": "S0",
        "phase": "production",
        "start_year": 120,
        "years": 6,
        "absolute_start_year": 120,
        "absolute_end_year": 126,
        "tau0_n_m2": 0.1,
        "long_truth_contract_sha256": contract_sha,
        "executable_sha256": _sha256(executable),
    }
    manifest_path = run_dir / "segment_manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if any(prior.get(key) != value for key, value in identity.items()):
            raise LongTruthContractError("existing long-truth identity changed")
        return prior

    run_dir.mkdir(parents=True, exist_ok=False)
    start_iteration = int(simulation["start_iteration"])
    end_iteration = int(simulation["end_iteration"])
    n_time_steps = end_iteration - start_iteration
    rendered = render_data(start_iteration, n_time_steps).replace(
        "AF--FNO S0",
        "AF--FNO S0 evaluation-only Bire day-2000 truth",
        1,
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
        raise LongTruthContractError(
            f"expected MITgcm {MITGCM_COMMIT}, found {revision}"
        )
    plan = project_root.resolve() / "docs/AF_FNO_Project_Plan.tex"
    manifest = {
        **identity,
        "end_iteration": end_iteration,
        "start_iteration": start_iteration,
        "n_time_steps": n_time_steps,
        "delta_t_seconds": DELTA_T_SECONDS,
        "expected_daily_records": int(simulation["expected_daily_records"]),
        "run_dir": str(run_dir),
        "parent_result": str(parent_path),
        "parent_result_sha256": contract["artifacts"]["parent_result"][
            "sha256"
        ],
        "parent_pickups": parent_pickups,
        "mitgcm_commit": revision,
        "long_truth_contract": str(resolved_contract),
        "ensemble": contract["ensemble"],
        "training_dataset_policy": {
            "append": False,
            "purpose": "evaluation_only_unseen_S0_truth",
        },
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


def run_continuation(
    project_root: Path,
    scratch_root: Path,
    contract_path: Path,
) -> dict[str, Any]:
    """Run and validate the evaluation-only S0 continuation."""

    manifest = prepare_continuation(
        project_root,
        scratch_root,
        contract_path,
    )
    result = run_segment(manifest)
    expected = int(manifest["expected_daily_records"])
    if result.get("diagnostics") != {
        "dynState": expected,
        "surfState": expected,
    }:
        raise LongTruthContractError("long-truth daily inventory is incomplete")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--contract", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--project-root", type=Path, required=True)
    run.add_argument("--scratch-root", type=Path, required=True)
    run.add_argument("--contract", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.contract)
    else:
        result = run_continuation(
            args.project_root.resolve(),
            args.scratch_root.resolve(),
            args.contract.resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
