"""Independent-seed replication of the pointwise-anomaly direct-state Model C.

The parent experiment selected every checkpoint using split 1 before opening the
fixed 15-member S2 characterization.  This runner changes only the initialization
and batch-order seed.  Each replica repeats the same training-only selection and
then the already frozen S2 characterization; inference, intermediate-wind,
response, and adjoint archives remain sealed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .af_model_c_anomaly_direct import (
    load_anomaly_direct_contract,
    preflight_anomaly_direct,
    run_anomaly_direct,
)

VERSION = "model_c_anomaly_direct_replication_v1"
CONTRACT_STATUS = (
    "frozen_after_seed20260723_pass_before_independent_seed_replication"
)
EXPECTED_DECLARED_SEEDS = (20260723, 20260724, 20260725)
EXPECTED_NEW_SEEDS = (20260724, 20260725)


class ModelCAnomalyDirectReplicationError(RuntimeError):
    """Raised when the replication contract or immutable parent changes."""


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _verify_reference_result(
    contract: Mapping[str, Any],
    root: Path,
) -> None:
    reference = contract["reference_result"]
    report_path = Path(reference["report"])
    checkpoint_path = Path(reference["checkpoint"])
    if not report_path.is_absolute():
        report_path = root / report_path
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    if (
        not report_path.is_file()
        or _file_sha256(report_path) != reference["report_sha256"]
        or not checkpoint_path.is_file()
        or _file_sha256(checkpoint_path) != reference["checkpoint_sha256"]
    ):
        raise ModelCAnomalyDirectReplicationError(
            "anomaly-direct reference result changed"
        )
    report = json.loads(report_path.read_text())
    decision = report.get("selection_decision", {})
    metrics = report.get("validation_figure", {}).get("metrics", {})
    expected_day200 = reference["day200_model_rmse"]
    if (
        report.get("status") != "complete"
        or int(report.get("seed", -1)) != 20260723
        or decision.get("classification")
        != "training_only_pushforward_gate_passed"
        or decision.get("passed") is not True
        or int(decision.get("selected_fine_tune_step", -1)) != 14400
        or report.get("save_reload_nine_step_bitwise_exact") is not True
        or report.get("validation_state_opened") is not True
        or report.get("inference_state_opened") is not False
    ):
        raise ModelCAnomalyDirectReplicationError(
            "anomaly-direct reference decision changed"
        )
    for field, expected in expected_day200.items():
        observed = metrics.get(field, {}).get("model", {}).get("day200_mean")
        if observed is None or float(observed) != float(expected):
            raise ModelCAnomalyDirectReplicationError(
                f"anomaly-direct reference day-200 metric changed: {field}"
            )


def load_replication_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load and validate the independent-seed replication contract."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    seeds = contract.get("seed_replication", {})
    reads = contract.get("read_contract", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or tuple(int(value) for value in seeds.get("declared_seeds", ()))
        != EXPECTED_DECLARED_SEEDS
        or tuple(int(value) for value in seeds.get("new_seeds", ()))
        != EXPECTED_NEW_SEEDS
        or seeds.get("change_from_parent") != "seed_only"
        or seeds.get("checkpoint_selection") != "training_split1_only"
        or reads.get("training_state") is not True
        or reads.get("fixed_S2_validation_figure_state") is not True
        or any(
            reads.get(name) is not False
            for name in (
                "inference_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
            )
        )
    ):
        raise ValueError("anomaly-direct replication contract changed")

    root = resolved.parents[1]
    parent = contract["parent_contract"]
    parent_path = root / parent["path"]
    if (
        not parent_path.is_file()
        or _file_sha256(parent_path) != parent["sha256"]
    ):
        raise ModelCAnomalyDirectReplicationError(
            "anomaly-direct parent contract changed"
        )
    load_anomaly_direct_contract(parent_path, verify_sources=verify_sources)
    _verify_reference_result(contract, root)

    if verify_sources:
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ValueError(
                    f"anomaly-direct replication source changed: {source}"
                )
    return contract, resolved, _file_sha256(resolved)


def resolve_replication_seed(
    contract_path: str | Path,
    array_index: int,
) -> int:
    """Map a zero-based Slurm array index to its frozen independent seed."""

    contract, _, _ = load_replication_contract(contract_path)
    seeds = tuple(
        int(value) for value in contract["seed_replication"]["new_seeds"]
    )
    if array_index < 0 or array_index >= len(seeds):
        raise IndexError(
            f"array index {array_index} is outside {len(seeds)} replication seeds"
        )
    return seeds[array_index]


def _paths_for_seed(
    contract: Mapping[str, Any],
    seed: int,
) -> tuple[Path, Path, Path]:
    outputs = contract["output_contract"]
    scratch_root = Path(outputs["scratch_root"]).resolve()
    project_root = Path(outputs["project_root"]).resolve()
    scratch_output = scratch_root / "seeds" / f"seed_{seed}"
    project_output = project_root / f"seed_{seed}"
    derived_contract = scratch_root / "contracts" / f"seed_{seed}.json"
    return scratch_output, project_output, derived_contract


def _derived_parent_contract(
    replication_contract: Mapping[str, Any],
    replication_path: Path,
    replication_sha256: str,
    seed: int,
) -> dict[str, Any]:
    root = replication_path.parents[1]
    parent_path = root / replication_contract["parent_contract"]["path"]
    parent = json.loads(parent_path.read_text())
    scratch_output, project_output, _ = _paths_for_seed(
        replication_contract,
        seed,
    )
    derived = copy.deepcopy(parent)
    derived["purpose"] = (
        "independent_seed_replication_of_pointwise_anomaly_direct_state_model_c"
    )
    derived["training"]["seed"] = seed
    derived["output_contract"]["scratch_output"] = str(scratch_output)
    derived["output_contract"]["project_output"] = str(project_output)
    derived["replication_provenance"] = {
        "version": VERSION,
        "replication_contract": str(replication_path),
        "replication_contract_sha256": replication_sha256,
        "parent_contract": str(parent_path),
        "parent_contract_sha256": replication_contract["parent_contract"][
            "sha256"
        ],
        "reference_seed": 20260723,
        "replication_seed": seed,
        "change_from_parent": "seed_only",
        "checkpoint_selection": "training_split1_only",
        "fixed_validation_members_changed": False,
    }
    # The replication contract independently verifies the complete immutable
    # source set.  The derived runtime contract lives in scratch, so retaining
    # repository-relative source paths would resolve against the wrong root.
    derived["source_hashes"] = {}
    return derived


def _materialize_derived_contract(
    contract: Mapping[str, Any],
    resolved: Path,
    digest: str,
    seed: int,
) -> tuple[Path, str]:
    _, _, path = _paths_for_seed(contract, seed)
    derived = _derived_parent_contract(
        contract,
        resolved,
        digest,
        seed,
    )
    text = json.dumps(derived, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise ModelCAnomalyDirectReplicationError(
                f"derived seed contract changed: {path}"
            )
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        os.replace(temporary, path)
    return path, _json_sha256(derived)


def preflight_replication(
    dataset_path: str | Path,
    quality_report_path: str | Path,
    contract_path: str | Path,
    array_index: int,
) -> dict[str, Any]:
    """Verify one array task without opening the fixed S2 member states."""

    contract, resolved, digest = load_replication_contract(contract_path)
    seed = resolve_replication_seed(resolved, array_index)
    scratch_output, project_output, _ = _paths_for_seed(contract, seed)
    if project_output.exists():
        raise FileExistsError(
            f"replication project output already exists: {project_output}"
        )
    derived_path, derived_content_sha = _materialize_derived_contract(
        contract,
        resolved,
        digest,
        seed,
    )
    result = preflight_anomaly_direct(
        dataset_path,
        quality_report_path,
        derived_path,
        scratch_output,
    )
    return {
        "status": "ready",
        "version": VERSION,
        "array_index": array_index,
        "seed": seed,
        "replication_contract": str(resolved),
        "replication_contract_sha256": digest,
        "derived_contract": str(derived_path),
        "derived_contract_content_sha256": derived_content_sha,
        "scratch_output": str(scratch_output),
        "project_output": str(project_output),
        "normalization": result["normalization"],
        "validation_state_opened": False,
        "inference_state_opened": False,
    }


def run_replication(
    dataset_path: str | Path,
    quality_report_path: str | Path,
    contract_path: str | Path,
    array_index: int,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run one seed-only replica under the frozen parent experiment."""

    contract, resolved, digest = load_replication_contract(contract_path)
    seed = resolve_replication_seed(resolved, array_index)
    scratch_output, project_output, _ = _paths_for_seed(contract, seed)
    if project_output.exists():
        raise FileExistsError(
            f"replication project output already exists: {project_output}"
        )
    derived_path, derived_content_sha = _materialize_derived_contract(
        contract,
        resolved,
        digest,
        seed,
    )
    report = run_anomaly_direct(
        dataset_path,
        quality_report_path,
        derived_path,
        scratch_output,
        device_name=device_name,
    )
    decision = report["selection_decision"]
    return {
        "status": report["status"],
        "version": VERSION,
        "array_index": array_index,
        "seed": seed,
        "replication_contract": str(resolved),
        "replication_contract_sha256": digest,
        "derived_contract": str(derived_path),
        "derived_contract_content_sha256": derived_content_sha,
        "scratch_output": str(scratch_output),
        "project_output": str(project_output),
        "selected_optimizer_step": decision["selected_fine_tune_step"],
        "classification": decision["classification"],
        "training_gate_passed": decision["passed"],
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "arrays_sha256": report["arrays_sha256"],
        "validation_state_opened": report["validation_state_opened"],
        "inference_state_opened": report["inference_state_opened"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser("resolve-seed")
    resolve.add_argument("--contract", type=Path, required=True)
    resolve.add_argument("--array-index", type=int, required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--dataset", type=Path, required=True)
        child.add_argument("--quality-report", type=Path, required=True)
        child.add_argument("--contract", type=Path, required=True)
        child.add_argument("--array-index", type=int, required=True)
        if command == "run":
            child.add_argument(
                "--device",
                choices=("auto", "cpu", "cuda"),
                default="auto",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "resolve-seed":
        print(resolve_replication_seed(args.contract, args.array_index))
        return 0
    if args.command == "preflight":
        result = preflight_replication(
            args.dataset,
            args.quality_report,
            args.contract,
            args.array_index,
        )
    else:
        result = run_replication(
            args.dataset,
            args.quality_report,
            args.contract,
            args.array_index,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
