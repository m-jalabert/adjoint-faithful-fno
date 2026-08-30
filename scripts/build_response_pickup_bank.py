"""Execution step 6 of docs/Adjoint_faithful_response_training_plan.md.

Resolves and hash-verifies the day-5,760 pickup for one regime, runs the
320-day unperturbed bridge segment via
``bire_repro.af_response_pickup_bank``, archives every 10-day pickup, and
verifies the three retained response-validation source days
(6,010/6,050/6,080) reproduce trajectory-v3's P32 projection exactly
(Gate D0, section 7.1).

This step is pure nominal MITgcm data generation: it never reads an adjoint,
TAF, or blind-response artifact, so it does not need the study's separate
development/evaluator identity machinery (that only protects the later
stages that touch adjoint truth or model selection -- see plan section
18.1/24). Keeping it light on purpose: this only reuses the source-pickup
resolver and grid/trajectory verifier from
``build_forward_response_inventory.py``, not that script's full
direction/quota-allocation machinery (which is Step 4's concern, not
Step 6's).

``resolve`` is read-only: it reports which source pickup each regime would
use and its hash, without staging or running anything.

``run`` stages and executes one regime's bridge segment plus the post-run
P32 verification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "archive" / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from bire_repro import af_response_pickup_bank as bank  # noqa: E402
from build_forward_response_inventory import (  # noqa: E402
    REGIMES,
    InventoryError,
    SourceError,
    _verified_chain_roots,
    load_json_strict,
    pickup_to_trajectory_p32,
    read_grid,
    resolve_annual_pickup,
    verify_trajectory_store,
)


DEFAULT_DATASET_CONTRACT = PROJECT_ROOT / "config" / "forward_response_dataset_v1.json"
DEFAULT_SCRATCH_ROOT = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno")
DEFAULT_EXECUTABLE = PROJECT_ROOT / "build" / "af_s0" / "mitgcmuv"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "outputs" / "af_fno" / "response" / "forward_response_v1"


class PickupBankOrchestratorError(RuntimeError):
    """Raised when the pickup bank cannot legitimately proceed."""


def _load_sources(dataset_contract_path: Path):
    """The minimal (contract, chain_roots, grid, trajectory_state) tuple."""

    contract = load_json_strict(dataset_contract_path)
    roots = _verified_chain_roots(contract)
    grid = read_grid(contract["sources"]["grid"]["canonical_root"])
    state, _report = verify_trajectory_store(contract, grid)
    return contract, roots, grid, state


def resolve_sources(dataset_contract_path: Path = DEFAULT_DATASET_CONTRACT) -> dict[str, Any]:
    """Report the day-5,760 source pickup each regime would use. Read-only."""

    _contract, roots, _grid, _state = _load_sources(dataset_contract_path)
    resolutions: dict[str, Any] = {}
    for regime in REGIMES:
        resolution = resolve_annual_pickup(regime, bank.SOURCE_DAY, roots)
        resolutions[regime] = {
            "meta_path": resolution.canonical.meta_path,
            "data_path": resolution.canonical.data_path,
            "meta_sha256": resolution.canonical.meta_sha256,
            "data_sha256": resolution.canonical.data_sha256,
            "canonical_choice_reason": resolution.canonical_choice_reason,
        }
    return {"mode": "resolve", "source_day": bank.SOURCE_DAY, "regimes": resolutions}


def run_regime(
    regime: str,
    *,
    project_root: Path = PROJECT_ROOT,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    executable: Path = DEFAULT_EXECUTABLE,
    dataset_contract_path: Path = DEFAULT_DATASET_CONTRACT,
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> dict[str, Any]:
    """Stage, run, and P32-verify one regime's bridge segment (Gate D0)."""

    _contract, roots, grid, state = _load_sources(dataset_contract_path)
    resolution = resolve_annual_pickup(regime, bank.SOURCE_DAY, roots)
    manifest = bank.prepare_segment(
        project_root,
        scratch_root,
        executable,
        regime,
        Path(resolution.canonical.meta_path),
        Path(resolution.canonical.data_path),
        resolution.canonical.meta_sha256,
        resolution.canonical.data_sha256,
    )
    result = bank.run_segment(manifest)

    projection_failures: list[str] = []
    verified_days: list[int] = []
    for entry in result["retained_pickups"]:
        day = int(entry["day"])
        projected = pickup_to_trajectory_p32(entry["meta_path"], grid.wet)
        truth = np.asarray(state[REGIMES.index(regime), day], dtype=np.float32)
        if not np.array_equal(projected, truth):
            projection_failures.append(f"{regime}/day{day}")
        verified_days.append(day)
    if set(verified_days) != set(bank.RETAINED_ANCHOR_DAYS):
        raise PickupBankOrchestratorError(
            f"expected P32 verification for days {bank.RETAINED_ANCHOR_DAYS}, got {verified_days}"
        )
    if projection_failures:
        raise SourceError(
            "response-validation pickup P32 projection differs from trajectory-v3: "
            + ", ".join(projection_failures)
        )

    report = {
        "mode": "run",
        "version": "af_response_pickup_bank_v1",
        "regime": regime,
        "manifest": manifest,
        "result": {key: value for key, value in result.items() if key != "archived_pickups"},
        "archived_pickup_count": len(result["archived_pickups"]),
        "retained_days_p32_verified_against_trajectory_v3": verified_days,
        "gate_d0_pickup_bank_status": "pass",
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / f"pickup_bank_verification_{regime}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("resolve")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--regime", choices=bank.REGIMES, required=True)
    run_parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    run_parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    run_parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    arguments = parser.parse_args(argv)
    try:
        if arguments.mode == "resolve":
            result = resolve_sources()
        else:
            result = run_regime(
                arguments.regime,
                project_root=arguments.project_root.resolve(),
                scratch_root=arguments.scratch_root.resolve(),
                executable=arguments.executable.resolve(),
            )
    except (InventoryError, bank.PickupBankError, PickupBankOrchestratorError) as error:
        print(f"RESPONSE PICKUP BANK: FAIL -- {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
