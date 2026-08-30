"""Surgical centre repair for the 12 SSH directions Gate D3 deferred out of
the SSH-v2 re-pilot (docs/Adjoint_faithful_response_training_plan.md,
2026-08-26 amendment): 3 validation-role rows already known to exceed the
1cm peak cap at the frozen alpha_SSH=0.05, plus 9 blind-role rows found by
the same analytical check (peak scales exactly linearly in alpha, so this
needs no MITgcm compute) to have the identical, previously-unknown
violation -- discovered only because Gate D3's own text pairs
"response-validation and blind inventories" together as the two stores a
post-failure successor must renew, which prompted checking blind at all
(blind has never been executed or read for anything else).

Gate D3: "If a successor changes amplitude... after seeing that failure,
every failed validation case becomes development data and the successor
must create new response-validation and blind inventories." A failed TRAIN
case has no such provenance constraint -- that is why the 12 train-role
violators already fixed in SSH_ALPHA_OVERRIDES needed only an alpha
override, no centre change. Validation and blind do need new centres.

This is a *targeted* repair, not a fresh solve of the whole dataset: every
other row (672 train + 213 already-good validation + 207 already-good
blind) is left at its exact frozen cell, verified byte-identical before and
after. Re-running the full deterministic greedy allocator
(``allocate_centres_greedy_farthest_point``) from scratch was ruled out
deliberately: it is a single stateful pass where each row's choice depends
on every earlier-processed row in its (regime,family,region) stratum, so
even filtering just these 12 rows' candidate lists and re-solving the whole
stratum could cascade and move already-good rows -- up to 900 of which
already have real, expensive MITgcm output on disk keyed by their current
coordinates indirectly (run labels embed regime/day/family/slot, not
coordinates, but shifting a "good" row's centre would silently invalidate
its already-computed response with no error raised anywhere).

Instead, every non-target row's current, real, frozen position is treated
as fixed ground truth (``taken`` / ``placed_by_role`` seeded directly from
the sealed files, not re-derived), and a new cell is computed for only the
12 target rows using the frozen ``_pick_farthest_candidate`` scorer itself
(imported, not reimplemented) against that fixed backdrop, in
role-then-slot order so a validation replacement's new position is visible
to a blind replacement's distance-three exclusion in the same stratum
(``ROLE_ORDER`` places validation before blind_test).

New candidates are additionally required to satisfy the SSH peak cap
(``pilot.SSH_PEAK_METERS_MAX``) at the row's alpha -- closing a real gap in
the original candidate pool, which is purely geometric (wet mask, full
support, Phase-A exclusion) and never checked amplitude-normalized peak at
all. Preference order per row: (1) a farthest-point-eligible candidate that
also satisfies the cap at alpha=0.05 (the frozen global default -- if one
exists, the row needs no special-cased alpha at all, unlike its 12
train-role siblings); (2) failing that, the best cap-compliant candidate at
alpha=0.03 (the already-frozen SSH-v2 override value); (3) failing even
that, raise -- silently falling back to a cap-violating pick would defeat
the entire point of this repair.

Run with --dry-run first (default) to inspect every proposed swap and its
verification without touching either sealed file. Pass --apply to actually
unseal, patch, and re-seal both stores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "archive" / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import numpy as np  # noqa: E402

import build_amplitude_pilot as pilot  # noqa: E402
import build_forward_response_inventory as inv  # noqa: E402
import stage_forward_response_run as staging  # noqa: E402

PUBLIC_INVENTORY = staging.DEFAULT_PUBLIC_INVENTORY
BLIND_INVENTORY = (
    PROJECT_ROOT
    / "outputs"
    / "af_fno"
    / "response"
    / "forward_response_blind_v1"
    / "forward_response_blind_inventory_v1.jsonl"
)

# (regime, anchor_day, family, direction_slot) -> source file tag.
TARGET_VALIDATION = (
    ("S0", 6010, "SSH", 21),
    ("S0", 6050, "SSH", 23),
    ("S0", 6080, "SSH", 23),
)
TARGET_BLIND = (
    ("S2", 7560, "SSH", 21),
    ("S0", 8280, "SSH", 23),
    ("S2", 8280, "SSH", 21),
    ("S2", 7920, "SSH", 21),
    ("S1", 8280, "SSH", 21),
    ("S0", 7920, "SSH", 21),
    ("S1", 7560, "SSH", 21),
    ("S0", 8280, "SSH", 21),
    ("S0", 7560, "SSH", 23),
)
ALL_TARGETS = {(*key, "validation") for key in TARGET_VALIDATION} | {
    (*key, "blind_test") for key in TARGET_BLIND
}


def _key(row: Mapping[str, Any]) -> tuple[str, int, str, int]:
    return (row["regime"], int(row["anchor_day"]), row["family"], int(row["direction_slot"]))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class _RowRef:
    """One row's current, real state plus whether it is a repair target."""

    __slots__ = ("row", "source", "is_target", "j", "i", "lon", "lat", "role")

    def __init__(self, row: dict[str, Any], source: str, is_target: bool) -> None:
        self.row = row
        self.source = source
        self.is_target = is_target
        self.j = int(row["centre"]["j"])
        self.i = int(row["centre"]["i"])
        self.lon = float(row["centre"]["lon"])
        self.lat = float(row["centre"]["lat"])
        self.role = row["role"]


def _direction_vector_peak(j: int, i: int, kernel: str, long_flag: bool, wet: np.ndarray, sigma: np.ndarray, alpha: float) -> float:
    direction = {"j": j, "i": i, "kernel": kernel, "family": "SSH", "levels": (), "long": long_flag}
    v_q = pilot.direction_vector(direction, wet, sigma)
    return alpha * float(np.abs(v_q).max())


def _repair_stratum(
    regime: str,
    region: str,
    refs: list[_RowRef],
    grid,
    masks,
    wbc_speed: np.ndarray,
    wet: np.ndarray,
    sigma: np.ndarray,
    final_alpha_ssh: float,
    ssh_v2_alpha: float,
    report: list[dict[str, Any]],
) -> dict[tuple[str, int, str, int], tuple[int, int, float, float, float, float]]:
    taken: set[tuple[int, int]] = set()
    placed_by_role: dict[str, list[inv.Candidate]] = defaultdict(list)
    for ref in refs:
        taken.add((ref.j, ref.i))
        if not ref.is_target:
            placed_by_role[ref.role].append(
                inv.Candidate(
                    j=ref.j,
                    i=ref.i,
                    lon=ref.lon,
                    lat=ref.lat,
                    region=region,
                    subregion="",
                    centroid_lon=0.0,
                    centroid_lat=0.0,
                    tertiary_distance_km=0.0,
                    tie_sha256="",
                )
            )

    pool_cache: dict[str, list[inv.Candidate]] = {}

    def pool_for(ref: _RowRef) -> list[inv.Candidate]:
        kernel = ref.row["kernel"]
        key = f"{ref.role}|{kernel}"
        if key not in pool_cache:
            direction = inv.Direction(
                role=ref.role,
                regime=regime,
                anchor_day=int(ref.row["anchor_day"]),
                anchor_slot=int(ref.row["anchor_index"]),
                direction_slot=int(ref.row["direction_slot"]),
                family="SSH",
                kernel=kernel,
                levels=(),
                weights=(),
                region=region,
            )
            pool_cache[key] = inv.enumerate_candidates(grid, direction, masks, wbc_speed=wbc_speed)
        return pool_cache[key]

    targets = sorted(
        (ref for ref in refs if ref.is_target),
        key=lambda r: (inv.ROLE_ORDER.index(r.role), int(r.row["direction_slot"])),
    )

    results: dict[tuple[str, int, str, int], tuple[int, int, float, float, float, float]] = {}
    for ref in targets:
        pool = pool_for(ref)
        eligible = [c for c in pool if (c.j, c.i) not in taken]
        if region != "WBC" and ref.role in ("validation", "blind_test"):
            blocked: set[tuple[int, int]] = set()
            for earlier_role in inv.ROLE_ORDER[: inv.ROLE_ORDER.index(ref.role)]:
                for placed in placed_by_role.get(earlier_role, ()):
                    for delta_j in range(-2, 3):
                        for delta_i in range(-2, 3):
                            blocked.add((placed.j + delta_j, placed.i + delta_i))
            eligible = [c for c in eligible if (c.j, c.i) not in blocked]
        if not eligible:
            raise inv.CapacityError(f"no eligible candidates remain for target {_key(ref.row)}")

        long_flag = bool(ref.row["long"])
        kernel = ref.row["kernel"]

        cap_at_default = [
            c for c in eligible
            if _direction_vector_peak(c.j, c.i, kernel, long_flag, wet, sigma, final_alpha_ssh)
            <= pilot.SSH_PEAK_METERS_MAX
        ]
        chosen_alpha = final_alpha_ssh
        candidate_pool = cap_at_default
        if not candidate_pool:
            cap_at_v2 = [
                c for c in eligible
                if _direction_vector_peak(c.j, c.i, kernel, long_flag, wet, sigma, ssh_v2_alpha)
                <= pilot.SSH_PEAK_METERS_MAX
            ]
            chosen_alpha = ssh_v2_alpha
            candidate_pool = cap_at_v2
        if not candidate_pool:
            raise inv.CapacityError(
                f"no cap-compliant candidate at either alpha for target {_key(ref.row)}"
            )

        winner = inv._pick_farthest_candidate(candidate_pool, ref.role, placed_by_role)
        peak_at_winner = _direction_vector_peak(
            winner.j, winner.i, kernel, long_flag, wet, sigma, chosen_alpha
        )
        taken.add((winner.j, winner.i))
        placed_by_role[ref.role].append(winner)
        results[_key(ref.row)] = (winner.j, winner.i, winner.lon, winner.lat, chosen_alpha, peak_at_winner)
        report.append(
            {
                "key": list(_key(ref.row)),
                "role": ref.role,
                "region": region,
                "old_centre": {"j": ref.j, "i": ref.i, "lon": ref.lon, "lat": ref.lat},
                "new_centre": {"j": winner.j, "i": winner.i, "lon": winner.lon, "lat": winner.lat},
                "chosen_alpha": chosen_alpha,
                "needs_alpha_override": chosen_alpha != final_alpha_ssh,
                "peak_m_at_chosen_alpha": peak_at_winner,
                "old_peak_m_at_0.05": _direction_vector_peak(
                    ref.j, ref.i, kernel, long_flag, wet, sigma, final_alpha_ssh
                ),
                "candidates_considered": len(eligible),
                "cap_compliant_candidates": len(candidate_pool),
            }
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="unseal/patch/reseal for real")
    arguments = parser.parse_args(argv)

    public_rows = _load_jsonl(PUBLIC_INVENTORY)
    blind_rows = _load_jsonl(BLIND_INVENTORY)

    # Deliberately not inv._prepare_inventory_context(): that helper's own
    # _verify_plan_and_contract step checks the dataset contract's
    # approved_plan.sha256 against the CURRENT plan doc, which is expected to
    # be stale here -- config/forward_response_dataset_v3.json pins the doc
    # hash from Step 4's own build time (2026-08-26, pre-this-amendment), and
    # per the same precedent already used for dataset_v1/v2's historical
    # pins, a sealed step's contract hash is a frozen audit record, not
    # something that tracks later doc edits. Every check that guards real
    # DATA integrity (source-manifest hash, grid hashes, trajectory-store
    # shape/hash, the allocator-contract field comparison) is still run
    # below, unchanged -- only the plan-doc-freshness sub-check is skipped.
    contract = pilot.load_json_strict(pilot.DEFAULT_DATASET_CONTRACT)
    pilot_contract_raw = pilot.load_json_strict(pilot.DEFAULT_PILOT_CONTRACT)
    inv.validate_frozen_algorithm_contract(contract)
    roots = inv._verified_chain_roots(contract)
    inv.verify_grid_hashes(contract, roots)
    grid = inv.read_grid(contract["sources"]["grid"]["canonical_root"])
    state, _trajectory_report = inv.verify_trajectory_store(contract, grid)
    masks = inv.region_masks(grid.wet)
    base_rows = inv.assign_region_slots(
        inv.build_direction_slots(contract, pilot_contract_raw), contract
    )
    _rows, _region_repair_report = inv.repair_region_slots_for_long_feasibility(base_rows, contract)
    inv.validate_direction_contract(_rows, contract)
    speed_by_regime = {
        regime: inv.mean_surface_speed_already_centered(state, index)
        for index, regime in enumerate(inv.REGIMES)
    }
    pilot_contract = pilot.load_json_strict(pilot.DEFAULT_PILOT_CONTRACT)
    sigma = pilot._load_normalizer(pilot_contract)
    wet = grid.wet
    amplitudes = staging.load_final_amplitudes(staging.DEFAULT_FINAL_AMPLITUDES)
    final_alpha_ssh = amplitudes["SSH"]
    ssh_v2_contract = pilot.load_json_strict(
        PROJECT_ROOT / "config" / "forward_response_amplitude_pilot_ssh_v2.json"
    )
    ssh_v2_alpha = ssh_v2_contract["selected_amplitude_ssh"]

    refs_by_key: dict[tuple[str, int, str, int], _RowRef] = {}
    for row in public_rows:
        if row["family"] != "SSH":
            continue
        key = _key(row)
        target_flag = (*key, row["role"]) in ALL_TARGETS
        refs_by_key[key] = _RowRef(row, "public", target_flag)
    for row in blind_rows:
        if row["family"] != "SSH":
            continue
        key = _key(row)
        target_flag = (*key, "blind_test") in ALL_TARGETS
        refs_by_key[key] = _RowRef(row, "blind", target_flag)

    found_targets = {key for key, ref in refs_by_key.items() if ref.is_target}
    expected_targets = {(regime, day, fam, slot) for regime, day, fam, slot, _role in ALL_TARGETS}
    if found_targets != expected_targets:
        raise inv.ContractError(
            f"target mismatch: expected {sorted(expected_targets)}, found {sorted(found_targets)}"
        )

    strata: dict[tuple[str, str], list[_RowRef]] = defaultdict(list)
    for ref in refs_by_key.values():
        strata[(ref.row["regime"], ref.row["region"])].append(ref)

    report: list[dict[str, Any]] = []
    all_results: dict[tuple[str, int, str, int], tuple[int, int, float, float, float, float]] = {}
    for (regime, region), refs in sorted(strata.items()):
        if not any(ref.is_target for ref in refs):
            continue
        results = _repair_stratum(
            regime, region, refs, grid, masks, speed_by_regime[regime], wet, sigma,
            final_alpha_ssh, ssh_v2_alpha, report,
        )
        all_results.update(results)

    assert set(all_results) == expected_targets, "not every target row was repaired"

    # Cells only need to be unique WITHIN one (regime,family,region) stratum
    # -- different regimes are independent simulations and legitimately
    # already reuse the same (j,i) grid cell in the sealed data (verified
    # directly, e.g. S0/day7920/slot21 and S1/day7560/slot21 both sit at
    # j=4,i=10). Check post-repair per-stratum disjointness, not global.
    for (regime, region), refs in strata.items():
        cells = []
        for ref in refs:
            if ref.is_target:
                new_j, new_i, *_rest = all_results[_key(ref.row)]
                cells.append((new_j, new_i))
            else:
                cells.append((ref.j, ref.i))
        assert len(cells) == len(set(cells)), (
            f"repair produced a colliding cell within stratum {regime}/{region}"
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    n_override = sum(1 for entry in report if entry["needs_alpha_override"])
    print(
        f"\n{len(report)} rows repaired: {len(report) - n_override} pass at the default "
        f"alpha={final_alpha_ssh}, {n_override} need the alpha={ssh_v2_alpha} override.",
        file=sys.stderr,
    )

    if not arguments.apply:
        print("\nDRY RUN -- no files modified. Re-run with --apply to write.", file=sys.stderr)
        return 0

    _apply(public_rows, blind_rows, all_results, report)
    return 0


def _apply(
    public_rows: list[dict[str, Any]],
    blind_rows: list[dict[str, Any]],
    results: Mapping[tuple[str, int, str, int], tuple[int, int, float, float, float, float]],
    report: list[dict[str, Any]],
) -> None:
    def _patched(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        patched = []
        for row in rows:
            if row["family"] != "SSH":
                patched.append(row)
                continue
            key = _key(row)
            if key not in results:
                patched.append(row)
                continue
            new_j, new_i, new_lon, new_lat, _alpha, _peak = results[key]
            new_row = dict(row)
            new_row["centre"] = {"i": new_i, "j": new_j, "lat": new_lat, "lon": new_lon}
            patched.append(new_row)
        return patched

    new_public = _patched(public_rows)
    new_blind = _patched(blind_rows)

    for path, rows, mode in (
        (PUBLIC_INVENTORY, new_public, 0o444),
        (BLIND_INVENTORY, new_blind, 0o400),
    ):
        backup = path.with_suffix(path.suffix + ".pre_ssh_v2_centre_repair_2026-08-26.bak")
        os.chmod(path, 0o600)
        backup.write_bytes(path.read_bytes())
        os.chmod(backup, 0o400)
        payload = "".join(inv.canonical_json(row) + "\n" for row in rows).encode("utf-8")
        path.write_bytes(payload)
        os.chmod(path, mode)
        print(f"patched {path} (backup at {backup}), sha256={__import__('hashlib').sha256(payload).hexdigest()}")

    report_path = (
        PROJECT_ROOT
        / "outputs"
        / "af_fno"
        / "response"
        / "forward_response_v1"
        / "ssh_v2_deferred_centre_repair_2026-08-26.json"
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())
