"""Check that the sharded sampler partitions every optimizer step exactly.

Needs no GPU and no trajectory store: it drives the real
:class:`ChunkAwareBatchSampler` and :class:`ShardedBatchSampler` over a stand-in
whose ``records`` have the production shape, and asserts three things a wrong
shard would break --- equal per-rank stream lengths, a disjoint and complete
cover of the global order, and, per optimizer step, a union over ranks equal to
exactly that step's global microbatches.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "turbulent_double_gyre" / "src"))

from turbfno.distributed import (  # noqa: E402
    DistributedError,
    ShardedBatchSampler,
    Topology,
    require_divisible,
)
from turbfno.runtime import ChunkAwareBatchSampler  # noqa: E402
from turbfno.train import (  # noqa: E402
    EXPERIMENTS,
    GRADIENT_ACCUMULATION_STEPS,
    MICROBATCH_SIZE,
    SEED,
    TRAINING_STARTS_PER_REGIME,
)


class _Records:
    """Only the attribute the sampler reads: regime-major, day-ascending."""

    def __init__(self) -> None:
        self.records = [
            (regime, day)
            for regime in range(len(EXPERIMENTS))
            for day in range(TRAINING_STARTS_PER_REGIME)
        ]


def check(world: int) -> bool:
    dataset = _Records()
    topologies = [Topology(rank, world, rank) for rank in range(world)]
    local = require_divisible(GRADIENT_ACCUMULATION_STEPS, topologies[0])
    streams = [
        list(
            iter(
                ShardedBatchSampler(
                    ChunkAwareBatchSampler(dataset, MICROBATCH_SIZE, SEED), topology
                )
            )
        )
        for topology in topologies
    ]
    reference = list(iter(ChunkAwareBatchSampler(dataset, MICROBATCH_SIZE, SEED)))

    equal_lengths = {len(s) for s in streams} == {len(reference) // world}
    flat = [tuple(b) for stream in streams for b in stream]
    covers = len(flat) == len(reference) and len(set(flat)) == len(reference)
    partitions = True
    for step in range(len(reference) // GRADIENT_ACCUMULATION_STEPS):
        lo = step * GRADIENT_ACCUMULATION_STEPS
        want = {tuple(b) for b in reference[lo : lo + GRADIENT_ACCUMULATION_STEPS]}
        got: set = set()
        for rank in range(world):
            got |= {tuple(b) for b in streams[rank][step * local : (step + 1) * local]}
        if want != got:
            partitions = False
            break

    ok = equal_lengths and covers and partitions
    print(
        f"world {world}: {len(reference) // world} microbatches per rank, "
        f"{local} per optimizer step | lengths "
        f"{'ok' if equal_lengths else 'BAD'} | cover "
        f"{'exact' if covers else 'BROKEN'} | step partition "
        f"{'exact' if partitions else 'BROKEN'}"
    )
    return ok


def refused(world: int) -> bool:
    """A world size the guards must reject rather than silently mis-shard."""

    try:
        check(world)
    except DistributedError as exc:
        print(f"world {world}: refused, as it must -- {exc}")
        return True
    print(f"world {world}: NOT refused; a divisibility guard is broken")
    return False


def main() -> int:
    # 8 microbatches per optimizer step and 17,820 in the epoch: a usable world
    # size has to divide both, which leaves 1, 2 and 4.
    ok = all(check(world) for world in (1, 2, 4))
    ok &= refused(3)   # does not divide the optimizer step
    ok &= refused(8)   # divides the step but not the epoch
    print("SAMPLER SHARDING:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
