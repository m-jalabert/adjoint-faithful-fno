"""Data-parallel training across the GPUs of one node.

The operator is small enough to replicate and the gradient small enough to move
that the textbook :class:`~torch.nn.parallel.DistributedDataParallel` buys
almost nothing here, and it carries real risk: the model is called seven times
per microbatch --- six graph-building rollout calls and one ``no_grad``
amplification probe --- before a single backward, and its spectral weight is a
:mod:`torch.nn.utils.parametrize` parametrization whose buffers advance in place
on every forward.  DDP's per-forward bookkeeping and buffer broadcast both
assume one forward per backward, so it would have to be disarmed on every call
that is not the last.

So this module does the one thing that is actually needed: after every rank has
finished its own share of a step's microbatches, average the gradients.  The
allreduce is 794 MiB, measured at 20.7 ms on two V100s and 10.5 ms on four over
this node's full NVLink mesh, against a step of 9.6 s and 4.8 s respectively --
0.2 per cent either way.  What DDP's overlap-with-backward would recover is
smaller than the noise, and the exactness is auditable in ten lines.

Sharding is exact rather than approximate because every term of the objective
reduces over its non-batch dimensions *before* the batch mean, so a sum of
per-rank means weighted equally is the mean over the whole batch --- provided
each rank carries the same number of samples.  That is required, not assumed:
:func:`shard_microbatches` refuses a world size that does not divide the step's
microbatch count.
"""
from __future__ import annotations

import datetime
import os
from typing import Any, Sequence

from .runtime import torch

try:  # pragma: no cover - environment dependent
    import torch.distributed as dist
except (ImportError, OSError):  # pragma: no cover
    dist = None  # type: ignore[assignment]

#: Collectives are only ever called between optimizer steps, so the default
#: half-hour is ample; the long tail (rank 0 validating alone) is deliberately
#: outside the process group's lifetime.
TIMEOUT = datetime.timedelta(minutes=30)


class DistributedError(RuntimeError):
    """Raised when the data-parallel launch is inconsistent with the contract."""


class Topology:
    """The rank's place in the run, and the collectives it needs.

    A single-process run builds one of these too, with ``world_size == 1``; every
    method is then a no-op, so the training loop has no branches in it.
    """

    def __init__(self, rank: int, world_size: int, local_rank: int) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def average_gradients(self, model: Any) -> None:
        """Replace every rank's gradient with the mean over ranks."""

        if not self.distributed:
            return
        for parameter in model.parameters():
            if parameter.grad is None:
                # A parameter with no gradient on *any* rank is fine; one with a
                # gradient on some ranks only would desynchronize the replicas,
                # and the spectral cap is the only thing that can produce that
                # (a mode already at or below one has a constant scale). It is
                # a scale, not a parameter, so `original` always receives one.
                raise DistributedError(
                    "a parameter carries no gradient; the replicas would diverge"
                )
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad /= self.world_size

    def average_scalar(self, value: float) -> float:
        """Mean of a per-rank scalar, for the reported training window."""

        if not self.distributed:
            return float(value)
        tensor = torch.tensor(
            [float(value)], dtype=torch.float64, device=f"cuda:{self.local_rank}"
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item()) / self.world_size

    def average_direction(self, direction: Any) -> Any:
        """Share one power-iteration direction across ranks.

        The single-process loop advances one direction sequentially through the
        step's microbatches. Sharded, each rank advances its own through its own
        subset, so they are averaged and renormalized back to the unit sphere at
        the end of every step. The quantity is a diagnostic --- it carries no
        gradient and enters no loss --- so this is a declared difference in how
        ``mean_single_call_amplification`` is estimated, not a change to training.
        """

        if not self.distributed:
            return direction
        shared = direction.clone()
        dist.all_reduce(shared, op=dist.ReduceOp.SUM)
        norm = shared.norm()
        return direction if float(norm) <= 0.0 else shared / norm

    def shutdown(self) -> None:
        if self.distributed and dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def initialize(device_name: str = "auto") -> Topology:
    """Join the process group described by the launcher's environment.

    Recognizes both ``torchrun`` and ``srun`` conventions. Absent either, this is
    a single-process run and no process group is created at all.
    """

    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID"))
    world = os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS"))
    local = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID"))
    if rank is None or world is None or int(world) <= 1:
        return Topology(0, 1, 0 if local is None else int(local))
    if dist is None:  # pragma: no cover - environment dependent
        raise DistributedError("torch.distributed is unavailable in this environment")
    if device_name == "cpu":
        raise DistributedError("data-parallel training requires CUDA devices")
    local_rank = int(local) if local is not None else int(rank)
    torch.cuda.set_device(local_rank)
    # Rank and size are passed explicitly rather than left to env:// so an srun
    # launch, which sets SLURM_PROCID but not RANK, works without a shim; only
    # the rendezvous address still comes from the environment.
    dist.init_process_group(
        backend="nccl",
        timeout=TIMEOUT,
        world_size=int(world),
        rank=int(rank),
    )
    topology = Topology(int(rank), int(world), local_rank)
    if dist.get_world_size() != topology.world_size:
        raise DistributedError("the launcher and the process group disagree on size")
    return topology


class ShardedBatchSampler:
    """One rank's stride through the global microbatch order.

    Wraps :class:`~turbfno.runtime.ChunkAwareBatchSampler` rather than replacing
    it, so the chunk-local batching and the per-epoch reshuffle are the parent's,
    unchanged.  Every rank reproduces the *same* shuffled order --- same seed,
    same epoch counter, and they iterate in lockstep --- and then takes the
    microbatches at positions congruent to its rank.  Microbatch ``j`` of every
    step therefore goes to rank ``j mod world_size``, so the union over ranks of
    one step is exactly the step the single-process loop would have run, summed
    in a different order and in no other way different.
    """

    def __init__(self, parent: Any, topology: Topology) -> None:
        total = len(parent.batches)
        # The ranks must also reach the end of an epoch together. If the stream
        # does not divide evenly, they exhaust their iterators at different
        # steps, restart at different times, and from the first epoch boundary
        # onward the per-step union is no longer the step the contract declares.
        # 17,820 training microbatches divide by 1, 2 and 4 but not by 8.
        if total % topology.world_size:
            raise DistributedError(
                f"{total} microbatches do not divide evenly across "
                f"{topology.world_size} ranks; the ranks would reach the epoch "
                f"boundary at different optimizer steps"
            )
        self.parent = parent
        self.topology = topology

    def __len__(self) -> int:
        return len(self.parent.batches) // self.topology.world_size

    def __iter__(self):
        import random as _random

        order = list(range(len(self.parent.batches)))
        _random.Random(self.parent.seed + self.parent.epoch).shuffle(order)
        self.parent.epoch += 1
        for index in order[self.topology.rank :: self.topology.world_size]:
            yield list(self.parent.batches[index])


def require_divisible(accumulation_total: int, topology: Topology) -> int:
    """Return this rank's microbatches per optimizer step.

    Exactness needs equal per-rank sample counts: the objective's terms are
    per-sample means, so an unweighted mean of per-rank means is the mean over
    the whole batch only when the ranks carry the same number of samples.  A
    world size that does not divide the step is refused rather than silently
    reweighted.
    """

    if accumulation_total % topology.world_size:
        raise DistributedError(
            f"{accumulation_total} microbatches per optimizer step do not divide "
            f"evenly across {topology.world_size} ranks; use a world size that "
            f"divides {accumulation_total}"
        )
    return accumulation_total // topology.world_size


__all__ = [
    "DistributedError",
    "Topology",
    "ShardedBatchSampler",
    "initialize",
    "require_divisible",
]
