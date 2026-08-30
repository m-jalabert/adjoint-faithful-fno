"""Spectral power-vector snapshot/restore around an auxiliary response chain
(plan section 15.2).

Every ``ModeSpectralNorm.forward`` call mutates its ``left``/``right``
power-iteration buffers *in place* (``spectral_norm.py``'s own docstring:
"This mutates ``self.left``/``self.right`` in place"), tracking the
operator's evolving dominant singular direction across every training
forward -- nominal and auxiliary alike. Section 15.2 requires the auxiliary
(response) chain to leave that persistent state exactly as if it had never
run: "Immediately before an auxiliary chain, snapshot every spectral
left/right power vector; allow the parent's usual two power iterations on
each batched lead; backpropagate the auxiliary loss; then restore every
vector bit-for-bit before the optimizer step, in a `finally` guard."

The reason is architectural, not incidental: the buffers are a *shared*,
order-dependent estimator of the current weights' dominant singular
direction. If the auxiliary chain's own forward passes were allowed to
advance them, the *next* nominal microbatch's power iteration would start
from a direction contaminated by the auxiliary batch's own statistics --
making B and C's nominal training path silently diverge in something
section 5.1 requires identical (nothing about the buffers is declared as a
permitted delta). Weight *gradients* from the auxiliary loss are kept
(``retain_weight_gradients``): only the estimator state is rolled back.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .runtime import torch
from .spectral_norm import NormalizedSpectralWeight, spectral_convolutions


class SpectralContextError(RuntimeError):
    """Raised when the spectral power-vector snapshot/restore cannot proceed."""


def _normalized_weights(model: Any) -> list[NormalizedSpectralWeight]:
    holders = []
    for conv in spectral_convolutions(model):
        holder = conv.weight
        if not isinstance(holder, NormalizedSpectralWeight):
            raise SpectralContextError("model has no installed per-mode spectral normalization")
        holders.append(holder)
    return holders


def snapshot_power_vectors(model: Any) -> list[tuple[Any, Any]]:
    """One ``(left, right)`` clone pair per spectral block, in block order."""

    with torch.no_grad():
        return [(holder.norm.left.clone(), holder.norm.right.clone()) for holder in _normalized_weights(model)]


def restore_power_vectors(model: Any, snapshot: list[tuple[Any, Any]]) -> None:
    """Bit-for-bit ``.copy_()`` restore -- must run in a ``finally`` guard so
    a non-finite auxiliary loss (caught upstream) can never leave the
    buffers advanced."""

    holders = _normalized_weights(model)
    if len(holders) != len(snapshot):
        raise SpectralContextError(f"snapshot has {len(snapshot)} blocks, model has {len(holders)}")
    with torch.no_grad():
        for holder, (left, right) in zip(holders, snapshot):
            holder.norm.left.copy_(left)
            holder.norm.right.copy_(right)


def buffer_hash(model: Any) -> str:
    """A cheap fingerprint of every block's buffers, for the branch-order-
    invariance / zero-mutation tests section 15.2 requires -- not used on
    the training hot path."""

    import hashlib

    digest = hashlib.sha256()
    with torch.no_grad():
        for holder in _normalized_weights(model):
            digest.update(holder.norm.left.detach().cpu().numpy().tobytes())
            digest.update(holder.norm.right.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def auxiliary_chain(model: Any) -> Iterator[None]:
    """Snapshot before, restore after -- in a ``finally`` guard so an
    exception raised mid-chain (e.g. a non-finite auxiliary loss) still
    leaves the buffers untouched, matching section 15.2's own guard
    requirement exactly."""

    snapshot = snapshot_power_vectors(model)
    try:
        yield
    finally:
        restore_power_vectors(model, snapshot)
