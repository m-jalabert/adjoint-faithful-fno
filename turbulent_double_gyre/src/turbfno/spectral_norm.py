"""Per-mode spectral normalization of the Fourier convolution operators.

Inside one ``SpectralConv`` the channel mixing at Fourier mode ``k = (ky, kx)``
is a dense complex matrix::

    y_hat(k) = R_k x_hat(k),        R_k in C^{128 x 128}

and there are 32 x 17 = 544 such matrices per block, 1,632 across the three
blocks. They hold 103,809,024 of the model's 104,368,296 parameters --- 99.46 % ---
and nothing in the objective constrains their gain. If ``sigma_max(R_k) > 1`` for
a dynamically important low-frequency mode, that block can amplify perturbations
in that Fourier/channel direction on every autoregressive call.

This module caps it, following McCabe et al., *Towards Stability of
Autoregressive Neural Operators* (arXiv:2306.10619)::

    R_k_tilde = R_k * min(1, 1 / sigma_max(R_k))

One-sided by construction: a mode whose largest singular value is already below
one is left exactly alone, and the scale is then a constant so no gradient flows
through it. Only modes that would amplify are touched.

**This is a reparameterization of the operator, not a trajectory penalty.** It
does *not* impose ``||J_F|| <= 1`` on the emulator as a whole: each block is
``GELU(K_R h + W h)`` followed by a Channel MLP and its own skip, so the model
can still represent legitimate transient amplification, waves and boundary
dynamics. It constrains the gain of one constituent --- the global Fourier
operator --- rather than declaring that perturbation growth is always wrong.

**Estimation.** A full SVD of 1,632 complex 128x128 matrices per step is out of
the question, so each mode carries a persistent pair of singular-vector
estimates advanced by alternating power iteration on every forward, as
``torch.nn.utils.spectral_norm`` does for a real weight matrix. With ``u`` and
``v`` detached, ``sigma = Re(u^H A v)`` still carries gradient through ``A``.

The consequence is that the cap holds on a running *estimate*, so the true
``sigma_max`` tracks one with a small lag rather than being clamped exactly.
Power iteration converges from below, so the lag is always an overshoot. At
construction, after the declared warmup, the true post-cap ``sigma_max`` is
1.00009; one optimizer step later it was measured at 1.0022 with a single
iteration per forward, which is why two are used.

**Convention.** neuralop stores the weight as ``(in, out, ky, kx)`` and contracts
``y_o = sum_i x_i W_{i,o}``, so the operator acting on channels is ``A = W^T``.
Since ``sigma_max(W^T) = sigma_max(W)`` the iteration runs on ``W`` directly.

**Materialization.** The normalized weights are baked into the tensor before any
checkpoint is written, so a published checkpoint loads into a plain
:class:`~turbfno.model.ProductionFNO` with no reparameterization attached and
the inference layer is simply ``y_hat(k) = R_k_tilde x_hat(k)``. Its adjoint is
then ``R_k_tilde^H``, with no clipping, no stochastic perturbation and no extra
branch --- which is the point, given the eventual adjoint study.

**Why not torch.nn.utils.parametrize.** It builds a dynamic subclass of the
module holding the parameter, and neuralop stores the spectral weight in
tltorch's ``DenseTensor``, whose ``FactorizedTensor.__init_subclass__`` requires
a ``name`` argument --- the injection raises ``TypeError``. The holder below
replaces the ``DenseTensor`` instead and reproduces the two pieces of its
interface ``SpectralConv`` actually uses: ``weight[slices]`` and ``.shape``.
``DenseTensor`` registers a non-leaf tensor as a *buffer*, so wrapping the
normalized tensor back up preserves the autograd graph.
"""
from __future__ import annotations

import copy
from typing import Any

from .runtime import nn, require_runtime, torch

#: Power iterations per forward once the estimate is warm. ``spectral_norm``
#: uses one; two is taken here because the cap is enforced on a *running*
#: estimate, so the true sigma trails the moving weights by an amount set by how
#: fast they move against how fast the estimate tracks. Measured one optimizer
#: step after a converged warmup, the published sigma_max was 1.0022 at one
#: iteration. The cost is negligible against the FNO forward.
POWER_ITERATIONS = 2

#: Iterations run once at construction. Power iteration converges from below, so
#: an under-converged sigma under-scales and leaves the true value above one.
#: Measured on the neuralop initialization, the true post-cap sigma_max is
#: 1.0350 after 24 iterations, 1.0050 after 100, and 1.00009 after 400.
WARMUP_ITERATIONS = 400

EPSILON = 1.0e-12

#: The parameter inside neuralop's ``DenseTensor`` weight wrapper.
WEIGHT_PARAMETER = "tensor"


class SpectralNormError(RuntimeError):
    """Raised when per-mode spectral normalization cannot be installed."""


if nn is not None:

    from tltorch.factorized_tensors.factorized_tensors import DenseTensor

    class ModeSpectralNorm(nn.Module):
        """Cap the largest singular value of every per-mode matrix at one.

        Registered with :mod:`torch.nn.utils.parametrize` on the spectral
        weight, so ``conv.weight.tensor`` returns the normalized tensor and the
        underlying free parameter keeps its original shape and count.
        """

        def __init__(
            self,
            weight: Any,
            *,
            power_iterations: int = POWER_ITERATIONS,
            warmup_iterations: int = WARMUP_ITERATIONS,
            epsilon: float = EPSILON,
        ) -> None:
            super().__init__()
            if weight.ndim != 4 or not weight.is_complex():
                raise SpectralNormError(
                    "per-mode spectral normalization expects a complex "
                    "(in, out, ky, kx) spectral weight"
                )
            self.in_channels, self.out_channels = int(weight.shape[0]), int(weight.shape[1])
            self.mode_shape = tuple(int(v) for v in weight.shape[2:])
            self.modes = int(weight.shape[2] * weight.shape[3])
            self.power_iterations = int(power_iterations)
            self.epsilon = float(epsilon)
            generator = torch.Generator(device="cpu").manual_seed(0)

            def _unit(width: int) -> Any:
                real = torch.randn((self.modes, width), generator=generator)
                imaginary = torch.randn((self.modes, width), generator=generator)
                value = torch.complex(real, imaginary).to(weight.device)
                return value / value.norm(dim=1, keepdim=True).clamp_min(epsilon)

            # Persistent so a reloaded model resumes from a converged estimate.
            self.register_buffer("left", _unit(self.out_channels))
            self.register_buffer("right", _unit(self.in_channels))
            with torch.no_grad():
                self._iterate(self._matrices(weight), int(warmup_iterations))

        def _matrices(self, weight: Any) -> Any:
            """``(in, out, ky, kx) -> (modes, in, out)``."""

            return weight.reshape(self.in_channels, self.out_channels, self.modes).permute(
                2, 0, 1
            )

        def _iterate(self, matrices: Any, iterations: int) -> None:
            left, right = self.left, self.right
            for _ in range(int(iterations)):
                # right <- A^H u = conj(W) u ; left <- A v = W^T v
                right = torch.einsum("mio,mo->mi", matrices.conj(), left)
                right = right / right.norm(dim=1, keepdim=True).clamp_min(self.epsilon)
                left = torch.einsum("mio,mi->mo", matrices, right)
                left = left / left.norm(dim=1, keepdim=True).clamp_min(self.epsilon)
            self.left.copy_(left)
            self.right.copy_(right)

        def sigma(self, weight: Any) -> Any:
            """Largest singular value of every mode's matrix, shape ``(modes,)``."""

            matrices = self._matrices(weight)
            product = torch.einsum("mio,mi->mo", matrices, self.right)
            return torch.einsum("mo,mo->m", self.left.conj(), product).real.abs()

        def forward(self, weight: Any) -> Any:
            matrices = self._matrices(weight)
            if self.training:
                with torch.no_grad():
                    self._iterate(matrices.detach(), self.power_iterations)
            # Snapshot the singular-vector estimates. The rollout calls this six
            # times per microbatch and each call advances the buffers in place;
            # without a clone the graph from an earlier call would hold a tensor
            # a later call has mutated, and backward fails on the version count.
            left, right = self.left.clone(), self.right.clone()
            product = torch.einsum("mio,mi->mo", matrices, right)
            sigma = torch.einsum("mo,mo->m", left.conj(), product).real.abs()
            # min(1, 1/sigma): identity where the mode already contracts, and a
            # constant there, so no gradient flows through an inactive mode.
            scale = torch.clamp(1.0 / sigma.clamp_min(self.epsilon), max=1.0)
            scaled = matrices * scale.reshape(-1, 1, 1)
            return scaled.permute(1, 2, 0).reshape(
                self.in_channels, self.out_channels, *self.mode_shape
            )

    class NormalizedSpectralWeight(nn.Module):
        """Drop-in replacement for neuralop's ``DenseTensor`` spectral weight.

        Holds the *unconstrained* parameter and exposes ``tensor`` as the
        normalized value, recomputed on every access so the cap is part of the
        forward graph. ``SpectralConv.forward`` only ever does
        ``self.weight[slices]``, and the contraction it selected at construction
        expects a ``DenseTensor``, so ``__getitem__`` hands one back.
        """

        def __init__(
            self, original: Any, *, warmup_iterations: int = WARMUP_ITERATIONS
        ) -> None:
            super().__init__()
            self.register_parameter("original", nn.Parameter(original.data.clone()))
            self.norm = ModeSpectralNorm(
                self.original.data, warmup_iterations=warmup_iterations
            )
            self.shape = self.original.shape
            self.order = self.original.ndim
            self.rank = None

        @property
        def tensor(self) -> Any:
            return self.norm(self.original)

        def __getitem__(self, indices: Any) -> Any:
            return DenseTensor(self.tensor[indices])

        def sigma(self) -> Any:
            """Unnormalized per-mode singular values of the free parameter."""

            with torch.no_grad():
                return self.norm.sigma(self.original)


else:  # pragma: no cover - environment dependent
    ModeSpectralNorm = None  # type: ignore[assignment,misc]
    NormalizedSpectralWeight = None  # type: ignore[assignment,misc]


def spectral_convolutions(model: Any) -> list[Any]:
    """The three ``SpectralConv`` layers, in block order."""

    return list(model.fno.fno_blocks.convs)


def apply_mode_spectral_norm(
    model: Any,
    *,
    warmup_iterations: int = WARMUP_ITERATIONS,
) -> dict[str, Any]:
    """Install the per-mode cap on every spectral convolution of ``model``.

    ``warmup_iterations`` is exposed only so tests that do not depend on a
    converged sigma can run cheaply; training always uses the declared value.
    """

    require_runtime()
    installed = []
    for index, conv in enumerate(spectral_convolutions(model)):
        holder = conv.weight
        if isinstance(holder, NormalizedSpectralWeight):
            raise SpectralNormError(f"block {index} is already normalized")
        if not isinstance(holder, nn.Module) or not hasattr(holder, WEIGHT_PARAMETER):
            raise SpectralNormError(
                "the spectral weight is not the expected DenseTensor wrapper"
            )
        replacement = NormalizedSpectralWeight(
            getattr(holder, WEIGHT_PARAMETER), warmup_iterations=warmup_iterations
        )
        conv.weight = replacement
        installed.append(
            {
                "block": index,
                "modes": replacement.norm.modes,
                "in_channels": replacement.norm.in_channels,
                "out_channels": replacement.norm.out_channels,
            }
        )
    return {
        "form": "R_k <- R_k * min(1, 1 / sigma_max(R_k)) for every Fourier mode",
        "estimator": "persistent alternating power iteration, one step per forward",
        "power_iterations_per_forward": POWER_ITERATIONS,
        "warmup_iterations": int(warmup_iterations),
        "blocks": installed,
        "matrices_total": sum(record["modes"] for record in installed),
        "adds_parameters": 0,
        "applies_to": "spectral_convolutions_only",
        "does_not_impose": "a global bound on the emulator Jacobian",
    }


def mode_sigma_summary(model: Any) -> dict[str, Any]:
    """Per-block statistics of the *unnormalized* mode singular values."""

    summary = []
    for index, conv in enumerate(spectral_convolutions(model)):
        holder = conv.weight
        if not isinstance(holder, NormalizedSpectralWeight):
            continue
        sigma = holder.sigma()
        summary.append(
            {
                "block": index,
                "modes": int(sigma.numel()),
                "sigma_max": float(sigma.max()),
                "sigma_mean": float(sigma.mean()),
                "sigma_median": float(sigma.median()),
                "modes_above_one": int((sigma > 1.0).sum()),
                "fraction_above_one": float((sigma > 1.0).float().mean()),
            }
        )
    return {
        "per_block": summary,
        "modes_above_one_total": sum(r["modes_above_one"] for r in summary),
        "note": (
            "sigma of the free parameter before the cap; a mode above one is "
            "one the cap is actively holding down"
        ),
    }


def materialized_state_dict(model: Any) -> dict[str, Any]:
    """State dict with the normalized weights baked in, plain-model compatible.

    Checkpoints are written from this, so a published checkpoint loads strictly
    into an unmodified :class:`~turbfno.model.ProductionFNO` and the inference
    operator is exactly the normalized one. The free parameter and the two
    singular-vector buffers do not enter the checkpoint.
    """

    normalized = {}
    for index, conv in enumerate(spectral_convolutions(model)):
        holder = conv.weight
        if not isinstance(holder, NormalizedSpectralWeight):
            continue
        with torch.no_grad():
            normalized[f"fno.fno_blocks.convs.{index}.weight.{WEIGHT_PARAMETER}"] = (
                holder.tensor.detach().clone()
            )
    result = {
        key: copy.deepcopy(value)
        for key, value in model.state_dict().items()
        if key != "_metadata"
        and ".weight.original" not in key
        and ".weight.norm." not in key
    }
    result.update(normalized)
    return result
