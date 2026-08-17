"""The production MITgcm double-gyre emulator.

One model, defined once::

    F_theta: [x_t, S] -> x_{t+10}

``x_t`` is the 46-channel prognostic state ``[U_1:15, V_1:15, Theta_1:15, eta]``
at a single time level, and ``S`` the five-channel physical static block
``[tau_x, wet mask, f(phi), dx(phi), theta_clim(x, y)]``. Two deterministic
sine/cosine position channels are appended inside the model, so 51 external
channels enter as 53 at lifting and 46 leave. The operator is a 32x32-mode,
width-128, three-block FNO with six pointwise LayerNorms, a 4C Channel MLP, 10 %
domain padding and a parallel bias-free 3x3 local correction: 27,297,960
parameters.

It is trained **once, from random initialization**, with six-step autoregression
and the complete physics-aware objective active from optimizer step one. There
is no lineage anywhere in this package --- no parent checkpoint, no migration, no
inherited normalization, no function-preserving load, no comparator model and no
staged fine-tuning. Both normalization artifacts are recomputed over training
days 0--5999.

Layout, in dependency order --- each module imports only from the ones above it::

    runtime               torch/neuralop guard, device, seeding, hashing, the
                          46-channel contract and the batch sampler
    diagnostics           streamfunction, hydrostatic pressure, SST, speed;
                          per-member RMSE and anomaly correlation
    plots                 the frozen figure definitions
    dataset               the trajectory-v3 store, the Bire Section 3.2 split,
                          the train-only normalizers, the physical static block
                          and the one-input rollout dataset
    perturbation_growth   growth diagnostics: single-call g and lambda_hat
    spectral_norm         per-mode cap on the Fourier convolution operators
    pressure_gradient     MITgcm-consistent PHIHYD gradient loss
    continuity            free-surface transport-continuity residual loss
    barotropic_transport  depth-integrated transport-tendency loss
    objective             the frozen production objective
    model                 architecture, position encoder, FNO, unroll, stepper
    validation            360-day validation rollouts, the selection rule,
                          the train-only climatology
    train                 contract, from-scratch training loop, selection
    figures               the S0 evaluation suite and the 2,000-day gate half
    anomaly               streamfunction-anomaly companions to figures 3 and 7

``diagnostics`` and ``plots`` import nothing from the package at all.

Entry points, each driven by a frozen contract in ``config/``::

    python -m oceanfno.train    preflight|run                 --contract ...
    python -m oceanfno.figures  finalize|preflight|run        --contract ...
    python -m oceanfno.anomaly  finalize|preflight|run        --contract ...
"""

from __future__ import annotations

__all__ = [
    "anomaly",
    "barotropic_transport",
    "continuity",
    "dataset",
    "diagnostics",
    "figures",
    "model",
    "objective",
    "perturbation_growth",
    "plots",
    "pressure_gradient",
    "runtime",
    "spectral_norm",
    "train",
    "validation",
]
