"""The production MITgcm double-gyre emulator.

One model, defined once::

    F_theta: [x_t, S] -> x_{t+10}

``x_t`` is the 46-channel prognostic state ``[U_1:15, V_1:15, Theta_1:15, eta]``
at a single time level, and ``S`` the five-channel physical static block
``[tau_x, wet mask, f(phi), dx(phi), theta_clim(x, y)]``. Two deterministic
sine/cosine position channels are appended inside the model, so 51 external
channels enter as 53 at lifting and 46 leave. The operator is a 64x64-mode,
width-128, three-block FNO with six pointwise LayerNorms, a 4C Channel MLP, 10 %
raised-cosine tapered replicate padding and a parallel bias-free 3x3 local
correction: 104,368,296 parameters on the 248 x 248 grid.

The **production** operator is trained once, from random initialization, with
six-step autoregression and the complete physics-aware objective active from
optimizer step one: no parent checkpoint, no migration, no inherited
normalization, no function-preserving load and no comparator model. Both
normalization artifacts are recomputed over training days 0--5999. That is
``train``, and it stays that way.

There is exactly **one** staged stage in the package, and it is separate by
construction. ``finetune`` continues a *published* production checkpoint for a
short second stage at nine-step (ninety-day) autoregression, reusing the
parent's normalization unchanged and starting Adam cold. It changes four things
--- initialization, rollout length, learning rate and step count --- and nothing
else: not the architecture, not the objective's weights, not the spectral cap,
not the split. It imports ``train`` rather than modifying it, so every hash the
parent's contract pins stays valid and the parent stays reproducible.

Layout, in dependency order --- each module imports only from the ones above it::

    runtime               torch/neuralop guard, device, seeding, hashing, the
                          46-channel contract and the batch sampler
    diagnostics           streamfunction, hydrostatic pressure, SST, speed;
                          per-member RMSE and anomaly correlation
    plots                 the frozen figure definitions
    dataset               the turbulent trajectory store, the Bire Section 3.2 split,
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
    finetune              contract, ninety-day staged fine-tune of a published
                          production checkpoint, selection
    figures               the S0 evaluation suite and the 2,000-day gate half
    anomaly               streamfunction-anomaly companions to figures 3 and 7
    figures_ft90          the same S0 suite for the fine-tuned checkpoint
    anomaly_ft90          the same anomaly companions for that suite

``diagnostics`` and ``plots`` import nothing from the package at all.

The two ``_ft90`` evaluation modules are **lineage adapters, not second
implementations**. Each imports its numerical routines from the production
package it shadows --- the rollout, the 2,000-day conditions, the member draw,
the reference mean, the anomaly diagnostics and every plot --- and reimplements
only the provenance: which arm produced the checkpoint, and which sealed package
feeds the next stage. Nothing they touch is duplicated, and ``figures.py``,
``anomaly.py`` and ``plots.py`` stay byte-identical to the hashes the parent's
contracts pin, so the parent's packages remain re-runnable. That matters here
beyond tidiness: the parent's day-2,000 numbers are the comparison the fine-tune
exists to move, and a comparison whose baseline can drift is not one.

Entry points, each driven by a frozen contract in ``config/``::

    python -m turbfno.train        preflight|run              --contract ...
    python -m turbfno.finetune     preflight|run              --contract ...
    python -m turbfno.figures      finalize|preflight|run     --contract ...
    python -m turbfno.anomaly      finalize|preflight|run     --contract ...
    python -m turbfno.figures_ft90 finalize|preflight|run     --contract ...
    python -m turbfno.anomaly_ft90 finalize|preflight|run     --contract ...
"""

from __future__ import annotations

__all__ = [
    "anomaly",
    "anomaly_ft90",
    "barotropic_transport",
    "continuity",
    "dataset",
    "diagnostics",
    "figures",
    "figures_ft90",
    "finetune",
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
