"""Canonical two-in / one-out six-step FNO for the 46-channel MITgcm double gyre.

A 32x32-mode Fourier Neural Operator with a bias-free local 3x3 correction and
the deterministic sine/cosine position encoder advances a closed 46-channel
ocean state by ten days. The canonical arm reads *two* consecutive time levels,
``(x_{t-10}, x_t) -> x_{t+10}``, and autoregresses by sliding the pair forward,
which gives the operator an empirical tendency instead of the current state
alone. This package trains it, selects a checkpoint, publishes the S0 figure
suite, and evaluates the acceptance gate. The one-input 32x32 architecture and
its own ancestors are represented only where checkpoint migration and literal
comparison require them.

Layout, in dependency order --- each module imports only from the ones above it::

    runtime      torch/neuralop guard, device, seeding, checkpoint state,
                 SHA-256 helpers, and the 46-channel contract every other
                 module is written against
    diagnostics  barotropic streamfunction, hydrostatic pressure, SST and
                 speed; per-member RMSE and anomaly correlation
    plots        the frozen figure definitions, unchanged across every package
    dataset      the trajectory-v3 store, the Bire Section 3.2 split, the
                 train-only normalizers, the rollout dataset and its sampler
    objective    the group-balanced Model C loss and its two configurations
    model        architecture, Bire positional encoder, the FNO, the
                 autoregressive unroll, and the evaluation stepper
    validation   360-day validation rollouts, the selection rule, the
                 train-only climatology
    train        the fine-tuning arm: contract, training loop, selection
    figures      the S0 evaluation, figure 6's pre-train/fine-tune pair, and
                 the 2,000-day half of the acceptance gate
    anomaly      streamfunction-anomaly companions to figures 3 and 7

``diagnostics`` and ``plots`` import nothing from the package at all. ``figures``
sits after ``train`` because it reads the arm's identifiers --- the selected
checkpoint steps, the objective hash, and the training report's name --- to
verify the checkpoint it is handed really is the one that arm published.

Entry points, each driven by a frozen contract in ``config/``::

    python -m oceanfno.train    preflight|run       --contract ...
    python -m oceanfno.figures  finalize|preflight|run --contract ...
    python -m oceanfno.anomaly  finalize|preflight|run --contract ...

The frozen contracts and reports remain the provenance record of the completed
experiments. The canonical modules consolidate the selected 32x32 workflows;
they do not rewrite those historical records.
"""

from __future__ import annotations

__all__ = [
    "anomaly",
    "dataset",
    "diagnostics",
    "figures",
    "model",
    "objective",
    "plots",
    "runtime",
    "train",
    "validation",
]
