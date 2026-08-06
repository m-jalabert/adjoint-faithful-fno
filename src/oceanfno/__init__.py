"""Six-step rollout fine-tune of the 46-channel MITgcm double-gyre emulator.

A Fourier Neural Operator advances a closed 46-channel ocean state by ten days.
This package trains it, selects a checkpoint, publishes the S0 figure suite, and
evaluates the acceptance gate. It contains only the code those three jobs
execute.

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
    python -m oceanfno.anomaly  preflight|run       --contract ...

Provenance: every function body here was copied verbatim from the arm that
produced the published checkpoint, and verified to reproduce it numerically ---
the same forward pass on the published weights, the same loss to the last bit,
the same objective hash, and a bit-identical rebuild of the anomaly package. The
superseded tree is kept under ``archive/`` together with the v1 contracts it is
hashed by.
"""

from __future__ import annotations

__all__ = ["anomaly", "dataset", "diagnostics", "figures", "model",
           "objective", "plots", "runtime", "train", "validation"]
