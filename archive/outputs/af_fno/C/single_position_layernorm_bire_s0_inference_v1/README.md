# Single-position LayerNorm Model C: S0 Bire-style Figures 3--8

This package evaluates the seed-20260724, step-14880 single-position LayerNorm
pointwise-anomaly direct-state Model C under the control wind
(tau0 = 0.1 N m-2). All forecasts use a ten-day prediction interval.

The arm retains the full 46-channel state and changes exactly two things from
the retained selected Model C: the duplicated positional encoding is removed
(position is supplied once, through the two static data channels, and the
appended grid embedding is disabled) and the Bire pointwise channel LayerNorm
is applied after spectral and Channel-MLP mixing.

The 15 initial conditions, the day-2000 MITgcm truth from evaluation-only job
304735, the climatology and persistence baselines, the metric reductions, and
the six figure filenames are identical to
`anomaly_direct_bire_s0_inference_v1`, so the two packages are directly
comparable field for field.

Climatology is the pointwise temporal mean over S0 split-1 training snapshots.
Persistence holds each member's initial physical field fixed. RMSE is first
computed over wet cells for each member; lines/bands are the mean and
10th/90th percentiles across the 15 members.

Figure 6 compares the prior residual rollout-conditioned Model C with this
arm. It is an architecture-direction comparison, not a literal
pretrained/fine-tuned pairing.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `ea987916d219dbdabe97971389fc4c46384d954b0e373e70c5c9c74456ff518e`.
