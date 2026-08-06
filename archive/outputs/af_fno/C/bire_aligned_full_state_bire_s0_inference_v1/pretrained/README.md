# Bire-aligned full-state FNO (one-step pretrained): S0 Bire-style Figures 3--8

This package evaluates the seed-20260724, step-3,840 **pretrained**
Bire-aligned full-state Model C under the control wind (tau0 = 0.1 N m-2). All
forecasts use a ten-day prediction interval.

The arm keeps the MITgcm trajectories, the 62x62 grid, the 24x16 retained
modes, the 46-channel state, the ten-day map, and the pointwise anomaly
normalization, and replaces the remaining project-specific architecture and
training choices with Bire-like ones: three FNO blocks with six pointwise
channel LayerNorms, position supplied exactly once by the deterministic
`oceanfourcast.PosEmbed` sine/cosine fields appended immediately before
lifting, no external 3x3 raw-input branch, and Adam(1e-2, betas 0.9/0.95, no
weight decay) at batch size 8. Training was 3,840 one-step updates under `MSE + 0.01 MAE`.

The 15 initial conditions, the day-2000 MITgcm truth from evaluation-only job
304735, the climatology and persistence baselines, the metric reductions, and
the six figure filenames are identical to
`single_position_layernorm_bire_s0_inference_v1`, so the packages are directly
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

Report content SHA-256: `cf613ca366a205942e9e747acb4c934a8895abcbb84353d7626a8023349a2931`.
