# Bire-aligned FNO, learning-rate control (two-step fine-tuned): S0 Figures 3--8

This package evaluates the seed-20260724, step-7,680 **finetuned** checkpoint of
the Bire-aligned full-state Model C **learning-rate control** under the control
wind (tau0 = 0.1 N m-2). All forecasts use a ten-day prediction interval.

The arm is a one-factor control against `model_c_bire_aligned_full_state_v1`:
the only declared change is the optimizer learning rate,
0.01 -> 0.0005. Architecture (three FNO
blocks, six pointwise channel LayerNorms, 49 external inputs, the deterministic
`oceanfourcast.PosEmbed` sine/cosine fields appended immediately before lifting,
no external 3x3 raw-input branch), the wet-cell `MSE + 0.01 MAE` objective, the
two-stage protocol, the seed, batch size 8, betas (0.9, 0.95), zero weight
decay, and the absence of gradient clipping are all frozen against that parent.
Training was 3,840 one-step pretraining updates followed by 3,840 two-step autoregressive fine-tuning updates.

The 1e-2 parent collapsed to climatology: its one-step normalized MSE settled at
the zero-anomaly value of 1.0 and its day-200 ACC was +0.06 to +0.11. Comparing
this package with
`bire_aligned_full_state_bire_s0_inference_v1/finetuned` isolates the effect of
the learning rate alone; comparing it with
`single_position_layernorm_bire_s0_inference_v1` compares the Bire-aligned
architecture package against the retained incumbent.

The 15 initial conditions, the day-2000 MITgcm truth from evaluation-only job
304735, the climatology and persistence baselines, the metric reductions, and
the six figure filenames are identical across all three packages, so they are
directly comparable field for field.

Climatology is the pointwise temporal mean over S0 split-1 training snapshots.
Persistence holds each member's initial physical field fixed. RMSE is first
computed over wet cells for each member; lines/bands are the mean and
10th/90th percentiles across the 15 members.

Figure 6 compares the prior residual rollout-conditioned Model C with this arm.
It is an architecture-direction comparison, not a literal pretrained/fine-tuned
pairing.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `f9daff10e6e3a9597ef1b22ec2403d44ec5ea9f4caaddce532d634c0e243c6bf`.
