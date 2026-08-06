# Bire-aligned architecture, incumbent objective: S0 Figures 3--8

This package evaluates the seed-20260724, step-7,680 gate-selected checkpoint
of the **loss-recovery control** under the control wind (tau0 = 0.1 N m-2). All
forecasts use a ten-day prediction interval.

The arm is an architecture-fixed control against
`model_c_bire_aligned_full_state_lr5e4_v1`. It keeps three FNO blocks, 24x16
modes, width 128, six pointwise channel LayerNorms, the deterministic
`oceanfourcast.PosEmbed` fields appended immediately before lifting, no external
3x3 raw-input branch, 10% padding, Adam(5e-4, betas 0.9/0.95, weight decay 0),
batch size 8, and the 7,680-step budget. It changes only the objective and the
rollout exposure, restoring the incumbent group-balanced Model C loss v1 --

    L_state = (L_U + L_V + L_Theta + L_eta) / 4

with its increment, rollout, spectral, and western-boundary terms -- over a
three-step unrolled rollout throughout.

Bire's `MSE + 0.01 MAE` averaged over all 46 normalized channels gives the
physical groups effective multiplicities `U:V:Theta:eta = 15:15:15:1`, so the
free surface received 1/46 of the channel-averaged loss. This arm tests whether
that weighting, rather than the architecture, cost the forecast skill.

Compare with `bire_aligned_full_state_lr5e4_bire_s0_inference_v1/finetuned` to
isolate the objective, and with `single_position_layernorm_bire_s0_inference_v1`
for the retained incumbent. The 15 initial conditions, the day-2000 MITgcm truth
from evaluation-only job 304735, the baselines, the reductions, and the six
figure filenames are identical across all packages, so they are directly
comparable field for field.

Climatology is the pointwise temporal mean over S0 split-1 training snapshots.
Persistence holds each member's initial physical field fixed. RMSE is first
computed over wet cells for each member; lines/bands are the mean and
10th/90th percentiles across the 15 members.

Figure 6 compares the prior residual rollout-conditioned Model C with this arm.
It is an architecture-direction comparison, not a literal pretrained/fine-tuned
pairing.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `6e6d64bbb1169d52d66cc31b43e52412c309f72b339d113118b7f3ebe0fa5a6f`.
