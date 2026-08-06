# Bire-faithful protocol arm (one-step pretrained): S0 Figures 3--8

This package evaluates the seed-20260724 **pretrained** checkpoint (optimizer step
3,840, selected by lowest validation loss within its stage) of the
Bire-faithful protocol arm under the control wind (tau0 = 0.1 N m-2). All
forecasts use a ten-day prediction interval.

The arm corrects, as one bundle on the working 5e-4 base, three unintended
divergences from the public `oceanfourcast` implementation:

| quantity | earlier arms | this arm |
| --- | --- | --- |
| MAE weight | 0.01 | **0.05** |
| LR schedule | step decay x0.2 at 75% | **cosine, T_max 3, eta_min 1e-05** |
| checkpoint selection | fixed steps | **lowest validation loss per stage** |

Architecture (three FNO blocks, six pointwise channel LayerNorms, 49 external
inputs, the deterministic `oceanfourcast.PosEmbed` fields appended immediately
before lifting, no external 3x3 branch), the two-stage protocol, seed, batch
size 8, betas (0.9, 0.95), zero weight decay, absent gradient clipping, and
lr0 = 5e-4 are frozen against `model_c_bire_aligned_full_state_lr5e4_v1`.
ChannelMLP dropout stays at zero.

Adopting validation-based selection holds out a seeded random
10% of the split-1 training records, so this arm trained on
90% of the records the earlier arms saw. The holdout is drawn from training-split
records only; no sealed archive was opened.

Compare with `bire_aligned_full_state_lr5e4_bire_s0_inference_v1/pretrained` to
isolate the effect of the three corrections, and with
`single_position_layernorm_bire_s0_inference_v1` for the retained incumbent. The
15 initial conditions, the day-2000 MITgcm truth from evaluation-only job 304735,
the baselines, the reductions, and the six figure filenames are identical across
all packages, so they are directly comparable field for field.

Climatology is the pointwise temporal mean over S0 split-1 training snapshots.
Persistence holds each member's initial physical field fixed. RMSE is first
computed over wet cells for each member; lines/bands are the mean and
10th/90th percentiles across the 15 members.

Figure 6 compares the prior residual rollout-conditioned Model C with this arm.
It is an architecture-direction comparison, not a literal pretrained/fine-tuned
pairing.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `85bed72245a70b5cb03cbc201fad6aea464dd2de3c13b4cdd9d17d4abc7f99c5`.
