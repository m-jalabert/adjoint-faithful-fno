# Chronological-split loss-recovery model: S0 Figures 3--8

This package evaluates the seed-20260724, step-7,680 checkpoint of the
**chronological-split** arm under the control wind (tau0 = 0.1 N m-2), selected
on a held 630-day validation block. All forecasts use a ten-day interval.

The model is identical to `model_c_bire_aligned_loss_recovery_v1` -- three FNO
blocks, six pointwise channel LayerNorms, modes 24x16, width 128, Bire
positional encoding, 10% padding, no external 3x3 branch, Model C loss v1 over a
three-step rollout, Adam 5e-4 decaying to 1e-4 at 75%, batch 8, 7,680 steps,
trained from scratch. What changed is the protocol:

| | stored split | this arm |
| --- | --- | --- |
| train | 0--2519 and 3690--6209 (interleaved) | **0--5039 (contiguous)** |
| validation | 6300--6569 | **5130--5759** |
| test | 2970--3599 and 6660--7199 | **5850--7199** |
| normalizer | shared seed-20260724 artifact | **recomputed from 0--5039** |
| checkpoint rule | 360-day gate over training records | **held validation block** |

**This is not a pure split-order ablation.** Both training sets hold 5,040 days
but only 3,870 overlap: 5040--6209 is exchanged for 2520--3689, changing 23.2%
of the training snapshots. The arm tests the chronological protocol *and*
sensitivity to which stretch of trajectory is used for training.

The climatology baseline in these figures is rebuilt from 0--5039. The frozen
runner derives it from the stored split-1 codes, which include 5850--6209 --
test days here -- so leaving it alone would have leaked 360 test days into the
baseline. Both intervals contain 5,040 days, so the runner's count check passes
either way and does not catch the substitution on its own.

The 15 initial conditions lie in 6660--7199, which is test under both splits, so
this package is directly comparable with
`bire_aligned_loss_recovery_bire_s0_inference_v1/selected`. Do **not** compare
the two models over the full new test interval 5850--7199: the parent trained on
5850--6209.

Persistence holds each member's initial physical field fixed. RMSE is first
computed over wet cells for each member; lines/bands are the mean and 10th/90th
percentiles across the 15 members.

Figure 6 compares the prior residual rollout-conditioned Model C with this arm.
It is an architecture-direction comparison, not a literal pretrained/fine-tuned
pairing.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `7711cad2654710932a3cded75b5c256dcf4fd0de7085f87661729c0366af767c`.
