# Matched-budget duration arm, S0: Figures 3--8

This package evaluates the seed-20260724, step-15,360
checkpoint of the pooled matched-budget duration model on the **S0**
inference set (indices 6200--7199), tau0 = 0.1 N m-2.

The arm is identical to `model_c_bire_protocol_pooled_v1` in every model
quantity except `maximum_steps`, which moves from 7,680 to 15,360. Validation
selected the final step 15,360 checkpoint, improving the worst
90--360-day ratio to climatology from 1.081 to 0.932.

**Directly comparable with the parent figure package.** The 15 members, the seed,
the lead grid, the truth window, the climatology and the persistence baseline are
all unchanged, so `outputs/af_fno/C/bire_protocol_s0_figures_v1/S0/` and this
package differ only in the checkpoint. The starts are 6200--6999, this draw
spanning 6263--6979, and every member has
lead-matched MITgcm truth to day 2,000 (6979 + 2000 =
8979 < 9000).

**Figure 6 comparator.** The black curve is this run's own step-7,680
checkpoint against the selected step-15,360 one.
Step 7,680 is the parent arm's entire budget, so the gap between the curves is
what doubling the budget bought within a single run. It is not the parent's
checkpoint: this run's decay falls at 11,520, so at step 7,680 it is still at
5e-4 where the parent had already dropped to 1e-4.

Climatology is the pointwise S0 mean over the Bire training block
(0--5999) only. Persistence holds each member's initial physical field fixed.
RMSE is computed over wet cells per member; lines and bands are the mean and
10th/90th percentiles across the 15 members.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `53233fa21f127576056ae1e0ecc049e9bd94e0bcb1efc9c495a334c434f45f7d`.
