# Trajectory-v3 pooled model, S1: S0-style Figures 3--8

This package evaluates the seed-20260724, step-7,680
checkpoint of the pooled trajectory-v3 model on the **S1** held test block
(indices 6480--8999), tau0 = 0.075 N m-2. S1 is the wind-regime robustness.

The model is the loss-recovery architecture and objective, unchanged: three FNO
blocks, six pointwise LayerNorms, modes 24x16, width 128, Bire positional
encoding, 10% padding, no external local branch, Model C loss v1 over a
three-step rollout, Adam 5e-4 decaying to 1e-4 at 75%, batch 8, 7,680 steps. One
FNO was trained on the pooled S0+S1+S2 training blocks (0--5039) and selected on
the pooled validation blocks (5130--6389).

Trajectory-v3 differs from v2 in the physics, not only the split: S1 and S2 were
equilibrated independently for 100 years from the tutorial initial condition
under their own wind, rather than branching from the S0 year-100 state with a
five-year adjustment.

**Not comparable byte-for-byte with the v2 packages.** The 15 starts are a new
fixed draw from 6480--6999, the only window admitting a complete 2,000-day
rollout inside a 2,520-day test block. The v2 suite's starts (6660--7199) index a
different record and would land in validation here. Comparison with earlier
packages is protocol-level only.

**Figure 6 comparator.** The prior residual Model C was trained on v2 with a v2
normalizer and branch-based S1/S2, so it cannot be run meaningfully on v3. The
black curve is instead this run's own step-1,920 checkpoint against the
selected step-7,680 checkpoint: a training-progress
comparison on identical data, not the frozen architecture pairing.

Climatology is the pointwise S1 mean over the v3 training block only.
Persistence holds each member's initial physical field fixed. RMSE is computed
over wet cells per member; lines and bands are the mean and 10th/90th
percentiles across the 15 members.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `f9b60889c49f0b87f5be2f4dee682b9ca1002d1d2b707fff24f27e982aaa718a`.
