# Bire Section 3.2 protocol, S0: Figures 3--8

This package evaluates the seed-20260724, step-7,680
checkpoint of the pooled Bire-protocol model on the **S0** inference set
(indices 6200--7199), tau0 = 0.1 N m-2.
S0 is the primary Bire-style comparison.

The model is the loss-recovery architecture and objective, unchanged: three FNO
blocks, six pointwise LayerNorms, modes 24x16, width 128, Bire positional
encoding, 10% padding, no external local branch, Model C loss v1 over a
three-step rollout, Adam 5e-4 decaying to 1e-4 at 75%, batch 8, 7,680 steps. One
FNO was trained on the pooled S0+S1+S2 training blocks (0--5999) and selected on
the pooled validation blocks (6000--7199).

**The split is Bire's, not a chronological variant.** 6,000 training + 1,200
validation is the paper's entire 7,200-day record, and the 1,000 inference days
are the final 1,000 of validation rather than a third block -- the paper states
it uses no third held-out set. There are no buffers, because the paper has none.
The inference set is therefore nested in validation; selection starts are drawn
from the 200 validation days outside it, so no selection start is also a member
start.

**Model visibility.** The model saw nothing at or beyond index 7200. Days
7200--8999 of trajectory-v3 carry no split code and are used only as evaluation
truth. That is what makes the day-2,000 ground-truth column of the paper's
Figure 7 reproducible here: the 15 starts are drawn from
6200--6999, the part of the inference set that admits a
complete 2,000-day rollout inside the store, and this draw spans
6263--6979. Every member therefore has lead-matched MITgcm truth at
every lead out to day 2,000 (6979 + 2000 = 8979 < 9000). Bire obtain the
same separation across simulations -- runs 2 and 4 are entirely held out -- and
we obtain it along time.

**Not comparable byte-for-byte with the v2 or v3 chronological packages.** The
15 starts are a new fixed draw under a different record indexing, and the
training and validation blocks differ, so comparison with earlier packages is
protocol-level only.

**Figure 6 comparator.** The prior residual Model C was trained on v2 with a v2
normalizer and branch-based S1/S2, so it cannot be run meaningfully here. The
black curve is instead this run's own step-1,920 checkpoint against the
selected step-7,680 checkpoint: a training-progress
comparison on identical data, not the frozen architecture pairing.

Climatology is the pointwise S0 mean over the Bire training block
(0--5999) only. Persistence holds each member's initial physical field fixed.
RMSE is computed over wet cells per member; lines and bands are the mean and
10th/90th percentiles across the 15 members.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `c6b60dba4e5f4a0b3e2dcd72ce01c8f3633cb26db4c0ad083d83cf9ec372224f`.
