# Loss-recovery model on trajectory-v3 (independent equilibria, pooled regimes)

Same model as `model_c_bire_aligned_loss_recovery_v1`: three FNO blocks,
six pointwise LayerNorms, modes 24x16, width 128, Bire positional encoding,
10% padding, no external local branch, Model C loss v1 over a three-step
rollout, Adam 5e-4 decaying to 1e-4 at 75%, batch 8, 7,680 steps, from
scratch on seed 20260724.

The data underneath it is new. S1 and S2 are no longer branches of the S0
year-100 state: each regime was equilibrated independently for 100 years
from the tutorial initial condition under its own wind, then run 25
production years.

| split | indices | days per regime |
| --- | --- | --- |
| train (pooled) | 0--5039 | 5040 |
| validation (pooled) | 5130--6389 | 1260 |
| test (sealed here) | 6480--8999 | 2520 |

One FNO trained on all three training blocks and selected on all three
validation blocks. Pointwise normalizer, per-regime climatology, and
increment scale all recomputed from the v3 training block.

| step | short AUC / persistence | long AUC / climatology | day-200 ACC (U/SST/P) |
| --- | --- | --- | --- |
| 1920 | 0.758/0.719/1.439 | 2.073/2.028/1.831 | +0.282/+0.288/+0.433 |
| 3840 | 0.617/0.619/1.103 | 1.840/2.042/1.732 | +0.293/+0.276/+0.527 |
| 5760 | 0.585/0.688/1.263 | 1.812/1.960/1.405 | +0.414/+0.342/+0.424 |
| 7680 | 0.402/0.412/0.436 | 1.365/1.515/0.731 | +0.547/+0.480/+0.802 |

Selected step 7680 via `primary_rule`.

Training and validation only; the test block remained sealed.

Report content SHA-256: `9aeaaa7fea851ee8ef03748fcfcd3d83d92182d882d8893380120ffc02483af0`.
