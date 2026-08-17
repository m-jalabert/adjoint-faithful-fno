# Matched-budget duration control for the Bire Section 3.2 protocol

Identical to `model_c_bire_protocol_pooled_v1` in every model quantity except one:

| quantity | parent | this arm |
| --- | --- | --- |
| `maximum_steps` | 7,680 | 15,360 |
| checkpoint steps | 1,920 / 3,840 / 5,760 / 7,680 | 3,840 / 7,680 / 11,520 / 15,360 |
| decay step (0.75 x budget) | 5,760 | 11,520 |

Architecture, Model C loss v1 over a three-step rollout, the Bire Section 3.2
split (train 0--5999, validation 6000--7199, inference 6200--7199 nested, no
buffers, pooled S0+S1+S2), the train-only normalizers, seed 20260724, batch 8,
Adam 5e-4 with betas (0.9, 0.95), zero weight decay, absent gradient clipping,
the 0.2 decay factor, and the selection rule are all unchanged and asserted
field by field against the parent contract.

**This run's step-7,680 checkpoint is not the parent's.** Holding
`decay_fraction` at 0.75 moves the decay from step 5,760 to 11,520, so at step
7,680 this run is still at 5e-4 while the parent had already dropped to 1e-4.
The schedule *shape* is preserved, which is the faithful reading of the same
experiment run longer; step 7,680 is a within-run landmark, not a comparator.

Selection is unchanged: minimise the worst 90--360-day RMSE-AUC relative to
climatology subject to each field's 10--90-day AUC staying within 5% of the best
checkpoint, on 102 pooled validation rollouts from starts outside the inference
set.

| step | short AUC / persistence | long AUC / climatology | day-200 ACC (U/SST/P) |
| --- | --- | --- | --- |
| 3,840 | 0.622/0.560/1.610 | 1.594/1.493/1.428 | +0.494/+0.462/+0.424 |
| 7,680 | 0.476/0.617/1.060 | 1.542/1.723/1.445 | +0.463/+0.469/+0.480 |
| 11,520 | 0.340/0.449/0.777 | 0.946/1.253/0.889 | +0.705/+0.587/+0.694 |
| 15,360 | 0.253/0.293/0.261 | 0.716/0.932/0.419 | +0.796/+0.719/+0.932 |

Selected step 15,360 via `primary_rule`.

Training and validation only; the inference set opens through the figure
contract, S0 only.

Report content SHA-256: `25e4e875d3f993298717d14b0f5aa3decc156eb1782752b786eac9a2b4bc93b2`.
