# Pooled loss-recovery model under the Bire et al. Section 3.2 arrangement

Same model as `model_c_bire_aligned_v3_pooled_v1`: three FNO blocks, six
pointwise LayerNorms, modes 24x16, width 128, Bire positional encoding,
10% padding, no external local branch, Model C loss v1 over a three-step
rollout, Adam 5e-4 decaying to 1e-4 at 75%, batch 8, 7,680 steps, from
scratch on seed 20260724. Only the data arrangement changes.

| split | indices | days per regime |
| --- | --- | --- |
| train (pooled) | 0--5999 | 6000 |
| validation (pooled) | 6000--7199 | 1200 |
| inference (nested in validation) | 6200--7199 | 1000 |

6,000 + 1,200 = 7,200 is Bire's entire record, and the 1,000 inference
days are the final 1,000 of validation, not a third block -- the paper
states it uses no third held-out set. No buffers: the paper has none, and
leakage is prevented instead by requiring a training rollout's whole
target sequence to stay inside training, so the latest start is
5969.

The model sees nothing at or beyond index 7200. Days
7200--8999 are evaluation truth only, never trained on,
validated on, or used as an inference start. Bire obtain the same
separation across simulations (runs 2 and 4 are entirely held out); we
obtain it along time, which is what allows the day-2,000 ground-truth
column their Figure 7 also shows.

Selection used the pooled validation block under the declared rule:
minimise the worst 90--360-day RMSE-AUC relative to climatology subject
to each field's 10--90-day AUC staying within 5% of the best checkpoint.

| step | short AUC / persistence | long AUC / climatology | day-200 ACC (U/SST/P) |
| --- | --- | --- | --- |
| 1920 | 0.719/0.693/1.172 | 1.928/2.119/1.921 | +0.332/+0.249/+0.642 |
| 3840 | 0.622/0.560/1.610 | 1.594/1.493/1.428 | +0.494/+0.462/+0.424 |
| 5760 | 0.480/0.552/1.249 | 1.305/1.554/1.769 | +0.606/+0.503/+0.568 |
| 7680 | 0.345/0.371/0.407 | 0.978/1.081/0.591 | +0.718/+0.656/+0.837 |

Selected step 7680 via `primary_rule`.

Training and validation only; the inference set opens through the figure
contract, S0 only.

Report content SHA-256: `59864c25904709672ebc33d6d2d4ecf5c147ab9a321e9165aff2a4891739ea3f`.
