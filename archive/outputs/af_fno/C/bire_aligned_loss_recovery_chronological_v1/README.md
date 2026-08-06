# Loss-recovery model under a strictly chronological split

Identical model to `model_c_bire_aligned_loss_recovery_v1` --- same seed,
three FNO blocks, six pointwise LayerNorms, modes 24x16, width 128, Bire
positional encoding, 10% padding, no external local branch, Model C loss
v1 over a three-step rollout, Adam 5e-4 decaying to 1e-4 at 75%, batch 8,
7,680 steps, trained from scratch.

| split | indices | days |
| --- | --- | --- |
| train | 0--5039 | 5040 |
| validation | 5130--5759 | 630 |
| test (sealed here) | 5850--7199 | 1350 |

Training strictly precedes validation, which strictly precedes test, with
90-day buffers at both boundaries (180 buffer days).

**This is not a pure split-order ablation.** Both training sets contain
5,040 days but only 3,870 overlap: 5040-6209 is exchanged for 2520-3689, changing 23.2% of the training snapshots. The arm
tests the chronological protocol *and* sensitivity to the training period.

All train-derived statistics were recomputed from 0--5039: pointwise mean,
pointwise scale, channel scale floors, per-regime climatology, and the
pointwise increment scale. Wind normalization is unchanged because
`static_features` has no time axis.

Checkpoints were selected on 90 held 360-day validation rollouts inside
5130--5759 using the declared rule: minimise the worst 90--360-day
RMSE-AUC relative to climatology, subject to each field's 10--90-day
RMSE-AUC staying within 5% of the best checkpoint's.

| step | short AUC / persistence | long AUC / climatology | day-200 ACC (U/SST/P) |
| --- | --- | --- | --- |
| 1920 | 0.909/0.793/1.247 | 2.170/1.686/0.898 | +0.321/+0.305/+0.507 |
| 3840 | 0.881/0.890/2.058 | 2.444/2.677/2.833 | +0.333/+0.188/+0.535 |
| 5760 | 0.773/0.905/2.074 | 2.168/2.330/1.347 | +0.341/+0.210/+0.361 |
| 7680 | 0.480/0.472/0.560 | 1.374/1.415/0.424 | +0.632/+0.559/+0.935 |

Selected step 7680 via `primary_rule`.

Training and validation only. The test block and the held S0 archive
remained sealed for this run.

Report content SHA-256: `8e5da2daa3e370b8b41063f5f0268d33f1df705b396ff6481566669f2b1eb9c2`.
