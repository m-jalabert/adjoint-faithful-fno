# Bire-aligned architecture with the incumbent Model C objective

Architecture-fixed loss-recovery control.  Everything architectural and
optimizer-side is frozen against
`model_c_bire_aligned_full_state_lr5e4_v1`; only the objective and the
rollout exposure change.

| quantity | 5e-4 Bire arm | this arm |
| --- | --- | --- |
| objective | wet-cell `MSE + 0.01 MAE` over 46 channels | **group-balanced Model C loss v1** |
| group weighting | `U:V:Theta:eta = 15:15:15:1` | **equal quarters** |
| rollout exposure | 1-step pretrain then 2-step fine-tune | **3-step unrolled throughout** |

Three FNO blocks, six pointwise LayerNorms, Bire PosEmbed, no external
3x3 branch, 24x16 modes, width 128, 10% padding, Adam(5e-4, betas
0.9/0.95, weight decay 0), batch size 8, and the 7,680-step budget are
all unchanged, so the optimizer-step and sequence-exposure budgets stay
comparable with every Bire-aligned arm.

It answers one question: is the bounded behaviour coming from the
Bire-aligned architecture while the loss of skill comes from the Bire
objective?

Gate instrument (unchanged 360-day split-1 spectral/primary summary):

| step | worst 10--90-day RMSE ratio | worst mid/bottom modewise | gate |
| --- | --- | --- | --- |
| 1920 | 1.3412 | 119.128 | False |
| 3840 | 1.3466 | 185.937 | False |
| 5760 | 1.0106 | 35.558 | False |
| 7680 | 0.4532 | 22.317 | False |

Selected: step 7680 (`no_arm_checkpoint_passed`).

Training split only; validation, inference, held S0, response, and
adjoint archives remained sealed for this run.

Report content SHA-256: `9f1e7ffc0ebe80bdd938dd1b1c8f4fb92c660f3facf3364aace3fb3a45a2c32d`.
