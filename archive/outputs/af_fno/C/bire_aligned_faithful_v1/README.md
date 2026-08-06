# Bire-faithful protocol arm (training split only)

Three unintended divergences from the public `oceanfourcast`
implementation, corrected as one bundle on the working 5e-4 base:

| quantity | earlier arms | this arm |
| --- | --- | --- |
| MAE weight | 0.01 | **0.05** |
| LR schedule | step decay x0.2 at 75% | **cosine, T_max 3, eta_min 1e-5** |
| checkpoint selection | fixed steps | **lowest validation loss per stage** |

Architecture, inputs, positional encoder, two-stage protocol, seed,
batch size 8, betas, weight decay, absent gradient clipping, and
lr0 = 5e-4 are frozen against `model_c_bire_aligned_full_state_lr5e4_v1`.
ChannelMLP dropout stays at zero.

Validation is a seeded random 10% of the split-1 training records; no
sealed archive is opened. Epoch = a fixed 1,920-step period, so the
optimizer-step and sequence-exposure budget matches the earlier arms.

Per-epoch validation loss and selection:

| epoch | step | stage | lr | train total | valid total |
| --- | --- | --- | --- | --- | --- |
| 1 | 1920 | pretrained | 0.0005 | 0.05357 | 0.01980 |
| 2 | 3840 | pretrained | 0.000378 | 0.01305 | 0.00984 |
| 3 | 5760 | finetuned | 0.000133 | 0.01949 | 0.01719 |
| 4 | 7680 | finetuned | 1e-05 | 0.01455 | 0.01444 |

Selected per stage:

* `pretrained` -- epoch 2, step 3840, validation 0.00984
* `finetuned` -- epoch 4, step 7680, validation 0.01444

Gate instrument (unchanged 360-day split-1 spectral/primary summary):

| stage | step | worst 10--90-day RMSE ratio | worst mid/bottom modewise | gate |
| --- | --- | --- | --- | --- |
| pretrained | 3840 | 1.9592 | 435.347 | False |
| finetuned | 7680 | 0.8926 | 133.349 | False |

Training split only; validation, inference, held S0, response, and
adjoint archives remained sealed for this run.

Report content SHA-256: `19d42d8cbc9d0434f7137d5e207ea1ceaf7eec3b5a1bcba21534a06ad28dc095`.
