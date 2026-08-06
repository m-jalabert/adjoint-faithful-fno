# Bire-aligned full-state FNO, learning-rate control (training split only)

One-factor control against `model_c_bire_aligned_full_state_v1`. The
single declared change is the optimizer learning rate:

| | parent arm | this control |
| --- | --- | --- |
| initial learning rate | 1e-2 | **5e-4** |
| everything else | — | identical |

Architecture, 49 external inputs, Bire sine/cosine position fields, six
pointwise LayerNorms, absent external 3x3 branch, wet-cell
`MSE + 0.01 MAE`, the 3,840 + 3,840 two-stage protocol, seed, batch size
8, betas (0.9, 0.95), zero weight decay, absent gradient clipping, and
the 0.75/0.2 decay schedule are all frozen against the parent.

The parent arm collapsed to climatology: its one-step normalized MSE
settled at the zero-anomaly value of 1.0 and its day-200 ACC was +0.06
to +0.11. This control tests whether that collapse was caused by the
learning rate rather than by the Bire architecture package.

Gate instrument (unchanged 360-day split-1 spectral/primary summary):

| stage | step | worst 10--90-day RMSE ratio | worst mid/bottom modewise | gate |
| --- | --- | --- | --- | --- |
| pretrained | 3840 | 3.4657 | 707.071 | False |
| finetuned | 7680 | 0.9518 | 32.449 | False |

Training split only; validation, inference, held S0, response, and
adjoint archives remained sealed for this run.

Report content SHA-256: `016058fdfc1887336dc704770471b798ce8addad92dac08f2ae06fb53a84556a`.
