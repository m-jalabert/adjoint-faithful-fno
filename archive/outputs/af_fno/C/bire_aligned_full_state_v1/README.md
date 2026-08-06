# Bire-aligned full-state FNO (training split only)

Three-block, width-128 FNO over the retained 46-channel closed state.
49 external inputs (46 state + wind stress + wet mask + distance to
wall); the two deterministic Bire sine/cosine position channels are
appended immediately before lifting, giving 51 lifting inputs. Six
pointwise channel LayerNorms, no external 3x3 raw-input branch, and
the pointwise residual retained inside each spectral block.

Training is Bire's two-stage protocol under `MSE + 0.01 MAE` on wet
cells: 3840 one-step pretraining updates then
3840 two-step autoregressive fine-tuning updates, with
Adam(1e-2, betas 0.9/0.95, weight decay 0) at batch size 8 and no
gradient clipping.

Gate instrument (unchanged 360-day split-1 spectral/primary summary):

| stage | step | worst 10--90-day RMSE ratio | worst mid/bottom modewise | gate |
| --- | --- | --- | --- | --- |
| pretrained | 3840 | 19.5907 | 1.033 | False |
| finetuned | 7680 | 19.5644 | 0.954 | False |

Training split only; validation, inference, held S0, response, and
adjoint archives remained sealed for this run.

Report content SHA-256: `6de25131135617392d4d182ec2d5f19368a0e87d7587481d968e2689f055dd6c`.
