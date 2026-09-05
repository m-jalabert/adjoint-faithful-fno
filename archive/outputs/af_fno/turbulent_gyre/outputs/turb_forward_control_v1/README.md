# Production emulator, smooth FFT-boundary padding

    F_theta: [x_t, S] -> x_(t+10)

`x_t` is the 46-channel prognostic state at one time only; `S` is the five
physical static channels. 51 external channels, plus two deterministic
sine/cosine position channels inside the model, so lifting sees 53 and the
output is 46. A 64 x 64-mode, width-128, three-block FNO with six pointwise
LayerNorms, a 4C Channel MLP, 10 % raised-cosine tapered replicate padding and a
parallel bias-free 3 x 3 local correction. Parameter count:
104,368,296.

## Production update

The first turbulent rollout developed narrowband zonal stripes next to the
domain edge. Disabling the local 3 x 3 path did not suppress them and degraded
SST and pressure skill. Replacing the constant-zero latent FFT halo with a
replicated halo brought smoothly to zero reduced the diagnostic stripe-band
power fraction from 0.478 to
0.261. The outer FFT boundary is exactly
zero and periodically continuous, while the original lifted field is untouched.

This is the only production revision. The local branch, Fourier modes, width,
blocks, loss, data, optimizer, learning-rate schedule and seed are unchanged;
the weights and training-only normalizers are recomputed from scratch.

## Retained spectral normalization

The channel mixing at Fourier mode `k` is a dense complex matrix
`R_k` in `C^(128 x 128)`; there are 6,336 of them across
the three blocks, holding 99.46 % of the parameters. Each remains capped:

    R_k <- R_k * min(1, 1 / sigma_max(R_k))

estimated by a persistent alternating power iteration, one step per forward.
Modes already at or below one are left exactly alone. This is a
reparameterization of the operator, **not** a bound on the emulator Jacobian:
each block is `GELU(K_R h + W h)` with its own skips, so legitimate transient
amplification remains representable.

At initialization every mode of block 0 had `sigma_max` above one
(mean 1.955, max 2.030); by the final
checkpoint the free parameter's mean was 2.916 with
2,112 of 2,112 modes above one and therefore
actively held down.

Checkpoints are written **materialized** --- the normalized weights are baked
into the tensor --- so a published checkpoint loads into an unmodified model and
the inference layer is simply `y_hat(k) = R_k_tilde x_hat(k)`, whose adjoint is
`R_k_tilde^H`.

The contraction *penalty* of the two preceding arms is removed. Perturbation
growth is still measured and still gates checkpoint selection, but contributes
no gradient, so this run isolates the operator constraint.

## Training

From random initialization: no parent checkpoint, no migration, no inherited
normalization. Both normalizers are recomputed over training days
0--5999 of S0, S1 and S2 only.

Six-step autoregressive, no teacher forcing after the initial state:
17,820 sequences
(5,940 starts per regime),
7,680 optimizer steps at batch 8
(microbatch 1 x accumulation 8) =
368,640 state transitions. Learning rate 0.0005 through
step 5,760, then
0.0001.

    L = L_state + 0.001 L_inc + 0.50 L_rollout + 1e-5 L_spectral
        + 0.065 L_boundary + 0.05 L_pressure + 0.05 L_continuity
        + 0.05 L_barotropic

## Selection

| step | short AUC 10--90 (speed / SST / pressure) | long / climatology | growth rate |
| --- | --- | --- | --- |
| 1,920 | 3.078 / 10.739 / 20.968 | 1.181 / 1.090 / 0.992 | 1.0419 |
| 3,840 | 3.038 / 10.315 / 20.157 | 1.190 / 1.085 / 0.982 | 1.0419 |
| 5,760 | 3.066 / 10.301 / 20.391 | 1.184 / 1.095 / 0.974 | 1.0428 |
| 7,680 | 3.080 / 10.230 / 20.138 | 1.198 / 1.091 / 0.966 | 1.0377 |

Selected step 7,680 via
`declared_fallback_no_checkpoint_met_the_growth_rate_ceiling`, growth rate 1.03770 per call
(ceiling 1.0; the unconstrained arm measured
1.0168). Validation gate:
**fail**.

Evaluation uses the nested validation/inference protocol; there is no
independent third test split.

Report content SHA-256: `0d4dabc6cc078dfa2144bfeb770353e4963f87b8866287faa052e6bb37330be8`.
