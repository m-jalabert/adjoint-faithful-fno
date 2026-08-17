# Two-in / one-out continuation of the 32 x 32 Model C

This model warm-starts `model_c_bire_protocol_rollout_ft_y32_x32_v1` at optimizer step
3,840. Only the input contract changes:

    one-in / one-out:   x_t                -> x_(t+10)
    two-in / one-out:  (x_(t-10), x_t)     -> x_(t+10)

and autoregression slides the pair forward, so a self-generated rollout reads
`(x_t, xhat_(t+10)) -> xhat_(t+20)` and then
`(xhat_(t+10), xhat_(t+20)) -> xhat_(t+30)`. The pair gives the operator an
empirical tendency `(x_t - x_(t-10)) / 10 days`, which is the multistep idea
behind Adams--Bashforth in spirit; it is not AB-II's algebra, since MITgcm
extrapolates from two stored *tendencies* rather than two states.

The external input block therefore grows from 46 + 3 = 49 channels to
2 x 46 + 3 = 95, and lifting from 51 to
97. The two input-facing tensors --- the lifting
weight and the bias-free local 3 x 3 weight --- gain 46 leading input channels
that begin at exact zero, and the parent's state, static and position columns
are copied into the trailing slice. At initialization the model therefore
*ignores* the history state and reproduces the parent's map on `x_t` for any
history whatsoever, up to float32 summation order.

The 32 x 32 Fourier modes, the trained local branch, the deterministic
sine/cosine position encoder, the dataset, the normalizers, the six-step
autoregressive loss, the optimizer reset, the schedule, the seed, the validation
starts and the checkpoint-selection rule are inherited unchanged, so temporal
context is the only thing this arm tests.

Training draws 5,930 starts per
regime (17,790 pooled), from
10 to
5,939: days 0--9 are the only starts the
one-input arm had that the history requirement removes, and no target moved.

| step | short AUC 10--90 (speed / SST / pressure) | long / climatology |
| --- | --- | --- |
| 960 | 0.098 / 1.065 / 0.536 | 0.440 / 0.543 / 0.196 |
| 1,920 | 0.098 / 1.051 / 0.411 | 0.471 / 0.592 / 0.193 |
| 2,880 | 0.097 / 1.031 / 0.431 | 0.448 / 0.523 / 0.199 |
| 3,840 | 0.095 / 1.021 / 0.360 | 0.436 / 0.541 / 0.166 |

The step-3,840 one-input baseline scores
0.099 / 1.073 / 0.377 short and
0.455 / 0.569 / 0.180 long on the
same 102 pooled rollouts, in the order
surface_speed, sst, phihyd_surface.

Selected step 3,840 via
`primary_rule`. Validation gate:
**pass**. The S0-only
2,000-day figures and anomaly diagnostics are evaluated by the canonical
held-inference package.

Parameter count: 27,327,440.
Report content SHA-256: `64a5c0aaceada98a7b914c339a939bc95777a711122fdb141ffb874fdf3bc8b4`.
