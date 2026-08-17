# Local24 streamfunction anomalies, S0 — companions to figures 3 and 7

These two plates **add to** the published figure package; they replace nothing.
`model_c_bire_figure3a_streamfunction_anomaly_1deg_s0_dt10.png` and `model_c_bire_figure7a_streamfunction_anomaly_day060_day2000_s0.png` show

    psi'(x, y, t) = psi(x, y, t) - psi_bar_S0(x, y)

with `psi_bar_S0` the **MITgcm** time-mean barotropic streamfunction over the S0
**training** block, days 0–5999, averaged over
6,000 days. The same two-dimensional field is
subtracted from truth and from the model. The model's own time mean is
deliberately *not* used: that would absorb any bias in the mean circulation the
model carries, which is the error most worth seeing.

The reference field itself is published as
`model_c_bire_reference_time_mean_streamfunction_s0.png`, range
-30.01 to 30.87 Sv,
RMS 10.67 Sv.

## Why the total-field plates were not enough

On a ±33 Sv scale the stationary double gyre fills the plate. The transients
this model actually had to learn are roughly
0.19 Sv RMS at day 2,000 — about
2% of
the mean field's own RMS. A model can carry the mean gyres correctly and still
get the variability badly wrong without the total-field plate showing it.

| lead | truth psi' RMS | model psi' RMS | ratio |
| --- | --- | --- | --- |
| day 60 | 0.241 Sv | 0.265 Sv | 1.097 |
| day 2,000 | 0.194 Sv | 0.748 Sv | 3.846 |

A ratio below one means damped transients; above one, manufactured ones.

## What this does not change

Amplitude diagnostics stay on the **total** field. In particular, the
acceptance gate's day-2,000 streamfunction minimum remains a statement about
mean-circulation intensity and is unaffected by anything here. These plates
answer a different question: whether the local24 model reproduces variability
*about* that mean.

Member 0 of the 15-member S0 ensemble, the same member figures 3 and 7 plot.

Report content SHA-256: `cee742f9968f1c51c708ce8ff82a997489586e5c907fdfc3e2fe779a7cb6e7a8`.
