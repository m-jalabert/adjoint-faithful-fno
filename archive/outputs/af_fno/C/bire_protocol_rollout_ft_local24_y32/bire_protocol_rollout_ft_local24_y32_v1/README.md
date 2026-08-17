# Meridional-32 continuation of the retained local24 model

This controlled arm warm-starts `model_c_bire_protocol_rollout_ft_local24_v1` at optimizer step
3,840. Only the Fourier mode count in tensor order
(Y, X) changes, from 24 x 24 to 32 x 24. The trained bias-free 49 -> 46 local
3 x 3 branch is copied unchanged. Each 24 x 13 complex spectral tensor is
embedded in indices 4:28 of a zeroed 32 x 13 tensor, preserving the initial
map exactly while opening four new negative and four new positive meridional
modes.

Dataset, split, normalization, six-step autoregressive loss, optimizer reset,
schedule, seed, validation starts, and checkpoint-selection rule are inherited
byte-for-byte from the local24 parent. Zonal capacity remains fixed because the
parent already carries excessive zonal high-wavenumber power.

| step | short AUC 10--90 (speed / SST / pressure) | long / climatology |
| --- | --- | --- |
| 960 | 0.107 / 1.200 / 0.561 | 0.520 / 0.662 / 0.247 |
| 1,920 | 0.107 / 1.170 / 0.438 | 0.498 / 0.610 / 0.198 |
| 2,880 | 0.105 / 1.141 / 0.487 | 0.475 / 0.616 / 0.220 |
| 3,840 | 0.103 / 1.134 / 0.406 | 0.487 / 0.619 / 0.203 |

Selected step 3,840 via
`primary_rule`. Validation gate:
**pass**. The S0-only
2,000-day figures and anomaly diagnostics are intentionally deferred to the
next versioned job.

Parameter count: 21,005,164.
Report content SHA-256: `3deca0d460ffa7806f64b5dba7641a5cdc160f912517878951cf2eabd2360bdc`.
