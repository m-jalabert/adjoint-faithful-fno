# Smooth-position continuation of the retained Y32 model

This model warm-starts `model_c_bire_protocol_rollout_ft_local24_y32_v1` at optimizer step
3,840. Every checkpoint tensor is copied bitwise and the
optimizer is reset. The only architecture change replaces the alternating
Bire position values with two smooth monotone Cartesian coordinates: x and y
each run from -1 to +1 on the 62 x 62 grid. They retain the same two channels,
order, shape, float32 dtype, and non-persistent-buffer status.

This is tensor-preserving but deliberately not function-preserving: the frozen
weights receive different deterministic position values before the first
optimizer update. The preflight report therefore records the initial map delta
on one training-only state from each wind regime instead of asserting zero.

Dataset, split, normalization, six-step autoregressive loss, optimizer reset,
schedule, seed, validation starts, and checkpoint-selection rule are inherited
byte-for-byte from the recursively materialized Y32 parent. Fourier modes stay
32 x 24 (Y, X), width stays 128, the three FNO blocks and six pointwise
LayerNorms are unchanged, and the trained bias-free 49 -> 46 local 3 x 3 branch
is copied exactly.

| step | short AUC 10--90 (speed / SST / pressure) | long / climatology |
| --- | --- | --- |
| 960 | 0.104 / 1.163 / 0.531 | 0.509 / 0.618 / 0.228 |
| 1,920 | 0.103 / 1.134 / 0.431 | 0.488 / 0.574 / 0.182 |
| 2,880 | 0.101 / 1.098 / 0.393 | 0.480 / 0.563 / 0.175 |
| 3,840 | 0.100 / 1.095 / 0.385 | 0.478 / 0.585 / 0.183 |

Selected step 2,880 via
`primary_rule`. Validation gate:
**pass**. The S0-only
2,000-day figures and anomaly diagnostics are evaluated by the canonical
held-inference package.

Parameter count: 21,005,164.
Parent checkpoint SHA-256: `238c255b9b7daf54a2ea2be5e815b65919642a46eb240fd7a0046ec1b88dba4e`.
Position encoding: `smooth_xy` on [-1, 1].
Report content SHA-256: `e7c9edd4519b3e7d7110bc1cd4a98c55151608c589e78effecd61495f323e66a`.
