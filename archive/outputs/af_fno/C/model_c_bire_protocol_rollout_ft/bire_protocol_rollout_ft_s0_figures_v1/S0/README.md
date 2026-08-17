# Six-step rollout fine-tune, S0: Figures 3--8

This package evaluates the step-3,840 checkpoint of the six-step rollout
fine-tune on the **S0** inference set (indices 6200--7199), tau0 =
0.1 N m-2.

The fine-tune started from the step-15,360 checkpoint of
`model_c_bire_protocol_duration_v1` and continued for 3,840 steps at 2e-5 with
the autoregressive rollout deepened from three ten-day calls to six and the
rollout weight raised from 0.15 to 0.50. Architecture, normalization, split,
Fourier modes, static inputs, positional encoding and the 46-channel output are
unchanged, so this package and
`outputs/af_fno/C/bire_protocol_duration_s0_figures_v1/S0/` differ only in the
checkpoint: same 15 members, same seed, same lead grid, same truth window, same
climatology and persistence baselines.

The starts are drawn from 6200--6999, this draw spanning 6263--6979,
so every member has lead-matched MITgcm truth to day 2,000
(6979 + 2000 = 8979 < 9000) from days 7200--8999, which the
model never saw in any capacity.

**Figure 6 is a literal pre-train / fine-tune pair.** The black curve is the
step-15,360 model the fine-tune started from; the red curve is the
selected step-3,840 fine-tuned model. Both were trained on the same data in
the same normalized coordinates by the same code path, so the gap between them
is what deepening the rollout bought and nothing else. Every earlier package in
this project could only offer a within-run training-progress comparison.

The two checkpoints carry different objective hashes -- v1 over three steps for
the comparator, the six-step contract for the selected model -- and each is
verified against its own.

Climatology is the pointwise S0 mean over the Bire training block
(0--5999) only. Persistence holds each member's initial physical field fixed.
RMSE is computed over wet cells per member; lines and bands are the mean and
10th/90th percentiles across the 15 members.

The 2,000-day half of the final acceptance gate is written beside this folder as
`bire_protocol_rollout_ft_acceptance_gate.json`.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `e54675e2a5bc0932ef673e2b6fb827e95ade807790fc47bfd597ed8869e3b57f`.
