# Local3x3 + 24x24-mode fine-tune, S0: Figures 3--8

This package evaluates the selected step-3,840 checkpoint of the local24
fine-tune on the **S0** inference set (indices 6200--7199), tau0 =
0.1 N m-2.

The fine-tune started from the step-3,840 checkpoint of
`model_c_bire_protocol_rollout_ft_v1`. The six-step objective, normalization,
split, static inputs, positional encoding and 46-channel direct-state output
are retained. The architecture adds a zero-initialized, bias-free 3x3 local
49-to-46 correction and expands the Fourier modes from 24x16 to 24x24. The
existing spectral coefficients are embedded unchanged and the new zonal
coefficients start at zero, so the new model initially reproduces its parent
exactly before fine-tuning.

The starts are drawn from 6200--6999, this draw spanning 6263--6979,
so every member has lead-matched MITgcm truth to day 2,000
(6979 + 2000 = 8979 < 9000) from days 7200--8999, which the
model never saw in any capacity.

**Figure 6 is a literal parent / fine-tune pair.** The black curve is the
archived step-3,840 rollout model the fine-tune started from; the
red curve is the selected step-3,840 local24 model. They use the same
data, normalized coordinates and six-step objective. The architecture of each
checkpoint is verified independently before its weights are loaded.

Both checkpoints carry the same six-step objective hash; their declared
architectures differ exactly in the local branch and retained zonal modes.

Climatology is the pointwise S0 mean over the Bire training block
(0--5999) only. Persistence holds each member's initial physical field fixed.
RMSE is computed over wet cells per member; lines and bands are the mean and
10th/90th percentiles across the 15 members.

The 2,000-day half of the final acceptance gate is written beside this folder as
`bire_protocol_rollout_ft_local24_acceptance_gate.json`.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `d6566b5ddc81978bd993e42cf5bbbcd8ef73e20e50bab8f7bd1406056ba42208`.
