# Production emulator, S0_turb: Figures 3--8

This held package evaluates the selected step-5,760 checkpoint of
`turb_forward_control_v1` on the 15-member S0_turb inference protocol.

The model is the one-input production operator: `F(x_t, S) -> x_(t+10)` with
`x_t` the 46-channel state at one time level and `S` the five physical static
channels `[tau_x, wet, f, dx, theta_clim]`; 32 x 32 Fourier modes, width 128,
three blocks, a bias-free local 3 x 3 path and the deterministic sine/cosine
position encoder. It was trained once from random initialization, so there is no
predecessor to compare against and every plate carries a single model curve
against persistence and climatology.

Starts span 6263--6979 and are drawn from the
inference block [6200, 7200] nested inside validation
[6000, 7200]; there is no independent third test split. Every member
has lead-matched MITgcm truth through day 2,000.

The measurable gate is written beside the S0_turb folder as `turb_forward_control_v1_acceptance_gate.json`. This
package performs no training, selection, or checkpoint promotion.

Report content SHA-256: `7f86f0a6e1c0802b1d224d1a5d5f9589c0c3820a69b40a10cd8ed214abe22db5`.
