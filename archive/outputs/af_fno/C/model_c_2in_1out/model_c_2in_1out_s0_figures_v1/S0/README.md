# Two-in / one-out Model C, S0: Figures 3--8

This held package evaluates the selected step-3,840 checkpoint of
`model_c_2in_1out_v1` on the exact 15-member S0 inference protocol used for
local24, Y32 and the 32x32 arm. The black Figure 6 curve is its literal
step-3,840 one-input parent; the red curve is the two-input
fine-tune. Both carry 32x32 Fourier modes, the trained bias-free local 3x3
path, the deterministic sine/cosine position encoder and the same six-step
objective. The only difference is what each call is handed: the parent reads
`x_t`, this arm reads the pair `(x_(t-10), x_t)` and slides it forward at every
autoregressive step.

Each member therefore reads one extra initial condition, the truth state ten
days before its start. That day is model-visible and is never a scored target,
so the leads, truth, climatology and persistence baselines are identical to the
compared packages. Starts span 6263--6979; every
member has lead-matched truth through day 2,000. The numerical reductions, plot
functions, filenames and lead grid are reused unchanged.

The measurable gate is written beside the S0 folder as `model_c_2in_1out_acceptance_gate.json`. This
package performs no training, selection, or checkpoint promotion.

Report content SHA-256: `df37a7625d9fc428faa1c100d4ff9a139de7c2f70e979304b43fc20934d78458`.
