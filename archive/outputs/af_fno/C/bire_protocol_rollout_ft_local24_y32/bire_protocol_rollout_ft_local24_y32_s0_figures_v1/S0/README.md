# Meridional-32 local Model C, S0: Figures 3--8

This held package evaluates the selected step-3,840 checkpoint of
`model_c_bire_protocol_rollout_ft_local24_y32_v1` on the exact 15-member S0 inference protocol used for
local24.  The black Figure 6 curve is its literal step-3,840
local24 parent; the red curve is the 32x24 fine-tune.  Both retain the trained
bias-free 49-to-46 local 3x3 path and the same six-step objective.  Only four
new meridional modes on each side were opened during fine-tuning; zonal modes
remain fixed at 24.

Starts span 6263--6979; every member has
lead-matched truth through day 2,000. Climatology remains the pointwise
S0 training-block mean and persistence holds the initial physical state.
The numerical reductions, plot functions, filenames and lead grid are reused
unchanged from `oceanfno.figures`.

The measurable gate is written beside the S0 folder as `bire_protocol_rollout_ft_local24_y32_acceptance_gate.json`. This
package performs no training, selection, or checkpoint promotion.

Report content SHA-256: `55a7cf0001e721d3b8a890201fc5777a6de4bda9cea50b4685d88a0ada04c768`.
