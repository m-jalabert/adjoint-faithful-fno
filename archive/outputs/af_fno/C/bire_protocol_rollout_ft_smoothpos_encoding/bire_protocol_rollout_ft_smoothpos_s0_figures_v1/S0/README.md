# Smooth-position local Model C, S0: Figures 3--8

This held package evaluates the selected step-2,880 checkpoint of
`model_c_bire_protocol_rollout_ft_smoothpos_v1` on the exact 15-member S0 inference protocol used for
local24 and Y32. The black Figure 6 curve is its literal step-3,840
Y32 parent; the red curve is the smooth-position fine-tune. Both retain the
trained bias-free 49-to-46 local 3x3 path, the 32x24 modes, and the same
six-step objective. The only change is the deterministic position encoding:
the alternating sine/cosine parity pattern is replaced by smooth monotone x
and y coordinates on [-1, 1].

Starts span 6263--6979; every member has
lead-matched truth through day 2,000. Climatology remains the pointwise
S0 training-block mean and persistence holds the initial physical state.
The numerical reductions, plot functions, filenames and lead grid are reused
unchanged from the preceding Y32 package.

The measurable gate is written beside the S0 folder as `bire_protocol_rollout_ft_smoothpos_acceptance_gate.json`. This
package performs no training, selection, or checkpoint promotion.

Report content SHA-256: `c99c1a7f2d125a64ab68e33cdeef6d5c99b16144297c67115193a4e627f625c9`.
