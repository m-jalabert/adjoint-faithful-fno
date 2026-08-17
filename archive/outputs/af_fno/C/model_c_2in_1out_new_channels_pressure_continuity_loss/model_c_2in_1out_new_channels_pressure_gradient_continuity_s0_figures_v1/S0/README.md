# Physical static channels, S0: Figures 3--8

This held package evaluates the selected step-3,840 checkpoint of
`model_c_2in_1out_new_channels_pressure_gradient_continuity_v1` on the exact 15-member S0 inference protocol used for every
preceding arm. The black Figure 6 curve is its literal step-3,840
parent; the red curve is this arm. Both read the same two time levels
`(x_(t-10), x_t)`, carry 32x32 Fourier modes, the trained bias-free local 3x3
path, the deterministic sine/cosine position encoder and the same six-step
objective. The only difference is which environmental fields they are handed:

    parent    tau_x, wet mask, distance to wall
    this arm  tau_x, wet mask, f(phi), dx(phi), theta_clim(x, y)

Both models read the same extra initial condition, the truth state ten days
before each start. That day is model-visible and is never a scored target, so
the leads, truth, climatology and persistence baselines are identical to the
compared packages. Starts span 6263--6979; every
member has lead-matched truth through day 2,000. The numerical reductions, plot
functions, filenames and lead grid are reused unchanged.

The measurable gate is written beside the S0 folder as `model_c_2in_1out_new_channels_pressure_gradient_continuity_acceptance_gate.json`. This
package performs no training, selection, or checkpoint promotion.

Report content SHA-256: `e5b5b8c487a585fca7438ce35b6a0df46cc020a2ca439a0a88e39e2464b0c384`.
