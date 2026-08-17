# Physical-static-channel streamfunction anomalies, S0 — companions to figures 3 and 7

These plates reuse the frozen anomaly definition exactly:

    psi'(x, y, t) = psi(x, y, t) - psi_bar_S0(x, y)

`psi_bar_S0` is MITgcm's two-dimensional time-mean barotropic streamfunction
over training days 0--5999. The identical field
is subtracted from truth and the selected new-channel model; the model's own
mean is never subtracted. The three PNG definitions and member 0 inputs are
unchanged from the preceding arm, while this package writes to its own root.

| lead | truth anomaly RMS | physical-statics anomaly RMS | ratio |
| --- | --- | --- | --- |
| day 60 | 0.241 Sv | 0.257 Sv | 1.065 |
| day 2,000 | 0.194 Sv | 0.459 Sv | 2.360 |

At day 2,000, normalized first-difference RMS is
0.615 meridionally and
0.761 zonally; directional high-wavenumber fractions
are 0.00313 and 0.02277.
The report also retains western-four-cell and interior anomaly RMS diagnostics.

This package reads the sealed new-channel figure arrays and model-visible MITgcm
training state only. It rolls out no model, promotes nothing, and does not
modify the total-field figures or acceptance gate.

Report content SHA-256: `74703eb74c6a0c81668baa8e86d7d013a00e84a58a4ecfaa4d5c52d9bd536692`.
