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
| day 60 | 0.241 Sv | 0.249 Sv | 1.032 |
| day 2,000 | 0.194 Sv | 0.495 Sv | 2.544 |

At day 2,000, normalized first-difference RMS is
0.522 meridionally and
0.634 zonally; directional high-wavenumber fractions
are 0.00257 and 0.02603.
The report also retains western-four-cell and interior anomaly RMS diagnostics.

This package reads the sealed new-channel figure arrays and model-visible MITgcm
training state only. It rolls out no model, promotes nothing, and does not
modify the total-field figures or acceptance gate.

Report content SHA-256: `8a8b64c74fc85339d80e39214fcf9d4a73e3b6a62284d4ed0fcc706309910bf1`.
