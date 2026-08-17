# Two-in / one-out streamfunction anomalies, S0 — companions to figures 3 and 7

These plates reuse the frozen anomaly definition exactly:

    psi'(x, y, t) = psi(x, y, t) - psi_bar_S0(x, y)

`psi_bar_S0` is MITgcm's two-dimensional time-mean barotropic streamfunction
over training days 0--5999. The identical field
is subtracted from truth and the selected two-input model; the model's own mean
is never subtracted. The three PNG definitions and member 0 inputs are unchanged
from the 32x32 arm, while this package writes to its own output root.

| lead | truth anomaly RMS | 2-in / 1-out anomaly RMS | ratio |
| --- | --- | --- | --- |
| day 60 | 0.241 Sv | 0.264 Sv | 1.096 |
| day 2,000 | 0.194 Sv | 0.530 Sv | 2.723 |

At day 2,000, normalized first-difference RMS is
0.509 meridionally and
0.733 zonally; directional high-wavenumber fractions
are 0.00311 and 0.09999.
The report also retains western-four-cell and interior anomaly RMS diagnostics.

This package reads the sealed two-input figure arrays and model-visible MITgcm
training state only. It rolls out no model, promotes nothing, and does not
modify the total-field figures or acceptance gate.

Report content SHA-256: `4dcb05bcb803ff1ae06c01c3c42b379675dff6a2bffad511933952104a29e147`.
