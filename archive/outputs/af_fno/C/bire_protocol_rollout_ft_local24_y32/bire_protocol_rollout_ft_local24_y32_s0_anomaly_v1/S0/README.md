# Y32 streamfunction anomalies, S0 — companions to figures 3 and 7

These plates reuse the frozen anomaly definition exactly:

    psi'(x, y, t) = psi(x, y, t) - psi_bar_S0(x, y)

`psi_bar_S0` is MITgcm's two-dimensional time-mean barotropic streamfunction
over training days 0--5999. The identical field
is subtracted from truth and the selected 32x24 model; the model's own mean is
never subtracted. The three PNG definitions and member 0 inputs are unchanged
from local24, while this package writes to a distinct Y32 output root.

| lead | truth anomaly RMS | Y32 anomaly RMS | ratio |
| --- | --- | --- | --- |
| day 60 | 0.241 Sv | 0.262 Sv | 1.088 |
| day 2,000 | 0.194 Sv | 0.674 Sv | 3.464 |

At day 2,000, normalized first-difference RMS is
0.427 meridionally and
0.791 zonally; directional high-wavenumber fractions
are 0.00230 and 0.07616.
The report also retains western-four-cell and interior anomaly RMS diagnostics.

This package reads the sealed Y32 figure arrays and model-visible MITgcm
training state only. It rolls out no model, promotes nothing, and does not
modify the total-field figures or acceptance gate.

Report content SHA-256: `a730a03b9ff05becf0cc63471490e3bf8400c1622642540fa708a98504c72047`.
