# Smooth-position streamfunction anomalies, S0 — companions to figures 3 and 7

These plates reuse the frozen anomaly definition exactly:

    psi'(x, y, t) = psi(x, y, t) - psi_bar_S0(x, y)

`psi_bar_S0` is MITgcm's two-dimensional time-mean barotropic streamfunction
over training days 0--5999. The identical field
is subtracted from truth and the selected smooth-position model; the model's
own mean is never subtracted. The three PNG definitions and member 0 inputs are
unchanged from Y32, while this package writes to a distinct smooth-position
output root.

| lead | truth anomaly RMS | smooth-position anomaly RMS | ratio |
| --- | --- | --- | --- |
| day 60 | 0.241 Sv | 0.295 Sv | 1.222 |
| day 2,000 | 0.194 Sv | 1.072 Sv | 5.511 |

At day 2,000, normalized first-difference RMS is
0.291 meridionally and
0.854 zonally; directional high-wavenumber fractions
are 0.00163 and 0.04475.
The report also retains western-four-cell and interior anomaly RMS diagnostics.

This package reads the sealed smooth-position figure arrays and model-visible MITgcm
training state only. It rolls out no model, promotes nothing, and does not
modify the total-field figures or acceptance gate.

Report content SHA-256: `8eeec291b0f2fb37be8e5a6212d98afa8dc59e46da7c68bca3570bcd85684bcb`.
