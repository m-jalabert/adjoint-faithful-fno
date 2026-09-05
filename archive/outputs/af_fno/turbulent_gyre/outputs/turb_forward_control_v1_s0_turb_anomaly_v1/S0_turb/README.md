# Production emulator streamfunction anomalies, S0 — companions to figures 3 and 7

    psi'(x, y, t) = psi(x, y, t) - psi_bar_S0(x, y)

`psi_bar_S0` is MITgcm's two-dimensional time-mean barotropic streamfunction
over training days 0--5999. The identical field is
subtracted from truth and from the model; the model's own mean is never
subtracted, so a bias in the standing gyre cannot hide inside the anomaly.

| lead | truth anomaly RMS | model anomaly RMS | ratio |
| --- | --- | --- | --- |
| day 60 | 9.044 Sv | 8.201 Sv | 0.907 |
| day 2,000 | 7.367 Sv | 9.641 Sv | 1.309 |

At day 2,000 the model's normalized first-difference RMS is
0.101 meridionally and 0.143
zonally, against 0.169 and
0.179 in truth; the directional high-wavenumber power
fractions are 0.00000 and 0.00008
against 0.00000 and 0.00000.

The western-boundary concentration is reported explicitly, because no gate scores
it: the model's first-four-wet-cell to interior anomaly RMS ratio is
1.967 against
2.378 in truth.

This package reads the sealed figure arrays and model-visible MITgcm training
state only. It rolls out no model, promotes nothing, and does not modify the
total-field figures or the acceptance gate.

Report content SHA-256: `2e0b67196fff05eb4a395b2bd5d24e71c259778d5c05cac405459aee040bead2`.
