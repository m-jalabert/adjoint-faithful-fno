# Model C Bire-style per-lead streamfunction maps

Status: complete descriptive characterization of the scientifically
rejected Model C successor. This package cannot authorize tuning or
inference.

The eight figures show S2 validation start 6335 separately at days
20, 30, ..., 90. Each contains MITgcm truth, Model C prediction, and
truth-minus-prediction at the native 1-degree resolution. Truth and
prediction share one scale across every lead. Each error map uses its
own labeled symmetric scale so the spatial error remains visible.

| Lead (days) | RMSE (Sv) | Max absolute error (Sv) | Relative RMSE |
|---:|---:|---:|---:|
| 20 | 0.093444 | 0.723955 | 0.7003% |
| 30 | 0.158044 | 1.128493 | 1.1844% |
| 40 | 0.176499 | 1.075172 | 1.3227% |
| 50 | 0.173707 | 0.942432 | 1.3017% |
| 60 | 0.181428 | 0.720797 | 1.3596% |
| 70 | 0.214835 | 0.745377 | 1.6100% |
| 80 | 0.258320 | 0.957977 | 1.9358% |
| 90 | 0.295096 | 1.171301 | 2.2112% |

The complete report and arrays remain immutable in scratch.
Figure-manifest content SHA-256: `3b74d759d551588f8edf6ff944a4177344ea8a8efc81703d38719e4a95450ac7`.
