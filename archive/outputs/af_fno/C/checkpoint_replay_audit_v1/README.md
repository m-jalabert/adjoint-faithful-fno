# Model C exact-replay checkpoint audit

Classification: `objective_correction_required`.

Exact replay passed: `True`.

| Optimizer step | Full gate | Worst primary RMSE-AUC ratio | Worst SST/P-rho lead ratio |
| ---: | :---: | ---: | ---: |
| 11520 | False | 9.259 | 17.632 |
| 13440 | False | 2.397 | 5.600 |
| 14400 | False | 2.366 | 5.513 |
| 14880 | False | 2.594 | 5.825 |
| 15120 | False | 2.338 | 5.533 |
| 15360 | False | 2.511 | 6.067 |

The figure shows mean SST and surface P/rho RMSE for all six replayed late
checkpoints against persistence and training climatology. Solid ratio curves
use persistence; dotted ratio curves use climatology. Values below one win.

The CSV contains every plotted mean and both ratios. The summary and manifest
bind these project-facing files to the immutable scratch report and arrays.
Manifest content SHA-256: `75a32afb5b2e829326b2ab6865f480db34b970d56d9495155ba859b8c57b62b2`.
