# Model C training-only rollout diagnosis

Status: operationally complete. The predeclared classification is
`training_objective_or_checkpoint_gate_mismatch`.

Every seed reproduces the fresh-validation pattern on fixed split-1 chronology:
the slow fields beat persistence at day 10, but lose over the 10--90-day curve
and at day 90.

| Field | Day 10 | 10--90-day RMSE-AUC | Day 90 |
| --- | ---: | ---: | ---: |
| SST | 0.809 | 2.021 | 4.221 |
| Surface PHIHYD | 0.552 | 1.867 | 3.873 |
| SSH | 0.532 | 1.844 | 3.835 |

Values are Model C RMSE divided by persistence; lower than one is better.
The result supports revising long-rollout supervision/checkpoint selection
before attributing the failure to inadequate data or opening inference.

`diagnosis_summary.json` is the lightweight numerical evidence. Three figures
show training curves, training-versus-validation slow-field drift, and all-field
RMSE-AUC ratios. `figure_manifest.json` binds them to the immutable report and
array hashes; its content SHA-256 is `664e66a0866d8af85ca33561b6892bf64e426c7049fff37691ab4094f5393671`.
