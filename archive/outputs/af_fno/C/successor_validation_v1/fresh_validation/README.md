# Model C successor fresh validation

Status: operationally complete and scientifically rejected by the frozen
validation-v2 gate.

GPU job 290738 applied the prospectively frozen contract to all three
width-128 checkpoints, 180 complete 10--90-day starts per regime, and the
persistence, training-climatology, and frozen-A0 baselines. Inference and all
later archives remained sealed.

The outcome is not an SSH-only rejection. All three seeds are excellent for
surface speed and beat A0 for every primary field. At day 10 they also beat
persistence for basin-mean SST and derived surface pressure, except for a
small low-wind SST miss in the regime breakdown. The slow fields then
accumulate autoregressive error: their persistence ratios cross one around
days 20--30 and reach roughly 4.5--5 at day 90.

The three-seed bootstrap-mean RMSE-AUC ratios are:

| Field | Persistence | Climatology | A0 |
| --- | ---: | ---: | ---: |
| Surface speed | 0.245 | 0.275 | 0.175 |
| SST | 2.313 | 2.824 | 0.220 |
| Surface PHIHYD | 2.198 | 0.682 | 0.165 |

Surface speed passes every comparison with confidence. SST fails the
persistence and climatology RMSE requirements despite positive ACC
improvements. Surface pressure fails persistence RMSE and ACC, while passing
climatology and A0.

Every rollout is finite, land remains exactly zero, SSH and streamfunction
amplitudes remain near truth, and the wind-response slope passes. The
predeclared normalized-maximum ceiling of 20 is exceeded by every seed
(`21.169--21.367`), but the corresponding validation truth itself reaches
`21.092`. This makes that ceiling non-discriminating; it does not change the
rejection because the primary slow-field metrics fail independently.

The result resolves the current diagnosis: doubling data and restoring
Bire-style capacity solved one-step fitting and generalization, but the
three-step/full-state rollout objective and amplitude-only long-run training
gate did not control 20--90-day slow-field trajectory error.

`validation_summary.json` is the lightweight project-facing evidence. The
complete report and member-level arrays remain immutable in scratch.

## Figures

The following figures were generated directly from the hash-verified immutable
member-level arrays:

| Figure | Contents |
| --- | --- |
| `model_c_primary_rmse_vs_lead.png` | Absolute RMSE curves for surface speed, SST, and derived surface pressure |
| `model_c_primary_acc_vs_lead.png` | ACC curves for the three primary fields |
| `model_c_primary_rmse_ratio_vs_persistence.png` | Lead-dependent RMSE divided by persistence; the horizontal line at one is the decision boundary |
| `model_c_primary_regime_rmse_ratio.png` | Primary-field persistence ratios separated by S0/S1/S2 wind regime |
| `model_c_all_field_rmse_auc_ratios.png` | 10--90-day RMSE-AUC ratios for every reported state and derived field |

`figure_manifest.json` records every figure hash, the plotted lead times and
field summaries, and the exact source report/array hashes. Its content SHA-256
is `8b24b6af068601b79fb8e7fda22009e3b6aa4c42f4e8b1d76360386101a0845f`.
The plotting step did not read inference or any later sealed archive.
