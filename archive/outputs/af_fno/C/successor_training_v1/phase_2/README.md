# Model C successor training v1 — phase 2

Status: operationally complete and scientifically passed the prospective
training-only capacity gate.

The width-128 candidate restores Bire's absolute latent width and uses
lift/projection width 256 with a `4C` Channel MLP. It retains the same trajectory-v2
training records, loss v1, `(24,16)` Fourier modes, four layers, 10% padding,
optimizer exposure, and seed used by the bounded successor contract.

Over all 15,060 training pairs, its U/V/temperature/SSH RMSE ratios to persistence
are `0.049040/0.121279/0.384816/0.522413`. Every group beats persistence separately
in S0, S1, and S2. The former bottleneck, S1 SSH, is now `0.762683`.

The checkpoint reloads bitwise exactly. Twelve training-only long rollouts remain
finite through 180 days, and every normalized amplitude ratio at days 90 and 180
lies inside the declared `[0.5, 2]` interval.

`phase_2_summary.json` is the lightweight project copy of the decision evidence.
The complete report and 116 MB checkpoint remain immutable under:

`/bigscratch/mjalabert314/bire_james25_repro/af_fno/models/C/successor_training_v1/phase_2/v2_bireprop_w128_mlp4/`

The result authorizes a prospectively frozen fresh-validation protocol. It does not
by itself freeze Model C or authorize inference. Fresh validation, inference,
intermediate-wind, response, and adjoint data were not read by this job.
