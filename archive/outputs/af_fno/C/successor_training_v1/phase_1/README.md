# Model C successor training v1 — phase 1

Status: both candidates completed operationally and were scientifically rejected by
the prospective training-only capacity gate.

The data-only width-64 control missed only low-wind S1 SSH, at 1.103435 times
persistence over all 5,020 S1 training pairs. Restoring the Bire-style Channel-MLP
expansion from `0.5C` to `4C`, with every other setting fixed, reduced this ratio to
1.003573. It also improved the aggregate U/V/temperature/SSH ratios by
23.6%/22.2%/9.4%/7.9%, respectively.

The `4C` result is meaningful evidence that insufficient pointwise channel mixing was
part of the bottleneck, but the exact unrounded S1 SSH ratio remains above one.
Consequently neither phase-1 candidate opens fresh validation. Both candidates passed
the exact three-step save/reload check, stayed finite for 180 days, and passed every
90/180-day amplitude check.

`phase_1_summary.json` is the lightweight project copy of the decision evidence.
Complete histories and checkpoints remain immutable under:

`/bigscratch/mjalabert314/bire_james25_repro/af_fno/models/C/successor_training_v1/phase_1/`

The reports are small but are not duplicated here; their exact paths and SHA-256
digests are recorded in the summary. The large checkpoints remain in scratch and are
addressed by SHA-256.

Fresh validation, inference, intermediate-wind, response, and adjoint data were not
read.
