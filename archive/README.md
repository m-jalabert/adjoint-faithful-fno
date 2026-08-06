# Archive — the superseded tree

`src/oceanfno/` is the live code. Everything here preceded it and is kept only
so the published results remain verifiable. Nothing was deleted; the structure
mirrors the repository, so any file can be restored with a single `git mv`.

| directory | contents |
| --- | --- |
| `archive/src/bire_repro/` | the previous package, 78 modules and ~33,000 lines |
| `archive/config/` | frozen contracts of every arm, including the v1 contracts of this one |
| `archive/outputs/af_fno/C/` | published result packages of superseded arms |
| `archive/slurm/`, `archive/tests/` | their launchers and tests |

## Why the v1 contracts are here rather than in `config/`

The published checkpoint `selected.pt` carries `contract_sha256` of
`model_c_bire_protocol_rollout_ft_v1.json`, and that contract hashes the module
paths of `src/bire_repro`. Both live here, so that chain still verifies from
this directory exactly as it did when the run completed.

`config/*_v2.json` are the live contracts. They hash `src/oceanfno` instead, and
each carries a `supersedes` block naming the v1 contract, its SHA-256, and the
fact that the published artifacts were produced under v1. The scientific arm did
not change — the code was consolidated and verified numerically identical:

* the FNO forward pass on the published weights,
* the six-step unroll,
* every term of the objective and its contract hash `0cc75764…`,
* the split arithmetic, selection rule, and derived diagnostics,
* a bit-identical rebuild of the whole streamfunction-anomaly package.

## What is still referenced from outside

Three things in this directory are read by the live tree, and moving them again
will break contract verification:

1. `archive/config/model_c_bire_protocol_duration_v1.json` — the v2 training
   contract hashes it and reads it to assert only the declared quantities moved.
2. `archive/config/model_c_bire_protocol_duration_s0_figures_v1.json` — the
   package the figure suite is asserted comparable with.
3. `outputs/af_fno/C/bire_protocol_duration_v1/` (not archived) — supplies the
   46 per-channel increment scales and the baseline validation summary the
   acceptance gate compares against.
