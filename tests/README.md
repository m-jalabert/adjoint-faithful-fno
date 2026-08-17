# Test layout

The production emulator is `F([x_t, S]) -> x_(t+10)`: 46 prognostic channels at
one time level plus five physical static channels in, 46 out, trained once from
random initialization. There is no lineage, so there are no migration,
inheritance, or comparator tests.

- `test_train.py`: the 51/53/46 channel contract, the exact 27,297,960-parameter
  architecture, the zero-initialized local branch, the self-generated rollout,
  the Bire split and its 5,940 starts per regime, the from-scratch training
  contract (which is rejected if it names any parent), the frozen schedule, and
  each of the eight objective terms.
- `test_figures.py`: the held S0 inference protocol, the 15 fixed members, the
  absence of any comparator series, and the 2,000-day acceptance-gate half.
- `test_anomaly.py`: sealed figure-package provenance, the MITgcm training-mean
  reference, and the anomaly RMS / structure diagnostics.

Everything above runs without a GPU and without the trajectory store: contracts
are read with `verify_sources=False`, and the objective tests use small
synthetic tensors whose answers are known by hand.

`test_fno_adjoint.py` and `test_mitgcm_adjoint.py` belong to the separate
adjoint study and are not part of the production emulator's suite.
