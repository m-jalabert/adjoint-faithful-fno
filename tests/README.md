# Test layout

The production emulator is `F([x_t, S]) -> x_(t+10)`: 46 prognostic channels at
one time level plus five physical static channels in, 46 out, trained once from
random initialization. That run has no lineage. Exactly one arm does --- the
ninety-day fine-tune of its published checkpoint --- and its lineage is the
thing under test rather than an implementation detail.

- `test_train.py`: the 51/53/46 channel contract, the exact 27,297,960-parameter
  architecture, the zero-initialized local branch, the self-generated rollout,
  the Bire split and its 5,940 starts per regime, the from-scratch training
  contract (which is rejected if it names any parent), the frozen schedule, and
  each of the eight objective terms.
- `test_finetune.py`: the staged ninety-day fine-tune. That it names and pins
  exactly one parent, loads it strictly, reuses its normalization and starts
  Adam cold; that only the four declared fields differ from the parent; that the
  objective is the parent's weights over nine calls and every physics term
  reaches the ninth; that reinstalling the spectral cap on already-capped
  weights preserves the forward map; and the five acceptance conditions,
  including the two new ones --- short-horizon skill against the parent, and a
  90--360-day RMSE curve that flattens rather than steepens.

  Its load-bearing test is `test_every_source_the_parent_contract_pins_is_
  unchanged`: the fine-tune was built by adding a module, not by generalizing
  `train.py`, so the parent must still hash to its own frozen contract. If that
  test fails, the parent has stopped being reproducible.
- `test_finetune_evaluation.py`: the fine-tune's S0 figure and anomaly packages.
  That their evaluation protocol is byte-for-byte the parent's (same 15 members,
  same seed, same leads, same fields) so the two arms are comparable; that the
  anomaly package removes the *same* MITgcm training-mean field the parent's
  removed, since two models' anomalies are only comparable if the standing gyre
  subtracted from both is identical; that the lineage checks are real, rejecting
  a contract that claims `from_scratch`, drops the parent, or omits the pinned
  comparison baseline; and that both adapters reuse the production numerics by
  identity rather than reimplementing them.

  It carries the same load-bearing guard for `figures.py`, `anomaly.py` and
  `plots.py`. Their being untouched is what keeps the parent's published
  day-2,000 numbers a fixed baseline rather than a moving one.
- `test_figures.py`: the held S0 inference protocol, the 15 fixed members, the
  absence of any comparator series, and the 2,000-day acceptance-gate half.
- `test_anomaly.py`: sealed figure-package provenance, the MITgcm training-mean
  reference, and the anomaly RMS / structure diagnostics.

Everything above runs without a GPU and without the trajectory store: contracts
are read with `verify_sources=False`, and the objective tests use small
synthetic tensors whose answers are known by hand.

`test_fno_adjoint.py`, `test_mitgcm_adjoint.py` and `test_adjoint_phase_a.py`
belong to the separate adjoint study and are not part of the production
emulator's suite.

`test_adjoint_phase_a.py` covers the ninety-day window of
`docs/Adjoint_study_Phase_A.md`: that every declared run closes exactly on its
cost day (an off-by-one there means the two sides evaluate different
functionals); that the smooth kernel's stencil is wet, normalized and centred on
the frozen target, and that the isotropic alternative is *refused* rather than
silently renormalized onto a different place; that ninety days fits the existing
checkpoint tape, since if it does not the study needs a TAF submission and the
plan is wrong; and that `neuralop`'s spectral working buffer is still hard-coded
to complex64, so a version upgrade that fixes it shows up as a test failure
rather than quietly making the workaround dead code.
