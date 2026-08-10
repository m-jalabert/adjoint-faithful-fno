# Test layout

- `test_train.py`: canonical two-in / one-out architecture, the zero-history
  input-channel checkpoint migration, the three-level contract inheritance, the
  pair-sliding autoregression, objective, split, selection, and
  training-contract checks.
- `test_figures.py`: held S0 inference, the one-input 32x32 comparator, the
  declared extra initial condition, acceptance-gate, and Figure 3--8
  publication checks.
- `test_anomaly.py`: sealed figure-package provenance and streamfunction-anomaly
  diagnostics.

The one-input 32x32 model appears only where a parent checkpoint or literal
comparison is required. There are no per-arm duplicate test modules.

Tests that bind to an artifact the current arm has not produced yet -- the
training report, the sealed figure package, the published anomaly package --
skip with a reason naming what is missing, so a clean run before training is
all skips rather than failures.
