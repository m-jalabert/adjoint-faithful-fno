# Test layout

- `test_train.py`: canonical 32x32 architecture, the zonal zero-extension
  checkpoint migration, the two-level contract inheritance, objective, split,
  selection, and training-contract checks.
- `test_figures.py`: held S0 inference, Y32 comparator, acceptance-gate,
  and Figure 3--8 publication checks.
- `test_anomaly.py`: sealed figure-package provenance and streamfunction-anomaly
  diagnostics.

The retained Y32 model appears only where a parent checkpoint or literal
comparison is required. There are no per-arm duplicate test modules.

Tests that bind to an artifact the current arm has not produced yet -- the
training report, the sealed figure package, the published anomaly package --
skip with a reason naming what is missing, so a clean run before training is
all skips rather than failures.
