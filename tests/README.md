# Test layout

- `test_train.py`: canonical Y32 architecture, centered checkpoint migration,
  objective, split, selection, and training-contract checks.
- `test_figures.py`: held S0 inference, local24 comparator, acceptance-gate,
  and Figure 3--8 publication checks.
- `test_anomaly.py`: sealed figure-package provenance and streamfunction-anomaly
  diagnostics.

The retained local24 model appears only where a parent checkpoint or literal
comparison is required. There are no per-arm duplicate test modules.
