# Test layout

- `core/`: reusable package utilities.
- `data/`: MITgcm, trajectory, and dataset workflows.
- `models/`: A0, Model A, Model B, and forward-comparison tests.
- `model_c/`: Model C diagnostics, evaluations, and plot-generation tests.
- test root: checks whose exact paths and bytes are fixed by immutable contracts.

Pytest discovers all subdirectories through the repository-level `testpaths`
setting.
