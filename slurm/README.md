# Slurm layout

- `mitgcm/`: model build, segmented integrations, wind runs, and trajectory
  expansion.
- `data/`: dataset conversion, validation, and coverage jobs.
- `models/a/`, `models/b/`, `models/c/`: model-specific training and diagnostics.
- `evaluation/`: shared forward-evaluation jobs.
- Slurm root: scripts whose historical paths and bytes are fixed by immutable
  experiment contracts.

Submit from the repository root so `SLURM_SUBMIT_DIR` resolves consistently.
Before moving a root script, check every `config/*source_hashes*` entry.
