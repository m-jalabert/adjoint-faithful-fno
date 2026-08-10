# Slurm layout

- `mitgcm/`: model build, segmented integrations, wind runs, and trajectory
  expansion.
- `data/`: dataset conversion, validation, and coverage jobs.
- `models/a/`, `models/b/`, `models/c/`: model-specific training and diagnostics.
  In `models/c/`, the arm-suffixed scripts (`*_2in_1out.sbatch`) are the current
  canonical launchers. The unsuffixed `train`/`figures`/`anomaly.sbatch` are the
  32x32 one-input arm's and name that arm's contracts, which the canonical
  modules no longer accept; they are kept as the record of how that arm ran.
- `evaluation/`: shared forward-evaluation jobs.
- Slurm root: scripts whose historical paths and bytes are fixed by immutable
  experiment contracts.

Submit from the repository root so `SLURM_SUBMIT_DIR` resolves consistently.
Before moving a root script, check every `config/*source_hashes*` entry.
