# Slurm layout

- `mitgcm/`: model build, segmented integrations, wind runs, and trajectory
  expansion.
- `data/`: dataset conversion, validation, and coverage jobs.
- `models/a/`, `models/b/`, `models/c/`: model-specific training and diagnostics.
  `models/c/` holds the production emulator's three launchers, to be submitted in
  order:

      train_production_1in_1out.sbatch      GPU, ~36 h
      figures_production_1in_1out.sbatch    GPU, after training completes
      anomaly_production_1in_1out.sbatch    CPU, after the figures complete

  The figure and anomaly jobs each begin with a `finalize` step that stamps the
  preceding stage's digests into their contract, so neither can be run early
  against a half-written artifact.
- `evaluation/`: shared forward-evaluation jobs.
- Slurm root: scripts whose historical paths and bytes are fixed by immutable
  experiment contracts.

Submit from the repository root so `SLURM_SUBMIT_DIR` resolves consistently.
Before moving a root script, check every `config/*source_hashes*` entry.
