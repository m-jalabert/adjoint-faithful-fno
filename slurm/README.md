# Slurm layout

- `mitgcm/`: model build, segmented integrations, wind runs, and trajectory
  expansion, plus the adjoint ground truth.

      af_s0_adjoint_{pickup,build,grdchk,run}.sbatch   the v1 10/20-day study
      af_s0_adjoint_phase_a.sbatch                     the 90-day Phase A window

  Phase A takes its run name in `AF_ADJ_RUN` and reads that run's window and
  weight field from `config/adjoint_phase_a_v1.json`, so the runs and
  `docs/Adjoint_study_Phase_A.md` cannot drift apart. Submit `F90` first: it is
  the only forward run, and it supplies both the day-7280 pickup `V10` needs
  and the daily snapshots gate G0 checks. It needs no TAF submission —
  `code_ad/tamc.h` is sized 72 x 200 x 1 = 14,400 >= 6,480 steps, so ninety days
  fits the existing tape and `mitgcmuv_ad` is reused byte-for-byte.

  Add `--exclusive` on the command line when a whole node is available; it is
  deliberately not a `#SBATCH` directive, because it cannot be turned off from
  the command line and this cluster is often fully allocated.
- `data/`: dataset conversion, validation, and coverage jobs.
- `models/a/`, `models/b/`, `models/c/`: model-specific training and diagnostics.
  `models/c/` holds the production emulator's launchers, each named for the arm
  it runs and submitted in order:

      train_..._spectralnorm_v1.sbatch        GPU, ~3.2 h  from scratch
      figures_..._spectralnorm_v1.sbatch      GPU, after training completes
      anomaly_..._spectralnorm_v1.sbatch      CPU, after the figures complete

  and the staged fine-tune of the checkpoint that run published, with its own
  three-stage sequence:

      train_..._spectralnorm_ft90_v1.sbatch     GPU, ~1.7 h  from the parent
      figures_..._spectralnorm_ft90_v1.sbatch   GPU, after the fine-tune
      anomaly_..._spectralnorm_ft90_v1.sbatch   CPU, after those figures

  The fine-tune reads the parent's `selected.pt`, normalization and report by
  digest and writes to its own roots, so the parent stays published and
  untouched. Its figure package evaluates the same 15 members on the same seed
  the parent used and pins the parent's sealed figure summary as an artifact, so
  the day-2,000 comparison is against published bytes rather than a remembered
  number; the anomaly package likewise removes the identical MITgcm
  training-mean field. Both run through `oceanfno.figures_ft90` and
  `oceanfno.anomaly_ft90`, which are lineage adapters over the production
  modules rather than copies of them.

  The figure and anomaly jobs each begin with a `finalize` step that stamps the
  preceding stage's digests into their contract, so neither can be run early
  against a half-written artifact.
- `evaluation/`: shared forward-evaluation jobs.
- Slurm root: scripts whose historical paths and bytes are fixed by immutable
  experiment contracts.

Submit from the repository root so `SLURM_SUBMIT_DIR` resolves consistently.
Before moving a root script, check every `config/*source_hashes*` entry.
