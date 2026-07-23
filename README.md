# Adjoint-Faithful Fourier Neural Operator for MITgcm

This project develops and evaluates an adjoint-faithful Fourier neural operator
(AF-FNO) for the MITgcm baroclinic double-gyre benchmark. It first reconstructs
an adapted version of the FNO used by Bire et al. (2025), then introduces modern
forward and perturbation-response objectives before comparing emulator
sensitivities with MITgcm adjoint ground truth.

The project is currently developed on the UCSB Pod HPC cluster. Source code,
configuration, experiment definitions, and lightweight provenance belong in
Git. Raw MITgcm trajectories, model checkpoints, and other large products stay
on HPC scratch storage.

## Project documents

- [Technical project plan](docs/AF_FNO_Project_Plan.pdf)
  ([TeX source](docs/AF_FNO_Project_Plan.tex))
- [Milestone and decision tracker](docs/Project_tracking.pdf)
  ([TeX source](docs/Project_tracking.tex))
- [Results and interpretation tracker](docs/Results_tracking.pdf)
  ([TeX source](docs/Results_tracking.tex))
- [Bire A0 reconstruction evidence](docs/bire_a0_reconstruction.md)
- [Historical Bire data schema](docs/bire_a0_data_schema.md)

The tracker is the authoritative record of completed jobs, validation evidence,
frozen decisions, and the next executable milestone.

## Current status

Status date: **23 July 2026**.

- The 1-degree MITgcm control, low-wind, and high-wind trajectories (S0--S2)
  are complete.
- S0 passed all seven numerical tutorial-validation gates.
- The shared preprocessing and frozen A0, Model A, and Model B baselines are
  complete. Model B materially improves Model A's 100-day rollout but still
  loses to persistence for temperature and SSH, so it remains a frozen
  forward-loss ablation.
- Forward-optimized Model C is active. Training-only jobs 284850/284857 found a
  147-fold range in channel increment scales, only about 278--398 effective
  increment samples, and stronger velocity-energy gains from extra meridional
  modes. The bounded search now includes `(24,16)`. Complex-safe CPU job 285116
  completed the training-only loss calibration in 4m39s. The frozen
  increment/rollout/spectral/boundary weights are
  `0.001/0.15/1e-5/0.065`. C1b job 285192 passed all gates except
  temperature/SSH increment skill, which remained 1.285/2.032 times
  persistence. Controlled-duration job 285265 changes only 160 to 320 epochs
  at learning rate `0.0005`.
- Model D will copy the selected Model C forward design and add local
  perturbation-response supervision. Intermediate-wind, response-inference,
  and adjoint data remain sealed until Model C passes the complete forward gate.
- Pressure is deliberately derived rather than learned. The common evaluator now
  reconstructs MITgcm `PHIHYD` at surface/mid/bottom levels 0/7/14 from all
  predicted temperature levels and SSH using the configured linear EOS. Against
  the archived S0 `PH` dump, the all-level maximum error is
  `3.89e-4 m2 s-2` and RMSE is `9.75e-6 m2 s-2`. Pressure-complete evaluation
  packages use non-overwriting contract `forward_complete_v2`; earlier v1
  packages remain pre-pressure evidence.
- The retired 0.25-degree MITgcm reconstruction campaign and its operational
  artifacts have been removed; only the recovered code evidence needed to
  define A0 is retained.

## Experimental ladder

| Model | Purpose |
| --- | --- |
| A0 | Adapted Bire architecture and training protocol; frozen historical baseline |
| A | Modern dense state-residual FNO baseline |
| B | Forward model with rollout, spectral, and boundary-aware losses |
| C | Validation-selected forward emulator with group-balanced physics losses |
| D | Selected Model C design plus perturbation-response supervision |

Models A0--D use the same MITgcm trajectories and frozen evaluation protocol.
MITgcm adjoints are generated only after the emulators and evaluation choices
are frozen.

## Active Model C workflow

Model C follows a staged, evidence-gated workflow based on the practical FNO
guidance of Duruisseaux et al. and the forward-evidence pattern of Bire et al.:

1. audit training-only autocorrelation, ten-day increment scales, and retained
   spectral energy for the declared 12/16/24-mode candidates;
2. freeze group-balanced state, increment, rollout, tapered-spectral, and
   boundary loss weights using training diagnostics;
3. require a stronger 96-sample memorization and exact-reload gate;
4. use bounded validation-only successive halving for modes, width, and only
   diagnostically justified padding/layer changes;
5. freeze three seeds and hashes before opening inference data or generating
   intermediate-wind and response experiments.

The C1a calibration is backend-independent science: CUDA is faster but not
required.
CPU job 284860 failed before output because a provisional gradient norm
overflowed float32. Job 284864 exposed and was cancelled for discarding the
imaginary part of complex spectral gradients. Diagnostic reductions now use
float64/complex128 without changing model precision; the large real/complex
regression test passes. Job 285116 completed successfully; its clipped
automatic proposal was rejected and the rounded unclipped gradient-balanced
weights are recorded in `config/model_c_loss_v1.json`. Pending GPU replica
284858 was cancelled without runtime.

C1b is retained as a scientific rejection: U/V increment errors beat
persistence at 0.254/0.442, while temperature/SSH did not. Because the better
attempt reached its best total at the final epoch, C1c tests undertraining
before changing loss weights or architecture. Validation and inference remain
sealed.

The current 7,530 overlapping ten-day training pairs are not treated as 7,530
independent samples. Dataset expansion requires learning-curve,
autocorrelation, and seed-spread evidence and, if approved, creates a versioned
`trajectories_v2` rather than overwriting version 1.

## Repository layout

| Path | Contents |
| --- | --- |
| `af_fno/mitgcm/` | Active 1-degree MITgcm code and input templates |
| `src/bire_repro/` | Recovered Bire baseline plus AF-FNO data, rollout, and analysis modules |
| `slurm/` | UCSB Slurm job templates |
| `scripts/` | Environment, upstream-fetch, and submission helpers |
| `config/` | Declared experiment configuration and A0 historical reference |
| `manifests/` | Pinned upstream versions and lightweight provenance |
| `tests/` | Unit and workflow tests |
| `docs/` | Project plan, tracker, and data/reconstruction documentation |
| `outputs/`, `work/` | Generated local products; ignored by Git |
| `external/` | Recreated upstream checkouts; ignored by Git |

The pressure reconstruction and validation entry point is
`python -m bire_repro.af_pressure`; its durable validation report is
`outputs/af_fno/pressure_validation_v1/pressure_validation.json`.

## UCSB setup

Fetch the pinned MITgcm and Bire repositories, then create the Python
environment:

```bash
./scripts/fetch_upstream.sh
./scripts/bootstrap_env.sh
.venv/bin/pytest -q
.venv/bin/ruff check .
```

The bootstrap script currently loads UCSB-specific compiler, CUDA, and Anaconda
modules. Upstream revisions are pinned in `manifests/upstream.json`; nested
upstream repositories are intentionally not committed.

Submit jobs from the repository root. For example:

```bash
sbatch slurm/af_s0_build.sbatch
./scripts/submit_af_s0_chain.sh
./scripts/submit_af_s1_s2.sh
```

Consult the tracker before submitting anything: completed jobs should not be
repeated unless a documented configuration change invalidates them.

## Data and provenance policy

Git tracks:

- source code and tests;
- Slurm templates and submission helpers;
- text configuration and small JSON manifests;
- project documentation and the three project-facing PDFs.

Git does not track:

- virtual environments or compiled MITgcm trees;
- scheduler output and language/tool caches;
- cloned upstream repositories;
- raw or reduced scientific arrays;
- checkpoints, trained weights, or generated rollouts;
- locally retained reference PDFs.

The canonical S0 catalog and validation figures currently live under
`outputs/af_fno/S0`, with raw MITgcm data linked to UCSB `/bigscratch`. These
products must be archived or transferred independently of Git.

## Moving to NASA HPC

Git will transfer the software and lightweight provenance, not the datasets.
Before the first NASA run:

1. clone the repository into a NASA project filesystem;
2. map the compiler, MPI, Python, CUDA, and scheduler modules available there;
3. rebuild the Python environment and required MITgcm executables locally;
4. replace UCSB accounts, partitions, and absolute paths in the configuration
   and Slurm templates;
5. transfer required raw/reduced datasets with the approved data-transfer
   service and verify checksums;
6. run unit tests, a short MITgcm restart test, and a small FNO smoke test before
   launching production jobs.

Do not copy `.venv`, `build`, or nested `external` checkouts between HPC
systems. Those contain platform-dependent artifacts and are reproducibly
rebuilt from the committed environment files and upstream manifest.

## Reproducibility conventions

- One scientific or infrastructure change per commit when practical.
- Record every completed campaign or frozen decision in
  `docs/Project_tracking.tex`.
- Never commit credentials, allocation identifiers that must remain private, or
  large scientific products.
- Use feature branches for model or workflow changes and merge them only after
  tests and smoke checks pass.
- Tag frozen experiment stages such as `s0-validated`, `a0-baseline`, and
  `adjoint-ground-truth-v1`.

## Citation

The historical baseline is based on Bire et al., *Ocean Emulation With Fourier
Neural Operators: Double Gyre*, Journal of Advances in Modeling Earth Systems,
DOI: [10.1029/2023MS004137](https://doi.org/10.1029/2023MS004137).

## License

The project code is released under the [MIT License](LICENSE). MITgcm,
NeuralOperator, and the recovered Bire code retain their respective licenses.
