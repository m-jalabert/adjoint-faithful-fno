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
- [Bire A0 reconstruction evidence](docs/bire_a0_reconstruction.md)
- [Historical Bire data schema](docs/bire_a0_data_schema.md)

The tracker is the authoritative record of completed jobs, validation evidence,
frozen decisions, and the next executable milestone.

## Current status

Status date: **21 July 2026**.

- The 1-degree MITgcm control, low-wind, and high-wind trajectories (S0--S2)
  are complete.
- S0 passed all seven numerical tutorial-validation gates.
- The next critical-path tasks are shared preprocessing and the frozen A0
  adapted-Bire baseline.
- The retired 0.25-degree MITgcm reconstruction campaign and its operational
  artifacts have been removed; only the recovered code evidence needed to
  define A0 is retained.

## Experimental ladder

| Model | Purpose |
| --- | --- |
| A0 | Adapted Bire architecture and training protocol; frozen historical baseline |
| A | Modern dense state-residual FNO baseline |
| B | Forward model with rollout, spectral, and boundary-aware losses |
| C | AF-FNO with perturbation-response supervision |

Models A0--C use the same MITgcm trajectories and frozen evaluation protocol.
MITgcm adjoints are generated only after the emulators and evaluation choices
are frozen.

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
- project documentation and the two project-facing PDFs.

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
