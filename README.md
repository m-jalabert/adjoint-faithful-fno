# Adjoint-Faithful Fourier Neural Operator for MITgcm

This project tests whether an ocean emulator trained on forward trajectories can also reproduce the ocean model's sensitivities. It compares a Fourier neural operator (FNO) trained with a forecast objective against an otherwise matched model with additional forward perturbation-response supervision. MITgcm/TAF adjoints provide an independent evaluation reference; no adjoint quantity enters training or checkpoint selection.

**Current finding:** response supervision reduces sensitivity error, but does not recover the spatial structure of the MITgcm adjoint. Forward forecast skill and correct reverse-mode differentiation are not sufficient evidence of physically faithful sensitivities.

## Start here

- [Project handover and reproduction guide](docs/ADJOINT_HANDOVER.md): environment, regeneration of MITgcm trajectories, all-seed training, forward figures, adjoints and the next architecture experiment.
- [Current progress report (PDF)](docs/progress_summary_control_response_seed20260911/progress_summary.pdf) and [LaTeX source](docs/progress_summary_control_response_seed20260911/progress_summary.tex): control/response comparison, including three spatial targets and two SSH objectives.
- [Response-study protocol](docs/Adjoint_faithful_response_training_plan.md): matched experiments, perturbation inventory, evaluation and confirmatory criteria.
- [MITgcm adjoint build and validation](docs/mitgcm_adjoint_ground_truth_plan.md).
- [Turbulent double-gyre study](turbulent_double_gyre/README.md): the separate 0.25-degree extension and its execution instructions.

The older project plans and trackers under `docs/` retain historical decisions. This README summarizes the current progress report rather than the earlier Model A–D development ladder.

## Experiments and results

The main sensitivity study uses the 1-degree MITgcm double gyre on a 62 × 62 grid. Its training trajectories cover independently equilibrated wind regimes S0, S1 and S2; the reported forward and adjoint comparisons use S0, a weakly turbulent regime.

| Study arm | Experiment | Training difference |
| --- | --- | --- |
| B: nominal control | `model_c_adjoint_faithful_nominal_control_v1` | Eight-term forward objective |
| C: response | `model_c_adjoint_faithful_response_v1` | Same objective plus signed forward-response mismatch, weighted by 0.001 |

Both arms train from random initialization with paired seeds **20260724, 20260911 and 20260912**. They share architecture, nominal data, normalization, batch schedule, optimizer and nominal-validation checkpoint selection. Both live under `outputs/af_fno/C/`: that directory denotes the production model family, whereas B/C denote arms of this study.

The current report illustrates **seed 20260911**, selected for its strong control forecast. These figures are descriptive; they do not replace the three-seed evaluation or change the predeclared primary seed, 20260724.

### What the report shows

- **Forward behavior:** for seed 20260911, the two arms evolve similarly until roughly day 700, then separate. At day 2,000 the response model exceeds climatology error in all three primary fields (surface speed, SST and surface pressure), while the control model remains below it. The sign of the forward difference changes across seeds, so this is not evidence of a uniform forward penalty from response supervision.
- **Adjoint error:** across the report's 18 combinations of target, objective and lead, the response arm lowers relative L2 sensitivity error in every cell, by up to 32%. These compare interior, eastern and western targets, raw-SSH and SSH-anomaly objectives, and leads of 10, 30 and 90 days.
- **Adjoint structure:** pattern correlations remain near zero, approximately −0.15 to +0.18 across the two arms. Reduced error largely reflects changed sensitivity magnitude rather than recovery of MITgcm's basin-wide spatial structure.
- **Mean conservation:** MITgcm conserves basin-mean sea level in this closed basin. Both emulators instead damp its sensitivity by approximately 0.38 per ten-day call.
- **Confirmatory verdict: negative.** The earlier blind comparison improved relative L2 in 24 of 24 confirmatory cells, with a median response/control ratio of 0.745. However, the primary-seed aggregate ratio was 0.837, missing the required ≤0.8 threshold. Five of six criteria passed; the favorable descriptive seed does not reverse the verdict.

The [report](docs/progress_summary_control_response_seed20260911/progress_summary.tex) gives the metrics, figures, limitations and numerical derivative checks. In particular, strong seed dependence in long rollouts means that a single favorable pair cannot establish a robust improvement.

## Model and code

One call advances a single ocean state by ten days. The 46 predicted channels are 15 vertical levels each of U, V and potential temperature, plus SSH. Five static/forcing fields accompany the state: zonal wind stress, wet mask, Coriolis parameter, zonal grid spacing and SST relaxation target. Two deterministic position channels are added inside the network. Pressure is derived from predicted temperature and SSH.

The 1-degree model uses three FNO blocks, width 128, 32 × 32 Fourier modes, channel MLPs, pointwise LayerNorm, 10% domain padding and an external bias-free 3 × 3 local convolution: 27,297,960 parameters. It predicts the next normalized state directly and feeds its own predictions into subsequent calls. Per-mode spectral normalization constrains the Fourier operators during training; its weights are materialized in published checkpoints.

| Path | Role |
| --- | --- |
| `src/oceanfno/model.py` | Architecture, autoregressive rollout and physical-state evaluation adapter |
| `src/oceanfno/dataset.py` | Trajectory access, training records, static fields and training-only normalization |
| `src/oceanfno/objective.py` | Forecast losses, combined with pressure-gradient, continuity and barotropic-transport modules |
| `src/oceanfno/train_response.py` | Shared from-scratch runner for both study arms |
| `src/oceanfno/response_*.py` | Response sampling, loss, spectral-buffer handling and response evaluation |
| `src/oceanfno/validation.py` | Nominal forecast evaluation and checkpoint selection |
| `src/oceanfno/figures_response.py`, `anomaly_response.py` | Per-seed forward figures and training-mean anomaly companions |
| `scripts/fno_adjoint_model.py` | Frozen seed/checkpoint registry and shared emulator derivative pipeline |
| `scripts/stage_adjoint_run.py`, `extract_mitgcm_adjoint_phase_a.py` | MITgcm adjoint run staging and reference extraction |
| `scripts/*posthoc_v1.py`, `scripts/*raw_ssh_v1.py` | Additional target/objective evaluations used by the current report |
| `af_fno/mitgcm/` | MITgcm forward/adjoint code modifications and input templates |
| `archive/src/bire_repro/`, `archive/slurm/` | Earlier infrastructure, including the trajectory-generation and v3 conversion utilities still needed for reproduction |
| `config/`, `slurm/models/c/` | Experiment contracts and cluster execution wrappers |
| `turbulent_double_gyre/` | Separate 0.25-degree experiment; its results are not the 1-degree adjoint-study results above |

Some shared adjoint utilities and output filenames retain an `ft90` label. The B/C seed registry uses that derivative implementation without fine-tuning the B/C models. Check the report and checkpoint identity to identify a run.

## Running the project

Use Python **3.11 or 3.12**. [pyproject.toml](pyproject.toml) declares NumPy, SciPy, xarray, Dask, Zarr 2.x, numcodecs, xmitgcm, Matplotlib, pandas and Jinja2. Emulator execution additionally requires PyTorch **>=2.5,<2.7** and NeuralOperator **2.0.0**; the latter is checked at runtime.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[fno,dev]'
export PYTHONPATH="$PWD/src:$PWD/archive/src:$PWD/scripts:${PYTHONPATH:-}"
```

Install a PyTorch build appropriate to the machine's CPU/CUDA environment. GPU training is the practical production route; the seed-adjoint runner uses CPU float64/complex128 for precision-sensitive derivative checks. The resolved environment is recorded in [env/requirements-resolved.txt](env/requirements-resolved.txt). [scripts/bootstrap_env.sh](scripts/bootstrap_env.sh) contains UCSB-specific modules and must be adapted elsewhere.

MITgcm also requires Make, a Fortran compiler and MPI. Fetch the upstream repositories using [scripts/fetch_upstream.sh](scripts/fetch_upstream.sh) and verify the revisions in [manifests/upstream.json](manifests/upstream.json). Generating the MITgcm adjoint executable requires licensed TAF (`staf`); differentiating the PyTorch emulator does not. Slurm is required for the supplied cluster wrappers, not for the Python modules themselves.

**A Git clone does not contain the raw trajectories, restart pickups, Zarr datasets or trained checkpoints.** Regenerate the MITgcm trajectories on the destination machine, then build `trajectories_v3.zarr` and the forward perturbation-response dataset before training. The handover gives the simulation chains, including independent 100-year equilibration and 25 production years per regime.

Contracts contain absolute project/scratch paths and pinned artifact hashes. Relocate them and record new provenance before running preflight. The older top-level submission helpers reference moved files and are not complete v3 launchers; use the archived entry points documented in the handover.

After data regeneration and contract preparation, the execution order is:

1. Run the paired-training equivalence checks.
2. Run `oceanfno.train_response preflight` and `run` for both arms and all three seeds.
3. Finalize and run each seed's `oceanfno.figures_response` package, then its `oceanfno.anomaly_response` companion.
4. Generate and validate the MITgcm reference adjoints; run `scripts/fno_adjoint_model.py` for each frozen B/C seed identity.
5. Compare sensitivity structure, amplitude, forced/free trajectories and mean conservation alongside the forward metrics.

The [handover](docs/ADJOINT_HANDOVER.md) contains the commands and file relationships. Newly trained checkpoints need new verified identities in the adjoint registry; the existing registry pins the original hashes.

## Results and storage

- `outputs/af_fno/C/model_c_adjoint_faithful_{nominal_control,response}_v1/`: per-seed training results, with separate sibling forward/anomaly figure packages.
- `outputs/af_fno/adjoint/fno_{b,c}_seed_<seed>_s0_adjoint_v1/`: original seed adjoints; `mitgcm_s0_adjoint_v2/` contains their reference package.
- `outputs/af_fno/adjoint/comparison_response_v1/`, `comparison_posthoc_v1/` and related target-specific folders: comparison products.
- `docs/progress_summary_control_response_seed20260911/`: the current report and its figure assets.
- `archive/outputs/af_fno/turbulent_gyre/`: earlier turbulent-run evidence retained separately from the current turbulent experiment.

Git retains code, configurations, documentation and selected figure/provenance products. Large arrays, trajectories and weights remain on scratch and require separate storage or regeneration. Do not copy `.venv`, compiled `build/` trees or upstream checkouts between machines; rebuild them locally.

## Next experiment

Retrain **both nominal-control and response arms without the external local 3 × 3 branch**, using all three paired seeds and otherwise matched data, losses, schedule, budget and evaluation starts. First separating the local and Fourier contributions to a trained model's ten-day sensitivity can help test the proposed explanation; it is not a substitute for retraining.

The current architecture enforces the local branch, so this requires a versioned code/configuration change and matching training, figure and adjoint loaders. Retain the FNO blocks' internal pointwise skip connections. Evaluate every seed for forward accuracy, long-rollout stability, response skill, adjoint spatial/sign/amplitude agreement and mean conservation. The report motivates this experiment but does not establish that removing the branch will solve the sensitivity problem.

## Attribution and license

The ocean-emulation backbone follows Bire et al., *Ocean Emulation With Fourier Neural Operators: Double Gyre* (2025), DOI: 10.1029/2023MS004137. The current report contains the complete references.

Project code is released under the [MIT License](LICENSE). MITgcm, NeuralOperator and the recovered Bire code retain their respective licenses.
