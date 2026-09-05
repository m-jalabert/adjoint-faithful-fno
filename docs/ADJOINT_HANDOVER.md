# Adjoint project handover

This guide covers the **from-scratch nominal-control and response v1 experiments**, their forward figures in `outputs/af_fno/C/`, and their three-seed adjoints in `outputs/af_fno/adjoint/`. The next experiment is to repeat both arms **without the local 3 × 3 convolution**, for every seed. Fine-tuning experiments are outside this guide.

## 1. Set up and reproduce the forward results

### Identify the runs

The study calls the matched nominal control **arm B** and the response-trained model **arm C**. Both are implementations of the production Model C architecture and both live under `outputs/af_fno/C/`; the directory letter and study-arm letters mean different things.

| Study arm | Training configuration in `config/` | Results under `outputs/af_fno/C/` |
| --- | --- | --- |
| B: nominal control | `model_c_adjoint_faithful_nominal_control_v1.json` | `model_c_adjoint_faithful_nominal_control_v1/seed_<seed>/` |
| C: response | `model_c_adjoint_faithful_response_v1.json` | `model_c_adjoint_faithful_response_v1/seed_<seed>/` |

Use all three paired seeds: **20260724, 20260911, 20260912**. Each run produces a `report.json`, numerical arrays, selection figure and provenance manifest. The actual `selected.pt` and `normalization.npz` are stored on scratch under `af_fno/models/C/<experiment>/seed_<seed>/`. A Git clone does not restore them; regenerate them by training if they have not been separately retained.

### Machine and Python environment

Run commands from the repository root. The dependencies declared in [pyproject.toml](../pyproject.toml) are:

- Python **3.11 or 3.12**; NumPy, SciPy, xarray, Dask, Zarr **2.x**, numcodecs, xmitgcm, Matplotlib, pandas and Jinja2.
- PyTorch **>=2.5,<2.7** and NeuralOperator **exactly 2.0.0** for the emulator. The code checks the NeuralOperator version.
- pytest, pytest-cov and Ruff for development checks.
- Git, Make, a Fortran compiler and MPI for MITgcm. The cluster forward-build scripts use GNU Fortran 14.2 and OpenMPI 5.0.7. GPU training scripts use CUDA 12.4; a compatible NVIDIA driver and PyTorch CUDA build are needed.
- **Licensed TAF and its `staf` command** to generate the MITgcm adjoint executable. PyTorch differentiation of the emulator does not require TAF.

For a new machine, install a PyTorch build appropriate to that machine, then install the project dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[fno,dev]'
export PYTHONPATH="$PWD/src:$PWD/archive/src:$PWD/scripts:${PYTHONPATH:-}"
export AF_PROJECT_ROOT="$PWD"
export AF_SCRATCH_ROOT="/path/to/your/scratch/af_fno"
```

Replace the scratch placeholder before running anything. `archive/src` is required for the MITgcm/data utilities, which are no longer under `src/`. The active neural-network package is `src/oceanfno/`.

[env/requirements-resolved.txt](../env/requirements-resolved.txt) records the resolved environment; [scripts/bootstrap_env.sh](../scripts/bootstrap_env.sh) is the UCSB setup recipe. Adapt its module commands on another machine. Slurm is needed only for the supplied batch wrappers; their Python commands can be run directly in a suitably allocated environment. GPU training is the practical route; the seed adjoint runner deliberately uses CPU double precision.

Fetch the upstream sources with `bash scripts/fetch_upstream.sh`. The expected commits are in [manifests/upstream.json](../manifests/upstream.json). The helper checks these commits, but clones the default branch of `oceanfourcast`; if that check fails, check out the manifest's exact revision in the fetched repository before proceeding.

### Recreate the MITgcm trajectories first

**The raw trajectories and restart pickups are on scratch and are not supplied by Git. They must be regenerated.** The nominal training input is **`trajectories_v3.zarr`**, not v1 or v2. It contains three wind regimes, 9,000 daily records per regime, 46 state channels and a 62 × 62 grid. Model years have 360 days.

Each wind regime must equilibrate independently for 100 model years, followed by 25 production years:

| Regime | Wind amplitude (N/m²) | Production records |
| --- | ---: | ---: |
| S0 | 0.100 | 9,000 |
| S1 | 0.075 | 9,000 |
| S2 | 0.125 | 9,000 |

The relevant code is in `af_fno/mitgcm/code/` and the input templates in `af_fno/mitgcm/`. The executable is built by [archive/slurm/mitgcm/af_s0_build.sbatch](../archive/slurm/mitgcm/af_s0_build.sbatch), using MITgcm's `genmake2`, `make depend` and `make`. On Slurm, after adapting modules/account/partition:

```bash
sbatch archive/slurm/mitgcm/af_s0_build.sbatch
```

Wait for the executable `build/af_s0/mitgcmuv` before submitting simulation segments. Follow these chains in order, with each segment starting from the preceding pickup:

| Chain | Files that run it |
| --- | --- |
| S0 spin-up: years 0–100, in ten-year segments; initial production: 100–110 | `archive/slurm/mitgcm/af_s0_segment.sbatch`, `archive/src/bire_repro/af_s0.py` |
| S0 continuation: 110–120 | `archive/slurm/mitgcm/af_trajectory_v2_expand.sbatch`, `archive/src/bire_repro/af_trajectory_expansion.py`, `archive/config/trajectories_v2_expansion.json` (S0 task only) |
| S0 long truth: 120–126 | `archive/slurm/mitgcm/af_model_c_bire_s0_long_truth.sbatch`, corresponding `af_model_c_bire_s0_long_truth.py` module and `archive/config/model_c_bire_s0_long_truth_v1.json` |
| S1 and S2: independent spin-up 0–100, production 100–110, 110–120, 120–125 | `archive/slurm/mitgcm/af_independent_wind_segment.sbatch`, `archive/src/bire_repro/af_independent_wind_trajectories.py` |

The two S0 continuation wrappers also retain pre-archive config defaults: set `AF_EXPANSION_CONTRACT` to the archived expansion JSON and `AF_CONTRACT` to the archived long-truth JSON, respectively, after relocating their contents.

The S0 wrapper accepts `AF_S0_PHASE`, `AF_S0_START_YEAR`, `AF_S0_YEARS`; the independent-wind wrapper accepts `AF_REGIME`, `AF_PHASE`, `AF_START_YEAR`, `AF_YEARS`. For example, the first S1 segment is submitted with:

```bash
sbatch --export=ALL,AF_REGIME=S1,AF_PHASE=spinup,AF_START_YEAR=0,AF_YEARS=10 \
  archive/slurm/mitgcm/af_independent_wind_segment.sbatch
```

Continue at years 10, 20, …, 90, then switch to production. Use scheduler `afterok` dependencies or wait for each predecessor to finish. Keep raw `.data`/`.meta` diagnostics, grid/input files and pickups: response generation and adjoint restarts need more than the reduced Zarr states.

The historical `scripts/submit_af_s0_chain.sh` and `scripts/submit_af_s1_s2.sh` are **not ready-to-run v3 launchers**: they reference moved wrappers, and the latter uses the older short wind-adjustment protocol. Use the archived wrappers above.

The v3 converter expects the historical directory layout, declared in `PRODUCTION_SEGMENTS` in [archive/src/bire_repro/af_data_v3.py](../archive/src/bire_repro/af_data_v3.py):

```text
mitgcm/S0/production/years_100_110
mitgcm_v2/S0/production/years_110_120
mitgcm_long_truth_v1/S0/production/years_120_126
mitgcm_independent_v1/S1/production/years_{100_110,110_120,120_125}
mitgcm_independent_v1/S2/production/years_{100_110,110_120,120_125}
```

These are relative to the scratch root. The converter takes only the first five years of the last S0 segment. Once all chains are complete:

```bash
python -m bire_repro.af_data_v3 build \
  --scratch-root "$AF_SCRATCH_ROOT" \
  --output "$AF_SCRATCH_ROOT/datasets/trajectories_v3.zarr"
python -m bire_repro.af_data_v3 validate \
  --dataset "$AF_SCRATCH_ROOT/datasets/trajectories_v3.zarr"
```

The store's original split metadata is historical. These production experiments define their own training days **0–5999**, validation starts **6000–6198 at stride 6**, and the fixed S0 figure ensemble in their contracts. Do not substitute the older v3 split for the production runner's split.

### Recreate the response training data

Arm C additionally requires **`datasets/forward_response_v1.zarr`**. This is generated with forward MITgcm perturbation experiments, not MITgcm adjoint labels. Keep the frozen direction inventory, amplitudes and train/validation roles from the supplied response contracts.

| Stage | Files and purpose |
| --- | --- |
| Restart bank | `scripts/build_response_pickup_bank.py` and `slurm/mitgcm/af_response_pickup_bank_segment.sbatch`: reproduce exact restart states at perturbation anchors |
| Perturbation definitions | `config/forward_response_dataset_v3.json`, `config/forward_response_schema_v1.json`, `scripts/build_forward_response_inventory.py`: fix regimes, dates, U/V/temperature/SSH directions, regions and levels |
| Amplitudes and numerical noise | `scripts/build_amplitude_pilot.py`, amplitude-pilot contracts/reports: reproduce nominal, signed and duplicate/tight-tolerance controls |
| Production runs | `scripts/stage_forward_response_run.py`, `scripts/submit_forward_response_run.py`, `slurm/mitgcm/af_forward_response_array.sbatch`: stage and execute nominal/plus/minus trajectories |
| Conversion and checks | `scripts/extract_forward_response_dataset.py`, `scripts/verify_forward_response_dataset.py`: package trajectories and verify the response store |

Start with `python scripts/build_response_pickup_bank.py --help` and `python scripts/build_forward_response_inventory.py --help` to inspect the stage-specific interfaces. After recreating the bank and pilot products, `python scripts/stage_forward_response_run.py list-work` enumerates production work; `python scripts/submit_forward_response_run.py --dry-run` previews submissions. The production helper relies on the pilot and inventory artifacts, so it is not a substitute for those preceding stages.

After the simulations finish:

```bash
python scripts/extract_forward_response_dataset.py \
  --dataset-path "$AF_SCRATCH_ROOT/datasets/forward_response_v1.zarr"
python scripts/verify_forward_response_dataset.py \
  --dataset-path "$AF_SCRATCH_ROOT/datasets/forward_response_v1.zarr"
```

Response utilities also contain absolute source paths/defaults. Adapt them before generation. The detailed inventory/pilot protocol is in [docs/Adjoint_faithful_response_training_plan.md](../docs/Adjoint_faithful_response_training_plan.md); the final training coefficient is already frozen at **0.001** in `config/forward_response_lambda_screen_v2.json`.

### Relocate the contracts, then train both arms

The supplied JSON files pin absolute `/home/mjalabert314/...` and `/bigscratch/mjalabert314/...` paths, source hashes, dataset hashes and reference artifacts. Setting `AF_SCRATCH_ROOT` alone does **not** rewrite these. Prepare a documented set of relocated contracts, including their referenced parent configuration, figure/anomaly contracts and source declarations. Keep the MITgcm `data`, `SST_relax.bin` and `DXF.data` source paths valid; they are used to construct physical static fields.

Preflight intentionally rejects inconsistent files. The parent configuration and deterministic-artifact references are needed for the equivalence checks, although the parent model weights are not loaded for B/C training. Regenerated data/checkpoints on a different compiler, MPI or GPU stack may differ numerically or in file hashes. Compare physical results and deterministic artifacts, record new provenance where necessary, and do not claim bitwise identity merely because a seed matches. Preserve the original contracts as the historical reference.

After relocation and data validation, run the common-runner equivalence check:

```bash
python scripts/verify_response_training_equivalence.py
```

The following is the direct execution sequence for all six runs, using the corresponding contracts at the paths shown:

```bash
for arm in nominal_control response; do
  contract="config/model_c_adjoint_faithful_${arm}_v1.json"
  for seed in 20260724 20260911 20260912; do
    python -m oceanfno.train_response preflight --contract "$contract" --seed "$seed"
    python -m oceanfno.train_response run --contract "$contract" --seed "$seed" --device cuda
  done
done
```

Alternatively, submit `slurm/models/c/train_adjoint_faithful_nominal_control_v1.sbatch` and `train_adjoint_faithful_response_v1.sbatch` once per seed with `--export=ALL,AF_SEED=<seed>`; `AF_CONTRACT` can select a relocated contract.

Both arms train for 7,680 optimizer steps, with effective batch 8, six autoregressive ten-day steps, and Adam. The learning rate drops from 0.0005 to 0.0001 after step 5,760. Both select checkpoints using nominal validation. Arm C adds a response update every fourth optimizer step; response metrics do not choose the checkpoint.

### Produce every seed's forward figures

Once training and selection have completed, run the six existing figure contracts and their anomaly companions:

```bash
for arm in nominal_control response; do
  for seed in 20260724 20260911 20260912; do
    stem="config/model_c_adjoint_faithful_${arm}_v1_seed_${seed}_s0"
    python -m oceanfno.figures_response finalize --contract "${stem}_figures_v1.json"
    python -m oceanfno.figures_response preflight --contract "${stem}_figures_v1.json"
    python -m oceanfno.figures_response run --contract "${stem}_figures_v1.json" --device cuda
    python -m oceanfno.anomaly_response finalize --contract "${stem}_anomaly_v1.json"
    python -m oceanfno.anomaly_response preflight --contract "${stem}_anomaly_v1.json"
    python -m oceanfno.anomaly_response run --contract "${stem}_anomaly_v1.json"
  done
done
```

`finalize` binds the contracts to the selected artifacts; use the relocated copies. The equivalent Slurm wrappers are `figures_adjoint_faithful_response_v1.sbatch` and `anomaly_adjoint_faithful_response_v1.sbatch`, with `AF_FIGURE_CONTRACT` and `AF_ANOMALY_CONTRACT` respectively.

Results go to `<experiment>_seed_<seed>_s0_figures_v1/S0/` and `<experiment>_seed_<seed>_s0_anomaly_v1/S0/` beneath `outputs/af_fno/C/`. They include streamfunction maps, surface-speed/SST/derived-pressure RMSE and ACC, short-range and 2,000-day rollout figures, and numerical arrays/CSV summaries. All use the same 15 S0 starts. Anomaly maps subtract the **MITgcm training mean from both truth and prediction**. The anomaly stage uses saved figure arrays and does not rerun the network.

Compare these numerical summaries with the original per-seed packages, including persistence and climatology curves. Retain every seed, the selected checkpoint identity, and the full lead-time curves. Existing output folders are protected against accidental overwrite; use new output roots for a new reproduction campaign.

## 2. Reproduce and understand the seed adjoints

### MITgcm reference

The seed comparison uses `outputs/af_fno/adjoint/mitgcm_s0_adjoint_v2/`. Its configuration is [config/adjoint_phase_a_v1.json](../config/adjoint_phase_a_v1.json); `config/mitgcm_adjoint_s0_target_v1.json` defines the original target. The source time is production day **7200**, with leads **10, 20, 30 and 90 days**.

Recreate the day-7200 S0 pickup from the forward chain. Build the adjoint with `slurm/mitgcm/af_s0_adjoint_build.sbatch`, which uses `af_fno/mitgcm/code_ad/` and licensed TAF. The forward restart stage `af_s0_adjoint_pickup.sbatch` supplies/checks the early reference window. Then use `af_s0_adjoint_phase_a.sbatch` with `AF_ADJ_RUN`:

1. Run **F90** first: it supplies the continuous forward truth and the late pickup needed by V10.
2. Run **P10/P20/P30/P90** for the point objective and **K10/K20/K30/K90** for the smooth-kernel objective.
3. Run **C90** for the basin-mean conservation probe and **V10** for the late-window consistency check.
4. Run the **G90** finite-difference checks using `af_s0_adjoint_phase_a_grdchk.sbatch`; inspect its point/epsilon array settings and require a resolved finite-difference plateau.

`scripts/stage_adjoint_run.py` translates each name into the correct restart, time window, cost weights and executable. Adapt its paths as well as the wrappers. After the runs and checks complete:

```bash
python scripts/extract_mitgcm_adjoint_phase_a.py
```

The extractor reads the raw MITgcm adjoint fields and builds the v2 reference arrays/report. Avoid `--allow-missing` for a final reference. Restart agreement, finite differences, land masking and conservation checks establish whether the numerical derivatives are trustworthy. [docs/mitgcm_adjoint_ground_truth_plan.md](../docs/mitgcm_adjoint_ground_truth_plan.md) describes the build and initial validation in more detail.

### All six emulator adjoints

For the original checkpoint identities:

```bash
for seed in 20260724 20260911 20260912; do
  python scripts/fno_adjoint_model.py --model "B_${seed}" --threads 8
  python scripts/fno_adjoint_model.py --model "C_${seed}" --threads 8
done
```

The Slurm equivalent is `slurm/models/c/fno_adjoint_model_v1.sbatch` with `AF_ADJOINT_MODEL=B_<seed>` or `C_<seed>`.

**Before using this on newly trained checkpoints**, update a versioned identity registry: `scripts/fno_adjoint_model.py::IDENTITIES` hard-codes the original checkpoint/normalization hashes, optimizer steps, reports and output directories. It will reject a different checkpoint. A new run must have its own verified identity and output location.

The resulting folders are `fno_b_seed_<seed>_s0_adjoint_v1/` and `fno_c_seed_<seed>_s0_adjoint_v1/`. Each contains a report, maps, finite-difference checks, backward and lead sweeps, conservation diagnostics and objective comparisons.

The registry delegates to `scripts/fno_adjoint_ft90.py` for shared derivative machinery. This dependency is necessary even though **these B/C runs are not fine-tuned**. The inherited array/figure filenames contain `fno_ft90_`; use the folder, registry and report checkpoint identity to identify the actual model.

The reported map is the sensitivity of a terminal SSH functional to initial SSH, `dJ/dη`. For the anomaly objectives, the basin-mean contribution is accounted for by the configured weights. Three objective types distinguish point/local behavior, a smooth spatial kernel, and global mean conservation.

- **Forced chain:** the emulator Jacobians are evaluated at MITgcm truth states. This is the primary comparison with the MITgcm adjoint.
- **Free chain:** the emulator Jacobians are evaluated along its own forecast. This represents deployed sensitivity and includes trajectory drift.

At ten days the two chains agree by construction. Their separation at later leads measures the trajectory effect. CPU float64 calculations also preserve complex128 Fourier buffers; retain that conversion in any successor implementation. Passing derivative checks establishes that the gradient is computed correctly, not that it agrees physically with MITgcm.

`scripts/compare_adjoint_models_response_v1.py` assembles the established multi-model comparison under `outputs/af_fno/adjoint/comparison_response_v1/`, using `scripts/adjoint_metrics.py`. Its existing model set also includes historical comparators; make a B/C-only version for the follow-up if those artifacts are unavailable. Report pattern agreement, amplitude and relative error, forced/free differences, and basin-mean conservation for **every seed**. Other posthoc subdirectories represent additional objective studies and are separate from these seed-v1 packages.

## 3. How the core code fits together

```text
MITgcm code + input templates → raw trajectories and pickups
  → trajectory Zarr + perturbed-response Zarr
  → dataset/normalization → model + objectives → train_response
  → nominal validation → selected.pt + normalization.npz
  → figures_response → anomaly_response
  → fno_adjoint_model → compare with validated MITgcm adjoints
```

The active emulator code is in `src/oceanfno/`:

| File(s) | Conceptual role |
| --- | --- |
| `runtime.py` | Runtime/version checks, seeds, hashes and common state conventions |
| `dataset.py` | Reads trajectory data, defines training records, computes training-only normalization and constructs the five physical static channels |
| `model.py` | Defines `ProductionArchitecture`, `ProductionFNO`, autoregressive unrolling and the physical-state evaluation adapter |
| `spectral_norm.py` | Caps each Fourier mode's channel-mixing matrix during training; normalized weights are materialized in saved checkpoints |
| `objective.py` | Combines state, increment, rollout, spectral and boundary losses with the physical losses below |
| `pressure_gradient.py`, `continuity.py`, `barotropic_transport.py` | Penalize pressure-gradient, free-surface continuity and depth-integrated transport errors without adding predicted channels |
| `train.py`, `train_response.py` | Base production protocol and the common paired-study runner; the latter handles both B and C through the response-enabled flag |
| `response_dataset.py`, `response_objective.py` | Deterministic perturbation sampling and signed perturbation-response mismatch, balanced across physical groups |
| `response_spectral_context.py` | Keeps auxiliary response evaluations from changing the spectral-normalization buffers relative to the matched control |
| `validation.py`, `perturbation_growth.py` | Nominal forecast checkpoint selection and perturbation-growth diagnostics |
| `response_validation.py`, `response_validation_blind.py` | Separate response-skill evaluation; these metrics do not select the nominal checkpoint |
| `diagnostics.py`, `plots.py` | Physical diagnostics and shared figure definitions |
| `figures.py`, `figures_response.py` | Shared forward evaluation and its per-arm/per-seed contract adapter |
| `anomaly.py`, `anomaly_response.py` | Training-mean streamfunction anomaly calculation and its study adapter |

One network call maps **one state to the state ten days later**. The 46 state channels are 15 U levels, 15 V levels, 15 temperature levels and SSH. Five static/forcing fields are appended: zonal wind stress, wet mask, Coriolis parameter, zonal grid spacing and SST relaxation target. Two deterministic position channels are added internally. Thus the external input has 51 channels, lifting sees 53, and output has 46. Pressure is derived from temperature and SSH.

The global network has three FNO blocks, width 128, 32 × 32 Fourier modes, channel MLPs, pointwise LayerNorm and 10% domain padding. In parallel, a bias-free **51→46, 3 × 3 convolution** adds a local correction. It begins at zero but learns during training. The network predicts the **next normalized state directly**, not a residual increment. Subsequent rollout calls consume its own predictions while holding static fields fixed and masking land.

Response supervision compares the signed change induced by the same perturbation in MITgcm and the emulator. It supplements the ordinary forward objective, with weight 0.001 and a mixture of short (10-day) and long (through 60-day) chains. B and C use the same nominal initialization, batch schedule, normalizer and checkpoint-selection procedure for each paired seed.

## 4. Immediate follow-up: remove the local 3 × 3 branch

Create new experiment versions and separate output roots for **both B and C**, retaining all three seeds. The intended comparison changes only the presence of the external local convolution; keep the trajectories, perturbation inventory, normalization procedure, global FNO, losses, schedule and evaluation starts fixed.

The current code does **not** expose a working “no local branch” switch. In `model.py`, `ProductionArchitecture.__post_init__` enforces a bias-free 3 × 3 branch, `ProductionFNO.__init__` creates `self.local`, and `forward` adds it to the global output. Introduce an explicit architecture option, omit the branch when disabled, and return the global output alone. Retain the FNO blocks' internal pointwise skip connections: they are a separate component.

Removing the external convolution removes `46 × 51 × 3 × 3 = 21,114` parameters, changing the expected total from **27,297,960 to 27,276,846**. Update architecture assertions, parameter-count checks, contract/source hashes and checkpoint loading accordingly. Training, figure and adjoint loaders must all construct the same new architecture. Merely setting `local_kernel_size=0`, or zeroing a previously trained branch at inference, does not implement the intended from-scratch experiment.

Recreate the data once, then:

1. Verify the new architecture, reload behavior and paired nominal training path. Freeze the successor contracts and their source hashes.
2. Train the two successor arms from scratch for seeds 20260724, 20260911 and 20260912.
3. Select checkpoints by the same nominal validation rule and reproduce each seed's forward and anomaly figure package.
4. Register all six new checkpoints in the adjoint identity adapter; repeat the derivative checks, forced/free maps and comparisons against the same validated MITgcm reference.
5. Compare original and no-local results seed by seed: forward RMSE/ACC, long-rollout and boundary anomaly structure, response skill, adjoint pattern/amplitude and conservation. Preserve all seeds rather than choosing a favorable one.

The handover deliverable for that experiment should retain the versioned configurations, reports, numerical figure arrays, plots and checkpoint/data provenance, with a documented external location for large scratch products. Those products remain necessary even when the code itself is in Git.
