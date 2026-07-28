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

Status date: **27 July 2026**.

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
  `0.001/0.15/1e-5/0.065`. C1b job 285192, 320-epoch C1c job 285265,
  and width-64 GPU job 285325 all failed only the complete groupwise increment
  gate; simply training longer or increasing capacity did not solve SSH.
  GPU audit array 287581 checked every saved evaluation and the exact late
  checkpoint gradients. CPU fallback scaling job 287556 measured 436/272/253
  seconds at 4/8/16 cores, so 16 cores is the fallback when GPU access would
  materially delay work; backend replication remains secondary. Bounded array
  287583 then isolated two changes:
  increasing the increment weight to `0.0025` was rejected, while retaining
  loss v1 and reducing the learning rate from `0.0005` to `0.0001` after epoch
  240 passed the complete 96-record gate at epoch 315. Its U/V/temperature/SSH
  increment ratios to persistence were `0.211/0.360/0.794/0.990`, with every
  other criterion and exact three-step reload also passing. The frozen
  validation search then completed in GPU jobs 287604, 288466, 289379, and
  289915. Successive halving selected the four-layer, width-64 `(24,16)` design
  without triggering the predeclared padding or depth branches. Its search-seed
  U/V/temperature/SSH validation ratios were
  `0.1473/0.3547/0.6597/1.00014`. Final-seed array 290382 completed cleanly,
  but all three seeds narrowly failed only SSH at
  `1.00014/1.00383/1.03818` times persistence. The immutable three-seed
  decision therefore scientifically rejects validation-search v1;
  configuration freezing and inference are not authorized.
- Post-search data-adequacy job 290415 then evaluated the same three immutable
  checkpoints by wind regime without reading inference. Its predeclared gate
  passed all four checks and authorizes a non-destructive `trajectories_v2`
  expansion. Slow-state effective sample counts are only 7.59 for temperature
  and 7.69 for SSH across all three regimes, and the chronology curve improves
  strictly with added data. The audit also found that low-wind S1 SSH is already
  worse than persistence on training (`1.32--1.36`) and validation
  (`1.50--1.55`), whereas S0/S2 are below persistence. Consequently, doubling
  coverage is justified but is not treated as a complete remedy: the successor
  must also test regime balance, channel mixing, and rollout stabilization.
  The immutable expansion contract added ten exact production years to each of
  S0--S2 in array job 290439. All three continuations ended normally in
  15m10s--15m31s with 3,600 daily `dynState`/`surfState` records and final
  pickups. Dataset job 290443 then built the 11 GB
  `trajectories_v2.zarr` store with shape `3x7200x46x62x62`; independent
  quality job 290444 verified finite/zero-land values, exact daily chronology,
  a bitwise-identical v1 prefix, bitwise raw-MDS extension samples, and
  independently reproduced train-only normalizers. The split has 15,060
  training, 780 fresh validation, and 3,450 sealed inference pairs.
- Training-only coverage job 290446 used a contract frozen after dataset
  quality control and before coverage metrics. Temperature/SSH state-proxy
  effective coverage increased by `2.142x/2.039x`, and their ten-day increment
  proxies by `2.106x/2.253x`, so the predeclared two-times slow-field target
  passed. Fresh v2 validation remains unread. Successor contract
  `config/model_c_successor_training_v1.json` (SHA-256
  `3e0a300e73ec4044705e29190473d0aee7308fb6ea7d12485346d1fe14159285`)
  now fixes a training-only sequence: exact width-64 data control, isolated
  width-64 `4C` Channel-MLP ablation, then a width-128/256-lift/256-projection/
  `4C` candidate only after phase 1. Both tasks in GPU array 290448 completed
  normally and passed reload and 180-day stability checks, but each missed only
  low-wind S1 SSH. On all 5,020 S1 training pairs, the data-only control scored
  `1.103435x` persistence and isolated `4C` channel mixing improved it to
  `1.003573x`; the latter also improved every aggregate group. This supports
  channel mixing as part of the bottleneck but does not pass the exact
  predeclared gate. Phase-2 width-128 job 290597 then completed in 1h20m22s and
  passed the entire gate. Its aggregate U/V/temperature/SSH ratios are
  `0.049040/0.121279/0.384816/0.522413`, every regime/group passes, and S1 SSH
  is now `0.762683x` persistence. Exact reload and all 180-day stability checks
  also pass. Durable phase summaries are saved under
  `outputs/af_fno/C/successor_training_v1/`.
  Corrected fresh-validation contract
  `config/model_c_successor_validation_v2.json` (SHA-256
  `56bb6ec597993b0689ab699a452ed189392f98806bf95b7add79062b070cf27b`)
  is frozen before any validation state metric. It supersedes an immutable
  off-by-one v1 metadata contract: the fresh block has 180 complete 90-day
  starts per regime, not 181. GPU array 290673 completed both missing replicas
  normally. All three seeds pass the full training gate; their worst
  regime/group ratios are `0.762683/0.732270/0.839507`, all from low-wind S1
  SSH. Exact reload and all 180-day stability checks pass for every seed.
  Lightweight gate evidence is saved under
  `outputs/af_fno/C/successor_validation_v1/training_gate/`.
  The all-group SSH requirement in this stage is a training-capacity diagnostic,
  not the final forward gate. Before fresh v2 validation opens, the latter is
  fixed to Bire's surface-speed/SST/surface-pressure 10--90-day RMSE/ACC curve
  evidence, with streamfunction used for circulation stability and wind
  response. SSH remains fully reported with uncertainty and stability checks
  but is no longer an independent one-number veto.
  Fresh-validation job 290738 completed normally but scientifically rejected
  the successor. Surface speed passes every baseline comparison, while the
  three-seed bootstrap-mean SST and surface-pressure RMSE-AUC ratios to
  persistence are `2.313` and `2.198`. Both are good at day 10 and accumulate
  error after days 20--30; this is not an SSH-only failure. All rollouts remain
  finite and circulation diagnostics pass. Inference remains sealed.
  Five decision-facing validation figures and a hash manifest are saved under
  `outputs/af_fno/C/successor_validation_v1/fresh_validation/`; they were
  generated from the immutable member-level validation arrays without opening
  any later archive.
  Training-only rollout-diagnosis contract
  `config/model_c_rollout_diagnosis_v1.json` (SHA-256
  `d0ecee0db2b233409af2cbd55b34cbe133b647e28bd0c17381956040595ff7ba`)
  was executed unchanged by GPU job 291102, which completed in 3m58s with exit
  zero. Every seed reproduces the slow-field drift on fixed training
  chronology. Mean day-10/AUC/day-90 ratios to persistence are
  `0.809/2.021/4.221` for SST and `0.552/1.867/3.873` for surface pressure.
  The predeclared classification is therefore
  `training_objective_or_checkpoint_gate_mismatch`, not a validation-only
  generalization gap. Three diagnosis figures, a lightweight summary, and a
  hash manifest are saved in `outputs/af_fno/C/rollout_diagnosis_v1/`.
  Model C is still temporally one-input/one-output: a call consumes one
  46-channel ocean state plus five forcing/static channels and predicts one
  46-channel ten-day residual. Its three-step training unroll does not make it
  Samudra's two-time-level/two-future-time-level formulation. That formulation
  is now a scientifically motivated but unselected bounded ablation because two
  consecutive inputs expose an explicit tendency and two direct future targets
  may reduce recursive error accumulation. Because job 291102 shows the error
  already in training, the first priority is a bounded long-rollout
  loss/checkpoint audit. The larger temporal interface should then be compared
  prospectively with the simpler one-input/one-output correction, not assumed
  to be a cure.
  A descriptive Bire-style figure contract is frozen in
  `config/model_c_bire_figures_v1.json` (SHA-256
  `791c84f12c2d8dc5b314ca3e24c56b8fcf8b817c2624f26ac0b204fc0feb696b`)
  before reading any 100--200-day validation value. It fixes checkpoint seed
  20260723, 15 prospectively sampled S2 fresh-validation starts, and one
  non-cherry-picked member. Figure 3 will compare MITgcm, Model C, and their
  difference for the 1-degree barotropic streamfunction at days 0--40, without
  a coarse-grid row. Figure 4 will reproduce the paper's prediction,
  climatology, and persistence mean and 10th--90th percentile curves at
  ten-day spacing through day 200 with the requested axes. Persistence repeats
  the initial condition. Climatology is the regime-specific pointwise mean of
  all 5,040 split-1 training snapshots, matching the paper's mathematical
  definition while preventing evaluation leakage. This characterizes an
  already rejected model and cannot authorize tuning or inference. GPU job
  291134 completed normally in 51 seconds.  For the fixed S2 member, the
  streamfunction RMSE is 0.0316, 0.0934, 0.1580, and 0.1765 Sv at days
  10/20/30/40, against a truth RMS of about 13.34 Sv.  The 15-member curves
  remain finite through day 200 but expose severe recursive drift: day-200
  Model-C/persistence RMSE is 0.07344/0.00480 m/s for surface speed,
  5.981/0.0505 degrees C for SST, and 3.747/0.0253 m2/s2 for surface pressure.
  The requested fixed SST and pressure axes are consequently exceeded from
  days 100 and 160; the speed curve remains within its requested axis.  The
  immutable report/arrays SHA-256 values are
  `b704b921444d...e300ac5`/`a73ac9bf2cc9...20b1`, and the two figures,
  lightweight summary, and hash manifest are under
  `outputs/af_fno/C/bire_figures_v1/` (manifest-content SHA-256
  `258be6fabe13...827f`).  Inference remains sealed.
  A separate immutable per-lead visualization contract is now frozen in
  `config/model_c_bire_streamfunction_leads_v1.json` (SHA-256
  `ccb965e58bdc053d286b77b86fa8fa347bd2d06cfbe430f6cbdc433c5ae1934c`).
  It fixes exactly eight new native-1-degree figures at days 20, 30, ..., 90
  for the same S2 start 6335 and checkpoint.  Each figure shows MITgcm,
  Model C, and truth-minus-prediction.  Truth and prediction share one common
  scale across all leads, while each difference panel uses its own labeled
  symmetric maximum-error scale and reports RMSE/max error so the small
  structure is visible without implying that colors are comparable between
  leads.  GPU job 291135 completed normally in 35 seconds.  Streamfunction
  RMSE grows from 0.0934 Sv (0.700% of truth RMS) at day 20 to 0.2951 Sv
  (2.211%) at day 90; the maximum pointwise error grows from 0.7240 to
  1.1713 Sv.  The circulation pattern therefore remains visually strong
  through day 90 even while the previously measured SST and pressure errors
  drift badly.  Day 10 remains in the original Figure-3 analogue.  The
  immutable report/arrays SHA-256 values are
  `71c6e1dc1f9a...89c206`/`b4126494d9c0...3bcd48`; all eight PNGs, a numerical
  summary, and the manifest are saved under
  `outputs/af_fno/C/bire_streamfunction_leads_v1/` (manifest-content SHA-256
  `3b74d759d551...450ac7`).  This characterization reads only the already
  opened S2 validation member and cannot tune the model or open inference.
- Model D remains blocked. It will copy only a Model C design that passes the
  complete forward gate and then add local perturbation-response supervision.
  Intermediate-wind, response-inference, and adjoint data remain sealed.
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

C1b, C1c, and the width-64 capacity run are retained as scientific rejections,
not operational failures. Across both 320-epoch histories, none of 64 evaluated
epochs passed every increment group, and width 64 did not improve the best SSH
ratio. The late-checkpoint audit found that the weighted increment gradient was
already about 0.41 times the state gradient and was dominated by an aligned SSH
component; the volatile late SSH history pointed to optimization stabilization.
The accepted training-only schedule keeps loss v1 and applies one fivefold
learning-rate decay after epoch 240. A separately versioned 2.5-times
increment-weight loss v2 is preserved as a rejected diagnostic branch. The
validation-only search was frozen in
`config/model_c_validation_search_v1.json` (SHA-256
`9e1d44299ae6cb36acbb3fc4ad158fb133cbb2d474f004cd2ef5f0e66dc4c6f6`).
It used four from-scratch successive-halving rounds, taking 10 candidates to
5, 3, 2, and 1 while chronology/step resources rise together from
25%/1,920 steps to 100%/7,680 steps. The full round exactly preserves the
accepted 75%-point learning-rate decay. Physical U/V/temperature/SSH
validation RMSE ratios rank first, followed by the declared 30/90/180-day
physics score. The stage-manifest SHA-256 values are
`40b54530d7d9...12d2f05`, `a2d639f8940c...d5e382`,
`9bbd4fc4ce3c...f30e17`, and `453c05ec408f...2fea2`.
The selected `(24,16)`, width-64 design did not pass the final three-seed rule:
every seed beat persistence for U, V, and temperature and reloaded exactly,
but no seed beat it for SSH. The freeze decision has SHA-256
`e19d502442fc...72155`; it records `configuration_frozen=false` and
`inference_authorized=false`.

Validation-search v1 remains closed as a scientific rejection, not an
operational failure. The separately frozen post-search contract
`config/model_c_data_adequacy_v1.json` (SHA-256
`e185301947b45e88cdd90c4d198ad46bfe4acc9364d366d641eb2676853c84a9`)
was executed in GPU job 290415. Its report (SHA-256
`be672face8c4deb5e15b72c2a09d0781a9d9830bd0984c0b5c05ca88f4e307ee`)
authorizes the bounded data expansion without changing the v1 rejection or
opening inference.

The original 7,530 overlapping ten-day training pairs were not treated as
7,530 independent samples. `config/trajectories_v2_expansion.json` (SHA-256
`7ee2ae10330afaefaea8a30b3fd535a468e90a5c349d1a2befffec469fe045ed`) freezes exact
continuous ten-year extensions for S0--S2 and a successor split with 15,060
training pairs, 780 fresh validation pairs, and 3,450 sealed inference pairs.
Jobs 290439/290443/290444 completed the extensions, conversion, and independent
quality gate. The immutable dataset metadata and quality-report SHA-256 values
are `ae7be47c4569...78b8cc` and `a6f7e5234ace...cf752`; the v1 state prefix and
sampled extension MDS records are bitwise exact.

Coverage contract `config/trajectories_v2_coverage_audit.json` (SHA-256
`e539ca1bb61a...9197f36`) was frozen before job 290446 read any coverage
metric. Its report (SHA-256 `7758cfb64e73...18e5`) meets the prospective
two-times slow-state target. The successor comparison is now frozen in
`config/model_c_successor_training_v1.json`: loss v1, `(24,16)` modes, four
layers, optimizer exposure, and training records remain fixed while data,
Channel-MLP expansion, then latent width are isolated in order. Phase-1 array
290448 completed: the width-64 control and `4C` variant failed only S1 SSH at
`1.103435x` and `1.003573x` persistence, respectively. Width-128 job 290597
passed every training-only criterion, including S1 SSH at `0.762683x`.
Validation contract v2 now fixes three seeds, all 180 complete 10--90-day starts
per regime, A0/persistence/climatology baselines, paired block-bootstrap
confidence, and Bire-primary surface-speed/SST/derived-pressure gates. Seed
replication array 290673 completed with exit zero, and all three seeds passed
the prospective training gate. Fresh-validation job 290738 then rejected the
configuration: day-10 slow-field skill is good, but SST and surface-pressure
errors grow beyond persistence over the 10--90-day curve. The next frozen
training-only audit evaluated the same lead-time behavior on balanced split-1
starts in GPU job 291102 and reproduced it for every seed, classifying an
objective/checkpoint-gate mismatch. Intermediate winds, inference, response
data, and adjoints remain sealed.

## Repository layout

| Path | Contents |
| --- | --- |
| `af_fno/mitgcm/` | Active 1-degree MITgcm code and input templates |
| `src/bire_repro/` | Contract-pinned workflow modules plus `core/`, `analysis/`, and `diagnostics/` packages |
| `slurm/` | Contract-pinned jobs plus MITgcm, data, evaluation, and model subdirectories |
| `scripts/` | Environment, upstream-fetch, and submission helpers |
| `config/` | Declared experiment configuration and A0 historical reference |
| `manifests/` | Pinned upstream versions and lightweight provenance |
| `tests/` | Contract-pinned checks plus core, data, model, and Model C suites |
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
sbatch slurm/mitgcm/af_s0_build.sbatch
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
