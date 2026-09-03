# Turbulent double gyre

The 0.25-degree counterpart of the 1-degree AF--FNO forward study. One model,
one seed, forward only: no adjoint study, no response training, no fine-tuning
stage.

Everything here is self-contained. `src/turbfno` is a copy of `src/oceanfno`
carrying only the modules a forward study needs, so no contract of the 1-degree
study is invalidated by anything in this tree and nothing here is hashed against
it.

## What changed, and why

The physics problem, the protocol and the objective are the 1-degree study's.
Both stores hold 9,000 days per regime, so the split, the 17,820 training
sequences and every evaluation lead carry over untouched. What changed:

| | 1 degree | 0.25 degree |
|---|---|---|
| grid | 62 x 62 | 248 x 248 |
| wet cells | 3,600 | 57,600 |
| `n_modes` | 32 x 32 | 64 x 64 |
| parameters | 27,297,960 | 104,368,296 |
| spectral share | 97.95 % | 99.46 % |
| microbatch | 4 | 1 |
| accumulation | 2 | 8 |
| `spectral_bins` | 12 | 48 |
| `western_boundary_width` | 4 cells (4 deg) | 16 cells (4 deg) |

`n_modes` is a cutoff in cycles per basin, and the basin is the same 62 x 62
degrees in both runs. At 32 modes the highest retained wavenumber is 16 cycles
per basin either way, so keeping 32 would have put the spectral cutoff at the
same 3.9-degree wavelength while the grid resolved down to 0.5 degrees -- every
newly resolved eddy scale would reach the output only through the 3 x 3 local
branch and the pointwise channel MLP, never through a Fourier convolution. 64
modes halves that cutoff to 1.94 degrees. 128 modes, which would preserve the
*fraction* of the spectrum, is 409 M parameters and does not fit a 32 GB V100 at
microbatch 1.

Three quantities were implicitly one degree in the 1-degree tree because
metres-per-degree and metres-per-cell coincided there. They are cell metrics,
they differ by four here, and all three are restated:

- the meridional step of the barotropic streamfunction (`diagnostics`)
- the meridional PHIHYD gradient in the pressure-gradient loss
- the meridional transport divergence in the continuity loss

Left uncorrected, every published streamfunction would be 4x out and the two
physics losses would silently weight meridional against zonal by 4x.

## Layout

```
config/    the frozen training contract, generated not hand-written
scripts/   dataset builder, contract generator, data-parallel equivalence check
slurm/     one submission script per stage
src/turbfno/  the forward-only package
work/      benchmark and verification output
outputs/   reports, figures, manifests
logs/      Slurm stdout and stderr
```

## Order of operations

```bash
# 1. Build the store (~1 h, ~207 GB).  chunk_days=1: each sample reads seven
#    non-adjacent day slices and this store is far too large to cache.
sbatch turbulent_double_gyre/slurm/build_turb_dataset.sbatch

# 2. Freeze the contract against the bytes that now exist on disk.
.venv/bin/python turbulent_double_gyre/scripts/make_turb_contract.py

# 3. Check the sharded step equals the single-process step (synthetic, ~10 min).
sbatch turbulent_double_gyre/slurm/verify_data_parallel.sbatch

# 4. Train.  The world size must divide both the eight microbatches per
#    optimizer step and the 17,820 in an epoch: 1, 2 or 4.
sbatch --gres=gpu:v100:2 --ntasks=2 --cpus-per-task=10 \
  turbulent_double_gyre/slurm/train_turb_forward_control_v1.sbatch
```

## Data parallelism

`turbfno.distributed` averages gradients between optimizer steps rather than
wrapping the model in `DistributedDataParallel`. The model is called seven times
per microbatch before a single backward, and its spectral weight is a
`parametrize` parametrization whose buffers advance in place on every forward;
DDP's per-forward bookkeeping assumes one forward per backward and would have to
be disarmed on all but the last call. The allreduce it replaces is 794 MiB,
measured at 20.7 ms on two V100s over this node's full NVLink mesh against a
9.6 s step -- 0.2 per cent, so the overlap DDP would buy is below the noise.

Sharding is exact, not approximate: every term of the objective reduces over its
non-batch dimensions *before* the batch mean, so an unweighted mean of per-rank
means is the mean over the whole batch, provided the ranks carry equal sample
counts. A world size that divides neither the eight microbatches of an optimizer step
nor the 17,820 of an epoch is refused rather than silently mis-sharded, which
leaves 1, 2 and 4. `scripts/verify_data_parallel_equivalence.py` checks the
claim end to end.

## Open questions carried into the first run

- **The growth-rate ceiling is inherited.** `growth_rate_ceiling = 1.0` was set
  where the 1-degree S0 showed no twin-perturbation growth at all. A 0.25-degree
  double gyre at `viscAh = 500` is expected to be chaotic, in which case a
  non-amplifying emulator is not the physical target and the filter may reject
  every checkpoint for a correct reason. It is left as-is so the first run
  measures against the same yardstick; re-derive it from the measured MITgcm
  twin growth before freezing a second arm.
- **The learning-rate schedule is inherited.** 5e-4 was tuned at 62 x 62 with
  27 M parameters against a 104 M-parameter model on a 16x larger field.
- **Validation does not shard.** Rank 0 runs the four 360-day checkpoint
  validations and the growth measurement alone, ~1.5 h. At four ranks that is
  already an eighth of the job.
