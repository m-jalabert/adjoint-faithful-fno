"""Emit the frozen training contract for the turbulent forward study.

Every hash the runner audits is computed here from the bytes on disk, so the
contract cannot drift from the tree it describes.  Run it once the trajectory
store exists; re-run it after any edit to a hashed source file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "turbulent_double_gyre"
sys.path.insert(0, str(STUDY / "src"))

from turbfno import train as T  # noqa: E402
from turbfno.model import EXPECTED_PARAMETER_COUNT, ProductionArchitecture  # noqa: E402
from turbfno.objective import (  # noqa: E402
    LOSS_CONTRACT_SHA256,
    SPECTRAL_BINS,
    WESTERN_BOUNDARY_WIDTH,
    production_loss_config,
)

SCRATCH = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno")
DATASET = SCRATCH / "datasets" / "trajectories_turb_v1.zarr"
TURB_SPINUP = SCRATCH / "mitgcm_turb_v1" / "S0_turb" / "spinup" / "years_000_010"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build() -> dict:
    config = production_loss_config()
    architecture = ProductionArchitecture().to_dict()
    return {
        "version": T.VERSION,
        "contract_status": T.CONTRACT_STATUS,
        "purpose": (
            "the 0.25-degree turbulent counterpart of "
            "model_c_adjoint_faithful_nominal_control_v1 seed 20260911. Same "
            "problem, same protocol, same objective and same schedule; the grid "
            "is 248x248 at 0.25 degrees instead of 62x62 at 1 degree, the "
            "spectral truncation is 64x64 instead of 32x32, and the three "
            "quantities that were implicitly one degree -- the meridional cell "
            "metric, the western-boundary band and the spectral bin width -- are "
            "restated in physical units so they mean what they meant before. "
            "Forward only: no adjoint study, no response training, one seed"
        ),
        "study": {
            "adjoint": False,
            "response_training": False,
            "seeds": [T.SEED],
            "separated_from": (
                "the 1-degree study in src/oceanfno; this tree shares no source "
                "file with it and no contract of it is re-hashed here"
            ),
        },
        "architecture": architecture,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "changed_relative_to_the_one_degree_arm": {
            "grid_shape": {"from": [62, 62], "to": [248, 248]},
            "n_modes": {"from": [32, 32], "to": [64, 64]},
            "parameter_count": {"from": 27_297_960, "to": EXPECTED_PARAMETER_COUNT},
            "spectral_share_of_parameters": {"from": 0.9795, "to": 0.9946},
            "domain_padding": {
                "from": "10% constant-zero latent padding",
                "to": "10% raised-cosine tapered replicate latent padding",
                "reason": (
                    "the first turbulent rollout developed narrowband zonal "
                    "stripes adjacent to the domain edge; continuing the lifted "
                    "field into the halo removes the hidden-to-zero jump while "
                    "retaining an exactly zero, periodically continuous FFT edge"
                ),
            },
            "microbatch_size": {
                "from": 4,
                "to": T.MICROBATCH_SIZE,
                "reason": (
                    "the six-step activation load at 248x248 with 64x64 modes "
                    "peaks at 20.8 GiB on a 32 GB V100; two samples need ~35 GiB"
                ),
            },
            "gradient_accumulation_steps": {
                "from": 2,
                "to": T.GRADIENT_ACCUMULATION_STEPS,
                "reason": "holds the effective batch at 8",
            },
            "spectral_bins": {
                "from": 12,
                "to": SPECTRAL_BINS,
                "reason": (
                    "the loss bins by radius normalized to Nyquist, and Nyquist "
                    "is four times higher here, so a fixed bin count would "
                    "quadruple the physical band each bin spans"
                ),
            },
            "western_boundary_width": {
                "from": 4,
                "to": WESTERN_BOUNDARY_WIDTH,
                "reason": (
                    "four cells was four degrees at 1 degree; sixteen keeps the "
                    "same physical band now that the boundary current is resolved"
                ),
            },
            "meridional_cell_metric": {
                "from": "R * deg2rad(1.0), which coincided with the cell spacing",
                "to": "R * deg2rad(0.25), the actual cell spacing",
                "affects": [
                    "the barotropic streamfunction in diagnostics",
                    "the meridional PHIHYD gradient in the pressure-gradient loss",
                    "the meridional transport divergence in the continuity loss",
                ],
                "reason": (
                    "these are cell differences divided by a cell spacing; on the "
                    "1-degree grid metres-per-degree and metres-per-cell were "
                    "equal, here they differ by exactly four"
                ),
            },
            "unchanged": [
                "the eight loss terms and their weights",
                "rollout_steps",
                "effective batch",
                "maximum_steps and the checkpoint steps",
                "learning rate and schedule",
                "the split, 17,820 training sequences and every evaluation lead",
                "the spectral cap and its estimator",
                "hidden_channels, n_layers and the local 3x3 branch",
            ],
        },
        "initialization": {
            "from_scratch": True,
            "load_model_state": False,
            "load_optimizer_state": False,
            "normalization_reused": False,
            "parent_checkpoint": None,
            "local_branch_initialization": "zeros",
            "local_branch_bias": False,
            "lineage": "none; this arm has no parent and no fine-tuning stage",
            "fno_weights": "neuraloperator_default_random_initialization",
            "spectral_weights": "random_initialization",
            "lifting_projection_and_channel_mlp": "random_initialization",
            "layer_norm_scale": 1.0,
            "layer_norm_bias": 0.0,
        },
        "training": {
            "optimizer": "adam",
            "rollout_steps": T.ROLLOUT_STEPS,
            "batch_size": T.BATCH_SIZE,
            "microbatch_size": T.MICROBATCH_SIZE,
            "gradient_accumulation_steps": T.GRADIENT_ACCUMULATION_STEPS,
            "initial_learning_rate": T.LEARNING_RATE,
            "adam_betas": list(T.ADAM_BETAS),
            "weight_decay": T.WEIGHT_DECAY,
            "maximum_steps": T.MAXIMUM_STEPS,
            "checkpoint_steps": list(T.CHECKPOINT_STEPS),
            "decay_fraction": T.DECAY_FRACTION,
            "decay_factor": T.DECAY_FACTOR,
            "seed": T.SEED,
            "gradient_clipping": False,
            "dropout": 0.0,
            "from_scratch": True,
            "state_transitions": T.STATE_TRANSITIONS,
            "schedule_note": (
                f"steps 1-{int(T.MAXIMUM_STEPS * T.DECAY_FRACTION)} at "
                f"{T.LEARNING_RATE}, the remainder at "
                f"{T.LEARNING_RATE * T.DECAY_FACTOR}"
            ),
            "data_parallel_note": (
                "gradient_accumulation_steps is the global count per optimizer "
                "step; a data-parallel launch divides it among the ranks and "
                "averages the gradients, which changes only the summation order"
            ),
        },
        "loss": {
            "contract_sha256": LOSS_CONTRACT_SHA256,
            "all_terms_active_from_step_1": True,
            "contraction_penalty": False,
            "increment_weight": config.increment_weight,
            "rollout_weight": config.rollout_weight,
            "spectral_weight": config.spectral_weight,
            "boundary_weight": config.boundary_weight,
            "pressure_gradient_weight": config.pressure_gradient_weight,
            "continuity_weight": config.continuity_weight,
            "barotropic_transport_weight": config.barotropic_transport_weight,
            "spectral_bins": SPECTRAL_BINS,
            "western_boundary_width": WESTERN_BOUNDARY_WIDTH,
            "physics_terms_add_no_output_channels": True,
            "staged_fine_tuning": False,
        },
        "spectral_normalization": {
            "applied": True,
            "applies_to": "spectral_convolutions_only",
            "form": "R_k <- R_k * min(1, 1 / sigma_max(R_k))",
            "estimator": "persistent alternating power iteration on the weight",
            "operator": "the dense complex 128x128 channel mixing at each Fourier mode",
            "power_iterations_per_forward": T.POWER_ITERATIONS,
            "warmup_iterations": T.WARMUP_ITERATIONS,
            "blocks": 3,
            "modes_per_block": 64 * (64 // 2 + 1),
            "matrices_total": T.SPECTRAL_MATRICES_TOTAL,
            "checkpoints_materialized": True,
            "adds_parameters": 0,
            "one_sided": True,
            "data_independent": (
                "the iteration reads the weight only, never a batch, so every "
                "data-parallel rank computes the same sigma from the same weights"
            ),
            "power_iterations_per_optimizer_step_note": (
                "two per forward and seven forwards per microbatch, so the count "
                "per optimizer step falls with the world size: 112 on one rank, "
                "56 on two, 28 on four. All act on the same frozen weight within "
                "a step and all are far past the warmup, so the estimate is "
                "converged in every case"
            ),
            "reference": "McCabe et al., arXiv:2306.10619",
        },
        "normalization": {
            "recomputed_from_training_days_only": True,
            "train_days": list(T.TRAIN_RANGE),
            "reused_from_a_previous_run": False,
            "definition": (
                "x_hat_c(y,x) = (x_c(y,x) - mu_c(y,x)) / sigma_c(y,x), land zero"
            ),
            "pointwise_shape": [46, 248, 248],
            "channel_scale_floor_quantile": 0.05,
            "absolute_scale_floor": 1e-06,
            "increment_scale": (
                "per_channel_rms_of_the_normalized_ten_day_increment_over_training_pairs"
            ),
            "static_channel_normalization": (
                "wet_mask stays a raw 0/1 indicator; the four physical "
                "coefficient fields are standardized over wet cells"
            ),
        },
        "checkpoint_selection": {
            "rule": (
                "keep every checkpoint within 5 percent of the best short AUC in "
                "each primary field, then keep only those at or below the growth "
                "ceiling, then minimize the worst 90-360 day RMSE-AUC ratio to "
                "climatology; publish exactly one selected.pt"
            ),
            "short_auc_tolerance": T.SHORT_AUC_TOLERANCE,
            "short_auc_window_days": [10, 90],
            "long_auc_window_days": [90, 360],
            "worst_long_ratio_ceiling": T.WORST_LONG_RATIO_CEILING,
            "primary_fields": list(T.PRIMARY_FIELDS),
            "rollout_days": 360,
            "growth_rate_ceiling": T.GROWTH_RATE_CEILING,
            "growth_rate_calls": T.DIAGNOSTIC_CALLS,
            "growth_rate_regime": "S0_turb",
            "open_question": (
                "the ceiling of 1.0 is inherited from the 1-degree arm, where the "
                "flow showed no twin-perturbation growth. A 0.25-degree double "
                "gyre at viscAh=500 is expected to be chaotic, in which case a "
                "non-amplifying emulator is not the physical target and this "
                "filter may reject every checkpoint for a correct reason. It is "
                "left inherited so the first run measures the growth rate against "
                "the same yardstick; the ceiling is to be re-derived from the "
                "measured MITgcm twin growth before any second arm is frozen"
            ),
        },
        "data": {
            "regimes": ["S0_turb", "S1_turb", "S2_turb"],
            "train": [0, 6000],
            "validation": [6000, 7200],
            "inference": [6200, 7200],
            "evaluation_truth_only": [7200, 9000],
            "horizon_days": 10,
            "training_sequences": T.TRAINING_RECORDS,
            "training_starts_per_regime": T.TRAINING_STARTS_PER_REGIME,
            "identical_to_the_one_degree_arm": (
                "9,000 days per regime in both stores, so every split boundary, "
                "the sequence count and every evaluation lead carry over"
            ),
        },
        "output": {
            "project_root": str(STUDY / "outputs" / T.VERSION),
            "scratch_root": str(SCRATCH / "models" / "turb" / T.VERSION),
            "artifacts": list(T.OUTPUT_ARTIFACTS),
        },
        "sources": {
            "dataset": {
                "path": str(DATASET),
                "metadata_sha256": sha256(DATASET / ".zmetadata"),
                "version": "trajectories_turb_v1",
            },
            "mitgcm_declaration": {
                "path": str(TURB_SPINUP / "data"),
                "sha256": sha256(TURB_SPINUP / "data"),
            },
            "mitgcm_sst_relaxation": {
                "path": str(TURB_SPINUP / "SST_relax.bin"),
                "sha256": sha256(TURB_SPINUP / "SST_relax.bin"),
            },
            "mitgcm_zonal_spacing": {
                "path": str(TURB_SPINUP / "DXF.data"),
                "sha256": sha256(TURB_SPINUP / "DXF.data"),
            },
        },
        # Relative to the study root, which is what load_contract resolves them
        # against: it takes the contract's own parents[1].
        "source_hashes": {
            name: sha256(STUDY / name) for name in sorted(T.REQUIRED_SOURCE_HASHES)
        },
        "read_contract": {
            "training_state": True,
            "validation_state": True,
            "inference_state": False,
            "adjoint_state": False,
            "response_state": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(
            STUDY / "config" / f"{T.VERSION}.json"
        ),
    )
    args = parser.parse_args()
    contract = build()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")
    print(f"  loss contract sha256   {contract['loss']['contract_sha256']}")
    print(f"  parameters             {contract['expected_parameter_count']:,}")
    print(f"  dataset metadata sha   {contract['sources']['dataset']['metadata_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
