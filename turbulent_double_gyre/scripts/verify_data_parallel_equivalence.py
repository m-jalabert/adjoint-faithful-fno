"""Prove the sharded step equals the single-process step, on synthetic data.

Runs one optimizer step over the *same* eight deterministic samples at whatever
world size the launcher provides, and prints the resulting gradient's summary
norms.  Launch it at world 1 and world 2 and compare: if the sharding, the
gradient averaging and the per-sample structure of the objective all hold, the
two must agree to float32 summation-order tolerance.

Deliberately synthetic --- it needs no trajectory store, so it can run while the
store is still building, and it isolates the parallel arithmetic from the data
pipeline.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "turbulent_double_gyre" / "src"))

from turbfno import model as M  # noqa: E402
from turbfno import distributed as D  # noqa: E402
from turbfno.barotropic_transport import barotropic_transport_relative_l2  # noqa: E402
from turbfno.continuity import ContinuityContext, continuity_relative_l2  # noqa: E402
from turbfno.dataset import western_boundary_mask  # noqa: E402
from turbfno.objective import production_loss_config, production_loss_terms  # noqa: E402
from turbfno.perturbation_growth import (  # noqa: E402
    EPSILON_RELATIVE,
    single_call_amplification,
)
from turbfno.pressure_gradient import (  # noqa: E402
    PressureGradientContext,
    pressure_gradient_relative_l2,
)
from turbfno.runtime import seed_everything  # noqa: E402
from turbfno.spectral_norm import apply_mode_spectral_norm  # noqa: E402
from turbfno.train import (  # noqa: E402
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    MICROBATCH_SIZE,
    SEED,
)

GRID = 248


def synthetic_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The same eight samples on every rank, drawn from one fixed seed."""

    generator = torch.Generator(device="cpu").manual_seed(20260911)
    features = torch.randn(
        BATCH_SIZE, 51, GRID, GRID, generator=generator, dtype=torch.float32
    ) * 0.3
    futures = torch.randn(
        BATCH_SIZE, 6, 46, GRID, GRID, generator=generator, dtype=torch.float32
    ) * 0.3
    return features.to(device), futures.to(device)


def main() -> int:
    topology = D.initialize("cuda")
    accumulation = D.require_divisible(GRADIENT_ACCUMULATION_STEPS, topology)
    device = torch.device(f"cuda:{topology.local_rank}")
    seed_everything(SEED)

    model = M.build_model(M.ProductionArchitecture()).to(device)
    apply_mode_spectral_norm(model, warmup_iterations=40)
    parameters = M.parameter_count(model)

    wet_np = np.ones((GRID, GRID), dtype=bool)
    wet_np[:4] = wet_np[-4:] = wet_np[:, :4] = wet_np[:, -4:] = False
    wet = torch.from_numpy(wet_np.astype(np.float32))[None, None].to(device)
    config = production_loss_config()
    boundary = torch.from_numpy(
        western_boundary_mask(wet_np, config.western_boundary_width).astype(np.float32)
    )[None, None].to(device)
    increment_scale = torch.ones(46, device=device)

    lat = np.linspace(14.125, 75.875, GRID, dtype=np.float32)
    dx = np.repeat((111000.0 * np.cos(np.deg2rad(lat)) * 0.25)[:, None], GRID, axis=1)
    fields = {
        "pointwise_mean": np.zeros((46, GRID, GRID), dtype=np.float32),
        "pointwise_scale": np.ones((46, GRID, GRID), dtype=np.float32),
        "zonal_spacing_m": dx.astype(np.float32),
        "wet_mask": wet_np,
    }
    pressure_context = PressureGradientContext(**fields)
    continuity_context = ContinuityContext(**fields)

    features, futures = synthetic_batch(device)
    direction = torch.randn(1, 46, GRID, GRID, generator=torch.Generator().manual_seed(7))
    direction = (direction / direction.norm()).to(device)

    # This rank's share of the eight microbatches, dealt exactly as the training
    # loop deals them: microbatch j of the step goes to rank j mod world_size.
    mine = list(range(topology.rank, GRADIENT_ACCUMULATION_STEPS, topology.world_size))
    assert len(mine) == accumulation

    model.train()
    model.zero_grad(set_to_none=True)
    totals: dict[str, float] = {}
    for index in mine:
        lo = index * MICROBATCH_SIZE
        sample = features[lo : lo + MICROBATCH_SIZE]
        target = futures[lo : lo + MICROBATCH_SIZE]
        predictions = M.state_unroll(model, sample, wet, config.rollout_steps)
        present, static = sample[:, :46], sample[:, 46:]
        _, direction = single_call_amplification(
            model, present, static, wet, predictions[:, 0], direction,
            epsilon_relative=EPSILON_RELATIVE,
        )
        auxiliary = {
            "pressure_gradient": pressure_gradient_relative_l2(
                predictions, target, pressure_context
            ),
            "continuity": continuity_relative_l2(
                predictions, target, present, continuity_context
            ),
            "barotropic_transport": barotropic_transport_relative_l2(
                predictions, target, present, continuity_context
            ),
        }
        terms = production_loss_terms(
            predictions, target, present, wet, boundary,
            increment_scale, config, auxiliary,
        )
        # Exactly as the training loop does it: divide by this rank's *local*
        # microbatch count, so each rank builds the mean over its own samples,
        # and the mean over ranks that follows is then the mean over the batch.
        # Dividing by the global count here instead and still averaging would
        # scale the update down by the world size -- which is what this check
        # caught the first time it ran.
        (terms["total"] / accumulation).backward()
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())

    topology.average_gradients(model)

    flat = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
    result = {
        "world_size": topology.world_size,
        "rank": topology.rank,
        "microbatches_this_rank": mine,
        "parameters": parameters,
        "grad_l2": float(flat.detach().float().norm().item()),
        "grad_l1": float(flat.detach().float().abs().sum().item()),
        "grad_max_abs": float(flat.detach().float().abs().max().item()),
        "mean_total_loss_this_rank": totals["total"] / len(mine),
    }
    if topology.is_primary:
        print(json.dumps(result, sort_keys=True), flush=True)
        out = Path(os.environ.get("AF_EQUIV_OUT", "/dev/null"))
        if str(out) != "/dev/null":
            out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    topology.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
