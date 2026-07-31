"""S0 Bire-style Figure 3--8 suite for the Bire-faithful protocol arm.

Publishes the same six figures, from the same 15 fixed S0 starts against the
same day-2000 truth and baselines, as the incumbent and the two earlier
Bire-aligned packages, for the arm trained by
:mod:`af_model_c_bire_aligned_faithful` -- one folder per stage.

That arm's architecture is identical to the 1e-2 and 5e-4 arms', so the stage
view, checkpoint loader, stepper, and stage binding in
:mod:`af_model_c_bire_aligned_s0_figures` all apply unchanged.

One thing does differ structurally.  The earlier arms checkpointed at fixed
optimizer steps, so that runner validates each stage's step against the
hard-coded ``CHECKPOINT_STEPS``.  This arm selects its checkpoints by validation
loss, so the retained step is whichever epoch won and is not known in advance.
The binding therefore rebinds ``CHECKPOINT_STEPS`` to the steps the contract
actually declares, read from the contract at bind time.  Without that the suite
would only work when validation happened to pick the last epoch of each stage.

Held-evaluation package: no training, no checkpoint selection, no promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import af_model_c_bire_aligned_s0_figures as aligned_figures
from .af_model_c_bire_aligned_faithful import (
    COSINE_ETA_MIN,
    COSINE_T_MAX,
    MAE_WEIGHT,
    PARENT_MAE_WEIGHT,
    VALIDATION_FRACTION,
)
from .af_model_c_bire_aligned_full_state import STAGE_NAMES

VERSION = "model_c_bire_aligned_faithful_s0_figures_v1"
CONTRACT_STATUS = (
    "frozen_after_the_bire_faithful_training_run_and_before_any_figure_metric"
)
DEFAULT_STAGE = aligned_figures.DEFAULT_STAGE


class BireAlignedFaithfulS0FigureError(RuntimeError):
    """Raised when the Bire-faithful figure contract is violated."""


def selected_steps(contract_path: str | Path) -> tuple[int, ...]:
    """Optimizer steps the contract declares, in stage order."""

    contract = json.loads(Path(contract_path).read_text())
    order = tuple(contract.get("stage_order", ()))
    if order != STAGE_NAMES:
        raise BireAlignedFaithfulS0FigureError("the two Bire stages changed")
    return tuple(
        int(contract["stages"][stage]["model"]["optimizer_step"]) for stage in order
    )


def _readme(report: Mapping[str, Any]) -> str:
    """Describe this arm's active stage and the three corrections."""

    stage = aligned_figures.active_stage()
    label = "one-step pretrained" if stage == "pretrained" else "two-step fine-tuned"
    step = aligned_figures.CHECKPOINT_STEPS[STAGE_NAMES.index(stage)]
    return f"""# Bire-faithful protocol arm ({label}): S0 Figures 3--8

This package evaluates the seed-20260724 **{stage}** checkpoint (optimizer step
{step:,}, selected by lowest validation loss within its stage) of the
Bire-faithful protocol arm under the control wind (tau0 = 0.1 N m-2). All
forecasts use a ten-day prediction interval.

The arm corrects, as one bundle on the working 5e-4 base, three unintended
divergences from the public `oceanfourcast` implementation:

| quantity | earlier arms | this arm |
| --- | --- | --- |
| MAE weight | {PARENT_MAE_WEIGHT} | **{MAE_WEIGHT}** |
| LR schedule | step decay x0.2 at 75% | **cosine, T_max {COSINE_T_MAX}, eta_min {COSINE_ETA_MIN:g}** |
| checkpoint selection | fixed steps | **lowest validation loss per stage** |

Architecture (three FNO blocks, six pointwise channel LayerNorms, 49 external
inputs, the deterministic `oceanfourcast.PosEmbed` fields appended immediately
before lifting, no external 3x3 branch), the two-stage protocol, seed, batch
size 8, betas (0.9, 0.95), zero weight decay, absent gradient clipping, and
lr0 = 5e-4 are frozen against `model_c_bire_aligned_full_state_lr5e4_v1`.
ChannelMLP dropout stays at zero.

Adopting validation-based selection holds out a seeded random
{VALIDATION_FRACTION:.0%} of the split-1 training records, so this arm trained on
90% of the records the earlier arms saw. The holdout is drawn from training-split
records only; no sealed archive was opened.

Compare with `bire_aligned_full_state_lr5e4_bire_s0_inference_v1/{stage}` to
isolate the effect of the three corrections, and with
`single_position_layernorm_bire_s0_inference_v1` for the retained incumbent. The
15 initial conditions, the day-2000 MITgcm truth from evaluation-only job 304735,
the baselines, the reductions, and the six figure filenames are identical across
all packages, so they are directly comparable field for field.

Climatology is the pointwise temporal mean over S0 split-1 training snapshots.
Persistence holds each member's initial physical field fixed. RMSE is first
computed over wet cells for each member; lines/bands are the mean and
10th/90th percentiles across the 15 members.

Figure 6 compares the prior residual rollout-conditioned Model C with this arm.
It is an architecture-direction comparison, not a literal pretrained/fine-tuned
pairing.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `{report["report_content_sha256"]}`.
"""


BINDINGS = ("VERSION", "CONTRACT_STATUS", "_readme", "CHECKPOINT_STEPS")


class _FaithfulBinding:
    """Bind version, status, README, and the *declared* checkpoint steps."""

    def __init__(self, contract_path: str | Path) -> None:
        self.steps = selected_steps(contract_path)

    def __enter__(self) -> None:
        self._saved = {name: getattr(aligned_figures, name) for name in BINDINGS}
        aligned_figures.VERSION = VERSION
        aligned_figures.CONTRACT_STATUS = CONTRACT_STATUS
        aligned_figures._readme = _readme
        aligned_figures.CHECKPOINT_STEPS = self.steps

    def __exit__(self, *exc: Any) -> None:
        for name, value in self._saved.items():
            setattr(aligned_figures, name, value)


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load and stage-resolve this arm's figure contract."""

    with _FaithfulBinding(path):
        return aligned_figures.load_contract(path, verify_sources=verify_sources)


def preflight(
    contract_path: str | Path,
    stage: str = DEFAULT_STAGE,
) -> dict[str, Any]:
    """Verify sources and the stage checkpoint identity without plotting."""

    with _FaithfulBinding(contract_path):
        result = dict(aligned_figures.preflight(contract_path, stage))
    result["mae_weight"] = MAE_WEIGHT
    result["learning_rate_schedule"] = "cosine_annealing"
    result["checkpoint_selection"] = "lowest_validation_loss_within_each_stage"
    result["declared_steps"] = list(selected_steps(contract_path))
    return result


def run(
    contract_path: str | Path,
    stage: str = DEFAULT_STAGE,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Publish the six S0 figures for one stage of the Bire-faithful arm."""

    with _FaithfulBinding(contract_path):
        return aligned_figures.run(contract_path, stage, device_name=device_name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        child.add_argument(
            "--stage",
            choices=(*STAGE_NAMES, "all"),
            default=DEFAULT_STAGE,
        )
        if command == "run":
            child.add_argument(
                "--device",
                choices=("auto", "cpu", "cuda"),
                default="auto",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stages = STAGE_NAMES if args.stage == "all" else (args.stage,)
    if args.command == "preflight":
        result = {stage: preflight(args.contract, stage) for stage in stages}
    else:
        result = {
            stage: run(args.contract, stage, device_name=args.device)
            for stage in stages
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
