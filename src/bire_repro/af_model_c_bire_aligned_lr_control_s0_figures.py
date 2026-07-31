"""S0 Bire-style Figure 3--8 suite for the Bire-aligned learning-rate control.

Publishes the same six figures, from the same 15 fixed S0 starts against the
same day-2000 truth and baselines, as
``single_position_layernorm_bire_s0_inference_v1`` and
``bire_aligned_full_state_bire_s0_inference_v1``, for the arm trained by
:mod:`af_model_c_bire_aligned_lr_control` -- one folder per stage.

The control's architecture is *identical* to the 1e-2 arm's, so the stage view,
checkpoint loader, stepper, and stage binding in
:mod:`af_model_c_bire_aligned_s0_figures` all apply unchanged.  Only three things
differ: the contract's version string, its status string, and the published
README, which must name this arm rather than the 1e-2 one.  Those are module
globals of that runner, resolved at call time, so rebinding the three reuses
everything else without editing a source file whose SHA-256 the completed 1e-2
figure contract certifies.

Held-evaluation package: no training, no checkpoint selection, no promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import af_model_c_bire_aligned_s0_figures as aligned_figures
from .af_model_c_bire_aligned_full_state import CHECKPOINT_STEPS, STAGE_NAMES
from .af_model_c_bire_aligned_lr_control import (
    CONTROL_LEARNING_RATE,
    PARENT_LEARNING_RATE,
)

VERSION = "model_c_bire_aligned_full_state_lr5e4_s0_figures_v1"
CONTRACT_STATUS = (
    "frozen_after_the_bire_aligned_lr_control_training_run_and_before_any_figure_metric"
)
DEFAULT_STAGE = aligned_figures.DEFAULT_STAGE

_STAGE_TEXT = {
    "pretrained": ("one-step pretrained", "3,840 one-step updates"),
    "finetuned": (
        "two-step fine-tuned",
        "3,840 one-step pretraining updates followed by 3,840 two-step "
        "autoregressive fine-tuning updates",
    ),
}


def _readme(report: Mapping[str, Any]) -> str:
    """Describe this arm's active stage, not the 1e-2 arm's."""

    stage = aligned_figures.active_stage()
    label, protocol = _STAGE_TEXT[stage]
    step = CHECKPOINT_STEPS[STAGE_NAMES.index(stage)]
    return f"""# Bire-aligned FNO, learning-rate control ({label}): S0 Figures 3--8

This package evaluates the seed-20260724, step-{step:,} **{stage}** checkpoint of
the Bire-aligned full-state Model C **learning-rate control** under the control
wind (tau0 = 0.1 N m-2). All forecasts use a ten-day prediction interval.

The arm is a one-factor control against `model_c_bire_aligned_full_state_v1`:
the only declared change is the optimizer learning rate,
{PARENT_LEARNING_RATE:g} -> {CONTROL_LEARNING_RATE:g}. Architecture (three FNO
blocks, six pointwise channel LayerNorms, 49 external inputs, the deterministic
`oceanfourcast.PosEmbed` sine/cosine fields appended immediately before lifting,
no external 3x3 raw-input branch), the wet-cell `MSE + 0.01 MAE` objective, the
two-stage protocol, the seed, batch size 8, betas (0.9, 0.95), zero weight
decay, and the absence of gradient clipping are all frozen against that parent.
Training was {protocol}.

The 1e-2 parent collapsed to climatology: its one-step normalized MSE settled at
the zero-anomaly value of 1.0 and its day-200 ACC was +0.06 to +0.11. Comparing
this package with
`bire_aligned_full_state_bire_s0_inference_v1/{stage}` isolates the effect of
the learning rate alone; comparing it with
`single_position_layernorm_bire_s0_inference_v1` compares the Bire-aligned
architecture package against the retained incumbent.

The 15 initial conditions, the day-2000 MITgcm truth from evaluation-only job
304735, the climatology and persistence baselines, the metric reductions, and
the six figure filenames are identical across all three packages, so they are
directly comparable field for field.

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


RUNNER_BINDINGS = ("VERSION", "CONTRACT_STATUS", "_readme")


class _ControlBinding:
    """Bind this arm's version, status, and README into the 1e-2 figure runner."""

    def __enter__(self) -> None:
        self._saved = {
            name: getattr(aligned_figures, name) for name in RUNNER_BINDINGS
        }
        aligned_figures.VERSION = VERSION
        aligned_figures.CONTRACT_STATUS = CONTRACT_STATUS
        aligned_figures._readme = _readme

    def __exit__(self, *exc: Any) -> None:
        for name, value in self._saved.items():
            setattr(aligned_figures, name, value)


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load and stage-resolve this arm's figure contract."""

    with _ControlBinding():
        return aligned_figures.load_contract(path, verify_sources=verify_sources)


def preflight(
    contract_path: str | Path,
    stage: str = DEFAULT_STAGE,
) -> dict[str, Any]:
    """Verify sources and the stage checkpoint identity without plotting."""

    with _ControlBinding():
        result = dict(aligned_figures.preflight(contract_path, stage))
    result["initial_learning_rate"] = CONTROL_LEARNING_RATE
    result["parent_initial_learning_rate"] = PARENT_LEARNING_RATE
    result["single_declared_change"] = "initial_learning_rate"
    return result


def run(
    contract_path: str | Path,
    stage: str = DEFAULT_STAGE,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Publish the six S0 figures for one stage of the learning-rate control."""

    with _ControlBinding():
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
