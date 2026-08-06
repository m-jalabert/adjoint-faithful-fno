"""S0 Bire-style Figure 3--8 suite for the loss-recovery control.

Publishes the same six figures, from the same 15 fixed S0 starts against the
same day-2000 truth and baselines, as every earlier package, for the arm trained
by :mod:`af_model_c_bire_aligned_loss_recovery`.

Two structural differences from the earlier Bire-aligned figure suites:

* this arm restores the incumbent three-step objective, so it has a single
  gate-selected checkpoint rather than a pretrained/fine-tuned pair.  The
  runner's two-stage vocabulary is therefore rebound to the single stage
  ``selected``, with its optimizer step read from the contract rather than
  assumed;
* its checkpoint payload carries no ``stage_id`` --- there are no stages --- so
  the checkpoint identity check is replaced by one that verifies the
  architecture, the optimizer step, and the base loss-v1 contract hash.  That
  last check is the stronger one here: it proves the published model was trained
  under the restored group-balanced objective and not under Bire's MSE+MAE.

Held-evaluation package: no training, no checkpoint selection, no promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import af_model_c_bire_aligned_s0_figures as aligned_figures
from .af_model_c import MODEL_C_LOSS_V1_CONTRACT_SHA256
from .af_model_c_bire_aligned_full_state import (
    BireAlignedArchitecture,
    BireAlignedStepper,
    build_bire_aligned_model,
)

try:  # Keep documentation-only imports usable without the optional ML stack.
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

VERSION = "model_c_bire_aligned_loss_recovery_s0_figures_v1"
CONTRACT_STATUS = (
    "frozen_after_the_loss_recovery_training_run_and_before_any_figure_metric"
)
STAGE_NAMES = ("selected",)
DEFAULT_STAGE = "selected"


class BireAlignedLossRecoveryS0FigureError(RuntimeError):
    """Raised when the loss-recovery figure contract is violated."""


def selected_steps(contract_path: str | Path) -> tuple[int, ...]:
    """The optimizer step the contract declares, as a one-element tuple."""

    contract = json.loads(Path(contract_path).read_text())
    if tuple(contract.get("stage_order", ())) != STAGE_NAMES:
        raise BireAlignedLossRecoveryS0FigureError(
            "the loss-recovery arm publishes exactly one selected checkpoint"
        )
    return (int(contract["stages"]["selected"]["model"]["optimizer_step"]),)


def _selected_stepper(
    contract: Mapping[str, Any],
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> BireAlignedStepper:
    """Build the loss-recovery map and verify it was trained under loss v1."""

    checkpoint = Path(contract["artifacts"]["selected_checkpoint"]["path"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture_dict = contract["selected_model"]["architecture"]
    if (
        payload.get("architecture") != architecture_dict
        or int(payload.get("optimizer_step", -1))
        != int(contract["selected_model"]["optimizer_step"])
        or payload.get("base_loss_contract_sha256")
        != MODEL_C_LOSS_V1_CONTRACT_SHA256
        or int(payload.get("rollout_steps", -1)) != 3
    ):
        raise BireAlignedLossRecoveryS0FigureError(
            "selected checkpoint identity or training objective changed"
        )
    model = build_bire_aligned_model(
        BireAlignedArchitecture(**architecture_dict)
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with np.load(
        Path(contract["artifacts"]["selected_normalization"]["path"])
    ) as artifact:
        mean = np.asarray(artifact["pointwise_mean"], dtype=np.float32)
        scale = np.asarray(artifact["pointwise_scale"], dtype=np.float32)
    return BireAlignedStepper(
        model=model,
        device=device,
        wet=wet,
        mean=mean,
        scale=scale,
        wind_mean=wind_mean,
        wind_scale=wind_scale,
    )


def _readme(report: Mapping[str, Any]) -> str:
    step = aligned_figures.CHECKPOINT_STEPS[0]
    return f"""# Bire-aligned architecture, incumbent objective: S0 Figures 3--8

This package evaluates the seed-20260724, step-{step:,} gate-selected checkpoint
of the **loss-recovery control** under the control wind (tau0 = 0.1 N m-2). All
forecasts use a ten-day prediction interval.

The arm is an architecture-fixed control against
`model_c_bire_aligned_full_state_lr5e4_v1`. It keeps three FNO blocks, 24x16
modes, width 128, six pointwise channel LayerNorms, the deterministic
`oceanfourcast.PosEmbed` fields appended immediately before lifting, no external
3x3 raw-input branch, 10% padding, Adam(5e-4, betas 0.9/0.95, weight decay 0),
batch size 8, and the 7,680-step budget. It changes only the objective and the
rollout exposure, restoring the incumbent group-balanced Model C loss v1 --

    L_state = (L_U + L_V + L_Theta + L_eta) / 4

with its increment, rollout, spectral, and western-boundary terms -- over a
three-step unrolled rollout throughout.

Bire's `MSE + 0.01 MAE` averaged over all 46 normalized channels gives the
physical groups effective multiplicities `U:V:Theta:eta = 15:15:15:1`, so the
free surface received 1/46 of the channel-averaged loss. This arm tests whether
that weighting, rather than the architecture, cost the forecast skill.

Compare with `bire_aligned_full_state_lr5e4_bire_s0_inference_v1/finetuned` to
isolate the objective, and with `single_position_layernorm_bire_s0_inference_v1`
for the retained incumbent. The 15 initial conditions, the day-2000 MITgcm truth
from evaluation-only job 304735, the baselines, the reductions, and the six
figure filenames are identical across all packages, so they are directly
comparable field for field.

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


BINDINGS = (
    "VERSION",
    "CONTRACT_STATUS",
    "_readme",
    "CHECKPOINT_STEPS",
    "STAGE_NAMES",
    "_selected_stepper",
    # The runner's active stage defaults to the two-stage arm's "finetuned",
    # which does not exist here.  It must be rebound too, or a standalone
    # ``load_contract`` --- outside the runner's own stage binding --- resolves
    # a stage view that this contract has no entry for.
    "_ACTIVE_STAGE",
)


class _LossRecoveryBinding:
    """Bind the single-stage vocabulary and checkpoint check into the runner."""

    def __init__(self, contract_path: str | Path) -> None:
        self.steps = selected_steps(contract_path)

    def __enter__(self) -> None:
        self._saved = {name: getattr(aligned_figures, name) for name in BINDINGS}
        aligned_figures.VERSION = VERSION
        aligned_figures.CONTRACT_STATUS = CONTRACT_STATUS
        aligned_figures._readme = _readme
        aligned_figures.CHECKPOINT_STEPS = self.steps
        aligned_figures.STAGE_NAMES = STAGE_NAMES
        aligned_figures._selected_stepper = _selected_stepper
        aligned_figures._ACTIVE_STAGE = DEFAULT_STAGE

    def __exit__(self, *exc: Any) -> None:
        for name, value in self._saved.items():
            setattr(aligned_figures, name, value)


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load and stage-resolve this arm's figure contract."""

    with _LossRecoveryBinding(path):
        return aligned_figures.load_contract(path, verify_sources=verify_sources)


def preflight(
    contract_path: str | Path,
    stage: str = DEFAULT_STAGE,
) -> dict[str, Any]:
    """Verify sources and the selected checkpoint identity without plotting."""

    with _LossRecoveryBinding(contract_path):
        result = dict(aligned_figures.preflight(contract_path, stage))
    result["objective"] = "incumbent_group_balanced_model_c_loss_v1"
    result["loss_contract_sha256"] = MODEL_C_LOSS_V1_CONTRACT_SHA256
    result["rollout_steps"] = 3
    return result


def run(
    contract_path: str | Path,
    stage: str = DEFAULT_STAGE,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Publish the six S0 figures for the loss-recovery control."""

    with _LossRecoveryBinding(contract_path):
        return aligned_figures.run(contract_path, stage, device_name=device_name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        child.add_argument("--stage", choices=(*STAGE_NAMES, "all"), default=DEFAULT_STAGE)
        if command == "run":
            child.add_argument(
                "--device", choices=("auto", "cpu", "cuda"), default="auto"
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stages = STAGE_NAMES if args.stage == "all" else (args.stage,)
    if args.command == "preflight":
        result = {s: preflight(args.contract, s) for s in stages}
    else:
        result = {
            s: run(args.contract, s, device_name=args.device) for s in stages
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
