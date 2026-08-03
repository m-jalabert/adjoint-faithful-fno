"""S0 Bire-style Figure 3--8 suite for the chronological-split arm.

Publishes the same six figures, from the same 15 fixed S0 starts against the
same day-2000 truth, as every earlier package, for the arm trained by
:mod:`af_model_c_bire_aligned_chronological`.  Those starts lie in 6660--7199,
which is test under the stored split *and* under the chronological one, so this
package is directly comparable with
``bire_aligned_loss_recovery_bire_s0_inference_v1/selected``.

Three bindings are required beyond the single-stage vocabulary:

* **the climatology baseline must be rebuilt.**  The frozen runner derives it
  from the stored split-1 snapshot codes, which cover 0--2519 and 3690--6209 and
  therefore include 5850--6209 --- test days under this split.  Leaving it alone
  would leak 360 test days into the baseline the model is scored against.  The
  replacement averages S0 over 0--5039 only.  Both intervals happen to contain
  5,040 days, so the runner's own count check passes either way and cannot be
  relied on to catch the substitution; the binding is what makes it correct.
* **the normalization is this arm's own**, recomputed from 0--5039, not the
  shared seed-20260724 artifact.
* **the checkpoint identity check** verifies the split version and the loss-v1
  contract hash, proving the published model was trained under the chronological
  protocol and the group-balanced objective.

Held-evaluation package: no training, no checkpoint selection, no promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import af_model_c_bire_aligned_s0_figures as aligned_figures
from . import af_model_c_bire_s0_figures as frozen_figures
from .af_model_c import MODEL_C_LOSS_V1_CONTRACT_SHA256
from .af_model_c_bire_aligned_full_state import (
    BireAlignedArchitecture,
    BireAlignedStepper,
    build_bire_aligned_model,
)
from .af_model_c_chronological_split import TRAIN_RANGE
from .af_model_c_chronological_split import VERSION as SPLIT_VERSION
from .af_model_c_chronological_split import snapshot_codes as chronological_snapshot_codes

try:  # Keep documentation-only imports usable without the optional ML stack.
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

VERSION = "model_c_bire_aligned_chronological_s0_figures_v1"
CONTRACT_STATUS = (
    "frozen_after_the_chronological_training_and_validation_and_before_any_test_metric"
)
STAGE_NAMES = ("selected",)
DEFAULT_STAGE = "selected"


class ChronologicalS0FigureError(RuntimeError):
    """Raised when the chronological figure contract is violated."""


def selected_steps(contract_path: str | Path) -> tuple[int, ...]:
    contract = json.loads(Path(contract_path).read_text())
    if tuple(contract.get("stage_order", ())) != STAGE_NAMES:
        raise ChronologicalS0FigureError(
            "the chronological arm publishes exactly one selected checkpoint"
        )
    return (int(contract["stages"]["selected"]["model"]["optimizer_step"]),)


def _train_only_s0_climatology(
    state: Any,
    snapshot_codes: Any,
    wet: np.ndarray,
    *,
    chunk_days: int = 60,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """S0 climatology over the chronological training interval only.

    Signature-compatible with the frozen runner's helper.  The ``snapshot_codes``
    argument it passes is the *stored* split and is deliberately ignored.
    """

    replacement = chronological_snapshot_codes(len(np.asarray(snapshot_codes)))
    state_mean, derived_mean, days = _FROZEN_CLIMATOLOGY(
        state, replacement, wet, chunk_days=chunk_days
    )
    if days != TRAIN_RANGE[1] - TRAIN_RANGE[0]:
        raise ChronologicalS0FigureError(
            "the train-only S0 climatology did not cover 0--5039"
        )
    return state_mean, derived_mean, days


#: Captured at import, before any binding, so the replacement above always
#: delegates to the frozen implementation rather than to itself.
_FROZEN_CLIMATOLOGY = frozen_figures._s0_training_climatology


def _selected_stepper(
    contract: Mapping[str, Any],
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> BireAlignedStepper:
    """Build the chronological map and verify its split and objective."""

    checkpoint = Path(contract["artifacts"]["selected_checkpoint"]["path"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture_dict = contract["selected_model"]["architecture"]
    if (
        payload.get("architecture") != architecture_dict
        or int(payload.get("optimizer_step", -1))
        != int(contract["selected_model"]["optimizer_step"])
        or payload.get("split_version") != SPLIT_VERSION
        or payload.get("base_loss_contract_sha256") != MODEL_C_LOSS_V1_CONTRACT_SHA256
        or int(payload.get("rollout_steps", -1)) != 3
    ):
        raise ChronologicalS0FigureError(
            "selected checkpoint identity, split, or objective changed"
        )
    model = build_bire_aligned_model(BireAlignedArchitecture(**architecture_dict)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with np.load(Path(contract["artifacts"]["selected_normalization"]["path"])) as artifact:
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
    return f"""# Chronological-split loss-recovery model: S0 Figures 3--8

This package evaluates the seed-20260724, step-{step:,} checkpoint of the
**chronological-split** arm under the control wind (tau0 = 0.1 N m-2), selected
on a held 630-day validation block. All forecasts use a ten-day interval.

The model is identical to `model_c_bire_aligned_loss_recovery_v1` -- three FNO
blocks, six pointwise channel LayerNorms, modes 24x16, width 128, Bire
positional encoding, 10% padding, no external 3x3 branch, Model C loss v1 over a
three-step rollout, Adam 5e-4 decaying to 1e-4 at 75%, batch 8, 7,680 steps,
trained from scratch. What changed is the protocol:

| | stored split | this arm |
| --- | --- | --- |
| train | 0--2519 and 3690--6209 (interleaved) | **0--5039 (contiguous)** |
| validation | 6300--6569 | **5130--5759** |
| test | 2970--3599 and 6660--7199 | **5850--7199** |
| normalizer | shared seed-20260724 artifact | **recomputed from 0--5039** |
| checkpoint rule | 360-day gate over training records | **held validation block** |

**This is not a pure split-order ablation.** Both training sets hold 5,040 days
but only 3,870 overlap: 5040--6209 is exchanged for 2520--3689, changing 23.2%
of the training snapshots. The arm tests the chronological protocol *and*
sensitivity to which stretch of trajectory is used for training.

The climatology baseline in these figures is rebuilt from 0--5039. The frozen
runner derives it from the stored split-1 codes, which include 5850--6209 --
test days here -- so leaving it alone would have leaked 360 test days into the
baseline. Both intervals contain 5,040 days, so the runner's count check passes
either way and does not catch the substitution on its own.

The 15 initial conditions lie in 6660--7199, which is test under both splits, so
this package is directly comparable with
`bire_aligned_loss_recovery_bire_s0_inference_v1/selected`. Do **not** compare
the two models over the full new test interval 5850--7199: the parent trained on
5850--6209.

Persistence holds each member's initial physical field fixed. RMSE is first
computed over wet cells for each member; lines/bands are the mean and 10th/90th
percentiles across the 15 members.

Figure 6 compares the prior residual rollout-conditioned Model C with this arm.
It is an architecture-direction comparison, not a literal pretrained/fine-tuned
pairing.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `{report["report_content_sha256"]}`.
"""


ALIGNED_BINDINGS = (
    "VERSION",
    "CONTRACT_STATUS",
    "_readme",
    "CHECKPOINT_STEPS",
    "STAGE_NAMES",
    "_selected_stepper",
    "_ACTIVE_STAGE",
)


class _ChronologicalBinding:
    """Bind the single stage, the checkpoint check, and the train-only climatology."""

    def __init__(self, contract_path: str | Path) -> None:
        self.steps = selected_steps(contract_path)

    def __enter__(self) -> None:
        self._saved = {n: getattr(aligned_figures, n) for n in ALIGNED_BINDINGS}
        self._saved_climatology = frozen_figures._s0_training_climatology
        aligned_figures.VERSION = VERSION
        aligned_figures.CONTRACT_STATUS = CONTRACT_STATUS
        aligned_figures._readme = _readme
        aligned_figures.CHECKPOINT_STEPS = self.steps
        aligned_figures.STAGE_NAMES = STAGE_NAMES
        aligned_figures._selected_stepper = _selected_stepper
        aligned_figures._ACTIVE_STAGE = DEFAULT_STAGE
        frozen_figures._s0_training_climatology = _train_only_s0_climatology

    def __exit__(self, *exc: Any) -> None:
        for name, value in self._saved.items():
            setattr(aligned_figures, name, value)
        frozen_figures._s0_training_climatology = self._saved_climatology


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    with _ChronologicalBinding(path):
        return aligned_figures.load_contract(path, verify_sources=verify_sources)


def preflight(
    contract_path: str | Path,
    stage: str = DEFAULT_STAGE,
) -> dict[str, Any]:
    with _ChronologicalBinding(contract_path):
        result = dict(aligned_figures.preflight(contract_path, stage))
    result["split_version"] = SPLIT_VERSION
    result["climatology"] = "rebuilt_from_chronological_train_only_0_5039"
    result["objective"] = "incumbent_group_balanced_model_c_loss_v1"
    return result


def run(
    contract_path: str | Path,
    stage: str = DEFAULT_STAGE,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    with _ChronologicalBinding(contract_path):
        return aligned_figures.run(contract_path, stage, device_name=device_name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        child.add_argument("--stage", choices=(*STAGE_NAMES, "all"), default=DEFAULT_STAGE)
        if command == "run":
            child.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stages = STAGE_NAMES if args.stage == "all" else (args.stage,)
    if args.command == "preflight":
        result = {s: preflight(args.contract, s) for s in stages}
    else:
        result = {s: run(args.contract, s, device_name=args.device) for s in stages}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
