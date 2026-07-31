"""S0 Bire-style Figure 3--8 suite for the Bire-aligned full-state FNO.

This publishes the exact figure package already published under
``single_position_layernorm_bire_s0_inference_v1`` -- the same 15 fixed S0
inference starts, the same day-2000 continuous truth, the same persistence and
split-1 climatology baselines, the same metric reductions, and the same six
filenames -- for the arm trained by :mod:`af_model_c_bire_aligned_full_state`.

Bire's protocol produces two models, so this runner publishes two complete
figure folders, one per stage:

* ``pretrained``  -- optimizer step 3,840, one-step objective only;
* ``finetuned``   -- optimizer step 7,680, after two-step autoregressive
  fine-tuning.

Selecting a stage rewrites only ``selected_model``, ``artifacts``, and
``output`` in the contract view handed to the frozen runner; the protocol,
truth, baselines, reductions, and the Figure-6 comparator are identical across
stages and identical to the published incumbent package.

Two properties of this arm prevent the frozen runner from being reused
directly: its map takes 49 external channels and appends its own position
fields, so the strict successor builder and the five-channel static block both
reject it; and its checkpoints sit at steps the frozen runner does not know.
Both are handled by binding, so no certified source file is modified.

This is a held-evaluation package: it performs no training, no checkpoint
selection, and promotes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import af_model_c_bire_s0_figures as figures
from .af_model_c_bire_aligned_full_state import (
    CHECKPOINT_STEPS,
    EXTERNAL_INPUT_CHANNELS,
    LIFTING_INPUT_CHANNELS,
    STAGE_NAMES,
    BireAlignedArchitecture,
    BireAlignedStepper,
    build_bire_aligned_model,
)
from .af_model_c_overfit import _file_sha256

try:  # Keep documentation-only imports usable without the optional ML stack.
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

VERSION = "model_c_bire_aligned_full_state_s0_figures_v1"
CONTRACT_STATUS = (
    "frozen_after_the_bire_aligned_training_run_and_before_any_figure_metric"
)
DEFAULT_STAGE = "finetuned"

_ACTIVE_STAGE = DEFAULT_STAGE


class BireAlignedS0FigureError(RuntimeError):
    """Raised when the Bire-aligned S0 figure contract is violated."""


def active_stage() -> str:
    """Return the stage the bound runner is currently publishing."""

    return _ACTIVE_STAGE


def stage_view(contract: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Return the contract as the frozen runner expects it for one stage."""

    if stage not in STAGE_NAMES:
        raise BireAlignedS0FigureError(f"unknown Bire-aligned stage: {stage}")
    record = contract["stages"][stage]
    view = dict(contract)
    view["artifacts"] = {
        **contract["artifacts"],
        "selected_checkpoint": record["checkpoint"],
    }
    view["selected_model"] = record["model"]
    view["output"] = record["output"]
    view["active_stage"] = stage
    return view


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the figure contract and return the active stage's view.

    The protocol clauses are deliberately identical to the frozen runner's, so
    each published package is comparable field for field with
    ``single_position_layernorm_bire_s0_inference_v1``.
    """

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
    ):
        raise BireAlignedS0FigureError("S0 figure contract is not frozen")
    protocol = contract["protocol"]
    if (
        float(protocol.get("tau0_n_m2", -1.0)) != 0.1
        or int(protocol.get("prediction_interval_days", -1)) != 10
        or int(protocol.get("maximum_lead_days", -1)) != 2000
        or int(protocol.get("member_count", -1)) != 15
        or tuple(protocol.get("start_draw_order", ())) != figures.EXPECTED_STARTS
        or int(protocol.get("single_member_start", -1)) != figures.EXPECTED_STARTS[0]
        or tuple(protocol.get("figure_names", ())) != figures.FIGURE_NAMES
        or tuple(protocol.get("rmse_fields", ())) != figures.RMSE_FIELDS
        or tuple(protocol.get("acc_fields", ())) != figures.ACC_FIELDS
    ):
        raise BireAlignedS0FigureError("S0 figure protocol changed")
    stages = contract["stages"]
    # The contract is stored as canonical sorted JSON, so key order carries no
    # meaning; the protocol order is declared explicitly instead.
    if set(stages) != set(STAGE_NAMES) or tuple(
        contract.get("stage_order", ())
    ) != STAGE_NAMES:
        raise BireAlignedS0FigureError("the two Bire-aligned stages changed")
    for stage, expected_step in zip(STAGE_NAMES, CHECKPOINT_STEPS):
        record = stages[stage]
        architecture = record["model"]["architecture"]
        if (
            int(record["model"].get("optimizer_step", -1)) != expected_step
            or int(architecture.get("in_channels", -1)) != EXTERNAL_INPUT_CHANNELS
            or int(architecture.get("lifting_in_channels", -1))
            != LIFTING_INPUT_CHANNELS
            or int(architecture.get("n_layers", -1)) != 3
            or architecture.get("local_kernel_size") is not None
            or architecture.get("positional_embedding") is not None
        ):
            raise BireAlignedS0FigureError(
                f"the declared {stage} Bire-aligned model changed"
            )
        for key in ("project", "scratch"):
            if not str(record["output"][key]).endswith(stage):
                raise BireAlignedS0FigureError(
                    f"the {stage} output folder must carry its stage name"
                )
    if contract["prior_model"]["architecture"].get("positional_embedding") != "grid":
        raise BireAlignedS0FigureError("prior comparator architecture changed")
    if contract["figure6"].get("literal_pretrain_finetune_pair") is not False:
        raise BireAlignedS0FigureError("Figure 6 comparison is mislabeled")
    view = stage_view(contract, _ACTIVE_STAGE)
    if verify_sources:
        for label, specification in view["artifacts"].items():
            figures._verify_file(specification, label)
        root = resolved.parents[1]
        for relative, expected in contract["source_hashes"].items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise BireAlignedS0FigureError(f"source changed: {relative}")
    return view, resolved, _file_sha256(resolved)


def _selected_stepper(
    contract: Mapping[str, Any],
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> BireAlignedStepper:
    """Build the Bire-aligned map for the active stage's checkpoint."""

    checkpoint = Path(contract["artifacts"]["selected_checkpoint"]["path"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture_dict = contract["selected_model"]["architecture"]
    expected_step = int(contract["selected_model"]["optimizer_step"])
    if (
        payload.get("architecture") != architecture_dict
        or int(payload.get("optimizer_step", -1)) != expected_step
        or str(payload.get("stage_id")) != str(contract["active_stage"])
    ):
        raise BireAlignedS0FigureError("selected checkpoint identity changed")
    architecture = BireAlignedArchitecture(**architecture_dict)
    model = build_bire_aligned_model(architecture).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    normalization_path = Path(
        contract["artifacts"]["selected_normalization"]["path"]
    )
    with np.load(normalization_path) as artifact:
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


_FROZEN_PRIOR_STEPPER = figures._prior_stepper


def _prior_stepper(contract: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
    """Delegate to the frozen prior stepper with its own architecture record.

    The frozen function reads ``selected_model.architecture`` for the prior
    checkpoint because both models shared an architecture in the original
    contract.  Here they do not, so it is given a contract view whose
    ``selected_model`` is the declared ``prior_model``.
    """

    view = dict(contract)
    view["selected_model"] = contract["prior_model"]
    return _FROZEN_PRIOR_STEPPER(view, *args, **kwargs)


_STAGE_TEXT = {
    "pretrained": (
        "one-step pretrained",
        "3,840 one-step updates under `MSE + 0.01 MAE`",
    ),
    "finetuned": (
        "two-step fine-tuned",
        "3,840 one-step pretraining updates followed by 3,840 two-step "
        "autoregressive fine-tuning updates under `MSE + 0.01 MAE`",
    ),
}


def _readme(report: Mapping[str, Any]) -> str:
    """Describe the active stage of *this* arm."""

    stage = _ACTIVE_STAGE
    label, protocol = _STAGE_TEXT[stage]
    step = CHECKPOINT_STEPS[STAGE_NAMES.index(stage)]
    return f"""# Bire-aligned full-state FNO ({label}): S0 Bire-style Figures 3--8

This package evaluates the seed-20260724, step-{step:,} **{stage}**
Bire-aligned full-state Model C under the control wind (tau0 = 0.1 N m-2). All
forecasts use a ten-day prediction interval.

The arm keeps the MITgcm trajectories, the 62x62 grid, the 24x16 retained
modes, the 46-channel state, the ten-day map, and the pointwise anomaly
normalization, and replaces the remaining project-specific architecture and
training choices with Bire-like ones: three FNO blocks with six pointwise
channel LayerNorms, position supplied exactly once by the deterministic
`oceanfourcast.PosEmbed` sine/cosine fields appended immediately before
lifting, no external 3x3 raw-input branch, and Adam(1e-2, betas 0.9/0.95, no
weight decay) at batch size 8. Training was {protocol}.

The 15 initial conditions, the day-2000 MITgcm truth from evaluation-only job
304735, the climatology and persistence baselines, the metric reductions, and
the six figure filenames are identical to
`single_position_layernorm_bire_s0_inference_v1`, so the packages are directly
comparable field for field.

Climatology is the pointwise temporal mean over S0 split-1 training snapshots.
Persistence holds each member's initial physical field fixed. RMSE is first
computed over wet cells for each member; lines/bands are the mean and
10th/90th percentiles across the 15 members.

Figure 6 compares the prior residual rollout-conditioned Model C with this
arm. It is an architecture-direction comparison, not a literal
pretrained/fine-tuned pairing.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `{report["report_content_sha256"]}`.
"""


class _FigureBinding:
    """Bind this arm's loader, steppers, README, and version into the runner."""

    _NAMES = (
        "load_contract",
        "_selected_stepper",
        "_prior_stepper",
        "_readme",
        "VERSION",
    )

    def __init__(self, stage: str = DEFAULT_STAGE) -> None:
        if stage not in STAGE_NAMES:
            raise BireAlignedS0FigureError(f"unknown Bire-aligned stage: {stage}")
        self.stage = stage

    def __enter__(self) -> None:
        global _ACTIVE_STAGE

        self._saved = {}
        for name in self._NAMES:
            if not hasattr(figures, name):
                raise BireAlignedS0FigureError(
                    f"{figures.__name__} no longer defines {name}"
                )
            self._saved[name] = getattr(figures, name)
        self._previous_stage = _ACTIVE_STAGE
        _ACTIVE_STAGE = self.stage
        figures.load_contract = load_contract
        figures._selected_stepper = _selected_stepper
        figures._prior_stepper = _prior_stepper
        figures._readme = _readme
        figures.VERSION = VERSION

    def __exit__(self, *exc: Any) -> None:
        global _ACTIVE_STAGE

        _ACTIVE_STAGE = self._previous_stage
        for name, value in self._saved.items():
            setattr(figures, name, value)


def preflight(
    contract_path: str | Path,
    stage: str = DEFAULT_STAGE,
) -> dict[str, Any]:
    """Verify sources and both checkpoint identities without plotting."""

    with _FigureBinding(stage):
        result = dict(figures.preflight(contract_path))
        contract, _, _ = load_contract(contract_path)
    result["stage"] = stage
    result["selected_optimizer_step"] = int(
        contract["selected_model"]["optimizer_step"]
    )
    result["external_input_channels"] = EXTERNAL_INPUT_CHANNELS
    result["lifting_input_channels"] = LIFTING_INPUT_CHANNELS
    result["fno_blocks"] = 3
    result["external_local_branch"] = False
    result["project_output"] = contract["output"]["project"]
    return result


def run(
    contract_path: str | Path,
    stage: str = DEFAULT_STAGE,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Publish the six S0 figures for one Bire-aligned stage."""

    with _FigureBinding(stage):
        report = figures.evaluate(contract_path, device_name=device_name)
    report = dict(report)
    report["stage"] = stage
    return report


def run_all(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Publish both stages' figure packages in the declared order."""

    return {
        stage: run(contract_path, stage, device_name=device_name)
        for stage in STAGE_NAMES
    }


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
