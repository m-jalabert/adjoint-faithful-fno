"""S0 Bire-style Figure 3--8 suite for the single-position LayerNorm arm.

This reproduces the exact figure package published under
``anomaly_direct_bire_s0_inference_v1`` -- the same 15 fixed S0 inference
starts, the same day-2000 continuous truth, the same persistence and split-1
climatology baselines, the same metric reductions, and the same six filenames --
for the arm trained by
:mod:`af_model_c_anomaly_direct_single_position_layernorm`.

Two properties of that arm prevent the frozen figure runner from being reused
directly:

* its checkpoint is optimizer step 14,880, while the frozen runner hard-codes
  step 13,440 for the selected model, and
* its map carries the Bire pointwise LayerNorm and disables the appended grid
  embedding, so the strict successor builder rejects it.

Neither can be reached from a contract, so this module replaces the selected
stepper and reuses everything else in the frozen runner by import.  The prior
Figure-6 comparator is unchanged: it remains the rollout-conditioned residual
Model C v3, and it is built by the frozen runner's own function through a thin
contract view that points it at its own architecture record.

No certified source file is modified.  This is a held-evaluation package: it
performs no training, no checkpoint selection, and promotes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import af_model_c_bire_s0_figures as figures
from .af_model_c_anomaly_direct import PointwiseDirectStepper
from .af_model_c_anomaly_direct_bire_regularization_controls import (
    RegularizationArm,
    build_regularized_model,
)
from .af_model_c_anomaly_direct_single_position_layernorm import (
    SinglePositionArchitecture,
)
from .af_model_c_overfit import _file_sha256

try:  # Keep documentation-only imports usable without the optional ML stack.
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

VERSION = "model_c_single_position_layernorm_s0_figures_v1"
CONTRACT_STATUS = "frozen_after_the_single_position_training_gate_and_before_any_figure_metric"
SELECTED_STEP = 14880
ARM = ("layernorm", True, 0.0)


class SinglePositionS0FigureError(RuntimeError):
    """Raised when the single-position S0 figure contract is violated."""


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the figure contract frozen before any single-position figure metric.

    The protocol clauses are deliberately identical to the frozen runner's, so
    the published package is comparable field for field with
    ``anomaly_direct_bire_s0_inference_v1``.
    """

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
    ):
        raise SinglePositionS0FigureError("S0 figure contract is not frozen")
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
        raise SinglePositionS0FigureError("S0 figure protocol changed")
    selected = contract["selected_model"]
    if (
        int(selected.get("optimizer_step", -1)) != SELECTED_STEP
        or selected["architecture"].get("positional_embedding") is not None
        or bool(selected.get("pointwise_layer_norm")) is not True
    ):
        raise SinglePositionS0FigureError("selected single-position model changed")
    if contract["prior_model"]["architecture"].get("positional_embedding") != "grid":
        raise SinglePositionS0FigureError("prior comparator architecture changed")
    if contract["figure6"].get("literal_pretrain_finetune_pair") is not False:
        raise SinglePositionS0FigureError("Figure 6 comparison is mislabeled")
    if verify_sources:
        for label, specification in contract["artifacts"].items():
            figures._verify_file(specification, label)
        root = resolved.parents[1]
        for relative, expected in contract["source_hashes"].items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise SinglePositionS0FigureError(f"source changed: {relative}")
    return contract, resolved, _file_sha256(resolved)


def _selected_stepper(
    contract: Mapping[str, Any],
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> PointwiseDirectStepper:
    """Build the single-position LayerNorm map from its own checkpoint record."""

    checkpoint = Path(contract["artifacts"]["selected_checkpoint"]["path"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture_dict = contract["selected_model"]["architecture"]
    if (
        payload.get("architecture") != architecture_dict
        or int(payload.get("optimizer_step", -1)) != SELECTED_STEP
    ):
        raise SinglePositionS0FigureError("selected checkpoint identity changed")
    architecture = SinglePositionArchitecture(**architecture_dict)
    model = build_regularized_model(architecture, RegularizationArm(*ARM)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    normalization_path = Path(
        contract["artifacts"]["selected_normalization"]["path"]
    )
    with np.load(normalization_path) as artifact:
        mean = np.asarray(artifact["pointwise_mean"], dtype=np.float32)
        scale = np.asarray(artifact["pointwise_scale"], dtype=np.float32)
    return PointwiseDirectStepper(
        model=model,
        device=device,
        wet=wet,
        mean=mean,
        scale=scale,
        wind_mean=wind_mean,
        wind_scale=wind_scale,
    )


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


_FROZEN_PRIOR_STEPPER = figures._prior_stepper


def _readme(report: Mapping[str, Any]) -> str:
    """Describe *this* arm.

    The frozen runner's README hard-codes ``step-13440`` and the selected
    anomaly-direct model, which would misdescribe this package.
    """

    return f"""# Single-position LayerNorm Model C: S0 Bire-style Figures 3--8

This package evaluates the seed-20260724, step-14880 single-position LayerNorm
pointwise-anomaly direct-state Model C under the control wind
(tau0 = 0.1 N m-2). All forecasts use a ten-day prediction interval.

The arm retains the full 46-channel state and changes exactly two things from
the retained selected Model C: the duplicated positional encoding is removed
(position is supplied once, through the two static data channels, and the
appended grid embedding is disabled) and the Bire pointwise channel LayerNorm
is applied after spectral and Channel-MLP mixing.

The 15 initial conditions, the day-2000 MITgcm truth from evaluation-only job
304735, the climatology and persistence baselines, the metric reductions, and
the six figure filenames are identical to
`anomaly_direct_bire_s0_inference_v1`, so the two packages are directly
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

    def __enter__(self) -> None:
        self._saved = {}
        for name in self._NAMES:
            if not hasattr(figures, name):
                raise SinglePositionS0FigureError(
                    f"{figures.__name__} no longer defines {name}"
                )
            self._saved[name] = getattr(figures, name)
        figures.load_contract = load_contract
        figures._selected_stepper = _selected_stepper
        figures._prior_stepper = _prior_stepper
        figures._readme = _readme
        figures.VERSION = VERSION

    def __exit__(self, *exc: Any) -> None:
        for name, value in self._saved.items():
            setattr(figures, name, value)


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify sources and the selected checkpoint identity without plotting."""

    with _FigureBinding():
        result = dict(figures.preflight(contract_path))
    contract, _, _ = load_contract(contract_path)
    result["selected_optimizer_step"] = SELECTED_STEP
    result["positional_embedding"] = contract["selected_model"]["architecture"][
        "positional_embedding"
    ]
    result["pointwise_layer_norm"] = True
    return result


def run(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Publish the six S0 figures for the single-position LayerNorm arm."""

    with _FigureBinding():
        return figures.evaluate(contract_path, device_name=device_name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        if command == "run":
            child.add_argument(
                "--device",
                choices=("auto", "cpu", "cuda"),
                default="auto",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.contract)
    else:
        result = run(args.contract, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
