"""Learning-rate control for the Bire-aligned full-state FNO.

The Bire-aligned package (:mod:`af_model_c_bire_aligned_full_state`) trained
without diverging but produced no forecast skill: its one-step normalized MSE
settled at the zero-anomaly value of 1.0, its day-200 ACC was +0.06 to +0.11
against the incumbent's +0.84 to +0.86, and both stage checkpoints emitted
essentially the climatological field.  A controlled 600-step comparison on the
identical code path, data, and seed separated the two candidate explanations::

    Adam lr 1e-2  ->  one-step MSE 1.03 -> 0.94   (stuck at the trivial solution)
    Adam lr 5e-4  ->  one-step MSE 0.17 -> 0.034  (still falling; persistence 0.0185)

so the collapse is attributable to the declared learning rate rather than to the
three-block architecture, the positional encoder, the removed local branch, or
the ``MSE + 0.01 MAE`` objective.

This arm changes **exactly one quantity**::

    initial_learning_rate:  1e-2  ->  5e-4

Everything else is frozen bit for bit against
``model_c_bire_aligned_full_state_v1``: the same architecture dataclass, the
same 49 external inputs, the same Bire position fields, the same six pointwise
LayerNorms, the same wet-cell objective, the same 3,840 + 3,840 two-stage
protocol, the same seed, batch size 8, betas (0.9, 0.95), zero weight decay,
absent gradient clipping, and the same 0.75/0.2 decay schedule.  Separating the
Bire *architecture* from the Bire *optimizer settings* is the whole point, so no
second knob may move.

The Bire-aligned module's SHA-256 is certified by two completed contracts, and
re-recording it would retroactively falsify the provenance of the results that
module already produced.  This arm therefore supplies its own contract loader
and README and reuses the parent's architecture, model builder, training loop,
and gate instrument by binding, exactly as the single-position LayerNorm arm
binds into its own parent.  No certified source file is modified.

Training split only.  The held S0 figure suite opens through its own contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import af_model_c_bire_aligned_full_state as aligned
from .af_model_c_bire_aligned_full_state import (
    CHECKPOINT_STEPS,
    EXTERNAL_INPUT_CHANNELS,
    LIFTING_INPUT_CHANNELS,
    MAE_WEIGHT,
    MAXIMUM_STEPS,
    BireAlignedArchitecture,
    BireAlignedFullStateError,
)
from .af_model_c_overfit import _file_sha256
from .af_model_c_successor import STATE_CHANNEL_COUNT

VERSION = "model_c_bire_aligned_full_state_lr5e4_v1"
CONTRACT_STATUS = "frozen_before_any_bire_aligned_lr_control_metric"
CONTROL_LEARNING_RATE = 5.0e-4
PARENT_LEARNING_RATE = 1.0e-2
#: Held as a literal rather than read from ``aligned.VERSION``: the loader runs
#: inside :class:`_ParentBinding`, where that global is this arm's own version.
PARENT_VERSION = "model_c_bire_aligned_full_state_v1"

#: Every training field that must remain identical to the parent arm.  The
#: learning rate is deliberately absent: it is the one declared change.
FROZEN_TRAINING_FIELDS = (
    "seed",
    "optimizer",
    "batch_size",
    "adam_betas",
    "weight_decay",
    "gradient_clipping",
    "maximum_steps",
    "checkpoint_steps",
    "decay_fraction",
    "decay_factor",
)


class BireAlignedLearningRateControlError(BireAlignedFullStateError):
    """Raised when the learning-rate control moves more than one quantity."""


def _parent_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Load and verify the parent arm this control is differenced against."""

    record = contract["sources"]["parent_contract"]
    path = Path(record["path"]).resolve()
    if not path.is_file() or _file_sha256(path) != record["sha256"]:
        raise BireAlignedLearningRateControlError(
            "the parent Bire-aligned contract changed"
        )
    parent = json.loads(path.read_text())
    if (
        parent.get("version") != PARENT_VERSION
        or float(parent["training"]["initial_learning_rate"])
        != PARENT_LEARNING_RATE
    ):
        raise BireAlignedLearningRateControlError(
            "the parent Bire-aligned arm is not the 1e-2 run"
        )
    return parent


def _assert_single_change(
    contract: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> None:
    """Fail unless the learning rate is the *only* difference from the parent."""

    if contract["architecture"] != parent["architecture"]:
        raise BireAlignedLearningRateControlError(
            "the learning-rate control must keep the parent architecture exactly"
        )
    if contract["loss"] != parent["loss"]:
        raise BireAlignedLearningRateControlError(
            "the learning-rate control must keep the parent objective exactly"
        )
    if contract["stages"] != parent["stages"]:
        raise BireAlignedLearningRateControlError(
            "the learning-rate control must keep the parent two-stage protocol"
        )
    if contract["selection"] != parent["selection"]:
        raise BireAlignedLearningRateControlError(
            "the learning-rate control must keep the parent gate instrument"
        )
    training = contract["training"]
    parent_training = parent["training"]
    for field in FROZEN_TRAINING_FIELDS:
        if training.get(field) != parent_training.get(field):
            raise BireAlignedLearningRateControlError(
                f"the learning-rate control moved a second quantity: {field}"
            )
    if float(training["initial_learning_rate"]) != CONTROL_LEARNING_RATE:
        raise BireAlignedLearningRateControlError(
            "the learning-rate control is declared at 5e-4"
        )


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the contract frozen before any learning-rate control metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    architecture = contract.get("architecture", {})
    training = contract.get("training", {})
    loss = contract.get("loss", {})
    read = contract.get("read_contract", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or int(architecture.get("in_channels", -1)) != EXTERNAL_INPUT_CHANNELS
        or int(architecture.get("lifting_in_channels", -1))
        != LIFTING_INPUT_CHANNELS
        or int(architecture.get("out_channels", -1)) != STATE_CHANNEL_COUNT
        or int(architecture.get("n_layers", -1)) != 3
        or architecture.get("local_kernel_size") is not None
        or architecture.get("positional_embedding") is not None
        or int(training.get("seed", -1)) != 20260724
        or int(training.get("batch_size", -1)) != 8
        or float(training.get("initial_learning_rate", -1.0))
        != CONTROL_LEARNING_RATE
        or tuple(float(value) for value in training.get("adam_betas", ()))
        != (0.9, 0.95)
        or float(training.get("weight_decay", -1.0)) != 0.0
        or training.get("optimizer") != "adam"
        or training.get("gradient_clipping") is not False
        or int(training.get("maximum_steps", -1)) != MAXIMUM_STEPS
        or tuple(training.get("checkpoint_steps", ())) != CHECKPOINT_STEPS
        or loss.get("objective") != "wet_cell_mse_plus_0p01_mae"
        or float(loss.get("mae_weight", -1.0)) != MAE_WEIGHT
        or read.get("training_state") is not True
        or any(
            read.get(name) is not False
            for name in (
                "validation_state",
                "inference_state",
                "held_s0_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
            )
        )
    ):
        raise BireAlignedLearningRateControlError(
            "Bire-aligned learning-rate control contract changed"
        )
    aligned.arm_stages(contract)
    BireAlignedArchitecture(**architecture)
    _assert_single_change(contract, _parent_contract(contract))
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise BireAlignedLearningRateControlError(
                    f"Bire-aligned learning-rate control source changed: {source}"
                )
    return contract, resolved, _file_sha256(resolved)


def _readme(report: Mapping[str, Any]) -> str:
    """Describe the control, not the parent arm whose README hard-codes 1e-2."""

    lines = [
        "# Bire-aligned full-state FNO, learning-rate control (training split only)",
        "",
        "One-factor control against `model_c_bire_aligned_full_state_v1`. The",
        "single declared change is the optimizer learning rate:",
        "",
        "| | parent arm | this control |",
        "| --- | --- | --- |",
        "| initial learning rate | 1e-2 | **5e-4** |",
        "| everything else | — | identical |",
        "",
        "Architecture, 49 external inputs, Bire sine/cosine position fields, six",
        "pointwise LayerNorms, absent external 3x3 branch, wet-cell",
        "`MSE + 0.01 MAE`, the 3,840 + 3,840 two-stage protocol, seed, batch size",
        "8, betas (0.9, 0.95), zero weight decay, absent gradient clipping, and",
        "the 0.75/0.2 decay schedule are all frozen against the parent.",
        "",
        "The parent arm collapsed to climatology: its one-step normalized MSE",
        "settled at the zero-anomaly value of 1.0 and its day-200 ACC was +0.06",
        "to +0.11. This control tests whether that collapse was caused by the",
        "learning rate rather than by the Bire architecture package.",
        "",
        "Gate instrument (unchanged 360-day split-1 spectral/primary summary):",
        "",
        "| stage | step | worst 10--90-day RMSE ratio | worst mid/bottom modewise | gate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for value in report["evaluation_summaries"]:
        lines.append(
            "| {stage} | {step} | {primary:.4f} | {spectral:.3f} | {gate} |".format(
                stage=value["stage_id"],
                step=value["optimizer_step"],
                primary=value["worst_primary_10_to_90_ratio"],
                spectral=value["worst_mid_bottom_modewise_ratio_all_leads"],
                gate=value["gate"]["pass"],
            )
        )
    lines += [
        "",
        "Training split only; validation, inference, held S0, response, and",
        "adjoint archives remained sealed for this run.",
        "",
        f"Report content SHA-256: `{report['content_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


PARENT_BINDINGS = ("load_contract", "_readme", "VERSION")


class _ParentBinding:
    """Bind this control's loader, README, and version into the parent module.

    The parent's ``preflight`` and ``run`` resolve ``load_contract``,
    ``_readme``, and ``VERSION`` as module globals at call time, so rebinding the
    three reuses the whole training loop, checkpoint writer, gate evaluation, and
    artifact package without editing a source file whose SHA-256 two completed
    contracts certify.
    """

    def __enter__(self) -> None:
        self._saved = {name: getattr(aligned, name) for name in PARENT_BINDINGS}
        aligned.load_contract = load_contract
        aligned._readme = _readme
        aligned.VERSION = VERSION

    def __exit__(self, *exc: Any) -> None:
        for name, value in self._saved.items():
            setattr(aligned, name, value)


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify sources and the single declared change without training."""

    with _ParentBinding():
        result = dict(aligned.preflight(contract_path))
    contract, _, _ = load_contract(contract_path)
    result["initial_learning_rate"] = CONTROL_LEARNING_RATE
    result["parent_initial_learning_rate"] = PARENT_LEARNING_RATE
    result["single_declared_change"] = "initial_learning_rate"
    return result


def run(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train and evaluate the learning-rate control on the training split."""

    with _ParentBinding():
        return aligned.run(contract_path, device_name=device_name)


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
