"""Matched-budget duration control for the Bire Section 3.2 protocol arm.

The Bire-protocol arm (:mod:`af_model_c_bire_protocol`) stopped at 7,680
optimizer steps with its training window still falling monotonically and with no
sign of flattening::

    step 1,920  lr 5e-4  total 0.334   spectral 20.70
    step 3,840  lr 5e-4  total 0.181   spectral  9.99
    step 5,760  lr 5e-4  total 0.144   spectral  8.25
    step 7,680  lr 1e-4  total 0.091   spectral  2.47

Validation selected the final checkpoint, which dominated every field and both
lead windows, so the optimum sat at the edge of the budget rather than inside
it.  Every Bire-aligned arm has used the same 7,680-step budget against the
incumbent residual model's 15,360, so no architecture comparison in this project
has yet been made at matched budget.

This arm changes **exactly one quantity**::

    maximum_steps:  7,680  ->  15,360

Everything else is frozen against ``model_c_bire_protocol_pooled_v1``: the same
architecture, the same Model C loss v1 over a three-step rollout, the same Bire
Section 3.2 split with training and validation pooled across S0/S1/S2, the same
train-only normalizers, the same seed 20260724, batch 8, Adam 5e-4, betas
(0.9, 0.95), zero weight decay, absent gradient clipping, and the same 0.75/0.2
decay schedule and selection rule.

Two consequences of holding ``decay_fraction`` rather than the decay step are
declared here, before any metric:

* The decay now falls at step 11,520 instead of 5,760, so this run's step-7,680
  checkpoint is **not** comparable to the parent's final checkpoint --- it is
  still at 5e-4 where the parent's had already dropped to 1e-4.  Preserving the
  schedule *shape* is the faithful reading of "the same experiment, longer";
  the alternative, pinning the decay to step 5,760, would instead run three
  quarters of the budget at the decayed rate and is not what was asked for.
* Checkpoints move to the quarters of the new budget, 3,840 / 7,680 / 11,520 /
  15,360, so the parent's four checkpoint steps are not reproduced.  Step 7,680
  is retained as a within-run landmark, not as a comparator.

Predeclared outcomes, so the result cannot be read after the fact:

1. **The budget was binding.**  Validation selects a step beyond 7,680 and the
   worst long-horizon ratio to climatology improves.  The 7,680-step family is
   then under-trained and every architecture comparison in this project needs
   re-running at matched budget.
2. **The budget was not binding.**  Validation selects 7,680 or the long ratio
   fails to improve.  Training loss and held skill have decoupled, which is the
   pattern this project has recorded four times, and duration is closed.
3. **The tail degrades.**  The long-horizon per-call gain or amplitude worsens
   while the training window keeps falling, in which case longer training buys
   short-lead skill at the cost of the bounded attractor.

The parent module's SHA-256 is certified by two completed contracts, so no
certified source is modified.  This arm supplies its own contract loader,
README, artifact names, and step constants, and binds them into the parent's
training loop exactly as the learning-rate control binds into its own parent.

Training and validation only.  The S0 figure suite opens through its own
contract, after this one completes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import af_model_c_bire_protocol as protocol
from .af_bire_protocol_split import (
    INFERENCE_RANGE,
    TRAIN_RANGE,
    VALIDATION_RANGE,
)
from .af_data_v3 import DATASET_VERSION, PRODUCTION_DAYS
from .af_model_c import MODEL_C_LOSS_V1_CONTRACT_SHA256
from .af_model_c_bire_aligned_full_state import (
    EXTERNAL_INPUT_CHANNELS,
    LIFTING_INPUT_CHANNELS,
    BireAlignedArchitecture,
)
from .af_model_c_overfit import _file_sha256

VERSION = "model_c_bire_protocol_duration_v1"
CONTRACT_STATUS = "frozen_before_any_bire_protocol_duration_metric"

#: The parent arm this control is derived from, one quantity at a time.
PARENT_VERSION = "model_c_bire_protocol_pooled_v1"
PARENT_MAXIMUM_STEPS = 7680

#: The single declared change.
MAXIMUM_STEPS = 15360
#: Quarters of the new budget; 7,680 is a within-run landmark, not a comparator.
CHECKPOINT_STEPS = (3840, 7680, 11520, 15360)

LEARNING_RATE = protocol.LEARNING_RATE
ROLLOUT_STEPS = protocol.ROLLOUT_STEPS

#: The one training field this arm is permitted to move.
DECLARED_CHANGE = "maximum_steps"

NORMALIZATION_NAME = "model_c_bire_protocol_duration_train_only_normalization.npz"
DIVERGENCE_NAME = "bire_protocol_duration_divergence.json"
CHECKPOINT_STEM = "model_c_bire_protocol_duration_step"
REPORT_NAME = "bire_protocol_duration_report.json"
ARRAYS_NAME = "bire_protocol_duration_arrays.npz"
FIGURE_NAME = "model_c_bire_protocol_duration_selection.png"


class BireProtocolDurationError(RuntimeError):
    """Raised when the duration control's single declared change is violated."""


def _assert_single_change(contract: Mapping[str, Any]) -> None:
    """Every frozen model quantity but the budget must equal the parent's."""

    record = contract["sources"]["parent_contract"]
    path = Path(record["path"]).resolve()
    if not path.is_file() or _file_sha256(path) != record["sha256"]:
        raise BireProtocolDurationError("the parent Bire-protocol contract changed")
    parent = json.loads(path.read_text())
    if parent.get("version") != PARENT_VERSION:
        raise BireProtocolDurationError("the parent arm is not the Bire-protocol arm")
    if contract["architecture"] != parent["architecture"]:
        raise BireProtocolDurationError("the duration control must keep the parent architecture")
    if contract["loss"] != parent["loss"]:
        raise BireProtocolDurationError("the duration control must keep the parent objective")
    if contract["dataset"] != parent["dataset"]:
        raise BireProtocolDurationError("the duration control must keep the parent split")
    moved = [
        field
        for field in protocol.FROZEN_TRAINING_FIELDS
        if contract["training"].get(field) != parent["training"].get(field)
    ]
    if moved != [DECLARED_CHANGE]:
        raise BireProtocolDurationError(
            f"the duration control must move exactly {DECLARED_CHANGE!r}, moved {moved!r}"
        )
    if int(parent["training"]["maximum_steps"]) != PARENT_MAXIMUM_STEPS:
        raise BireProtocolDurationError("the parent budget is not 7,680 steps")


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the duration contract frozen before any metric of this arm."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    architecture = contract.get("architecture", {})
    training = contract.get("training", {})
    dataset = contract.get("dataset", {})
    selection = contract.get("checkpoint_selection", {})
    read = contract.get("read_contract", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or dataset.get("version") != DATASET_VERSION
        or int(dataset.get("production_days", -1)) != PRODUCTION_DAYS
        or tuple(dataset.get("train", ())) != TRAIN_RANGE
        or tuple(dataset.get("validation", ())) != VALIDATION_RANGE
        or tuple(dataset.get("inference", ())) != INFERENCE_RANGE
        or dataset.get("pooled_regimes") is not True
        or int(architecture.get("in_channels", -1)) != EXTERNAL_INPUT_CHANNELS
        or int(architecture.get("lifting_in_channels", -1)) != LIFTING_INPUT_CHANNELS
        or int(architecture.get("n_layers", -1)) != 3
        or architecture.get("local_kernel_size") is not None
        or architecture.get("positional_embedding") is not None
        or int(training.get("seed", -1)) != 20260724
        or int(training.get("batch_size", -1)) != 8
        or float(training.get("initial_learning_rate", -1.0)) != LEARNING_RATE
        or int(training.get("maximum_steps", -1)) != MAXIMUM_STEPS
        or tuple(training.get("checkpoint_steps", ())) != CHECKPOINT_STEPS
        or int(training.get("rollout_steps", -1)) != ROLLOUT_STEPS
        or contract.get("loss", {}).get("contract_sha256") != MODEL_C_LOSS_V1_CONTRACT_SHA256
        or contract.get("normalization", {}).get("recomputed_from") != "bire_protocol_train_only_0_5999"
        or selection.get("rule") != "min_worst_long_climatology_ratio_subject_to_short_guard"
        or selection.get("evaluated_on") != "pooled_bire_protocol_validation_6000_6199"
        or read.get("training_state") is not True
        or read.get("validation_state") is not True
        or any(
            read.get(name) is not False
            for name in ("inference_state", "intermediate_wind_state", "response_state", "adjoint_state")
        )
    ):
        raise BireProtocolDurationError("Bire-protocol duration contract changed")
    BireAlignedArchitecture(**architecture)
    _assert_single_change(contract)
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise BireProtocolDurationError(f"duration source changed: {source}")
    return contract, resolved, _file_sha256(resolved)


def _readme(report: Mapping[str, Any]) -> str:
    decay = int(report["optimizer"]["decay_step"])
    rows = "\n".join(
        "| {step} | {short} | {long} | {acc} |".format(
            step=f"{int(s['optimizer_step']):,}",
            short="/".join(f"{s['short_ratio_to_persistence'][f]:.3f}" for f in protocol.PRIMARY_FIELDS),
            long="/".join(f"{s['long_ratio_to_climatology'][f]:.3f}" for f in protocol.PRIMARY_FIELDS),
            acc="/".join(f"{s['acc_day200'][f]:+.3f}" for f in ("surface_u", "sst", "phihyd_surface")),
        )
        for s in report["validation_summaries"]
    )
    return f"""# Matched-budget duration control for the Bire Section 3.2 protocol

Identical to `{PARENT_VERSION}` in every model quantity except one:

| quantity | parent | this arm |
| --- | --- | --- |
| `maximum_steps` | {PARENT_MAXIMUM_STEPS:,} | {MAXIMUM_STEPS:,} |
| checkpoint steps | 1,920 / 3,840 / 5,760 / 7,680 | {" / ".join(f"{s:,}" for s in CHECKPOINT_STEPS)} |
| decay step (0.75 x budget) | 5,760 | {decay:,} |

Architecture, Model C loss v1 over a three-step rollout, the Bire Section 3.2
split (train 0--5999, validation 6000--7199, inference 6200--7199 nested, no
buffers, pooled S0+S1+S2), the train-only normalizers, seed 20260724, batch 8,
Adam 5e-4 with betas (0.9, 0.95), zero weight decay, absent gradient clipping,
the 0.2 decay factor, and the selection rule are all unchanged and asserted
field by field against the parent contract.

**This run's step-7,680 checkpoint is not the parent's.** Holding
`decay_fraction` at 0.75 moves the decay from step 5,760 to {decay:,}, so at step
7,680 this run is still at 5e-4 while the parent had already dropped to 1e-4.
The schedule *shape* is preserved, which is the faithful reading of the same
experiment run longer; step 7,680 is a within-run landmark, not a comparator.

Selection is unchanged: minimise the worst 90--360-day RMSE-AUC relative to
climatology subject to each field's 10--90-day AUC staying within 5% of the best
checkpoint, on 102 pooled validation rollouts from starts outside the inference
set.

| step | short AUC / persistence | long AUC / climatology | day-200 ACC (U/SST/P) |
| --- | --- | --- | --- |
{rows}

Selected step {int(report['selection_decision']['selected_optimizer_step']):,} via `{report['selection_decision']['branch']}`.

Training and validation only; the inference set opens through the figure
contract, S0 only.

Report content SHA-256: `{report['content_sha256']}`.
"""


#: Globals the parent's ``preflight`` and ``run`` resolve at call time.
PARENT_BINDINGS = (
    "VERSION",
    "CHECKPOINT_STEM",
    "CONTRACT_STATUS",
    "CHECKPOINT_STEPS",
    "MAXIMUM_STEPS",
    "NORMALIZATION_NAME",
    "DIVERGENCE_NAME",
    "REPORT_NAME",
    "ARRAYS_NAME",
    "FIGURE_NAME",
    "load_contract",
    "_readme",
)


class _ParentBinding:
    """Bind this control's constants and loader into the parent module.

    The parent resolves each of these as a module global at call time, so
    rebinding them reuses the whole training loop, checkpoint writer, validation
    pass, selection rule, and artifact package without editing a source file
    whose SHA-256 completed contracts certify.
    """

    def __enter__(self) -> "_ParentBinding":
        here = globals()
        self._saved = {name: getattr(protocol, name) for name in PARENT_BINDINGS}
        for name in PARENT_BINDINGS:
            setattr(protocol, name, here[name])
        return self

    def __exit__(self, *exc: Any) -> None:
        for name, value in self._saved.items():
            setattr(protocol, name, value)


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify sources, the split, and the single declared change without training."""

    with _ParentBinding():
        result = dict(protocol.preflight(contract_path))
    result["maximum_steps"] = MAXIMUM_STEPS
    result["parent_maximum_steps"] = PARENT_MAXIMUM_STEPS
    result["checkpoint_steps"] = list(CHECKPOINT_STEPS)
    result["single_declared_change"] = DECLARED_CHANGE
    return result


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    """Train the matched-budget control and select on the pooled validation block."""

    with _ParentBinding():
        result = dict(protocol.run(contract_path, device_name=device_name))
    result["single_declared_change"] = DECLARED_CHANGE
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("preflight")
    check.add_argument("--contract", required=True)
    train = subparsers.add_parser("run")
    train.add_argument("--contract", required=True)
    train.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = preflight(args.contract) if args.command == "preflight" else run(
        args.contract, device_name=args.device
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
