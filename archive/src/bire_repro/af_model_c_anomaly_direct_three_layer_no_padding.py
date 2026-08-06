"""Training-only three-layer/no-padding Bire architecture control."""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import af_model_c_anomaly_direct_bire_regularization_controls as base
from .af_model_c_overfit import _file_sha256


VERSION = "model_c_anomaly_direct_three_layer_no_padding_v1"
CONTROL_ID = "three_layer_no_padding"
CHECKPOINT_STEPS = (3840, 7680, 11520, 13440, 14400, 14880, 15360)
REPORT_NAME = "three_layer_no_padding_report.json"
ARRAYS_NAME = "three_layer_no_padding_arrays.npz"
FIGURE_NAME = "model_c_three_layer_no_padding_selection.png"
README_NAME = "README.md"
MANIFEST_NAME = "manifest.json"
CHECKPOINT_DIRECTORY = "training_checkpoints"
BEST_NAME = "model_c_three_layer_no_padding_best_diagnostic.pt"


class ThreeLayerNoPaddingError(RuntimeError):
    """Raised when the frozen architecture control changes."""


@dataclass(frozen=True)
class ArchitectureControl:
    """Plain three-block FNO with no domain padding or extra regularizer."""

    arm_id: str = CONTROL_ID
    pointwise_layer_norm: bool = False
    channel_mlp_dropout: float = 0.0


@dataclass(frozen=True)
class ThreeLayerNoPaddingArchitecture:
    """The existing dense Model C architecture with only two Bire changes."""

    in_channels: int = 51
    out_channels: int = 46
    n_modes: tuple[int, int] = (24, 16)
    hidden_channels: int = 128
    n_layers: int = 3
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    channel_mlp_expansion: float = 4.0
    domain_padding: float | None = None
    positional_embedding: str = "grid"
    use_channel_mlp: bool = True
    local_kernel_size: int = 3
    fno_block_precision: str = "full"
    factorization: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "n_modes", tuple(int(value) for value in self.n_modes)
        )
        if (
            self.in_channels != 51
            or self.out_channels != 46
            or self.n_modes != (24, 16)
            or self.hidden_channels != 128
            or self.n_layers != 3
            or self.lifting_channel_ratio != 2
            or self.projection_channel_ratio != 2
            or self.channel_mlp_expansion != 4.0
            or self.domain_padding is not None
            or self.positional_embedding != "grid"
            or not self.use_channel_mlp
            or self.local_kernel_size != 3
            or self.fno_block_precision != "full"
            or self.factorization is not None
        ):
            raise ValueError("three-layer/no-padding architecture changed")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_modes"] = list(self.n_modes)
        return result


def architecture_from_contract(
    contract: Mapping[str, Any],
) -> ThreeLayerNoPaddingArchitecture:
    return ThreeLayerNoPaddingArchitecture(**contract["architecture"])


def _load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    training = contract.get("training", {})
    read = contract.get("read_contract", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status")
        != "frozen_after_negative_regularization_aggregate_before_control_metrics"
        or int(training.get("seed", -1)) != 20260724
        or int(training.get("maximum_steps", -1)) != 15360
        or tuple(training.get("checkpoint_steps", ())) != CHECKPOINT_STEPS
        or int(training.get("batch_size", -1)) != 4
        or float(training.get("initial_learning_rate", -1.0)) != 5.0e-4
        or read.get("training_state") is not True
        or any(
            read.get(name) is not False
            for name in (
                "validation_state",
                "inference_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
                "long_term_state",
            )
        )
    ):
        raise ValueError("three-layer/no-padding contract changed")
    architecture_from_contract(contract)
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise ThreeLayerNoPaddingError(
                    f"architecture-control source changed: {source}"
                )
    return contract, resolved, _file_sha256(resolved)


def _verify_aggregate(contract: Mapping[str, Any]) -> None:
    for name in ("aggregate_report", "aggregate_manifest"):
        record = contract["sources"][name]
        path = Path(record["path"]).resolve()
        if not path.is_file() or _file_sha256(path) != record["sha256"]:
            raise ThreeLayerNoPaddingError(f"{name} changed")
        value = json.loads(path.read_text())
        if value["content_sha256"] != record["content_sha256"]:
            raise ThreeLayerNoPaddingError(f"{name} content changed")
        if name == "aggregate_report":
            decision = value["selection_decision"]
            if (
                decision["status"] != "no_regularization_arm_passed"
                or decision["retain_original_model"] is not True
                or decision["next_action"]
                != "freeze_three_layer_no_padding_training_only_control"
            ):
                raise ThreeLayerNoPaddingError(
                    "regularization aggregate no longer authorizes control"
                )
            if any(
                value.get(field) is not False
                for field in (
                    "validation_state_opened",
                    "inference_state_opened",
                    "response_or_adjoint_state_opened",
                    "long_term_state_opened",
                )
            ):
                raise ThreeLayerNoPaddingError(
                    "regularization aggregate opened a later archive"
                )


def _verify_artifacts(
    contract: Mapping[str, Any],
    dataset: Path,
) -> tuple[Path, dict[str, Any]]:
    result = _ORIGINALS["_verify_artifacts"](contract, dataset)
    _verify_aggregate(contract)
    return result


def _arm_from_contract(
    contract: Mapping[str, Any],
    arm_index: int,
) -> ArchitectureControl:
    if arm_index != 0 or contract["control"]["control_id"] != CONTROL_ID:
        raise ValueError("architecture control index changed")
    control = ArchitectureControl()
    if (
        contract["control"]["pointwise_layer_norm"]
        != control.pointwise_layer_norm
        or float(contract["control"]["channel_mlp_dropout"])
        != control.channel_mlp_dropout
    ):
        raise ValueError("architecture control regularization changed")
    return control


def select_control_checkpoint(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Use the inherited frozen gate and expose control-specific semantics."""

    decision = _ORIGINALS["select_arm_checkpoint"](summaries)
    passed = bool(decision["arm_training_gate_passed"])
    return {
        "status": (
            "architecture_control_training_gate_passed"
            if passed
            else "no_architecture_control_checkpoint_passed"
        ),
        "control_training_gate_passed": passed,
        "selected_optimizer_step": decision["selected_optimizer_step"],
        "selected_fine_tune_step": decision["selected_fine_tune_step"],
        "best_diagnostic_optimizer_step": (
            decision["best_diagnostic_optimizer_step"]
        ),
        "next_action": (
            "freeze_replication_before_any_later_archive_read"
            if passed
            else "retain_original_model_and_advance_to_group_specific_heads"
        ),
    }


def _checkpoint_path(directory: Path, step: int) -> Path:
    return directory / f"model_c_three_layer_no_padding_step_{step:05d}.pt"


_PATCHED_NAMES = (
    "VERSION",
    "CHECKPOINT_STEPS",
    "ARM_IDS",
    "REPORT_NAME",
    "ARRAYS_NAME",
    "FIGURE_NAME",
    "README_NAME",
    "MANIFEST_NAME",
    "CHECKPOINT_DIRECTORY",
    "BEST_NAME",
    "ModelCSuccessorArchitecture",
    "load_contract",
    "_verify_artifacts",
    "arm_from_contract",
    "select_arm_checkpoint",
    "_checkpoint_path",
)
_ORIGINALS = {name: getattr(base, name) for name in _PATCHED_NAMES}


@contextlib.contextmanager
def _patched_base() -> Iterator[None]:
    replacements = {
        "VERSION": VERSION,
        "CHECKPOINT_STEPS": CHECKPOINT_STEPS,
        "ARM_IDS": (CONTROL_ID,),
        "REPORT_NAME": REPORT_NAME,
        "ARRAYS_NAME": ARRAYS_NAME,
        "FIGURE_NAME": FIGURE_NAME,
        "README_NAME": README_NAME,
        "MANIFEST_NAME": MANIFEST_NAME,
        "CHECKPOINT_DIRECTORY": CHECKPOINT_DIRECTORY,
        "BEST_NAME": BEST_NAME,
        "ModelCSuccessorArchitecture": ThreeLayerNoPaddingArchitecture,
        "load_contract": _load_contract,
        "_verify_artifacts": _verify_artifacts,
        "arm_from_contract": _arm_from_contract,
        "select_arm_checkpoint": select_control_checkpoint,
        "_checkpoint_path": _checkpoint_path,
    }
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in _ORIGINALS.items():
            setattr(base, name, value)


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify the sealed sources and production selection records."""

    with _patched_base():
        result = base.preflight(contract_path, 0)
    result["control"] = result.pop("arm")
    return result


def run(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train and evaluate the one frozen architecture control."""

    with _patched_base():
        return base.run(contract_path, 0, device_name=device_name)


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
