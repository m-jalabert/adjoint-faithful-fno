"""Training-only group-specific projection-head control for Model C."""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import af_model_c_anomaly_direct_bire_regularization_controls as base
from .af_model_c_overfit import _file_sha256

try:
    import torch
    import torch.nn as nn
    from neuralop.models import FNO
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    FNO = None  # type: ignore[assignment]


VERSION = "model_c_anomaly_direct_group_heads_v1"
CONTROL_ID = "group_specific_heads"
CHECKPOINT_STEPS = (3840, 7680, 11520, 13440, 14400, 14880, 15360)
REPORT_NAME = "group_heads_report.json"
ARRAYS_NAME = "group_heads_arrays.npz"
FIGURE_NAME = "model_c_group_heads_selection.png"
README_NAME = "README.md"
MANIFEST_NAME = "manifest.json"
CHECKPOINT_DIRECTORY = "training_checkpoints"
BEST_NAME = "model_c_group_heads_best_diagnostic.pt"
GROUP_NAMES = ("u", "v", "temperature", "ssh")
GROUP_SIZES = (15, 15, 15, 1)


class GroupHeadsControlError(RuntimeError):
    """Raised when the frozen group-head control changes."""


@dataclass(frozen=True)
class GroupHeadsControl:
    """No LayerNorm or dropout; only the declared projection heads change."""

    arm_id: str = CONTROL_ID
    pointwise_layer_norm: bool = False
    channel_mlp_dropout: float = 0.0


@dataclass(frozen=True)
class GroupHeadsArchitecture:
    """The source trunk followed by four independent two-layer heads."""

    in_channels: int = 51
    out_channels: int = 46
    n_modes: tuple[int, int] = (24, 16)
    hidden_channels: int = 128
    n_layers: int = 4
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    channel_mlp_expansion: float = 4.0
    domain_padding: float = 0.1
    positional_embedding: str = "grid"
    use_channel_mlp: bool = True
    local_kernel_size: int = 3
    fno_block_precision: str = "full"
    factorization: str | None = None
    trunk_out_channels: int = 128
    head_hidden_channels: int = 256
    head_groups: tuple[str, ...] = GROUP_NAMES
    head_group_sizes: tuple[int, ...] = GROUP_SIZES

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "n_modes", tuple(int(value) for value in self.n_modes)
        )
        object.__setattr__(self, "head_groups", tuple(self.head_groups))
        object.__setattr__(
            self,
            "head_group_sizes",
            tuple(int(value) for value in self.head_group_sizes),
        )
        if (
            self.in_channels != 51
            or self.out_channels != 46
            or self.n_modes != (24, 16)
            or self.hidden_channels != 128
            or self.n_layers != 4
            or self.lifting_channel_ratio != 2
            or self.projection_channel_ratio != 2
            or self.channel_mlp_expansion != 4.0
            or self.domain_padding != 0.1
            or self.positional_embedding != "grid"
            or not self.use_channel_mlp
            or self.local_kernel_size != 3
            or self.fno_block_precision != "full"
            or self.factorization is not None
            or self.trunk_out_channels != 128
            or self.head_hidden_channels != 256
            or self.head_groups != GROUP_NAMES
            or self.head_group_sizes != GROUP_SIZES
            or sum(self.head_group_sizes) != self.out_channels
        ):
            raise ValueError("group-specific-head architecture changed")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_modes"] = list(self.n_modes)
        result["head_groups"] = list(self.head_groups)
        result["head_group_sizes"] = list(self.head_group_sizes)
        return result


if nn is not None:

    class GroupSpecificHeadDirectFNO(nn.Module):
        """Shared FNO dynamics with independent U/V/T/SSH projections."""

        def __init__(
            self,
            architecture: GroupHeadsArchitecture,
            arm: GroupHeadsControl,
        ) -> None:
            super().__init__()
            if arm != GroupHeadsControl():
                raise ValueError("group-head regularization control changed")
            self.architecture = architecture
            self.arm = arm
            self.fno = FNO(
                n_modes=architecture.n_modes,
                in_channels=architecture.in_channels,
                out_channels=architecture.trunk_out_channels,
                hidden_channels=architecture.hidden_channels,
                n_layers=architecture.n_layers,
                lifting_channel_ratio=architecture.lifting_channel_ratio,
                projection_channel_ratio=architecture.projection_channel_ratio,
                positional_embedding=architecture.positional_embedding,
                use_channel_mlp=architecture.use_channel_mlp,
                channel_mlp_dropout=arm.channel_mlp_dropout,
                channel_mlp_expansion=architecture.channel_mlp_expansion,
                domain_padding=architecture.domain_padding,
                fno_block_precision=architecture.fno_block_precision,
                factorization=architecture.factorization,
            )
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            architecture.trunk_out_channels,
                            architecture.head_hidden_channels,
                            kernel_size=1,
                        ),
                        nn.GELU(),
                        nn.Conv2d(
                            architecture.head_hidden_channels,
                            group_size,
                            kernel_size=1,
                        ),
                    )
                    for group_size in architecture.head_group_sizes
                ]
            )
            self.local_heads = nn.ModuleList(
                [
                    nn.Conv2d(
                        architecture.in_channels,
                        group_size,
                        kernel_size=architecture.local_kernel_size,
                        padding=architecture.local_kernel_size // 2,
                    )
                    for group_size in architecture.head_group_sizes
                ]
            )

        def forward(self, features: Any) -> Any:
            if (
                features.ndim != 4
                or features.shape[1] != self.architecture.in_channels
            ):
                raise ValueError("group-head Model C expects N,51,Y,X")
            latent = self.fno(features)
            groups = [
                head(latent) + local(features)
                for head, local in zip(
                    self.heads,
                    self.local_heads,
                    strict=True,
                )
            ]
            return torch.cat(groups, dim=1)

else:  # pragma: no cover
    GroupSpecificHeadDirectFNO = None  # type: ignore[assignment,misc]


def build_group_head_model(
    architecture: GroupHeadsArchitecture,
    arm: GroupHeadsControl,
) -> Any:
    if GroupSpecificHeadDirectFNO is None:  # pragma: no cover
        raise RuntimeError("group-head control requires PyTorch")
    return GroupSpecificHeadDirectFNO(architecture, arm)


def architecture_from_contract(
    contract: Mapping[str, Any],
) -> GroupHeadsArchitecture:
    return GroupHeadsArchitecture(**contract["architecture"])


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
        != "frozen_after_negative_three_layer_control_before_group_head_metrics"
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
        raise ValueError("group-head control contract changed")
    architecture_from_contract(contract)
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise GroupHeadsControlError(
                    f"group-head source changed: {source}"
                )
    return contract, resolved, _file_sha256(resolved)


def _verify_predecessor(contract: Mapping[str, Any]) -> None:
    for name in ("three_layer_report", "three_layer_manifest"):
        record = contract["sources"][name]
        path = Path(record["path"]).resolve()
        if not path.is_file() or _file_sha256(path) != record["sha256"]:
            raise GroupHeadsControlError(f"{name} changed")
        value = json.loads(path.read_text())
        if value["content_sha256"] != record["content_sha256"]:
            raise GroupHeadsControlError(f"{name} content changed")
        if name == "three_layer_report":
            decision = value["selection_decision"]
            if (
                decision["status"]
                != "no_architecture_control_checkpoint_passed"
                or decision["control_training_gate_passed"] is not False
                or decision["next_action"]
                != "retain_original_model_and_advance_to_group_specific_heads"
                or value["retraining_steps"] != 0
            ):
                raise GroupHeadsControlError(
                    "three-layer result no longer authorizes group heads"
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
                raise GroupHeadsControlError(
                    "three-layer result opened a later archive"
                )


def _verify_artifacts(
    contract: Mapping[str, Any],
    dataset: Path,
) -> tuple[Path, dict[str, Any]]:
    result = _ORIGINALS["_verify_artifacts"](contract, dataset)
    _verify_predecessor(contract)
    return result


def _arm_from_contract(
    contract: Mapping[str, Any],
    arm_index: int,
) -> GroupHeadsControl:
    if arm_index != 0 or contract["control"]["control_id"] != CONTROL_ID:
        raise ValueError("group-head control index changed")
    arm = GroupHeadsControl()
    if (
        contract["control"]["pointwise_layer_norm"]
        != arm.pointwise_layer_norm
        or float(contract["control"]["channel_mlp_dropout"])
        != arm.channel_mlp_dropout
    ):
        raise ValueError("group-head control regularization changed")
    return arm


def select_control_checkpoint(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the inherited gate with group-head-specific next actions."""

    decision = _ORIGINALS["select_arm_checkpoint"](summaries)
    passed = bool(decision["arm_training_gate_passed"])
    return {
        "status": (
            "group_head_training_gate_passed"
            if passed
            else "no_group_head_checkpoint_passed"
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
            else "retain_original_model_and_freeze_slow_fast_split_control"
        ),
    }


def _checkpoint_path(directory: Path, step: int) -> Path:
    return directory / f"model_c_group_heads_step_{step:05d}.pt"


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
    "build_regularized_model",
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
        "ModelCSuccessorArchitecture": GroupHeadsArchitecture,
        "build_regularized_model": build_group_head_model,
        "load_contract": _load_contract,
        "_verify_artifacts": _verify_artifacts,
        "arm_from_contract": _arm_from_contract,
        "select_arm_checkpoint": select_control_checkpoint,
        "_checkpoint_path": _checkpoint_path,
    }
    original_attribution_architecture = (
        base.attribution.ModelCSuccessorArchitecture
    )
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        base.attribution.ModelCSuccessorArchitecture = GroupHeadsArchitecture
        yield
    finally:
        base.attribution.ModelCSuccessorArchitecture = (
            original_attribution_architecture
        )
        for name, value in _ORIGINALS.items():
            setattr(base, name, value)


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify sealed sources and the frozen group-head architecture."""

    with _patched_base():
        result = base.preflight(contract_path, 0)
    result["control"] = result.pop("arm")
    return result


def run(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train and evaluate the one frozen group-head control."""

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
