"""Single-position-encoding LayerNorm arm for anomaly-direct Model C.

The retained full 46-channel pointwise-anomaly direct-state map is trained again
with exactly two declared changes: the duplicated positional encoding is removed
and the Bire pointwise channel LayerNorm is applied after spectral and
Channel-MLP mixing.

Position currently enters the map twice.  The 51 input channels already carry
``longitude_normalized`` and ``latitude_normalized`` from :mod:`af_data`, and the
FNO is additionally built with ``positional_embedding="grid"`` so that
``GridEmbeddingND`` appends two more coordinate channels immediately before
lifting.  Both encodings are computed on the same unpadded 62x62 grid, so the
lifting block receives 53 channels carrying 51 distinct fields.  Bire et al.
encode position exactly once (Section 3.3.1).

The shared :class:`ModelCSuccessorArchitecture` cannot simply be relaxed to allow
``positional_embedding=None``: five frozen contracts certify that module's
SHA-256, and re-recording them would retroactively falsify the provenance of
results the earlier code produced.  This module therefore supplies its own
permissive architecture type and its own contract loader, and reuses the parent
regularization control's training loop, LayerNorm module, and gate evaluator by
import.  No certified source file is modified.

Training split only.  The held S0 figure suite opens solely through the separate
frozen figure contract, and only after this training-only gate passes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import (
    af_model_c_anomaly_direct_bire_regularization_controls as controls,
)
from . import (
    af_model_c_anomaly_direct_training_spectral_attribution_v2 as attribution,
)
from .af_model_c_overfit import _file_sha256
from .af_model_c_successor import INPUT_CHANNEL_COUNT, STATE_CHANNEL_COUNT

VERSION = "model_c_anomaly_direct_single_position_layernorm_v1"
CONTRACT_STATUS = "frozen_before_any_single_position_layernorm_metric"
EXPECTED_ARM = ("layernorm", True, 0.0)
CHECKPOINT_STEPS = controls.CHECKPOINT_STEPS


class SinglePositionLayerNormError(RuntimeError):
    """Raised when the single-position LayerNorm arm violates its contract."""


@dataclass(frozen=True)
class SinglePositionArchitecture:
    """Retained Model C proportions with the appended grid embedding removed.

    Field-for-field compatible with
    :class:`~bire_repro.af_model_c_successor.ModelCSuccessorArchitecture`, but
    ``positional_embedding`` may be ``None`` so that position is supplied once,
    through the two existing static data channels.
    """

    in_channels: int = INPUT_CHANNEL_COUNT
    out_channels: int = STATE_CHANNEL_COUNT
    n_modes: tuple[int, int] = (24, 16)
    hidden_channels: int = 128
    n_layers: int = 4
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    channel_mlp_expansion: float = 4.0
    domain_padding: float = 0.1
    positional_embedding: str | None = None
    use_channel_mlp: bool = True
    local_kernel_size: int = 3
    fno_block_precision: str = "full"
    factorization: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "n_modes", tuple(int(value) for value in self.n_modes)
        )
        if self.in_channels != INPUT_CHANNEL_COUNT or self.out_channels != 46:
            raise SinglePositionLayerNormError(
                "single-position arm keeps the 46 state + 5 static -> 46 contract"
            )
        if self.n_modes != (24, 16) or self.n_layers != 4:
            raise SinglePositionLayerNormError(
                "single-position arm fixes modes=(24,16) and four layers"
            )
        if self.hidden_channels != 128 or self.channel_mlp_expansion != 4.0:
            raise SinglePositionLayerNormError(
                "single-position arm keeps the width-128 Bire latent proportions"
            )
        if self.lifting_channel_ratio != 2 or self.projection_channel_ratio != 2:
            raise SinglePositionLayerNormError(
                "single-position arm keeps lifting and projection width 256"
            )
        if (
            self.domain_padding != 0.1
            or not self.use_channel_mlp
            or self.local_kernel_size != 3
        ):
            raise SinglePositionLayerNormError(
                "single-position arm keeps padding, Channel MLP, and the 3x3 branch"
            )
        if self.positional_embedding is not None:
            raise SinglePositionLayerNormError(
                "the single-position arm exists to disable the appended grid embedding"
            )
        if self.fno_block_precision != "full" or self.factorization is not None:
            raise SinglePositionLayerNormError(
                "single-position arm remains a dense float32 FNO"
            )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_modes"] = list(self.n_modes)
        return result


def arm_from_contract(
    contract: Mapping[str, Any],
    arm_index: int,
) -> controls.RegularizationArm:
    """Return the single declared arm, which must be the Bire LayerNorm arm."""

    arms = contract.get("arms", ())
    if arm_index != 0 or len(arms) != 1:
        raise SinglePositionLayerNormError(
            "the single-position arm declares exactly one LayerNorm arm"
        )
    record = arms[0]
    arm = controls.RegularizationArm(
        arm_id=str(record["arm_id"]),
        pointwise_layer_norm=bool(record["pointwise_layer_norm"]),
        channel_mlp_dropout=float(record["channel_mlp_dropout"]),
    )
    if (
        arm.arm_id,
        arm.pointwise_layer_norm,
        arm.channel_mlp_dropout,
    ) != EXPECTED_ARM:
        raise SinglePositionLayerNormError("single-position arm definition changed")
    return arm


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the contract frozen before any single-position LayerNorm metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    training = contract.get("training", {})
    read = contract.get("read_contract", {})
    architecture = contract.get("architecture", {})
    selection = contract.get("selection", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or len(contract.get("arms", ())) != 1
        or architecture.get("positional_embedding") is not None
        or int(architecture.get("in_channels", -1)) != INPUT_CHANNEL_COUNT
        or int(architecture.get("out_channels", -1)) != STATE_CHANNEL_COUNT
        or int(training.get("seed", -1)) != 20260724
        or int(training.get("maximum_steps", -1)) != 15360
        or tuple(training.get("checkpoint_steps", ())) != CHECKPOINT_STEPS
        or int(training.get("batch_size", -1)) != 4
        or float(training.get("initial_learning_rate", -1.0)) != 5.0e-4
        or float(selection.get("primary_ratio_limit", -1.0)) != 1.0
        or read.get("training_state") is not True
        or any(
            read.get(name) is not False
            for name in (
                "validation_state",
                "inference_state",
                "intermediate_wind_state",
                "response_state",
                "adjoint_state",
            )
        )
    ):
        raise SinglePositionLayerNormError(
            "single-position LayerNorm contract changed"
        )
    arm_from_contract(contract, 0)
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise SinglePositionLayerNormError(
                    f"single-position LayerNorm source changed: {source}"
                )
    return contract, resolved, _file_sha256(resolved)


CONTROLS_BINDINGS = (
    "ModelCSuccessorArchitecture",
    "load_contract",
    "arm_from_contract",
    "VERSION",
)
ARCHITECTURE_MODULES = (controls, attribution)


class _ParentBinding:
    """Bind this arm's architecture, loader, and version into the parent modules.

    The parent training loop resolves ``ModelCSuccessorArchitecture``,
    ``load_contract``, ``arm_from_contract``, and ``VERSION`` as module globals at
    call time, so rebinding them is sufficient to reuse the loop without editing
    the certified parent source.  The same rebinding idiom is already used inside
    that module for ``attribution.build_successor``.

    Checkpoint evaluation is delegated to
    :mod:`af_model_c_anomaly_direct_training_spectral_attribution_v2`, which
    reconstructs the architecture from the saved payload using *its own* module
    global.  That symbol must be bound as well, otherwise training completes and
    the run only fails afterwards, when the first checkpoint is reloaded and the
    strict validator rejects ``positional_embedding=None``.
    """

    def __enter__(self) -> None:
        self._saved: dict[tuple[str, str], Any] = {}
        for module in ARCHITECTURE_MODULES:
            names = (
                CONTROLS_BINDINGS
                if module is controls
                else ("ModelCSuccessorArchitecture",)
            )
            for name in names:
                if not hasattr(module, name):
                    raise SinglePositionLayerNormError(
                        f"{module.__name__} no longer defines {name}"
                    )
                self._saved[(module.__name__, name)] = getattr(module, name)
            module.ModelCSuccessorArchitecture = SinglePositionArchitecture
        controls.load_contract = load_contract
        controls.arm_from_contract = arm_from_contract
        controls.VERSION = VERSION

    def __exit__(self, *exc: Any) -> None:
        modules = {module.__name__: module for module in ARCHITECTURE_MODULES}
        for (module_name, name), value in self._saved.items():
            setattr(modules[module_name], name, value)


def preflight(contract_path: str | Path, arm_index: int = 0) -> dict[str, Any]:
    """Verify sources and the single-encoding model without training."""

    with _ParentBinding():
        result = dict(controls.preflight(contract_path, arm_index))
    contract, _, _ = load_contract(contract_path)
    architecture = SinglePositionArchitecture(**contract["architecture"])
    result["positional_embedding"] = architecture.positional_embedding
    result["duplicate_positional_encoding_removed"] = True
    result["state_channels"] = int(architecture.out_channels)
    return result


def run(
    contract_path: str | Path,
    arm_index: int = 0,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train and select the single-position LayerNorm arm on the training split."""

    with _ParentBinding():
        return controls.run(contract_path, arm_index, device_name=device_name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        child.add_argument("--arm-index", type=int, default=0)
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
        result = preflight(args.contract, args.arm_index)
    else:
        result = run(args.contract, args.arm_index, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
