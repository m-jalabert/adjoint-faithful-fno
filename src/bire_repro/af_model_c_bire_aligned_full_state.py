"""Bire-aligned full-state FNO for the 46-channel closed-state emulator.

This arm answers one focused question: does a substantially more faithful Bire
architecture and training protocol improve the short- and long-term behaviour of
the retained 46-channel closed-state emulator?  The MITgcm trajectories, the
62x62 grid, the ten-day map, the 24x16 retained modes, the pointwise anomaly
normalization, and the held S0 evaluation are all unchanged.  Everything that
remains project-specific in the architecture and the training protocol is
replaced by the Bire choice.

Relative to the retained single-position LayerNorm arm this changes, as one
declared package:

* four FNO blocks become three, so eight pointwise LayerNorms become six;
* the linear longitude/latitude data channels are removed and position enters
  exactly once, through the deterministic ``oceanfourcast.PosEmbed``
  sine/cosine fields appended immediately before lifting;
* the external 3x3 raw-input-to-output convolution is removed, while Bire's
  pointwise residual path *inside* each spectral block is retained;
* the five-term Model C objective becomes ``MSE + 0.01 MAE`` on wet cells;
* three-step training from scratch becomes one-step pretraining followed by
  two-step autoregressive fine-tuning;
* AdamW(5e-4, wd 1e-5, batch 4) becomes Adam(1e-2, wd 0, batch 8).

Channel bookkeeping::

    46 state + 3 geometry/forcing  = 49 external inputs
    49 + 2 Bire position channels  = 51 lifting inputs
    51 -> 256 -> 128 -> 3 blocks -> 256 -> 46 direct normalized future state

The 49 external inputs keep the wet mask and the distance-to-wall field, which
encode basin geometry rather than generic absolute coordinates, and drop
``longitude_normalized`` and ``latitude_normalized`` so that absolute position
enters the map exactly once.

Training split only.  The held S0 figure suite opens through the separate
figure contract, and only after this run completes.  No certified source file
is modified: the checkpoint gate instrument is reused by binding this arm's
architecture, builder, and stepper into it for the duration of the call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from . import (
    af_model_c_anomaly_direct_training_spectral_attribution_v2 as attribution,
)
from .af_a0_evaluate import _normalizers
from .af_data import STATIC_FEATURES
from .af_model_a import (
    ChunkAwareBatchSampler,
    _checkpoint_state_dict,
    require_model_a_runtime,
    seed_everything,
)
from .af_model_b import records_for_rollout_split
from .af_model_c_anomaly_direct import (
    ModelCAnomalyRolloutDataset,
    PointwiseDirectStepper,
    direct_state_unroll,
)
from .af_model_c_anomaly_direct_bire_regularization_controls import (
    PointwiseChannelLayerNorm,
)
from .af_model_c_anomaly_direct_deep_pressure_spectral_regularization import (
    summarize_evaluation,
)
from .af_model_c_overfit import _device, _file_sha256
from .af_model_c_successor import STATE_CHANNEL_COUNT

try:
    import torch
    import torch.nn as nn
    from neuralop.models import FNO
    from torch.utils.data import DataLoader
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    FNO = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


VERSION = "model_c_bire_aligned_full_state_v1"
CONTRACT_STATUS = "frozen_before_any_bire_aligned_full_state_metric"

#: ``wind_stress_x``, ``wet_mask``, ``distance_to_wall_normalized``.  The two
#: linear coordinate fields are deliberately absent: position enters only
#: through :class:`BirePositionalEncoding`.
RETAINED_STATIC_FEATURES = (
    "wind_stress_x",
    "wet_mask",
    "distance_to_wall_normalized",
)
RETAINED_STATIC_INDICES = tuple(
    STATIC_FEATURES.index(name) for name in RETAINED_STATIC_FEATURES
)
EXTERNAL_INPUT_CHANNELS = STATE_CHANNEL_COUNT + len(RETAINED_STATIC_FEATURES)
POSITIONAL_CHANNELS = 2
LIFTING_INPUT_CHANNELS = EXTERNAL_INPUT_CHANNELS + POSITIONAL_CHANNELS

PRETRAIN_STEPS = 3840
FINE_TUNE_STEPS = 3840
MAXIMUM_STEPS = PRETRAIN_STEPS + FINE_TUNE_STEPS
CHECKPOINT_STEPS = (PRETRAIN_STEPS, MAXIMUM_STEPS)
STAGE_NAMES = ("pretrained", "finetuned")
STAGE_BY_STEP = {PRETRAIN_STEPS: "pretrained", MAXIMUM_STEPS: "finetuned"}
MAE_WEIGHT = 0.01

REPORT_NAME = "bire_aligned_full_state_report.json"
ARRAYS_NAME = "bire_aligned_full_state_arrays.npz"
FIGURE_NAME = "model_c_bire_aligned_full_state_selection.png"
DIVERGENCE_NAME = "bire_aligned_full_state_divergence.json"
README_NAME = "README.md"
MANIFEST_NAME = "manifest.json"
CHECKPOINT_DIRECTORY = "training_checkpoints"


class BireAlignedFullStateError(RuntimeError):
    """Raised when the Bire-aligned full-state arm violates its contract."""


class BireAlignedDivergenceError(BireAlignedFullStateError):
    """Raised when the declared learning rate produces a non-finite update."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BireAlignedArchitecture:
    """Three-block Bire FNO over the retained 46-channel state.

    ``in_channels`` counts the *external* inputs only.  The two deterministic
    Bire position channels are appended inside the model, immediately before
    lifting, so the lifting network still sees 51 channels.
    """

    in_channels: int = EXTERNAL_INPUT_CHANNELS
    out_channels: int = STATE_CHANNEL_COUNT
    positional_channels: int = POSITIONAL_CHANNELS
    lifting_in_channels: int = LIFTING_INPUT_CHANNELS
    grid_shape: tuple[int, int] = (62, 62)
    n_modes: tuple[int, int] = (24, 16)
    hidden_channels: int = 128
    n_layers: int = 3
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    channel_mlp_expansion: float = 4.0
    channel_mlp_dropout: float = 0.0
    domain_padding: float = 0.1
    positional_embedding: str | None = None
    use_channel_mlp: bool = True
    pointwise_layer_norm: bool = True
    local_kernel_size: int | None = None
    fno_block_precision: str = "full"
    factorization: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "n_modes", tuple(int(value) for value in self.n_modes)
        )
        object.__setattr__(
            self, "grid_shape", tuple(int(value) for value in self.grid_shape)
        )
        if (
            self.in_channels != EXTERNAL_INPUT_CHANNELS
            or self.out_channels != STATE_CHANNEL_COUNT
            or self.positional_channels != POSITIONAL_CHANNELS
            or self.lifting_in_channels != LIFTING_INPUT_CHANNELS
        ):
            raise BireAlignedFullStateError(
                "the Bire-aligned arm maps 46 state + 3 static (+2 position) -> 46"
            )
        if self.grid_shape != (62, 62) or self.n_modes != (24, 16):
            raise BireAlignedFullStateError(
                "the Bire-aligned arm keeps the 62x62 grid and 24x16 modes"
            )
        if self.n_layers != 3:
            raise BireAlignedFullStateError(
                "the Bire-aligned arm uses exactly three FNO blocks"
            )
        if self.hidden_channels != 128 or self.channel_mlp_expansion != 4.0:
            raise BireAlignedFullStateError(
                "the Bire-aligned arm keeps width 128 and the 4C Channel MLP"
            )
        if self.lifting_channel_ratio != 2 or self.projection_channel_ratio != 2:
            raise BireAlignedFullStateError(
                "the Bire-aligned arm keeps lifting and projection width 256"
            )
        if self.domain_padding != 0.1 or not self.use_channel_mlp:
            raise BireAlignedFullStateError(
                "the Bire-aligned arm keeps 10% padding and the Channel MLP"
            )
        if self.channel_mlp_dropout != 0.0:
            raise BireAlignedFullStateError(
                "the first clean Bire-aligned test freezes Channel MLP dropout at zero"
            )
        if self.positional_embedding is not None:
            raise BireAlignedFullStateError(
                "position enters only through the Bire sine/cosine encoder"
            )
        if not self.pointwise_layer_norm:
            raise BireAlignedFullStateError(
                "Bire normalizes channel-wise after both mixing operations"
            )
        if self.local_kernel_size is not None:
            raise BireAlignedFullStateError(
                "the external 3x3 raw-input branch is removed in this arm"
            )
        if self.fno_block_precision != "full" or self.factorization is not None:
            raise BireAlignedFullStateError(
                "the Bire-aligned arm remains a dense float32 FNO"
            )

    @property
    def layer_norm_count(self) -> int:
        return 2 * self.n_layers

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_modes"] = list(self.n_modes)
        result["grid_shape"] = list(self.grid_shape)
        return result


if nn is not None:

    class BirePositionalEncoding(nn.Module):
        """Deterministic ``oceanfourcast.PosEmbed`` position channels.

        Each direction contributes one channel whose entries alternate between
        sine and cosine of ``pi/(2 N) * index``::

            p(j) = sin(w_j) for even j,  cos(w_j) for odd j.

        The two fields are concatenated after the external inputs, in the
        published order ``(x, p_x, p_y)``, immediately before lifting.  They are
        constants, so they are registered as non-persistent buffers and never
        enter a checkpoint.
        """

        def __init__(self, height: int, width: int) -> None:
            super().__init__()
            self.height = int(height)
            self.width = int(width)
            self.register_buffer(
                "position_x",
                self._axis(self.width).view(1, 1, 1, self.width).expand(
                    1, 1, self.height, self.width
                ).contiguous(),
                persistent=False,
            )
            self.register_buffer(
                "position_y",
                self._axis(self.height).view(1, 1, self.height, 1).expand(
                    1, 1, self.height, self.width
                ).contiguous(),
                persistent=False,
            )

        @staticmethod
        def _axis(size: int) -> Any:
            frequency = math.pi / (2 * size) * torch.arange(
                size,
                dtype=torch.float32,
            )
            grid = torch.zeros(size, dtype=torch.float32)
            grid[::2] = torch.sin(frequency[::2])
            grid[1::2] = torch.cos(frequency[1::2])
            return grid

        def forward(self, value: Any) -> Any:
            if value.ndim != 4 or value.shape[-2:] != (self.height, self.width):
                raise ValueError("Bire position encoder expects N,C,62,62")
            batch = value.shape[0]
            return torch.cat(
                (
                    value,
                    self.position_x.expand(batch, -1, -1, -1),
                    self.position_y.expand(batch, -1, -1, -1),
                ),
                dim=1,
            )

    class BireAlignedFullStateFNO(nn.Module):
        """Three-block Bire FNO with six pointwise LayerNorms and no 3x3 branch.

        The pointwise residual ``W h`` around each truncated spectral operator is
        NeuralOperator's default ``fno_skip="linear"`` path and is retained; only
        the external raw-input-to-output convolution is removed.
        """

        def __init__(self, architecture: BireAlignedArchitecture) -> None:
            super().__init__()
            self.architecture = architecture
            self.positional_encoding = BirePositionalEncoding(
                *architecture.grid_shape
            )
            self.fno = FNO(
                n_modes=architecture.n_modes,
                in_channels=architecture.lifting_in_channels,
                out_channels=architecture.out_channels,
                hidden_channels=architecture.hidden_channels,
                n_layers=architecture.n_layers,
                lifting_channel_ratio=architecture.lifting_channel_ratio,
                projection_channel_ratio=architecture.projection_channel_ratio,
                positional_embedding=architecture.positional_embedding,
                use_channel_mlp=architecture.use_channel_mlp,
                channel_mlp_dropout=architecture.channel_mlp_dropout,
                channel_mlp_expansion=architecture.channel_mlp_expansion,
                domain_padding=architecture.domain_padding,
                fno_block_precision=architecture.fno_block_precision,
                factorization=architecture.factorization,
            )
            self.fno.fno_blocks.norm = nn.ModuleList(
                [
                    PointwiseChannelLayerNorm(architecture.hidden_channels)
                    for _ in range(architecture.layer_norm_count)
                ]
            )

        def forward(self, features: Any) -> Any:
            if (
                features.ndim != 4
                or features.shape[1] != self.architecture.in_channels
            ):
                raise ValueError("Bire-aligned Model C expects N,49,Y,X")
            return self.fno(self.positional_encoding(features))

else:  # pragma: no cover - environment dependent
    BirePositionalEncoding = None  # type: ignore[assignment,misc]
    BireAlignedFullStateFNO = None  # type: ignore[assignment,misc]


def build_bire_aligned_model(architecture: BireAlignedArchitecture) -> Any:
    """Build the Bire-aligned map, matching ``build_successor``'s signature."""

    require_model_a_runtime()
    if BireAlignedFullStateFNO is None:  # pragma: no cover
        raise RuntimeError("the Bire-aligned arm requires PyTorch")
    return BireAlignedFullStateFNO(architecture)


class BireAlignedStepper(PointwiseDirectStepper):
    """Evaluation adapter that supplies only the three retained static fields."""

    def normalized_static(self, static: Any, experiments: np.ndarray) -> Any:
        value = super().normalized_static(static, experiments)
        index = torch.as_tensor(
            RETAINED_STATIC_INDICES,
            dtype=torch.long,
            device=value.device,
        )
        return value.index_select(1, index)


def retained_features(batch: Any) -> Any:
    """Drop the two linear coordinate channels from a 51-channel batch."""

    if batch.ndim != 4 or batch.shape[1] != STATE_CHANNEL_COUNT + len(
        STATIC_FEATURES
    ):
        raise ValueError("the Bire-aligned arm reduces N,51,Y,X inputs")
    index = torch.as_tensor(
        [STATE_CHANNEL_COUNT + offset for offset in RETAINED_STATIC_INDICES],
        dtype=torch.long,
        device=batch.device,
    )
    return torch.cat(
        (batch[:, :STATE_CHANNEL_COUNT], batch.index_select(1, index)),
        dim=1,
    )


def bire_loss_terms(
    prediction: Any,
    target: Any,
    wet: Any,
    *,
    mae_weight: float = MAE_WEIGHT,
) -> dict[str, Any]:
    """Wet-cell ``MSE + 0.01 MAE`` over the 46 normalized state channels.

    Both reductions divide by ``B * C * |wet|`` so that land cells contribute
    neither error nor denominator, which is the necessary adaptation of Bire's
    dense-grid objective to a basin with land.
    """

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Bire loss expects matching N,C,Y,X tensors")
    error = (prediction - target) * wet
    denominator = (
        float(prediction.shape[0])
        * float(prediction.shape[1])
        * wet.sum()
    )
    mse = error.square().sum() / denominator
    mae = error.abs().sum() / denominator
    return {"mse": mse, "mae": mae, "total": mse + mae_weight * mae}


LOSS_TERMS = ("total", "mse", "mae")


def arm_stages(contract: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return the two declared training stages in execution order."""

    stages = contract.get("stages", ())
    if len(stages) != 2:
        raise BireAlignedFullStateError(
            "the Bire-aligned protocol declares exactly two stages"
        )
    expected = (
        ("pretrained", 1, 1, PRETRAIN_STEPS),
        ("finetuned", 2, PRETRAIN_STEPS + 1, MAXIMUM_STEPS),
    )
    resolved = []
    for record, reference in zip(stages, expected):
        value = (
            str(record["stage_id"]),
            int(record["autoregressive_steps"]),
            int(record["first_optimizer_step"]),
            int(record["last_optimizer_step"]),
        )
        if value != reference:
            raise BireAlignedFullStateError("the Bire-aligned stage plan changed")
        resolved.append(dict(record))
    return tuple(resolved)


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the contract frozen before any Bire-aligned full-state metric."""

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
        or float(training.get("initial_learning_rate", -1.0)) != 1.0e-2
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
        raise BireAlignedFullStateError("Bire-aligned full-state contract changed")
    arm_stages(contract)
    BireAlignedArchitecture(**architecture)
    if verify_sources:
        root = resolved.parents[1]
        for relative, expected in contract.get("source_hashes", {}).items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise BireAlignedFullStateError(
                    f"Bire-aligned full-state source changed: {source}"
                )
    return contract, resolved, _file_sha256(resolved)


def _verify_artifacts(
    contract: Mapping[str, Any],
    dataset: Path,
) -> tuple[Path, dict[str, Any]]:
    """Verify the immutable dataset and the reused gate instrument."""

    sources = contract["sources"]
    if (
        dataset != Path(sources["dataset"]["path"]).resolve()
        or _file_sha256(dataset / ".zmetadata")
        != sources["dataset"]["metadata_sha256"]
    ):
        raise BireAlignedFullStateError("dataset source changed")
    for name in (
        "normalization",
        "attribution_report",
        "attribution_arrays",
        "attribution_manifest",
        "incumbent_report",
    ):
        record = sources[name]
        path = Path(record["path"]).resolve()
        if not path.is_file() or _file_sha256(path) != record["sha256"]:
            raise BireAlignedFullStateError(f"source artifact changed: {name}")
    attribution_contract, _, digest = attribution.load_contract(
        sources["attribution_contract"]["path"]
    )
    if digest != sources["attribution_contract"]["sha256"]:
        raise BireAlignedFullStateError("attribution contract changed")
    attribution._verify_sources(attribution_contract, dataset)
    return Path(sources["normalization"]["path"]).resolve(), attribution_contract


ATTRIBUTION_BINDINGS = (
    "ModelCSuccessorArchitecture",
    "build_successor",
    "PointwiseDirectStepper",
)


class _GateBinding:
    """Bind this arm's architecture, builder, and stepper into the instrument.

    :func:`attribution._evaluate_seed` resolves ``ModelCSuccessorArchitecture``,
    ``build_successor``, and ``PointwiseDirectStepper`` as module globals at call
    time, so rebinding the three for the duration of the call reuses the frozen
    360-day spectral/primary gate without editing the certified source.  The
    stepper must be bound as well: the Bire-aligned map takes 49 external
    channels, so the five-channel static block would otherwise be concatenated
    unchanged and the forward pass would fail on the first checkpoint reload.
    """

    def __enter__(self) -> None:
        self._saved: dict[str, Any] = {}
        for name in ATTRIBUTION_BINDINGS:
            if not hasattr(attribution, name):
                raise BireAlignedFullStateError(
                    f"{attribution.__name__} no longer defines {name}"
                )
            self._saved[name] = getattr(attribution, name)
        attribution.ModelCSuccessorArchitecture = BireAlignedArchitecture
        attribution.build_successor = build_bire_aligned_model
        attribution.PointwiseDirectStepper = BireAlignedStepper

    def __exit__(self, *exc: Any) -> None:
        for name, value in self._saved.items():
            setattr(attribution, name, value)


def _source_evaluation(sources: Mapping[str, Any]) -> dict[str, Any]:
    """The fixed unregularized seed-20260724 step-13440 comparator."""

    with np.load(sources["attribution_arrays"]["path"], allow_pickle=False) as data:
        seeds = np.asarray(data["seeds"])
        index = int(np.flatnonzero(seeds == 20260724)[0])
        return {
            "optimizer_step": 13440,
            "ratio": np.asarray(data["frozen_median_modewise_ratio"][index]),
            "integrated": np.asarray(data["integrated_energy_ratio"][index]),
            "tail_model": np.asarray(data["tail_model_fraction"][index]),
            "tail_truth": np.asarray(data["tail_truth_fraction"][index]),
            "model_rmse": np.asarray(data["primary_model_rmse"][index]),
            "persistence_rmse": np.asarray(data["primary_persistence_rmse"][index]),
        }


def _checkpoint_path(directory: Path, step: int) -> Path:
    return directory / f"model_c_bire_aligned_step_{step:05d}.pt"


def _evaluate_checkpoint(
    checkpoint: Path,
    optimizer_step: int,
    normalization: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    record = {
        "seed": 20260724,
        "optimizer_step": optimizer_step,
        "checkpoint": {"path": str(checkpoint)},
        "normalization": {"path": str(normalization)},
    }
    with _GateBinding():
        return attribution._evaluate_seed(record, **kwargs)


def _plot(
    path: Path,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([value["fine_tune_step"] for value in summaries])
    labels = [value.get("stage_id", "source") for value in summaries]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(
        steps,
        [value["day360_mid_modewise_ratio"] for value in summaries],
        "o-",
        label="PHIHYD mid",
    )
    axes[0].plot(
        steps,
        [value["day360_bottom_modewise_ratio"] for value in summaries],
        "s-",
        label="PHIHYD bottom",
    )
    axes[0].axhline(4.0, color="black", linestyle="--", label="factor-four")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Day-360 median modewise ratio")
    axes[0].legend()
    for field in ("surface_speed", "sst", "phihyd_surface"):
        axes[1].plot(
            steps,
            [
                value["primary_10_to_90_rmse_ratio_to_persistence"][field]
                for value in summaries
            ],
            "o-",
            label=field.replace("_", " "),
        )
    axes[1].axhline(1.0, color="black", linestyle="--", label="persistence")
    axes[1].set_ylabel("10--90-day RMSE ratio")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("Optimizer step (0 = fixed source)")
        axis.set_xticks(steps)
        axis.set_xticklabels(
            [f"{step}\n{label}" for step, label in zip(steps, labels)]
        )
        axis.grid(alpha=0.25)
    figure.suptitle("Bire-aligned full-state FNO: pretraining and fine-tuning")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify sources and build the map once, without training."""

    contract, resolved, digest = load_contract(contract_path)
    dataset = Path(contract["sources"]["dataset"]["path"]).resolve()
    _, attribution_contract = _verify_artifacts(contract, dataset)
    group = zarr.open_consolidated(str(dataset), mode="r")
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    records = attribution.training_records(attribution_contract, split)
    architecture = BireAlignedArchitecture(**contract["architecture"])
    model = build_bire_aligned_model(architecture)
    parameters = int(sum(value.numel() for value in model.parameters()))
    with torch.no_grad():
        probe = model(
            torch.zeros(1, EXTERNAL_INPUT_CHANNELS, *architecture.grid_shape)
        )
    if tuple(probe.shape) != (1, STATE_CHANNEL_COUNT, *architecture.grid_shape):
        raise BireAlignedFullStateError("the Bire-aligned map output shape changed")
    return {
        "status": "ready",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "external_input_channels": EXTERNAL_INPUT_CHANNELS,
        "lifting_input_channels": LIFTING_INPUT_CHANNELS,
        "retained_static_features": list(RETAINED_STATIC_FEATURES),
        "fno_blocks": architecture.n_layers,
        "layer_norm_modules": len(model.fno.fno_blocks.norm),
        "external_local_branch": False,
        "parameter_count": parameters,
        "selection_records": int(records.shape[0]),
        "inference_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }


def run(
    contract_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Pretrain one step, fine-tune two steps, and evaluate both checkpoints."""

    if torch is None or DataLoader is None:  # pragma: no cover
        raise RuntimeError("the Bire-aligned arm requires PyTorch")
    require_model_a_runtime()
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    stages = arm_stages(contract)
    dataset = Path(contract["sources"]["dataset"]["path"]).resolve()
    normalization, attribution_contract = _verify_artifacts(contract, dataset)
    scratch = Path(contract["output"]["scratch_root"]).resolve()
    project = Path(contract["output"]["project_root"]).resolve()
    scratch_tmp = scratch.with_name(scratch.name + ".tmp")
    project_tmp = project.with_name(project.name + ".tmp")
    if any(path.exists() for path in (scratch, project, scratch_tmp, project_tmp)):
        raise FileExistsError("refusing to overwrite Bire-aligned output")

    training = contract["training"]
    seed_everything(int(training["seed"]))
    device = _device(device_name)
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    pair_codes = np.asarray(group["pair_split"][:], dtype=np.uint8)
    split = np.asarray(group["snapshot_split"][:], dtype=np.uint8)
    _, _, wet_array, _, wind_mean, wind_scale = _normalizers(group)
    wet_array = np.asarray(wet_array, dtype=bool)
    with np.load(normalization, allow_pickle=False) as values:
        point_mean = np.asarray(values["pointwise_mean"], dtype=np.float32)
        point_scale = np.asarray(values["pointwise_scale"], dtype=np.float32)

    # The record population is the unchanged split-1 three-step rollout set, so
    # the chronology matches the incumbent arm exactly.  Only two of the three
    # stored futures are ever read.
    training_records = records_for_rollout_split(pair_codes, 1, rollout_steps=3)
    training_dataset = ModelCAnomalyRolloutDataset(
        dataset,
        training_records,
        point_mean,
        point_scale,
        rollout_steps=2,
    )
    batch_size = int(training["batch_size"])
    loader = DataLoader(
        training_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            training_dataset,
            batch_size,
            int(training["seed"]),
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    architecture = BireAlignedArchitecture(**contract["architecture"])
    model = build_bire_aligned_model(architecture).to(device)
    parameter_count = int(sum(value.numel() for value in model.parameters()))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["initial_learning_rate"]),
        betas=tuple(float(value) for value in training["adam_betas"]),
        weight_decay=float(training["weight_decay"]),
    )
    wet = torch.from_numpy(wet_array.astype(np.float32))[None, None].to(device)
    maximum_steps = int(training["maximum_steps"])
    decay_step = int(round(maximum_steps * float(training["decay_fraction"])))
    stage_by_step = {
        step: record
        for record in stages
        for step in range(
            int(record["first_optimizer_step"]),
            int(record["last_optimizer_step"]) + 1,
        )
    }

    scratch_tmp.parent.mkdir(parents=True, exist_ok=True)
    project_tmp.parent.mkdir(parents=True, exist_ok=True)
    scratch_tmp.mkdir()
    project_tmp.mkdir()
    checkpoint_directory = scratch_tmp / CHECKPOINT_DIRECTORY
    checkpoint_directory.mkdir()
    iterator = iter(loader)
    totals = {name: 0.0 for name in LOSS_TERMS}
    samples = 0
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    def _diverged(step: int, reason: str) -> None:
        """Promote a divergence diagnosis, then fail fast without tuning."""

        record = {
            "status": "diverged",
            "version": VERSION,
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "reason": reason,
            "optimizer_step": int(step),
            "stage": stage_by_step[step]["stage_id"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_clipping": False,
            "declared_response": (
                "the contract declares fail_fast_on_non_finite so that no "
                "untested stabilizer is introduced by this first test"
            ),
        }
        (project_tmp / DIVERGENCE_NAME).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        (project_tmp / README_NAME).write_text(
            "# Bire-aligned full-state FNO: diverged\n\n"
            f"Training became non-finite at optimizer step {step} "
            f"({record['stage']} stage) with learning rate "
            f"{record['learning_rate']:g} and no gradient clipping. "
            "The contract declares fail-fast rather than adding an untested "
            "stabilizing mechanism.\n"
        )
        shutil.rmtree(scratch_tmp, ignore_errors=True)
        os.replace(project_tmp, project)
        raise BireAlignedDivergenceError(
            f"{reason} at optimizer step {step} ({record['stage']} stage) "
            f"with learning rate {record['learning_rate']:g}"
        )

    for step in range(1, maximum_steps + 1):
        if step == decay_step + 1:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= float(training["decay_factor"])
        stage = stage_by_step[step]
        rollout_steps = int(stage["autoregressive_steps"])
        try:
            raw_features, futures = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_features, futures = next(iterator)
        raw_features = raw_features.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        futures = futures.to(device=device, dtype=torch.float32, non_blocking=True)
        features = retained_features(raw_features)
        model.train()
        predictions = direct_state_unroll(model, features, wet, rollout_steps)
        accumulated = {name: None for name in LOSS_TERMS}
        for index in range(rollout_steps):
            terms = bire_loss_terms(
                predictions[:, index],
                futures[:, index],
                wet,
                mae_weight=float(contract["loss"]["mae_weight"]),
            )
            for name in LOSS_TERMS:
                accumulated[name] = (
                    terms[name]
                    if accumulated[name] is None
                    else accumulated[name] + terms[name]
                )
        if not all(
            bool(torch.isfinite(accumulated[name]).item()) for name in LOSS_TERMS
        ):
            _diverged(step, "training objective became non-finite")
        optimizer.zero_grad(set_to_none=True)
        accumulated["total"].backward()
        if not all(
            bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        ):
            _diverged(step, "training gradients became non-finite")
        optimizer.step()

        batch = int(features.shape[0])
        for name in LOSS_TERMS:
            totals[name] += float(accumulated[name].detach().cpu()) * batch
        samples += batch
        if step not in CHECKPOINT_STEPS:
            continue

        window = {name: totals[name] / samples for name in LOSS_TERMS}
        history_record = {
            "optimizer_step": step,
            "stage_id": stage["stage_id"],
            "autoregressive_steps": rollout_steps,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_window": window,
        }
        history.append(history_record)
        checkpoint_path = _checkpoint_path(checkpoint_directory, step)
        payload = {
            "version": VERSION,
            "stage_id": stage["stage_id"],
            "autoregressive_steps": rollout_steps,
            "optimizer_step": step,
            "fine_tune_step": step,
            "architecture": architecture.to_dict(),
            "contract": str(resolved_contract),
            "contract_sha256": contract_sha,
            "loss": contract["loss"],
            "optimizer": {
                "name": "adam",
                "initial_learning_rate": float(
                    training["initial_learning_rate"]
                ),
                "adam_betas": [float(value) for value in training["adam_betas"]],
                "weight_decay": float(training["weight_decay"]),
                "batch_size": batch_size,
                "gradient_clipping": False,
            },
            "training_history_record": history_record,
            "model_state_dict": _checkpoint_state_dict(model),
        }
        torch.save(payload, checkpoint_path)
        checkpoints.append(
            {
                "stage_id": stage["stage_id"],
                "optimizer_step": step,
                "fine_tune_step": step,
                "checkpoint": str(
                    scratch / CHECKPOINT_DIRECTORY / checkpoint_path.name
                ),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
            }
        )
        totals = {name: 0.0 for name in LOSS_TERMS}
        samples = 0

    if len(checkpoints) != len(CHECKPOINT_STEPS):
        raise BireAlignedFullStateError("not every declared stage was checkpointed")
    del model, optimizer, loader, training_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    records = attribution.training_records(attribution_contract, split)
    initial = attribution.base._gather_states(state, records, 0)
    raw_static = attribution.base._gather_static(static, records)
    common = {
        "device": device,
        "initial": initial,
        "raw_static": raw_static,
        "experiments": records[:, 0],
        "state": state,
        "records": records,
        "wet": wet_array,
        "wind_mean": float(wind_mean),
        "wind_scale": float(wind_scale),
        "modes": np.arange(1, 31, dtype=np.float32),
    }
    source = _source_evaluation(contract["sources"])
    source_summary = summarize_evaluation(source)
    source_primary = source_summary["primary_10_to_90_rmse_ratio_to_persistence"]
    evaluated = []
    summaries = []
    for record in checkpoints:
        value = _evaluate_checkpoint(
            checkpoint_directory / Path(record["checkpoint"]).name,
            int(record["optimizer_step"]),
            normalization,
            **common,
        )
        evaluated.append(value)
        summary = summarize_evaluation(
            value,
            source_primary_ratios=source_primary,
            selection=contract["selection"],
        )
        summary["fine_tune_step"] = int(record["optimizer_step"])
        summary["stage_id"] = record["stage_id"]
        summaries.append(summary)

    published = {
        record["stage_id"]: {
            "optimizer_step": int(record["optimizer_step"]),
            "checkpoint": str(scratch / f"{record['stage_id']}.pt"),
            "checkpoint_sha256": None,
        }
        for record in checkpoints
    }
    for record in checkpoints:
        destination = scratch_tmp / f"{record['stage_id']}.pt"
        shutil.copy2(
            checkpoint_directory / Path(record["checkpoint"]).name,
            destination,
        )
        published[record["stage_id"]]["checkpoint_sha256"] = _file_sha256(
            destination
        )

    steps = np.asarray(
        [value["optimizer_step"] for value in summaries],
        dtype=np.int32,
    )
    arrays_path = scratch_tmp / ARRAYS_NAME
    np.savez_compressed(
        arrays_path,
        optimizer_steps=steps,
        lead_days=np.arange(10, 361, 10, dtype=np.int16),
        frozen_median_modewise_ratio=np.stack(
            [value["ratio"] for value in evaluated]
        ).astype(np.float32),
        integrated_energy_ratio=np.stack(
            [value["integrated"] for value in evaluated]
        ).astype(np.float32),
        primary_model_rmse=np.stack(
            [value["model_rmse"] for value in evaluated]
        ).astype(np.float32),
        primary_persistence_rmse=np.asarray(
            source["persistence_rmse"],
            dtype=np.float32,
        ),
        source_frozen_median_modewise_ratio=np.asarray(
            source["ratio"],
            dtype=np.float32,
        ),
        selection_records=records.astype(np.int32),
    )
    report = {
        "status": "complete",
        "version": VERSION,
        "contract": str(resolved_contract),
        "contract_sha256": contract_sha,
        "architecture": architecture.to_dict(),
        "external_input_channels": EXTERNAL_INPUT_CHANNELS,
        "lifting_input_channels": LIFTING_INPUT_CHANNELS,
        "retained_static_features": list(RETAINED_STATIC_FEATURES),
        "removed_static_features": [
            name
            for name in STATIC_FEATURES
            if name not in RETAINED_STATIC_FEATURES
        ],
        "positional_encoding": "oceanfourcast_PosEmbed_sine_cosine_before_lifting",
        "external_local_branch": False,
        "layer_norm_modules": architecture.layer_norm_count,
        "parameter_count": parameter_count,
        "loss": contract["loss"],
        "optimizer": {
            "name": "adam",
            "initial_learning_rate": float(training["initial_learning_rate"]),
            "adam_betas": [float(value) for value in training["adam_betas"]],
            "weight_decay": float(training["weight_decay"]),
            "batch_size": batch_size,
            "gradient_clipping": False,
            "decay_step": decay_step,
            "decay_factor": float(training["decay_factor"]),
        },
        "stages": [dict(record) for record in stages],
        "training_history": history,
        "checkpoints": checkpoints,
        "published_checkpoints": published,
        "source_summary": source_summary,
        "evaluation_summaries": summaries,
        "gate_note": (
            "both stage checkpoints are retained and reported; neither is "
            "promoted here and the held S0 figure suite publishes both"
        ),
        "arrays": str(scratch / ARRAYS_NAME),
        "arrays_sha256": _file_sha256(arrays_path),
        "read_contract": contract["read_contract"],
        "validation_state_opened": False,
        "inference_state_opened": False,
        "held_s0_state_opened": False,
        "response_or_adjoint_state_opened": False,
        "elapsed_seconds": time.monotonic() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    report["content_sha256"] = _json_sha256(report)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (scratch_tmp / REPORT_NAME).write_text(rendered)
    (project_tmp / REPORT_NAME).write_text(rendered)
    shutil.copy2(arrays_path, project_tmp / ARRAYS_NAME)
    _plot(
        project_tmp / FIGURE_NAME,
        [{**source_summary, "fine_tune_step": 0, "stage_id": "source"}, *summaries],
    )
    (project_tmp / README_NAME).write_text(_readme(report))
    artifacts = {
        name: _file_sha256(project_tmp / name)
        for name in (REPORT_NAME, ARRAYS_NAME, FIGURE_NAME, README_NAME)
    }
    manifest = {
        "status": "complete",
        "version": VERSION,
        "contract_sha256": contract_sha,
        "artifacts": artifacts,
        "content_sha256": _json_sha256(artifacts),
        "inference_state_opened": False,
        "held_s0_state_opened": False,
        "response_or_adjoint_state_opened": False,
    }
    (project_tmp / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(scratch_tmp, scratch)
    os.replace(project_tmp, project)
    return report


def _readme(report: Mapping[str, Any]) -> str:
    lines = [
        "# Bire-aligned full-state FNO (training split only)",
        "",
        "Three-block, width-128 FNO over the retained 46-channel closed state.",
        "49 external inputs (46 state + wind stress + wet mask + distance to",
        "wall); the two deterministic Bire sine/cosine position channels are",
        "appended immediately before lifting, giving 51 lifting inputs. Six",
        "pointwise channel LayerNorms, no external 3x3 raw-input branch, and",
        "the pointwise residual retained inside each spectral block.",
        "",
        "Training is Bire's two-stage protocol under `MSE + 0.01 MAE` on wet",
        f"cells: {PRETRAIN_STEPS} one-step pretraining updates then",
        f"{FINE_TUNE_STEPS} two-step autoregressive fine-tuning updates, with",
        "Adam(1e-2, betas 0.9/0.95, weight decay 0) at batch size 8 and no",
        "gradient clipping.",
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
