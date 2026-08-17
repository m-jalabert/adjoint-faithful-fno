"""The production operator: ``F_theta: [x_t, S] -> x_{t+10}``.

One architecture, one builder, one unroll, one evaluation adapter. The map is

    x_t = [U_1:15, V_1:15, Theta_1:15, eta]            46 channels
    S   = [tau_x, wet, f, dx, theta_clim]               5 channels
                                                       --
    external input                                     51 channels
    + two deterministic Bire position channels          2
                                                       --
    lifting input                                      53 channels
    output x_{t+10}                                    46 channels

A 32x32-mode, width-128, three-block FNO with six pointwise LayerNorms and a
4C Channel MLP carries the global path; a bias-free 3x3 convolution from the raw
51-channel input carries a local correction and is initialized to exact zeros,
so at construction the FNO alone defines the map and the correction is learned
from nothing. Prediction is of the **state**, not of a residual, and rollout
feeds the model its own output from the second call onward.

There is no checkpoint inheritance anywhere in this module: no migration, no
parent architecture, no function-preserving load. The model is built randomly
initialized and trained once.
"""
from __future__ import annotations

import math
from typing import Any
from dataclasses import asdict, dataclass

import numpy as np

from .runtime import (
    FNO,
    STATE_CHANNEL_COUNT,
    STORE_STATIC_COUNT,
    functional,
    nn,
    require_runtime,
    torch,
)
from .dataset import STATIC_CHANNEL_COUNT, STATIC_FEATURES

POSITIONAL_CHANNELS = 2

EXTERNAL_INPUT_CHANNELS = STATE_CHANNEL_COUNT + STATIC_CHANNEL_COUNT

LIFTING_INPUT_CHANNELS = EXTERNAL_INPUT_CHANNELS + POSITIONAL_CHANNELS

#: Layout of the 51-channel external block, in the one published order.
STATE_SLICE = slice(0, STATE_CHANNEL_COUNT)

STATIC_SLICE = slice(STATE_CHANNEL_COUNT, EXTERNAL_INPUT_CHANNELS)

#: The exact parameter count the declared architecture must produce. Asserted at
#: preflight and in the test suite so a silent NeuralOperator layout change
#: cannot alter the model without anyone noticing.
EXPECTED_PARAMETER_COUNT = 27_297_960

README_NAME = "README.md"

MANIFEST_NAME = "manifest.json"

CHECKPOINT_DIRECTORY = "training_checkpoints"


class ArchitectureError(RuntimeError):
    """Raised when the production architecture declaration is violated."""


class DivergenceError(ArchitectureError):
    """Raised when the declared learning rate produces a non-finite update."""


@dataclass(frozen=True)
class ProductionArchitecture:
    """The frozen 1-in / 1-out 32x32 operator.

    ``in_channels`` counts the *external* inputs only. The two deterministic
    Bire position channels are appended inside the model, immediately before
    lifting, so the lifting network sees 53.
    """

    in_channels: int = EXTERNAL_INPUT_CHANNELS
    out_channels: int = STATE_CHANNEL_COUNT
    input_states: int = 1
    static_channels: tuple[str, ...] = STATIC_FEATURES
    positional_channels: int = POSITIONAL_CHANNELS
    lifting_in_channels: int = LIFTING_INPUT_CHANNELS
    grid_shape: tuple[int, int] = (62, 62)
    n_modes: tuple[int, int] = (32, 32)
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
    local_kernel_size: int = 3
    local_bias: bool = False
    fno_block_precision: str = "full"
    factorization: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_modes", tuple(int(v) for v in self.n_modes))
        object.__setattr__(self, "grid_shape", tuple(int(v) for v in self.grid_shape))
        object.__setattr__(
            self, "static_channels", tuple(str(v) for v in self.static_channels)
        )
        if self.static_channels != STATIC_FEATURES:
            raise ArchitectureError(
                f"the production arm declares exactly {list(STATIC_FEATURES)}"
            )
        if self.input_states != 1:
            raise ArchitectureError(
                "the production arm reads one time level; there is no history channel"
            )
        if (
            self.in_channels != EXTERNAL_INPUT_CHANNELS
            or self.out_channels != STATE_CHANNEL_COUNT
            or self.positional_channels != POSITIONAL_CHANNELS
            or self.lifting_in_channels != LIFTING_INPUT_CHANNELS
        ):
            raise ArchitectureError(
                "the production arm maps 46 state + 5 static (+2 position) -> 46"
            )
        if self.grid_shape != (62, 62) or self.n_modes != (32, 32):
            raise ArchitectureError(
                "the production arm keeps the 62x62 grid and the 32x32 Y,X modes"
            )
        if self.n_layers != 3:
            raise ArchitectureError("the production arm uses exactly three FNO blocks")
        if self.hidden_channels != 128 or self.channel_mlp_expansion != 4.0:
            raise ArchitectureError(
                "the production arm keeps width 128 and the 4C Channel MLP"
            )
        if self.lifting_channel_ratio != 2 or self.projection_channel_ratio != 2:
            raise ArchitectureError(
                "the production arm keeps lifting and projection width 256"
            )
        if self.domain_padding != 0.1 or not self.use_channel_mlp:
            raise ArchitectureError(
                "the production arm keeps 10% padding and the Channel MLP"
            )
        if self.channel_mlp_dropout != 0.0:
            raise ArchitectureError("the production arm runs without dropout")
        if self.positional_embedding is not None:
            raise ArchitectureError(
                "position enters only through the Bire sine/cosine encoder"
            )
        if not self.pointwise_layer_norm:
            raise ArchitectureError(
                "the production arm normalizes channel-wise after both mixing operations"
            )
        if self.local_kernel_size != 3 or self.local_bias:
            raise ArchitectureError(
                "the production arm keeps exactly one bias-free 3x3 local correction"
            )
        if self.fno_block_precision != "full" or self.factorization is not None:
            raise ArchitectureError("the production arm remains a dense float32 FNO")

    @property
    def layer_norm_count(self) -> int:
        return 2 * self.n_layers

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_modes"] = list(self.n_modes)
        result["grid_shape"] = list(self.grid_shape)
        result["static_channels"] = list(self.static_channels)
        return result


if nn is not None:

    class PointwiseChannelLayerNorm(nn.Module):
        """LayerNorm over channels independently at every grid point."""

        def __init__(self, channels: int, epsilon: float = 1.0e-5) -> None:
            super().__init__()
            self.channels = int(channels)
            self.epsilon = float(epsilon)
            self.weight = nn.Parameter(torch.ones(self.channels))
            self.bias = nn.Parameter(torch.zeros(self.channels))

        def forward(self, value: Any) -> Any:
            if value.ndim != 4 or value.shape[1] != self.channels:
                raise ValueError("pointwise LayerNorm expects N,C,Y,X")
            channels_last = value.permute(0, 2, 3, 1)
            normalized = functional.layer_norm(
                channels_last,
                (self.channels,),
                self.weight,
                self.bias,
                self.epsilon,
            )
            return normalized.permute(0, 3, 1, 2)

    class BirePositionalEncoding(nn.Module):
        """Deterministic ``oceanfourcast.PosEmbed`` position channels.

        Each direction contributes one channel whose entries alternate between
        sine and cosine of ``pi/(2 N) * index``::

            p(j) = sin(w_j) for even j,  cos(w_j) for odd j.

        The two fields are concatenated after the external inputs, in the
        published order ``(x, p_x, p_y)``, immediately before lifting. They are
        constants, so they are registered as non-persistent buffers and never
        enter a checkpoint.
        """

        def __init__(self, height: int, width: int) -> None:
            super().__init__()
            self.height = int(height)
            self.width = int(width)
            self.register_buffer(
                "position_x",
                self._axis(self.width)
                .view(1, 1, 1, self.width)
                .expand(1, 1, self.height, self.width)
                .contiguous(),
                persistent=False,
            )
            self.register_buffer(
                "position_y",
                self._axis(self.height)
                .view(1, 1, self.height, 1)
                .expand(1, 1, self.height, self.width)
                .contiguous(),
                persistent=False,
            )

        @staticmethod
        def _axis(size: int) -> Any:
            frequency = math.pi / (2 * size) * torch.arange(size, dtype=torch.float32)
            grid = torch.zeros(size, dtype=torch.float32)
            grid[::2] = torch.sin(frequency[::2])
            grid[1::2] = torch.cos(frequency[1::2])
            return grid

        def forward(self, value: Any) -> Any:
            if value.ndim != 4 or value.shape[-2:] != (self.height, self.width):
                raise ValueError("the Bire position encoder expects N,C,62,62")
            batch = value.shape[0]
            return torch.cat(
                (
                    value,
                    self.position_x.expand(batch, -1, -1, -1),
                    self.position_y.expand(batch, -1, -1, -1),
                ),
                dim=1,
            )

    class ProductionFNO(nn.Module):
        """The 32x32 three-block FNO with its zero-initialized local correction.

        The pointwise residual ``W h`` around each truncated spectral operator is
        NeuralOperator's default ``fno_skip="linear"`` path and is retained. The
        external ``local`` convolution is bias-free and starts at exact zeros, so
        every parameter that is not the FNO's own random initialization begins
        with no contribution at all.
        """

        def __init__(self, architecture: ProductionArchitecture) -> None:
            super().__init__()
            if not isinstance(architecture, ProductionArchitecture):
                raise ArchitectureError(
                    "the production model requires ProductionArchitecture"
                )
            self.architecture = architecture
            self.positional_encoding = BirePositionalEncoding(*architecture.grid_shape)
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
            self.local = nn.Conv2d(
                architecture.in_channels,
                architecture.out_channels,
                kernel_size=architecture.local_kernel_size,
                padding=architecture.local_kernel_size // 2,
                bias=architecture.local_bias,
            )
            nn.init.zeros_(self.local.weight)

        def forward(self, features: Any) -> Any:
            if features.ndim != 4 or features.shape[1] != self.architecture.in_channels:
                raise ValueError(
                    f"the production model expects N,{self.architecture.in_channels},Y,X"
                )
            global_state = self.fno(self.positional_encoding(features))
            return global_state + self.local(features)

else:  # pragma: no cover - environment dependent
    PointwiseChannelLayerNorm = None  # type: ignore[assignment,misc]
    BirePositionalEncoding = None  # type: ignore[assignment,misc]
    ProductionFNO = None  # type: ignore[assignment,misc]


def build_model(architecture: ProductionArchitecture) -> Any:
    """Build the randomly-initialized production operator."""

    require_runtime()
    if ProductionFNO is None:  # pragma: no cover - environment dependent
        raise RuntimeError("the production model requires PyTorch")
    return ProductionFNO(architecture)


def state_unroll(model: Any, features: Any, wet: Any, steps: int) -> Any:
    """Autoregress the direct state map, feeding the model its own output.

    ``features`` carries ``(x_t, static)``. Only the first call sees truth; from
    the second onward the input state is the previous prediction, so training is
    exposed to its own accumulated error rather than teacher-forced.
    """

    if steps <= 0:
        raise ValueError("the rollout needs at least one step")
    declared = getattr(getattr(model, "architecture", None), "in_channels", None)
    if declared is not None and features.shape[1] != declared:
        raise ValueError(f"the unroll expects N,{declared},Y,X features")
    current = features[:, STATE_SLICE]
    static = features[:, STATE_CHANNEL_COUNT:]
    predictions = []
    for _ in range(steps):
        current = model(torch.cat((current, static), dim=1)) * wet
        predictions.append(current)
    return torch.stack(predictions, dim=1)


@dataclass
class ProductionStepper:
    """Evaluation adapter: physical state in, one ten-day step, physical out.

    The five static channels do not all exist in the trajectory store, so the
    normalized block is built once by :func:`oceanfno.dataset.static_block` and
    handed here; the shared 360-day and 2,000-day rollout loops call
    :meth:`normalized_state`, :meth:`normalized_static`, :meth:`step` and
    :meth:`physical` and need to know nothing else.
    """

    model: Any
    device: Any
    wet: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    statics: np.ndarray

    def __post_init__(self) -> None:
        block = np.ascontiguousarray(self.statics, dtype=np.float32)
        if (
            block.ndim != 4
            or block.shape[1] != STATIC_CHANNEL_COUNT
            or block.shape[2:] != tuple(self.wet.shape)
        ):
            raise ArchitectureError(
                "the stepper needs an (experiments, 5, Y, X) physical static block"
            )
        self.statics = block
        self._wet = torch.from_numpy(self.wet.astype(np.float32))[None, None].to(
            self.device
        )

    def normalized_state(self, physical: np.ndarray) -> Any:
        value = (physical - self.mean[None]) / self.scale[None]
        value[:, :, ~self.wet] = 0.0
        return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(
            device=self.device, dtype=torch.float32
        )

    def normalized_static(self, static: Any, experiments: np.ndarray) -> Any:
        """Select the derived static block for each member's regime.

        ``static`` is the store's own array, which the shared loops still pass.
        It is checked for shape and then superseded: three of the five production
        channels are not in the store at all.
        """

        if static is not None and int(static.shape[1]) != STORE_STATIC_COUNT:
            raise ArchitectureError(
                "the store's static contract changed underneath the derived block"
            )
        indices = np.asarray(experiments, dtype=np.int64)
        if indices.size and int(indices.max()) >= self.statics.shape[0]:
            raise ArchitectureError(
                "a requested regime is outside the derived static block"
            )
        value = np.stack([self.statics[int(index)] for index in indices])
        return torch.from_numpy(np.ascontiguousarray(value)).to(
            device=self.device, dtype=torch.float32
        )

    def step(self, current: Any, static: Any) -> Any:
        return self.model(torch.cat((current, static), dim=1)) * self._wet

    def physical(self, normalized: Any) -> np.ndarray:
        value = normalized.detach().cpu().numpy()
        value = value * self.scale[None] + self.mean[None]
        value[:, :, ~self.wet] = 0.0
        return np.ascontiguousarray(value, dtype=np.float32)


def parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))
