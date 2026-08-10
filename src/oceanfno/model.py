"""Canonical 32x32 Model C and its explicit checkpoint ancestors.

The active emulator is :class:`BireY32FullStateFNO` built from
:class:`BireY32X32Architecture`. The retained 32x24 Y32 model, the 24x24 local
model and the original 24x16 Bire-aligned model remain here only so archived
checkpoints can be migrated and used as literal comparators.
"""
from __future__ import annotations

import math
from typing import Any, Mapping
from dataclasses import asdict, dataclass

import numpy as np

from .runtime import FNO, functional, nn, torch
from .runtime import STATE_CHANNEL_COUNT, STATIC_FEATURES, require_model_a_runtime
from .dataset import ModelCSuccessorArchitecture

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

#: The two-input arm hands the operator two consecutive time levels instead of
#: one.  ``x_{t-10}`` is prepended to ``x_t``, so the external block becomes
#: ``46 + 46 + 3 = 95`` and lifting sees ``97``.  The lag is the prediction
#: interval itself, which is what makes ``(x_t - x_{t-10}) / 10 days`` an
#: empirical tendency the operator can read.
INPUT_STATE_COUNT = 2

INPUT_LAG_DAYS = 10

TWO_IN_EXTERNAL_INPUT_CHANNELS = (
    INPUT_STATE_COUNT * STATE_CHANNEL_COUNT + len(RETAINED_STATIC_FEATURES)
)

TWO_IN_LIFTING_INPUT_CHANNELS = (
    TWO_IN_EXTERNAL_INPUT_CHANNELS + POSITIONAL_CHANNELS
)

#: Channel layout of the 95-channel external block, in the one published order.
HISTORY_SLICE = slice(0, STATE_CHANNEL_COUNT)

PRESENT_SLICE = slice(STATE_CHANNEL_COUNT, 2 * STATE_CHANNEL_COUNT)

TWO_IN_STATIC_SLICE = slice(
    2 * STATE_CHANNEL_COUNT, TWO_IN_EXTERNAL_INPUT_CHANNELS
)

README_NAME = "README.md"

MANIFEST_NAME = "manifest.json"

CHECKPOINT_DIRECTORY = "training_checkpoints"

class BireAlignedFullStateError(RuntimeError):
    """Raised when the Bire-aligned full-state arm violates its contract."""

class BireAlignedDivergenceError(BireAlignedFullStateError):
    """Raised when the declared learning rate produces a non-finite update."""

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


@dataclass(frozen=True)
class BireLocal24Architecture:
    """The local-correction successor to :class:`BireAlignedArchitecture`.

    NeuralOperator consumes modes in tensor spatial order.  The model tensors
    are ``N,C,Y,X``, so ``(24, 24)`` retains 24 modes in both the meridional and
    zonal directions.  Keeping the parent's 24 meridional modes while expanding
    the zonal side from 16 to 24 makes a lossless checkpoint migration possible:
    the four newly represented real-FFT coefficients can begin at zero.

    ``local_kernel_size`` declares a raw-external-input correction in parallel
    with the FNO.  Its implementation is deliberately fixed to a bias-free,
    zero-initialized 49-to-46 convolution, so constructing this architecture
    cannot perturb the parent map before fine-tuning.
    """

    in_channels: int = EXTERNAL_INPUT_CHANNELS
    out_channels: int = STATE_CHANNEL_COUNT
    positional_channels: int = POSITIONAL_CHANNELS
    lifting_in_channels: int = LIFTING_INPUT_CHANNELS
    grid_shape: tuple[int, int] = (62, 62)
    n_modes: tuple[int, int] = (24, 24)
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
                "the local-24 arm maps 46 state + 3 static (+2 position) -> 46"
            )
        if self.grid_shape != (62, 62) or self.n_modes != (24, 24):
            raise BireAlignedFullStateError(
                "the local-24 arm keeps the 62x62 grid and 24x24 Y,X modes"
            )
        if self.n_layers != 3:
            raise BireAlignedFullStateError(
                "the local-24 arm uses exactly three FNO blocks"
            )
        if self.hidden_channels != 128 or self.channel_mlp_expansion != 4.0:
            raise BireAlignedFullStateError(
                "the local-24 arm keeps width 128 and the 4C Channel MLP"
            )
        if self.lifting_channel_ratio != 2 or self.projection_channel_ratio != 2:
            raise BireAlignedFullStateError(
                "the local-24 arm keeps lifting and projection width 256"
            )
        if self.domain_padding != 0.1 or not self.use_channel_mlp:
            raise BireAlignedFullStateError(
                "the local-24 arm keeps 10% padding and the Channel MLP"
            )
        if self.channel_mlp_dropout != 0.0:
            raise BireAlignedFullStateError(
                "the local-24 arm freezes Channel MLP dropout at zero"
            )
        if self.positional_embedding is not None:
            raise BireAlignedFullStateError(
                "position enters only through the Bire sine/cosine encoder"
            )
        if not self.pointwise_layer_norm:
            raise BireAlignedFullStateError(
                "the local-24 arm normalizes channel-wise after both mixing operations"
            )
        if self.local_kernel_size != 3:
            raise BireAlignedFullStateError(
                "the local-24 arm uses exactly one external 3x3 correction"
            )
        if self.fno_block_precision != "full" or self.factorization is not None:
            raise BireAlignedFullStateError(
                "the local-24 arm remains a dense float32 FNO"
            )

    @property
    def layer_norm_count(self) -> int:
        return 2 * self.n_layers

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_modes"] = list(self.n_modes)
        result["grid_shape"] = list(self.grid_shape)
        return result


@dataclass(frozen=True)
class BireY32Architecture:
    """Canonical Model C: local correction with 32x24 Y,X modes."""

    in_channels: int = EXTERNAL_INPUT_CHANNELS
    out_channels: int = STATE_CHANNEL_COUNT
    positional_channels: int = POSITIONAL_CHANNELS
    lifting_in_channels: int = LIFTING_INPUT_CHANNELS
    grid_shape: tuple[int, int] = (62, 62)
    n_modes: tuple[int, int] = (32, 24)
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
                "the canonical Y32 model maps 46 state + 3 static (+2 position) -> 46"
            )
        if self.grid_shape != (62, 62) or self.n_modes != (32, 24):
            raise BireAlignedFullStateError(
                "the canonical Y32 model keeps the 62x62 grid and uses 32x24 Y,X modes"
            )
        if self.n_layers != 3:
            raise BireAlignedFullStateError(
                "the canonical Y32 model uses exactly three FNO blocks"
            )
        if self.hidden_channels != 128 or self.channel_mlp_expansion != 4.0:
            raise BireAlignedFullStateError(
                "the canonical Y32 model keeps width 128 and the 4C Channel MLP"
            )
        if self.lifting_channel_ratio != 2 or self.projection_channel_ratio != 2:
            raise BireAlignedFullStateError(
                "the canonical Y32 model keeps lifting and projection width 256"
            )
        if self.domain_padding != 0.1 or not self.use_channel_mlp:
            raise BireAlignedFullStateError(
                "the canonical Y32 model keeps 10% padding and the Channel MLP"
            )
        if self.channel_mlp_dropout != 0.0:
            raise BireAlignedFullStateError(
                "the canonical Y32 model freezes Channel MLP dropout at zero"
            )
        if self.positional_embedding is not None:
            raise BireAlignedFullStateError(
                "position enters only through the Bire sine/cosine encoder"
            )
        if not self.pointwise_layer_norm:
            raise BireAlignedFullStateError(
                "the canonical Y32 model normalizes channel-wise after both mixing operations"
            )
        if self.local_kernel_size != 3:
            raise BireAlignedFullStateError(
                "the canonical Y32 model retains exactly one external 3x3 correction"
            )
        if self.fno_block_precision != "full" or self.factorization is not None:
            raise BireAlignedFullStateError(
                "the canonical Y32 model remains a dense float32 FNO"
            )

    @property
    def layer_norm_count(self) -> int:
        return 2 * self.n_layers

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_modes"] = list(self.n_modes)
        result["grid_shape"] = list(self.grid_shape)
        return result


@dataclass(frozen=True)
class BireY32X32Architecture:
    """Canonical Model C with the zonal band opened to 32x32 Y,X modes.

    Field for field the retained Y32 model, including the deterministic
    sine/cosine position encoder and the bias-free local 3x3 correction. Only
    ``n_modes`` moves, from 32x24 to 32x32.
    """

    in_channels: int = EXTERNAL_INPUT_CHANNELS
    out_channels: int = STATE_CHANNEL_COUNT
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
                "the 32x32 model maps 46 state + 3 static (+2 position) -> 46"
            )
        if self.grid_shape != (62, 62) or self.n_modes != (32, 32):
            raise BireAlignedFullStateError(
                "the 32x32 model keeps the 62x62 grid and uses 32x32 Y,X modes"
            )
        if self.n_layers != 3:
            raise BireAlignedFullStateError(
                "the 32x32 model uses exactly three FNO blocks"
            )
        if self.hidden_channels != 128 or self.channel_mlp_expansion != 4.0:
            raise BireAlignedFullStateError(
                "the 32x32 model keeps width 128 and the 4C Channel MLP"
            )
        if self.lifting_channel_ratio != 2 or self.projection_channel_ratio != 2:
            raise BireAlignedFullStateError(
                "the 32x32 model keeps lifting and projection width 256"
            )
        if self.domain_padding != 0.1 or not self.use_channel_mlp:
            raise BireAlignedFullStateError(
                "the 32x32 model keeps 10% padding and the Channel MLP"
            )
        if self.channel_mlp_dropout != 0.0:
            raise BireAlignedFullStateError(
                "the 32x32 model freezes Channel MLP dropout at zero"
            )
        if self.positional_embedding is not None:
            raise BireAlignedFullStateError(
                "position enters only through the Bire sine/cosine encoder"
            )
        if not self.pointwise_layer_norm:
            raise BireAlignedFullStateError(
                "the 32x32 model normalizes channel-wise after both mixing operations"
            )
        if self.local_kernel_size != 3:
            raise BireAlignedFullStateError(
                "the 32x32 model retains exactly one external 3x3 correction"
            )
        if self.fno_block_precision != "full" or self.factorization is not None:
            raise BireAlignedFullStateError(
                "the 32x32 model remains a dense float32 FNO"
            )

    @property
    def layer_norm_count(self) -> int:
        return 2 * self.n_layers

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_modes"] = list(self.n_modes)
        result["grid_shape"] = list(self.grid_shape)
        return result

@dataclass(frozen=True)
class BireTwoInOneOutArchitecture:
    """Canonical Model C given two consecutive time levels instead of one.

    Field for field the 32x32 model, including the 32x32 modes, the
    deterministic sine/cosine position encoder and the bias-free local 3x3
    correction. Only the *input* contract moves: the operator is handed
    ``(x_{t-10}, x_t)`` rather than ``x_t`` alone, so the external block grows
    from 49 to 95 channels and lifting from 51 to 97. The output is still one
    state, ten days ahead, and autoregression slides the pair forward::

        (x_{t-10}, x_t) -> x_{t+10},  (x_t, x_{t+10}) -> x_{t+20},  ...

    This is the multistep analogue of Adams--Bashforth in spirit, not in
    algebra: MITgcm's AB-II extrapolates from two stored *tendencies*, whereas
    the pair here lets the operator infer one empirical tendency,
    ``(x_t - x_{t-10}) / 10 days``.
    """

    in_channels: int = TWO_IN_EXTERNAL_INPUT_CHANNELS
    out_channels: int = STATE_CHANNEL_COUNT
    input_states: int = INPUT_STATE_COUNT
    input_lag_days: int = INPUT_LAG_DAYS
    positional_channels: int = POSITIONAL_CHANNELS
    lifting_in_channels: int = TWO_IN_LIFTING_INPUT_CHANNELS
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
    fno_block_precision: str = "full"
    factorization: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "n_modes", tuple(int(value) for value in self.n_modes)
        )
        object.__setattr__(
            self, "grid_shape", tuple(int(value) for value in self.grid_shape)
        )
        if self.input_states != INPUT_STATE_COUNT or (
            self.input_lag_days != INPUT_LAG_DAYS
        ):
            raise BireAlignedFullStateError(
                "the two-input arm reads exactly two states ten days apart"
            )
        if (
            self.in_channels != TWO_IN_EXTERNAL_INPUT_CHANNELS
            or self.out_channels != STATE_CHANNEL_COUNT
            or self.positional_channels != POSITIONAL_CHANNELS
            or self.lifting_in_channels != TWO_IN_LIFTING_INPUT_CHANNELS
        ):
            raise BireAlignedFullStateError(
                "the two-input arm maps 2x46 state + 3 static (+2 position) -> 46"
            )
        if self.grid_shape != (62, 62) or self.n_modes != (32, 32):
            raise BireAlignedFullStateError(
                "the two-input arm keeps the 62x62 grid and the 32x32 Y,X modes"
            )
        if self.n_layers != 3:
            raise BireAlignedFullStateError(
                "the two-input arm uses exactly three FNO blocks"
            )
        if self.hidden_channels != 128 or self.channel_mlp_expansion != 4.0:
            raise BireAlignedFullStateError(
                "the two-input arm keeps width 128 and the 4C Channel MLP"
            )
        if self.lifting_channel_ratio != 2 or self.projection_channel_ratio != 2:
            raise BireAlignedFullStateError(
                "the two-input arm keeps lifting and projection width 256"
            )
        if self.domain_padding != 0.1 or not self.use_channel_mlp:
            raise BireAlignedFullStateError(
                "the two-input arm keeps 10% padding and the Channel MLP"
            )
        if self.channel_mlp_dropout != 0.0:
            raise BireAlignedFullStateError(
                "the two-input arm freezes Channel MLP dropout at zero"
            )
        if self.positional_embedding is not None:
            raise BireAlignedFullStateError(
                "position enters only through the Bire sine/cosine encoder"
            )
        if not self.pointwise_layer_norm:
            raise BireAlignedFullStateError(
                "the two-input arm normalizes channel-wise after both mixing operations"
            )
        if self.local_kernel_size != 3:
            raise BireAlignedFullStateError(
                "the two-input arm retains exactly one external 3x3 correction"
            )
        if self.fno_block_precision != "full" or self.factorization is not None:
            raise BireAlignedFullStateError(
                "the two-input arm remains a dense float32 FNO"
            )

    @property
    def layer_norm_count(self) -> int:
        return 2 * self.n_layers

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_modes"] = list(self.n_modes)
        result["grid_shape"] = list(self.grid_shape)
        return result

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


def retained_two_in_features(batch: Any) -> Any:
    """Drop the two linear coordinate channels from a 97-channel pair batch.

    The dataset stacks ``(x_{t-10}, x_t)`` ahead of all five static fields, so
    the reduction is the single-state one with the static block found 46
    channels further along; both state blocks are kept whole.
    """

    states = INPUT_STATE_COUNT * STATE_CHANNEL_COUNT
    if batch.ndim != 4 or batch.shape[1] != states + len(STATIC_FEATURES):
        raise ValueError("the two-input arm reduces N,97,Y,X inputs")
    index = torch.as_tensor(
        [states + offset for offset in RETAINED_STATIC_INDICES],
        dtype=torch.long,
        device=batch.device,
    )
    return torch.cat(
        (batch[:, :states], batch.index_select(1, index)),
        dim=1,
    )

ARM_IDS = ("layernorm", "dropout", "layernorm_dropout")

@dataclass(frozen=True)
class RegularizationArm:
    """One factorial arm in the frozen LayerNorm/dropout comparison."""

    arm_id: str
    pointwise_layer_norm: bool
    channel_mlp_dropout: float

    def __post_init__(self) -> None:
        if self.arm_id not in ARM_IDS:
            raise ValueError("unknown Bire regularization arm")
        if self.channel_mlp_dropout not in (0.0, 0.5):
            raise ValueError("dropout control must use zero or archived default 0.5")

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


    class BireRegularizedDirectFNO(nn.Module):
        """The fixed width-128 Model C with explicit Bire regularizers."""

        def __init__(
            self,
            architecture: ModelCSuccessorArchitecture,
            arm: RegularizationArm,
        ) -> None:
            super().__init__()
            self.architecture = architecture
            self.arm = arm
            self.fno = FNO(
                n_modes=architecture.n_modes,
                in_channels=architecture.in_channels,
                out_channels=architecture.out_channels,
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
            if arm.pointwise_layer_norm:
                self.fno.fno_blocks.norm = nn.ModuleList(
                    [
                        PointwiseChannelLayerNorm(
                            architecture.hidden_channels,
                        )
                        for _ in range(architecture.n_layers * 2)
                    ]
                )
            self.local = nn.Conv2d(
                architecture.in_channels,
                architecture.out_channels,
                kernel_size=architecture.local_kernel_size,
                padding=architecture.local_kernel_size // 2,
            )

        def forward(self, features: Any) -> Any:
            if (
                features.ndim != 4
                or features.shape[1] != self.architecture.in_channels
            ):
                raise ValueError("regularized Model C expects N,51,Y,X")
            return self.fno(features) + self.local(features)

else:  # pragma: no cover
    PointwiseChannelLayerNorm = None  # type: ignore[assignment,misc]
    BireRegularizedDirectFNO = None  # type: ignore[assignment,misc]

def direct_state_unroll(
    model: Any,
    features: Any,
    wet: Any,
    steps: int,
) -> Any:
    """Autoregress a direct normalized future-state map."""

    if steps <= 0:
        raise ValueError("direct-state rollout needs at least one step")
    current = features[:, :STATE_CHANNEL_COUNT]
    static = features[:, STATE_CHANNEL_COUNT:]
    predictions = []
    for _ in range(steps):
        current = model(torch.cat((current, static), dim=1)) * wet
        predictions.append(current)
    return torch.stack(predictions, dim=1)


def two_in_state_unroll(
    model: Any,
    features: Any,
    wet: Any,
    steps: int,
) -> Any:
    """Autoregress the two-time-level map, sliding the pair forward.

    ``features`` carries ``(x_{t-10}, x_t, static)``. The first call is the only
    one with two truth states; from the second call onward the pair is
    ``(previous prediction, latest prediction)``, so there is no teacher forcing
    after the initial condition --- exactly the exposure the one-input arm had.
    """

    if steps <= 0:
        raise ValueError("two-input rollout needs at least one step")
    if features.shape[1] != TWO_IN_EXTERNAL_INPUT_CHANNELS:
        raise ValueError("the two-input unroll expects N,95,Y,X features")
    previous = features[:, HISTORY_SLICE]
    current = features[:, PRESENT_SLICE]
    static = features[:, TWO_IN_STATIC_SLICE]
    predictions = []
    for _ in range(steps):
        future = model(torch.cat((previous, current, static), dim=1)) * wet
        predictions.append(future)
        previous, current = current, future
    return torch.stack(predictions, dim=1)

@dataclass
class PointwiseDirectStepper:
    """Evaluation adapter for a direct map with pointwise normalization."""

    model: Any
    device: Any
    wet: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    wind_mean: float
    wind_scale: float

    #: One-input maps need no history, so shared evaluation code can ask before
    #: paying for a second truth read.
    requires_history = False

    def begin(self, history: np.ndarray | None = None) -> None:
        """Accept the optional ``t - 10`` physical state; this map ignores it."""

        return None

    def normalized_state(self, physical: np.ndarray) -> Any:
        value = (physical - self.mean[None]) / self.scale[None]
        value[:, :, ~self.wet] = 0.0
        return torch.from_numpy(
            np.ascontiguousarray(value, dtype=np.float32)
        ).to(device=self.device, dtype=torch.float32)

    def normalized_static(self, static: Any, experiments: np.ndarray) -> Any:
        value = np.stack(
            [
                np.asarray(static[int(experiment)], dtype=np.float32)
                for experiment in experiments
            ]
        )
        value[:, 0] = (value[:, 0] - self.wind_mean) / self.wind_scale
        value[:, 0, ~self.wet] = 0.0
        return torch.from_numpy(np.ascontiguousarray(value)).to(
            device=self.device,
            dtype=torch.float32,
        )

    def step(self, current: Any, static: Any) -> Any:
        wet = torch.from_numpy(self.wet.astype(np.float32))[None, None].to(
            self.device
        )
        return self.model(torch.cat((current, static), dim=1)) * wet

    def physical(self, normalized: Any) -> np.ndarray:
        value = normalized.detach().cpu().numpy()
        value = value * self.scale[None] + self.mean[None]
        value[:, :, ~self.wet] = 0.0
        return np.ascontiguousarray(value, dtype=np.float32)

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


    class BireLocal24FullStateFNO(nn.Module):
        """The Bire FNO with 24x24 modes and a local raw-input correction.

        The global path deliberately has the same module names as
        :class:`BireAlignedFullStateFNO`.  That makes every compatible tensor
        identifiable by name during migration.  ``local`` is bias-free and its
        only parameter is initialized to exact zeros, so it initially adds no
        correction at all.
        """

        def __init__(self, architecture: BireLocal24Architecture) -> None:
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
            self.local = nn.Conv2d(
                architecture.in_channels,
                architecture.out_channels,
                kernel_size=architecture.local_kernel_size,
                padding=architecture.local_kernel_size // 2,
                bias=False,
            )
            nn.init.zeros_(self.local.weight)

        def forward(self, features: Any) -> Any:
            if (
                features.ndim != 4
                or features.shape[1] != self.architecture.in_channels
            ):
                raise ValueError("local-24 Model C expects N,49,Y,X")
            global_state = self.fno(self.positional_encoding(features))
            return global_state + self.local(features)


    class BireY32FullStateFNO(nn.Module):
        """Canonical local Model C with expanded meridional bandwidth."""

        def __init__(self, architecture: BireY32Architecture) -> None:
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
            self.local = nn.Conv2d(
                architecture.in_channels,
                architecture.out_channels,
                kernel_size=architecture.local_kernel_size,
                padding=architecture.local_kernel_size // 2,
                bias=False,
            )
            nn.init.zeros_(self.local.weight)

        def forward(self, features: Any) -> Any:
            if (
                features.ndim != 4
                or features.shape[1] != self.architecture.in_channels
            ):
                raise ValueError("canonical Y32 Model C expects N,49,Y,X")
            global_state = self.fno(self.positional_encoding(features))
            return global_state + self.local(features)


    class BireTwoInOneOutFNO(nn.Module):
        """The 32x32 Model C reading two consecutive time levels.

        Deliberately the same module names as :class:`BireY32FullStateFNO`: the
        FNO body, the six pointwise LayerNorms, the position encoder and the
        bias-free local branch are identical objects, and only the two
        input-facing tensors --- ``fno.lifting.fcs.0.weight`` and
        ``local.weight`` --- carry 46 extra input channels. Sharing the names
        is what makes every other parameter identifiable during migration.
        """

        def __init__(self, architecture: BireTwoInOneOutArchitecture) -> None:
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
            self.local = nn.Conv2d(
                architecture.in_channels,
                architecture.out_channels,
                kernel_size=architecture.local_kernel_size,
                padding=architecture.local_kernel_size // 2,
                bias=False,
            )
            nn.init.zeros_(self.local.weight)

        def forward(self, features: Any) -> Any:
            if (
                features.ndim != 4
                or features.shape[1] != self.architecture.in_channels
            ):
                raise ValueError("two-input Model C expects N,95,Y,X")
            global_state = self.fno(self.positional_encoding(features))
            return global_state + self.local(features)

else:  # pragma: no cover - environment dependent
    BirePositionalEncoding = None  # type: ignore[assignment,misc]
    BireAlignedFullStateFNO = None  # type: ignore[assignment,misc]
    BireLocal24FullStateFNO = None  # type: ignore[assignment,misc]
    BireY32FullStateFNO = None  # type: ignore[assignment,misc]
    BireTwoInOneOutFNO = None  # type: ignore[assignment,misc]

def build_bire_aligned_model(architecture: BireAlignedArchitecture) -> Any:
    """Build the Bire-aligned map, matching ``build_successor``'s signature."""

    require_model_a_runtime()
    if BireAlignedFullStateFNO is None:  # pragma: no cover
        raise RuntimeError("the Bire-aligned arm requires PyTorch")
    return BireAlignedFullStateFNO(architecture)


def build_bire_local24_model(architecture: BireLocal24Architecture) -> Any:
    """Build the 24x24 FNO plus its zero-initialized local correction."""

    require_model_a_runtime()
    if BireLocal24FullStateFNO is None:  # pragma: no cover
        raise RuntimeError("the local-24 arm requires PyTorch")
    return BireLocal24FullStateFNO(architecture)


def build_bire_y32_model(architecture: BireY32Architecture) -> Any:
    """Build the retained 32x24 FNO with its local correction topology."""

    require_model_a_runtime()
    if BireY32FullStateFNO is None:  # pragma: no cover
        raise RuntimeError("the retained Y32 model requires PyTorch")
    if not isinstance(architecture, BireY32Architecture):
        raise BireAlignedFullStateError(
            "build_bire_y32_model is reserved for the retained 32x24 parent"
        )
    return BireY32FullStateFNO(architecture)


def build_bire_y32_x32_model(architecture: BireY32X32Architecture) -> Any:
    """Build the 32x32 successor on the unchanged Y32 topology.

    The module class is shared with the parent deliberately: every layer, the
    position encoder and the local branch are identical, and only the FNO's
    declared ``n_modes`` differs, so a second class would duplicate code without
    describing a second model.
    """

    require_model_a_runtime()
    if BireY32FullStateFNO is None:  # pragma: no cover
        raise RuntimeError("the 32x32 model requires PyTorch")
    if not isinstance(architecture, BireY32X32Architecture):
        raise BireAlignedFullStateError(
            "the 32x32 builder requires BireY32X32Architecture"
        )
    return BireY32FullStateFNO(architecture)


def build_bire_two_in_one_out_model(
    architecture: BireTwoInOneOutArchitecture,
) -> Any:
    """Build the two-time-level 32x32 model."""

    require_model_a_runtime()
    if BireTwoInOneOutFNO is None:  # pragma: no cover
        raise RuntimeError("the two-input model requires PyTorch")
    if not isinstance(architecture, BireTwoInOneOutArchitecture):
        raise BireAlignedFullStateError(
            "the two-input builder requires BireTwoInOneOutArchitecture"
        )
    return BireTwoInOneOutFNO(architecture)


_BIRE_SPECTRAL_WEIGHT_KEYS = tuple(
    f"fno.fno_blocks.convs.{layer}.weight.tensor" for layer in range(3)
)


def migrate_bire_aligned_state_dict(
    parent_state: Mapping[str, Any],
    target_model: Any,
) -> dict[str, Any]:
    """Strictly migrate a 24x16 Bire checkpoint into the local-24 model.

    The old real-valued 2-D transform stores its 16 requested zonal modes as
    nine non-redundant complex coefficients.  The new 24-mode zonal dimension
    stores thirteen.  For each of the three spectral blocks this function
    copies the complete 24x9 parent tensor into ``[..., :9]`` and sets the four
    new coefficients to zero.  The sole newly introduced parameter,
    ``local.weight``, is also set to zero.

    Every key and shape is audited before anything is installed.  A parent with
    an extra key, a target with any added parameter besides ``local.weight``, or
    any shape change besides the exact three 24x9-to-24x13 expansions is
    rejected.  The migrated dictionary is then loaded with ``strict=True`` and
    returned with serializable provenance for the training checkpoint/report.
    """

    if BireLocal24FullStateFNO is None or not isinstance(
        target_model, BireLocal24FullStateFNO
    ):
        raise BireAlignedFullStateError(
            "the Bire checkpoint migration target must be BireLocal24FullStateFNO"
        )
    if target_model.architecture != BireLocal24Architecture():
        raise BireAlignedFullStateError(
            "the Bire checkpoint migration target architecture changed"
        )

    # NeuralOperator 2.0 injects a non-parameter top-level `_metadata` entry in
    # state_dict().  Production checkpoints deliberately omit it; accepting and
    # dropping precisely this known entry also makes the helper safe for a raw
    # in-memory parent state_dict without weakening the parameter-key audit.
    parent = {key: value for key, value in parent_state.items() if key != "_metadata"}
    raw_target = target_model.state_dict()
    target = {key: value for key, value in raw_target.items() if key != "_metadata"}
    parent_keys = set(parent)
    target_keys = set(target)
    unexpected = sorted(parent_keys - target_keys)
    added = sorted(target_keys - parent_keys)
    if unexpected or added != ["local.weight"]:
        raise BireAlignedFullStateError(
            "the Bire checkpoint key migration is not exactly +local.weight: "
            f"unexpected_parent={unexpected!r}, added_target={added!r}"
        )

    expected_expansions = set(_BIRE_SPECTRAL_WEIGHT_KEYS)
    observed_expansions: set[str] = set()
    migrated: dict[str, Any] = {}
    expansion_records: list[dict[str, Any]] = []
    for key in sorted(parent_keys):
        source = parent[key]
        destination = target[key]
        if not hasattr(source, "shape") or not hasattr(destination, "shape"):
            raise BireAlignedFullStateError(
                f"the Bire checkpoint entry {key!r} is not a tensor"
            )
        source_shape = tuple(int(value) for value in source.shape)
        destination_shape = tuple(int(value) for value in destination.shape)
        if source.dtype != destination.dtype:
            raise BireAlignedFullStateError(
                f"the Bire checkpoint dtype changed for {key}: "
                f"{source.dtype} -> {destination.dtype}"
            )
        if source_shape == destination_shape:
            value = destination.detach().clone()
            value.copy_(source)
            migrated[key] = value
            continue
        if key not in expected_expansions:
            raise BireAlignedFullStateError(
                f"undeclared Bire checkpoint shape change for {key}: "
                f"{source_shape} -> {destination_shape}"
            )
        if source_shape != (128, 128, 24, 9) or destination_shape != (
            128,
            128,
            24,
            13,
        ):
            raise BireAlignedFullStateError(
                f"the declared spectral expansion has unexpected shapes for {key}: "
                f"{source_shape} -> {destination_shape}"
            )
        value = destination.detach().clone()
        value.zero_()
        value[..., : source_shape[-1]].copy_(source)
        migrated[key] = value
        observed_expansions.add(key)
        expansion_records.append(
            {
                "key": key,
                "source_shape": list(source_shape),
                "target_shape": list(destination_shape),
                "copied_nonredundant_zonal_coefficients": source_shape[-1],
                "zero_initialized_new_zonal_coefficients": (
                    destination_shape[-1] - source_shape[-1]
                ),
            }
        )

    if observed_expansions != expected_expansions:
        missing = sorted(expected_expansions - observed_expansions)
        extra = sorted(observed_expansions - expected_expansions)
        raise BireAlignedFullStateError(
            "the Bire checkpoint did not perform exactly three spectral expansions: "
            f"missing={missing!r}, extra={extra!r}"
        )

    local = target["local.weight"].detach().clone()
    if tuple(local.shape) != (STATE_CHANNEL_COUNT, EXTERNAL_INPUT_CHANNELS, 3, 3):
        raise BireAlignedFullStateError(
            f"the local correction has unexpected shape {tuple(local.shape)!r}"
        )
    local.zero_()
    migrated["local.weight"] = local
    # This is both a final PyTorch-level key audit and the intended installation
    # step.  Callers can immediately construct an optimizer over target_model.
    incompatible = target_model.load_state_dict(migrated, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise BireAlignedFullStateError(
            "strict Bire checkpoint loading reported incompatible keys: "
            f"missing={incompatible.missing_keys!r}, "
            f"unexpected={incompatible.unexpected_keys!r}"
        )

    return {
        "state_dict": migrated,
        "provenance": {
            "migration": "bire_aligned_24x16_to_local24_24x24",
            "source_n_modes_tensor_order_y_x": [24, 16],
            "target_n_modes_tensor_order_y_x": [24, 24],
            "spectral_expansions": expansion_records,
            "added_parameter": {
                "key": "local.weight",
                "shape": list(local.shape),
                "bias": False,
                "initialization": "zeros",
            },
            "strict_load": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "initial_map_preserved_by_zero_extension": True,
        },
    }


_Y32_SPECTRAL_WEIGHT_KEYS = tuple(
    f"fno.fno_blocks.convs.{layer}.weight.tensor" for layer in range(3)
)


def migrate_local24_state_dict(
    parent_state: Mapping[str, Any],
    target_model: Any,
) -> dict[str, Any]:
    """Strictly migrate the retained local24 state into canonical Y32.

    NeuralOperator fft-shifts the first spatial Fourier dimension. The 24
    parent entries therefore occupy the centered slice ``4:28`` of each new
    32-entry meridional tensor; the four lower and four upper sidebands begin at
    exact zero. All same-shaped parameters, including the trained
    ``local.weight``, are copied without modification.
    """

    if BireY32FullStateFNO is None or not isinstance(
        target_model, BireY32FullStateFNO
    ):
        raise BireAlignedFullStateError(
            "the local24 migration target must be BireY32FullStateFNO"
        )
    if target_model.architecture != BireY32Architecture():
        raise BireAlignedFullStateError(
            "the local24 migration target architecture changed"
        )

    parent = {key: value for key, value in parent_state.items() if key != "_metadata"}
    raw_target = target_model.state_dict()
    target = {key: value for key, value in raw_target.items() if key != "_metadata"}
    parent_keys = set(parent)
    target_keys = set(target)
    missing = sorted(target_keys - parent_keys)
    unexpected = sorted(parent_keys - target_keys)
    if missing or unexpected:
        raise BireAlignedFullStateError(
            "the local24-to-Y32 migration must retain exactly the same keys: "
            f"missing_parent={missing!r}, unexpected_parent={unexpected!r}"
        )
    if "local.weight" not in parent_keys:
        raise BireAlignedFullStateError(
            "the local24 checkpoint has no trained local.weight"
        )

    expected_expansions = set(_Y32_SPECTRAL_WEIGHT_KEYS)
    observed_expansions: set[str] = set()
    migrated: dict[str, Any] = {}
    expansion_records: list[dict[str, Any]] = []
    for key in sorted(parent_keys):
        source = parent[key]
        destination = target[key]
        if not hasattr(source, "shape") or not hasattr(destination, "shape"):
            raise BireAlignedFullStateError(
                f"the local24 checkpoint entry {key!r} is not a tensor"
            )
        source_shape = tuple(int(value) for value in source.shape)
        destination_shape = tuple(int(value) for value in destination.shape)
        if source.dtype != destination.dtype:
            raise BireAlignedFullStateError(
                f"the local24 checkpoint dtype changed for {key}: "
                f"{source.dtype} -> {destination.dtype}"
            )
        if source_shape == destination_shape:
            value = destination.detach().clone()
            value.copy_(source)
            migrated[key] = value
            continue
        if key not in expected_expansions:
            raise BireAlignedFullStateError(
                f"undeclared local24 checkpoint shape change for {key}: "
                f"{source_shape} -> {destination_shape}"
            )
        if source_shape != (128, 128, 24, 13) or destination_shape != (
            128,
            128,
            32,
            13,
        ):
            raise BireAlignedFullStateError(
                f"the declared Y32 spectral expansion has unexpected shapes for {key}: "
                f"{source_shape} -> {destination_shape}"
            )
        value = destination.detach().clone()
        value.zero_()
        value[:, :, 4:28, :].copy_(source)
        migrated[key] = value
        observed_expansions.add(key)
        expansion_records.append(
            {
                "key": key,
                "source_shape": list(source_shape),
                "target_shape": list(destination_shape),
                "copied_meridional_slice": [4, 28],
                "zero_initialized_lower_meridional_modes": 4,
                "zero_initialized_upper_meridional_modes": 4,
            }
        )

    if observed_expansions != expected_expansions:
        missing_expansions = sorted(expected_expansions - observed_expansions)
        extra_expansions = sorted(observed_expansions - expected_expansions)
        raise BireAlignedFullStateError(
            "the local24 checkpoint did not perform exactly three Y32 expansions: "
            f"missing={missing_expansions!r}, extra={extra_expansions!r}"
        )

    local_shape = tuple(int(value) for value in migrated["local.weight"].shape)
    if local_shape != (STATE_CHANNEL_COUNT, EXTERNAL_INPUT_CHANNELS, 3, 3):
        raise BireAlignedFullStateError(
            f"the migrated trained local correction has unexpected shape {local_shape!r}"
        )
    incompatible = target_model.load_state_dict(migrated, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise BireAlignedFullStateError(
            "strict Y32 checkpoint loading reported incompatible keys: "
            f"missing={incompatible.missing_keys!r}, "
            f"unexpected={incompatible.unexpected_keys!r}"
        )

    return {
        "state_dict": migrated,
        "provenance": {
            "migration": "bire_local24_24x24_to_y32_32x24",
            "source_n_modes_tensor_order_y_x": [24, 24],
            "target_n_modes_tensor_order_y_x": [32, 24],
            "spectral_expansions": expansion_records,
            "retained_parameter": {
                "key": "local.weight",
                "shape": list(local_shape),
                "bias": False,
                "copied_without_modification": True,
            },
            "strict_load": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "initial_map_preserved_by_centered_zero_extension": True,
        },
    }


_Y32_X32_SPECTRAL_WEIGHT_KEYS = tuple(
    f"fno.fno_blocks.convs.{layer}.weight.tensor" for layer in range(3)
)


def migrate_y32_state_dict(
    parent_state: Mapping[str, Any],
    target_model: Any,
) -> dict[str, Any]:
    """Strictly migrate the retained Y32 state into the 32x32 model.

    The zonal dimension is the real-transform axis, so its coefficients are the
    non-redundant half spectrum: 24 requested zonal modes are stored as 13
    coefficients and 32 as 17. Unlike the meridional axis, this one is *not*
    fft-shifted, so the parent's 13 coefficients are the 13 lowest zonal
    wavenumbers and occupy the leading slice ``[..., :13]`` of each new 32x17
    tensor; the four added coefficients begin at exact zero.

    Every other parameter, including the trained ``local.weight``, is copied
    without modification, so the migration adds no parameter and the initial map
    is preserved exactly.
    """

    if BireY32FullStateFNO is None or not isinstance(
        target_model, BireY32FullStateFNO
    ):
        raise BireAlignedFullStateError(
            "the Y32 migration target must be BireY32FullStateFNO"
        )
    if target_model.architecture != BireY32X32Architecture():
        raise BireAlignedFullStateError(
            "the Y32 migration target architecture changed"
        )

    parent = {key: value for key, value in parent_state.items() if key != "_metadata"}
    raw_target = target_model.state_dict()
    target = {key: value for key, value in raw_target.items() if key != "_metadata"}
    parent_keys = set(parent)
    target_keys = set(target)
    missing = sorted(target_keys - parent_keys)
    unexpected = sorted(parent_keys - target_keys)
    if missing or unexpected:
        raise BireAlignedFullStateError(
            "the Y32-to-32x32 migration must retain exactly the same keys: "
            f"missing_parent={missing!r}, unexpected_parent={unexpected!r}"
        )
    if "local.weight" not in parent_keys:
        raise BireAlignedFullStateError(
            "the Y32 checkpoint has no trained local.weight"
        )

    expected_expansions = set(_Y32_X32_SPECTRAL_WEIGHT_KEYS)
    observed_expansions: set[str] = set()
    migrated: dict[str, Any] = {}
    expansion_records: list[dict[str, Any]] = []
    for key in sorted(parent_keys):
        source = parent[key]
        destination = target[key]
        if not hasattr(source, "shape") or not hasattr(destination, "shape"):
            raise BireAlignedFullStateError(
                f"the Y32 checkpoint entry {key!r} is not a tensor"
            )
        source_shape = tuple(int(value) for value in source.shape)
        destination_shape = tuple(int(value) for value in destination.shape)
        if source.dtype != destination.dtype:
            raise BireAlignedFullStateError(
                f"the Y32 checkpoint dtype changed for {key}: "
                f"{source.dtype} -> {destination.dtype}"
            )
        if source_shape == destination_shape:
            value = destination.detach().clone()
            value.copy_(source)
            migrated[key] = value
            continue
        if key not in expected_expansions:
            raise BireAlignedFullStateError(
                f"undeclared Y32 checkpoint shape change for {key}: "
                f"{source_shape} -> {destination_shape}"
            )
        if source_shape != (128, 128, 32, 13) or destination_shape != (
            128,
            128,
            32,
            17,
        ):
            raise BireAlignedFullStateError(
                f"the declared 32x32 zonal expansion has unexpected shapes for {key}: "
                f"{source_shape} -> {destination_shape}"
            )
        value = destination.detach().clone()
        value.zero_()
        value[:, :, :, :13].copy_(source)
        migrated[key] = value
        observed_expansions.add(key)
        expansion_records.append(
            {
                "key": key,
                "source_shape": list(source_shape),
                "target_shape": list(destination_shape),
                "copied_zonal_slice": [0, 13],
                "zero_initialized_new_zonal_coefficients": (
                    destination_shape[-1] - source_shape[-1]
                ),
            }
        )

    if observed_expansions != expected_expansions:
        missing_expansions = sorted(expected_expansions - observed_expansions)
        extra_expansions = sorted(observed_expansions - expected_expansions)
        raise BireAlignedFullStateError(
            "the Y32 checkpoint did not perform exactly three zonal expansions: "
            f"missing={missing_expansions!r}, extra={extra_expansions!r}"
        )

    local_shape = tuple(int(value) for value in migrated["local.weight"].shape)
    if local_shape != (STATE_CHANNEL_COUNT, EXTERNAL_INPUT_CHANNELS, 3, 3):
        raise BireAlignedFullStateError(
            f"the migrated trained local correction has unexpected shape {local_shape!r}"
        )
    incompatible = target_model.load_state_dict(migrated, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise BireAlignedFullStateError(
            "strict 32x32 checkpoint loading reported incompatible keys: "
            f"missing={incompatible.missing_keys!r}, "
            f"unexpected={incompatible.unexpected_keys!r}"
        )

    return {
        "state_dict": migrated,
        "provenance": {
            "migration": "bire_y32_32x24_to_y32_x32_32x32",
            "source_n_modes_tensor_order_y_x": [32, 24],
            "target_n_modes_tensor_order_y_x": [32, 32],
            "spectral_expansions": expansion_records,
            "retained_parameter": {
                "key": "local.weight",
                "shape": list(local_shape),
                "bias": False,
                "copied_without_modification": True,
            },
            "strict_load": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "initial_map_preserved_by_zero_extension": True,
        },
    }

#: The only two tensors that face the external input block, and therefore the
#: only two whose shape can move when a second time level is prepended.
_TWO_IN_INPUT_FACING_KEYS = {
    "fno.lifting.fcs.0.weight": (
        LIFTING_INPUT_CHANNELS,
        TWO_IN_LIFTING_INPUT_CHANNELS,
    ),
    "local.weight": (
        EXTERNAL_INPUT_CHANNELS,
        TWO_IN_EXTERNAL_INPUT_CHANNELS,
    ),
}


def migrate_y32_x32_state_dict(
    parent_state: Mapping[str, Any],
    target_model: Any,
) -> dict[str, Any]:
    """Strictly migrate the 32x32 one-input state into the two-input model.

    The two-input external block is ``(x_{t-10}, x_t, static)``, so the parent's
    ``(x_t, static)`` block keeps its order and simply moves 46 channels to the
    right; the position channels the encoder appends move with it. Both
    input-facing tensors are therefore zeroed and the whole parent tensor is
    copied into the trailing slice. The new leading 46 input channels are exact
    zeros, so at initialization the model *ignores* the history state and
    reproduces the parent map on ``x_t`` for any history whatsoever.

    Every other parameter --- the three spectral tensors, the Channel MLPs, the
    six LayerNorms, the projection and the skips --- is copied unchanged, so no
    Fourier capacity is added and temporal context is the sole change.
    """

    if BireTwoInOneOutFNO is None or not isinstance(
        target_model, BireTwoInOneOutFNO
    ):
        raise BireAlignedFullStateError(
            "the two-input migration target must be BireTwoInOneOutFNO"
        )
    if target_model.architecture != BireTwoInOneOutArchitecture():
        raise BireAlignedFullStateError(
            "the two-input migration target architecture changed"
        )

    parent = {key: value for key, value in parent_state.items() if key != "_metadata"}
    raw_target = target_model.state_dict()
    target = {key: value for key, value in raw_target.items() if key != "_metadata"}
    parent_keys = set(parent)
    target_keys = set(target)
    missing = sorted(target_keys - parent_keys)
    unexpected = sorted(parent_keys - target_keys)
    if missing or unexpected:
        raise BireAlignedFullStateError(
            "the 32x32-to-two-input migration must retain exactly the same keys: "
            f"missing_parent={missing!r}, unexpected_parent={unexpected!r}"
        )
    if "local.weight" not in parent_keys:
        raise BireAlignedFullStateError(
            "the 32x32 checkpoint has no trained local.weight"
        )

    expected_expansions = set(_TWO_IN_INPUT_FACING_KEYS)
    observed_expansions: set[str] = set()
    migrated: dict[str, Any] = {}
    expansion_records: list[dict[str, Any]] = []
    for key in sorted(parent_keys):
        source = parent[key]
        destination = target[key]
        if not hasattr(source, "shape") or not hasattr(destination, "shape"):
            raise BireAlignedFullStateError(
                f"the 32x32 checkpoint entry {key!r} is not a tensor"
            )
        source_shape = tuple(int(value) for value in source.shape)
        destination_shape = tuple(int(value) for value in destination.shape)
        if source.dtype != destination.dtype:
            raise BireAlignedFullStateError(
                f"the 32x32 checkpoint dtype changed for {key}: "
                f"{source.dtype} -> {destination.dtype}"
            )
        if source_shape == destination_shape:
            value = destination.detach().clone()
            value.copy_(source)
            migrated[key] = value
            continue
        if key not in expected_expansions:
            raise BireAlignedFullStateError(
                f"undeclared two-input shape change for {key}: "
                f"{source_shape} -> {destination_shape}"
            )
        parent_width, target_width = _TWO_IN_INPUT_FACING_KEYS[key]
        if (
            source_shape[1] != parent_width
            or destination_shape[1] != target_width
            or source_shape[:1] != destination_shape[:1]
            or source_shape[2:] != destination_shape[2:]
        ):
            raise BireAlignedFullStateError(
                f"the declared history expansion has unexpected shapes for {key}: "
                f"{source_shape} -> {destination_shape}"
            )
        value = destination.detach().clone()
        value.zero_()
        value[:, STATE_CHANNEL_COUNT:].copy_(source)
        migrated[key] = value
        observed_expansions.add(key)
        expansion_records.append(
            {
                "key": key,
                "source_shape": list(source_shape),
                "target_shape": list(destination_shape),
                "copied_into_input_slice": [STATE_CHANNEL_COUNT, target_width],
                "zero_initialized_history_input_channels": STATE_CHANNEL_COUNT,
            }
        )

    if observed_expansions != expected_expansions:
        missing_expansions = sorted(expected_expansions - observed_expansions)
        extra_expansions = sorted(observed_expansions - expected_expansions)
        raise BireAlignedFullStateError(
            "the 32x32 checkpoint did not perform exactly two input expansions: "
            f"missing={missing_expansions!r}, extra={extra_expansions!r}"
        )

    local_shape = tuple(int(value) for value in migrated["local.weight"].shape)
    if local_shape != (
        STATE_CHANNEL_COUNT,
        TWO_IN_EXTERNAL_INPUT_CHANNELS,
        3,
        3,
    ):
        raise BireAlignedFullStateError(
            f"the migrated local correction has unexpected shape {local_shape!r}"
        )
    incompatible = target_model.load_state_dict(migrated, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise BireAlignedFullStateError(
            "strict two-input checkpoint loading reported incompatible keys: "
            f"missing={incompatible.missing_keys!r}, "
            f"unexpected={incompatible.unexpected_keys!r}"
        )

    return {
        "state_dict": migrated,
        "provenance": {
            "migration": "bire_y32_x32_one_in_to_two_in_one_out",
            "source_external_input_channels": EXTERNAL_INPUT_CHANNELS,
            "target_external_input_channels": TWO_IN_EXTERNAL_INPUT_CHANNELS,
            "source_lifting_input_channels": LIFTING_INPUT_CHANNELS,
            "target_lifting_input_channels": TWO_IN_LIFTING_INPUT_CHANNELS,
            "input_expansions": expansion_records,
            "n_modes_tensor_order_y_x": [32, 32],
            "spectral_capacity_unchanged": True,
            "retained_parameter": {
                "key": "local.weight",
                "shape": list(local_shape),
                "bias": False,
                "trained_columns_copied_without_modification": True,
            },
            "strict_load": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "initial_map_ignores_history_and_equals_the_parent": True,
        },
    }


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


class BireTwoInStepper(BireAlignedStepper):
    """Evaluation adapter that carries the previous time level across a rollout.

    ``begin`` is handed the physical ``t - 10`` truth state once, before the
    first call; every ``step`` afterwards consumes the pair it holds and slides
    it forward, so the shared 360-day and 2,000-day rollout loops need no other
    change. ``requires_history`` is what lets those loops decide whether the
    extra truth read is needed at all.
    """

    requires_history = True

    _previous: Any = None

    def begin(self, history: np.ndarray | None = None) -> None:
        if history is None:
            raise BireAlignedFullStateError(
                "the two-input stepper needs the t-10 physical state"
            )
        self._previous = self.normalized_state(
            np.asarray(history, dtype=np.float32)
        )

    def step(self, current: Any, static: Any) -> Any:
        if self._previous is None:
            raise BireAlignedFullStateError(
                "the two-input stepper was stepped before begin() supplied history"
            )
        if self._previous.shape != current.shape:
            raise BireAlignedFullStateError(
                "the two-input stepper's history and present states disagree"
            )
        wet = torch.from_numpy(self.wet.astype(np.float32))[None, None].to(
            self.device
        )
        future = self.model(
            torch.cat((self._previous, current, static), dim=1)
        ) * wet
        self._previous = current
        return future
