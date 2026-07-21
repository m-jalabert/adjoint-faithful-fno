"""Paper-faithful Fourier neural operator used by Bire et al. (2025).

The network in the paper is not the stock :class:`neuralop.models.FNO`.
Its Fourier multiplier is channel diagonal (``separable=True``), followed by
an explicit pointwise channel mixer.  This module keeps that distinction
visible and uses only NeuralOperator's public ``SpectralConv`` primitive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any, Mapping, Sequence


_TORCH_IMPORT_ERROR: BaseException | None = None
_NEURALOP_IMPORT_ERROR: BaseException | None = None

try:  # Keep lightweight commands such as ``repro --help`` importable on login nodes.
    import torch
    from torch import nn
    from torch.nn import functional as F
except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

if torch is not None:
    try:
        from neuralop.layers.spectral_convolution import SpectralConv
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        SpectralConv = None  # type: ignore[assignment]
        _NEURALOP_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised only without torch
    SpectralConv = None  # type: ignore[assignment]


DEPENDENCY_MESSAGE = (
    "PaperFNO2d requires PyTorch and NeuralOperator 2.0.0. Activate the "
    "project's CUDA/PyTorch environment and install the locked dependencies "
    "before training or rollout."
)


def require_fno_dependencies() -> None:
    """Raise an actionable error when the ML runtime is unavailable."""

    cause = _TORCH_IMPORT_ERROR or _NEURALOP_IMPORT_ERROR
    if torch is None or SpectralConv is None:
        raise RuntimeError(DEPENDENCY_MESSAGE) from cause

    try:
        neuralop_version = metadata.version("neuraloperator")
    except metadata.PackageNotFoundError:  # source-checkout development fallback
        try:
            import neuralop

            neuralop_version = neuralop.__version__
        except (ImportError, AttributeError) as fallback_exc:  # pragma: no cover
            raise RuntimeError(DEPENDENCY_MESSAGE) from fallback_exc
    if neuralop_version.split(".", maxsplit=1)[0] != "2":
        raise RuntimeError(
            f"{DEPENDENCY_MESSAGE} Found neuraloperator {neuralop_version}; "
            "this implementation targets the 2.x SpectralConv API."
        )


def _pair(value: int | Sequence[int], *, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    else:
        result = tuple(int(item) for item in value)
        if len(result) != 2:
            raise ValueError(f"{name} must contain exactly two integers")
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive, got {result}")
    return result


def effective_n_modes(
    requested: int | Sequence[int], spatial_shape: Sequence[int]
) -> tuple[int, int]:
    """Return the even mode counts supported by a particular spatial grid.

    NeuralOperator interprets the final count before applying the real-FFT
    redundancy.  A 31 x 31 stride-8 grid therefore receives ``(30, 30)`` and
    uses the centered crop of weights learned with ``(64, 64)``.
    """

    modes = _pair(requested, name="requested modes")
    shape = _pair(spatial_shape, name="spatial shape")
    result = []
    for wanted, size in zip(modes, shape):
        supported = min(wanted, size)
        if supported > 1 and supported % 2:
            supported -= 1
        result.append(max(1, supported))
    return result[0], result[1]


@dataclass(frozen=True)
class PaperFNOConfig:
    """Architecture settings, with production defaults from the paper."""

    in_channels: int = 11
    out_channels: int = 10
    lifting_channels: int = 256
    hidden_channels: int = 128
    projection_channels: int = 256
    channel_mlp_channels: int = 512
    n_layers: int = 3
    n_modes: tuple[int, int] = (64, 64)
    norm_eps: float = 1.0e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_modes", _pair(self.n_modes, name="n_modes"))
        integer_fields = (
            "in_channels",
            "out_channels",
            "lifting_channels",
            "hidden_channels",
            "projection_channels",
            "channel_mlp_channels",
            "n_layers",
        )
        for field_name in integer_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "PaperFNOConfig":
        if values is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        kwargs = {key: value for key, value in values.items() if key in allowed}
        if "n_modes" in kwargs:
            kwargs["n_modes"] = _pair(kwargs["n_modes"], name="n_modes")
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_modes"] = list(self.n_modes)
        return result


if nn is not None:

    class ChannelLayerNorm2d(nn.Module):
        """Non-affine LayerNorm over channels of an NCHW tensor."""

        def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(channels, eps=eps, elementwise_affine=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


    class AlternatingSineCosinePosition2d(nn.Module):
        """Two positional fields recovered from oceanfourcast v0.1.1.

        Along each axis, even coordinate indices contain ``sin(pi*i/(2*n))``
        and odd indices contain ``cos(pi*i/(2*n))``.  Coordinates are created
        from the actual input shape, so the same model can run on 248 x 248 or
        the paper's stride-8 31 x 31 grid.
        """

        @staticmethod
        def coordinates_like(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if x.ndim != 4:
                raise ValueError(f"expected NCHW input, got shape {tuple(x.shape)}")
            height, width = x.shape[-2:]
            x_index = torch.arange(width, dtype=x.dtype, device=x.device)
            y_index = torch.arange(height, dtype=x.dtype, device=x.device)
            x_phase = torch.pi * x_index / (2.0 * width)
            y_phase = torch.pi * y_index / (2.0 * height)
            x_coord = torch.where(
                (x_index.to(torch.int64) % 2) == 0,
                torch.sin(x_phase),
                torch.cos(x_phase),
            ).view(1, 1, 1, width)
            y_coord = torch.where(
                (y_index.to(torch.int64) % 2) == 0,
                torch.sin(y_phase),
                torch.cos(y_phase),
            ).view(1, 1, height, 1)
            return (
                x_coord.expand(x.shape[0], 1, height, width),
                y_coord.expand(x.shape[0], 1, height, width),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_coord, y_coord = self.coordinates_like(x)
            return torch.cat((x, x_coord, y_coord), dim=1)


    class SoftGatedResidual2d(nn.Module):
        """Learned per-channel affine residual initialized as the identity."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.weight * x + self.bias


    class PaperFourierBlock2d(nn.Module):
        """One spatial-mixing and channel-mixing block from Figure 1."""

        def __init__(
            self,
            hidden_channels: int,
            n_modes: tuple[int, int],
            channel_mlp_channels: int,
            norm_eps: float,
        ) -> None:
            super().__init__()
            require_fno_dependencies()
            self.max_n_modes = _pair(n_modes, name="n_modes")
            self.spectral = SpectralConv(
                hidden_channels,
                hidden_channels,
                n_modes=self.max_n_modes,
                separable=True,
                bias=False,
                factorization=None,
                implementation="reconstructed",
                fno_block_precision="full",
                fft_norm="forward",
            )
            self.pointwise = nn.Conv2d(hidden_channels, hidden_channels, 1, bias=True)
            self.spatial_norm = ChannelLayerNorm2d(hidden_channels, eps=norm_eps)
            self.channel_mlp = nn.Sequential(
                nn.Conv2d(hidden_channels, channel_mlp_channels, 1, bias=True),
                nn.GELU(),
                nn.Conv2d(channel_mlp_channels, hidden_channels, 1, bias=True),
            )
            self.channel_skip = SoftGatedResidual2d(hidden_channels)
            self.channel_norm = ChannelLayerNorm2d(hidden_channels, eps=norm_eps)
            self.last_effective_n_modes = self.max_n_modes

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            active_modes = effective_n_modes(self.max_n_modes, x.shape[-2:])
            # SpectralConv keeps a max-sized parameter tensor. Updating n_modes
            # only selects its centered low-mode crop and never reallocates weights.
            self.spectral.n_modes = active_modes
            self.last_effective_n_modes = active_modes
            x = self.spatial_norm(F.gelu(self.spectral(x) + self.pointwise(x)))
            return self.channel_norm(self.channel_mlp(x) + self.channel_skip(x))


    class PaperFNO2d(nn.Module):
        """Resolution-flexible, paper-faithful emulator with no patching/dropout."""

        def __init__(self, config: PaperFNOConfig | Mapping[str, Any] | None = None) -> None:
            super().__init__()
            require_fno_dependencies()
            if isinstance(config, PaperFNOConfig):
                self.config = config
            else:
                self.config = PaperFNOConfig.from_mapping(config)

            cfg = self.config
            self.position = AlternatingSineCosinePosition2d()
            self.lifting = nn.Sequential(
                nn.Conv2d(cfg.in_channels + 2, cfg.lifting_channels, 1, bias=True),
                nn.GELU(),
                nn.Conv2d(cfg.lifting_channels, cfg.hidden_channels, 1, bias=True),
            )
            self.blocks = nn.ModuleList(
                PaperFourierBlock2d(
                    hidden_channels=cfg.hidden_channels,
                    n_modes=cfg.n_modes,
                    channel_mlp_channels=cfg.channel_mlp_channels,
                    norm_eps=cfg.norm_eps,
                )
                for _ in range(cfg.n_layers)
            )
            self.projection = nn.Sequential(
                nn.Conv2d(cfg.hidden_channels, cfg.projection_channels, 1, bias=True),
                nn.GELU(),
                nn.Conv2d(cfg.projection_channels, cfg.out_channels, 1, bias=True),
            )

        @property
        def in_channels(self) -> int:
            return self.config.in_channels

        @property
        def out_channels(self) -> int:
            return self.config.out_channels

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 4:
                raise ValueError(f"expected NCHW input, got shape {tuple(x.shape)}")
            if x.shape[1] != self.in_channels:
                raise ValueError(
                    f"expected {self.in_channels} input channels, got {x.shape[1]}"
                )
            x = self.lifting(self.position(x))
            for block in self.blocks:
                x = block(x)
            return self.projection(x)

        def architecture_dict(self) -> dict[str, Any]:
            return self.config.to_dict()


else:  # pragma: no cover - only used on environments lacking torch

    class _MissingTorchClass:
        def __init__(self, *_: Any, **__: Any) -> None:
            require_fno_dependencies()

    ChannelLayerNorm2d = _MissingTorchClass
    AlternatingSineCosinePosition2d = _MissingTorchClass
    SoftGatedResidual2d = _MissingTorchClass
    PaperFourierBlock2d = _MissingTorchClass
    PaperFNO2d = _MissingTorchClass


def build_paper_fno(
    config: PaperFNOConfig | Mapping[str, Any] | None = None,
) -> PaperFNO2d:
    """Construct the emulator and provide one stable public factory."""

    require_fno_dependencies()
    return PaperFNO2d(config)


__all__ = [
    "AlternatingSineCosinePosition2d",
    "ChannelLayerNorm2d",
    "DEPENDENCY_MESSAGE",
    "PaperFNO2d",
    "PaperFNOConfig",
    "PaperFourierBlock2d",
    "SoftGatedResidual2d",
    "build_paper_fno",
    "effective_n_modes",
    "require_fno_dependencies",
]
