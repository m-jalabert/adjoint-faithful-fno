"""Runtime guard, determinism, hashing, and the 46-channel state contract.

Every other module in the package is written against the names defined here:
the ordered 46-channel prognostic state, the four physical groups the objective
balances, the ten-day prediction interval, and the audited loss terms. Nothing
in this module knows about any particular model.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

from importlib import metadata

import numpy as np

try:  # Keep documentation-only imports working on login nodes without the ML stack.
    import torch
    from torch import nn
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, Dataset
    from neuralop.models import FNO
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    FNO = None  # type: ignore[assignment]

NEURALOPERATOR_VERSION = "2.0.0"


def require_runtime() -> None:
    """Verify the pinned PyTorch / NeuralOperator environment is importable."""

    if torch is None or nn is None or FNO is None or DataLoader is None:
        raise RuntimeError(
            "the production emulator requires the project PyTorch and "
            f"NeuralOperator {NEURALOPERATOR_VERSION} environment"
        )
    try:
        version = metadata.version("neuraloperator")
    except metadata.PackageNotFoundError as exc:  # pragma: no cover - installation failure
        raise RuntimeError(
            "the production emulator requires the project PyTorch and "
            f"NeuralOperator {NEURALOPERATOR_VERSION} environment"
        ) from exc
    if version != NEURALOPERATOR_VERSION:
        raise RuntimeError(
            f"this tree is pinned to neuraloperator {NEURALOPERATOR_VERSION}, found {version}"
        )


def seed_everything(seed: int) -> None:
    """Make initialization, batch order and evaluation reproducible."""

    require_runtime()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def json_safe(value: Any) -> Any:
    """Replace non-finite floats with ``None`` so a report is always writable.

    Infinity is a legitimate *value* for several diagnostics --- a growth rate
    that never doubles, a ratio whose denominator is zero --- but it is not
    representable in JSON, and ``allow_nan=False`` turns it into a hard failure.
    A v3 training run lost 2 h 46 m of completed work to exactly that: every
    checkpoint was written and selection had finished when serializing the
    report raised ``Out of range float values are not JSON compliant: inf``.
    Sanitizing at the boundary means no diagnostic can ever do that again.
    """

    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _device(name: str) -> Any:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested without a visible GPU")
    return torch.device(name)


def _checkpoint_state_dict(model: Any) -> dict[str, Any]:
    """Copy only loadable parameters from NeuralOperator's augmented state dict."""

    # neuraloperator 2.0's FNO includes a top-level ``_metadata`` value in its
    # state dictionary. PyTorch's load_state_dict treats it as an unexpected
    # parameter key, so it must not enter a portable experiment checkpoint.
    return copy.deepcopy(
        {key: value for key, value in model.state_dict().items() if key != "_metadata"}
    )


#: The prediction interval of the emulator, in days.
HORIZON_DAYS = 10

#: Number of MITgcm vertical levels carried by the prognostic state.
LEVEL_COUNT = 15


def state_channels(nr: int = LEVEL_COUNT) -> tuple[str, ...]:
    """Names for ``x = [U_1:Nr, V_1:Nr, Theta_1:Nr, eta]``."""

    return tuple(
        [f"U_{level:02d}" for level in range(1, nr + 1)]
        + [f"V_{level:02d}" for level in range(1, nr + 1)]
        + [f"Theta_{level:02d}" for level in range(1, nr + 1)]
        + ["Eta"]
    )


STATE_CHANNELS = state_channels()

STATE_CHANNEL_COUNT = len(STATE_CHANNELS)

#: The static-feature contract of the trajectory-v3 *store*. The production
#: model reads only two of these directly and derives three further physical
#: coefficients from MITgcm's own inputs, but the store's own order is checked
#: so a regenerated dataset cannot silently permute what is read.
STORE_STATIC_FEATURES = (
    "wind_stress_x",
    "longitude_normalized",
    "latitude_normalized",
    "wet_mask",
    "distance_to_wall_normalized",
)

STORE_STATIC_COUNT = len(STORE_STATIC_FEATURES)

WIND_FEATURE_INDEX = STORE_STATIC_FEATURES.index("wind_stress_x")

WET_MASK_FEATURE_INDEX = STORE_STATIC_FEATURES.index("wet_mask")

#: The four physical groups the objective weights equally.
GROUP_SLICES = {
    "u": slice(0, 15),
    "v": slice(15, 30),
    "temperature": slice(30, 45),
    "ssh": slice(45, 46),
}

#: Every loss component recorded in the training window and checked for
#: finiteness at each optimizer step. The three physics terms are active from
#: step one, so they are audited on the same footing as the base terms.
AUDIT_TERMS = (
    "total",
    "state",
    "increment",
    "rollout",
    "spectral",
    "boundary",
    "pressure_gradient",
    "continuity",
    "barotropic_transport",
    *(f"state_{name}" for name in GROUP_SLICES),
    *(f"increment_{name}" for name in GROUP_SLICES),
)


class ChunkAwareBatchSampler:
    """Shuffle contiguous compressed-Zarr batches while preserving local reads.

    Records are laid out regime-major and day-ascending, so a batch drawn from
    one contiguous run touches a small number of Zarr chunks. Batches never
    straddle a regime boundary, and the batch *order* is reshuffled every epoch
    from the declared seed.
    """

    def __init__(self, dataset: Any, batch_size: int, seed: int) -> None:
        if int(batch_size) <= 0:
            raise ValueError("the batch sampler needs a positive batch size")
        self.seed, self.epoch = int(seed), 0
        batches: list[tuple[int, ...]] = []
        current: list[int] = []
        previous: int | None = None
        for index, (experiment, _) in enumerate(dataset.records):
            if previous is not None and experiment != previous and current:
                batches.append(tuple(current))
                current = []
            current.append(index)
            if len(current) == int(batch_size):
                batches.append(tuple(current))
                current = []
            previous = experiment
        if current:
            batches.append(tuple(current))
        self.batches = tuple(batches)

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        order = list(range(len(self.batches)))
        random.Random(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        yield from (list(self.batches[index]) for index in order)


__all__: Sequence[str] = (
    "AUDIT_TERMS",
    "ChunkAwareBatchSampler",
    "DataLoader",
    "Dataset",
    "FNO",
    "GROUP_SLICES",
    "HORIZON_DAYS",
    "LEVEL_COUNT",
    "NEURALOPERATOR_VERSION",
    "STATE_CHANNELS",
    "STATE_CHANNEL_COUNT",
    "STORE_STATIC_COUNT",
    "STORE_STATIC_FEATURES",
    "WET_MASK_FEATURE_INDEX",
    "WIND_FEATURE_INDEX",
    "functional",
    "nn",
    "require_runtime",
    "seed_everything",
    "state_channels",
    "torch",
)
