"""Small, dependency-light readers for MITgcm MDS metadata/data pairs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MDSMeta:
    """Metadata required to reshape and label one global MDS array."""

    dimensions: tuple[int, ...]
    nrecords: int
    dtype: np.dtype
    fields: tuple[str, ...]
    timestep: int | None


def parse_mds_meta(path: str | Path) -> MDSMeta:
    """Parse the subset of an MDS ``.meta`` file used by AF-FNO analysis."""

    metadata_path = Path(path)
    text = metadata_path.read_text()

    dim_match = re.search(r"dimList\s*=\s*\[(.*?)\];", text, re.DOTALL)
    if not dim_match:
        raise ValueError(f"Missing dimList in {metadata_path}")
    dim_values = [int(value) for value in re.findall(r"[-+]?\d+", dim_match.group(1))]
    if not dim_values or len(dim_values) % 3:
        raise ValueError(f"Invalid dimList in {metadata_path}")
    dimensions = tuple(dim_values[index] for index in range(0, len(dim_values), 3))

    record_match = re.search(r"nrecords\s*=\s*\[\s*(\d+)\s*\]", text)
    nrecords = int(record_match.group(1)) if record_match else 1
    precision_match = re.search(r"dataprec\s*=\s*\[\s*'([^']+)'", text)
    precision = precision_match.group(1).strip().lower() if precision_match else "float32"
    precision_map = {
        "float32": np.dtype(">f4"),
        "real*4": np.dtype(">f4"),
        "float64": np.dtype(">f8"),
        "real*8": np.dtype(">f8"),
    }
    if precision not in precision_map:
        raise ValueError(f"Unsupported MDS precision {precision!r}")

    fields_match = re.search(r"fldList\s*=\s*\{(.*?)\};", text, re.DOTALL)
    fields = (
        tuple(value.strip() for value in re.findall(r"'([^']+)'", fields_match.group(1)))
        if fields_match
        else ()
    )
    timestep_match = re.search(r"timeStepNumber\s*=\s*\[\s*(\d+)\s*\]", text)
    timestep = int(timestep_match.group(1)) if timestep_match else None
    return MDSMeta(dimensions, nrecords, precision_map[precision], fields, timestep)


def read_mds(path: str | Path) -> tuple[MDSMeta, np.ndarray]:
    """Read a global MDS pair as ``(record, z, y, x)``-ordered data."""

    metadata_path = Path(path)
    if metadata_path.suffix == ".data":
        metadata_path = metadata_path.with_suffix(".meta")
    meta = parse_mds_meta(metadata_path)
    data_path = metadata_path.with_suffix(".data")
    count = meta.nrecords * math.prod(meta.dimensions)
    values = np.fromfile(data_path, dtype=meta.dtype, count=count)
    if values.size != count:
        raise ValueError(
            f"MDS size mismatch for {data_path}: expected {count}, got {values.size}"
        )
    shape = (meta.nrecords, *reversed(meta.dimensions))
    return meta, values.reshape(shape).astype(np.float64, copy=False)


def mds_fields(meta: MDSMeta, values: np.ndarray) -> dict[str, np.ndarray]:
    """Split a diagnostics MDS array into its named fields."""

    if not meta.fields:
        raise ValueError("MDS metadata contains no fldList")
    if meta.nrecords % len(meta.fields):
        raise ValueError("MDS record count is not divisible by fldList length")
    records_per_field = meta.nrecords // len(meta.fields)
    fields: dict[str, np.ndarray] = {}
    for index, name in enumerate(meta.fields):
        value = values[index * records_per_field : (index + 1) * records_per_field]
        fields[name] = value[0] if records_per_field == 1 else value
    return fields
