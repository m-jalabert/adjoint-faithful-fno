"""Tests for the compact shared AF--FNO dataset conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from bire_repro.af_data import DatasetSpec, build_dataset, validate_dataset


def _write_mds(
    root: Path,
    stem: str,
    values: np.ndarray,
    fields: tuple[str, ...] = (),
) -> None:
    """Write the small MDS subset consumed by the production converter."""

    data = root / f"{stem}.data"
    meta = root / f"{stem}.meta"
    nrecords, *shape = values.shape
    dim_list: list[int] = []
    for dimension in reversed(shape):
        dim_list.extend((dimension, 1, dimension))
    field_list = "".join(f" '{field:<8}'" for field in fields)
    field_block = f"\nfldList = {{\n{field_list}\n }};" if fields else ""
    meta.write_text(
        "nDims = [ %d ];\n" % len(shape)
        + "dimList = [ "
        + ", ".join(str(value) for value in dim_list)
        + " ];\n"
        + "dataprec = [ 'float32' ];\n"
        + "nrecords = [ %d ];" % nrecords
        + field_block
        + "\n"
    )
    values.astype(">f4").tofile(data)


def _write_trajectory(root: Path, experiment: str, spec: DatasetSpec, offset: int) -> None:
    production = root / "mitgcm" / experiment / "production" / "years_000_001"
    production.mkdir(parents=True)
    y, x = np.indices((spec.ny, spec.nx), dtype=np.float32)
    wet = np.ones((spec.ny, spec.nx), dtype=np.float32)
    wet[[0, -1], :] = 0.0
    wet[:, [0, -1]] = 0.0
    _write_mds(production, "Depth", wet[None])
    _write_mds(production, "XC", x[None])
    _write_mds(production, "YC", y[None])
    (0.01 * (x + 1.0)).astype(">f4").tofile(production / "windx_cosy.bin")
    (production / "segment_result.json").write_text(json.dumps({"end_iteration": offset + 1000}))

    level = np.arange(spec.nr, dtype=np.float32)[:, None, None]
    for record in range(spec.expected_records):
        iteration = offset + record * 72
        u = np.broadcast_to(100.0 * level + x[None] + record, (spec.nr, spec.ny, spec.nx)).astype(np.float32)
        v = np.broadcast_to(200.0 * level + y[None] + record, (spec.nr, spec.ny, spec.nx)).astype(np.float32)
        theta = np.broadcast_to(10.0 * level + 0.1 * x[None] + record, (spec.nr, spec.ny, spec.nx)).astype(np.float32)
        eta = (x + y + record).astype(np.float32)
        _write_mds(
            production,
            f"dynState.{iteration:010d}",
            np.stack((u, v, theta), axis=0),
            ("UVEL", "VVEL", "THETA"),
        )
        _write_mds(
            production,
            f"surfState.{iteration:010d}",
            eta[None],
            ("ETAN",),
        )


def test_build_and_validate_shared_dataset(tmp_path: Path) -> None:
    spec = DatasetSpec(
        expected_records=12,
        nr=2,
        ny=4,
        nx=5,
        horizon_days=1,
        train_stop=5,
        validation_start=6,
        validation_stop=9,
        inference_start=10,
    )
    scratch = tmp_path / "scratch"
    for number, experiment in enumerate(("S0", "S1", "S2")):
        _write_trajectory(scratch, experiment, spec, offset=1000 * (number + 1))

    destination = tmp_path / "trajectories.zarr"
    manifest = build_dataset(scratch, destination, spec=spec, chunk_days=3)
    report = validate_dataset(destination, spec=spec)
    group = zarr.open_consolidated(str(destination), mode="r")

    assert manifest["experiments"] == ["S0", "S1", "S2"]
    assert report["status"] == "valid"
    assert group["state"].shape == (3, 12, 7, 4, 5)
    assert group["state"][0, 0, :, 0, 0].tolist() == [0.0] * 7
    # U centers x-facing velocities; V centers y-facing velocities.
    assert group["state"][0, 0, 0, 1, 1] == 1.5
    assert group["state"][0, 0, 2, 1, 1] == 1.5
    assert group["pair_split"][:].tolist().count(1) == 4
    assert group["pair_split"][:].tolist().count(2) == 2
    assert group["pair_split"][:].tolist().count(3) == 1
    assert np.isfinite(group["state_mean"][:]).all()
    assert np.all(group["state_scale"][:] > 0.0)
