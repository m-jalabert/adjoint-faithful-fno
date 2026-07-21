import json
from pathlib import Path

import numpy as np
import pytest

import bire_repro.data as data_module
from bire_repro.config import load_config
from bire_repro.config import config_sha256
from bire_repro.data import (
    STATE_CHANNELS,
    DataError,
    archive_manifest,
    center_staggered,
    cleanup_raw,
    wind_stress,
)
from bire_repro.manifest import sha256_file


def test_center_staggered_n_plus_one():
    faces = np.asarray([0.0, 2.0, 4.0, 0.0])
    np.testing.assert_allclose(center_staggered(faces, 0, 3), [1.0, 3.0, 2.0])


def test_center_staggered_mds_closed_boundary():
    faces = np.asarray([0.0, 2.0, 4.0])
    np.testing.assert_allclose(center_staggered(faces, 0, 3), [1.0, 3.0, 2.0])


def test_wind_channel_shape_and_amplitude():
    config = load_config()
    wind = wind_stress(config, 0.1)
    assert wind.shape == (248, 248)
    assert len(STATE_CHANNELS) == 11
    assert np.max(np.abs(wind)) <= 0.1


def test_convert_experiment_broadcasts_wind_across_time_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zarr = pytest.importorskip("zarr")
    config = load_config()
    config["grid"].update({"nx": 3, "ny": 2})
    config["mitgcm"]["production_days"] = 10
    config["data"].update(
        {
            "zarr_time_chunk": 8,
            "zarr_channel_chunk": 1,
            "zarr_y_chunk": 2,
            "zarr_x_chunk": 3,
        }
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    for iteration in range(10):
        (raw / f"dynDiag.{iteration:010d}.meta").write_text("metadata\n")
    wind = wind_stress(config, config["experiments"][0]["tau0_n_m2"])
    np.save(raw / "wind_for_fno.npy", wind, allow_pickle=False)

    class FakeField:
        dims = ("time",)
        sizes = {"time": 10}

    monkeypatch.setattr(data_module, "open_mds", lambda *_args: {"THETA": FakeField()})
    monkeypatch.setattr(
        data_module,
        "_channel_chunk",
        lambda _dataset, _name, time_slice, _mid, ny, nx: np.zeros(
            (time_slice.stop - time_slice.start, ny, nx), dtype="f4"
        ),
    )
    store = tmp_path / "state.zarr"
    data_module.convert_experiment(config, 1, raw, store)

    root = zarr.open_group(str(store), mode="r")
    state = root["state"]
    assert state.chunks[1] == 8
    assert root.attrs["experiment_1_records"] == 10
    np.testing.assert_array_equal(
        state[0, :10, -1, :, :], np.repeat(wind[None, :, :], 10, axis=0)
    )


def _raw_cleanup_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project = tmp_path / "project"
    scratch = tmp_path / "scratch"
    raw = scratch / "raw"
    store = scratch / "reduced" / "state.zarr"
    store.mkdir(parents=True)
    reduced_file = store / "state-chunk"
    reduced_file.write_bytes(b"sealed reduced state")
    config = {
        "paths": {
            "project_root": str(project),
            "scratch_root": str(scratch),
            "raw": str(raw),
        },
        "mitgcm": {"production_days": 2},
        "experiments": [{"id": 1, "slug": "low_wind"}],
    }
    monkeypatch.setattr(
        data_module,
        "validate_store",
        lambda _config, path: {
            "path": str(path),
            "completed_experiments": [1],
            "errors": [],
            "valid": True,
        },
    )

    production = scratch / "mitgcm" / "production" / "exp01_low_wind"
    staging = raw / "low_wind" / "production"
    staging.mkdir(parents=True)
    chunks: list[Path] = []
    diagnostic_paths: list[Path] = []
    preserved_paths: list[Path] = []
    for index, timestep in enumerate((288, 576)):
        chunk = production / "chunks" / f"chunk_{index:04d}_{index + 1:04d}"
        chunk.mkdir(parents=True)
        chunks.append(chunk.resolve())
        meta = chunk / f"dynDiag.{timestep:010d}.meta"
        binary = meta.with_suffix(".data")
        meta.write_text(f"timeStepNumber = [ {timestep} ];\n")
        binary.write_bytes(f"diagnostic-{timestep}".encode())
        diagnostic_paths.extend((meta, binary))

        run_manifest = chunk / "run_manifest.json"
        run_manifest.write_text(json.dumps({"chunk": index}))
        pickup_meta = chunk / f"pickup.{timestep:010d}.meta"
        pickup_data = pickup_meta.with_suffix(".data")
        pickup_meta.write_text("permanent restart metadata")
        pickup_data.write_bytes(b"permanent restart state")
        run_result = chunk / "run_result.json"
        run_result.write_text(
            json.dumps(
                {
                    "pickup_meta": str(pickup_meta.resolve()),
                    "pickup_data": str(pickup_data.resolve()),
                    "pickup_sha256": {
                        "meta": sha256_file(pickup_meta),
                        "data": sha256_file(pickup_data),
                    },
                }
            )
        )
        run_log = chunk / "run.log"
        run_log.write_text("immutable model log")
        grid = chunk / "XC.data"
        grid.write_bytes(b"preserved grid")
        preserved_paths.extend(
            (run_manifest, run_result, pickup_meta, pickup_data, run_log, grid)
        )

    completion = {
        "experiment_id": 1,
        "status": "complete",
        "daily_states": 2,
        "chunks": [str(path) for path in chunks],
        "raw_dir": str(staging.resolve()),
        "reproduction_config": {"canonical_sha256": config_sha256(config)},
    }
    production.mkdir(parents=True, exist_ok=True)
    completion_path = production / "production_complete.json"
    completion_path.write_text(json.dumps(completion))
    staging_manifest_path = staging / "staging_manifest.json"
    staging_manifest_path.write_text(
        json.dumps(
            {
                "source_chunks": [str(path) for path in chunks],
                "daily_meta_count": 2,
            }
        )
    )
    for source in diagnostic_paths:
        (staging / source.name).symlink_to(source.resolve())

    manifest_path = project / "manifests" / "reduced-data.json"
    return {
        "config": config,
        "store": store,
        "reduced_file": reduced_file,
        "manifest": manifest_path,
        "staging": staging,
        "chunks": chunks,
        "diagnostics": diagnostic_paths,
        "preserved": preserved_paths + [completion_path, staging_manifest_path],
    }


def test_archive_cleanup_is_explicit_dry_run_safe_and_preserves_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _raw_cleanup_fixture(tmp_path, monkeypatch)
    manifest_path = archive_manifest(
        fixture["config"], fixture["store"], fixture["manifest"]
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    source = manifest["raw_production_sources"][0]
    assert source["staging_directory"] == str(fixture["staging"].resolve())
    assert source["source_chunk_directories"] == [str(path) for path in fixture["chunks"]]
    assert len(source["cleanup_targets"]) == 3
    assert len(source["preserved_restart_files"]) == 4

    dry_run_targets = cleanup_raw(fixture["config"], manifest_path, execute=False)
    assert dry_run_targets == [fixture["staging"].resolve(), *fixture["chunks"]]
    assert all(path.exists() for path in fixture["diagnostics"])
    assert all((fixture["staging"] / path.name).is_symlink() for path in fixture["diagnostics"])

    # A file created after sealing is not an explicit manifest target and must survive.
    unlisted = fixture["chunks"][0] / "dynDiag.9999999999.data"
    unlisted.write_bytes(b"not in sealed inventory")
    cleanup_raw(fixture["config"], manifest_path, execute=True)
    assert all(not path.exists() for path in fixture["diagnostics"])
    assert all(not (fixture["staging"] / path.name).exists() for path in fixture["diagnostics"])
    assert unlisted.is_file()
    assert all(path.exists() for path in fixture["preserved"])
    assert all(path.is_dir() for path in [fixture["staging"], *fixture["chunks"]])

    # Missing listed entries are treated as already cleaned, making resumption safe.
    assert cleanup_raw(fixture["config"], manifest_path, execute=True) == dry_run_targets


def test_cleanup_verifies_reduced_checksums_before_any_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _raw_cleanup_fixture(tmp_path, monkeypatch)
    manifest_path = archive_manifest(
        fixture["config"], fixture["store"], fixture["manifest"]
    )
    fixture["reduced_file"].write_bytes(b"tampered reduced state")
    with pytest.raises(DataError, match="reduced product verification failed"):
        cleanup_raw(fixture["config"], manifest_path, execute=True)
    assert all(path.exists() for path in fixture["diagnostics"])
    assert all((fixture["staging"] / path.name).is_symlink() for path in fixture["diagnostics"])


def test_cleanup_refuses_changed_production_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _raw_cleanup_fixture(tmp_path, monkeypatch)
    manifest_path = archive_manifest(
        fixture["config"], fixture["store"], fixture["manifest"]
    )
    run_manifest = fixture["chunks"][0] / "run_manifest.json"
    run_manifest.write_text('{"changed": true}')
    with pytest.raises(DataError, match="production provenance verification failed"):
        cleanup_raw(fixture["config"], manifest_path, execute=True)
    assert all(path.exists() for path in fixture["diagnostics"])
