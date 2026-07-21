"""MITgcm MDS to canonical Zarr conversion and reproducible normalization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import config_sha256, experiment
from .manifest import file_record, sha256_file, tree_records, verify_records, write_manifest


STATE_CHANNELS = (
    "U_surface",
    "U_mid",
    "V_surface",
    "V_mid",
    "T_surface",
    "T_mid",
    "PHIHYD_surface",
    "PHIHYD_mid",
    "PHIHYD_bottom",
    "barotropic_streamfunction",
    "wind_stress_x",
)

_DYNAMIC_MDS_NAME = re.compile(r"^dynDiag\.\d+\.(?:meta|data)$")


class DataError(RuntimeError):
    """Raised when raw or reduced data violate the paper schema."""


def _zarr():
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:  # pragma: no cover - exercised on bare login nodes
        raise DataError("data commands require `pip install -e .` in the project environment") from exc
    return zarr, Blosc


def canonical_store_path(config: Mapping[str, Any]) -> Path:
    return Path(config["paths"]["reduced"]) / "mitgcm_state.zarr"


def stats_store_path(config: Mapping[str, Any]) -> Path:
    return Path(config["paths"]["reduced"]) / "normalization.zarr"


def initialize_store(config: Mapping[str, Any], path: str | Path | None = None) -> Path:
    """Create metadata/chunk layout without allocating the full data volume."""
    zarr, Blosc = _zarr()
    path = Path(path or canonical_store_path(config)).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="a")
    n_exp = len(config["experiments"])
    n_time = int(config["mitgcm"]["production_days"])
    n_chan = len(STATE_CHANNELS)
    ny, nx = int(config["grid"]["ny"]), int(config["grid"]["nx"])
    chunks = (
        1,
        int(config["data"]["zarr_time_chunk"]),
        int(config["data"]["zarr_channel_chunk"]),
        int(config["data"]["zarr_y_chunk"]),
        int(config["data"]["zarr_x_chunk"]),
    )
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    state = root.require_dataset(
        "state",
        shape=(n_exp, n_time, n_chan, ny, nx),
        chunks=chunks,
        dtype="f4",
        compressor=compressor,
        fill_value=np.nan,
    )
    state.attrs["dimensions"] = ["experiment", "time", "channel", "y", "x"]
    root.require_dataset("experiment", shape=(n_exp,), dtype="i2")[:] = [
        item["id"] for item in config["experiments"]
    ]
    root.require_dataset("tau0_n_m2", shape=(n_exp,), dtype="f8")[:] = [
        item["tau0_n_m2"] for item in config["experiments"]
    ]
    root.require_dataset("time_day", shape=(n_time,), dtype="i4")[:] = np.arange(n_time)
    root.require_dataset("channel", shape=(n_chan,), dtype="<U32")[:] = STATE_CHANNELS
    grid = config["grid"]
    lon = grid["lon0_deg"] + (np.arange(nx) + 0.5) * grid["dx_deg"]
    lat = grid["lat0_deg"] + (np.arange(ny) + 0.5) * grid["dy_deg"]
    root.require_dataset("lon_deg_e", shape=(nx,), dtype="f8")[:] = lon
    root.require_dataset("lat_deg_n", shape=(ny,), dtype="f8")[:] = lat
    if "completed_experiments" not in root.attrs:
        root.attrs["completed_experiments"] = []
    root.attrs.update(
        {
            "schema_version": 1,
            "config_sha256": config_sha256(config),
            "channel_order": list(STATE_CHANNELS),
            "units": [
                "m s-1",
                "m s-1",
                "m s-1",
                "m s-1",
                "degC",
                "degC",
                "m2 s-2",
                "m2 s-2",
                "m2 s-2",
                "m2 s-1",
                "N m-2",
            ],
            "vertical_interpretation": "15 layers, thicknesses 50..190 m, total 1800 m",
            "midlevel_zero_based_index": int(grid["midlevel_index"]),
            "inference_overlaps_validation": True,
        }
    )
    return path


def wind_stress(config: Mapping[str, Any], tau0: float) -> np.ndarray:
    """Equation (1), evaluated at tracer-cell centers."""
    grid = config["grid"]
    lat = grid["lat0_deg"] + (np.arange(grid["ny"]) + 0.5) * grid["dy_deg"]
    meridional_extent = grid["ny"] * grid["dy_deg"]
    profile = -tau0 * np.cos(2.0 * np.pi * (lat - grid["lat0_deg"]) / meridional_extent)
    return np.repeat(profile[:, None], grid["nx"], axis=1).astype("f4")


def center_staggered(values: np.ndarray, axis: int, target_size: int) -> np.ndarray:
    """Match the archived adjacent-face averaging, including a closed high boundary."""
    axis %= values.ndim
    size = values.shape[axis]
    if size == target_size + 1:
        left = np.take(values, np.arange(target_size), axis=axis)
        right = np.take(values, np.arange(1, target_size + 1), axis=axis)
        return 0.5 * (left + right)
    if size == target_size:
        # MDS stores N west/south faces; the omitted east/north boundary has zero normal flow.
        shifted = np.zeros_like(values)
        source = [slice(None)] * values.ndim
        destination = [slice(None)] * values.ndim
        source[axis] = slice(1, None)
        destination[axis] = slice(None, -1)
        shifted[tuple(destination)] = values[tuple(source)]
        return 0.5 * (values + shifted)
    raise DataError(f"staggered axis has size {size}; expected {target_size} or {target_size + 1}")


def open_mds(raw_dir: str | Path, delta_t_s: float = 300.0):
    try:
        from xmitgcm import open_mdsdataset
    except ImportError as exc:  # pragma: no cover
        raise DataError("MDS conversion requires xmitgcm") from exc
    raw_dir = Path(raw_dir).resolve()
    return open_mdsdataset(
        str(raw_dir),
        grid_dir=str(raw_dir),
        prefix=["dynDiag"],
        iters="all",
        delta_t=delta_t_s,
        geometry="sphericalpolar",
        read_grid=True,
        ignore_unknown_vars=False,
    )


def _dimension(field, kind: str) -> str:
    candidates = [dim for dim in field.dims if dim.lower().startswith(kind.lower())]
    if not candidates:
        raise DataError(f"cannot identify {kind!r} dimension in {field.dims}")
    return candidates[0]


def _time_dimension(field) -> str:
    for dim in field.dims:
        if dim.lower() in {"time", "t"} or dim.lower().startswith("time"):
            return dim
    raise DataError(f"cannot identify time dimension in {field.dims}")


def _vertical_dimension(field) -> str:
    for dim in field.dims:
        if dim.lower().startswith("z") or "md" in dim.lower():
            return dim
    raise DataError(f"cannot identify vertical dimension in {field.dims}")


def _to_tyx(field, time_slice: slice) -> np.ndarray:
    tdim = _time_dimension(field)
    ydim = _dimension(field, "y")
    xdim = _dimension(field, "x")
    selected = field.isel({tdim: time_slice}).transpose(tdim, ydim, xdim)
    return np.asarray(selected.values)


def _level(field, index: int):
    return field.isel({_vertical_dimension(field): index})


def _field(dataset, name: str):
    aliases = {"T": ("THETA", "Theta", "T"), "PHIHYD": ("PHIHYD", "PhiHyd")}
    for candidate in aliases.get(name, (name,)):
        if candidate in dataset:
            return dataset[candidate]
    raise DataError(f"diagnostic {name} not found; available: {sorted(dataset.data_vars)}")


def _channel_chunk(
    dataset, name: str, time_slice: slice, mid: int, ny: int, nx: int
) -> np.ndarray:
    if name == "U_surface":
        return center_staggered(_to_tyx(_level(_field(dataset, "UVEL"), 0), time_slice), -1, nx)
    if name == "U_mid":
        return center_staggered(_to_tyx(_level(_field(dataset, "UVEL"), mid), time_slice), -1, nx)
    if name == "V_surface":
        return center_staggered(_to_tyx(_level(_field(dataset, "VVEL"), 0), time_slice), -2, ny)
    if name == "V_mid":
        return center_staggered(_to_tyx(_level(_field(dataset, "VVEL"), mid), time_slice), -2, ny)
    if name == "T_surface":
        return _to_tyx(_level(_field(dataset, "THETA"), 0), time_slice)
    if name == "T_mid":
        return _to_tyx(_level(_field(dataset, "THETA"), mid), time_slice)
    if name == "PHIHYD_surface":
        return _to_tyx(_level(_field(dataset, "PHIHYD"), 0), time_slice)
    if name == "PHIHYD_mid":
        return _to_tyx(_level(_field(dataset, "PHIHYD"), mid), time_slice)
    if name == "PHIHYD_bottom":
        return _to_tyx(_level(_field(dataset, "PHIHYD"), -1), time_slice)
    if name == "barotropic_streamfunction":
        psi = _field(dataset, "PsiVEL").sum(_vertical_dimension(_field(dataset, "PsiVEL")))
        values = _to_tyx(psi, time_slice)
        values = center_staggered(values, -1, nx)
        return center_staggered(values, -2, ny)
    raise DataError(f"unsupported dynamic channel {name}")


def convert_experiment(
    config: Mapping[str, Any],
    selector: int | str,
    raw_dir: str | Path | None = None,
    store_path: str | Path | None = None,
    allow_partial: bool = False,
) -> Path:
    """Stream one experiment's diagnostics into its canonical Zarr region."""
    zarr, _ = _zarr()
    item = experiment(config, selector)
    exp_index = int(item["id"]) - 1
    raw_dir = Path(
        raw_dir or Path(config["paths"]["raw"]) / item["slug"] / "production"
    ).resolve()
    if list(raw_dir.glob("dynDiag.*.meta")):
        chunk_dirs = [raw_dir]
    else:
        completion_path = raw_dir / "production_complete.json"
        if completion_path.is_file():
            completion = json.loads(completion_path.read_text())
            chunk_dirs = [Path(value).resolve() for value in completion.get("chunks", [])]
            if completion.get("status") != "complete" and not allow_partial:
                raise DataError(f"MITgcm production is not complete: {completion_path}")
        else:
            chunk_dirs = sorted((raw_dir / "chunks").glob("chunk_*"))
            chunk_dirs = [path.resolve() for path in chunk_dirs if (path / "run_result.json").is_file()]
    if not chunk_dirs:
        raise DataError(f"no completed MITgcm diagnostic chunks found under {raw_dir}")
    n_raw_records = sum(len(list(path.glob("dynDiag.*.meta"))) for path in chunk_dirs)
    forcing_candidates = [raw_dir / "wind_for_fno.npy"] + [
        path / "wind_for_fno.npy" for path in chunk_dirs
    ]
    forcing_path = next((path for path in forcing_candidates if path.is_file()), None)
    if forcing_path is None:
        raise DataError(f"exact staged wind_for_fno.npy is missing under {raw_dir}")
    exact_wind = np.load(forcing_path, allow_pickle=False)
    expected_wind_shape = (int(config["grid"]["ny"]), int(config["grid"]["nx"]))
    if exact_wind.shape != expected_wind_shape or not np.isfinite(exact_wind).all():
        raise DataError(f"invalid exact forcing {forcing_path}: shape={exact_wind.shape}")
    forcing_json = forcing_path.with_name("forcing.json")
    if forcing_json.is_file():
        forcing_metadata = json.loads(forcing_json.read_text())
        expected_sha = forcing_metadata.get("fno_forcing", {}).get("sha256")
        if expected_sha and sha256_file(forcing_path) != expected_sha:
            raise DataError(f"forcing checksum mismatch: {forcing_path}")
    analytic_wind = wind_stress(config, float(item["tau0_n_m2"]))
    if not np.allclose(exact_wind, analytic_wind, rtol=1e-6, atol=1e-8):
        raise DataError("staged MITgcm forcing does not match locked Equation (1)")
    path = initialize_store(config, store_path)
    root = zarr.open_group(str(path), mode="a")
    n_expected = int(config["mitgcm"]["production_days"])
    if n_raw_records != n_expected and not allow_partial:
        raise DataError(f"{item['slug']} has {n_raw_records} daily records; expected {n_expected}")
    n_write = min(n_raw_records, n_expected)
    chunk = int(config["data"]["zarr_time_chunk"])
    ny, nx = int(config["grid"]["ny"]), int(config["grid"]["nx"])
    mid = int(config["grid"]["midlevel_index"])
    state = root["state"]
    global_start = 0
    for chunk_dir in chunk_dirs:
        dataset = open_mds(chunk_dir, float(config["mitgcm"]["delta_t_s"]))
        first = _field(dataset, "THETA")
        tdim = _time_dimension(first)
        local_count = min(int(first.sizes[tdim]), n_write - global_start)
        for local_start in range(0, local_count, chunk):
            local_stop = min(local_start + chunk, local_count)
            output_start = global_start + local_start
            output_stop = global_start + local_stop
            for channel_index, name in enumerate(STATE_CHANNELS[:-1]):
                values = _channel_chunk(dataset, name, slice(local_start, local_stop), mid, ny, nx)
                if values.shape != (local_stop - local_start, ny, nx):
                    raise DataError(
                        f"{name} produced {values.shape}, expected "
                        f"{(local_stop-local_start, ny, nx)} in {chunk_dir}"
                    )
                state[exp_index, output_start:output_stop, channel_index, :, :] = values.astype("f4")
            wind_chunk = np.repeat(
                exact_wind[None, :, :], local_stop - local_start, axis=0
            )
            state[exp_index, output_start:output_stop, -1, :, :] = wind_chunk
        global_start += local_count
        close = getattr(dataset, "close", None)
        if close is not None:
            close()
        if global_start >= n_write:
            break
    if global_start != n_write:
        raise DataError(f"converted {global_start} records but expected to write {n_write}")
    completed = set(root.attrs.get("completed_experiments", []))
    if n_write == n_expected:
        completed.add(int(item["id"]))
    root.attrs["completed_experiments"] = sorted(completed)
    root.attrs[f"experiment_{item['id']}_records"] = n_write
    return path


def compute_stats(
    config: Mapping[str, Any],
    store_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Pointwise mean/std over exactly 3 x 6000 training states."""
    zarr, Blosc = _zarr()
    source_path = Path(store_path or canonical_store_path(config)).resolve()
    root = zarr.open_group(str(source_path), mode="r")
    state = root["state"]
    train_ids = [int(value) for value in config["data"]["training_experiments"]]
    start = int(config["data"]["training_start"])
    stop = int(config["data"]["training_stop"])
    expected = set(train_ids)
    completed = set(root.attrs.get("completed_experiments", []))
    if not expected.issubset(completed):
        raise DataError(f"training experiments missing from store: {sorted(expected - completed)}")
    shape = state.shape[2:]
    total = np.zeros(shape, dtype="f8")
    total_sq = np.zeros(shape, dtype="f8")
    count = 0
    chunk = max(1, int(config["data"]["zarr_time_chunk"]))
    for exp_id in train_ids:
        for chunk_start in range(start, stop, chunk):
            chunk_stop = min(chunk_start + chunk, stop)
            values = np.asarray(state[exp_id - 1, chunk_start:chunk_stop], dtype="f8")
            if not np.isfinite(values).all():
                raise DataError(f"non-finite training data in experiment {exp_id}, day {chunk_start}")
            total += values.sum(axis=0)
            total_sq += np.square(values).sum(axis=0)
            count += values.shape[0]
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    epsilon = float(config["data"]["normalization_epsilon"])
    std_safe = np.maximum(std, epsilon)
    output_path = Path(output_path or stats_store_path(config)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = zarr.open_group(str(output_path), mode="w")
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    chunks = (1, shape[1], shape[2])
    out.create_dataset("mean", data=mean.astype("f4"), chunks=chunks, compressor=compressor)
    out.create_dataset("std", data=std_safe.astype("f4"), chunks=chunks, compressor=compressor)
    out.create_dataset("raw_std", data=std.astype("f4"), chunks=chunks, compressor=compressor)
    out.create_dataset("channel", data=np.asarray(STATE_CHANNELS, dtype="<U32"))
    out.attrs.update(
        {
            "schema_version": 1,
            "sample_count": count,
            "expected_sample_count": 18000,
            "experiment_ids": train_ids,
            "time_slice_python": [start, stop],
            "epsilon": epsilon,
            "config_sha256": config_sha256(config),
        }
    )
    if count != 18000:
        raise DataError(f"normalization used {count} states instead of the locked 18000")
    return output_path


def validate_store(config: Mapping[str, Any], store_path: str | Path | None = None) -> dict[str, Any]:
    zarr, _ = _zarr()
    path = Path(store_path or canonical_store_path(config)).resolve()
    root = zarr.open_group(str(path), mode="r")
    expected_shape = (
        5,
        int(config["mitgcm"]["production_days"]),
        11,
        int(config["grid"]["ny"]),
        int(config["grid"]["nx"]),
    )
    errors: list[str] = []
    if tuple(root["state"].shape) != expected_shape:
        errors.append(f"shape {root['state'].shape} != {expected_shape}")
    if tuple(root["channel"][:].tolist()) != STATE_CHANNELS:
        errors.append("channel order mismatch")
    completed = list(root.attrs.get("completed_experiments", []))
    for exp_id in completed:
        sample = np.asarray(root["state"][exp_id - 1, [0, -1], :, :, :])
        if not np.isfinite(sample).all():
            errors.append(f"experiment {exp_id} has non-finite first/last records")
        expected_wind = wind_stress(config, config["experiments"][exp_id - 1]["tau0_n_m2"])
        if not np.allclose(sample[:, -1], expected_wind, atol=1e-7):
            errors.append(f"experiment {exp_id} forcing channel mismatch")
    report = {
        "path": str(path),
        "shape": list(root["state"].shape),
        "completed_experiments": completed,
        "config_sha256": root.attrs.get("config_sha256"),
        "errors": errors,
        "valid": not errors and completed == [1, 2, 3, 4, 5],
    }
    return report


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise DataError(f"{description} is not a JSON object: {path}")
    return value


def _dynamic_mds_files(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("dynDiag.*.meta")) + sorted(
        directory.glob("dynDiag.*.data")
    )
    return sorted(paths, key=lambda path: path.name)


def _paired_dynamic_names(paths: list[Path], description: str) -> None:
    meta = {path.name.removesuffix(".meta") for path in paths if path.suffix == ".meta"}
    data = {path.name.removesuffix(".data") for path in paths if path.suffix == ".data"}
    if meta != data:
        missing_data = sorted(meta - data)[:5]
        missing_meta = sorted(data - meta)[:5]
        raise DataError(
            f"incomplete dynDiag pairs in {description}: "
            f"missing data={missing_data}, missing metadata={missing_meta}"
        )


def _restart_provenance(chunk: Path, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    hashes = result.get("pickup_sha256")
    if not isinstance(hashes, Mapping):
        raise DataError(f"run result has no pickup checksums: {chunk / 'run_result.json'}")
    records: list[dict[str, Any]] = []
    for key, hash_key in (("pickup_meta", "meta"), ("pickup_data", "data")):
        if key not in result or hash_key not in hashes:
            raise DataError(f"run result has incomplete pickup provenance: {chunk}")
        path = Path(str(result[key])).expanduser().resolve()
        if path.parent != chunk or not path.is_file() or path.is_symlink():
            raise DataError(f"unsafe or missing permanent pickup: {path}")
        records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": str(hashes[hash_key]),
            }
        )
    return records


def _raw_experiment_record(
    config: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    """Inventory one completed production without hashing large raw diagnostics."""

    experiment_id = int(item["id"])
    slug = str(item["slug"])
    raw_root = Path(config["paths"]["raw"]).expanduser().resolve()
    scratch_root = Path(config["paths"]["scratch_root"]).expanduser().resolve()
    staging = (raw_root / slug / "production").resolve()
    production = (
        scratch_root / "mitgcm" / "production" / f"exp{experiment_id:02d}_{slug}"
    ).resolve()
    chunks_root = (production / "chunks").resolve()
    if not staging.is_dir() or staging.is_symlink():
        raise DataError(f"raw staging directory is missing or unsafe: {staging}")
    if not production.is_dir() or production.is_symlink():
        raise DataError(f"production directory is missing or unsafe: {production}")

    staging_manifest_path = staging / "staging_manifest.json"
    complete_path = production / "production_complete.json"
    staging_manifest = _load_json(staging_manifest_path, "staging manifest")
    completion = _load_json(complete_path, "production completion manifest")
    expected_days = int(config["mitgcm"]["production_days"])
    expected_config_hash = config_sha256(config)
    if completion.get("status") != "complete":
        raise DataError(f"MITgcm production is not complete: {complete_path}")
    if int(completion.get("experiment_id", -1)) != experiment_id:
        raise DataError(f"production experiment mismatch: {complete_path}")
    if int(completion.get("daily_states", -1)) != expected_days:
        raise DataError(f"production record count mismatch: {complete_path}")
    provenance = completion.get("reproduction_config")
    if not isinstance(provenance, Mapping) or provenance.get(
        "canonical_sha256"
    ) != expected_config_hash:
        raise DataError(f"production configuration mismatch: {complete_path}")
    if Path(str(completion.get("raw_dir", ""))).expanduser().resolve() != staging:
        raise DataError(f"production raw staging path mismatch: {complete_path}")
    if int(staging_manifest.get("daily_meta_count", -1)) != expected_days:
        raise DataError(f"staging record count mismatch: {staging_manifest_path}")

    completion_chunks = [
        Path(str(value)).expanduser().resolve() for value in completion.get("chunks", [])
    ]
    staged_chunks = [
        Path(str(value)).expanduser().resolve()
        for value in staging_manifest.get("source_chunks", [])
    ]
    if not completion_chunks or completion_chunks != staged_chunks:
        raise DataError(
            f"production and staging chunk inventories disagree for experiment {experiment_id}"
        )
    if len(set(completion_chunks)) != len(completion_chunks):
        raise DataError(f"duplicate source chunks for experiment {experiment_id}")

    source_targets: list[dict[str, Any]] = []
    source_by_name: dict[str, Path] = {}
    provenance_files = [file_record(complete_path), file_record(staging_manifest_path)]
    restart_files: list[dict[str, Any]] = []
    for chunk in completion_chunks:
        if chunk.parent != chunks_root or not chunk.is_dir() or chunk.is_symlink():
            raise DataError(f"unsafe production chunk: {chunk}")
        manifest_path = chunk / "run_manifest.json"
        result_path = chunk / "run_result.json"
        if not manifest_path.is_file() or not result_path.is_file():
            raise DataError(f"production chunk lacks immutable run metadata: {chunk}")
        provenance_files.extend((file_record(manifest_path), file_record(result_path)))
        restart_files.extend(_restart_provenance(chunk, _load_json(result_path, "run result")))

        diagnostics = _dynamic_mds_files(chunk)
        if not diagnostics or any(path.is_symlink() or not path.is_file() for path in diagnostics):
            raise DataError(f"production diagnostics are missing or unsafe: {chunk}")
        _paired_dynamic_names(diagnostics, str(chunk))
        entries: list[dict[str, Any]] = []
        for path in diagnostics:
            if not _DYNAMIC_MDS_NAME.fullmatch(path.name):
                raise DataError(f"unexpected diagnostic filename: {path}")
            if path.name in source_by_name:
                raise DataError(f"duplicate diagnostic filename across chunks: {path.name}")
            source_by_name[path.name] = path.resolve()
            entries.append({"name": path.name, "bytes": path.stat().st_size})
        source_targets.append(
            {
                "kind": "source_chunk_diagnostics",
                "directory": str(chunk),
                "entries": entries,
            }
        )

    source_meta_count = sum(
        1 for name in source_by_name if name.endswith(".meta")
    )
    if source_meta_count != expected_days:
        raise DataError(
            f"source chunks contain {source_meta_count} daily records; expected {expected_days}"
        )

    staged_diagnostics = _dynamic_mds_files(staging)
    _paired_dynamic_names(staged_diagnostics, str(staging))
    if {path.name for path in staged_diagnostics} != set(source_by_name):
        raise DataError(f"staging links do not exactly match source chunks: {staging}")
    staging_entries: list[dict[str, Any]] = []
    for path in staged_diagnostics:
        source = source_by_name[path.name]
        if not path.is_symlink() or path.resolve() != source:
            raise DataError(f"staging entry is not the expected source link: {path}")
        staging_entries.append(
            {"name": path.name, "target": str(source), "bytes": source.stat().st_size}
        )

    return {
        "experiment_id": experiment_id,
        "slug": slug,
        "production_directory": str(production),
        "staging_directory": str(staging),
        "source_chunk_directories": [str(path) for path in completion_chunks],
        "daily_records": source_meta_count,
        "preserved_provenance_files": provenance_files,
        "preserved_restart_files": restart_files,
        "cleanup_targets": [
            {
                "kind": "staging_diagnostic_links",
                "directory": str(staging),
                "entries": staging_entries,
            },
            *source_targets,
        ],
    }


def archive_manifest(
    config: Mapping[str, Any],
    store_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> Path:
    """Seal a valid reduced store and its exact, cleanup-safe raw lineage."""

    store_path = Path(store_path or canonical_store_path(config)).resolve()
    manifest_path = Path(
        manifest_path
        or Path(config["paths"]["project_root"]) / "manifests" / "reduced-data.json"
    )
    validation = validate_store(config, store_path)
    if not validation["valid"]:
        raise DataError("refusing to seal an incomplete or invalid reduced store")
    raw_sources = [
        _raw_experiment_record(config, item) for item in config["experiments"]
    ]
    payload = {
        "kind": "reduced_data",
        "schema_version": 2,
        "config_sha256": config_sha256(config),
        "files": tree_records(store_path),
        "validation": validation,
        "raw_production_sources": raw_sources,
    }
    return write_manifest(manifest_path, payload)


def _cleanup_inventory(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[list[Path], list[tuple[str, Path, Mapping[str, Any]]]]:
    raw_root = Path(config["paths"]["raw"]).expanduser().resolve()
    scratch_root = Path(config["paths"]["scratch_root"]).expanduser().resolve()
    expected_experiments = {
        int(item["id"]): item for item in config["experiments"]
    }
    raw_sources = manifest.get("raw_production_sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != len(expected_experiments):
        raise DataError("manifest has no complete raw production source inventory")

    directories: list[Path] = []
    operations: list[tuple[str, Path, Mapping[str, Any]]] = []
    seen_experiments: set[int] = set()
    seen_paths: set[Path] = set()
    staging_operations: list[tuple[str, Path, Mapping[str, Any]]] = []
    source_operations: list[tuple[str, Path, Mapping[str, Any]]] = []

    for record in raw_sources:
        if not isinstance(record, Mapping):
            raise DataError("invalid raw production source record")
        experiment_id = int(record.get("experiment_id", -1))
        if experiment_id not in expected_experiments or experiment_id in seen_experiments:
            raise DataError(f"invalid or duplicate raw experiment record: {experiment_id}")
        seen_experiments.add(experiment_id)
        item = expected_experiments[experiment_id]
        slug = str(item["slug"])
        if record.get("slug") != slug:
            raise DataError(f"raw source slug mismatch for experiment {experiment_id}")

        expected_staging = (raw_root / slug / "production").resolve()
        expected_production = (
            scratch_root
            / "mitgcm"
            / "production"
            / f"exp{experiment_id:02d}_{slug}"
        ).resolve()
        staging = Path(str(record.get("staging_directory", ""))).resolve()
        production = Path(str(record.get("production_directory", ""))).resolve()
        if staging != expected_staging or production != expected_production:
            raise DataError(f"unsafe raw source roots for experiment {experiment_id}")
        source_directories = {
            Path(str(path)).resolve()
            for path in record.get("source_chunk_directories", [])
        }
        if not source_directories or any(
            path.parent != (production / "chunks").resolve()
            for path in source_directories
        ):
            raise DataError(f"unsafe source chunk inventory for experiment {experiment_id}")

        targets = record.get("cleanup_targets")
        if not isinstance(targets, list) or not targets:
            raise DataError(f"raw experiment has no cleanup targets: {experiment_id}")
        record_staging_operations: list[
            tuple[str, Path, Mapping[str, Any]]
        ] = []
        record_source_operations: list[
            tuple[str, Path, Mapping[str, Any]]
        ] = []
        source_target_directories: set[Path] = set()
        for target in targets:
            if not isinstance(target, Mapping):
                raise DataError("invalid cleanup target record")
            kind = str(target.get("kind", ""))
            directory = Path(str(target.get("directory", ""))).resolve()
            if kind == "staging_diagnostic_links":
                if directory != staging:
                    raise DataError(f"unsafe staging cleanup target: {directory}")
            elif kind == "source_chunk_diagnostics":
                if directory not in source_directories:
                    raise DataError(f"unlisted source chunk cleanup target: {directory}")
                source_target_directories.add(directory)
            else:
                raise DataError(f"unsupported raw cleanup target kind: {kind!r}")
            if directory not in directories:
                directories.append(directory)
            entries = target.get("entries")
            if not isinstance(entries, list) or not entries:
                raise DataError(f"cleanup target has no explicit entries: {directory}")
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise DataError(f"invalid cleanup entry in {directory}")
                name = str(entry.get("name", ""))
                if not _DYNAMIC_MDS_NAME.fullmatch(name) or Path(name).name != name:
                    raise DataError(f"unsafe cleanup entry name: {name!r}")
                path = directory / name
                if path in seen_paths:
                    raise DataError(f"duplicate cleanup entry: {path}")
                seen_paths.add(path)
                operation = (kind, path, entry)
                if kind == "staging_diagnostic_links":
                    record_staging_operations.append(operation)
                else:
                    record_source_operations.append(operation)

        if source_target_directories != source_directories:
            raise DataError(
                f"cleanup targets do not cover every source chunk for experiment "
                f"{experiment_id}"
            )
        if not record_staging_operations:
            raise DataError(f"no staging cleanup entries for experiment {experiment_id}")
        record_source_entries = {path for _, path, _ in record_source_operations}
        for _, _, entry in record_staging_operations:
            target = Path(str(entry.get("target", ""))).resolve()
            if target not in record_source_entries:
                raise DataError(
                    f"staging link target is not an explicit source entry: {target}"
                )
        staging_operations.extend(record_staging_operations)
        source_operations.extend(record_source_operations)

        provenance_records = record.get("preserved_provenance_files")
        if not isinstance(provenance_records, list) or not provenance_records:
            raise DataError(f"no preserved production provenance for experiment {experiment_id}")
        provenance_errors = verify_records(provenance_records)
        if provenance_errors:
            raise DataError(
                "production provenance verification failed: "
                + "; ".join(provenance_errors[:5])
            )
        restart_records = record.get("preserved_restart_files")
        if not isinstance(restart_records, list) or len(restart_records) != 2 * len(
            source_directories
        ):
            raise DataError(f"incomplete restart provenance for experiment {experiment_id}")
        for restart in restart_records:
            if not isinstance(restart, Mapping):
                raise DataError(f"invalid restart provenance for experiment {experiment_id}")
            path = Path(str(restart.get("path", ""))).resolve()
            checksum = str(restart.get("sha256", ""))
            if (
                path.parent not in source_directories
                or not path.name.startswith("pickup.")
                or path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != int(restart.get("bytes", -1))
                or len(checksum) != 64
            ):
                raise DataError(f"restart provenance changed or is unsafe: {path}")

    if seen_experiments != set(expected_experiments):
        raise DataError("raw source inventory does not cover every experiment")
    operations.extend(staging_operations)
    operations.extend(source_operations)
    return directories, operations


def _validate_cleanup_entry(
    kind: str, path: Path, entry: Mapping[str, Any]
) -> bool:
    """Return whether an exact manifest entry still exists and is safe to unlink."""

    if not path.exists() and not path.is_symlink():
        return False
    try:
        expected_bytes = int(entry["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(f"cleanup entry has no valid sealed size: {path}") from exc
    if kind == "staging_diagnostic_links":
        try:
            expected_target = Path(str(entry["target"])).resolve()
        except KeyError as exc:
            raise DataError(f"staging cleanup entry has no target: {path}") from exc
        if not path.is_symlink() or path.resolve() != expected_target:
            raise DataError(f"staging cleanup entry changed since sealing: {path}")
        if not expected_target.is_file() or expected_target.stat().st_size != expected_bytes:
            raise DataError(f"staging source size changed since sealing: {expected_target}")
    else:
        if path.is_symlink() or not path.is_file():
            raise DataError(f"source cleanup entry changed type since sealing: {path}")
        if path.stat().st_size != expected_bytes:
            raise DataError(f"source size changed since sealing: {path}")
    return True


def cleanup_raw(
    config: Mapping[str, Any], manifest_path: str | Path, execute: bool = False
) -> list[Path]:
    """Remove only sealed dynDiag files/links; preserve run and restart provenance."""

    manifest = _load_json(Path(manifest_path).expanduser().resolve(), "data manifest")
    if manifest.get("kind") != "reduced_data" or manifest.get("schema_version") != 2:
        raise DataError("manifest is not a cleanup-capable reduced-data seal")
    if manifest.get("config_sha256") != config_sha256(config):
        raise DataError("manifest configuration does not match the active configuration")
    reduced_records = manifest.get("files")
    if not isinstance(reduced_records, list) or not reduced_records:
        raise DataError("manifest contains no reduced product checksums")
    errors = verify_records(reduced_records)
    if errors:
        raise DataError("reduced product verification failed: " + "; ".join(errors[:5]))
    if not manifest.get("validation", {}).get("valid"):
        raise DataError("manifest does not contain a passing data validation")

    directories, operations = _cleanup_inventory(config, manifest)
    existing = [
        (kind, path, entry)
        for kind, path, entry in operations
        if _validate_cleanup_entry(kind, path, entry)
    ]
    if execute:
        for kind, path, entry in existing:
            if _validate_cleanup_entry(kind, path, entry):
                path.unlink()
    return directories
