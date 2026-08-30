"""Byte-level contract tests for sparse forward-response pickup edits."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SRC = PROJECT_ROOT / "archive" / "src"
if str(ARCHIVE_SRC) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_SRC))

from bire_repro.af_s0_twin import (  # noqa: E402
    DEFAULT_SPEC,
    PICKUP_DTYPE,
    PICKUP_FIELD_LAYOUT,
    PICKUP_GRID,
    TWIN_START_ITERATION,
    PickupEdit,
    TwinExperimentError,
    pickup_record_index,
    write_declared_pickup_edits,
    write_perturbed_pickup,
)
from bire_repro.af_s0_twin2 import SPEC as TWIN2_SPEC  # noqa: E402


GOLDEN_SOURCE = Path(
    "/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm/S0/"
    "spinup/years_090_100/pickup.0002592000.meta"
)
GOLDEN_HASHES = {
    "source_meta": "0898b7d838b7eb53cbaa7fdd3a98d065e4715d9f67105f3c6b5de9eaae55659a",
    "source_data": "0b1f35893e06650c56c81e24e0371097e727d0afc228e80c7a60132fb94584f4",
    "epsilon_1e-6": "97ef9b8eb846ace99c333c7777b78c102a089a7757105ec537af915b0184141c",
    "epsilon_1e-3": "a1ee0e6aef27c6754c398268ca0592fbd426f939c5cf6f6ad5b4b99c8fdc59f2",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_synthetic_pickup(root: Path, *, iteration: int = 1234) -> Path:
    meta_path = root / f"pickup.{iteration:010d}.meta"
    data_path = meta_path.with_suffix(".data")
    fields = " ".join(f"'{name:<8}'" for name, _ in PICKUP_FIELD_LAYOUT)
    meta_path.write_text(
        " nDims = [   2 ];\n"
        " dimList = [\n"
        "  62,    1,   62,\n"
        "  62,    1,   62\n"
        " ];\n"
        " dataprec = [ 'float64' ];\n"
        f" nrecords = [ {sum(count for _, count in PICKUP_FIELD_LAYOUT)} ];\n"
        f" timeStepNumber = [ {iteration} ];\n"
        f" fldList = {{ {fields} }};\n"
    )
    shape = (sum(count for _, count in PICKUP_FIELD_LAYOUT), *PICKUP_GRID)
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    values = (values % 4093) / 4093.0 + 0.125
    values[pickup_record_index("Theta", 1) : pickup_record_index("Theta", 15) + 1] = 8.0
    values[0, 0, 0] = 0.0
    values.astype(PICKUP_DTYPE).tofile(data_path)
    return meta_path


@pytest.fixture
def synthetic_pickup(tmp_path: Path) -> Path:
    return _write_synthetic_pickup(tmp_path / "source")


@pytest.fixture(autouse=True)
def _make_source_directory(tmp_path: Path) -> None:
    (tmp_path / "source").mkdir(exist_ok=True)


def test_pickup_record_layout_is_frozen() -> None:
    assert pickup_record_index("Uvel", 1) == 0
    assert pickup_record_index("Uvel", 15) == 14
    assert pickup_record_index("Vvel", 1) == 15
    assert pickup_record_index("Theta", 1) == 30
    assert pickup_record_index("Salt", 1) == 45
    assert pickup_record_index("GuNm1", 1) == 60
    assert pickup_record_index("GvNm1", 1) == 75
    assert pickup_record_index("GtNm1", 1) == 90
    assert pickup_record_index("EtaN", 1) == 105
    assert pickup_record_index("dEtaHdt", 1) == 106
    assert pickup_record_index("EtaH", 1) == 107
    with pytest.raises(TwinExperimentError):
        pickup_record_index("Uvel", 0)
    with pytest.raises(TwinExperimentError):
        pickup_record_index("EtaN", 2)
    with pytest.raises(TwinExperimentError):
        pickup_record_index("not-a-field", 1)


def test_additive_cell_edit_changes_exactly_one_eight_byte_word(
    synthetic_pickup: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "edited"
    run_dir.mkdir()
    record, j, i = pickup_record_index("EtaN", 1), 7, 11
    source_data = synthetic_pickup.with_suffix(".data")
    before = np.fromfile(source_data, dtype=PICKUP_DTYPE).reshape(-1, *PICKUP_GRID)

    manifest = write_declared_pickup_edits(
        synthetic_pickup,
        run_dir,
        expected_iteration=1234,
        declared_fields=("EtaN",),
        edits=(PickupEdit(record=record, j=j, i=i, value=0.25),),
        operation="add",
    )

    output_meta = Path(manifest["edited_pickup_meta"])
    output_data = Path(manifest["edited_pickup_data"])
    after = np.fromfile(output_data, dtype=PICKUP_DTYPE).reshape(-1, *PICKUP_GRID)
    changed = np.flatnonzero(
        np.frombuffer(source_data.read_bytes(), dtype="V8")
        != np.frombuffer(output_data.read_bytes(), dtype="V8")
    )
    expected_word = record * PICKUP_GRID[0] * PICKUP_GRID[1] + j * PICKUP_GRID[1] + i
    assert changed.tolist() == [expected_word]
    assert after[record, j, i] == before[record, j, i] + 0.25
    assert output_data.stat().st_size == source_data.stat().st_size
    assert output_meta.read_bytes() == synthetic_pickup.read_bytes()
    assert manifest["changed_value_count"] == 1
    assert manifest["declared_fields"] == ["EtaN"]


def test_multiple_declared_cell_edits_preserve_every_other_raw_word(
    synthetic_pickup: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "edited"
    run_dir.mkdir()
    edits = (
        PickupEdit(pickup_record_index("Uvel", 4), 2, 3, 0.125),
        PickupEdit(pickup_record_index("Theta", 9), 31, 17, -0.5),
        PickupEdit(pickup_record_index("EtaN", 1), 61, 61, 0.01),
    )
    manifest = write_declared_pickup_edits(
        synthetic_pickup,
        run_dir,
        expected_iteration=1234,
        declared_fields=("Uvel", "Theta", "EtaN"),
        edits=edits,
        operation="add",
    )
    source_words = np.frombuffer(synthetic_pickup.with_suffix(".data").read_bytes(), dtype="V8")
    output_words = np.frombuffer(Path(manifest["edited_pickup_data"]).read_bytes(), dtype="V8")
    changed = set(np.flatnonzero(source_words != output_words).tolist())
    plane_size = PICKUP_GRID[0] * PICKUP_GRID[1]
    expected = {edit.record * plane_size + edit.j * PICKUP_GRID[1] + edit.i for edit in edits}
    assert changed == expected


def test_legacy_whole_record_multiplication_is_still_available(
    synthetic_pickup: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "edited"
    run_dir.mkdir()
    record = pickup_record_index("Uvel", 2)
    source = np.fromfile(synthetic_pickup.with_suffix(".data"), dtype=PICKUP_DTYPE).reshape(
        -1, *PICKUP_GRID
    )
    manifest = write_declared_pickup_edits(
        synthetic_pickup,
        run_dir,
        expected_iteration=1234,
        declared_fields=("Uvel",),
        edits=(PickupEdit(record, None, None, 1.001),),
        operation="multiply",
    )
    output = np.fromfile(manifest["edited_pickup_data"], dtype=PICKUP_DTYPE).reshape(
        -1, *PICKUP_GRID
    )
    assert np.array_equal(output[record], source[record] * 1.001)
    assert np.array_equal(output[:record], source[:record])
    assert np.array_equal(output[record + 1 :], source[record + 1 :])


@pytest.mark.parametrize(
    ("declared", "edits", "operation"),
    [
        (("EtaN",), (PickupEdit(105, 1, None, 0.1),), "add"),
        (("EtaN",), (PickupEdit(105, None, None, 0.1),), "add"),
        (("EtaN",), (PickupEdit(105, 1, 1, 0.0),), "add"),
        (("EtaN",), (PickupEdit(105, 1, 1, "0.1"),), "add"),
        (("EtaN",), (PickupEdit(105, 1, 1, float("nan")),), "add"),
        (("EtaN",), (PickupEdit(105, 1, 1, float("inf")),), "add"),
        (("EtaN",), (PickupEdit(105, 62, 1, 0.1),), "add"),
        (("EtaN",), (PickupEdit(105, 1, -1, 0.1),), "add"),
        (("EtaN",), (PickupEdit(108, 1, 1, 0.1),), "add"),
        (("EtaN",), (PickupEdit(105, 1, 1, 1.0),), "multiply"),
        (("Theta",), (PickupEdit(30, None, None, 1.001),), "multiply"),
        (("Uvel",), (PickupEdit(0, 1, 1, 1.001),), "multiply"),
        (("Salt",), (PickupEdit(45, 1, 1, 0.1),), "add"),
        (("EtaN",), (PickupEdit(105, 1, 1, 0.1), PickupEdit(105, 1, 1, 0.2)), "add"),
        (("Theta",), (PickupEdit(105, 1, 1, 0.1),), "add"),
        (("Theta", "EtaN"), (PickupEdit(105, 1, 1, 0.1),), "add"),
    ],
)
def test_ambiguous_or_invalid_edits_fail_closed(
    synthetic_pickup: Path,
    tmp_path: Path,
    declared: tuple[str, ...],
    edits: tuple[PickupEdit, ...],
    operation: str,
) -> None:
    run_dir = tmp_path / "edited"
    run_dir.mkdir()
    with pytest.raises(TwinExperimentError):
        write_declared_pickup_edits(
            synthetic_pickup,
            run_dir,
            expected_iteration=1234,
            declared_fields=declared,
            edits=edits,
            operation=operation,
        )
    assert not (run_dir / synthetic_pickup.name).exists()
    assert not (run_dir / synthetic_pickup.with_suffix(".data").name).exists()


def test_numerically_ineffective_cell_edit_fails_before_copy(
    synthetic_pickup: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "edited"
    run_dir.mkdir()
    with pytest.raises(TwinExperimentError, match="produces no byte change"):
        write_declared_pickup_edits(
            synthetic_pickup,
            run_dir,
            expected_iteration=1234,
            declared_fields=("EtaN",),
            edits=(PickupEdit(105, 1, 1, np.finfo(np.float64).tiny),),
            operation="add",
        )
    assert not any(run_dir.iterdir())


def test_edit_that_would_write_nonfinite_data_fails_before_copy(
    synthetic_pickup: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "edited"
    run_dir.mkdir()
    with pytest.raises(TwinExperimentError, match="is non-finite"):
        write_declared_pickup_edits(
            synthetic_pickup,
            run_dir,
            expected_iteration=1234,
            declared_fields=("Uvel",),
            edits=(
                PickupEdit(
                    pickup_record_index("Uvel", 1),
                    None,
                    None,
                    np.finfo(float).max,
                ),
            ),
            operation="multiply",
        )
    assert not any(run_dir.iterdir())


def test_iteration_mismatch_fails_before_copy(synthetic_pickup: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "edited"
    run_dir.mkdir()
    with pytest.raises(TwinExperimentError, match="expected 1235"):
        write_declared_pickup_edits(
            synthetic_pickup,
            run_dir,
            expected_iteration=1235,
            declared_fields=("EtaN",),
            edits=(PickupEdit(105, 1, 1, 0.1),),
            operation="add",
        )
    assert not any(run_dir.iterdir())


def test_existing_destination_is_never_overwritten(
    synthetic_pickup: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "edited"
    run_dir.mkdir()
    destination = run_dir / synthetic_pickup.name
    destination.write_bytes(b"preexisting")
    with pytest.raises(TwinExperimentError, match="refusing to overwrite"):
        write_declared_pickup_edits(
            synthetic_pickup,
            run_dir,
            expected_iteration=1234,
            declared_fields=("EtaN",),
            edits=(PickupEdit(105, 1, 1, 0.1),),
            operation="add",
        )
    assert destination.read_bytes() == b"preexisting"
    assert not (run_dir / synthetic_pickup.with_suffix(".data").name).exists()


@pytest.mark.parametrize("extra_bytes", [-8, 8])
def test_source_data_must_have_exact_declared_size(
    synthetic_pickup: Path, tmp_path: Path, extra_bytes: int
) -> None:
    data_path = synthetic_pickup.with_suffix(".data")
    raw = data_path.read_bytes()
    data_path.write_bytes(raw[:extra_bytes] if extra_bytes < 0 else raw + b"\0" * extra_bytes)
    run_dir = tmp_path / "edited"
    run_dir.mkdir()
    with pytest.raises(TwinExperimentError, match="expected exactly"):
        write_declared_pickup_edits(
            synthetic_pickup,
            run_dir,
            expected_iteration=1234,
            declared_fields=("EtaN",),
            edits=(PickupEdit(105, 1, 1, 0.1),),
            operation="add",
        )


@pytest.mark.skipif(not GOLDEN_SOURCE.is_file(), reason="immutable production source unavailable")
@pytest.mark.parametrize(
    ("spec", "expected_hash"),
    [
        (DEFAULT_SPEC, GOLDEN_HASHES["epsilon_1e-6"]),
        (TWIN2_SPEC, GOLDEN_HASHES["epsilon_1e-3"]),
    ],
)
def test_legacy_wrapper_matches_immutable_golden_pickups(
    tmp_path: Path, spec: object, expected_hash: str
) -> None:
    assert _sha256(GOLDEN_SOURCE) == GOLDEN_HASHES["source_meta"]
    assert _sha256(GOLDEN_SOURCE.with_suffix(".data")) == GOLDEN_HASHES["source_data"]
    run_dir = tmp_path / spec.label
    run_dir.mkdir()

    manifest = write_perturbed_pickup(GOLDEN_SOURCE, run_dir, spec=spec)

    assert _sha256(Path(manifest["twin_pickup_meta"])) == GOLDEN_HASHES["source_meta"]
    assert _sha256(Path(manifest["twin_pickup_data"])) == expected_hash
    assert manifest["source_pickup_sha256"] == {
        "meta": GOLDEN_HASHES["source_meta"],
        "data": GOLDEN_HASHES["source_data"],
    }
    assert set(manifest) == {
        "applied",
        "epsilon",
        "mode",
        "formula",
        "fields",
        "iteration",
        "dataprec",
        "meta_copied_verbatim",
        "source_pickup_meta",
        "source_pickup_data",
        "twin_pickup_meta",
        "twin_pickup_data",
        "source_pickup_sha256",
        "twin_pickup_sha256",
        "field_statistics",
    }
    assert manifest["iteration"] == TWIN_START_ITERATION
