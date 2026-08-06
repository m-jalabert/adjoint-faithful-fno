from pathlib import Path

import numpy as np

from bire_repro.mds import MDSMeta, mds_fields, parse_mds_meta, read_mds


def test_mds_reader_parses_records_and_field_groups(tmp_path: Path) -> None:
    meta_path = tmp_path / "dynState.0000000010.meta"
    meta_path.write_text(
        """
 nDims = [ 2 ];
 dimList = [
  3, 1, 3,
  2, 1, 2
 ];
 dataprec = [ 'float32' ];
 nrecords = [ 4 ];
 timeStepNumber = [ 10 ];
 fldList = { 'THETA   ' 'UVEL    ' };
"""
    )
    expected = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    expected.astype(">f4").tofile(meta_path.with_suffix(".data"))

    parsed = parse_mds_meta(meta_path)
    assert parsed.dimensions == (3, 2)
    assert parsed.fields == ("THETA", "UVEL")
    assert parsed.timestep == 10
    meta, values = read_mds(meta_path)
    np.testing.assert_array_equal(values, expected)
    fields = mds_fields(meta, values)
    assert fields["THETA"].shape == (2, 2, 3)
    np.testing.assert_array_equal(fields["UVEL"], expected[2:])


def test_mds_fields_squeezes_record_not_vertical_dimension() -> None:
    meta = MDSMeta(
        dimensions=(3, 2, 15),
        nrecords=2,
        dtype=np.dtype(">f4"),
        fields=("THETA", "UVEL"),
        timestep=288,
    )
    values = np.zeros((2, 15, 2, 3))
    fields = mds_fields(meta, values)
    assert fields["THETA"].shape == (15, 2, 3)
    assert fields["UVEL"].shape == (15, 2, 3)
