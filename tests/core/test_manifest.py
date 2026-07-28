from bire_repro.core.manifest import file_record, verify_records


def test_checksum_verification(tmp_path):
    product = tmp_path / "product.bin"
    product.write_bytes(b"paper reproduction")
    record = file_record(product)
    assert verify_records([record]) == []
    product.write_bytes(b"changed")
    assert "mismatch" in verify_records([record])[0]
