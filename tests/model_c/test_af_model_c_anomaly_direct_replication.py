from __future__ import annotations

import json
from pathlib import Path

import pytest

from bire_repro import af_model_c_anomaly_direct_replication as replication


def _minimal_replication_contract(project_root: Path) -> dict[str, object]:
    return {
        "parent_contract": {
            "path": "config/model_c_anomaly_direct_v1.json",
            "sha256": "parent-sha",
        },
        "output_contract": {
            "scratch_root": "/tmp/anomaly-direct-replication",
            "project_root": str(
                project_root
                / "outputs"
                / "af_fno"
                / "C"
                / "anomaly_direct_replication_v1"
            ),
        },
    }


def test_resolve_replication_seed_uses_frozen_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = {
        "seed_replication": {
            "new_seeds": [20260724, 20260725],
        }
    }
    monkeypatch.setattr(
        replication,
        "load_replication_contract",
        lambda path: (contract, Path(path), "contract-sha"),
    )
    assert replication.resolve_replication_seed("contract.json", 0) == 20260724
    assert replication.resolve_replication_seed("contract.json", 1) == 20260725
    with pytest.raises(IndexError, match="outside 2 replication seeds"):
        replication.resolve_replication_seed("contract.json", 2)


def test_derived_contract_changes_only_replication_provenance_seed_and_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    replication_path = root / "config" / "replication-placeholder.json"
    contract = _minimal_replication_contract(root)
    derived = replication._derived_parent_contract(
        contract,
        replication_path,
        "replication-sha",
        20260724,
    )
    parent = json.loads(
        (root / "config" / "model_c_anomaly_direct_v1.json").read_text()
    )

    assert derived["architecture"] == parent["architecture"]
    assert derived["normalization"] == parent["normalization"]
    assert derived["prediction"] == parent["prediction"]
    assert derived["training"]["optimizer"] == parent["training"]["optimizer"]
    assert (
        derived["training"]["checkpoint_steps"]
        == parent["training"]["checkpoint_steps"]
    )
    assert derived["training"]["seed"] == 20260724
    assert parent["training"]["seed"] == 20260723
    assert derived["replication_provenance"]["change_from_parent"] == "seed_only"
    assert derived["replication_provenance"]["fixed_validation_members_changed"] is False
    assert derived["source_hashes"] == {}
    assert derived["output_contract"]["scratch_output"].endswith(
        "/seeds/seed_20260724"
    )
    assert derived["output_contract"]["project_output"].endswith(
        "/seed_20260724"
    )


def test_seed_paths_are_disjoint() -> None:
    root = Path(__file__).resolve().parents[2]
    contract = _minimal_replication_contract(root)
    first = replication._paths_for_seed(contract, 20260724)
    second = replication._paths_for_seed(contract, 20260725)
    assert len(set(first + second)) == 6
