"""Split and source-inventory contracts for trajectory dataset version 2."""

import json
from pathlib import Path

import numpy as np

from bire_repro.af_data_v2 import (
    TOTAL_RECORDS,
    dataset_v2_split,
    inventory_extension,
)
from bire_repro.af_trajectory_expansion import load_expansion_contract


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_ROOT = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno")


def test_dataset_v2_split_is_complete_buffered_and_reproducible() -> None:
    contract, _, _ = load_expansion_contract(
        PROJECT_ROOT / "config/trajectories_v2_expansion.json"
    )
    snapshot, pair, counts = dataset_v2_split(contract)
    assert snapshot.shape == pair.shape == (TOTAL_RECORDS,)
    assert counts == {"training": 5020, "validation": 260, "inference": 1150}
    assert np.count_nonzero(pair == 1) == 5020
    assert np.count_nonzero(pair == 2) == 260
    assert np.count_nonzero(pair == 3) == 1150
    assert np.all(snapshot[2610:2880] == 0)
    assert np.all(snapshot[2970:3600] == 3)
    assert np.all(snapshot[6300:6570] == 2)


def test_completed_extensions_match_the_frozen_contract() -> None:
    contract, _, digest = load_expansion_contract(
        PROJECT_ROOT / "config/trajectories_v2_expansion.json"
    )
    for experiment in ("S0", "S1", "S2"):
        inventory = inventory_extension(
            SCRATCH_ROOT,
            contract,
            digest,
            experiment,
        )
        result = json.loads(inventory.result_path.read_text())
        assert len(inventory.iterations) == 3600
        assert result["returncode"] == 0
        assert result["diagnostics"] == {"dynState": 3600, "surfState": 3600}
