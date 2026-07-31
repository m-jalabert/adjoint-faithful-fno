from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bire_repro.af_model_c_bire_s0_long_truth import (
    EXPECTED_STARTS,
    LongTruthContractError,
    load_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = PROJECT_ROOT / "config/model_c_bire_s0_long_truth_v1.json"


def test_frozen_s0_long_truth_contract() -> None:
    contract, path, digest = load_contract(CONTRACT)
    assert path.is_absolute()
    assert len(digest) == 64
    assert contract["simulation"]["tau0_n_m2"] == 0.1
    assert contract["simulation"]["expected_daily_records"] == 2160
    assert tuple(contract["ensemble"]["start_draw_order"]) == EXPECTED_STARTS
    assert max(EXPECTED_STARTS) + 2000 <= 7199 + 2160


def test_ensemble_reproduces_from_rng_without_replacement() -> None:
    expected = np.random.default_rng(20260729).choice(
        np.arange(6660, 7200, dtype=np.int64),
        size=15,
        replace=False,
    )
    assert tuple(expected.tolist()) == EXPECTED_STARTS
    encoded = json.dumps(
        expected.tolist(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == (
        "c8756f5d3f4b0ac381b52a597b181049dc170b59a0a2b3d7f7dfe85750cb1241"
    )


def test_contract_rejects_changed_ensemble(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text())
    value["ensemble"]["start_draw_order"][0] += 1
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(value))
    with pytest.raises(LongTruthContractError, match="ensemble changed"):
        load_contract(changed, verify_sources=False)
