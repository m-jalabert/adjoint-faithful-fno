from __future__ import annotations

import json
from pathlib import Path

import pytest

from bire_repro import af_model_c_bire_s0_figures as figures
from bire_repro.af_model_c_single_position_s0_figures import (
    CONTRACT_STATUS,
    SELECTED_STEP,
    VERSION,
    SinglePositionS0FigureError,
    _FigureBinding,
    _readme,
    load_contract,
)

CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "model_c_single_position_layernorm_s0_figures_v1.json"
)
FROZEN = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "model_c_bire_s0_figures_v1.json"
)


def test_contract_targets_the_single_position_checkpoint() -> None:
    contract, resolved, digest = load_contract(CONTRACT, verify_sources=False)
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64
    assert contract["version"] == VERSION
    assert contract["contract_status"] == CONTRACT_STATUS
    selected = contract["selected_model"]
    assert selected["optimizer_step"] == SELECTED_STEP == 14880
    assert selected["architecture"]["positional_embedding"] is None
    assert selected["pointwise_layer_norm"] is True
    assert contract["prior_model"]["architecture"]["positional_embedding"] == "grid"


def test_protocol_and_filenames_match_the_frozen_package() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    frozen = json.loads(FROZEN.read_text())
    assert contract["protocol"] == frozen["protocol"]
    assert contract["baselines"] == frozen["baselines"]
    assert contract["truth"] == frozen["truth"]
    assert contract["output"]["required"] == frozen["output"]["required"]
    assert tuple(contract["protocol"]["figure_names"]) == figures.FIGURE_NAMES
    assert tuple(contract["protocol"]["rmse_fields"]) == figures.RMSE_FIELDS
    assert tuple(contract["protocol"]["acc_fields"]) == figures.ACC_FIELDS


def test_output_is_a_new_folder_and_does_not_overwrite_the_frozen_package() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    frozen = json.loads(FROZEN.read_text())
    assert contract["output"]["overwrite"] is False
    assert contract["output"]["project"] != frozen["output"]["project"]
    assert contract["output"]["scratch"] != frozen["output"]["scratch"]
    assert contract["output"]["project"].endswith(
        "single_position_layernorm_bire_s0_inference_v1"
    )


def test_held_evaluation_promotes_nothing() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    read = contract["read_contract"]
    assert read["held_s0_state"] is True
    assert read["promotes_no_checkpoint"] is True
    for sealed in ("intermediate_wind_state", "response_state", "adjoint_state"):
        assert read[sealed] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("selected_model", "optimizer_step"), 13440),
        (("selected_model", "architecture", "positional_embedding"), "grid"),
        (("prior_model", "architecture", "positional_embedding"), None),
    ],
)
def test_rejects_a_contract_that_drifts_from_the_arm(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    contract = json.loads(CONTRACT.read_text())
    target = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    written = tmp_path / "contract.json"
    written.write_text(json.dumps(contract))
    with pytest.raises(SinglePositionS0FigureError):
        load_contract(written, verify_sources=False)


def test_figure_binding_installs_and_restores_the_frozen_runner() -> None:
    before = {
        name: getattr(figures, name)
        for name in (
            "load_contract",
            "_selected_stepper",
            "_prior_stepper",
            "_readme",
            "VERSION",
        )
    }
    with _FigureBinding():
        assert figures.load_contract is load_contract
        assert figures.VERSION == VERSION
        assert figures._selected_stepper is not before["_selected_stepper"]
        assert figures._prior_stepper is not before["_prior_stepper"]
        assert figures._readme is not before["_readme"]
    for name, value in before.items():
        assert getattr(figures, name) is value


def test_published_readme_describes_this_arm_not_the_frozen_one() -> None:
    """The frozen runner's README hard-codes step-13440; ours must not."""

    text = _readme({"report_content_sha256": "0" * 64})
    assert "step-14880" in text
    assert "13440" not in text
    assert "single-position" in text.lower()
    frozen = figures._readme({"report_content_sha256": "0" * 64})
    assert "step-13440" in frozen


def test_frozen_runner_entry_points_still_exist() -> None:
    """The binding replaces names in the frozen runner; catch renames early."""

    for name in ("evaluate", "preflight", "load_contract", "_verify_file"):
        assert callable(getattr(figures, name))
    assert isinstance(figures.EXPECTED_STARTS, tuple)
