from __future__ import annotations

import json
from pathlib import Path

import pytest

from bire_repro import af_model_c_bire_s0_figures as figures
from bire_repro.af_model_c_bire_aligned_full_state import (
    CHECKPOINT_STEPS,
    EXTERNAL_INPUT_CHANNELS,
    LIFTING_INPUT_CHANNELS,
    STAGE_NAMES,
)
from bire_repro.af_model_c_bire_aligned_s0_figures import (
    CONTRACT_STATUS,
    DEFAULT_STAGE,
    VERSION,
    BireAlignedS0FigureError,
    _FigureBinding,
    _readme,
    active_stage,
    load_contract,
    stage_view,
)

CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "model_c_bire_aligned_full_state_s0_figures_v1.json"
)
FROZEN = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "model_c_single_position_layernorm_s0_figures_v1.json"
)

pytestmark = pytest.mark.skipif(
    not CONTRACT.is_file(),
    reason="the figure contract is written from the completed training report",
)


def test_contract_declares_both_products_of_the_bire_protocol() -> None:
    contract, resolved, digest = load_contract(CONTRACT, verify_sources=False)
    assert resolved == CONTRACT.resolve()
    assert len(digest) == 64
    raw = json.loads(CONTRACT.read_text())
    assert raw["version"] == VERSION
    assert raw["contract_status"] == CONTRACT_STATUS
    assert set(raw["stages"]) == set(STAGE_NAMES)
    assert tuple(raw["stage_order"]) == STAGE_NAMES == ("pretrained", "finetuned")
    for stage, step in zip(STAGE_NAMES, CHECKPOINT_STEPS):
        model = raw["stages"][stage]["model"]
        assert model["optimizer_step"] == step
        assert model["architecture"]["in_channels"] == EXTERNAL_INPUT_CHANNELS
        assert (
            model["architecture"]["lifting_in_channels"] == LIFTING_INPUT_CHANNELS
        )
        assert model["architecture"]["n_layers"] == 3
        assert model["architecture"]["local_kernel_size"] is None
        assert model["architecture"]["positional_embedding"] is None
    # The default view is the endpoint of the protocol.
    assert contract["active_stage"] == DEFAULT_STAGE == "finetuned"
    assert contract["selected_model"]["optimizer_step"] == CHECKPOINT_STEPS[-1]


def test_protocol_and_filenames_match_the_incumbent_package() -> None:
    contract, _, _ = load_contract(CONTRACT, verify_sources=False)
    incumbent = json.loads(FROZEN.read_text())
    assert contract["protocol"] == incumbent["protocol"]
    assert contract["baselines"] == incumbent["baselines"]
    assert contract["truth"] == incumbent["truth"]
    assert contract["prior_model"] == incumbent["prior_model"]
    assert tuple(contract["protocol"]["figure_names"]) == figures.FIGURE_NAMES
    assert tuple(contract["protocol"]["rmse_fields"]) == figures.RMSE_FIELDS
    assert tuple(contract["protocol"]["acc_fields"]) == figures.ACC_FIELDS


def test_each_stage_publishes_into_its_own_new_folder() -> None:
    raw = json.loads(CONTRACT.read_text())
    incumbent = json.loads(FROZEN.read_text())
    projects = set()
    for stage in STAGE_NAMES:
        output = raw["stages"][stage]["output"]
        assert output["project"].endswith(stage)
        assert output["scratch"].endswith(stage)
        assert output["project"] != incumbent["output"]["project"]
        assert output["scratch"] != incumbent["output"]["scratch"]
        assert output["required"] == incumbent["output"]["required"]
        projects.add(output["project"])
    assert len(projects) == 2
    assert raw["output_policy"]["overwrite"] is False


def test_held_evaluation_promotes_nothing() -> None:
    raw = json.loads(CONTRACT.read_text())
    read = raw["read_contract"]
    assert read["held_s0_state"] is True
    assert read["promotes_no_checkpoint"] is True
    assert read["training_state"] is False
    for sealed in ("intermediate_wind_state", "response_state", "adjoint_state"):
        assert read[sealed] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("stages", "finetuned", "model", "optimizer_step"), 14880),
        (("stages", "pretrained", "model", "architecture", "n_layers"), 4),
        (
            ("stages", "pretrained", "model", "architecture", "local_kernel_size"),
            3,
        ),
        (("prior_model", "architecture", "positional_embedding"), None),
        (("figure6", "literal_pretrain_finetune_pair"), True),
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
    with pytest.raises(BireAlignedS0FigureError):
        load_contract(written, verify_sources=False)


def test_stage_view_swaps_only_the_model_artifacts_and_output() -> None:
    raw = json.loads(CONTRACT.read_text())
    pretrained = stage_view(raw, "pretrained")
    finetuned = stage_view(raw, "finetuned")
    assert pretrained["protocol"] == finetuned["protocol"] == raw["protocol"]
    assert pretrained["prior_model"] == finetuned["prior_model"]
    assert (
        pretrained["artifacts"]["selected_checkpoint"]
        != finetuned["artifacts"]["selected_checkpoint"]
    )
    assert pretrained["output"]["project"] != finetuned["output"]["project"]
    assert (
        pretrained["artifacts"]["prior_checkpoint"]
        == finetuned["artifacts"]["prior_checkpoint"]
    )
    with pytest.raises(BireAlignedS0FigureError):
        stage_view(raw, "selected")


def test_figure_binding_selects_a_stage_and_restores_the_frozen_runner() -> None:
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
    assert active_stage() == DEFAULT_STAGE
    with _FigureBinding("pretrained"):
        assert active_stage() == "pretrained"
        assert figures.load_contract is load_contract
        assert figures.VERSION == VERSION
        contract, _, _ = load_contract(CONTRACT, verify_sources=False)
        assert contract["selected_model"]["optimizer_step"] == CHECKPOINT_STEPS[0]
    assert active_stage() == DEFAULT_STAGE
    for name, value in before.items():
        assert getattr(figures, name) is value
    with pytest.raises(BireAlignedS0FigureError):
        _FigureBinding("selected")


def test_published_readme_names_the_stage_it_describes() -> None:
    with _FigureBinding("pretrained"):
        pretrained = _readme({"report_content_sha256": "0" * 64})
    with _FigureBinding("finetuned"):
        finetuned = _readme({"report_content_sha256": "0" * 64})
    assert "3,840" in pretrained and "pretrained" in pretrained
    assert "7,680" in finetuned and "finetuned" in finetuned
    assert "14880" not in pretrained and "14880" not in finetuned
    # The frozen runner's own README must remain untouched.
    assert "step-13440" in figures._readme({"report_content_sha256": "0" * 64})


def test_frozen_runner_entry_points_still_exist() -> None:
    for name in ("evaluate", "preflight", "load_contract", "_verify_file"):
        assert callable(getattr(figures, name))
    assert isinstance(figures.EXPECTED_STARTS, tuple)
