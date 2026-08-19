"""Tests for the fine-tune's S0 figure and anomaly packages.

Both are *lineage adapters*: they reuse every numerical routine from
:mod:`oceanfno.figures` and :mod:`oceanfno.anomaly` and reimplement only the
part that says which training arm produced the checkpoint. So these tests ask
three things:

1. that the evaluation protocol is byte-for-byte the parent's, because a
   fine-tune scored on different members would not be comparable to the model
   it fine-tuned;
2. that the lineage checks are real -- a package cannot claim the fine-tune's
   provenance without the whole chain back to the published parent;
3. that ``figures.py``, ``anomaly.py`` and ``plots.py`` are untouched, so the
   parent's own packages stay reproducible and the comparison has a fixed end.

Runnable without a GPU: contracts are read with ``verify_sources=False`` and
nothing here rolls anything out.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from oceanfno import anomaly, anomaly_ft90, figures, figures_ft90, plots
from oceanfno.model import ProductionArchitecture
from oceanfno.runtime import _file_sha256
import oceanfno.finetune as finetune

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "config/model_c_production_1in_1out_spectralnorm_ft90_v1_s0_figures_v1.json"
ANOMALY = ROOT / "config/model_c_production_1in_1out_spectralnorm_ft90_v1_s0_anomaly_v1.json"
PARENT_FIGURES = ROOT / "config/model_c_production_1in_1out_spectralnorm_v1_s0_figures_v1.json"
PARENT_ANOMALY = ROOT / "config/model_c_production_1in_1out_spectralnorm_v1_s0_anomaly_v1.json"
FIG_SBATCH = ROOT / "slurm/models/c/figures_production_1in_1out_spectralnorm_ft90_v1.sbatch"
ANOM_SBATCH = ROOT / "slurm/models/c/anomaly_production_1in_1out_spectralnorm_ft90_v1.sbatch"


# ---------------------------------------------------------------------------
# the parent's evaluation code must stay untouched
# ---------------------------------------------------------------------------


def test_the_parent_evaluation_contracts_still_pin_their_own_sources() -> None:
    """figures.py, anomaly.py and plots.py must be byte-identical.

    The fine-tune's packages were built by adding adapters, not by generalizing
    these modules, precisely so the parent's figure and anomaly packages remain
    re-runnable -- which is what makes them a fixed comparison rather than a
    moving one.
    """

    changed: dict[str, Any] = {}
    for contract_path in (PARENT_FIGURES, PARENT_ANOMALY):
        pinned = json.loads(contract_path.read_text())["source_hashes"]
        assert pinned, f"{contract_path.name} pins no sources"
        for relative, expected in pinned.items():
            actual = _file_sha256(ROOT / relative)
            if actual != expected:
                changed[f"{contract_path.name}:{relative}"] = (expected, actual)
    assert not changed, f"the fine-tune modified parent-pinned sources: {changed}"


def test_the_parent_figure_module_still_declares_the_parent_lineage() -> None:
    assert figures.VERSION == "model_c_production_1in_1out_spectralnorm_v1_s0_figures_v1"
    assert figures.TRAINING_VERSION == finetune.PARENT_VERSION
    assert anomaly.VERSION == "model_c_production_1in_1out_spectralnorm_v1_s0_anomaly_v1"
    assert anomaly.FIGURE_PACKAGE_VERSION == figures.VERSION


# ---------------------------------------------------------------------------
# the protocol is the parent's, exactly
# ---------------------------------------------------------------------------


def test_the_figure_protocol_is_identical_to_the_parents() -> None:
    """A different member draw would make the two packages incomparable."""

    child = json.loads(FIGURES.read_text())["protocol"]
    parent = json.loads(PARENT_FIGURES.read_text())["protocol"]
    for key in (
        "member_count",
        "start_seed",
        "start_draw_order",
        "start_window",
        "inference_set",
        "regimes",
        "primary_regime",
        "figure_names",
        "figure3_lead_days",
        "figure7_lead_days",
        "rmse_fields",
        "acc_fields",
        "maximum_lead_days",
        "prediction_interval_days",
        "short_lead_days",
        "long_lead_days",
        "comparator_model",
        "nesting",
        "static_channels",
    ):
        assert child[key] == parent[key], key
    assert child["identical_to_the_parent_figure_protocol"] is True


def test_the_figure_baselines_and_truth_are_the_parents() -> None:
    child = json.loads(FIGURES.read_text())
    parent = json.loads(PARENT_FIGURES.read_text())
    assert child["baselines"] == parent["baselines"]
    assert child["truth"] == parent["truth"]
    assert child["dataset"]["tau0_n_m2"] == parent["dataset"]["tau0_n_m2"]
    for key in ("train", "validation", "inference", "version"):
        assert child["dataset"][key] == parent["dataset"][key], key


def test_the_figure_package_reuses_the_parents_numerics() -> None:
    """The adapter must not have reimplemented the evaluation itself."""

    assert figures_ft90.evaluate_regime is figures.evaluate_regime
    assert figures_ft90.long_rollout_gate is figures.long_rollout_gate
    assert figures_ft90.declared_inference_starts is figures.declared_inference_starts
    assert figures_ft90.MEMBER_COUNT == figures.MEMBER_COUNT == 15
    assert figures_ft90.START_SEED == figures.START_SEED
    assert figures_ft90.REGIMES == figures.REGIMES


def test_the_anomaly_package_reuses_the_parents_numerics() -> None:
    assert anomaly_ft90.training_mean_streamfunction is anomaly.training_mean_streamfunction
    assert anomaly_ft90.day2000_structure_summary is anomaly.day2000_structure_summary
    assert anomaly_ft90.variability_summary is anomaly.variability_summary
    assert anomaly_ft90.FIGURE_NAMES == anomaly.FIGURE_NAMES
    assert anomaly_ft90.REGIME == anomaly.REGIME == "S0"


# ---------------------------------------------------------------------------
# the lineage
# ---------------------------------------------------------------------------


def test_the_figure_contract_declares_the_fine_tuned_lineage() -> None:
    selected = json.loads(FIGURES.read_text())["selected_model"]
    assert selected["version"] == finetune.VERSION
    assert selected["from_scratch"] is False
    assert selected["parent_version"] == finetune.PARENT_VERSION
    assert selected["parent_optimizer_step"] == finetune.PARENT_OPTIMIZER_STEP
    assert selected["rollout_steps"] == 9
    assert selected["loss_contract_sha256"] == finetune.FINETUNE_LOSS_CONTRACT_SHA256
    assert selected["architecture"] == ProductionArchitecture().to_dict()
    # The parent's figure contract says the opposite, and must keep saying it.
    assert json.loads(PARENT_FIGURES.read_text())["selected_model"]["from_scratch"] is True


def test_a_figure_contract_claiming_from_scratch_is_rejected(tmp_path: Path) -> None:
    contract = json.loads(FIGURES.read_text())
    contract["selected_model"]["from_scratch"] = True
    target = tmp_path / "scratch.json"
    target.write_text(json.dumps(contract))
    with pytest.raises(figures.FigureContractError):
        figures_ft90.load_contract(target, verify_sources=False)


def test_a_figure_contract_without_the_parent_is_rejected(tmp_path: Path) -> None:
    contract = json.loads(FIGURES.read_text())
    del contract["selected_model"]["parent_version"]
    target = tmp_path / "orphan.json"
    target.write_text(json.dumps(contract))
    with pytest.raises(figures.FigureContractError):
        figures_ft90.load_contract(target, verify_sources=False)


def test_a_figure_contract_missing_the_parent_summary_is_rejected(tmp_path: Path) -> None:
    """The comparison baseline is a pinned artifact, not an optional extra."""

    contract = json.loads(FIGURES.read_text())
    del contract["artifacts"][figures_ft90.PARENT_SUMMARY_ARTIFACT]
    target = tmp_path / "nobaseline.json"
    target.write_text(json.dumps(contract))
    with pytest.raises(figures.FigureContractError):
        figures_ft90.load_contract(target, verify_sources=False)


def test_the_two_packages_write_to_different_roots_than_the_parent() -> None:
    for child_path, parent_path in ((FIGURES, PARENT_FIGURES), (ANOMALY, PARENT_ANOMALY)):
        child = json.loads(child_path.read_text())["output"]
        parent = json.loads(parent_path.read_text())["output"]
        assert child["project_root"] != parent["project_root"]
        assert child["scratch_root"] != parent["scratch_root"]
        assert "outputs/af_fno/C/" in child["project_root"]
        assert child["overwrite"] is False


def test_the_figure_gate_artifact_is_named_for_the_fine_tune() -> None:
    assert figures_ft90.GATE_NAME == f"{finetune.VERSION}_acceptance_gate.json"
    assert figures_ft90.GATE_NAME != figures.GATE_NAME


# ---------------------------------------------------------------------------
# the anomaly package's reference field
# ---------------------------------------------------------------------------


def test_the_anomaly_reference_is_declared_identical_to_the_parents() -> None:
    """Two models' anomalies are comparable only if the same mean is removed."""

    child = json.loads(ANOMALY.read_text())["reference"]
    parent = json.loads(PARENT_ANOMALY.read_text())["reference"]
    for key in (
        "source",
        "days",
        "regime",
        "subtracted_from",
        "model_own_mean_used",
        "is_two_dimensional_field",
        "not_a_scalar_spatial_mean",
    ):
        assert child[key] == parent[key], key
    assert child["identical_to_the_parent_packages_reference"] is True
    assert child["model_own_mean_used"] is False


def test_an_anomaly_contract_using_the_models_own_mean_is_rejected(tmp_path: Path) -> None:
    contract = json.loads(ANOMALY.read_text())
    contract["reference"]["model_own_mean_used"] = True
    target = tmp_path / "ownmean.json"
    target.write_text(json.dumps(contract))
    with pytest.raises(anomaly.AnomalyContractError):
        anomaly_ft90.load_contract(target, verify_sources=False)


def test_an_anomaly_contract_that_drops_the_shared_reference_claim_is_rejected(
    tmp_path: Path,
) -> None:
    contract = json.loads(ANOMALY.read_text())
    contract["reference"]["identical_to_the_parent_packages_reference"] = False
    target = tmp_path / "different_reference.json"
    target.write_text(json.dumps(contract))
    with pytest.raises(anomaly.AnomalyContractError):
        anomaly_ft90.load_contract(target, verify_sources=False)


def test_the_anomaly_contract_reads_the_fine_tuned_figure_package() -> None:
    contract = json.loads(ANOMALY.read_text())
    artifacts = contract["artifacts"]
    assert artifacts["figure_package_contract"]["path"].endswith(
        f"{figures_ft90.VERSION}.json"
    )
    assert figures_ft90.VERSION in artifacts["figure_package_arrays"]["path"]
    assert artifacts["figure_package_arrays"]["path"].endswith(plots.ARRAYS_NAME)
    assert contract["protocol"]["training_version"] == finetune.VERSION
    assert contract["protocol"]["parent_version"] == finetune.PARENT_VERSION
    assert anomaly_ft90.PARENT_ANOMALY_ARTIFACT in artifacts
    assert contract["modifies_published_figures"] is False
    assert contract["adds_only"] is True
    assert contract["protocol"]["reads_model_weights"] is False
    assert contract["protocol"]["rolls_nothing_out"] is True


def test_the_anomaly_package_points_at_the_child_not_the_parent_figures() -> None:
    assert anomaly_ft90.FIGURE_PACKAGE_VERSION == figures_ft90.VERSION
    assert anomaly_ft90.FIGURE_PACKAGE_VERSION != figures.VERSION
    assert anomaly_ft90.TRAINING_VERSION == finetune.VERSION


# ---------------------------------------------------------------------------
# the declared-pending pipeline
# ---------------------------------------------------------------------------


def test_the_anomaly_contract_defers_the_sealed_digests_until_the_figures_exist() -> None:
    contract = json.loads(ANOMALY.read_text())
    deferred = anomaly_ft90.unfilled_fields(contract)
    package = json.loads(FIGURES.read_text())
    figures_are_done = (
        Path(package["output"]["project_root"]).resolve() / "S0" / plots.ARRAYS_NAME
    ).is_file()
    if figures_are_done:
        # Once `anomaly_ft90 finalize` has run there is nothing left deferred.
        assert deferred == [] or all("figure_package" in name for name in deferred)
    else:
        assert deferred, "the sealed digests cannot be known before the figures exist"
        assert all("figure_package" in name for name in deferred)


def test_a_partially_filled_figure_contract_refuses_to_load(tmp_path: Path) -> None:
    contract = json.loads(FIGURES.read_text())
    contract["selected_model"]["optimizer_step"] = figures_ft90.PENDING
    target = tmp_path / "pending.json"
    target.write_text(json.dumps(contract))
    with pytest.raises(figures.FigureContractError):
        figures_ft90.load_contract(target, verify_sources=False)


# ---------------------------------------------------------------------------
# the jobs
# ---------------------------------------------------------------------------


def test_the_slurm_jobs_use_the_fine_tuned_entrypoints() -> None:
    figure_text = FIG_SBATCH.read_text()
    for stage in ("finalize", "preflight", "run"):
        assert f"-m oceanfno.figures_ft90 {stage}" in figure_text
    assert "ft90_v1_s0_figures_v1.json" in figure_text
    # It must not drive the parent's package.
    assert "-m oceanfno.figures " not in figure_text

    anomaly_text = ANOM_SBATCH.read_text()
    for stage in ("finalize", "preflight", "run"):
        assert f"-m oceanfno.anomaly_ft90 {stage}" in anomaly_text
    assert "ft90_v1_s0_anomaly_v1.json" in anomaly_text
    assert "-m oceanfno.anomaly " not in anomaly_text
    # The anomaly stage loads no weights, so it must not request a GPU.
    assert "--gres=gpu" not in anomaly_text
