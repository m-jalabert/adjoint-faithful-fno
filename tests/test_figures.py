"""Tests for the physical-static-channel model's S0 figure package.

These figures are a static-channel ablation, not a new evaluation protocol. They
must therefore reuse the preceding arm's starts, truth, baselines, normalizers,
lead grid, and six plot definitions exactly.  Only the selected checkpoint
changes: the new-statics fine-tune is red and the old-statics checkpoint it
started from is black.  Both sides read the same two time levels, so the
temporal context is asserted equal rather than compared.

Each member still reads a second initial condition, the truth state ten days
before its start.  It is an initial condition and never a scored target, so the
declared starts, leads and baselines are asserted identical to the compared
package.

The tests that bind to the training report are skipped until the arm has
actually been trained; everything that is a property of the declaration alone
runs immediately.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/model_c_2in_1out_new_channels_s0_figures_v1.json"
COMPARED = ROOT / "config/model_c_2in_1out_s0_figures_v1.json"
TRAINING = ROOT / "config/model_c_2in_1out_new_channels_v1.json"
COMPARATOR_TRAINING = ROOT / "config/model_c_2in_1out_v1.json"
TRAINING_REPORT = (
    ROOT
    / "outputs/af_fno/C/model_c_2in_1out_new_channels_v1/"
    "model_c_2in_1out_new_channels_report.json"
)
MODULE = ROOT / "src/oceanfno/figures.py"
SBATCH = ROOT / "slurm/models/c/figures_2in_1out_new_channels.sbatch"

VERSION = "model_c_2in_1out_new_channels_s0_figures_v1"
TRAINING_VERSION = "model_c_2in_1out_new_channels_v1"
COMPARATOR_VERSION = "model_c_2in_1out_v1"
PENDING = "PENDING_AFTER_TRAINING"
#: Which checkpoint selection lands on is not known before the run; only that
#: it is one of the four declared ones. The comparator step is fixed.
CHECKPOINT_STEPS = (960, 1920, 2880, 3840)
COMPARATOR_STEP = 3840
#: An arbitrary declared step, used only where a caption's formatting is under
#: test rather than the arm's actual selection.
EXAMPLE_STEP = 2880

#: The three architecture fields this arm is allowed to move.
STATIC_CONTRACT_FIELDS = {"in_channels", "lifting_in_channels", "static_channels"}
NEW_STATIC_CHANNELS = [
    "wind_stress_x",
    "wet_mask",
    "coriolis_parameter",
    "zonal_grid_spacing",
    "sst_relaxation_target",
]

STARTS = (
    6263,
    6293,
    6331,
    6389,
    6579,
    6593,
    6598,
    6601,
    6651,
    6661,
    6694,
    6707,
    6711,
    6968,
    6979,
)
FIGURE3_LEADS = (0, 10, 20, 30, 40)
FIGURE7_LEADS = (60, 2000)
FIGURE_NAMES = (
    "model_c_bire_figure3_streamfunction_1deg_s0_dt10.png",
    "model_c_bire_figure4_dt10_rmse_0_200_days_s0.png",
    "model_c_bire_figure5_dt10_single_member_rmse_s0.png",
    "model_c_bire_figure6_dt10_acc_0_200_days_s0.png",
    "model_c_bire_figure7_dt10_streamfunction_day060_day2000_s0.png",
    "model_c_bire_figure8_dt10_rmse_0_2000_days_s0.png",
)

PROJECT_ROOT = (
    "/home/mjalabert314/bire_james25_repro/outputs/af_fno/C/"
    "model_c_2in_1out_new_channels_s0_figures_v1"
)
SCRATCH_ROOT = (
    "/bigscratch/mjalabert314/bire_james25_repro/af_fno/models/C/"
    "model_c_2in_1out_new_channels_s0_figures_v1"
)

requires_contract = pytest.mark.skipif(
    not CONTRACT.is_file(), reason="the new-channel figure contract is absent"
)
requires_module = pytest.mark.skipif(
    not MODULE.is_file(), reason="the canonical figure module is absent"
)
requires_report = pytest.mark.skipif(
    not TRAINING_REPORT.is_file(),
    reason="the new-channel arm has not been trained yet",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw() -> dict:
    return json.loads(CONTRACT.read_text())


def _pending() -> dict:
    contract = _raw()
    contract["selected_model"]["optimizer_step"] = PENDING
    for key in ("selected_checkpoint", "selected_normalization", "selected_report"):
        contract["artifacts"][key]["sha256"] = PENDING
    return contract


def _filled() -> dict:
    """Materialize the four post-training fields from the run's own report."""

    contract = _pending()
    report = json.loads(TRAINING_REPORT.read_text())
    published = report["published_checkpoint"]
    contract["selected_model"]["optimizer_step"] = int(
        published["optimizer_step"]
    )
    contract["artifacts"]["selected_checkpoint"]["sha256"] = published[
        "checkpoint_sha256"
    ]
    contract["artifacts"]["selected_normalization"]["sha256"] = published[
        "normalization_sha256"
    ]
    contract["artifacts"]["selected_report"]["sha256"] = _sha256(TRAINING_REPORT)
    return contract


def _written(contract: dict, directory: Path) -> Path:
    path = directory / "new_channel_figures.json"
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return path


def _suite():
    return importlib.import_module("oceanfno.figures")


@requires_contract
def test_new_channels_reuses_the_exact_s0_protocol_and_six_plots() -> None:
    mine = _raw()
    compared = json.loads(COMPARED.read_text())
    compared_protocol = compared["protocol"]
    protocol = mine["protocol"]

    shared = (
        "member_count",
        "maximum_lead_days",
        "prediction_interval_days",
        "start_draw_order",
        "figure_names",
        "rmse_fields",
        "acc_fields",
        "regimes",
        "primary_regime",
        "inference_set",
        "start_seed",
        "start_window",
        "figure3_lead_days",
        "figure7_lead_days",
        "short_lead_days",
        "long_lead_days",
    )
    for key in shared:
        assert protocol[key] == compared_protocol[key], key

    assert protocol["static_channels"] == NEW_STATIC_CHANNELS
    assert protocol["member_count"] == len(STARTS) == 15
    assert tuple(protocol["start_draw_order"]) == STARTS
    assert tuple(protocol["figure3_lead_days"]) == FIGURE3_LEADS
    assert tuple(protocol["figure7_lead_days"]) == FIGURE7_LEADS
    assert protocol["maximum_lead_days"] == 2000
    assert protocol["prediction_interval_days"] == 10
    assert protocol["inference_set"] == [6200, 7200]
    assert protocol["start_window"] == [6200, 7000]
    assert tuple(protocol["figure_names"]) == FIGURE_NAMES
    assert tuple(
        name for name in mine["output"]["required"] if name.endswith(".png")
    ) == FIGURE_NAMES


@requires_contract
def test_the_extra_initial_condition_is_declared_and_is_never_a_target() -> None:
    """The one protocol addition must be stated, and stated as an input only."""

    history = _raw()["protocol"]["history_initial_condition"]
    assert history["lag_days"] == 10
    assert history["states_per_call"] == 2
    assert history["is_scored_target"] is False
    assert history["earliest_history_day"] == min(STARTS) - 10
    # Model-visible means < 7200; the leads themselves are unchanged.
    assert history["earliest_history_day"] < 7200
    assert history["leads_unchanged"] is True


@requires_contract
@requires_report
def test_new_channels_reuses_truth_baselines_dataset_and_normalization() -> None:
    mine = _raw()
    compared = json.loads(COMPARED.read_text())
    report = json.loads(TRAINING_REPORT.read_text())

    assert mine["truth"] == compared["truth"]
    assert mine["baselines"] == compared["baselines"]
    assert mine["dataset"] == compared["dataset"]
    assert mine["artifacts"]["dataset_metadata"] == compared["artifacts"][
        "dataset_metadata"
    ]

    selected = mine["artifacts"]["selected_normalization"]
    published = report["published_checkpoint"]
    assert selected["path"] == published["normalization"]
    # The normalizers are republished bytes of the same train-only coordinates
    # the parent weights were trained in, so the digest must not have moved.
    assert published["normalization_sha256"] == compared["artifacts"][
        "selected_normalization"
    ]["sha256"]
    assert selected["sha256"] in (PENDING, published["normalization_sha256"])


@requires_contract
@requires_report
def test_figure6_is_the_new_channel_arm_against_its_literal_old_static_parent() -> None:
    contract = _filled()
    training = json.loads(TRAINING.read_text())
    comparator_training = json.loads(COMPARATOR_TRAINING.read_text())
    report = json.loads(TRAINING_REPORT.read_text())
    selected = contract["selected_model"]
    comparator = contract["comparator_model"]

    assert contract["version"] == VERSION
    assert selected["version"] == TRAINING_VERSION
    assert comparator["version"] == COMPARATOR_VERSION
    assert selected["architecture"] == training["architecture"]
    assert comparator["architecture"] == comparator_training["architecture"]
    # Spatial capacity is identical; only the input contract differs.
    assert selected["architecture"]["n_modes"] == [32, 32]
    assert comparator["architecture"]["n_modes"] == [32, 32]
    # Temporal context is identical on both sides; only the statics differ.
    assert selected["architecture"]["input_states"] == 2
    assert comparator["architecture"]["input_states"] == 2
    assert selected["architecture"]["input_lag_days"] == 10
    assert comparator["architecture"]["input_lag_days"] == 10
    assert selected["architecture"]["static_channels"] == NEW_STATIC_CHANNELS
    assert selected["architecture"]["in_channels"] == 97
    assert selected["architecture"]["lifting_in_channels"] == 99
    assert comparator["architecture"]["in_channels"] == 95
    assert comparator["architecture"]["lifting_in_channels"] == 97
    assert "static_channels" not in comparator["architecture"]
    assert selected["architecture"]["local_kernel_size"] == 3
    assert comparator["architecture"]["local_kernel_size"] == 3
    # The reverted smooth-position field must not have come back either.
    assert "position_encoding" not in selected["architecture"]
    assert "position_encoding" not in comparator["architecture"]
    assert {
        key: value
        for key, value in selected["architecture"].items()
        if key not in STATIC_CONTRACT_FIELDS
    } == {
        key: value
        for key, value in comparator["architecture"].items()
        if key not in STATIC_CONTRACT_FIELDS
    }
    assert selected["optimizer_step"] in CHECKPOINT_STEPS
    assert comparator["optimizer_step"] == COMPARATOR_STEP
    assert selected["rollout_steps"] == comparator["rollout_steps"] == 6
    assert selected["base_loss_contract_sha256"] == comparator[
        "base_loss_contract_sha256"
    ]

    published = report["published_checkpoint"]
    assert contract["artifacts"]["selected_checkpoint"]["path"] == published[
        "checkpoint"
    ]
    assert contract["artifacts"]["selected_checkpoint"]["sha256"] == published[
        "checkpoint_sha256"
    ]
    for key in ("path", "sha256"):
        assert contract["artifacts"]["comparator_checkpoint"][key] == training[
            "sources"
        ]["initialization_checkpoint"][key]
    assert "model_c_2in_1out_new_channels_v1/selected.pt" in published["checkpoint"]
    assert "model_c_2in_1out_v1/selected.pt" in contract[
        "artifacts"
    ]["comparator_checkpoint"]["path"]
    assert contract["figure6"]["literal_pretrain_finetune_pair"] is True
    assert contract["figure6"]["comparator_optimizer_step"] == COMPARATOR_STEP


@requires_contract
def test_new_channel_figure_outputs_have_exact_noncolliding_roots() -> None:
    mine = _raw()
    compared = json.loads(COMPARED.read_text())
    training = json.loads(TRAINING.read_text())
    output = mine["output"]

    assert output["project_root"] == PROJECT_ROOT
    assert output["scratch_root"] == SCRATCH_ROOT
    assert output["overwrite"] is False
    assert output["one_folder_per_regime"] is True
    assert output["required"] == compared["output"]["required"]
    for key in ("project_root", "scratch_root"):
        assert output[key] != compared["output"][key]
        assert output[key] != training["output"][key]
    assert mine["comparability"]["directly_comparable_with_one_input_package"] is True


@requires_contract
@requires_module
@requires_report
def test_a_filled_new_channel_figure_contract_loads_strictly(tmp_path) -> None:
    suite = _suite()
    contract, resolved, digest = suite.load_contract(
        _written(_filled(), tmp_path), verify_sources=False
    )
    assert contract["version"] == suite.VERSION == VERSION
    assert contract["selected_model"]["version"] == TRAINING_VERSION
    assert contract["comparator_model"]["version"] == COMPARATOR_VERSION
    assert resolved.is_file()
    assert len(digest) == 64


@requires_contract
@requires_module
@requires_report
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda c: c["protocol"].update(start_draw_order=list(reversed(STARTS))),
            id="starts_reordered",
        ),
        pytest.param(
            lambda c: c["protocol"].update(figure7_lead_days=[60, 1000]),
            id="day2000_removed",
        ),
        pytest.param(
            lambda c: c["protocol"].update(figure_names=list(FIGURE_NAMES[:-1])),
            id="sixth_plot_removed",
        ),
        pytest.param(
            lambda c: c["selected_model"].update(version=COMPARATOR_VERSION),
            id="selected_version_reverted",
        ),
        pytest.param(
            lambda c: c["comparator_model"].update(version=TRAINING_VERSION),
            id="comparator_version_moved",
        ),
        pytest.param(
            lambda c: c["selected_model"]["architecture"].update(
                static_channels=["wind_stress_x", "wet_mask",
                                 "distance_to_wall_normalized"]
            ),
            id="selected_statics_reverted",
        ),
        pytest.param(
            lambda c: c["selected_model"]["architecture"].update(n_modes=[32, 24]),
            id="selected_modes_moved",
        ),
        pytest.param(
            lambda c: c["selected_model"]["architecture"].update(
                position_encoding="smooth_xy"
            ),
            id="reverted_position_encoding_reintroduced",
        ),
        pytest.param(
            lambda c: c["comparator_model"]["architecture"].update(
                static_channels=NEW_STATIC_CHANNELS
            ),
            id="wrong_comparator_architecture",
        ),
        pytest.param(
            lambda c: c["comparator_model"]["architecture"].update(
                local_kernel_size=None
            ),
            id="comparator_local_branch_removed",
        ),
        pytest.param(
            lambda c: c["artifacts"]["selected_checkpoint"].update(
                path=c["artifacts"]["comparator_checkpoint"]["path"]
            ),
            id="selected_checkpoint_replaced",
        ),
        pytest.param(
            lambda c: c["artifacts"]["comparator_checkpoint"].update(
                path="/bigscratch/wrong-parent/selected.pt"
            ),
            id="comparator_checkpoint_replaced",
        ),
        pytest.param(
            lambda c: c["artifacts"]["selected_normalization"].update(
                sha256="0" * 64
            ),
            id="normalizer_changed",
        ),
        pytest.param(
            lambda c: c.update(truth={"source": "another_trajectory"}),
            id="truth_changed",
        ),
        pytest.param(
            lambda c: c["baselines"].update(climatology="full_record_mean"),
            id="climatology_changed",
        ),
        pytest.param(
            lambda c: c["output"].update(
                project_root=PROJECT_ROOT.replace("new_channels_", "")
            ),
            id="project_root_changed",
        ),
        pytest.param(
            lambda c: c["output"].update(scratch_root=SCRATCH_ROOT + "_other"),
            id="scratch_root_changed",
        ),
        pytest.param(
            lambda c: c["output"].update(overwrite=True),
            id="overwrite_enabled",
        ),
        pytest.param(
            lambda c: c["figure6"].update(literal_pretrain_finetune_pair=False),
            id="literal_pair_denied",
        ),
        pytest.param(
            lambda c: c["selected_model"].update(
                base_loss_contract_sha256="0" * 64
            ),
            id="objective_changed",
        ),
    ],
)
def test_new_channel_figure_contract_rejects_protocol_model_and_output_tampering(
    mutate, tmp_path
) -> None:
    suite = _suite()
    contract = _filled()
    mutate(contract)
    with pytest.raises(suite.ModelCNewChannelsFigureError):
        suite.load_contract(_written(contract, tmp_path), verify_sources=False)


@requires_contract
@requires_module
@requires_report
def test_finalize_binds_only_to_the_new_channel_training_report(tmp_path) -> None:
    suite = _suite()
    path = _written(_pending(), tmp_path)
    result = suite.finalize(path)
    written = json.loads(path.read_text())
    expected = _filled()

    assert result["status"] == "filled"
    assert written["selected_model"]["optimizer_step"] in CHECKPOINT_STEPS
    for key in ("selected_checkpoint", "selected_normalization", "selected_report"):
        assert written["artifacts"][key] == expected["artifacts"][key]
    suite.load_contract(path, verify_sources=False)



def _gate_inputs(
    *,
    finite: bool = True,
    magnitude: float = 3.0,
    minimum: float = -20.0,
    std_scale: float = 1.0,
) -> tuple[dict, dict]:
    suite = _suite()
    rng = np.random.default_rng(0)
    grid = 32
    wet = np.ones((grid, grid), dtype=np.uint8)
    truth = rng.normal(0.0, 5.0, size=(grid, grid))
    model = truth * std_scale
    model = model - model.min() + minimum
    model = (model - model.mean()) * (
        (truth.std() * std_scale) / max(model.std(), 1e-12)
    ) + model.mean()
    model[0, 0] = minimum
    arrays = {
        "wet_mask": wet,
        "figure7_model_streamfunction": np.stack([model * 0.5, model]),
        "figure7_truth_streamfunction": np.stack([truth * 0.5, truth]),
    }
    summary = {
        "all_selected_states_finite": finite,
        "maximum_selected_normalized_abs": magnitude,
        "rmse": {
            field: {
                "model": {"day2000_mean": 1.0},
                "climatology": {"day2000_mean": 1.25},
            }
            for field in suite.RMSE_FIELDS
        },
    }
    return arrays, summary


def test_long_rollout_gate_passes_and_uses_the_frozen_thresholds() -> None:
    suite = _suite()
    arrays, summary = _gate_inputs(magnitude=8.0, minimum=-33.0)
    gate = suite.long_rollout_gate(arrays, summary)
    assert gate["long_rollout_conditions_pass"]
    assert all(gate["conditions"].values())
    assert suite.MAXIMUM_NORMALIZED_MAGNITUDE == 8.0
    assert suite.MINIMUM_STREAMFUNCTION_SV == -33.0
    assert suite.DAY2000_STD_RATIO_RANGE == (0.80, 1.25)


@pytest.mark.parametrize(
    "kwargs,failing",
    [
        ({"finite": False}, "all_values_finite"),
        ({"magnitude": 8.5}, "maximum_normalized_magnitude_at_most_8"),
        ({"minimum": -40.0}, "streamfunction_minimum_at_least_minus_33_sv"),
        ({"std_scale": 0.5}, "day2000_spatial_std_ratio_in_range"),
        ({"std_scale": 1.5}, "day2000_spatial_std_ratio_in_range"),
    ],
)
def test_each_long_rollout_condition_can_fail_independently(kwargs, failing) -> None:
    arrays, summary = _gate_inputs(**kwargs)
    gate = _suite().long_rollout_gate(arrays, summary)
    assert gate["conditions"][failing] is False
    assert not gate["long_rollout_conditions_pass"]
    assert sum(not value for value in gate["conditions"].values()) == 1


def test_day2000_climatology_ratio_remains_advisory() -> None:
    suite = _suite()
    arrays, summary = _gate_inputs()
    for field in suite.RMSE_FIELDS:
        summary["rmse"][field]["model"]["day2000_mean"] = 1.25
    gate = suite.long_rollout_gate(arrays, summary)
    assert all(
        value == pytest.approx(1.0)
        for value in gate["advisory_day2000_rmse_ratio_to_climatology"].values()
    )
    assert gate["long_rollout_conditions_pass"]
    assert "not gated" in gate["advisory_note"]


def test_captions_and_readme_name_the_static_channel_comparison() -> None:
    suite = _suite()
    with suite.FineTuneLabels("S0", 0.1, EXAMPLE_STEP) as labels:
        assert (
            labels.rewrite("S0 architecture-direction comparison")
            == "S0 engineered vs physical static channels"
        )
        assert labels.rewrite("Prior residual Model C") == (
            "tau, wet, d_wall (step 3,840)"
        )
        assert labels.rewrite("Selected anomaly-direct Model C") == (
            "tau, wet, f, dx, theta_clim (step 2,880)"
        )
    text = suite._readme(
        "S0",
        {
            "selected_optimizer_step": EXAMPLE_STEP,
            "report_content_sha256": "0" * 64,
        },
    )
    flat = " ".join(text.split())
    assert "literal step-3,840" in flat
    assert "tau_x, wet mask, f(phi), dx(phi), theta_clim(x, y)" in flat
    assert "`(x_(t-10), x_t)`" in flat
    assert "6263--6979" in flat
    assert "never a scored target" in flat
    assert suite.GATE_NAME in text


def test_both_steppers_are_history_aware_and_only_one_reads_the_new_block() -> None:
    """Figure 6's two curves differ in statics, not in temporal context."""

    pytest.importorskip("neuralop")
    from oceanfno.model import BireTwoInNewChannelsStepper, BireTwoInStepper

    assert BireTwoInStepper.requires_history is True
    assert BireTwoInNewChannelsStepper.requires_history is True
    assert issubclass(BireTwoInNewChannelsStepper, BireTwoInStepper)
    # Only the new-channel adapter carries a derived block.
    assert "static_block" in BireTwoInNewChannelsStepper.__init__.__code__.co_varnames


def test_canonical_figure_module_has_no_wrapper_context() -> None:
    assert not hasattr(_suite(), "_y32_figures_runner")


@pytest.mark.skipif(
    not SBATCH.is_file(), reason="the two-input figure launcher is absent"
)
def test_launcher_finalizes_then_runs_only_the_canonical_module() -> None:
    text = SBATCH.read_text()
    invoked = {
        line.split("-m", 1)[1].strip().split()[0]
        for line in text.splitlines()
        if " -m " in f" {line} " and "oceanfno." in line
    }
    assert invoked == {"oceanfno.figures"}
    assert "model_c_2in_1out_new_channels_s0_figures_v1.json" in text
    assert text.index("finalize") < text.index("  preflight") < text.index("  run")
