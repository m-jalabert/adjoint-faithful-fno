"""S0 Bire-style Figure 3--8 suite and acceptance gate for the rollout fine-tune.

Publishes the same six figures, on the same 15 members drawn from 6200--6999 by
the same seed, over the same lead grid to day 2,000, against the same
train-only climatology and persistence baselines as every earlier package.  Only
the checkpoint differs, so this package and
``outputs/af_fno/C/bire_protocol_duration_s0_figures_v1/S0`` are directly
comparable field for field.

Figure 6 is, for the first time in this project, a **literal pre-train /
fine-tune pair**: the black curve is the step-15,360 checkpoint the fine-tune
started from and the red curve is the selected fine-tuned checkpoint, both
trained on the same data in the same normalized coordinates by the same code
path.  Earlier packages could only offer a within-run training-progress
comparison, and their contracts record ``literal_pretrain_finetune_pair: false``
for that reason; this one records ``true``.

Two consequences follow, and both are handled here rather than in the frozen
suite:

* The two checkpoints carry **different objective hashes** --- the comparator's
  is Model C loss v1 over three steps, the selected one's is the six-step
  contract --- so the suite's single-hash identity check cannot serve both.
  :func:`_stepper` verifies each against the hash the contract declares for it.
* The frozen captions name a training-progress comparison.  :class:`FineTuneLabels`
  rewrites them to name the fine-tune pairing instead.  Nothing about the data,
  axes, or reductions changes.

After the figures are published this module evaluates the **2,000-day half of
the final acceptance gate**, which is the half no earlier stage can reach:
finiteness, normalized magnitude, the day-2,000 streamfunction minimum in Sv,
and the day-2,000 spatial-standard-deviation ratio to truth.  It is written
beside the regime folder, not inside it, so the sealed regime manifest stays
valid.  The gate report also carries an advisory collapse indicator --- the
day-2,000 model RMSE divided by climatology's --- because a model that has
relaxed onto climatology can satisfy every bound on amplitude and finiteness
while carrying no circulation at all, and this project has recorded that
failure before.

Held-evaluation package: no training, no checkpoint selection, no promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import af_model_c_bire_protocol_figures as suite
from . import af_model_c_bire_s0_figures as figures
from .af_data_v3 import DATASET_VERSION
from .af_model_c import MODEL_C_LOSS_V1_CONTRACT_SHA256
from .af_model_c_bire_aligned_full_state import (
    BireAlignedArchitecture,
    BireAlignedStepper,
    _json_sha256,
    build_bire_aligned_model,
)
from .af_model_c_bire_protocol_rollout_ft import (
    BASELINE_OPTIMIZER_STEP,
    CHECKPOINT_STEPS,
    FINE_TUNE_LOSS_CONTRACT_SHA256,
    REPORT_NAME as TRAINING_REPORT_NAME,
    ROLLOUT_STEPS,
)
from .af_model_c_overfit import _file_sha256

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

VERSION = "model_c_bire_protocol_rollout_ft_s0_figures_v1"
CONTRACT_STATUS = (
    "frozen_after_the_bire_protocol_rollout_ft_training_and_validation"
    "_and_before_any_inference_metric"
)

MEMBER_COUNT = suite.MEMBER_COUNT
START_SEED = suite.START_SEED
REGIMES = suite.REGIMES

#: The checkpoint the fine-tune started from; figure 6's black curve.
COMPARATOR_STEP = BASELINE_OPTIMIZER_STEP
#: Distinguishes this package's red curve from the baseline package's.
MODEL_LABEL = f"Bire-protocol Model C ({ROLLOUT_STEPS}-step fine-tune)"

#: Contract fields that cannot be known until training has completed.  The
#: contract ships with them unfilled and `finalize` writes them from the run's
#: own report, so the figure job never starts against a half-declared contract.
PENDING = "PENDING_AFTER_TRAINING"
PENDING_PATHS: tuple[tuple[str, ...], ...] = (
    ("selected_model", "optimizer_step"),
    ("artifacts", "selected_checkpoint", "sha256"),
    ("artifacts", "selected_normalization", "sha256"),
    ("artifacts", "selected_report", "sha256"),
)

#: Section 6 of the arm declaration, the 2,000-day half.
MAXIMUM_NORMALIZED_MAGNITUDE = 8.0
MINIMUM_STREAMFUNCTION_SV = -33.0
DAY2000_STD_RATIO_RANGE = (0.80, 1.25)
GATE_NAME = "bire_protocol_rollout_ft_acceptance_gate.json"


class BireProtocolRolloutFineTuneFigureError(RuntimeError):
    """Raised when the rollout fine-tune figure contract is violated."""


def declared_inference_starts() -> np.ndarray:
    """The parent suite's 15 members, unchanged, so the packages compare."""

    return suite.declared_inference_starts()


def _read(contract: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = contract
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def unfilled_fields(contract: Mapping[str, Any]) -> list[str]:
    """Declared-pending contract fields that training has not yet supplied."""

    return [
        ".".join(path)
        for path in PENDING_PATHS
        if _read(contract, path) in (None, PENDING)
    ]


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the figure contract frozen after training and before any inference metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    protocol = contract.get("protocol", {})
    figure6 = contract.get("figure6", {})
    selected = contract.get("selected_model", {})
    pending = unfilled_fields(contract)
    if pending:
        raise BireProtocolRolloutFineTuneFigureError(
            "the figure contract still carries post-training fields: "
            + ", ".join(pending)
            + " -- run `finalize` against the training report first"
        )
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or contract.get("dataset", {}).get("version") != DATASET_VERSION
        or int(protocol.get("member_count", -1)) != MEMBER_COUNT
        or tuple(protocol.get("start_draw_order", ()))
        != tuple(int(v) for v in declared_inference_starts())
        or tuple(protocol.get("regimes", ())) != REGIMES
        or protocol.get("primary_regime") != "S0"
        or int(figure6.get("comparator_optimizer_step", -1)) != COMPARATOR_STEP
        or figure6.get("literal_pretrain_finetune_pair") is not True
        or int(selected.get("rollout_steps", -1)) != ROLLOUT_STEPS
        or selected.get("base_loss_contract_sha256") != FINE_TUNE_LOSS_CONTRACT_SHA256
        or contract.get("comparator_model", {}).get("base_loss_contract_sha256")
        != MODEL_C_LOSS_V1_CONTRACT_SHA256
        or int(selected.get("optimizer_step", -1)) not in CHECKPOINT_STEPS
    ):
        raise BireProtocolRolloutFineTuneFigureError("rollout fine-tune figure contract changed")
    if verify_sources:
        for label, specification in contract["artifacts"].items():
            figures._verify_file(specification, label)
        root = resolved.parents[1]
        for relative, expected in contract["source_hashes"].items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise BireProtocolRolloutFineTuneFigureError(f"source changed: {relative}")
    return contract, resolved, _file_sha256(resolved)


def _stepper(
    contract: Mapping[str, Any],
    key: str,
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> BireAlignedStepper:
    """Build one checkpoint's stepper against the objective declared for *it*.

    The suite's own ``_stepper`` checks every checkpoint against a single loss
    hash.  That is right when both curves come from one run; here the comparator
    is a three-step v1 model and the selected one is its six-step fine-tune, so
    each is verified against the hash its own contract block declares.
    """

    record = contract["artifacts"][key]
    payload = torch.load(Path(record["path"]), map_location=device, weights_only=False)
    if key == "selected_checkpoint":
        declared = contract["selected_model"]
        expected_step = int(declared["optimizer_step"])
    else:
        declared = contract["comparator_model"]
        expected_step = COMPARATOR_STEP
    architecture_dict = contract["selected_model"]["architecture"]
    if (
        payload.get("architecture") != architecture_dict
        or int(payload.get("optimizer_step", -1)) != expected_step
        or payload.get("dataset_version") != DATASET_VERSION
        or payload.get("base_loss_contract_sha256") != declared["base_loss_contract_sha256"]
        or int(payload.get("rollout_steps", -1)) != int(declared["rollout_steps"])
    ):
        raise BireProtocolRolloutFineTuneFigureError(
            f"{key} identity, dataset, or objective changed"
        )
    model = build_bire_aligned_model(BireAlignedArchitecture(**architecture_dict)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with np.load(Path(contract["artifacts"]["selected_normalization"]["path"])) as artifact:
        mean = np.asarray(artifact["pointwise_mean"], dtype=np.float32)
        scale = np.asarray(artifact["pointwise_scale"], dtype=np.float32)
    return BireAlignedStepper(
        model=model, device=device, wet=wet, mean=mean, scale=scale,
        wind_mean=wind_mean, wind_scale=wind_scale,
    )


class FineTuneLabels(suite.RegimeLabels):
    """The frozen captions, rewritten for a literal pre-train / fine-tune pair.

    The suite rewrites ``S0 architecture-direction comparison`` into a
    *training-progress* comparison, which is what an earlier arm's own-run
    comparator was.  Here the two curves really are the model before and after
    fine-tuning, so the caption says so.
    """

    def __init__(self, regime: str, tau0: float, selected_step: int) -> None:
        super().__init__(regime, tau0, selected_step)
        self.rules = (
            (
                "S0 architecture-direction comparison",
                f"{regime} three-step model vs six-step fine-tune",
            ),
            (
                "Prior residual Model C",
                f"Before fine-tuning (step {COMPARATOR_STEP:,})",
            ),
            (
                "Selected anomaly-direct Model C",
                f"After {ROLLOUT_STEPS}-step fine-tune (step {selected_step:,})",
            ),
            *(
                rule
                for rule in self.rules
                if rule[0]
                not in (
                    "S0 architecture-direction comparison",
                    "Prior residual Model C",
                    "Selected anomaly-direct Model C",
                )
            ),
        )


def long_rollout_gate(
    arrays: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """The 2,000-day half of the final acceptance gate, from the published arrays."""

    wet = np.asarray(arrays["wet_mask"], dtype=bool)
    day2000 = figures.FIGURE_7_LEADS.index(2000)
    model = np.asarray(arrays["figure7_model_streamfunction"], dtype=np.float64)[day2000][wet]
    truth = np.asarray(arrays["figure7_truth_streamfunction"], dtype=np.float64)[day2000][wet]
    finite = bool(summary["all_selected_states_finite"])
    magnitude = float(summary["maximum_selected_normalized_abs"])
    minimum = float(model.min())
    truth_std = float(truth.std())
    ratio = float(model.std() / truth_std) if truth_std > 0.0 else float("inf")
    low, high = DAY2000_STD_RATIO_RANGE
    conditions = {
        "all_values_finite": finite,
        "maximum_normalized_magnitude_at_most_8": magnitude <= MAXIMUM_NORMALIZED_MAGNITUDE,
        "streamfunction_minimum_at_least_minus_33_sv": minimum >= MINIMUM_STREAMFUNCTION_SV,
        "day2000_spatial_std_ratio_in_range": low <= ratio <= high,
    }
    collapse = {
        field: float(summary["rmse"][field]["model"]["day2000_mean"])
        / float(summary["rmse"][field]["climatology"]["day2000_mean"])
        for field in figures.RMSE_FIELDS
    }
    return {
        "measured": {
            "all_values_finite": finite,
            "maximum_normalized_magnitude": magnitude,
            "day2000_streamfunction_minimum_sv": minimum,
            "day2000_streamfunction_maximum_sv": float(model.max()),
            "day2000_truth_streamfunction_minimum_sv": float(truth.min()),
            "day2000_spatial_std_ratio_to_truth": ratio,
        },
        "thresholds": {
            "maximum_normalized_magnitude": MAXIMUM_NORMALIZED_MAGNITUDE,
            "streamfunction_minimum_sv": MINIMUM_STREAMFUNCTION_SV,
            "day2000_spatial_std_ratio": list(DAY2000_STD_RATIO_RANGE),
        },
        "conditions": conditions,
        "long_rollout_conditions_pass": all(conditions.values()),
        "advisory_day2000_rmse_ratio_to_climatology": collapse,
        "advisory_note": (
            "a ratio at or near 1.0 means the day-2,000 field is indistinguishable "
            "from climatology; it is reported, not gated, because the arm "
            "declaration does not require day-2,000 skill over persistence"
        ),
        "streamfunction_basis": (
            "member 0's day-2,000 barotropic streamfunction, the field figure 7 "
            "publishes and the visual criterion inspects"
        ),
        "visual_criterion": "by_inspection_of_figures_3_and_7",
    }


def acceptance_gate(contract: Mapping[str, Any], regime: str = "S0") -> dict[str, Any]:
    """Join the training arm's validation half with this package's 2,000-day half."""

    output = Path(contract["output"]["project_root"]).resolve() / regime
    with np.load(output / figures.ARRAYS_NAME) as stored:
        arrays = {name: stored[name] for name in stored.files}
    summary = json.loads((output / figures.SUMMARY_NAME).read_text())
    training = json.loads(
        Path(contract["artifacts"]["selected_report"]["path"]).read_text()
    )
    validation = training["acceptance_gate"]
    long_half = long_rollout_gate(arrays, summary)
    gate = {
        "version": VERSION,
        "regime": regime,
        "selected_optimizer_step": int(contract["selected_model"]["optimizer_step"]),
        "comparator_optimizer_step": COMPARATOR_STEP,
        "validation_half": validation,
        "long_rollout_half": long_half,
        "measurable_conditions_pass": bool(
            validation["validation_conditions_pass"]
            and long_half["long_rollout_conditions_pass"]
        ),
        "remaining_by_inspection": (
            "western boundary sharp, gyre structure identifiable, no basin-wide "
            "smoothing -- figures 3 and 7"
        ),
        "decision_note": (
            "the arm declaration freezes this checkpoint and opens the adjoint "
            "study after the evaluation regardless of whether this gate passes"
        ),
    }
    gate["content_sha256"] = _json_sha256(gate)
    return gate


#: The suite instantiates its captions by the name ``RegimeLabels``, so the
#: binding below resolves that name in this module's namespace.
RegimeLabels = FineTuneLabels

PARENT_BINDINGS = (
    "VERSION",
    "CONTRACT_STATUS",
    "COMPARATOR_STEP",
    "MODEL_LABEL",
    "RegimeLabels",
    "load_contract",
    "_stepper",
    "_readme",
)


class _SuiteBinding:
    """Bind this arm's loader, stepper, captions, comparator and README into the suite.

    ``_stepper`` and ``RegimeLabels`` are bound in addition to the usual
    constants because a literal pre-train / fine-tune pair is the one case the
    frozen suite does not express: it assumes both curves share an objective
    hash and describes the comparison as within-run progress.
    """

    def __enter__(self) -> "_SuiteBinding":
        here = globals()
        missing = [name for name in PARENT_BINDINGS if name not in here]
        if missing:
            # Resolve everything before assigning anything: a half-applied
            # binding never runs __exit__, so it would leak into the suite for
            # the rest of the process.
            raise BireProtocolRolloutFineTuneFigureError(
                f"this arm defines no replacement for {missing!r}"
            )
        self._saved = {name: getattr(suite, name) for name in PARENT_BINDINGS}
        for name in PARENT_BINDINGS:
            setattr(suite, name, here[name])
        return self

    def __exit__(self, *exc: Any) -> None:
        for name, value in self._saved.items():
            setattr(suite, name, value)


def _readme(regime: str, report: Mapping[str, Any]) -> str:
    starts = declared_inference_starts()
    selected = int(report["selected_optimizer_step"])
    return f"""# Six-step rollout fine-tune, {regime}: Figures 3--8

This package evaluates the step-{selected:,} checkpoint of the six-step rollout
fine-tune on the **{regime}** inference set (indices 6200--7199), tau0 =
{report['tau0_n_m2']} N m-2.

The fine-tune started from the step-{COMPARATOR_STEP:,} checkpoint of
`model_c_bire_protocol_duration_v1` and continued for 3,840 steps at 2e-5 with
the autoregressive rollout deepened from three ten-day calls to six and the
rollout weight raised from 0.15 to 0.50. Architecture, normalization, split,
Fourier modes, static inputs, positional encoding and the 46-channel output are
unchanged, so this package and
`outputs/af_fno/C/bire_protocol_duration_s0_figures_v1/S0/` differ only in the
checkpoint: same 15 members, same seed, same lead grid, same truth window, same
climatology and persistence baselines.

The starts are drawn from 6200--6999, this draw spanning {int(starts.min())}--{int(starts.max())},
so every member has lead-matched MITgcm truth to day 2,000
({int(starts.max())} + 2000 = {int(starts.max()) + 2000} < 9000) from days 7200--8999, which the
model never saw in any capacity.

**Figure 6 is a literal pre-train / fine-tune pair.** The black curve is the
step-{COMPARATOR_STEP:,} model the fine-tune started from; the red curve is the
selected step-{selected:,} fine-tuned model. Both were trained on the same data in
the same normalized coordinates by the same code path, so the gap between them
is what deepening the rollout bought and nothing else. Every earlier package in
this project could only offer a within-run training-progress comparison.

The two checkpoints carry different objective hashes -- v1 over three steps for
the comparator, the six-step contract for the selected model -- and each is
verified against its own.

Climatology is the pointwise {regime} mean over the Bire training block
(0--5999) only. Persistence holds each member's initial physical field fixed.
RMSE is computed over wet cells per member; lines and bands are the mean and
10th/90th percentiles across the 15 members.

The 2,000-day half of the final acceptance gate is written beside this folder as
`{GATE_NAME}`.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `{report['report_content_sha256']}`.
"""


def finalize(contract_path: str | Path) -> dict[str, Any]:
    """Fill the declared-pending fields from the training run's own report.

    Idempotent: a field already equal to what the report says is left alone, and
    a field that disagrees is refused rather than overwritten, so this can be run
    before every figure job without becoming a way to edit a frozen contract.
    """

    resolved = Path(contract_path).resolve()
    contract = json.loads(resolved.read_text())
    report_path = Path(contract["artifacts"]["selected_report"]["path"])
    if not report_path.is_file():
        raise BireProtocolRolloutFineTuneFigureError(
            f"the training report is not on disk yet: {report_path}"
        )
    if report_path.name != TRAINING_REPORT_NAME:
        raise BireProtocolRolloutFineTuneFigureError(
            f"the declared report is not {TRAINING_REPORT_NAME}"
        )
    report = json.loads(report_path.read_text())
    if report.get("version") != "model_c_bire_protocol_rollout_ft_v1":
        raise BireProtocolRolloutFineTuneFigureError("the report is not this arm's")
    published = report["published_checkpoint"]
    resolutions = {
        ("selected_model", "optimizer_step"): int(published["optimizer_step"]),
        ("artifacts", "selected_checkpoint", "sha256"): str(published["checkpoint_sha256"]),
        ("artifacts", "selected_normalization", "sha256"): str(published["normalization_sha256"]),
        ("artifacts", "selected_report", "sha256"): _file_sha256(report_path),
    }
    applied: dict[str, Any] = {}
    for path, value in resolutions.items():
        current = _read(contract, path)
        if current not in (None, PENDING) and current != value:
            raise BireProtocolRolloutFineTuneFigureError(
                f"{'.'.join(path)} is already {current!r}, not {value!r}; "
                "refusing to overwrite a filled contract field"
            )
        node = contract
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        if current != value:
            applied[".".join(path)] = value
    for key, declared in (
        ("selected_checkpoint", published["checkpoint"]),
        ("selected_normalization", published["normalization"]),
    ):
        if contract["artifacts"][key]["path"] != declared:
            raise BireProtocolRolloutFineTuneFigureError(
                f"{key} path disagrees with the training report: "
                f"{contract['artifacts'][key]['path']} vs {declared}"
            )
    if applied:
        resolved.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return {
        "status": "filled" if applied else "already_complete",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": _file_sha256(resolved),
        "applied": applied,
        "selected_optimizer_step": int(published["optimizer_step"]),
    }


def preflight(contract_path: str | Path) -> dict[str, Any]:
    with _SuiteBinding():
        result = dict(suite.preflight(contract_path))
    contract, _, _ = load_contract(contract_path, verify_sources=False)
    result["comparator_optimizer_step"] = COMPARATOR_STEP
    result["literal_pretrain_finetune_pair"] = True
    result["selected_rollout_steps"] = ROLLOUT_STEPS
    result["selected_base_loss_contract_sha256"] = FINE_TUNE_LOSS_CONTRACT_SHA256
    result["acceptance_gate_artifact"] = str(
        Path(contract["output"]["project_root"]).resolve() / GATE_NAME
    )
    return result


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    """Publish the six figures, then evaluate the 2,000-day acceptance gate."""

    with _SuiteBinding():
        published = dict(suite.run(contract_path, device_name=device_name))
    contract, _, _ = load_contract(contract_path, verify_sources=False)
    root = Path(contract["output"]["project_root"]).resolve()
    gate = acceptance_gate(contract, "S0")
    # Beside the regime folder, never inside it: that package's manifest is
    # already sealed over its own file list.
    (root / GATE_NAME).write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    published["acceptance_gate"] = gate
    return published


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fill = subparsers.add_parser("finalize")
    fill.add_argument("--contract", required=True)
    check = subparsers.add_parser("preflight")
    check.add_argument("--contract", required=True)
    plot = subparsers.add_parser("run")
    plot.add_argument("--contract", required=True)
    plot.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "finalize":
        result: Any = finalize(args.contract)
    elif args.command == "preflight":
        result = preflight(args.contract)
    else:
        result = run(args.contract, device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
