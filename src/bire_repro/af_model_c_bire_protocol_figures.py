"""S0 Bire-style Figure 3--8 suite for the Bire Section 3.2 protocol arm.

Publishes the six figures for the pooled Bire-protocol model on S0's inference
set, using the frozen plotting functions from :mod:`af_model_c_bire_s0_figures`
so the figure definitions, axes, reductions, and filenames stay identical to
every earlier package.  S0 only, as the primary Bire-style comparison.

The 15 members are drawn from the inference set 6200--6999.  That set is the
final 1,000 days of validation, exactly as the paper has it, restricted to the
starts that complete a 2,000-day rollout inside the store.

Lead-matched truth to day 2,000 comes from days 7200--8999, which the model
never saw in any capacity --- not trained on, not validated on, not an inference
start.  The paper has the same separation: its Figure 7 carries a ground-truth
column at the 2,000th day, obtained because simulations 2 and 4 are entirely
held out.  We obtain it along time rather than across simulations.

Figure 6's black curve is this run's own step-1,920 checkpoint against the
selected step-7,680 checkpoint --- a training-progress comparison on identical
data, since the prior residual Model C belongs to a different dataset and
normalizer.  The frozen plotters' hard-coded captions are rewritten accordingly.

Held-evaluation package: no training, no checkpoint selection, no promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from . import af_model_c_bire_s0_figures as figures
from .af_a0_evaluate import _normalizers
from .af_bire_protocol_split import INFERENCE_RANGE, TRAIN_RANGE
from .af_bire_protocol_split import inference_starts as protocol_inference_starts
from .af_bire_protocol_split import (
    INFERENCE_START_RANGE,
    MAXIMUM_INFERENCE_ROLLOUT_DAYS,
    RECORD_DAYS,
    STORE_DAYS,
    assert_model_visible,
    assert_truth_available,
)
from .af_data_v3 import DATASET_VERSION
from .af_forward_complete import _member_acc, _member_rmse, derived_fields
from .af_model_c import MODEL_C_LOSS_V1_CONTRACT_SHA256
from .af_model_c_bire_aligned_chronological import _gather
from .af_model_c_bire_aligned_full_state import (
    BireAlignedArchitecture,
    BireAlignedStepper,
    _json_sha256,
    build_bire_aligned_model,
)
from .af_model_c_bire_protocol import train_only_climatology
from .af_model_c_overfit import _device, _file_sha256

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

VERSION = "model_c_bire_protocol_s0_figures_v1"
CONTRACT_STATUS = "frozen_after_the_bire_protocol_training_and_validation_and_before_any_inference_metric"

MEMBER_COUNT = 15
START_SEED = 20260802
COMPARATOR_STEP = 1920
LEAD_DAYS = figures.LEAD_DAYS
SHORT_LEAD_DAYS = figures.SHORT_LEAD_DAYS
RMSE_FIELDS = figures.RMSE_FIELDS
ACC_FIELDS = figures.ACC_FIELDS


class BireProtocolFigureError(RuntimeError):
    """Raised when the Bire-protocol figure contract is violated."""


#: S0 is the primary Bire-style comparison; this suite publishes it alone.
REGIMES = ("S0",)
REGIME_INDEX = {"S0": 0}


def declared_inference_starts() -> np.ndarray:
    """15 members from the inference set, fixed by the declared seed."""

    starts = protocol_inference_starts(MEMBER_COUNT, START_SEED)
    # The starts are what the model is handed; truth may run past the record.
    assert_model_visible(starts, "inference starts")
    assert_truth_available(
        starts + MAXIMUM_INFERENCE_ROLLOUT_DAYS, "day-2,000 lead-matched truth"
    )
    return starts


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the figure contract frozen before any Bire-protocol inference metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    protocol = contract.get("protocol", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or contract.get("dataset", {}).get("version") != DATASET_VERSION
        or int(protocol.get("member_count", -1)) != MEMBER_COUNT
        or int(protocol.get("maximum_lead_days", -1)) != max(LEAD_DAYS)
        or int(protocol.get("prediction_interval_days", -1)) != 10
        or tuple(protocol.get("start_draw_order", ())) != tuple(int(v) for v in declared_inference_starts())
        or tuple(protocol.get("figure_names", ())) != figures.FIGURE_NAMES
        or tuple(protocol.get("rmse_fields", ())) != RMSE_FIELDS
        or tuple(protocol.get("acc_fields", ())) != ACC_FIELDS
        or tuple(protocol.get("regimes", ())) != REGIMES
        or protocol.get("primary_regime") != "S0"
        or tuple(protocol.get("inference_set", ())) != INFERENCE_RANGE
        or contract.get("figure6", {}).get("literal_pretrain_finetune_pair") is not False
        or int(contract.get("figure6", {}).get("comparator_optimizer_step", -1))
        != COMPARATOR_STEP
    ):
        raise BireProtocolFigureError("Bire-protocol figure contract changed")
    if verify_sources:
        for label, specification in contract["artifacts"].items():
            figures._verify_file(specification, label)
        root = resolved.parents[1]
        for relative, expected in contract["source_hashes"].items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise BireProtocolFigureError(f"source changed: {relative}")
    return contract, resolved, _file_sha256(resolved)


def _stepper(
    contract: Mapping[str, Any],
    key: str,
    device: Any,
    wet: np.ndarray,
    wind_mean: float,
    wind_scale: float,
) -> BireAlignedStepper:
    """Build one Bire-protocol checkpoint's stepper and verify its identity."""

    record = contract["artifacts"][key]
    payload = torch.load(Path(record["path"]), map_location=device, weights_only=False)
    architecture_dict = contract["selected_model"]["architecture"]
    expected_step = int(
        contract["selected_model"]["optimizer_step"]
        if key == "selected_checkpoint"
        else COMPARATOR_STEP
    )
    if (
        payload.get("architecture") != architecture_dict
        or int(payload.get("optimizer_step", -1)) != expected_step
        or payload.get("dataset_version") != DATASET_VERSION
        or payload.get("base_loss_contract_sha256") != MODEL_C_LOSS_V1_CONTRACT_SHA256
    ):
        raise BireProtocolFigureError(f"{key} identity, dataset, or objective changed")
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


#: Wind label per regime, for titles the frozen plotters hard-code as S0's.
REGIME_WIND_LABEL = {"S0": "Control wind", "S1": "Low wind", "S2": "High wind"}


class RegimeLabels:
    """Rewrite the frozen plotters' hard-coded S0 control-wind captions.

    The frozen figure functions embed ``$\\tau_0=0.1$``, ``One S0 inference
    member``, ``S0 architecture-direction comparison``, and the v2 legend names
    directly in their title strings.  Reusing those functions verbatim --- which
    is what keeps the figure definitions identical across every package --- would
    therefore label the S1 and S2 packages with S0's wind and with a comparison
    this run does not perform.

    Copying the plotters instead would let the definitions drift, so the captions
    are rewritten in place for the duration of one regime's plotting.  The
    substitutions are explicit and asserted by tests; nothing about the data,
    axes, or reductions changes.
    """

    def __init__(self, regime: str, tau0: float, selected_step: int) -> None:
        self.regime = regime
        tau = f"{tau0:g}"
        wind = REGIME_WIND_LABEL[regime]
        self.rules: tuple[tuple[str, str], ...] = (
            # Longest first: the bare tau rule would otherwise consume this one.
            (
                r"Control wind $\tau_0=0.1$ N m$^{-2}$",
                rf"{wind} $\tau_0={tau}$ N m$^{{-2}}$",
            ),
            (r"$\tau_0=0.1$ N m$^{-2}$", rf"$\tau_0={tau}$ N m$^{{-2}}$"),
            # "inference" is Bire's own word for this block, so unlike the v3
            # chronological arm -- which called it the test block -- the captions
            # keep it; only the regime name is substituted.
            ("One S0 inference member", f"One {regime} inference member"),
            (
                "S0 architecture-direction comparison",
                f"{regime} training-progress comparison",
            ),
            ("Prior residual Model C", f"Step {COMPARATOR_STEP:,} checkpoint"),
            (
                "Selected anomaly-direct Model C",
                f"Selected step {selected_step:,} checkpoint",
            ),
        )

    def rewrite(self, text: Any) -> Any:
        if not isinstance(text, str):
            return text
        for old, new in self.rules:
            text = text.replace(old, new)
        return text

    def __enter__(self) -> "RegimeLabels":
        import matplotlib.axes
        import matplotlib.figure

        self._axes = matplotlib.axes.Axes
        self._figure = matplotlib.figure.Figure
        self._set_title = matplotlib.axes.Axes.set_title
        self._suptitle = matplotlib.figure.Figure.suptitle
        self._plot = matplotlib.axes.Axes.plot
        self._method_labels = figures.METHOD_LABELS
        rewrite, set_title, suptitle, plot = (
            self.rewrite,
            self._set_title,
            self._suptitle,
            self._plot,
        )

        def patched_set_title(axis, label, *args, **kwargs):
            return set_title(axis, rewrite(label), *args, **kwargs)

        def patched_suptitle(figure, text, *args, **kwargs):
            return suptitle(figure, rewrite(text), *args, **kwargs)

        def patched_plot(axis, *args, **kwargs):
            if "label" in kwargs:
                kwargs["label"] = rewrite(kwargs["label"])
            return plot(axis, *args, **kwargs)

        matplotlib.axes.Axes.set_title = patched_set_title
        matplotlib.figure.Figure.suptitle = patched_suptitle
        matplotlib.axes.Axes.plot = patched_plot
        figures.METHOD_LABELS = {
            **self._method_labels,
            "model": "Bire-protocol Model C",
        }
        return self

    def __exit__(self, *exc: Any) -> None:
        self._axes.set_title = self._set_title
        self._figure.suptitle = self._suptitle
        self._axes.plot = self._plot
        figures.METHOD_LABELS = self._method_labels


def _fields(states: np.ndarray, wet: np.ndarray) -> dict[str, np.ndarray]:
    result = derived_fields(states, wet)
    result["surface_u"] = np.asarray(states[:, 0], dtype=np.float32)
    result["surface_v"] = np.asarray(states[:, 15], dtype=np.float32)
    return result


def evaluate_regime(
    selected: BireAlignedStepper,
    comparator: BireAlignedStepper,
    state: Any,
    static: Any,
    regime_index: int,
    starts: np.ndarray,
    climatology_state: np.ndarray,
    climatology_derived: Mapping[str, np.ndarray],
    wet: np.ndarray,
) -> dict[str, np.ndarray]:
    """Roll both checkpoints to day 2,000 and reduce exactly as the frozen suite."""

    records = np.stack(
        [np.full(starts.shape, regime_index, dtype=np.int64), starts], axis=1
    )
    initial = _gather(state, records, 0)
    experiments = records[:, 0]
    selected_current = selected.normalized_state(initial)
    selected_static = selected.normalized_static(static, experiments)
    comparator_current = comparator.normalized_state(initial)
    comparator_static = comparator.normalized_static(static, experiments)
    initial_fields = _fields(initial, wet)
    climate = np.repeat(climatology_state[regime_index][None], starts.size, axis=0)
    climate_fields = _fields(climate, wet)
    for name, value in climatology_derived.items():
        climate_fields[name] = np.repeat(value[regime_index][None], starts.size, axis=0)

    arrays: dict[str, np.ndarray] = {
        "lead_days": np.asarray(LEAD_DAYS, dtype=np.int16),
        "short_lead_days": np.asarray(SHORT_LEAD_DAYS, dtype=np.int16),
        "start_draw_order": starts.astype(np.int32),
        "finite": np.empty((starts.size, len(LEAD_DAYS)), dtype=np.uint8),
        "normalized_max_abs": np.empty((starts.size, len(LEAD_DAYS)), dtype=np.float32),
    }
    for method in figures.METHODS:
        for field in RMSE_FIELDS:
            arrays[f"rmse__{method}__{field}"] = np.empty(
                (starts.size, len(LEAD_DAYS)), dtype=np.float32
            )
    for name in ("selected", "prior"):
        for field in ACC_FIELDS:
            arrays[f"acc__{name}__{field}"] = np.empty(
                (starts.size, len(SHORT_LEAD_DAYS)), dtype=np.float32
            )
    arrays["single_rmse__streamfunction"] = np.empty(len(SHORT_LEAD_DAYS), dtype=np.float32)
    arrays["single_rmse__sst"] = np.empty_like(arrays["single_rmse__streamfunction"])
    for tag, leads in (("figure3", figures.FIGURE_3_LEADS), ("figure7", figures.FIGURE_7_LEADS)):
        arrays[f"{tag}_truth_streamfunction"] = np.empty((len(leads), *wet.shape), dtype=np.float32)
        arrays[f"{tag}_model_streamfunction"] = np.empty_like(arrays[f"{tag}_truth_streamfunction"])
    figure3 = {lead: index for index, lead in enumerate(figures.FIGURE_3_LEADS)}
    figure7 = {lead: index for index, lead in enumerate(figures.FIGURE_7_LEADS)}
    wet_tensor = torch.from_numpy(wet).to(selected.device)

    with torch.no_grad():
        for lead_index, lead in enumerate(LEAD_DAYS):
            if lead:
                selected_current = selected.step(selected_current, selected_static)
                selected_prediction = selected.physical(selected_current)
                if lead <= 200:
                    comparator_current = comparator.step(comparator_current, comparator_static)
                    comparator_prediction = comparator.physical(comparator_current)
            else:
                selected_prediction = initial.copy()
                comparator_prediction = initial.copy()
            truth = _gather(state, records, lead)
            truth_fields = _fields(truth, wet)
            selected_fields = _fields(selected_prediction, wet)
            for field in RMSE_FIELDS:
                arrays[f"rmse__model__{field}"][:, lead_index] = _member_rmse(
                    selected_fields[field], truth_fields[field], wet)
                arrays[f"rmse__persistence__{field}"][:, lead_index] = _member_rmse(
                    initial_fields[field], truth_fields[field], wet)
                arrays[f"rmse__climatology__{field}"][:, lead_index] = _member_rmse(
                    climate_fields[field], truth_fields[field], wet)
            arrays["finite"][:, lead_index] = np.isfinite(selected_prediction).all(axis=(1, 2, 3))
            arrays["normalized_max_abs"][:, lead_index] = (
                torch.amax(torch.abs(selected_current[:, :, wet_tensor]), dim=(1, 2))
                .detach().cpu().numpy()
            )
            if lead <= 200:
                short_index = lead // 10
                comparator_fields = _fields(comparator_prediction, wet)
                for field in ACC_FIELDS:
                    arrays[f"acc__selected__{field}"][:, short_index] = _member_acc(
                        selected_fields[field], truth_fields[field], climate_fields[field], wet)
                    arrays[f"acc__prior__{field}"][:, short_index] = _member_acc(
                        comparator_fields[field], truth_fields[field], climate_fields[field], wet)
                arrays["single_rmse__streamfunction"][short_index] = _member_rmse(
                    selected_fields["streamfunction"], truth_fields["streamfunction"], wet)[0]
                arrays["single_rmse__sst"][short_index] = _member_rmse(
                    selected_fields["sst"], truth_fields["sst"], wet)[0]
            if lead in figure3:
                arrays["figure3_truth_streamfunction"][figure3[lead]] = truth_fields["streamfunction"][0]
                arrays["figure3_model_streamfunction"][figure3[lead]] = selected_fields["streamfunction"][0]
            if lead in figure7:
                arrays["figure7_truth_streamfunction"][figure7[lead]] = truth_fields["streamfunction"][0]
                arrays["figure7_model_streamfunction"][figure7[lead]] = selected_fields["streamfunction"][0]
    return arrays


def _readme(regime: str, report: Mapping[str, Any]) -> str:
    role = "primary Bire-style comparison" if regime == "S0" else "wind-regime robustness"
    starts = declared_inference_starts()
    return f"""# Bire Section 3.2 protocol, {regime}: Figures 3--8

This package evaluates the seed-20260724, step-{report['selected_optimizer_step']:,}
checkpoint of the pooled Bire-protocol model on the **{regime}** inference set
(indices {INFERENCE_RANGE[0]}--{INFERENCE_RANGE[1] - 1}), tau0 = {report['tau0_n_m2']} N m-2.
{regime} is the {role}.

The model is the loss-recovery architecture and objective, unchanged: three FNO
blocks, six pointwise LayerNorms, modes 24x16, width 128, Bire positional
encoding, 10% padding, no external local branch, Model C loss v1 over a
three-step rollout, Adam 5e-4 decaying to 1e-4 at 75%, batch 8, 7,680 steps. One
FNO was trained on the pooled S0+S1+S2 training blocks (0--5999) and selected on
the pooled validation blocks (6000--7199).

**The split is Bire's, not a chronological variant.** 6,000 training + 1,200
validation is the paper's entire 7,200-day record, and the 1,000 inference days
are the final 1,000 of validation rather than a third block -- the paper states
it uses no third held-out set. There are no buffers, because the paper has none.
The inference set is therefore nested in validation; selection starts are drawn
from the 200 validation days outside it, so no selection start is also a member
start.

**Model visibility.** The model saw nothing at or beyond index 7200. Days
7200--8999 of trajectory-v3 carry no split code and are used only as evaluation
truth. That is what makes the day-2,000 ground-truth column of the paper's
Figure 7 reproducible here: the 15 starts are drawn from
{INFERENCE_START_RANGE[0]}--{INFERENCE_START_RANGE[1] - 1}, the part of the inference set that admits a
complete 2,000-day rollout inside the store, and this draw spans
{int(starts.min())}--{int(starts.max())}. Every member therefore has lead-matched MITgcm truth at
every lead out to day 2,000 ({int(starts.max())} + 2000 = {int(starts.max()) + 2000} < {STORE_DAYS}). Bire obtain the
same separation across simulations -- runs 2 and 4 are entirely held out -- and
we obtain it along time.

**Not comparable byte-for-byte with the v2 or v3 chronological packages.** The
15 starts are a new fixed draw under a different record indexing, and the
training and validation blocks differ, so comparison with earlier packages is
protocol-level only.

**Figure 6 comparator.** The prior residual Model C was trained on v2 with a v2
normalizer and branch-based S1/S2, so it cannot be run meaningfully here. The
black curve is instead this run's own step-{COMPARATOR_STEP:,} checkpoint against the
selected step-{report['selected_optimizer_step']:,} checkpoint: a training-progress
comparison on identical data, not the frozen architecture pairing.

Climatology is the pointwise {regime} mean over the Bire training block
(0--5999) only. Persistence holds each member's initial physical field fixed.
RMSE is computed over wet cells per member; lines and bands are the mean and
10th/90th percentiles across the 15 members.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `{report["report_content_sha256"]}`.
"""


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    """Publish the six figures for every regime on the v3 test block."""

    if torch is None:
        raise RuntimeError("v3 figure evaluation requires PyTorch")
    started = time.monotonic()
    contract, resolved_contract, contract_sha = load_contract(contract_path)
    device = _device(device_name)
    dataset = Path(contract["dataset"]["path"]).resolve()
    group = zarr.open_consolidated(str(dataset), mode="r")
    state = group["state"]
    static = group["static_features"]
    wet = np.asarray(group["wet_mask"][:], dtype=bool)
    longitude = np.asarray(group["longitude_deg"][:], dtype=np.float32)
    latitude = np.asarray(group["latitude_deg"][:], dtype=np.float32)
    _, _, _, _, wind_mean, wind_scale = _normalizers(group)
    starts = declared_inference_starts()
    climatology_state, climatology_derived, climatology_days = train_only_climatology(state, wet)
    if climatology_days != TRAIN_RANGE[1] - TRAIN_RANGE[0]:
        raise BireProtocolFigureError("the v3 train-only climatology did not cover 0--5039")

    selected = _stepper(contract, "selected_checkpoint", device, wet, wind_mean, wind_scale)
    comparator = _stepper(contract, "comparator_checkpoint", device, wet, wind_mean, wind_scale)

    published: dict[str, Any] = {}
    figures._style()
    for regime in REGIMES:
        regime_index = REGIME_INDEX[regime]
        output = Path(contract["output"]["project_root"]).resolve() / regime
        scratch = Path(contract["output"]["scratch_root"]).resolve() / regime
        for path in (output, scratch):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
        arrays = evaluate_regime(
            selected, comparator, state, static, regime_index, starts,
            climatology_state, climatology_derived, wet,
        )
        arrays["longitude_deg"] = longitude
        arrays["latitude_deg"] = latitude
        arrays["wet_mask"] = wet.astype(np.uint8)
        summary = figures._summary(arrays)
        output_tmp = output.with_name(output.name + ".tmp")
        scratch_tmp = scratch.with_name(scratch.name + ".tmp")
        output_tmp.parent.mkdir(parents=True, exist_ok=True)
        scratch_tmp.parent.mkdir(parents=True, exist_ok=True)
        output_tmp.mkdir()
        scratch_tmp.mkdir()
        try:
            scratch_arrays = scratch_tmp / figures.ARRAYS_NAME
            np.savez_compressed(scratch_arrays, **arrays)
            with RegimeLabels(
                regime,
                float(contract["dataset"]["tau0_n_m2"][regime]),
                int(contract["selected_model"]["optimizer_step"]),
            ):
                figures._plot_streamfunction_grid(output_tmp, arrays, longitude, latitude, wet)
                figures._plot_rmse(output_tmp, arrays, long=False)
                figures._plot_single_member(output_tmp, arrays)
                figures._plot_acc(output_tmp, arrays)
                figures._plot_day60_day2000(output_tmp, arrays, longitude, latitude, wet)
                figures._plot_rmse(output_tmp, arrays, long=True)
            figures._write_csv(output_tmp / figures.CSV_NAME, arrays)
            (output_tmp / figures.SUMMARY_NAME).write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
            report = {
                "version": VERSION,
                "status": "complete",
                "regime": regime,
                "tau0_n_m2": contract["dataset"]["tau0_n_m2"][regime],
                "role": "primary" if regime == "S0" else "robustness",
                "contract": str(resolved_contract),
                "contract_sha256": contract_sha,
                "dataset": str(dataset),
                "dataset_version": DATASET_VERSION,
                "selected_optimizer_step": int(contract["selected_model"]["optimizer_step"]),
                "comparator_optimizer_step": COMPARATOR_STEP,
                "start_draw_order": starts.astype(int).tolist(),
                "inference_start_range": list(INFERENCE_START_RANGE),
                "summary": summary,
                "arrays": str(scratch / figures.ARRAYS_NAME),
                "arrays_sha256": _file_sha256(scratch_arrays),
                "figures": list(figures.FIGURE_NAMES),
                "figure6": contract["figure6"],
                "regime_captions": "frozen S0 control-wind captions rewritten for this regime",
                "elapsed_seconds": time.monotonic() - started,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
            report["report_content_sha256"] = _json_sha256(report)
            (scratch_tmp / figures.REPORT_NAME).write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
            shutil.copy2(scratch_tmp / figures.REPORT_NAME, output_tmp / figures.REPORT_NAME)
            shutil.copy2(scratch_arrays, output_tmp / figures.ARRAYS_NAME)
            (output_tmp / figures.README_NAME).write_text(_readme(regime, report))
            manifest = {
                "version": VERSION,
                "regime": regime,
                "contract_sha256": contract_sha,
                "report_content_sha256": report["report_content_sha256"],
                "artifacts": {
                    path.name: {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}
                    for path in sorted(output_tmp.iterdir()) if path.is_file()
                },
            }
            manifest["manifest_content_sha256"] = _json_sha256(manifest)
            (output_tmp / figures.MANIFEST_NAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            scratch_tmp.replace(scratch)
            output_tmp.replace(output)
        except Exception:
            shutil.rmtree(scratch_tmp, ignore_errors=True)
            shutil.rmtree(output_tmp, ignore_errors=True)
            raise
        published[regime] = report
    return published


def preflight(contract_path: str | Path) -> dict[str, Any]:
    """Verify sources, starts, and both checkpoint identities without plotting."""

    contract, resolved, digest = load_contract(contract_path)
    starts = declared_inference_starts()
    return {
        "status": "pass",
        "version": VERSION,
        "contract": str(resolved),
        "contract_sha256": digest,
        "dataset_version": DATASET_VERSION,
        "regimes": list(REGIMES),
        "primary_regime": "S0",
        "member_count": MEMBER_COUNT,
        "start_draw_order": starts.astype(int).tolist(),
        "inference_start_range": list(INFERENCE_START_RANGE),
        "maximum_lead_days": max(LEAD_DAYS),
        "selected_optimizer_step": int(contract["selected_model"]["optimizer_step"]),
        "comparator_optimizer_step": COMPARATOR_STEP,
        "continuation_required": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--contract", type=Path, required=True)
        if command == "run":
            child.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = preflight(args.contract) if args.command == "preflight" else run(
        args.contract, device_name=args.device
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
