"""360-day validation rollouts, the checkpoint-selection rule, and climatology.

Every checkpoint is autoregressed 360 days at ten-day steps from all 102 pooled
validation starts (34 per regime, stride 6, days 6000--6198), and scored against
lead-matched MITgcm truth with persistence and a train-only climatology as the
two reference curves. The selection rule is then:

1. compute the 10--90-day RMSE-AUC of each primary field;
2. keep only checkpoints within 5 % of the best short AUC in *every* field;
3. **keep only checkpoints whose measured perturbation growth rate is at or
   below one**;
4. among those, take the one minimizing the worst 90--360-day RMSE-AUC ratio to
   climatology.

Condition 3 exists because v1 was blind to it: every one of its four checkpoints
had a growth rate above one (1.0146--1.0227) and the rule still returned the
best-RMSE checkpoint, which then amplified 28x over the 2,000-day inference.
A criterion that cannot see the failure cannot avoid selecting it.

Exactly one checkpoint is published as ``selected.pt``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .runtime import STATE_CHANNEL_COUNT, torch
from .dataset import TRAIN_RANGE, VALIDATION_ROLLOUT_DAYS
from .diagnostics import _member_acc, _member_rmse, derived_fields
from .model import ProductionStepper
from .perturbation_growth import GROWTH_RATE_CEILING

LEAD_DAYS = tuple(range(10, VALIDATION_ROLLOUT_DAYS + 1, 10))

SHORT_LEADS = tuple(lead for lead in LEAD_DAYS if 10 <= lead <= 90)

LONG_LEADS = tuple(lead for lead in LEAD_DAYS if 90 <= lead <= 360)

ACC_LEAD_LIMIT = 200

#: Validation members carried through the 360-day rollout at once. Independent
#: of the result -- every reduction over members is taken after all chunks have
#: contributed -- so this is purely a memory knob.
VALIDATION_MEMBER_CHUNK = 4


def validation_member_chunk() -> int:
    """The member chunk, overridable for a card with more or less memory."""

    import os

    return max(1, int(os.environ.get("AF_VALIDATION_MEMBER_CHUNK", VALIDATION_MEMBER_CHUNK)))

PRIMARY_FIELDS = ("surface_speed", "sst", "phihyd_surface")

ACC_FIELDS = ("surface_u", "surface_v", "phihyd_surface", "sst")

BIAS_FIELDS = ("sst", "eta", "phihyd_surface", "streamfunction")

SHORT_SKILL_TOLERANCE = 1.05


class ValidationError(RuntimeError):
    """Raised when the validation protocol is violated."""


def _gather(state: Any, records: np.ndarray, offset: int) -> np.ndarray:
    """Truth states at ``start + offset`` for every ``(experiment, start)`` record."""

    return np.stack(
        [
            np.asarray(state[int(experiment), int(start) + int(offset)], dtype=np.float32)
            for experiment, start in records
        ]
    )


def lead_auc(
    curve: Sequence[float], leads: Sequence[float], window: Sequence[int]
) -> float:
    """Trapezoidal RMSE-AUC of ``curve`` over the leads inside ``window``."""

    lead_array = np.asarray(leads, dtype=np.float64)
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != lead_array.shape:
        raise ValueError("RMSE curve and lead axis must have the same length")
    mask = np.isin(lead_array, np.asarray(window, dtype=np.float64))
    if int(mask.sum()) < 2:
        raise ValueError("an AUC window needs at least two leads")
    integrate = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed it
    return float(integrate(values[mask], lead_array[mask]))


def _evaluation_fields(states: np.ndarray, wet: np.ndarray) -> dict[str, np.ndarray]:
    fields = derived_fields(states, wet)
    fields["surface_u"] = np.asarray(states[:, 0], dtype=np.float32)
    fields["surface_v"] = np.asarray(states[:, 15], dtype=np.float32)
    fields["eta"] = np.asarray(states[:, 45], dtype=np.float32)
    return fields


def train_only_climatology(
    state: Any,
    wet: np.ndarray,
    *,
    chunk_days: int = 60,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Per-regime pointwise climatology over the training block 0--5999 only."""

    start_index, stop_index = TRAIN_RANGE
    experiments = int(state.shape[0])
    state_sum = np.zeros(
        (experiments, STATE_CHANNEL_COUNT, *wet.shape), dtype=np.float64
    )
    derived_sum = {
        name: np.zeros((experiments, *wet.shape), dtype=np.float64)
        for name in ("surface_speed", "phihyd_surface", "sst", "streamfunction")
    }
    count = 0
    for experiment in range(experiments):
        count = 0
        for begin in range(start_index, stop_index, chunk_days):
            end = min(begin + chunk_days, stop_index)
            raw = np.asarray(state[experiment, begin:end], dtype=np.float32)
            state_sum[experiment] += raw.sum(axis=0, dtype=np.float64)
            fields = derived_fields(raw, wet)
            for name in derived_sum:
                derived_sum[name][experiment] += fields[name].sum(axis=0, dtype=np.float64)
            count += int(raw.shape[0])
    if count != stop_index - start_index:
        raise ValidationError("the train-only climatology day count changed")
    state_mean = (state_sum / count).astype(np.float32)
    state_mean[:, :, ~wet] = 0.0
    derived_mean = {}
    for name, value in derived_sum.items():
        mean = (value / count).astype(np.float32)
        mean[:, ~wet] = 0.0
        derived_mean[name] = mean
    return state_mean, derived_mean, int(count)


def validate_checkpoint(
    stepper: ProductionStepper,
    state: Any,
    static: Any,
    records: np.ndarray,
    climatology_state: np.ndarray,
    climatology_derived: Mapping[str, np.ndarray],
    wet: np.ndarray,
) -> dict[str, Any]:
    """360-day validation rollout metrics for one checkpoint.

    The rollout is chunked over *members*, not leads: each lead's state depends
    on the previous one, so the sequence cannot be split, but the members are
    independent and can be carried a few at a time. On the 1-degree grid all 102
    started together and cost 72 MB; at 248 x 248 with 64 x 64 modes one padded
    spectral buffer for 102 members is 4.35 GiB, which is what exhausted a 32 GB
    card after the training loop had already finished.
    """

    experiments = records[:, 0]
    members, leads = int(records.shape[0]), len(LEAD_DAYS)
    chunk = validation_member_chunk()

    rmse = {
        method: {
            field: np.empty((members, leads), dtype=np.float64)
            for field in PRIMARY_FIELDS
        }
        for method in ("model", "persistence", "climatology")
    }
    acc_leads = [lead for lead in LEAD_DAYS if lead <= ACC_LEAD_LIMIT]
    acc_members = {
        field: np.empty((members, len(acc_leads)), dtype=np.float64)
        for field in ACC_FIELDS
    }
    amplitude = np.zeros(leads, dtype=np.float64)
    bias_sum = {field: 0.0 for field in BIAS_FIELDS}
    bias_count = {field: 0 for field in BIAS_FIELDS}
    wet_tensor = torch.from_numpy(wet).to(stepper.device)

    for lo in range(0, members, chunk):
        hi = min(lo + chunk, members)
        block = records[lo:hi]
        block_experiments = experiments[lo:hi]
        initial = _gather(state, block, 0)
        current = stepper.normalized_state(initial)
        forcing = stepper.normalized_static(static, block_experiments)
        initial_fields = _evaluation_fields(initial, wet)
        climate_state = np.stack([climatology_state[int(e)] for e in block_experiments])
        climate_fields = _evaluation_fields(climate_state, wet)
        for name, value in climatology_derived.items():
            climate_fields[name] = np.stack([value[int(e)] for e in block_experiments])

        with torch.no_grad():
            for index, lead in enumerate(LEAD_DAYS):
                current = stepper.step(current, forcing)
                prediction = stepper.physical(current)
                truth = _gather(state, block, lead)
                predicted = _evaluation_fields(prediction, wet)
                observed = _evaluation_fields(truth, wet)
                for field in PRIMARY_FIELDS:
                    rmse["model"][field][lo:hi, index] = _member_rmse(
                        predicted[field], observed[field], wet
                    )
                    rmse["persistence"][field][lo:hi, index] = _member_rmse(
                        initial_fields[field], observed[field], wet
                    )
                    rmse["climatology"][field][lo:hi, index] = _member_rmse(
                        climate_fields[field], observed[field], wet
                    )
                # A maximum over every member, accumulated across the chunks.
                amplitude[index] = max(
                    amplitude[index],
                    float(torch.amax(torch.abs(current[:, :, wet_tensor])).detach().cpu()),
                )
                if lead <= ACC_LEAD_LIMIT:
                    for field in ACC_FIELDS:
                        acc_members[field][lo:hi, index] = _member_acc(
                            predicted[field],
                            observed[field],
                            climate_fields[field],
                            wet,
                        )
                if lead == LEAD_DAYS[-1]:
                    # A mean over wet cells and members: summed here, divided once
                    # every chunk has contributed.
                    for field in BIAS_FIELDS:
                        error = (predicted[field] - observed[field])[:, wet]
                        bias_sum[field] += float(np.sum(error))
                        bias_count[field] += int(error.size)
        del current, forcing
        if stepper.device.type == "cuda":
            torch.cuda.empty_cache()

    # Means over members, taken once, so the chunking cannot change the answer.
    acc = {
        field: [float(v) for v in acc_members[field].mean(axis=0)] for field in ACC_FIELDS
    }
    bias = {
        field: bias_sum[field] / bias_count[field] if bias_count[field] else float("nan")
        for field in BIAS_FIELDS
    }

    mean_rmse = {
        method: {field: rmse[method][field].mean(axis=0) for field in PRIMARY_FIELDS}
        for method in rmse
    }
    short_auc = {
        f: lead_auc(mean_rmse["model"][f], LEAD_DAYS, SHORT_LEADS) for f in PRIMARY_FIELDS
    }
    long_auc = {
        f: lead_auc(mean_rmse["model"][f], LEAD_DAYS, LONG_LEADS) for f in PRIMARY_FIELDS
    }
    long_clim = {
        f: lead_auc(mean_rmse["climatology"][f], LEAD_DAYS, LONG_LEADS)
        for f in PRIMARY_FIELDS
    }
    short_persistence = {
        f: lead_auc(mean_rmse["persistence"][f], LEAD_DAYS, SHORT_LEADS)
        for f in PRIMARY_FIELDS
    }
    gain = {}
    for field in PRIMARY_FIELDS:
        curve = mean_rmse["model"][field]
        gain[field] = (
            float((curve[-1] / curve[-4]) ** (1.0 / 3.0))
            if curve[-4] > 0
            else float("nan")
        )
    return {
        "lead_days": list(LEAD_DAYS),
        "mean_rmse": {
            m: {f: v.tolist() for f, v in d.items()} for m, d in mean_rmse.items()
        },
        "short_auc_10_90": short_auc,
        "long_auc_90_360": long_auc,
        "long_auc_90_360_climatology": long_clim,
        "long_ratio_to_climatology": {
            f: float(long_auc[f] / long_clim[f]) for f in PRIMARY_FIELDS
        },
        "short_ratio_to_persistence": {
            f: float(short_auc[f] / short_persistence[f]) for f in PRIMARY_FIELDS
        },
        "acc_through_day200": {f: acc[f] for f in ACC_FIELDS},
        "acc_day200": {f: float(acc[f][-1]) for f in ACC_FIELDS},
        "per_call_gain_330_360": gain,
        "maximum_normalized_amplitude": float(np.max(amplitude)),
        "slow_field_bias_day360": bias,
        "arrays": {method: {f: rmse[method][f] for f in PRIMARY_FIELDS} for method in rmse},
    }


def _growth_rate(summary: Mapping[str, Any]) -> float:
    """The worst measured per-call growth rate, or +inf if it was not measured."""

    record = summary.get("perturbation_growth")
    if not isinstance(record, Mapping):
        return float("inf")
    value = record.get("worst_growth_rate_per_call")
    # None means the measurement failed; treat it as maximally unstable so it
    # sorts last and can never be selected as "the least unstable".
    return float("inf") if value is None else float(value)


def select_by_validation(
    summaries: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = SHORT_SKILL_TOLERANCE,
    growth_ceiling: float = GROWTH_RATE_CEILING,
) -> dict[str, Any]:
    """Apply the declared checkpoint-selection rule."""

    if not summaries:
        raise ValidationError("checkpoint selection needs summaries")
    best_short = {
        field: min(float(s["short_auc_10_90"][field]) for s in summaries)
        for field in PRIMARY_FIELDS
    }
    short_feasible = [
        s
        for s in summaries
        if all(
            float(s["short_auc_10_90"][field]) <= tolerance * best_short[field]
            for field in PRIMARY_FIELDS
        )
    ]
    stable = [s for s in short_feasible if _growth_rate(s) <= growth_ceiling]
    feasible = stable
    if feasible:
        selected = min(
            feasible,
            key=lambda s: (
                max(float(v) for v in s["long_ratio_to_climatology"].values()),
                int(s["optimizer_step"]),
            ),
        )
        branch = "primary_rule"
    elif short_feasible:
        # The short guard was met but nothing was stable. Report that plainly
        # and fall back to the least unstable checkpoint rather than silently
        # returning the best-RMSE one, which is exactly the v1 failure.
        #
        # The fallback pool is deliberately the *short-feasible* set, not every
        # checkpoint: the short guard stays a hard constraint and the growth
        # ceiling is the second filter. In the v2 run this mattered --- step
        # 5,760 was the least unstable at 1.0036, but its short AUCs were
        # 1.58x/1.55x/2.57x the best, far outside the 1.05 tolerance, so it was
        # never eligible and step 7,680 was correctly returned.
        selected = min(
            short_feasible,
            key=lambda s: (_growth_rate(s), int(s["optimizer_step"])),
        )
        branch = "declared_fallback_no_checkpoint_met_the_growth_rate_ceiling"
    else:
        selected = min(
            summaries,
            key=lambda s: (
                max(
                    float(s["short_auc_10_90"][f]) / best_short[f] for f in PRIMARY_FIELDS
                ),
                int(s["optimizer_step"]),
            ),
        )
        branch = "declared_fallback_no_checkpoint_met_the_short_guard"
    return {
        "branch": branch,
        "short_skill_tolerance": float(tolerance),
        "growth_rate_ceiling": float(growth_ceiling),
        "best_short_auc_10_90": best_short,
        "short_feasible_steps": [int(s["optimizer_step"]) for s in short_feasible],
        "stable_steps": [int(s["optimizer_step"]) for s in stable],
        "feasible_steps": [int(s["optimizer_step"]) for s in feasible],
        "growth_rate_by_step": {
            str(int(s["optimizer_step"])): _growth_rate(s) for s in summaries
        },
        "selected_optimizer_step": int(selected["optimizer_step"]),
        "selected_growth_rate_per_call": _growth_rate(selected),
        "selected_worst_long_ratio_to_climatology": max(
            float(v) for v in selected["long_ratio_to_climatology"].values()
        ),
    }


def _plot(path: Path, summaries: Sequence[Mapping[str, Any]], selected_step: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(s["optimizer_step"]) for s in summaries]
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    for field in PRIMARY_FIELDS:
        axes[0].plot(
            steps,
            [s["short_ratio_to_persistence"][field] for s in summaries],
            "o-",
            label=field.replace("_", " "),
        )
        axes[1].plot(
            steps,
            [s["long_ratio_to_climatology"][field] for s in summaries],
            "o-",
            label=field.replace("_", " "),
        )
    axes[0].axhline(1.0, color="black", linestyle="--")
    axes[0].set_ylabel("10--90-day RMSE-AUC / persistence")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_ylabel("90--360-day RMSE-AUC / climatology")
    for field in ACC_FIELDS:
        axes[2].plot(
            steps,
            [s["acc_day200"][field] for s in summaries],
            "o-",
            label=field.replace("_", " "),
        )
    axes[2].set_ylabel("Day-200 ACC")
    axes[2].set_ylim(-0.1, 1.0)
    for axis in axes:
        axis.axvline(selected_step, color="#d95f02", alpha=0.7)
        axis.set_xlabel("Optimizer step")
        axis.set_xticks(steps)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Held-validation checkpoint selection, 102 pooled S0/S1/S2 rollouts "
        f"(selected step {selected_step:,})"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)
