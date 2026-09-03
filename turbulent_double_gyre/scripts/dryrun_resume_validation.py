"""Exercise the whole resume path in minutes, with the slow parts stubbed.

The resume path's expensive stages -- a climatology scan over 204 GB and four
360-day validation rollouts -- take about an hour, and everything structural
happens *after* them: the selection, the report, the README, the manifest and
the publish. Two runs died there on a one-line mistake an hour in, which is a
bad way to find a ``KeyError``.

So this runs the real ``run(..., resume_validation=True)`` against a throwaway
output tree whose checkpoints are symlinks to the real ones, with exactly three
functions replaced by correctly-shaped stubs: the climatology scan, the
per-checkpoint validation and the growth measurement. Everything else is the
production code path -- contract audit, normalizer read-back, static block,
model build, checkpoint loading, selection, report, publish.

It proves the path runs to completion. It proves nothing about the numbers, and
the tree it writes is deleted at the end.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "turbulent_double_gyre"
sys.path.insert(0, str(STUDY / "src"))

from turbfno import train as T  # noqa: E402
from turbfno.perturbation_growth import GROWTH_RATE_CEILING  # noqa: E402

REAL_SCRATCH = Path(
    "/bigscratch/mjalabert314/bire_james25_repro/af_fno/models/turb"
) / f"{T.VERSION}.tmp"
DRY = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno/models/turb_dryrun")


def _fake_validation(*args, **kwargs) -> dict:
    leads = list(T.LEAD_DAYS) if hasattr(T, "LEAD_DAYS") else list(range(10, 361, 10))
    from turbfno.validation import ACC_FIELDS, LEAD_DAYS, PRIMARY_FIELDS

    leads = list(LEAD_DAYS)
    members = 102
    rng = np.random.default_rng(0)
    curve = {f: np.linspace(0.1, 0.9, len(leads)) for f in PRIMARY_FIELDS}
    return {
        "lead_days": leads,
        "mean_rmse": {
            m: {f: curve[f].tolist() for f in PRIMARY_FIELDS}
            for m in ("model", "persistence", "climatology")
        },
        "short_auc_10_90": {f: 0.5 for f in PRIMARY_FIELDS},
        "long_auc_90_360": {f: 0.6 for f in PRIMARY_FIELDS},
        "long_auc_90_360_climatology": {f: 1.0 for f in PRIMARY_FIELDS},
        "long_ratio_to_climatology": {f: 0.6 for f in PRIMARY_FIELDS},
        "short_ratio_to_persistence": {f: 0.4 for f in PRIMARY_FIELDS},
        "acc_through_day200": {f: [0.9] * 20 for f in ACC_FIELDS},
        "acc_day200": {f: 0.9 for f in ACC_FIELDS},
        "per_call_gain_330_360": {f: 1.0 for f in PRIMARY_FIELDS},
        "maximum_normalized_amplitude": 3.0,
        "slow_field_bias_day360": {
            f: 0.0 for f in ("sst", "eta", "phihyd_surface", "streamfunction")
        },
        "arrays": {
            m: {
                f: rng.random((members, len(leads))).astype(np.float32)
                for f in PRIMARY_FIELDS
            }
            for m in ("model", "persistence", "climatology")
        },
    }


def _fake_growth(*args, **kwargs) -> dict:
    return {
        "growth_rate_per_call": 1.02,
        "worst_growth_rate_per_call": 1.02,
        "measurement_failed_on_a_start": False,
        "per_start": [{"growth_rate_per_call": 1.02, "finite": True}],
        "starts": 2,
        "ceiling": GROWTH_RATE_CEILING,
        "at_or_below_ceiling": False,
        "definition": "stubbed for the dry run",
    }


def _fake_climatology(state, wet, **kwargs):
    experiments = int(state.shape[0])
    from turbfno.runtime import STATE_CHANNEL_COUNT

    climate = np.zeros((experiments, STATE_CHANNEL_COUNT, *wet.shape), dtype=np.float32)
    derived = {
        name: np.zeros((experiments, *wet.shape), dtype=np.float32)
        for name in ("surface_speed", "phihyd_surface", "sst", "streamfunction")
    }
    return climate, derived, 6000


def build_tree() -> Path:
    """A throwaway scratch tree whose heavy files are symlinks to the real ones."""

    if DRY.exists():
        shutil.rmtree(DRY)
    tmp = DRY / f"{T.VERSION}.tmp"
    (tmp / T.CHECKPOINT_DIRECTORY).mkdir(parents=True)
    (tmp / T.NORMALIZATION_NAME).symlink_to(REAL_SCRATCH / T.NORMALIZATION_NAME)
    for step in T.CHECKPOINT_STEPS:
        name = f"{T.CHECKPOINT_STEM}_{step:05d}.pt"
        (tmp / T.CHECKPOINT_DIRECTORY / name).symlink_to(
            REAL_SCRATCH / T.CHECKPOINT_DIRECTORY / name
        )
    return tmp


def build_contract() -> Path:
    """The real contract with its output roots pointed at the throwaway tree."""

    source = STUDY / "config" / f"{T.VERSION}.json"
    contract = json.loads(source.read_text())
    contract["output"]["scratch_root"] = str(DRY / T.VERSION)
    contract["output"]["project_root"] = str(DRY / "project" / T.VERSION)
    # It has to live beside the real one: load_contract resolves the pinned
    # source hashes against the contract's own parents[1], which is the study
    # root only when the file sits in config/.
    target = STUDY / "config" / f"{T.VERSION}.dryrun.json"
    target.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return target


def main() -> int:
    build_tree()
    contract = build_contract()

    T.train_only_climatology = _fake_climatology
    T.validate_checkpoint = _fake_validation
    T.growth_rate_summary = _fake_growth

    print("running the real resume path with three stages stubbed ...", flush=True)
    report = T.run(contract, device_name="cuda", resume_validation=True)

    published = Path(report["published_checkpoint"]["checkpoint"]).parent
    project = DRY / "project" / T.VERSION
    missing = [
        name for name in T.OUTPUT_ARTIFACTS if not (project / name).is_file()
    ]
    print(f"  status                {report['status']}")
    print(f"  selected step         {report['published_checkpoint']['optimizer_step']}")
    print(f"  resumed block present {report.get('resumed') is not None}")
    print(f"  training history      {len(report['training_history'])} records")
    print(f"  artifacts             {'all present' if not missing else missing}")
    ok = report["status"] == "complete" and not missing
    shutil.rmtree(DRY, ignore_errors=True)
    (STUDY / "config" / f"{T.VERSION}.dryrun.json").unlink(missing_ok=True)
    print("RESUME DRY RUN:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
