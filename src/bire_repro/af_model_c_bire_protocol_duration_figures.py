"""S0 Bire-style Figure 3--8 suite for the matched-budget duration arm.

Publishes the six figures for the 15,360-step checkpoint of
:mod:`af_model_c_bire_protocol_duration` on S0's inference set, reusing the
frozen plotting functions so the figure definitions, axes, reductions, and
filenames stay identical to every earlier package.

Everything about the evaluation is unchanged from the parent figure suite: the
same 15 members drawn from 6200--6999 by the same seed, the same lead grid out
to 2,000 days, the same lead-matched truth from days 7200--8999, the same
train-only climatology and persistence baselines.  Only the checkpoint differs,
so the two packages are directly comparable field for field.

Figure 6's black comparator is **this run's own step-7,680 checkpoint** against
the selected step-15,360 one.  That is the most informative within-run pairing
available because 7,680 is the parent arm's whole budget, so the black-versus-red
gap is what doubling the budget bought.  It is *not* the parent's checkpoint:
holding ``decay_fraction`` at 0.75 moves this run's decay to step 11,520, so at
step 7,680 this run is still at 5e-4 where the parent had already dropped to
1e-4.  The captions say training-progress comparison for that reason.

Held-evaluation package: no training, no checkpoint selection, no promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import af_model_c_bire_protocol_figures as suite
from .af_model_c_overfit import _file_sha256

VERSION = "model_c_bire_protocol_duration_s0_figures_v1"
CONTRACT_STATUS = (
    "frozen_after_the_bire_protocol_duration_training_and_validation"
    "_and_before_any_inference_metric"
)

MEMBER_COUNT = suite.MEMBER_COUNT
START_SEED = suite.START_SEED
#: The parent arm's entire budget, used here as this run's own comparator.
COMPARATOR_STEP = 7680
SELECTED_STEP = 15360
REGIMES = suite.REGIMES


class BireProtocolDurationFigureError(RuntimeError):
    """Raised when the duration figure contract is violated."""


def declared_inference_starts():
    """The parent suite's 15 members, unchanged, so the packages compare."""

    return suite.declared_inference_starts()


def load_contract(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], Path, str]:
    """Load the duration figure contract frozen before any inference metric."""

    resolved = Path(path).resolve()
    contract = json.loads(resolved.read_text())
    protocol = contract.get("protocol", {})
    figure6 = contract.get("figure6", {})
    if (
        contract.get("version") != VERSION
        or contract.get("contract_status") != CONTRACT_STATUS
        or int(protocol.get("member_count", -1)) != MEMBER_COUNT
        or tuple(protocol.get("start_draw_order", ()))
        != tuple(int(v) for v in declared_inference_starts())
        or tuple(protocol.get("regimes", ())) != REGIMES
        or protocol.get("primary_regime") != "S0"
        or int(figure6.get("comparator_optimizer_step", -1)) != COMPARATOR_STEP
        or figure6.get("literal_pretrain_finetune_pair") is not False
        or int(contract.get("selected_model", {}).get("optimizer_step", -1)) != SELECTED_STEP
    ):
        raise BireProtocolDurationFigureError("duration figure contract changed")
    if verify_sources:
        for label, specification in contract["artifacts"].items():
            suite.figures._verify_file(specification, label)
        root = resolved.parents[1]
        for relative, expected in contract["source_hashes"].items():
            source = root / relative
            if not source.is_file() or _file_sha256(source) != expected:
                raise BireProtocolDurationFigureError(f"source changed: {relative}")
    return contract, resolved, _file_sha256(resolved)


def _readme(regime: str, report: Mapping[str, Any]) -> str:
    starts = declared_inference_starts()
    return f"""# Matched-budget duration arm, {regime}: Figures 3--8

This package evaluates the seed-20260724, step-{int(report['selected_optimizer_step']):,}
checkpoint of the pooled matched-budget duration model on the **{regime}**
inference set (indices 6200--7199), tau0 = {report['tau0_n_m2']} N m-2.

The arm is identical to `model_c_bire_protocol_pooled_v1` in every model
quantity except `maximum_steps`, which moves from 7,680 to 15,360. Validation
selected the final step {SELECTED_STEP:,} checkpoint, improving the worst
90--360-day ratio to climatology from 1.081 to 0.932.

**Directly comparable with the parent figure package.** The 15 members, the seed,
the lead grid, the truth window, the climatology and the persistence baseline are
all unchanged, so `outputs/af_fno/C/bire_protocol_s0_figures_v1/S0/` and this
package differ only in the checkpoint. The starts are 6200--6999, this draw
spanning {int(starts.min())}--{int(starts.max())}, and every member has
lead-matched MITgcm truth to day 2,000 ({int(starts.max())} + 2000 =
{int(starts.max()) + 2000} < 9000).

**Figure 6 comparator.** The black curve is this run's own step-{COMPARATOR_STEP:,}
checkpoint against the selected step-{int(report['selected_optimizer_step']):,} one.
Step 7,680 is the parent arm's entire budget, so the gap between the curves is
what doubling the budget bought within a single run. It is not the parent's
checkpoint: this run's decay falls at 11,520, so at step 7,680 it is still at
5e-4 where the parent had already dropped to 1e-4.

Climatology is the pointwise {regime} mean over the Bire training block
(0--5999) only. Persistence holds each member's initial physical field fixed.
RMSE is computed over wet cells per member; lines and bands are the mean and
10th/90th percentiles across the 15 members.

This is a held-evaluation package. It performs no training, no checkpoint
selection, and promotes nothing.

Report content SHA-256: `{report['report_content_sha256']}`.
"""


PARENT_BINDINGS = ("VERSION", "CONTRACT_STATUS", "COMPARATOR_STEP", "load_contract", "_readme")


class _SuiteBinding:
    """Bind this arm's loader, version, comparator and README into the suite."""

    def __enter__(self) -> "_SuiteBinding":
        here = globals()
        self._saved = {name: getattr(suite, name) for name in PARENT_BINDINGS}
        for name in PARENT_BINDINGS:
            setattr(suite, name, here[name])
        return self

    def __exit__(self, *exc: Any) -> None:
        for name, value in self._saved.items():
            setattr(suite, name, value)


def preflight(contract_path: str | Path) -> dict[str, Any]:
    with _SuiteBinding():
        result = dict(suite.preflight(contract_path))
    result["selected_optimizer_step"] = SELECTED_STEP
    result["comparator_optimizer_step"] = COMPARATOR_STEP
    return result


def run(contract_path: str | Path, *, device_name: str = "auto") -> dict[str, Any]:
    with _SuiteBinding():
        return dict(suite.run(contract_path, device_name=device_name))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("preflight")
    check.add_argument("--contract", required=True)
    plot = subparsers.add_parser("run")
    plot.add_argument("--contract", required=True)
    plot.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = preflight(args.contract) if args.command == "preflight" else run(
        args.contract, device_name=args.device
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
