"""Second MITgcm twin of the S0 control trajectory, at epsilon = 1e-3.

This is the *same* experiment as :mod:`af_s0_twin`, run at a perturbation
amplitude one thousand times larger:

    x'_100 = x_100 + delta x,   delta x = epsilon * (Uvel, Vvel),   epsilon = 1e-3

Same year-100 S0 pickup, same executable, same forcing, same physics, same
segment plan, same perturbation geometry.  Only the amplitude differs, which is
what makes the pair interpretable: any difference in the outcome is
attributable to the amplitude alone.

The motivation is measurement, not dynamics.  At epsilon = 1e-6 the initial
velocity difference is ~1e-8 m/s, which is a handful of float32 quantisation
steps at the magnitude of the S0 velocities, so once MITgcm damps the
perturbation the daily float32 diagnostics stop resolving it and the difference
curves pin to discrete values.  At epsilon = 1e-3 the initial difference is
~1e-5 m/s RMS, some three to four decades above the float32 step, so the daily
diagnostics can follow the perturbation down through several decades of decay
before quantisation becomes a concern.

Everything below delegates to :mod:`af_s0_twin` through a :class:`TwinSpec`.
Nothing is reimplemented -- in particular the byte-level pickup surgery and its
five verification steps are the identical, already-exercised code path -- so
the two amplitudes cannot silently drift apart in anything but epsilon.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .af_s0_twin import (
    TWIN_ROOT_NAME,
    TwinSpec,
    prepare_segment as _prepare_segment,
    run_segment as _run_segment,
    segment_plan as _segment_plan,
)


EXPERIMENT = "S0_twin2"
EPSILON = 1.0e-3
TWIN_LABEL = "S0_eps1e-3"

#: Sibling of the epsilon = 1e-6 twin under the same twin root, so the pair sits
#: side by side on scratch and is discovered by the same glob.
SPEC = TwinSpec(
    experiment=EXPERIMENT,
    epsilon=EPSILON,
    root_name=TWIN_ROOT_NAME,
    label=TWIN_LABEL,
)


def segment_plan() -> dict[str, Any]:
    """Return the immutable twin2 segment plan and its iteration boundaries."""

    return _segment_plan(spec=SPEC)


def prepare_segment(
    project_root: Path,
    scratch_root: Path,
    executable: Path,
    start_year: int,
    years: int,
) -> dict[str, Any]:
    """Create one immutable twin2 segment and its provenance manifest."""

    return _prepare_segment(
        project_root, scratch_root, executable, start_year, years, spec=SPEC
    )


def run_segment(manifest, launcher=None) -> dict[str, Any]:
    """Run and validate one prepared twin2 segment."""

    return _run_segment(manifest, launcher=launcher)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-segment", "run-segment"):
        segment = commands.add_parser(name)
        segment.add_argument("--project-root", type=Path, required=True)
        segment.add_argument("--scratch-root", type=Path, required=True)
        segment.add_argument("--executable", type=Path, required=True)
        segment.add_argument("--start-year", type=int, required=True)
        segment.add_argument("--years", type=int, required=True)
    commands.add_parser("plan")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        result: Any = segment_plan()
    else:
        manifest = prepare_segment(
            args.project_root.resolve(),
            args.scratch_root.resolve(),
            args.executable.resolve(),
            args.start_year,
            args.years,
        )
        result = manifest if args.command == "prepare-segment" else run_segment(manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
