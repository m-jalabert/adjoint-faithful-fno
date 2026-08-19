"""Acceptance gate G1: the adjoint gradient matches the finite difference.

Implements section 8 / gate G1 of docs/mitgcm_adjoint_ground_truth_plan.md.

Parses the `grad-res` blocks that pkg/grdchk writes to STDOUT, one archived
file per (test point, epsilon) combination, and reports the ratio table.

The gate is not "one epsilon agreed".  A correct adjoint shows a PLATEAU: the
ratio FD/adjoint sits at 1 across the middle of the epsilon range and degrades
only at the small end where float round-off in the two forward runs takes over.
A single agreeing epsilon is weak evidence; the plateau is strong evidence, and
its width is also the measurement of the linear range that section 12.3 says to
report rather than assert.

The land test point of the plan's eight-point table is deliberately absent:
GRDCHK_GET_POSITION only ever lands on wet cells, so land is checked by gate G4
over all 244 land cells of the extracted map instead.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

#: pkg/grdchk prints these three lines after each check; they are the only
#: numbers that matter.  grdchk_print.F writes them at full double precision.
ADM_PATTERNS = {
    "fc_ref": re.compile(r"ADM\s+ref_cost_function\s*=\s*(\S+)"),
    "adjoint_gradient": re.compile(r"ADM\s+adjoint_gradient\s*=\s*(\S+)"),
    "finite_difference": re.compile(r"ADM\s+finite-diff_grad\s*=\s*(\S+)"),
}

#: "<label>_eps<tag>.stdout", where the tag is the Fortran literal with '.'->'p'
NAME = re.compile(r"^(?P<label>.+)_eps(?P<eps>[0-9]+pd-[0-9]+)\.stdout$")

#: Gate G1 threshold and the epsilons it is asserted at.
TOLERANCE = 1.0e-4
GATE_EPSILONS = (1.0e-4, 1.0e-5)


def fortran_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def epsilon_of_tag(tag: str) -> float:
    return fortran_float(tag.replace("p", "."))


def parse(path: Path) -> dict | None:
    text = path.read_text(errors="replace")
    values: dict[str, float] = {}
    for key, pattern in ADM_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            return None
        values[key] = fortran_float(match.group(1))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--report", default=None)
    arguments = parser.parse_args()

    rows: list[dict] = []
    for path in sorted(Path(arguments.results).glob("*.stdout")):
        match = NAME.match(path.name)
        if match is None:
            continue
        values = parse(path)
        label = match.group("label")
        epsilon = epsilon_of_tag(match.group("eps"))
        if values is None:
            rows.append({"point": label, "epsilon": epsilon, "status": "NO GRAD-RES"})
            continue
        adjoint = values["adjoint_gradient"]
        # A zero adjoint makes the ratio meaningless, and it is exactly the
        # failure mode the ALLOW_ETAN0_CONTROL bug produced, so name it.
        ratio = values["finite_difference"] / adjoint if adjoint != 0.0 else float("nan")
        rows.append(
            {
                "point": label,
                "epsilon": epsilon,
                "status": "zero adjoint" if adjoint == 0.0 else "ok",
                "fc_ref": values["fc_ref"],
                "adjoint_gradient": adjoint,
                "finite_difference": values["finite_difference"],
                "fd_over_adjoint": ratio,
                "abs_error": abs(ratio - 1.0),
            }
        )

    points = sorted({row["point"] for row in rows})
    epsilons = sorted({row["epsilon"] for row in rows}, reverse=True)

    print("Gate G1  --  FD / adjoint")
    header = f"{'test point':<24}" + "".join(f"{e:>14.0e}" for e in epsilons)
    print(header)
    print("-" * len(header))
    for point in points:
        cells = []
        for epsilon in epsilons:
            row = next(
                (r for r in rows if r["point"] == point and r["epsilon"] == epsilon), None
            )
            if row is None:
                cells.append(f"{'-':>14}")
            elif row["status"] != "ok":
                cells.append(f"{row['status']:>14}")
            else:
                cells.append(f"{row['fd_over_adjoint']:>14.9f}")
        print(f"{point:<24}" + "".join(cells))

    # An empty sweep must never report PASS: a gate that passes when it checked
    # nothing is worse than no gate at all.
    failures: list[str] = [] if points else ["no test points found; the sweep produced nothing"]
    for point in points:
        for epsilon in GATE_EPSILONS:
            row = next(
                (r for r in rows if r["point"] == point and abs(r["epsilon"] - epsilon) < 1e-18),
                None,
            )
            if row is None:
                failures.append(f"{point} @ eps={epsilon:g}: missing")
            elif row["status"] != "ok":
                failures.append(f"{point} @ eps={epsilon:g}: {row['status']}")
            elif not (row["abs_error"] < TOLERANCE):
                failures.append(
                    f"{point} @ eps={epsilon:g}: |FD/adj - 1| = {row['abs_error']:.3e}"
                )

    print()
    print(f"  points          {len(points)}")
    print(f"  gate epsilons   {', '.join(f'{e:g}' for e in GATE_EPSILONS)}")
    print(f"  tolerance       |FD/adjoint - 1| < {TOLERANCE:g}")
    for failure in failures:
        print(f"  FAIL  {failure}")
    verdict = not failures
    print(f"  GATE G1: {'PASS' if verdict else 'FAIL'}")
    if not verdict:
        print()
        print("  Section 12 lists the causes in order.  First check whether the failing")
        print("  points are convecting columns (ivdc_kappa = 1. is not differentiable,")
        print("  section 12.1); the northern test point exists to make that fast.")

    if arguments.report:
        Path(arguments.report).write_text(
            json.dumps(
                {
                    "gate": "G1",
                    "tolerance": TOLERANCE,
                    "gate_epsilons": list(GATE_EPSILONS),
                    "rows": rows,
                    "failures": failures,
                    "pass": verdict,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"  wrote {arguments.report}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
