"""Generate the 0.25-degree bathymetry, wind and SST-restoring fields.

This is the tutorial ``gendata.m`` of ``tutorial_baroclinic_gyre`` rewritten so
that the grid spacing is a parameter.  The physical domain is unchanged: a
60 deg x 60 deg ocean spanning (0E,15N)-(60E,75N), flat bottom at 1800 m,
surrounded by a one-degree land rim.  At 1 deg that rim is one cell and the grid
is 62 x 62; at 0.25 deg it is four cells and the grid is 248 x 248, which is the
resolution locked in ``config/bire_a0_reference.toml``.

``--check`` regenerates the 1-degree fields and compares them byte for byte with
the tutorial binaries, so the generator is validated on the grid whose answer is
already known before it is trusted on the grid whose answer is not.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

OCEAN_DEPTH_M = 1800.0
OCEAN_X0_DEG = 0.0  # south-western corner of the ocean domain
OCEAN_Y0_DEG = 15.0
OCEAN_EXTENT_DEG = 60.0
RIM_DEG = 1.0  # land rim width, one degree on every side
T_MAX_C = 30.0  # restoring temperature at the southern edge of the ocean
T_MIN_C = 0.0  # ... and at the northern edge


def grid(spacing_deg: float) -> tuple[int, float, float]:
    """Cell count and grid origin for one horizontal spacing."""

    cells = round((OCEAN_EXTENT_DEG + 2.0 * RIM_DEG) / spacing_deg)
    if not np.isclose(cells * spacing_deg, OCEAN_EXTENT_DEG + 2.0 * RIM_DEG):
        raise ValueError(f"{spacing_deg} does not tile the 62-degree domain")
    return cells, OCEAN_X0_DEG - RIM_DEG, OCEAN_Y0_DEG - RIM_DEG


def fields(spacing_deg: float, tau0_n_m2: float) -> dict[str, np.ndarray]:
    """Bathymetry, zonal wind stress and restoring temperature on one grid.

    Arrays are indexed ``[y, x]`` so that a C-order write puts x fastest, which
    is the layout MITgcm expects from an MDS input file.
    """

    n, xg_origin, yg_origin = grid(spacing_deg)
    rim = round(RIM_DEG / spacing_deg)

    depth = np.full((n, n), -OCEAN_DEPTH_M)
    depth[:rim, :] = 0.0
    depth[-rim:, :] = 0.0
    depth[:, :rim] = 0.0
    depth[:, -rim:] = 0.0

    # Wind stress and restoring temperature are functions of latitude only and
    # are evaluated at cell centres, exactly as the tutorial generator does.
    y_centre = yg_origin + (np.arange(n) + 0.5) * spacing_deg
    tau = -tau0_n_m2 * np.cos(
        2.0 * np.pi * (y_centre - OCEAN_Y0_DEG) / OCEAN_EXTENT_DEG
    )
    y_north = OCEAN_Y0_DEG + OCEAN_EXTENT_DEG
    restore = (T_MAX_C - T_MIN_C) / OCEAN_EXTENT_DEG * (y_north - y_centre) + T_MIN_C

    broadcast = np.ones((1, n))
    return {
        "bathy.bin": depth,
        "windx_cosy.bin": tau[:, None] * broadcast,
        "SST_relax.bin": restore[:, None] * broadcast,
    }


def write(destination: Path, spacing_deg: float, tau0_n_m2: float) -> dict[str, str]:
    """Write the three MDS input files as big-endian float32 and hash them."""

    destination.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, array in fields(spacing_deg, tau0_n_m2).items():
        payload = np.asarray(array, dtype=">f4").tobytes(order="C")
        (destination / name).write_bytes(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()
    return digests


def check(tutorial_input: Path) -> None:
    """Reproduce the 1-degree tutorial binaries exactly, or fail loudly."""

    generated = fields(1.0, 0.1)
    for name, array in generated.items():
        reference = np.fromfile(tutorial_input / name, dtype=">f4")
        mine = np.asarray(array, dtype=">f4").reshape(-1)
        if reference.shape != mine.shape:
            raise SystemExit(f"{name}: shape {mine.shape} != tutorial {reference.shape}")
        if not np.array_equal(reference, mine):
            worst = int(np.argmax(np.abs(reference - mine)))
            raise SystemExit(
                f"{name}: differs from the tutorial file, worst element {worst} "
                f"({mine[worst]} vs {reference[worst]})"
            )
        print(f"{name}: byte-identical to the tutorial 1-degree file ({mine.size} values)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="tutorial input directory to validate against")
    parser.add_argument("--out", type=Path, help="directory to write the generated fields into")
    parser.add_argument("--spacing-deg", type=float, default=0.25)
    parser.add_argument("--tau0", type=float, default=0.1)
    args = parser.parse_args(argv)

    if args.check:
        check(args.check)
    if args.out:
        n, xg_origin, yg_origin = grid(args.spacing_deg)
        digests = write(args.out, args.spacing_deg, args.tau0)
        print(f"grid {n} x {n} at {args.spacing_deg} deg, origin ({xg_origin}, {yg_origin})")
        for name, digest in digests.items():
            print(f"  {name} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
