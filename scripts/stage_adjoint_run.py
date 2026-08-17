"""Stage a MITgcm run directory for the adjoint ground-truth study.

Implements stages 6 to 8 of docs/mitgcm_adjoint_ground_truth_plan.md.  Four
modes, all sharing one physics renderer so the adjoint can never drift from the
forward trajectory it is supposed to linearise about:

    pickup   forward exe, day 7200 -> 7220 (1440 steps).  Produces the day-7210
             pickup that Run A starts from, the day-7220 pickup, and daily
             snapshots for gate G0.
    grdchk   adjoint exe, day 7210 -> 7220, useGrdchk=.TRUE.   (gate G1)
    runA     adjoint exe, day 7210 -> 7220, the primary 10-day map
    runB     adjoint exe, day 7200 -> 7220, the 20-day map and lead sweep

Physics in PARM01/02/04/05 is copied verbatim from
archive/src/bire_repro/af_s0.py::render_data, which rendered every S0 forward
segment.  Only PARM03 differs between modes.

Nothing here reads or writes the FNO tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from select_adjoint_target import BASE_ITERATION, CONTRACT_VERSION, STEPS_PER_DAY

#: Reference S0 run directory: forcing files, grid metrics, and the day-7200 pickup.
DEFAULT_SOURCE = Path(
    "/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_long_truth_v1/S0/production/years_120_126"
)
DEFAULT_SCRATCH = Path("/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v1")

#: ctrl_readparms.F sets ctrlprec = 64 unless CTRL_SET_PREC_32; the control and
#: the returned gradient are therefore big-endian float64.
CTRL_DTYPE = ">f8"

FORCING_FILES = ("bathy.bin", "windx_cosy.bin", "SST_relax.bin")

MODES: dict[str, dict[str, Any]] = {
    "pickup": {
        "start_day": 7200,
        "days": 20,
        "adjoint": False,
        "grdchk": False,
        # 864000 s = 10 days: pickups land at day 7210 and day 7220
        "pchkpt_freq": 864000.0,
        "adj_dump_freq": None,
    },
    "grdchk": {
        "start_day": 7210,
        "days": 10,
        "adjoint": True,
        "grdchk": True,
        "pchkpt_freq": 0.0,
        "adj_dump_freq": None,
    },
    "runA": {
        "start_day": 7210,
        "days": 10,
        "adjoint": True,
        "grdchk": False,
        "pchkpt_freq": 0.0,
        "adj_dump_freq": 86400.0,
    },
    "runB": {
        "start_day": 7200,
        "days": 20,
        "adjoint": True,
        "grdchk": False,
        "pchkpt_freq": 0.0,
        "adj_dump_freq": 86400.0,
    },
}


class StagingError(RuntimeError):
    """Raised when a run directory cannot be staged consistently."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iteration_of_day(day: int) -> int:
    return BASE_ITERATION + STEPS_PER_DAY * day


def render_data(
    n_iter0: int,
    n_time_steps: int,
    pchkpt_freq: float,
    adj_dump_freq: float | None,
) -> str:
    """The official S0 tutorial physics, with PARM03 set for this segment.

    PARM01/02/04/05 are byte-identical to af_s0.render_data.  Changing anything
    outside PARM03 would mean differentiating a model the FNO was not trained
    on.
    """

    if n_iter0 < 0 or n_time_steps <= 0:
        raise StagingError("n_iter0 must be nonnegative and n_time_steps positive")
    adj_line = "" if adj_dump_freq is None else f" adjDumpFreq={adj_dump_freq!r},\n"
    return f"""# AF--FNO S0 adjoint: official 1-degree baroclinic-gyre tutorial physics
 &PARM01
 viscAh=5000.,
 viscAr=1.E-2,
 no_slip_sides=.TRUE.,
 no_slip_bottom=.FALSE.,
 diffKhT=1000.,
 diffKrT=1.E-5,
 ivdc_kappa=1.,
 implicitDiffusion=.TRUE.,
 eosType='LINEAR',
 tRef=30.,27.,24.,21.,18.,15.,13.,11.,9.,7.,6.,5.,4.,3.,2.,
 tAlpha=2.E-4,
 sBeta=0.,
 rhoNil=999.8,
 gravity=9.81,
 rigidLid=.FALSE.,
 implicitFreeSurface=.TRUE.,
 exactConserv=.TRUE.,
 saltStepping=.FALSE.,
 useSingleCpuIO=.TRUE.,
 &
 &PARM02
 cg2dTargetResidual=1.E-7,
 cg2dMaxIters=1000,
 &
 &PARM03
 nIter0={n_iter0},
 nTimeSteps={n_time_steps},
 deltaT=1200.,
 pChkptFreq={pchkpt_freq!r},
 chkptFreq=0.,
 dumpFreq=0.,
 monitorFreq=2592000.,
 monitorSelect=2,
 tauThetaClimRelax=2592000.,
{adj_line} &
 &PARM04
 usingSphericalPolarGrid=.TRUE.,
 delX=62*1.,
 delY=62*1.,
 xgOrigin=-1.,
 ygOrigin=14.,
 delR=50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,
 &
 &PARM05
 bathyFile='bathy.bin',
 zonalWindFile='windx_cosy.bin',
 thetaClimFile='SST_relax.bin',
 &
"""


def render_data_pkg(adjoint: bool, grdchk: bool) -> str:
    lines = [" &PACKAGES", " useMNC=.FALSE.,", " useDiagnostics=.TRUE.,"]
    if adjoint:
        lines.append(f" useGrdchk={'.TRUE.' if grdchk else '.FALSE.'},")
    lines.extend([" &", ""])
    return "\n".join(lines)


def stage(
    mode: str,
    project_root: Path,
    scratch_root: Path,
    source_dir: Path,
    executable: Path,
    weight_file: Path,
    force: bool,
) -> Path:
    if mode not in MODES:
        raise StagingError(f"unknown mode {mode!r}, expected one of {sorted(MODES)}")
    settings = MODES[mode]

    contract_path = project_root / "config" / f"{CONTRACT_VERSION}.json"
    contract = json.loads(contract_path.read_text())

    start_day = int(settings["start_day"])
    n_iter0 = iteration_of_day(start_day)
    n_time_steps = int(settings["days"]) * STEPS_PER_DAY
    end_day = start_day + int(settings["days"])

    run_dir = scratch_root / mode
    if run_dir.exists():
        if not force:
            raise StagingError(f"{run_dir} exists; pass --force to replace it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    input_ad = project_root / "af_fno" / "mitgcm" / "input_ad"
    forward_input = project_root / "af_fno" / "mitgcm" / "input"

    # --- namelists -------------------------------------------------------
    (run_dir / "data").write_text(
        render_data(
            n_iter0,
            n_time_steps,
            float(settings["pchkpt_freq"]),
            settings["adj_dump_freq"],
        )
    )
    (run_dir / "data.pkg").write_text(
        render_data_pkg(bool(settings["adjoint"]), bool(settings["grdchk"]))
    )
    shutil.copy2(forward_input / "eedata", run_dir / "eedata")
    shutil.copy2(forward_input / "data.diagnostics.production", run_dir / "data.diagnostics")

    if settings["adjoint"]:
        for name in ("data.cost", "data.ctrl", "data.autodiff", "data.optim"):
            shutil.copy2(input_ad / name, run_dir / name)
        if settings["grdchk"]:
            shutil.copy2(input_ad / "data.grdchk", run_dir / "data.grdchk")

    # --- forcing and bathymetry -----------------------------------------
    for name in FORCING_FILES:
        source = source_dir / name
        if not source.is_file():
            raise StagingError(f"missing forcing file {source}")
        shutil.copy2(source, run_dir / name)

    # --- initial condition ----------------------------------------------
    for suffix in ("data", "meta"):
        source = source_dir / f"pickup.{n_iter0:010d}.{suffix}"
        if not source.is_file():
            raise StagingError(
                f"missing {source}.  For mode {mode!r} the day-{start_day} pickup must exist; "
                "run mode 'pickup' first if this is day 7210."
            )
        shutil.copy2(source.resolve(), run_dir / source.name)

    # --- adjoint-only inputs ---------------------------------------------
    staged_weight: Path | None = None
    if settings["adjoint"]:
        if not weight_file.is_file():
            raise StagingError(f"missing weight field {weight_file}; run scripts/make_cost_weight.py")
        staged_weight = run_dir / "costWeight.bin"
        shutil.copy2(weight_file, staged_weight)

        # ctrl_map_ini.F reads xx_etan.<optimcycle> with ladinit=.FALSE., so the
        # control file must exist.  A pure sensitivity run needs it identically
        # zero: the gradient is taken about the unperturbed trajectory.
        nx, ny = int(contract["grid"]["nx"]), int(contract["grid"]["ny"])
        zeros = np.zeros((ny, nx), dtype=CTRL_DTYPE)
        zeros.tofile(run_dir / "xx_etan.0000000000.data")

    (run_dir / executable.name).symlink_to(executable)

    manifest = {
        "version": "mitgcm_adjoint_s0_run_v1",
        "mode": mode,
        "run_dir": str(run_dir),
        "start_day": start_day,
        "end_day": end_day,
        "n_iter0": n_iter0,
        "n_time_steps": n_time_steps,
        "end_iteration": n_iter0 + n_time_steps,
        "adjoint": bool(settings["adjoint"]),
        "grdchk": bool(settings["grdchk"]),
        "adj_dump_freq": settings["adj_dump_freq"],
        "executable": str(executable),
        "executable_sha256": _sha256(executable),
        "source_dir": str(source_dir),
        "target_contract": str(contract_path),
        "target_i_global": contract["i_global"],
        "target_j_global": contract["j_global"],
        "weight_file": str(weight_file) if staged_weight else None,
        "weight_sha256": _sha256(staged_weight) if staged_weight else None,
        "checksums": {
            name: _sha256(run_dir / name)
            for name in sorted(p.name for p in run_dir.iterdir() if p.is_file())
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"staged {mode} -> {run_dir}")
    print(f"  days       {start_day} -> {end_day}   ({n_time_steps} steps)")
    print(f"  nIter0     {n_iter0}   end {n_iter0 + n_time_steps}")
    print(f"  executable {executable}")
    if staged_weight:
        print(f"  weight     {weight_file.name} -> costWeight.bin")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--scratch-root", default=str(DEFAULT_SCRATCH))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--executable", default=None)
    parser.add_argument("--weight-file", default=None)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    project_root = (
        Path(arguments.project_root) if arguments.project_root else Path(__file__).resolve().parent.parent
    )
    if arguments.executable:
        executable = Path(arguments.executable)
    elif MODES[arguments.mode]["adjoint"]:
        executable = project_root / "build" / "af_s0_ad" / "mitgcmuv_ad"
    else:
        executable = project_root / "build" / "af_s0" / "mitgcmuv"
    if not executable.is_file():
        raise StagingError(
            f"executable {executable} not found."
            + (
                "  Run 'make adall' in build/af_s0_ad -- that step needs the TAF licence."
                if MODES[arguments.mode]["adjoint"]
                else ""
            )
        )

    weight_file = (
        Path(arguments.weight_file)
        if arguments.weight_file
        else project_root / "work" / "costWeight_ssh_anomaly.bin"
    )

    stage(
        arguments.mode,
        project_root,
        Path(arguments.scratch_root),
        Path(arguments.source_dir),
        executable,
        weight_file,
        arguments.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
