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

Phase A (docs/Adjoint_study_Phase_A.md) adds eleven more, all reading their
window and weight field from config/adjoint_phase_a_v1.json rather than from a
hard-coded table:

    P10/P20/P30/P90   the point objective, source day 7200, cost at 7210 /
                      7220 / 7230 / 7290.  P90 also yields the whole
                      backward sweep from its 91 ADJetan dumps
    K10/K20/K30/K90   the same four with the smooth-kernel weight field
    C90               the mean-only conservation probe over 91 dumps, gate G3
    V10               day 7280 -> 7290, gate G2b: a fresh ten-day run must
                      return the same adjoint state a ninety-day tape does
    G90               grdchk at the ninety-day window, gate G1-90

**No TAF resubmission is involved.**  code_ad/tamc.h is sized
72 x 200 x 1 = 14,400 >= 6,480 steps, so ninety days fits the existing tape and
the v1 executable is reused byte-for-byte.  Nothing in the differentiated
source changes.

Physics in PARM01/02/04/05 is copied verbatim from
archive/src/bire_repro/af_s0.py::render_data, which rendered every S0 forward
segment.  Only PARM03 differs between modes.

Nothing here reads or writes the FNO tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
        # data.diagnostics sets dumpAtLast=.FALSE., so the state at the final
        # iteration is never written.  The archived segments only hold a day-7220
        # snapshot because 7220 is interior to a six-year production block.  To
        # reproduce that snapshot by the same mechanism the re-run has to step
        # one day past it; otherwise gate G0 reports day 7220 MISSING.
        "tail_days": 1,
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
    # Gate G3.  Identical to runA in every respect except the weight field,
    # which must be staged as work/costWeight_mean_only.bin.  Because the area
    # integral of eta is exactly conserved by this configuration, the adjoint
    # of the mean functional is provably constant in time, so the returned map
    # must equal the weight field itself at every dump time.  No finite
    # differences, sharp, and available at every lead.
    "runG3": {
        "start_day": 7210,
        "days": 10,
        "adjoint": True,
        "grdchk": False,
        "pchkpt_freq": 0.0,
        "adj_dump_freq": 86400.0,
    },
}


#: Phase A writes to its own scratch root so the validated v1 products under
#: mitgcm_adjoint_v1/ are never touched.
DEFAULT_SCRATCH_PHASE_A = Path(
    "/bigscratch/mjalabert314/bire_james25_repro/af_fno/mitgcm_adjoint_v2"
)

PHASE_A_CONTRACT = "adjoint_phase_a_v1"

#: Objective name -> the weight file scripts/make_cost_weight.py writes.
WEIGHT_FILES = {
    "ssh_anomaly": "costWeight_ssh_anomaly.bin",
    "ssh_anomaly_kernel": "costWeight_ssh_anomaly_kernel.bin",
    "mean_only": "costWeight_mean_only.bin",
}


class StagingError(RuntimeError):
    """Raised when a run directory cannot be staged consistently."""


def phase_a_modes(project_root: Path) -> dict[str, dict[str, Any]]:
    """Build the Phase A mode table from the frozen contract.

    Read rather than hard-coded, so the runs and the plan cannot drift: the
    contract's ``mitgcm_runs`` block is the single declaration of which windows
    exist, and it is the same block the extractor and the comparison read.
    """

    path = project_root / "config" / f"{PHASE_A_CONTRACT}.json"
    if not path.is_file():
        return {}
    contract = json.loads(path.read_text())
    steps_per_day = int(contract["window"]["steps_per_day"])
    modes: dict[str, dict[str, Any]] = {}

    # Forward runs first.  F90 is not optional bookkeeping: V10 starts at day
    # 7280, which no archive segment holds a pickup for, and data.diagnostics
    # sets dumpAtLast=.FALSE. so no adjoint run ever dumps day 7290 either.
    # One forward pass supplies both, plus the daily snapshots gate G0 needs
    # across the whole ninety-day window.
    for run in contract.get("mitgcm_forward_runs", []):
        modes[run["name"]] = {
            "start_day": (int(run["n_iter0"]) - BASE_ITERATION) // steps_per_day,
            "days": int(run["days"]),
            "tail_days": int(run.get("tail_days", 0)),
            "adjoint": False,
            "grdchk": False,
            "pchkpt_freq": float(run["pchkpt_freq_seconds"]),
            "adj_dump_freq": None,
            "phase_a": True,
        }

    for run in contract["mitgcm_runs"]:
        steps = int(run["n_time_steps"])
        if steps % steps_per_day:
            raise StagingError(f"run {run['name']} is not a whole number of days")
        start_day = (int(run["n_iter0"]) - BASE_ITERATION) // steps_per_day
        if iteration_of_day(start_day) != int(run["n_iter0"]):
            raise StagingError(f"run {run['name']} does not start on a day boundary")
        if int(run["n_iter0"]) + steps != iteration_of_day(int(run["cost_day"])):
            raise StagingError(
                f"run {run['name']} does not end at its declared cost day {run['cost_day']}"
            )
        modes[run["name"]] = {
            "start_day": start_day,
            "days": steps // steps_per_day,
            "adjoint": True,
            "grdchk": bool(run.get("grdchk", False)),
            "pchkpt_freq": 0.0,
            # One dump per model day.  For P90 that is 91 maps -- the complete
            # backward evolution of the day-7290 target's sensitivity, for the
            # price of the run that was happening anyway.
            "adj_dump_freq": 86400.0,
            "weight": run["weight"],
            "cost_day": int(run["cost_day"]),
            "phase_a": True,
        }
    return modes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iteration_of_day(day: int) -> int:
    return BASE_ITERATION + STEPS_PER_DAY * day


def fortran_real(value: float) -> str:
    """Render a real in MITgcm's own namelist style, e.g. 1e-7 -> '1.E-7'.

    Byte-identity with af_s0.render_data is asserted by
    test_staged_physics_is_identical_to_the_forward_renderer, so the default
    must reproduce the forward run's literal exactly rather than merely parse
    to the same number.
    """

    mantissa, _, exponent = f"{value:E}".partition("E")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}.E{int(exponent)}"


def render_data(
    n_iter0: int,
    n_time_steps: int,
    pchkpt_freq: float,
    adj_dump_freq: float | None,
    cg2d_target_residual: float = 1.0e-7,
) -> str:
    """The official S0 tutorial physics, with PARM03 set for this segment.

    PARM01/02/04/05 are byte-identical to af_s0.render_data.  Changing anything
    outside PARM03 would mean differentiating a model the FNO was not trained
    on.
    """

    if n_iter0 < 0 or n_time_steps <= 0:
        raise StagingError("n_iter0 must be nonnegative and n_time_steps positive")
    adj_line = "" if adj_dump_freq is None else f" adjDumpFreq={adj_dump_freq!r},\n"

    # DUMP_ADJ_XY writes ADJetan through WRITE_FLD_XY_RL, which uses
    # writeBinaryPrec; the MITgcm default is 32, which would put a float32
    # quantisation floor (~1e-7 relative) directly under gates G2 and G3 -- the
    # same trap as the s0-twin-float32-floor finding.  adxx_etan is already
    # float64 (it goes through ctrlprec), so without this the two halves of G2
    # are not even the same precision.
    #
    # This is safe for the forward snapshots: data.diagnostics sets
    # fileFlags='R', and diagnostics_out.F:396 lets that flag override
    # writeBinaryPrec, so dynState/surfState stay float32 and stay comparable
    # to the archive.  The adjoint runs write no pickups (pChkptFreq = 0).
    precision_line = " writeBinaryPrec=64,\n" if adj_dump_freq is not None else ""
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
{precision_line} &
 &PARM02
 cg2dTargetResidual={fortran_real(cg2d_target_residual)},
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
    cg2d_target_residual: float = 1.0e-7,
    settings: dict[str, Any] | None = None,
    grdchk_eps: float | None = None,
    grdchk_point: tuple[int, int] | None = None,
    run_name: str | None = None,
) -> Path:
    if settings is None:
        available = {**MODES, **phase_a_modes(project_root)}
        if mode not in available:
            raise StagingError(f"unknown mode {mode!r}, expected one of {sorted(available)}")
        settings = available[mode]

    contract_path = project_root / "config" / f"{CONTRACT_VERSION}.json"
    contract = json.loads(contract_path.read_text())

    start_day = int(settings["start_day"])
    n_iter0 = iteration_of_day(start_day)
    end_day = start_day + int(settings["days"])
    n_time_steps = (int(settings["days"]) + int(settings.get("tail_days", 0))) * STEPS_PER_DAY

    run_dir = scratch_root / (run_name or mode)
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
            cg2d_target_residual,
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
            grdchk = (input_ad / "data.grdchk").read_text()
            if grdchk_point is not None:
                # iGloPos/jGloPos are TILE-LOCAL (grdchk_readparms.F:116 rejects
                # iGloPos > sNx), and with sNx = sNy = 31 on a 2x2 decomposition
                # a global (i,j) maps to tile ((i-1)//31 + 1, (j-1)//31 + 1) at
                # local ((i-1)%31 + 1, (j-1)%31 + 1).
                i_global, j_global = grdchk_point
                i_tile, j_tile = (i_global - 1) // 31 + 1, (j_global - 1) // 31 + 1
                i_local, j_local = (i_global - 1) % 31 + 1, (j_global - 1) % 31 + 1
                for key, value in (
                    ("iGloPos", i_local),
                    ("jGloPos", j_local),
                    ("iGloTile", i_tile),
                    ("jGloTile", j_tile),
                ):
                    # Count the matches rather than comparing the text: a
                    # substitution that writes back the value already there is
                    # still a successful one, and for points whose tile-local
                    # index happens to equal the default it is the common case.
                    grdchk, count = re.subn(
                        rf"^( {key}\s*=\s*)\d+,$",
                        rf"\g<1>{value},",
                        grdchk,
                        flags=re.MULTILINE,
                    )
                    if count != 1:
                        raise StagingError(
                            f"{key} matched {count} times in data.grdchk, expected exactly 1"
                        )
            if grdchk_eps is not None:
                # A ninety-day window's sensitivity magnitude is not known in
                # advance, and the finite difference cannot resolve better than
                # the noise floor of fc divided by 2*eps.  The epsilon is
                # therefore chosen from the measured |S| after P90 rather than
                # frozen here.
                # Fortran wants a D exponent for a double literal.  Build the
                # literal on its own and only then splice it in: applying the
                # e->d substitution to the whole line rewrites the key as
                # "grdchk_dps", which MITgcm would either reject or ignore in
                # favour of the default epsilon -- a silent wrong answer of
                # exactly the kind this study keeps finding.
                literal = f"{grdchk_eps:.3e}".replace("e", "d")
                grdchk, count = re.subn(
                    r"^( grdchk_eps\s*=\s*)\S+,$",
                    rf"\g<1>{literal},",
                    grdchk,
                    flags=re.MULTILINE,
                )
                if count != 1:
                    raise StagingError(
                        f"grdchk_eps matched {count} times in data.grdchk, expected exactly 1"
                    )
            (run_dir / "data.grdchk").write_text(grdchk)

    # --- forcing and bathymetry -----------------------------------------
    for name in FORCING_FILES:
        source = source_dir / name
        if not source.is_file():
            raise StagingError(f"missing forcing file {source}")
        shutil.copy2(source, run_dir / name)

    # --- initial condition ----------------------------------------------
    # Day 7200 is archived with the production segments; day 7210 exists only
    # because the 'pickup' mode run produced it, so search the scratch pickup
    # directory too.  Order matters: the archive is authoritative where it has
    # the file.
    pickup_dirs = (source_dir, scratch_root / "pickup", scratch_root / "F90")
    for suffix in ("data", "meta"):
        name = f"pickup.{n_iter0:010d}.{suffix}"
        source = next((d / name for d in pickup_dirs if (d / name).is_file()), None)
        if source is None:
            searched = ", ".join(str(d) for d in pickup_dirs)
            raise StagingError(
                f"missing {name} in {searched}.  For mode {mode!r} the day-{start_day} "
                "pickup must exist; run mode 'pickup' (v1, day 7210) or 'F90' "
                "(Phase A, days 7210-7290) first."
            )
        shutil.copy2(source.resolve(), run_dir / name)

    # --- adjoint-only inputs ---------------------------------------------
    staged_weight: Path | None = None
    if settings["adjoint"]:
        if not weight_file.is_file():
            raise StagingError(f"missing weight field {weight_file}; run scripts/make_cost_weight.py")
        staged_weight = run_dir / "costWeight.bin"
        shutil.copy2(weight_file, staged_weight)

        # ctrl_map_genarr.F ACTIVE_READ_XY's xx_etan.<optimcycle> with
        # ladinit=.FALSE., so the control file must exist.  A pure sensitivity
        # run needs it identically zero: the gradient is taken about the
        # unperturbed trajectory.
        nx, ny = int(contract["grid"]["nx"]), int(contract["grid"]["ny"])
        zeros = np.zeros((ny, nx), dtype=CTRL_DTYPE)
        zeros.tofile(run_dir / "xx_etan.0000000000.data")

        # xx_genarr2d_weight names a file that ctrl_map_genarr.F reads
        # unconditionally with READ_REC_3D_RL at ctrlprec.  data.ctrl sets
        # preproc 'noscaling', so wgenarr2d never divides the control and these
        # values are not used numerically -- but the name must be non-blank for
        # ctrl_init.F to register the control at all, and the file must be
        # readable.  Ones, so that a dropped 'noscaling' is a no-op rather than
        # a silent rescaling of the gradient.
        ones = np.ones((ny, nx), dtype=CTRL_DTYPE)
        ones.tofile(run_dir / "wetan_ones.bin")

    (run_dir / executable.name).symlink_to(executable)

    manifest = {
        "version": "adjoint_phase_a_v1" if settings.get("phase_a") else "mitgcm_adjoint_s0_run_v1",
        "mode": mode,
        "run_name": run_name or mode,
        "grdchk_point_global": list(grdchk_point) if grdchk_point else None,
        "cost_day": settings.get("cost_day", end_day),
        "objective": settings.get("weight"),
        "grdchk_eps": grdchk_eps,
        "run_dir": str(run_dir),
        "start_day": start_day,
        "end_day": end_day,
        "n_iter0": n_iter0,
        "n_time_steps": n_time_steps,
        "end_iteration": n_iter0 + n_time_steps,
        "adjoint": bool(settings["adjoint"]),
        "grdchk": bool(settings["grdchk"]),
        "adj_dump_freq": settings["adj_dump_freq"],
        # Provenance: anything other than 1e-7 is a section 12.2 diagnostic and
        # must not be reported as ground truth.
        "cg2d_target_residual": cg2d_target_residual,
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
    parser.add_argument("mode", help="a v1 mode (pickup/grdchk/runA/runB/runG3) or a Phase A run (P10 ... G90)")
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--scratch-root",
        default=None,
        help="defaults to mitgcm_adjoint_v1 for the v1 modes and mitgcm_adjoint_v2 for Phase A",
    )
    parser.add_argument("--run-name", default=None, help="run directory name (defaults to the mode)")
    parser.add_argument(
        "--grdchk-point",
        default=None,
        help="global 1-based 'i,j' to test; converted to the tile-local position grdchk wants",
    )
    parser.add_argument(
        "--grdchk-eps",
        type=float,
        default=None,
        help="override grdchk_eps; Phase A chooses it from the measured |S| after P90",
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--executable", default=None)
    parser.add_argument("--weight-file", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--cg2d-target-residual",
        type=float,
        default=1.0e-7,
        # DIAGNOSTIC ONLY (plan section 12.2).  1e-7 is the value every S0
        # forward segment was integrated with and the only value the production
        # runs may use -- tightening it means differentiating a model the FNO
        # was not trained on.  A tighter setting exists solely to separate
        # solver-tolerance error in the FINITE DIFFERENCE from a genuine error
        # in the adjoint during the gate G1 sweep.
    )
    arguments = parser.parse_args()

    project_root = (
        Path(arguments.project_root) if arguments.project_root else Path(__file__).resolve().parent.parent
    )
    phase_a = phase_a_modes(project_root)
    available = {**MODES, **phase_a}
    if arguments.mode not in available:
        parser.error(f"unknown mode {arguments.mode!r}; choose from {sorted(available)}")
    settings = available[arguments.mode]
    scratch_root = Path(
        arguments.scratch_root
        if arguments.scratch_root
        else (DEFAULT_SCRATCH_PHASE_A if settings.get("phase_a") else DEFAULT_SCRATCH)
    )
    if arguments.executable:
        executable = Path(arguments.executable)
    elif settings["adjoint"]:
        executable = project_root / "build" / "af_s0_ad" / "mitgcmuv_ad"
    else:
        executable = project_root / "build" / "af_s0" / "mitgcmuv"
    if not executable.is_file():
        raise StagingError(
            f"executable {executable} not found."
            + (
                "  Run 'make adall' in build/af_s0_ad -- that step needs the TAF licence."
                if settings["adjoint"]
                else ""
            )
        )

    # Phase A runs name their objective in the contract, so the weight field
    # follows from the run rather than from a flag that can be forgotten -- the
    # single largest silent-failure risk in this study is the two sides
    # weighting eta differently.
    if arguments.weight_file:
        weight_file = Path(arguments.weight_file)
    elif settings.get("weight"):
        weight_file = project_root / "work" / WEIGHT_FILES[settings["weight"]]
    else:
        weight_file = project_root / "work" / "costWeight_ssh_anomaly.bin"

    stage(
        arguments.mode,
        project_root,
        scratch_root,
        Path(arguments.source_dir),
        executable,
        weight_file,
        arguments.force,
        arguments.cg2d_target_residual,
        settings=settings,
        grdchk_eps=arguments.grdchk_eps,
        grdchk_point=(
            tuple(int(v) for v in arguments.grdchk_point.split(","))
            if arguments.grdchk_point
            else None
        ),
        run_name=arguments.run_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
