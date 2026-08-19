CBOP
C !ROUTINE: CTRL_OPTIONS.h
C !INTERFACE:
C #include "CTRL_OPTIONS.h"

C !DESCRIPTION:
C *==================================================================*
C | CPP options file for Control (ctrl) package:
C | Control which optional features to compile in this package code.
C *==================================================================*
CEOP

#ifndef CTRL_OPTIONS_H
#define CTRL_OPTIONS_H
#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

#ifdef ALLOW_CTRL
#ifdef ECCO_CPPOPTIONS_H

C-- When multi-package option-file ECCO_CPPOPTIONS.h is used (directly included
C    in CPP_OPTIONS.h), this option file is left empty since all options that
C   are specific to this package are assumed to be set in ECCO_CPPOPTIONS.h

#else /* ndef ECCO_CPPOPTIONS_H */
C   ==================================================================
C-- Package-specific Options & Macros go here

C  o  Re-activate deprecated codes in pkg/ecco & pkg/ctrl (but not recommended)
C     and since pkg/ctrl can be used without pkg/ecco, better to have it here
#undef ECCO_CTRL_DEPRECATED

#undef EXCLUDE_CTRL_PACK
#undef ALLOW_NONDIMENSIONAL_CONTROL_IO

C       >>> Initial values.
#undef ALLOW_THETA0_CONTROL
#undef ALLOW_SALT0_CONTROL
#undef ALLOW_UVEL0_CONTROL
#undef ALLOW_VVEL0_CONTROL
C--   AF--FNO: ALLOW_ETAN0_CONTROL is DEAD CODE in c68j and must stay undefined.
C     Every one of its blocks -- ctrl_init.F:552, ctrl_map_ini.F:531,
C     ctrl_pack.F:613, ctrl_unpack.F:703, grdchk_getxx.F:601 -- sits inside
C     #ifdef ECCO_CTRL_DEPRECATED, which is defined nowhere in this checkout.
C     Defining ALLOW_ETAN0_CONTROL therefore compiles CTRL_MAP_INI down to an
C     empty subroutine: xx_etan_dummy never reaches etaN, TAF reports
C     "the independent variables have no influence on the variables : fc",
C     and the generated adjoint is identically zero.  See the 2026-08-17
C     entry in docs/mitgcm_adjoint_ground_truth_plan.md.
#undef ALLOW_ETAN0_CONTROL
#undef ALLOW_TR10_CONTROL
#undef ALLOW_TAUU0_CONTROL
#undef ALLOW_TAUV0_CONTROL
#undef ALLOW_SFLUX0_CONTROL
#undef ALLOW_HFLUX0_CONTROL
#undef ALLOW_SSS0_CONTROL
#undef ALLOW_SST0_CONTROL

C       >>> Surface fluxes.
#undef ALLOW_HFLUX_CONTROL
#undef ALLOW_SFLUX_CONTROL
#undef ALLOW_USTRESS_CONTROL
#undef ALLOW_VSTRESS_CONTROL
#undef ALLOW_SWFLUX_CONTROL
#undef ALLOW_LWFLUX_CONTROL

C       >>> Atmospheric state.
#undef ALLOW_ATEMP_CONTROL
#undef ALLOW_AQH_CONTROL
#undef ALLOW_UWIND_CONTROL
#undef ALLOW_VWIND_CONTROL
#undef ALLOW_PRECIP_CONTROL

C       >>> Other Control.
#undef ALLOW_DIFFKR_CONTROL
#undef ALLOW_KAPGM_CONTROL
#undef ALLOW_KAPREDI_CONTROL
#undef ALLOW_BOTTOMDRAG_CONTROL

C       >>> Backward compatibility option (before checkpoint 65p)
#undef ALLOW_KAPGM_CONTROL_OLD
#undef ALLOW_KAPREDI_CONTROL_OLD

C       >>> Generic Control.
C--   AF--FNO: this is the live initial-SSH control path in c68j, and the only
C     one.  CTRL_MAP_INI_GENARR matches xx_genarr2d_file(iarr)(1:7)=='xx_etan'
C     and calls CTRL_MAP_GENARR2D( etaN, ... ), which does the ACTIVE_READ_XY
C     of xx_etan.<optimcycle> against xx_genarr2d_dummy(iarr) -- already in the
C     TAF -input set of tools/adjoint_options/adjoint_default.  Defining this
C     also flips ctrlUseGen to .TRUE. by default (ctrl_readparms.F:176), which
C     is what routes CTRL_INIT_VARIABLES to the generic path.
C
C     Unlike the deprecated ALLOW_ETAN0_CONTROL path, this perturbs etaN only,
C     not etaH.  That is not a loss: INITIALISE_VARIA calls INTEGR_CONTINUITY
C     after PACKAGES_INIT_VARIABLES, and with implicDiv2Dflow = 1 (the default
C     here) UPDATE_ETAH sets etaH = etaN before the first timestep.  etaH is a
C     dependent diagnostic, not a second independent control.
#define ALLOW_GENARR2D_CONTROL
#undef ALLOW_GENARR3D_CONTROL
#undef ALLOW_GENTIM2D_CONTROL

C  o Rotation of wind/stress controls adjustments
C    from Eastward/Northward to model grid directions
#undef ALLOW_ROTATE_UV_CONTROLS

C  o Originally the first two time-reccords of control
C    variable tau u and tau v were skipped.
C    The CTRL_SKIP_FIRST_TWO_ATM_REC_ALL option extends this
C    to the other the time variable atmospheric controls.
#undef CTRL_SKIP_FIRST_TWO_ATM_REC_ALL

C  o use pkg/smooth correlation operator (incl. smoother) for 2D controls (Weaver, Courtier 01)
C    This CPP option just sets the default for ctrlSmoothCorrel2D to .TRUE.
#undef ALLOW_SMOOTH_CORREL2D
C  o use pkg/smooth correlation operator (incl. smoother) for 3D controls (Weaver, Courtier 01)
C    This CPP option just sets the default for ctrlSmoothCorrel3D to .TRUE.
#undef ALLOW_SMOOTH_CORREL3D

C  o apply pkg/ctrl/ctrl_smooth.F to 2D controls (outside of ctrlSmoothCorrel2D)
#undef ALLOW_CTRL_SMOOTH
C  o apply pkg/smooth/smooth_diff2d.F to 2D controls (outside of ctrlSmoothCorrel2D)
#undef ALLOW_SMOOTH_CTRL2D
C  o apply pkg/smooth/smooth_diff3d.F to 3D controls (outside of ctrlSmoothCorrel3D)
#undef ALLOW_SMOOTH_CTRL3D

C   ==================================================================
#endif /* ndef ECCO_CPPOPTIONS_H */
#endif /* ALLOW_CTRL */
#endif /* CTRL_OPTIONS_H */
