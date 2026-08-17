#include "COST_OPTIONS.h"

CBOP
C     !ROUTINE: COST_TEST
C     !INTERFACE:
      SUBROUTINE COST_TEST( myThid )

C     !DESCRIPTION:
C     *==========================================================*
C     | AF--FNO adjoint ground truth: the quantity of interest.
C     |
C     |     J = sum_ij w_ij * etaN_ij(T)
C     |
C     | w is read at runtime from afCostWeightFile, so every linear
C     | functional of the final sea surface height is a runtime
C     | choice and one TAF build serves the whole study.  For the
C     | SSH anomaly at the frozen target cell p*,
C     |
C     |     w = delta_{p*} - rA * maskC / A_wet
C     |
C     | built by scripts/make_cost_weight.py.
C     |
C     | Accumulating into the per-tile objf_test is deliberate:
C     | COST_FINAL already forms  fc = mult_test * sum_tiles
C     | objf_test  and then the global sum, so J is assembled
C     | exactly with no global sum inside the differentiated code.
C     |
C     | COST_TEST is called once, from COST_FUNCTION, which
C     | THE_MAIN_LOOP invokes as COST_FINAL after the time loop.  J
C     | is therefore evaluated on the final state and needs no
C     | myIter guard.  objf_test is zeroed by COST_INIT_VARIA.
C     *==========================================================*

C     !USES:
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "DYNVARS.h"
#include "GRID.h"
#include "cost.h"

C     !INPUT PARAMETERS:
C     myThid :: my Thread Id number
      INTEGER myThid
CEOP

#ifdef ALLOW_COST_TEST
C     !LOCAL VARIABLES:
C     afCostW :: the weight field w, passive
      INTEGER bi, bj, i, j
      INTEGER itlo, ithi, jtlo, jthi
      _RL     afCostW(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)

      jtlo = myByLo(myThid)
      jthi = myByHi(myThid)
      itlo = myBxLo(myThid)
      ithi = myBxHi(myThid)

C--   Passive read.  READ_REC_XY_RL, not ACTIVE_READ_XY: w defines the
C     functional, it is not a control, and it carries no dependence on any
C     independent variable.  This is the same idiom ctrl_map_ini_gen.F uses
C     for its weight fields on the TAF path.
      CALL READ_REC_XY_RL( afCostWeightFile, afCostW, 1, 0, myThid )

      DO bj = jtlo,jthi
       DO bi = itlo,ithi
        DO j = 1,sNy
         DO i = 1,sNx
          objf_test(bi,bj) = objf_test(bi,bj)
     &                     + afCostW(i,j,bi,bj) * etaN(i,j,bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO

#endif /* ALLOW_COST_TEST */

      RETURN
      END

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
