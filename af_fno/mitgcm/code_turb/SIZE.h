CBOP
C    !ROUTINE: SIZE.h
C    !DESCRIPTION: AF--FNO 0.25-degree turbulent grid on 64 MPI ranks.
C                  248 = 8 x 31, so an 8 x 8 rank decomposition gives the same
C                  31 x 31 tile the validated 1-degree configuration used.
CEOP
      INTEGER sNx, sNy, OLx, OLy
      INTEGER nSx, nSy, nPx, nPy
      INTEGER Nx, Ny, Nr
      PARAMETER (
     &           sNx = 31,
     &           sNy = 31,
     &           OLx =  2,
     &           OLy =  2,
     &           nSx =  1,
     &           nSy =  1,
     &           nPx =  8,
     &           nPy =  8,
     &           Nx  = sNx*nSx*nPx,
     &           Ny  = sNy*nSy*nPy,
     &           Nr  = 15)

      INTEGER MAX_OLX, MAX_OLY
      PARAMETER ( MAX_OLX = OLx,
     &            MAX_OLY = OLy )
