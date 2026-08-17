CBOP
C    !ROUTINE: SIZE.h
C    !DESCRIPTION: AF--FNO 0.25-degree turbulent grid on 16 MPI ranks
C                  (4 x 4 decomposition, 62 x 62 tiles).
CEOP
      INTEGER sNx, sNy, OLx, OLy
      INTEGER nSx, nSy, nPx, nPy
      INTEGER Nx, Ny, Nr
      PARAMETER (
     &           sNx = 62,
     &           sNy = 62,
     &           OLx =  2,
     &           OLy =  2,
     &           nSx =  1,
     &           nSy =  1,
     &           nPx =  4,
     &           nPy =  4,
     &           Nx  = sNx*nSx*nPx,
     &           Ny  = sNy*nSy*nPy,
     &           Nr  = 15)

      INTEGER MAX_OLX, MAX_OLY
      PARAMETER ( MAX_OLX = OLx,
     &            MAX_OLY = OLy )
