CBOP
C    !ROUTINE: SIZE.h
C    !DESCRIPTION: AF--FNO 1-degree tutorial grid on four MPI ranks.
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
     &           nPx =  2,
     &           nPy =  2,
     &           Nx  = sNx*nSx*nPx,
     &           Ny  = sNy*nSy*nPy,
     &           Nr  = 15)

      INTEGER MAX_OLX, MAX_OLY
      PARAMETER ( MAX_OLX = OLx,
     &            MAX_OLY = OLy )

