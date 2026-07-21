C     Diagnostics storage for AF--FNO spin-up and production streams.
      INTEGER    ndiagMax
      INTEGER    numlists, numperlist, numLevels
      INTEGER    numDiags
      INTEGER    nRegions, sizRegMsk, nStats
      INTEGER    diagSt_size
      PARAMETER( ndiagMax = 500 )
      PARAMETER( numlists = 4, numperlist = 10, numLevels = 2*Nr )
      PARAMETER( numDiags = 10*Nr )
      PARAMETER( nRegions = 0, sizRegMsk = 1, nStats = 4 )
      PARAMETER( diagSt_size = 1 )

