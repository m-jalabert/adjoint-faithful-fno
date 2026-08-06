# Model C S0 boundary/checkpoint stability diagnosis

This package performs zero retraining. It evaluates all seven stored
seed-20260724 anomaly-direct checkpoints on the same 15 fresh S0 inference
initializations and evaluation-only day-2000 truth used by job 304736.

The four-cell boundary band is defined inside the 60-by-60 wet rectangle.
Qx and Qy are depth-integrated zonal and meridional transports. Streamfunction
is reconstructed by cumulatively integrating Qx from south to north and is not
a neural-network output.

The package diagnoses whether earlier checkpoints avoid the runaway and
whether boundary transport error precedes interior error. It is descriptive
and cannot reselect a checkpoint.

Report content SHA-256: `0a370f256a39004573ecba883dc50b82c82b363c29ee32860e929e33a7155345`.
