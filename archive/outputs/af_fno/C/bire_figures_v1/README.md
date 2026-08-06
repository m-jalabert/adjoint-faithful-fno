# Model C Bire-style Figures 3 and 4

Status: complete descriptive characterization of the scientifically rejected
Model C successor. This package cannot authorize tuning or inference.

`model_c_bire_figure3_streamfunction_1deg.png` follows Bire Figure 3 at the project's native 1-degree
resolution: MITgcm truth, Model C prediction, and truth-minus-prediction
barotropic streamfunction at 0--40 days. The prospectively selected member is
S2 validation start 6335.

`model_c_bire_figure4_dt10_rmse_0_200_days.png` follows the middle column of Bire Figure 4 for
delta_t = 10 days. It uses 15 prospectively selected S2 fresh-validation
initial conditions and reports the member mean with 10th--90th percentile
shading from 0 to 200 days.

Persistence repeats each member's initial condition at every lead. Climatology
is the S2 pointwise time mean over all split-1 training snapshots; nonlinear
derived fields are time-averaged after derivation. This is the paper's
mathematical definition with a stricter training-only, leakage-free source.

The complete report and arrays remain immutable in scratch. Figure-manifest
content SHA-256: `258be6fabe136c0b7a505616925b4d0bb4e7670a63d50d4b70160719f036827f`.
