# Model C SST and surface-pressure baseline companion

This immutable descriptive package makes the climatology and persistence
comparisons from the original fixed-axis Bire-style Figure 4 easier to inspect.
It does not rerun or select a model and does not open any new dataset split.

The left column of `model_c_bire_figure4_dt10_sst_phihyd_full_and_zoom.png` shows the complete finite Model C RMSE range.
The right column zooms to the baseline scale. Vertical lines mark the first lead
where Model C mean RMSE becomes worse than persistence or climatology.

SST first loses to persistence/climatology at
20/
20 days. Surface PHIHYD first loses
at 30/
70 days.

`model_c_bire_figure4_dt10_sst_phihyd_rmse.csv` contains all member-mean and 10th--90th percentile values.
Manifest content SHA-256: `01433936e406dc9e699c9751a47091168599472fcb6c7b5d164db4aa7da99063`.
