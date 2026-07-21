# Bire A0 reconstruction evidence

This workflow targets the scientific experiment described in Bire et al. (2025), not an
unattainable byte-identical replay. The paper and archived repository omit the MITgcm input
deck and conflict in several places. Every choice below is therefore locked in
`config/bire_a0_reference.toml` and recorded in product manifests. The retired
0.25-degree MITgcm campaign is not active; this document is retained only as
historical evidence for the adapted A0 architecture and preprocessing choices.

| Ambiguity or defect | Locked reconstruction | Consequence |
|---|---|---|
| Paper says 15 levels and 2,000 m, but lists 17 interfaces | 15 layers with thicknesses 50, 60, ..., 190 m (1,800 m total) | Matches both “15” and “bottom thickness 190 m”; does not silently invent layer 16 |
| Printed SST-restoring equation has the wrong sign | Linear 30 °C south to 10 °C north | Matches the prose and archived generator |
| No MITgcm version or namelists | MITgcm `checkpoint68j`, based on `tutorial_baroclinic_gyre` | Exact source SHA and generated input hashes are retained |
| No spin-up duration or criterion | Adaptive 20-year blocks; two consecutive passing blocks | Control minimum 100 years, branches minimum 40 years, with documented caps |
| 6,000 train + 1,200 validation + 1,000 inference exceeds 7,200 | Train `[0,6000)`, validation `[6000,7200)`, inference `[6200,7200)` | The inference window is explicitly an overlap, consistent with no separate test set |
| Archived 15 “inference” indices range to 2,030 and are needed for 2,000-day truth | Treat recovered indices as absolute post-spin-up production indices | Long truth exists, but these indices overlap the declared training window; reports flag this leakage |
| Normalization absent from paper but present in code history | Pointwise mean/std over 18,000 training states, epsilon `1e-5` | Statistics are stored as a checksummed Zarr product |
| Public converter emits 12 channels but training expects 11 | Ten dynamic channels plus wind; omit SST restoring field | Matches Figure 1 and downstream code |
| Public `calculate_rmse_acc.py` calls MSE “RMSE” | True cosine-latitude-weighted RMSE is primary | Legacy unweighted MSE is still exported with an explicit label |
| Figure captions swap SST and pressure rows | Follow the visible panel order and label it correctly | Reproduced captions do not repeat the publication error |
| Low-resolution scripts default to stride 2 | Fixed stride 8 (0.25° to 2°) | Gives a 31 × 31 grid; spectral modes are cropped to available modes |
| Tutorial uses a one-degree land rim while the paper only states 248² | Four 0.25° land cells on every side | Gives the archived code's 240 × 240 active ocean interior |
| Diagnostic precision is unstated | R32 daily diagnostics; model arithmetic remains the compiled default | Cuts raw storage in half without reducing the canonical float32 product |
| Tutorial density differs from the paper's Sverdrup calculation | MITgcm `rhoNil=999.8`; Figure 9 theory uses `rho0=1000` kg m⁻³ | Prevents a model constant and an analytic plotting constant from being conflated |
| Published operator differs from stock `neuraloperator.FNO` | Custom `PaperFNO2d` built from NeuralOperator-compatible spectral components | Preserves depthwise Fourier mixing and separate pointwise channel mixing |
| Epochs, seed, optimizer, and schedule omitted | Archived seed/optimizer values plus bounded early stopping | All deviations/retries are written into training manifests |

Raw MDS output is eligible for deletion only after the canonical reduced store and restart
lineage have checksummed manifests and validation passes. Cleanup is dry-run by default. It
unlinks only the exact sealed `dynDiag` files and their staging symlinks; pickups, forcing,
grids, logs, manifests, directories, and post-seal/unlisted files are preserved.
