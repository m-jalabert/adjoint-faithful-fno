# Physical static channels for the two-in / one-out Model C

This model warm-starts `model_c_2in_1out_v1` at optimizer step
3,840. The map, the temporal context and the spatial
bandwidth are all unchanged --- still `(x_(t-10), x_t) -> x_(t+10)` on 32 x 32
Fourier modes with the trained bias-free local 3 x 3 branch and the
deterministic sine/cosine position encoder. Only the description of the
environment moves:

    parent    [tau_x, wet mask, distance to wall]
    this arm  [tau_x, wet mask, f(phi), dx(phi), theta_clim(x, y)]

`tau_x` is the actual momentum forcing and the wet mask the actual basin
geometry, so both are kept. The three added fields are coefficients that appear
in the governing equations: the Coriolis parameter `f = 2 Omega sin(phi)`, the
zonal grid spacing `dx = R cos(phi) dlambda` that makes the spherical grid
physically non-uniform in x, and the SST relaxation target the setup restores
towards on a 30-day timescale. `distance_to_wall_normalized` is removed as an
engineered heuristic rather than a term in those equations. Raw longitude and
latitude are deliberately not added: position already enters through the
encoder, and latitude's physical role is now carried by `f` and `dx`.

The external block therefore grows from 95 to
97 channels and lifting from
97 to 99.

**The warm start is not function-preserving, and could not be.** The parent
carries trained weights on the removed channel, so the initial map loses exactly
that field's contribution. The three new coefficient columns begin at zero, the
94 shared state/wind/wet columns and the position tail are copied unchanged, and
the size of the resulting step is measured rather than assumed: mean absolute
change 0.00988 against a mean
absolute output of 0.48719
in normalized state units, i.e.
2.03%.

Dataset, split, normalizers, six-step autoregressive loss, optimizer reset,
schedule, seed, validation starts and checkpoint-selection rule are inherited
byte-for-byte from the two-input parent.

| step | short AUC 10--90 (speed / SST / pressure) | long / climatology |
| --- | --- | --- |
| 960 | 0.095 / 1.026 / 0.436 | 0.443 / 0.553 / 0.177 |
| 1,920 | 0.093 / 1.005 / 0.481 | 0.427 / 0.563 / 0.193 |
| 2,880 | 0.094 / 1.002 / 0.460 | 0.463 / 0.536 / 0.252 |
| 3,840 | 0.091 / 0.983 / 0.342 | 0.419 / 0.520 / 0.153 |

The step-3,840 parent scores
0.095 / 1.021 / 0.360 short and
0.436 / 0.541 / 0.166 long on the
same 102 pooled rollouts, in the order
surface_speed, sst, phihyd_surface.

Selected step 3,840 via
`primary_rule`. Validation gate:
**pass**.

Static channels: wind_stress_x, wet_mask, coriolis_parameter, zonal_grid_spacing, sst_relaxation_target.
Parameter count: 27,328,780.
Report content SHA-256: `6ff261509c15cafb865fd6787c8c73196fd761a7da910ed764ebb95b7e55f3fd`.
