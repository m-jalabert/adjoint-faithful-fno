# Historical Bire data contract (A0 reference)

This is the recovered paper-oriented contract used as evidence when defining A0.
It is not the active 1-degree AF-FNO data contract.

The reduced MITgcm store is `mitgcm_state.zarr` with array:

```text
state[experiment, time, channel, y, x]
      [5,          7200, 11,      248, 248] float32
```

Coordinates are `experiment`, `tau0_n_m2`, `time_day`, `channel`, `lat_deg_n`, and
`lon_deg_e`. Daily state index zero is the first post-spin-up production output. The channel
order is immutable:

1. `U_surface`
2. `U_mid`
3. `V_surface`
4. `V_mid`
5. `T_surface`
6. `T_mid`
7. `PHIHYD_surface`
8. `PHIHYD_mid`
9. `PHIHYD_bottom`
10. `barotropic_streamfunction`
11. `wind_stress_x`

The midlevel is zero-based vertical index 7. Its actual MITgcm cell-center depth is recorded
from the generated grid; the paper's nominal label is 950 m. U is averaged to tracer centers
in x, V in y, and `PsiVEL` is summed vertically and centered in both directions, matching the
archived converter.

The four-cell (one-degree) land rim leaves a 240 × 240 wet interior. MDS diagnostics are R32;
checksums and record counts, rather than higher raw precision, guard the conversion.
Raw data arrive in restartable 360-day directories beneath
`mitgcm/production/expNN_slug/chunks/`. Collision-checked symbolic links expose them through
the flat `raw/slug/production` layout expected by xmitgcm, without a second full raw copy. The
converter uses the exact staged `wind_for_fno.npy` rather than independently regenerating it.

An FNO rollout group contains:

```text
prediction[member, day_index, channel, y, x]  # ten dynamic channels
truth[member, day_index, channel, y, x]
initial_index[member]
day[day_index]
```

Required rollout attributes are `experiment_id`, `lag_days`, `checkpoint`, `config_sha256`,
and `resolution`. Plot discovery uses attributes rather than filenames.
