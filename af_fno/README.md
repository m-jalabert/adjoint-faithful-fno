# AF--FNO active experiment track

This directory implements the minimum-data programme in
`docs/AF_FNO_Project_Plan.tex`.  It is the active MITgcm experiment track; the
retired 0.25-degree MITgcm attempt and its operational artifacts have been
removed.

S0 uses the official 62 x 62 x 15 baroclinic-gyre tutorial, a 1,200 s time
step, four MPI ranks in a balanced 2 x 2 decomposition, and the tutorial
control wind stress (`tau0 = 0.1 N m-2`).  The literal project-plan schedule is
100 spin-up years followed by 10 years of daily production.

The short-partition workflow divides S0 into eleven immutable ten-year
segments.  Every segment writes annual permanent pickups; Slurm `afterok`
dependencies prevent a child segment from starting unless its parent is
complete and verified.

Submit the complete chain from the repository root with:

```bash
./scripts/submit_af_s0_chain.sh
```

The canonical UCSB runtime location for the completed data is recorded in
`docs/Project_tracking.tex`.  New sites should set `AF_SCRATCH_ROOT` explicitly
rather than relying on the UCSB default embedded in the Slurm wrappers.

After S0 passes the tutorial validation, submit the low- and high-wind branches
with `./scripts/submit_af_s1_s2.sh`.  S1 (`tau0=0.075 N m-2`) and S2
(`tau0=0.125 N m-2`) both branch from the validated S0 year-100 pickup, adjust
for five model years, and then produce ten years of daily output.  Each branch
is split into an adjustment and production job linked by an `afterok`
dependency.
