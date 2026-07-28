# Python package layout

- `core/`: reusable configuration, canonical-data, metrics, training, rollout,
  plotting, and reporting utilities.
- `analysis/`: evaluation and project-facing plotting entry points.
- `diagnostics/`: training-only audits and bounded Model C diagnostics.
- package root: scientific workflow modules whose historical paths are fixed by
  immutable experiment contracts.

Do not move a root workflow module without first checking every
`config/*source_hashes*` entry. Completed contracts intentionally retain their
original paths and file hashes.
