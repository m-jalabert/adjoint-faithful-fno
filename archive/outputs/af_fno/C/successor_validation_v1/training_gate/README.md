# Model C successor validation — three-seed training gate

Status: operationally complete and scientifically passed.

GPU array 290673 trained the two missing replicas of the already selected
width-128 successor. Seed 20260723 is the hash-pinned phase-2 checkpoint from
job 290597; array tasks 0 and 1 produced seeds 20260724 and 20260725 without
reading validation or any later archive.

Every seed:

- beats persistence for U, V, temperature, and SSH separately in S0, S1, and
  S2 over all 15,060 training pairs;
- reloads its three-step prediction bitwise exactly;
- remains finite through the declared 180-day training-only rollout; and
- passes every 90- and 180-day amplitude bound.

The worst regime/group ratios to persistence are `0.762683`, `0.732270`, and
`0.839507` for seeds 20260723, 20260724, and 20260725. In each case the worst
group is low-wind S1 SSH.

`three_seed_training_gate.json` is the lightweight project-facing evidence.
The complete reports and 116 MB checkpoints remain immutable in scratch. This
result authorizes one execution of the prospectively frozen fresh-validation
contract. It does not yet freeze Model C or authorize inference.
