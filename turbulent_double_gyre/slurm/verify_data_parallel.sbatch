#!/bin/bash -l
#SBATCH --job-name=turb-verify-dp
#SBATCH --account=pod-users
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
set -euo pipefail
module purge
module load gnu12/12.2.0 cuda/12.4
R=/home/mjalabert314/bire_james25_repro
PY="${R}/.venv/bin/python"
S="${R}/turbulent_double_gyre/scripts/verify_data_parallel_equivalence.py"
W="${R}/turbulent_double_gyre/work"
mkdir -p "$W"
export PYTHONPATH="${R}/turbulent_double_gyre/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4

echo "=== world 1 ==="
AF_EQUIV_OUT="$W/equiv_world1.json" "$PY" "$S"
echo "=== world 2 ==="
AF_EQUIV_OUT="$W/equiv_world2.json" "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node=2 "$S"

"$PY" - <<'PYEOF'
import json, pathlib
W = pathlib.Path("/home/mjalabert314/bire_james25_repro/turbulent_double_gyre/work")
a = json.loads((W / "equiv_world1.json").read_text())
b = json.loads((W / "equiv_world2.json").read_text())
print(f"{'metric':24s} {'world 1':>18s} {'world 2':>18s} {'rel diff':>12s}")
ok = True
for key in ("grad_l2", "grad_l1", "grad_max_abs"):
    x, y = a[key], b[key]
    rel = abs(x - y) / max(abs(x), 1e-30)
    ok &= rel < 1e-4
    print(f"{key:24s} {x:18.9e} {y:18.9e} {rel:12.3e}")
print(f"\nparameters {a['parameters']:,} == {b['parameters']:,}: {a['parameters']==b['parameters']}")
print("EQUIVALENCE:", "PASS" if ok and a["parameters"] == b["parameters"] else "FAIL")
PYEOF
