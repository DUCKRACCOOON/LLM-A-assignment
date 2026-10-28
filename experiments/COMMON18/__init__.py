"""Common-host comparison; optional local dependency directories are ignored."""
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
_root=Path(__file__).resolve().parents[2]
for name in ('plot_deps','common_deps','torch_deps'):
    p=_root/'local_private'/name
    if p.exists():sys.path.insert(0,str(p))
