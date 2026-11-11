"""Allowlisted source package; no credentials, prior raw responses or weights."""
import hashlib
import json
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
files=list((root/'reproduction').glob('next_*.py'))+[root/'reproduction/test_next_search.py',root/'reproduction/core.py',
      root/'results/next_experiments_plan/plan.md',root/'results/next_experiments_plan/protocol.json',
      root/'vendor/llm-astar/llmastar/pather/a_star/prompt.py',
      root/'vendor/llm-astar/dataset/environment_50_30.json',root/'vendor/llm-astar/LICENSE']
# core imports this particular frozen prompt file at module load.
files=[p for p in files if p.exists()]+[root/'vendor/llm-astar/llmastar/pather/llm_a_star/prompt.py']
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
archive=root/'dist/next-search-v2.zip';archive.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(root).as_posix())
    z.writestr('deployment_manifest.json',json.dumps(manifest,indent=2))
result=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files))
(root/'results/next_search_deployment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
