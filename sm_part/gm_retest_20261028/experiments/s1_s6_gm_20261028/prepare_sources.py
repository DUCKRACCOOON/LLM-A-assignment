"""Export fixed GM source and inputs, without touching existing worktrees."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent / "vendor" / "gm"
REPO = ROOT / "tmp" / "sm-publish"
REF = "c56eb83b7df4be06d402daba36e0a2b45afa950b"
GIT = ["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO)]
files = subprocess.check_output(GIT + ["ls-tree", "-r", "--name-only", REF], text=True).splitlines()
explicit = {
    "source_code/dataset/environment_50_30.json",
    "source_code/llmastar/pather/llm_a_star/prompt.py",
    "results/generalization/cases.json",
    "experiments/K3/results/run_20261025/public_cases.json",
    "experiments/K3/results/run_20261025/synthetic_cases.json",
    "experiments/K4/results/run_20261025/transfer_cases.json",
    "experiments/K3/results/run_20261025/public/per_case.csv",
    "experiments/K4/results/run_20261025/responses.jsonl",
}
chosen = [name for name in files if name in explicit or
          (name.endswith(".py") and name.startswith(("reproduction/", "experiments/K3/", "experiments/K4/")))]
manifest = {"gm_commit": REF, "files": {}}
for name in chosen:
    data = subprocess.check_output(GIT + ["show", f"{REF}:{name}"])
    out = DEST / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.read_bytes() != data:
        raise RuntimeError(f"Existing snapshot differs: {name}")
    out.write_bytes(data)
    manifest["files"][name] = hashlib.sha256(data).hexdigest()
out = ROOT / "results" / "s1_s6_gm_20261028" / "source_snapshot.json"
out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({"exported_files": len(chosen), "gm_commit": REF}))
