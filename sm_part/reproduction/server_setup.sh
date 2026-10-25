#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${LLAMA_PYTHON:-/usr/bin/python3}" -m venv --system-site-packages --without-pip .venv-llama
# NGC's global constraint file pins other applications' package versions.
# Scope this override to pip inside our isolated venv; retain system CUDA torch.
env -u PIP_CONSTRAINT .venv-llama/bin/python -m pip install -r requirements-llama.txt
.venv-llama/bin/python reproduction/llama_run.py preflight
.venv-llama/bin/python -m pip freeze > results/llama/server-requirements-lock.txt
echo 'Environment ready. Run download with --prompt-token in the foreground, then server_start.sh.'
