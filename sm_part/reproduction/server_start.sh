#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/llama
if [[ ! -f results/llama/model_manifest.json ]]; then
    echo 'Download first: .venv-llama/bin/python reproduction/llama_run.py download --prompt-token'
    exit 1
fi
if [[ -f results/llama/runner.pid ]] && kill -0 "$(cat results/llama/runner.pid)" 2>/dev/null; then
    echo 'A recorded runner PID is alive. Inspect it before launching another worker.'
    exit 1
fi
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
nohup .venv-llama/bin/python reproduction/llama_run.py all --device "${1:-0}" >> results/llama/run.log 2>&1 < /dev/null &
echo "$!" > results/llama/runner.pid
echo "Started PID $(cat results/llama/runner.pid). Status: results/llama/status.json"
