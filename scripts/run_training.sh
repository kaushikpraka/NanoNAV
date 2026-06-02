#!/usr/bin/env bash
# Launch NanoWM-B/2 training on the LeKiwi nav dataset (RunPod H100).
# Env is the uv venv at /workspace/nanowm-venv (not conda — see context/runpod-setup.md notes).
# Usage: tmux new-session -d -s train 'bash /workspace/NanoNAV/scripts/run_training.sh'
# NOTE: no `set -u` — the venv activate script references unbound vars (PS1, etc.).
set -eo pipefail

export WORKDIR=/workspace
export REPO_DIR=/workspace/NanoNAV
export RESULTS_DIR=/workspace/results
export LEKIWI_DATA_ROOT=/workspace/data/lekiwi
export WANDB_PROJECT=nanonav
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RESULTS_DIR"

source /workspace/nanowm-venv/bin/activate
cd "$REPO_DIR/external/nanowm"

python -c "import pytorch_lightning as pl, torch; print('[env] PL', pl.__version__, '| torch', torch.__version__)"

# python -u so the log streams live to train.log (tail -f it to monitor). tee (not -a) = fresh log each launch.
exec python -u src/main.py experiment=lekiwi_nav dataset=lerobot/lekiwi model=nanowm_b2 \
    2>&1 | tee "$RESULTS_DIR/train.log"
