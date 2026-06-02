# RunPod Setup Runbook

**Audience: a Claude session on a freshly-provisioned RunPod H100.** Your job here is one-time
machine bring-up: install prerequisites, get the code, build the dataset, and launch training. Work
through the steps **in order, verifying each gate before proceeding**. A bare RunPod template ships
almost nothing (often no conda/tmux/git) — *check what exists and install only what's missing*, rather
than assuming. Once training is launched and healthy, switch to [[runpod-operator-guide]] for
monitoring, and log the run in [[training-runs]].

> Bootstrapping note: if you're reading this from inside `/workspace/NanoNAV`, the repo is already
> cloned (skip to step 3). If the user pasted this file's contents, clone the repo in step 2 first.

## Pod prerequisites (provisioned via the RunPod UI — confirm, don't assume)

- 1× **H100 80 GB**, a CUDA-enabled PyTorch template.
- A **persistent / network volume mounted at `/workspace`** — the dataset and checkpoints must live
  here so they survive a pod stop/restart. Verify with `df -h /workspace`.

## Environment contract (export these once, reuse everywhere)

```bash
export WORKDIR=/workspace
export REPO_DIR=$WORKDIR/NanoNAV
export RESULTS_DIR=$WORKDIR/results            # checkpoints + logs (on the volume)
export LEKIWI_DATA_ROOT=$WORKDIR/data/lekiwi   # derived v2.1 dataset (on the volume)
export WANDB_PROJECT=nanonav
export CONDA_ROOT=$WORKDIR/miniconda3          # install conda here so it persists on the volume
mkdir -p "$RESULTS_DIR" "$(dirname "$LEKIWI_DATA_ROOT")"
```

## Step 1 — System prerequisites (install only what's missing)

Check each with `command -v <tool>`; install the absent ones.

```bash
# git, tmux, curl — system packages. Use sudo only if not already root.
if ! command -v git >/dev/null || ! command -v tmux >/dev/null || ! command -v curl >/dev/null; then
  SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq git tmux curl ca-certificates
fi
```

```bash
# conda — NanoWM ships a conda env. If absent, install Miniconda onto the volume.
if ! command -v conda >/dev/null 2>&1 && [ ! -d "$CONDA_ROOT" ]; then
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_ROOT"
fi
# Make conda usable in this shell:
source "${CONDA_ROOT}/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"
```

**Gate:** `command -v git tmux conda` all resolve, `df -h /workspace` shows the volume mounted.
Video decoding (av1) and `wandb`/`huggingface-cli` are NOT installed here — they come from the conda
env in step 3 (they are in `environment.yml`).

## Step 2 — Get the code (clone with the submodule)

Skip if `$REPO_DIR/.git` already exists (pod restarted with the volume intact) — just update instead.

```bash
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --recurse-submodules https://github.com/KaushikTheProgrammer/NanoNAV.git "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --recurse-submodules && git -C "$REPO_DIR" submodule update --init --recursive
fi
```

**Gate:** `$REPO_DIR/external/nanowm/environment.yml` exists (submodule populated). If not, run
`git -C "$REPO_DIR" submodule update --init --recursive`. See [[nanowm-integration]] for the layout.

## Step 3 — Conda environment

```bash
cd "$REPO_DIR/external/nanowm"
conda env list | grep -q '^nanowm ' || conda env create -f environment.yml -n nanowm
conda activate nanowm
```

This installs the pinned stack (`lerobot-datasets==2.1.0`, `pytorch-lightning==1.9.5`,
`diffusers==0.24.0`, `wandb`, `av`, `decord`, `hydra-core`, …). Creation is slow (minutes) — expected.

**Gate:** `python -c "import torch; print(torch.cuda.is_available())"` → `True`; `wandb --version`
and `huggingface-cli version` both resolve.

## Step 4 — Auth (paste tokens once)

```bash
huggingface-cli whoami >/dev/null 2>&1 || huggingface-cli login   # reads the source dataset
wandb login                                                       # for run monitoring (optional)
```

## Step 5 — Build the derived dataset (skip if present)

CPU-bound av1 decode, ~10–20 min. Writes a LeRobot v2.1 dataset NanoWM reads natively. **Always run
the dry-run first** — it verifies the v3.0→v2.1 frame-offset math without writing anything.

```bash
cd "$REPO_DIR"
if [ ! -d "$LEKIWI_DATA_ROOT/meta" ]; then
  python scripts/build_lekiwi_nav_dataset.py --limit 2 --dry-run     # GATE: must print "dry-run OK"
  python scripts/build_lekiwi_nav_dataset.py --out-root "$LEKIWI_DATA_ROOT"
fi
```

**Gate:** dry-run prints `dry-run OK`; after the full build, `$LEKIWI_DATA_ROOT/meta` exists and
loads. See [[nanowm-integration]] for what the dataset contains.

## Step 6 — Launch training (detached, so it survives SSH drops)

```bash
tmux new-session -d -s train "cd '$REPO_DIR/external/nanowm' && conda activate nanowm && \
  RESULTS_DIR='$RESULTS_DIR' LEKIWI_DATA_ROOT='$LEKIWI_DATA_ROOT' WANDB_PROJECT='$WANDB_PROJECT' \
  python src/main.py experiment=lekiwi_nav dataset=lerobot/lekiwi model=nanowm_b2 \
  2>&1 | tee -a '$RESULTS_DIR/train.log'"
```

**Gate (first ~500 steps):** loss is finite and trending down, `nvidia-smi` shows high GPU util, no
shape errors, checkpoints appearing under `$RESULTS_DIR`. Watch with `tmux attach -t train`
(detach: `Ctrl-b d`), `tail -f $RESULTS_DIR/train.log`, `watch -n2 nvidia-smi`.

## Step 7 — Hand off to monitoring

Training is running. Now follow [[runpod-operator-guide]] (healthy signature, failure playbook,
escalation) and record the run in [[training-runs]]. After ~50K steps, run the Stage-5 gate:

```bash
python src/sample/action_diagnostic.py --ckpt "$RESULTS_DIR"/<run>/checkpoints/latest-*.ckpt \
    --out "$RESULTS_DIR"/<run>/action_diag
```
