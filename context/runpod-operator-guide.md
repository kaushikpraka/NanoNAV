# RunPod Operator Guide

**Audience: the Claude session running *on the RunPod VM*** that babysits the training run. You were
started inside the pod and pointed at this repo. Read [[overview]], [[nanowm-integration]],
[[training]], and [[roadmap]] first for design context. Your job is **operational**, not design.

## Your role & scope

- **Keep the training run healthy:** monitor it, diagnose problems, self-heal where safe, escalate
  otherwise.
- **Record everything** in [[training-runs]] (append-only) — config, milestones, anomalies, the final
  diagnostic.
- **Do NOT change design/architecture decisions.** Action representation, model size, latent space,
  camera, frame interval, etc. are settled in `context/`. Operational fixes only (batch size, workers,
  resume, disk hygiene). If something seems to require a design change, **escalate to the user**.

## Environment contract

| Thing | Expected value |
|---|---|
| GPU | 1× H100 80 GB |
| Conda env | `nanowm` (`conda activate nanowm`) |
| NanoWM repo | the fork `KaushikTheProgrammer/nano-world-model` |
| `RESULTS_DIR` | `/workspace/results` (checkpoints + logs, on the persistent volume) |
| `LEKIWI_DATA_ROOT` | `/workspace/data/lekiwi` (derived v2.1 dataset, on the volume) |
| Session multiplexer | `tmux` (training runs in a named session so SSH drops don't kill it) |
| Experiment tracking | **wandb** — note the run URL in [[training-runs]] |

If env vars are unset, re-export them per [[runpod-setup]] (the bring-up runbook).

## Launch / resume

**First launch** (dataset already built into `$LEKIWI_DATA_ROOT`):
```bash
tmux new -s train
conda activate nanowm
cd external/nanowm   # or the fork clone
python src/main.py experiment=lekiwi_nav dataset=lerobot/lekiwi model=nanowm_b2
```
**Dataset missing?** Build it once (CPU-bound, ~10–20 min) before training:
```bash
python scripts/build_lekiwi_nav_dataset.py   # writes $LEKIWI_DATA_ROOT
```
**Resume after a crash/restart:** point at the latest checkpoint in `$RESULTS_DIR` (NanoWM/Lightning
writes periodic checkpoints there). Verify the resumed step matches the last logged step before
declaring recovery successful.

## "Healthy" signature

- **Loss** (v-prediction) trends down then plateaus; no sudden spikes to NaN/Inf.
- **Throughput / GPU util:** steps/sec stable; `nvidia-smi` shows sustained high util. Low util ⇒
  dataloader starvation (see playbook).
- **No NaN/Inf** in loss or grad norm.
- **Disk:** `$RESULTS_DIR` volume has headroom; checkpoints are being written on schedule.
- **Action-conditioning (project-specific, watch this):** the action-embedding RMS must stay healthy
  (~0.1+). Collapse toward ~0.002 is the **Finding-#4 atrophy** failure — the model is learning to
  ignore actions. See [[training]] (Table 5/6). If you see it trending toward zero, flag it.

## Failure playbook (symptom → check → fix)

| Symptom | Check | Fix |
|---|---|---|
| `CUDA out of memory` | batch size vs 80 GB | Lower `training.batch_size`, add grad-accum to keep **eff-bs 64**. Confirm bf16/amp on. |
| GPU util low, steps/sec poor | `nvidia-smi`, CPU load | Raise `infra.num_workers`; AV1 decode is CPU-heavy. Don't exceed pod vCPUs. |
| `NaN`/`Inf` loss | when it started; lr | Resume from last good ckpt; if persistent, lower lr; inspect for a corrupt batch. **Escalate if it recurs.** |
| Disk full on volume | `df -h /workspace` | Prune old checkpoints in `$RESULTS_DIR` (keep latest + best). |
| SSH dropped, is it dead? | `tmux ls`; `nvidia-smi` | Training survives in tmux. Reattach `tmux attach -t train`. |
| `wandb`/`huggingface` auth error | token present | `wandb login` / `huggingface-cli login` (user supplies token). |
| lerobot version / v3.0 read error | building dataset | The build script reads v3.0 raw and writes v2.1; do NOT `pip install` a different lerobot into the `nanowm` env (it pins 2.1.0). |
| Dataset won't load in NanoWM | `image_key`, `observation.state`, `episode_data_index` | Verify the v2.1 dataset has all three (see [[nanowm-integration]]). |

## Stop / success criteria

- Target steps reached (~50K for the first checkpoint) **AND** the Table 5/6 diagnostic passes
  (`python src/sample/action_diagnostic.py ...`): GT latent-L2 clearly below zero/random, action RMS
  ~0.1+. Save the checkpoint, record results in [[training-runs]], stop the run.

## Escalate to the user when

- Repeated OOM after batch/grad-accum mitigation, or you cannot reach eff-bs 64.
- Persistent NaN that survives a resume + lr reduction.
- The **diagnostic FAILS** (action atrophy) — this is a design-level outcome, not an op fix.
- Dataset integrity problems (missing episodes, wrong action shape, decode failures).
- Anything that would require changing a `context/` design decision.

Include the **wandb run URL** and the last ~50 log lines when escalating.
