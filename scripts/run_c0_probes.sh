#!/usr/bin/env bash
# C0 probe matrix (Option C / semantic WM retrain — context/semantic-wm-retrain.md).
# Four short trainings over frozen DINOv2 tokens that answer ONE question: does the action
# branch stay alive (Finding-#4 kill-switch) under {flow,x0} x {adaln_fuse,cross_attention,additive}?
# Each run: ~3k steps (~1-2 h H100), then action_diagnostic on its latest checkpoint.
# Verdict per run: PASS = action-emb RMS > 0.05 AND gt-action latent-L2 < zero/random baselines.
#
# Encoder = facebook/dinov2-small == the Gate-A-validated dinov2_vits14 weights (224px, 16x16
# patches, 384d). latent_scale=2.4 brings token std to ~1.0 (measured; the SD-VAE
# scaling_factor analog). model NanoWM-B/1: patch 1 over the 16x16 token grid -> 256 model
# tokens = per-DINO-token correspondence (RAE-NWM's choice).
#
# Usage: tmux new-session -d -s c0 'bash /workspace/NanoNAV/scripts/run_c0_probes.sh'
set -eo pipefail

export WORKDIR=/workspace
export REPO_DIR=/workspace/NanoNAV
export RESULTS_DIR=/workspace/results
export LEKIWI_DATA_ROOT=/workspace/data/lekiwi
export WANDB_PROJECT=nanonav
export WEBDINO_MODEL_PATH=facebook/dinov2-small
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

[ -f /workspace/secrets/env.sh ] && source /workspace/secrets/env.sh
source /workspace/nanowm-venv/bin/activate
cd "$REPO_DIR/external/nanowm"

SUMMARY="$RESULTS_DIR/c0_probe_summary.md"
echo "# C0 probe matrix — $(date -u +%Y-%m-%dT%H:%MZ)" > "$SUMMARY"

COMMON=(
  experiment=lekiwi_nav dataset=lerobot/lekiwi model=nanowm_b2
  latent_codec=webdino latent_codec.latent_scale=2.4
  model.arch=NanoWM-B/1 model.latent_size=16 model.latent_channels=384
  experiment.training.max_steps=3000
)
FLOW=(
  experiment.diffusion.pred_name=flow
  experiment.diffusion.snr_gamma=0.0
  experiment.diffusion.zero_terminal_snr=false
)

run_probe() {
  local name=$1; shift
  echo "=== [$name] overrides: $* ==="
  python -u src/main.py "${COMMON[@]}" model.name="$name" "$@" \
      2>&1 | tee "$RESULTS_DIR/${name}.log" || { echo "## $name: TRAIN CRASHED" >> "$SUMMARY"; return 0; }

  # newest run dir for this probe (hydra stamps <ts>-<model.name>-F4S10-lekiwi)
  local rundir
  rundir=$(ls -dt "$RESULTS_DIR"/*-"$name"-F*-lekiwi 2>/dev/null | head -1)
  local ckpt
  ckpt=$(find "$rundir/checkpoints" -name "*.ckpt" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  if [ -z "$ckpt" ]; then
    echo "## $name: NO CHECKPOINT FOUND ($rundir)" >> "$SUMMARY"; return 0
  fi
  echo "=== [$name] action_diagnostic on $ckpt ==="
  python -u src/sample/action_diagnostic.py --ckpt "$ckpt" \
      --out "$RESULTS_DIR/c0_diag_${name}" \
      2>&1 | tee "$RESULTS_DIR/c0_diag_${name}.log" || true
  {
    echo "## $name"
    echo '```'
    tail -20 "$RESULTS_DIR/c0_diag_${name}.log"
    echo '```'
  } >> "$SUMMARY"
}

run_probe C0a-dinoB1-flow-adalnfuse "${FLOW[@]}" model.action_injection.type=adaln_fuse
run_probe C0b-dinoB1-flow-xattn     "${FLOW[@]}" model.action_injection.type=cross_attention
run_probe C0c-dinoB1-x0-adalnfuse   experiment.diffusion.pred_name=x model.action_injection.type=adaln_fuse
run_probe C0d-dinoB1-flow-additive  "${FLOW[@]}" model.action_injection.type=additive

echo "=== C0 matrix complete -> $SUMMARY ==="
[ -x /workspace/bin/notify.sh ] && /workspace/bin/notify.sh "NanoNAV C0 probe matrix complete — summary at results/c0_probe_summary.md" || true
