#!/usr/bin/env bash
# 6b.3 — Mac-side SSH reverse tunnel: pod ↔ LeKiwi bridge (no Tailscale, no TUN).
#
# Run this ON THE MAC (which is on the LAN with the Pi). It dials into the RunPod pod over its
# exposed SSH/TCP port and REVERSE-forwards the two lerobot ZMQ ports back out to the Pi, so the
# pod can run `lekiwi_mpc.py --planner wm --ip 127.0.0.1` as the lerobot client unchanged.
#
#   pod:127.0.0.1:5555  --(-R, over ssh)-->  this Mac  -->  PI:5555   (cmd)
#   pod:127.0.0.1:5556  --(-R, over ssh)-->  this Mac  -->  PI:5556   (observations)
#   pod:127.0.0.1:9876  --(-R, over ssh)-->  this Mac  -->  Mac:9876  (rerun viewer; optional)
#
# Uses autossh if installed (auto-reconnect), else a plain-ssh retry loop. Keep this terminal open
# for the whole session. See context/tailscale-setup.md "Recommended: RunPod TCP port + SSH reverse tunnel".
#
# Usage:
#   scripts/tunnel_up.sh --pod-host <RUNPOD_PUBLIC_IP> --pod-port <SSH_TCP_PORT> [opts]
#   (get host+port from the RunPod console -> Connect -> "SSH over exposed TCP" — NOT ssh.runpod.io)
# Opts:
#   --pi-ip IP       LeKiwi Pi LAN IP            (default 10.0.0.125)
#   --user U         pod ssh user               (default root)
#   --key PATH       ssh identity file          (default: ssh's own resolution)
#   --rerun-port N   also reverse-forward N for rerun telemetry (default 9876; --rerun-port 0 to skip)
#   --cmd-port N     ZMQ command port           (default 5555)
#   --obs-port N     ZMQ observation port       (default 5556)
set -euo pipefail

POD_HOST=""; POD_PORT=""; PI_IP="10.0.0.125"; USER_="root"; KEY=""
RERUN_PORT="9876"; CMD_PORT="5555"; OBS_PORT="5556"

while [ $# -gt 0 ]; do
  case "$1" in
    --pod-host)   POD_HOST="$2"; shift 2 ;;
    --pod-port)   POD_PORT="$2"; shift 2 ;;
    --pi-ip)      PI_IP="$2";    shift 2 ;;
    --user)       USER_="$2";    shift 2 ;;
    --key)        KEY="$2";      shift 2 ;;
    --rerun-port) RERUN_PORT="$2"; shift 2 ;;
    --cmd-port)   CMD_PORT="$2"; shift 2 ;;
    --obs-port)   OBS_PORT="$2"; shift 2 ;;
    -h|--help)    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[tunnel] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$POD_HOST" ] || [ -z "$POD_PORT" ]; then
  echo "[tunnel] --pod-host and --pod-port are required (RunPod console -> Connect -> SSH over exposed TCP)." >&2
  echo "[tunnel] e.g. scripts/tunnel_up.sh --pod-host 31.24.80.34 --pod-port 12345" >&2
  exit 2
fi

# Reverse forwards: pod localhost:<port> -> (tunnel) -> Mac -> target.
FORWARDS=( -R "${CMD_PORT}:${PI_IP}:${CMD_PORT}" -R "${OBS_PORT}:${PI_IP}:${OBS_PORT}" )
if [ "${RERUN_PORT}" != "0" ]; then
  FORWARDS+=( -R "${RERUN_PORT}:127.0.0.1:${RERUN_PORT}" )   # pod -> Mac's local rerun viewer
fi

SSH_OPTS=(
  -N                                   # no remote command, just forward
  -o ServerAliveInterval=30            # keep warm during the ~8s CEM compute (no traffic) so NAT won't reap it
  -o ServerAliveCountMax=3
  -o ExitOnForwardFailure=yes          # fail fast if a remote port is already bound (stale tunnel)
  -o StrictHostKeyChecking=accept-new
  -p "${POD_PORT}"
)
[ -n "$KEY" ] && SSH_OPTS+=( -i "$KEY" )
TARGET="${USER_}@${POD_HOST}"

echo "[tunnel] pod ${TARGET}:${POD_PORT}  |  Pi ${PI_IP}"
echo "[tunnel] reverse-forwarding pod:127.0.0.1 -> ${CMD_PORT}(cmd) ${OBS_PORT}(obs)$([ "$RERUN_PORT" != 0 ] && echo " ${RERUN_PORT}(rerun)")"
echo "[tunnel] then ON THE POD:"
echo "         python scripts/lekiwi_mpc.py --planner wm --ip 127.0.0.1 \\"
echo "           --ckpt /workspace/results/20260603_160326-NanoWM-B-2-F4S10-lekiwi/checkpoints/across_timesteps/epoch=13-step=8000.ckpt \\"
echo "           --nanowm-src /workspace/NanoNAV/external/nanowm/src --goal goals/run1/goal.png"
[ "$RERUN_PORT" != "0" ] && echo "         (rerun: start the viewer on the Mac, add --rerun --rerun-addr 127.0.0.1:${RERUN_PORT} on the pod)"
echo "[tunnel] Ctrl-C to tear down. Keep this terminal open."

if command -v autossh >/dev/null 2>&1; then
  echo "[tunnel] using autossh (auto-reconnect)."
  exec autossh -M 0 "${SSH_OPTS[@]}" "${FORWARDS[@]}" "${TARGET}"
else
  echo "[tunnel] autossh not found — plain ssh in a retry loop (brew install autossh for auto-reconnect)."
  trap 'echo; echo "[tunnel] torn down."; exit 0' INT TERM
  while true; do
    ssh "${SSH_OPTS[@]}" "${FORWARDS[@]}" "${TARGET}" || true
    echo "[tunnel] ssh exited; reconnecting in 2s (Ctrl-C to stop)…"
    sleep 2
  done
fi
