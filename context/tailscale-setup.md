# Pod ↔ LeKiwi bridge (Stage 6b.3)

Goal: let the **GPU pod** (NanoWM + CEM) reach the **LeKiwi Pi** (lerobot host, LAN `10.0.0.125`) so
`scripts/lekiwi_mpc.py --planner wm` runs as the lerobot `LeKiwiClient` in the same process as the WM
(the 6b design — no bespoke inference API). The pod and the robot are on different networks; this doc covers
the options to bridge them so lerobot's ZMQ (cmd `5555` + observations `5556`) reaches the Pi.

> **Status:** the pod can capture goals from the Mac today ([[planning]] 6b.4). The closed loop is gated on
> this bridge. See [[roadmap]] Stage 6b, [[experiment-log]] 6b.2.
>
> **Recommended path = SSH reverse tunnel over RunPod's exposed TCP port** (below): no TUN, no new code,
> reuses the validated pod-as-client design. The Tailscale paths are alternatives; kernel mode is **blocked**
> on this pod (no `/dev/net/tun`).

---

## ✅ Recommended: RunPod TCP port + SSH reverse tunnel (no TUN, no code)

SSH needs no TUN, so this sidesteps the blocker entirely, and the pod stays the lerobot client exactly as
6b.2/6b.3 validated — just pointed at `127.0.0.1`. The Mac (already on the LAN with the Pi, 6b.0/6b.1) dials
into the pod over RunPod's SSH and **reverse-forwards** the two ZMQ ports back out to the Pi:

```
   Pod (GPU)                          Mac (dials out, on LAN)          Pi robot
   lekiwi_mpc --ip 127.0.0.1   ──►   ssh -R 5555 / -R 5556    ──►   10.0.0.125:5555/5556
   (connects to its own localhost)   (relays back over tunnel)       (lerobot lekiwi host)
```

Facts confirmed on this pod (2026-06-05): sshd running, `AllowTcpForwarding` at default `yes`, public IP
`31.24.80.34`, `RUNPOD_POD_ID=ctmyc6ld7ht5zd`; lerobot `LeKiwiClientConfig` → `port_zmq_cmd=5555`,
`port_zmq_observations=5556`. lerobot's client only ever *connects out* (PUSH cmd / SUB obs), so both ports
ride the reverse tunnel — identical to the direct connect that worked in 6b.0/6b.1.

**1. On the Mac** — keep this terminal open for the whole session. One command (`scripts/tunnel_up.sh`
wraps the `ssh -R` with keepalives + autossh auto-reconnect; get `<ip> <port>` from the RunPod console →
**Connect → SSH over exposed TCP**, NOT the `ssh.runpod.io` proxy — the proxy blocks `-R`/`-L`):
```bash
scripts/tunnel_up.sh --pod-host <pod-public-ip> --pod-port <pod-ssh-tcp-port>
# (defaults: --pi-ip 10.0.0.125, --user root, reverse-forwards 5555 cmd / 5556 obs / 9876 rerun)
```
Equivalent raw command if you'd rather not use the helper:
```bash
ssh -N -R 5555:10.0.0.125:5555 -R 5556:10.0.0.125:5556 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  root@<pod-public-ip> -p <pod-ssh-tcp-port>
```
(`ServerAliveInterval` keeps the tunnel warm during the ~8 s CEM compute, when no traffic flows, so NAT
doesn't reap it.)

**2. On the pod** — the closed loop against the tunnelled robot (`--ip 127.0.0.1`):
```bash
cd /workspace/NanoNAV
python scripts/lekiwi_mpc.py --planner wm \
  --ip 127.0.0.1 \
  --ckpt /workspace/results/20260603_160326-NanoWM-B-2-F4S10-lekiwi/checkpoints/across_timesteps/epoch=13-step=8000.ckpt \
  --nanowm-src /workspace/NanoNAV/external/nanowm/src \
  --goal goals/run1/goal.png
```
The integrate_se2 action stats are already wired as `--action-mean/--action-std` defaults (6b.3 pre-wiring),
so this is turnkey. Optional rerun telemetry can stream to the Mac viewer over the same SSH (`-R 9876:...`).

**Requirements / gotchas:**
- Use the **direct TCP SSH** endpoint (exposed port → 22), not `ssh.runpod.io` (no forwarding there).
- Your Mac's public key must be in the pod's `authorized_keys` (RunPod installs it if you added your SSH key).
- `sshd_config` must keep `AllowTcpForwarding yes` (default; confirmed here).
- Latency: observation frames go Pi→Mac (LAN)→internet→pod. Stop-and-plan pulls one frame per ~8–9 s cycle,
  so it's a non-issue; lerobot already compresses camera frames over ZMQ.
- This is **inbound-to-pod only** (RunPod TCP ports are inbound) — which is exactly why the Mac must be the
  SSH *initiator* and use `-R` (reverse), letting the pod reach the Pi without the pod dialing out.

---

## ⚠️ Tailscale prerequisite: the pod needs a TUN device

The Tailscale paths below are alternatives to the SSH tunnel. Kernel-mode Tailscale creates a `tailscale0`
interface via `/dev/net/tun`, which apps then use to reach tailnet IPs (`100.x.y.z`) transparently — what
lerobot's raw ZMQ needs.

**This pod (as probed 2026-06-05) cannot do that out of the box:**
```
$ ls -l /dev/net/tun            # -> no such file
$ mknod /dev/net/tun c 10 200   # -> Operation not permitted  (no CAP_MKNOD/NET_ADMIN; not privileged)
```
So **kernel mode will fail at `tailscale up`** here, and userspace mode does *not* transparently carry ZMQ
(see Path B). Resolve this first. Check on any fresh pod:
```bash
ls -l /dev/net/tun && echo "TUN ok (Path A)" || echo "no TUN — enable it, or use Path B/alternative"
```

**How to get TUN on RunPod:**
- Re-deploy the pod on a template/config that exposes `/dev/net/tun` (privileged / TUN-enabled). On RunPod
  this is a pod-creation setting, not something we can grant from inside the container.
- If the container has the cap but the node is missing, create the device once per boot:
  `mkdir -p /dev/net && mknod /dev/net/tun c 10 200 && chmod 600 /dev/net/tun` (only works if privileged —
  it failed here, so this pod must be re-created with TUN).

If you genuinely cannot get TUN, prefer the **Recommended alternative** below over fighting Path B.

---

## Path A — kernel mode (TUN available) — the clean path

Everything persists on `/workspace` so it survives RunPod restarts (only `/workspace` does — same reason
`tmux` lives in `/workspace/bin`). Binaries → `/workspace/bin` (already on PATH via `bin/env.sh`); node
state → `/workspace/secrets` (0700, holds the node key).

**1. Install the static binaries onto the volume (once):**
```bash
TS_VER=1.98.4   # check https://pkgs.tailscale.com/stable/ for the current amd64 tgz
curl -fsSL -o /tmp/ts.tgz https://pkgs.tailscale.com/stable/tailscale_${TS_VER}_amd64.tgz
tar -xzf /tmp/ts.tgz -C /tmp
cp /tmp/tailscale_${TS_VER}_amd64/tailscale /tmp/tailscale_${TS_VER}_amd64/tailscaled /workspace/bin/
chmod +x /workspace/bin/tailscale /workspace/bin/tailscaled
source /workspace/bin/env.sh   # puts /workspace/bin on PATH
```

**2. Start the daemon (every fresh pod) — run under tmux so it survives the shell:**
```bash
tmux new-session -d -s tailscaled \
  'tailscaled --state=/workspace/secrets/tailscaled.state \
              --socket=/var/run/tailscale/tailscaled.sock \
   > /workspace/tailscaled.log 2>&1'
```
(`--state` on the volume = the node stays authenticated across restarts; re-running step 3 is then a no-op.)

**3. Authenticate (first pod only; state restores it afterwards):**
```bash
# put TS_AUTHKEY in /workspace/secrets/env.sh (ephemeral or reusable, tagged). NEVER commit it.
source /workspace/secrets/env.sh
tailscale up --authkey="${TS_AUTHKEY}" --hostname=nanonav-pod --accept-routes=false
```
No auth key? `tailscale up --hostname=nanonav-pod` prints a login URL to paste in a browser.

**4. Put the Pi on the same tailnet** (do this once, on the Pi — it keeps its existing LAN host):
```bash
# on the Raspberry Pi:
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --hostname=lekiwi-pi
tailscale ip -4        # -> the Pi's 100.x.y.z  (use this as --ip below)
```
Verify the lerobot lekiwi **host binds all interfaces** (`tcp://*:<port>`, the default) so it accepts the
tailnet connection, not only the LAN IP.

**5. Verify pod → Pi:**
```bash
tailscale status                 # both nodes Active; note the Pi's 100.x IP
tailscale ping <pi-100.x.y.z>    # want "direct" (not "via DERP") for lowest latency
ping -c3 <pi-100.x.y.z>
```
DERP-relayed still works — **stop-and-plan tolerates it** (~8–9 s/cycle, one frame per cycle) — but direct
is better for the observation-image hop.

**6. Run the closed loop** (point `--ip` at the Pi's tailscale IP instead of the LAN `10.0.0.125`):
```bash
cd /workspace/NanoNAV
python scripts/lekiwi_mpc.py --planner wm \
  --ip <pi-100.x.y.z> \
  --ckpt /workspace/results/20260603_160326-NanoWM-B-2-F4S10-lekiwi/checkpoints/across_timesteps/epoch=13-step=8000.ckpt \
  --nanowm-src /workspace/NanoNAV/external/nanowm/src \
  --goal goals/run1/goal.png \
  --rerun --rerun-addr <mac-tailscale-ip>:9876   # optional live telemetry to the Mac viewer
```
Tailscale carries both ZMQ ports (cmd + observations) over the tunnel — no port-forwarding. The integrate_se2
action stats are already wired as `--action-mean/--action-std` defaults (6b.3 pre-wiring), so this is turnkey.

---

## Path B — userspace fallback (no TUN) — fragile, last resort

```bash
tmux new-session -d -s tailscaled \
  'tailscaled --tun=userspace-networking --socks5-server=localhost:1055 \
              --state=/workspace/secrets/tailscaled.state > /workspace/tailscaled.log 2>&1'
tailscale up --authkey="${TS_AUTHKEY}" --hostname=nanonav-pod
```
**The catch:** userspace mode creates **no** `tailscale0` interface. Tailnet traffic is reachable only via the
SOCKS5 proxy at `localhost:1055`. lerobot's ZMQ has no SOCKS support, so `LeKiwiClient(remote_ip=100.x)` will
**not** connect. You'd have to forward each ZMQ port (cmd + observations) through the proxy with a SOCKS5-aware
TCP forwarder (e.g. nmap `ncat --proxy localhost:1055 --proxy-type socks5 …`; neither `ncat` nor `socat` is on
the pod, so you'd install one) and point lerobot at `127.0.0.1`. Two ports, keep-open listeners, easy to get
subtly wrong on the streaming-video port. **Treat as a stopgap; don't ship the live run on it.**

---

## Further fallback: flip the topology (Mac-as-client + pod inference server)

Only if the SSH reverse tunnel is somehow unavailable AND TUN can't be enabled. Keep the lerobot client on the
Mac (the **Mac↔robot LAN path already works**, 6b.0/6b.1, RTT ~15 ms) and move only the *inference* across:

- **Mac** runs `lekiwi_mpc.py` as the lerobot client (robot stays local LAN).
- **Pod** runs a tiny WM inference server wrapping `LekiwiPlanner` (`plan(frame, goal) -> action + viz`),
  exposed on a **RunPod TCP port** (inbound to the pod — the Mac connects in).
- The Mac's planner becomes a thin `--planner remote` stub POSTing `(frame, goal)` to that port.

Cost: a remote-planner stub on the Mac + a server on the pod — strictly more code than the SSH tunnel, which
needs none. Prefer the SSH reverse tunnel; reach for this only if you want the pod fully decoupled from the
robot's transport. Either way the validated 6b.2 engine is reused unchanged. Spec as 6b.3-alt before building.

---

## Restart persistence (RunPod)

Each fresh pod: `/workspace` survives (binaries + `tailscaled.state`), but the **`tailscaled` process is gone**
and **`/dev/net/tun` is gone** (re-create per Path A prereq if privileged). Re-run Path A step 2 (daemon); the
persisted state means step 3 (auth) is skipped — the node comes back authenticated. Consider a
`/workspace/bin/tailscale-up.sh` that does TUN-create + daemon-launch idempotently.

## Security

- `TS_AUTHKEY` lives in `/workspace/secrets/env.sh` (same place as the HF/GH/WANDB keys); **never commit it**.
- Use an **ephemeral, pre-authorized, tagged** key from the admin console (e.g. `tag:nanonav-pod`) so a torn-down
  pod's node auto-expires and ACLs scope it to just the Pi.
- The node key in `tailscaled.state` is sensitive — `secrets/` is `0700`; keep it there, not in the repo.
