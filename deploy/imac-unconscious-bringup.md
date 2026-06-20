# iMac CPU workers → Prithvi's unconscious (bring-up runbook)

State after 2026-06-19: all 3 iMacs (mac3/mac4/bishwa-mac) are on the home mesh, fully provisioned
(cmake + the nakshatra `llama-nakshatra-worker` daemon built w/ Metal, Python venv, the full
DSR1-8B Q4 gguf, and a packaged `~/pkg`). **PROVEN:** Mac4 serves DSR1-8B via a 2-node nakshatra
**CPU** chain (coherent output, 5.8 tok/s). Reach: `ssh mac3|mac4|bishwa-mac` (via Pi jump host).

**Why CPU:** Metal is numerically broken on these Intel-iMac Radeon Pro 5700 XT GPUs (vanilla
llama.cpp garbles on Metal; CPU is correct). Run workers with `--n-gpu-layers 0`.

## The ONE thing that needs Biswa's sudo (on the GPU box, 10.42.0.1)
The unconscious coordinator (client.py, on this box) must reach the Macs at 10.51.0.x over WG, but
this box's WG peer for the Pi lacks that subnet. In a terminal ON the GPU box:

```sh
# pi peer pubkey = i2g15kFmkv4p04rAZOXV3n/y/tnaJBSecuOeT1rcyQI=  (from mesh-roster.tsv)
sudo wg set prithvi-wg0 peer i2g15kFmkv4p04rAZOXV3n/y/tnaJBSecuOeT1rcyQI= \
     allowed-ips 10.42.0.3/32,10.51.0.0/24
sudo ip route replace 10.51.0.0/24 dev prithvi-wg0
# verify: ssh mac4 from the box directly should now work without the Pi jump host
```
(The Pi side already routes/forwards mesh→client — `prithvi-mesh-door.sh` persists it.)

## Then (all from here, no Mac physical access — `ssh mac4` etc.)
1. **Run a CPU worker on each Mac** (2-node split shown; a 3-way is fine too). On each Mac, the
   worker self-provisions its slice from `~/pkg`:
   ```sh
   ssh mac4 '~/nakshatra-venv/bin/python ~/nakshatra-scripts/worker.py \
     --port 5540 --sub-gguf ~/slices/a.gguf --package-url ~/pkg --mode first \
     --layer-start 0 --layer-end 16 --model-id dsr1-8b \
     --daemon-bin ~/llama-nak/build/bin/llama-nakshatra-worker \
     --n-ctx 2048 --n-gpu-layers 0 --node-id mac4-a --no-file-server --skip-sha256 &'
   # ...and a `last` worker (layers 16-32) on another Mac (or same Mac, port 5541).
   ```
   NOTE: cross-Mac (worker→worker push, or client→multiple Macs) needs the Pi to forward
   client↔client — NOT enabled (the VPS-reflector path made it asymmetric). For a multi-Mac chain,
   either (a) run all workers' ports on ONE Mac, or (b) put the coordinator on the Pi, or (c) make
   the Macs full mesh nodes (10.42.0.x) instead of relay clients. Single-Mac 2-node works today.
2. **Roster:** add the Mac worker(s) to `~/.nakshatra/unconscious-roster.tsv` as `tier=self`
   (coord = `10.51.0.17:5540` etc.), and flip the unconscious model to `from_roster` + enable the
   lifecycle gate (`spec.lifecycle_gate=True` / `NAKSHATRA_UNCONSCIOUS_LIFECYCLE`). The lifecycle
   (`scripts/fabric/unconscious_lifecycle.py`) then auto-places onto the self-tier Mac workers.
3. The hub toggle (`/unconscious` on infra.html) reflects the placement.

## Cleanup if needed
- Stop Mac workers: `ssh mac4 'pkill -f worker.py; pkill -f llama-nakshatra-worker'`
- Pi routes are reboot-persistent (in `/usr/local/sbin/prithvi-mesh-door.sh`).
