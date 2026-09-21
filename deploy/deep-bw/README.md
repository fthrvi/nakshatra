# deep-bw — the blackwell + ijru `qwen3-30b-q3` chain, alive on demand

Snapshot of the live config (was NOT in git until 2026-09-21). No secrets in here: paths, overlay IPs and key *ids* only — keys live in
each node's `~/.nakshatra/keys` and the pillar.

```
request ─► nakshatra-deep-bw.service (hub :11601, scripts/nakshatra_serve.py + serve_lifecycle.py)
             │ cold?  ssh-summon both workers  (lifecycle-deep30b-bw.json → nks-q3a / nks-q3b)
             │ ready = each worker's Info() answers (a TLS handshake or TCP accept is NOT readiness)
             │ POST /lease to the Sthambha pillar (pi 10.42.0.3:7777) → planned chain → client.py signed gRPC
             └ idle 600 s  → stop both workers (pillar lease expiry, or the local idle clock if no lease was granted)
```

| file | lives at |
|---|---|
| `nakshatra-deep-bw.service` | hub `~/.config/systemd/user/` (`systemctl --user daemon-reload && systemctl --user enable --now nakshatra-deep-bw`) |
| `lifecycle-deep30b-bw.json`, `serve_models.deepbw.yaml` | hub `~/.nakshatra/` |
| `ssh-config.ijru-nks.snippet` | hub `~/.ssh/config` |
| `nks-q3a.blackwell.sh` | blackwell WSL `~/nks-q3a.sh` |
| `nks-wsl-portproxy.blackwell.ps1` | blackwell Windows `C:\ProgramData\nakshatra\nks-wsl-portproxy.ps1` |
| `nks-q3b.ijru.sh` | ijru `~/nks-q3b.sh` |
| `deploy-workers.sh` | run from any checkout: pushes the worker code to both nodes and verifies sha256 of every file the worker imports |

## Gotchas (each cost real time)
- **Worker code drifts.** The nodes run hand-copied trees. After ANY change to `scripts/worker.py` (or a `nakshatra_*` module it imports) run
  `bash deploy/deep-bw/deploy-workers.sh`; `--check` verifies without touching anything.
- **blackwell = Windows + WSL2 (NAT).** The Windows localhost relay does not forward the worker's IPv6-wildcard `:5562`, so
  `netsh portproxy 10.42.0.7:5562 → <WSL eth0 IP>` is required; the WSL IP changes on every WSL restart, so the launch command re-points it
  first. The ssh login shell is PowerShell: pipe scripts into `wsl -e bash -l -s` (`bash -l` is needed for `systemd-run --user`).
- **A worker binds its port before it can authenticate anyone** (sha256 of a 10 GB slice + pillar registration ≈ 5–8 s). It answers Info
  `UNAVAILABLE "worker starting: <why>"` until registered AND its peer-key cache holds its own key; `/healthz` shows `starting`.
- **blackwell's 12 GB cannot hold this worker (~6.5 GB) and the Ollama coder (10 GB) together.** The chain is on-demand only.
  ⚠️ Nothing yet stops a summon while Ollama holds the card (open item: refuse in `nks-q3a.sh start` when free VRAM < 7 GB).
- The hub's `launch`/`stop` strings are arbitrary shell on both nodes. Bounding that door (forced command) is a decision for the owner of
  each machine — see `infra/mesh-agent/install-agent-reach.sh` for the pattern.
