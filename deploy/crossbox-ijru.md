# Cross-box unconscious / federation proof (ijru = operator-B) — STAGED, deferred

The last open piece of Fork A: a **true cross-box run** — a 2nd operator's box (`ijru`) serving part of
a chain over the mesh. **Deferred** 2026-06-16 (Biswa + the federation lane): `ijru` is doing Phase 2
(verified distro PULL) first, not serving yet. Everything below is staged so wiring it is fast.

## The proof (what we run when ijru is up)
NOT "ijru serves the unconscious" — the firewall forbids that, by design:
- `prithvi-unconscious` is unlisted → **strictest tier = self**. `ijru` = **trusted** < self → **firewalled OUT**.
  The deeper-brain stays on Prithvi's own nodes. Proving ijru is *excluded* from it is half the proof.
- So the cross-box span uses a **general/public model** ijru's trusted tier satisfies — e.g. `general-7b`
  (`known` in `trisul/infra/control-plane/models.tsv`; trusted ≥ known ✓). Hub (self) + ijru (trusted)
  both eligible → the chain spans both boxes → a real token flows across two operators.

Full federation thesis in one run: **shared compute for shareable models, sovereign isolation for the self.**

## Prerequisites (NOT yet met)
1. **ijru box online + rostered as a WORKER** (not just the operator key). Today ijru is an operator row
   with coord `-` (no serving address). Needs the worker-join path (federation/onboarding lane):
   `worker-invite trusted` → ijru runs `/worker.sh` → generates its own key → rostered with a real
   `host:port` serving coord reachable from the hub over the mesh/relay.
2. **Roster approach = SERVE-LOCAL** (federation lane's call): add ijru's worker row to a serve-local
   `~/.nakshatra/crossbox-roster.tsv` (copy pubkey+coord from the junction roster). Do NOT point the serve
   at the shared VPS roster, and do NOT `scp` git over it — until the **roster-merge-on-deploy** that
   preserves the VPS-only self-join rows (3 stranger workers + 2 models) is built. That merge is the
   federation lane's prerequisite for the junction path; unbuilt as of 2026-06-16.

## Steps (when prereqs met)
1. `~/.nakshatra/crossbox-roster.tsv`:
   ```
   pk-uncon-a  unconscious-a  me    self     t-self  127.0.0.1:5540     # hub self slot
   <ijru-worker-pubkey>  ijru-gpu  ijru  trusted  t-ijru  <ijru-host>:<port>   # the 2nd operator's worker
   ```
2. `~/.nakshatra/serve_models.crossbox.yaml`: model `general-7b`, `from_roster: true`,
   `package: ~/.nakshatra/packages/dsr1-llama8b` (weights immaterial — proving the path/firewall),
   `hidden_size: 4096`, `num_layers: 32`.
3. Start a test serve (`:11699`) with `MESH_PEERS=~/.nakshatra/crossbox-roster.tsv` +
   `NAKSHATRA_LIFECYCLE_ROSTER_MODEL=general-7b` (+ HIDDEN_SIZE/NUM_LAYERS/PACKAGE/DAEMON_BIN/PYTHON_BIN).
   The hub slot is summoned locally by `RosterWorkerController`; **ijru's slot is REMOTE → the controller
   skips it** (already coded) — ijru's own worker (from worker-join) serves it. client.py dials ijru's coord.
4. Verify: a real token flows through the hub+ijru chain (`general-7b`), AND `prithvi-unconscious`
   still generates a chain with ONLY the hub (ijru firewalled out). Both = the complete proof.

## ijru-side worker launch (the commands to relay to Biswa)
ijru has the python repo (cloned) + is rostered trusted. To become a meshd-tunneled serving worker it
needs FOUR things. **It is NOT one command — (B) the daemon build is the gating step.**

**(A) Python env** (deps; the worker spawns the daemon, needs no llama_cpp python):
```
python3 -m venv ~/nks-venv && ~/nks-venv/bin/pip install grpcio protobuf cryptography pyyaml gguf numpy
```

**(B) THE DAEMON — `llama-nakshatra-worker` — NOT in the repo; BUILD it on ijru.**
ijru = **Linux x86-64 + NVIDIA** (confirmed 2026-06-17) → **CUDA build**. The hub's binary is gfx1201-ROCm
and won't run elsewhere, but the partial-load patches are backend-agnostic — the only difference is the
`-DGGML_CUDA=ON` configure flag. **A verified, idempotent build script is committed: `deploy/build-ijru-cuda.sh`.**
On ijru, from its nakshatra checkout: `nvcc --version` (CUDA toolkit must be installed) then
`bash deploy/build-ijru-cuda.sh` → produces `~/llama.cpp/build/bin/llama-nakshatra-worker`. The script
pins llama.cpp to base `c46583b` (tag b8445), applies the 5 Llama m4 partial-load patches (verified clean
at `-p4` on that base 2026-06-17; NOT the qwen3moe patches — DeepSeek-R1-Distill-Llama is Llama arch),
drops the fabric example + registers it, CUDA-configures and builds. Override the SM arch with
`CUDA_ARCH=86 bash deploy/build-ijru-cuda.sh` if `native` autodetect fails. The patch set lives at
`experiments/v0.0/m4_patches/`; the worker source at `experiments/v0.0/{worker_daemon.cpp,shm_ring.hpp}`.
⚠ **drift-class (CORRECTED 2026-06-17 — it is NOT a build string).** `drift_compatible`
(`scripts/recovery/drift_aware.py`) compares a **behavioral fingerprint**, not an engine label: the
hub's live class is `prithvi-q8@gauge1:327418908285` — a hash of the **greedy token-id sequence** the
node produces for a fixed canonical prompt (`scripts/discovery/drift_gauge.py`). Per
`docs/cross-machine-validation.md`, two boxes with byte-identical weights but different CPUs/builds
still diverge on a close-call argmax → different fingerprint → refused (a bit-deterministic chain
needs same-class nodes). **Consequence:** a heterogeneous ijru (CPU, or any non-gfx1201 build) will
**almost certainly MISMATCH** the hub's fingerprint → meshd skips it (`drift_aware.py:38`,
`meshd.py:159`). So for the first proof the hub side **MUST run with drift-class UNSET** (legacy branch
`if not primary_class: return True` → any peer allowed) — it is **required**, not merely recommended,
for a heterogeneous box. Same-class production chaining across different hardware is a genuine open
limitation (the cross-machine-determinism finding), not a config flag. **Hub-side UNSET = staged below.**

**(C) The model slice** — ijru self-provisions via `--package-url`. **⚠ The hub is NOT on the VPS relay
net** (hub = `10.42.0.1` home-mesh + `10.0.0.210` LAN; ijru = `10.51.0.14` VPS junction client — no shared
L3). So `http.server` on a hub addr is **not reachable from ijru** without routing over the junction.
**Preferred for ijru = SCP the package dir + use a LOCAL `--package-url` path** (the slicer accepts a
local dir):
```
# from the HUB (over the existing junction/relay or any reachable hop), one-time:
scp -r ~/.nakshatra/packages/dsr1-llama8b  ijru:~/.nakshatra/packages/dsr1-llama8b
# ijru worker then uses:  --package-url ~/.nakshatra/packages/dsr1-llama8b   (local path, no http)
```
(If a hub addr IS routable from ijru later, `python3 -m http.server 8099 --bind <routable-hub-addr>` +
`--package-url http://<addr>:8099/dsr1-llama8b` also works — but scp is the robust default for a NAT'd box.)

**(D) meshd publish + worker run** — so the hub's meshd discovers + tunnels ijru's worker. ijru's layer
range is assigned by ME at chain-build time (proof: hub=[0,16) first, ijru=[16,32) last, general-7b/32L):
```
# meshd (publish to the SHARED cross-NAT rendezvous — NOT the hub's local 127.0.0.1:51820; the mesh
# lane owns this addr — the VPS relay/junction). Same --mesh-id (prithvi-q8) as the hub.
# ⚠ OMIT --drift-class on the FIRST proof: ijru's CUDA build won't match the hub's behavioral
#   fingerprint (prithvi-q8@gauge1:…); the hub side runs drift-UNSET so this side can too.
~/nks-venv/bin/python ~/nakshatra/scripts/mesh/meshd.py --relay-dir ~/.nakshatra/relay \
  --rendezvous <SHARED-VPS-RENDEZVOUS:PORT> --worker-addr 127.0.0.1:5570 \
  --mesh-id prithvi-q8   # & (background) — no --drift-class for the proof
# the worker (last slot [16,32); NVIDIA → offload with --n-gpu-layers 99):
NAKSHATRA_TLS_REQUIRED=false NAKSHATRA_AUTH_REQUIRED=false NAKSHATRA_REFUSE_UNREGISTERED_PEERS=false \
~/nks-venv/bin/python ~/nakshatra/scripts/worker.py --port 5570 \
  --sub-gguf ~/.nakshatra/ijru-L16-32.gguf --package-url ~/.nakshatra/packages/dsr1-llama8b \
  --mode last --layer-start 16 --layer-end 32 --model-id general-7b \
  --daemon-bin ~/llama.cpp/build/bin/llama-nakshatra-worker --n-ctx 2048 --n-gpu-layers 99 --node-id ijru-w
```
Then ping me: the hub's meshd tunnels ijru's worker to a local `127.0.0.1:<local_port>`; I put that port
in the chain as the `last` worker (hub serves `first` [0,16)), run client.py → the real two-box token.

**Cross-lane prerequisites (NOT serve-layers):** (1) the SHARED cross-NAT rendezvous — DIAGNOSED FULLY
2026-06-17 (see below); (2) ijru's daemon build — CLOSED (`build-ijru-cuda.sh`). Data-plane de-risked
(`257ff02`).

### Gate (b) FULLY DIAGNOSED 2026-06-17 — it's a firewall boundary, not a missing address
Topology: relay VPS `45.63.109.137`, roaming-client pool `10.51.0.0/24` (relay self `10.51.0.1`),
bridged into the home mesh `10.42.0.0/24` by the **Pi** (`10.42.0.3`, holds both nets:
prithvi-wg0 + wg-egress). ijru = roaming client `10.51.0.14`.
- The hub's nakshatra-relay (`relay.py`, the rendezvous) ALREADY listens on `*:51820` → reachable at
  `10.42.0.1:51820`; meshd merely *connects* to `127.0.0.1`. Routing hub↔ijru exists via the Pi.
- **BUT** the Pi's `PRITHVI_RELAY` FORWARD chain scopes roaming clients to ONLY `10.42.0.2:80,443`
  (kali apps) + `10.42.0.3:53` (DNS), else **DROP**. So **ijru CANNOT reach `10.42.0.1:51820` today** —
  the rendezvous is firewalled off by design (the VPS-relay multi-tenant scoping). This is gate (b).
- **Two ways to open it (Biswa's security call):**
  - **(b2, RECOMMENDED, sovereign-correct):** run `relay.py` on the VPS `45.63.109.137` as the neutral
    blind-L4 junction (it's an *untrusted byte-forwarder* by design; the meshd tunnel is E2E-encrypted
    over it, so the VPS sees only ciphertext). Both hub (route via wg) + ijru (relay client) reach it
    WITHOUT exposing any home-mesh port to the roaming net. Matches `infra/nakshatra-junction`'s
    "a blind-L4 junction must be listening". Rendezvous = `45.63.109.137:<port>`. Requires a VPS deploy.
  - **(b1, faster, weaker):** add one scoped rule to the Pi's `PRITHVI_RELAY`:
    `-d 10.42.0.1/32 -p tcp --dport 51820 -j ACCEPT` (insert before the trailing DROP). Then rendezvous
    = `10.42.0.1:51820`. Pokes a single hole into the home mesh for the roaming net — reversible, but
    weakens the boundary the audit established. Acceptable for a one-off proof; revert after.

## HUB-SIDE proof prep (my lane — staged, run at proof time; reversible)
All three are non-destructive and leave the LIVE consumer meshd + the live unconscious untouched.

1. **drift-UNSET meshd for the proof.** Do NOT edit the live `nakshatra-meshd.service` (it's the
   consumer-only daemon, drift-class `prithvi-q8@gauge1:327418908285`). Run a SEPARATE proof meshd
   pointed at the shared VPS rendezvous with drift-class omitted (→ legacy branch admits any peer):
   ```
   ~/nakshatra/.venv/bin/python ~/nakshatra/scripts/mesh/meshd.py \
     --mesh-id prithvi-q8 --relay-dir ~/.nakshatra/relay \
     --rendezvous <SHARED-VPS-RENDEZVOUS:PORT>  --worker-addr 127.0.0.1:5540 \
     --endpoint hub-proof   # NO --drift-class  → drift_compatible() returns True for any ijru class
   ```
   (`--worker-addr` = the hub's first-half worker so meshd advertises it; the proof chain itself is
   wired by me via client.py once meshd reports ijru's tunnel local-port.)
2. **Serve-local crossbox roster** (`~/.nakshatra/crossbox-roster.tsv`) + `general-7b` from_roster model
   yaml — already specified in "Steps" above; touches neither git nor the VPS roster.
3. **Chain wiring** — when meshd logs ijru's tunnel up (`127.0.0.1:<local_port>`), I put that port as
   the `last` worker [16,32), hub serves `first` [0,16), run client.py → real two-box token. Minutes.

## State
- Mechanism: 100% built + proven single-box (slices 1–4b, live cutover `a9354f3`). Only the real 2nd
  serving box is missing.
- Memory: `project_serve_layers_next`. Coordination thread: `prithvi/INBOX.md` (2026-06-16).
