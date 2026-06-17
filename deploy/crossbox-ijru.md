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

**(B) ⚠ THE DAEMON — `llama-nakshatra-worker` — NOT in the repo; must be BUILT for ijru's arch.**
The hub's binary is gfx1201-ROCm and won't run elsewhere. ijru builds the patched llama.cpp (partial-load)
for ITS backend (CPU if no GPU). **Need ijru's OS/arch/GPU to give the exact recipe** (Linux-x86-CPU vs
macOS-Metal differ; macOS uses the lab-Mac recipe in [[project_l2_to_l4_connected]] — copy build +
`install_name_tool -add_rpath` + `codesign`). The patch lives at `experiments/v0.0/m4_patches/`.
⚠ **drift-class:** meshd's `drift_compatible` is EXACT-MATCH — ijru's build's drift-class must equal the
hub's (`rocm-gfx1201-...`) OR **run the hub side with drift-class UNSET (legacy mode → any peer allowed)
for the first proof.** Recommend UNSET for the first cross-box token; align builds for production.

**(C) The model slice** — ijru self-provisions via `--package-url`. The package must be mesh-reachable
from ijru (relay client `10.51.0.14`). Hub serves it on a relay-reachable addr:
```
# on the HUB:  cd ~/.nakshatra/packages && python3 -m http.server 8099 --bind <hub-relay-addr>
# ijru worker: --package-url http://<hub-relay-addr>:8099/dsr1-llama8b
```
(or scp the package dir to ijru and use a local path).

**(D) meshd publish + worker run** — so the hub's meshd discovers + tunnels ijru's worker. ijru's layer
range is assigned by ME at chain-build time (proof: hub=[0,16) first, ijru=[16,32) last, general-7b/32L):
```
# meshd (publish to the SHARED cross-NAT rendezvous — NOT the hub's local 127.0.0.1:51820; the mesh
# lane owns this addr — likely the VPS relay/junction). Same --mesh-id as the hub.
~/nks-venv/bin/python ~/nakshatra/scripts/mesh/meshd.py --relay-dir ~/.nakshatra/relay \
  --rendezvous <SHARED-RENDEZVOUS-HOST:PORT> --worker-addr 127.0.0.1:5570 \
  --mesh-id <hub-mesh-id> --drift-class <ijru-build-class>   # &  (background)
# the worker (last slot [16,32); --n-gpu-layers 0 if CPU):
NAKSHATRA_TLS_REQUIRED=false NAKSHATRA_AUTH_REQUIRED=false NAKSHATRA_REFUSE_UNREGISTERED_PEERS=false \
~/nks-venv/bin/python ~/nakshatra/scripts/worker.py --port 5570 \
  --sub-gguf ~/.nakshatra/ijru-L16-32.gguf --package-url http://<hub-relay-addr>:8099/dsr1-llama8b \
  --mode last --layer-start 16 --layer-end 32 --model-id general-7b \
  --daemon-bin ~/<built-daemon-path> --n-ctx 2048 --n-gpu-layers 0 --node-id ijru-w
```
Then ping me: the hub's meshd tunnels ijru's worker to a local `127.0.0.1:<local_port>`; I put that port
in the chain as the `last` worker (hub serves `first` [0,16)), run client.py → the real two-box token.

**Cross-lane prerequisites (NOT serve-layers):** (1) the SHARED cross-NAT rendezvous addr both meshds
use — mesh lane (hub meshd currently local-only); (2) ijru's daemon build matching drift-class, or
hub-side drift UNSET for the proof. My data-plane is de-risked (`257ff02`); these two gate the run.

## State
- Mechanism: 100% built + proven single-box (slices 1–4b, live cutover `a9354f3`). Only the real 2nd
  serving box is missing.
- Memory: `project_serve_layers_next`. Coordination thread: `prithvi/INBOX.md` (2026-06-16).
