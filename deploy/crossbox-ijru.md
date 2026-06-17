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

## State
- Mechanism: 100% built + proven single-box (slices 1–4b, live cutover `a9354f3`). Only the real 2nd
  serving box is missing.
- Memory: `project_serve_layers_next`. Coordination thread: `prithvi/INBOX.md` (2026-06-16).
