# v1.0 §10 acceptance — cluster validation run

Validates the merged P1–P4 work (`docs/v1.0-discovery-and-distribution.md` §10):
*two machines with no shared static config discover each other, mutually
authenticate, self-provision their layer range from a verified package, negotiate
the wire version, and produce the byte-identical greedy token.*

It's split into two parts because the control plane is pure-Python (runs anywhere)
but the token needs the gRPC runtime.

## Part A — control plane (runnable anywhere, no daemon)

```
python -m validate.acceptance_controlplane --package <pkg_dir> [--require-signature]
```

Proves, live: two nodes (separate Ed25519 keys, sharing **only** a relay dir)
→ signed listings → discover + rank by measured Fᵢ + **pin each other's keys**
→ ALPN version negotiation → **self-provision only the assigned range** from the
package (fail-closed SHA-256) → assembled sub-GGUFs with correct slice metadata.

Build a package first (signed):
```
python -m packaging.package_gguf <full-model.gguf> <pkg_dir> --model-id <id> --sign-worker-key
```

Part A prints `NODE_A_GGUF` / `NODE_B_GGUF` — the provisioned sub-GGUFs Part B consumes.
**Status: PASSES** on this box against the 1B (16 layers, split [0,8)+[8,16)).

## Part B — token parity (needs the gRPC runtime)

```
NODE_A_GGUF=… NODE_B_GGUF=… REF_GGUF=<full-model.gguf> \
DAEMON_BIN=<…/llama-nakshatra-worker> N_LAYERS=16 PROMPT="The capital of France is" \
./validate/cluster_token_parity.sh
```

Launches a 2-worker localhost chain on the provisioned sub-GGUFs (the P4 control-
version handshake negotiates as the client reads each worker's `Info`), runs
`client.py`, and compares the distributed first token to the single-machine
`llama-cli` reference — the v0.1 parity bar, now provisioned with **zero hand-cut
sub-GGUFs**.

### Prerequisites (the gate the script enforces, exit 2 if unmet)
1. **A venv with the runtime deps.** ✅ Set up on the trisul box at
   `nakshatra/.venv` — worker.py/client.py need only `grpcio protobuf
   cryptography numpy<2 gguf pyyaml` (NOT the full torch/hivemind setup.cfg
   stack). `source .venv/bin/activate`.
2. **The gRPC worker daemon.** ✅ Built: `cmake --build <llama.cpp>/build
   --target llama-nakshatra-worker` → `/home/prithvi/llama.cpp/build/bin/
   llama-nakshatra-worker`. **Verified loading a P2-provisioned sub-GGUF:** it
   runs the partial-load patch (`nakshatra partial-load: layers [0,8) of 16`) and
   reports `[daemon] ready`. So package→provision→daemon-load is proven against
   the real daemon.
3. **A full-model GGUF** for the reference token.

> **Known blocker on the reference step (this box):** `llama-cli` hangs in
> interactive mode even with `-no-cnv` on this llama.cpp build (a tooling quirk —
> same one that pushed the spec-decode bench to `llama-bench`). The reference
> token capture needs a non-`llama-cli` path (e.g. `llama-simple`, the server, or
> a token-id dump) before Part B runs unattended. The daemon + chain side is
> unaffected.

Workers run CPU-only (`--n-gpu-layers 0`) so the run does **not** contend with
Prithvi's GPU.

## What this run is for

It's the honest gap from PR #1: the worker/client integration paths were
compile-checked + CI-guarded but never executed live or on two machines. Part A
closes the control-plane half today; Part B closes the token half the moment a
Nakshatra runtime (venv + gRPC daemon) is available — on this box or a real
two-machine cluster.
```
two machines → discover → pin → negotiate → self-provision → ' Paris'
   Part A: ✅ here          Part B: ready, needs runtime
```
