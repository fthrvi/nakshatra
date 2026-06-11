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

## §10 — PASSED end-to-end (2026-06-11)

Ran the full gate on this box against the 1B (split `[0,8)`+`[8,16)`):

```
discover → pin → negotiate(control/v1) → self-provision(signed pkg) → ' Paris'
   Part A: ✅            Part B: ✅
```

- **Reference** (`llama-simple`, full model): `The capital of France is Paris. The Eiffel Tower is`
- **Chain** (2 workers, each self-provisioned its slice from a SIGNED package):
  `step 1: id=12366 ' Paris'` → `The capital of France is Paris. The Eiffel Tower is`
- **Byte-identical**, and `12366 ' Paris'` is exactly the v0.1 parity token. The
  client logged `caps=[… control/v1]` and `OK: contiguous coverage of [0,16)` —
  P4 negotiation live.

### Bug this run found + fixed (weight-tied models)
Small Llamas (1B/3B) **tie** `output.weight` to `token_embd.weight` (no separate
output head). The slice-cutters dropped `token_embd` from non-first slices, so a
tied model's LAST worker couldn't find its output weights — a latent bug in
`partial_gguf.py` too, not just P2. Fixed in two places:
- **packager** (`tied_embeddings` flag → the last worker also fetches the
  embeddings fragment; assembler dedups so the whole-model slice isn't doubled),
- **daemon** (`m4_patches/llama-model.cpp.patch`: the tied-output fallback loads
  `token_embd` for real on a last-only worker instead of aliasing a tensor that
  was never loaded).
Larger models (the live 8B/70B cluster) have a real `output.weight` and never hit
this.
