# What joining exposes on your machine — audited, not asserted

**2026-09-04. Read the code, ran the checks. Two findings were serious and are fixed in the
same commit as this file.**

The question a stranger asks before running `nakshatra join`: *what can the network do to my
machine?* Here is the answer from the code, with the parts that were wrong called out.

## What a peer can make your worker do

Exactly seven RPCs, and none of them runs code you did not install:

| RPC | what it does on your box | can a peer abuse it? |
|---|---|---|
| `Info` | reports your slice's layer range, backend, capabilities | read-only |
| `Forward` / `Inference` | runs a tensor through **your local model slice** and returns the result | it is compute on weights you fetched; it cannot run anything else. Messages capped at 256 MB |
| `TruncateKV` | drops speculative tokens from your KV cache | bounded by the cache |
| `Sleep` / `Wake` | releases / reloads the GPU | affects only your daemon |
| `SignParticipation` | signs, with your key, the span **you** believe you served | signs your own belief, never the caller's claim |

**No RPC carries a filesystem path from the peer.** The worker never `exec`s anything a peer
sent; its only subprocesses are the llama daemon it started and `partial_gguf` on files it
already holds, path-guarded by `validate_slice_path()`.

**What the worker touches on disk:** the slice it fetched, its own Ed25519 key, its log.
It reads no home directory, opens no user files. Its outbound connections are to pinned peers
only — and the "push to next stage" address a peer supplies is refused unless that address is
in the registered roster (`is_registered_address`), so a peer cannot use your worker as a
probe against `127.0.0.1:22`.

**The optional file server** (`--file-server-port`, off by default) serves model slices —
public weights — from one directory, path-guarded. It leaks nothing you did not choose to host.

## ⛔ Finding 1 — the join path produced UNAUTHENTICATED workers

`resolve_auth_required()` has this truth table:

    env NAKSHATRA_AUTH_REQUIRED    pillar_url    result
    unset                          ""            False   ← "Mode A legacy"
    unset                          set           True

The join path — `servecmd.serve_argv`, `observe`, `act` — contained **zero references to
auth or a pillar URL.** So every node produced by `nakshatra join` came up in Mode A: TLS on
the wire, but **no peer authentication** and, because `peer_resolver` is `None` in that mode,
**the push-address SSRF gate switched off too.** Anyone who could reach the port could call
`Forward` on a stranger's GPU for free, and could aim its next-hop push anywhere.

Fixed: the join path now sets `NAKSHATRA_AUTH_REQUIRED=true` in the daemon's environment and
passes the coordinator as `--pillar-url`. A joined node authenticates every non-`Info` call.

## ⛔ Finding 2 — the provisioner executed unverified downloads

`provision-worker.sh` fetched four archives over HTTPS and extracted or ran them with **no
hash, no signature, nothing:**

    cmake.tgz     verification: none
    stack.tgz     verification: none   ← contains the engine it BUILDS AND RUNS
    scripts.tgz   verification: none   ← contains worker.py, which it EXECUTES
    k.deb         verification: none

TLS was the only integrity guarantee. A compromised `prithviloka.net`, a mis-issued
certificate, or a proxy that terminates TLS (common on corporate and university networks)
meant **remote code execution on every machine that ever joined**, silently, forever. This is
the single largest exposure a joining machine had, and it is the one a careful stranger would
find first.

Fixed: the two archives that contain code (`stack.tgz`, `scripts.tgz`) are now verified
against `WORKER_STACK_SHA256` / `WORKER_SCRIPTS_SHA256` **before extraction, fail-closed**.
If the expected hash is not configured the provisioner **refuses** rather than proceeding —
a missing hash is not permission. `WORKER_ALLOW_UNVERIFIED=1` exists for development and
prints a warning that cannot be missed.

⚠️ This breaks the current hosted flow until hashes are published alongside the archives.
That is the correct order: code that runs on strangers' machines does not ship before its
hash does.

## What remains, honestly

- **The model slice is untrusted input to llama.cpp.** A malicious GGUF exploiting a parser
  bug is a real class of attack. Mitigation today: slices come from **signed packages**
  (Ed25519, `--require-signature`) and every fragment is SHA-verified. That bounds it to
  "a package signer you pinned" — not zero.
- **Activations are visible to the worker that computes them.** Encryption hides prompts from
  relays; it can never hide them from the node doing the math. This is why
  `eligible_workers()` keeps sensitive models on trusted tiers. A stranger's node serves
  general/public inference only. Make sure that stays true.
- **A joined node is a process running as your user.** It is bounded by what the code does,
  not by a sandbox. Running it in a container or a dedicated user is a reasonable operator
  choice, and `containerplan` exists for exactly that.

## The invariant

**We use the compute. We never touch the owner's data. We never keep the owner waiting.**
The first two are enforced by what the worker *is* — a slice loader that speaks gRPC to
pinned peers. This audit found the two places that promise was weaker than it claimed, and
closed them.
