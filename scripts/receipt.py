"""
receipt.py — verifiable receipt of a distributed inference run.

Speed-stack finding #20 (trisul research 2026-06-19). A per-run JSON artifact a skeptic can
INDEPENDENTLY check, mirroring shard's "proofs against fakes": the run was (1) served by
distinct machines, (2) really distributed, (3) produced this exact output, (4) re-runnable
against a named engine build. Today nakshatra has worker attestation + build-provenance but
no per-run receipt — this is it.

HONESTY SCOPE (what a receipt can and cannot prove — verify_receipt never overclaims):
  • output_sha256 — THIRD-PARTY CHECKABLE, zero trust: anyone recomputes the hash over the
    token list and confirms it matches.
  • structural/math consistency — CHECKABLE: token count, tok/s math, monotonic timestamps,
    contiguous layer-map [0, N), worker distinctness (ids AND TLS SPKI / address).
  • reproducibility — scoped to "WITHIN-ENGINE, same-provenance re-runnable": re-running the
    named engine build (engine_provenance) reproduces this output. NOT "matches your laptop's
    HF decode token-for-token" — a quantized model decoded batched-vs-single-token rounds FP
    differently across engines/backends, so cross-engine bit-exactness is not claimable.
  • model identity — ASSERTED via model_id only. The gRPC model_content_hash is a zero stub
    today (worker.py returns b"\\x00"*32), so it is deliberately NOT treated as proof.
  • participation — v1 is COORDINATOR-ASSERTED (signed_by="coordinator"). `worker_signatures`
    is a non-breaking placeholder; later each worker Ed25519-signs "{run_id}|{node_id}|
    [{a},{b})|{output_sha256}" (reusing nakshatra_auth's per-node key) so participation
    becomes cryptographically attested rather than coordinator-claimed.

Pure: no GPU, no network, no proto change. `engine_provenance` / `worker_signatures` are
passthrough (populated by the caller when available). Fully unit-tested.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

RECEIPT_VERSION = "1.0"

# What each claim actually means — embedded verbatim so a reader of the receipt sees the
# honesty scope without needing this source. verify_receipt enforces only the checkable ones.
CLAIMS = {
    "output_hash": "third-party checkable: recompute sha256 over generated_tokens",
    "reproducibility": "within-engine, same-provenance (engine_provenance) re-runnable; "
                       "NOT bit-exact across engines/backends (quantized kernels round "
                       "batched-vs-single-token differently)",
    "model_identity": "asserted via model_id; served weights NOT cryptographically verified "
                      "over the wire (gRPC model_content_hash is a stub)",
    "participation": "coordinator-asserted; per-worker signatures absent in v1 "
                     "(worker_signatures placeholder)",
}


def output_sha256(tokens: Sequence[int]) -> str:
    """Stable, order-sensitive hash of a token id list. Canonical form: comma-joined ints."""
    canonical = ",".join(str(int(t)) for t in tokens)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_receipt(*, run_id: str, model_id: str,
                  prompt_tokens: Sequence[int], generated_tokens: Sequence[int],
                  workers: Sequence[Dict[str, Any]], elapsed_s: float,
                  started_at: float, ended_at: float,
                  engine_provenance: Optional[List[str]] = None,
                  worker_signatures: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Assemble a run receipt.

    workers: per-stage dicts with at least node_id, layer_start, layer_end; ideally also
             address, spki_hash, backend, mean_rpc_ms. Order = pipeline order.
    """
    gen = [int(t) for t in generated_tokens]
    n_gen = len(gen)
    tok_per_s = (n_gen / elapsed_s) if elapsed_s > 0 else 0.0
    chain = [{
        "node_id": w.get("node_id"),
        "address": w.get("address"),
        "spki_hash": w.get("spki_hash"),
        "layer_start": w.get("layer_start"),
        "layer_end": w.get("layer_end"),
        "backend": w.get("backend"),
        "mean_rpc_ms": w.get("mean_rpc_ms"),
    } for w in workers]
    return {
        "receipt_version": RECEIPT_VERSION,
        "run_id": run_id,
        "model_id": model_id,
        "signed_by": "coordinator",
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_s": elapsed_s,
        "n_prompt": len(list(prompt_tokens)),
        "prompt_sha256": output_sha256(prompt_tokens),
        "n_generated": n_gen,
        "generated_tokens": gen,
        "output_sha256": output_sha256(gen),
        "tok_per_s": tok_per_s,
        "chain": chain,
        "engine_provenance": list(engine_provenance or []),
        "worker_signatures": list(worker_signatures or []),
        "claims": dict(CLAIMS),
    }


def verify_receipt(receipt: Dict[str, Any], *, tol: float = 1e-6) -> Tuple[bool, List[str]]:
    """Re-check everything a third party can check WITHOUT the cluster. Returns (ok, problems).

    Enforces only the honest claims: output hash recompute, count/tok-s/time math, worker
    distinctness, contiguous layer-map. Does NOT assert weights or cross-engine bit-exactness.
    """
    problems: List[str] = []

    def need(key):
        if key not in receipt:
            problems.append(f"missing field: {key}")
            return False
        return True

    for k in ("generated_tokens", "output_sha256", "n_generated", "elapsed_s",
              "tok_per_s", "started_at", "ended_at", "chain"):
        need(k)
    if problems:
        return False, problems

    gen = receipt["generated_tokens"]
    # 1. output hash recompute (zero-trust)
    if output_sha256(gen) != receipt["output_sha256"]:
        problems.append("output_sha256 does not match generated_tokens")
    # 2. count consistency
    if len(gen) != receipt["n_generated"]:
        problems.append(f"n_generated {receipt['n_generated']} != len(generated_tokens) {len(gen)}")
    # 3. tok/s math
    elapsed = receipt["elapsed_s"]
    if elapsed <= 0:
        if receipt["n_generated"] > 0:
            problems.append("elapsed_s must be > 0 for a non-empty generation")
    else:
        expected_tps = receipt["n_generated"] / elapsed
        if abs(receipt["tok_per_s"] - expected_tps) > max(tol, tol * expected_tps):
            problems.append(f"tok_per_s {receipt['tok_per_s']} != n_generated/elapsed_s {expected_tps}")
    # 4. timestamps monotonic
    if receipt["ended_at"] < receipt["started_at"]:
        problems.append("ended_at precedes started_at")
    # 5. worker distinctness — distinct ids AND distinct TLS keys (distinct keys ≠ one box)
    chain = receipt["chain"]
    ids = [c.get("node_id") for c in chain]
    if len(set(ids)) != len(ids):
        problems.append("duplicate node_id in chain (not distinct machines)")
    spkis = [c.get("spki_hash") for c in chain if c.get("spki_hash")]
    if spkis and len(set(spkis)) != len(spkis):
        problems.append("duplicate spki_hash in chain (same TLS key on multiple stages)")
    # 6. contiguous layer-map [0, N), no gap/overlap
    try:
        ranges = sorted((int(c["layer_start"]), int(c["layer_end"])) for c in chain)
    except (KeyError, TypeError, ValueError):
        ranges = None
        problems.append("chain has missing/invalid layer_start/layer_end")
    if ranges:
        if ranges[0][0] != 0:
            problems.append(f"layer-map does not start at 0 (starts at {ranges[0][0]})")
        for (s, e), (ns, ne) in zip(ranges, ranges[1:]):
            if ns != e:
                problems.append(f"layer-map gap/overlap: [{s},{e}) then [{ns},{ne})")

    return (len(problems) == 0, problems)
