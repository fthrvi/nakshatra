"""Drift-class conformance gauge (v1.x — from the cross-machine determinism finding).

`docs/cross-machine-validation.md` showed: two heterogeneous nodes holding
byte-identical weights still diverge in multi-token generation, because their
CPUs/builds round floating-point slightly differently and a close-call argmax
flips. So **bit-identical generation requires same-class nodes**, and the mesh
needs a cheap way to tell which nodes are in the same class.

This is that gauge — the "interchangeable parts / tolerance bin" lever from the
by-analogy work. A node runs a FIXED canonical computation (a pinned prompt →
greedy token-id sequence at temp 0, fixed length) on the served model; the hash
of that sequence is its **drift-class fingerprint**. Two nodes with the same
fingerprint agree on every argmax for that reference and can safely form a
bit-deterministic chain; different fingerprints ⇒ different drift class.

Pure + dependency-light: the fingerprint is computed from a token-id sequence the
node already knows how to produce (via its engine / a single-node llama-simple /
the chain itself). This module owns the canonical inputs + the fingerprint
contract, not the inference — so a worker, a CLI probe, and a test all agree on
what "the same class" means.

Design notes:
  • Greedy token-ids (not raw logits) are the right granularity: what makes
    generation diverge is argmax *decisions* flipping, and the token sequence is
    exactly the record of those decisions. Two nodes that pick the same tokens for
    GAUGE_TOKENS steps will track each other for at least that far.
  • The fingerprint is per (canonical_prompt, model_id, length). A node advertises
    one fingerprint per model it serves. Discovery can then pre-filter peers to a
    compatible class, the same way `supported_protocol` pre-filters wire versions.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

GAUGE_VERSION = 1

# The canonical probe. FIXED forever at a given GAUGE_VERSION — changing it
# changes every fingerprint, so bump GAUGE_VERSION if you ever must.
CANONICAL_PROMPT = "The capital of France is"
GAUGE_TOKENS = 16          # greedy steps to fingerprint (longer = stricter class)
GAUGE_TEMPERATURE = 0.0    # deterministic argmax


@dataclass(frozen=True)
class DriftFingerprint:
    """A node's drift-class id for one model."""
    model_id: str
    gauge_version: int
    fingerprint: str        # hex sha256 over the canonical token-id sequence
    n_tokens: int

    def same_class(self, other: "DriftFingerprint") -> bool:
        """True iff the two nodes agree on the canonical generation — i.e. they
        are in the same drift class for this model. Different model_id /
        gauge_version are never comparable (returns False)."""
        return (
            self.gauge_version == other.gauge_version
            and self.model_id == other.model_id
            and self.fingerprint == other.fingerprint
        )

    def short(self) -> str:
        return f"{self.model_id}@gauge{self.gauge_version}:{self.fingerprint[:12]}"


def fingerprint_from_token_ids(model_id: str, token_ids: Sequence[int],
                               gauge_version: int = GAUGE_VERSION) -> DriftFingerprint:
    """Build a fingerprint from the greedy token-id sequence a node produced for
    the canonical prompt. `token_ids` MUST be the temp-0 greedy continuation of
    CANONICAL_PROMPT, exactly GAUGE_TOKENS long (or fewer if the model stops)."""
    ids = list(int(t) for t in token_ids)
    payload = f"{gauge_version}\n{model_id}\n" + ",".join(str(i) for i in ids)
    fp = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return DriftFingerprint(model_id=model_id, gauge_version=gauge_version,
                            fingerprint=fp, n_tokens=len(ids))


def classes_of(fingerprints: list[DriftFingerprint]) -> dict[str, list[DriftFingerprint]]:
    """Group fingerprints into drift classes (same fingerprint hash). Returns
    {fingerprint_hash: [members]} — each key is one bit-deterministic class."""
    out: dict[str, list[DriftFingerprint]] = {}
    for f in fingerprints:
        out.setdefault(f.fingerprint, []).append(f)
    return out


# ── producing the canonical token-ids (optional helper, needs llama_cpp) ──

def canonical_token_ids_via_llama_cpp(model_path: str) -> list[int]:
    """Run the canonical greedy probe on `model_path` via llama-cpp-python and
    return the GAUGE_TOKENS token ids. Requires `llama_cpp` (the gauge contract
    above is dependency-free; only this convenience producer needs it). A worker
    can instead feed token-ids it already computed to fingerprint_from_token_ids."""
    from llama_cpp import Llama
    llm = Llama(model_path=model_path, n_ctx=256, logits_all=False, verbose=False)
    ids: list[int] = []
    toks = llm.tokenize(CANONICAL_PROMPT.encode("utf-8"))
    for _ in range(GAUGE_TOKENS):
        llm.eval(toks)
        nxt = int(llm.sample(temp=GAUGE_TEMPERATURE))   # greedy at temp 0
        if nxt == llm.token_eos():
            break
        ids.append(nxt)
        toks = [nxt]
    return ids


def gauge_model(model_path: str, model_id: str) -> DriftFingerprint:
    """One-call: run the canonical probe on a full model and return this node's
    drift fingerprint for it."""
    return fingerprint_from_token_ids(model_id, canonical_token_ids_via_llama_cpp(model_path))
