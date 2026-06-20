"""
identity_binding.py — bind a peer's Ed25519 identity to (a) a holder-of-key proof, (b) the
distributed-run work it performed, and (c) a credit-ledger account. ONE key, three jobs.

Sourced from `leyten/shard`'s identity-binding prove/verify primitive (sidecar `main.go:-prove/-verify`:
a node signs a challenge with its key; a verifier checks the sig against the pinned key alone). Here we
apply it to our own posture:

  • prove/verify(challenge)      — generic holder-of-key (a stolen roster row is useless without the key).
  • sign/verify_participation()  — fills the `receipt.py` placeholder: each worker Ed25519-signs the exact
    stage it served (`{run_id}|{node_id}|[{a},{b})|{output_sha256}`), upgrading participation from
    COORDINATOR-ASSERTED to holder-of-key PROVEN. You cannot earn credit for work you can't prove.
  • account_id(pubkey)           — the credit-ledger account IS the identity (v1: account == pubkey).
  • creditable_accounts(receipt) — the metering bridge: which bound accounts a verified receipt may credit.

This is the trust primitive between the admission junction (admits by pubkey), the #20 run receipts
(what was served), and the reciprocal-compute credit ledger (who earns). Keys are the SAME format as
`trisul/infra/control-plane/admission.py` (raw 32-byte hex priv/pub, base64 sigs) — so a peer's admission
identity, its receipt-signing identity, and its ledger account are one and the same.

Pure crypto: no GPU, no network, no proto change. Default-deny / fail-closed on every check.
"""
from __future__ import annotations

import base64
import json
import secrets
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cryptography.hazmat.primitives.asymmetric import ed25519

# A live challenge older than this (seconds) is rejected — bounds replay of a captured proof.
CHALLENGE_MAX_AGE = 120


# ── low-level Ed25519 (same conventions as admission.py: hex keys, base64 sigs) ──────────────────────
def _ed_sign(priv_hex: str, data: bytes) -> str:
    return base64.b64encode(
        ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex)).sign(data)).decode()


def _ed_verify(pub_hex: str, b64sig: str, data: bytes) -> bool:
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex)).verify(
            base64.b64decode(b64sig), data)
        return True
    except Exception:
        return False


def pub_of(priv_hex: str) -> str:
    """The public key (raw hex) for a raw-hex private key — the peer's stable identity."""
    return ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(priv_hex)).public_key().public_bytes_raw().hex()


# ── (a) generic holder-of-key proof (shard's prove/verify) ───────────────────────────────────────────
def make_challenge() -> Dict[str, Any]:
    """A fresh challenge for a live holder-of-key handshake (e.g. binding a session/account to a key)."""
    return {"nonce": secrets.token_hex(16), "ts": int(time.time())}


def _challenge_bytes(challenge: Dict[str, Any]) -> bytes:
    return json.dumps({"nonce": challenge.get("nonce", ""), "ts": int(challenge.get("ts", 0))},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def prove(priv_hex: str, challenge: Dict[str, Any]) -> str:
    """Sign a challenge — proves the presenter holds the private key for its claimed identity."""
    return _ed_sign(priv_hex, _challenge_bytes(challenge))


def verify(pub_hex: str, challenge: Dict[str, Any], b64sig: str, *, now: Optional[int] = None) -> bool:
    """Verify a holder-of-key proof against the pinned pubkey. Fail-closed; rejects stale challenges."""
    now = int(now if now is not None else time.time())
    if abs(now - int(challenge.get("ts", 0))) > CHALLENGE_MAX_AGE:
        return False
    return _ed_verify(pub_hex, b64sig, _challenge_bytes(challenge))


# ── (b) the ledger account == the identity (v1) ──────────────────────────────────────────────────────
def account_id(pub_hex: str) -> str:
    """The credit-ledger account for an identity. v1: the account IS the pubkey (no separate mapping),
    namespaced so it's unambiguous in ledger records and future-proof if we ever derive instead."""
    return f"nak:{pub_hex}"


# ── (c) receipt participation — holder-of-key proof of WHAT a worker served ──────────────────────────
def participation_message(run_id: str, node_id: str, layer_start: int, layer_end: int,
                          output_sha256: str) -> bytes:
    """The exact canonical the worker signs (matches receipt.py's documented placeholder format)."""
    return f"{run_id}|{node_id}|[{int(layer_start)},{int(layer_end)})|{output_sha256}".encode("utf-8")


def sign_participation(priv_hex: str, *, run_id: str, node_id: str, layer_start: int, layer_end: int,
                       output_sha256: str) -> Dict[str, Any]:
    """A worker self-signs the stage it served. Goes into a receipt's `worker_signatures` list."""
    pub = pub_of(priv_hex)
    return {"node_id": node_id, "pubkey": pub, "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "sig": _ed_sign(priv_hex, participation_message(run_id, node_id, layer_start, layer_end,
                                                            output_sha256))}


def verify_participation(entry: Dict[str, Any], *, run_id: str, output_sha256: str,
                         pinned: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """Verify one worker's participation signature.

    Checks the signature binds the claimed pubkey to THIS run + THIS exact stage + THIS output. If
    `pinned` (node_id -> trusted pubkey, e.g. from the admission roster) is given, the signing key MUST
    match the pinned one — default-deny: an unpinned or mismatched key earns nothing.
    Returns (ok, reason).
    """
    try:
        node_id = entry["node_id"]; pub = entry["pubkey"]
        a = int(entry["layer_start"]); b = int(entry["layer_end"]); sig = entry["sig"]
    except (KeyError, TypeError, ValueError) as e:
        return False, f"malformed participation entry ({e})"
    if pinned is not None:
        want = pinned.get(node_id)
        if want is None:
            return False, f"node '{node_id}' not in the pinned roster (default-deny)"
        if want != pub:
            return False, f"key for '{node_id}' does not match the pinned identity"
    if not _ed_verify(pub, sig, participation_message(run_id, node_id, a, b, output_sha256)):
        return False, f"participation signature invalid for '{node_id}'"
    return True, "ok"


def creditable_accounts(receipt: Dict[str, Any], *,
                        pinned: Optional[Dict[str, str]] = None) -> Tuple[List[str], List[str]]:
    """The metering bridge the credit ledger consumes. Given a run receipt, return the list of bound
    account_ids that may be credited — ONLY stages whose worker_signature holder-of-key-verifies against
    this run's id + output (and, if `pinned` given, against the rostered key). Stages without a valid
    proof earn NOTHING (no coordinator-asserted credit). Returns (accounts, problems).

    Non-breaking: a receipt with no `worker_signatures` simply yields no creditable accounts + a note —
    it does not error, so legacy receipts and `verify_receipt()` are unaffected.
    """
    problems: List[str] = []
    run_id = receipt.get("run_id", "")
    out = receipt.get("output_sha256", "")
    sigs = receipt.get("worker_signatures") or []
    if not run_id or not out:
        return [], ["receipt missing run_id/output_sha256"]
    if not sigs:
        return [], ["no worker_signatures present (participation coordinator-asserted only; nothing to credit)"]
    accounts: List[str] = []
    seen: set = set()
    for entry in sigs:
        ok, why = verify_participation(entry, run_id=run_id, output_sha256=out, pinned=pinned)
        if not ok:
            problems.append(why)
            continue
        acct = account_id(entry["pubkey"])
        if acct in seen:                       # one account credited once per run, even if it claims 2 stages
            continue
        seen.add(acct)
        accounts.append(acct)
    return accounts, problems
