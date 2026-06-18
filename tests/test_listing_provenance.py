"""Provenance on the signed NakshatraListing — peers can verify each other's BUILD.
Covers: signed round-trip, backward-compat (a pre-provenance listing still verifies
and produces identical canonical bytes), and tamper-evidence (provenance is signed).
Run: python -m pytest tests/test_listing_provenance.py -q
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402
from discovery.nakshatra_listing import NakshatraListing  # noqa: E402


def _key():
    k = ed25519.Ed25519PrivateKey.generate()
    return k, k.public_key().public_bytes_raw().hex()


def _listing(pub_hex, provenance=None):
    return NakshatraListing(
        mesh_id="m", node_id="n", ed25519_pubkey_hex=pub_hex,
        serving=["general-7b"], supported_protocol=[1], drift_class="m@gauge1:abc",
        provenance=provenance, created_unix=int(time.time()))


def test_provenance_signs_verifies_and_roundtrips():
    k, pub = _key()
    L = _listing(pub, provenance="prov1:" + "a" * 64)
    L.sign(k.private_bytes_raw())
    assert L.verify()
    back = NakshatraListing.from_json(L.to_json())
    assert back.provenance == "prov1:" + "a" * 64
    assert back.verify()                       # signature survives serialization


def test_unset_provenance_is_backward_compatible():
    # A listing that doesn't advertise provenance must produce canonical bytes with
    # NO 'provenance' key — byte-identical to a pre-provenance listing — so existing
    # signatures keep verifying and old verifiers don't choke.
    k, pub = _key()
    L = _listing(pub, provenance=None)
    assert b"provenance" not in L.canonical_bytes()
    L.sign(k.private_bytes_raw())
    assert L.verify()
    assert "provenance" not in L.to_json()     # omitted on the wire when unset


def test_provenance_is_signed_tamper_evident():
    # Flipping the advertised build after signing must invalidate the signature —
    # a peer can't be fooled about another node's build in transit.
    k, pub = _key()
    L = _listing(pub, provenance="prov1:" + "a" * 64)
    L.sign(k.private_bytes_raw())
    assert L.verify()
    L.provenance = "prov1:" + "b" * 64         # tamper
    assert not L.verify()


def test_setting_provenance_changes_canonical_bytes():
    _, pub = _key()
    without = _listing(pub, provenance=None).canonical_bytes()
    with_ = _listing(pub, provenance="prov1:" + "c" * 64).canonical_bytes()
    assert without != with_
    assert b"provenance" in with_


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
