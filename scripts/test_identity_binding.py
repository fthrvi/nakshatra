"""Tests for identity_binding.py — holder-of-key proofs, receipt participation, credit accounts.
Pure crypto, no hardware. Run: python3 -m pytest scripts/test_identity_binding.py -q"""
from cryptography.hazmat.primitives.asymmetric import ed25519

import identity_binding as ib


def _key():
    priv = ed25519.Ed25519PrivateKey.generate().private_bytes_raw().hex()
    return priv, ib.pub_of(priv)


# ── generic holder-of-key proof ──────────────────────────────────────────────────────────────────────
def test_prove_verify_roundtrip():
    priv, pub = _key()
    ch = ib.make_challenge()
    assert ib.verify(pub, ch, ib.prove(priv, ch)) is True


def test_verify_rejects_wrong_key():
    priv, _ = _key()
    _, other_pub = _key()
    ch = ib.make_challenge()
    assert ib.verify(other_pub, ch, ib.prove(priv, ch)) is False


def test_verify_rejects_tampered_challenge():
    priv, pub = _key()
    ch = ib.make_challenge()
    sig = ib.prove(priv, ch)
    ch2 = dict(ch); ch2["nonce"] = "deadbeef" * 4
    assert ib.verify(pub, ch2, sig) is False


def test_verify_rejects_stale_challenge():
    priv, pub = _key()
    ch = {"nonce": "ab" * 16, "ts": 1_000_000}          # ancient
    assert ib.verify(pub, ch, ib.prove(priv, ch), now=1_000_000 + ib.CHALLENGE_MAX_AGE + 5) is False
    # but fresh within the window passes
    assert ib.verify(pub, ch, ib.prove(priv, ch), now=1_000_000 + 5) is True


def test_verify_rejects_garbage_sig():
    _, pub = _key()
    assert ib.verify(pub, ib.make_challenge(), "not-base64!!") is False


# ── account binding ──────────────────────────────────────────────────────────────────────────────────
def test_account_id_is_namespaced_identity():
    _, pub = _key()
    assert ib.account_id(pub) == f"nak:{pub}"
    # distinct keys → distinct accounts
    _, pub2 = _key()
    assert ib.account_id(pub) != ib.account_id(pub2)


# ── receipt participation ─────────────────────────────────────────────────────────────────────────────
RUN = "run-123"
OUT = "a" * 64


def test_sign_verify_participation_roundtrip():
    priv, pub = _key()
    e = ib.sign_participation(priv, run_id=RUN, node_id="w0", layer_start=0, layer_end=16, output_sha256=OUT)
    assert e["pubkey"] == pub and e["node_id"] == "w0"
    ok, why = ib.verify_participation(e, run_id=RUN, output_sha256=OUT)
    assert ok, why


def test_participation_rejects_wrong_output():
    priv, _ = _key()
    e = ib.sign_participation(priv, run_id=RUN, node_id="w0", layer_start=0, layer_end=16, output_sha256=OUT)
    ok, _ = ib.verify_participation(e, run_id=RUN, output_sha256="b" * 64)
    assert ok is False


def test_participation_rejects_wrong_run():
    priv, _ = _key()
    e = ib.sign_participation(priv, run_id=RUN, node_id="w0", layer_start=0, layer_end=16, output_sha256=OUT)
    ok, _ = ib.verify_participation(e, run_id="other-run", output_sha256=OUT)
    assert ok is False


def test_participation_rejects_tampered_stage():
    priv, _ = _key()
    e = ib.sign_participation(priv, run_id=RUN, node_id="w0", layer_start=0, layer_end=16, output_sha256=OUT)
    e["layer_end"] = 24                                  # claim more layers than signed for
    ok, _ = ib.verify_participation(e, run_id=RUN, output_sha256=OUT)
    assert ok is False


def test_participation_pinned_match_and_mismatch():
    priv, pub = _key()
    e = ib.sign_participation(priv, run_id=RUN, node_id="w0", layer_start=0, layer_end=16, output_sha256=OUT)
    # pinned to the real key → ok
    ok, _ = ib.verify_participation(e, run_id=RUN, output_sha256=OUT, pinned={"w0": pub})
    assert ok
    # pinned to a different key → default-deny
    _, other = _key()
    ok, why = ib.verify_participation(e, run_id=RUN, output_sha256=OUT, pinned={"w0": other})
    assert not ok and "pinned" in why
    # node not in roster → default-deny
    ok, why = ib.verify_participation(e, run_id=RUN, output_sha256=OUT, pinned={"someone-else": pub})
    assert not ok and "roster" in why


# ── creditable_accounts (the metering bridge) ─────────────────────────────────────────────────────────
def _receipt(sigs):
    return {"run_id": RUN, "output_sha256": OUT, "worker_signatures": sigs}


def test_creditable_accounts_credits_valid_only():
    p0, pub0 = _key(); p1, pub1 = _key()
    good = ib.sign_participation(p0, run_id=RUN, node_id="w0", layer_start=0, layer_end=16, output_sha256=OUT)
    bad = ib.sign_participation(p1, run_id="WRONG", node_id="w1", layer_start=16, layer_end=32, output_sha256=OUT)
    accts, problems = ib.creditable_accounts(_receipt([good, bad]))
    assert ib.account_id(pub0) in accts
    assert ib.account_id(pub1) not in accts            # bad proof earns nothing
    assert problems                                     # the bad one is reported


def test_creditable_accounts_dedups_per_account():
    p0, pub0 = _key()
    s_a = ib.sign_participation(p0, run_id=RUN, node_id="w0", layer_start=0, layer_end=16, output_sha256=OUT)
    s_b = ib.sign_participation(p0, run_id=RUN, node_id="w0", layer_start=16, layer_end=32, output_sha256=OUT)
    accts, _ = ib.creditable_accounts(_receipt([s_a, s_b]))
    assert accts == [ib.account_id(pub0)]               # one account, credited once


def test_creditable_accounts_enforces_pinned_roster():
    p0, pub0 = _key()
    e = ib.sign_participation(p0, run_id=RUN, node_id="w0", layer_start=0, layer_end=16, output_sha256=OUT)
    accts, _ = ib.creditable_accounts(_receipt([e]), pinned={"w0": pub0})
    assert accts == [ib.account_id(pub0)]
    _, other = _key()
    accts2, problems = ib.creditable_accounts(_receipt([e]), pinned={"w0": other})
    assert accts2 == [] and problems


def test_creditable_accounts_no_signatures_is_nonbreaking():
    accts, problems = ib.creditable_accounts({"run_id": RUN, "output_sha256": OUT})
    assert accts == [] and any("coordinator-asserted" in p for p in problems)


def test_creditable_accounts_missing_fields():
    accts, problems = ib.creditable_accounts({"worker_signatures": []})
    assert accts == [] and problems
