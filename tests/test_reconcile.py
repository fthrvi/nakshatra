"""Unit tests for reconcile.py covering all specified scenarios."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from identity_binding import account_id, pub_of, sign_participation  # noqa: E402
from receipt import build_receipt  # noqa: E402
from reconcile import reconcile  # noqa: E402

GEN = [3, 4, 5]  # n_generated == 3


def _make_receipt(node_id, layer_start, layer_end, priv_key, run_id="r", output_sha256=None):
    """Helper to build a receipt with a participation signature."""
    chain = [{"node_id": node_id, "layer_start": layer_start, "layer_end": layer_end}]
    kw = dict(run_id=run_id, model_id="m", prompt_tokens=[1, 2], generated_tokens=GEN,
              workers=chain, elapsed_s=1.0, started_at=0.0, ended_at=1.0)
    
    # Build receipt to get output_sha256 if not provided
    if output_sha256 is None:
        base_receipt = build_receipt(**kw)
        output_sha256 = base_receipt["output_sha256"]
    
    # Sign the participation
    sig = sign_participation(priv_key, run_id=run_id, node_id=node_id,
                            layer_start=layer_start, layer_end=layer_end,
                            output_sha256=output_sha256)
    
    return build_receipt(**kw, worker_signatures=[sig])


def test_perfect_agreement_clean_true():
    """Perfect agreement → clean=True"""
    priv = os.urandom(32).hex()
    pub = pub_of(priv)
    acct = account_id(pub)
    receipt = _make_receipt("node1", 0, 13, priv)
    roster = {"node1": pub}
    
    # Expected: 3 * (13 - 0) = 39
    res = reconcile({acct: 39}, [receipt], roster)
    assert res["clean"] is True
    assert res["agree"] == {acct: 39}
    assert res["over"] == {}
    assert res["under"] == {}
    assert res["unknown_accounts"] == []


def test_one_account_over():
    """One account over: ledger holds MORE than receipts justify"""
    priv = os.urandom(32).hex()
    pub = pub_of(priv)
    acct = account_id(pub)
    receipt = _make_receipt("node1", 0, 13, priv)
    roster = {"node1": pub}
    
    # Ledger has 39 + 100 = 139 but computed is 39
    res = reconcile({acct: 139}, [receipt], roster)
    assert res["clean"] is False
    assert acct in res["over"]
    assert res["over"][acct] == (139, 39)
    assert res["under"] == {}
    assert res["unknown_accounts"] == []


def test_one_account_under():
    """One account under: ledger holds LESS than receipts justify"""
    priv = os.urandom(32).hex()
    pub = pub_of(priv)
    acct = account_id(pub)
    receipt = _make_receipt("node1", 0, 13, priv)
    roster = {"node1": pub}
    
    # Ledger has 1 but computed is 39
    res = reconcile({acct: 1}, [receipt], roster)
    assert res["clean"] is False
    assert acct in res["under"]
    assert res["under"][acct] == (1, 39)
    assert res["over"] == {}


def test_account_with_no_receipts_unknown_not_over():
    """Account with no supporting receipts → unknown_accounts, NOT over"""
    priv = os.urandom(32).hex()
    pub = pub_of(priv)
    acct = account_id(pub)
    receipt = _make_receipt("node1", 0, 13, priv)
    roster = {"node1": pub}
    
    # Different account with balance
    fake_acct = "nak:" + "f" * 64
    res = reconcile({fake_acct: 5}, [receipt], roster)
    assert fake_acct in res["unknown_accounts"]
    assert res["over"] == {}
    assert res["clean"] is False


def test_unverifiable_signature_no_under():
    """Unverifiable signature should not create under"""
    priv = os.urandom(32).hex()
    pub = pub_of(priv)
    acct = account_id(pub)
    
    # Build receipt with valid signature first
    receipt = _make_receipt("node1", 0, 13, priv)
    
    # Corrupt the signature
    receipt["worker_signatures"][0]["sig"] = "invalid_signature"
    
    roster = {"node1": pub}
    
    # Even though ledger has balance, unverifiable signature means no computed credit
    res = reconcile({acct: 39}, [receipt], roster)
    # Should be unknown because signature didn't verify
    assert acct in res["unknown_accounts"]
    assert res["over"] == {}
    assert res["under"] == {}


def test_malformed_receipt_counted_skipped():
    """Malformed receipt counted in skipped without raising"""
    priv = os.urandom(32).hex()
    pub = pub_of(priv)
    acct = account_id(pub)
    
    # Valid receipt
    valid_receipt = _make_receipt("node1", 0, 13, priv)
    
    # Malformed receipts
    malformed1 = "not a dict"
    malformed2 = {"worker_signatures": "not a list"}
    malformed3 = {"worker_signatures": [], "run_id": "", "output_sha256": ""}
    
    res = reconcile({acct: 39}, [valid_receipt, malformed1, malformed2, malformed3], 
                   {"node1": pub})
    
    assert res["skipped"] == 3
    assert res["agree"] == {acct: 39}
    assert res["clean"] is True


def test_several_receipts_sum_for_one_account():
    """Several receipts summing for one account"""
    priv = os.urandom(32).hex()
    pub = pub_of(priv)
    acct = account_id(pub)
    
    # Two receipts for same account
    receipt1 = _make_receipt("node1", 0, 10, priv, run_id="r1")
    receipt2 = _make_receipt("node1", 10, 20, priv, run_id="r2")
    
    roster = {"node1": pub}
    
    # Expected: 3 * 10 + 3 * 10 = 60
    res = reconcile({acct: 60}, [receipt1, receipt2], roster)
    assert res["clean"] is True
    assert res["agree"] == {acct: 60}


def test_empty_ledger_no_receipts_clean():
    """Empty ledger with no receipts → clean"""
    res = reconcile({}, [], {})
    assert res["clean"] is True
    assert res["agree"] == {}
    assert res["over"] == {}
    assert res["under"] == {}
    assert res["unknown_accounts"] == []


def test_multiple_accounts_mixed_results():
    """Multiple accounts with mixed agreement states"""
    priv1 = os.urandom(32).hex()
    priv2 = os.urandom(32).hex()
    priv3 = os.urandom(32).hex()
    
    pub1 = pub_of(priv1)
    pub2 = pub_of(priv2)
    pub3 = pub_of(priv3)
    
    acct1 = account_id(pub1)
    acct2 = account_id(pub2)
    acct3 = account_id(pub3)
    
    # Receipts for acct1 and acct2
    receipt1 = _make_receipt("node1", 0, 10, priv1, run_id="r1")
    receipt2 = _make_receipt("node2", 0, 5, priv2, run_id="r2")
    
    roster = {"node1": pub1, "node2": pub2}
    
    # acct1: exact match (30)
    # acct2: under (5 vs computed 15)
    # acct3: unknown (no receipt)
    res = reconcile({acct1: 30, acct2: 5, acct3: 10}, 
                   [receipt1, receipt2], roster)
    
    assert res["clean"] is False
    assert res["agree"] == {acct1: 30}
    assert acct2 in res["under"]
    assert acct3 in res["unknown_accounts"]
    assert res["over"] == {}