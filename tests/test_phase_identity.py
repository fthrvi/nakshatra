import pytest
from phase_identity import identity


def test_clean_identity():
    facts = {
        "identity_pubkey": "a" * 64,
        "identity_is_new": False,
        "registered": True,
        "roster_pubkey": None
    }
    success, problems, updates = identity(facts)
    assert success is True
    assert problems == []
    assert updates == {"identity_ok": True, "account": "nak:" + "a" * 64}


def test_malformed_pubkey():
    facts = {
        "identity_pubkey": "GG" * 32,  # uppercase not allowed
        "identity_is_new": False,
        "registered": True,
        "roster_pubkey": None
    }
    success, problems, updates = identity(facts)
    assert success is False
    assert len(problems) == 1
    assert "identity_pubkey must be 64 lowercase hex characters" in problems[0]
    assert updates == {}


def test_not_registered():
    facts = {
        "identity_pubkey": "a" * 64,
        "identity_is_new": False,
        "registered": False,
        "roster_pubkey": None
    }
    success, problems, updates = identity(facts)
    assert success is False
    assert len(problems) == 1
    assert "node is not registered" in problems[0]
    assert updates == {}


def test_roster_mismatch_rekey():
    facts = {
        "identity_pubkey": "b" * 64,
        "identity_is_new": False,
        "registered": True,
        "roster_pubkey": "a" * 64
    }
    success, problems, updates = identity(facts)
    assert success is False
    assert len(problems) == 1
    assert "node appears to have re-keyed: everything earned under the old key is now unreachable" in problems[0]
    assert updates == {}


def test_new_key_with_existing_roster():
    facts = {
        "identity_pubkey": "b" * 64,
        "identity_is_new": True,
        "registered": True,
        "roster_pubkey": "a" * 64
    }
    success, problems, updates = identity(facts)
    assert success is False
    assert len(problems) == 1
    assert "fresh key minted for node that already had an identity on file" in problems[0]
    assert updates == {}


def test_absent_roster_pubkey_first_join():
    facts = {
        "identity_pubkey": "a" * 64,
        "identity_is_new": True,
        "registered": True,
        "roster_pubkey": None
    }
    success, problems, updates = identity(facts)
    assert success is True
    assert problems == []
    assert updates == {"identity_ok": True, "account": "nak:" + "a" * 64}


def test_account_id_formed_correctly():
    facts = {
        "identity_pubkey": "deadbeef" * 8,
        "identity_is_new": False,
        "registered": True,
        "roster_pubkey": None
    }
    success, problems, updates = identity(facts)
    assert success is True
    assert updates["account"] == "nak:" + "deadbeef" * 8


def test_missing_observations():
    facts = {
        "identity_pubkey": None,
        "identity_is_new": False,
        "registered": True,
        "roster_pubkey": None
    }
    success, problems, updates = identity(facts)
    assert success is False
    assert len(problems) >= 1
    assert "identity_pubkey must be a string" in problems or "identity_pubkey must be 64 characters" in problems or "identity_pubkey must be 64 lowercase hex characters" in problems
    assert updates == {}


def test_hostile_types():
    facts = {
        "identity_pubkey": 12345,
        "identity_is_new": "yes",
        "registered": "true",
        "roster_pubkey": ["a", "b"]
    }
    success, problems, updates = identity(facts)
    assert success is False
    assert len(problems) >= 1
    assert updates == {}


def test_determinism():
    facts = {
        "identity_pubkey": "a" * 64,
        "identity_is_new": False,
        "registered": True,
        "roster_pubkey": None
    }
    results = [identity(facts) for _ in range(5)]
    assert all(r == results[0] for r in results)