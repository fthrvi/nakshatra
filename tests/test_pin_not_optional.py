"""The roster pin is not optional — and the wire format did not move to achieve that.

⚠️ Both halves matter and they pull against each other. Making the check mandatory is easy if
you are willing to change what gets signed; the temptation is to fold the pubkey into the
canonical message so the signature binds the key directly. That is a defensible design and it
invalidates every signature ever produced. This change is deliberately narrower: it alters WHO
IS TRUSTED, never WHAT IS SIGNED. The golden-vector tests below are what hold that line.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from identity_binding import (  # noqa: E402
    UNPINNED_ACCEPT_ANY_KEY, account_id, creditable_accounts, participation_message,
    pub_of, sign_participation, verify_participation,
)

# Produced by the PRE-CHANGE module, pasted as literals on purpose. Regenerating them from the
# module under test would make this test agree with whatever the module does today.
GOLDEN = [
    (("r", "alice", 0, 13, "h"), b"r|alice|[0,13)|h"),
    (("run-2", "bob", 5, 64, "deadbeef"), b"run-2|bob|[5,64)|deadbeef"),
    (("x", "node|pipe", 0, 1, ""), b"x|node|pipe|[0,1)|"),
]


@pytest.mark.parametrize("args,expected", GOLDEN)
def test_canonical_bytes_are_frozen(args, expected):
    assert participation_message(*args) == expected


def _entry(priv, **kw):
    base = dict(run_id="r", node_id="alice", layer_start=0, layer_end=13, output_sha256="h")
    base.update(kw)
    return sign_participation(priv, **base)


def test_omitting_pinned_is_a_typeerror():
    """The defect was that forgetting one keyword silently disabled authentication."""
    priv = os.urandom(32).hex()
    with pytest.raises(TypeError):
        verify_participation(_entry(priv), run_id="r", output_sha256="h")
    with pytest.raises(TypeError):
        creditable_accounts({"run_id": "r", "output_sha256": "h", "worker_signatures": []})


def test_the_forgery_that_motivated_this_is_rejected():
    honest, attacker = os.urandom(32).hex(), os.urandom(32).hex()
    forged = _entry(attacker, layer_end=64)          # attacker's key, alice's name, 64 layers
    ok, why = verify_participation(forged, run_id="r", output_sha256="h",
                                   pinned={"alice": pub_of(honest)})
    assert not ok and "does not match the pinned identity" in why


def test_honest_worker_still_verifies():
    priv = os.urandom(32).hex()
    ok, why = verify_participation(_entry(priv), run_id="r", output_sha256="h",
                                   pinned={"alice": pub_of(priv)})
    assert ok, why


@pytest.mark.parametrize("bad", [None, {}, [], "roster", 0])
def test_a_roster_we_cannot_read_denies(bad):
    """None is no longer 'no roster, allow'. {} is a roster with nobody in it."""
    priv = os.urandom(32).hex()
    ok, _ = verify_participation(_entry(priv), run_id="r", output_sha256="h", pinned=bad)
    assert not ok


def test_the_escape_hatch_exists_and_is_visibly_unsafe():
    """A caller with no roster can still opt out — deliberately, in a form that greps."""
    attacker = os.urandom(32).hex()
    forged = _entry(attacker, layer_end=64)
    ok, _ = verify_participation(forged, run_id="r", output_sha256="h",
                                 pinned=UNPINNED_ACCEPT_ANY_KEY)
    assert ok, "the hatch must reproduce the old permissive behaviour, not quietly vanish"


def test_creditable_accounts_credits_only_rostered_keys():
    honest, attacker = os.urandom(32).hex(), os.urandom(32).hex()
    receipt = {"run_id": "r", "output_sha256": "h",
               "worker_signatures": [_entry(honest), _entry(attacker, node_id="mallory")]}
    accounts, problems = creditable_accounts(receipt, pinned={"alice": pub_of(honest)})
    assert accounts == [account_id(pub_of(honest))]
    assert any("not in the pinned roster" in p for p in problems)
