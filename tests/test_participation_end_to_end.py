"""The whole honest-accounting path, once, on real objects.

Every module in this chain has its own unit tests. None of them proves the chain HOLDS —
that what one produces is what the next can consume. Nine modules that each pass in isolation
can still disagree at every seam, and two of them in this very batch did exactly that: they
passed their own suites while being wrong about the receipt shape, because their fixtures
were invented rather than built.

So this walks one run from generation to settlement using the real builder, the real signer
and the real verifier at every step, and asserts the thing the mission actually rests on:

    a stranger's node earns credit for work it can prove it did,
    and earns nothing for work it cannot.

⚠️ It is deliberately ONE narrative rather than parametrised cases. The value is in the
ordering — output_sha256 exists only after generation, signatures bind to it, coverage
compares two independent stories about the same run, units apportion, reconcile audits. A
seam that breaks shows up as a failure at the step where the handoff is wrong.
"""
import os

from coverage import signature_coverage
from dispute import should_dispute
from identity_binding import account_id, creditable_accounts, pub_of
from receipt import build_receipt, verify_receipt
from reconcile import reconcile
from settlekey import is_duplicate, settlement_key
from units import credit_units
from worker_sigs import build_worker_signatures

GENERATED = [11, 22, 33, 44]          # n_generated == 4
CHAIN = [("alice", 0, 13), ("bob", 13, 26), ("carol", 26, 32)]


def _run(keys, sign_only=None):
    """One real run: build, learn the output hash, sign against it, rebuild."""
    chain = [{"node_id": n, "layer_start": a, "layer_end": b} for n, a, b in CHAIN]
    kw = dict(run_id="run-42", model_id="llama-32L", prompt_tokens=[1, 2, 3],
              generated_tokens=GENERATED, workers=chain, elapsed_s=2.0,
              started_at=100.0, ended_at=102.0)
    out = build_receipt(**kw)["output_sha256"]
    stages = [{"node_id": n, "layer_start": a, "layer_end": b}
              for n, a, b in (sign_only or CHAIN)]
    sigs = build_worker_signatures(stages, keys, run_id="run-42", output_sha256=out)
    return build_receipt(**kw, worker_signatures=sigs)


def test_the_whole_chain_holds():
    keys = {n: os.urandom(32).hex() for n, _, _ in CHAIN}
    roster = {n: pub_of(k) for n, k in keys.items()}
    rcpt = _run(keys)

    # 1. the receipt is internally consistent, and says who signed it
    ok, problems = verify_receipt(rcpt)
    assert ok, problems
    assert rcpt["signed_by"] == "workers"

    # 2. the signatures cover every layer the chain claims — no unattested work
    cov = signature_coverage(rcpt, roster)
    assert cov["fully_covered"] is True, cov
    assert cov["covered"] == [(0, 32)] and cov["gaps"] == [] and cov["unclaimed"] == []

    # 3. every worker is creditable, and the accounts are their identities
    accounts, why = creditable_accounts(rcpt, pinned=roster)
    assert sorted(accounts) == sorted(account_id(pub_of(k)) for k in keys.values()), why

    # 4. apportionment: units = n_generated x layers served
    units = credit_units(rcpt, roster)
    assert units[account_id(pub_of(keys["alice"]))] == len(GENERATED) * 13
    assert units[account_id(pub_of(keys["bob"]))] == len(GENERATED) * 13
    assert units[account_id(pub_of(keys["carol"]))] == len(GENERATED) * 6

    # 5. a ledger holding exactly that reconciles clean
    assert reconcile(units, [rcpt], roster)["clean"] is True

    # 6. and the same receipt delivered twice settles once
    key = settlement_key(rcpt)
    assert is_duplicate(key, {key: "2026-09-04T00:00:00Z"})[0] is True


def test_an_unsigned_stage_earns_nothing_and_is_visible():
    """The failure the mission cares about: work nobody proved must not be paid,
    AND must not be silent — an unpaid gap that nothing reports is indistinguishable
    from a node that was never in the run."""
    keys = {n: os.urandom(32).hex() for n, _, _ in CHAIN}
    roster = {n: pub_of(k) for n, k in keys.items()}
    rcpt = _run(keys, sign_only=[("alice", 0, 13), ("bob", 13, 26)])   # carol does not sign

    cov = signature_coverage(rcpt, roster)
    assert cov["fully_covered"] is False
    assert cov["gaps"] == [(26, 32)], cov          # visible, and named

    units = credit_units(rcpt, roster)
    assert account_id(pub_of(keys["carol"])) not in units
    assert sum(units.values()) == len(GENERATED) * 26


def test_a_key_nobody_rostered_earns_nothing():
    """The defect this whole branch exists to close: an unregistered key certifying work."""
    keys = {n: os.urandom(32).hex() for n, _, _ in CHAIN}
    roster = {n: pub_of(k) for n, k in keys.items()}
    keys["alice"] = os.urandom(32).hex()           # alice re-keys; the roster does not know
    rcpt = _run(keys)

    accounts, why = creditable_accounts(rcpt, pinned=roster)
    assert account_id(pub_of(keys["alice"])) not in accounts, why
    assert signature_coverage(rcpt, roster)["gaps"] == [(0, 13)]
    assert credit_units(rcpt, roster).get(account_id(pub_of(keys["alice"]))) is None


def test_a_failing_spot_check_run_claws_back():
    """The audit loop closes: repeated genuine failures dispute, one blip does not."""
    blip = [{"ok": False, "checked_at": 1, "reason": "timeout"}]
    assert should_dispute(blip)[0] is False

    real = [{"ok": False, "checked_at": t, "reason": f"mismatch-{t}"} for t in (1, 2, 3)]
    fired, why = should_dispute(real)
    assert fired is True and "3 of 3" in why
