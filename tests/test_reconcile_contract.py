"""Reconciliation against receipts the REAL builder produced. Harness-owned; not yours to edit.

⚠️⚠️ A previous attempt passed its own suite and was wrong on every real receipt: an exact
match was not reported `clean`, and an over-credited account was not reported in `over`. Its
fixtures were invented, so they matched its implementation and nothing else. The same thing
happened to the coverage module in the same batch — checking `isinstance(worker_signatures,
dict)` when it is a LIST.

A test whose fixture is invented cannot detect a wrong guess about the format. It can only
confirm the guess. So this file BUILDS its receipts with `receipt.build_receipt` and signs
with `identity_binding.sign_participation` — the shape is produced here, not described here.

Read it first to learn what a receipt actually looks like.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from identity_binding import account_id, pub_of, sign_participation  # noqa: E402
from receipt import build_receipt  # noqa: E402

from reconcile import reconcile  # noqa: E402

GEN = [3, 4, 5]          # n_generated == 3


def _receipt(spans):
    """spans = [(node_id, start, end)] -> (receipt, roster, {node: account_id})"""
    keys = {n: os.urandom(32).hex() for n, _, _ in spans}
    chain = [{"node_id": n, "layer_start": a, "layer_end": b} for n, a, b in spans]
    kw = dict(run_id="r", model_id="m", prompt_tokens=[1, 2], generated_tokens=GEN,
              workers=chain, elapsed_s=1.0, started_at=0.0, ended_at=1.0)
    out = build_receipt(**kw)["output_sha256"]
    sigs = [sign_participation(keys[n], run_id="r", node_id=n, layer_start=a,
                               layer_end=b, output_sha256=out) for n, a, b in spans]
    return (build_receipt(**kw, worker_signatures=sigs),
            {n: pub_of(k) for n, k in keys.items()},
            {n: account_id(pub_of(k)) for n, k in keys.items()})


def test_exact_match_is_clean():
    r, roster, acct = _receipt([("a", 0, 13)])
    res = reconcile({acct["a"]: len(GEN) * 13}, [r], roster)
    assert res["clean"] is True, res
    assert res["over"] == {} and res["under"] == {}, res


def test_over_credited_account_is_reported_as_over():
    """The finding that matters: the ledger paid for work no receipt supports."""
    r, roster, acct = _receipt([("a", 0, 13)])
    res = reconcile({acct["a"]: len(GEN) * 13 + 100}, [r], roster)
    assert acct["a"] in res["over"], res
    assert res["clean"] is False, res


def test_under_credited_is_separate_from_over():
    r, roster, acct = _receipt([("a", 0, 13)])
    res = reconcile({acct["a"]: 1}, [r], roster)
    assert acct["a"] in res["under"] and res["over"] == {}, res


def test_account_with_no_receipts_is_unknown_not_over():
    """It may simply predate the receipts handed in. Report what you observed."""
    r, roster, _ = _receipt([("a", 0, 13)])
    res = reconcile({"nak:" + "f" * 64: 5}, [r], roster)
    assert "nak:" + "f" * 64 in res["unknown_accounts"], res
    assert res["over"] == {}, res


def test_two_receipts_sum_for_one_account():
    keys = os.urandom(32).hex()
    rs = []
    for a, b in [(0, 10), (10, 20)]:
        chain = [{"node_id": "a", "layer_start": a, "layer_end": b}]
        kw = dict(run_id=f"r{a}", model_id="m", prompt_tokens=[1], generated_tokens=GEN,
                  workers=chain, elapsed_s=1.0, started_at=0.0, ended_at=1.0)
        out = build_receipt(**kw)["output_sha256"]
        sig = sign_participation(keys, run_id=f"r{a}", node_id="a", layer_start=a,
                                 layer_end=b, output_sha256=out)
        rs.append(build_receipt(**kw, worker_signatures=[sig]))
    res = reconcile({account_id(pub_of(keys)): len(GEN) * 20}, rs, {"a": pub_of(keys)})
    assert res["clean"] is True, res


def test_empty_everything_is_clean():
    assert reconcile({}, [], {})["clean"] is True
