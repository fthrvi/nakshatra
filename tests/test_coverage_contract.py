"""The REAL receipt shape, produced by the real builder. Harness-owned; not yours to edit.

⚠️⚠️ WHY THIS EXISTS. A previous attempt passed all twelve of its own tests and was wrong on
every real receipt, because it invented the shape it tested against:

    if not isinstance(worker_signatures, dict):   # it is a LIST
        return <empty result>
    ... 'start' in stage and 'end' in stage       # they are layer_start / layer_end

Its fixtures matched its implementation, so the suite agreed with itself while disagreeing
with `receipt.build_receipt`. A test whose fixture is invented cannot detect a wrong guess
about the format — it can only confirm the guess.

So this file builds its receipts with the ACTUAL builder and signs with the ACTUAL signer.
The shape is not described here, it is produced here. If your module disagrees with what
`build_receipt` emits, that is your bug, and no amount of internally-consistent testing will
show it to you.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from identity_binding import pub_of, sign_participation  # noqa: E402
from receipt import build_receipt  # noqa: E402

from coverage import signature_coverage  # noqa: E402


def _run(spans, *, sign=None):
    """Build a genuine receipt for `spans` = [(node_id, start, end), ...]."""
    keys = {n: os.urandom(32).hex() for n, _, _ in spans}
    chain = [{"node_id": n, "layer_start": a, "layer_end": b} for n, a, b in spans]
    skeleton = build_receipt(run_id="r", model_id="m", prompt_tokens=[1, 2],
                             generated_tokens=[3, 4, 5], workers=chain, elapsed_s=1.0,
                             started_at=0.0, ended_at=1.0)
    out = skeleton["output_sha256"]
    to_sign = spans if sign is None else sign
    sigs = [sign_participation(keys[n], run_id="r", node_id=n, layer_start=a,
                               layer_end=b, output_sha256=out) for n, a, b in to_sign]
    rcpt = build_receipt(run_id="r", model_id="m", prompt_tokens=[1, 2],
                         generated_tokens=[3, 4, 5], workers=chain, elapsed_s=1.0,
                         started_at=0.0, ended_at=1.0, worker_signatures=sigs)
    return rcpt, {n: pub_of(k) for n, k in keys.items()}


def test_a_fully_signed_real_receipt_is_fully_covered():
    rcpt, roster = _run([("a", 0, 13), ("b", 13, 32)])
    r = signature_coverage(rcpt, roster)
    assert r["fully_covered"] is True, r
    assert r["covered"] == [(0, 32)], r
    assert r["gaps"] == [] and r["unclaimed"] == [], r


def test_worker_signatures_is_a_list_not_a_dict():
    """Pinning the exact wrong guess that made the first attempt inert."""
    rcpt, _ = _run([("a", 0, 13)])
    assert isinstance(rcpt["worker_signatures"], list)


def test_chain_stages_use_layer_start_and_layer_end():
    """The other wrong guess: the keys are not `start`/`end`."""
    rcpt, _ = _run([("a", 0, 13)])
    stage = rcpt["chain"][0]
    assert "layer_start" in stage and "layer_end" in stage, stage
    assert "start" not in stage and "end" not in stage, stage


def test_an_unsigned_stage_becomes_a_gap():
    rcpt, roster = _run([("a", 0, 13), ("b", 13, 32)], sign=[("a", 0, 13)])
    r = signature_coverage(rcpt, roster)
    assert r["fully_covered"] is False, r
    assert r["gaps"] == [(13, 32)], r


def test_an_empty_chain_is_not_fully_covered():
    rcpt, roster = _run([])
    assert signature_coverage(rcpt, roster)["fully_covered"] is False
