"""`signed_by` must describe the receipt it is on, not a constant.

⚠️ It was hardcoded to "coordinator" while `worker_signatures` sat beside it as a documented
placeholder. So a receipt that genuinely carried worker signatures would still have announced
itself as coordinator-asserted — and a consumer deciding whether to trust it reads exactly
that field. The bug only becomes visible the moment the feature starts working, which is the
worst possible time for a provenance field to be wrong.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from identity_binding import (  # noqa: E402
    UNPINNED_ACCEPT_ANY_KEY, creditable_accounts, pub_of, sign_participation,
)
from receipt import build_receipt  # noqa: E402


def _receipt(worker_signatures=None):
    return build_receipt(
        run_id="r", model_id="m",
        prompt_tokens=[1, 2, 3], generated_tokens=[4, 5],
        workers=[{"node_id": "alice", "layer_start": 0, "layer_end": 13}],
        elapsed_s=1.0, started_at=0.0, ended_at=1.0,
        worker_signatures=worker_signatures,
    )


def test_no_signatures_is_coordinator_asserted():
    assert _receipt()["signed_by"] == "coordinator"
    assert _receipt([])["signed_by"] == "coordinator"


def test_signatures_present_means_workers_signed():
    priv = os.urandom(32).hex()
    r = _receipt([sign_participation(priv, run_id="r", node_id="alice",
                                     layer_start=0, layer_end=13,
                                     output_sha256="unused-for-this-assertion")])
    assert r["signed_by"] == "workers"


def test_signed_by_workers_is_not_a_claim_that_they_verify():
    """PROVENANCE, NOT VALIDITY — the distinction the credit path depends on.

    A receipt full of garbage signatures still says "workers": that field reports what the
    receipt CONTAINS. Whether any of it is true is `creditable_accounts`' job, against a
    pinned roster. Conflating the two would let a receipt vouch for itself.
    """
    junk = [{"node_id": "alice", "pubkey": "00" * 32, "layer_start": 0,
             "layer_end": 13, "sig": "bm90YXNpZw=="}]
    r = _receipt(junk)
    assert r["signed_by"] == "workers"

    accounts, problems = creditable_accounts(r, pinned={"alice": "11" * 32})
    assert accounts == [], "a receipt must not be able to certify itself"
    assert problems


def test_a_real_signature_over_the_receipts_own_output_is_creditable():
    """End to end on the honest path: sign the receipt's actual output hash, then credit."""
    priv = os.urandom(32).hex()
    skeleton = _receipt()                       # build once to learn output_sha256
    entry = sign_participation(priv, run_id="r", node_id="alice", layer_start=0,
                               layer_end=13, output_sha256=skeleton["output_sha256"])
    r = _receipt([entry])
    accounts, problems = creditable_accounts(r, pinned={"alice": pub_of(priv)})
    assert accounts == [f"nak:{pub_of(priv)}"], problems
