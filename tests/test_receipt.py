"""
Unit tests for verifiable run receipts (scripts/receipt.py). Pure — no GPU/cluster.

Proves: the output hash is stable + order-sensitive; a well-formed receipt verifies; and
EVERY checkable claim has a tamper test that trips verify_receipt (hash, count, tok/s math,
timestamps, worker distinctness, layer-map contiguity).
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from receipt import build_receipt, verify_receipt, output_sha256, RECEIPT_VERSION  # noqa: E402


def _workers():
    # 3 distinct stages covering a 24-layer model contiguously.
    return [
        {"node_id": "w0", "address": "10.0.0.1:5530", "spki_hash": "aaa",
         "layer_start": 0, "layer_end": 8, "backend": "rocm", "mean_rpc_ms": 12.0},
        {"node_id": "w1", "address": "10.0.0.2:5531", "spki_hash": "bbb",
         "layer_start": 8, "layer_end": 16, "backend": "metal", "mean_rpc_ms": 40.0},
        {"node_id": "w2", "address": "10.0.0.3:5532", "spki_hash": "ccc",
         "layer_start": 16, "layer_end": 24, "backend": "metal", "mean_rpc_ms": 38.0},
    ]


def _receipt(**over):
    kw = dict(
        run_id="run-123", model_id="dsr1-llama-8b-q4",
        prompt_tokens=[1, 2, 3, 4], generated_tokens=[10, 11, 12, 13, 14],
        workers=_workers(), elapsed_s=2.0, started_at=1000.0, ended_at=1002.0,
    )
    kw.update(over)
    return build_receipt(**kw)


# ---------------------------------------------------------------- hash

def test_output_sha256_stable_and_order_sensitive():
    assert output_sha256([1, 2, 3]) == output_sha256([1, 2, 3])
    assert output_sha256([1, 2, 3]) != output_sha256([1, 3, 2])   # order matters
    assert output_sha256([]) == output_sha256([])                 # empty is well-defined
    assert len(output_sha256([1])) == 64                          # hex sha256


# ---------------------------------------------------------------- well-formed

def test_valid_receipt_verifies():
    r = _receipt()
    ok, problems = verify_receipt(r)
    assert ok, problems
    assert problems == []
    assert r["receipt_version"] == RECEIPT_VERSION
    assert r["signed_by"] == "coordinator"
    assert r["worker_signatures"] == []          # v1 placeholder present
    assert r["tok_per_s"] == 2.5                  # 5 tokens / 2.0 s
    assert "reproducibility" in r["claims"]       # honesty scope embedded


def test_empty_generation_is_valid():
    r = _receipt(generated_tokens=[], elapsed_s=0.0)
    ok, problems = verify_receipt(r)
    assert ok, problems
    assert r["tok_per_s"] == 0.0


# ---------------------------------------------------------------- tamper detection

def test_tamper_flip_token_fails_hash():
    r = _receipt()
    r["generated_tokens"][2] = 999          # change output but not the stored hash
    ok, problems = verify_receipt(r)
    assert not ok
    assert any("output_sha256" in p for p in problems)


def test_tamper_corrupt_hash_fails():
    r = _receipt()
    r["output_sha256"] = "0" * 64
    ok, problems = verify_receipt(r)
    assert not ok and any("output_sha256" in p for p in problems)


def test_tamper_count_mismatch_fails():
    r = _receipt()
    r["n_generated"] = 99
    ok, problems = verify_receipt(r)
    assert not ok and any("n_generated" in p for p in problems)


def test_tamper_tok_per_s_fails_math():
    r = _receipt()
    r["tok_per_s"] = 100.0                   # inconsistent with 5/2.0
    ok, problems = verify_receipt(r)
    assert not ok and any("tok_per_s" in p for p in problems)


def test_tamper_timestamps_non_monotonic_fails():
    r = _receipt(started_at=1002.0, ended_at=1000.0)
    ok, problems = verify_receipt(r)
    assert not ok and any("ended_at precedes" in p for p in problems)


def test_duplicate_node_id_fails_distinctness():
    ws = _workers()
    ws[1]["node_id"] = "w0"                   # two stages claim the same machine id
    r = _receipt(workers=ws)
    ok, problems = verify_receipt(r)
    assert not ok and any("duplicate node_id" in p for p in problems)


def test_duplicate_spki_fails_distinctness():
    ws = _workers()
    ws[2]["spki_hash"] = "aaa"                # same TLS key on two stages = likely one box
    r = _receipt(workers=ws)
    ok, problems = verify_receipt(r)
    assert not ok and any("spki_hash" in p for p in problems)


def test_layer_map_gap_fails():
    ws = _workers()
    ws[1]["layer_end"] = 15                   # leaves a gap [15,16)
    r = _receipt(workers=ws)
    ok, problems = verify_receipt(r)
    assert not ok and any("gap/overlap" in p for p in problems)


def test_layer_map_not_starting_at_zero_fails():
    ws = _workers()
    ws[0]["layer_start"] = 1                  # doesn't cover layer 0
    r = _receipt(workers=ws)
    ok, problems = verify_receipt(r)
    assert not ok and any("does not start at 0" in p for p in problems)


def test_overlap_fails():
    ws = _workers()
    ws[1]["layer_start"] = 6                  # overlaps w0's [0,8)
    r = _receipt(workers=ws)
    ok, problems = verify_receipt(r)
    assert not ok and any("gap/overlap" in p for p in problems)


def test_missing_required_field_fails():
    r = _receipt()
    del r["output_sha256"]
    ok, problems = verify_receipt(r)
    assert not ok and any("missing field" in p for p in problems)


def test_passthrough_provenance_and_signatures():
    r = _receipt(engine_provenance=["Nakshatra-Prov:engine=abc123;code=def456"],
                 worker_signatures=[{"node_id": "w0", "sig": "deadbeef"}])
    ok, problems = verify_receipt(r)
    assert ok, problems
    assert r["engine_provenance"] == ["Nakshatra-Prov:engine=abc123;code=def456"]
    assert r["worker_signatures"][0]["node_id"] == "w0"


def test_verify_does_not_mutate():
    r = _receipt()
    before = copy.deepcopy(r)
    verify_receipt(r)
    assert r == before


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
