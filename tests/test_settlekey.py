import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
from settlekey import settlement_key, is_duplicate


def test_same_receipt_twice_gives_same_key():
    receipt = {
        "run_id": "run-123",
        "output_sha256": "abc123",
        "worker_signatures": [
            {"node_id": "node1", "layer_start": 0, "layer_end": 10},
            {"node_id": "node2", "layer_start": 10, "layer_end": 20}
        ]
    }
    key1 = settlement_key(receipt)
    key2 = settlement_key(receipt)
    assert key1 == key2


def test_differing_only_in_elapsed_s_started_at_tok_per_s_gives_same_key():
    receipt1 = {
        "run_id": "run-123",
        "output_sha256": "abc123",
        "worker_signatures": [
            {"node_id": "node1", "layer_start": 0, "layer_end": 10}
        ],
        "elapsed_s": 1.5,
        "started_at": "2024-01-01T00:00:00Z",
        "tok_per_s": 100.0
    }
    receipt2 = {
        "run_id": "run-123",
        "output_sha256": "abc123",
        "worker_signatures": [
            {"node_id": "node1", "layer_start": 0, "layer_end": 10}
        ],
        "elapsed_s": 2.0,
        "started_at": "2024-01-01T00:00:01Z",
        "tok_per_s": 99.0
    }
    key1 = settlement_key(receipt1)
    key2 = settlement_key(receipt2)
    assert key1 == key2


def test_different_run_id_differs():
    receipt1 = {
        "run_id": "run-123",
        "output_sha256": "abc123",
        "worker_signatures": [
            {"node_id": "node1", "layer_start": 0, "layer_end": 10}
        ]
    }
    receipt2 = {
        "run_id": "run-456",
        "output_sha256": "abc123",
        "worker_signatures": [
            {"node_id": "node1", "layer_start": 0, "layer_end": 10}
        ]
    }
    key1 = settlement_key(receipt1)
    key2 = settlement_key(receipt2)
    assert key1 != key2


def test_different_output_sha256_differs():
    receipt1 = {
        "run_id": "run-123",
        "output_sha256": "abc123",
        "worker_signatures": [
            {"node_id": "node1", "layer_start": 0, "layer_end": 10}
        ]
    }
    receipt2 = {
        "run_id": "run-123",
        "output_sha256": "def456",
        "worker_signatures": [
            {"node_id": "node1", "layer_start": 0, "layer_end": 10}
        ]
    }
    key1 = settlement_key(receipt1)
    key2 = settlement_key(receipt2)
    assert key1 != key2


def test_same_run_different_signature_set_differs():
    receipt1 = {
        "run_id": "run-123",
        "output_sha256": "abc123",
        "worker_signatures": [
            {"node_id": "node1", "layer_start": 0, "layer_end": 10}
        ]
    }
    receipt2 = {
        "run_id": "run-123",
        "output_sha256": "abc123",
        "worker_signatures": [
            {"node_id": "node1", "layer_start": 0, "layer_end": 10},
            {"node_id": "node2", "layer_start": 10, "layer_end": 20}
        ]
    }
    key1 = settlement_key(receipt1)
    key2 = settlement_key(receipt2)
    assert key1 != key2


def test_signature_order_does_not_affect_key():
    receipt1 = {
        "run_id": "run-123",
        "output_sha256": "abc123",
        "worker_signatures": [
            {"node_id": "node2", "layer_start": 10, "layer_end": 20},
            {"node_id": "node1", "layer_start": 0, "layer_end": 10}
        ]
    }
    receipt2 = {
        "run_id": "run-123",
        "output_sha256": "abc123",
        "worker_signatures": [
            {"node_id": "node1", "layer_start": 0, "layer_end": 10},
            {"node_id": "node2", "layer_start": 10, "layer_end": 20}
        ]
    }
    key1 = settlement_key(receipt1)
    key2 = settlement_key(receipt2)
    assert key1 == key2


def test_is_duplicate_on_fresh_key():
    seen = {}
    key = "some-key"
    is_dup, timestamp = is_duplicate(key, seen)
    assert is_dup is False
    assert timestamp == ""


def test_is_duplicate_on_seen_key():
    seen = {"some-key": "2024-01-01T00:00:00Z"}
    key = "some-key"
    is_dup, timestamp = is_duplicate(key, seen)
    assert is_dup is True
    assert timestamp == "2024-01-01T00:00:00Z"


def test_is_duplicate_does_not_mutate_seen():
    seen = {"some-key": "2024-01-01T00:00:00Z"}
    original_seen = dict(seen)
    key = "another-key"
    is_duplicate(key, seen)
    assert seen == original_seen


def test_is_duplicate_does_not_mutate_seen_on_duplicate():
    seen = {"some-key": "2024-01-01T00:00:00Z"}
    original_seen = dict(seen)
    key = "some-key"
    is_duplicate(key, seen)
    assert seen == original_seen


def test_malformed_receipt_still_produces_key():
    malformed_receipts = [
        None,
        "not a dict",
        123,
        [],
        {"run_id": "run-123"},  # missing output_sha256 and worker_signatures
        {"run_id": "run-123", "output_sha256": "abc", "worker_signatures": "not a list"},
        {"run_id": "run-123", "output_sha256": "abc", "worker_signatures": [123, "string"]},
    ]
    
    keys = []
    for receipt in malformed_receipts:
        try:
            key = settlement_key(receipt)
            keys.append(key)
            # Should not raise
        except Exception:
            pytest.fail(f"settlement_key raised on malformed receipt: {receipt}")
    
    # All malformed receipts should produce deterministic keys
    assert len(keys) == len(malformed_receipts)