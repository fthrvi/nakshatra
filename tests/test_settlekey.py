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


def test_non_dict_receipts_raise_instead_of_colliding_on_one_fake_key():
    """⚠️ Regression for the collision bug: `except Exception: return sha256(b"")` used to
    collapse EVERY non-dict input onto the SAME constant hash, so an unrelated `None` receipt
    and a `123` receipt would "settle" as duplicates of each other. Fail loud instead — a
    receipt that isn't even a dict is a bug upstream, not a key to mint."""
    for receipt in [None, "not a dict", 123, []]:
        with pytest.raises(TypeError):
            settlement_key(receipt)


def test_incomplete_but_dict_shaped_receipts_still_produce_distinct_keys():
    """A dict missing fields (or with a malformed `worker_signatures`) is not the pathological
    case above — it still has `.get`, so it still produces a real, order-independent key, and
    a genuinely different receipt must not collide with it."""
    same_key_variants = [
        {"run_id": "run-123"},  # missing output_sha256 and worker_signatures
        {"run_id": "run-123", "worker_signatures": "not a list"},
        {"run_id": "run-123", "worker_signatures": [123, "string"]},
    ]
    different = {"run_id": "run-456"}

    keys = [settlement_key(r) for r in same_key_variants]  # must not raise
    # The malformed "worker_signatures" variants both normalize to [] (non-dict entries are
    # dropped, a non-list collapses to []), so they legitimately collide with the missing-
    # field case — but a genuinely different run_id must not collide with any of them.
    assert len(set(keys)) == 1
    assert settlement_key(different) not in keys