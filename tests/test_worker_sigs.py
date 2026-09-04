import pytest
from identity_binding import verify_participation, UNPINNED_ACCEPT_ANY_KEY, pub_of
from worker_sigs import build_worker_signatures


def test_clean_three_stage_run_round_trips_and_verifies():
    """A clean three-stage run round-trips and all three verify."""
    stages = [
        {"node_id": "node_a", "layer_start": 0, "layer_end": 10},
        {"node_id": "node_b", "layer_start": 10, "layer_end": 20},
        {"node_id": "node_c", "layer_start": 20, "layer_end": 30},
    ]
    
    # Generate keys for each node
    keys = {}
    roster = {}
    for node_id in ["node_a", "node_b", "node_c"]:
        priv_hex = "a" * 64  # dummy key for testing
        keys[node_id] = priv_hex
        roster[node_id] = pub_of(priv_hex)
    
    run_id = "test_run_123"
    output_sha256 = "output_hash_abc123"
    
    signatures = build_worker_signatures(stages, keys, run_id=run_id, output_sha256=output_sha256)
    
    assert len(signatures) == 3
    
    # Verify each signature
    for entry in signatures:
        ok, reason = verify_participation(entry, run_id=run_id, output_sha256=output_sha256, pinned=roster)
        assert ok, f"Signature failed verification: {reason}"


def test_stage_with_no_key_is_skipped():
    """A stage with no key is skipped."""
    stages = [
        {"node_id": "node_a", "layer_start": 0, "layer_end": 10},
        {"node_id": "node_unknown", "layer_start": 10, "layer_end": 20},  # no key
        {"node_id": "node_b", "layer_start": 20, "layer_end": 30},
    ]
    
    keys = {
        "node_a": "a" * 64,
        "node_b": "b" * 64,
    }
    roster = {
        "node_a": pub_of(keys["node_a"]),
        "node_b": pub_of(keys["node_b"]),
    }
    
    run_id = "test_run_456"
    output_sha256 = "output_hash_def456"
    
    signatures = build_worker_signatures(stages, keys, run_id=run_id, output_sha256=output_sha256)
    
    # Only 2 signatures should be generated (node_unknown is skipped)
    assert len(signatures) == 2
    assert signatures[0]["node_id"] == "node_a"
    assert signatures[1]["node_id"] == "node_b"
    
    # Verify both signatures
    for entry in signatures:
        ok, reason = verify_participation(entry, run_id=run_id, output_sha256=output_sha256, pinned=roster)
        assert ok, f"Signature failed verification: {reason}"


def test_zero_width_stage_is_skipped():
    """A zero-width stage is skipped."""
    stages = [
        {"node_id": "node_a", "layer_start": 0, "layer_end": 10},
        {"node_id": "node_b", "layer_start": 10, "layer_end": 10},  # zero width
        {"node_id": "node_c", "layer_start": 10, "layer_end": 20},
    ]
    
    keys = {
        "node_a": "a" * 64,
        "node_b": "b" * 64,
        "node_c": "c" * 64,
    }
    roster = {
        "node_a": pub_of(keys["node_a"]),
        "node_b": pub_of(keys["node_b"]),
        "node_c": pub_of(keys["node_c"]),
    }
    
    run_id = "test_run_789"
    output_sha256 = "output_hash_ghi789"
    
    signatures = build_worker_signatures(stages, keys, run_id=run_id, output_sha256=output_sha256)
    
    # Only 2 signatures should be generated (zero-width stage is skipped)
    assert len(signatures) == 2
    assert signatures[0]["node_id"] == "node_a"
    assert signatures[1]["node_id"] == "node_c"
    
    # Verify both signatures
    for entry in signatures:
        ok, reason = verify_participation(entry, run_id=run_id, output_sha256=output_sha256, pinned=roster)
        assert ok, f"Signature failed verification: {reason}"


def test_overlapping_stages_raise_value_error():
    """Overlapping stages for one node raise ValueError."""
    stages = [
        {"node_id": "node_a", "layer_start": 0, "layer_end": 10},
        {"node_id": "node_a", "layer_start": 5, "layer_end": 15},  # overlaps with first
    ]
    
    keys = {
        "node_a": "a" * 64,
    }
    
    run_id = "test_run_overlap"
    output_sha256 = "output_hash_overlap"
    
    with pytest.raises(ValueError, match="Overlapping layer ranges"):
        build_worker_signatures(stages, keys, run_id=run_id, output_sha256=output_sha256)


def test_non_overlapping_stages_for_same_node_both_signed():
    """Non-overlapping stages for the same node are both signed."""
    stages = [
        {"node_id": "node_a", "layer_start": 0, "layer_end": 10},
        {"node_id": "node_a", "layer_start": 20, "layer_end": 30},  # non-overlapping
    ]
    
    keys = {
        "node_a": "a" * 64,
    }
    roster = {
        "node_a": pub_of(keys["node_a"]),
    }
    
    run_id = "test_run_non_overlap"
    output_sha256 = "output_hash_non_overlap"
    
    signatures = build_worker_signatures(stages, keys, run_id=run_id, output_sha256=output_sha256)
    
    # Both stages should be signed
    assert len(signatures) == 2
    assert all(sig["node_id"] == "node_a" for sig in signatures)
    
    # Verify both signatures
    for entry in signatures:
        ok, reason = verify_participation(entry, run_id=run_id, output_sha256=output_sha256, pinned=roster)
        assert ok, f"Signature failed verification: {reason}"


def test_order_is_preserved():
    """Order is preserved."""
    stages = [
        {"node_id": "node_c", "layer_start": 0, "layer_end": 10},
        {"node_id": "node_a", "layer_start": 10, "layer_end": 20},
        {"node_id": "node_b", "layer_start": 20, "layer_end": 30},
    ]
    
    keys = {
        "node_a": "a" * 64,
        "node_b": "b" * 64,
        "node_c": "c" * 64,
    }
    
    run_id = "test_run_order"
    output_sha256 = "output_hash_order"
    
    signatures = build_worker_signatures(stages, keys, run_id=run_id, output_sha256=output_sha256)
    
    # Order should be preserved
    assert len(signatures) == 3
    assert signatures[0]["node_id"] == "node_c"
    assert signatures[1]["node_id"] == "node_a"
    assert signatures[2]["node_id"] == "node_b"


def test_entry_signed_for_run_id_a_does_not_verify_under_run_id_b():
    """An entry signed for run_id="a" does not verify under run_id="b"."""
    stages = [
        {"node_id": "node_a", "layer_start": 0, "layer_end": 10},
    ]
    
    keys = {
        "node_a": "a" * 64,
    }
    roster = {
        "node_a": pub_of(keys["node_a"]),
    }
    
    run_id_a = "run_a"
    run_id_b = "run_b"
    output_sha256 = "output_hash"
    
    signatures = build_worker_signatures(stages, keys, run_id=run_id_a, output_sha256=output_sha256)
    
    assert len(signatures) == 1
    
    # Should verify with correct run_id
    ok, reason = verify_participation(signatures[0], run_id=run_id_a, output_sha256=output_sha256, pinned=roster)
    assert ok, f"Signature should verify with correct run_id: {reason}"
    
    # Should NOT verify with wrong run_id
    ok, reason = verify_participation(signatures[0], run_id=run_id_b, output_sha256=output_sha256, pinned=roster)
    assert not ok, f"Signature should not verify with wrong run_id, but got: {reason}"