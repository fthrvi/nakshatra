import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
from identity_binding import sign_participation, pub_of, account_id, UNPINNED_ACCEPT_ANY_KEY
from units import credit_units


def make_receipt(n_generated: int, signatures: list[dict], run_id: str = "run1", output_sha256: str = "output1"):
    return {
        "run_id": run_id,
        "output_sha256": output_sha256,
        "n_generated": n_generated,
        "worker_signatures": signatures,
    }


def test_single_verified_stage():
    # Create keys and sign a stage
    priv1 = "a" * 64
    pub1 = pub_of(priv1)
    pinned = {"node1": pub1}
    
    sig = sign_participation(
        priv1,
        run_id="run1",
        node_id="node1",
        layer_start=0,
        layer_end=10,
        output_sha256="output1"
    )
    
    receipt = make_receipt(100, [sig])
    result = credit_units(receipt, pinned)
    
    expected_account = account_id(pub1)
    assert result == {expected_account: 100 * 10}  # 100 * (10-0)


def test_three_stages_three_accounts():
    # Create three different key pairs
    privs = ["a" * 64, "b" * 64, "c" * 64]
    pubs = [pub_of(p) for p in privs]
    pinned = {
        "node1": pubs[0],
        "node2": pubs[1],
        "node3": pubs[2]
    }
    
    sigs = [
        sign_participation(privs[0], run_id="run1", node_id="node1", layer_start=0, layer_end=5, output_sha256="output1"),
        sign_participation(privs[1], run_id="run1", node_id="node2", layer_start=5, layer_end=10, output_sha256="output1"),
        sign_participation(privs[2], run_id="run1", node_id="node3", layer_start=10, layer_end=15, output_sha256="output1"),
    ]
    
    receipt = make_receipt(10, sigs)
    result = credit_units(receipt, pinned)
    
    expected = {
        account_id(pubs[0]): 10 * 5,
        account_id(pubs[1]): 10 * 5,
        account_id(pubs[2]): 10 * 5,
    }
    assert result == expected


def test_two_disjoint_spans_one_account():
    # One account with two disjoint spans
    priv1 = "a" * 64
    pub1 = pub_of(priv1)
    pinned = {"node1": pub1, "node2": pub1}
    
    sigs = [
        sign_participation(priv1, run_id="run1", node_id="node1", layer_start=0, layer_end=5, output_sha256="output1"),
        sign_participation(priv1, run_id="run1", node_id="node2", layer_start=10, layer_end=15, output_sha256="output1"),
    ]
    
    receipt = make_receipt(10, sigs)
    result = credit_units(receipt, pinned)
    
    expected_account = account_id(pub1)
    # Total span: (5-0) + (15-10) = 10, units = 10 * 10 = 100
    assert result == {expected_account: 10 * 10}


def test_overlapping_spans_raises():
    # One account with overlapping spans
    priv1 = "a" * 64
    pub1 = pub_of(priv1)
    pinned = {"node1": pub1, "node2": pub1}
    
    sigs = [
        sign_participation(priv1, run_id="run1", node_id="node1", layer_start=0, layer_end=10, output_sha256="output1"),
        sign_participation(priv1, run_id="run1", node_id="node2", layer_start=5, layer_end=15, output_sha256="output1"),
    ]
    
    receipt = make_receipt(10, sigs)
    
    with pytest.raises(ValueError, match=account_id(pub1)):
        credit_units(receipt, pinned)


def test_unverified_entry_contributes_nothing():
    priv1 = "a" * 64
    pub1 = pub_of(priv1)
    pinned = {"node1": pub1}
    
    # Valid signature
    sig1 = sign_participation(priv1, run_id="run1", node_id="node1", layer_start=0, layer_end=5, output_sha256="output1")
    
    # Invalid signature (wrong node_id)
    sig2 = sign_participation(priv1, run_id="run1", node_id="node2", layer_start=5, layer_end=10, output_sha256="output1")
    
    receipt = make_receipt(10, [sig1, sig2])
    result = credit_units(receipt, pinned)
    
    expected_account = account_id(pub1)
    assert result == {expected_account: 10 * 5}


def test_two_node_ids_one_pubkey_collapses():
    # Two node_ids using the same key
    priv1 = "a" * 64
    pub1 = pub_of(priv1)
    pinned = {"node1": pub1, "node2": pub1}
    
    sigs = [
        sign_participation(priv1, run_id="run1", node_id="node1", layer_start=0, layer_end=5, output_sha256="output1"),
        sign_participation(priv1, run_id="run1", node_id="node2", layer_start=10, layer_end=15, output_sha256="output1"),
    ]
    
    receipt = make_receipt(10, sigs)
    result = credit_units(receipt, pinned)
    
    expected_account = account_id(pub1)
    assert result == {expected_account: 10 * 10}


def test_n_generated_zero_gives_empty_dict():
    priv1 = "a" * 64
    pub1 = pub_of(priv1)
    pinned = {"node1": pub1}
    
    sig = sign_participation(priv1, run_id="run1", node_id="node1", layer_start=0, layer_end=10, output_sha256="output1")
    
    receipt = make_receipt(0, [sig])
    result = credit_units(receipt, pinned)
    
    assert result == {}


def test_malformed_receipt_gives_empty_dict():
    # Non-dict receipt
    assert credit_units("not a dict", {}) == {}
    
    # Missing worker_signatures
    assert credit_units({"run_id": "run1", "output_sha256": "output1", "n_generated": 10}, {}) == {}
    
    # Non-list worker_signatures
    assert credit_units({"run_id": "run1", "output_sha256": "output1", "n_generated": 10, "worker_signatures": "not a list"}, {}) == {}
    
    # Missing n_generated
    assert credit_units({"run_id": "run1", "output_sha256": "output1", "worker_signatures": []}, {}) == {}
    
    # n_generated not int
    assert credit_units({"run_id": "run1", "output_sha256": "output1", "n_generated": "10", "worker_signatures": []}, {}) == {}
    
    # n_generated negative
    assert credit_units({"run_id": "run1", "output_sha256": "output1", "n_generated": -1, "worker_signatures": []}, {}) == {}


def test_missing_run_id_or_output_sha256():
    priv1 = "a" * 64
    pub1 = pub_of(priv1)
    pinned = {"node1": pub1}
    
    sig = sign_participation(priv1, run_id="run1", node_id="node1", layer_start=0, layer_end=10, output_sha256="output1")
    
    # Missing run_id
    receipt = {"output_sha256": "output1", "n_generated": 10, "worker_signatures": [sig]}
    assert credit_units(receipt, pinned) == {}
    
    # Missing output_sha256
    receipt = {"run_id": "run1", "n_generated": 10, "worker_signatures": [sig]}
    assert credit_units(receipt, pinned) == {}


def test_unverified_entry_skipped():
    # Create two keys
    priv1 = "a" * 64
    priv2 = "b" * 64
    pub1 = pub_of(priv1)
    pub2 = pub_of(priv2)
    
    pinned = {"node1": pub1}
    
    # Valid signature
    sig1 = sign_participation(priv1, run_id="run1", node_id="node1", layer_start=0, layer_end=5, output_sha256="output1")
    
    # Invalid signature (wrong pinned key)
    sig2 = sign_participation(priv2, run_id="run1", node_id="node2", layer_start=5, layer_end=10, output_sha256="output1")
    
    receipt = make_receipt(10, [sig1, sig2])
    result = credit_units(receipt, pinned)
    
    expected_account = account_id(pub1)
    assert result == {expected_account: 10 * 5}