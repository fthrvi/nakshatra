import pytest
from assign import assign_layers


def test_one_node_holding_everything():
    nodes = [
        {"node_id": "node1", "max_layers": 12, "rtt_ms": 10.0}
    ]
    result = assign_layers(nodes, 12)
    assert result == [
        {"node_id": "node1", "layer_start": 0, "layer_end": 12}
    ]


def test_three_nodes_splitting_exactly():
    nodes = [
        {"node_id": "node1", "max_layers": 4, "rtt_ms": 10.0},
        {"node_id": "node2", "max_layers": 4, "rtt_ms": 20.0},
        {"node_id": "node3", "max_layers": 4, "rtt_ms": 30.0}
    ]
    result = assign_layers(nodes, 12)
    assert result == [
        {"node_id": "node1", "layer_start": 0, "layer_end": 4},
        {"node_id": "node2", "layer_start": 4, "layer_end": 8},
        {"node_id": "node3", "layer_start": 8, "layer_end": 12}
    ]


def test_rtt_ordering_places_closest_node_first():
    nodes = [
        {"node_id": "far_node", "max_layers": 6, "rtt_ms": 50.0},
        {"node_id": "near_node", "max_layers": 6, "rtt_ms": 10.0},
        {"node_id": "mid_node", "max_layers": 6, "rtt_ms": 30.0}
    ]
    result = assign_layers(nodes, 12)
    # Near node should get first 6 layers, mid node next 6
    assert result == [
        {"node_id": "near_node", "layer_start": 0, "layer_end": 6},
        {"node_id": "mid_node", "layer_start": 6, "layer_end": 12}
    ]


def test_node_with_max_layers_zero_is_skipped():
    nodes = [
        {"node_id": "node1", "max_layers": 0, "rtt_ms": 10.0},
        {"node_id": "node2", "max_layers": 6, "rtt_ms": 20.0},
        {"node_id": "node3", "max_layers": 6, "rtt_ms": 30.0}
    ]
    result = assign_layers(nodes, 12)
    assert result == [
        {"node_id": "node2", "layer_start": 0, "layer_end": 6},
        {"node_id": "node3", "layer_start": 6, "layer_end": 12}
    ]


def test_total_capacity_one_short_returns_empty():
    nodes = [
        {"node_id": "node1", "max_layers": 5, "rtt_ms": 10.0},
        {"node_id": "node2", "max_layers": 5, "rtt_ms": 20.0}
    ]
    result = assign_layers(nodes, 11)
    assert result == []


def test_excess_capacity_no_empty_trailing_spans():
    nodes = [
        {"node_id": "node1", "max_layers": 10, "rtt_ms": 10.0},
        {"node_id": "node2", "max_layers": 10, "rtt_ms": 20.0},
        {"node_id": "node3", "max_layers": 10, "rtt_ms": 30.0}
    ]
    result = assign_layers(nodes, 12)
    assert result == [
        {"node_id": "node1", "layer_start": 0, "layer_end": 10},
        {"node_id": "node2", "layer_start": 10, "layer_end": 12}
    ]


def test_contiguous_cover_no_gaps_or_overlaps():
    nodes = [
        {"node_id": "node1", "max_layers": 3, "rtt_ms": 10.0},
        {"node_id": "node2", "max_layers": 4, "rtt_ms": 20.0},
        {"node_id": "node3", "max_layers": 5, "rtt_ms": 30.0}
    ]
    result = assign_layers(nodes, 12)
    
    # Check contiguity
    assert len(result) > 0
    assert result[0]["layer_start"] == 0
    assert result[-1]["layer_end"] == 12
    
    for i in range(len(result) - 1):
        assert result[i]["layer_end"] == result[i+1]["layer_start"], "Gap or overlap detected"
    
    # Verify no overlaps
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert result[i]["layer_end"] <= result[j]["layer_start"], "Overlap detected"


def test_deterministic_tie_breaking_on_equal_rtt():
    nodes = [
        {"node_id": "b_node", "max_layers": 6, "rtt_ms": 10.0},
        {"node_id": "a_node", "max_layers": 6, "rtt_ms": 10.0},
        {"node_id": "c_node", "max_layers": 6, "rtt_ms": 10.0}
    ]
    result = assign_layers(nodes, 12)
    
    # Should be ordered by node_id when rtt_ms is equal
    assert result == [
        {"node_id": "a_node", "layer_start": 0, "layer_end": 6},
        {"node_id": "b_node", "layer_start": 6, "layer_end": 12}
    ]


def test_malformed_input_non_list_nodes():
    result = assign_layers("not a list", 10)
    assert result == []


def test_malformed_input_non_dict_entries():
    nodes = [
        {"node_id": "node1", "max_layers": 5, "rtt_ms": 10.0},
        "not a dict",
        {"node_id": "node2", "max_layers": 5, "rtt_ms": 20.0}
    ]
    result = assign_layers(nodes, 10)
    assert result == []


def test_malformed_input_missing_keys():
    nodes = [
        {"node_id": "node1", "max_layers": 5},  # missing rtt_ms
        {"node_id": "node2", "max_layers": 5, "rtt_ms": 20.0}
    ]
    result = assign_layers(nodes, 10)
    assert result == []


def test_malformed_input_non_positive_n_layers():
    nodes = [
        {"node_id": "node1", "max_layers": 5, "rtt_ms": 10.0}
    ]
    assert assign_layers(nodes, 0) == []
    assert assign_layers(nodes, -1) == []


def test_malformed_input_invalid_types():
    nodes = [
        {"node_id": 123, "max_layers": 5, "rtt_ms": 10.0}  # node_id not string
    ]
    assert assign_layers(nodes, 10) == []
    
    nodes = [
        {"node_id": "node1", "max_layers": "five", "rtt_ms": 10.0}  # max_layers not int
    ]
    assert assign_layers(nodes, 10) == []
    
    nodes = [
        {"node_id": "node1", "max_layers": 5, "rtt_ms": "ten"}  # rtt_ms not number
    ]
    assert assign_layers(nodes, 10) == []


def test_empty_nodes_list():
    result = assign_layers([], 10)
    assert result == []


def test_all_nodes_skipped_due_to_zero_max_layers():
    nodes = [
        {"node_id": "node1", "max_layers": 0, "rtt_ms": 10.0},
        {"node_id": "node2", "max_layers": 0, "rtt_ms": 20.0}
    ]
    result = assign_layers(nodes, 10)
    assert result == []


def test_partial_assignment_returns_empty():
    # Total capacity is 8 but n_layers is 10
    nodes = [
        {"node_id": "node1", "max_layers": 3, "rtt_ms": 10.0},
        {"node_id": "node2", "max_layers": 5, "rtt_ms": 20.0}
    ]
    result = assign_layers(nodes, 10)
    assert result == []