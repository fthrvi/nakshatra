import pytest
from listingcheck import validate_listing


def test_clean_listing():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert problems == []
    assert listing == content


def test_oversize_content_refused_without_parsing():
    # Create content longer than max_bytes that is also invalid JSON
    long_content = "x" * (8193)  # > max_bytes
    listing, problems = validate_listing(long_content, max_bytes=8192)
    assert listing is None
    assert problems == ["content exceeds maximum size"]


def test_invalid_json():
    content = "not valid json"
    listing, problems = validate_listing(content)
    assert listing is None
    assert problems == ["invalid JSON"]


def test_top_level_array():
    content = "[1, 2, 3]"
    listing, problems = validate_listing(content)
    assert listing is None
    assert problems == ["top-level is not an object"]


def test_missing_node_id():
    content = {
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "missing required field: node_id" in problems


def test_missing_pubkey():
    content = {
        "node_id": "node123",
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "missing required field: pubkey" in problems


def test_missing_accel():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "missing required field: accel" in problems


def test_missing_max_layers():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "missing required field: max_layers" in problems


def test_missing_ctx_tokens():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "missing required field: ctx_tokens" in problems


def test_missing_addrs():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "missing required field: addrs" in problems


def test_missing_relay():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "missing required field: relay" in problems


def test_missing_version():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "missing required field: version" in problems


def test_node_id_wrong_type():
    content = {
        "node_id": 123,
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'node_id' has wrong type" in problems


def test_pubkey_wrong_type():
    content = {
        "node_id": "node123",
        "pubkey": 123,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'pubkey' has wrong type" in problems


def test_accel_wrong_type():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": 123,
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'accel' has wrong type" in problems


def test_max_layers_wrong_type():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": "100",
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'max_layers' has wrong type" in problems


def test_ctx_tokens_wrong_type():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": "1000000",
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'ctx_tokens' has wrong type" in problems


def test_addrs_wrong_type():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": "127.0.0.1:8080",
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'addrs' has wrong type" in problems


def test_relay_wrong_type():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": 123,
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'relay' has wrong type" in problems


def test_version_wrong_type():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": 123
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'version' has wrong type" in problems


def test_unexpected_field():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0",
        "extra_field": "should_not_be_here"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "unexpected field: extra_field" in problems


def test_node_id_with_slash():
    content = {
        "node_id": "node/123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'node_id' has invalid value" in problems


def test_pubkey_uppercase():
    content = {
        "node_id": "node123",
        "pubkey": "A" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'pubkey' has invalid value" in problems


def test_addrs_too_many_entries():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": [f"addr{i}" for i in range(9)],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'addrs' has invalid value" in problems


def test_string_with_newline():
    content = {
        "node_id": "node\n123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'node_id' contains control characters" in problems


def test_string_with_escape_sequence():
    # The test expects a string that contains the literal ESC character (0x1b)
    # We construct it directly using chr(27) to ensure the actual escape character is present
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": chr(27) + "[31mred"
    }
    import json
    content_json = json.dumps(content)
    listing, problems = validate_listing(content_json)
    assert listing is None
    assert "field 'version' contains control characters" in problems


def test_nested_json_deep():
    # Create deeply nested JSON (50 levels)
    nested = "x"
    for _ in range(50):
        nested = {"nested": nested}
    content = str(nested).replace("'", '"')
    listing, problems = validate_listing(content)
    assert listing is None
    assert "unexpected field: nested" in problems


def test_none_input():
    listing, problems = validate_listing(None)
    assert listing is None
    assert "invalid JSON" in problems


def test_bytes_input():
    listing, problems = validate_listing(b"bytes")
    assert listing is None
    assert "invalid JSON" in problems


def test_node_id_too_short():
    content = {
        "node_id": "",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'node_id' has invalid value" in problems


def test_node_id_too_long():
    content = {
        "node_id": "a" * 65,
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'node_id' has invalid value" in problems


def test_pubkey_wrong_length():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 63,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'pubkey' has invalid value" in problems


def test_accel_invalid_value():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "opencl",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'accel' has invalid value" in problems


def test_max_layers_out_of_range():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 1001,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'max_layers' has invalid value" in problems


def test_ctx_tokens_out_of_range():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 10_000_001,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'ctx_tokens' has invalid value" in problems


def test_addrs_item_too_long():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["a" * 129],
        "relay": "wss://relay.example.com",
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'addrs' has invalid value" in problems


def test_relay_too_long():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "a" * 257,
        "version": "1.0"
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'relay' has invalid value" in problems


def test_version_too_long():
    content = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "cuda",
        "max_layers": 100,
        "ctx_tokens": 1000000,
        "addrs": ["127.0.0.1:8080"],
        "relay": "wss://relay.example.com",
        "version": "a" * 33
    }
    listing, problems = validate_listing(str(content).replace("'", '"'))
    assert listing is None
    assert "field 'version' has invalid value" in problems