import pytest
from registration import registration_payload


def test_clean_payload_exact_keys():
    """A clean payload asserting the EXACT key set."""
    facts = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "accel": "some_accel",
        "max_layers": 5,
        "ctx_tokens": 100,
        "addrs": ["127.0.0.1:8080"],
        "version": "1.0.0",
    }
    result = registration_payload(facts)
    assert set(result.keys()) == {"node_id", "pubkey", "accel", "max_layers", "ctx_tokens", "addrs", "version"}
    assert result["node_id"] == "node123"
    assert result["pubkey"] == "a" * 64
    assert result["accel"] == "some_accel"
    assert result["max_layers"] == 5
    assert result["ctx_tokens"] == 100
    assert result["addrs"] == ["127.0.0.1:8080"]
    assert result["version"] == "1.0.0"


def test_private_key_not_in_output():
    """Private key under three different names must all be absent from the output."""
    private_key = "a" * 64  # 64-hex private key
    facts = {
        "node_id": "node123",
        "pubkey": "b" * 64,
        "priv": private_key,
        "private_key": private_key,
        "secret": private_key,
    }
    result = registration_payload(facts)
    assert "priv" not in result
    assert "private_key" not in result
    assert "secret" not in result
    assert private_key not in result.values()


def test_join_token_absent():
    """Join token must be absent from the output."""
    facts = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "join_token": "secret_token_123",
    }
    result = registration_payload(facts)
    assert "join_token" not in result
    assert "secret_token_123" not in result.values()


def test_hostname_absent():
    """Hostname must be absent from the output."""
    facts = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "hostname": "myhost.example.com",
    }
    result = registration_payload(facts)
    assert "hostname" not in result
    assert "myhost.example.com" not in result.values()


def test_bad_pubkey_raises():
    """A bad pubkey raises ValueError."""
    # Too short
    with pytest.raises(ValueError):
        registration_payload({"node_id": "node123", "pubkey": "abc"})
    
    # Too long
    with pytest.raises(ValueError):
        registration_payload({"node_id": "node123", "pubkey": "a" * 65})
    
    # Uppercase
    with pytest.raises(ValueError):
        registration_payload({"node_id": "node123", "pubkey": "A" * 64})
    
    # Non-hex
    with pytest.raises(ValueError):
        registration_payload({"node_id": "node123", "pubkey": "g" * 64})
    
    # Not a string
    with pytest.raises(ValueError):
        registration_payload({"node_id": "node123", "pubkey": 12345})


def test_missing_node_id_raises():
    """Missing node_id raises ValueError."""
    facts = {"pubkey": "a" * 64}
    with pytest.raises(ValueError):
        registration_payload(facts)


def test_negative_numbers_floored():
    """Negative numbers for max_layers and ctx_tokens are floored at 0."""
    facts = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "max_layers": -5,
        "ctx_tokens": -10,
    }
    result = registration_payload(facts)
    assert result["max_layers"] == 0
    assert result["ctx_tokens"] == 0


def test_addrs_dict_becomes_empty_list():
    """addrs given as a dict becomes []."""
    facts = {
        "node_id": "node123",
        "pubkey": "a" * 64,
        "addrs": {"key": "value"},
    }
    result = registration_payload(facts)
    assert result["addrs"] == []


def test_missing_version_becomes_empty_string():
    """Missing version becomes ""."""
    facts = {
        "node_id": "node123",
        "pubkey": "a" * 64,
    }
    result = registration_payload(facts)
    assert result["version"] == ""