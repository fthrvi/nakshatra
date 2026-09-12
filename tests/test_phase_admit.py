import pytest
from join.phase_admit import admit


def test_clean_admit():
    facts = {
        "join": {
            "coordinator": "coord123",
            "token": "secrettoken",
            "relay": "relay.example.com",
            "expires_at": 0
        },
        "now": 1000,
        "node_version": "1.2.3",
        "coordinator_version": "1.4.5",
        "package_url": "https://example.com/slice.tar.gz"
    }
    ok, problems, updates = admit(facts)
    assert ok is True
    assert problems == []
    assert updates == {"coordinator": "coord123", "admitted": True}
    assert "token" not in updates


def test_missing_join():
    facts = {
        "now": 1000,
        "node_version": "1.2.3",
        "coordinator_version": "1.4.5",
        "package_url": "https://example.com/slice.tar.gz"
    }
    ok, problems, updates = admit(facts)
    assert ok is False
    assert "missing join" in problems


def test_missing_token():
    facts = {
        "join": {
            "coordinator": "coord123",
            "relay": "relay.example.com",
            "expires_at": 0
        },
        "now": 1000,
        "node_version": "1.2.3",
        "coordinator_version": "1.4.5",
        "package_url": "https://example.com/slice.tar.gz"
    }
    ok, problems, updates = admit(facts)
    assert ok is False
    assert "missing token" in problems
    # Token should never appear in problem string
    for p in problems:
        assert "secrettoken" not in p


def test_expired_code():
    facts = {
        "join": {
            "coordinator": "coord123",
            "token": "secrettoken",
            "relay": "relay.example.com",
            "expires_at": 1000
        },
        "now": 1000,
        "node_version": "1.2.3",
        "coordinator_version": "1.4.5",
        "package_url": "https://example.com/slice.tar.gz"
    }
    ok, problems, updates = admit(facts)
    assert ok is False
    assert "join code expired" in problems


def test_code_expiring_one_second_in_future():
    facts = {
        "join": {
            "coordinator": "coord123",
            "token": "secrettoken",
            "relay": "relay.example.com",
            "expires_at": 1001
        },
        "now": 1000,
        "node_version": "1.2.3",
        "coordinator_version": "1.4.5",
        "package_url": "https://example.com/slice.tar.gz"
    }
    ok, problems, updates = admit(facts)
    assert ok is True
    assert problems == []
    assert updates == {"coordinator": "coord123", "admitted": True}


def test_major_mismatch():
    facts = {
        "join": {
            "coordinator": "coord123",
            "token": "secrettoken",
            "relay": "relay.example.com",
            "expires_at": 0
        },
        "now": 1000,
        "node_version": "2.2.3",
        "coordinator_version": "1.4.5",
        "package_url": "https://example.com/slice.tar.gz"
    }
    ok, problems, updates = admit(facts)
    assert ok is False
    assert "major version mismatch" in problems


def test_minor_mismatch_admitted():
    facts = {
        "join": {
            "coordinator": "coord123",
            "token": "secrettoken",
            "relay": "relay.example.com",
            "expires_at": 0
        },
        "now": 1000,
        "node_version": "1.2.3",
        "coordinator_version": "1.4.5",
        "package_url": "https://example.com/slice.tar.gz"
    }
    ok, problems, updates = admit(facts)
    assert ok is True
    assert problems == []


def test_http_rejected():
    facts = {
        "join": {
            "coordinator": "coord123",
            "token": "secrettoken",
            "relay": "relay.example.com",
            "expires_at": 0
        },
        "now": 1000,
        "node_version": "1.2.3",
        "coordinator_version": "1.4.5",
        "package_url": "http://example.com/slice.tar.gz"
    }
    ok, problems, updates = admit(facts)
    assert ok is False
    assert "package_url must use https" in problems


def test_private_ip_rejected():
    private_ips = [
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "127.0.0.1",
        "169.254.1.1"
    ]
    for ip in private_ips:
        facts = {
            "join": {
                "coordinator": "coord123",
                "token": "secrettoken",
                "relay": "relay.example.com",
                "expires_at": 0
            },
            "now": 1000,
            "node_version": "1.2.3",
            "coordinator_version": "1.4.5",
            "package_url": f"https://{ip}/slice.tar.gz"
        }
        ok, problems, updates = admit(facts)
        assert ok is False
        assert "package_url host is a private or reserved IP" in problems


def test_ipv6_private_rejected():
    ipv6_private = [
        "::1",  # loopback
        "fc00::1",  # unique local
        "fd00::1",  # unique local
        "fe80::1"  # link-local
    ]
    for ip in ipv6_private:
        facts = {
            "join": {
                "coordinator": "coord123",
                "token": "secrettoken",
                "relay": "relay.example.com",
                "expires_at": 0
            },
            "now": 1000,
            "node_version": "1.2.3",
            "coordinator_version": "1.4.5",
            "package_url": f"https://[{ip}]/slice.tar.gz"
        }
        ok, problems, updates = admit(facts)
        assert ok is False
        assert "package_url host is a private or reserved IP" in problems


def test_public_ip_accepted():
    facts = {
        "join": {
            "coordinator": "coord123",
            "token": "secrettoken",
            "relay": "relay.example.com",
            "expires_at": 0
        },
        "now": 1000,
        "node_version": "1.2.3",
        "coordinator_version": "1.4.5",
        "package_url": "https://8.8.8.8/slice.tar.gz"
    }
    ok, problems, updates = admit(facts)
    assert ok is True
    assert problems == []
    assert updates == {"coordinator": "coord123", "admitted": True}


def test_token_never_in_problem_string():
    facts = {
        "join": {
            "coordinator": "coord123",
            "token": "supersecrettoken123",
            "relay": "relay.example.com",
            "expires_at": 1000
        },
        "now": 1000,
        "node_version": "1.2.3",
        "coordinator_version": "1.4.5",
        "package_url": "https://example.com/slice.tar.gz"
    }
    ok, problems, updates = admit(facts)
    assert ok is False
    for p in problems:
        assert "supersecrettoken123" not in p


def test_updates_never_contains_token():
    facts = {
        "join": {
            "coordinator": "coord123",
            "token": "supersecrettoken123",
            "relay": "relay.example.com",
            "expires_at": 0
        },
        "now": 1000,
        "node_version": "1.2.3",
        "coordinator_version": "1.4.5",
        "package_url": "https://example.com/slice.tar.gz"
    }
    ok, problems, updates = admit(facts)
    assert ok is True
    assert "token" not in updates
    assert updates == {"coordinator": "coord123", "admitted": True}