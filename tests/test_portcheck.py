import pytest
from portcheck import port_conflicts


def test_free_port():
    """A free port should return (True, "")"""
    listeners = []
    result = port_conflicts(8080, listeners)
    assert result == (True, "")


def test_same_port_on_0_0_0_0():
    """Port on 0.0.0.0 conflicts with any address"""
    listeners = [
        {"port": 8080, "addr": "0.0.0.0", "pid": 1234, "name": "myserver"}
    ]
    result = port_conflicts(8080, listeners)
    assert result[0] is False
    assert "8080" in result[1]
    assert "myserver" in result[1]
    assert "1234" in result[1]


def test_same_port_on_specific_address():
    """Port on specific address conflicts since want binds all interfaces"""
    listeners = [
        {"port": 8080, "addr": "127.0.0.1", "pid": 5678, "name": "another-server"}
    ]
    result = port_conflicts(8080, listeners)
    assert result[0] is False
    assert "8080" in result[1]
    assert "another-server" in result[1]
    assert "5678" in result[1]


def test_different_port_not_conflicting():
    """Different port should not conflict"""
    listeners = [
        {"port": 9000, "addr": "127.0.0.1", "pid": 1111, "name": "other-server"}
    ]
    result = port_conflicts(8080, listeners)
    assert result == (True, "")


def test_process_named_in_why():
    """Process name and pid should be in the reason string"""
    listeners = [
        {"port": 8080, "addr": "0.0.0.0", "pid": 4242, "name": "llama-server"}
    ]
    result = port_conflicts(8080, listeners)
    assert result[0] is False
    assert "llama-server" in result[1]
    assert "4242" in result[1]


def test_port_zero():
    """Port 0 is out of range"""
    result = port_conflicts(0, [])
    assert result[0] is False
    assert "0" in result[1]


def test_port_80():
    """Port 80 is privileged"""
    result = port_conflicts(80, [])
    assert result[0] is False
    assert "80" in result[1]
    assert "elevated privileges" in result[1] or "privileged" in result[1]


def test_port_65536():
    """Port 65536 is out of range"""
    result = port_conflicts(65536, [])
    assert result[0] is False
    assert "65536" in result[1]


def test_port_negative():
    """Negative port is out of range"""
    result = port_conflicts(-1, [])
    assert result[0] is False
    assert "-1" in result[1]


def test_malformed_listener_entries_skipped():
    """Malformed entries should be skipped without raising"""
    listeners = [
        "not a dict",
        {"port": "not an int", "addr": "127.0.0.1"},
        {"port": 8080, "addr": 123},
        {"port": 8080},  # missing addr
        {"addr": "127.0.0.1"},  # missing port
        {"port": 9000, "addr": "127.0.0.1", "pid": 1111, "name": "other-server"}
    ]
    result = port_conflicts(8080, listeners)
    assert result == (True, "")


def test_empty_list():
    """Empty listeners list should return free port"""
    result = port_conflicts(8080, [])
    assert result == (True, "")


def test_never_raising():
    """Function should never raise on malformed input"""
    malformed_inputs = [
        None,
        "string",
        123,
        [None, "string", 123],
        [{"port": "not int", "addr": "127.0.0.1"}],
    ]
    for inp in malformed_inputs:
        try:
            port_conflicts(8080, inp)
        except Exception:
            pytest.fail(f"port_conflicts raised on input: {inp}")