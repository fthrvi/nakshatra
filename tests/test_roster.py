import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import json
from pathlib import Path

import pytest

from roster import load_roster


def test_clean_roster(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": "node1", "pubkey": "a" * 64, "status": "active"},
        {"node_id": "node2", "pubkey": "b" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {"node1": "a" * 64, "node2": "b" * 64}


def test_revoked_record_excluded(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": "node1", "pubkey": "a" * 64, "status": "revoked"},
        {"node_id": "node2", "pubkey": "b" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {"node2": "b" * 64}


def test_pending_record_excluded(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": "node1", "pubkey": "a" * 64, "status": "pending"},
        {"node_id": "node2", "pubkey": "b" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {"node2": "b" * 64}


def test_suspended_record_excluded(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": "node1", "pubkey": "a" * 64, "status": "suspended"},
        {"node_id": "node2", "pubkey": "b" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {"node2": "b" * 64}


def test_missing_status_excluded(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": "node1", "pubkey": "a" * 64},  # missing status
        {"node_id": "node2", "pubkey": "b" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {"node2": "b" * 64}


def test_short_pubkey_excluded(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": "node1", "pubkey": "a" * 63, "status": "active"},
        {"node_id": "node2", "pubkey": "b" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {"node2": "b" * 64}


def test_long_pubkey_excluded(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": "node1", "pubkey": "a" * 65, "status": "active"},
        {"node_id": "node2", "pubkey": "b" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {"node2": "b" * 64}


def test_uppercase_pubkey_excluded(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": "node1", "pubkey": "A" * 64, "status": "active"},
        {"node_id": "node2", "pubkey": "b" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {"node2": "b" * 64}


def test_non_hex_pubkey_excluded(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": "node1", "pubkey": "g" * 64, "status": "active"},  # 'g' is not hex
        {"node_id": "node2", "pubkey": "b" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {"node2": "b" * 64}


def test_non_string_pubkey_excluded(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": "node1", "pubkey": 12345, "status": "active"},
        {"node_id": "node2", "pubkey": "b" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {"node2": "b" * 64}


def test_missing_pubkey_excluded(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": "node1", "status": "active"},  # missing pubkey
        {"node_id": "node2", "pubkey": "b" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {"node2": "b" * 64}


def test_duplicate_node_id_same_key_collapses(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": "node1", "pubkey": "a" * 64, "status": "active"},
        {"node_id": "node1", "pubkey": "a" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {"node1": "a" * 64}


def test_duplicate_node_id_different_key_raises(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": "node1", "pubkey": "a" * 64, "status": "active"},
        {"node_id": "node1", "pubkey": "b" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    with pytest.raises(ValueError, match="Duplicate node_id"):
        load_roster(str(roster_file))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_roster(str(tmp_path / "nonexistent.json"))


def test_malformed_json_raises(tmp_path: Path) -> None:
    roster_file = tmp_path / "roster.json"
    roster_file.write_text("{invalid json")

    with pytest.raises(ValueError, match="Malformed JSON"):
        load_roster(str(roster_file))


def test_empty_roster_with_only_revoked_returns_empty_dict(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": "node1", "pubkey": "a" * 64, "status": "revoked"},
        {"node_id": "node2", "pubkey": "b" * 64, "status": "pending"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {}


def test_non_list_json_raises(tmp_path: Path) -> None:
    roster_data = {"node1": "a" * 64}  # not a list
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    with pytest.raises(ValueError, match="must contain a JSON array"):
        load_roster(str(roster_file))


def test_non_dict_record_ignored(tmp_path: Path) -> None:
    roster_data = [
        "not a dict",
        {"node_id": "node1", "pubkey": "a" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {"node1": "a" * 64}


def test_non_string_node_id_excluded(tmp_path: Path) -> None:
    roster_data = [
        {"node_id": 123, "pubkey": "a" * 64, "status": "active"},
        {"node_id": "node2", "pubkey": "b" * 64, "status": "active"},
    ]
    roster_file = tmp_path / "roster.json"
    roster_file.write_text(json.dumps(roster_data))

    result = load_roster(str(roster_file))
    assert result == {"node2": "b" * 64}