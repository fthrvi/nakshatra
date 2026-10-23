"""The person-custody seam is the only path from Nakshatra to person authority."""
from __future__ import annotations

from pathlib import Path

import pytest

from test_nakd import signer_mod  # noqa: F401

from network import join_setup, nak, person_custody


BROKER_ERROR = (
    "person custody is set to broker, but the phone-approval service is not available yet; "
    "nothing was signed."
)


def _identity(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NAK_NET_DIR", raising=False)
    monkeypatch.delenv("NAK_SIGNER_DIR", raising=False)
    join_setup.ensure_identity(tmp_path)
    return tmp_path / ".sthambha" / "signer"


def _snapshot(root: Path) -> dict[str, tuple[int, bytes | None]]:
    return {
        path.relative_to(root).as_posix(): (
            path.lstat().st_mode,
            path.read_bytes() if path.is_file() else None,
        )
        for path in sorted(root.rglob("*"))
    }


@pytest.mark.parametrize("kind", sorted(person_custody.KINDS))
def test_file_backend_matches_the_old_direct_ed25519_signature_for_every_kind(
        kind, tmp_path, monkeypatch):
    signer_dir = _identity(tmp_path, monkeypatch)
    old_key, person_pub = signer_mod.load_key(tmp_path / ".sthambha" / "person-TEST.key")
    statement = b"byte-for-byte deterministic person statement"

    assert person_custody.person_sign(kind, statement, summary=f"Test {kind}") == old_key.sign(statement)
    assert person_custody.PersonKey(kind, f"Test {kind}").sign(statement) == old_key.sign(statement)
    assert person_custody.PersonKey(kind, f"Test {kind}").public_key().public_bytes_raw().hex() == person_pub
    assert (signer_dir / "person.pub").read_text().strip() == person_pub
    assert person_custody.custody_label() == "test"


def test_unknown_kind_and_bad_custody_value_are_refused(tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="unknown person-signing kind 'unknown'"):
        person_custody.person_sign("unknown", b"statement", summary="Unknown action")

    custody = tmp_path / ".nakshatra" / "net" / "person-custody"
    custody.write_text("cloud\n")
    with pytest.raises(SystemExit) as refused:
        person_custody.custody_mode()
    assert str(custody) in str(refused.value)
    assert "expected one word: file or broker" in str(refused.value)


@pytest.mark.parametrize("kind", sorted(person_custody.KINDS))
def test_broker_stub_refuses_every_person_act(kind, tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    custody = tmp_path / ".nakshatra" / "net" / "person-custody"
    custody.write_text("broker\n")
    before = _snapshot(tmp_path)

    with pytest.raises(SystemExit) as refused:
        person_custody.person_sign(kind, b"statement", summary=f"Test {kind}")

    assert str(refused.value) == BROKER_ERROR
    assert _snapshot(tmp_path) == before


def test_invite_file_override_bypasses_broker_without_changing_its_mode(tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    custody = tmp_path / ".nakshatra" / "net" / "person-custody"
    custody.write_text("broker\n")
    override, _pub = signer_mod.new_key()
    override_path = tmp_path / "invite.key"
    signer_mod.save_key(override_path, override)
    statement = b"explicit invite"

    key = person_custody.PersonKey(
        "invite", "Create explicit invite", key_path=override_path)

    assert key.sign(statement) == override.sign(statement)
    assert key.public_key().public_bytes_raw() == override.public_key().public_bytes_raw()
    assert person_custody.custody_mode() == "broker"


def test_default_person_key_rechecks_custody_mode_at_the_signature_boundary(tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    key = person_custody.PersonKey("invite", "Create contact invite")
    custody = tmp_path / ".nakshatra" / "net" / "person-custody"
    custody.write_text("broker\n")

    with pytest.raises(SystemExit) as refused:
        key.sign(b"statement")

    assert str(refused.value) == BROKER_ERROR


def test_broker_mode_refuses_real_accept_without_recording_it(tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    custody = tmp_path / ".nakshatra" / "net" / "person-custody"
    custody.write_text("broker\n")
    before = _snapshot(tmp_path)
    calls = []

    class Terminal:
        def isatty(self):
            return True

    def fake_call(_sock, req):
        calls.append(req)
        if req["op"] == "requests":
            return {"requests": [{
                "id": "484f48ab", "person": "48" * 32, "nickname": "strangerbox",
                "requires_person": True,
            }]}
        raise AssertionError("accept must not reach the daemon in broker-stub mode")

    monkeypatch.setattr(nak.sys, "stdin", Terminal())
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    monkeypatch.setattr(nak, "call", fake_call)

    with pytest.raises(SystemExit) as refused:
        nak.main(["accept", "484f48ab", "--as", "strangerbox"])

    assert str(refused.value) == BROKER_ERROR
    assert [call["op"] for call in calls] == ["requests"]
    assert _snapshot(tmp_path) == before


def test_broker_mode_refuses_real_grant_renew_without_changing_the_grant(
        tmp_path, monkeypatch):
    signer_dir = _identity(tmp_path, monkeypatch)
    custody = tmp_path / ".nakshatra" / "net" / "person-custody"
    custody.write_text("broker\n")
    grant = signer_dir / "agents" / "agent.grant.json"
    before_grant = grant.read_bytes()
    before = _snapshot(tmp_path)

    with pytest.raises(SystemExit) as refused:
        nak.main(["grant", "renew", "--yes"])

    assert str(refused.value) == BROKER_ERROR
    assert grant.read_bytes() == before_grant
    assert _snapshot(tmp_path) == before
    assert person_custody.custody_label() == "broker"


def test_person_test_key_literal_is_confined_to_custody_and_tiny_help_allowlist():
    network = Path(__file__).resolve().parents[1] / "scripts" / "network"
    allowed = {
        ("join_setup.py", "2. person key      ~/.sthambha/person-TEST.key           TEST custody: a file, not a vault"),
        ("join_setup.py", "custody_note = \"your person key is a TEST file at ~/.sthambha/person-TEST.key: back it up, and never share it\""),
        ("nak.py", "nak invite --person-key ~/.sthambha/person-TEST.key [--ttl 86400] [--note \"for Rajesh\"]"),
    }
    found = set()
    for path in network.rglob("*.py"):
        if path.name == "person_custody.py":
            continue
        for line in path.read_text().splitlines():
            if "person-TEST.key" in line:
                found.add((path.relative_to(network).as_posix(), line.strip()))
    assert found == allowed
