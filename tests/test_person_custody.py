"""The person-custody seam is the only path from Nakshatra to person authority."""
from __future__ import annotations

import ast
import base64
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


def test_invite_file_override_cannot_bypass_broker(tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    custody = tmp_path / ".nakshatra" / "net" / "person-custody"
    custody.write_text("broker\n")
    override, _pub = signer_mod.new_key()
    override_path = tmp_path / "invite.key"
    signer_mod.save_key(override_path, override)
    with pytest.raises(SystemExit, match="person custody is broker; --person-key is not allowed"):
        person_custody.PersonKey("invite", "Create explicit invite", key_path=override_path)
    assert person_custody.custody_mode() == "broker"


def test_person_key_preflights_broker_without_calling_signing_transport(tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    custody = tmp_path / ".nakshatra" / "net" / "person-custody"
    custody.write_text("broker\n")
    monkeypatch.setattr(person_custody, "_broker_sign", lambda *_a: pytest.fail("sign transport called"))

    with pytest.raises(SystemExit) as refused:
        person_custody.PersonKey("invite", "Create contact invite")

    assert str(refused.value) == BROKER_ERROR


def test_custody_path_honours_nak_net_dir_and_signing_home_is_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    net = tmp_path / "exact-net"
    net.mkdir()
    (net / "person-custody").write_text("broker\n")
    monkeypatch.setenv("NAK_NET_DIR", str(net))

    with pytest.raises(SystemExit, match="phone-approval service"):
        person_custody.broker_preflight("invite", "anything")

    other = tmp_path / "other-home"
    with person_custody._signing_home(other):
        assert person_custody.custody_mode() == "file"


def test_plain_drops_controls_bidi_collapses_and_truncates():
    dirty = "  a\n\tb\x00\x85\u202ec\u2066   d  "
    assert person_custody.plain(dirty, 100) == "abc d"
    assert person_custody.plain("abcdef", 5) == "abcd…"
    assert person_custody.plain("abcdef", 1) == "…"


def test_summary_content_never_refuses_and_broker_stub_never_gets_empty_statement(
        tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    summary = "\n\x00\u202e"
    assert person_custody.person_sign("invite", b"statement", summary=summary)
    with pytest.raises(SystemExit, match="unknown person-signing kind"):
        person_custody.broker_preflight("made-up", summary)

    custody = tmp_path / ".nakshatra" / "net" / "person-custody"
    custody.write_text("broker\n")
    seen = []

    def broker(kind, statement, text):
        seen.append((kind, statement, text))
        raise SystemExit(BROKER_ERROR)

    monkeypatch.setattr(person_custody, "_broker_sign", broker)
    with pytest.raises(SystemExit, match="statement must not be empty"):
        person_custody.person_sign("invite", b"", summary=summary)
    assert seen == []


def test_person_key_loads_and_matches_once_and_approval_shares_one_load(tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    real_load = signer_mod.load_key
    loads = []

    def counted(path):
        loads.append(Path(path))
        return real_load(path)

    monkeypatch.setattr(signer_mod, "load_key", counted)
    key = person_custody.PersonKey("contact-accept", "one")
    key.sign(b"first")
    key.sign(b"second")
    assert len(loads) == 1

    loads.clear()
    with person_custody.approval("address-control", "one approval"):
        first = person_custody.PersonKey("address-control", "one approval")
        second = person_custody.PersonKey("address-control", "one approval")
        first.sign(b"address")
        second.sign(b"register")
    assert len(loads) == 1


def test_join_setup_broker_preflight_happens_before_first_identity_write(tmp_path, monkeypatch):
    custody = tmp_path / ".nakshatra" / "net" / "person-custody"
    custody.parent.mkdir(parents=True)
    custody.write_text("broker\n")
    before = _snapshot(tmp_path)
    monkeypatch.setattr(person_custody, "_broker_sign", lambda *_a: pytest.fail("sign transport called"))

    with pytest.raises(SystemExit) as refused:
        join_setup.ensure_identity(tmp_path)

    assert str(refused.value) == BROKER_ERROR
    assert _snapshot(tmp_path) == before


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


@pytest.mark.parametrize(
    ("argv", "read_ops"),
    [
        (["address", "on", "--name", "Never written"], []),
        (["address", "off"], ["address_status"]),
        (["task", "decide", "abcd", "accept"], []),
        (["invite", "--note", "secret"], []),
        (["revoke", "0123456789abcdef"], []),
        (["accept", "484f48ab"], ["requests"]),
    ],
)
def test_broker_cli_person_acts_refuse_before_prompt_write_state_call_or_code(
        argv, read_ops, tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch)
    custody = tmp_path / ".nakshatra" / "net" / "person-custody"
    custody.write_text("broker\n")
    before = _snapshot(tmp_path)
    calls = []

    class Terminal:
        def isatty(self):
            return True

    def no_prompt(_prompt):
        pytest.fail("broker preflight must happen before input()")

    def fake_call(_sock, req):
        calls.append(req)
        if req["op"] == "address_status":
            return {"enabled": True, "aid": "ab" * 16}
        if req["op"] == "tasks":
            return {"tasks": [{"task_hash": "abcd" + "1" * 60}]}
        if req["op"] == "requests":
            return {"requests": [{"id": "484f48ab", "person": "48" * 32,
                                   "nickname": "stranger", "requires_person": True}]}
        raise AssertionError(f"state-changing daemon call reached: {req}")

    monkeypatch.setattr(nak.sys, "stdin", Terminal())
    monkeypatch.setattr("builtins.input", no_prompt)
    monkeypatch.setattr(nak, "call", fake_call)

    with pytest.raises(SystemExit) as refused:
        nak.main(argv)

    assert str(refused.value) == BROKER_ERROR
    assert [req["op"] for req in calls] == read_ops
    assert _snapshot(tmp_path) == before
    captured = capsys.readouterr()
    assert "nki1." not in captured.out + captured.err
    assert "nka1." not in captured.out + captured.err


def test_broker_invite_person_key_is_refused_before_any_escape_or_daemon_call(
        tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch)
    custody = tmp_path / ".nakshatra" / "net" / "person-custody"
    custody.write_text("broker\n")
    override, _pub = signer_mod.new_key()
    override_path = tmp_path / "override.key"
    signer_mod.save_key(override_path, override)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(nak, "call", lambda *_a: pytest.fail("daemon call reached"))

    with pytest.raises(SystemExit) as refused:
        nak.main(["invite", "--person-key", str(override_path)])

    assert str(refused.value) == "person custody is broker; --person-key is not allowed"
    assert _snapshot(tmp_path) == before
    assert "nki1." not in "".join(capsys.readouterr())


def test_broker_grant_renew_refuses_before_input(tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    custody = tmp_path / ".nakshatra" / "net" / "person-custody"
    custody.write_text("broker\n")
    before = _snapshot(tmp_path)

    class Terminal:
        def isatty(self):
            return True

    monkeypatch.setattr(nak.sys, "stdin", Terminal())
    monkeypatch.setattr("builtins.input", lambda _p: pytest.fail("input called"))
    with pytest.raises(SystemExit) as refused:
        nak.main(["grant", "renew"])
    assert str(refused.value) == BROKER_ERROR
    assert _snapshot(tmp_path) == before


def test_file_invite_default_raw_key_needs_no_person_pub_and_preserves_old_bytes(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NAK_NET_DIR", raising=False)
    person, _pub = signer_mod.new_key()
    key_path = tmp_path / ".sthambha" / "person-TEST.key"
    signer_mod.save_key(key_path, person)
    calls = []
    monkeypatch.setattr(nak.time, "time", lambda: 1234)
    monkeypatch.setattr(nak, "_release_for_invite", lambda *_a: (None, ""))

    def fake_call(_sock, req):
        calls.append(req)
        if req["op"] == "status":
            return {"node": "ab" * 32, "relay": "relay.example:1"}
        if req["op"] == "register_invite":
            return {}
        raise AssertionError(req)

    monkeypatch.setattr(nak, "call", fake_call)
    assert nak.main(["invite", "--code-only", "--note", "a\nb"]) == 0
    code = calls[-1]["code"]
    decoded = __import__("joincode").decode_invite(code, now=1234)
    assert decoded["inviter"] == person.public_key().public_bytes_raw().hex()
    assert decoded["note"] == "a\nb"
    signature = base64.b64decode(decoded["sig"]["value"])
    unsigned = dict(decoded)
    unsigned.pop("sig")
    assert signature == person.sign(__import__("joincode")._canon(unsigned))
    assert not (tmp_path / ".sthambha" / "signer" / "person.pub").exists()
    assert code in capsys.readouterr().out


def test_accept_dirty_stranger_nickname_has_clean_provenance_summary(
        tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch)
    request_id = "484f48ab"
    nickname = "  Mo\nm\u202e  "
    captured = []
    real_key = signer_mod.load_key(tmp_path / ".sthambha" / "person-TEST.key")[0]

    class Terminal:
        def isatty(self):
            return True

    class CapturingKey:
        def __init__(self, kind, summary, **_kwargs):
            captured.append((kind, summary))

        def sign(self, statement):
            return real_key.sign(statement)

    def fake_call(_sock, req):
        if req["op"] == "requests":
            return {"requests": [{"id": request_id, "person": "48" * 32,
                                   "nickname": nickname, "requires_person": True}]}
        if req["op"] == "accept":
            return {"person": "48" * 32, "petname": nickname}
        raise AssertionError(req)

    monkeypatch.setattr(nak.sys, "stdin", Terminal())
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    monkeypatch.setattr(nak, "call", fake_call)
    monkeypatch.setattr(nak, "PersonKey", CapturingKey)
    assert nak.main(["accept", request_id]) == 0
    assert captured[0][0] == "contact-accept"
    assert captured[0][1] == 'Accept stranger 4848484848484848… (calls themselves "Mom")'
    assert "\n" not in captured[0][1] and "\u202e" not in captured[0][1]
    assert request_id not in capsys.readouterr().err


def test_person_test_key_literal_is_confined_to_custody_and_tiny_help_allowlist():
    network = Path(__file__).resolve().parents[1] / "scripts" / "network"
    allowed = {
        ("join_setup.py", "2. person key      ~/.sthambha/person-TEST.key           TEST custody: a file, not a vault"),
        ("join_setup.py", "custody_note = \"your person key is a TEST file at ~/.sthambha/person-TEST.key: back it up, and never share it\""),
        ("nak.py", "nak invite --person-key ~/.sthambha/person-TEST.key [--ttl 86400] [--note \"for Rajesh\"]"),
        ("nak.py", 'help="explicit file key in file custody only (default: ~/.sthambha/person-TEST.key)")'),
    }
    found = set()
    for path in network.rglob("*.py"):
        if path.name == "person_custody.py":
            continue
        for line in path.read_text().splitlines():
            if "person-TEST.key" in line:
                found.add((path.relative_to(network).as_posix(), line.strip()))
    assert found == allowed


def test_person_authority_call_sites_are_explicitly_confined():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    found = set()
    for path in scripts.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"issue_grant", "encode_invite", "encode_address"}):
                operation = node.func.attr
            elif isinstance(node.func, ast.Name) and node.func.id == "make_record":
                operation = node.func.id
            else:
                continue
            first = node.args[0]
            if isinstance(first, ast.Name):
                origin = first.id
            elif (isinstance(first, ast.Attribute) and first.attr == "sign"
                  and isinstance(first.value, ast.Name)):
                origin = f"{first.value.id}.sign"
            elif (isinstance(first, ast.Call) and isinstance(first.func, ast.Name)
                  and first.func.id == "PersonKey"):
                origin = "PersonKey(...)"
            else:
                origin = ast.dump(first, include_attributes=False)
            found.add((path.relative_to(scripts).as_posix(), operation, origin))
    assert found == {
        ("network/join_setup.py", "issue_grant", "PersonKey(...)"),
        ("network/nak.py", "issue_grant", "person_priv"),
        ("network/nak.py", "encode_invite", "priv"),
        ("network/nak.py", "encode_address", "person_priv"),
        ("network/nak.py", "make_record", "person_priv.sign"),
        ("network/nak.py", "make_record", "person_key.sign"),
    }

    for path in scripts.rglob("*.py"):
        if path.name == "person_custody.py":
            continue
        source = path.read_text()
        assert "from_private_bytes(bytes.fromhex(person" not in source
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "load_key" and node.args:
                assert not (isinstance(node.args[0], ast.Name)
                            and node.args[0].id in {"person_path", "person_key"}), path
