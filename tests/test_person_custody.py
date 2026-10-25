"""The person-custody seam is the only path from Nakshatra to person authority."""
from __future__ import annotations

import ast
import base64
from contextlib import contextmanager, nullcontext
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import socket
import threading
import time
import urllib.error
import urllib.parse

import pytest

from test_nakd import signer_mod  # noqa: F401

import joincode
from network import join_setup, nak, nakd, person_custody, revocation, suspension, verdict


TOKEN = "test-broker-token"


def _canon(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _identity(tmp_path: Path, monkeypatch) -> tuple[Path, object]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NAK_NET_DIR", raising=False)
    monkeypatch.delenv("NAK_SIGNER_DIR", raising=False)
    join_setup.ensure_identity(tmp_path)
    private, _pub = signer_mod.load_key(tmp_path / ".sthambha" / "person-TEST.key")
    return tmp_path / ".sthambha" / "signer", private


def _snapshot(root: Path) -> dict[str, tuple[int, bytes | None]]:
    return {
        path.relative_to(root).as_posix(): (
            path.lstat().st_mode,
            path.read_bytes() if path.is_file() else None,
        )
        for path in sorted(root.rglob("*"))
    }


def _independent_sign(private, items):
    """Protocol implementation used by the fake broker, separate from code under test."""
    finals = []
    signatures = []
    for item in items:
        raw = base64.b64decode(item["b64"], validate=True)
        if "record_ref" in item:
            ref = item["record_ref"]
            record = json.loads(finals[ref].decode())
            record["sig"] = signatures[ref].hex()
            signed_record = _canon(record)
            proof = json.loads(raw.decode())
            assert proof["payload"] == {"record_sha256": "0" * 64}
            proof["payload"] = {"record_sha256": hashlib.sha256(signed_record).hexdigest()}
            raw = _canon(proof)
        finals.append(raw)
        signatures.append(private.sign(raw))
    return finals, signatures


class FakeBroker:
    def __init__(self, private, *, status="approved", corrupt="",
                 principal="nakshatra-person", pending_polls=0):
        self.private = private
        self.status = status
        self.corrupt = corrupt
        self.principal = principal
        self.pending_polls = pending_polls
        self.enqueues = []
        self.sign_calls = 0
        self.principal_calls = 0
        self.result_calls = 0
        self._requests = {}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def _reply(self, status, value):
                raw = json.dumps(value, separators=(",", ":")).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _authorized(self):
                if self.headers.get("Authorization") == f"Bearer {TOKEN}":
                    return True
                self._reply(401, {"error": "wrong bearer"})
                return False

            def _body(self):
                return json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))

            def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
                if not self._authorized():
                    return
                parsed = urllib.parse.urlsplit(self.path)
                if parsed.path == "/principal/v1":
                    owner.principal_calls += 1
                    self._reply(200, {"principal": owner.principal})
                    return
                if parsed.path == "/result/v1":
                    owner.result_calls += 1
                    request_id = urllib.parse.parse_qs(parsed.query).get("request_id", [""])[0]
                    state = owner._requests.get(request_id)
                    if state and owner.pending_polls:
                        owner.pending_polls -= 1
                        self._reply(200, {"status": "pending"})
                    else:
                        self._reply(200, {"status": owner.status if state else "unknown"})
                    return
                self._reply(404, {"error": "unknown route"})

            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
                if not self._authorized():
                    return
                body = self._body()
                if self.path == "/enqueue/v1":
                    exact = body.get("exact_content")
                    if (body.get("what") != "nakshatra.person.v1" or body.get("tier") != "high"
                            or not isinstance(exact, dict) or exact.get("v") != 1
                            or not 1 <= len(exact.get("items") or []) <= 8
                            or not isinstance(body.get("idempotency_key"), str)
                            or body.get("ttl") != person_custody._approval_wait_s(exact.get("kind"))):
                        self._reply(400, {"error": "invalid enqueue"})
                        return
                    request_id = f"request-{len(owner.enqueues) + 1}"
                    owner.enqueues.append(body)
                    owner._requests[request_id] = exact
                    self._reply(200, {"request_id": request_id})
                    return
                if self.path == "/person-sign/v1":
                    owner.sign_calls += 1
                    request_id = body.get("request_id")
                    exact = owner._requests.get(request_id)
                    if exact is None or body.get("exact_content") != exact or owner.status != "approved":
                        self._reply(409, {"error": "not approved"})
                        return
                    finals, signatures = _independent_sign(owner.private, exact["items"])
                    if owner.corrupt == "signature":
                        signatures[0] = b"\0" * 64
                    elif owner.corrupt == "item":
                        finals[0] += b" substituted"
                        signatures[0] = owner.private.sign(finals[0])
                    self._reply(200, {
                        "request_id": request_id,
                        "signatures": [signature.hex() for signature in signatures],
                        "items_b64": [base64.b64encode(item).decode() for item in finals],
                    })
                    return
                self._reply(404, {"error": "unknown route"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@contextmanager
def _broker(tmp_path, monkeypatch, *, status="approved", corrupt="", remove_private=True,
            principal="nakshatra-person", pending_polls=0):
    signer_dir, private = _identity(tmp_path, monkeypatch)
    broker = FakeBroker(
        private, status=status, corrupt=corrupt, principal=principal,
        pending_polls=pending_polls).start()
    try:
        net = tmp_path / ".nakshatra" / "net"
        token = tmp_path / "broker.token"
        token.write_text(TOKEN)
        token.chmod(0o600)
        config = net / "person-broker.json"
        config.write_text(json.dumps({
            "url": broker.url, "token_file": str(token),
        }))
        config.chmod(0o600)
        marker = net / "person-custody"
        marker.write_text("broker\n")
        marker.chmod(0o600)
        if remove_private:
            (tmp_path / ".sthambha" / "person-TEST.key").unlink()
        assert (signer_dir / "person.pub").is_file()
        yield broker, private
    finally:
        broker.close()


@pytest.mark.parametrize("kind", sorted(person_custody.KINDS))
def test_file_backend_matches_direct_ed25519_for_every_kind(kind, tmp_path, monkeypatch):
    _signer, private = _identity(tmp_path, monkeypatch)
    statement = f"deterministic {kind} statement".encode()
    assert person_custody.person_sign(kind, statement, summary=f"Test {kind}") == private.sign(statement)
    key = person_custody.PersonKey(kind, f"Test {kind}")
    assert key.sign(statement) == private.sign(statement)
    assert key.public_key().public_bytes_raw() == private.public_key().public_bytes_raw()


def _statement(kind, private):
    person = private.public_key().public_bytes_raw().hex()
    proof_now = int(time.time())
    if kind == "contact-accept":
        return nakd.accept_statement("484f48ab", "friend", proof_now)
    if kind == "address-control":
        return nakd.address_control_statement("address_off", "12" * 16, proof_now)
    if kind == "task-verdict":
        return verdict.decision_statement("ab" * 32, "accept", "looks good", proof_now)
    if kind in {"grant-issue", "grant-renew", "spend-cap"}:
        return _canon({
            "schema": signer_mod.GRANT_SCHEMA, "subject": "agent",
            "subject_pub": "23" * 32, "caps": ["nak.msg"], "node": "34" * 32,
            "issued_at": 123, "expires_at": 456, "grant_id": "45" * 8,
        })
    if kind == "invite":
        return _canon({
            "v": 1, "inviter": person, "inviter_node": "56" * 32,
            "peers": [{"node": "56" * 32, "addrs": ["relay.example:1"]}],
            "nonce": "67" * 16, "expires_at": 456, "scope": "member", "ik": "78" * 32,
        })
    if kind == "revoke":
        record = revocation.build_unsigned_record(person, 1, ["89" * 8], 123)
        return revocation.revocation_statement(record)
    if kind == "report":
        return nakd.person_action_statement(
            "report-cli", nakd.report_proof_payload("9a" * 32, "abuse", "", {}), proof_now)
    if kind == "suspend":
        return nakd.person_action_statement(
            "operator-trust", {"person": "ab" * 32}, proof_now)
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", sorted(person_custody.KINDS))
def test_broker_signs_each_custody_kind_with_one_enqueue(kind, tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch) as (broker, private):
        statement = _statement(kind, private)
        context = {"verdict_reason": "looks good"} if kind == "task-verdict" else None
        signature = person_custody.person_sign(
            kind, statement, summary=f"Test {kind}", context=context)
        private.public_key().verify(signature, statement)
        assert len(broker.enqueues) == 1
        assert broker.sign_calls == 1
        request = broker.enqueues[0]
        assert request["what"] == "nakshatra.person.v1"
        assert request["tier"] == "high"
        assert request["exact_content"]["kind"] == kind
        expected_ttl = 90 if kind in {
            "address-control", "contact-accept", "report", "suspend", "task-verdict",
        } else 300
        assert request["ttl"] == expected_ttl
        if context:
            assert request["exact_content"]["context"] == context


def test_person_key_broker_adapter_uses_public_file_and_never_private_key(
        tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch) as (broker, private):
        key = person_custody.PersonKey("invite", "Invite one friend")
        statement = b"invite bytes"
        signature = key.sign(statement)
        private.public_key().verify(signature, statement)
        assert key.public_key().public_bytes_raw() == private.public_key().public_bytes_raw()
        assert len(broker.enqueues) == 1


def test_file_batch_template_output_matches_independent_old_sequence(tmp_path, monkeypatch):
    monkeypatch.setattr(person_custody.time, "time", lambda: 123)
    _signer, private = _identity(tmp_path, monkeypatch)
    person = private.public_key().public_bytes_raw().hex()
    unsigned = suspension.build_unsigned_record(
        person, 3, ["71" * 32], {"71" * 32: "abuse"}, 123)
    record_statement = suspension.suspension_statement(unsigned)
    stage = nakd.person_action_statement(
        "stage-suspension", {"record_sha256": "0" * 64}, 123)
    install = nakd.person_action_statement(
        "install-suspension", {"record_sha256": "0" * 64}, 123)
    items = [record_statement, ("ref", 0, stage), ("ref", 0, install)]
    actual = person_custody.person_sign_batch("suspend", items, summary="Suspend one node")
    wire = [
        {"b64": base64.b64encode(record_statement).decode()},
        {"b64": base64.b64encode(stage).decode(), "record_ref": 0},
        {"b64": base64.b64encode(install).decode(), "record_ref": 0},
    ]
    finals, signatures = _independent_sign(private, wire)
    assert actual == list(zip(finals, signatures))


def test_address_split_helpers_preserve_the_old_encoded_bytes(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import ed25519

    private = ed25519.Ed25519PrivateKey.generate()
    invite_key = ed25519.Ed25519PrivateKey.generate()

    class FixedEd25519:
        @staticmethod
        def generate():
            return invite_key

    monkeypatch.setattr(ed25519, "Ed25519PrivateKey", FixedEd25519)
    monkeypatch.setattr(joincode.secrets, "token_hex", lambda _size: "12" * 16)
    release = {"url": "https://release.example/nak", "channel": "stable",
               "pubkey": "23" * 32, "version": "1.2.3"}
    peers = [{"node": "34" * 32, "addrs": ["relay.example:1"]}]
    person = private.public_key().public_bytes_raw().hex()
    unsigned = joincode.build_unsigned_address(
        person, "45" * 32, peers, days=30, now=123,
        from_name="Biswa", release=release)
    expected = joincode.attach_address_signature(
        unsigned, private.sign(joincode._canon(unsigned)))
    actual = joincode.encode_address(
        private, "45" * 32, peers, days=30, now=123,
        from_name="Biswa", release=release)
    assert actual == expected


def test_broker_suspend_chain_is_one_enqueue_and_passes_real_verifiers(tmp_path, monkeypatch):
    monkeypatch.setattr(person_custody.time, "time", lambda: 456)
    with _broker(tmp_path, monkeypatch) as (broker, private):
        person = private.public_key().public_bytes_raw().hex()
        unsigned = suspension.build_unsigned_record(
            person, 9, ["72" * 32], {"72" * 32: "operator reason"}, 456)
        templates = [nakd.person_action_statement(
            kind, {"record_sha256": "0" * 64}, 456)
            for kind in ("stage-suspension", "install-suspension")]
        signed = person_custody.person_sign_batch(
            "suspend",
            [suspension.suspension_statement(unsigned),
             ("ref", 0, templates[0]), ("ref", 0, templates[1])],
            summary="Suspend one node")
        record = suspension.attach_signature(unsigned, signed[0][1])
        suspension.verify_record(record, person)
        for index, kind in enumerate(("stage-suspension", "install-suspension"), start=1):
            expected = nakd.person_action_statement(
                kind, nakd.suspension_record_proof_payload(record), 456)
            assert signed[index][0] == expected
            private.public_key().verify(signed[index][1], expected)
        assert len(broker.enqueues) == 1
        assert broker.sign_calls == 1


@pytest.mark.parametrize("corrupt,match", [
    ("signature", "wrong signature"),
    ("item", "substituted signing item"),
])
def test_broker_wrong_signature_or_substituted_item_is_refused(
        corrupt, match, tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch, corrupt=corrupt) as (broker, _private):
        with pytest.raises(SystemExit, match=match):
            person_custody.person_sign("invite", b"exact invite", summary="Invite")
        assert len(broker.enqueues) == 1 and broker.sign_calls == 1


def test_declined_act_signs_nothing_and_makes_no_state_change(tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch, status="declined") as (broker, _private):
        before = _snapshot(tmp_path)
        with pytest.raises(SystemExit, match="not approved,?;? nothing was signed"):
            person_custody.person_sign("contact-accept", b"accept", summary="Accept")
        assert _snapshot(tmp_path) == before
        assert len(broker.enqueues) == 1
        assert broker.sign_calls == 0


def test_expired_act_is_not_approved_and_signs_nothing(tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch, status="expired") as (broker, _private):
        with pytest.raises(SystemExit, match="not approved"):
            person_custody.person_sign("invite", b"invite", summary="Invite")
        assert len(broker.enqueues) == 1 and broker.sign_calls == 0


def test_late_real_broker_wait_refuses_stale_proof_before_local_send_or_write(
        tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch, pending_polls=1) as (broker, _private):
        statement = nakd.accept_statement("484f48ab", "friend", 1000)
        monkeypatch.setattr(person_custody, "_BROKER_POLL_S", 0)
        monkeypatch.setattr(person_custody.time, "time", lambda: 1121)
        with pytest.raises(SystemExit, match="120s nakd proof window.*nothing was sent or written"):
            person_custody.person_sign(
                "contact-accept", statement, summary="Accept after delayed approval")
        assert broker.result_calls == 2  # pending first: the real broker polling loop waited.
        assert broker.sign_calls == 1


def test_preflight_rejects_a_token_registered_to_the_wrong_principal(tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch, principal="somebody-else") as (broker, _private):
        with pytest.raises(SystemExit, match="wrong principal.*nakshatra-person"):
            person_custody.broker_preflight("invite", "Invite")
        assert broker.enqueues == [] and broker.sign_calls == 0


def test_broker_act_cannot_be_switched_to_file_by_marker_change(tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch) as (broker, private):
        act = person_custody.broker_preflight("invite", "Invite")
        key = person_custody.PersonKey("invite", "Invite", act=act)
        (tmp_path / ".nakshatra" / "net" / "person-custody").write_text("file\n")
        statement = b"the act remains broker-backed"
        signature = key.sign(statement)
        private.public_key().verify(signature, statement)
        assert broker.sign_calls == 1


def test_declined_cli_accept_never_calls_state_changing_daemon_op(tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch, status="declined") as (broker, _private):
        calls = []

        class Terminal:
            def isatty(self):
                return True

        def fake_call(_sock, request):
            calls.append(request["op"])
            if request["op"] == "requests":
                return {"requests": [{"id": "484f48ab", "person": "48" * 32,
                                       "nickname": "stranger", "requires_person": True}]}
            raise AssertionError("declined signature must stop before accept")

        monkeypatch.setattr(nak.sys, "stdin", Terminal())
        monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
        monkeypatch.setattr(nak, "call", fake_call)
        with pytest.raises(SystemExit, match="not approved"):
            nak.main(["accept", "484f48ab"])
        assert calls == ["requests"]
        assert len(broker.enqueues) == 1 and broker.sign_calls == 0


def test_broker_address_on_batches_code_and_control_into_one_enqueue(
        tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch) as (broker, private):
        calls = []
        release = {"url": "https://releases.example/nak", "channel": "stable",
                   "pubkey": "12" * 32, "version": "1.2.3"}

        def fake_call(_sock, request):
            calls.append(request)
            if request["op"] == "status":
                return {"node": "ab" * 32, "relay": "relay.example:51820"}
            if request["op"] == "register_address":
                address = joincode.decode_address(request["code"])
                private.public_key().verify(
                    bytes.fromhex(request["signature"]),
                    nakd.address_control_statement(
                        "register_address", address["aid"], request["ts"]))
                return {"aid": address["aid"]}
            raise AssertionError(request)

        monkeypatch.setattr(nak, "call", fake_call)
        monkeypatch.setattr(nak, "_release_for_invite",
                            lambda *_args: (release, "install {CODE}"))
        assert nak.main(["address", "on", "--name", "Biswa"]) == 0
        assert len(broker.enqueues) == 1 and broker.sign_calls == 1
        assert len(broker.enqueues[0]["exact_content"]["items"]) == 2
        assert calls[-1]["op"] == "register_address"


def test_declined_revoke_writes_no_revoked_file_or_pending_record(tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch, status="declined") as (broker, _private):
        class Terminal:
            def isatty(self):
                return True

        monkeypatch.setattr(nak.sys, "stdin", Terminal())
        monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
        monkeypatch.setattr(nak, "call", lambda *_args: pytest.fail("install reached after decline"))
        with pytest.raises(SystemExit, match="not approved"):
            nak.main(["revoke", "0123456789abcdef"])
        assert not (tmp_path / ".sthambha" / "signer" / "revoked.txt").exists()
        assert not (tmp_path / ".nakshatra" / "net" / "pending-revocation.json").exists()
        assert len(broker.enqueues) == 1 and broker.sign_calls == 0


def test_broker_down_preflight_refuses_before_prompt_or_write(tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    net = tmp_path / ".nakshatra" / "net"
    token = tmp_path / "broker.token"
    token.write_text(TOKEN)
    token.chmod(0o600)
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    config = net / "person-broker.json"
    config.write_text(json.dumps({
        "url": f"http://127.0.0.1:{port}", "token_file": str(token),
    }))
    config.chmod(0o600)
    marker = net / "person-custody"
    marker.write_text("broker\n")
    marker.chmod(0o600)
    before = _snapshot(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("prompted before preflight"))
    with pytest.raises(SystemExit, match="unavailable.*nothing was signed"):
        nak.main(["address", "on", "--name", "must not be stored"])
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("mode", ["missing", "bad-json", "bad-token-mode"])
def test_invalid_broker_configuration_is_clear_and_fail_closed(mode, tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    net = tmp_path / ".nakshatra" / "net"
    marker = net / "person-custody"
    marker.write_text("broker\n")
    marker.chmod(0o600)
    token = tmp_path / "token"
    token.write_text(TOKEN)
    token.chmod(0o600)
    config = net / "person-broker.json"
    if mode == "bad-json":
        config.write_text("{")
        config.chmod(0o600)
    elif mode == "bad-token-mode":
        token.chmod(0o644)
        config.write_text(json.dumps({"url": "http://127.0.0.1:9", "token_file": str(token)}))
        config.chmod(0o600)
    with pytest.raises(SystemExit) as refused:
        person_custody.broker_preflight("invite", "Invite")
    message = str(refused.value)
    assert "person broker" in message or "person custody is broker" in message
    assert ("missing" in message if mode == "missing" else "invalid" in message or "0600" in message)


@pytest.mark.parametrize("target,mutation,match", [
    ("marker", "symlink", "regular file"),
    ("marker", "writable", "group/world writable"),
    ("config", "symlink", "regular file"),
    ("config", "writable", "group/world writable"),
    ("token", "symlink", "regular file"),
])
def test_custody_files_are_owned_regular_nofollow_and_private(
        target, mutation, match, tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    net = tmp_path / ".nakshatra" / "net"
    token = tmp_path / "broker.token"
    token.write_text(TOKEN)
    token.chmod(0o600)
    config = net / "person-broker.json"
    config.write_text(json.dumps({"url": "http://127.0.0.1:9", "token_file": str(token)}))
    config.chmod(0o600)
    marker = net / "person-custody"
    marker.write_text("broker\n")
    marker.chmod(0o600)
    path = {"marker": marker, "config": config, "token": token}[target]
    if mutation == "symlink":
        original = path.with_name(path.name + ".real")
        path.rename(original)
        path.symlink_to(original)
    else:
        path.chmod(0o666)
    with pytest.raises(SystemExit, match=match):
        person_custody.broker_preflight("invite", "Invite")


def test_custody_marker_must_be_owned_by_effective_uid(tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    marker = tmp_path / ".nakshatra" / "net" / "person-custody"
    marker.write_text("file\n")
    marker.chmod(0o600)
    monkeypatch.setattr(person_custody.os, "geteuid", lambda: marker.stat().st_uid + 1)
    with pytest.raises(SystemExit, match="owned by the effective user"):
        person_custody.custody_mode()


def test_broker_http_error_is_fixed_bounded_and_never_echoes_body_or_headers(monkeypatch):
    error = urllib.error.HTTPError(
        "http://broker.invalid/enqueue/v1", 418, "header-secret", {"X-Secret": "header-value"},
        io.BytesIO(b'{"error":"safe-code","detail":"body-secret"}'))
    monkeypatch.setattr(person_custody._BROKER_OPENER, "open", lambda *_args, **_kw: (_ for _ in ()).throw(error))
    with pytest.raises(SystemExit) as refused:
        person_custody._broker_json(
            person_custody._BrokerConfig("http://broker.invalid", "token"),
            "POST", "/enqueue/v1", {})
    message = str(refused.value)
    assert message == "person broker refused the request (HTTP 418, code safe-code); nothing was signed"
    assert "body-secret" not in message and "header-secret" not in message and "header-value" not in message


def test_broker_redirects_are_disabled():
    assert person_custody._NoBrokerRedirects().redirect_request(
        None, None, 302, "move", {}, "http://elsewhere.invalid") is None


def test_broker_minimum_ttl_incompatibility_is_loud(monkeypatch):
    monkeypatch.setattr(person_custody, "_BROKER_V1_MIN_TTL_S", 91)
    with pytest.raises(SystemExit, match="incompatible.*skew=120s.*broker_min=91s"):
        person_custody._approval_wait_s("contact-accept")


def test_batch_ref_rules_are_identical_in_file_mode(tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="earlier plain item"):
        person_custody.person_sign_batch(
            "suspend", [("ref", 0, b"bad")], summary="bad reference")
    with pytest.raises(SystemExit, match="1 through 8"):
        person_custody.person_sign_batch("suspend", [], summary="empty")
    with pytest.raises(SystemExit, match="1 through 8"):
        person_custody.person_sign_batch("suspend", [b"x"] * 9, summary="too many")


def test_missing_marker_keeps_file_compatibility_but_imported_key_fails_closed(
        tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    (tmp_path / ".sthambha" / "person-TEST.key").unlink()
    assert person_custody.custody_mode() == "file"
    with pytest.raises(SystemExit, match="cannot load the person key.*decision"):
        person_custody.person_sign("invite", b"invite", summary="Invite")


def test_invite_override_cannot_bypass_broker(tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch) as (_broker_server, _private):
        override, _pub = signer_mod.new_key()
        override_path = tmp_path / "override.key"
        signer_mod.save_key(override_path, override)
        with pytest.raises(SystemExit, match="--person-key is not allowed"):
            person_custody.PersonKey("invite", "Invite", key_path=override_path)


def test_plain_drops_controls_bidi_collapses_and_truncates():
    dirty = "  a\n\tb\x00\x85\u202ec\u2066   d  "
    assert person_custody.plain(dirty, 100) == "abc d"
    assert person_custody.plain("abcdef", 5) == "abcd…"
    assert person_custody.plain("abcdef", 1) == "…"


def test_broker_recovery_commands_never_emit_private_person_key(tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch) as (_broker_server, _private):
        reuse = nak._delegate_reuse_command(
            Path("state"), Path("person.key"), "agent", ["nak.msg"], 365, 30, 7, False)
        join = join_setup._delegate_command(
            Path("state"), Path("person.key"), "agent", ["nak.msg"],
            reuse_key=True, broker=True)
        assert reuse.startswith("nak grant renew") and "--person-key" not in reuse
        assert join.startswith("nak grant renew") and "--person-key" not in join


@pytest.mark.parametrize("allowlisted_entry", [
    ("network/join_setup.py", "_established_identity"),
    ("network/join_setup.py", "ensure_identity"),
])
def test_join_allowlist_entries_never_read_a_private_person_key_in_broker_mode(
        allowlisted_entry, tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch) as (_broker_server, _private):
        assert allowlisted_entry in _PRIVATE_CONSTRUCTOR_ALLOWLIST
        private_path = tmp_path / ".sthambha" / "person-TEST.key"
        private_path.write_text("not a key and deliberately unreadable by the broker path")
        monkeypatch.setattr(
            join_setup, "_raw_file_key",
            lambda *_args: pytest.fail("broker mode loaded the private person key"))
        result = join_setup.ensure_identity(tmp_path)
        assert result["person"] == (
            tmp_path / ".sthambha" / "signer" / "person.pub").read_text().strip()


def test_grant_allowlist_entry_loads_only_agent_key_in_broker_mode(tmp_path, monkeypatch):
    entry = ("network/nak.py", "_grant_renew_confined")
    assert entry in _PRIVATE_CONSTRUCTOR_ALLOWLIST
    with _broker(tmp_path, monkeypatch) as (broker, _private):
        monkeypatch.setattr(
            person_custody, "_load_matching_person_key",
            lambda **_kwargs: pytest.fail("broker renewal loaded the private person key"))
        assert nak.main(["grant", "renew", "--yes", "--no-lease"]) == 0
        assert broker.sign_calls == 1


@pytest.mark.parametrize("mode", ["file", "broker"])
def test_cli_report_resolves_contact_to_node_before_building_signed_statement(
        mode, tmp_path, monkeypatch):
    if mode == "broker":
        context = _broker(tmp_path, monkeypatch)
    else:
        _signer, private = _identity(tmp_path, monkeypatch)
        context = nullcontext((None, private))
    with context as (broker, private):
        node = "9b" * 32
        seen = []

        class Terminal:
            def isatty(self):
                return True

        def fake_call(_sock, request):
            seen.append(request)
            if request["op"] == "resolve_report_subject":
                assert request["subject"] == "friend"
                return {"subject": node}
            if request["op"] == "report_cli":
                assert request["subject"] == node
                statement = nakd.person_action_statement(
                    "report-cli",
                    nakd.report_proof_payload(node, "abuse", "", {}), request["ts"])
                private.public_key().verify(bytes.fromhex(request["signature"]), statement)
                return {"subject": node, "status": "received"}
            raise AssertionError(request)

        monkeypatch.setattr(nak.sys, "stdin", Terminal())
        monkeypatch.setattr(nak, "call", fake_call)
        assert nak.main(["report", "friend", "--reason", "abuse"]) == 0
        assert [request["op"] for request in seen] == [
            "resolve_report_subject", "report_cli"]
        if broker is not None:
            raw = base64.b64decode(
                broker.enqueues[0]["exact_content"]["items"][0]["b64"], validate=True)
            assert json.loads(raw)["payload"]["subject"] == node
        # Deliberate parity exception: file custody now signs the resolved node too; the old
        # typed-petname bytes were not the bytes nakd ultimately resolved and transmitted.


@pytest.mark.parametrize("kind", ["worker", "consumer"])
def test_declined_member_invite_cancels_all_staged_server_state(kind, tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch, status="declined") as (broker, _private):
        monkeypatch.setenv("NAK_ONBOARD_TOKEN", "operator-secret")
        monkeypatch.setattr(nak, "_release_for_invite", lambda _url="": ({}, ""))
        staged = set()
        calls = []

        def fake_call(_sock, request):
            calls.append(request["op"])
            if request["op"] == "status":
                return {"node": "ab" * 32, "relay": "relay.example:51820"}
            raise AssertionError("declined invite must not be registered locally")

        def fake_post(url, body):
            if url.endswith(f"/{kind}-stage"):
                expected = {"token": "operator-secret", "tier": "known"}
                if kind == "consumer":
                    expected.update({"starter": 0, "inviter_pubkey": "ab" * 32})
                assert body == expected
                staged.add("declined-code")
                return {"code": "declined-code", "expires_at": int(time.time()) + 600}
            if url.endswith("/invite-cancel"):
                assert body == {"token": "operator-secret", "code": "declined-code"}
                staged.discard(body["code"])
                return {"cancelled": True}
            raise AssertionError(url)

        monkeypatch.setattr(nak, "call", fake_call)
        with pytest.raises(SystemExit, match="not approved"):
            nak.main(["invite", "--kind", kind, "--tier", "known",
                      "--redeem", "https://onboard.example/base", "--code-only"],
                     http_post=fake_post)
        assert staged == set()
        assert calls == ["status"]
        assert broker.sign_calls == 0


_PRIVATE_CONSTRUCTOR_ALLOWLIST = {
    ("joincode.py", "encode_invite"): "generates the invite's one-use request secret",
    ("joincode.py", "build_unsigned_address"): "generates the address's one-use request secret",
    ("joincode.py", "address_pub"): "derives only the embedded address request public key",
    ("joincode.py", "invite_pub"): "derives only the embedded invite request public key",
    ("sidecar_key.py", "libp2p_ed25519_private"): "converts the node seed for libp2p",
    ("worker.py", "Info"): "uses the worker node identity",
    ("worker.py", "__init__"): "uses the worker node identity",
    ("paid_admission.py", "_receipt"): "signs with a worker payment key supplied to the class",
    ("identity_binding.py", "_ed_sign"): "generic worker-participation signing helper",
    ("identity_binding.py", "pub_of"): "generic worker-participation public-key helper",
    ("nakshatra_auth.py", "generate_keypair"): "creates a node/service keypair",
    ("nakshatra_auth.py", "public_key_hex_from_private"): "derives a node/service public key",
    ("nakshatra_auth.py", "sign_request"): "signs node/service HTTP requests",
    ("network/join_setup.py", "redeem_worker_invite"): "loads the node key for worker redemption",
    ("network/join_setup.py", "redeem_consumer_invite"): "loads the node key for consumer redemption",
    ("network/nakd.py", "__init__"): "derives nakd's node identity",
    ("network/nakd.py", "node_sign_stage"): "signs staging with nakd's node key",
    ("network/nakd.py", "report"): "signs the outbound report with nakd's node key",
    ("discovery/nakshatra_listing.py", "sign"): "signs discovery with its supplied node key",
    ("packaging/nakshatra_package.py", "sign"): "signs packages with a dedicated release key",
    ("transport/identity_handshake.py", "make_proof"): "signs transport proof with a node key",
    ("transport/secure_channel.py", "secure_handshake"): "converts the transport node seed",
    ("transport/connect.py", "_pub_of"): "derives a transport node public key",
    ("test_identity_binding.py", "_key"): "script-local test generates a disposable key",
    ("packaging/test_package_tool.py", "test_certify_signs_a_valid_package"): "test-only package key",
    ("packaging/test_package_tool.py", "test_certify_refuses_invalid_package"): "test-only package key",
    ("network/join_setup.py", "_established_identity"): "legacy person file read is file-mode-only",
    ("network/join_setup.py", "ensure_identity"): "legacy person file creation/read is file-mode-only",
    ("network/nak.py", "_grant_renew_confined"): "loads only the delegated agent key",
}

_PERSON_PATH_ALLOWLIST = {
    ("network/join_setup.py", "<module>"): "operator-facing legacy file-custody documentation",
    ("network/join_setup.py", "main"): "operator-facing legacy backup reminder",
    ("network/nak.py", "<module>"): "operator-facing CLI example for explicit file custody",
    ("network/join_setup.py", "ensure_identity"): "file-mode bootstrap compatibility path",
    ("network/nak.py", "_grant_renew_confined"): "file-mode fallback command construction",
    ("network/nak.py", "main"): "file-mode invite override and help text",
}


def _authority_guard_findings(scripts: Path):
    calls, private_pairs, path_pairs = set(), set(), set()
    for path in scripts.rglob("*.py"):
        relative = path.relative_to(scripts).as_posix()
        if relative == "network/person_custody.py":
            continue
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        parents = {child: parent for parent in ast.walk(tree)
                   for child in ast.iter_child_nodes(parent)}

        def owner(node):
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return node.name
            return "<module>"

        aliases = set()
        for node in ast.walk(tree):
            if (isinstance(node, (ast.ImportFrom,)) and node.module
                    and any(alias.name in {"load_key", "_raw_file_key"} for alias in node.names)):
                aliases.update(alias.asname or alias.name for alias in node.names
                               if alias.name in {"load_key", "_raw_file_key"})
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute) \
                    and node.value.attr in {"load_key", "_raw_file_key", "from_private_bytes"}:
                aliases.update(target.id for target in node.targets if isinstance(target, ast.Name))

        for node in ast.walk(tree):
            fn = owner(node)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and "person-TEST.key" in node.value:
                path_pairs.add((relative, fn))
            if isinstance(node, ast.Name) and node.id == "_person_key_path":
                path_pairs.add((relative, fn))
            if isinstance(node, ast.BinOp):
                rendered = ast.unparse(node)
                if ".sthambha" in rendered and "person" in rendered:
                    path_pairs.add((relative, fn))
            if not isinstance(node, ast.Call):
                continue
            call_name = (node.func.id if isinstance(node.func, ast.Name)
                         else node.func.attr if isinstance(node.func, ast.Attribute) else "")
            if call_name in aliases | {"load_key", "_raw_file_key", "from_private_bytes"}:
                private_pairs.add((relative, fn))
            if (isinstance(node.func, ast.Attribute)
                    and ast.unparse(node.func.value).endswith("Ed25519PrivateKey")):
                private_pairs.add((relative, fn))
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
            calls.add((relative, operation, origin))
    return calls, private_pairs, path_pairs


def test_person_authority_call_sites_and_private_loads_are_explicitly_confined():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    calls, private_pairs, path_pairs = _authority_guard_findings(scripts)
    assert calls == {
        ("network/join_setup.py", "issue_grant", "PersonKey(...)"),
        ("network/nak.py", "issue_grant", "person_priv"),
        ("network/nak.py", "encode_invite", "priv"),
    }
    assert all(_PRIVATE_CONSTRUCTOR_ALLOWLIST.values())
    assert all(_PERSON_PATH_ALLOWLIST.values())
    assert private_pairs == set(_PRIVATE_CONSTRUCTOR_ALLOWLIST)
    assert path_pairs == set(_PERSON_PATH_ALLOWLIST)


def test_authority_guard_catches_injected_aliased_private_load(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "scripts" / "network" / "nak.py").read_text()
    copy = tmp_path / "network"
    copy.mkdir()
    (copy / "nak.py").write_text(source + "\n"
        "def injected_alias_load():\n"
        "    alias = signer_mod.load_key\n"
        "    return alias(_person_key_path())\n")
    _calls, private_pairs, path_pairs = _authority_guard_findings(tmp_path)
    assert ("network/nak.py", "injected_alias_load") in private_pairs
    assert ("network/nak.py", "injected_alias_load") in path_pairs
    assert ("network/nak.py", "injected_alias_load") not in _PRIVATE_CONSTRUCTOR_ALLOWLIST
