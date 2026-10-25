"""The person-custody seam is the only path from Nakshatra to person authority."""
from __future__ import annotations

import ast
import base64
from collections import Counter
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
                 principal="nakshatra-person", pending_polls=0,
                 result_failures=0, sign_failures=0,
                 drop_approved_result=False, drop_sign_response=False,
                 returned_request_id=None):
        self.private = private
        self.status = status
        self.corrupt = corrupt
        self.principal = principal
        self.pending_polls = pending_polls
        self.result_failures = result_failures
        self.sign_failures = sign_failures
        self.drop_approved_result = drop_approved_result
        self.dropped_approved_result = False
        self.drop_sign_response = drop_sign_response
        self.dropped_sign_response = False
        self.returned_request_id = returned_request_id
        self.enqueues = []
        self.sign_calls = 0
        self.principal_calls = 0
        self.result_calls = 0
        self._requests = {}
        self._signed_requests = set()
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
                    if state and owner.result_failures:
                        owner.result_failures -= 1
                        self._reply(503, {"error": "temporary", "retryable": True})
                    elif state and owner.pending_polls:
                        owner.pending_polls -= 1
                        self._reply(200, {"status": "pending"})
                    elif state and owner.drop_approved_result and not owner.dropped_approved_result:
                        owner.dropped_approved_result = True
                        self.close_connection = True
                        self.connection.shutdown(socket.SHUT_RDWR)
                        self.connection.close()
                    else:
                        status = ("consumed" if state and owner.dropped_approved_result
                                  else owner.status if state else "unknown")
                        self._reply(200, {"status": status})
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
                    request_id = (owner.returned_request_id
                                  or f"{len(owner.enqueues) + 1:016x}")
                    owner.enqueues.append(body)
                    owner._requests[request_id] = exact
                    self._reply(200, {"request_id": request_id})
                    return
                if self.path == "/person-sign/v1":
                    owner.sign_calls += 1
                    if owner.sign_failures:
                        owner.sign_failures -= 1
                        self._reply(503, {"error": "temporary", "retryable": True})
                        return
                    request_id = body.get("request_id")
                    if request_id in owner._signed_requests:
                        self._reply(409, {"error": "person signing request was already signed"})
                        return
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
                    owner._signed_requests.add(request_id)
                    if owner.drop_sign_response and not owner.dropped_sign_response:
                        owner.dropped_sign_response = True
                        self.close_connection = True
                        self.connection.shutdown(socket.SHUT_RDWR)
                        self.connection.close()
                        return
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
            principal="nakshatra-person", pending_polls=0,
            result_failures=0, sign_failures=0, drop_approved_result=False,
            drop_sign_response=False, returned_request_id=None):
    signer_dir, private = _identity(tmp_path, monkeypatch)
    broker = FakeBroker(
        private, status=status, corrupt=corrupt, principal=principal,
        pending_polls=pending_polls, result_failures=result_failures,
        sign_failures=sign_failures,
        drop_approved_result=drop_approved_result,
        drop_sign_response=drop_sign_response,
        returned_request_id=returned_request_id).start()
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


def test_broker_success_body_never_reflects_an_unknown_status(tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch, status=TOKEN) as (broker, _private):
        with pytest.raises(SystemExit) as refused:
            person_custody.person_sign("invite", b"invite", summary="Invite")
        assert str(refused.value) == (
            "person broker returned an invalid terminal status; nothing was signed; "
            "approval may already have been granted; request_id=0000000000000001 can be retried "
            "without a new prompt")
        assert TOKEN not in str(refused.value)
        assert len(broker.enqueues) == 1 and broker.sign_calls == 0


def test_broker_returned_bearer_as_request_id_is_refused_without_leaking(
        tmp_path, monkeypatch):
    with _broker(
            tmp_path, monkeypatch, returned_request_id=TOKEN) as (broker, _private):
        with pytest.raises(SystemExit) as refused:
            person_custody.person_sign("invite", b"invite", summary="Invite")
        assert str(refused.value) == (
            "person broker returned an invalid request id; nothing was signed")
        assert TOKEN not in str(refused.value)
        assert broker.result_calls == 0 and broker.sign_calls == 0


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


def test_resolve_is_authenticated_preflight_only_and_not_a_signing_kind(
        tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch) as (broker, _private):
        act = person_custody.broker_preflight("resolve", "Resolve disputed hold")
        assert act.kind == "resolve" and act.mode == "broker"
        act.assert_current()
        assert broker.enqueues == [] and broker.sign_calls == 0

        with pytest.raises(SystemExit, match="unknown person-signing kind 'resolve'"):
            person_custody.person_sign(
                "resolve", b"must never become a broker shape", summary="Resolve disputed hold")
        assert broker.enqueues == [] and broker.sign_calls == 0


def test_broker_act_refuses_if_marker_changes_before_signing(tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch) as (broker, _private):
        act = person_custody.broker_preflight("invite", "Invite")
        key = person_custody.PersonKey("invite", "Invite", act=act)
        (tmp_path / ".nakshatra" / "net" / "person-custody").write_text("file\n")
        with pytest.raises(SystemExit, match="changed from broker to file.*nothing was signed"):
            key.sign(b"the stale act must not sign")
        assert broker.enqueues == [] and broker.sign_calls == 0


def test_person_act_constructor_is_private():
    with pytest.raises(TypeError, match="construction is private"):
        person_custody.PersonAct("invite", "Invite", "file", None, None)


def test_forged_file_act_under_broker_custody_refuses_before_private_key_load(
        tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch) as (broker, _private):
        forged = object.__new__(person_custody.PersonAct)
        for name, value in {
            "kind": "invite", "summary": "Invite", "mode": "file",
            "config": None, "public_key": None,
            "_sentinel": person_custody._PERSON_ACT_SENTINEL,
        }.items():
            object.__setattr__(forged, name, value)
        monkeypatch.setattr(
            person_custody, "_load_matching_person_key",
            lambda **_kwargs: pytest.fail("forged act reached the private person key"))
        with pytest.raises(SystemExit, match="changed from file to broker.*nothing was signed"):
            person_custody.person_sign(
                "invite", b"reviewer forged act", summary="Invite", act=forged)
        assert broker.enqueues == [] and broker.sign_calls == 0


@pytest.mark.parametrize("endpoint", ["result", "person-sign"])
def test_broker_retries_two_503s_within_the_same_human_act(
        endpoint, tmp_path, monkeypatch):
    options = {"result_failures": 2} if endpoint == "result" else {"sign_failures": 2}
    monkeypatch.setattr(person_custody, "_BROKER_RETRY_INITIAL_S", 0)
    monkeypatch.setattr(person_custody, "_BROKER_RETRY_MAX_S", 0)
    with _broker(tmp_path, monkeypatch, **options) as (broker, private):
        statement = b"one act, one request id"
        signature = person_custody.person_sign("invite", statement, summary="Invite")
        private.public_key().verify(signature, statement)
        assert len(broker.enqueues) == 1
        if endpoint == "result":
            assert broker.result_calls == 3 and broker.sign_calls == 1
        else:
            assert broker.result_calls == 1 and broker.sign_calls == 3


def test_broker_lost_approved_result_proceeds_from_consumed_with_same_request_id(
        tmp_path, monkeypatch):
    monkeypatch.setattr(person_custody, "_BROKER_RETRY_INITIAL_S", 0)
    monkeypatch.setattr(person_custody, "_BROKER_RETRY_MAX_S", 0)
    with _broker(tmp_path, monkeypatch, drop_approved_result=True) as (broker, private):
        statement = b"approved response was lost"
        signature = person_custody.person_sign("invite", statement, summary="Invite")
        private.public_key().verify(signature, statement)
        assert broker.dropped_approved_result is True
        assert broker.result_calls == 2 and broker.sign_calls == 1
        assert len(broker.enqueues) == 1


def test_broker_lost_sign_response_reports_consumed_and_requires_new_approval(
        tmp_path, monkeypatch):
    monkeypatch.setattr(person_custody, "_BROKER_RETRY_INITIAL_S", 0)
    monkeypatch.setattr(person_custody, "_BROKER_RETRY_MAX_S", 0)
    with _broker(tmp_path, monkeypatch, drop_sign_response=True) as (broker, _private):
        with pytest.raises(SystemExit) as refused:
            person_custody.person_sign(
                "invite", b"signed response was lost", summary="Invite")
        message = str(refused.value)
        assert "request was signed and consumed" in message
        assert "signatures were lost in transit" in message
        assert "nothing was sent or written" in message
        assert "Starting the act again will ask for a new approval" in message
        assert "retried without another prompt" not in message
        assert "without a new prompt" not in message
        assert broker.dropped_sign_response is True
        assert broker.sign_calls == 2 and len(broker.enqueues) == 1


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
        io.BytesIO(json.dumps({"error": "test-broker-token", "detail": "body-secret"}).encode()))
    monkeypatch.setattr(person_custody._BROKER_OPENER, "open", lambda *_args, **_kw: (_ for _ in ()).throw(error))
    with pytest.raises(SystemExit) as refused:
        person_custody._broker_json(
            person_custody._BrokerConfig("http://broker.invalid", "token"),
            "POST", "/enqueue/v1", {})
    message = str(refused.value)
    assert message == "approval service error (HTTP 418)"
    assert TOKEN not in message
    assert "body-secret" not in message and "header-secret" not in message and "header-value" not in message


def test_broker_known_error_uses_only_the_closed_local_identifier(monkeypatch):
    error = urllib.error.HTTPError(
        "http://broker.invalid/person-sign/v1", 409, "remote-reason", {},
        io.BytesIO(b'{"error":"person signing request was already signed"}'))
    monkeypatch.setattr(
        person_custody._BROKER_OPENER, "open",
        lambda *_args, **_kw: (_ for _ in ()).throw(error))
    with pytest.raises(SystemExit) as refused:
        person_custody._broker_json(
            person_custody._BrokerConfig("http://broker.invalid", "token"),
            "POST", "/person-sign/v1", {})
    assert str(refused.value) == "approval service error (HTTP 409, code already-signed)"


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


@pytest.mark.parametrize("marker_state", ["missing", "unreadable"])
def test_broker_configuration_without_a_readable_marker_never_falls_back_to_file(
        marker_state, tmp_path, monkeypatch):
    _identity(tmp_path, monkeypatch)
    net = tmp_path / ".nakshatra" / "net"
    (net / "person-broker.json").write_text("{}")
    if marker_state == "unreadable":
        (net / "person-custody").write_bytes(b"\xff")
    with pytest.raises(SystemExit, match="custody marker|refusing file custody"):
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


@pytest.mark.parametrize("allowlisted_finding", [
    ("network/join_setup.py", "_established_identity", "_raw_file_key", "person_path"),
    ("network/join_setup.py", "ensure_identity", "_raw_file_key", "person_path"),
])
def test_join_allowlist_entries_never_read_a_private_person_key_in_broker_mode(
        allowlisted_finding, tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch) as (_broker_server, _private):
        assert _PRIVATE_CONSTRUCTOR_ALLOWLIST[allowlisted_finding] >= 1
        private_path = tmp_path / ".sthambha" / "person-TEST.key"
        private_path.write_text("not a key and deliberately unreadable by the broker path")
        monkeypatch.setattr(
            join_setup, "_raw_file_key",
            lambda *_args: pytest.fail("broker mode loaded the private person key"))
        result = join_setup.ensure_identity(tmp_path)
        assert result["person"] == (
            tmp_path / ".sthambha" / "signer" / "person.pub").read_text().strip()


def test_grant_allowlist_entry_loads_only_agent_key_in_broker_mode(tmp_path, monkeypatch):
    finding = ("network/nak.py", "_grant_renew_confined", "agent_files.load_key", "")
    assert _PRIVATE_CONSTRUCTOR_ALLOWLIST[finding] == 1
    with _broker(tmp_path, monkeypatch) as (broker, _private):
        monkeypatch.setattr(
            person_custody, "_load_matching_person_key",
            lambda **_kwargs: pytest.fail("broker renewal loaded the private person key"))
        assert nak.main(["grant", "renew", "--yes", "--no-lease"]) == 0
        assert broker.sign_calls == 1


def test_declined_grant_renew_never_creates_a_lease_key(tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch, status="declined") as (broker, _private):
        lease_path = tmp_path / ".sthambha" / "signer" / "lease.key"
        lease_path.unlink(missing_ok=True)
        before = _snapshot(tmp_path)
        with pytest.raises(SystemExit, match="not approved"):
            nak.main(["grant", "renew", "--yes"])
        assert not lease_path.exists()
        assert _snapshot(tmp_path) == before
        assert broker.sign_calls == 0


def test_approved_grant_renew_persists_the_in_memory_lease_key_after_signature(
        tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch) as (broker, _private):
        from cryptography.hazmat.primitives.asymmetric import ed25519
        signer_dir = tmp_path / ".sthambha" / "signer"
        lease_path = signer_dir / "lease.key"
        lease_path.unlink(missing_ok=True)
        assert nak.main(["grant", "renew", "--yes"]) == 0
        seed = lease_path.read_bytes()
        renewed = json.loads((signer_dir / "agents" / "agent.grant.json").read_text())
        assert len(seed) == 32
        assert ed25519.Ed25519PrivateKey.from_private_bytes(
            seed).public_key().public_bytes_raw().hex() == renewed["lease"]["issuer_pub"]
        assert broker.sign_calls == 1


@pytest.mark.parametrize("mode", ["file", "broker"])
def test_cli_report_preserves_file_bytes_and_broker_resolves_from_local_store(
        mode, tmp_path, monkeypatch):
    if mode == "broker":
        context = _broker(tmp_path, monkeypatch)
    else:
        _signer, private = _identity(tmp_path, monkeypatch)
        context = nullcontext((None, private))
    with context as (broker, private):
        node = "9b" * 32
        seen = []
        if broker is not None:
            from network.store import Store
            signer_dir = tmp_path / ".sthambha" / "signer"
            data_key = signer_mod.load_or_create_data_key(signer_dir)
            Store(tmp_path / ".nakshatra" / "net", data_key).upsert_contact(
                "8a" * 32, node, "active", petname="friend")

        class Terminal:
            def isatty(self):
                return True

        def fake_call(_sock, request):
            seen.append(request)
            if request["op"] == "report_cli":
                expected_subject = node if broker is not None else "friend"
                assert request["subject"] == expected_subject
                statement = nakd.person_action_statement(
                    "report-cli",
                    nakd.report_proof_payload(expected_subject, "abuse", "", {}), request["ts"])
                private.public_key().verify(bytes.fromhex(request["signature"]), statement)
                return {"subject": node, "status": "received"}
            raise AssertionError(request)

        monkeypatch.setattr(nak.sys, "stdin", Terminal())
        monkeypatch.setattr(nak, "call", fake_call)
        assert nak.main(["report", "friend", "--reason", "abuse"]) == 0
        assert [request["op"] for request in seen] == ["report_cli"]
        if broker is not None:
            raw = base64.b64decode(
                broker.enqueues[0]["exact_content"]["items"][0]["b64"], validate=True)
            assert json.loads(raw)["payload"]["subject"] == node


def test_broker_report_refuses_an_unresolved_local_contact_without_daemon_lookup(
        tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch) as (broker, _private):
        from network.store import Store
        signer_dir = tmp_path / ".sthambha" / "signer"
        data_key = signer_mod.load_or_create_data_key(signer_dir)
        Store(tmp_path / ".nakshatra" / "net", data_key)
        monkeypatch.setattr(nak.sys, "stdin", type("Terminal", (), {"isatty": lambda self: True})())
        monkeypatch.setattr(nak, "call", lambda *_args: pytest.fail("broker report used a daemon lookup"))
        with pytest.raises(SystemExit, match="local contact name.*nothing was signed"):
            nak.main(["report", "not-a-contact", "--reason", "abuse"])
        assert broker.enqueues == [] and broker.sign_calls == 0


@pytest.mark.parametrize("kind", ["worker", "consumer"])
def test_declined_broker_invite_cancels_the_already_staged_server_code(
        kind, tmp_path, monkeypatch):
    with _broker(tmp_path, monkeypatch, status="declined") as (broker, _private):
        monkeypatch.setenv("NAK_ONBOARD_TOKEN", "operator-secret")
        monkeypatch.setattr(nak, "_release_for_invite", lambda _url="": ({}, ""))
        calls = []
        posts = []

        def fake_call(_sock, request):
            calls.append(request["op"])
            if request["op"] == "status":
                return {"node": "ab" * 32, "relay": "relay.example:51820"}
            if request["op"] == "node_sign_stage":
                raise AssertionError("operator-token staging must not request a node signature")
            raise AssertionError("declined invite must not be registered locally")

        def fake_post(url, body):
            posts.append((url, body))
            if url.endswith(f"/{kind}-stage"):
                return {"code": f"{kind}-server-code", "expires_at": int(time.time()) + 3600}
            if url.endswith("/invite-cancel"):
                return {"cancelled": True}
            raise AssertionError(url)

        monkeypatch.setattr(nak, "call", fake_call)
        with pytest.raises(SystemExit, match="not approved"):
            nak.main(["invite", "--kind", kind, "--tier", "known",
                      "--redeem", "https://onboard.example/base", "--code-only"],
                     http_post=fake_post)
        assert calls == ["status"]
        assert posts[0][1]["token"] == "operator-secret"
        assert posts[1] == (
            "https://onboard.example/base/invite-cancel",
            {"token": "operator-secret", "code": f"{kind}-server-code"})
        assert broker.sign_calls == 0


@pytest.mark.parametrize("kind", ["worker", "consumer"])
def test_stage_post_failure_happens_before_approval_and_outputs_no_invite(
        kind, tmp_path, monkeypatch, capsys):
    with _broker(tmp_path, monkeypatch) as (broker, _private):
        monkeypatch.setenv("NAK_ONBOARD_TOKEN", "operator-secret")
        monkeypatch.setattr(nak, "_release_for_invite", lambda _url="": ({}, ""))
        calls = []
        posts = []

        def fake_call(_sock, request):
            calls.append(request["op"])
            if request["op"] == "status":
                return {"node": "ab" * 32, "relay": "relay.example:51820"}
            if request["op"] == "node_sign_stage":
                raise AssertionError("operator-token staging must not request a node signature")
            raise AssertionError("failed stage must not register an invite")

        def fake_post(url, body):
            posts.append((url, body))
            if url.endswith(f"/{kind}-stage"):
                raise SystemExit("stage unavailable")
            raise AssertionError(url)

        monkeypatch.setattr(nak, "call", fake_call)
        with pytest.raises(SystemExit, match="stage unavailable"):
            nak.main(["invite", "--kind", kind, "--tier", "known",
                      "--redeem", "https://onboard.example/base", "--code-only"],
                     http_post=fake_post)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert posts[0][0].endswith(f"/{kind}-stage")
        assert len(posts) == 1
        assert calls == ["status"]
        assert broker.enqueues == [] and broker.sign_calls == 0


_PERSON_LITERAL_ALLOWLIST = {
    ("network/join_setup.py",
     "2. person key      ~/.sthambha/person-TEST.key           TEST custody: a file, not a vault"):
        (1, "operator-facing bootstrap layout documents the legacy file-custody location"),
    ("network/join_setup.py",
     "3. signer state    ~/.sthambha/signer/                   node.pub, person.pub"):
        (1, "operator-facing bootstrap layout documents public signer identity state"),
    ("network/join_setup.py",
     '"""Whether ~/.sthambha contains anything except an empty signer directory."""'):
        (1, "the broker migration fence documents the authority directory it inspects"),
    ("network/join_setup.py", 'state = home / ".sthambha" / "signer"'):
        (1, "join setup must initialize the established public signer-state directory"),
    ("network/join_setup.py",
     'sthambha_authority_present = _sthambha_authority_present(home / ".sthambha", state)'):
        (1, "broker migration must refuse residual authority material in the legacy directory"),
    ("network/join_setup.py",
     'custody_note = "your person key is a TEST file at ~/.sthambha/person-TEST.key: back it up, and never share it"'):
        (1, "file-custody setup warning tells the operator what must be backed up"),
    ("network/nak.py",
     'nak invite --person-key ~/.sthambha/person-TEST.key [--ttl 86400] [--note "for Rajesh"]'):
        (1, "CLI module help preserves the established explicit-key example"),
    ("network/nak.py",
     'signer_dir = Path(os.environ.get("NAK_SIGNER_DIR", Path.home() / ".sthambha" / "signer"))'):
        (2, "two confined grant flows need the public signer and agent-key state directory"),
    ("network/nak.py",
     'help="explicit file key in file custody only (default: ~/.sthambha/person-TEST.key)")'):
        (1, "the file-only --person-key option documents its legacy default"),
    ("network/nakd.py",
     'signer_dir = Path(os.environ.get("NAK_SIGNER_DIR", Path.home() / ".sthambha" / "signer"))'):
        (1, "the daemon loads only its public identity and delegated node/agent state here"),
    ("network/nakd.py",
     '"NAK_SIGNER_SOCK", home / ".sthambha" / "signer" / "signer.sock")))'):
        (1, "the daemon connects to the separately confined signer socket"),
}


_PRIVATE_CONSTRUCTOR_ALLOWLIST = Counter({
    ("discovery/nakshatra_listing.py", "sign", "ed25519.Ed25519PrivateKey.from_private_bytes", "priv_bytes"): 1,
    ("identity_binding.py", "_ed_sign", "ed25519.Ed25519PrivateKey.from_private_bytes", "bytes.fromhex(priv_hex)"): 1,
    ("identity_binding.py", "pub_of", "ed25519.Ed25519PrivateKey.from_private_bytes", "bytes.fromhex(priv_hex)"): 1,
    ("joincode.py", "address_pub", "_ed.Ed25519PrivateKey.from_private_bytes", "bytes.fromhex(address['ik'])"): 1,
    ("joincode.py", "build_unsigned_address", "_ed.Ed25519PrivateKey.generate", ""): 1,
    ("joincode.py", "encode_invite", "_ed.Ed25519PrivateKey.generate", ""): 1,
    ("joincode.py", "invite_pub", "_ed.Ed25519PrivateKey.from_private_bytes", "bytes.fromhex(inv['ik'])"): 1,
    ("nakshatra_auth.py", "generate_keypair", "ed25519.Ed25519PrivateKey.generate", ""): 1,
    ("nakshatra_auth.py", "public_key_hex_from_private", "ed25519.Ed25519PrivateKey.from_private_bytes", "priv_bytes"): 1,
    ("nakshatra_auth.py", "sign_request", "ed25519.Ed25519PrivateKey.from_private_bytes", "priv_bytes"): 1,
    ("network/join_setup.py", "_established_identity", "_raw_file_key", "person_path"): 1,
    ("network/join_setup.py", "_established_identity", "agent_files.load_key", ""): 1,
    ("network/join_setup.py", "ensure_identity", "_raw_file_key", "person_path"): 2,
    ("network/join_setup.py", "redeem_consumer_invite", "ed25519.Ed25519PrivateKey.from_private_bytes", "node_key.read_bytes()"): 1,
    ("network/join_setup.py", "redeem_worker_invite", "ed25519.Ed25519PrivateKey.from_private_bytes", "node_key.read_bytes()"): 1,
    ("network/nak.py", "_grant_renew_confined", "agent_files.load_key", ""): 1,
    ("network/nak.py", "_grant_renew_confined", "_ed25519.Ed25519PrivateKey.from_private_bytes", "lease_seed"): 1,
    ("network/nakd.py", "__init__", "ed25519.Ed25519PrivateKey.from_private_bytes", "node_key"): 1,
    ("network/nakd.py", "node_sign_stage", "ed25519.Ed25519PrivateKey.from_private_bytes", "self._key"): 1,
    ("network/nakd.py", "report", "ed25519.Ed25519PrivateKey.from_private_bytes", "self._key"): 1,
    ("packaging/nakshatra_package.py", "sign", "ed25519.Ed25519PrivateKey.from_private_bytes", "priv_bytes"): 1,
    ("packaging/test_package_tool.py", "test_certify_refuses_invalid_package", "ed25519.Ed25519PrivateKey.generate", ""): 1,
    ("packaging/test_package_tool.py", "test_certify_signs_a_valid_package", "ed25519.Ed25519PrivateKey.generate", ""): 1,
    ("paid_admission.py", "_receipt", "ed25519.Ed25519PrivateKey.from_private_bytes", "self.private_key"): 1,
    ("sidecar_key.py", "libp2p_ed25519_private", "ed25519.Ed25519PrivateKey.from_private_bytes", "seed"): 1,
    ("test_identity_binding.py", "_key", "ed25519.Ed25519PrivateKey.generate", ""): 1,
    ("transport/connect.py", "_pub_of", "ed25519.Ed25519PrivateKey.from_private_bytes", "key"): 1,
    ("transport/identity_handshake.py", "make_proof", "ed25519.Ed25519PrivateKey.from_private_bytes", "priv_bytes"): 1,
    ("transport/secure_channel.py", "secure_handshake", "ed25519.Ed25519PrivateKey.from_private_bytes", "priv_ed_bytes"): 1,
    ("worker.py", "Info", "ed25519.Ed25519PrivateKey.from_private_bytes", "self.worker_priv"): 1,
    ("worker.py", "__init__", "ed25519.Ed25519PrivateKey.from_private_bytes", "worker_priv"): 1,
})

_PERSON_PATH_ALLOWLIST = Counter({
    ("network/join_setup.py", "ensure_identity", "_person_key_path", "home"): 1,
    ("network/join_setup.py", "main", "_person_key_path", "a.home"): 1,
    ("network/nak.py", "_grant_renew_confined", "_person_key_path", ""): 1,
    ("network/nak.py", "main", "_person_key_path", ""): 1,
})

_AUTHORITY_CALL_ALLOWLIST = Counter({
    ("network/join_setup.py", "ensure_identity", "S.issue_grant",
     "PersonKey('grant-issue', f'Give agent {agent} {','.join(AGENT_CAPS)}', "
     "act=custody_act.for_signing('grant-issue', f'Give agent {agent} {','.join(AGENT_CAPS)}'))"): 1,
    ("network/nak.py", "_grant_renew_confined", "signer_mod.issue_grant", "person_priv"): 1,
    ("network/nak.py", "main", "joincode.encode_invite", "priv"): 1,
})


def _normalized(node) -> str:
    return " ".join(ast.unparse(node).split()) if node is not None else ""


def _folded_string(node) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _folded_string(node.left), _folded_string(node.right)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                folded = _folded_string(value.value)
                if folded is None:
                    return None
                parts.append(folded)
            else:
                return None
        return "".join(parts)
    return None


def _has_private_person_fragment(node) -> bool:
    for candidate in ast.walk(node):
        folded = _folded_string(candidate)
        if folded is None:
            continue
        lowered = folded.lower()
        if "person" in lowered and (".key" in lowered or "-test" in lowered):
            return True
    return False


def _person_literal_findings(scripts: Path) -> Counter:
    """Find every forbidden marker in each string literal or f-string fragment."""
    markers = ("test.key", "person-test", ".sthambha")
    findings = Counter()
    for path in scripts.rglob("*.py"):
        relative = path.relative_to(scripts).as_posix()
        if relative == "network/person_custody.py":
            continue
        source = path.read_text()
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
        matched_lines = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            lowered = node.value.lower()
            if not any(marker in lowered for marker in markers):
                continue
            span = range(node.lineno - 1, (node.end_lineno or node.lineno))
            relevant = []
            for index in span:
                line_lower = lines[index].lower()
                if any(marker in line_lower for marker in markers):
                    relevant.append(index)
            # Constant folding can combine adjacent fragments whose individual source text does
            # not contain the complete marker. Still report the literal's opening line.
            matched_lines.update(relevant or [node.lineno - 1])
        for index in sorted(matched_lines):
            findings[(relative, lines[index].strip())] += 1
    return findings


def _authority_guard_findings(scripts: Path):
    calls, private_findings, path_findings = Counter(), Counter(), Counter()
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
            if not isinstance(node, ast.Call):
                continue
            call_text = _normalized(node.func)
            first_arg = _normalized(node.args[0]) if node.args else ""
            call_name = (node.func.id if isinstance(node.func, ast.Name)
                         else node.func.attr if isinstance(node.func, ast.Attribute) else "")
            private_call = call_name in aliases | {"load_key", "_raw_file_key", "from_private_bytes"}
            if (isinstance(node.func, ast.Attribute)
                    and _normalized(node.func.value).endswith("Ed25519PrivateKey")):
                private_call = True
            if private_call:
                private_findings[(relative, fn, call_text, first_arg)] += 1

            path_expression = None
            if call_name in {"read_bytes", "read_text"} and isinstance(node.func, ast.Attribute):
                path_expression = node.func.value
            elif call_name in {"open", "Path"} and node.args:
                path_expression = node.args[0]
            if path_expression is not None and (
                    any(isinstance(part, ast.Name) and part.id == "_person_key_path"
                        for part in ast.walk(path_expression))
                    or _has_private_person_fragment(path_expression)):
                path_findings[(relative, fn, call_text, _normalized(path_expression))] += 1
            if call_name == "_person_key_path":
                path_findings[(relative, fn, call_text, first_arg)] += 1

            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"issue_grant", "encode_invite", "encode_address"}):
                operation = node.func.attr
            elif isinstance(node.func, ast.Name) and node.func.id == "make_record":
                operation = node.func.id
            else:
                continue
            calls[(relative, fn, call_text, first_arg)] += 1
    return calls, private_findings, path_findings


def test_person_authority_call_sites_and_private_loads_are_explicitly_confined():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    calls, private_findings, path_findings = _authority_guard_findings(scripts)
    literal_findings = _person_literal_findings(scripts)
    literal_allowlist = Counter({
        finding: count for finding, (count, _reason) in _PERSON_LITERAL_ALLOWLIST.items()
    })
    assert calls == _AUTHORITY_CALL_ALLOWLIST
    assert all(count >= 1 and reason
               for count, reason in _PERSON_LITERAL_ALLOWLIST.values())
    assert literal_findings == literal_allowlist
    assert all(_PRIVATE_CONSTRUCTOR_ALLOWLIST.values())
    assert all(_PERSON_PATH_ALLOWLIST.values())
    assert private_findings == _PRIVATE_CONSTRUCTOR_ALLOWLIST
    assert path_findings == _PERSON_PATH_ALLOWLIST


def test_authority_guard_catches_injected_aliased_private_load(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "scripts" / "network" / "nak.py").read_text()
    copy = tmp_path / "network"
    copy.mkdir()
    (copy / "nak.py").write_text(source + "\n"
        "def injected_alias_load():\n"
        "    alias = signer_mod.load_key\n"
        "    return alias(_person_key_path())\n")
    _calls, private_findings, path_findings = _authority_guard_findings(tmp_path)
    private = ("network/nak.py", "injected_alias_load", "alias", "_person_key_path()")
    person_path = ("network/nak.py", "injected_alias_load", "_person_key_path", "")
    assert private_findings[private] == 1
    assert path_findings[person_path] == 1
    assert private not in _PRIVATE_CONSTRUCTOR_ALLOWLIST


def test_authority_guard_multiset_rejects_one_more_load_inside_allowlisted_function(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "scripts" / "network" /
              "join_setup.py").read_text()
    copy = tmp_path / "network"
    copy.mkdir()
    # Same function, same normalized call and argument: only a multiset notices the added load.
    source = source.replace(
        "_, stored_person_pub = _raw_file_key(person_path)",
        "_, stored_person_pub = _raw_file_key(person_path)\n"
        "            _raw_file_key(person_path)", 1)
    (copy / "join_setup.py").write_text(source)
    _calls, private_findings, _paths = _authority_guard_findings(tmp_path)
    finding = ("network/join_setup.py", "_established_identity", "_raw_file_key", "person_path")
    assert private_findings[finding] == _PRIVATE_CONSTRUCTOR_ALLOWLIST[finding] + 1
    assert private_findings != _PRIVATE_CONSTRUCTOR_ALLOWLIST


@pytest.mark.parametrize("expression", [
    "Path('person-' + 'TEST.key').read_bytes()",
    "open(Path(f\"{'person'}-{'TEST'}.key\"), 'rb')",
])
def test_authority_guard_constant_folds_private_person_paths(expression, tmp_path):
    (tmp_path / "probe.py").write_text(
        "from pathlib import Path\n"
        "def injected_path():\n"
        f"    return {expression}\n")
    _calls, _private, path_findings = _authority_guard_findings(tmp_path)
    assert path_findings
    assert all(finding[0:2] == ("probe.py", "injected_path") for finding in path_findings)


def test_authority_literal_guard_catches_reviewer_fstring_mutation(tmp_path):
    line = 'return Path(f"{Path.home()}/.sthambha/person-{suffix}.key").read_bytes()'
    (tmp_path / "probe.py").write_text(
        "from pathlib import Path\n"
        "def injected_path(suffix):\n"
        f"    {line}\n")
    findings = _person_literal_findings(tmp_path)
    assert findings == Counter({("probe.py", line): 1})
    assert not set(findings) <= set(_PERSON_LITERAL_ALLOWLIST)


def test_authority_literal_guard_catches_reviewer_split_literal_mutation(tmp_path):
    line = ('private_path = Path(".sthambha") / ("person-" + "TEST.key"); '
            'private_path.read_bytes()')
    (tmp_path / "probe.py").write_text(
        "from pathlib import Path\n"
        "def injected_path():\n"
        f"    {line}\n")
    findings = _person_literal_findings(tmp_path)
    assert findings == Counter({("probe.py", line): 1})
    assert not set(findings) <= set(_PERSON_LITERAL_ALLOWLIST)
