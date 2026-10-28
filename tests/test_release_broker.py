"""Approval-broker release signing tests; every broker is loopback and in-process."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519


REL = Path(__file__).resolve().parent.parent / "release"
sys.path.insert(0, str(REL))

import broker_signer  # noqa: E402
import build as B  # noqa: E402
import keyroll as KR  # noqa: E402
import releasekit as rk  # noqa: E402


TOKEN = "release-test-token-never-real"
REQUEST_ID = "0123456789abcdef"


class FakeReleaseBroker:
    def __init__(self, private, *, corrupt=None, redirect=False):
        self.private = private
        self.corrupt = corrupt
        self.redirect = redirect
        self.enqueues = []
        self.sign_calls = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _reply(self, status, document, **headers):
                encoded = json.dumps(document).encode()
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def _body(self):
                return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

            def do_GET(self):
                assert self.headers.get("Authorization") == f"Bearer {TOKEN}"
                if self.path == "/principal/v1":
                    if outer.redirect:
                        self._reply(302, {}, Location="http://127.0.0.1:1/principal/v1")
                    else:
                        self._reply(200, {"principal": "nakshatra-release"})
                elif self.path == f"/result/v1?request_id={REQUEST_ID}":
                    self._reply(200, {"status": "approved"})
                else:
                    self._reply(404, {"error": "not found"})

            def do_POST(self):
                assert self.headers.get("Authorization") == f"Bearer {TOKEN}"
                body = self._body()
                if self.path == "/enqueue/v1":
                    outer.enqueues.append(body)
                    self._reply(200, {"request_id": REQUEST_ID})
                    return
                if self.path != "/release-sign/v1":
                    self._reply(404, {"error": "not found"})
                    return
                outer.sign_calls += 1
                assert body["request_id"] == REQUEST_ID
                exact = body["exact_content"]
                assert exact == outer.enqueues[-1]["exact_content"]
                items = [base64.b64decode(item["b64"], validate=True)
                         for item in exact["items"]]
                if exact["kind"] == "release":
                    manifest = json.loads(items[0])
                    first_signature = outer.private.sign(items[0])
                    signed_manifest = dict(manifest, sig={
                        "alg": "Ed25519",
                        "keyid": outer.private.public_key().public_bytes_raw().hex()[:16],
                        "value": base64.b64encode(first_signature).decode(),
                    })
                    latest = json.loads(items[1])
                    latest["manifest_sha256"] = hashlib.sha256(
                        json.dumps(signed_manifest, indent=1, sort_keys=True).encode()).hexdigest()
                    if outer.corrupt == "wrong_sha":
                        latest["manifest_sha256"] = "f" * 64
                    items = [items[0], rk.canonical(latest)]
                signatures = [outer.private.sign(item) for item in items]
                if outer.corrupt == "wrong_signature":
                    signatures[0] = bytes(64)
                elif outer.corrupt == "wrong_items":
                    items[0] = b'{"substituted":true}'
                elif outer.corrupt == "swapped":
                    items.reverse()
                    signatures.reverse()
                self._reply(200, {
                    "request_id": REQUEST_ID,
                    "signatures": [signature.hex() for signature in signatures],
                    "items_b64": [base64.b64encode(item).decode() for item in items],
                })

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


@contextmanager
def _broker_config(tmp_path, *, corrupt=None, redirect=False, token_mode=0o600):
    private = ed25519.Ed25519PrivateKey.generate()
    broker = FakeReleaseBroker(private, corrupt=corrupt, redirect=redirect)
    directory = tmp_path / "broker-config"
    directory.mkdir()
    token = directory / "token"
    token.write_text(TOKEN)
    token.chmod(token_mode)
    public_path = directory / "release.pub"
    public_path.write_text(private.public_key().public_bytes_raw().hex() + "\n")
    public_path.chmod(0o644)
    config_path = directory / "release-broker.json"
    config_path.write_text(json.dumps({
        "url": broker.url,
        "token_file": str(token),
    }))
    config_path.chmod(0o600)
    try:
        yield broker, directory, private
    finally:
        broker.close()


@pytest.fixture
def release_env(tmp_path, monkeypatch):
    repo = tmp_path / "component"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "module.py").write_text("OK = True\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run([
        "git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "fixture"], check=True)
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then echo 'uv 0.0.0'; exit 0; fi\n"
        "while [ $# -gt 0 ]; do if [ \"$1\" = \"-o\" ]; then "
        "echo 'cryptography==46.0.6 --hash=sha256:00' > \"$2\"; fi; shift; done\n")
    uv.chmod(0o755)
    monkeypatch.setattr(B, "SPEC", {
        "name": "nakshatra-node",
        "python": "3.12",
        "requirements": ["cryptography==46.0.6"],
        "components": [{
            "name": "component", "repo": str(repo), "paths": ["pkg"], "pythonpath": "pkg",
        }],
        "services": {},
        "health": [],
    })
    return {"dist": tmp_path / "dist", "uv": uv, "tmp": tmp_path}


def _build(env, config, version="1.2.3"):
    return B.build(
        version, "canary", {}, None, env["dist"], env["uv"], now=123,
        release_broker=config)


def test_release_broker_happy_path_is_one_enqueue_and_writes_both(release_env):
    with _broker_config(release_env["tmp"]) as (broker, config, private):
        version_dir = _build(release_env, config)
        manifest = json.loads((version_dir / "manifest.json").read_text())
        latest = json.loads((release_env["dist"] / "canary" / "latest.json").read_text())
        public_hex = private.public_key().public_bytes_raw().hex()
        assert rk.verify(manifest, public_hex)
        assert rk.verify(latest, public_hex)
        assert latest["manifest_sha256"] == rk.sha256_file(version_dir / "manifest.json")
        assert len(broker.enqueues) == 1 and broker.sign_calls == 1
        assert broker.enqueues[0]["exact_content"]["kind"] == "release"
        assert len(broker.enqueues[0]["exact_content"]["items"]) == 2


@pytest.mark.parametrize("corrupt", [
    "wrong_signature", "wrong_items", "wrong_sha", "swapped",
])
def test_release_broker_bad_batch_publishes_nothing(release_env, corrupt):
    with _broker_config(release_env["tmp"], corrupt=corrupt) as (broker, config, _private):
        with pytest.raises(broker_signer.BrokerSigningError):
            _build(release_env, config)
        assert not (release_env["dist"] / "canary" / "1.2.3").exists()
        assert not (release_env["dist"] / "canary" / "latest.json").exists()
        assert len(broker.enqueues) == 1 and broker.sign_calls == 1


def test_release_broker_redirect_is_refused_without_publication(release_env):
    with _broker_config(release_env["tmp"], redirect=True) as (broker, config, _private):
        with pytest.raises(broker_signer.BrokerSigningError, match="HTTP 302"):
            _build(release_env, config)
        assert broker.enqueues == []
        assert not (release_env["dist"] / "canary" / "1.2.3").exists()
        assert not (release_env["dist"] / "canary" / "latest.json").exists()


def test_release_broker_token_must_be_0600(release_env):
    with _broker_config(release_env["tmp"], token_mode=0o640) as (broker, config, _private):
        with pytest.raises(broker_signer.BrokerSigningError, match="0600"):
            _build(release_env, config)
        assert broker.enqueues == []
        assert not (release_env["dist"] / "canary" / "1.2.3").exists()
        assert not (release_env["dist"] / "canary" / "latest.json").exists()


def test_keyroll_wrong_new_pubkey_is_refused_before_enqueue(tmp_path):
    old = ed25519.Ed25519PrivateKey.generate()
    old_path = tmp_path / "old.key"
    old_path.write_text(old.private_bytes_raw().hex())
    with _broker_config(tmp_path) as (broker, config, _private):
        with pytest.raises(SystemExit):
            KR.main([
                "make", "--old-key", str(old_path), "--new-broker", str(config),
                "--new-pubkey", "0" * 64, "--issued-at", "123", "--reason", "test roll",
                "--out", str(tmp_path / "keyroll.json"),
            ])
        assert broker.enqueues == []
        assert not (tmp_path / "keyroll.json").exists()


def test_refresh_latest_happy_path_uses_refresh_kind(release_env):
    with _broker_config(release_env["tmp"]) as (broker, config, private):
        version_dir = _build(release_env, config)
        path = B.refresh_latest_pointer(
            "1.2.3", "canary", release_env["dist"], config, now=456)
        latest = json.loads(path.read_text())
        assert latest["created"] == 456
        assert latest["manifest_sha256"] == rk.sha256_file(version_dir / "manifest.json")
        assert rk.verify(latest, private.public_key().public_bytes_raw().hex())
        assert [request["exact_content"]["kind"] for request in broker.enqueues] == [
            "release", "refresh"]
        assert broker.sign_calls == 2


def test_refresh_refuses_signed_latest_masquerading_as_manifest(release_env):
    with _broker_config(release_env["tmp"]) as (broker, config, _private):
        version_dir = _build(release_env, config)
        latest_path = release_env["dist"] / "canary" / "latest.json"
        signed_latest = latest_path.read_bytes()
        latest_path.unlink()
        (version_dir / "manifest.json").write_bytes(signed_latest)

        with pytest.raises(RuntimeError, match="not valid for the broker release key"):
            B.refresh_latest_pointer(
                "1.2.3", "canary", release_env["dist"], config, now=456)

        assert not latest_path.exists()
        assert len(broker.enqueues) == 1
        assert broker.sign_calls == 1


def test_installed_node_imports_need_only_scripts_and_sthambha(tmp_path):
    repo = Path(__file__).resolve().parent.parent
    candidates = (
        os.environ.get("NAK_STHAMBHA_PATH", ""),
        *os.environ.get("PYTHONPATH", "").split(os.pathsep),
        str(Path.home() / "sthambha"),
        str(Path.home() / "sthambha-signer"),
        str(repo.parent / "sthambha"),
        "/home/prithvi/sthambha",
    )
    sthambha = next(
        (Path(candidate) for candidate in candidates
         if candidate and (Path(candidate) / "sthambha" / "signer.py").exists()),
        None,
    )
    assert sthambha is not None, "the node-import regression needs a Sthambha checkout"
    scripts = repo / "scripts"
    code = (
        "import sys; "
        f"roots = [{str(scripts)!r}, {str(sthambha)!r}]; "
        "sys.path[:0] = roots; "
        "assert sys.path[:2] == roots; "
        f"assert {str(repo)!r} not in sys.path; "
        "import network.person_custody, network.nak, network.nakd"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=tmp_path,
        text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_malformed_broker_status_line_is_never_reflected():
    import socket
    import threading

    from network import broker_transport

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def serve():
        conn, _ = listener.accept()
        with conn:
            conn.recv(65536)
            conn.sendall(b"BROKER-OUTPUT-SECRET\r\n\r\n")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    config = broker_transport.BrokerConfig(f"http://127.0.0.1:{listener.getsockname()[1]}", "tok")
    with pytest.raises(SystemExit) as caught:
        broker_transport.request_json(
            config, "GET", "/pending/v1", error_info=lambda _p, _e: ("", False),
            unavailable_message="approval service unavailable",
            invalid_message="approval service returned an invalid reply")
    thread.join(timeout=5)
    listener.close()
    assert "BROKER-OUTPUT" not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
