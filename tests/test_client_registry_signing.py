"""client.py registry mode must SIGN its pillar GETs (NKS_REGISTRY_KEYID), over the full path including the query.

Since the Sthambha pillar's auth hardening, /chain and /peers reject an unsigned GET with 401, so `--registry` (and the
serve's `registry_url` model entries) silently stopped working. Unset = unsigned, the old behaviour.
"""
import base64
import hashlib
import http.server
import sys
import threading
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import client  # noqa: E402
import nakshatra_auth as na  # noqa: E402


class _Capture(http.server.BaseHTTPRequestHandler):
    seen = []

    def do_GET(self):
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        body = b'{"chain": []}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    _Capture.seen = []
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Capture)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_without_a_keyid_the_request_is_unsigned_as_before(server, monkeypatch):
    monkeypatch.delenv("NKS_REGISTRY_KEYID", raising=False)
    client._registry_urlopen(f"{server}/chain?model=m").read()
    assert _Capture.seen == [("/chain?model=m", None)]


def test_with_a_keyid_the_get_is_signed_over_the_full_path_including_the_query(server, monkeypatch):
    priv = ed25519.Ed25519PrivateKey.generate()
    priv_bytes = priv.private_bytes_raw()
    monkeypatch.setenv("NKS_REGISTRY_KEYID", "hub")
    monkeypatch.setattr(na, "load_or_create_worker_key", lambda *a, **k: (priv_bytes, "unused"))

    client._registry_urlopen(f"{server}/chain?model=qwen3-30b-q3").read()

    path, header = _Capture.seen[0]
    assert path == "/chain?model=qwen3-30b-q3"
    assert header.startswith("Sthambha-Ed25519 ") and 'keyid="hub"' in header
    sig = base64.b64decode(header.split('sig="')[1].split('"')[0])
    ts = header.split('ts="')[1].split('"')[0]
    pub = priv.public_key()
    good = f"GET\n/chain?model=qwen3-30b-q3\n{hashlib.sha256(b'').hexdigest()}\n{ts}".encode()
    pub.verify(sig, good)  # verifies over the FULL path, query included
    with pytest.raises(Exception):
        pub.verify(sig, f"GET\n/chain\n{hashlib.sha256(b'').hexdigest()}\n{ts}".encode())  # NOT over the bare path


def test_a_401_from_the_pillar_explains_how_to_fix_it(monkeypatch):
    class _Refuse(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Refuse)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.delenv("NKS_REGISTRY_KEYID", raising=False)
    try:
        with pytest.raises(RuntimeError, match="NKS_REGISTRY_KEYID"):
            client._try_pillar_chain(f"http://127.0.0.1:{srv.server_port}", "m")
    finally:
        srv.shutdown()


# ── gRPC: an authenticated worker (registered with a pillar => auth_required) needs signed calls ──────────────────────
# The driver had no way to send them, so it could only drive unauthenticated (static-YAML, Mode A) workers.

import nakshatra_grpc_auth as ga  # noqa: E402
import nakshatra_pb2 as pb  # noqa: E402


def _signer(monkeypatch):
    priv = ed25519.Ed25519PrivateKey.generate()
    monkeypatch.setenv("NKS_REGISTRY_KEYID", "hub")
    monkeypatch.setattr(na, "load_or_create_worker_key", lambda *a, **k: (priv.private_bytes_raw(), "unused"))
    return priv.public_key().public_bytes_raw().hex()


def test_without_a_keyid_grpc_calls_carry_no_metadata(monkeypatch):
    monkeypatch.delenv("NKS_REGISTRY_KEYID", raising=False)
    assert client._grpc_auth_metadata("Forward", b"x") is None


def test_the_forward_metadata_verifies_with_the_workers_own_check(monkeypatch):
    pub_hex = _signer(monkeypatch)
    req = pb.ForwardRequest(hidden_in=b"abc", batch=1, n_tokens=1)
    (key, header), = client._grpc_auth_metadata("Forward", req.SerializeToString())
    assert key == "authorization"
    resolver = lambda keyid: pub_hex if keyid == "hub" else None  # noqa: E731
    assert ga.verify_grpc_call("/nakshatra.Nakshatra/Forward", header, req.SerializeToString(), resolver) == "hub"
    with pytest.raises(ga.AuthError):  # signed for a DIFFERENT body -> a replayed header cannot authorise another request
        ga.verify_grpc_call("/nakshatra.Nakshatra/Forward", header, b"other", resolver)
    with pytest.raises(ga.AuthError):  # and not valid for another method
        ga.verify_grpc_call("/nakshatra.Nakshatra/Sleep", header, req.SerializeToString(), resolver)


def test_the_streaming_inference_call_is_signed_over_its_first_frame(monkeypatch):
    pub_hex = _signer(monkeypatch)
    seen = {}

    class _FakeStub:
        def Inference(self, gen, metadata=None):
            seen["metadata"] = metadata
            seen["first"] = next(gen)
            return iter([pb.InferenceStep(session_id="s", step_id="1")])

    stream = client.InferenceStream(_FakeStub(), "w0")
    assert "metadata" not in seen  # not started until the first step: the first frame has to exist to be signed
    step = pb.InferenceStep(session_id="s", step_id="1", prefix_length=3)
    stream.step(step)
    (_, header), = seen["metadata"]
    assert ga.verify_grpc_call("/nakshatra.Nakshatra/Inference", header, step.SerializeToString(),
                               lambda k: pub_hex, is_streaming=True) == "hub"
    assert seen["first"] == step


def test_an_unsigned_stream_still_starts_with_no_metadata(monkeypatch):
    monkeypatch.delenv("NKS_REGISTRY_KEYID", raising=False)
    seen = {}

    class _FakeStub:
        def Inference(self, gen, metadata=None):
            seen["metadata"] = metadata
            next(gen)
            return iter([pb.InferenceStep(session_id="s", step_id="1")])

    client.InferenceStream(_FakeStub(), "w0").step(pb.InferenceStep(session_id="s", step_id="1", prefix_length=3))
    assert seen["metadata"] is None
