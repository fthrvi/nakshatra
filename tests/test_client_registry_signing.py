"""client.py signs pillar GETs over the full path, including the query.

Since the Sthambha pillar's auth hardening, /chain and /peers reject an unsigned GET with 401, so `--registry` (and the
serve's `registry_url` model entries) need a keyid resolved from release state.
"""
import base64
import hashlib
import io
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import client  # noqa: E402
import nakshatra_auth as na  # noqa: E402


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@pytest.fixture()
def pillar(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    seen = []

    def open_pillar_url(target, **_kwargs):
        seen.append(target)
        return _Response(b'{"chain": []}')

    monkeypatch.setattr(na, "open_pillar_url", open_pillar_url)
    return seen


def test_explicit_blank_keyid_keeps_the_request_unsigned(pillar, monkeypatch):
    monkeypatch.setenv("NKS_REGISTRY_KEYID", "")
    client._registry_urlopen("https://pillar.invalid/chain?model=m").read()
    assert pillar == ["https://pillar.invalid/chain?model=m"]


def test_with_a_keyid_the_get_is_signed_over_the_full_path_including_the_query(pillar, monkeypatch):
    priv = ed25519.Ed25519PrivateKey.generate()
    priv_bytes = priv.private_bytes_raw()
    monkeypatch.setenv("NKS_REGISTRY_KEYID", "hub")
    monkeypatch.setattr(na, "load_or_create_worker_key", lambda *a, **k: (priv_bytes, "unused"))

    client._registry_urlopen("https://pillar.invalid/chain?model=qwen3-30b-q3").read()

    request = pillar[0]
    assert isinstance(request, Request)
    path = request.selector
    header = request.get_header("Authorization")
    assert path == "/chain?model=qwen3-30b-q3"
    assert header.startswith("Sthambha-Ed25519 ") and 'keyid="hub"' in header
    sig = base64.b64decode(header.split('sig="')[1].split('"')[0])
    ts = header.split('ts="')[1].split('"')[0]
    pub = priv.public_key()
    good = f"GET\n/chain?model=qwen3-30b-q3\n{hashlib.sha256(b'').hexdigest()}\n{ts}".encode()
    pub.verify(sig, good)  # verifies over the FULL path, query included
    with pytest.raises(Exception):
        pub.verify(sig, f"GET\n/chain\n{hashlib.sha256(b'').hexdigest()}\n{ts}".encode())  # NOT over the bare path


def test_public_key_fallback_uses_the_same_signing_key(pillar, monkeypatch):
    monkeypatch.delenv("NKS_REGISTRY_KEYID", raising=False)
    priv = ed25519.Ed25519PrivateKey.generate()
    public_key_hex = priv.public_key().public_bytes_raw().hex()
    monkeypatch.setattr(
        na, "load_or_create_worker_key",
        lambda *_a, **_k: (priv.private_bytes_raw(), public_key_hex),
    )

    client._registry_urlopen("https://pillar.invalid/chain?model=m").read()

    request = pillar[0]
    assert isinstance(request, Request)
    assert f'keyid="{public_key_hex}"' in request.get_header("Authorization")


def test_a_401_names_the_attempted_keyid_and_its_source(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NKS_REGISTRY_KEYID", raising=False)
    registrar = tmp_path / ".nakshatra" / "registrar.env"
    registrar.parent.mkdir()
    registrar.write_text("NKS_NODE_ID=hub\n")
    priv = ed25519.Ed25519PrivateKey.generate().private_bytes_raw()
    monkeypatch.setattr(na, "load_or_create_worker_key", lambda *_a, **_k: (priv, "unused"))

    def refuse(target, **_kwargs):
        url = target.full_url if isinstance(target, Request) else target
        raise HTTPError(url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(na, "open_pillar_url", refuse)
    with pytest.raises(RuntimeError) as error:
        client._try_pillar_chain("https://pillar.invalid", "m")
    message = str(error.value)
    assert "tried keyid 'hub'" in message
    assert f"NKS_NODE_ID in {registrar}" in message


# ── gRPC: an authenticated worker (registered with a pillar => auth_required) needs signed calls ──────────────────────
# The driver had no way to send them, so it could only drive unauthenticated (static-YAML, Mode A) workers.

import nakshatra_grpc_auth as ga  # noqa: E402
import nakshatra_pb2 as pb  # noqa: E402


def _signer(monkeypatch):
    priv = ed25519.Ed25519PrivateKey.generate()
    monkeypatch.setenv("NKS_REGISTRY_KEYID", "hub")
    monkeypatch.setattr(na, "load_or_create_worker_key", lambda *a, **k: (priv.private_bytes_raw(), "unused"))
    return priv.public_key().public_bytes_raw().hex()


def test_explicit_blank_keyid_grpc_calls_carry_no_metadata(monkeypatch):
    monkeypatch.setenv("NKS_REGISTRY_KEYID", "")
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
    monkeypatch.setenv("NKS_REGISTRY_KEYID", "")
    seen = {}

    class _FakeStub:
        def Inference(self, gen, metadata=None):
            seen["metadata"] = metadata
            next(gen)
            return iter([pb.InferenceStep(session_id="s", step_id="1")])

    client.InferenceStream(_FakeStub(), "w0").step(pb.InferenceStep(session_id="s", step_id="1", prefix_length=3))
    assert seen["metadata"] is None
