"""U4 B3: remotely fetched model bytes always have trusted SHA-256 provenance."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

pytest.importorskip("grpc", reason="worker.py imports grpc at module load")

import worker  # noqa: E402

_PRIV = bytes(range(32))
_NODE_ID = "bootstrap-worker"


class _Response:
    def __init__(self, body: bytes, headers=None):
        self._body = io.BytesIO(body)
        self.headers = headers or {}

    def read(self, size=-1):
        return self._body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _pillar_response(model_sha256=None):
    record = {
        "model_id": "model",
        "layer_start": 0,
        "layer_end": 1,
        "is_online": True,
        "node_id": "peer-a",
        "address": "peer.example:5500",
        "file_path": "/cache/slice.gguf",
    }
    if model_sha256 is not None:
        record["model_sha256"] = model_sha256
    return _Response(json.dumps({"files": [record]}).encode())


def test_remote_slice_hash_mismatch_is_refused_without_installing_file(
        monkeypatch, tmp_path):
    expected = "0" * 64
    payload = b"poisoned model bytes"
    pillar_calls = []
    peer_calls = []

    def fake_pillar_open(req, *, timeout, expected_spki_hash=None):
        pillar_calls.append((req, timeout, expected_spki_hash))
        return _pillar_response(expected)

    def fake_urlopen(url, timeout):
        peer_calls.append(url)
        return _Response(payload, {"Content-Length": str(len(payload))})

    monkeypatch.setattr(worker._wauth, "open_pillar_url", fake_pillar_open)
    monkeypatch.setattr(worker.urlrequest, "urlopen", fake_urlopen)
    dest = tmp_path / "slice.gguf"
    streamed = {}
    pin = "a" * 64

    with pytest.raises(RuntimeError, match="sha mismatch"):
        worker.fetch_sub_gguf_from_peer(
            "https://pillar.example", "model", 0, 1, str(dest),
            own_node_id=_NODE_ID, priv_key=_PRIV,
            spki_hash=pin,
            verified_sha_out=streamed,
        )

    assert len(pillar_calls) == 1
    assert pillar_calls[0][1:] == (10, pin)
    assert len(peer_calls) == 1
    keyid, sig, ts = worker._parse_http_auth_header(
        pillar_calls[0][0].get_header("Authorization"))
    assert keyid == _NODE_ID
    pub = worker._wauth.public_key_hex_from_private(_PRIV)
    assert worker._wauth.verify_request(
        pub, "GET", "/files?model=model", b"", ts, sig)
    assert not dest.exists()
    assert not Path(str(dest) + ".tmp").exists()
    assert streamed == {}


@pytest.mark.parametrize("bad_sha", [None, "", "abc", "z" * 64])
def test_remote_slice_missing_or_malformed_pillar_hash_is_refused_before_fetch(
        monkeypatch, tmp_path, bad_sha):
    pillar_calls = []
    peer_calls = []

    def fake_pillar_open(req, *, timeout, expected_spki_hash=None):
        pillar_calls.append((req, timeout, expected_spki_hash))
        return _pillar_response(bad_sha)

    monkeypatch.setattr(worker._wauth, "open_pillar_url", fake_pillar_open)
    monkeypatch.setattr(
        worker.urlrequest, "urlopen",
        lambda *args, **kwargs: peer_calls.append((args, kwargs)),
    )
    # The retired environment escape hatch must not change this result.
    monkeypatch.setenv("STHAMBHA_REFUSE_UNVERIFIED_FETCH", "false")

    with pytest.raises(RuntimeError, match="valid pillar-attested model_sha256"):
        worker.fetch_sub_gguf_from_peer(
            "https://pillar.example", "model", 0, 1,
            str(tmp_path / "slice.gguf"),
            own_node_id=_NODE_ID, priv_key=_PRIV, spki_hash="a" * 64,
        )

    assert len(pillar_calls) == 1
    assert peer_calls == []


def test_streamed_remote_slice_hash_is_returned_for_registration(
        monkeypatch, tmp_path):
    payload = b"verified model bytes"
    expected = worker.hashlib.sha256(payload).hexdigest()
    pillar_calls = []

    def fake_pillar_open(req, *, timeout, expected_spki_hash=None):
        pillar_calls.append((req, timeout, expected_spki_hash))
        return _pillar_response(expected)

    def fake_urlopen(url, timeout):
        return _Response(payload, {"Content-Length": str(len(payload))})

    monkeypatch.setattr(worker._wauth, "open_pillar_url", fake_pillar_open)
    monkeypatch.setattr(worker.urlrequest, "urlopen", fake_urlopen)
    dest = tmp_path / "slice.gguf"
    streamed = {}

    assert worker.fetch_sub_gguf_from_peer(
        "https://pillar.example", "model", 0, 1, str(dest),
        own_node_id=_NODE_ID, priv_key=_PRIV,
        spki_hash="a" * 64,
        verified_sha_out=streamed,
    ) == str(dest)
    assert dest.read_bytes() == payload
    assert streamed == {"sha256": expected}


def test_remote_slice_refuses_unsigned_pillar_query_before_network(
        monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        worker._wauth, "open_pillar_url",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        worker.urlrequest, "urlopen",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="requires the worker key and node id"):
        worker.fetch_sub_gguf_from_peer(
            "https://pillar.example", "model", 0, 1,
            str(tmp_path / "slice.gguf"), own_node_id=_NODE_ID,
        )

    assert calls == []


def test_remote_slice_https_pillar_uses_shared_pinned_no_redirect_transport(
        monkeypatch, tmp_path):
    payload = b"verified model bytes"
    expected = worker.hashlib.sha256(payload).hexdigest()
    observed = []

    def fake_pillar_open(req, *, timeout, expected_spki_hash=None):
        observed.append((req, timeout, expected_spki_hash))
        return _pillar_response(expected)

    def fake_urlopen(url, timeout):
        return _Response(payload, {"Content-Length": str(len(payload))})

    monkeypatch.setattr(worker._wauth, "open_pillar_url", fake_pillar_open)
    monkeypatch.setattr(worker.urlrequest, "urlopen", fake_urlopen)
    pin = "a" * 64

    worker.fetch_sub_gguf_from_peer(
        "https://pillar.example", "model", 0, 1,
        str(tmp_path / "slice.gguf"), own_node_id=_NODE_ID,
        priv_key=_PRIV, spki_hash=pin,
    )

    assert len(observed) == 1
    request, timeout, observed_pin = observed[0]
    assert isinstance(request, worker.urlrequest.Request)
    assert request.full_url == "https://pillar.example/files?model=model"
    assert timeout == 10
    assert observed_pin == pin


def test_remote_slice_refuses_unpinned_https_before_files_query(
        monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        worker._wauth, "open_pillar_url",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="HTTPS PILLAR_URL requires"):
        worker.fetch_sub_gguf_from_peer(
            "https://pillar.example", "model", 0, 1,
            str(tmp_path / "slice.gguf"),
            own_node_id=_NODE_ID, priv_key=_PRIV,
        )

    assert calls == []


def test_remote_slice_refuses_http_outside_trusted_overlay_before_files_query(
        monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        worker._wauth, "open_pillar_url",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="outside loopback/TRUSTED_PILLAR_NETS"):
        worker.fetch_sub_gguf_from_peer(
            "http://198.51.100.10:7777", "model", 0, 1,
            str(tmp_path / "slice.gguf"),
            own_node_id=_NODE_ID, priv_key=_PRIV,
        )

    assert calls == []


def test_remote_slice_accepts_trusted_overlay_http_for_files_query(
        monkeypatch, tmp_path):
    payload = b"verified model bytes"
    expected = worker.hashlib.sha256(payload).hexdigest()
    observed = []

    def fake_pillar_open(req, *, timeout, expected_spki_hash=None):
        observed.append((req.full_url, timeout, expected_spki_hash))
        return _pillar_response(expected)

    monkeypatch.setattr(worker._wauth, "open_pillar_url", fake_pillar_open)
    monkeypatch.setattr(
        worker.urlrequest, "urlopen",
        lambda url, timeout: _Response(
            payload, {"Content-Length": str(len(payload))}),
    )

    worker.fetch_sub_gguf_from_peer(
        "http://10.42.0.9:7777", "model", 0, 1,
        str(tmp_path / "slice.gguf"),
        own_node_id=_NODE_ID, priv_key=_PRIV,
    )

    assert observed == [
        ("http://10.42.0.9:7777/files?model=model", 10, None),
    ]


def test_skip_sha256_local_file_requires_explicit_dev_mode(tmp_path):
    local_package = str(tmp_path / "package")
    with pytest.raises(ValueError, match="requires explicit --local-dev"):
        worker.validate_local_dev_fetch_policy(
            local_dev=False, skip_sha256=True,
            package_url=local_package, auto_fetch=False,
        )

    worker.validate_local_dev_fetch_policy(
        local_dev=True, skip_sha256=True,
        package_url=local_package, auto_fetch=False,
    )


@pytest.mark.parametrize("source", [
    "https://models.example/package",
    "http://10.0.0.8/package",
    r"\\server\share\package",
    "//server/share/package",
])
def test_local_dev_refuses_non_loopback_package_url(source):
    with pytest.raises(ValueError, match="local path"):
        worker.validate_local_dev_fetch_policy(
            local_dev=True, skip_sha256=True,
            package_url=source, auto_fetch=False,
        )


@pytest.mark.parametrize("source", [
    "file:///tmp/package",
    "http://localhost:8080/package",
    "http://127.0.0.1:8080/package",
    "https://[::1]/package",
])
def test_local_dev_accepts_only_file_or_loopback_package_source(source):
    worker.validate_local_dev_fetch_policy(
        local_dev=True, skip_sha256=True,
        package_url=source, auto_fetch=False,
    )
