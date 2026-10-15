"""Consumer-only tokenizer provisioning from a signed layer package."""
from __future__ import annotations

import fcntl
import hashlib
import json
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "release"))

import client  # noqa: E402
import gateway_tokenizer  # noqa: E402
import nakshatra_serve as serve  # noqa: E402
import release_trust  # noqa: E402
import releasekit  # noqa: E402
from packaging import fetch_package  # noqa: E402
from packaging.nakshatra_package import (  # noqa: E402
    Artifact, KIND_SHARED, NakshatraPackage, PackageError, ROLE_METADATA,
)


class _RecordingHandler(SimpleHTTPRequestHandler):
    requests: list[str] = []

    def do_GET(self):
        type(self).requests.append(self.path)
        super().do_GET()

    def log_message(self, _format, *_args):
        pass


@pytest.fixture
def package_server(tmp_path):
    package = tmp_path / "package"
    metadata = package / "shared" / "metadata.gguf"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"tokenizer metadata only")
    artifact = Artifact(
        path="shared/metadata.gguf",
        sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        size=metadata.stat().st_size,
        kind=KIND_SHARED,
        role=ROLE_METADATA,
    )
    pkg = NakshatraPackage(
        model_id="llama32-3b", arch="llama", n_layers=28,
        artifacts=[artifact])
    private = ed25519.Ed25519PrivateKey.generate()
    pkg.sign(private.private_bytes_raw())
    (package / "package.json").write_text(pkg.to_json())

    _RecordingHandler.requests = []
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_RecordingHandler, directory=str(package)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "url": f"http://127.0.0.1:{server.server_port}",
            "pubkey": pkg.signer_pubkey_hex,
            "package": package,
            "requests": _RecordingHandler.requests,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_auto_discovery_fetches_only_verified_metadata(package_server, tmp_path, monkeypatch):
    tokenizer_dir = tmp_path / "home" / ".nakshatra" / "gateway" / "models"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NAKSHATRA_TOKENIZER_DIR", str(tokenizer_dir))
    monkeypatch.setenv("PACKAGE_URL", package_server["url"])
    monkeypatch.setattr(
        gateway_tokenizer, "load_release_package_signers",
        lambda: frozenset({package_server["pubkey"]}))
    monkeypatch.setattr(client, "_fetch_registry_peers", lambda _url: [{
        "is_online": True,
        "layer_offerings": [{
            "model_id": "llama32-3b", "layer_start": 0, "layer_end": 28,
        }],
    }])

    models = serve._discover_auto_models("https://pillar.example:7778")

    dest = gateway_tokenizer.tokenizer_path("llama32-3b", tokenizer_dir)
    provenance = gateway_tokenizer.tokenizer_manifest_path("llama32-3b", tokenizer_dir)
    assert models["llama32-3b"].tokenizer_gguf == str(dest)
    assert dest.read_bytes() == b"tokenizer metadata only"
    assert dest.stat().st_mode & 0o777 == 0o644
    cached_manifest = json.loads(provenance.read_text())
    assert cached_manifest["signer_pubkey_hex"] == package_server["pubkey"]
    assert cached_manifest["signature_b64"]
    assert package_server["requests"] == ["/package.json", "/shared/metadata.gguf"]


def test_tampered_metadata_is_refused_and_not_installed(package_server, tmp_path):
    (package_server["package"] / "shared" / "metadata.gguf").write_bytes(b"tampered")
    root = tmp_path / "models"
    root.mkdir()
    dest = gateway_tokenizer.tokenizer_path("llama32-3b", root)
    dest.write_bytes(b"previous verified tokenizer")

    with pytest.raises(PackageError, match="SHA-256 mismatch"):
        gateway_tokenizer.fetch_model_tokenizer(
            "llama32-3b", package_url=package_server["url"],
            release_pubkeys={package_server["pubkey"]}, root=root)

    assert dest.read_bytes() == b"previous verified tokenizer"


def test_untrusted_manifest_signer_is_refused(package_server, tmp_path):
    with pytest.raises(PackageError, match="untrusted key"):
        gateway_tokenizer.fetch_model_tokenizer(
            "llama32-3b", package_url=package_server["url"],
            release_pubkeys={"89" * 32}, root=tmp_path / "models")


def test_tampered_cached_metadata_is_never_advertised(package_server, tmp_path, monkeypatch):
    root = tmp_path / "models"
    gateway_tokenizer.fetch_model_tokenizer(
        "llama32-3b", package_url=package_server["url"],
        release_pubkeys={package_server["pubkey"]}, root=root)
    cached = gateway_tokenizer.tokenizer_path("llama32-3b", root)
    cached.write_bytes(b"tampered cache")

    with pytest.raises(PackageError, match="cached tokenizer (size|SHA-256) mismatch"):
        gateway_tokenizer.verify_model_tokenizer(
            "llama32-3b", release_pubkeys={package_server["pubkey"]}, root=root)

    monkeypatch.setenv("NAKSHATRA_TOKENIZER_DIR", str(root))
    monkeypatch.delenv("PACKAGE_URL", raising=False)
    monkeypatch.setattr(
        gateway_tokenizer, "load_release_package_signers",
        lambda: frozenset({package_server["pubkey"]}))
    assert serve._local_tokenizers({"llama32-3b"}) == {}


def test_cached_tokenizer_verification_follows_release_switch(
        package_server, tmp_path, monkeypatch):
    prefix = tmp_path / "node"
    release_key = ed25519.Ed25519PrivateKey.generate()
    private_hex = release_key.private_bytes_raw().hex()
    public_hex = release_key.public_key().public_bytes_raw().hex()

    def write_release(version: str) -> None:
        release_dir = prefix / "releases" / version
        release_dir.mkdir(parents=True)
        manifest = releasekit.sign({
            "schema": "nak-release/1",
            "version": version,
            "package_signers": {
                "schema": "nak-package-signers/1",
                "keys": [package_server["pubkey"]],
            },
        }, private_hex)
        (release_dir / "manifest.json").write_text(json.dumps(manifest))

    write_release("1.2.3")
    (prefix / "current").symlink_to("releases/1.2.3")
    (prefix / "config.json").write_text(json.dumps({
        "current": "1.2.3", "pubkey": public_hex,
    }))
    monkeypatch.setenv("NAK_NODE_PREFIX", str(prefix))
    root = tmp_path / "models"
    gateway_tokenizer.fetch_model_tokenizer(
        "llama32-3b", package_url=package_server["url"], root=root)

    write_release("1.2.4")
    state = prefix / "state"
    state.mkdir()
    transition = state / release_trust.TRANSITION_FILE
    transition.write_text(json.dumps({
        "schema": release_trust.TRANSITION_SCHEMA,
        "pubkey": public_hex,
        "from": "1.2.3",
        "to": "1.2.4",
    }))
    transition.chmod(0o600)
    next_current = prefix / "current.next"
    next_current.symlink_to("releases/1.2.4")
    next_current.replace(prefix / "current")

    # Services verify their existing tokenizer cache after `current` switches
    # but before install.py commits the new version to config.json. The installer
    # remains live and holds the same prefix flock throughout this window.
    with (prefix / ".install.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            assert gateway_tokenizer.verify_model_tokenizer(
                "llama32-3b", root=root) == str(
                    gateway_tokenizer.tokenizer_path("llama32-3b", root))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def test_model_ids_have_collision_free_cache_names(tmp_path):
    assert gateway_tokenizer.tokenizer_path("a/b", tmp_path) != \
        gateway_tokenizer.tokenizer_path("a_b", tmp_path)


def test_release_pin_can_only_be_narrowed_to_a_trusted_key(package_server, tmp_path):
    with pytest.raises(PackageError, match="may only narrow"):
        gateway_tokenizer.fetch_model_tokenizer(
            "llama32-3b", package_url=package_server["url"],
            trusted_pubkey="89" * 32,
            release_pubkeys={package_server["pubkey"]}, root=tmp_path)


def test_package_redirect_is_refused(package_server, tmp_path):
    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", package_server["url"] + "/package.json")
            self.end_headers()

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(PackageError, match="refused redirect"):
            gateway_tokenizer.fetch_model_tokenizer(
                "llama32-3b", package_url=f"http://127.0.0.1:{server.server_port}",
                release_pubkeys={package_server["pubkey"]}, root=tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_oversize_artifact_aborts_after_first_extra_byte(tmp_path, monkeypatch):
    class OversizeResponse:
        headers = {}

        def __init__(self):
            self.remaining = b"x" * 1000
            self.bytes_read = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size):
            chunk, self.remaining = self.remaining[:size], self.remaining[size:]
            self.bytes_read += len(chunk)
            return chunk

    response = OversizeResponse()
    monkeypatch.setattr(fetch_package, "_open_url", lambda *_args, **_kwargs: response)
    dest = tmp_path / "oversize.gguf"

    with pytest.raises(PackageError, match="exceeds signed size"):
        fetch_package._fetch_one(
            "https://packages.example/", "shared/metadata.gguf", dest,
            expected_size=4)

    assert response.bytes_read == 5
    assert dest.read_bytes() == b""
