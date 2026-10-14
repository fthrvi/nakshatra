"""Consumer-only tokenizer provisioning from a signed layer package."""
from __future__ import annotations

import hashlib
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import client  # noqa: E402
import gateway_tokenizer  # noqa: E402
import nakshatra_serve as serve  # noqa: E402
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
    monkeypatch.setenv("PACKAGE_TRUSTED_PUBKEY", package_server["pubkey"])
    monkeypatch.setattr(client, "_fetch_registry_peers", lambda _url: [{
        "is_online": True,
        "layer_offerings": [{
            "model_id": "llama32-3b", "layer_start": 0, "layer_end": 28,
        }],
    }])

    models = serve._discover_auto_models("https://pillar.example:7778")

    dest = tokenizer_dir / "llama32-3b.metadata.gguf"
    assert models["llama32-3b"].tokenizer_gguf == str(dest)
    assert dest.read_bytes() == b"tokenizer metadata only"
    assert dest.stat().st_mode & 0o777 == 0o644
    assert package_server["requests"] == ["/package.json", "/shared/metadata.gguf"]


def test_tampered_metadata_is_refused_and_not_installed(package_server, tmp_path):
    (package_server["package"] / "shared" / "metadata.gguf").write_bytes(b"tampered")
    root = tmp_path / "models"
    root.mkdir()
    dest = root / "llama32-3b.metadata.gguf"
    dest.write_bytes(b"previous verified tokenizer")

    with pytest.raises(PackageError, match="SHA-256 mismatch"):
        gateway_tokenizer.fetch_model_tokenizer(
            "llama32-3b", package_url=package_server["url"],
            trusted_pubkey=package_server["pubkey"], root=root)

    assert dest.read_bytes() == b"previous verified tokenizer"


def test_untrusted_manifest_signer_is_refused(package_server, tmp_path):
    with pytest.raises(PackageError, match="untrusted key"):
        gateway_tokenizer.fetch_model_tokenizer(
            "llama32-3b", package_url=package_server["url"],
            trusted_pubkey="89" * 32, root=tmp_path / "models")
