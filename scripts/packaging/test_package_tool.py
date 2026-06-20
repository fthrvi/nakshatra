"""Tests for package_tool.py — validate + certify a layer-package. Builds a real on-disk package
(no network/GPU). Run: python3 -m pytest scripts/packaging/test_package_tool.py -q"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # packaging/ on path

from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402

import package_tool as pt  # noqa: E402
from nakshatra_package import (  # noqa: E402
    Artifact, NakshatraPackage, KIND_LAYER, KIND_SHARED,
    ROLE_METADATA, ROLE_EMBEDDINGS, ROLE_HEAD,
)


def _write(root: Path, rel: str, data: bytes) -> Artifact:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return Artifact(path=rel, sha256=hashlib.sha256(data).hexdigest(), size=len(data),
                    kind=KIND_LAYER if rel.startswith("layers/") else KIND_SHARED)


def _build(root: Path, n_layers=2):
    arts = []
    a = _write(root, "shared/metadata.gguf", b"META"); a.role = ROLE_METADATA; arts.append(a)
    a = _write(root, "shared/embeddings.gguf", b"EMBED"); a.role = ROLE_EMBEDDINGS; arts.append(a)
    a = _write(root, "shared/head.gguf", b"HEAD"); a.role = ROLE_HEAD; arts.append(a)
    for i in range(n_layers):
        a = _write(root, f"layers/layer-{i}.gguf", b"L" * (i + 3)); a.layer_idx = i; arts.append(a)
    return NakshatraPackage(model_id="test-model", arch="llama", n_layers=n_layers, artifacts=arts)


def test_valid_package_structural_and_ondisk(tmp_path):
    pkg = _build(tmp_path)
    assert pt.validate_package(pkg) == []                       # structural-only
    assert pt.validate_package(pkg, tmp_path) == []             # + on-disk bytes


def test_detects_size_and_sha_drift(tmp_path):
    pkg = _build(tmp_path)
    (tmp_path / "layers/layer-0.gguf").write_bytes(b"TAMPERED-LONGER")   # bytes drift from manifest
    probs = pt.validate_package(pkg, tmp_path)
    assert any("size mismatch" in p for p in probs)
    assert any("sha256 mismatch" in p for p in probs)


def test_detects_missing_file(tmp_path):
    pkg = _build(tmp_path)
    (tmp_path / "layers/layer-1.gguf").unlink()
    assert any("file not found" in p for p in pt.validate_package(pkg, tmp_path))


def test_detects_missing_layer_artifact(tmp_path):
    pkg = _build(tmp_path)
    pkg.artifacts = [a for a in pkg.artifacts if not (a.kind == KIND_LAYER and a.layer_idx == 1)]
    assert any("missing 1 layer" in p for p in pt.validate_package(pkg))


def test_detects_missing_shared_role(tmp_path):
    pkg = _build(tmp_path)
    pkg.artifacts = [a for a in pkg.artifacts if a.role != ROLE_HEAD]
    assert any("missing required shared artifact role: 'head'" in p for p in pt.validate_package(pkg))


def test_detects_duplicate_path(tmp_path):
    pkg = _build(tmp_path)
    dup = pkg.artifacts[-1]
    pkg.artifacts.append(Artifact(path=dup.path, sha256=dup.sha256, size=dup.size,
                                  kind=KIND_LAYER, layer_idx=dup.layer_idx))
    assert any("duplicate artifact path" in p for p in pt.validate_package(pkg))


def test_certify_signs_a_valid_package(tmp_path):
    pkg = _build(tmp_path)
    # write manifest + a key file, then run the CLI
    man = tmp_path / "model-package.json"; man.write_text(pkg.to_json())
    priv = ed25519.Ed25519PrivateKey.generate().private_bytes_raw().hex()
    key = tmp_path / "op.key"; key.write_text(priv)
    rc = pt.main(["certify", str(man), "--key", str(key), "--root", str(tmp_path)])
    assert rc == 0
    signed = NakshatraPackage.from_json(man.read_text())
    assert signed.signature_b64 and signed.verify_signature()


def test_certify_refuses_invalid_package(tmp_path):
    pkg = _build(tmp_path)
    (tmp_path / "layers/layer-0.gguf").unlink()                 # break on-disk integrity
    man = tmp_path / "model-package.json"; man.write_text(pkg.to_json())
    priv = ed25519.Ed25519PrivateKey.generate().private_bytes_raw().hex()
    key = tmp_path / "op.key"; key.write_text(priv)
    rc = pt.main(["certify", str(man), "--key", str(key), "--root", str(tmp_path)])
    assert rc == 1                                              # refused
    assert not NakshatraPackage.from_json(man.read_text()).signature_b64   # left unsigned


def test_validate_cli_exit_codes(tmp_path):
    pkg = _build(tmp_path)
    man = tmp_path / "m.json"; man.write_text(pkg.to_json())
    assert pt.main(["validate", str(man), "--root", str(tmp_path)]) == 0
    assert pt.main(["validate", str(man), "--require-signature"]) == 1     # unsigned + required
