"""Tests for v1.0 §5 layer-package weight distribution (P2).

Covers the schema (sign/verify/tamper/path-safety/range-selection) and the full
package → verified-fetch → assemble round-trip against a tiny synthetic GGUF,
asserting tensor+KV parity with experiments/v0.0/partial_gguf.py and fail-closed
behaviour on SHA mismatch / bad signature / untrusted signer.

Self-contained: builds its own ~few-KB GGUF fixture, no external model needed.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from gguf import GGUFReader, GGUFWriter  # noqa: E402
from packaging.nakshatra_package import (  # noqa: E402
    Artifact, NakshatraPackage, new_package, PackageError,
    KIND_SHARED, KIND_LAYER, ROLE_METADATA, ROLE_EMBEDDINGS, ROLE_HEAD)
from packaging.fetch_package import fetch_and_assemble  # noqa: E402
from nakshatra_auth import generate_keypair  # noqa: E402

PARTIAL_GGUF = _REPO / "experiments" / "v0.0" / "partial_gguf.py"
N_LAYERS = 4


# ── fixtures ──────────────────────────────────────────────────────────

def _make_source_gguf(path: Path) -> None:
    """A tiny but structurally real llama GGUF: block_count KV + token_embd,
    rope_freqs, output head, and 2 tensors per block. Enough to exercise the
    packager + partial_gguf.py."""
    w = GGUFWriter(str(path), arch="llama")
    w.add_uint32("llama.block_count", N_LAYERS)
    w.add_uint32("llama.context_length", 128)
    w.add_uint32("llama.embedding_length", 8)
    rng = np.arange(64, dtype=np.float32)
    w.add_tensor("token_embd.weight", rng.reshape(8, 8).copy())
    w.add_tensor("rope_freqs.weight", np.arange(4, dtype=np.float32))
    w.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))
    w.add_tensor("output.weight", (rng + 100).reshape(8, 8).copy())
    for i in range(N_LAYERS):
        w.add_tensor(f"blk.{i}.attn_norm.weight", np.full(8, i + 1, dtype=np.float32))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", np.full(8, (i + 1) * 10, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture
def source_gguf(tmp_path):
    p = tmp_path / "src.gguf"
    _make_source_gguf(p)
    return p


@pytest.fixture
def package_dir(source_gguf, tmp_path):
    from packaging.package_gguf import package_gguf
    out = tmp_path / "pkg"
    package_gguf(source_gguf, out, "test-tiny")
    return out


def _tensor_bytes(path) -> dict:
    return {t.name: bytes(t.data.tobytes()) for t in GGUFReader(str(path)).tensors}


def _nks_kvs(path) -> dict:
    out = {}
    for f in GGUFReader(str(path)).fields.values():
        if f.name.startswith("nakshatra."):
            v = f.parts[f.data[0]]
            out[f.name] = v.tolist()[0] if hasattr(v, "tolist") else v[0]
    return out


# ── schema unit tests ─────────────────────────────────────────────────

def _art(path, kind, idx=None, role=None, body=b"x"):
    return Artifact(path, hashlib.sha256(body).hexdigest(), len(body), kind, idx, role)


def _toy_pkg():
    arts = [_art("shared/metadata.gguf", KIND_SHARED, role=ROLE_METADATA, body=b"m"),
            _art("shared/embeddings.gguf", KIND_SHARED, role=ROLE_EMBEDDINGS, body=b"e"),
            _art("shared/output.gguf", KIND_SHARED, role=ROLE_HEAD, body=b"o")]
    arts += [_art(f"layers/layer-{i:03d}.gguf", KIND_LAYER, idx=i, body=bytes([i])) for i in range(4)]
    return new_package("toy", "llama", 4, arts)


def test_sign_verify_roundtrip():
    pkg = _toy_pkg()
    priv, _ = generate_keypair()
    pkg.sign(priv)
    assert pkg.verify_signature()
    pkg2 = NakshatraPackage.from_json(pkg.to_json())
    pkg2.validate()
    assert pkg2.verify_signature()
    assert pkg2.revision == pkg.revision


def test_tamper_breaks_revision():
    pkg = _toy_pkg()
    bad = NakshatraPackage.from_json(pkg.to_json())
    bad.artifacts[-1].sha256 = "0" * 64
    with pytest.raises(PackageError, match="revision mismatch"):
        bad.validate()


def test_bad_signature_fails_verify():
    pkg = _toy_pkg()
    priv, _ = generate_keypair()
    pkg.sign(priv)
    bad = NakshatraPackage.from_json(pkg.to_json())
    bad.signature_b64 = "A" * 88
    assert not bad.verify_signature()


@pytest.mark.parametrize("bad", ["/abs.gguf", "../esc.gguf", "a/../b.gguf", "a\\b.gguf", "a//b.gguf"])
def test_unsafe_paths_rejected(bad):
    with pytest.raises(PackageError):
        Artifact(bad, "0" * 64, 1, KIND_SHARED, role=ROLE_METADATA).validate()


def test_position_aware_range_selection():
    pkg = _toy_pkg()
    first = {a.path for a in pkg.artifacts_for_range(0, 2)}
    mid = {a.path for a in pkg.artifacts_for_range(1, 3)}
    last = {a.path for a in pkg.artifacts_for_range(2, 4)}
    assert "shared/embeddings.gguf" in first and "shared/output.gguf" not in first
    assert "shared/embeddings.gguf" not in mid and "shared/output.gguf" not in mid
    assert "shared/output.gguf" in last and "shared/embeddings.gguf" not in last
    assert "shared/metadata.gguf" in first & mid & last  # always


# ── round-trip / parity ───────────────────────────────────────────────

def test_whole_assembly_matches_source(package_dir, source_gguf, tmp_path):
    dest = tmp_path / "whole.gguf"
    fetch_and_assemble(str(package_dir), 0, N_LAYERS, str(dest))
    assert _tensor_bytes(dest) == _tensor_bytes(source_gguf)


@pytest.mark.parametrize("start,end", [(0, 2), (2, 4), (1, 3), (0, N_LAYERS)])
def test_range_assembly_matches_partial_gguf(package_dir, source_gguf, tmp_path, start, end):
    asm = tmp_path / f"asm_{start}_{end}.gguf"
    pg = tmp_path / f"pg_{start}_{end}.gguf"
    fetch_and_assemble(str(package_dir), start, end, str(asm))
    subprocess.run([sys.executable, str(PARTIAL_GGUF), str(source_gguf), str(pg),
                    "--start", str(start), "--end", str(end)],
                   check=True, capture_output=True)
    assert _tensor_bytes(asm) == _tensor_bytes(pg)
    assert _nks_kvs(asm) == _nks_kvs(pg)


def test_sha_mismatch_fails_closed(package_dir, tmp_path):
    # Corrupt a layer fragment on disk; fetch must refuse.
    frag = package_dir / "layers" / "layer-001.gguf"
    data = bytearray(frag.read_bytes())
    data[-1] ^= 0xFF
    frag.write_bytes(data)
    with pytest.raises(PackageError, match="SHA-256 mismatch"):
        fetch_and_assemble(str(package_dir), 0, N_LAYERS, str(tmp_path / "x.gguf"))


def test_require_signature_refuses_unsigned(package_dir, tmp_path):
    with pytest.raises(PackageError, match="unsigned"):
        fetch_and_assemble(str(package_dir), 0, 2, str(tmp_path / "x.gguf"),
                           require_signature=True)


def _make_tied_source_gguf(path: Path) -> None:
    """A weight-TIED model: token_embd present, NO separate output.weight (the
    output projection is tied to token_embd). Like every small Llama (1B/3B)."""
    w = GGUFWriter(str(path), arch="llama")
    w.add_uint32("llama.block_count", N_LAYERS)
    w.add_tensor("token_embd.weight", np.arange(64, dtype=np.float32).reshape(8, 8).copy())
    w.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))   # NO output.weight
    for i in range(N_LAYERS):
        w.add_tensor(f"blk.{i}.attn_norm.weight", np.full(8, i + 1, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def test_tied_model_last_slice_includes_token_embd(tmp_path):
    """Regression for the weight-tied bug §10 surfaced: the LAST worker of a tied
    model needs token_embd (the tied output weights), and the whole-model slice
    must not duplicate it."""
    from packaging.package_gguf import package_gguf
    src = tmp_path / "tied.gguf"
    _make_tied_source_gguf(src)
    pkg_dir = tmp_path / "pkg"
    package_gguf(src, pkg_dir, "tied")
    pkg = NakshatraPackage.from_json((pkg_dir / "package.json").read_text())
    assert pkg.tied_embeddings is True

    def names(path):
        return [t.name for t in GGUFReader(str(path)).tensors]

    # last slice [2,4) (has lm_head, NOT first) must carry token_embd for the tied output
    last = tmp_path / "last.gguf"
    fetch_and_assemble(str(pkg_dir), 2, N_LAYERS, str(last))
    assert "token_embd.weight" in names(last)

    # middle slice [1,3) (neither first nor last) must NOT carry it
    mid = tmp_path / "mid.gguf"
    fetch_and_assemble(str(pkg_dir), 1, 3, str(mid))
    assert "token_embd.weight" not in names(mid)

    # whole model [0,N) carries token_embd exactly ONCE (no duplicate)
    whole = tmp_path / "whole.gguf"
    fetch_and_assemble(str(pkg_dir), 0, N_LAYERS, str(whole))
    assert names(whole).count("token_embd.weight") == 1


def test_untrusted_signer_refused(source_gguf, tmp_path):
    from packaging.package_gguf import package_gguf
    priv, pub = generate_keypair()
    out = tmp_path / "signed_pkg"
    package_gguf(source_gguf, out, "test-tiny", sign_priv=priv)
    # correct signer accepted
    fetch_and_assemble(str(out), 0, 2, str(tmp_path / "ok.gguf"),
                       require_signature=True, trusted_pubkeys={pub})
    # a different key is rejected
    _, other_pub = generate_keypair()
    with pytest.raises(PackageError, match="untrusted"):
        fetch_and_assemble(str(out), 0, 2, str(tmp_path / "no.gguf"),
                           require_signature=True, trusted_pubkeys={other_pub})
