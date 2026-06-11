"""Worker auto-provisioning from a layer package (v1.0 §5 / P2 integration).

Guarded: worker.py imports grpc at module load, so this only runs where grpc is
installed (the cluster/CI venv). It exercises the actual worker entry point
worker.provision_from_package against a real package built from a synthetic GGUF,
asserting a loader-ready sub-GGUF lands with the right nakshatra.layer_range_*
metadata and fail-closed SHA verification.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

pytest.importorskip("grpc", reason="worker.py imports grpc at module load")

from gguf import GGUFReader, GGUFWriter  # noqa: E402
import worker  # noqa: E402
from packaging.package_gguf import package_gguf  # noqa: E402
from packaging.nakshatra_package import PackageError  # noqa: E402

N_LAYERS = 4


def _make_source_gguf(path: Path) -> None:
    w = GGUFWriter(str(path), arch="llama")
    w.add_uint32("llama.block_count", N_LAYERS)
    w.add_tensor("token_embd.weight", np.arange(64, dtype=np.float32).reshape(8, 8).copy())
    w.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))
    w.add_tensor("output.weight", np.arange(64, dtype=np.float32).reshape(8, 8).copy())
    for i in range(N_LAYERS):
        w.add_tensor(f"blk.{i}.attn_norm.weight", np.full(8, i + 1, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


@pytest.fixture
def package_dir(tmp_path):
    src = tmp_path / "src.gguf"
    _make_source_gguf(src)
    out = tmp_path / "pkg"
    package_gguf(src, out, "test-tiny")
    return out


def _nks(path):
    out = {}
    for f in GGUFReader(str(path)).fields.values():
        if f.name.startswith("nakshatra."):
            v = f.parts[f.data[0]]
            out[f.name] = v.tolist()[0] if hasattr(v, "tolist") else v[0]
    return out


def test_worker_provisions_middle_range(package_dir, tmp_path):
    dest = tmp_path / "slice.gguf"
    worker.provision_from_package(str(package_dir), 1, 3, str(dest))
    assert dest.exists()
    nks = _nks(dest)
    assert nks["nakshatra.layer_range_start"] == 1
    assert nks["nakshatra.layer_range_end"] == 3
    assert nks["nakshatra.has_token_embd"] in (0, False)   # middle worker
    assert nks["nakshatra.has_lm_head"] in (0, False)


def test_worker_provision_fail_closed_on_corruption(package_dir, tmp_path):
    frag = package_dir / "layers" / "layer-001.gguf"
    data = bytearray(frag.read_bytes()); data[-1] ^= 0xFF
    frag.write_bytes(data)
    with pytest.raises(PackageError, match="SHA-256 mismatch"):
        worker.provision_from_package(str(package_dir), 0, N_LAYERS, str(tmp_path / "x.gguf"))
