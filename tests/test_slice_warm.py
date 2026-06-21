"""slice_warm — page-cache warming of GGUF slices, tested without GPU/network."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import slice_warm as sw


def _mk(tmp_path, name="slice.gguf", mb=4):
    p = tmp_path / name
    p.write_bytes(os.urandom(mb << 20))
    return str(p)


def test_warm_reads_whole_file(tmp_path):
    p = _mk(tmp_path, mb=4)
    r = sw.warm_path(p)
    assert r["ok"] is True
    assert r["bytes"] == 4 << 20
    assert r["error"] is None


def test_warm_makes_file_resident(tmp_path):
    p = _mk(tmp_path, mb=8)
    # evict best-effort isn't available unprivileged; just assert warm -> resident
    sw.warm_path(p)
    frac = sw.resident_fraction(p)
    # mincore may be unavailable on some kernels (-1.0); otherwise should be hot
    assert frac == -1.0 or frac > 0.9


def test_missing_file_does_not_raise(tmp_path):
    r = sw.warm_path(str(tmp_path / "nope.gguf"))
    assert r["ok"] is False
    assert r["error"] is not None


def test_warm_paths_skips_empty_and_returns_per_path(tmp_path):
    a = _mk(tmp_path, "a.gguf", 2)
    b = _mk(tmp_path, "b.gguf", 2)
    out = sw.warm_paths([a, "", b, None])
    assert len(out) == 2
    assert all(r["ok"] for r in out)


def test_mlock_is_best_effort(tmp_path):
    # mlock may fail under RLIMIT_MEMLOCK when unprivileged — must still warm ok.
    p = _mk(tmp_path, mb=2)
    r = sw.warm_path(p, mlock=True)
    assert r["ok"] is True
    assert isinstance(r["mlocked"], bool)   # True if allowed, False if not — never raises


def test_zero_byte_file(tmp_path):
    p = tmp_path / "empty.gguf"
    p.write_bytes(b"")
    r = sw.warm_path(str(p))
    assert r["ok"] is True and r["bytes"] == 0
