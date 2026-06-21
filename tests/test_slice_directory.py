"""slice_directory — TTL who-holds-what directory, tested with a fake clock."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import slice_directory as sd
import slice_fetch as sf


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t
    def __call__(self):
        return self.t


def test_publish_and_lookup():
    clk = _Clock()
    d = sd.SliceDirectory(ttl_s=100, now_fn=clk)
    d.publish("nodeA:8077", ["m@h-L0-16.gguf", "m@h-L16-32.gguf"])
    assert d.holders("m@h-L0-16.gguf") == ["nodeA:8077"]
    assert d.holders("nonexistent.gguf") == []


def test_ttl_expiry_evicts_dead_node():
    clk = _Clock()
    d = sd.SliceDirectory(ttl_s=100, now_fn=clk)
    d.publish("nodeA:8077", ["m@h-L0-16.gguf"])
    clk.t += 50
    assert d.holders("m@h-L0-16.gguf") == ["nodeA:8077"]   # still fresh
    clk.t += 60                                            # now 110 > ttl 100
    assert d.holders("m@h-L0-16.gguf") == []               # evicted


def test_freshest_holder_first():
    clk = _Clock()
    d = sd.SliceDirectory(ttl_s=1000, now_fn=clk)
    d.publish("old:8077", ["m@h-L0-16.gguf"])
    clk.t += 10
    d.publish("new:8077", ["m@h-L0-16.gguf"])
    assert d.holders("m@h-L0-16.gguf") == ["new:8077", "old:8077"]


def test_to_holders_fn_plugs_into_slice_fetch():
    clk = _Clock()
    d = sd.SliceDirectory(ttl_s=1000, now_fn=clk)
    ref = sf.SliceRef("m", "h", 0, 16)            # filename m@h-L0-16.gguf
    d.publish("peerX:8077", [ref.filename])
    holders_fn = d.to_holders_fn()
    assert holders_fn(ref) == ["peerX:8077"]


def test_persistence_roundtrip(tmp_path):
    p = str(tmp_path / "dir.json")
    clk = _Clock()
    d1 = sd.SliceDirectory(ttl_s=1000, path=p, now_fn=clk)
    d1.publish("nodeA:8077", ["m@h-L0-16.gguf"])
    # a fresh instance loads the persisted records
    d2 = sd.SliceDirectory(ttl_s=1000, path=p, now_fn=clk)
    assert d2.holders("m@h-L0-16.gguf") == ["nodeA:8077"]


def test_prune_drops_expired(tmp_path):
    clk = _Clock()
    d = sd.SliceDirectory(ttl_s=100, now_fn=clk)
    d.publish("a:8077", ["x.gguf"])
    clk.t += 200
    d.publish("b:8077", ["x.gguf"])               # b fresh, a expired
    d.prune()
    assert d.all_live_nodes() == ["b:8077"]


def test_publish_self_lists_gguf(tmp_path):
    clk = _Clock()
    d = sd.SliceDirectory(ttl_s=1000, now_fn=clk)
    sdir = tmp_path / "slices"; sdir.mkdir()
    (sdir / "m@h-L0-16.gguf").write_bytes(b"GGUF")
    (sdir / "notes.txt").write_text("ignore me")
    n = sd.publish_self(d, "me:8077", str(sdir))
    assert n == 1
    assert d.holders("m@h-L0-16.gguf") == ["me:8077"]
