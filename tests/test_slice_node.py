"""slice_directory shared-dir functions + slice_node heartbeat, no network."""
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import slice_directory as sd
import slice_fetch as sf


class _Clock:
    def __init__(self, t=1000.0): self.t = t
    def __call__(self): return self.t


REF = sf.SliceRef("m", "h", 0, 16)   # filename m@h-L0-16.gguf


def test_publish_to_dir_per_node_files(tmp_path):
    d = str(tmp_path / "dir")
    sd.publish_to_dir(d, "10.0.0.1:8077", ["m@h-L0-16.gguf"])
    sd.publish_to_dir(d, "10.0.0.2:8077", ["m@h-L16-32.gguf"])
    files = sorted(os.listdir(d))
    assert len(files) == 2            # one file per node, no clobber
    assert all(f.endswith(".json") for f in files)


def test_concurrent_publishers_dont_clobber(tmp_path):
    d = str(tmp_path / "dir")
    # both publish the SAME slice; both must remain holders
    sd.publish_to_dir(d, "nodeA:8077", ["m@h-L0-16.gguf"])
    sd.publish_to_dir(d, "nodeB:8077", ["m@h-L0-16.gguf"])
    holders = sd.holders_from_dir(d)(REF)
    assert set(holders) == {"nodeA:8077", "nodeB:8077"}


def test_holders_from_dir_ttl_and_freshness(tmp_path):
    d = str(tmp_path / "dir")
    clk = _Clock()
    sd.publish_to_dir(d, "old:8077", ["m@h-L0-16.gguf"], now_fn=clk)
    clk.t += 10
    sd.publish_to_dir(d, "new:8077", ["m@h-L0-16.gguf"], now_fn=clk)
    fn = sd.holders_from_dir(d, ttl_s=100, now_fn=clk)
    assert fn(REF) == ["new:8077", "old:8077"]     # freshest first
    clk.t += 200                                    # both now stale
    assert fn(REF) == []


def test_holders_from_dir_plugs_into_make_ensure_fn(tmp_path):
    d = str(tmp_path / "dir")
    sd.publish_to_dir(d, "peer:8077", ["m@h-L0-16.gguf"])
    holders_fn = sd.holders_from_dir(d)
    # build an ensure_fn using the directory's holders (no real fetch invoked
    # because the file is already in cache here)
    cache = str(tmp_path / "cache"); os.makedirs(cache)
    with open(os.path.join(cache, REF.filename), "wb") as f:
        f.write(b"GGUF" + b"\x00" * 64)
    ensure = sf.make_ensure_fn([REF], cache, holders_fn=holders_fn, log=lambda *_: None)
    assert ensure() == [os.path.join(cache, REF.filename)]


def test_slice_node_heartbeat_publishes(tmp_path):
    import slice_node
    slices = tmp_path / "slices"; slices.mkdir()
    (slices / "m@h-L0-16.gguf").write_bytes(b"GGUF" + b"\x00" * 16)
    d = str(tmp_path / "dir")
    httpd = slice_node.run(str(slices), "127.0.0.1:0", d, port=0, interval=0.2)
    try:
        # heartbeat runs on a timer; give it a moment to write the first record
        deadline = time.time() + 5
        while time.time() < deadline and not (os.path.isdir(d) and os.listdir(d)):
            time.sleep(0.1)
        holders = sd.holders_from_dir(d)(REF)
        assert holders == ["127.0.0.1:0"]
    finally:
        httpd._heartbeat_stop.set()
        httpd.shutdown()
