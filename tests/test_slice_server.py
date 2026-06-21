"""slice_server + http_pull + ensure_present — real HTTP transport, end to end,
tested against a live in-process server (no external network)."""
import json
import os
import sys
import threading
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import slice_server
import slice_fetch as sf
import pytest


REF = sf.SliceRef("deepseek", "abc123", 0, 16)


def _gguf(path, body=b"\x07" * 4096):
    with open(path, "wb") as f:
        f.write(b"GGUF" + body)


@pytest.fixture
def live_server(tmp_path):
    served = tmp_path / "served"; served.mkdir()
    _gguf(served / REF.filename)
    httpd = slice_server.serve(str(served), host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    yield f"127.0.0.1:{port}", served
    httpd.shutdown()


def test_slices_listing(live_server):
    peer, _ = live_server
    with urllib.request.urlopen(f"http://{peer}/slices", timeout=5) as r:
        data = json.load(r)
    names = [s["name"] for s in data["slices"]]
    assert REF.filename in names


def test_http_pull_fetches_slice(live_server, tmp_path):
    peer, _ = live_server
    dest = str(tmp_path / "out.gguf")
    assert sf.http_pull(peer, REF, dest) is True
    with open(dest, "rb") as f:
        assert f.read(4) == b"GGUF"


def test_pull_missing_slice_raises(live_server, tmp_path):
    peer, _ = live_server
    missing = sf.SliceRef("nope", "zzz", 0, 8)
    with pytest.raises(Exception):  # 404 → HTTPError
        sf.http_pull(peer, missing, str(tmp_path / "x.gguf"))


def test_path_traversal_blocked(live_server, tmp_path):
    peer, _ = live_server
    with pytest.raises(Exception):  # 400 bad name
        urllib.request.urlopen(f"http://{peer}/slice/..%2f..%2fetc%2fpasswd", timeout=5)


def test_ensure_present_pulls_from_live_peer(live_server, tmp_path):
    peer, _ = live_server
    cache = tmp_path / "cache"
    src = sf.mesh_peer_source(sf.static_holders([peer]), sf.http_pull)
    p = sf.ensure_present(REF, str(cache), sources=[src], log=lambda *_: None)
    assert os.path.exists(p)
    with open(p, "rb") as f:
        assert f.read(4) == b"GGUF"


def test_ensure_falls_through_dead_peer_to_live(live_server, tmp_path):
    peer, _ = live_server
    cache = tmp_path / "cache"
    # first peer is dead (connection refused), second is the live server
    src = sf.mesh_peer_source(sf.static_holders(["127.0.0.1:1", peer]), sf.http_pull)
    p = sf.ensure_present(REF, str(cache), sources=[src], log=lambda *_: None)
    assert os.path.exists(p)


def test_make_ensure_fn_end_to_end(live_server, tmp_path):
    # the production builder: refs + live peer → ensure() returns local paths,
    # fetching the missing slice from the peer.
    peer, _ = live_server
    cache = tmp_path / "cache"
    ensure = sf.make_ensure_fn([REF], str(cache), peers=[peer], log=lambda *_: None)
    paths = ensure()
    assert len(paths) == 1 and os.path.exists(paths[0])
    # second call is a pure cache hit (no server needed) — still returns the path
    paths2 = ensure()
    assert paths2 == paths
