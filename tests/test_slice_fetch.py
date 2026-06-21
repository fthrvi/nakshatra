"""slice_fetch — content-addressed slice acquisition (local→peer→origin),
tested without network: fake sources are local files."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import slice_fetch as sf


REF = sf.SliceRef(model="deepseek", model_hash="abc123", layer_start=0, layer_end=16)


def _gguf(path, body=b"\x00" * 4096):
    with open(path, "wb") as f:
        f.write(b"GGUF" + body)


def test_filename_is_content_addressed():
    assert REF.filename == "deepseek@abc123-L0-16.gguf"


def test_cache_hit_returns_without_sources(tmp_path):
    cache = tmp_path / "cache"; cache.mkdir()
    _gguf(cache / REF.filename)
    # no sources at all — must still return the cached file
    p = sf.ensure_present(REF, str(cache), sources=[], log=lambda *_: None)
    assert p == str(cache / REF.filename)


def test_fetches_from_first_available_source(tmp_path):
    cache = tmp_path / "cache"
    peer = tmp_path / "peer"; peer.mkdir()
    _gguf(peer / REF.filename)
    src = sf.local_dir_source(str(peer), name="peer")
    p = sf.ensure_present(REF, str(cache), sources=[src], log=lambda *_: None)
    assert os.path.exists(p)
    with open(p, "rb") as f:
        assert f.read(4) == b"GGUF"


def test_source_ordering_prefers_earlier(tmp_path):
    cache = tmp_path / "cache"
    empty = tmp_path / "empty"; empty.mkdir()      # has nothing
    peer = tmp_path / "peer"; peer.mkdir()
    _gguf(peer / REF.filename)
    calls = []
    def tracking(name, d):
        s = sf.local_dir_source(str(d), name=name)
        return (name, lambda ref, dst: (calls.append(name), s[1](ref, dst))[1])
    sources = [tracking("empty", empty), tracking("peer", peer)]
    sf.ensure_present(REF, str(cache), sources=sources, log=lambda *_: None)
    assert calls == ["empty", "peer"]              # tried empty first, then peer


def test_rejects_bad_magic_and_tries_next(tmp_path):
    cache = tmp_path / "cache"
    bad = tmp_path / "bad"; bad.mkdir()
    with open(bad / REF.filename, "wb") as f:
        f.write(b"NOTGGUF" + b"\x00" * 100)        # wrong magic
    good = tmp_path / "good"; good.mkdir()
    _gguf(good / REF.filename)
    p = sf.ensure_present(REF, str(cache),
                          sources=[sf.local_dir_source(str(bad), "bad"),
                                   sf.local_dir_source(str(good), "good")],
                          log=lambda *_: None)
    with open(p, "rb") as f:
        assert f.read(4) == b"GGUF"                # got the good one


def test_verifies_sha256_when_known(tmp_path):
    cache = tmp_path / "cache"
    peer = tmp_path / "peer"; peer.mkdir()
    _gguf(peer / "deepseek@abc123-L0-16.gguf", body=b"\x01" * 1000)
    good_sha = sf._sha256(str(peer / "deepseek@abc123-L0-16.gguf"))
    ref_ok = sf.SliceRef("deepseek", "abc123", 0, 16, sha256=good_sha)
    ref_bad = sf.SliceRef("deepseek", "abc123", 0, 16, sha256="deadbeef")
    assert sf.ensure_present(ref_ok, str(cache),
                             [sf.local_dir_source(str(peer), "peer")], log=lambda *_: None)
    import pytest
    with pytest.raises(FileNotFoundError):
        sf.ensure_present(ref_bad, str(cache),
                          [sf.local_dir_source(str(peer), "peer")], log=lambda *_: None)


def test_all_sources_fail_raises(tmp_path):
    cache = tmp_path / "cache"
    empty = tmp_path / "empty"; empty.mkdir()
    import pytest
    with pytest.raises(FileNotFoundError):
        sf.ensure_present(REF, str(cache),
                          [sf.local_dir_source(str(empty), "empty")], log=lambda *_: None)


def test_source_error_does_not_abort_chain(tmp_path):
    cache = tmp_path / "cache"
    good = tmp_path / "good"; good.mkdir()
    _gguf(good / REF.filename)
    def boom(ref, dst):
        raise RuntimeError("peer unreachable")
    p = sf.ensure_present(REF, str(cache),
                          sources=[("flaky", boom),
                                   sf.local_dir_source(str(good), "good")],
                          log=lambda *_: None)
    assert os.path.exists(p)


def test_mesh_peer_source_tries_holders_in_order(tmp_path):
    cache = tmp_path / "cache"
    _gguf(tmp_path / REF.filename)
    tried = []
    def holders(ref):
        return ["peerA", "peerB"]
    def pull(peer, ref, dst):
        tried.append(peer)
        if peer == "peerB":
            import shutil
            shutil.copyfile(str(tmp_path / ref.filename), dst)
            return True
        return False
    src = sf.mesh_peer_source(holders, pull)
    sf.ensure_present(REF, str(cache), sources=[src], log=lambda *_: None)
    assert tried == ["peerA", "peerB"]            # fell through A to B
