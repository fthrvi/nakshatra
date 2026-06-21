"""slice_cdc — content-defined chunking dedup, tested on small synthetic data."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import slice_cdc as cdc


def _rand(n, seed=0):
    import random
    r = random.Random(seed)
    return bytes(r.getrandbits(8) for _ in range(n))


def test_chunk_boundaries_cover_and_are_contiguous():
    data = _rand(200_000)
    bs = cdc.chunk_boundaries(data, avg_bits=12)
    assert bs[0][0] == 0 and bs[-1][1] == len(data)
    for (a, b), (c, d) in zip(bs, bs[1:]):
        assert b == c                       # contiguous, no gaps/overlap
    assert len(bs) > 1                       # actually chunked


def test_chunking_is_deterministic():
    data = _rand(120_000, seed=7)
    assert cdc.manifest(data, avg_bits=12) == cdc.manifest(data, avg_bits=12)


def test_reconstruct_equals_original(tmp_path):
    data = _rand(150_000, seed=3)
    src = tmp_path / "slice.gguf"; src.write_bytes(data)
    store = cdc.ChunkStore(str(tmp_path / "store"))
    man = cdc.index_slice(str(src), store, avg_bits=12)
    out = str(tmp_path / "rebuilt.gguf")
    assert cdc.reconstruct(man, store, out) is True
    assert open(out, "rb").read() == data    # byte-identical rebuild


def test_dedup_shared_prefix_only_fetches_novel_chunks(tmp_path):
    # two slices sharing a large identical region (quant-family analogue)
    shared = _rand(120_000, seed=1)
    a = shared + _rand(20_000, seed=2)
    b = shared + _rand(20_000, seed=9)       # same big prefix, different tail
    pa = tmp_path / "a.gguf"; pa.write_bytes(a)
    pb = tmp_path / "b.gguf"; pb.write_bytes(b)
    store = cdc.ChunkStore(str(tmp_path / "store"))
    cdc.index_slice(str(pa), store, avg_bits=12)         # store A's chunks
    man_b = cdc.manifest(b, avg_bits=12)
    miss = cdc.missing_chunks(man_b, store)              # what B would need to fetch
    # B shares the prefix → most of B's chunks are already stored from A
    assert len(miss) < len(man_b)                        # genuine dedup
    shared_frac = 1 - len(miss) / len(man_b)
    assert shared_frac > 0.4                             # big shared region reused


def test_missing_then_fetch_reconstructs(tmp_path):
    # simulate fetching only the missing chunks from a "peer" store
    data = _rand(140_000, seed=5)
    peer = cdc.ChunkStore(str(tmp_path / "peer"))
    man = []
    for s, e in cdc.chunk_boundaries(data, avg_bits=12):
        man.append(peer.add(data[s:e]))
    local = cdc.ChunkStore(str(tmp_path / "local"))      # empty
    miss = cdc.missing_chunks(man, local)
    assert miss == [s for s in dict.fromkeys(man)]       # all missing locally
    out = str(tmp_path / "out.gguf")
    ok = cdc.reconstruct(man, local, out, fetch_fn=lambda sha: peer.get(sha))
    assert ok and open(out, "rb").read() == data
