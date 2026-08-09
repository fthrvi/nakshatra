"""The --nostr-relay flag exists and does what its docstring always claimed.

Background (2026-08-09 discovery audit): meshd's docstring and INFRA-MAP both
advertised `--nostr-relay wss://…`, but _parse_args never defined it and
MeshNode hardcoded FileRelay — the entire Nostr discovery stack was dark. These
tests pin the wire-up AND the key-persistence contract (an ephemeral event key
would defeat NIP-33 replacement, so it must load from disk).

No network: NostrRelay only touches the websocket in _connect(), never in
__init__.
"""
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from mesh.meshd import MeshConfig, MeshNode, _parse_args  # noqa: E402

pytest.importorskip("coincurve", reason="nostr path needs coincurve")
pytest.importorskip("websocket", reason="nostr path needs websocket-client")

from discovery import nostr  # noqa: E402
from discovery.relay import FileRelay, NostrRelay  # noqa: E402


def _cfg(tmp_path, **over):
    base = dict(
        mesh_id="m", serving=[], relay_dir=str(tmp_path / "relay"),
        rendezvous_host="127.0.0.1", rendezvous_port=51820, worker_addr=None,
        drift_class=None, endpoint_hint="", decode_ms_per_layer=None,
        refresh=10.0, identity_file=tmp_path / "id.key",
        status_file=tmp_path / "st.json", once=True)
    base.update(over)
    return MeshConfig(**base)


# ── the flag exists and lands in the config ──

def test_parse_args_default_is_file_relay():
    cfg = _parse_args(["--mesh-id", "m"])
    assert cfg.nostr_relay is None          # empty string normalized to None


def test_parse_args_nostr_relay_and_key_file():
    cfg = _parse_args(["--mesh-id", "m", "--nostr-relay", "wss://relay.example",
                       "--nostr-key-file", "/tmp/k.hex"])
    assert cfg.nostr_relay == "wss://relay.example"
    assert cfg.nostr_key_file == Path("/tmp/k.hex")


# ── relay selection ──

def test_meshnode_default_uses_file_relay(tmp_path):
    node = MeshNode(_cfg(tmp_path))
    assert isinstance(node.relay, FileRelay)


def test_meshnode_nostr_relay_selected_with_persisted_key(tmp_path):
    key_file = tmp_path / "keys" / "nostr.secp256k1"
    node = MeshNode(_cfg(tmp_path, nostr_relay="wss://relay.example",
                         nostr_key_file=key_file))
    assert isinstance(node.relay, NostrRelay)
    assert node.relay.relay_url == "wss://relay.example"
    # the event key was created on disk and is what the relay signs with
    assert key_file.exists()
    assert node.relay.nostr_privkey_hex == key_file.read_text().strip()


def test_meshnode_nostr_key_stable_across_restarts(tmp_path):
    """The whole point: same pubkey across restarts → listings replace."""
    key_file = tmp_path / "nostr.hex"
    a = MeshNode(_cfg(tmp_path, nostr_relay="wss://r", nostr_key_file=key_file))
    b = MeshNode(_cfg(tmp_path, nostr_relay="wss://r", nostr_key_file=key_file))
    assert a.relay.nostr_privkey_hex == b.relay.nostr_privkey_hex
    assert nostr.pubkey_of(a.relay.nostr_privkey_hex) == \
           nostr.pubkey_of(b.relay.nostr_privkey_hex)


# ── key persistence contract ──

def test_load_or_create_key_creates_0600(tmp_path):
    p = tmp_path / "deep" / "nostr.hex"
    key = nostr.load_or_create_key(p)
    assert nostr.pubkey_of(key)                       # valid curve key
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_load_or_create_key_rejects_corrupt_file(tmp_path):
    p = tmp_path / "nostr.hex"
    p.write_text("not-a-key\n")
    with pytest.raises(Exception):
        nostr.load_or_create_key(p)                   # corrupt key must never sign


# ── audit gap (b): per-node NIP-33 d-tag ──

def test_nostr_d_tag_scoped_to_node(tmp_path):
    """One shared event key publishing two nodes in one mesh must not have the
    second listing replace the first: d must be mesh_id/node_id, not mesh_id."""
    from nakshatra_auth import generate_keypair
    from discovery.nakshatra_listing import NakshatraListing
    from discovery.relay import listing_to_nostr_event_content

    def _l(node_id):
        priv, pub = generate_keypair()
        l = NakshatraListing(mesh_id="m1", node_id=node_id, ed25519_pubkey_hex=pub)
        l.sign(priv)
        return l

    d_of = lambda ev: [t for t in ev["tags"] if t[0] == "d"][0][1]
    a, b = listing_to_nostr_event_content(_l("nks-aaa")), listing_to_nostr_event_content(_l("nks-bbb"))
    assert d_of(a) == "m1/nks-aaa" and d_of(b) == "m1/nks-bbb"
    assert d_of(a) != d_of(b)      # distinct replaceable addresses under one pubkey


# ── audit gap (c): capacity fields populated ──

def test_listing_carries_probed_vram_and_node_count(tmp_path, monkeypatch):
    import fabric.worker_join as wj
    monkeypatch.setattr(wj, "detect_capabilities",
                        lambda: {"gpu": "FakeGPU", "vram_mb": 20480, "backend": "cuda"})
    node = MeshNode(_cfg(tmp_path))
    node._last_peers = [{"node_id": "p1"}, {"node_id": "p2"}]
    listing = node._build_listing()
    assert listing.total_vram_bytes == 20480 * 1024 * 1024
    assert listing.node_count == 3          # self + 2 admitted peers last loop
    assert listing.verify()                 # capacity fields ride inside the signature


def test_probe_failure_lists_at_zero_not_crash(tmp_path, monkeypatch):
    import fabric.worker_join as wj
    monkeypatch.setattr(wj, "detect_capabilities",
                        lambda: (_ for _ in ()).throw(RuntimeError("no smi")))
    node = MeshNode(_cfg(tmp_path))
    assert node.total_vram_bytes == 0
    assert node._build_listing().total_vram_bytes == 0
