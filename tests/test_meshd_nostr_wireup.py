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


def test_load_or_create_key_regenerates_corrupt_file(tmp_path):
    # A malformed/half-written key must REGENERATE, never raise — else a
    # Restart=always daemon hot-loops forever on a bricked key file.
    p = tmp_path / "nostr.hex"
    p.write_text("not-a-key\n")
    key = nostr.load_or_create_key(p)
    assert nostr.pubkey_of(key)                        # a fresh, valid key
    assert p.read_text().strip() == key                # bad content replaced
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_load_or_create_key_regenerates_empty_file(tmp_path):
    p = tmp_path / "nostr.hex"
    p.write_text("")                                   # 0-byte = interrupted create
    key = nostr.load_or_create_key(p)
    assert nostr.pubkey_of(key)


def test_load_or_create_key_tightens_loose_perms(tmp_path):
    p = tmp_path / "nostr.hex"
    good = nostr.keygen()[0]
    p.write_text(good + "\n")
    import os as _os
    _os.chmod(p, 0o644)
    key = nostr.load_or_create_key(p)                  # loads the good key…
    assert key == good
    assert stat.S_IMODE(p.stat().st_mode) == 0o600     # …and tightens perms in place


def test_load_or_create_key_stable_second_call(tmp_path):
    p = tmp_path / "nostr.hex"
    assert nostr.load_or_create_key(p) == nostr.load_or_create_key(p)


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


# ── audit gap (c): capacity fields populated (probe lazy, cached, all-GPU) ──

def test_vram_probed_lazily_not_in_init(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(MeshNode, "_probe_total_vram_bytes",
                        lambda self: (calls.append(1), 20480 * 1024 * 1024)[1])
    node = MeshNode(_cfg(tmp_path))
    assert node._total_vram_bytes == -1 and calls == []      # NOT probed at construction
    listing = node._build_listing()
    assert listing.total_vram_bytes == 20480 * 1024 * 1024   # probed on first build
    assert node._total_vram_bytes == 20480 * 1024 * 1024     # cached
    assert listing.node_count == 1                           # self-reported count reverted
    assert listing.verify()                                  # rides inside the signature
    node._build_listing()
    assert len(calls) == 1                                    # cached, not re-probed


def test_vram_probe_failure_lists_at_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(MeshNode, "_probe_total_vram_bytes", lambda self: 0)
    node = MeshNode(_cfg(tmp_path))
    assert node._build_listing().total_vram_bytes == 0       # honest 0, no crash


def test_vram_probe_retries_until_nonzero(tmp_path, monkeypatch):
    # GPU cold at boot → 0; must re-probe (not freeze 0) until a real reading.
    seq = iter([0, 0, 8192 * 1024 * 1024])
    monkeypatch.setattr(MeshNode, "_probe_total_vram_bytes", lambda self: next(seq))
    node = MeshNode(_cfg(tmp_path))
    assert node._build_listing().total_vram_bytes == 0
    assert node._build_listing().total_vram_bytes == 0
    assert node._build_listing().total_vram_bytes == 8192 * 1024 * 1024


def test_vram_sums_all_gpus(tmp_path, monkeypatch):
    # detect_capabilities is card0-only; _probe_total_vram_bytes must sum rows.
    import shutil, subprocess
    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/nvidia-smi" if n == "nvidia-smi" else None)

    class _R:  # 3 cards, one per line
        stdout = "20480\n16320\n2048\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R())
    node = MeshNode(_cfg(tmp_path))
    assert node._probe_total_vram_bytes() == (20480 + 16320 + 2048) * 1024 * 1024


# ── audit gap (a) trust boundary: auto-tunnel allowlist over a public relay ──

def _peer_listing(pub="ab" * 32, node_id="nks-peer"):
    from discovery.nakshatra_listing import NakshatraListing
    return NakshatraListing(mesh_id="m", node_id=node_id, ed25519_pubkey_hex=pub)


def test_filerelay_tunnels_freely(tmp_path):
    node = MeshNode(_cfg(tmp_path))                          # FileRelay: ACL is the gate
    assert node._tunnel_permitted(_peer_listing()) is True


def test_nostr_without_allowlist_is_observe_only(tmp_path):
    node = MeshNode(_cfg(tmp_path, nostr_relay="wss://r",
                         nostr_key_file=tmp_path / "k.hex"))
    assert node._tunnel_permitted(_peer_listing()) is False  # discovered, NOT tunneled


def test_nostr_with_allowlist_permits_only_listed(tmp_path):
    allow = tmp_path / "allow.txt"
    allow.write_text("# my peers\nAB" + "AB" * 31 + "\n")     # case-insensitive
    node = MeshNode(_cfg(tmp_path, nostr_relay="wss://r",
                         nostr_key_file=tmp_path / "k.hex", peer_allowlist=allow))
    assert node._tunnel_permitted(_peer_listing(pub="ab" * 32)) is True
    assert node._tunnel_permitted(_peer_listing(pub="cd" * 32, node_id="nks-x")) is False
