"""U2: one node identity — meshd defaults to the same key as the worker, pillar auth and nakd."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_meshd_identity_defaults_to_the_node_key(monkeypatch):
    import argparse
    from mesh import meshd
    from nakshatra_auth import WORKER_KEY_PATH
    seen = {}
    orig = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        ns = orig(self, args, namespace)
        seen["identity_file"] = ns.identity_file
        raise SystemExit(0)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    try:
        meshd.main([])
    except SystemExit:
        pass
    assert seen["identity_file"] == str(WORKER_KEY_PATH)
    assert "mesh.key" not in seen["identity_file"]


def test_sidecar_key_is_the_node_key_in_libp2p_form():
    import os
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from sidecar_key import libp2p_ed25519_private
    seed = os.urandom(32)
    blob = libp2p_ed25519_private(seed)
    assert blob[:4] == bytes([0x08, 0x01, 0x12, 64]) and len(blob) == 68
    assert blob[4:36] == seed
    assert blob[36:] == ed25519.Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()


def test_sidecar_peer_id_matches_go_libp2p_fixed_vector():
    from sidecar_key import peer_id_from_node_pub
    # RFC 8032 test key; expected value is also derived with peer.IDFromPublicKey in the sidecar's Go tests.
    pub = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    assert peer_id_from_node_pub(pub) == "12D3KooWQK1wnefoLrcVHbbnf5tLzbopUd3K3bFAoJpA7YJgL5pV"


def test_sidecar_if_missing_repairs_mismatched_or_corrupt_key_and_keeps_backup(tmp_path):
    import os
    from sidecar_key import libp2p_ed25519_private, main
    node_key, out = tmp_path / "node.key", tmp_path / "sidecar" / "node.key"
    seed = os.urandom(32)
    node_key.write_bytes(seed)
    out.parent.mkdir()
    out.write_bytes(b"truncated legacy key")
    assert main(["--node-key", str(node_key), "--out", str(out), "--if-missing"]) == 0
    assert out.read_bytes() == libp2p_ed25519_private(seed)
    assert out.with_name("node.key.bak").read_bytes() == b"truncated legacy key"

    # A valid key for a different node is also repaired, not accepted just because it parses/exists.
    wrong = libp2p_ed25519_private(os.urandom(32))
    out.write_bytes(wrong)
    assert main(["--node-key", str(node_key), "--out", str(out), "--if-missing"]) == 0
    assert out.read_bytes() == libp2p_ed25519_private(seed)
    assert out.with_name("node.key.bak").read_bytes() == wrong
