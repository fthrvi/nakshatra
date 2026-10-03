"""sidecar_key.py — the libp2p sidecar's key FROM the node key (unification U2).

The sidecar (`third_party/shard-libp2p-sidecar`) reads a libp2p-marshalled private key
(`crypto.UnmarshalPrivateKey`): protobuf {1: KeyType=Ed25519(1), 2: Data = seed(32) || pubkey(32)}.
Writing that from `~/.nakshatra/keys/worker.ed25519` makes the sidecar's PeerId derive from the SAME
node key as meshd, the worker, pillar auth and nakd — one node identity, not four.

    python3 scripts/sidecar_key.py [--node-key PATH] --out ~/.config/nakshatra-sidecar/node.key

⚠️ Switching a RUNNING sidecar to this key changes its PeerId; anything that dials it by PeerId (e.g.
Sutra's tunnel to blackwell) must be updated in the same step.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nakshatra_auth import WORKER_KEY_PATH  # noqa: E402


def libp2p_ed25519_private(seed: bytes) -> bytes:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    if len(seed) != 32:
        raise ValueError("node key must be a 32-byte Ed25519 seed")
    pub = ed25519.Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()
    data = seed + pub
    return bytes([0x08, 0x01, 0x12, len(data)]) + data       # field1 varint 1 (Ed25519), field2 bytes(64)


def _base58btc(data: bytes) -> str:
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    zeroes = len(data) - len(data.lstrip(b"\0"))
    n, out = int.from_bytes(data, "big"), ""
    while n:
        n, digit = divmod(n, 58)
        out = alphabet[digit] + out
    return "1" * zeroes + out


def peer_id_from_node_pub(pub_hex: str) -> str:
    """The identity-multihash PeerId go-libp2p derives from a 32-byte Ed25519 node key."""
    try:
        pub = bytes.fromhex(pub_hex)
    except (TypeError, ValueError) as e:
        raise ValueError("node public key must be 32-byte hex") from e
    if len(pub) != 32:
        raise ValueError("node public key must be 32-byte hex")
    public_key_proto = bytes([0x08, 0x01, 0x12, 0x20]) + pub
    identity_multihash = bytes([0x00, len(public_key_proto)]) + public_key_proto
    return _base58btc(identity_multihash)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--node-key", type=Path, default=WORKER_KEY_PATH)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--if-missing", action="store_true",
                    help="leave an existing key only when it already derives from the current node key")
    a = ap.parse_args(argv)
    blob = libp2p_ed25519_private(a.node_key.read_bytes())
    if a.if_missing and a.out.exists():
        try:
            if a.out.read_bytes() == blob:
                return 0
        except OSError:
            pass
        backup = a.out.with_name(a.out.name + ".bak")
        os.replace(a.out, backup)
        print(f"backed up mismatched/corrupt key to {backup}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(a.out), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(blob)
    print(f"wrote {a.out} (sidecar identity = node key {blob[-32:].hex()[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
