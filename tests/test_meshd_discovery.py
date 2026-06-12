"""meshd discovery: heartbeat-TTL staleness + drift-class admission (v1.1 capstone)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402
from discovery.nakshatra_listing import NakshatraListing  # noqa: E402
from discovery.relay import FileRelay  # noqa: E402
from mesh.meshd import MeshNode, MeshConfig  # noqa: E402


def _signed(node_id, drift, age_s, serving=("prithvi-q8",)):
    k = ed25519.Ed25519PrivateKey.generate()
    L = NakshatraListing(
        mesh_id="m", node_id=node_id,
        ed25519_pubkey_hex=k.public_key().public_bytes_raw().hex(),
        serving=list(serving), supported_protocol=[1], drift_class=drift,
        measured_decode_ms_per_layer=2.0, created_unix=int(time.time()) - age_s)
    L.sign(k.private_bytes_raw())
    return L


def _node(tmp_path, drift):
    cfg = MeshConfig(
        mesh_id="m", serving=["prithvi-q8"], relay_dir=str(tmp_path / "relay"),
        rendezvous_host="127.0.0.1", rendezvous_port=51820, worker_addr=None,
        drift_class=drift, endpoint_hint="", decode_ms_per_layer=None,
        refresh=10.0, identity_file=tmp_path / "id.key",
        status_file=tmp_path / "st.json", once=True, peer_ttl=60.0)
    return MeshNode(cfg)


def test_fresh_same_class_peer_admitted(tmp_path):
    node = _node(tmp_path, "classA")
    relay = FileRelay(str(tmp_path / "relay"))
    relay.publish(_signed("peer-fresh", "classA", age_s=5))
    admitted = node._discover()
    assert [l.node_id for l in admitted] == ["peer-fresh"]


def test_stale_peer_aged_out(tmp_path):
    node = _node(tmp_path, "classA")               # peer_ttl=60
    relay = FileRelay(str(tmp_path / "relay"))
    relay.publish(_signed("peer-dead", "classA", age_s=120))   # 2 min old > TTL
    assert node._discover() == []                  # stale heartbeat → not admitted


def test_wrong_drift_class_rejected(tmp_path):
    node = _node(tmp_path, "classA")
    relay = FileRelay(str(tmp_path / "relay"))
    relay.publish(_signed("peer-other-build", "classB", age_s=5))
    assert node._discover() == []                  # fresh but wrong build → rejected


def test_effective_ttl_auto_scales_with_refresh():
    cfg = MeshConfig(mesh_id="m", serving=[], relay_dir="/tmp/x",
                     rendezvous_host="127.0.0.1", rendezvous_port=1, worker_addr=None,
                     drift_class=None, endpoint_hint="", decode_ms_per_layer=None,
                     refresh=30.0, identity_file=Path("/tmp/k"),
                     status_file=Path("/tmp/s"), peer_ttl=0.0)
    assert cfg.effective_ttl() == 120.0            # max(90, 4*30)
    cfg2 = MeshConfig(**{**cfg.__dict__, "refresh": 5.0})
    assert cfg2.effective_ttl() == 90.0            # floor
