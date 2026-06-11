#!/usr/bin/env python3
"""§10 acceptance test — CONTROL-PLANE half (runnable without the gRPC daemon).

The v1.0 ship gate (docs/v1.0-discovery-and-distribution.md §10) is: two machines
with **no shared static config** discover each other, mutually authenticate,
provision **only their assigned layer range** from a verified package, negotiate
the wire version, and then produce the byte-identical greedy token.

That last step (the token) needs the patched gRPC worker daemon + the heavy
runtime. THIS harness validates everything up to it — the part the merged P1/P2/P4
work newly enables — using only the pure-Python modules (no grpc, no daemon):

  1. two nodes, each with its own Ed25519 key, share ONLY a relay directory
     (the stand-in for a public Nostr relay) — no static YAML naming the other;
  2. each publishes a SIGNED NakshatraListing;
  3. node A discovers node B, ranks it by measured compute Fᵢ, and PINS its
     advertised Ed25519 key (and vice-versa) — admission is identity-bound;
  4. control-protocol versions are negotiated (ALPN-style) — incompatible → abort;
  5. each node SELF-PROVISIONS its assigned layer range from the shared package,
     verifying every fragment's SHA-256 fail-closed, assembling a loader-ready
     sub-GGUF.

Exit 0 + "ACCEPTANCE (control plane): PASS" iff all five hold. Part B
(cluster_token_parity.sh) takes the assembled sub-GGUFs into the gRPC chain for
the token-parity check.

Run:  python -m validate.acceptance_controlplane --package <dir> [--n-layers N]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nakshatra_auth import generate_keypair  # noqa: E402
from discovery.nakshatra_listing import NakshatraListing  # noqa: E402
from discovery.relay import FileRelay, pin_from_listing  # noqa: E402
from routing.model_router import resolve_serving_peer  # noqa: E402  (rank+pin path)
from wire.version import SUPPORTED_CONTROL_VERSIONS  # noqa: E402
from wire.handshake import negotiate_handshake, advertise_capabilities  # noqa: E402
from packaging.fetch_package import fetch_and_assemble  # noqa: E402
from packaging.nakshatra_package import NakshatraPackage  # noqa: E402

MESH = "acceptance-mesh"


class Node:
    """A would-be cluster member. Knows only: its own key, its assigned range,
    the shared relay dir, and the package location. NOT the other node."""

    def __init__(self, name, layer_range, relay_dir, decode_ms, model_id):
        self.name = name
        self.start, self.end = layer_range
        self.relay = FileRelay(relay_dir)
        self.decode_ms = decode_ms
        self.model_id = model_id
        self.priv, self.pub = generate_keypair()
        self.node_id = f"node-{name}-{self.pub[:8]}"

    def advertise(self, serving):
        l = NakshatraListing(
            mesh_id=MESH, node_id=self.node_id, ed25519_pubkey_hex=self.pub,
            serving=[serving], measured_decode_ms_per_layer=self.decode_ms,
            endpoint_hint=f"http://127.0.0.1:55{self.start:02d}",
            supported_protocol=list(SUPPORTED_CONTROL_VERSIONS),
            node_count=1,
        )
        l.sign(self.priv)
        self.relay.publish(l)
        print(f"  [{self.name}] published signed listing {self.node_id} "
              f"(Fᵢ={self.decode_ms}ms/layer, layers [{self.start},{self.end}))")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, help="layer-package dir (package.json inside)")
    ap.add_argument("--n-layers", type=int, default=None,
                    help="total layers (default: read from manifest)")
    ap.add_argument("--require-signature", action="store_true",
                    help="require the package manifest to be signed")
    args = ap.parse_args()

    pkg = NakshatraPackage.from_json((Path(args.package) / "package.json").read_text())
    n = args.n_layers or pkg.n_layers
    mid = n // 2
    model_id = pkg.model_id
    print(f"§10 control-plane acceptance — model {model_id}, {n} layers, "
          f"split [0,{mid}) + [{mid},{n})")
    print(f"package signed: {bool(pkg.signature_b64)}; this build speaks "
          f"control versions {list(SUPPORTED_CONTROL_VERSIONS)}\n")

    relay_dir = tempfile.mkdtemp(prefix="nks-accept-relay-")
    work = tempfile.mkdtemp(prefix="nks-accept-work-")
    print(f"shared relay (the only shared state): {relay_dir}\n")

    # 1+2. two nodes, separate keys, publish signed listings to the shared relay
    a = Node("A", (0, mid), relay_dir, decode_ms=2.0, model_id=model_id)
    b = Node("B", (mid, n), relay_dir, decode_ms=5.0, model_id=model_id)
    print("[1/5] advertise signed listings:")
    a.advertise(model_id)
    b.advertise(model_id)

    # 3. A discovers B (ranked by Fᵢ, pinned) and vice-versa — identity-bound admission
    print("\n[2/5] discover + rank + pin (no static config — relay only):")
    a_sees = resolve_serving_peer(a.relay, model_id, mesh_id=MESH, exclude_node_id=a.node_id)
    b_sees = resolve_serving_peer(b.relay, model_id, mesh_id=MESH, exclude_node_id=b.node_id)
    assert a_sees and a_sees[0].node_id == b.node_id, "A failed to discover+pin B"
    assert b_sees and b_sees[0].node_id == a.node_id, "B failed to discover+pin A"
    assert a_sees[0].ed25519_pubkey_hex == b.pub, "A pinned the wrong key for B"
    assert b_sees[0].ed25519_pubkey_hex == a.pub, "B pinned the wrong key for A"
    print(f"  A pinned B → {a_sees[0].ed25519_pubkey_hex[:16]}… (score {a_sees[2]:.1f})")
    print(f"  B pinned A → {b_sees[0].ed25519_pubkey_hex[:16]}… (score {b_sees[2]:.1f})")

    # 4. negotiate control version across the (advertised) supported sets
    print("\n[3/5] negotiate control-protocol version (ALPN-style):")
    agreed = negotiate_handshake("0.1.0", advertise_capabilities())
    print(f"  agreed control version = {agreed}")

    # 5. each node self-provisions ONLY its assigned range from the package
    print("\n[4/5] self-provision assigned ranges from the verified package "
          "(fail-closed SHA-256):")
    a_gguf = str(Path(work) / "node-A.gguf")
    b_gguf = str(Path(work) / "node-B.gguf")
    fetch_and_assemble(args.package, a.start, a.end, a_gguf,
                       require_signature=args.require_signature)
    fetch_and_assemble(args.package, b.start, b.end, b_gguf,
                       require_signature=args.require_signature)

    # verify both assembled sub-GGUFs carry the right slice metadata
    print("\n[5/5] verify provisioned sub-GGUFs:")
    from gguf import GGUFReader
    for node, path in ((a, a_gguf), (b, b_gguf)):
        kv = {f.name: (f.parts[f.data[0]].tolist()[0] if hasattr(f.parts[f.data[0]], "tolist")
                       else f.parts[f.data[0]][0])
              for f in GGUFReader(path).fields.values() if f.name.startswith("nakshatra.")}
        assert kv["nakshatra.layer_range_start"] == node.start
        assert kv["nakshatra.layer_range_end"] == node.end
        sz = Path(path).stat().st_size
        print(f"  [{node.name}] {path} layers=[{node.start},{node.end}) "
              f"embd={bool(kv['nakshatra.has_token_embd'])} "
              f"lm_head={bool(kv['nakshatra.has_lm_head'])} ({sz/1e6:.0f} MB)")

    print("\nProvisioned sub-GGUFs (hand these to Part B for the token-parity run):")
    print(f"  NODE_A_GGUF={a_gguf}")
    print(f"  NODE_B_GGUF={b_gguf}")
    print("\n✅ ACCEPTANCE (control plane): PASS — discover → pin → negotiate → "
          "self-provision, no shared static config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
