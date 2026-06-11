#!/usr/bin/env python3
"""discover.py — publish this node's listing and find/rank peers (v1.0 §4 CLI).

Uses the worker's Ed25519 mesh key as the signed listing identity, and ranks
discovered peers by measured compute Fᵢ (the Sthambha decode-ms/layer signal).

    # advertise this node
    python -m discovery.discover publish --relay-dir /shared/nks-relay \
        --mesh-id home --serving llama-3.3-70b --decode-ms-per-layer 1.8

    # find the best peers to grow the mesh toward
    python -m discovery.discover find --relay-dir /shared/nks-relay \
        --mesh-id home --want-model llama-3.3-70b

Transport here is FileRelay (a shared dir) — zero-dependency, good for a
Tailscale-mounted share or local dev. Swap in NostrRelay for a public relay
once secp256k1 (coincurve) is installed; the listing/rank/pin logic is identical.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nakshatra_auth import load_or_create_worker_key  # noqa: E402
from discovery.nakshatra_listing import NakshatraListing, rank_listings  # noqa: E402
from discovery.relay import FileRelay, pin_from_listing  # noqa: E402
from wire.version import SUPPORTED_CONTROL_VERSIONS  # noqa: E402


def _node_id(pub_hex: str) -> str:
    return f"nks-{pub_hex[:12]}"


def cmd_publish(args: argparse.Namespace) -> int:
    priv, pub = load_or_create_worker_key()
    listing = NakshatraListing(
        mesh_id=args.mesh_id,
        node_id=args.node_id or _node_id(pub),
        ed25519_pubkey_hex=pub,
        serving=args.serving or [],
        wanted=args.wanted or [],
        total_vram_bytes=args.vram_bytes,
        node_count=args.node_count,
        measured_decode_ms_per_layer=args.decode_ms_per_layer,
        endpoint_hint=args.endpoint or "",
        capacity_full=args.capacity_full,
        supported_protocol=list(SUPPORTED_CONTROL_VERSIONS),
        drift_class=args.drift_class,
        created_unix=int(time.time()),
    )
    listing.sign(priv)
    FileRelay(args.relay_dir).publish(listing)
    print(f"[discover] published {listing.node_id} mesh={listing.mesh_id} "
          f"serving={listing.serving} Fᵢ-signal={listing.measured_decode_ms_per_layer} "
          f"→ {args.relay_dir}")
    return 0


def cmd_find(args: argparse.Namespace) -> int:
    _, pub = load_or_create_worker_key()
    me = args.node_id or _node_id(pub)
    listings = FileRelay(args.relay_dir).query(mesh_id=args.mesh_id)
    ranked = rank_listings(
        listings, exclude_node_id=me,
        want_mesh_id=args.mesh_id, want_model=args.want_model,
        offer_model=args.offer_model,
    )
    if not ranked:
        print("[discover] no verified peers found")
        return 0
    print(f"[discover] {len(ranked)} verified peer(s), best-first:")
    for l, score in ranked:
        pin = pin_from_listing(l)  # demonstrate the admission pin
        ms = l.measured_decode_ms_per_layer
        print(f"  {score:9.1f}  {l.node_id}  mesh={l.mesh_id}  "
              f"decode_ms/layer={ms if ms is not None else '—'}  "
              f"serving={l.serving}  pin={pin.ed25519_pubkey_hex[:12]}…")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Nakshatra discovery (publish/find).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("publish", help="advertise this node's signed listing")
    p.add_argument("--relay-dir", required=True)
    p.add_argument("--mesh-id", required=True)
    p.add_argument("--node-id", default=None)
    p.add_argument("--serving", nargs="*", default=None)
    p.add_argument("--wanted", nargs="*", default=None)
    p.add_argument("--vram-bytes", type=int, default=0)
    p.add_argument("--node-count", type=int, default=1)
    p.add_argument("--drift-class", default=None,
                   help="drift_gauge fingerprint for the served model (v1.1 §8.1); "
                        "lets peers pre-filter to a bit-deterministic engine-build class")
    p.add_argument("--decode-ms-per-layer", type=float, default=None,
                   help="measured compute Fᵢ signal (lower = faster)")
    p.add_argument("--endpoint", default=None)
    p.add_argument("--capacity-full", action="store_true")
    p.set_defaults(func=cmd_publish)

    f = sub.add_parser("find", help="discover + rank peers by measured compute")
    f.add_argument("--relay-dir", required=True)
    f.add_argument("--mesh-id", default=None)
    f.add_argument("--node-id", default=None)
    f.add_argument("--want-model", default=None)
    f.add_argument("--offer-model", default=None)
    f.set_defaults(func=cmd_find)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
