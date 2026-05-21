"""Cluster smoke harness — probe peers' SPKI through PillarPeerKeyResolver.

The "client" half of the 2026-05-21 SPKI federation cluster smoke. Pair
with smoke_spki_register.py running on the target machine(s).

Runs the four falsifiable checks named in the sprint plan:

  1. handshake-success-on-match — open_pinned_channel against a peer
     whose pillar-attested SPKI matches its actual cert.
  2. refuse-on-mismatch — open_pinned_channel against a peer where
     expected SPKI differs from observed cert (we override expected
     in-line; the operator's real-world equivalent is rotating a peer's
     cert without re-registering).
  3. refuse-on-unknown-peer — open_pinned_channel against an address
     not in the pillar's roster.
  4. refuse-on-stale-cache — short stale_cache_deadline; sleep past it;
     verify expected_spki returns None.

Each check prints its expected vs. actual outcome. Exits 0 if all four
behave as expected; non-zero otherwise.

Usage:
    python scripts/smoke_spki_probe.py \\
        --pillar-url http://home-pc:7778 \\
        --target-addr 203.0.113.11:5550 \\
        --target-node-id smoke-node-e \\
        [--unknown-addr 127.0.0.1:9999]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nakshatra_auth as auth
import nakshatra_grpc_auth as ga
import nakshatra_tls as nt


def _resolve_with_retries(resolver: ga.PillarPeerKeyResolver,
                            node_id: str, addr: str,
                            attempts: int = 5) -> str | None:
    """Refresh the cache + check for the target's SPKI. The pillar
    may not have observed our register-fetch round-trip yet; retry."""
    for i in range(attempts):
        try:
            resolver.refresh_once()
        except Exception as e:
            print(f"  [warn] refresh attempt {i+1} failed: {e}")
            time.sleep(1)
            continue
        spki = resolver.expected_spki(addr)
        if spki is not None:
            return spki
        print(f"  [retry {i+1}] {addr} not yet showing SPKI; waiting...")
        time.sleep(2)
    return resolver.expected_spki(addr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pillar-url", required=True)
    ap.add_argument("--target-addr", required=True,
                    help="host:port for the register-side peer "
                         "(e.g. 203.0.113.11:5550)")
    ap.add_argument("--target-node-id", required=True)
    ap.add_argument("--unknown-addr", default="127.0.0.1:65535",
                    help="An address the pillar does NOT have in its "
                         "roster (used for check 3)")
    ap.add_argument("--probe-timeout", type=float, default=5.0)
    args = ap.parse_args()

    # Identity for the probe client (so the pillar accepts our /peers GET).
    # We don't register ourselves — just sign with a fresh ephemeral key
    # if the pillar requires auth on /peers, or skip if it doesn't.
    priv_bytes, pub_hex = auth.generate_keypair()
    probe_node_id = "smoke-probe-client"

    # First, register the probe client so the pillar accepts authenticated
    # /peers reads. We don't need to listen on anything — just need a
    # registered identity. address can be a placeholder.
    import json
    from urllib import request as urlrequest
    body = json.dumps({
        "node_id": probe_node_id, "node_type": "compute",
        "address": "probe-only:0",
        "public_key_hex": pub_hex,
    }).encode("utf-8")
    ts = int(time.time())
    sig = auth.sign_request(priv_bytes, "POST", "/peer", body, ts)
    header = f'Sthambha-Ed25519 keyid="{probe_node_id}",sig="{sig}",ts="{ts}"'
    try:
        with urlrequest.urlopen(urlrequest.Request(
                f"{args.pillar_url.rstrip('/')}/peer", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": header},
                method="POST"), timeout=10) as resp:
            print(f"[probe] registered probe client: {resp.read().decode()[:120]}")
    except Exception as e:
        print(f"[probe] WARN: could not register probe client: {e}")

    resolver = ga.PillarPeerKeyResolver(
        args.pillar_url,
        refresh_interval_s=10.0,
        stale_cache_deadline_s=300.0,
        priv_key=priv_bytes,
        own_node_id=probe_node_id,
    )
    print(f"[probe] pillar URL: {args.pillar_url}")
    print(f"[probe] target: {args.target_addr} (node_id={args.target_node_id})")
    print(f"[probe] unknown: {args.unknown_addr}")
    print()

    failures = 0
    results: list[tuple[str, bool, str]] = []

    # Wait for the target's SPKI to appear in the roster.
    expected = _resolve_with_retries(
        resolver, args.target_node_id, args.target_addr)
    if expected is None:
        print(f"[FAIL] target {args.target_node_id} @ {args.target_addr} "
              f"has no SPKI in pillar roster — register side may not be "
              f"up, or address mismatch.")
        return 2
    print(f"[probe] pillar-attested SPKI for {args.target_node_id}: "
          f"{expected}")
    print()

    # ── Check 1: handshake-success-on-match ──────────────────────────
    print("[check 1] handshake-success-on-match")
    try:
        ch = nt.open_pinned_channel(
            args.target_addr, expected,
            refuse_unpinned=True,
            probe_timeout_s=args.probe_timeout,
        )
        ch.close()
        print(f"  ✓ open_pinned_channel succeeded")
        results.append(("1 match handshake", True, "ok"))
    except nt.PinError as e:
        print(f"  ✗ unexpected refusal: {e}")
        results.append(("1 match handshake", False, str(e)))
        failures += 1
    print()

    # ── Check 2: refuse-on-mismatch ──────────────────────────────────
    print("[check 2] refuse-on-mismatch (wrong expected SPKI)")
    wrong_spki = "0" * 64
    try:
        nt.open_pinned_channel(
            args.target_addr, wrong_spki,
            refuse_unpinned=True,
            probe_timeout_s=args.probe_timeout,
        )
        print(f"  ✗ unexpectedly succeeded with wrong SPKI")
        results.append(("2 spki mismatch", False, "did not refuse"))
        failures += 1
    except nt.PinError as e:
        if e.reason == "spki_mismatch":
            assert e.details.get("expected") == wrong_spki
            assert e.details.get("actual") == expected
            print(f"  ✓ refused with spki_mismatch "
                  f"(expected={wrong_spki[:8]}…, actual={expected[:8]}…)")
            results.append(("2 spki mismatch", True, "ok"))
        else:
            print(f"  ✗ wrong reason: {e.reason} ({e})")
            results.append(("2 spki mismatch", False, e.reason))
            failures += 1
    print()

    # ── Check 3: refuse-on-unknown-peer ──────────────────────────────
    print("[check 3] refuse-on-unknown-peer (address not in roster)")
    unknown_expected = resolver.expected_spki(args.unknown_addr)
    if unknown_expected is not None:
        print(f"  [warn] {args.unknown_addr} unexpectedly in roster "
              f"with SPKI {unknown_expected[:8]}… — pick a different "
              f"--unknown-addr next run")
        results.append(("3 unknown peer", False, "addr was in roster"))
        failures += 1
    else:
        # Caller treats None as "refuse if unpinned policy". Verify
        # open_pinned_channel refuses on the unknown path.
        try:
            nt.open_pinned_channel(
                args.unknown_addr, unknown_expected,
                refuse_unpinned=True,
                probe_timeout_s=args.probe_timeout,
            )
            print(f"  ✗ unexpectedly opened channel to unknown peer")
            results.append(("3 unknown peer", False, "did not refuse"))
            failures += 1
        except nt.PinError as e:
            if e.reason == "unpinned_peer":
                print(f"  ✓ refused with unpinned_peer (= unknown to pillar)")
                results.append(("3 unknown peer", True, "ok"))
            else:
                print(f"  ✗ wrong reason: {e.reason} ({e})")
                results.append(("3 unknown peer", False, e.reason))
                failures += 1
    print()

    # ── Check 4: refuse-on-stale-cache ───────────────────────────────
    print("[check 4] refuse-on-stale-cache (cache age past deadline)")
    stale_resolver = ga.PillarPeerKeyResolver(
        args.pillar_url,
        refresh_interval_s=10.0,
        stale_cache_deadline_s=0.5,
        priv_key=priv_bytes,
        own_node_id=probe_node_id,
    )
    try:
        stale_resolver.refresh_once()
    except Exception as e:
        print(f"  ✗ unable to refresh: {e}")
        failures += 1
    else:
        fresh = stale_resolver.expected_spki(args.target_addr)
        if fresh != expected:
            print(f"  ✗ stale_resolver fresh lookup mismatched expected")
            results.append(("4 stale cache", False, "fresh mismatch"))
            failures += 1
        else:
            time.sleep(1.0)  # pass the 0.5s deadline
            stale_lookup = stale_resolver.expected_spki(args.target_addr)
            if stale_lookup is None:
                print(f"  ✓ refused on stale cache (lookup returned None)")
                results.append(("4 stale cache", True, "ok"))
            else:
                print(f"  ✗ stale cache returned {stale_lookup[:8]}…, "
                      f"expected None")
                results.append(("4 stale cache", False, "no refuse"))
                failures += 1
    print()

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Summary:")
    for name, ok, note in results:
        print(f"  {'✓' if ok else '✗'} {name}: {note}")
    print(f"\n{len(results) - failures}/{len(results)} checks passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
