"""Always-on mesh layer (v1.1 capstone).

Turns the proven discovery + transport components into long-running daemons that
form a mesh WITHOUT hand-holding:
  • pairing.py — deterministic, coordination-free rendezvous id + role for a pair
    of nodes (both sides compute the same id and agree on who dials);
  • meshd.py   — the node daemon: publish-heartbeat → discover (verify+pin+rank,
    same-drift-class only) → auto-bring-up an encrypted tunnel to the chosen peer.

The pieces it stands on (all proven on real hardware — see docs/INFRA-MAP.md):
discovery/{nostr,nakshatra_listing,relay}.py, transport/{relay,secure_channel,
mux_tunnel}.py, recovery/drift_aware.py.
"""
