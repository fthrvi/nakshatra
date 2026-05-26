"""Nakshatra `fabric_lite` Python prototype.

Implements the Sthambha network-fabric wire contract (raw-UDP data plane
with fixed-schema authenticated packets) as a drop-in alternative to the
existing gRPC `Forward` / `Inference` path. Boot path is gated by the
``--transport=fabric`` worker flag; default ``--transport=grpc`` keeps
today's clusters unchanged.

Specs this package implements:
- ``~/sthambha/docs/network-fabric.md`` — overview, three topology
  modes, §4 join handshake, §5 data plane discipline.
- ``~/sthambha/docs/fabric-packet-schema.md`` — wire contract (header
  layout, AES-128-GCM authentication, chunked-datagram rules,
  conformance §11).
- ``~/trisul/decisions/0005-fabric-section-8-ratified.md`` — design
  decisions (per-session keys, two-tier _lite/_fast, counter-snapshot
  observability, mixed-cluster forbidden).
- ``~/trisul/plans/2026-05-26-nakshatra-fabric-lite-prototype-sprint.md``
  — this sprint's phase plan.

Phase A (this commit): packet schema codec only. Transport / join /
backend / link_stats land in later commits in this sprint.
"""
from __future__ import annotations
