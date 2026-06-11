"""Control-version handshake glue (v1.0 §7) — the live-path check.

The gRPC `Info` RPC already carries `protocol_version` (legacy "0.1.0") and a
repeated `protocol_capabilities`. This module maps those wire fields to the
negotiation in wire/version.py so the chain handshake actually *enforces* a
mutually-supported control version instead of sending it and ignoring it.

  • A worker advertises its supported versions as `control/vN` capability tokens
    (additive — alongside "streaming", "rpc_push", …).
  • The client, when it reads a worker's Info during chain setup, calls
    negotiate_handshake() and refuses to build a chain with an incompatible peer.

Back-compat: a pre-§7 worker advertises no `control/vN` token; its bare
`protocol_version` ("0.1.0") maps to control v1, which this build supports — so
the existing v0.1 cluster keeps working untouched.
"""
from __future__ import annotations

from typing import Iterable, Optional

if __package__:
    from .version import (SUPPORTED_CONTROL_VERSIONS, parse_version,
                          negotiate_or_raise)
else:  # pragma: no cover
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from wire.version import (SUPPORTED_CONTROL_VERSIONS, parse_version,
                              negotiate_or_raise)

CAP_CONTROL_PREFIX = "control/v"   # e.g. "control/v1"


def advertise_capabilities() -> list[str]:
    """Capability tokens this build advertises for control-version negotiation.
    Appended to a worker's Info.protocol_capabilities."""
    return [f"{CAP_CONTROL_PREFIX}{v}" for v in SUPPORTED_CONTROL_VERSIONS]


def peer_supported_versions(protocol_version: str,
                            capabilities: Iterable[str]) -> list[int]:
    """Recover a peer's supported control versions from its Info fields.
    Prefers explicit `control/vN` capability tokens; falls back to the bare
    `protocol_version` string (legacy "0.1.0" → [1]) when none are present."""
    versions: list[int] = []
    for cap in capabilities or []:
        if isinstance(cap, str) and cap.startswith(CAP_CONTROL_PREFIX):
            tail = cap[len(CAP_CONTROL_PREFIX):]
            if tail.isdigit():
                versions.append(int(tail))
    if versions:
        return sorted(set(versions))
    try:
        return [parse_version(protocol_version)]
    except ValueError:
        return [1]  # unparseable legacy field ⇒ assume v1


def negotiate_handshake(protocol_version: str, capabilities: Iterable[str],
                        local: Optional[Iterable] = None) -> int:
    """Agree a control version with a peer from its Info, or raise
    VersionNegotiationError (caller refuses the chain). `local` defaults to this
    build's SUPPORTED_CONTROL_VERSIONS."""
    remote = peer_supported_versions(protocol_version, capabilities)
    return negotiate_or_raise(
        local if local is not None else SUPPORTED_CONTROL_VERSIONS, remote)
