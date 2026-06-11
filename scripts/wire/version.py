"""Control-plane wire-version negotiation (v1.0 §7).

The DATA plane already versions hard: fabric/packet.py rejects any
`version_major != 0x01` (ADR 0005 forbids mixed-fabric clusters) — a v2 packet on
a v1 cluster is a typed drop, never silent corruption. Good.

The CONTROL plane did not: registration/handshake/gossip send
`protocol_version="0.1.0"` but **nothing checks it**. That was safe only while
the cluster was hand-wired. The moment P1 discovery lets a stranger running last
month's build try to join, an unchecked version is exactly the silent-corruption
risk §7 warns about.

This module is the ALPN-style negotiation that closes it: each side advertises an
ordered set of supported control-protocol versions; negotiation picks the highest
mutually-supported version, or **cleanly rejects** (typed error) when there is no
overlap. Discovery carries the supported set (NakshatraListing.supported_protocol)
so an incompatible peer is filtered out *before* a join is attempted; the
handshake calls negotiate_or_raise() as the belt-and-braces check at admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

# This build's current control-protocol version and the full set it can speak.
# Bump CONTROL_PROTOCOL_VERSION on a breaking control-plane change; keep older
# versions in SUPPORTED_CONTROL_VERSIONS for as long as back-compat is offered.
CONTROL_PROTOCOL_VERSION: int = 1
SUPPORTED_CONTROL_VERSIONS: tuple[int, ...] = (1,)

# The legacy free-text version that registration/handshake have shipped so far.
# It maps to control-protocol v1 so an un-upgraded peer negotiates cleanly.
LEGACY_VERSION_STRING: str = "0.1.0"


class VersionNegotiationError(Exception):
    """No mutually-supported control-protocol version — admission must refuse.
    Carries the two advertised sets so the rejection is debuggable, never silent."""

    def __init__(self, local: Iterable[int], remote: Iterable[int]):
        self.local = sorted(set(local))
        self.remote = sorted(set(remote))
        super().__init__(
            f"no common control-protocol version: local supports {self.local}, "
            f"peer supports {self.remote}")


def parse_version(v) -> int:
    """Normalise a wire-advertised version into an int. Accepts an int, a bare
    numeric string, or the legacy dotted '0.1.0' (→ 1). Raises ValueError on
    anything else so a garbage version is a hard error, not a 0."""
    if isinstance(v, bool):
        raise ValueError(f"invalid version: {v!r}")
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s == LEGACY_VERSION_STRING:
            return 1
        if s.isdigit():
            return int(s)
        # dotted forms like "1.2.0" → major component
        head = s.split(".", 1)[0]
        if head.isdigit():
            return int(head)
    raise ValueError(f"unparseable protocol version: {v!r}")


def normalize_supported(versions: Optional[Iterable]) -> list[int]:
    """Normalise an advertised supported-set. Empty/None ⇒ legacy [1] (a peer
    that advertises nothing is assumed to be a pre-§7 v1 node)."""
    if not versions:
        return [1]
    out: list[int] = []
    for v in versions:
        try:
            out.append(parse_version(v))
        except ValueError:
            continue
    return sorted(set(out)) or [1]


@dataclass(frozen=True)
class NegotiationResult:
    compatible: bool
    agreed: Optional[int]      # highest mutually-supported version, if any
    local: tuple[int, ...]
    remote: tuple[int, ...]


def negotiate(local: Optional[Iterable] = None,
              remote: Optional[Iterable] = None) -> NegotiationResult:
    """ALPN-style: pick the HIGHEST version both sides support. Default `local`
    is this build's SUPPORTED_CONTROL_VERSIONS. Never raises — returns a result
    with compatible=False when there's no overlap (use negotiate_or_raise for the
    fail-closed variant)."""
    loc = normalize_supported(local if local is not None else SUPPORTED_CONTROL_VERSIONS)
    rem = normalize_supported(remote)
    common = set(loc) & set(rem)
    agreed = max(common) if common else None
    return NegotiationResult(compatible=bool(common), agreed=agreed,
                             local=tuple(loc), remote=tuple(rem))


def negotiate_or_raise(local: Optional[Iterable] = None,
                       remote: Optional[Iterable] = None) -> int:
    """Return the agreed version, or raise VersionNegotiationError. This is the
    fail-closed call the handshake uses at admission."""
    res = negotiate(local, remote)
    if not res.compatible:
        raise VersionNegotiationError(res.local, res.remote)
    return res.agreed  # type: ignore[return-value]


def is_compatible(remote: Optional[Iterable],
                  local: Optional[Iterable] = None) -> bool:
    """True iff there's a mutually-supported version — the cheap pre-join filter
    discovery/routing use to skip a peer we could never speak to."""
    return negotiate(local, remote).compatible
