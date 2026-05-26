"""Fabric packet schema codec — wire contract per
``~/sthambha/docs/fabric-packet-schema.md`` v1.

Implements the 64-byte authenticated header + AES-128-GCM-sealed
payload that flows between Nakshatra workers on the fabric data plane.
This module is a pure codec — no sockets, no I/O, no state beyond the
NamedTuple it returns. Transport (UDP socket lifecycle, chunked
reassembly, monotonic seq counters, ring buffers) belongs to
``fabric/transport.py`` (Phase B).

Conformance § 11 of the schema doc — what this module guarantees on
the receive path:

  1. Drop packets whose first two bytes aren't ``0xF4 0xB1`` (``MagicError``)
  2. Drop packets with ``version_major != 0x01`` (``VersionError``)
  3. Verify AES-128-GCM tag using per-pair key + deterministic
     ``chain_id || seq`` nonce; failure raises ``AuthError``
  7. Use little-endian for all multi-byte integer fields

Conformance items 4–6 (chain_id active, dtype-matches-plan, chunked
reassembly) belong to the transport layer, not this codec.
"""
from __future__ import annotations

import struct
from typing import NamedTuple

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ── Wire constants ──────────────────────────────────────────────────


# Magic bytes at offset 0–1. Specified by the doc as the two-byte
# sequence ``0xF4 0xB1`` (spells "FAB1"). Kept as a raw bytes literal
# rather than a u16 to avoid endianness ambiguity — the conformance
# check is a literal byte-sequence match, not an integer compare.
MAGIC: bytes = b"\xF4\xB1"

VERSION_MAJOR: int = 0x01
VERSION_MINOR: int = 0x00

# Packet types (offset 4).
PACKET_TYPE_FORWARD:  int = 0x01
PACKET_TYPE_FEEDBACK: int = 0x02
PACKET_TYPE_CONTROL:  int = 0x03

# Dtype enum (offset 5). Decorative — receiver already knows the dtype
# from the chain plan, but carried so mismatch surfaces as a typed
# drop counter rather than silent corruption.
DTYPE_FP16:   int = 0x01
DTYPE_BF16:   int = 0x02
DTYPE_FP32:   int = 0x03
DTYPE_INT8:   int = 0x04
DTYPE_OPAQUE: int = 0xFF

# Flag bits (offset 6).
FLAG_TRUNCATED_CONTINUED: int = 0b0000_0001   # bit 0: more chunks coming
FLAG_LAST_IN_STEP:        int = 0b0000_0010   # bit 1: forward only

# Layer-idx sentinel for FEEDBACK packets (schema §4: "FEEDBACK sets to
# 0xFFFFFFFF"). Forward packets carry the real layer index.
LAYER_IDX_FEEDBACK: int = 0xFFFFFFFF

HEADER_SIZE: int = 64
AUTH_TAG_SIZE: int = 16
AAD_SIZE: int = 48                          # first 48 bytes — see §5

# Struct format (schema §4 + §6 little-endian rule).
#
# Total = 2 + 1*6 + 8 + 8 + 4 + 4 + 8 + 8 + 16 = 64 bytes
#
# Field order matches offsets exactly; the unit test in
# tests/test_fabric_packet.py asserts byte offsets so a drift in this
# format string would be caught immediately.
_HEADER_STRUCT = struct.Struct("<2sBBBBBBQQIIQQ16s")
assert _HEADER_STRUCT.size == HEADER_SIZE, (
    f"header struct must be {HEADER_SIZE} bytes, got {_HEADER_STRUCT.size}"
)


# ── Exceptions ──────────────────────────────────────────────────────


class FabricError(Exception):
    """Base for all fabric-codec failures so a generic except clause at
    the transport layer can catch the whole family with one line."""


class MagicError(FabricError):
    """Schema §11 item 1 — packet's first two bytes weren't ``0xF4
    0xB1``. The transport counts these as garbage UDP that found the
    fabric port and drops them without further processing."""


class VersionError(FabricError):
    """Schema §11 item 2 — ``version_major`` was not ``0x01``. ADR 0005
    decision 6 forbids mixed-fabric clusters, so seeing a v2 packet on
    a v1 cluster is a configuration error worth a counter, not a
    decode."""


class AuthError(FabricError):
    """Schema §11 item 3 — AES-128-GCM authentication failed. Could be:
    wrong per-pair key (out-of-sync rekey), replay with mutated bytes,
    or genuine wire corruption. Transport increments
    ``recv_auth_fails`` and drops; payload bytes MUST NOT be logged."""


class TruncatedError(FabricError):
    """The packet is shorter than the 64-byte header. Not in the
    schema's conformance list explicitly but implicit in "drop garbage
    UDP" — a malformed datagram that hits the port but can't even
    accommodate the header is rejected up-front."""


# ── Header NamedTuple ────────────────────────────────────────────────


class FabricHeader(NamedTuple):
    """64-byte fabric packet header, fields in wire order. See
    ``~/sthambha/docs/fabric-packet-schema.md`` §4 for the byte layout.

    The ``auth_tag`` field is part of the NamedTuple for symmetry on
    the unpack side, but :func:`seal` overwrites it during encryption;
    callers passing a header into ``seal`` should set ``auth_tag`` to
    ``b'\\x00' * 16`` (no security impact — the seal computes the
    correct tag regardless of the input value)."""
    magic: bytes              # 2 bytes — MUST be MAGIC for valid packets
    version_major: int        # 1 byte
    version_minor: int        # 1 byte
    packet_type: int          # 1 byte — see PACKET_TYPE_*
    dtype: int                # 1 byte — see DTYPE_*
    flags: int                # 1 byte — see FLAG_*
    reserved: int             # 1 byte — MUST be 0 in v1
    chain_id: int             # 8 bytes — u64 LE
    step_id: int              # 8 bytes — u64 LE
    layer_idx: int            # 4 bytes — u32 LE (FEEDBACK = LAYER_IDX_FEEDBACK)
    seq: int                  # 4 bytes — u32 LE, monotonic per link
    payload_length: int       # 8 bytes — u64 LE
    payload_offset: int       # 8 bytes — u64 LE (chunked datagram offset)
    auth_tag: bytes           # 16 bytes — AES-GCM tag


# ── Pack / unpack ────────────────────────────────────────────────────


def pack_header(h: FabricHeader) -> bytes:
    """Serialize ``h`` to its 64-byte on-wire representation. Pure —
    does not validate the magic / version / reserved fields, on the
    theory that the caller is responsible for constructing valid
    headers and the receive-side parser will catch any drift."""
    return _HEADER_STRUCT.pack(
        h.magic, h.version_major, h.version_minor,
        h.packet_type, h.dtype, h.flags, h.reserved,
        h.chain_id, h.step_id,
        h.layer_idx, h.seq,
        h.payload_length, h.payload_offset,
        h.auth_tag,
    )


def unpack_header(buf: bytes) -> FabricHeader | None:
    """Parse the first 64 bytes of ``buf`` as a fabric header. Returns
    ``None`` on magic mismatch OR ``version_major`` mismatch — those
    are the two "this packet isn't ours, drop it" conditions in the
    schema's conformance §11 items 1+2.

    All other validation (auth, chain-active, dtype-matches-plan,
    chunked reassembly) is the transport layer's job. This parser only
    answers "is this byte string a v1 fabric header?".

    Raises ``TruncatedError`` if ``buf`` is shorter than the 64-byte
    header — the caller passed something that couldn't possibly have
    been a fabric packet."""
    if len(buf) < HEADER_SIZE:
        raise TruncatedError(
            f"buffer too short for fabric header: {len(buf)} < {HEADER_SIZE}"
        )
    fields = _HEADER_STRUCT.unpack_from(buf, 0)
    h = FabricHeader._make(fields)
    if h.magic != MAGIC:
        return None
    if h.version_major != VERSION_MAJOR:
        return None
    return h


# ── Seal / open (authenticated encryption) ──────────────────────────


def _nonce_for(chain_id: int, seq: int) -> bytes:
    """Build the deterministic AES-GCM nonce per schema §5:
    ``chain_id u64 LE || seq u32 LE`` = 12 bytes total.

    Per-pair keys + monotonic per-link ``seq`` guarantees nonce
    uniqueness without per-packet randomness, which is what makes
    AES-GCM safe at fabric speeds (per-call nonce generation +
    transmission would either need a separate field or an extra
    handshake)."""
    return chain_id.to_bytes(8, "little") + seq.to_bytes(4, "little")


def seal(header: FabricHeader, payload: bytes, key: bytes) -> bytes:
    """Authenticate + encrypt ``payload`` under ``key`` per schema §5.

    Returns the full ``HEADER_SIZE + len(payload)`` on-wire packet
    bytes. The returned header has the computed AES-GCM tag written to
    offset 48–63; whatever ``auth_tag`` was on ``header`` is ignored.

    AAD is the first 48 bytes of the header (everything up to the tag
    slot itself). ``payload_length`` SHOULD equal ``len(payload)``;
    senders that ship a chunk smaller than a full activation use
    ``payload_offset`` + ``FLAG_TRUNCATED_CONTINUED`` to indicate
    that, but the field still describes the chunk's bytes.

    The key MUST be exactly 16 bytes (AES-128). Larger keys would
    silently work as AES-192/256 — refuse explicitly to keep the
    fabric to a single ciphersuite per ADR 0005 design lock."""
    if len(key) != 16:
        raise ValueError(
            f"fabric per-pair key must be 16 bytes (AES-128); got {len(key)}"
        )
    # Build the AAD: header bytes 0–47 with tag zeroed. Whatever the
    # caller put in header.auth_tag is irrelevant because we never
    # serialize beyond offset 48 here.
    header_for_aad = header._replace(auth_tag=b"\x00" * AUTH_TAG_SIZE)
    aad = pack_header(header_for_aad)[:AAD_SIZE]
    nonce = _nonce_for(header.chain_id, header.seq)
    sealed = AESGCM(key).encrypt(nonce, payload, aad)
    # AESGCM.encrypt returns ciphertext || tag concatenated.
    ciphertext = sealed[:-AUTH_TAG_SIZE]
    tag = sealed[-AUTH_TAG_SIZE:]
    header_with_tag = header._replace(auth_tag=tag)
    return pack_header(header_with_tag) + ciphertext


def open(packet: bytes, key: bytes) -> tuple[FabricHeader, bytes]:
    """Verify + decrypt a sealed fabric packet. Returns
    ``(header, plaintext_payload)``.

    Raises:
      - :class:`TruncatedError` — packet shorter than 64 bytes
      - :class:`MagicError` — first 2 bytes aren't ``0xF4 0xB1``
      - :class:`VersionError` — ``version_major != 0x01``
      - :class:`AuthError` — AES-GCM tag verification failed

    The caller (transport layer) maps each exception to a counter
    increment per schema §11. Plaintext bytes MUST NOT be logged on
    AuthError; the failure mode is "wrong key, replay, or wire
    corruption", any of which could leak attacker-controlled bytes
    into operator logs."""
    if len(packet) < HEADER_SIZE:
        raise TruncatedError(
            f"packet too short: {len(packet)} < {HEADER_SIZE}"
        )
    if packet[:2] != MAGIC:
        raise MagicError(
            f"bad magic: {packet[:2].hex()} != {MAGIC.hex()}"
        )
    # unpack_header validates version too and would return None if we
    # let it. We want the typed exception here for the counter
    # discrimination, so re-check explicitly before the parse.
    if packet[2] != VERSION_MAJOR:
        raise VersionError(
            f"unsupported version_major: {packet[2]:#04x} (expect "
            f"{VERSION_MAJOR:#04x})"
        )
    header = unpack_header(packet[:HEADER_SIZE])
    # Both validation arms above already returned, so unpack_header
    # cannot legitimately return None here. Belt-and-braces in case a
    # future change adds a None branch we forgot to mirror.
    assert header is not None, "header validation desync"
    ciphertext = packet[HEADER_SIZE:HEADER_SIZE + header.payload_length]
    # Rebuild AAD the same way seal() did: header bytes 0–47 with the
    # tag zeroed. We DON'T use the on-wire tag bytes for AAD; they
    # never participated in the MAC computation.
    header_for_aad = header._replace(auth_tag=b"\x00" * AUTH_TAG_SIZE)
    aad = pack_header(header_for_aad)[:AAD_SIZE]
    nonce = _nonce_for(header.chain_id, header.seq)
    try:
        plaintext = AESGCM(key).decrypt(
            nonce, ciphertext + header.auth_tag, aad,
        )
    except InvalidTag as e:
        raise AuthError("AES-GCM authentication failed") from e
    return header, plaintext
