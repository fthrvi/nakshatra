"""Phase A tests for ``scripts/fabric/packet.py`` — wire-codec only.

Covers the falsifiable checks in
``~/trisul/plans/2026-05-26-nakshatra-fabric-lite-prototype-sprint.md``
Phase A:

  * ``pack`` / ``unpack`` round-trip for FORWARD + FEEDBACK + CONTROL
  * Header offsets match ``fabric-packet-schema.md`` §4 byte-exactly
  * AES-128-GCM seal / open round-trip
  * Tampered-byte rejection
  * Wrong-key rejection
  * Magic mismatch → ``MagicError`` on ``open``; ``None`` on ``unpack_header``
  * v2 packet → ``VersionError`` on ``open``; ``None`` on ``unpack_header``
  * v1.1 future-minor accepted (forward-compat per §7)
  * Truncated buffer rejected
  * Short / oversized key rejected (AES-128 only)
  * Deterministic nonce construction
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fabric import packet as fp  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────


KEY_A = b"A" * 16
KEY_B = b"B" * 16


def _h(*, packet_type=fp.PACKET_TYPE_FORWARD, dtype=fp.DTYPE_FP16,
       flags=0, chain_id=1, step_id=0, layer_idx=0, seq=0,
       payload_length=0, payload_offset=0,
       auth_tag=b"\x00" * 16,
       version_major=fp.VERSION_MAJOR,
       version_minor=fp.VERSION_MINOR) -> fp.FabricHeader:
    """Build a default-valued FabricHeader with overridable fields. Keeps
    each test focused on the one or two fields it actually exercises."""
    return fp.FabricHeader(
        magic=fp.MAGIC,
        version_major=version_major,
        version_minor=version_minor,
        packet_type=packet_type,
        dtype=dtype,
        flags=flags,
        reserved=0,
        chain_id=chain_id,
        step_id=step_id,
        layer_idx=layer_idx,
        seq=seq,
        payload_length=payload_length,
        payload_offset=payload_offset,
        auth_tag=auth_tag,
    )


# ── 1. Header struct size + offsets match the spec byte-exactly ──────


def test_header_size_is_64_bytes():
    """Schema §4 + §6 — 64 bytes, exactly one cache line. Any drift here
    breaks the wire contract on every other implementation."""
    assert fp.HEADER_SIZE == 64
    assert fp._HEADER_STRUCT.size == 64


def test_header_field_offsets_match_spec():
    """Schema §4 — every field at its specified offset. The test packs a
    header with distinctive byte patterns per field and then asserts
    specific bytes are at specific offsets. If a struct format-string
    typo shifts a field by one, this fails loud."""
    h = fp.FabricHeader(
        magic=b"\xF4\xB1",
        version_major=0x01, version_minor=0x00,
        packet_type=0x42,      # offset 4
        dtype=0x43,            # offset 5
        flags=0x44,            # offset 6
        reserved=0x00,         # offset 7
        chain_id=0x0807060504030201,        # offset 8, u64 LE
        step_id=0x1817161514131211,         # offset 16, u64 LE
        layer_idx=0x24232221,               # offset 24, u32 LE
        seq=0x28272625,                     # offset 28, u32 LE
        payload_length=0x3837363534333231,  # offset 32, u64 LE
        payload_offset=0x4847464544434241,  # offset 40, u64 LE
        auth_tag=bytes(range(16)),          # offset 48
    )
    buf = fp.pack_header(h)
    assert len(buf) == 64
    # Magic at 0..2
    assert buf[0:2] == b"\xF4\xB1"
    # Version + packet_type + dtype + flags + reserved at 2..8
    assert buf[2] == 0x01
    assert buf[3] == 0x00
    assert buf[4] == 0x42
    assert buf[5] == 0x43
    assert buf[6] == 0x44
    assert buf[7] == 0x00
    # u64 chain_id LE at 8..16 — least-significant byte first
    assert buf[8:16] == bytes([1, 2, 3, 4, 5, 6, 7, 8])
    # u64 step_id LE at 16..24
    assert buf[16:24] == bytes([0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18])
    # u32 layer_idx LE at 24..28
    assert buf[24:28] == bytes([0x21, 0x22, 0x23, 0x24])
    # u32 seq LE at 28..32
    assert buf[28:32] == bytes([0x25, 0x26, 0x27, 0x28])
    # u64 payload_length LE at 32..40
    assert buf[32:40] == bytes([0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38])
    # u64 payload_offset LE at 40..48
    assert buf[40:48] == bytes([0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48])
    # 16-byte auth_tag at 48..64
    assert buf[48:64] == bytes(range(16))


# ── 2. Pack / unpack round-trip ──────────────────────────────────────


def test_pack_unpack_roundtrip_forward():
    h = _h(packet_type=fp.PACKET_TYPE_FORWARD, dtype=fp.DTYPE_FP16,
            chain_id=42, step_id=7, layer_idx=14, seq=99,
            payload_length=16384, payload_offset=0,
            auth_tag=b"T" * 16)
    out = fp.unpack_header(fp.pack_header(h))
    assert out == h


def test_pack_unpack_roundtrip_feedback():
    """FEEDBACK packets carry a 4-byte token and set layer_idx to the
    sentinel 0xFFFFFFFF (schema §4)."""
    h = _h(packet_type=fp.PACKET_TYPE_FEEDBACK,
            dtype=fp.DTYPE_INT8,
            layer_idx=fp.LAYER_IDX_FEEDBACK,
            payload_length=4)
    out = fp.unpack_header(fp.pack_header(h))
    assert out == h
    assert out.layer_idx == 0xFFFFFFFF


def test_pack_unpack_roundtrip_control():
    """CONTROL packets carry an opaque payload (rekey notifications,
    drain signals). dtype = OPAQUE per the schema §4 table."""
    h = _h(packet_type=fp.PACKET_TYPE_CONTROL, dtype=fp.DTYPE_OPAQUE)
    out = fp.unpack_header(fp.pack_header(h))
    assert out == h


# ── 3. unpack_header magic + version handling ────────────────────────


def test_unpack_header_magic_mismatch_returns_none():
    """Schema §11 item 1 — garbage UDP that landed on the port gets
    silently dropped, NOT raised. unpack_header is the parser layer;
    raising belongs to ``open`` (which gets the typed-error treatment)."""
    h = _h()
    buf = bytearray(fp.pack_header(h))
    buf[0] = 0xDE  # corrupt magic
    assert fp.unpack_header(bytes(buf)) is None


def test_unpack_header_v2_returns_none():
    """Schema §7 — v1 receivers MUST drop v2 packets. Mixed-fabric
    clusters are forbidden by ADR 0005 decision 6, so this case is a
    configuration error, but a worker still has to refuse the packet
    cleanly without crashing the parser."""
    h = _h(version_major=0x02)
    assert fp.unpack_header(fp.pack_header(h)) is None


def test_unpack_header_accepts_future_minor_version():
    """Schema §7 — v1.x packets are forward-compatible with v1.0
    receivers: receivers ignore trailing fields they don't recognise.
    A v1.5 packet must still parse as a valid v1 header here."""
    h = _h(version_minor=0x05)
    out = fp.unpack_header(fp.pack_header(h))
    assert out is not None
    assert out.version_major == 0x01
    assert out.version_minor == 0x05


def test_unpack_header_truncated_buffer_raises():
    """Schema §11 implicit — a UDP datagram too short to even hold a
    header isn't a valid fabric packet. Raise rather than return None
    because the cause is qualitatively different from "wrong magic"
    (which is "this is some other protocol's packet")."""
    short = fp.pack_header(_h())[:32]
    with pytest.raises(fp.TruncatedError):
        fp.unpack_header(short)


# ── 4. seal / open round-trip + AAD coverage ─────────────────────────


def test_seal_open_roundtrip_small_payload():
    payload = b"hello fabric"
    h = _h(payload_length=len(payload))
    sealed = fp.seal(h, payload, KEY_A)
    assert len(sealed) == fp.HEADER_SIZE + len(payload)
    out_h, out_p = fp.open(sealed, KEY_A)
    assert out_p == payload
    # The on-wire header gets the computed tag written in; everything
    # else round-trips identically.
    assert out_h.chain_id == h.chain_id
    assert out_h.seq == h.seq
    assert out_h.payload_length == h.payload_length
    assert out_h.auth_tag != b"\x00" * 16  # seal wrote a real tag


def test_seal_open_roundtrip_typical_activation_size():
    """A Llama-3.3 70B hidden state at fp16 is 8192 dims × 2 bytes =
    16384 payload bytes. Make sure seal/open handle the realistic
    case, not just a tiny test payload."""
    payload = b"\xAB" * 16384
    h = _h(payload_length=len(payload), chain_id=1234567890,
            step_id=42, layer_idx=39, seq=100)
    sealed = fp.seal(h, payload, KEY_A)
    out_h, out_p = fp.open(sealed, KEY_A)
    assert out_p == payload
    assert out_h.layer_idx == 39


def test_open_wrong_key_raises_auth_error():
    """Per-pair keys make AES-GCM safe at fabric speeds. If a sender
    rotates its key but the receiver hasn't fetched the new keyring
    yet, every packet fails auth — which the transport counts and
    eventually triggers a /fabric/rekey re-query."""
    payload = b"secret activation"
    sealed = fp.seal(_h(payload_length=len(payload)), payload, KEY_A)
    with pytest.raises(fp.AuthError):
        fp.open(sealed, KEY_B)


def test_open_tampered_payload_byte_raises_auth_error():
    """AES-GCM is an AEAD — flipping a single ciphertext bit fails the
    tag verification. Schema §11 item 3."""
    payload = b"X" * 64
    sealed = bytearray(fp.seal(_h(payload_length=len(payload)),
                                  payload, KEY_A))
    sealed[fp.HEADER_SIZE + 10] ^= 0x01  # flip a byte deep in the ciphertext
    with pytest.raises(fp.AuthError):
        fp.open(bytes(sealed), KEY_A)


def test_open_tampered_header_aad_byte_raises_auth_error():
    """AAD covers header bytes 0..48. Mutating a header field (e.g.
    chain_id) without re-computing the tag must fail auth, because a
    plain receiver would otherwise accept a packet rerouted to a
    different chain."""
    payload = b"Y" * 64
    sealed = bytearray(fp.seal(_h(payload_length=len(payload),
                                    chain_id=1), payload, KEY_A))
    # Flip a byte inside chain_id (offset 8..16) without touching the
    # tag at offset 48..64.
    sealed[12] ^= 0x01
    with pytest.raises(fp.AuthError):
        fp.open(bytes(sealed), KEY_A)


def test_open_bad_magic_raises_magic_error():
    """Schema §11 item 1 (typed-error variant at the ``open`` layer)."""
    sealed = bytearray(fp.seal(_h(payload_length=0), b"", KEY_A))
    sealed[0] = 0xDE
    with pytest.raises(fp.MagicError):
        fp.open(bytes(sealed), KEY_A)


def test_open_v2_raises_version_error():
    """Schema §11 item 2 (typed-error variant at the ``open`` layer).
    Distinguishing this from MagicError lets the transport increment a
    separate counter so operators can tell "wrong protocol" from
    "wrong version" at debug time."""
    # We have to build the v2 packet bytes manually — seal() would
    # produce a v1 packet because that's what _h defaults to, but the
    # caller could legitimately have passed an unexpected version.
    h = _h(version_major=0x02, payload_length=0)
    sealed = fp.seal(h, b"", KEY_A)
    with pytest.raises(fp.VersionError):
        fp.open(sealed, KEY_A)


def test_open_truncated_packet_raises_truncated_error():
    sealed = fp.seal(_h(payload_length=0), b"", KEY_A)
    with pytest.raises(fp.TruncatedError):
        fp.open(sealed[:30], KEY_A)


# ── 5. Key size policy (AES-128 only) ────────────────────────────────


def test_seal_rejects_non_16_byte_key():
    """ADR 0005 + schema §5 — AES-128-GCM only. Pillar issues 16-byte
    keys; refuse anything else explicitly rather than silently
    promoting to AES-192/256."""
    with pytest.raises(ValueError, match="16 bytes"):
        fp.seal(_h(payload_length=0), b"", b"short")
    with pytest.raises(ValueError, match="16 bytes"):
        fp.seal(_h(payload_length=0), b"", b"A" * 32)


# ── 6. Deterministic nonce ───────────────────────────────────────────


def test_nonce_construction_is_chain_id_then_seq():
    """Schema §5 — nonce = chain_id u64 LE || seq u32 LE = 12 bytes.
    Asserting the construction explicitly so a future refactor that
    swaps to a different scheme would break the cross-repo wire
    contract loudly. Other v1 implementations rely on this exact
    layout."""
    n = fp._nonce_for(chain_id=0x0102030405060708, seq=0x11121314)
    assert len(n) == 12
    assert n[:8] == bytes([0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01])
    assert n[8:] == bytes([0x14, 0x13, 0x12, 0x11])


def test_nonce_differs_across_seq_for_same_chain():
    """Sanity: the per-(link, seq) nonce uniqueness property AES-GCM
    requires. The schema relies on this implicitly via "seq monotonic
    per link"; assert it here so a future refactor can't accidentally
    nullify the per-seq part."""
    n1 = fp._nonce_for(chain_id=42, seq=1)
    n2 = fp._nonce_for(chain_id=42, seq=2)
    assert n1 != n2


# ── 7. Seal preserves header fields end-to-end ───────────────────────


def test_seal_writes_tag_but_preserves_all_other_fields():
    """seal() should ONLY modify the auth_tag slot. A regression that
    silently mutated, say, chain_id during seal would corrupt the
    receive-side lookup without surfacing as an auth failure (because
    the tag was computed over the post-mutation bytes)."""
    payload = b"Z" * 32
    h_in = _h(packet_type=fp.PACKET_TYPE_FORWARD,
               chain_id=999, step_id=88, seq=7,
               payload_length=len(payload),
               auth_tag=b"\x00" * 16)
    sealed = fp.seal(h_in, payload, KEY_A)
    h_out = fp.unpack_header(sealed[:64])
    assert h_out is not None
    # Every field except auth_tag must match.
    assert h_out._replace(auth_tag=b"\x00" * 16) == h_in
    # And the tag was actually computed (not left zero).
    assert h_out.auth_tag != b"\x00" * 16
