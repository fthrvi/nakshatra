"""Phase B tests for ``scripts/fabric/transport.py`` — UDP transport,
chunking, and reassembly. Sits on top of the Phase A ``fabric.packet``
codec and below the Phase D Forward/Inference shim.

Covers the falsifiable checks from the sprint plan's Phase B:

  * MTU math: max chunk payload subtracts IP+UDP+fabric overhead
  * Sender chunk count matches expected
  * send_seq is monotonic per link, doesn't reset on send failure
  * Single-datagram round-trip (small payload, no chunking)
  * 16 KB at MTU 1500 chunks correctly and reassembles
  * Out-of-order chunk arrival still reassembles
  * Deliberately-dropped chunk → recv_gaps increments
  * Oversize payload → recv_dropped_alloc
  * Wrong-key incoming → recv_auth_fails
  * Magic-mismatch garbage → recv_dropped_magic
  * Pinned-peer mismatch → datagram silently dropped
  * Counter set matches schema §9 names
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fabric import packet as fp  # noqa: E402
from fabric import transport as ft  # noqa: E402


KEY_A = b"A" * 16
KEY_B = b"B" * 16


# ── helpers ──────────────────────────────────────────────────────────


def _link_pair(*, mtu=1500, key=KEY_A, chain_id=42,
                 max_slot_bytes=ft.DEFAULT_MAX_SLOT_BYTES,
                 max_slots=ft.DEFAULT_MAX_SLOTS):
    """Build two FabricLinks bound to ephemeral 127.0.0.1 UDP ports,
    each pointing at the other. Returns ``(a, b)`` where a.send goes
    to b's socket and vice versa. Caller closes both."""
    sock_a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_a.bind(("127.0.0.1", 0))
    sock_b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_b.bind(("127.0.0.1", 0))
    a = ft.FabricLink(sock_a, sock_b.getsockname(), key, chain_id,
                       mtu=mtu, max_slot_bytes=max_slot_bytes,
                       max_slots=max_slots)
    b = ft.FabricLink(sock_b, sock_a.getsockname(), key, chain_id,
                       mtu=mtu, max_slot_bytes=max_slot_bytes,
                       max_slots=max_slots)
    return a, b


# ── 1. MTU math ──────────────────────────────────────────────────────


def test_max_chunk_payload_for_mtu_1500():
    """At MTU 1500: 1500 - 28 (IP+UDP) - 64 (fabric) = 1408."""
    assert ft.max_chunk_payload_for_mtu(1500) == 1408


def test_max_chunk_payload_for_mtu_9000_jumbo():
    """Mode A lab cluster on jumbo frames. 9000 - 28 - 64 = 8908."""
    assert ft.max_chunk_payload_for_mtu(9000) == 8908


def test_max_chunk_payload_for_mtu_too_small():
    """A pathological MTU below the IP+UDP+fabric overhead must raise.
    Operator-typo guard — a misconfigured ``--fabric-mtu 50`` would
    otherwise compute a negative chunk size and silently break."""
    with pytest.raises(ValueError, match="too small"):
        ft.max_chunk_payload_for_mtu(50)


# ── 2. Sender path ───────────────────────────────────────────────────


def test_build_chunks_single_datagram_for_small_payload():
    """Payload that fits in one chunk yields one datagram with both
    LAST_IN_STEP set and TRUNCATED_CONTINUED clear (it IS the last,
    AND there's nothing more coming)."""
    a, b = _link_pair()
    try:
        chunks = a.build_chunks(b"hello", step_id=1, layer_idx=2)
        assert len(chunks) == 1
        header = fp.unpack_header(chunks[0][:64])
        assert header is not None
        assert header.flags & fp.FLAG_LAST_IN_STEP
        assert not (header.flags & fp.FLAG_TRUNCATED_CONTINUED)
        assert header.payload_offset == 0
        assert header.payload_length == len(b"hello")
    finally:
        a.close()
        b.close()


def test_build_chunks_splits_at_mtu_boundary():
    """16 KB activation at MTU 1500 chunks to ceil(16384 / 1408) = 12
    datagrams. The last chunk has LAST_IN_STEP set + chunked-flag
    clear; all preceding chunks have TRUNCATED_CONTINUED set +
    LAST_IN_STEP clear."""
    a, b = _link_pair(mtu=1500)
    try:
        payload = b"\xAB" * 16384
        chunks = a.build_chunks(payload, step_id=0, layer_idx=39)
        assert len(chunks) == 12
        # Earlier chunks
        for i, c in enumerate(chunks[:-1]):
            h = fp.unpack_header(c[:64])
            assert h is not None
            assert h.flags & fp.FLAG_TRUNCATED_CONTINUED, f"chunk {i}"
            assert not (h.flags & fp.FLAG_LAST_IN_STEP), f"chunk {i}"
            assert h.payload_offset == i * 1408
        # Last chunk
        last = fp.unpack_header(chunks[-1][:64])
        assert last is not None
        assert last.flags & fp.FLAG_LAST_IN_STEP
        assert not (last.flags & fp.FLAG_TRUNCATED_CONTINUED)
        assert last.payload_offset == 11 * 1408   # 15488
    finally:
        a.close()
        b.close()


def test_send_seq_monotonic_across_chunks():
    """schema §5 — nonce uniqueness depends on seq monotonicity per
    link. A 12-chunk send consumes seqs 0..11 in order."""
    a, b = _link_pair(mtu=1500)
    try:
        chunks = a.build_chunks(b"X" * 16384)
        seqs = [fp.unpack_header(c[:64]).seq for c in chunks]
        assert seqs == list(range(12))
        assert a.send_seq == 12
    finally:
        a.close()
        b.close()


def test_send_seq_does_not_reset_across_calls():
    """seq is per-link-per-rekey, not per-send. Two separate send()
    calls each consume distinct seq ranges."""
    a, b = _link_pair()
    try:
        a.build_chunks(b"first")     # consumes seq 0
        a.build_chunks(b"second")    # consumes seq 1
        assert a.send_seq == 2
    finally:
        a.close()
        b.close()


# ── 3. Round-trip — single datagram ──────────────────────────────────


def test_roundtrip_single_datagram():
    """Smoke: a.send(payload) → b.recv() → bytes match. Uses real
    127.0.0.1 UDP sockets so we cover the actual recvfrom + auth path
    end-to-end."""
    a, b = _link_pair()
    try:
        a.send(b"the wormhole works", step_id=1, layer_idx=7)
        result = b.recv(timeout=1.0)
        assert result is not None
        header, plaintext = result
        assert plaintext == b"the wormhole works"
        assert header.step_id == 1
        assert header.layer_idx == 7
        assert b.counters["recv_packets"] == 1
        assert a.counters["sent_packets"] == 1
    finally:
        a.close()
        b.close()


def test_roundtrip_chunked_16kb_at_mtu_1500():
    """The plan's headline check — 16 KB activation chunks to 12
    datagrams and reassembles to the original bytes."""
    a, b = _link_pair(mtu=1500)
    try:
        payload = bytes(i & 0xFF for i in range(16384))  # distinctive pattern
        n = a.send(payload, step_id=42, layer_idx=39)
        assert n == 12
        result = b.recv(timeout=2.0)
        assert result is not None
        header, plaintext = result
        assert plaintext == payload
        assert header.payload_length == len(payload)
        assert b.counters["recv_packets"] == 12
        assert b.counters["recv_gaps"] == 0
    finally:
        a.close()
        b.close()


def test_roundtrip_chunked_at_mtu_9000_jumbo():
    """Mode A lab cluster path. Same 16 KB activation needs only 2
    chunks at jumbo MTU."""
    a, b = _link_pair(mtu=9000)
    try:
        payload = b"Q" * 16384
        n = a.send(payload)
        assert n == 2
        result = b.recv(timeout=2.0)
        assert result is not None
        _, plaintext = result
        assert plaintext == payload
    finally:
        a.close()
        b.close()


# ── 4. Out-of-order arrival ──────────────────────────────────────────


def test_out_of_order_chunks_still_reassemble():
    """UDP doesn't promise order; reassembly is keyed on
    ``payload_offset``, not arrival order. Send chunks reversed; recv
    must still produce the original bytes intact."""
    a, b = _link_pair(mtu=1500)
    try:
        payload = bytes(range(256)) * 30           # 7680 bytes — 6 chunks
        chunks = a.build_chunks(payload)
        assert len(chunks) == 6
        # Send reversed — last chunk first, first chunk last.
        for c in reversed(chunks):
            a.sock.sendto(c, a.peer_addr)
            a.counters["sent_packets"] += 1
            a.counters["sent_bytes"] += len(c)
        result = b.recv(timeout=2.0)
        assert result is not None
        _, plaintext = result
        assert plaintext == payload
    finally:
        a.close()
        b.close()


# ── 5. Loss detection — recv_gaps ────────────────────────────────────


def test_dropped_middle_chunk_increments_recv_gaps():
    """Skip a chunk in the middle of a multi-chunk send. The receiver
    sees an offset coverage hole AND a seq gap; ``recv_gaps`` counts
    the missing seqs; recv() times out because the assembly never
    completes (LAST_IN_STEP arrives but the hole stays unfilled)."""
    a, b = _link_pair(mtu=1500)
    try:
        payload = b"M" * 8192                       # 6 chunks at MTU 1500
        chunks = a.build_chunks(payload)
        assert len(chunks) == 6
        # Send all but chunk index 2 (seq 2).
        for i, c in enumerate(chunks):
            if i == 2:
                continue
            a.sock.sendto(c, a.peer_addr)
            a.counters["sent_packets"] += 1
            a.counters["sent_bytes"] += len(c)
        # Recv should block until timeout (LAST_IN_STEP seen but slot
        # coverage incomplete).
        result = b.recv(timeout=0.5)
        assert result is None
        # The seq gap from 1 → 3 (skipping 2) is one missing seq.
        assert b.counters["recv_gaps"] == 1
        assert b.counters["recv_packets"] == 5
    finally:
        a.close()
        b.close()


def test_seq_gap_counted_exactly():
    """A two-seq skip (drop two consecutive packets) counts 2 gaps."""
    a, b = _link_pair(mtu=1500)
    try:
        payload = b"G" * (1408 * 5)                  # exactly 5 chunks
        chunks = a.build_chunks(payload)
        # Send chunks 0, 3, 4 (skipping 1 + 2).
        for i in (0, 3, 4):
            a.sock.sendto(chunks[i], a.peer_addr)
        # Drain the recvable datagrams; recv() will time out because
        # the assembly is incomplete (offsets 1408..4224 are missing).
        b.recv(timeout=0.5)
        assert b.counters["recv_gaps"] == 2
    finally:
        a.close()
        b.close()


# ── 6. Capacity limits — recv_dropped_alloc ──────────────────────────


def test_oversize_single_chunk_drops_with_alloc_counter():
    """A single chunk whose payload_length exceeds max_slot_bytes is
    refused up front — we don't allocate a never-completable slot.
    Built by crafting a packet manually because the normal send path
    won't construct one this big at typical MTU."""
    a, b = _link_pair(mtu=9000, max_slot_bytes=4096)
    try:
        big = b"X" * 8000
        chunks = a.build_chunks(big)
        # Single chunk at jumbo MTU; chunk's payload_length = 8000 >
        # max_slot_bytes 4096 → drop on receive.
        assert len(chunks) == 1
        a.sock.sendto(chunks[0], a.peer_addr)
        result = b.recv(timeout=0.5)
        assert result is None
        assert b.counters["recv_dropped_alloc"] == 1
    finally:
        a.close()
        b.close()


def test_concurrent_assemblies_exceed_max_slots():
    """If too many distinct ``(step_id, layer_idx)`` assemblies are in
    flight at once, new ones are refused via ``recv_dropped_alloc``.
    Better than evicting an in-flight slot (which would silently
    corrupt one chain's activation while another's progresses)."""
    a, b = _link_pair(mtu=1500, max_slots=2)
    try:
        # Open 3 distinct assemblies, each by sending only a non-last
        # chunk so they stay in the slot map.
        payload = b"P" * 2816                        # 2 chunks at MTU 1500
        for step in (1, 2, 3):
            chunks = a.build_chunks(payload, step_id=step)
            # Send only the first chunk; LAST_IN_STEP is on chunk 1
            # so the slot stays open after seeing chunk 0.
            a.sock.sendto(chunks[0], a.peer_addr)
        # Drain everything — recv times out (no assembly completes).
        b.recv(timeout=0.3)
        # Third assembly's first chunk hits max_slots and counts as
        # a drop.
        assert b.counters["recv_dropped_alloc"] >= 1
    finally:
        a.close()
        b.close()


# ── 7. Auth + crypto failure paths ───────────────────────────────────


def test_wrong_key_increments_recv_auth_fails():
    """Sender encrypts under KEY_A, receiver pinned to KEY_B. Every
    packet drops at AES-GCM tag verify; recv times out; counter
    increments per dropped packet."""
    sock_a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_a.bind(("127.0.0.1", 0))
    sock_b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_b.bind(("127.0.0.1", 0))
    a = ft.FabricLink(sock_a, sock_b.getsockname(), KEY_A, chain_id=1)
    b = ft.FabricLink(sock_b, sock_a.getsockname(), KEY_B, chain_id=1)
    try:
        a.send(b"will not decrypt")
        result = b.recv(timeout=0.5)
        assert result is None
        assert b.counters["recv_auth_fails"] == 1
    finally:
        a.close()
        b.close()


def test_garbage_udp_increments_recv_dropped_magic():
    """A datagram lacking the 0xF4 0xB1 magic — e.g., some other
    protocol's stray packet landing on the fabric port — drops
    silently. ``recv_dropped_magic`` is a private counter (not in
    schema §9) but useful for operator debug."""
    a, b = _link_pair()
    try:
        a.sock.sendto(b"\xDE\xAD\xBE\xEF" + b"\x00" * 100, a.peer_addr)
        result = b.recv(timeout=0.3)
        assert result is None
        assert b.counters["recv_dropped_magic"] == 1
    finally:
        a.close()
        b.close()


def test_datagram_from_wrong_source_silently_ignored():
    """The link pins ``peer_addr``. A datagram from a different source
    is dropped without even attempting auth — defense in depth against
    spoofed UDP, plus a small CPU savings (AES-GCM verify is wasted
    cycles on packets we'd never trust anyway).

    Counters reflect: no recv_packets increment, no auth_fails."""
    a, b = _link_pair()
    intruder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    intruder.bind(("127.0.0.1", 0))
    try:
        intruder.sendto(b"junk from elsewhere", b.sock.getsockname())
        result = b.recv(timeout=0.3)
        assert result is None
        # Crucially: didn't even count toward recv_packets because we
        # filter on source address before unpacking.
        assert b.counters["recv_packets"] == 0
        assert b.counters["recv_auth_fails"] == 0
    finally:
        intruder.close()
        a.close()
        b.close()


# ── 8. Construction guards + counter shape ───────────────────────────


def test_link_rejects_non_16_byte_key():
    """Same policy as fabric.packet.seal — AES-128 only."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    try:
        with pytest.raises(ValueError, match="16 bytes"):
            ft.FabricLink(sock, ("127.0.0.1", 0), b"too short", 1)
    finally:
        sock.close()


def test_counter_set_matches_schema_names():
    """Schema §9 names the counters that ship over /fabric/link_stats
    in Phase E. Asserting the names now means Phase E's payload
    shape is byte-identical without further code changes here."""
    a, _ = _link_pair()
    try:
        names = set(a.counters.keys())
        for required in ("sent_packets", "sent_bytes",
                          "recv_packets", "recv_bytes",
                          "recv_auth_fails", "recv_gaps",
                          "recv_dropped_alloc", "recv_dropped_dtype"):
            assert required in names
    finally:
        a.close()
