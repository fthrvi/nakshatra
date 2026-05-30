"""Phase A.1 tests for ``scripts/fabric/shm_ring.py`` — SPSC ring
buffer over an mmap'd file. Foundation for Phase A.2's DaemonClient
shm-mode wiring + Phase A.3's C++ daemon patches.

Covers:
  * create/attach round-trip (sane header)
  * MagicError / VersionError on attach to bad file
  * Single message write→read
  * Multiple messages preserve FIFO order
  * Ring wrap: write past capacity boundary, payload intact across
    the discontinuity
  * Wrap with the LENGTH PREFIX itself straddling the boundary
    (the trickiest case — easy to silently corrupt)
  * Empty ring read returns None
  * Full ring write returns False (caller retries)
  * Oversize message raises CapacityError
  * Partial commit invisible to reader (producer wrote bytes but
    hasn't advanced cursor → reader still sees None)
  * Threaded producer / consumer round-trip
  * Cursor monotonicity across many writes (no premature wrap)
"""
from __future__ import annotations

import struct
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fabric.shm_ring import (   # noqa: E402
    HEADER_SIZE, MAGIC, VERSION_MAJOR,
    CapacityError, MagicError, ShmRing, ShmRingError, VersionError,
)


# ── helpers ──────────────────────────────────────────────────────────


def _ring_path(tmp_path) -> Path:
    return tmp_path / "test.ring"


# ── 1. create / attach ──────────────────────────────────────────────


def test_create_writes_valid_header(tmp_path):
    """create() truncates the file + writes a v1 header with the
    correct magic, version, capacity, and zeroed cursors."""
    path = _ring_path(tmp_path)
    with ShmRing.create(path, capacity=1024) as r:
        assert r.capacity == 1024
        assert r.available_to_read == 0
        assert r.available_to_write == 1024


def test_attach_validates_magic(tmp_path):
    """attach() refuses a file whose first 4 bytes aren't ``RING`` —
    catches "wrong file" + "corrupt header" + "stale region from a
    previous incompatible run"."""
    path = _ring_path(tmp_path)
    path.write_bytes(b"NOPE" + b"\x00" * (HEADER_SIZE - 4) + b"\x00" * 64)
    with pytest.raises(MagicError):
        ShmRing.attach(path)


def test_attach_refuses_unsupported_version(tmp_path):
    """attach() refuses a v2 ring on a v1 client — same forbidden-
    mixed-version stance as ADR 0005 decision 6 for the wire schema."""
    path = _ring_path(tmp_path)
    # Build a header with bumped version_major.
    payload_area = b"\x00" * 1024
    bogus_v2 = struct.pack("<4sIQQQ", MAGIC, 2, 1024, 0, 0) \
        + b"\x00" * (HEADER_SIZE - 32) \
        + payload_area
    path.write_bytes(bogus_v2)
    with pytest.raises(VersionError):
        ShmRing.attach(path)


def test_attach_round_trips_with_create(tmp_path):
    """Producer creates, consumer attaches, both see the same
    capacity + can interact through the same memory."""
    path = _ring_path(tmp_path)
    producer = ShmRing.create(path, capacity=2048)
    consumer = ShmRing.attach(path)
    try:
        assert producer.capacity == consumer.capacity == 2048
        assert producer.write_message(b"hello")
        # Consumer sees it.
        assert consumer.read_message() == b"hello"
        assert consumer.read_message() is None
    finally:
        consumer.close()
        producer.close()


def test_create_rejects_zero_or_negative_capacity(tmp_path):
    path = _ring_path(tmp_path)
    with pytest.raises(ValueError):
        ShmRing.create(path, capacity=0)
    with pytest.raises(ValueError):
        ShmRing.create(path, capacity=-1)


# ── 2. Basic write / read ───────────────────────────────────────────


def test_single_message_round_trip(tmp_path):
    path = _ring_path(tmp_path)
    with ShmRing.create(path, capacity=1024) as r:
        assert r.write_message(b"the wormhole works")
        out = r.read_message()
        assert out == b"the wormhole works"
        assert r.available_to_read == 0


def test_multiple_messages_preserve_fifo_order(tmp_path):
    path = _ring_path(tmp_path)
    with ShmRing.create(path, capacity=1024) as r:
        for i in range(10):
            assert r.write_message(f"msg-{i:02d}".encode())
        for i in range(10):
            assert r.read_message() == f"msg-{i:02d}".encode()
        assert r.read_message() is None


def test_empty_ring_read_returns_none(tmp_path):
    path = _ring_path(tmp_path)
    with ShmRing.create(path, capacity=1024) as r:
        assert r.read_message() is None


def test_zero_byte_payload_supported(tmp_path):
    """A zero-byte payload is a valid framed message (only the
    length prefix). Useful for CMD-style messages with no payload."""
    path = _ring_path(tmp_path)
    with ShmRing.create(path, capacity=64) as r:
        assert r.write_message(b"")
        assert r.read_message() == b""


# ── 3. Capacity + framing ───────────────────────────────────────────


def test_full_ring_write_returns_false(tmp_path):
    """When the ring lacks space for the new message, write_message
    returns False — caller retries (Phase A.2's DaemonClient will
    spin or sleep). Does NOT raise; full-ring is expected back-
    pressure under producer-faster-than-consumer."""
    path = _ring_path(tmp_path)
    with ShmRing.create(path, capacity=64) as r:
        # Fill the ring with messages until write_message refuses.
        msg = b"X" * 16                          # 4-byte len + 16 = 20 bytes
        written = 0
        while r.write_message(msg):
            written += 1
            if written > 100:
                pytest.fail("ring should be full long before this")
        # We pushed at most floor(64 / 20) = 3 messages.
        assert written >= 1
        assert written <= 3
        # Subsequent writes still refuse until the consumer drains.
        assert not r.write_message(msg)
        # After a single read, one slot free again.
        assert r.read_message() == msg
        assert r.write_message(msg)


def test_oversize_message_raises_capacity_error(tmp_path):
    """A message that's larger than the ring's payload area can NEVER
    fit, even if the ring is empty. Raise rather than return False —
    that would loop forever in a caller using a retry pattern."""
    path = _ring_path(tmp_path)
    with ShmRing.create(path, capacity=64) as r:
        with pytest.raises(CapacityError):
            r.write_message(b"X" * 128)


def test_message_exactly_at_capacity_minus_prefix_fits(tmp_path):
    """Edge: a message of size (capacity - 4) is the largest single
    write that fits. Confirms the off-by-one math in CapacityError /
    available_to_write."""
    path = _ring_path(tmp_path)
    with ShmRing.create(path, capacity=64) as r:
        assert r.write_message(b"X" * (64 - 4))
        assert r.read_message() == b"X" * (64 - 4)


# ── 4. Wrap-around ──────────────────────────────────────────────────


def test_wrap_around_payload_split(tmp_path):
    """Producer writes enough messages to push the cursor past
    capacity; the next write's payload straddles the wrap boundary
    in the underlying memory. Reader must reassemble it correctly."""
    path = _ring_path(tmp_path)
    # Capacity 256 — small enough to wrap quickly with our test
    # messages (each 20 bytes wire-side).
    with ShmRing.create(path, capacity=256) as r:
        msg = b"ABCDEFGHIJKLMNOP"            # 16 bytes; wire = 20
        # Push 10 messages through; ring wraps at message ~13.
        for i in range(40):
            assert r.write_message(msg + bytes([i & 0xFF]))
            got = r.read_message()
            assert got == msg + bytes([i & 0xFF])


def test_wrap_with_length_prefix_straddling_boundary(tmp_path):
    """The trickiest case: the 4-byte length prefix itself spans the
    wrap boundary (3 bytes before, 1 byte after). Easy to silently
    corrupt with off-by-one math; explicit test guards it."""
    path = _ring_path(tmp_path)
    # Capacity such that after a sequence of pre-fills, the next
    # write's length prefix straddles. Compute deliberately.
    cap = 64
    with ShmRing.create(path, capacity=cap) as r:
        # Write 3 bytes from the END of the buffer first by:
        # 1. Fill so that read_cursor advances past 0
        # 2. Resulting available room places a length prefix near
        #    capacity boundary
        # Construct: write 13-byte payload (wire 17) then read.
        # After this, both cursors = 17. Next write at offset 17.
        # Write 40-byte payload (wire 44) → fills 17..61.
        # Read, both cursors = 61. Next write at offset 61.
        # Write 1-byte payload (wire 5) → straddles 61..64 + 0..1.
        # The 4-byte LENGTH PREFIX spans 61, 62, 63, 0.
        assert r.write_message(b"X" * 13)
        assert r.read_message() == b"X" * 13
        assert r.write_message(b"Y" * 40)
        assert r.read_message() == b"Y" * 40
        # Now write a tiny message whose LENGTH PREFIX straddles.
        assert r.write_message(b"Z")
        assert r.read_message() == b"Z"


def test_wrap_with_payload_straddling_boundary(tmp_path):
    """The more common wrap case — length prefix lands cleanly but
    payload straddles. Same caveat (easy to silently corrupt)."""
    path = _ring_path(tmp_path)
    cap = 64
    with ShmRing.create(path, capacity=cap) as r:
        assert r.write_message(b"X" * 10)
        assert r.read_message() == b"X" * 10
        assert r.write_message(b"Y" * 40)
        assert r.read_message() == b"Y" * 40
        # Cursors at 58; next write of 6-byte payload (wire 10) spans
        # 58..64 + 0..4. Payload bytes 0..1 land before the wrap;
        # bytes 2..5 land after.
        payload = b"\xAA\xBB\xCC\xDD\xEE\xFF"
        assert r.write_message(payload)
        assert r.read_message() == payload


# ── 5. Partial-write invisibility ───────────────────────────────────


def test_partial_payload_invisible_to_reader(tmp_path):
    """The cursor advance is the COMMIT. Until the producer advances
    write_cursor, the reader sees no message — even if length-prefix
    bytes are already on disk. Critical correctness property: SPSC
    ordering relies on the cursor being the publication barrier."""
    path = _ring_path(tmp_path)
    with ShmRing.create(path, capacity=1024) as r:
        # Write the LENGTH PREFIX manually but DON'T advance the
        # cursor — simulates a producer interrupted mid-write.
        struct.pack_into("<I", r._buf, HEADER_SIZE, 16)
        # Even though the bytes are present, no message is visible.
        assert r.read_message() is None
        # Real write commits via cursor advance + reader sees it.
        assert r.write_message(b"committed")
        assert r.read_message() == b"committed"


# ── 6. Threaded SPSC ────────────────────────────────────────────────


def test_threaded_producer_consumer_round_trip(tmp_path):
    """Real SPSC use case: producer thread writes N messages, consumer
    thread reads them. Validates the cursor visibility across thread
    boundaries (on x86 + ARM-acquire/release this Just Works for
    aligned u64 loads; the test guards a regression that broke
    that assumption)."""
    path = _ring_path(tmp_path)
    n_messages = 200
    received: list[bytes] = []

    with ShmRing.create(path, capacity=4096) as producer:
        with ShmRing.attach(path) as consumer:
            def consume():
                deadline = time.time() + 5.0
                while len(received) < n_messages:
                    msg = consumer.read_message()
                    if msg is None:
                        if time.time() > deadline:
                            return
                        time.sleep(0.001)
                        continue
                    received.append(msg)

            t = threading.Thread(target=consume, daemon=True)
            t.start()

            for i in range(n_messages):
                payload = f"msg-{i:05d}".encode()
                # Retry on full ring.
                while not producer.write_message(payload):
                    time.sleep(0.0005)

            t.join(timeout=5.0)
            assert len(received) == n_messages
            for i, msg in enumerate(received):
                assert msg == f"msg-{i:05d}".encode()


# ── 7. Cursor headroom ──────────────────────────────────────────────


def test_cursors_are_monotonic_no_premature_wrap(tmp_path):
    """The cursors are u64 monotonic — actual wrap math happens at
    the offset compute, not at the cursor itself. Test that pushing
    many messages doesn't trip cursor arithmetic before u64 actually
    exhausts (which would take ~years at realistic rates).

    Sanity: after 1000 write+read cycles, cursors are at 1000 *
    (payload_size + 4) and still equal (consumer fully drained)."""
    path = _ring_path(tmp_path)
    with ShmRing.create(path, capacity=128) as r:
        for _ in range(1000):
            assert r.write_message(b"X" * 16)
            assert r.read_message() == b"X" * 16
        # Each iteration advances both cursors by 20 (4-byte prefix
        # + 16-byte payload). 1000 iterations → 20000.
        assert r._read_write_cursor() == 20000
        assert r._read_read_cursor() == 20000
