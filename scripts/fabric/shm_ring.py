"""Shared-memory ring buffer — primitive for the fabric C++ sprint
Phase A (daemon shm I/O).

The fabric_lite Python prototype's `DaemonClient` ships activations
into the C++ daemon via stdin/stdout pipes, paying a ~100µs marshal
cost per hop. Phase A replaces the pipe with a memory region both
sides mmap, eliminating the kernel copy entirely on each
producer→consumer handoff.

This module is the **transport-neutral primitive**: a single-producer
single-consumer ring buffer over an mmap'd file. Both the C++ daemon
(Phase A.3) and the Python `DaemonClient` (Phase A.2) consume the
same header layout + framing, so the two sides can interoperate on
their first integration test.

**Why a file-backed mmap and not memfd/SCM_RIGHTS?** The sprint plan
recommends SCM_RIGHTS for production (cleaner fd lifetime, no on-disk
artifact). For Phase A.1 we want the simplest thing that round-trips
correctly; a `/tmp/<uuid>.ring` file mmap'd by both sides is trivially
debuggable + works identically on Linux + macOS. SCM_RIGHTS lands
when Phase A.2/A.3 actually integrate the daemon (one localised diff,
not load-bearing for the ring primitive itself).

**SPSC ordering caveat.** This module assumes one producer + one
consumer per ring; the cursors are monotonic u64 (~584 years of headroom
at 1 GB/s before wrap). Memory ordering relies on x86's TSO + ARM's
acquire/release semantics on aligned 8-byte loads/stores; Python's
mmap doesn't expose explicit fences, but the existing cluster is all
x86 (Linux + Intel Macs) so TSO covers us. Apple Silicon would need a
revisit (mlock + a small ctypes fence wrapper) — not in scope here
because the cluster doesn't have any.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Optional, Union


# ── Wire constants ──────────────────────────────────────────────────


# 64-byte header — one cache line on every relevant ISA. Layout is
# fixed at v1; new fields go in the reserved padding without breaking
# v1 consumers.
HEADER_SIZE: int = 64

# "RING" little-endian — first 4 header bytes. Mismatch on attach
# means the file isn't a ShmRing (corrupted, wrong format, or stale).
MAGIC: bytes = b"RING"

VERSION_MAJOR: int = 1

# Each framed message is preceded by a u32 length. Keep small — the
# fabric activation envelope is already u32-framed at the daemon level
# (CMD/n_tokens/start_pos/flags/payload_bytes/payload), so this length
# prefix is a single byte tax on each ring write.
_LENGTH_PREFIX_SIZE: int = 4


# Header field offsets — explicit so the C++ side can mirror them
# verbatim without ambiguity. Anything beyond offset 32 is reserved
# v1 padding (future cursor types, sequence numbers, etc.).
_OFF_MAGIC = 0           # 4 bytes
_OFF_VERSION = 4         # 4 bytes (u32)
_OFF_CAPACITY = 8        # 8 bytes (u64)
_OFF_WRITE_CURSOR = 16   # 8 bytes (u64) — monotonic
_OFF_READ_CURSOR = 24    # 8 bytes (u64) — monotonic


# ── Errors ──────────────────────────────────────────────────────────


class ShmRingError(Exception):
    """Base for ring failures so transport callers can catch the whole
    family with one line."""


class MagicError(ShmRingError):
    """Header's first 4 bytes weren't ``RING``. Wrong file, corrupted
    ring, or a stale region from a previous incompatible run."""


class VersionError(ShmRingError):
    """Header version_major didn't match. v1 attaches refuse v2 rings
    — the C++ daemon and Python side MUST be at the same major; mixed
    versions are forbidden (mirrors ADR 0005 decision 6's stance for
    the fabric wire schema)."""


class CapacityError(ShmRingError):
    """A single message exceeds the ring's payload capacity. Caller
    should size the ring for the largest expected payload + the
    4-byte length prefix; this raises rather than silently dropping
    so the operator sees the misconfiguration."""


# ── ShmRing ─────────────────────────────────────────────────────────


class ShmRing:
    """Single-producer single-consumer ring buffer over an mmap'd file.

    Two-step construction matches typical SPSC usage:

      producer = ShmRing.create("/tmp/daemon-in.ring", capacity=8<<20)
      consumer = ShmRing.attach("/tmp/daemon-in.ring")

    Producer writes via :meth:`write_message`; consumer reads via
    :meth:`read_message`. Both calls are O(1) amortized plus one
    memcpy of the payload. Neither blocks: full→False on write,
    empty→None on read. Caller wraps with retry/sleep loops where
    needed (Phase A.2's DaemonClient will).
    """

    __slots__ = ("_path", "_capacity", "_size", "_fd", "_buf",
                 "_owned_file")

    def __init__(self, path: Path, capacity: int, fd: int,
                 buf: memoryview, owned_file: bool):
        # Internal — use ``create`` / ``attach`` classmethods.
        self._path = path
        self._capacity = capacity
        self._size = HEADER_SIZE + capacity
        self._fd = fd
        self._buf = buf
        self._owned_file = owned_file

    # ── Construction ──────────────────────────────────────────────

    @classmethod
    def create(cls, path: Union[str, Path], capacity: int) -> "ShmRing":
        """Allocate + initialise a new ring at ``path``. Truncates any
        existing file at the path. Returns a ring with both cursors
        at zero, ready for the producer to write into.

        ``capacity`` is the PAYLOAD area size in bytes; the actual
        file size is ``capacity + HEADER_SIZE``. Typical sizing for
        the daemon: 8 MiB — comfortably above any single activation
        and gives the consumer plenty of slack to drain bursts."""
        import mmap as mmap_mod
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0; got {capacity}")
        path = Path(path)
        size = HEADER_SIZE + capacity
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT | os.O_TRUNC,
                     0o600)
        os.ftruncate(fd, size)
        mm = mmap_mod.mmap(fd, size,
                            mmap_mod.MAP_SHARED,
                            mmap_mod.PROT_READ | mmap_mod.PROT_WRITE)
        buf = memoryview(mm)
        # Write header. Cursors start at 0; capacity is fixed at create.
        struct.pack_into("<4sIQQQ", buf, 0,
                          MAGIC, VERSION_MAJOR, capacity, 0, 0)
        return cls(path=path, capacity=capacity, fd=fd,
                    buf=buf, owned_file=True)

    @classmethod
    def attach(cls, path: Union[str, Path]) -> "ShmRing":
        """Attach to an existing ring. Reads + validates the header;
        raises ``MagicError`` / ``VersionError`` on mismatch."""
        import mmap as mmap_mod
        path = Path(path)
        fd = os.open(str(path), os.O_RDWR)
        st = os.fstat(fd)
        size = st.st_size
        if size < HEADER_SIZE:
            os.close(fd)
            raise ShmRingError(
                f"file too small for ring header: {size} < {HEADER_SIZE}"
            )
        mm = mmap_mod.mmap(fd, size,
                            mmap_mod.MAP_SHARED,
                            mmap_mod.PROT_READ | mmap_mod.PROT_WRITE)
        buf = memoryview(mm)
        magic, version, capacity, _w, _r = struct.unpack_from(
            "<4sIQQQ", buf, 0)
        if magic != MAGIC:
            # Release the memoryview before closing the mmap —
            # mm.close() raises BufferError while exported views
            # exist.
            buf.release()
            mm.close()
            os.close(fd)
            raise MagicError(
                f"bad magic: {magic!r} != {MAGIC!r} at {path}"
            )
        if version != VERSION_MAJOR:
            buf.release()
            mm.close()
            os.close(fd)
            raise VersionError(
                f"unsupported version_major: {version} (expect "
                f"{VERSION_MAJOR})"
            )
        # Sanity: file size should equal HEADER_SIZE + capacity. Don't
        # fail hard if it's larger (could happen if the producer
        # over-ftruncated); the extra bytes are simply unused.
        return cls(path=path, capacity=capacity, fd=fd,
                    buf=buf, owned_file=False)

    # ── Cursor accessors (internal) ───────────────────────────────

    def _read_write_cursor(self) -> int:
        return struct.unpack_from("<Q", self._buf, _OFF_WRITE_CURSOR)[0]

    def _read_read_cursor(self) -> int:
        return struct.unpack_from("<Q", self._buf, _OFF_READ_CURSOR)[0]

    def _store_write_cursor(self, value: int) -> None:
        struct.pack_into("<Q", self._buf, _OFF_WRITE_CURSOR, value)

    def _store_read_cursor(self, value: int) -> None:
        struct.pack_into("<Q", self._buf, _OFF_READ_CURSOR, value)

    # ── State queries ─────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        """Total payload area in bytes (header excluded)."""
        return self._capacity

    @property
    def available_to_read(self) -> int:
        """Bytes that the consumer can read right now (committed by
        the producer; not yet consumed). Includes message framing
        bytes — a single 100-byte payload is reported as 104."""
        return self._read_write_cursor() - self._read_read_cursor()

    @property
    def available_to_write(self) -> int:
        """Bytes the producer can write right now without overrunning
        the unread tail."""
        return self._capacity - self.available_to_read

    # ── Write / read ──────────────────────────────────────────────

    def write_message(self, payload: bytes) -> bool:
        """Write one framed message. Returns ``True`` on success,
        ``False`` when the ring lacks space (caller retries).

        Raises :class:`CapacityError` if the message would never fit
        — better to fail loud at the caller's misconfigured size than
        to spin in a full-loop forever."""
        needed = _LENGTH_PREFIX_SIZE + len(payload)
        if needed > self._capacity:
            raise CapacityError(
                f"message size {needed} exceeds ring capacity "
                f"{self._capacity}"
            )
        if needed > self.available_to_write:
            return False
        w = self._read_write_cursor()
        # Length prefix.
        self._write_at(w, struct.pack("<I", len(payload)))
        # Payload.
        self._write_at(w + _LENGTH_PREFIX_SIZE, payload)
        # Commit: advancing the cursor is what makes the message
        # visible to the consumer. On x86 + ARM-acquire/release this
        # is safe because the consumer's cursor load that precedes
        # the payload read can't be reordered past it.
        self._store_write_cursor(w + needed)
        return True

    def read_message(self) -> Optional[bytes]:
        """Read one framed message. Returns ``None`` when no complete
        message is available (empty ring OR producer hasn't finished
        the next write yet)."""
        avail = self.available_to_read
        if avail < _LENGTH_PREFIX_SIZE:
            return None
        r = self._read_read_cursor()
        (length,) = struct.unpack(
            "<I", self._read_at(r, _LENGTH_PREFIX_SIZE))
        if length > self._capacity - _LENGTH_PREFIX_SIZE:
            # Defensive: a corrupted length would otherwise read
            # past the buffer + cause a hard crash. Raise instead so
            # the operator sees something is wrong.
            raise ShmRingError(
                f"corrupt length field at offset {r}: {length} > "
                f"capacity {self._capacity}"
            )
        if avail < _LENGTH_PREFIX_SIZE + length:
            return None                              # partial write
        payload = bytes(self._read_at(
            r + _LENGTH_PREFIX_SIZE, length))
        self._store_read_cursor(r + _LENGTH_PREFIX_SIZE + length)
        return payload

    # ── Buffer access — wrap-aware ────────────────────────────────

    def _write_at(self, cursor: int, data: bytes) -> None:
        """Write ``data`` to the payload area starting at the offset
        ``cursor % capacity``. Splits across the wrap boundary if
        needed."""
        offset = cursor % self._capacity
        n = len(data)
        end = offset + n
        if end <= self._capacity:
            self._buf[HEADER_SIZE + offset:HEADER_SIZE + end] = data
            return
        # Wraps: write [offset..capacity) then [0..end - capacity)
        first = self._capacity - offset
        self._buf[HEADER_SIZE + offset:HEADER_SIZE + self._capacity] = data[:first]
        self._buf[HEADER_SIZE:HEADER_SIZE + (n - first)] = data[first:]

    def _read_at(self, cursor: int, n: int) -> bytes:
        """Read ``n`` bytes starting at offset ``cursor % capacity``,
        splicing the wrap if needed."""
        offset = cursor % self._capacity
        end = offset + n
        if end <= self._capacity:
            return bytes(self._buf[HEADER_SIZE + offset:HEADER_SIZE + end])
        first = self._capacity - offset
        return (bytes(self._buf[HEADER_SIZE + offset:HEADER_SIZE + self._capacity])
                + bytes(self._buf[HEADER_SIZE:HEADER_SIZE + (n - first)]))

    # ── Lifecycle ─────────────────────────────────────────────────

    def close(self) -> None:
        """Release the mmap + close the fd. ``owned_file=True`` rings
        (the creator) also unlink the backing file so a re-run starts
        clean. Attached rings (consumer side) leave the file in place;
        the creator's close removes it."""
        try:
            self._buf.release()
        except Exception:
            pass
        try:
            os.close(self._fd)
        except Exception:
            pass
        if self._owned_file:
            try:
                os.unlink(self._path)
            except FileNotFoundError:
                pass

    def __enter__(self) -> "ShmRing":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
