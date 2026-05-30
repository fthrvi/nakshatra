"""Fabric transport — UDP socket lifecycle, MTU-aware chunking, and
per-link chunk reassembly. Sits on top of ``fabric.packet`` (the wire
codec) and underneath ``fabric.backend`` (which is the gRPC-compatible
shim coming in Phase D).

Per the sprint plan
(``~/trisul/plans/2026-05-26-nakshatra-fabric-lite-prototype-sprint.md``)
this module owns the following pieces of the fabric data plane:

  - One UDP socket per ordered ``(self, peer)`` pair (a `FabricLink`).
  - The monotonic ``seq`` counter for that link (schema §5: nonce uses
    ``chain_id || seq``; per-pair keys + monotonic seq is what makes
    AES-GCM safe).
  - MTU-aware sender split + receiver reassembly using the
    ``payload_offset`` field and ``FLAG_LAST_IN_STEP`` flag.
  - The counter set from schema §9 (sent/recv bytes/packets, auth
    fails, gaps, alloc drops, dtype drops).

It does NOT own:

  - Pillar ``/join`` handshake or keyring distribution — Phase C.
  - The Forward/Inference RPC shim — Phase D.
  - Periodic snapshot reporting to ``/fabric/link_stats`` — Phase E.

Each ``FabricLink`` is single-threaded by design — callers serialize
sends through one link and recv is called from one thread. ADR 0005
decision 6 forbids mixed-fabric clusters, so there's no
multiplexing-with-grpc concern at this layer.
"""
from __future__ import annotations

import os
import socket
import time
from typing import Optional

from fabric import packet as fp


# 2026-05-29 transport hardening — env-tunable knobs surfaced after the
# Phase F.3 cluster smoke found ~75% chunk loss when 12-chunk activations
# fired back-to-back over Tailscale. Wire schema is fine (every arrival
# parses + AES-GCM-verifies); the loss is at the receiver's UDP socket
# buffer drainage rate. Two knobs:
#
#   1. SO_RCVBUF bump: absorbs the burst kernel-side without slowing the
#      sender. macOS default ~200 KB; bumping to 4 MB swallows a 12-chunk
#      (~18 KB) burst trivially with room for many concurrent assemblies.
#   2. Sender pacing: a tunable inter-chunk sleep. Trades sender latency
#      for delivery rate when the receiver buffer route isn't enough.
#
# Both default OFF (0) so localhost smoke + unit tests are unchanged
# and the same-LAN Mode A path doesn't pay a latency cost it doesn't need.
ENV_RECV_BUF_BYTES = "NAKSHATRA_FABRIC_RECV_BUF_BYTES"
ENV_SEND_PACE_S = "NAKSHATRA_FABRIC_SEND_PACE_S"


def _env_int(name: str, default: int = 0) -> int:
    """Typo-safe env int reader. Same fallback stance as
    nakshatra_tls._probe_timeout_from_env."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(0, v)


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return max(0.0, v)


# ── MTU math ────────────────────────────────────────────────────────


# Per-datagram overhead the fabric pays before its own header.
# 20-byte IPv4 header + 8-byte UDP header = 28 bytes that we never see
# but the kernel + NIC consume out of the link MTU.
_IP_UDP_OVERHEAD_BYTES = 28


def max_chunk_payload_for_mtu(mtu: int) -> int:
    """Return the maximum fabric-payload bytes that fit in one UDP
    datagram at the given link MTU. Subtracts the IP/UDP overhead and
    the fixed 64-byte fabric header.

    Common values:
      - MTU 1500 (Ethernet default, WireGuard MTU 1420):  ~1408 / 1328
      - MTU 9000 (Mode A jumbo frames on the lab switch): 8908
    """
    payload = mtu - _IP_UDP_OVERHEAD_BYTES - fp.HEADER_SIZE
    if payload <= 0:
        raise ValueError(
            f"MTU {mtu} too small for fabric header + IP/UDP overhead "
            f"(need > {_IP_UDP_OVERHEAD_BYTES + fp.HEADER_SIZE})"
        )
    return payload


# ── Counter set per schema §9 ───────────────────────────────────────


# Names match the schema's prescribed counter set so Phase E's periodic
# snapshot to /fabric/link_stats has byte-identical field names to what
# the pillar's GET /fabric/links projection serves.
_COUNTER_NAMES = (
    "sent_packets",
    "sent_bytes",
    "recv_packets",
    "recv_bytes",
    "recv_auth_fails",
    "recv_gaps",
    "recv_dropped_alloc",
    "recv_dropped_dtype",
    # Schema §9 — RTT fields. Populated by FabricBackend's RTT ring
    # buffer on the first worker (the only side that observes the
    # full forward→feedback round trip); zero on every other link.
    # Default-initialised to 0 here so callers can read the keys
    # without a .get fallback even before the first round trip.
    "rtt_ns_p50",
    "rtt_ns_p99",
    # Magic + version drops aren't in the schema's prescribed set
    # (those are "garbage UDP found the port" cases that the schema
    # treats as silent drops). Track them privately for operator
    # debug; Phase E decides whether to surface them in snapshots.
    "recv_dropped_magic",
    "recv_dropped_version",
    "recv_dropped_truncated",
)


def _empty_counters() -> dict[str, int]:
    return {name: 0 for name in _COUNTER_NAMES}


# ── Per-assembly reassembly state ───────────────────────────────────


class _Assembly:
    """One in-flight chunked activation. Keyed in the link's slot map
    by ``(step_id, layer_idx)``. Single-datagram packets bypass this
    machinery (they're complete on first arrival).

    Stores chunks by ``payload_offset`` rather than by ``seq`` so
    out-of-order arrival reassembles correctly (the schema permits any
    arrival order; only ``payload_offset`` is authoritative for slot
    placement).
    """

    __slots__ = ("first_header", "chunks", "expected_total",
                 "last_in_step_seen")

    def __init__(self, first_header: fp.FabricHeader):
        self.first_header = first_header
        self.chunks: dict[int, bytes] = {}     # offset → plaintext bytes
        self.expected_total: Optional[int] = None
        self.last_in_step_seen: bool = False

    def add(self, header: fp.FabricHeader, plaintext: bytes) -> None:
        self.chunks[header.payload_offset] = plaintext
        if header.flags & fp.FLAG_LAST_IN_STEP:
            self.last_in_step_seen = True
            # The final chunk's offset + length tells us the total
            # activation size — receiver doesn't need a separate
            # contract field for this.
            self.expected_total = (
                header.payload_offset + header.payload_length
            )

    def total_bytes(self) -> int:
        return sum(len(c) for c in self.chunks.values())

    def is_complete(self) -> bool:
        """Schema §11 item 6 — dispatch when ``last_in_step`` has been
        observed AND every byte in ``[0, expected_total)`` has been
        filled by some chunk. Out-of-order chunks are fine; missing
        chunks (gap in offset coverage) block dispatch.
        """
        if not self.last_in_step_seen or self.expected_total is None:
            return False
        # Cheap full-coverage check: sort offsets, walk them, confirm
        # they tile [0, expected_total) without gaps and without
        # overlaps. Tiny number of chunks per activation in practice
        # (16 KB / 1408 = 12 at MTU 1500); not worth a more clever
        # data structure.
        offsets = sorted(self.chunks.keys())
        cursor = 0
        for off in offsets:
            if off != cursor:
                return False
            cursor = off + len(self.chunks[off])
        return cursor == self.expected_total

    def assemble(self) -> bytes:
        """Concatenate chunks in offset order. Caller MUST check
        :meth:`is_complete` first."""
        return b"".join(
            self.chunks[off] for off in sorted(self.chunks.keys())
        )


# ── FabricLink ─────────────────────────────────────────────────────


# Default per-link policy. Real values come from /join + chain plan
# in Phase C; for Phase B we expose them as constructor args so unit
# tests can pin them low to provoke the alloc-drop path.
DEFAULT_MAX_SLOT_BYTES = 32 * 1024 * 1024   # 32 MiB — fits Llama-3.3 70B fp16
DEFAULT_MAX_SLOTS = 16                       # concurrent in-flight assemblies


class FabricLink:
    """One ordered ``(self, peer)`` link over UDP, carrying fabric
    packets sealed under the per-pair key.

    The link owns:
      - A bound UDP socket (caller passes it in so tests can use
        ``socket.socketpair``-style two-port setups without binding
        to ephemeral ports separately).
      - A pinned ``peer_addr`` tuple — every send goes here, every
        recv asserts arrival from this address.
      - The monotonic ``send_seq`` counter (initialised to 0;
        increments per chunk sent).
      - The receive-side gap-tracking state (``_max_recv_seq``).
      - The reassembly slot map.
      - The counter dictionary.

    Symmetry: a peer-pair conceptually has two of these, one in each
    direction, each with its own socket + key + seq counter.
    """

    def __init__(
        self,
        sock: socket.socket,
        peer_addr: tuple[str, int],
        key: bytes,
        chain_id: int,
        *,
        mtu: int = 1500,
        max_slot_bytes: int = DEFAULT_MAX_SLOT_BYTES,
        max_slots: int = DEFAULT_MAX_SLOTS,
        recv_buf_bytes: Optional[int] = None,
        send_pace_s: Optional[float] = None,
    ):
        if len(key) != 16:
            raise ValueError(
                f"FabricLink key must be 16 bytes (AES-128); got {len(key)}"
            )
        self.sock = sock
        self.peer_addr = peer_addr
        self.key = key
        self.chain_id = chain_id
        self.mtu = mtu
        self.chunk_payload_max = max_chunk_payload_for_mtu(mtu)
        self.max_slot_bytes = max_slot_bytes
        self.max_slots = max_slots
        # 2026-05-29 transport hardening (see ENV_* docs above). Default
        # OFF — explicit constructor arg overrides env; both off is the
        # zero-overhead loopback path the unit tests + same-LAN
        # production rely on.
        self.send_pace_s = (
            send_pace_s if send_pace_s is not None
            else _env_float(ENV_SEND_PACE_S)
        )
        rcv = (recv_buf_bytes if recv_buf_bytes is not None
               else _env_int(ENV_RECV_BUF_BYTES))
        if rcv > 0:
            # Best-effort — the kernel may cap below the request
            # (sysctl net.core.rmem_max on Linux, kern.ipc.maxsockbuf
            # on macOS). setsockopt itself doesn't raise on cap;
            # the resulting effective buffer is whatever the OS gave.
            self.sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, rcv)
        self.recv_buf_bytes_requested = rcv
        self.send_seq: int = 0
        # Sentinel — no packets received yet, so the first packet
        # establishes the baseline without firing a spurious gap.
        self._max_recv_seq: int = -1
        self._slots: dict[tuple[int, int], _Assembly] = {}
        self.counters: dict[str, int] = _empty_counters()

    # ── Send side ──────────────────────────────────────────────────

    def build_chunks(
        self,
        payload: bytes,
        *,
        packet_type: int = fp.PACKET_TYPE_FORWARD,
        step_id: int = 0,
        layer_idx: int = 0,
        dtype: int = fp.DTYPE_FP16,
    ) -> list[bytes]:
        """Split ``payload`` into MTU-bounded chunks, seal each, and
        return the list of on-wire datagrams in send order.

        Exposed publicly (rather than inline in :meth:`send`) so tests
        can craft out-of-order arrivals and dropped-chunk scenarios
        deterministically. Production callers should use :meth:`send`.

        Side effect: increments ``self.send_seq`` by the number of
        chunks returned. A failed sendto by the caller doesn't roll
        the seq back — the schema requires monotonic seq per link
        even across send failures, because the per-link rekey is what
        resets seq, not packet success.
        """
        if not payload:
            # Zero-length payloads are allowed for CONTROL packets;
            # ship one datagram with last_in_step set.
            chunks = [b""]
        else:
            chunks = [
                payload[i:i + self.chunk_payload_max]
                for i in range(0, len(payload), self.chunk_payload_max)
            ]
        out: list[bytes] = []
        for idx, chunk in enumerate(chunks):
            is_last = (idx == len(chunks) - 1)
            flags = 0
            if not is_last:
                flags |= fp.FLAG_TRUNCATED_CONTINUED
            if is_last:
                flags |= fp.FLAG_LAST_IN_STEP
            offset = idx * self.chunk_payload_max
            header = fp.FabricHeader(
                magic=fp.MAGIC,
                version_major=fp.VERSION_MAJOR,
                version_minor=fp.VERSION_MINOR,
                packet_type=packet_type,
                dtype=dtype,
                flags=flags,
                reserved=0,
                chain_id=self.chain_id,
                step_id=step_id,
                layer_idx=layer_idx,
                seq=self.send_seq,
                payload_length=len(chunk),
                payload_offset=offset,
                auth_tag=b"\x00" * 16,
            )
            out.append(fp.seal(header, chunk, self.key))
            self.send_seq += 1
        return out

    def send(
        self,
        payload: bytes,
        *,
        packet_type: int = fp.PACKET_TYPE_FORWARD,
        step_id: int = 0,
        layer_idx: int = 0,
        dtype: int = fp.DTYPE_FP16,
    ) -> int:
        """Send ``payload`` as one or more chunked datagrams to the
        link's peer. Returns the number of chunks transmitted.

        No retransmit / no ACK / no flow control — schema §8 ("Loss
        handling at the schema layer: none"). A chronic loss pattern
        surfaces via the ``recv_gaps`` counter at the receiver and is
        the planner's signal to re-plan, not this layer's signal to
        recover.
        """
        chunks = self.build_chunks(
            payload, packet_type=packet_type, step_id=step_id,
            layer_idx=layer_idx, dtype=dtype,
        )
        for idx, c in enumerate(chunks):
            # Inter-chunk pacing (Phase F.3 burst-loss mitigation —
            # see fabric_burst_loss_finding memory). Only sleeps when
            # send_pace_s > 0 AND there's another chunk after this
            # one, so single-datagram sends pay no cost.
            if idx > 0 and self.send_pace_s > 0:
                time.sleep(self.send_pace_s)
            self.sock.sendto(c, self.peer_addr)
            self.counters["sent_packets"] += 1
            self.counters["sent_bytes"] += len(c)
        return len(chunks)

    # ── Receive side ──────────────────────────────────────────────

    def recv(
        self,
        *,
        timeout: Optional[float] = None,
    ) -> Optional[tuple[fp.FabricHeader, bytes]]:
        """Block until one complete activation is reassembled and
        return ``(metadata_header, plaintext)``. ``metadata_header`` is
        the FIRST chunk's header with ``payload_length`` rewritten to
        the total reassembled size — i.e., as if the whole activation
        had been one giant packet — so callers see a single-message
        view regardless of chunking.

        Returns ``None`` on timeout. The loop pumps datagrams through
        ``_handle_datagram`` (which updates state + counters) and
        returns when ``_handle_datagram`` yields a completed assembly.

        Single-datagram packets are completed assemblies of one chunk
        and return on the first iteration.
        """
        self.sock.settimeout(timeout)
        while True:
            try:
                buf, src = self.sock.recvfrom(65535)
            except socket.timeout:
                return None
            # Drop datagrams from any address other than the pinned
            # peer — defense in depth against spoofed UDP, plus auth
            # would reject them anyway on key mismatch.
            if src != self.peer_addr:
                continue
            result = self._handle_datagram(buf)
            if result is not None:
                return result

    def _handle_datagram(
        self, buf: bytes,
    ) -> Optional[tuple[fp.FabricHeader, bytes]]:
        """Parse + auth-verify one wire datagram. Updates counters and
        the reassembly slot map. Returns a completed assembly tuple
        ``(metadata_header, plaintext)`` if this datagram completes
        one, else ``None``.

        Visible as a public-ish underscore method because tests inject
        crafted bytes here without going through the real socket.
        """
        self.counters["recv_packets"] += 1
        self.counters["recv_bytes"] += len(buf)
        try:
            header, plaintext = fp.open(buf, self.key)
        except fp.MagicError:
            self.counters["recv_dropped_magic"] += 1
            return None
        except fp.VersionError:
            self.counters["recv_dropped_version"] += 1
            return None
        except fp.AuthError:
            self.counters["recv_auth_fails"] += 1
            return None
        except fp.TruncatedError:
            self.counters["recv_dropped_truncated"] += 1
            return None

        # Gap detection. The schema lives on (chain_id, sender_id,
        # receiver_id) seq monotonicity; this link IS one such triple
        # so a header.seq > _max_recv_seq + 1 means at least one prior
        # datagram never arrived. Out-of-order arrivals (seq < max)
        # don't trigger a fresh gap count — the gap was already
        # counted when the higher seq was first seen.
        if self._max_recv_seq >= 0 and header.seq > self._max_recv_seq + 1:
            self.counters["recv_gaps"] += (
                header.seq - self._max_recv_seq - 1
            )
        if header.seq > self._max_recv_seq:
            self._max_recv_seq = header.seq

        # Capacity guard applies to every packet, single-datagram or
        # chunked (schema §11 item 8). Single-datagram fast path
        # below would otherwise bypass it and return oversized
        # payloads to the caller.
        if header.payload_length > self.max_slot_bytes:
            self.counters["recv_dropped_alloc"] += 1
            return None

        # Single-datagram fast path: one chunk, last_in_step set,
        # offset 0. Skip the slot map entirely; just return.
        if (header.payload_offset == 0
                and header.flags & fp.FLAG_LAST_IN_STEP
                and not (header.flags & fp.FLAG_TRUNCATED_CONTINUED)):
            # The metadata header IS the chunk header in this case.
            return header, plaintext

        slot_key = (header.step_id, header.layer_idx)
        slot = self._slots.get(slot_key)
        if slot is None:
            if len(self._slots) >= self.max_slots:
                # Slot map full — refuse the new assembly. Beats
                # evicting an in-flight one (which would silently
                # corrupt that activation). Loss-of-this-activation
                # surfaces as never-completing; the gap counter
                # already incremented; planner will re-plan if this
                # is chronic.
                self.counters["recv_dropped_alloc"] += 1
                return None
            slot = _Assembly(first_header=header)
            self._slots[slot_key] = slot
        slot.add(header, plaintext)

        # If accumulated bytes blow past the per-slot cap (e.g., a
        # sender mis-sized the activation), drop the partial.
        if slot.total_bytes() > self.max_slot_bytes:
            del self._slots[slot_key]
            self.counters["recv_dropped_alloc"] += 1
            return None

        if slot.is_complete():
            del self._slots[slot_key]
            plaintext_full = slot.assemble()
            # Synthesize the "as-if-one-packet" metadata header from
            # the first chunk: same chain_id/step_id/layer_idx/type/
            # dtype, but payload_length = total and payload_offset = 0
            # and flags = LAST_IN_STEP (no TRUNCATED_CONTINUED).
            meta = slot.first_header._replace(
                payload_offset=0,
                payload_length=len(plaintext_full),
                flags=fp.FLAG_LAST_IN_STEP,
                # seq + auth_tag carry the first-chunk's values; they
                # apply to the wire packet, not the logical message,
                # so they're a touch ambiguous here. Document via the
                # docstring rather than mutate to a meaningless value.
            )
            return meta, plaintext_full
        return None

    # ── Lifecycle ─────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying socket + drop any partial assemblies.
        Counters are preserved so a snapshot can still be taken
        post-close."""
        self.sock.close()
        self._slots.clear()
