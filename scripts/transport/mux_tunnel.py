"""Multiplexing tunnel (v1.1 §8.4) — many TCP streams over one authenticated pipe.

The rendezvous relay (`relay.py`) pairs two peers into ONE byte pipe; the identity
handshake (`identity_handshake.py`) authenticates it. But the gRPC data plane
opens *several* connections to a worker (client→worker channel, worker→worker
forward, …). This layer carries all of them over the single relayed pipe — the
job WireGuard/QUIC do for free, done minimally here so the spike can push real
inference through the relay.

Topology (point a chain's worker-B address at the local listener):

    client → 127.0.0.1:LOCAL ─┐                       ┌─ 127.0.0.1:TARGET (worker B)
                       MuxTunnel(client) ── relay pipe ── MuxTunnel(server)
    each local TCP conn ──────┘   (one auth'd pipe)      └── one dial per stream

Frame: `>IBI` = (stream_id u32, type u8, length u32) + payload.
Types: OPEN(1) — open a stream (server dials TARGET); DATA(2); CLOSE(3).

Pure stdlib, no deps. One reader thread demuxes; writes are serialized by a lock.

Hardening (2026-10-02 design council), wire format unchanged so old peers still interoperate:
- MAX_FRAME: a frame longer than this closes the tunnel before its body is read.
- MAX_STREAMS: OPENs beyond this are refused with CLOSE.
- The reader never writes into a local socket: each stream has a queue drained by its own writer
  thread, so one slow consumer no longer stalls the others until it is MAX_QUEUED bytes behind.
  Past that the reader PAUSES (backpressure, as the old direct-write code did). A stream that is
  slow but still draining is never dropped; one that makes NO progress for STALL_S while over the
  cap is stuck, and is closed so the rest of the tunnel can move again.
- A peer's CLOSE is graceful: it is queued BEHIND the stream's data, and the writer closes the
  socket only after delivering everything (data-then-close is the normal request/response shape).
- Dials for incoming OPENs run off the reader thread.
Still no credit-based windows (that needs a protocol change); production = WireGuard/QUIC.
"""
from __future__ import annotations

import queue
import socket
import struct
import threading
import time
from typing import Optional

_HDR = struct.Struct(">IBI")
OPEN, DATA, CLOSE = 1, 2, 3
_CHUNK = 65536
MAX_FRAME = 1 << 20            # senders emit <= _CHUNK; anything over 1 MiB is hostile or broken
MAX_STREAMS = 256
MAX_QUEUED = 4 << 20           # bytes buffered per stream before the reader waits for it to drain
_FIN = object()                # queue marker: deliver what is queued, then close (graceful CLOSE)
STALL_S = 30.0                 # over the cap AND no drain progress this long = stuck, not slow: close it


class MuxTunnel:
    def __init__(self, pipe: socket.socket):
        self._pipe = pipe
        self._wlock = threading.Lock()
        self._streams: dict[int, socket.socket] = {}
        self._queues: dict[int, "queue.Queue[Optional[bytes]]"] = {}
        self._qbytes: dict[int, int] = {}
        self._slock = threading.Lock()
        self._drained = threading.Condition(self._slock)   # writers signal the paused reader
        self._next_id = 1
        self._closed = threading.Event()

    # ── framed writes (serialized) ──
    def _send(self, sid: int, typ: int, payload: bytes = b"") -> None:
        with self._wlock:
            try:
                self._pipe.sendall(_HDR.pack(sid, typ, len(payload)) + payload)
            except OSError:
                self._closed.set()

    def _recv_exact(self, n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n:
            try:
                chunk = self._pipe.recv(n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    # ── pump a local socket's bytes onto a mux stream ──
    def _pump_local_to_stream(self, sid: int, sock: socket.socket) -> None:
        try:
            while not self._closed.is_set():
                data = sock.recv(_CHUNK)
                if not data:
                    break
                self._send(sid, DATA, data)
        except OSError:
            pass
        finally:
            self._send(sid, CLOSE)
            self._drop(sid)

    def _register(self, sid: int, sock: Optional[socket.socket] = None) -> None:
        """Track a stream's queue now (so early DATA is kept); start its writer once a socket exists."""
        q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        with self._slock:
            self._queues[sid] = q
            self._qbytes[sid] = 0
        if sock is not None:
            self._attach(sid, sock)

    def _attach(self, sid: int, sock: socket.socket) -> None:
        with self._slock:
            q = self._queues.get(sid)
            if q is not None:
                self._streams[sid] = sock
        if q is None:                       # stream was closed while we were dialing
            try:
                sock.close()
            except OSError:
                pass
            return
        threading.Thread(target=self._writer, args=(sid, sock, q), daemon=True).start()

    def _writer(self, sid: int, sock: socket.socket, q) -> None:
        while True:
            data = q.get()
            if data is None:                 # aborted (error path): _drop already closed the socket
                return
            if data is _FIN:                 # the peer closed: everything before it is delivered
                self._drop(sid)
                return
            try:
                sock.sendall(data)
            except OSError:
                self._send(sid, CLOSE)
                self._drop(sid)
                return
            with self._slock:
                if sid in self._qbytes:
                    self._qbytes[sid] -= len(data)
                self._drained.notify_all()

    def _drop(self, sid: int) -> None:
        """Abort a stream NOW (errors, refusals, our own side ending). Queued data is discarded."""
        with self._slock:
            s = self._streams.pop(sid, None)
            q = self._queues.pop(sid, None)
            self._qbytes.pop(sid, None)
            self._drained.notify_all()
        if q is not None:
            q.put(None)
        if s:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass

    # ── the demux reader loop (run on a thread) ──
    def _reader(self, on_open) -> None:
        while not self._closed.is_set():
            hdr = self._recv_exact(_HDR.size)
            if hdr is None:
                break
            sid, typ, length = _HDR.unpack(hdr)
            if length > MAX_FRAME:
                break                                   # refuse before reading the body
            payload = self._recv_exact(length) if length else b""
            if length and payload is None:
                break
            if typ == OPEN:
                with self._slock:
                    full = len(self._queues) >= MAX_STREAMS or sid in self._queues
                if full:
                    self._send(sid, CLOSE)
                else:
                    self._register(sid)          # queue first: DATA may arrive before the dial ends
                    threading.Thread(target=on_open, args=(sid,), daemon=True).start()
            elif typ == DATA:
                stuck = False
                with self._slock:
                    # Backpressure: wait for this stream's writer to drain below the cap. The wait
                    # ends if the stream is dropped, the tunnel closes, or the stream is STUCK (no
                    # progress for STALL_S). A slow-but-moving stream is never cut.
                    last, since = self._qbytes.get(sid), time.monotonic()
                    while (sid in self._queues and self._qbytes[sid] > 0
                           and self._qbytes[sid] + len(payload) > MAX_QUEUED
                           and not self._closed.is_set()):
                        self._drained.wait(timeout=0.5)
                        now_q = self._qbytes.get(sid)
                        if now_q != last:
                            last, since = now_q, time.monotonic()
                        elif time.monotonic() - since > STALL_S:
                            stuck = True
                            break
                    q = None if stuck else self._queues.get(sid)
                    if q is not None:
                        self._qbytes[sid] += len(payload)
                        q.put(payload)
                if stuck:
                    self._send(sid, CLOSE)
                    self._drop(sid)
            elif typ == CLOSE:
                with self._slock:
                    q = self._queues.get(sid)
                if q is not None:
                    q.put(_FIN)   # graceful: the writer closes after the queued data (or once dialed)
        self._closed.set()

    # ── client side: listen locally, each conn → a stream ──
    def run_client(self, listen_host: str, listen_port: int) -> int:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((listen_host, listen_port))
        srv.listen(64)
        bound = srv.getsockname()[1]

        # client never receives OPENs (server dials); ignore.
        threading.Thread(target=self._reader, args=(lambda sid: None,), daemon=True).start()

        def accept_loop():
            while not self._closed.is_set():
                try:
                    conn, _ = srv.accept()
                except OSError:
                    break
                with self._slock:
                    sid = self._next_id
                    self._next_id += 2          # client uses odd ids
                self._register(sid, conn)
                self._send(sid, OPEN)
                threading.Thread(target=self._pump_local_to_stream,
                                 args=(sid, conn), daemon=True).start()

        threading.Thread(target=accept_loop, daemon=True).start()
        return bound

    # ── server side: each OPEN → dial TARGET, pipe ──
    def run_server(self, target_host: str, target_port: int) -> None:
        def on_open(sid: int):
            try:
                t = socket.create_connection((target_host, target_port), timeout=10)
            except OSError:
                self._send(sid, CLOSE)
                self._drop(sid)
                return
            self._attach(sid, t)
            threading.Thread(target=self._pump_local_to_stream,
                             args=(sid, t), daemon=True).start()

        self._reader(on_open)   # blocks; run in a thread if you need to return

    def close(self) -> None:
        self._closed.set()
        with self._slock:
            self._drained.notify_all()
        try:
            self._pipe.close()
        except OSError:
            pass
