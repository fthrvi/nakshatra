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
This is a *spike* mux (no flow control / window) — fine for the bursty, modest
activation traffic of a 1B chain. Production = WireGuard/QUIC.
"""
from __future__ import annotations

import socket
import struct
import threading
from typing import Optional

_HDR = struct.Struct(">IBI")
OPEN, DATA, CLOSE = 1, 2, 3
_CHUNK = 65536


class MuxTunnel:
    def __init__(self, pipe: socket.socket):
        self._pipe = pipe
        self._wlock = threading.Lock()
        self._streams: dict[int, socket.socket] = {}
        self._slock = threading.Lock()
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

    def _drop(self, sid: int) -> None:
        with self._slock:
            s = self._streams.pop(sid, None)
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
            payload = self._recv_exact(length) if length else b""
            if length and payload is None:
                break
            if typ == OPEN:
                on_open(sid)
            elif typ == DATA:
                with self._slock:
                    s = self._streams.get(sid)
                if s:
                    try:
                        s.sendall(payload)
                    except OSError:
                        self._drop(sid)
            elif typ == CLOSE:
                self._drop(sid)
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
                    self._streams[sid] = conn
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
                return
            with self._slock:
                self._streams[sid] = t
            threading.Thread(target=self._pump_local_to_stream,
                             args=(sid, t), daemon=True).start()

        self._reader(on_open)   # blocks; run in a thread if you need to return

    def close(self) -> None:
        self._closed.set()
        try:
            self._pipe.close()
        except OSError:
            pass
