"""ShmDaemonClient — drop-in mirror of worker.DaemonClient that talks
to the worker daemon over two shared-memory rings instead of stdin/
stdout pipes.

Phase A.2 of the C++ kernel-bypass sprint. Builds on Phase A.1's
ShmRing primitive; the next phase (A.3) replaces the C++ daemon's
stdio I/O with shm I/O so the Python ShmDaemonClient and the C++
daemon meet in the middle.

**Same `call()` signature as DaemonClient.** Callers (`WorkerServicer
._run_forward`, `Forward` / `Inference` RPCs, the fabric backend)
never know they're talking to a shm daemon vs a stdio daemon. The
existing daemon wire envelope is preserved byte-identically:

  Request frame  = u32 cmd | u32 n_tokens | u32 start_pos | u32 flags
                   | u32 payload_bytes | <payload>
  Response frame = u32 status | u32 payload_bytes | <payload>

The ring's own length prefix wraps the entire request/response frame;
the daemon's wire format doesn't need to know about shm framing.

**A.2 uses two file-backed rings** at `/tmp/<uuid>-{req,resp}.ring`,
passing paths via daemon argv. Phase A.2.x (or a later hardening
pass) may switch to SCM_RIGHTS over a Unix socket — the daemon-side
diff is small and isolated; the ring wire layout is unchanged either
way. Sprint plan OQ7.

**Polling loop**, not eventfd/futex. A.3 measurement will tell us
whether the few µs of poll overhead is worth replacing with a real
notification primitive (Linux eventfd vs Mac kevent — platform split).
For A.2's prototype scope, a 50µs sleep loop is fine.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fabric.shm_ring import ShmRing


# ── Wire constants (mirror worker.DaemonClient verbatim) ────────────


# Daemon commands. Same values worker.py's CMD_TOKEN_DECODE / _EMBD_
# DECODE / _INFO carry — keep in sync if those move.
CMD_TOKEN_DECODE = 1
CMD_EMBD_DECODE = 2
CMD_INFO = 3

# 5 × u32 request header — exactly what worker.DaemonClient packs.
_REQ_HEADER_FMT = "<IIIII"
_REQ_HEADER_SIZE = struct.calcsize(_REQ_HEADER_FMT)
assert _REQ_HEADER_SIZE == 20

# 2 × u32 response header — daemon's reply prefix.
_RESP_HEADER_FMT = "<II"
_RESP_HEADER_SIZE = struct.calcsize(_RESP_HEADER_FMT)
assert _RESP_HEADER_SIZE == 8


# ── Defaults ────────────────────────────────────────────────────────


# Ring size — must comfortably fit the largest activation + envelope.
# 8 MiB carries a 16 KB activation (typical Llama-class layer
# boundary) ~512× over, leaves headroom for slow consumers without
# back-pressuring producers. Constructor arg overrides.
DEFAULT_RING_CAPACITY = 8 * 1024 * 1024

# Poll interval between empty reads on the response ring. 50µs is well
# below typical decode latencies (ms-range) — operator never notices
# the poll overhead, and we avoid a tight CPU-burning spin loop.
# A.3 measurement may justify replacing with eventfd-based wakeup.
_POLL_INTERVAL_S = 5e-5

# Where temp rings live. /tmp is the cross-platform default; operators
# can override via env if a different volume is mounted (e.g. tmpfs
# explicit, or a faster scratch SSD).
RING_DIR = Path(os.environ.get(
    "NAKSHATRA_FABRIC_RING_DIR", "/tmp"))


# ── ShmDaemonClient ─────────────────────────────────────────────────


class ShmDaemonClient:
    """Manages a long-lived worker daemon subprocess that consumes /
    produces framed messages over two shared-memory rings.

    Construction signature mirrors worker.DaemonClient so a future
    refactor can swap them behind a feature flag without callers
    seeing the diff. The daemon binary contract is extended (not
    broken) — Phase A.3 will teach `llama-nakshatra-worker` to accept
    the new shm-path args; until then this client is exercised
    against `scripts/fabric/_fake_daemon.py`.
    """

    def __init__(
        self,
        daemon_bin: str,
        sub_gguf: str,
        mode: str,
        n_ctx: int,
        n_threads: int = 0,
        n_gpu_layers: int = 0,
        *,
        ring_capacity: int = DEFAULT_RING_CAPACITY,
        ring_dir: Optional[Path] = None,
        ready_timeout_s: float = 60.0,
    ):
        if ring_dir is None:
            ring_dir = RING_DIR
        ring_dir.mkdir(parents=True, exist_ok=True)
        # Random per-instance ring paths — multiple workers on the
        # same host don't collide. Parent owns; close() unlinks.
        run_id = uuid.uuid4().hex[:12]
        self._req_path = ring_dir / f"nakshatra-shm-{run_id}-req.ring"
        self._resp_path = ring_dir / f"nakshatra-shm-{run_id}-resp.ring"
        self._req_ring = ShmRing.create(self._req_path, ring_capacity)
        self._resp_ring = ShmRing.create(self._resp_path, ring_capacity)
        # Daemon argv: keep the existing positional args (sub_gguf,
        # mode, n_ctx, n_threads, n_gpu_layers) for back-compat with
        # the stdio path; add --fabric-shm-req/resp as keyword flags
        # the daemon recognises in shm-mode.
        self.proc = subprocess.Popen(
            [
                daemon_bin, sub_gguf, mode,
                str(n_ctx), str(n_threads), str(n_gpu_layers),
                "--fabric-shm-req", str(self._req_path),
                "--fabric-shm-resp", str(self._resp_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.lock = threading.Lock()
        # Mirror worker.DaemonClient's stderr buffer + drain thread.
        import collections
        self.stderr_lines = collections.deque(maxlen=500)
        self.recent_rpc_ms = collections.deque(maxlen=20)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        # Stay-in-sync with the stdio DaemonClient's ready check:
        # poll CMD_INFO until the daemon responds or we time out.
        self._wait_ready(timeout_s=ready_timeout_s)

    # ── stderr drain (mirrors worker.DaemonClient._drain_stderr) ──

    def _drain_stderr(self) -> None:
        if self.proc.stderr is None:
            return
        for line in iter(self.proc.stderr.readline, b""):
            text = line.decode("utf-8", "replace")
            self.stderr_lines.append(text)
            sys.stderr.write(f"[daemon] {text}")
            sys.stderr.flush()

    # ── ready handshake ──────────────────────────────────────────

    def _wait_ready(self, *, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        # Each probe gets a short per-attempt cap so a slow daemon
        # can't wedge the call's inner poll loop forever (which would
        # bypass the outer ``deadline`` check entirely).
        per_attempt = max(0.05, min(timeout_s / 4.0, 1.0))
        while time.monotonic() < deadline:
            try:
                self.call(CMD_INFO, 0, b"",
                          response_timeout_s=per_attempt,
                          _internal_ready_probe=True)
                return
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"daemon exited rc={self.proc.returncode} "
                        f"before becoming ready"
                    )
                time.sleep(0.02)
        raise TimeoutError("shm daemon never became ready")

    # ── The actually-load-bearing API ────────────────────────────

    def call(
        self,
        cmd: int,
        n_tokens: int,
        payload: bytes,
        start_pos: int = 0,
        flags: int = 0,
        *,
        response_timeout_s: Optional[float] = None,
        _internal_ready_probe: bool = False,
    ) -> tuple[int, bytes]:
        """Send one request frame, block until the response frame
        arrives. Returns ``(status, response_payload)`` — same
        contract as worker.DaemonClient.call.

        The lock serialises concurrent callers; each call is one
        request frame in + one response frame out, in order. The
        underlying SPSC ring relies on this — multiple producers on
        the same ring would corrupt the cursors.

        ``_internal_ready_probe`` is the ready-handshake escape
        hatch: it skips the proc-died check + timing accounting so
        repeated probes during startup don't fail loud or pollute
        recent_rpc_ms with cold-start latency.
        """
        with self.lock:
            t0 = time.monotonic()
            # Pack request envelope.
            hdr = struct.pack(_REQ_HEADER_FMT,
                              cmd, n_tokens, start_pos, flags, len(payload))
            req_frame = hdr + payload
            # Write into the request ring. Spin on full (the daemon
            # may be slow to drain on the first call; subsequent
            # calls should land instantly with 8 MiB rings).
            while not self._req_ring.write_message(req_frame):
                if not _internal_ready_probe and self.proc.poll() is not None:
                    raise RuntimeError(
                        f"daemon exited rc={self.proc.returncode} "
                        f"during write"
                    )
                time.sleep(_POLL_INTERVAL_S)
            # Poll for the response. ``response_timeout_s``, when set,
            # caps how long we'll wait — ready probes use a short cap
            # so a daemon that's silently mid-load can't hang the
            # ready handshake's outer timeout loop.
            response_frame: Optional[bytes] = None
            resp_deadline = (time.monotonic() + response_timeout_s
                             if response_timeout_s is not None else None)
            while True:
                response_frame = self._resp_ring.read_message()
                if response_frame is not None:
                    break
                if self.proc.poll() is not None:
                    if _internal_ready_probe:
                        raise RuntimeError("daemon not ready yet")
                    raise RuntimeError(
                        f"daemon exited rc={self.proc.returncode} "
                        f"during recv"
                    )
                if resp_deadline is not None and time.monotonic() > resp_deadline:
                    raise TimeoutError(
                        f"daemon response not received within "
                        f"{response_timeout_s}s"
                    )
                time.sleep(_POLL_INTERVAL_S)
            # Unpack response envelope.
            if len(response_frame) < _RESP_HEADER_SIZE:
                raise RuntimeError(
                    f"daemon response frame too short: "
                    f"{len(response_frame)} < {_RESP_HEADER_SIZE}"
                )
            status, plen = struct.unpack(
                _RESP_HEADER_FMT, response_frame[:_RESP_HEADER_SIZE])
            data = response_frame[_RESP_HEADER_SIZE:_RESP_HEADER_SIZE + plen]
            if len(data) != plen:
                raise RuntimeError(
                    f"daemon response payload length mismatch: "
                    f"declared {plen}, got {len(data)}"
                )
            # Timing accounting — same shape as DaemonClient. Inline
            # the NaN/Inf guard rather than late-importing worker.py;
            # ShmDaemonClient gets deployed standalone (Phase A.3
            # cluster smokes don't ship worker.py).
            if cmd != CMD_INFO and not _internal_ready_probe:
                raw = (time.monotonic() - t0) * 1000.0
                try:
                    clean = float(raw)
                    if clean == clean and -1e9 < clean < 1e9 and clean >= 0:
                        self.recent_rpc_ms.append(clean)
                except (ValueError, TypeError):
                    pass
            return status, data

    # ── Mirror worker.DaemonClient.info() ────────────────────────

    def info(self) -> dict:
        s, p = self.call(CMD_INFO, 0, b"")
        if s != 0 or len(p) < 24:
            raise RuntimeError(f"info call failed status={s} len={len(p)}")
        layer_start, layer_end, n_embd, has_embd, has_lm, n_vocab = \
            struct.unpack("<6i", p[:24])
        return dict(layer_start=layer_start, layer_end=layer_end,
                    n_embd=n_embd, has_token_embd=bool(has_embd),
                    has_lm_head=bool(has_lm), n_vocab=n_vocab)

    # ── Teardown ─────────────────────────────────────────────────

    def close(self) -> None:
        """Terminate the daemon + close + unlink both rings. Safe to
        call multiple times."""
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        # Close rings — these unlink the files (parent owns both).
        try:
            self._req_ring.close()
        except Exception:
            pass
        try:
            self._resp_ring.close()
        except Exception:
            pass
