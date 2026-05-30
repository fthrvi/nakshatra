"""Phase A.2 integration tests for ``scripts/fabric/shm_daemon_client.py``.

Real subprocess round-trips through the Python fake daemon
(``scripts/fabric/_fake_daemon.py``) — proves the wire envelope
preservation + the ShmDaemonClient ↔ daemon handshake without
requiring a built C++ binary (that's Phase A.3).

Covers:
  * Construction + ready handshake (proc spawn, ring create, CMD_INFO
    probe succeeds, info() returns the fake's expected struct)
  * CMD_INFO matches worker.DaemonClient.info()'s shape byte-for-byte
  * EMBD_DECODE round-trip: payload XOR'd by 0x55 (fake transform)
    arrives back intact after stripping the rtype prefix
  * Many sequential calls — ring cursor advances, no premature wrap
  * Lock serialises concurrent callers (multi-thread test)
  * Slow daemon startup: ready_timeout honoured + raises cleanly
  * Daemon-died-mid-call: caller gets RuntimeError, not a hang
  * close() unlinks both ring files (no /tmp leakage)
"""
from __future__ import annotations

import os
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fabric.shm_daemon_client import (   # noqa: E402
    CMD_EMBD_DECODE, CMD_INFO, CMD_TOKEN_DECODE,
    ShmDaemonClient,
)


# The fake daemon is the "binary" we point ShmDaemonClient at. Use
# the python3 interpreter from this env to ensure it runs against
# the same fabric.shm_ring module the test imports.
_FAKE_DAEMON_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "fabric" / "_fake_daemon.py"
)
assert _FAKE_DAEMON_PATH.exists()


def _build_client(*, ring_dir, mode="middle", ring_capacity=1 << 20,
                   ready_timeout_s=5.0, startup_delay_s=0.0,
                   n_embd=4):
    """Spawn a ShmDaemonClient pointed at the fake daemon. Returns
    the client; caller is responsible for ``client.close()``.

    We use the current Python interpreter as the "daemon binary" and
    run _fake_daemon.py as its argv[0]. argv[1..5] match the real
    daemon's positional args; --fabric-shm-{req,resp} are appended
    by ShmDaemonClient itself.

    The fake-only flags (--fake-startup-delay-s, --fake-n-embd) ride
    along — the real daemon will ignore them or fail loud; the fake
    accepts them. This is OK because the fake is the only thing the
    test ever spawns.
    """
    # Wrap sys.executable + the fake script as a single "daemon
    # binary". ShmDaemonClient does Popen([daemon_bin, sub_gguf,
    # mode, ...]); the fake just needs argv[0] to be the script.
    # Easiest: write a tiny shim that calls python on the script.
    shim = ring_dir / "fake-daemon-shim.sh"
    shim.write_text(
        "#!/bin/sh\n"
        f"exec {sys.executable} {_FAKE_DAEMON_PATH} "
        f"--fake-startup-delay-s {startup_delay_s} "
        f"--fake-n-embd {n_embd} \"$@\"\n"
    )
    shim.chmod(0o755)
    return ShmDaemonClient(
        daemon_bin=str(shim),
        sub_gguf="/dev/null",                # fake doesn't load it
        mode=mode,
        n_ctx=256, n_threads=1, n_gpu_layers=0,
        ring_capacity=ring_capacity,
        ring_dir=ring_dir,
        ready_timeout_s=ready_timeout_s,
    )


# ── 1. Construction + ready handshake ───────────────────────────────


def test_construction_spawns_fake_daemon_and_completes_ready_probe(tmp_path):
    """Constructor blocks until CMD_INFO succeeds — no other test
    needs to think about ready handshake, but assert it works at
    least once."""
    client = _build_client(ring_dir=tmp_path)
    try:
        # If we got here, _wait_ready returned cleanly. info() should
        # also work since the constructor's probe used it.
        info = client.info()
        assert info["n_embd"] == 4
        assert info["layer_start"] == 0
        assert info["layer_end"] == 14
    finally:
        client.close()


def test_info_shape_matches_daemon_client_contract(tmp_path):
    """The CMD_INFO response must parse via the same 6×i32 layout
    worker.DaemonClient.info() uses — keeps the cross-implementation
    contract honest."""
    client = _build_client(ring_dir=tmp_path, mode="first", n_embd=8)
    try:
        info = client.info()
        assert info["n_embd"] == 8
        assert info["has_token_embd"] is True   # mode=first
        assert info["has_lm_head"] is False
        assert info["n_vocab"] == 32000
    finally:
        client.close()


def test_construction_raises_on_slow_daemon_when_timeout_short(tmp_path):
    """A daemon that's still loading the model when the ready
    timeout elapses must raise TimeoutError — operators expect a
    bounded boot time. The fake's --fake-startup-delay-s gives us a
    deterministic slow start."""
    with pytest.raises(TimeoutError):
        _build_client(
            ring_dir=tmp_path,
            startup_delay_s=3.0,
            ready_timeout_s=0.5,
        )


# ── 2. CMD round-trips ──────────────────────────────────────────────


def test_embd_decode_round_trip_preserves_payload(tmp_path):
    """Send a 64-byte fp32 input (4 tokens × 4 n_embd × 4 bytes); the
    fake XORs each byte with 0x55; assert the output matches the
    transform AND the rtype prefix was stripped."""
    client = _build_client(ring_dir=tmp_path, n_embd=4)
    try:
        payload_in = bytes(range(64))
        status, payload_out = client.call(
            CMD_EMBD_DECODE, n_tokens=4, payload=payload_in,
            start_pos=0, flags=0,
        )
        assert status == 0
        # The fake's _handle_request prefixes its decode output with
        # a 4-byte rtype tag; the client returns the FULL payload
        # (including the prefix) per the existing contract — callers
        # like _run_forward strip it explicitly.
        assert len(payload_out) == 4 + 64
        # First 4 bytes are the rtype prefix (0x00000000 in the fake).
        assert payload_out[:4] == b"\x00" * 4
        # Remaining bytes are XOR'd input.
        assert payload_out[4:] == bytes((b ^ 0x55) for b in payload_in)
    finally:
        client.close()


def test_last_worker_token_decode_returns_single_int32(tmp_path):
    """mode=last + CMD_TOKEN_DECODE produces a single token id —
    matches the real daemon's mode-last decode contract."""
    client = _build_client(ring_dir=tmp_path, mode="last")
    try:
        status, payload = client.call(
            CMD_TOKEN_DECODE, n_tokens=1, payload=struct.pack("<i", 7),
        )
        assert status == 0
        # 4-byte rtype prefix + 4-byte int32 token id.
        assert len(payload) == 8
        (token,) = struct.unpack("<i", payload[4:])
        assert token == 42   # the fake's hardcoded token
    finally:
        client.close()


def test_unknown_command_returns_nonzero_status(tmp_path):
    """A command code the daemon doesn't recognise should surface as
    status != 0 rather than crash the daemon or hang the client."""
    client = _build_client(ring_dir=tmp_path)
    try:
        status, payload = client.call(cmd=99, n_tokens=0, payload=b"")
        assert status != 0
        assert payload == b""
    finally:
        client.close()


# ── 3. Many sequential calls ────────────────────────────────────────


def test_hundred_sequential_decodes(tmp_path):
    """Stress: 100 round trips through the same client. Catches
    ring-cursor drift, lock contention bugs, daemon-side state leaks."""
    client = _build_client(ring_dir=tmp_path, n_embd=4)
    try:
        payload = bytes([0xAB]) * 16    # 1 token × 4 n_embd × 4 = 16 bytes
        for i in range(100):
            status, out = client.call(
                CMD_EMBD_DECODE, n_tokens=1, payload=payload, start_pos=i,
            )
            assert status == 0
            assert out[4:] == bytes((b ^ 0x55) for b in payload)
    finally:
        client.close()


# ── 4. Concurrent callers (lock contract) ───────────────────────────


def test_concurrent_threads_serialised_by_lock(tmp_path):
    """ShmDaemonClient.call holds an internal lock so multiple threads
    can call() simultaneously without corrupting the SPSC rings. The
    lock is the contract; this test catches a regression that drops
    it (which would interleave req frames and reorder responses)."""
    client = _build_client(ring_dir=tmp_path, n_embd=4)
    failures: list[str] = []
    seen_outputs: list[int] = []
    lock = threading.Lock()

    def worker(tag: int):
        try:
            payload = bytes([tag & 0xFF]) * 16
            status, out = client.call(
                CMD_EMBD_DECODE, n_tokens=1, payload=payload,
            )
            if status != 0:
                with lock:
                    failures.append(f"tag {tag}: status {status}")
                return
            # Round trip should yield (tag ^ 0x55) bytes — note
            # bytes([N]) gives a 1-byte buffer containing N, where
            # bytes(N) gives N zero bytes.
            expected = bytes([(tag & 0xFF) ^ 0x55]) * 16
            if out[4:] != expected:
                with lock:
                    failures.append(
                        f"tag {tag}: out mismatch")
                return
            with lock:
                seen_outputs.append(tag)
        except Exception as e:
            with lock:
                failures.append(f"tag {tag}: raised {e!r}")

    try:
        threads = [
            threading.Thread(target=worker, args=(i,), daemon=True)
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        assert not failures, failures
        assert sorted(seen_outputs) == list(range(20))
    finally:
        client.close()


# ── 5. Daemon-died-mid-flight ───────────────────────────────────────


def test_daemon_kill_surfaces_runtime_error(tmp_path):
    """If the daemon dies between a successful ready handshake and a
    later call, the next call must raise RuntimeError quickly — NOT
    hang waiting for a response that never arrives. Catches a
    poll-loop missing-the-died-check regression."""
    client = _build_client(ring_dir=tmp_path)
    try:
        # First call works.
        client.info()
        # Kill the daemon process out from under us.
        client.proc.kill()
        client.proc.wait(timeout=2.0)
        # Next call should detect the dead proc + raise.
        with pytest.raises(RuntimeError, match="daemon exited"):
            client.call(CMD_EMBD_DECODE, n_tokens=1, payload=b"\x00" * 16)
    finally:
        client.close()


# ── 6. Cleanup ──────────────────────────────────────────────────────


def test_close_unlinks_both_ring_files(tmp_path):
    """close() must unlink both /tmp/<uuid>-req.ring and -resp.ring
    so a long-running worker process doesn't leak ring files into
    /tmp on every daemon restart."""
    client = _build_client(ring_dir=tmp_path)
    req_path = client._req_path
    resp_path = client._resp_path
    assert req_path.exists()
    assert resp_path.exists()
    client.close()
    assert not req_path.exists()
    assert not resp_path.exists()


def test_close_is_idempotent(tmp_path):
    client = _build_client(ring_dir=tmp_path)
    client.close()
    client.close()      # no exception
