"""Phase D tests for ``scripts/fabric/backend.py`` — the FabricBackend
Forward/Inference shim. Builds on the Phase A codec + Phase B transport.

Covers the falsifiable checks from the sprint plan Phase D:

  * handle_forward_packet maps a FORWARD packet → forward_fn call →
    outbound (FORWARD mid-chain, FEEDBACK at the last worker)
  * KV mapping: first packet per chain_id is cold prefill
    (keep_kv=False, start_pos=0); subsequent keep the cache + advance
  * Per-chain isolation: two chain_ids track independent start_pos
  * Failed decode doesn't advance the KV timeline (retry-safe)
  * Non-FORWARD packets + malformed payloads → drop (None)
  * End-to-end serve() loopback over real UDP FabricLinks: inbound
    FORWARD → daemon stub → forward neighbor receives the result
  * mode=last feedback wormhole: FORWARD → feedback link gets the token
"""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fabric import packet as fp  # noqa: E402
from fabric import backend as fb  # noqa: E402
from fabric.transport import FabricLink  # noqa: E402


KEY = b"K" * 16
CHAIN = 0xABCDEF
N_EMBD = 4
ELEM_BYTES = N_EMBD * 4               # f32 → 16 bytes per token


# ── fakes ────────────────────────────────────────────────────────────


class _FakeResult:
    """Duck-typed stand-in for worker.ForwardResult."""
    def __init__(self, ok, payload, error="", client_error=False):
        self.ok = ok
        self.payload = payload
        self.error = error
        self.client_error = client_error


def _echo_fn(transform=lambda b: b):
    """Build a forward_fn that records its calls + returns a transformed
    payload. The transform lets a test distinguish 'the daemon ran' from
    'the input passed through'."""
    calls = []

    def fn(payload, n_tokens, has_token_ids, keep_kv, start_pos):
        calls.append({
            "payload": payload, "n_tokens": n_tokens,
            "has_token_ids": has_token_ids, "keep_kv": keep_kv,
            "start_pos": start_pos,
        })
        return _FakeResult(True, transform(payload))

    fn.calls = calls
    return fn


def _header(*, packet_type=fp.PACKET_TYPE_FORWARD, chain_id=CHAIN,
            step_id=0, layer_idx=7, payload_length=ELEM_BYTES):
    return fp.FabricHeader(
        magic=fp.MAGIC, version_major=fp.VERSION_MAJOR,
        version_minor=fp.VERSION_MINOR, packet_type=packet_type,
        dtype=fp.DTYPE_FP32, flags=fp.FLAG_LAST_IN_STEP, reserved=0,
        chain_id=chain_id, step_id=step_id, layer_idx=layer_idx,
        seq=0, payload_length=payload_length, payload_offset=0,
        auth_tag=b"\x00" * 16,
    )


# ── 1. Construction guard ───────────────────────────────────────────


def test_backend_rejects_bad_mode():
    with pytest.raises(ValueError, match="first/middle/last"):
        fb.FabricBackend(_echo_fn(), "sideways", N_EMBD)


# ── 2. handle_forward_packet — dispatch + KV mapping ────────────────


def test_handle_forward_midchain_returns_forward_packet():
    """A mid-chain worker turns a FORWARD activation into another
    FORWARD activation for the next hop."""
    fn = _echo_fn(transform=lambda b: bytes((x + 1) & 0xFF for x in b))
    be = fb.FabricBackend(fn, "middle", N_EMBD)
    payload = b"\x01\x02\x03\x04" * N_EMBD       # 16 bytes = 1 token
    out = be.handle_forward_packet(_header(), payload)
    assert out is not None
    out_type, out_bytes = out
    assert out_type == fp.PACKET_TYPE_FORWARD
    assert out_bytes == bytes((x + 1) & 0xFF for x in payload)
    # First packet for this chain → cold prefill.
    assert fn.calls[0]["keep_kv"] is False
    assert fn.calls[0]["start_pos"] == 0
    assert fn.calls[0]["has_token_ids"] is False
    assert fn.calls[0]["n_tokens"] == 1


def test_handle_forward_last_worker_returns_feedback_packet():
    """The last worker produces a sampled token id → FEEDBACK wormhole,
    not another FORWARD."""
    fn = _echo_fn(transform=lambda b: b"\x07\x00\x00\x00")  # token id 7
    be = fb.FabricBackend(fn, "last", N_EMBD)
    out = be.handle_forward_packet(_header(), b"\x00" * ELEM_BYTES)
    assert out is not None
    out_type, out_bytes = out
    assert out_type == fp.PACKET_TYPE_FEEDBACK
    assert out_bytes == b"\x07\x00\x00\x00"


def test_kv_timeline_advances_across_steps():
    """Second+ packets for the same chain keep the KV cache and advance
    start_pos by the prior token count — mirrors the streaming
    Inference RPC's first_step + prefix_length accumulation."""
    fn = _echo_fn()
    be = fb.FabricBackend(fn, "middle", N_EMBD)
    # Step 0: 2 tokens (prefill).
    be.handle_forward_packet(
        _header(step_id=0, payload_length=2 * ELEM_BYTES),
        b"\x00" * (2 * ELEM_BYTES))
    # Step 1: 1 token (decode).
    be.handle_forward_packet(
        _header(step_id=1, payload_length=ELEM_BYTES),
        b"\x00" * ELEM_BYTES)
    assert fn.calls[0]["keep_kv"] is False
    assert fn.calls[0]["start_pos"] == 0
    assert fn.calls[0]["n_tokens"] == 2
    assert fn.calls[1]["keep_kv"] is True
    assert fn.calls[1]["start_pos"] == 2          # advanced by prefill's 2
    assert fn.calls[1]["n_tokens"] == 1


def test_per_chain_kv_isolation():
    """Two concurrent chains routed through one worker keep independent
    KV timelines — a step on chain A doesn't bump chain B's start_pos."""
    fn = _echo_fn()
    be = fb.FabricBackend(fn, "middle", N_EMBD)
    be.handle_forward_packet(_header(chain_id=111, step_id=0), b"\x00" * ELEM_BYTES)
    be.handle_forward_packet(_header(chain_id=222, step_id=0), b"\x00" * ELEM_BYTES)
    be.handle_forward_packet(_header(chain_id=111, step_id=1), b"\x00" * ELEM_BYTES)
    # chain 111: two steps → second sees start_pos=1
    # chain 222: one step → still a cold prefill
    assert fn.calls[0]["start_pos"] == 0          # 111 step 0
    assert fn.calls[1]["start_pos"] == 0          # 222 step 0 (independent)
    assert fn.calls[2]["start_pos"] == 1          # 111 step 1
    assert fn.calls[2]["keep_kv"] is True


def test_failed_decode_does_not_advance_kv():
    """A decode that fails must NOT bump prefix_length — otherwise the
    retry (or re-plan) would resume at the wrong start_pos and desync
    the KV cache."""
    results = [_FakeResult(False, b"", "daemon wedged"),
               _FakeResult(True, b"\x00" * ELEM_BYTES)]

    def fn(payload, n_tokens, has_token_ids, keep_kv, start_pos):
        fn.seen.append(start_pos)
        return results.pop(0)
    fn.seen = []

    be = fb.FabricBackend(fn, "middle", N_EMBD)
    out1 = be.handle_forward_packet(_header(step_id=0), b"\x00" * ELEM_BYTES)
    assert out1 is None                            # failed → dropped
    out2 = be.handle_forward_packet(_header(step_id=0), b"\x00" * ELEM_BYTES)
    assert out2 is not None
    # Both calls saw start_pos=0 — the failed first didn't advance it.
    assert fn.seen == [0, 0]


# ── 3. handle_forward_packet — drop paths ───────────────────────────


def test_non_forward_packet_dropped():
    """CONTROL / FEEDBACK aren't handled by the forward path — they're
    routed elsewhere (sampler / rekey). handle_forward_packet drops."""
    be = fb.FabricBackend(_echo_fn(), "middle", N_EMBD)
    assert be.handle_forward_packet(
        _header(packet_type=fp.PACKET_TYPE_CONTROL), b"\x00" * ELEM_BYTES
    ) is None


def test_payload_not_whole_vectors_dropped():
    """A payload that isn't a whole number of hidden vectors is a
    chain-plan dtype/shape mismatch — drop rather than feed garbage to
    the daemon."""
    be = fb.FabricBackend(_echo_fn(), "middle", N_EMBD)
    # 17 bytes — not a multiple of 16 (n_embd * 4)
    assert be.handle_forward_packet(_header(payload_length=17),
                                     b"\x00" * 17) is None


def test_empty_payload_dropped():
    be = fb.FabricBackend(_echo_fn(), "middle", N_EMBD)
    assert be.handle_forward_packet(_header(payload_length=0), b"") is None


# ── 4. serve() loop end-to-end over real UDP links ──────────────────


def _udp_sock():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    return s


def test_serve_loop_forwards_decoded_activation():
    """The headline Phase D check: a FORWARD activation arriving on the
    backend's inbound link is decoded and the result lands on the
    forward neighbor's link, all over real loopback UDP.

    Topology:
        prev_worker --FORWARD--> [backend inbound]
                                      decode (echo+1)
        [backend forward] --FORWARD--> next_worker
    """
    backend_sock = _udp_sock()
    prev_sock = _udp_sock()
    next_sock = _udp_sock()
    backend_addr = backend_sock.getsockname()
    prev_addr = prev_sock.getsockname()
    next_addr = next_sock.getsockname()

    # Backend's two links share its one UDP socket (mirrors
    # setup_fabric). Source-pinning keeps inbound recv'ing only from
    # the prev worker.
    inbound = FabricLink(backend_sock, prev_addr, KEY, CHAIN)
    forward = FabricLink(backend_sock, next_addr, KEY, CHAIN)
    # The prev worker sends INTO the backend; the next worker recvs
    # what the backend forwards (source will be backend_addr).
    prev_link = FabricLink(prev_sock, backend_addr, KEY, CHAIN)
    next_link = FabricLink(next_sock, backend_addr, KEY, CHAIN)

    fn = _echo_fn(transform=lambda b: bytes((x + 1) & 0xFF for x in b))
    be = fb.FabricBackend(fn, "middle", N_EMBD)
    be.set_links(inbound=inbound, forward=forward)

    t = threading.Thread(target=be.serve, kwargs={"recv_timeout": 0.2},
                          daemon=True)
    t.start()
    try:
        payload = b"\x10\x20\x30\x40" * N_EMBD     # 16 bytes, 1 token
        prev_link.send(payload, packet_type=fp.PACKET_TYPE_FORWARD,
                       step_id=0, layer_idx=7)
        got = next_link.recv(timeout=2.0)
        assert got is not None
        _, out_payload = got
        assert out_payload == bytes((x + 1) & 0xFF for x in payload)
        assert len(fn.calls) == 1
    finally:
        be.stop()
        t.join(timeout=2.0)
        for s in (backend_sock, prev_sock, next_sock):
            s.close()


def test_serve_loop_last_worker_sends_feedback():
    """A last-worker backend ships its sampled token over the feedback
    link, not the forward link."""
    backend_sock = _udp_sock()
    prev_sock = _udp_sock()
    head_sock = _udp_sock()                        # the first worker
    backend_addr = backend_sock.getsockname()
    prev_addr = prev_sock.getsockname()
    head_addr = head_sock.getsockname()

    inbound = FabricLink(backend_sock, prev_addr, KEY, CHAIN)
    feedback = FabricLink(backend_sock, head_addr, KEY, CHAIN)
    prev_link = FabricLink(prev_sock, backend_addr, KEY, CHAIN)
    head_link = FabricLink(head_sock, backend_addr, KEY, CHAIN)

    fn = _echo_fn(transform=lambda b: b"\x2a\x00\x00\x00")   # token 42
    be = fb.FabricBackend(fn, "last", N_EMBD)
    be.set_links(inbound=inbound, feedback=feedback)

    t = threading.Thread(target=be.serve, kwargs={"recv_timeout": 0.2},
                          daemon=True)
    t.start()
    try:
        prev_link.send(b"\x00" * ELEM_BYTES,
                       packet_type=fp.PACKET_TYPE_FORWARD, step_id=0)
        got = head_link.recv(timeout=2.0)
        assert got is not None
        header, token_bytes = got
        assert header.packet_type == fp.PACKET_TYPE_FEEDBACK
        assert token_bytes == b"\x2a\x00\x00\x00"
    finally:
        be.stop()
        t.join(timeout=2.0)
        for s in (backend_sock, prev_sock, head_sock):
            s.close()


def test_serve_requires_inbound_link():
    be = fb.FabricBackend(_echo_fn(), "middle", N_EMBD)
    with pytest.raises(RuntimeError, match="inbound"):
        be.serve()


def test_stop_is_idempotent():
    be = fb.FabricBackend(_echo_fn(), "middle", N_EMBD)
    be.stop()
    be.stop()        # no exception


# ── 5. _send_onward guards ──────────────────────────────────────────


def test_send_onward_drops_when_link_missing():
    """A misconfigured bringup (last worker with no feedback link, or
    mid-chain with no forward link) drops rather than crashing the
    serve loop."""
    be = fb.FabricBackend(_echo_fn(), "last", N_EMBD)
    # No feedback link wired — _send_onward should no-op, not raise.
    be._send_onward(_header(), fp.PACKET_TYPE_FEEDBACK, b"\x00\x00\x00\x00")
    be2 = fb.FabricBackend(_echo_fn(), "middle", N_EMBD)
    be2._send_onward(_header(), fp.PACKET_TYPE_FORWARD, b"\x00" * ELEM_BYTES)
