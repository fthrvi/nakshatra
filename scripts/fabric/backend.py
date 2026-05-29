"""Fabric backend — the Forward/Inference shim that runs the worker's
data plane over UDP fabric links instead of gRPC.

Phase D of the fabric_lite sprint. Builds on:
  - Phase A (`fabric.packet`) — the wire codec
  - Phase B (`fabric.transport.FabricLink`) — UDP link + chunking
  - Phase C (`fabric.join.JoinClient`) — keyring + neighbor blocks

The backend is deliberately transport-mechanics only. It does NOT run
the model — it calls a ``forward_fn`` callable (in practice
``WorkerServicer._run_forward``) that owns the daemon subprocess and
the actual decode. This keeps the daemon-call semantics in ONE place
(worker.py), so the gRPC path and the fabric path can never drift in
how they invoke the model.

Data flow for a mid-chain worker:

    backward neighbor --FORWARD--> [inbound_link.recv]
                                        |
                                   forward_fn(activation)   # daemon decode
                                        |
    [forward_link.send] --FORWARD--> forward neighbor

For the last worker, the produced token goes back to the first worker
over a feedback link (the 4-byte FEEDBACK wormhole), not onward.

The first worker's inbound side is the client's gRPC ``Forward`` call,
not a fabric link — per the sprint plan's open question 8, the first
worker translates gRPC-in to fabric-out. That bridge lives in
worker.py's boot wiring (Phase D step 7), not here; this module
handles the worker↔worker fabric hops.
"""
from __future__ import annotations

import struct
import sys
import threading
from pathlib import Path
from typing import Callable, Optional, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fabric import packet as fp
from fabric.transport import FabricLink


# ── forward_fn contract ─────────────────────────────────────────────


class _ForwardResultLike(Protocol):
    """Duck-typed view of worker.ForwardResult. Defined here so the
    backend never has to import worker.py (which pulls in grpc + the
    daemon). worker.py passes its ``_run_forward`` bound method as
    ``forward_fn`` and the backend reads these attributes off the
    result. Keeps the dependency arrow one-way: worker → fabric."""
    ok: bool
    payload: bytes
    error: str
    client_error: bool


ForwardFn = Callable[[bytes, int, bool, bool, int], _ForwardResultLike]


# ── per-chain decode state ──────────────────────────────────────────


class _ChainState:
    """Tracks the KV-cache position for one chain_id. Mirrors the
    streaming Inference RPC's ``first_step`` + ``prefix_length``
    accumulation (worker.py): the first FORWARD for a chain is a cold
    prefill (keep_kv=False, start_pos=0); each subsequent step keeps
    the KV cache and advances start_pos by the token count.

    Keyed per chain_id rather than globally because one worker may
    participate in multiple concurrent chains (different inferences
    routed through the same physical worker) — each has its own KV
    timeline. (The daemon itself is single-context today, so concurrent
    chains would interleave KV state incorrectly — that's a known
    fabric_lite v0 limitation matching the existing gRPC behavior, not
    a regression this layer introduces.)"""

    __slots__ = ("prefix_length",)

    def __init__(self):
        self.prefix_length: int = 0


# ── FabricBackend ────────────────────────────────────────────────────


class FabricBackend:
    """Runs the worker's forward-step data plane over fabric links.

    Construction (from worker.py boot, --transport=fabric):

        backend = FabricBackend(
            forward_fn=servicer._run_forward,
            mode="middle",
            n_embd=info.hidden_size,
            wire_dtype_bytes=4,            # f32 today
        )
        backend.set_links(inbound=backward_link, forward=forward_link)
        backend.serve()                   # blocks; run in a thread

    The backend reads FORWARD packets off the inbound link, runs the
    decode via forward_fn, and ships the result to the forward link
    (mid-chain) or the feedback link (last worker). Errors increment
    the inbound link's counters and drop the step — schema §8's
    "no retransmit; chronic loss → re-plan" stance.
    """

    def __init__(
        self,
        forward_fn: ForwardFn,
        mode: str,
        n_embd: int,
        *,
        wire_dtype_bytes: int = 4,
        dtype: int = fp.DTYPE_FP32,
    ):
        if mode not in ("first", "middle", "last"):
            raise ValueError(f"mode must be first/middle/last, got {mode!r}")
        self.forward_fn = forward_fn
        self.mode = mode
        self.n_embd = n_embd
        self.wire_dtype_bytes = wire_dtype_bytes
        self.dtype = dtype

        self.inbound_link: Optional[FabricLink] = None
        self.forward_link: Optional[FabricLink] = None
        self.feedback_link: Optional[FabricLink] = None

        self._chains: dict[int, _ChainState] = {}
        self._stop = threading.Event()

    # ── Link wiring ────────────────────────────────────────────────

    def set_links(
        self,
        *,
        inbound: Optional[FabricLink] = None,
        forward: Optional[FabricLink] = None,
        feedback: Optional[FabricLink] = None,
    ) -> None:
        """Attach the directional links. A mid-chain worker has inbound
        (recv from backward neighbor) + forward (send to next). The
        last worker has inbound + feedback (4-byte token back to the
        first worker). The first worker has forward only — its inbound
        side is the client's gRPC call, bridged in worker.py."""
        if inbound is not None:
            self.inbound_link = inbound
        if forward is not None:
            self.forward_link = forward
        if feedback is not None:
            self.feedback_link = feedback

    # ── Core dispatch (unit-testable without sockets) ──────────────

    def handle_forward_packet(
        self, header: fp.FabricHeader, payload: bytes,
    ) -> Optional[tuple[int, bytes]]:
        """Map one inbound FORWARD packet to a decode + produce the
        outbound payload. Returns ``(out_packet_type, out_bytes)`` to
        send onward, or ``None`` on error / unexpected packet (caller
        counts + drops).

        KV mapping: the chain's ``_ChainState`` decides keep_kv +
        start_pos exactly as the streaming Inference RPC does — first
        packet for a chain_id is a cold prefill, rest keep the cache.

        Mid-chain hidden_state arrives as fabric FORWARD; this worker
        always treats fabric input as hidden_state (``has_token_ids =
        False``) — the first worker's token-id input comes over gRPC,
        not fabric (sprint open question 8).
        """
        if header.packet_type != fp.PACKET_TYPE_FORWARD:
            # CONTROL / FEEDBACK aren't handled by the forward path.
            # FEEDBACK is consumed by the first worker's sampler (not
            # this method); CONTROL is rekey/drain (Phase E+).
            return None

        elem_bytes = self.n_embd * self.wire_dtype_bytes
        if elem_bytes <= 0 or len(payload) % elem_bytes != 0:
            # Payload isn't a whole number of hidden vectors — a
            # chain-plan dtype/shape mismatch. Drop; caller counts.
            return None
        n_tokens = len(payload) // elem_bytes
        if n_tokens == 0:
            return None

        chain = self._chains.get(header.chain_id)
        if chain is None:
            chain = _ChainState()
            self._chains[header.chain_id] = chain
        keep_kv = chain.prefix_length > 0
        start_pos = chain.prefix_length

        result = self.forward_fn(
            payload, n_tokens, False, keep_kv, start_pos,
        )
        if not result.ok:
            return None
        # Advance the KV timeline only after a successful decode — a
        # failed step must not desync start_pos for the retry/re-plan.
        chain.prefix_length += n_tokens

        if self.mode == "last":
            # Produced a single int32 token id — ships as FEEDBACK.
            return fp.PACKET_TYPE_FEEDBACK, result.payload
        # Mid-chain: produced a hidden_state — ships as FORWARD.
        return fp.PACKET_TYPE_FORWARD, result.payload

    # ── Serve loop ─────────────────────────────────────────────────

    def serve(self, *, recv_timeout: float = 1.0) -> None:
        """Blocking recv→decode→send loop. Run in a daemon thread.
        ``stop()`` breaks the loop at the next timeout boundary.

        Requires an inbound link. The outbound link is mode-dependent:
        last worker ships FEEDBACK on ``feedback_link``; everyone else
        ships FORWARD on ``forward_link``.
        """
        if self.inbound_link is None:
            raise RuntimeError("serve() requires an inbound link")
        while not self._stop.is_set():
            got = self.inbound_link.recv(timeout=recv_timeout)
            if got is None:
                continue  # timeout — re-check stop flag
            header, payload = got
            out = self.handle_forward_packet(header, payload)
            if out is None:
                continue  # error already counted on the inbound link
            out_type, out_bytes = out
            self._send_onward(header, out_type, out_bytes)

    def _send_onward(
        self, in_header: fp.FabricHeader, out_type: int, out_bytes: bytes,
    ) -> None:
        """Ship the decode result to the correct downstream link. The
        outbound step_id mirrors the inbound (same autoregression
        step); layer_idx for FORWARD is this worker's layer_end (the
        boundary the activation now crosses), carried through from the
        inbound header for the prototype since the receiver re-derives
        its own layer context from the chain plan anyway."""
        if out_type == fp.PACKET_TYPE_FEEDBACK:
            if self.feedback_link is None:
                # Last worker with no feedback link wired — nothing to
                # do but drop. A real chain always wires this; the
                # guard keeps a misconfigured bringup from crashing.
                return
            self.feedback_link.send(
                out_bytes,
                packet_type=fp.PACKET_TYPE_FEEDBACK,
                step_id=in_header.step_id,
                layer_idx=fp.LAYER_IDX_FEEDBACK,
                dtype=fp.DTYPE_INT8,
            )
            return
        if self.forward_link is None:
            return
        self.forward_link.send(
            out_bytes,
            packet_type=fp.PACKET_TYPE_FORWARD,
            step_id=in_header.step_id,
            layer_idx=in_header.layer_idx,
            dtype=self.dtype,
        )

    def stop(self) -> None:
        """Signal the serve loop to exit at the next recv timeout.
        Idempotent + non-blocking."""
        self._stop.set()
