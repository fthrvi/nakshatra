#!/usr/bin/env python3
"""Nakshatra client (M5) — chain walker over a multi-worker gRPC chain.

Two ways to discover the chain:
  - --config <yaml>      static cluster YAML (M5 default)
  - --registry <url>     query Sthambha pillar for live peer registry
                         and build a chain plan dynamically (Phase 3b)

After discovery, queries Info on each worker, validates the layer
partition is contiguous, then walks the chain:
  worker[0]  : tokens IN, hidden state OUT
  worker[1..N-2] (middle): hidden IN, hidden OUT
  worker[N-1] (last): hidden IN, top-1 token id OUT

For multi-token generation: streaming KV reuse — first step is full
prompt with keep_kv=False; later steps ship 1 token with keep_kv=True
and start_pos=prefix_length.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import queue
import re
import socket
import stat
import struct
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib import request as urlrequest

sys.path.insert(0, str(Path(__file__).parent))

import yaml

from client_defaults import (
    _environment_file_value,
    resolve_p2p_socket,
    resolve_registry_keyid,
)
import paid_client
from step_text import escape_step_text   # one token = one output line (see step_text.py)
from edge_health import EdgeError, EdgeFailureKind   # #17 live seam: typed edge faults so the
# recovery loop can swap a dead worker instead of the process dying. Pure stdlib (duck-types grpc).

# grpc + the generated stubs are needed only by the chain-walking / RPC
# paths, NOT by registry discovery: build_chain_from_registry and its
# helpers (_peer_chain_score, _try_pillar_chain, _sanitize_spki) are pure
# urllib + stdlib. Optional-import them — same shape as the nakshatra_tls
# block below — so the discovery logic stays importable (and unit-testable)
# on a checkout without the gRPC runtime. The RPC paths NameError only if
# actually exercised without grpc, which can't happen in a real run (the
# deps ship together; this only matters for grpc-free unit tests).
try:
    import grpc
    import nakshatra_pb2 as pb
    import nakshatra_pb2_grpc as pb_grpc
    _GRPC_AVAILABLE = True
except ImportError:
    grpc = pb = pb_grpc = None  # type: ignore[assignment]
    _GRPC_AVAILABLE = False

# 2026-05-26: SPKI-pinned outbound channels. Lazy-imported so the client
# still runs on a checkout that pre-dates the Phase 2 TLS module (in which
# case --tls-mode falls back to insecure regardless of the flag value).
try:
    import nakshatra_tls as _wtls
    _TLS_AVAILABLE = True
except ImportError:
    _wtls = None
    _TLS_AVAILABLE = False


# End-of-generation token ids, multi-arch. Llama-3 (~128k vocab) and Qwen3 (~151k vocab) live in
# disjoint id ranges, so the union never false-stops either model:
#   Llama-3: 128001 <|end_of_text|>, 128008 <|eom_id|>, 128009 <|eot_id|>
#   Qwen3:   151643 <|endoftext|>,   151645 <|im_end|>   (ChatML end-of-turn — needed for chat)
LLAMA3_EOS_IDS = {128001, 128008, 128009, 151643, 151645}

# One worker step normally has five minutes to answer. A standby worker gets
# an additive allowance on its first inference RPC only: loading the slice is
# expected work, not a stalled inference or a reason to advance recovery.
STEP_TIMEOUT_S = 300.0
DEFAULT_WARMUP_TIMEOUT_S = 60.0
WARMUP_UNAVAILABLE_BACKOFF_S = 0.25


def _residency(value) -> str:
    """Return the pillar residency vocabulary, treating old/missing data as unknown."""
    value = str(value or "").strip().lower()
    return value if value in ("loaded", "standby", "away") else "unknown"


def _warmup_timeout_s() -> float:
    raw = os.environ.get("NAKSHATRA_WARMUP_TIMEOUT_S", str(DEFAULT_WARMUP_TIMEOUT_S))
    try:
        return max(0.0, float(raw))
    except ValueError:
        raise ValueError(
            f"NAKSHATRA_WARMUP_TIMEOUT_S must be a number, got {raw!r}"
        )


def _is_unavailable(exc: BaseException) -> bool:
    code_fn = getattr(exc, "code", None)
    if not callable(code_fn):
        return False
    try:
        code = code_fn()
    except Exception:
        return False
    return getattr(code, "name", str(code)) == "UNAVAILABLE"


class HopRpcPolicy:
    """Deadlines and the one cold-start retry for a single active chain hop.

    ``invoke`` receives the per-step timeout in seconds. Keeping the policy
    transport-neutral lets unary Forward and the persistent Inference stream
    share exactly the same first-RPC semantics.
    """

    def __init__(self, worker_id: str, residency: str, *, warmup_timeout_s=None,
                 now=time.monotonic, sleep=time.sleep, log=None):
        self.worker_id = worker_id
        self.residency = _residency(residency)
        self.warmup_timeout_s = (
            _warmup_timeout_s() if warmup_timeout_s is None
            else max(0.0, float(warmup_timeout_s))
        )
        self._now = now
        self._sleep = sleep
        self._log = log or (lambda line: print(line, flush=True))
        self._first_rpc_consumed = False

    def call(self, invoke, normal_timeout_s: float = STEP_TIMEOUT_S):
        warming = self.residency == "standby" and not self._first_rpc_consumed
        # Consume before invoking so an exhausted warm-up falls through to the
        # existing recovery path; a recovery replay to the same hop is normal.
        self._first_rpc_consumed = True
        timeout_s = normal_timeout_s + (self.warmup_timeout_s if warming else 0.0)
        started = self._now()
        try:
            try:
                return invoke(timeout_s)
            except Exception as exc:
                if not (warming and _is_unavailable(exc)):
                    raise
                self._sleep(WARMUP_UNAVAILABLE_BACKOFF_S)
                return invoke(timeout_s)
        finally:
            if warming:
                elapsed = max(0.0, self._now() - started)
                self._log(
                    f"[chain] warming {self.worker_id} (standby) … {elapsed:.2f}s"
                )


def _hidden_bytes(n, n_embd):
    """Expected wire bytes of an n-token hidden state — int8 blob when NAKSHATRA_ACT_QUANT=int8
    (#16, the client just relays the opaque blob; workers quant/dequant), else fp32."""
    if os.environ.get("NAKSHATRA_ACT_QUANT", "").strip().lower() == "int8":
        from act_quant import quant_blob_size
        return quant_blob_size(n, n_embd)
    return n * n_embd * 4


class PushFailure(RuntimeError):
    """v0.5 §9.5: a worker advertised rpc_push but its peer connection failed.
    Caught by the recovery loop to downgrade the session to streaming-only
    (client-relay) mode without swapping workers — the broken link is between
    the two peer workers, not in either worker itself."""


def _open_chain_channel(address: str, expected_spki: str | None,
                        tls_mode: str):
    """2026-05-26 — open a gRPC channel to one chain worker, mirroring the
    shape of worker.py:_open_outbound_channel so client.py can talk to
    workers that have flipped to TLS-only after the 2026-05-21 SPKI sprint.

    ``tls_mode`` is the operator policy knob:

    - ``"off"``    — force ``grpc.insecure_channel``. Escape hatch for
      bringup or for legacy pre-Phase-2 chains the operator knows are
      plaintext-only.
    - ``"auto"``   — pin against ``expected_spki`` when one is available;
      fall through to insecure when the hash is missing. Back-compat with
      the existing v0.1 70B cluster on home-pc:5530, which hasn't been
      Phase-2'd.
    - ``"required"`` — refuse anything that isn't an SPKI-pinned channel.
      Operator-strict mode for Prithvi-style production use.

    PinError is surfaced via ``sys.exit`` because the chain is unusable
    if any single worker fails the handshake — there's no per-worker
    recovery story at preflight time (alternate-cursor recovery starts
    later, after _setup_chain returns).
    """
    if tls_mode == "off" or not _TLS_AVAILABLE:
        # Raise the gRPC message cap from the 4MiB default: a prefill hidden-state is
        # n_prompt_tokens × hidden × 4 bytes (e.g. 400 tok × 5120 × 4 = 8.4MB), which blows the
        # default → "RESOURCE_EXHAUSTED" empty gen on any real prompt. 256MB covers a full 2048-ctx.
        return grpc.insecure_channel(address, options=[
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ("grpc.max_send_message_length", 256 * 1024 * 1024)])
    try:
        return _wtls.open_pinned_channel(
            address, expected_spki,
            refuse_unpinned=(tls_mode == "required"),
            # Read this at the gateway-request boundary instead of relying on
            # nakshatra_tls's import-time default. The gateway passes its full
            # EnvironmentFile to this subprocess, so an operator override is
            # now explicit in the call that owns the actual SPKI probe.
            probe_timeout_s=_wtls._probe_timeout_from_env(),
        )
    except _wtls.PinError as e:
        sys.exit(f"[chain] TLS pin failure for {address}: {e}")


class RemoteTokenizer:
    """Drop-in replacement for the pieces of a vocab-only `Llama` object this file actually
    uses (`.tokenize()`, `.detokenize()`) — backed by a resident tokenizer_daemon.py over a Unix
    socket instead of loading the model in THIS process. 2026-09-12: measured ~0.3-0.5s per
    request for `Llama(vocab_only=True)` to load, paid on every chat turn because
    nakshatra_serve.py spawns a fresh client.py subprocess per request. A resident daemon this
    connects to pays that cost once, at daemon startup, not per turn — verified round-trip
    latency ~0.6ms per call once connected. Every OTHER caller of `llama.tokenize`/
    `llama.detokenize` in this file (detok_one, the final full/gen detokenize) is unchanged;
    they just see an object with the same two methods.

    One connection per client.py invocation (opened once, reused for every tokenize/detokenize
    call during that run — a chat turn does many small detokenize calls, one per generated
    token, so re-connecting per call would defeat the purpose)."""

    def __init__(self, socket_path: str, timeout: float = 10.0):
        import socket as _socket
        self._sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect(socket_path)
        self._buf = b""

    def _call(self, req: dict) -> dict:
        import json as _json
        self._sock.sendall((_json.dumps(req) + "\n").encode("utf-8"))
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("tokenizer daemon closed the connection")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        resp = _json.loads(line.decode("utf-8"))
        if "error" in resp:
            raise RuntimeError(f"tokenizer daemon: {resp['error']}")
        return resp

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = True):
        resp = self._call({"op": "tokenize", "text": text.decode("utf-8"),
                          "add_bos": add_bos, "special": special})
        return resp["ids"]

    def detokenize(self, ids) -> bytes:
        resp = self._call({"op": "detokenize", "ids": list(ids)})
        return resp["text"].encode("utf-8")

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass


def tokenize_local(model_path, prompt, tokenizer_socket: str = ""):
    """tokenizer_socket, when given, uses a resident tokenizer_daemon.py instead of loading
    the model in this process — see RemoteTokenizer's docstring for why. Any connection failure
    (daemon not running, stale socket, etc.) falls back to the normal local load rather than
    failing the whole request — the daemon is a latency optimization, never a hard dependency."""
    if tokenizer_socket:
        try:
            llama = RemoteTokenizer(tokenizer_socket)
            return llama.tokenize(prompt.encode("utf-8"), add_bos=True, special=True), llama
        except Exception as e:
            print(f"[chain] tokenizer daemon unavailable ({e!r}) — loading locally instead",
                  file=sys.stderr)
    from llama_cpp import Llama
    llama = Llama(model_path=model_path, vocab_only=True, verbose=False)
    return llama.tokenize(prompt.encode("utf-8"), add_bos=True, special=True), llama


def detok_one(llama, tid):
    try:
        return llama.detokenize([tid]).decode("utf-8", errors="replace")
    except Exception:
        return "?"


# The exact method strings worker.py signs/verifies (NOT the protobuf service name nakshatra.v0_1.Nakshatra).
_GRPC_AUTH_METHODS = {"Forward": "/nakshatra.Nakshatra/Forward", "Inference": "/nakshatra.Nakshatra/Inference",
                      "SignParticipation": "/nakshatra.Nakshatra/SignParticipation"}


def _grpc_auth_metadata(method: str, message_bytes: bytes, streaming: bool = False):
    """`authorization` gRPC metadata for a call to an AUTHENTICATED worker, or None.

    A worker that registered with a Sthambha pillar defaults to auth_required (Mode B/C) and rejects any call without a
    signed `Sthambha-Ed25519` header: this driver had no way to send one, so it could only drive unauthenticated
    (static-YAML, Mode A) workers. When NKS_REGISTRY_KEYID names a registered peer whose key this box holds, the call is
    signed over the serialized request (Forward) or the FIRST frame (Inference, streaming), exactly as the worker verifies.
    When the variable is absent, the release registrar id or this node's public-key id is used."""
    keyid, _keyid_source = resolve_registry_keyid()
    if not keyid:
        return None
    import nakshatra_auth as _na
    import nakshatra_grpc_auth as _ga
    key_path = Path.home() / ".nakshatra" / "keys" / "worker.ed25519"
    priv, _pub = _na.load_or_create_worker_key(key_path)
    header = _ga.build_grpc_auth_header(priv, keyid, _GRPC_AUTH_METHODS[method], message_bytes, is_streaming=streaming)
    return [("authorization", header)]


def call_forward(stub, payload, n_tokens, has_token_ids, worker_id="<unknown>",
                 keep_kv=False, start_pos=0, timing=None, all_logits=False,
                 eagle_hidden=False, rpc_policy=None):
    req = pb.ForwardRequest(
        hidden_in=payload, batch=1, n_tokens=n_tokens, has_token_ids=has_token_ids,
        keep_kv=keep_kv, start_pos=start_pos, all_logits=all_logits,
        eagle_hidden=eagle_hidden,
    )
    t0 = time.time()
    try:
        metadata = _grpc_auth_metadata("Forward", req.SerializeToString())
        invoke = lambda timeout: stub.Forward(req, timeout=timeout, metadata=metadata)
        resp = (rpc_policy or HopRpcPolicy(worker_id, "unknown")).call(invoke)
    except grpc.RpcError as e:
        # RuntimeError subclass: the existing outer recovery loop can swap an
        # alternate. The warm-up policy has already spent its one local retry.
        raise EdgeError.from_exc(e, peer_id=worker_id, phase="forward") from e
    if timing is not None:
        timing.setdefault(worker_id, []).append(time.time() - t0)
    return resp.hidden_out


class InferenceStream:
    """One bidi Inference stream to one worker, exposed as a sync send→recv.

    v0.5 M0.5.1: the worker keeps KV state for the lifetime of this stream,
    so the client doesn't pass keep_kv — the worker treats the first
    InferenceStep on the stream as a cold-prefill and every subsequent step
    as a decode that reuses the existing KV cache.

    The request side is a generator that pulls from a thread-safe queue;
    grpc reads from that generator on its own thread, calling us back with
    responses one at a time. Putting them through a single queue makes the
    stream look like a synchronous send-then-recv to chain-walk code.
    """

    def __init__(self, stub, worker_id, rpc_policy=None):
        self.worker_id = worker_id
        self._closed = False
        self._stub = stub
        self._rpc_policy = rpc_policy or HopRpcPolicy(worker_id, "unknown")
        self._req_q = None
        self._responses = None
        self._resp_q = None

    @staticmethod
    def _request_gen(req_q):
        while True:
            step = req_q.get()
            if step is None:
                return
            yield step

    @staticmethod
    def _pump_responses(responses, resp_q):
        try:
            for response in responses:
                resp_q.put((True, response))
        except Exception as exc:
            resp_q.put((False, exc))
        else:
            resp_q.put((False, RuntimeError("Inference stream ended without a response")))

    def _discard_stream(self):
        if self._req_q is not None:
            self._req_q.put(None)
        cancel = getattr(self._responses, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass
        self._req_q = self._responses = self._resp_q = None

    def _start_stream(self, request_step):
        req_q = queue.Queue()
        resp_q = queue.Queue()
        req_q.put(request_step)
        responses = self._stub.Inference(
            self._request_gen(req_q),
            metadata=_grpc_auth_metadata(
                "Inference", request_step.SerializeToString(), streaming=True),
        )
        self._req_q, self._responses, self._resp_q = req_q, responses, resp_q
        threading.Thread(
            target=self._pump_responses, args=(responses, resp_q), daemon=True,
        ).start()

    def _wait_response(self, timeout_s):
        try:
            ok, value = self._resp_q.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError(
                f"no response from {self.worker_id} in {timeout_s:.1f}s"
            )
        if ok:
            return value
        raise value

    def step(self, request_step):
        if self._closed:
            raise RuntimeError(f"stream to {self.worker_id} already closed")
        try:
            if self._responses is None:
                def first_attempt(timeout_s):
                    self._discard_stream()
                    self._start_stream(request_step)
                    return self._wait_response(timeout_s)
                return self._rpc_policy.call(first_attempt)
            self._req_q.put(request_step)
            return self._wait_response(STEP_TIMEOUT_S)
        except (grpc.RpcError, TimeoutError) as e:
            # #17 live seam: classify HERE, where the real grpc .code() is live (the
            # plain-RuntimeError rewrap downstream would lose it). EdgeError is a
            # RuntimeError subclass, so the recovery loop catches it and swaps to a
            # healthy alternate instead of the process dying on a sys.exit.
            raise EdgeError.from_exc(e, peer_id=self.worker_id, phase="forward") from e

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._discard_stream()


def call_inference_step(streamer, payload, n_tokens, has_token_ids,
                         session_id, step_idx, prefix_length, timing=None,
                         next_server=None, chain=None, all_logits=False,
                         spend_claim_token=""):
    """v0.5 M0.5.1 streaming equivalent of call_forward. Returns raw bytes
    matching Forward's return shape (hidden state OR int32 token id), so
    chain-walk code consumes the result identically.

    Push-related kwargs (mutually exclusive):
      - next_server (v0.5 M0.5.3 v1): single immediate-next worker. The
        receiving worker forwards there. Sufficient for 2-worker chains.
      - chain (v0.5 M0.5.3 v2): list of NextServer for the WHOLE remaining
        chain after the receiving worker. Each downstream worker pops the
        head and forwards the rest. Works for chains of any length.

    all_logits (2026-07-29 stream-spec): the streaming twin of call_forward's
    all_logits — request one argmax per input position instead of only the
    last. Default False -> identical to the plain streaming path today. When
    the receiving worker is mode=last with all_logits=True, `resp.token_ids.ids`
    carries n_tokens ids and the returned bytes are n_tokens*4 (K+1 verify
    argmaxes), which the generic packing below already handles unchanged.
    """
    step = pb.InferenceStep(
        session_id=session_id,
        step_id=f"step-{step_idx}",
        prefix_length=prefix_length,
        all_logits=all_logits,
        spend_claim_token=spend_claim_token,
    )
    if has_token_ids:
        ids = list(struct.unpack(f"<{n_tokens}i", payload))
        step.token_ids.ids.extend(ids)
    else:
        step.hidden_state.raw = payload
        step.hidden_state.batch = 1
        step.hidden_state.n_tokens = n_tokens
    if next_server is not None:
        step.next_server.CopyFrom(next_server)
    if chain:
        step.chain.extend(chain)

    t0 = time.time()
    # #17 live seam: a transport fault surfaces as EdgeError (from StreamHandle.step's grpc
    # handler) or a plain RuntimeError ("stream already closed"). Both are RuntimeError, so we
    # let them PROPAGATE to the recovery loop (which swaps/retries) instead of sys.exit-killing
    # the process here.
    resp = streamer.step(step)
    if timing is not None:
        timing.setdefault(streamer.worker_id, []).append(time.time() - t0)

    if resp.HasField("error"):
        err_str = resp.error.decode("utf-8", "replace")
        # v0.5 §9.5: structured push-failure signal — propagates to the
        # recovery loop, which downgrades the session to relay mode.
        if err_str.startswith("push_failed:"):
            raise PushFailure(f"worker {streamer.worker_id!r}: {err_str}")
        # #17 live seam: peer-side error → INTERNAL (not a transport fault, but a swap to a
        # healthy alternate may still complete the request — strictly better than dying here).
        raise EdgeError(peer_id=streamer.worker_id, kind=EdgeFailureKind.INTERNAL,
                        reason=f"worker reported error: {err_str}", phase="forward")
    if resp.HasField("token_ids"):
        ids = list(resp.token_ids.ids)
        return struct.pack(f"<{len(ids)}i", *ids)
    if resp.HasField("hidden_state"):
        return resp.hidden_state.raw
    # #17 live seam: PROTOCOL kind. v1's recovery loop still swaps (bounded by
    # --max-recovery-attempts then surfaces) — better than instant death — and the typed kind
    # lets the v2 is_transport_fault gate skip the pointless churn in one line.
    raise EdgeError(peer_id=streamer.worker_id, kind=EdgeFailureKind.PROTOCOL,
                    reason="returned an empty step", phase="forward")


# ── Speculative decode ON THE STREAM (2026-07-29) ──────────────────────────────
# docs/findings/cuda-chain-51-tok-s.md: unary --speculative pays per-call RPC setup
# on every one of K+1 verify positions and loses to plain streaming even with a GPU
# draft (12.20 vs 51-53 tok/s). This carries the SAME verify traversal (accept()/
# kv_keep_after() from speculative.py, unchanged) over the persistent per-worker
# Inference streams instead, using InferenceStep.all_logits + a prefix_length
# rewind on the next step (worker.py already trims KV to start_pos==prefix_length
# before decoding — the same primitive the unary M3 fusion uses, now driven
# through the stream).

def stream_spec_disable_reasons(use_streaming: bool, use_streaming_push: bool,
                                draft_ready: bool, worker_caps) -> list[str]:
    """Pure gate — no gRPC, no draft model. `worker_caps` is an iterable of
    (worker_id, capabilities_list) pairs, one per chain member (mirrors the
    `sorted_stubs` shape client.py already has: `[(w, stub, info), ...]`, so
    call sites pass `[(w["id"], info.protocol_capabilities) for w, _, info in
    sorted_stubs]`). Returns an empty list when stream-spec may engage;
    otherwise a list of human-readable reasons it must fall back to plain
    streaming decode. Protobuf silently ignores fields it doesn't recognize,
    so an old worker missing 'stream_spec' would return ONE token where K+1
    were expected — this refusal is what makes that failure mode impossible
    instead of a silent wrong answer."""
    reasons = []
    if not use_streaming or use_streaming_push:
        reasons.append("--stream-spec requires --use-streaming (not --use-streaming-push)")
    if not draft_ready:
        reasons.append("no --draft-model-path")
    non_ss = [wid for wid, caps in worker_caps if "stream_spec" not in caps]
    if non_ss:
        reasons.append(f"workers lack 'stream_spec' capability: {non_ss}")
    return reasons


def stream_spec_verify_fn(streamers, sorted_stubs, n_embd, session_id, step_idx,
                          prefix_length, timing=None):
    """Build a verify_fn(token_ids) -> List[int] closure matching
    speculative.speculative_round's contract, driven over the persistent
    per-worker Inference streams instead of unary Forward. `prefix_length` is
    held FIXED for the whole verify round (the workers' KV already covers
    [0, prefix_length) plus whatever the LAST round's rejected tail wrote —
    the daemon call at this prefix_length re-decodes from there); the caller
    is responsible for advancing prefix_length via kv_keep_after() after
    accept() decides how many of this round's positions to keep — THAT
    reassignment is the KV rewind, and it happens once, up in the decode
    loop, not per worker call here."""
    def verify_fn(verify_tokens):
        n_v = len(verify_tokens)
        payload = struct.pack(f"<{n_v}i", *verify_tokens)
        if len(sorted_stubs) == 1:
            # Solo chain: the one worker holds every layer, so its first response IS the
            # verify-round argmaxes — no hidden-state relay. See the plain-decode loop's
            # same branch for the full explanation.
            solo_resp = call_inference_step(streamers[0], payload, n_v, True,
                                            session_id=session_id, step_idx=step_idx,
                                            prefix_length=prefix_length, timing=timing,
                                            all_logits=True)
            if len(solo_resp) != n_v * 4:
                raise RuntimeError(f"stream-spec: solo worker returned {len(solo_resp)} bytes, "
                                   f"expected {n_v * 4} verify argmaxes")
            return list(struct.unpack(f"<{n_v}i", solo_resp))
        hidden = call_inference_step(streamers[0], payload, n_v, True,
                                     session_id=session_id, step_idx=step_idx,
                                     prefix_length=prefix_length, timing=timing,
                                     all_logits=True)
        if len(hidden) != _hidden_bytes(n_v, n_embd):
            raise RuntimeError(f"stream-spec: first worker returned {len(hidden)} bytes")
        for idx, stub_tup in enumerate(sorted_stubs[1:-1], start=1):
            hidden = call_inference_step(streamers[idx], hidden, n_v, False,
                                         session_id=session_id, step_idx=step_idx,
                                         prefix_length=prefix_length, timing=timing,
                                         all_logits=True)
            if len(hidden) != _hidden_bytes(n_v, n_embd):
                raise RuntimeError(f"stream-spec: middle worker returned {len(hidden)} bytes")
        last_i = len(sorted_stubs) - 1
        last_resp = call_inference_step(streamers[last_i], hidden, n_v, False,
                                        session_id=session_id, step_idx=step_idx,
                                        prefix_length=prefix_length, timing=timing,
                                        all_logits=True)
        if len(last_resp) != n_v * 4:
            raise RuntimeError(f"stream-spec: last worker returned {len(last_resp)} bytes, "
                               f"expected {n_v * 4} verify argmaxes")
        return list(struct.unpack(f"<{n_v}i", last_resp))
    return verify_fn


def _peer_chain_score(peer: dict) -> int:
    """Phase 5: lower is better. Prefers peers whose GPUs are not flagged
    as 'drifty' in the registry (Vega-class etc.). Ties broken by ordering
    GPU > CPU since GPU is faster when correct.

    Returns:
        0  for verified-GPU peers with chain_status='ok'
        1  for CPU peers (no drift risk, but slow)
        2  for GPU peers with chain_status='drifty' (use only as last resort)
        3  for peers we can't classify (unknown hardware)
    """
    hw = peer.get("hardware") or {}
    gpus = hw.get("gpus") or []
    if not gpus:
        return 1  # plain CPU peer
    g = gpus[0]
    backend = (g.get("backend") or "cpu").lower()
    chain_status = (g.get("chain_status") or "ok").lower()
    actual = int(g.get("actual_layers_offloaded", 0))
    if backend != "cpu" and actual > 0:
        if chain_status == "drifty":
            return 2  # GPU but known to drift — last resort
        return 0      # verified GPU, ok
    return 1          # declared GPU but didn't actually offload → CPU


def _sanitize_spki(raw) -> str:
    """Normalize a ``peer_spki_hash`` field from arbitrary input.
    Returns 64-lowercase-hex or ``""``. Same shape PillarPeerKeyResolver
    enforces on the worker side."""
    spki = str(raw or "").strip().lower()
    if len(spki) != 64:
        return ""
    try:
        bytes.fromhex(spki)
    except ValueError:
        return ""
    return spki


ACTIVATION_ENDPOINT_KIND = "libp2p-activation-v1"
_CLIENT_DIAL_ADDRESS = "_client_dial_address"
_PUSH_DIAL_ADDRESS = "_push_dial_address"
# Keep this paired with the sidecar's Go defaultDirectWait (15 seconds).
# FORWARD does not reply until that direct-path wait completes.
SIDECAR_DIRECT_WAIT_S = 15.0
SIDECAR_FORWARD_CONTROL_TIMEOUT_S = SIDECAR_DIRECT_WAIT_S + 5.0


def _own_node_and_sidecar_ids(*, home: Path | None = None) -> tuple[str, str]:
    """Return this uid's node public key and deterministically-derived PeerId.

    Identity discovery is read-only and deliberately strict: a gateway must
    never create or repair a node key merely because it is resolving a chain.
    An absent or unusable key means that no hop is classified as local.
    """
    base = Path.home() if home is None else Path(home)
    try:
        import nakshatra_auth
        from sidecar_key import peer_id_from_node_pub
        _private, node_public_key = nakshatra_auth.load_worker_key(
            base / ".nakshatra" / "keys" / "worker.ed25519"
        )
        return node_public_key, peer_id_from_node_pub(node_public_key)
    except (OSError, TypeError, ValueError):
        return "", ""


def _self_hop_loopback(spec: dict, endpoint: dict, *, home: Path | None = None
                       ) -> tuple[str, int] | None:
    """Resolve this node's typed activation hop without asking its sidecar.

    ``nak worker on`` is the authority for both the bind family and port. We
    therefore require valid ``LISTEN`` and ``PORT`` assignments in the worker
    opt-in instead of falling back to worker.py defaults. Wildcard binds map
    to loopback in the same address family; non-loopback-specific binds are
    refused because silently dialing them would violate the local-only path.
    """
    base = Path.home() if home is None else Path(home)
    node_public_key, sidecar_peer_id = _own_node_and_sidecar_ids(home=base)
    if not node_public_key:
        return None
    hop_node_id = str(spec.get("node_id") or spec.get("id") or "").strip()
    is_self = (
        hop_node_id.lower() == node_public_key.lower()
        or endpoint["peer_id"] == sidecar_peer_id
    )
    if not is_self:
        return None

    env_path = base / ".nakshatra" / "worker.env"
    port_text = _environment_file_value(env_path, "PORT")
    listen = _environment_file_value(env_path, "LISTEN")
    if not port_text.isascii() or not port_text.isdigit():
        raise RuntimeError(
            f"self activation hop requires a valid PORT in {env_path}"
        )
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise RuntimeError(
            f"self activation hop has an out-of-range PORT in {env_path}"
        )
    if not listen:
        raise RuntimeError(
            f"self activation hop requires an explicit LISTEN in {env_path}"
        )

    candidate = (listen[1:-1]
                 if listen.startswith("[") and listen.endswith("]") else listen)
    if candidate.lower() == "localhost":
        host = "127.0.0.1"
    else:
        try:
            configured = ipaddress.ip_address(candidate)
        except ValueError as e:
            raise RuntimeError(
                f"self activation hop has an invalid LISTEN in {env_path}"
            ) from e
        if configured.is_unspecified:
            host = "::1" if configured.version == 6 else "127.0.0.1"
        elif configured.is_loopback:
            host = configured.compressed
        else:
            raise RuntimeError(
                f"self activation hop LISTEN in {env_path} does not include loopback"
            )
    return host, port


def _parse_activation_endpoint(raw) -> dict | None:
    """Validate the additive registry endpoint used for activation streams.

    Peers without an endpoint retain the legacy literal-address path.  An
    endpoint that is present is never silently treated as a literal address:
    unknown kinds and malformed peer ids fail closed.
    """
    if raw in (None, ""):
        return None
    if not isinstance(raw, dict):
        raise RuntimeError("registry endpoint must be an object")
    kind = str(raw.get("kind") or "").strip()
    if kind != ACTIVATION_ENDPOINT_KIND:
        raise RuntimeError(f"unsupported registry endpoint kind {kind!r}")
    peer_id = str(raw.get("peer_id") or "").strip()
    if (not peer_id or not peer_id.isascii() or len(peer_id) > 160
            or any(ch.isspace() or ord(ch) < 33 for ch in peer_id)):
        raise RuntimeError("libp2p activation endpoint has an invalid peer_id")
    addrs = raw.get("addrs", [])
    if addrs is None:
        addrs = []
    if (not isinstance(addrs, list) or len(addrs) > 8
            or any(not isinstance(a, str) or not a or len(a) > 512
                   for a in addrs)):
        raise RuntimeError("libp2p activation endpoint has invalid addrs")
    endpoint = {"kind": kind, "peer_id": peer_id}
    if addrs:
        endpoint["addrs"] = list(addrs)
    return endpoint


def _sidecar_forward(peer_id: str, socket_path: str | None = None,
                     timeout: float = SIDECAR_FORWARD_CONTROL_TIMEOUT_S,
                     *, require_uid_owned: bool = False) -> tuple[str, int]:
    """Ask the uid-protected sidecar for a cached activation forward.

    The sidecar is a byte pipe only.  TLS and the registered worker SPKI are
    still checked by :func:`_open_chain_channel` after this returns.  The
    control timeout exceeds the sidecar's 15-second direct wait.  If a caller
    overrides it with a shorter value and times out after listener creation,
    the cached forward has no accepted client, remains idle, and is LRU-
    evictable.
    """
    endpoint = _parse_activation_endpoint(
        {"kind": ACTIVATION_ENDPOINT_KIND, "peer_id": peer_id})
    socket_path = socket_path.strip() if socket_path is not None else resolve_p2p_socket()
    if not socket_path:
        raise RuntimeError(
            "libp2p activation endpoint requires NAKSHATRA_P2P_SOCKET")
    if not os.path.isabs(socket_path):
        raise RuntimeError("NAKSHATRA_P2P_SOCKET must be an absolute path")
    control = None
    path_handle = None
    try:
        socket_info = None
        if require_uid_owned:
            try:
                socket_info = os.lstat(socket_path)
                if (not stat.S_ISSOCK(socket_info.st_mode)
                        or socket_info.st_uid != os.getuid()):
                    raise RuntimeError(
                        "paid libp2p activation requires a uid-owned Unix sidecar socket"
                    )
                path_handle = os.open(
                    socket_path,
                    os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC,
                )
                pinned_info = os.fstat(path_handle)
            except RuntimeError:
                raise
            except OSError as exc:
                raise RuntimeError(
                    "paid libp2p activation requires a live uid-owned "
                    f"sidecar socket: {exc}"
                ) from exc
            if ((pinned_info.st_dev, pinned_info.st_ino)
                    != (socket_info.st_dev, socket_info.st_ino)
                    or not stat.S_ISSOCK(pinned_info.st_mode)
                    or pinned_info.st_uid != os.getuid()):
                raise RuntimeError(
                    "paid libp2p activation sidecar socket changed before connect"
                )

        control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        control.settimeout(timeout)
        control.connect(socket_path)
        if require_uid_owned:
            try:
                credentials = control.getsockopt(
                    socket.SOL_SOCKET, socket.SO_PEERCRED,
                    struct.calcsize("3i"),
                )
                _pid, peer_uid, _gid = struct.unpack("3i", credentials)
                pinned_after = os.fstat(path_handle)
                current_info = os.lstat(socket_path)
            except (OSError, struct.error) as exc:
                raise RuntimeError(
                    f"paid libp2p activation could not authenticate sidecar: {exc}"
                ) from exc
            if peer_uid != os.getuid():
                raise RuntimeError(
                    "paid libp2p activation requires a same-uid sidecar listener"
                )
            pinned_identity = (pinned_after.st_dev, pinned_after.st_ino)
            if (pinned_identity != (socket_info.st_dev, socket_info.st_ino)
                    or pinned_identity
                    != (current_info.st_dev, current_info.st_ino)
                    or not stat.S_ISSOCK(current_info.st_mode)
                    or current_info.st_uid != os.getuid()):
                raise RuntimeError(
                    "paid libp2p activation sidecar socket changed during connect"
                )
        control.sendall(f"FORWARD {endpoint['peer_id']}\n".encode("ascii"))
        response = b""
        while b"\n" not in response and len(response) <= 4096:
            chunk = control.recv(4096)
            if not chunk:
                break
            response += chunk
    except OSError as e:
        raise RuntimeError(f"sidecar FORWARD failed: {e}") from e
    finally:
        if control is not None:
            control.close()
        if path_handle is not None:
            os.close(path_handle)
    if len(response) > 4096 or b"\n" not in response:
        raise RuntimeError("sidecar FORWARD returned an unterminated response")
    try:
        line = response.split(b"\n", 1)[0].decode("utf-8")
    except UnicodeDecodeError as e:
        raise RuntimeError("sidecar FORWARD returned non-UTF-8 data") from e
    if line.startswith("ERR "):
        raise RuntimeError(f"sidecar FORWARD refused {peer_id}: {line[4:]}")
    if not line.startswith("OK "):
        raise RuntimeError(f"sidecar FORWARD returned malformed response: {line!r}")
    host, sep, port_text = line[3:].rpartition(":")
    if sep != ":" or host != "127.0.0.1" or not port_text.isdigit():
        raise RuntimeError("sidecar FORWARD did not return a loopback TCP address")
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise RuntimeError("sidecar FORWARD returned an invalid TCP port")
    return host, port


def _resolve_activation_endpoint(spec: dict, *, socket_path: str | None = None,
                                 tls_mode: str = "auto",
                                 home: Path | None = None,
                                 require_uid_sidecar: bool = False) -> dict:
    """Resolve every supported hop to its authoritative dial addresses.

    This is the single address-selection boundary for client.py.  Legacy hops
    retain their configured literal address (and optional co-located push
    address).  Typed activation hops *only* expose the loopback listener
    returned by sidecar FORWARD; their legacy address is never copied into a
    dial field. The one exception is this node's own typed hop: it uses the
    worker opt-in's explicit port over loopback, avoiding a nonsensical
    sidecar self-dial while retaining the projected worker SPKI below.

    A FORWARD listener belongs to the hub's network namespace, so it cannot be
    sent to a remote worker as a server-to-server push target.  Such hops have
    no ``_push_dial_address`` and the capability gate below downgrades the
    session to client-relayed streaming.  Unary and ordinary streaming still
    use the returned listener with end-to-end worker authentication: TLS is
    pinned to the SPKI projected by the Pillar for that worker.
    """
    endpoint = _parse_activation_endpoint(spec.get("endpoint"))
    resolved = dict(spec)
    if endpoint is None:
        host = str(spec.get("address") or "").strip()
        try:
            port = int(spec.get("port"))
        except (TypeError, ValueError) as e:
            raise RuntimeError("legacy activation hop has an invalid port") from e
        if not host or not 1 <= port <= 65535:
            raise RuntimeError("legacy activation hop has an invalid literal address")
        push_host = str(spec.get("internal_address") or host).strip()
        if not push_host:
            raise RuntimeError("legacy activation hop has an invalid push address")
        resolved["address"] = host
        resolved["port"] = port
        resolved[_CLIENT_DIAL_ADDRESS] = f"{host}:{port}"
        resolved[_PUSH_DIAL_ADDRESS] = f"{push_host}:{port}"
        return resolved
    if tls_mode == "off":
        raise RuntimeError(
            "libp2p activation endpoints require TLS + worker SPKI pinning")
    if not _TLS_AVAILABLE:
        raise RuntimeError(
            "libp2p activation endpoint cannot be used without nakshatra_tls")
    if not _sanitize_spki(spec.get("peer_spki_hash")):
        raise RuntimeError(
            "libp2p activation endpoint has no valid registered worker SPKI")
    local = _self_hop_loopback(spec, endpoint, home=home)
    host, port = local or _sidecar_forward(
        endpoint["peer_id"], socket_path=socket_path,
        require_uid_owned=require_uid_sidecar,
    )
    resolved["address"] = host
    resolved["port"] = port
    dial_host = f"[{host}]" if ":" in host else host
    resolved[_CLIENT_DIAL_ADDRESS] = f"{dial_host}:{port}"
    resolved[_PUSH_DIAL_ADDRESS] = None
    return resolved


def _push_dial_address(resolved_hop: dict) -> str:
    """Return a resolver-produced server-to-server target or fail closed."""
    address = resolved_hop.get(_PUSH_DIAL_ADDRESS)
    if address:
        return str(address)
    endpoint = _parse_activation_endpoint(resolved_hop.get("endpoint"))
    if endpoint is not None:
        raise RuntimeError(
            "typed activation endpoint cannot be used as a downstream push "
            "target; use client-relayed streaming")
    raise RuntimeError("push hop was not passed through the activation resolver")


def _typed_push_downstream_workers(sorted_stubs: list) -> list[str]:
    """Return typed hops that a push-mode worker, rather than the hub, would dial."""
    return [w.get("id") or w.get("node_id") or "<unknown>"
            for w, _, _ in sorted_stubs[1:]
            if _parse_activation_endpoint(w.get("endpoint")) is not None]


def _apply_streaming_push_gate(use_streaming_push: bool, sorted_stubs: list,
                               *, context: str = "chain") -> bool:
    """Return whether push remains safe for this fully resolved chain.

    This gate belongs to chain setup rather than one-time session setup: an
    alternate selected during recovery can have a different endpoint kind or
    capability set from the primary it replaces.  Calling this after every
    rebuild makes a legacy-to-typed downstream replacement take the same safe
    client-relay downgrade as a typed downstream present at startup.
    """
    if not use_streaming_push:
        return False

    non_pushers = [w["id"] for w, _, info in sorted_stubs
                   if "rpc_push" not in info.protocol_capabilities]
    if non_pushers:
        print(
            f"[{context}] WARNING: push requested but workers lack rpc_push "
            f"capability: {non_pushers} — downgrading session to streaming-only "
            "(client relays per-step).",
            file=sys.stderr,
        )
        return False

    typed_downstream = _typed_push_downstream_workers(sorted_stubs)
    if typed_downstream:
        print(
            f"[{context}] WARNING: push requested with typed activation "
            f"downstream hops {typed_downstream}; hub-side FORWARD listeners "
            "cannot be dialed from remote workers — downgrading session to "
            "streaming-only (client relays per-step).",
            file=sys.stderr,
        )
        return False
    return True


def _fetch_registry_peers(registry_url: str, model_id: str = "") -> list[dict]:
    suffix = f"?model={model_id}" if model_id else ""
    url = f"{registry_url.rstrip('/')}/peers{suffix}"
    with _registry_urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())
    peers = (data or {}).get("peers", [])
    return [p for p in peers if isinstance(p, dict)]


def _require_pinned_pillar_projection(registry_url: str) -> str:
    """Return the configured Pillar SPKI or refuse paid projection data.

    ``_registry_urlopen`` supplies request authentication and passes this pin
    to the HTTPS opener. Paid planning must never import identity, endpoint,
    or worker-pin facts from an unpinned response.
    """
    from urllib.parse import urlsplit

    if urlsplit(registry_url).scheme.lower() != "https":
        raise PermissionError(
            "paid chain projection requires an HTTPS Pillar URL"
        )
    pin = _sanitize_spki(os.environ.get("STHAMBHA_PILLAR_SPKI_SHA256"))
    if not pin:
        raise PermissionError(
            "paid chain projection requires a valid pinned Pillar SPKI"
        )
    return pin


def _paid_worker_key(record: dict) -> str:
    """Return one canonical Ed25519 worker key, or ``""`` when absent."""
    value = str(
        record.get("public_key_hex") or record.get("worker_keyid") or ""
    ).strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else ""


def _paid_chain_identity(record: dict, label: str) -> tuple[str, str, dict | None]:
    """Validate one paid hop's key-derived node and activation identities."""
    key = _paid_worker_key({"public_key_hex": record.get("public_key_hex")})
    if not key:
        raise PermissionError(
            f"paid SPKI enrichment refused for {label!r}: "
            "chain worker has no full Ed25519 public key"
        )

    node_id = str(record.get("node_id") or "").strip().lower()
    if node_id and (re.fullmatch(r"[0-9a-f]{12,64}", node_id) is None
                    or not key.startswith(node_id)):
        raise PermissionError(
            f"paid SPKI enrichment refused for {label!r}: node identity "
            "is not a prefix of the worker public key"
        )

    try:
        endpoint = _parse_activation_endpoint(record.get("endpoint"))
    except RuntimeError as exc:
        raise PermissionError(
            f"paid SPKI enrichment refused for {label!r}: invalid activation endpoint"
        ) from exc
    if endpoint is not None:
        from sidecar_key import peer_id_from_node_pub
        if endpoint["peer_id"] != peer_id_from_node_pub(key):
            raise PermissionError(
                f"paid SPKI enrichment refused for {label!r}: activation "
                "PeerId does not match the worker public key"
            )

    record["public_key_hex"] = key
    return key, node_id, endpoint


def _merge_paid_pillar_projection(workers: list[dict], peers: list[dict], *,
                                  enrich_reachability: bool = False) -> None:
    """Strictly bind paid workers to the pinned Pillar ``/peers`` view.

    The caller owns transport authentication and Pillar SPKI pinning. This
    function owns the cross-projection join. The signed ``/chain`` hop's full
    worker key is primary and must bind its node-id prefix and activation
    PeerId. A matching ``/peers`` row is an additional pinned cross-check;
    its absence does not erase a complete chain hop. Only reachability and TLS
    SPKI are imported; roster trust tier and placement remain untouched.
    There is deliberately no address-keyed or TOFU path.
    """
    by_key: dict[str, list[dict]] = {}
    by_node: dict[str, list[dict]] = {}
    for peer in peers:
        node_id = str(peer.get("node_id") or peer.get("id") or "").strip().lower()
        if node_id:
            by_node.setdefault(node_id, []).append(peer)
        key = _paid_worker_key(peer)
        if key:
            by_key.setdefault(key, []).append(peer)

    for worker in workers:
        label = str(worker.get("node_id") or worker.get("id") or "<unknown>")
        key, worker_node, worker_endpoint = _paid_chain_identity(worker, label)
        matches = by_key.get(key, [])
        node_matches = by_node.get(worker_node, []) if worker_node else []
        if len(matches) > 1 or len(node_matches) > 1:
            raise PermissionError(
                f"paid SPKI enrichment refused for {label!r}: worker key is "
                "ambiguous in the pinned Pillar /peers projection"
            )
        if node_matches and _paid_worker_key(node_matches[0]) != key:
            raise PermissionError(
                f"paid SPKI enrichment refused for {label!r}: worker public "
                "key disagrees between chain and /peers"
            )
        peer = matches[0] if matches else (node_matches[0] if node_matches else None)

        worker_raw_spki = str(worker.get("peer_spki_hash") or "").strip()
        worker_spki = _sanitize_spki(worker_raw_spki)
        if worker_raw_spki and not worker_spki:
            raise PermissionError(
                f"paid SPKI enrichment refused for {label!r}: chain has an "
                "invalid worker SPKI"
            )

        if peer is None:
            if not worker_spki:
                raise PermissionError(
                    f"paid SPKI enrichment refused for {label!r}: neither chain "
                    "nor /peers has a valid worker SPKI; worker must re-register"
                )
            worker["peer_spki_hash"] = worker_spki
            continue

        peer_label = str(peer.get("node_id") or peer.get("id") or label)
        peer_key, peer_node, peer_endpoint = _paid_chain_identity(peer, peer_label)
        if peer_key != key:
            raise PermissionError(
                f"paid SPKI enrichment refused for {label!r}: worker public "
                "key disagrees between chain and /peers"
            )
        if not peer_node:
            raise PermissionError(
                f"paid SPKI enrichment refused for {label!r}: pinned Pillar "
                "peer has no node identity"
            )
        if worker_node and worker_node != peer_node:
            raise PermissionError(
                f"paid SPKI enrichment refused for {label!r}: node identity "
                "disagrees between chain and /peers"
            )

        if worker_endpoint is not None:
            if (peer_endpoint is None
                    or worker_endpoint["peer_id"] != peer_endpoint["peer_id"]):
                raise PermissionError(
                    f"paid SPKI enrichment refused for {label!r}: activation "
                    "PeerId disagrees between chain and /peers"
                )

        peer_raw_spki = str(peer.get("peer_spki_hash") or "").strip()
        peer_spki = _sanitize_spki(peer_raw_spki)
        if peer_raw_spki and not peer_spki:
            raise PermissionError(
                f"paid SPKI enrichment refused for {label!r}: /peers has an "
                "invalid worker SPKI"
            )
        if worker_spki and peer_spki and worker_spki != peer_spki:
            raise PermissionError(
                f"paid SPKI enrichment refused for {label!r}: worker SPKI "
                "conflicts between chain and /peers"
            )
        merged_spki = worker_spki or peer_spki
        if not merged_spki:
            raise PermissionError(
                f"paid SPKI enrichment refused for {label!r}: neither chain "
                "nor /peers has a valid worker SPKI; worker must re-register"
            )
        worker["peer_spki_hash"] = merged_spki

        if enrich_reachability:
            worker["node_id"] = peer_node
            if peer_endpoint is not None:
                worker["endpoint"] = peer_endpoint
            address = str(peer.get("address") or "").strip()
            valid_address = False
            if address:
                host, separator, port_text = address.rpartition(":")
                if (separator and host and port_text.isdigit()
                        and 1 <= int(port_text) <= 65535):
                    worker["address"] = host
                    worker["port"] = int(port_text)
                    valid_address = True
                elif peer_endpoint is None:
                    raise PermissionError(
                        f"paid SPKI enrichment refused for {label!r}: /peers "
                        "has invalid legacy reachability"
                    )
            if peer_endpoint is None and not valid_address:
                raise PermissionError(
                    f"paid SPKI enrichment refused for {label!r}: /peers has "
                    "no usable worker reachability"
                )
            if "residency" in peer:
                worker["residency"] = _residency(peer.get("residency"))


def _admit_activation_workers(model_id: str, workers: list[dict],
                              peers: list[dict], *, roster_loader=None,
                              eligible_fn=None) -> None:
    """Apply the operator roster firewall to every typed activation hop.

    The Pillar supplies identity and reachability facts.  Trust tier comes
    exclusively from the operator-curated roster; a peer-projected ``tier``
    field is deliberately ignored.
    """
    typed = [w for w in workers if w.get("endpoint") is not None]
    if not typed:
        return
    # An outsider consumer does not possess Biswa's private local roster. Its authority is the
    # signed, roster-gated pillar projection reached through an SPKI-pinned public front. This
    # narrowly-scoped mode is written only by `nak gateway on`; without BOTH the flag and a valid
    # TLS pin, the historical local-roster firewall below remains mandatory.
    trust_pillar = os.environ.get("NKS_TRUST_PINNED_PILLAR", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    pillar_pin = os.environ.get("STHAMBHA_PILLAR_SPKI_SHA256", "").strip()
    if trust_pillar and re.fullmatch(r"[0-9a-fA-F]{64}", pillar_pin):
        return
    from fabric import worker_join as wj

    by_node = {str(p.get("node_id") or p.get("id") or ""): p for p in peers}

    if roster_loader is None:
        admission = wj._load_admission()
        roster_loader = admission.load_peers
    roster = roster_loader() or {}
    standings = []
    for hop in typed:
        endpoint = _parse_activation_endpoint(hop.get("endpoint"))
        node_id = str(hop.get("node_id") or hop.get("id") or "")
        peer = by_node.get(node_id)
        if peer is None:
            raise PermissionError(
                f"identity firewall: activation peer {node_id!r} is absent from /peers")
        projected = _parse_activation_endpoint(peer.get("endpoint"))
        if projected is None or projected["peer_id"] != endpoint["peer_id"]:
            raise PermissionError(
                f"identity firewall: endpoint mismatch for activation peer {node_id!r}")
        pubkey = str(peer.get("public_key_hex") or "").strip().lower()
        if not pubkey:
            raise PermissionError(
                f"identity firewall: activation peer {node_id!r} has no node public key")
        registered_spki = _sanitize_spki(peer.get("peer_spki_hash"))
        inline_spki = _sanitize_spki(hop.get("peer_spki_hash"))
        if not registered_spki:
            raise PermissionError(
                f"identity firewall: activation peer {node_id!r} has no registered worker SPKI")
        if inline_spki and inline_spki != registered_spki:
            raise PermissionError(
                f"identity firewall: SPKI mismatch for activation peer {node_id!r}")
        hop["peer_spki_hash"] = registered_spki
        roster_peer = roster.get(pubkey)
        worker_role = (roster_peer or {}).get("role", "worker")
        standing = wj.WorkerStanding(
            node_id=node_id, pubkey=pubkey,
            admitted=roster_peer is not None and worker_role != "consumer",
            tier=(roster_peer or {}).get("tier", "denied"),
            tenant=(roster_peer or {}).get("tenant"), capabilities={})
        standings.append(standing)

    if eligible_fn is None:
        eligible, rejected = wj.eligible_workers(model_id, standings)
    else:
        eligible, rejected = eligible_fn(model_id, standings)
    eligible_identities = {(w.node_id, w.pubkey) for w in eligible}
    refused = [w for w in standings
               if (w.node_id, w.pubkey) not in eligible_identities]
    if refused:
        reasons = {r.get("worker"): r.get("reason") for r in rejected}
        detail = ", ".join(
            f"{w.node_id}: {reasons.get(w.node_id, 'not eligible')}" for w in refused)
        raise PermissionError(
            f"identity firewall: activation chain is not eligible for {model_id!r}: {detail}")


def _fetch_spki_index(registry_url: str) -> dict[str, str]:
    """2026-05-26 SPKI — query the pillar's /peers projection once and
    build an identity-first ``node_id/PeerId → peer_spki_hash`` index.

    The pillar's /chain endpoint doesn't include peer_spki_hash in its
    chain entries (that's a sthambha-side extension we haven't shipped
    yet), so we get the hashes the same way :class:`PillarPeerKeyResolver`
    does on the worker side — a separate /peers fetch. Literal address keys
    remain as a legacy fallback, but typed activation endpoints use stable
    node/peer identities because their loopback port is dynamic.

    Returns ``{}`` on failure or missing fields rather than raising —
    --tls-mode=auto then transparently degrades to plaintext for any
    worker whose hash we couldn't recover, while --tls-mode=required
    will surface a clear PinError("unpinned_peer") at _setup_chain time.
    """
    url = f"{registry_url.rstrip('/')}/peers"
    try:
        with _registry_urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"[chain] WARNING: SPKI index fetch failed ({e}); chain "
              f"will run unpinned in --tls-mode=auto.", file=sys.stderr)
        return {}
    index: dict[str, str] = {}
    for peer in (data or {}).get("peers", []):
        if not isinstance(peer, dict):
            continue
        address = str(peer.get("address") or "").strip()
        node_id = str(peer.get("node_id") or "").strip()
        try:
            endpoint = _parse_activation_endpoint(peer.get("endpoint"))
        except RuntimeError:
            endpoint = None
        spki = _sanitize_spki(peer.get("peer_spki_hash"))
        if spki:
            if node_id:
                index[node_id] = spki
            if endpoint is not None:
                index[endpoint["peer_id"]] = spki
            if address:  # legacy pillars/chains remain address-keyed
                index[address] = spki
    return index


def info_with_startup_wait(stub, request, timeout: float = 10.0, *, wait_s: "float | None" = None,
                           sleep=None, now=None):
    """stub.Info(), retrying while the worker says it is still STARTING.

    A worker binds its gRPC port before it has registered with the pillar and filled its key cache, and answers Info
    UNAVAILABLE "worker starting: <why>" until it can authenticate callers. That is a transient of a cold start (the chain
    lifecycle normally waits for it, but a driver can race a worker that was summoned a moment ago), so retry for up to
    NKS_INFO_STARTUP_WAIT_S (default 30 s). Anything else - refused connection, auth failure, version mismatch - raises at once."""
    import time as _time
    import grpc
    sleep = sleep or _time.sleep
    now = now or _time.monotonic
    if wait_s is None:
        wait_s = float(os.environ.get("NKS_INFO_STARTUP_WAIT_S", "30"))
    deadline = now() + wait_s
    while True:
        try:
            return stub.Info(request, timeout=timeout)
        except grpc.RpcError as e:
            code = e.code() if callable(getattr(e, "code", None)) else None
            details = str(e.details() if callable(getattr(e, "details", None)) else "")
            if code != grpc.StatusCode.UNAVAILABLE or "starting" not in details or now() >= deadline:
                raise
            print(f"[chain] worker still starting ({details}); retrying Info", flush=True)
            sleep(1.0)


def _registry_urlopen(url: str, timeout: float = 10):
    """urlopen for a Sthambha registry GET. Since the pillar's auth hardening /chain and /peers need a signed
    `Authorization: Sthambha-Ed25519` header (an unsigned GET is a 401), so registry mode was silently broken.
    When NKS_REGISTRY_KEYID names a registered peer whose key this box holds (nakshatra_auth's worker key), the request
    is signed - over the FULL path INCLUDING the query string, which is what the pillar verifies. When unset, the
    release registrar id or the release worker's public-key id is selected automatically.

    Sthambha's envelope has only an integer-second timestamp, not a nonce. Two
    same-key GETs for the same path in one second would therefore be identical
    and the pillar correctly rejects the second as a replay. A random query
    nonce makes every signed canonical path unique; the pillar's /peers and
    /chain handlers ignore unknown query keys while authenticating the complete
    path, so this needs no protocol or server change.
    """
    import nakshatra_auth as _na
    import secrets
    pin = os.environ.get("STHAMBHA_PILLAR_SPKI_SHA256", "").strip() or None
    keyid, keyid_source = resolve_registry_keyid()
    try:
        if not keyid:
            return _na.open_pillar_url(
                url, timeout=timeout, expected_spki_hash=pin,
            )
        from urllib.parse import urlsplit, urlunsplit
        parts = urlsplit(url)
        nonce_query = f"nks_nonce={secrets.token_hex(16)}"
        parts = parts._replace(
            query=f"{parts.query}&{nonce_query}" if parts.query else nonce_query)
        url = urlunsplit(parts)
        signed_path = parts.path + (f"?{parts.query}" if parts.query else "")
        key_path = Path.home() / ".nakshatra" / "keys" / "worker.ed25519"
        priv, _pub = _na.load_or_create_worker_key(key_path)
        header, _ts = _na.build_signed_envelope(priv, keyid, "GET", signed_path, b"")
        return _na.open_pillar_url(
            urlrequest.Request(url, headers={"Authorization": header}),
            timeout=timeout, expected_spki_hash=pin,
        )
    except urlrequest.HTTPError as e:
        if e.code == 401:
            attempted = repr(keyid) if keyid else "<unsigned>"
            raise RuntimeError(
                "the Sthambha pillar refused the registry request (401); "
                f"tried keyid {attempted} from {keyid_source}. "
                "Set NKS_REGISTRY_KEYID to a node id registered with the "
                "pillar whose key this box holds, or use --config."
            ) from e
        raise


def _try_pillar_chain(registry_url: str, model_id: str) -> list | None:
    """Phase I: ask the pillar to assemble the chain itself.

    Pillars from Phase I onward expose GET /chain?model=<id>. Returns
    None if the pillar is older (404) or returns an empty chain (so
    callers fall back to the local builder, which can produce a
    clearer error message). Other failures propagate."""
    url = f"{registry_url.rstrip('/')}/chain?model={model_id}"
    try:
        with _registry_urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
    except urlrequest.HTTPError as e:
        if e.code == 404:
            return None
        raise
    chain = data.get("chain") or []
    if not chain:
        return None
    for warn in data.get("warnings") or []:
        print(f"[chain] WARNING (from pillar): {warn}", file=sys.stderr)
    workers = []
    for c in chain:
        endpoint = _parse_activation_endpoint(c.get("endpoint"))
        host, _, port = (c.get("address") or "").rpartition(":")
        if endpoint is None and not port:
            raise RuntimeError(
                f"pillar returned malformed chain entry for "
                f"{c.get('node_id')!r}: address={c.get('address')!r}"
            )
        workers.append({
            "id": (c.get("node_id") or "")[:14],
            "node_id": str(c.get("node_id") or ""),
            "address": host,
            "port": int(port) if port else 0,
            "layer_start": int(c["layer_start"]),
            "layer_end": int(c["layer_end"]),
            "endpoint": endpoint,
            "public_key_hex": str(c.get("public_key_hex") or ""),
            "residency": _residency(c.get("residency")),
            # 2026-05-26 SPKI Phase 4 — pillar's /chain projection ships
            # peer_spki_hash inline (sthambha commit added the field
            # alongside this one). Older pillars omit it; that's still
            # safe — _sanitize_spki returns "" for missing/malformed,
            # and main()'s registry-side merge then backfills from a
            # /peers fetch as a fallback.
            "peer_spki_hash": _sanitize_spki(c.get("peer_spki_hash")),
        })
    return workers


def build_chain_from_registry(registry_url: str, model_id: str, *,
                              roster_loader=None, eligible_fn=None,
                              strict_pinned_projection: bool = False) -> list:
    """Query Sthambha registry, return a contiguous chain plan for `model_id`.

    Phase I: prefer the pillar's own /chain endpoint; if absent, fall
    back to the local algorithm (verified-GPU > CPU > drifty-GPU,
    rpc_ms tiebreak inside a tier).

    Returns list of dicts in cluster-YAML format (id, address, port,
    layer_start, layer_end) so downstream code stays identical between
    YAML mode and registry mode.

    Raises RuntimeError if no contiguous chain can be built.
    """
    if strict_pinned_projection:
        _require_pinned_pillar_projection(registry_url)
    pillar_chain = _try_pillar_chain(registry_url, model_id)
    if pillar_chain is not None:
        peers = []
        if (strict_pinned_projection
                or any(w.get("endpoint") is not None for w in pillar_chain)):
            peers = _fetch_registry_peers(registry_url, model_id)
        if strict_pinned_projection:
            _merge_paid_pillar_projection(
                pillar_chain, peers, enrich_reachability=True)
        if any(w.get("endpoint") is not None for w in pillar_chain):
            _admit_activation_workers(
                model_id, pillar_chain, peers, roster_loader=roster_loader,
                eligible_fn=eligible_fn)
        print(f"[chain] using pillar-served chain plan ({len(pillar_chain)} workers)")
        return pillar_chain

    peers = _fetch_registry_peers(registry_url, model_id)

    # Collect (peer, offering) pairs from online compute peers
    pairs = []
    away = 0
    for p in peers:
        if not p.get("is_online"):
            continue
        # An owner-reclaimed worker refuses all inference; the pillar's /chain already skips it,
        # and this fallback must too (it used to dial it and fail with "UNAVAILABLE: owner paused").
        if p.get("residency") == "away":
            away += any(o.get("model_id") == model_id for o in p.get("layer_offerings", []))
            continue
        for o in p.get("layer_offerings", []):
            if o.get("model_id") == model_id:
                pairs.append((p, o))

    if not pairs:
        why = f" ({away} away: their owners are using them)" if away else ""
        raise RuntimeError(f"no online peers advertise model_id={model_id!r} on {registry_url}{why}")

    # Sort: layer_start ascending, then chain-quality score ascending,
    # then recent_rpc_ms ascending (lower latency wins WITHIN a quality
    # tier — Phase H tiebreaker; never overrides drift status). Peers
    # with no rpc data yet (recent_rpc_ms == 0) get a high stand-in so
    # we prefer peers with KNOWN good latency.
    def _rpc_key(p):
        v = p.get("recent_rpc_ms") or 0.0
        return v if v > 0 else 9e9
    pairs.sort(key=lambda po: (po[1]["layer_start"],
                                _peer_chain_score(po[0]),
                                _rpc_key(po[0])))

    chain = []
    cursor = 0
    used_node_ids = set()
    saw_drifty = False
    for peer, offering in pairs:
        if offering["layer_start"] == cursor and peer["node_id"] not in used_node_ids:
            endpoint = _parse_activation_endpoint(peer.get("endpoint"))
            host, _, port = str(peer.get("address") or "").rpartition(":")
            if endpoint is None and not port:
                raise RuntimeError(f"peer {peer['node_id']!r} has malformed address {peer['address']!r} (expected host:port)")
            score = _peer_chain_score(peer)
            if score == 2:
                saw_drifty = True
                gpus = (peer.get("hardware") or {}).get("gpus") or [{}]
                print(f"[chain] WARNING: including drifty-flagged peer {peer['node_id']} "
                      f"({gpus[0].get('model','?')}) in chain — "
                      f"output may be incoherent. No alternative for layers "
                      f"[{offering['layer_start']},{offering['layer_end']}).",
                      file=sys.stderr)
            chain.append({
                "id": peer["node_id"][:14],
                "node_id": str(peer["node_id"]),
                "address": host,
                "port": int(port) if port else 0,
                "layer_start": offering["layer_start"],
                "layer_end": offering["layer_end"],
                "endpoint": endpoint,
                "public_key_hex": str(peer.get("public_key_hex") or ""),
                "residency": _residency(peer.get("residency")),
                # 2026-05-26 SPKI: take it straight from the /peers
                # payload we already loaded. defensive parse mirrors
                # _fetch_spki_index for the pillar-served chain path.
                "peer_spki_hash": _sanitize_spki(peer.get("peer_spki_hash")),
                # v1.1 hardening: carry the peer's drift class (P1 §8.1 Nostr
                # listing / gauge fingerprint) so drift-class-constrained
                # recovery engages on registry-built chains too. None for an
                # unclassified legacy peer → legacy any-alternate behaviour.
                "drift_class": peer.get("drift_class"),
            })
            used_node_ids.add(peer["node_id"])
            cursor = offering["layer_end"]

    if not chain:
        raise RuntimeError(f"no peer offers layer 0 of model {model_id!r}; cannot start chain")

    _admit_activation_workers(
        model_id, chain, peers, roster_loader=roster_loader,
        eligible_fn=eligible_fn)
    if strict_pinned_projection:
        _merge_paid_pillar_projection(
            chain, peers, enrich_reachability=True)

    return chain


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=str, help="cluster YAML (static)")
    src.add_argument("--registry", type=str,
                     help="Sthambha pillar URL (e.g. http://node-pi:7777). Builds "
                          "chain plan from live registry.")
    ap.add_argument("--model-id", type=str, default="",
                    help="Model id to query in the registry (required if --registry)")
    ap.add_argument("--model-path", type=str, required=True, help="full GGUF path for tokenizer (must match the model that was split)")
    ap.add_argument("--tokenizer-socket", type=str, default="",
                    help="2026-09-12: Unix socket of a resident tokenizer_daemon.py serving "
                         "--model-path's vocab — skips the ~0.3-0.5s per-process Llama(vocab_"
                         "only=True) reload. Falls back to a local load if the daemon isn't "
                         "reachable; never a hard requirement.")
    ap.add_argument("--prompt", type=str, default="The capital of France is")
    ap.add_argument("--max-tokens", "-n", type=int, default=1)
    ap.add_argument("--use-streaming", action="store_true",
                    help="v0.5 M0.5.1: use the streaming Inference RPC instead "
                         "of per-step Forward calls. One bidi stream per worker "
                         "for the entire session; KV state lives in the stream. "
                         "Output must be identical to the non-streaming path.")
    ap.add_argument("--use-streaming-push", action="store_true",
                    help="v0.5 M0.5.3: server-to-server push mode. Implies "
                         "--use-streaming. Client opens ONE stream to the first "
                         "worker; each step carries next_server pointing at the "
                         "next worker. Workers chain via peer connections; the "
                         "response unwinds back. v1 supports 2-worker chains; "
                         "longer chains need v2 (multi-hop next_server).")
    ap.add_argument("--simulate-fail-step", type=int, default=-1,
                    help="v0.5 M0.5.4 v0: inject a synthetic failure AFTER this "
                         "step (1-indexed). Triggers the recovery path: close "
                         "streams, re-open, replay history (prompt + tokens "
                         "generated so far), continue. -1 disables. Only honored "
                         "in streaming mode.")
    ap.add_argument("--max-recovery-attempts", type=int, default=3,
                    help="v0.5 M0.5.4 v0: how many times the client retries via "
                         "recovery before surfacing failure to the caller.")
    ap.add_argument("--tls-mode", choices=("off", "auto", "required"),
                    default="auto",
                    help="2026-05-26 SPKI: outbound TLS policy. 'auto' "
                         "(default): pin against peer_spki_hash when one is "
                         "available, else fall through to plaintext (back-"
                         "compat with pre-Phase-2 workers + the existing "
                         "v0.1 70B cluster). 'required': refuse to talk to "
                         "any worker without a declared SPKI hash. 'off': "
                         "force plaintext channels (escape hatch).")
    ap.add_argument("--receipt-out", type=str, default="",
                    help="speed-stack #20: write a verifiable run receipt (JSON) to this "
                         "path after the run — distinct workers + per-stage timing + output "
                         "token sha256 + layer-map; verify with scripts/receipt.verify_receipt")
    ap.add_argument("--participation-keyring", type=str, default="",
                    help="JSON {node_id: priv_hex} of nodes THIS operator owns. Each listed "
                         "stage is signed for, upgrading the receipt's participation from "
                         "coordinator-asserted to holder-of-key proven. Single-operator only: "
                         "a stage whose node_id is absent is left unsigned, never faked — a "
                         "coordinator that could mint a worker's signature would defeat the "
                         "purpose of having signatures. Signing for someone else's node needs "
                         "the worker to sign and return it (a proto change), because "
                         "output_sha256 covers the whole run and is unknown until it ends.")
    # Speculative decoding (slice 1). Default OFF; also enabled by env
    # NAKSHATRA_SPECULATIVE=1. Engages only in unary mode when a draft model is
    # given AND every worker advertises the "speculative" capability; any failure
    # falls back to plain decode via the existing recovery path.
    ap.add_argument("--speculative", action="store_true",
                    help="enable speculative decoding (needs --draft-model-path; "
                         "unary Forward mode only for slice 1)")
    ap.add_argument("--draft-model-path", type=str, default="",
                    help="full GGUF of a small same-tokenizer draft model "
                         "(e.g. Llama-3.2-1B); runs locally on the coordinator")
    ap.add_argument("--draft-max", type=int, default=4,
                    help="K: number of tokens the draft proposes per step")
    # Speculative decode ON THE STREAM (2026-07-29, docs/findings/cuda-chain-51-tok-s.md):
    # the unary --speculative path pays per-call RPC setup on every one of K+1 verify
    # positions and loses even to plain streaming with a GPU draft. This reuses the same
    # accept()/DraftModel machinery but rides the persistent Inference stream (all_logits
    # on InferenceStep + a prefix_length rewind) instead of unary Forward. Default OFF;
    # also enabled by env NKS_STREAM_SPEC=1. Needs --draft-model-path (or --eagle-head is
    # NOT yet wired for this path) and --use-streaming (not push); refuses and falls back
    # to plain streaming unless EVERY worker advertises the "stream_spec" capability.
    ap.add_argument("--stream-spec", action="store_true",
                    default=os.environ.get("NKS_STREAM_SPEC") == "1",
                    help="speculative decode over the persistent streaming Inference RPC "
                         "instead of unary Forward (needs --use-streaming + "
                         "--draft-model-path; env NKS_STREAM_SPEC=1)")
    # Async pipelining (Shard's 2.94→16.6 tok/s technique): keep several spec verify-chunks
    # in flight so every worker stage stays busy instead of one-at-a-time (the pipeline
    # bubble). Correctness stays byte-identical to greedy (commit-only-confirmed invariant in
    # async_pipeline.py). Default OFF — the proven sequential path is untouched. Applies only
    # to the explicit-stage streaming path with spec active (not --use-streaming-push).
    ap.add_argument("--async-pipeline", action="store_true",
                    default=os.environ.get("NKS_ASYNC_PIPELINE") == "1",
                    help="pipeline speculative decode: keep N verify-chunks in flight "
                         "(env NKS_ASYNC_PIPELINE=1)")
    ap.add_argument("--async-inflight", type=int,
                    default=int(os.environ.get("NKS_ASYNC_INFLIGHT", "3")),
                    help="max speculative verify-chunks in flight (pipeline depth target)")
    # EAGLE→live (2026-06-21): swap the GGUF draft for the matched EAGLE-3 head,
    # fed by the FIRST worker's cmd=5 hidden states (isolated on scratch seq 1).
    # Requires every worker to advertise the "eagle_hidden" capability.
    ap.add_argument("--eagle-head", type=str, default="",
                    help="EAGLE-3 draft head checkpoint (.pt). When set, replaces "
                         "--draft-model-path; the draft runs on the first worker's "
                         "cmd=5 hidden states (KV-safe scratch sequence).")
    ap.add_argument("--eagle-base", type=str, default="",
                    help="EAGLE base model dir (HF config source) for the head")
    ap.add_argument("--eagle-config", type=str, default="",
                    help="EAGLE head config json (draft_vocab etc.)")
    ap.add_argument("--paid-admission-fd", type=int, default=-1,
                    help=argparse.SUPPRESS)
    ap.add_argument("--paid-progress-fd", type=int, default=-1,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    paid = None
    if args.paid_admission_fd >= 0:
        try:
            paid = paid_client.read_paid_envelope(args.paid_admission_fd)
        except paid_client.PaidAdmissionError as exc:
            sys.exit(f"[paid] admission envelope refused: {exc}")
        refusal = paid_client.paid_mode_refusal(
            push=args.use_streaming_push,
            speculative=args.speculative,
            stream_spec=args.stream_spec,
            async_pipeline=args.async_pipeline,
            eagle=bool(args.eagle_head),
        )
        if refusal:
            sys.exit(f"[paid] paid sessions refuse {refusal} execution")
        if not args.use_streaming:
            sys.exit("[paid] paid sessions require direct streaming calls")
        if args.paid_progress_fd < 0:
            sys.exit("[paid] paid sessions require an invocation progress descriptor")
    if os.environ.get("NAKSHATRA_SPECULATIVE", "") in ("1", "true", "True"):
        args.speculative = True
        if paid is not None:
            sys.exit("[paid] paid sessions refuse non-greedy execution")
    # --use-streaming-push implies --use-streaming
    if args.use_streaming_push:
        args.use_streaming = True
    # --tls-mode=required is only meaningful if the TLS module is importable.
    # Fail loud rather than silently downgrading to insecure.
    if paid is not None and not _TLS_AVAILABLE:
        sys.exit("[paid] pinned TLS support is unavailable")
    if args.tls_mode == "required" and not _TLS_AVAILABLE:
        sys.exit("--tls-mode=required but nakshatra_tls module is not "
                 "importable; cannot pin channels. Install the Phase 2 TLS "
                 "module or drop the flag.")

    if args.registry:
        if not args.model_id:
            sys.exit("--registry requires --model-id")
        print(f"[chain] querying Sthambha registry: {args.registry} (model={args.model_id})")
        workers = build_chain_from_registry(args.registry, args.model_id)
        print(f"[chain] registry returned a chain of {len(workers)} workers covering layers [0,{workers[-1]['layer_end']})")
        # 2026-05-26 SPKI Phase 4 — happy path: a Phase-4-or-newer
        # pillar ships peer_spki_hash inline on every /chain entry,
        # so we skip the second /peers fetch entirely. The backfill
        # below only runs when some entry is missing a hash, which
        # means we're talking to an older pillar OR a Phase-2 worker
        # that doesn't run TLS yet. Avoids an HTTP roundtrip on every
        # chain build in the common case.
        if any(not (w.get("peer_spki_hash") or "") for w in workers):
            spki_index = _fetch_spki_index(args.registry)
            for w in workers:
                addr = f"{w['address']}:{w['port']}"
                endpoint = w.get("endpoint") or {}
                from_index = (
                    spki_index.get(w.get("node_id") or w.get("id") or "", "")
                    or spki_index.get(endpoint.get("peer_id") or "", "")
                    or spki_index.get(addr, "")
                )
                # Don't clobber a hash the chain entry already
                # populated with one that disagrees — surface it so
                # the operator sees something is wrong. Equal values
                # are fine; missing-in-index is fine (leave whatever's
                # there).
                existing = w.get("peer_spki_hash") or ""
                if existing and from_index and existing != from_index:
                    print(f"[chain] WARNING: SPKI for {addr} disagrees: "
                          f"chain={existing[:8]}… /peers={from_index[:8]}…; "
                          f"using chain value.", file=sys.stderr)
                elif from_index and not existing:
                    w["peer_spki_hash"] = from_index
    else:
        cfg = yaml.safe_load(Path(args.config).read_text())
        workers = cfg["workers"]
        print(f"[chain] {len(workers)} workers in config")
        # Normalize whatever the YAML carried into the same shape the
        # registry path emits — defensive parse rejects malformed hex.
        config_candidates = []
        for w in workers:
            w["peer_spki_hash"] = _sanitize_spki(w.get("peer_spki_hash"))
            w["endpoint"] = _parse_activation_endpoint(w.get("endpoint"))
            w["residency"] = _residency(w.get("residency"))
            config_candidates.append(w)
            for alt in list(w.get("alternates") or []):
                alt["peer_spki_hash"] = _sanitize_spki(
                    alt.get("peer_spki_hash"))
                alt["endpoint"] = _parse_activation_endpoint(
                    alt.get("endpoint"))
                alt["residency"] = _residency(alt.get("residency"))
                config_candidates.append(alt)
        if any(w.get("endpoint") is not None for w in config_candidates):
            configured_model = str(
                args.model_id or (cfg.get("model") or {}).get("id") or "")
            if not configured_model:
                raise PermissionError(
                    "identity firewall: typed activation config needs a model id")
            for candidate in config_candidates:
                if candidate.get("endpoint") is not None:
                    _admit_activation_workers(
                        configured_model, [candidate], [candidate])

    if paid is not None and any(worker.get("alternates") for worker in workers):
        # A paid authorization freezes one exact worker identity/span plan.
        # Typed PRIMARY hops are safe through the uid-protected sidecar because
        # its loopback FORWARD remains a byte pipe: the channel still pins and
        # attests the remote worker. Alternate selection would instead mutate
        # that authorized topology and is never allowed.
        sys.exit("[paid] paid sessions refuse alternate failover execution")

    # v0.5 M0.5.4 v1: alternate-worker recovery. Each worker has a list of
    # candidates: primary first, then optional alternates from yaml. A
    # per-worker cursor selects the active candidate. On failure, the
    # recovery handler advances cursors and rebuilds the chain.
    for w in workers:
        # v1.1 hardening: carry `drift_class` onto the primary candidate so the
        # drift-class-constrained recovery (recovery/drift_aware.py) actually
        # engages — next_compatible_cursor reads candidates[0]["drift_class"] as
        # the class an alternate must match. Omitting it here silently degraded
        # the constraint to the legacy "any alternate" path (primary class None
        # → allow anything). Alternates already pass through verbatim below.
        primary = {k: w[k] for k in ("id", "node_id", "address", "port",
                                       "layer_range", "mode", "peer_spki_hash",
                                       "drift_class", "endpoint", "public_key_hex",
                                       "internal_address", "residency")
                   if k in w}
        primary["sub_gguf_path"] = w.get("sub_gguf_path", "")
        # Normalize per-candidate hashes too: alternates may omit the
        # field, in which case --tls-mode=required will refuse the
        # downgrade rather than silently switching to plaintext.
        for alt in list(w.get("alternates") or []):
            alt["peer_spki_hash"] = _sanitize_spki(alt.get("peer_spki_hash"))
            alt["endpoint"] = _parse_activation_endpoint(alt.get("endpoint"))
            alt["residency"] = _residency(alt.get("residency"))
        w["candidates"] = [primary] + list(w.get("alternates") or [])
        w["cursor"] = 0

    def _spec(w):
        return w["candidates"][w["cursor"]]

    rpc_policies = {}

    def _setup_chain(context="chain"):
        """Open channels + call Info on each worker's current candidate.
        Returns (sorted_stubs, n_embd). Validates chain partition and reruns
        every transport gate for both initial setup and recovery rebuilds."""
        stubs_local = []
        n_embd_local = None
        for w in workers:
            spec = _spec(w)
            resolved_spec = _resolve_activation_endpoint(
                spec, tls_mode=args.tls_mode,
                require_uid_sidecar=(paid is not None))
            addr = resolved_spec[_CLIENT_DIAL_ADDRESS]
            if paid is not None and not resolved_spec.get("peer_spki_hash"):
                sys.exit(
                    f"[paid] pre-compute admission refused: {w['id']} has no valid TLS SPKI pin")
            # 2026-05-26 SPKI: build a TLS-pinned channel when we have a
            # hash, falling back per --tls-mode policy. Same channel is
            # reused by every later Forward / Inference RPC, so fixing
            # this single call site lights up the whole chain.
            ch = _open_chain_channel(
                addr, resolved_spec.get("peer_spki_hash") or None,
                ("required" if paid is not None or spec.get("endpoint") is not None
                 else args.tls_mode),
            )
            stub = pb_grpc.NakshatraStub(ch)
            identity_challenge = (
                paid_client.make_info_challenge() if paid is not None else b"")
            info = info_with_startup_wait(
                stub, pb.InfoRequest(identity_challenge=identity_challenge),
                timeout=10.0)
            attested_worker_keyid = None
            if paid is not None:
                try:
                    attested_worker_keyid = paid_client.verify_info_attestation(
                        info, identity_challenge)
                except paid_client.PaidAdmissionError as exc:
                    sys.exit(f"[paid] pre-compute admission refused: {exc}")
            caps = list(info.protocol_capabilities)
            # v1.0 §7 — negotiate the control-protocol version on the same
            # handshake. Refuse to build a chain with a worker we can't speak
            # to (clean reject, never a silent mixed-version chain).
            try:
                from wire.handshake import negotiate_handshake
                negotiate_handshake(info.protocol_version, caps)
            except ImportError:
                pass  # wire module unavailable — fall back to legacy (no check)
            except Exception as e:
                sys.exit(f"[chain] {w['id']} @ {addr}: control-version negotiation "
                         f"failed: {e}")
            cap_tag = f"  caps={caps}" if caps else "  caps=<v0.1>"
            print(f"  {w['id']:12s} {addr:25s}  layers=[{info.layer_start},{info.layer_end})  "
                  f"embd={info.has_token_embd}  lm={info.has_lm_head}  hidden={info.hidden_size}"
                  + cap_tag
                  + (f"  [alt cursor={w['cursor']}/{len(w['candidates'])-1}]" if w['cursor'] > 0 else ""))
            if n_embd_local is None:
                n_embd_local = info.hidden_size
            elif info.hidden_size != n_embd_local:
                sys.exit(f"hidden_size mismatch across workers: {n_embd_local} vs {info.hidden_size}")
            # All later paths receive the active, resolver-produced candidate.
            # In particular, push construction must not rediscover an address
            # from the outer worker record (which still contains the primary's
            # legacy address and may no longer even be the active alternate).
            active_w = dict(w)
            active_w.update(resolved_spec)
            if attested_worker_keyid is not None:
                active_w["_paid_attested_worker_keyid"] = attested_worker_keyid
            policy_key = (
                active_w.get("node_id") or active_w.get("id"),
                active_w.get(_CLIENT_DIAL_ADDRESS),
            )
            active_w["_rpc_policy"] = rpc_policies.setdefault(
                policy_key,
                HopRpcPolicy(
                    active_w.get("node_id") or active_w.get("id") or "<unknown>",
                    active_w.get("residency"),
                ),
            )
            stubs_local.append((active_w, stub, info))
        sorted_local = sorted(stubs_local, key=lambda x: x[2].layer_start)
        prev_end = sorted_local[0][2].layer_start
        for w, _, info in sorted_local:
            if info.layer_start != prev_end:
                sys.exit(f"chain GAP between previous and {w['id']}")
            prev_end = info.layer_end
        if not sorted_local[0][2].has_token_embd:
            sys.exit("first worker must have token_embd")
        if not sorted_local[-1][2].has_lm_head:
            sys.exit("last worker must have lm_head")
        args.use_streaming_push = _apply_streaming_push_gate(
            args.use_streaming_push, sorted_local, context=context)
        return sorted_local, n_embd_local

    def _advance_one_alternate():
        """Find the first worker with a remaining alternate candidate; advance
        its cursor. Returns the worker dict that was advanced, or None if no
        alternates remain anywhere.

        v1.1 hardening — DRIFT-CLASS-CONSTRAINED: only fail over onto a
        same-drift-class (= same engine build) alternate; a different-build peer
        holding the same layers would silently diverge the generation
        (docs/cross-machine-validation.md §2a). Drift class travels in the
        candidate's `drift_class` (Nostr listing, P1 §8.1). Falls back to the
        legacy any-alternate behaviour if the recovery module is unavailable or
        no candidate declares a class."""
        try:
            from recovery.drift_aware import first_advanceable_worker
            return first_advanceable_worker(workers)
        except ImportError:
            for w in workers:
                if w["cursor"] + 1 < len(w["candidates"]):
                    w["cursor"] += 1
                    return w
            return None

    sorted_stubs, n_embd = _setup_chain()
    print(f"[chain] OK: contiguous coverage of [{sorted_stubs[0][2].layer_start}, {sorted_stubs[-1][2].layer_end})")

    # Paid authorization freezes both topology and the complete enforcement
    # capability set.  Info calls are read-only; no activation/KV/daemon work
    # happens until both comparisons pass.
    if paid is not None:
        try:
            paid_client.require_paid_capabilities(sorted_stubs)
            paid_client.require_exact_runtime_plan(paid["plan"], sorted_stubs)
            if args.model_id and args.model_id != paid["model_id"]:
                raise paid_client.PaidAdmissionError(
                    "paid model differs from the authorized model")
        except paid_client.PaidAdmissionError as exc:
            sys.exit(f"[paid] pre-compute admission refused: {exc}")

    # Tokenize
    print(f"[chain] tokenizing locally")
    tokens, llama = tokenize_local(args.model_path, args.prompt, args.tokenizer_socket)
    if paid is None:
        print(f"[chain] {len(tokens)} prompt tokens: {tokens}")
    else:
        print(f"[chain] {len(tokens)} prompt tokens")
    if paid is not None:
        if len(tokens) != paid["max_prompt_tokens"]:
            sys.exit("[paid] tokenized prompt differs from the authorized prompt cap")
        if args.max_tokens != paid["max_output_tokens"]:
            sys.exit("[paid] output cap differs from authorization")

    # ── Speculative decoding setup (slice 1) ──────────────────────────────────
    # Engages only when explicitly enabled, a draft model is given, we're in unary
    # Forward mode, and EVERY worker advertises the "speculative" capability. Any
    # miss → plain decode (no error). Mirrors the push capability-negotiation above.
    draft = None
    accept = kv_keep_after = None          # bound when spec engages; used only then
    spec_active = False
    spec_k = max(1, int(args.draft_max))
    if args.speculative:
        reasons = []
        use_eagle = bool(args.eagle_head)
        if not args.draft_model_path and not use_eagle:
            reasons.append("no --draft-model-path / --eagle-head")
        if args.use_streaming or args.use_streaming_push:
            reasons.append("streaming/push mode (slice 1 verify is unary-only)")
        non_spec = [w["id"] for w, _, info in sorted_stubs
                    if "speculative" not in info.protocol_capabilities]
        if non_spec:
            reasons.append(f"workers lack 'speculative' capability: {non_spec}")
        if use_eagle:
            no_eagle = [w["id"] for w, _, info in sorted_stubs
                        if "eagle_hidden" not in info.protocol_capabilities]
            if no_eagle:
                reasons.append(f"workers lack 'eagle_hidden' capability: {no_eagle}")
        if reasons:
            print(f"[spec] requested but disabled: {'; '.join(reasons)} — plain decode.",
                  file=sys.stderr)
        elif use_eagle:
            try:
                from speculative import accept, kv_keep_after
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))
                from eagle_speculative import EagleDraft
                first_w, first_stub, _ = sorted_stubs[0]
                def _eagle_cmd5_fn(prefix_ids, _stub=first_stub, _wid=first_w["id"]):
                    n = len(prefix_ids)
                    raw = call_forward(_stub, struct.pack(f"<{n}i", *prefix_ids), n,
                                       has_token_ids=True, worker_id=_wid,
                                       eagle_hidden=True,
                                       rpc_policy=first_w["_rpc_policy"])
                    return list(struct.unpack(f"<{len(raw) // 4}f", raw))
                draft = EagleDraft(args.eagle_head, args.eagle_base,
                                   args.eagle_config, _eagle_cmd5_fn)
                spec_active = True
                print(f"[spec] ON: EAGLE head={args.eagle_head} "
                      f"(cmd=5 via {first_w['id']}) K={spec_k}", flush=True)
            except Exception as e:
                print(f"[spec] EAGLE head load failed ({e!r}) — plain decode.",
                      file=sys.stderr)
        else:
            try:
                from speculative import DraftModel, accept, kv_keep_after
                draft = DraftModel(args.draft_model_path)
                spec_active = True
                print(f"[spec] ON: draft={args.draft_model_path} K={spec_k}", flush=True)
            except Exception as e:
                print(f"[spec] draft load failed ({e!r}) — plain decode.", file=sys.stderr)

    # ── Speculative decode ON THE STREAM setup ─────────────────────────────────
    # Independent of the unary --speculative gate above (which explicitly disables
    # itself in streaming mode — "slice 1 verify is unary-only"). Engages only when
    # requested, a draft model loads, and every worker in the chain advertises
    # "stream_spec". Any miss -> plain streaming decode (no error).
    stream_spec_active = False
    stream_draft = None
    if args.stream_spec:
        worker_caps = [(w["id"], info.protocol_capabilities) for w, _, info in sorted_stubs]
        ss_reasons = stream_spec_disable_reasons(
            args.use_streaming, args.use_streaming_push,
            bool(args.draft_model_path), worker_caps,
        )
        if ss_reasons:
            print(f"[stream-spec] requested but disabled: {'; '.join(ss_reasons)} — "
                  f"falling back to plain streaming decode.", file=sys.stderr)
        else:
            try:
                from speculative import DraftModel, accept, kv_keep_after, speculative_round
                stream_draft = DraftModel(args.draft_model_path)
                stream_spec_active = True
                print(f"[stream-spec] ON: draft={args.draft_model_path} K={spec_k}", flush=True)
            except Exception as e:
                print(f"[stream-spec] draft load failed ({e!r}) — plain streaming decode.",
                      file=sys.stderr)

    # Streaming KV reuse: workers keep their KV cache across steps. Step 0 is
    # the cold prefill (full prompt, keep_kv=False, start_pos=0). Each later
    # step ships only the previous newly-generated token (n=1, keep_kv=True,
    # start_pos=prefix_length) — workers append to their existing KV cache.
    #
    # In streaming mode (v0.5 M0.5.1), the worker tracks first-step-vs-rest
    # implicitly from the position in the stream; the client doesn't pass
    # keep_kv at all. Otherwise the chain walk is structurally identical.
    generated = []
    prefix_length = 0
    timing = {}  # worker_id -> [per-call seconds]

    streamers: list = []
    session_id = ""
    paid_invoked: set[str] = set()
    # Streamers are opened/re-opened by the recovery loop below — see M0.5.4 v0.

    def _next_server_for(idx):
        """Return a NextServer proto pointing at the worker AFTER idx, or None
        if idx is the last worker. v0.5 M0.5.3 v1."""
        if idx + 1 >= len(sorted_stubs):
            return None
        nxt_w = sorted_stubs[idx + 1][0]
        return pb.NextServer(address=_push_dial_address(nxt_w),
                              session_id=session_id)

    def _chain_from(idx):
        """Return the full remaining chain after worker idx — i.e. workers
        idx+1, idx+2, ..., last. Empty if idx is the last worker. v0.5 M0.5.3 v2."""
        out = []
        for w, _, _ in sorted_stubs[idx + 1:]:
            out.append(pb.NextServer(address=_push_dial_address(w),
                                      session_id=session_id))
        return out

    def _step_call(idx, stub_tup, payload, n, has_tok, keep_kv, prefix_length,
                   all_logits=False):
        w = stub_tup[0]
        if args.use_streaming_push:
            # Only the first worker is contacted directly. The step carries
            # the FULL remaining chain (v2) so each worker downstream knows
            # who to forward to. The response we receive on our single stream
            # is the last worker's token_id, unwound back through the chain.
            chain = _chain_from(0)
            return call_inference_step(streamers[0], payload, n, has_tok,
                                       session_id=session_id, step_idx=step,
                                       prefix_length=prefix_length, timing=timing,
                                       chain=chain)
        if args.use_streaming:
            spend_claim_token = ""
            if paid is not None:
                worker_key = paid_client.worker_keyid(w)
                if worker_key not in paid_invoked:
                    # Record conservatively before the RPC. If the process
                    # dies after this write, the gateway will never abandon
                    # an entry that might have reached its worker.
                    paid_client.write_invocation_event(
                        args.paid_progress_fd, worker_key)
                    paid_invoked.add(worker_key)
                    spend_claim_token = paid["claim_token"]
            return call_inference_step(streamers[idx], payload, n, has_tok,
                                       session_id=session_id, step_idx=step,
                                       prefix_length=prefix_length, timing=timing,
                                       spend_claim_token=spend_claim_token)
        return call_forward(stub_tup[1], payload, n, has_token_ids=has_tok,
                            worker_id=w["id"],
                            keep_kv=keep_kv, start_pos=prefix_length, timing=timing,
                            all_logits=all_logits, rpc_policy=w["_rpc_policy"])

    def _run_async_pipeline():
        """Async-pipelined speculative decode over the unary-Forward chain (spec's transport).

        Keeps up to args.async_inflight verify-chunks in flight so every worker stage stays busy
        instead of one-at-a-time. Output is byte-identical to the sequential spec path — the
        scheduler (async_pipeline.pipelined_spec_decode) commits only verify-confirmed tokens and
        flushes mispredicted speculative chunks, whose KV writes the daemon rewinds via start_pos.
        Appends committed tokens to `generated`; raises on transport fault so the caller can fall
        back to sequential decode."""
        from async_pipeline import PipelineChain, pipelined_spec_decode

        stubs = [t[1] for t in sorted_stubs]
        wids = [t[0]["id"] for t in sorted_stubs]

        # Stage adapter for worker i: verify traversal ⇒ all_logits=True, keep_kv=True. The
        # daemon trims its KV to start_pos before decoding (M3 fusion) — which is exactly the
        # rewind a flushed chunk's corrected re-issue relies on. `meta` (unique per issued chunk)
        # is unused here because unary Forward has no step_id idempotency cache; it exists so the
        # streaming variant can key on it without a corrupt cache hit.
        def make_stage(i):
            def stage(payload, n, start_pos, first, last, meta):
                return call_forward(stubs[i], payload, n, has_token_ids=first,
                                    worker_id=wids[i], keep_kv=True, start_pos=start_pos,
                                    all_logits=True, timing=timing,
                                    rpc_policy=sorted_stubs[i][0]["_rpc_policy"])
            return stage
        chain = PipelineChain([make_stage(i) for i in range(len(sorted_stubs))])
        try:
            # ── cold prefill (sequential, plain): establish KV on every worker ──────────
            pre = list(tokens) + list(generated)
            n_pre = len(pre)
            buf = struct.pack(f"<{n_pre}i", *pre)
            for i, (w, stub, _) in enumerate(sorted_stubs):
                buf = call_forward(stub, buf, n_pre, has_token_ids=(i == 0), worker_id=w["id"],
                                   keep_kv=False, start_pos=0, timing=timing,
                                   rpc_policy=w["_rpc_policy"])
                if i < len(sorted_stubs) - 1 and len(buf) != _hidden_bytes(n_pre, n_embd):
                    raise RuntimeError(f"async prefill: worker {w['id']!r} returned {len(buf)} bytes")
            first_id = struct.unpack("<i", buf)[0]
            generated.append(first_id)
            print(f"[async] prefill → id={first_id} '{detok_one(llama, first_id)}'; "
                  f"pipelining spec-decode, depth={args.async_inflight}, K={spec_k}", flush=True)
            if first_id in LLAMA3_EOS_IDS:
                return

            def _emit(t):
                print(f"[async] +'{detok_one(llama, t)}' (id={t})", flush=True)

            out = pipelined_spec_decode(
                chain=chain, propose=draft.propose, prompt=pre, first_cur=first_id,
                start_pos0=n_pre, spec_k=spec_k,
                max_new=max(0, args.max_tokens - len(generated)),
                max_inflight=max(1, int(args.async_inflight)),
                eos_ids=LLAMA3_EOS_IDS, on_token=_emit)
            generated.extend(out)
        finally:
            chain.close()

    # v0.5 M0.5.4 v0: recovery loop. Wraps the chain walk in an outer
    # retry that, on stream/RPC failure, closes streams, opens fresh ones,
    # and re-prefills with (prompt + tokens already generated). The model
    # picks up generation from where we left off; the next produced token
    # will differ from what the dead chain would have produced (per the
    # Metal non-determinism finding — see v0.5-design-lock.md preface),
    # but the request completes without surfacing an error to the caller.
    recovery_attempts = 0
    restart_requested = True

    class SimulatedFailure(Exception):
        """Test-only synthetic failure for --simulate-fail-step."""

    t0 = time.time()
    while restart_requested:
        restart_requested = False
        # Open / re-open streamers on every (re-)attempt. Fresh session_id
        # so the worker idempotency cache from the dead session can't poison
        # the new one.
        for s in streamers:
            s.close()
        streamers.clear()
        if args.use_streaming:
            session_id = paid["session_id"] if paid is not None else uuid.uuid4().hex
            if args.use_streaming_push:
                first_w, first_stub, _ = sorted_stubs[0]
                streamer = InferenceStream(first_stub, first_w["id"])
                streamer._rpc_policy = first_w["_rpc_policy"]
                streamers.append(streamer)
                tag = f"streaming-push, 1 stream to {first_w['id']}"
            else:
                for w, stub, _ in sorted_stubs:
                    streamer = InferenceStream(stub, w["id"])
                    streamer._rpc_policy = w["_rpc_policy"]
                    streamers.append(streamer)
                tag = f"streaming, {len(streamers)} streams"
            if recovery_attempts == 0:
                print(f"[chain] {tag}  session_id={session_id[:8]}…")
            else:
                print(f"[recovery] attempt {recovery_attempts}: {tag}; "
                      f"replaying {len(generated)} pre-failure tokens",
                      file=sys.stderr)

        prefix_length = 0  # workers' KV is fresh on (re-)open
        already_done = len(generated)

        # ── Async-pipelined speculative decode (opt-in) ────────────────────────────────
        # Spec-decode runs over the unary Forward path, so async pipelining applies there
        # (not streaming/push). On any transport fault, drop spec + fall back to the proven
        # sequential loop via the normal recovery machinery.
        if (args.async_pipeline and spec_active
                and not args.use_streaming and not args.use_streaming_push):
            try:
                _run_async_pipeline()
                break  # generation complete → leave the recovery loop
            except (grpc.RpcError, RuntimeError) as e:
                print(f"[async] pipeline failed ({type(e).__name__}: {e}); "
                      f"falling back to sequential decode", file=sys.stderr)
                spec_active = False   # a mid-pipeline fault may leave KV inconsistent
                recovery_attempts += 1
                if recovery_attempts > args.max_recovery_attempts:
                    for s in streamers:
                        s.close()
                    raise
                restart_requested = True
                continue

        try:
            for step in range(already_done, args.max_tokens):
                # ── Speculative decode ON THE STREAM (2026-07-29) ───────────────
                # Same shape as the unary spec branch below, but the verify traversal
                # rides the persistent per-worker Inference streams (stream_spec_verify_fn)
                # instead of unary Forward — that's the whole point (no per-call RPC setup
                # on each of the K+1 verify positions). Reuses speculative_round (unchanged,
                # already proven byte-identical-to-greedy) so only the transport differs.
                if stream_spec_active and step > already_done:
                    verify_fn = stream_spec_verify_fn(
                        streamers, sorted_stubs, n_embd, session_id, step,
                        prefix_length, timing=timing)
                    res, drafts = speculative_round(
                        stream_draft, tokens + generated, spec_k, verify_fn)
                    # KV rewind: the workers' streams just wrote all n_v verify positions;
                    # the NEXT step's prefix_length is the kept-KV length, smaller when the
                    # draft was wrong anywhere — worker.py's Inference handler passes this
                    # straight through as start_pos to the daemon (keep_kv=True), which
                    # trims to it before decoding. Same primitive as the unary M3 fusion,
                    # now driven over the stream.
                    prefix_length = kv_keep_after(prefix_length, res.n_accepted)
                    stop = False
                    for t in res.committed:
                        if len(generated) >= args.max_tokens:
                            stop = True
                            break
                        generated.append(t)
                        if t in LLAMA3_EOS_IDS:
                            print(f"[stream-spec] step {step+1}: EOS {t} — stopping")
                            stop = True
                            break
                        print(f"[stream-spec] step {step+1}: +'{detok_one(llama, t)}' "
                              f"(accept {res.n_accepted}/{spec_k})", flush=True)
                    if stop or len(generated) >= args.max_tokens:
                        break
                    continue

                # ── Speculative decode (slice 1) ──────────────────────────────
                # Decode phase only; the cold prefill (step == already_done) stays
                # plain. The draft proposes K tokens; ONE verify traversal (all_logits)
                # returns K+1 greedy argmaxes; accept() commits the longest matching
                # prefix + correction; TruncateKV fans the rejected tail out to EVERY
                # worker. Any error propagates to the recovery except below, which
                # resets all KV and disables spec → plain decode resumes (never raises
                # into Prithvi's path). Output is identical to plain greedy decode.
                if spec_active and step > already_done:
                    cur = generated[-1]
                    drafts = draft.propose(tokens + generated, spec_k)
                    verify = [cur] + drafts
                    n_v = len(verify)
                    payload = struct.pack(f"<{n_v}i", *verify)
                    if len(sorted_stubs) == 1:
                        # Solo chain: same reasoning as the plain-decode loop and
                        # stream_spec_verify_fn — the one worker's first response IS the
                        # verify argmaxes, no hidden-state relay.
                        last_resp = _step_call(0, sorted_stubs[0], payload, n_v, True, True,
                                               prefix_length, all_logits=True)
                        if len(last_resp) != n_v * 4:
                            raise RuntimeError(f"spec: solo worker returned {len(last_resp)} bytes, "
                                               f"expected {n_v * 4} verify argmaxes")
                    else:
                        hidden = _step_call(0, sorted_stubs[0], payload, n_v, True, True,
                                            prefix_length, all_logits=True)
                        if len(hidden) != _hidden_bytes(n_v, n_embd):
                            raise RuntimeError(f"spec: first worker returned {len(hidden)} bytes")
                        for idx, stub_tup in enumerate(sorted_stubs[1:-1], start=1):
                            hidden = _step_call(idx, stub_tup, hidden, n_v, False, True,
                                                prefix_length, all_logits=True)
                            if len(hidden) != _hidden_bytes(n_v, n_embd):
                                raise RuntimeError(f"spec: middle worker returned {len(hidden)} bytes")
                        last_i = len(sorted_stubs) - 1
                        last_resp = _step_call(last_i, sorted_stubs[last_i], hidden, n_v, False,
                                               True, prefix_length, all_logits=True)
                        if len(last_resp) != n_v * 4:
                            raise RuntimeError(f"spec: last worker returned {len(last_resp)} bytes, "
                                               f"expected {n_v * 4} verify argmaxes")
                    target_argmax = list(struct.unpack(f"<{n_v}i", last_resp))
                    res = accept(drafts, target_argmax)
                    n_keep = kv_keep_after(prefix_length, res.n_accepted)
                    # M3 FUSION (2026-06-21): no separate TruncateKV round-trip.
                    # Setting prefix_length = n_keep makes the NEXT forward's start_pos
                    # = n_keep, and the daemon now truncates its KV to start_pos before
                    # decoding (worker_daemon.cpp keep_kv path) — folding the rejected-
                    # tail discard into the next verify. ONE round-trip per speculative
                    # step instead of two (RT/token 2.0 -> 1.0).
                    prefix_length = n_keep   # next start_pos == kept KV length; daemon trims to it
                    stop = False
                    for t in res.committed:
                        if len(generated) >= args.max_tokens:
                            stop = True
                            break
                        generated.append(t)
                        if t in LLAMA3_EOS_IDS:
                            print(f"[spec] step {step+1}: EOS {t} — stopping")
                            stop = True
                            break
                        print(f"[spec] step {step+1}: +'{detok_one(llama, t)}' "
                              f"(accept {res.n_accepted}/{spec_k})", flush=True)
                    if stop or len(generated) >= args.max_tokens:
                        break
                    continue

                if step == already_done:
                    # Cold prefill: prompt + any tokens already generated
                    # before this attempt. This rebuilds KV state on the
                    # fresh streams so generation can resume at step+1.
                    input_tokens = tokens + list(generated)
                    keep_kv = False
                else:
                    input_tokens = [generated[-1]]
                    keep_kv = True
                n_step = len(input_tokens)

                # v0.5 M0.5.3 push mode: client only contacts the FIRST worker.
                # The response we receive comes from the last worker (via the chain
                # of peer pushes); it's a token id, not a hidden state.
                token_payload = struct.pack(f"<{n_step}i", *input_tokens)
                if args.use_streaming_push:
                    last_resp = _step_call(0, sorted_stubs[0], token_payload, n_step, True,
                                           keep_kv, prefix_length)
                    if len(last_resp) != 4:
                        sys.exit(f"[chain] push mode: expected 4-byte token id from chain, got {len(last_resp)} bytes")
                    next_id = struct.unpack("<i", last_resp)[0]
                    generated.append(next_id)
                elif len(sorted_stubs) == 1:
                    # Solo chain (route-whole placement, 2026-09-12): the one worker holds
                    # EVERY layer (has_token_embd AND has_lm_head both true — see worker.py's
                    # mode="solo" handling), so it returns a token id directly from the first
                    # and only call. There is no hidden-state relay to do. Treat this exactly
                    # like the "last worker" case below, just with zero prior hops.
                    solo_resp = _step_call(0, sorted_stubs[0], token_payload, n_step, True,
                                           keep_kv, prefix_length)
                    if len(solo_resp) != 4:
                        sys.exit(f"[chain] solo worker {sorted_stubs[0][0]['id']!r} returned {len(solo_resp)} bytes, expected 4")
                    next_id = struct.unpack("<i", solo_resp)[0]
                    generated.append(next_id)
                else:
                    # Step 1: tokens → first worker → hidden.
                    # NOTE (#17): a wrong hidden byte-count is CORRUPTION/config (e.g. a worker
                    # with a different NAKSHATRA_ACT_QUANT setting than the client), NOT a
                    # transport fault — swapping alternates can't fix it and would just churn.
                    # So these size-mismatch checks stay fail-fast (sys.exit), unlike the
                    # transport faults above which now raise into the recovery loop.
                    hidden = _step_call(0, sorted_stubs[0], token_payload, n_step, True,
                                        keep_kv, prefix_length)
                    if len(hidden) != _hidden_bytes(n_step, n_embd):
                        sys.exit(f"[chain] first worker {sorted_stubs[0][0]['id']!r} returned {len(hidden)} bytes, expected {_hidden_bytes(n_step, n_embd)}")

                    # Steps 2..N-1: middle workers
                    for idx, stub_tup in enumerate(sorted_stubs[1:-1], start=1):
                        hidden = _step_call(idx, stub_tup, hidden, n_step, False,
                                            keep_kv, prefix_length)
                        if len(hidden) != _hidden_bytes(n_step, n_embd):
                            sys.exit(f"[chain] middle worker {stub_tup[0]['id']!r} returned {len(hidden)} bytes")

                    # Step N: last worker → token id
                    last_idx = len(sorted_stubs) - 1
                    last_resp = _step_call(last_idx, sorted_stubs[last_idx], hidden, n_step, False,
                                           keep_kv, prefix_length)
                    if len(last_resp) != 4:
                        sys.exit(f"[chain] last worker {sorted_stubs[last_idx][0]['id']!r} returned {len(last_resp)} bytes, expected 4")
                    next_id = struct.unpack("<i", last_resp)[0]
                    generated.append(next_id)
                prefix_length += n_step

                if next_id in LLAMA3_EOS_IDS:
                    print(f"[chain] step {step+1}: EOS {next_id} — stopping")
                    break
                if paid is None:
                    print(f"[chain] step {step+1}: id={next_id} '{escape_step_text(detok_one(llama, next_id))}'", flush=True)
                else:
                    print(f"[chain] step {step+1}: '{escape_step_text(detok_one(llama, next_id))}'", flush=True)

                # v0.5 M0.5.4 v0: synthetic failure injection for testing.
                # Triggers AFTER step `simulate_fail_step` completes, so the
                # token from that step is already in `generated[]` — the
                # recovery loop will replay it as part of the new prefill.
                if (args.simulate_fail_step >= 0
                        and step + 1 == args.simulate_fail_step
                        and recovery_attempts == 0):
                    raise SimulatedFailure(
                        f"injected failure after step {step+1} "
                        f"(--simulate-fail-step {args.simulate_fail_step})"
                    )
        except (grpc.RpcError, RuntimeError, SimulatedFailure) as e:
            recovery_attempts += 1
            if paid is not None:
                # Authorization pins one topology and one stable session. A
                # paid run never swaps candidates, changes capabilities, or
                # replays on a new session after any compute may have run.
                for s in streamers:
                    s.close()
                raise
            if spec_active:
                # A speculative step may have left KV inconsistent across workers
                # (partial TruncateKV). Recovery resets ALL KV via cold replay; also
                # drop to plain decode so we can't re-hit the same spec failure.
                spec_active = False
                print("[spec] disabling speculative decode after failure; "
                      "recovery replays with plain decode.", file=sys.stderr)
            if stream_spec_active:
                # Same reasoning as spec_active above: a verify round may have left
                # per-worker KV inconsistent (rewind not yet applied everywhere).
                # Recovery re-opens fresh streams and cold-replays, so plain streaming
                # decode is the safe resume path.
                stream_spec_active = False
                print("[stream-spec] disabling stream-spec after failure; "
                      "recovery replays with plain streaming decode.", file=sys.stderr)
            kind = "simulated" if isinstance(e, SimulatedFailure) else type(e).__name__
            print(f"[recovery] {kind} failure: {e}", file=sys.stderr)
            if recovery_attempts > args.max_recovery_attempts:
                print(f"[recovery] {recovery_attempts-1} attempts exhausted; "
                      f"surfacing failure to caller", file=sys.stderr)
                for s in streamers:
                    s.close()
                raise
            # v0.5 §9.5: push failure is a session-scope downgrade, not a
            # worker swap. The broken link is between two healthy workers —
            # advancing to an alternate wouldn't help and might mask the
            # underlying connectivity issue. We just turn push off for the
            # rest of this session and let streaming-only relay continue.
            if isinstance(e, PushFailure) and args.use_streaming_push:
                print(f"[recovery] push failed mid-session — downgrading to "
                      f"streaming-only (client relays per-step) for the rest "
                      f"of this session", file=sys.stderr)
                args.use_streaming_push = False
                restart_requested = True
                continue
            # v0.5 M0.5.4 v1: try advancing to a fresh alternate worker first;
            # only fall back to retry-same-worker if no alternates remain.
            advanced = _advance_one_alternate()
            if advanced is not None:
                new_spec = _spec(advanced)
                print(f"[recovery] swapping worker {advanced['id']!r} to alternate "
                      f"{new_spec['address']}:{new_spec['port']} "
                      f"(candidate {advanced['cursor']}/{len(advanced['candidates'])-1})",
                      file=sys.stderr)
                # Re-setup chain with new candidate(s)
                sorted_stubs[:], n_embd_new = _setup_chain("recovery")
                # n_embd shouldn't change across same-arch alternates; if it does, error.
                if n_embd_new != n_embd:
                    sys.exit(f"[recovery] alternate has different hidden_size "
                             f"({n_embd_new} vs {n_embd}); cannot continue")
            else:
                print(f"[recovery] no alternates remaining; retrying same workers",
                      file=sys.stderr)
            restart_requested = True
            # streams are closed at the top of the next iteration

    # Successful exit from the recovery loop
    for s in streamers:
        s.close()

    elapsed = time.time() - t0
    full = llama.detokenize(tokens + generated).decode("utf-8", errors="replace")
    gen = llama.detokenize(generated).decode("utf-8", errors="replace") if generated else ""
    print(f"[chain] generated {len(generated)} tokens in {elapsed:.2f}s  ({len(generated)/elapsed:.2f} tok/s)")

    # speed-stack #20: optional verifiable run receipt. Additive + best-effort — a receipt
    # failure must never break the run (wrapped). Assembled from data the coordinator already
    # holds: generated tokens, per-worker `timing`, and each stage's Info (id/spki/layer-map).
    if args.receipt_out:
        try:
            from receipt import build_receipt
            chain = []
            for w, _stub, info in sorted_stubs:
                ts = timing.get(w["id"], [])
                chain.append({
                    "node_id": w["id"],
                    "address": f'{w.get("address")}:{w.get("port")}',
                    "spki_hash": w.get("peer_spki_hash"),
                    "layer_start": info.layer_start,
                    "layer_end": info.layer_end,
                    "backend": info.backend,
                    "mean_rpc_ms": (sum(ts) / len(ts) * 1000.0) if ts else None,
                })
            run_id = (session_id or uuid.uuid4().hex)
            # Build once WITHOUT signatures to learn the run's output_sha256 — it covers the
            # whole generated sequence, so it does not exist until generation is finished.
            # ⚠️ That ordering is the reason a worker cannot sign inline on ForwardResponse or
            # the Inference stream, and why attesting someone else's node needs a post-run RPC.
            rcpt = build_receipt(
                run_id=run_id,
                model_id=(args.model_id or args.model_path),
                prompt_tokens=tokens, generated_tokens=generated,
                workers=chain, elapsed_s=elapsed,
                started_at=t0, ended_at=t0 + elapsed,
            )
            # ⚠️⚠️ ASK EACH WORKER TO SIGN FOR ITSELF. This is the half a keyring cannot
            # do: the coordinator holds no stranger's private key — deliberately, since one
            # that could mint a worker's signature defeats the purpose of signatures. Each
            # worker signs with ITS OWN key and returns what IT believes it served.
            #
            # ⚠️ Additive and best-effort by design. A worker predating the RPC returns
            # UNIMPLEMENTED; one that has exited raises; one with no identity returns
            # FAILED_PRECONDITION. NONE is fatal — the stage goes unsigned,
            # `signature_coverage` reports it as a gap, and that stage earns nothing. A run
            # that produced tokens must not be failed because a worker could not be paid.
            collected_sigs = []
            for w, _stub_sp, info in sorted_stubs:
                try:
                    sp_req = pb.SignParticipationRequest(
                        run_id=run_id, output_sha256=rcpt["output_sha256"],
                        layer_start=info.layer_start, layer_end=info.layer_end)
                    # An authenticated worker (every pillar-registered one) refuses an unsigned
                    # call with UNAUTHENTICATED, so the stage went unsigned and earned nothing
                    # (found live on bro, 2026-10-13). Sign it exactly like Forward.
                    r = _stub_sp.SignParticipation(
                        sp_req, timeout=20,
                        metadata=_grpc_auth_metadata("SignParticipation", sp_req.SerializeToString()))
                except Exception as e:                       # noqa: BLE001
                    code = getattr(e, "code", lambda: None)()
                    note = getattr(code, "name", None) or type(e).__name__
                    print(f"[receipt] {w['id']}: unsigned ({note}) — that stage earns nothing",
                          file=sys.stderr)
                    continue
                if not r.sig:
                    continue
                collected_sigs.append({"node_id": r.node_id, "pubkey": r.pubkey,
                                       "layer_start": r.layer_start,
                                       "layer_end": r.layer_end, "sig": r.sig})
                if (r.layer_start, r.layer_end) != (info.layer_start, info.layer_end):
                    # ⚠️ NOT an error, and deliberately not corrected. The worker signed what
                    # it believes; the disagreement is evidence and `signature_coverage`
                    # surfaces it. Trusting the coordinator here would erase the only signal
                    # that the two disagree.
                    print(f"[receipt] {w['id']} signed [{r.layer_start},{r.layer_end}) but the "
                          f"chain says [{info.layer_start},{info.layer_end}) — coverage will "
                          "show the gap", file=sys.stderr)
            if collected_sigs:
                rcpt = build_receipt(
                    run_id=run_id, model_id=(args.model_id or args.model_path),
                    prompt_tokens=tokens, generated_tokens=generated,
                    workers=chain, elapsed_s=elapsed,
                    started_at=t0, ended_at=t0 + elapsed,
                    worker_signatures=collected_sigs)
                print(f"[receipt] {len(collected_sigs)}/{len(chain)} stages signed by their "
                      "own workers (participation PROVEN, not coordinator-asserted)")

            if args.participation_keyring:
                # ⚠️ NEVER FALL BACK TO AN UNSIGNED RECEIPT. An operator who passed this flag
                # asked for attested participation; quietly writing a coordinator-asserted one
                # hands them a file saying `signed_by="coordinator"` while they believe it is
                # proven. Anything raised here is caught by the handler at the bottom of this
                # block, which writes NO receipt and prints a banner — see the note there.
                from worker_sigs import build_worker_signatures
                keyring = json.loads(Path(args.participation_keyring).read_text())
                if not isinstance(keyring, dict):
                    raise ValueError("--participation-keyring must be a JSON object "
                                     "{node_id: priv_hex}")
                sigs = build_worker_signatures(
                    [{"node_id": c["node_id"], "layer_start": c["layer_start"],
                      "layer_end": c["layer_end"]} for c in chain],
                    keyring, run_id=run_id, output_sha256=rcpt["output_sha256"])
                rcpt = build_receipt(
                    run_id=run_id,
                    model_id=(args.model_id or args.model_path),
                    prompt_tokens=tokens, generated_tokens=generated,
                    workers=chain, elapsed_s=elapsed,
                    started_at=t0, ended_at=t0 + elapsed,
                    worker_signatures=sigs,
                )
                unsigned = [c["node_id"] for c in chain
                            if c["node_id"] not in {s["node_id"] for s in sigs}]
                print(f"[receipt] signed {len(sigs)}/{len(chain)} stages"
                      + (f"; NO KEY for {unsigned} — those stages earn nothing" if unsigned else ""))
            Path(args.receipt_out).write_text(json.dumps(rcpt, indent=2))
            print(f"[receipt] wrote {args.receipt_out} "
                  f"(output_sha256={rcpt['output_sha256'][:12]}…, {len(chain)} stages)")
        except Exception as e:
            # ⚠️ An operator who passed --participation-keyring asked for ATTESTED
            # participation. One more stderr line in a long run is missable, and the failure
            # is silent where it matters: no receipt means the ledger credits nobody, weeks
            # later, with nothing pointing back to a typo in a keyring path. So that case gets
            # its own unmissable banner. Either way no receipt is written — an unsigned
            # receipt is never substituted for the signed one that was asked for.
            if args.participation_keyring:
                print("\n" + "!" * 72, file=sys.stderr)
                print(f"[receipt] SIGNED RECEIPT REQUESTED BUT NOT WRITTEN: {e!r}",
                      file=sys.stderr)
                print(f"[receipt]   keyring: {args.participation_keyring}", file=sys.stderr)
                print("[receipt]   no receipt was written — these stages will earn NOTHING",
                      file=sys.stderr)
                print("!" * 72 + "\n", file=sys.stderr)
            else:
                print(f"[receipt] failed to write receipt: {e!r}", file=sys.stderr)
    print(f"[chain] per-worker total RPC time:")
    for wid, ts in timing.items():
        avg_first = ts[0] if ts else 0.0
        avg_rest = sum(ts[1:]) / max(1, len(ts) - 1)
        print(f"    {wid:14s}  step1(prefill)={avg_first*1000:.0f}ms  steps2-N(stream) avg={avg_rest*1000:.0f}ms  total={sum(ts):.2f}s  n_calls={len(ts)}")
    if paid is None:
        print(f"[chain] full: {full!r}")
    print(f"[chain] gen:  {gen!r}")
    if paid is None:
        print(f"TOPTOKS_CHAIN {' '.join(str(t) for t in generated)}")


if __name__ == "__main__":
    main()
