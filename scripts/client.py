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
import json
import queue
import struct
import sys
import time
import uuid
from pathlib import Path
from urllib import request as urlrequest

sys.path.insert(0, str(Path(__file__).parent))

import grpc
import yaml
import nakshatra_pb2 as pb
import nakshatra_pb2_grpc as pb_grpc


LLAMA3_EOS_IDS = {128001, 128008, 128009}


def tokenize_local(model_path, prompt):
    from llama_cpp import Llama
    llama = Llama(model_path=model_path, vocab_only=True, verbose=False)
    return llama.tokenize(prompt.encode("utf-8"), add_bos=True, special=True), llama


def detok_one(llama, tid):
    try:
        return llama.detokenize([tid]).decode("utf-8", errors="replace")
    except Exception:
        return "?"


def call_forward(stub, payload, n_tokens, has_token_ids, worker_id="<unknown>",
                 keep_kv=False, start_pos=0, timing=None):
    req = pb.ForwardRequest(
        hidden_in=payload, batch=1, n_tokens=n_tokens, has_token_ids=has_token_ids,
        keep_kv=keep_kv, start_pos=start_pos,
    )
    t0 = time.time()
    try:
        resp = stub.Forward(req, timeout=300.0)
    except grpc.RpcError as e:
        sys.exit(
            f"[chain] Forward RPC to worker {worker_id!r} failed: "
            f"{e.code().name} — {e.details()}"
        )
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

    def __init__(self, stub, worker_id):
        self.worker_id = worker_id
        self._req_q = queue.Queue()
        self._closed = False
        self._responses = stub.Inference(self._request_gen())

    def _request_gen(self):
        while True:
            step = self._req_q.get()
            if step is None:
                return
            yield step

    def step(self, request_step):
        if self._closed:
            raise RuntimeError(f"stream to {self.worker_id} already closed")
        self._req_q.put(request_step)
        try:
            return next(self._responses)
        except grpc.RpcError as e:
            raise RuntimeError(
                f"Inference stream to {self.worker_id!r} failed: "
                f"{e.code().name} — {e.details()}"
            ) from e

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._req_q.put(None)


def call_inference_step(streamer, payload, n_tokens, has_token_ids,
                         session_id, step_idx, prefix_length, timing=None,
                         next_server=None, chain=None):
    """v0.5 M0.5.1 streaming equivalent of call_forward. Returns raw bytes
    matching Forward's return shape (hidden state OR int32 token id), so
    chain-walk code consumes the result identically.

    Push-related kwargs (mutually exclusive):
      - next_server (v0.5 M0.5.3 v1): single immediate-next worker. The
        receiving worker forwards there. Sufficient for 2-worker chains.
      - chain (v0.5 M0.5.3 v2): list of NextServer for the WHOLE remaining
        chain after the receiving worker. Each downstream worker pops the
        head and forwards the rest. Works for chains of any length.
    """
    step = pb.InferenceStep(
        session_id=session_id,
        step_id=f"step-{step_idx}",
        prefix_length=prefix_length,
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
    try:
        resp = streamer.step(step)
    except RuntimeError as e:
        sys.exit(f"[chain] {e}")
    if timing is not None:
        timing.setdefault(streamer.worker_id, []).append(time.time() - t0)

    if resp.HasField("error"):
        sys.exit(f"[chain] worker {streamer.worker_id!r} reported error: "
                 f"{resp.error.decode('utf-8', 'replace')}")
    if resp.HasField("token_ids"):
        ids = list(resp.token_ids.ids)
        return struct.pack(f"<{len(ids)}i", *ids)
    if resp.HasField("hidden_state"):
        return resp.hidden_state.raw
    sys.exit(f"[chain] worker {streamer.worker_id!r} returned an empty step")


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


def _try_pillar_chain(registry_url: str, model_id: str) -> list | None:
    """Phase I: ask the pillar to assemble the chain itself.

    Pillars from Phase I onward expose GET /chain?model=<id>. Returns
    None if the pillar is older (404) or returns an empty chain (so
    callers fall back to the local builder, which can produce a
    clearer error message). Other failures propagate."""
    url = f"{registry_url.rstrip('/')}/chain?model={model_id}"
    try:
        with urlrequest.urlopen(url, timeout=10) as resp:
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
        host, _, port = (c.get("address") or "").rpartition(":")
        if not port:
            raise RuntimeError(
                f"pillar returned malformed chain entry for "
                f"{c.get('node_id')!r}: address={c.get('address')!r}"
            )
        workers.append({
            "id": (c.get("node_id") or "")[:14],
            "address": host,
            "port": int(port),
            "layer_start": int(c["layer_start"]),
            "layer_end": int(c["layer_end"]),
        })
    return workers


def build_chain_from_registry(registry_url: str, model_id: str) -> list:
    """Query Sthambha registry, return a contiguous chain plan for `model_id`.

    Phase I: prefer the pillar's own /chain endpoint; if absent, fall
    back to the local algorithm (verified-GPU > CPU > drifty-GPU,
    rpc_ms tiebreak inside a tier).

    Returns list of dicts in cluster-YAML format (id, address, port,
    layer_start, layer_end) so downstream code stays identical between
    YAML mode and registry mode.

    Raises RuntimeError if no contiguous chain can be built.
    """
    pillar_chain = _try_pillar_chain(registry_url, model_id)
    if pillar_chain is not None:
        print(f"[chain] using pillar-served chain plan ({len(pillar_chain)} workers)")
        return pillar_chain

    url = f"{registry_url.rstrip('/')}/peers?model={model_id}"
    with urlrequest.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())

    # Collect (peer, offering) pairs from online compute peers
    pairs = []
    for p in data.get("peers", []):
        if not p.get("is_online"):
            continue
        for o in p.get("layer_offerings", []):
            if o.get("model_id") == model_id:
                pairs.append((p, o))

    if not pairs:
        raise RuntimeError(f"no online peers advertise model_id={model_id!r} on {registry_url}")

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
            host, _, port = peer["address"].rpartition(":")
            if not port:
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
                "address": host,
                "port": int(port),
                "layer_start": offering["layer_start"],
                "layer_end": offering["layer_end"],
            })
            used_node_ids.add(peer["node_id"])
            cursor = offering["layer_end"]

    if not chain:
        raise RuntimeError(f"no peer offers layer 0 of model {model_id!r}; cannot start chain")

    return chain


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=str, help="cluster YAML (static)")
    src.add_argument("--registry", type=str,
                     help="Sthambha pillar URL (e.g. http://umbrel:7777). Builds "
                          "chain plan from live registry.")
    ap.add_argument("--model-id", type=str, default="",
                    help="Model id to query in the registry (required if --registry)")
    ap.add_argument("--model-path", type=str, required=True, help="full GGUF path for tokenizer (must match the model that was split)")
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
    args = ap.parse_args()
    # --use-streaming-push implies --use-streaming
    if args.use_streaming_push:
        args.use_streaming = True

    if args.registry:
        if not args.model_id:
            sys.exit("--registry requires --model-id")
        print(f"[chain] querying Sthambha registry: {args.registry} (model={args.model_id})")
        workers = build_chain_from_registry(args.registry, args.model_id)
        print(f"[chain] registry returned a chain of {len(workers)} workers covering layers [0,{workers[-1]['layer_end']})")
    else:
        cfg = yaml.safe_load(Path(args.config).read_text())
        workers = cfg["workers"]
        print(f"[chain] {len(workers)} workers in config")

    # Open channels + Info on each
    stubs = []
    n_embd = None
    for w in workers:
        addr = f"{w['address']}:{w['port']}"
        ch = grpc.insecure_channel(addr)
        stub = pb_grpc.NakshatraStub(ch)
        info = stub.Info(pb.InfoRequest(), timeout=10.0)
        print(f"  {w['id']:12s} {addr:25s}  layers=[{info.layer_start},{info.layer_end})  embd={info.has_token_embd}  lm={info.has_lm_head}  hidden={info.hidden_size}")
        if n_embd is None:
            n_embd = info.hidden_size
        elif info.hidden_size != n_embd:
            sys.exit(f"hidden_size mismatch across workers: {n_embd} vs {info.hidden_size}")
        stubs.append((w, stub, info))

    # Validate chain partition
    n_workers = len(stubs)
    sorted_stubs = sorted(stubs, key=lambda x: x[2].layer_start)
    prev_end = sorted_stubs[0][2].layer_start
    for w, _, info in sorted_stubs:
        if info.layer_start != prev_end:
            sys.exit(f"chain GAP between previous and {w['id']}")
        prev_end = info.layer_end
    if not sorted_stubs[0][2].has_token_embd:
        sys.exit("first worker must have token_embd")
    if not sorted_stubs[-1][2].has_lm_head:
        sys.exit("last worker must have lm_head")
    print(f"[chain] OK: contiguous coverage of [{sorted_stubs[0][2].layer_start}, {sorted_stubs[-1][2].layer_end})")

    # Tokenize
    print(f"[chain] tokenizing locally")
    tokens, llama = tokenize_local(args.model_path, args.prompt)
    print(f"[chain] {len(tokens)} prompt tokens: {tokens}")

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
    # Streamers are opened/re-opened by the recovery loop below — see M0.5.4 v0.

    def _next_server_for(idx):
        """Return a NextServer proto pointing at the worker AFTER idx, or None
        if idx is the last worker. v0.5 M0.5.3 v1."""
        if idx + 1 >= len(sorted_stubs):
            return None
        nxt_w = sorted_stubs[idx + 1][0]
        return pb.NextServer(address=f"{nxt_w['address']}:{nxt_w['port']}",
                              session_id=session_id)

    def _chain_from(idx):
        """Return the full remaining chain after worker idx — i.e. workers
        idx+1, idx+2, ..., last. Empty if idx is the last worker. v0.5 M0.5.3 v2."""
        out = []
        for w, _, _ in sorted_stubs[idx + 1:]:
            out.append(pb.NextServer(address=f"{w['address']}:{w['port']}",
                                      session_id=session_id))
        return out

    def _step_call(idx, stub_tup, payload, n, has_tok, keep_kv, prefix_length):
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
            return call_inference_step(streamers[idx], payload, n, has_tok,
                                       session_id=session_id, step_idx=step,
                                       prefix_length=prefix_length, timing=timing)
        return call_forward(stub_tup[1], payload, n, has_token_ids=has_tok,
                            worker_id=w["id"],
                            keep_kv=keep_kv, start_pos=prefix_length, timing=timing)

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
            session_id = uuid.uuid4().hex
            if args.use_streaming_push:
                first_w, first_stub, _ = sorted_stubs[0]
                streamers.append(InferenceStream(first_stub, first_w["id"]))
                tag = f"streaming-push, 1 stream to {first_w['id']}"
            else:
                for w, stub, _ in sorted_stubs:
                    streamers.append(InferenceStream(stub, w["id"]))
                tag = f"streaming, {len(streamers)} streams"
            if recovery_attempts == 0:
                print(f"[chain] {tag}  session_id={session_id[:8]}…")
            else:
                print(f"[recovery] attempt {recovery_attempts}: {tag}; "
                      f"replaying {len(generated)} pre-failure tokens",
                      file=sys.stderr)

        prefix_length = 0  # workers' KV is fresh on (re-)open
        already_done = len(generated)

        try:
            for step in range(already_done, args.max_tokens):
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
                else:
                    # Step 1: tokens → first worker → hidden
                    hidden = _step_call(0, sorted_stubs[0], token_payload, n_step, True,
                                        keep_kv, prefix_length)
                    if len(hidden) != n_step * n_embd * 4:
                        sys.exit(f"[chain] first worker {sorted_stubs[0][0]['id']!r} returned {len(hidden)} bytes, expected {n_step*n_embd*4}")

                    # Steps 2..N-1: middle workers
                    for idx, stub_tup in enumerate(sorted_stubs[1:-1], start=1):
                        hidden = _step_call(idx, stub_tup, hidden, n_step, False,
                                            keep_kv, prefix_length)
                        if len(hidden) != n_step * n_embd * 4:
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
                print(f"[chain] step {step+1}: id={next_id} '{detok_one(llama, next_id)}'", flush=True)

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
            kind = "simulated" if isinstance(e, SimulatedFailure) else type(e).__name__
            print(f"[recovery] {kind} failure: {e}", file=sys.stderr)
            if recovery_attempts > args.max_recovery_attempts:
                print(f"[recovery] {recovery_attempts-1} attempts exhausted; "
                      f"surfacing failure to caller", file=sys.stderr)
                for s in streamers:
                    s.close()
                raise
            restart_requested = True
            # streams are closed at the top of the next iteration

    # Successful exit from the recovery loop
    for s in streamers:
        s.close()

    elapsed = time.time() - t0
    full = llama.detokenize(tokens + generated).decode("utf-8", errors="replace")
    gen = llama.detokenize(generated).decode("utf-8", errors="replace") if generated else ""
    print(f"[chain] generated {len(generated)} tokens in {elapsed:.2f}s  ({len(generated)/elapsed:.2f} tok/s)")
    print(f"[chain] per-worker total RPC time:")
    for wid, ts in timing.items():
        avg_first = ts[0] if ts else 0.0
        avg_rest = sum(ts[1:]) / max(1, len(ts) - 1)
        print(f"    {wid:14s}  step1(prefill)={avg_first*1000:.0f}ms  steps2-N(stream) avg={avg_rest*1000:.0f}ms  total={sum(ts):.2f}s  n_calls={len(ts)}")
    print(f"[chain] full: {full!r}")
    print(f"[chain] gen:  {gen!r}")
    print(f"TOPTOKS_CHAIN {' '.join(str(t) for t in generated)}")


if __name__ == "__main__":
    main()
