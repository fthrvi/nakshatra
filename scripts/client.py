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
import argparse
import json
import struct
import sys
import time
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
    args = ap.parse_args()

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
    generated = []
    prefix_length = 0
    timing = {}  # worker_id -> [per-call seconds]
    t0 = time.time()
    for step in range(args.max_tokens):
        if step == 0:
            input_tokens = tokens
            keep_kv = False
        else:
            input_tokens = [generated[-1]]
            keep_kv = True
        n_step = len(input_tokens)

        # Step 1: tokens → first worker → hidden
        token_payload = struct.pack(f"<{n_step}i", *input_tokens)
        first_w, first_stub, _ = sorted_stubs[0]
        hidden = call_forward(first_stub, token_payload, n_step, has_token_ids=True,
                              worker_id=first_w["id"],
                              keep_kv=keep_kv, start_pos=prefix_length, timing=timing)
        if len(hidden) != n_step * n_embd * 4:
            sys.exit(f"[chain] first worker {first_w['id']!r} returned {len(hidden)} bytes, expected {n_step*n_embd*4}")

        # Steps 2..N-1: middle workers
        for w, stub, info in sorted_stubs[1:-1]:
            hidden = call_forward(stub, hidden, n_step, has_token_ids=False,
                                  worker_id=w["id"],
                                  keep_kv=keep_kv, start_pos=prefix_length, timing=timing)
            if len(hidden) != n_step * n_embd * 4:
                sys.exit(f"[chain] middle worker {w['id']!r} returned {len(hidden)} bytes")

        # Step N: last worker → token id
        last_w, last_stub, _ = sorted_stubs[-1]
        last_resp = call_forward(last_stub, hidden, n_step, has_token_ids=False,
                                 worker_id=last_w["id"],
                                 keep_kv=keep_kv, start_pos=prefix_length, timing=timing)
        if len(last_resp) != 4:
            sys.exit(f"[chain] last worker {last_w['id']!r} returned {len(last_resp)} bytes, expected 4")
        next_id = struct.unpack("<i", last_resp)[0]
        generated.append(next_id)
        prefix_length += n_step

        if next_id in LLAMA3_EOS_IDS:
            print(f"[chain] step {step+1}: EOS {next_id} — stopping")
            break
        print(f"[chain] step {step+1}: id={next_id} '{detok_one(llama, next_id)}'", flush=True)

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
