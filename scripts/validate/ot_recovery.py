#!/usr/bin/env python3
"""O(t) in-flight recovery — proven against live workers (v1.1 hardening).

Demonstrates the dual-cache recovery end to end on a real 2-worker chain + an
alternate, using the pieces built earlier:
  • ActivationReplayCache (PR #16) — the client caches each input it relays to the
    last worker;
  • drift_aware.drift_compatible (PR #15) — the replacement must be same-class.

Flow (Forward-relay mode — the client is in the per-hop loop, so it HAS the input
to each worker, the v1.1 §8.5 client-side O(t) path):
  1. drive the chain A→B step by step, caching the input to B per step;
  2. at FAIL_AT, simulate B's death → pick a same-drift-class alternate B';
  3. CATCH UP only B': replay the cached inputs to B for steps 0..T (build its KV),
     discard the tokens. The SURVIVOR A is never touched (keeps its KV);
  4. resume A(untouched) → B'(caught-up).

Proof of O(t): the survivor A is called exactly once per step (never re-run for
the prefix on recovery), and B' replays exactly T steps — O(t) on the one failed
link, not O(chain × T). And the full token sequence is byte-identical to the
no-failure baseline.

Usage: ot_recovery.py A_ADDR B_ADDR BPRIME_ADDR MODEL_PATH FAIL_AT
       (FAIL_AT = -1 → no-failure baseline)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import grpc  # noqa: E402
import nakshatra_pb2 as pb  # noqa: E402
import nakshatra_pb2_grpc as pb_grpc  # noqa: E402
from recovery.activation_cache import ActivationReplayCache  # noqa: E402
from recovery.drift_aware import drift_compatible  # noqa: E402

A_ADDR, B_ADDR, BP_ADDR, MODEL, FAIL_AT = sys.argv[1:6]
FAIL_AT = int(FAIL_AT)
MAX_TOKENS = 8
PROMPT = b"The capital of France is"
SESSION = "ot-demo"
# drift classes (in production these come from each peer's Nostr listing /
# gauge fingerprint; here B and B' run the same build → same class)
PRIMARY_CLASS, ALT_CLASS = "classB", "classB"


def stub(addr):
    return pb_grpc.NakshatraStub(grpc.insecure_channel(addr))


def fwd(s, payload, n, has_tok, keep_kv, start_pos):
    req = pb.ForwardRequest(hidden_in=payload, batch=1, n_tokens=n,
                            has_token_ids=has_tok, keep_kv=keep_kv, start_pos=start_pos)
    return s.Forward(req, timeout=300.0).hidden_out


def main() -> int:
    from llama_cpp import Llama
    llm = Llama(model_path=MODEL, n_ctx=256, verbose=False)
    prompt_ids = list(llm.tokenize(PROMPT))

    A, B, Bp = stub(A_ADDR), stub(B_ADDR), stub(BP_ADDR)
    cache = ActivationReplayCache()
    counts = {"A": 0, "B": 0, "Bprime_catchup": 0}

    last = B
    generated, plen = [], 0
    for step in range(MAX_TOKENS):
        inp = prompt_ids if step == 0 else [generated[-1]]
        keep = step != 0
        n = len(inp)
        tok_payload = struct.pack(f"<{n}i", *inp)

        counts["A"] += 1
        hidden = fwd(A, tok_payload, n, True, keep, plen)          # A: tokens → hidden
        # cache the input WE relay to the last worker (metadata-prefixed)
        cache.record(SESSION, step, struct.pack("<iii", n, int(keep), plen) + hidden)

        counts["B"] += 1
        tok_bytes = fwd(last, hidden, n, False, keep, plen)        # B: hidden → token id
        tid = struct.unpack("<i", tok_bytes)[0]
        generated.append(tid)
        plen += n
        print(f"[ot] step {step+1}: id={tid}", flush=True)

        if step + 1 == FAIL_AT:
            print(f"[ot] !! simulating last-worker failure after step {step+1} → O(t) recovery", flush=True)
            # 1) admission: the replacement must be SAME drift class (PR #15)
            if not drift_compatible(PRIMARY_CLASS, ALT_CLASS):
                print("[ot] no same-drift-class alternate → would clean-restart; abort", flush=True)
                return 2
            # 2) sanity: we hold the full prefix (else a partial splice is unsafe)
            assert cache.has_full_prefix(SESSION, step), "prefix hole → unsafe splice"
            # 3) CATCH UP only B' — replay cached inputs to B for steps 0..T.
            #    The SURVIVOR A is NOT touched (counts['A'] does not increase here).
            for s_step, blob in cache.get_replay(SESSION, 0):
                n_s, keep_s, plen_s = struct.unpack("<iii", blob[:12])
                fwd(Bp, blob[12:], n_s, False, bool(keep_s), plen_s)  # build B' KV, discard token
                counts["Bprime_catchup"] += 1
            last = Bp
            print(f"[ot] B' caught up via {counts['Bprime_catchup']} replayed steps; "
                  f"survivor A untouched. resuming on A(untouched)→B'", flush=True)

    seq = " ".join(str(t) for t in generated)
    print(f"TOPTOKS_OT {seq}")
    print(f"[ot] A forwards={counts['A']} (== {MAX_TOKENS} steps, NEVER re-run on recovery) "
          f"| B forwards={counts['B']} | B' catch-up={counts['Bprime_catchup']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
