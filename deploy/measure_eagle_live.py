"""measure_eagle_live.py — live tok/s: plain greedy vs EAGLE speculative, over the
worker daemon's stdio protocol, on a WHOLE-model daemon (one process, all layers +
embd + lm_head). Tests the REAL wired mechanism: cmd=5 (scratch-seq hidden) ->
EagleDraft head -> cmd=1 all_logits verify (seq 0, M3-fused KV trim) -> accept.

Output identical to plain greedy by construction (verify corrects every reject);
the speedup is wall-clock tok/s. Reports plain tok/s, EAGLE tok/s, mean accepted/step.

Run on ijru (conda eagle env: torch+CUDA). Usage:
  ~/miniconda3/envs/eagle/bin/python measure_eagle_live.py <whole_gguf> <head.pt> <base> <eagle_cfg> [prompt]
"""
import os, sys, time, struct, subprocess, threading, glob

# APPEND (not insert-0) the repo dirs: ~/nakshatra/scripts has a packaging/ dir
# that would shadow the real `packaging` (transformers dep) if it preceded
# site-packages. Append → site-packages wins for stdlib/third-party modules.
for _p in ("~/nakshatra/scripts", "~/nakshatra/deploy", "~/EAGLE", "~"):
    _ap = os.path.expanduser(_p)
    if _ap not in sys.path:
        sys.path.append(_ap)

GGUF = sys.argv[1]
HEAD = sys.argv[2]
BASE = sys.argv[3]
CFG = sys.argv[4]
PROMPT = sys.argv[5] if len(sys.argv) > 5 else "Explain why the sky is blue in one paragraph."
K = int(os.environ.get("SPEC_K", "4"))
MAX_NEW = int(os.environ.get("MAX_NEW", "48"))
BIN = os.path.expanduser("~/llama.cpp/build-cuda/bin/llama-nakshatra-worker")
N_CTX = 512
# On a single 12GB card the q8 target (~8GB) + the EAGLE head (~3.5GB w/ embedding
# + CUDA ctx) don't both fit. Offload a few target layers to CPU to make room — the
# SAME daemon serves plain and EAGLE, so the speedup RATIO stays fair (only absolute
# tok/s drops). NGL=99 = all-GPU (needs a 2nd card / smaller target).
NGL = os.environ.get("NGL", "28")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from speculative import accept, kv_keep_after
from eagle_speculative import EagleDraft, daemon_cmd5_fn

# tokenizer via llama_cpp if available, else a trivial fallback
def get_tokenizer():
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=GGUF, n_ctx=N_CTX, n_gpu_layers=0, vocab_only=True, verbose=False)
        return (lambda s: llm.tokenize(s.encode())), llm
    except Exception as e:
        print(f"[warn] llama_cpp tokenizer unavailable ({e!r}); using HF tokenizer", flush=True)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE)
        return (lambda s: tok(s)["input_ids"]), None

tokenize, _llm = get_tokenizer()
prompt_ids = list(tokenize(PROMPT))
print(f"prompt='{PROMPT}' ({len(prompt_ids)} tokens)  K={K}  max_new={MAX_NEW}", flush=True)

# ── launch whole-model daemon ────────────────────────────────────────────────
# "last" mode → the worker emits TOKENS (lm_head argmax) for cmd=1, and argmax-
# per-position for all_logits verify. cmd=1 still accepts token input (mode only
# gates OUTPUT), and cmd=5 returns its hidden3 early (before the mode logic), so a
# single "last"-mode whole-model worker is the complete loop: tokens in → tokens
# out, plus draft hidden. (A "first"-mode worker would return hidden states, which
# the decode-feedback loop would misread as a token — the bug this fixes.)
p = subprocess.Popen([BIN, GGUF, "last", str(N_CTX), "0", NGL],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
ready = threading.Event()
def watch():
    for line in p.stderr:
        s = line.decode("utf-8", "replace").rstrip()
        if "ready" in s: print("  [daemon]", s, flush=True); ready.set()
        elif any(k in s for k in ("error", "fail", "Fail")): print("  [daemon]", s, flush=True)
threading.Thread(target=watch, daemon=True).start()
if not ready.wait(180): sys.exit("daemon never ready")
time.sleep(0.5)

_lock = threading.Lock()
def send_recv(cmd, n_tokens, start_pos, flags, payload=b""):
    with _lock:
        hdr = struct.pack("<IIIII", cmd, n_tokens, start_pos, flags, len(payload))
        p.stdin.write(hdr + payload); p.stdin.flush()
        status = struct.unpack("<I", p.stdout.read(4))[0]
        plen = struct.unpack("<I", p.stdout.read(4))[0]
        data = p.stdout.read(plen) if plen else b""
        return status, data

KEEP, ALL = 0x1, 0x2

def decode_argmax(tokens, start_pos, keep):
    """cmd=1 on seq 0; returns the last-position argmax (plain) — result_type+token."""
    n = len(tokens)
    st, d = send_recv(1, n, start_pos, (KEEP if keep else 0), struct.pack(f"<{n}i", *tokens))
    assert st == 0, f"decode failed st={st}"
    # result_type prefix (0=hidden,1=token); last worker → token id
    return struct.unpack("<i", d[4:8])[0]

def verify_argmax(tokens, start_pos):
    """cmd=1 all_logits on seq 0; returns argmax at EVERY position (the verify)."""
    n = len(tokens)
    st, d = send_recv(1, n, start_pos, KEEP | ALL, struct.pack(f"<{n}i", *tokens))
    assert st == 0, f"verify failed st={st}"
    return list(struct.unpack(f"<{n}i", d[4:4 + n * 4]))

# ── PLAIN greedy baseline (seq 0) ────────────────────────────────────────────
def run_plain():
    send_recv(4, 0, 0, 0, struct.pack("<I", 0))  # clear via truncate-to-0 (cmd=4)
    # cold prefill
    nxt = decode_argmax(prompt_ids, 0, keep=False)
    gen = [nxt]; pos = len(prompt_ids)
    t0 = time.time()
    for _ in range(MAX_NEW - 1):
        nxt = decode_argmax([gen[-1]], pos, keep=True)
        pos += 1; gen.append(nxt)
    dt = time.time() - t0
    return gen, (MAX_NEW - 1) / dt, dt

# ── EAGLE speculative (seq 0 verify + seq 1 draft hidden) ────────────────────
def run_eagle(draft):
    send_recv(4, 0, 0, 0, struct.pack("<I", 0))
    nxt = decode_argmax(prompt_ids, 0, keep=False)
    # prefill established KV for the prompt (pos 0..P-1); next decode starts at P.
    # `nxt` is the prefill's output token (no KV yet) — fed as `cur` next step.
    gen = [nxt]; prefix_length = len(prompt_ids)
    accepted_total = 0; steps = 0
    t0 = time.time()
    while len(gen) < MAX_NEW:
        cur = gen[-1]
        drafts = draft.propose(prompt_ids + gen, K)      # cmd=5 on scratch seq 1
        verify = [cur] + list(drafts)
        targ = verify_argmax(verify, prefix_length)       # seq 0, M3-fused trim
        res = accept(drafts, targ)
        prefix_length = kv_keep_after(prefix_length, res.n_accepted)
        accepted_total += res.n_accepted; steps += 1
        for t in res.committed:
            gen.append(t)
            if len(gen) >= MAX_NEW: break
    dt = time.time() - t0
    return gen, (len(gen) - 1) / dt, dt, accepted_total / max(steps, 1)

print("\n=== PLAIN greedy ===", flush=True)
plain_gen, plain_tps, plain_dt = run_plain()
print(f"  {MAX_NEW-1} tokens in {plain_dt:.2f}s = {plain_tps:.2f} tok/s", flush=True)

print("\n=== loading EAGLE head ===", flush=True)
draft = EagleDraft(HEAD, BASE, CFG, daemon_cmd5_fn(send_recv))
print("  head loaded; running EAGLE speculative…", flush=True)
eagle_gen, eagle_tps, eagle_dt, mean_acc = run_eagle(draft)
print(f"  {len(eagle_gen)-1} tokens in {eagle_dt:.2f}s = {eagle_tps:.2f} tok/s  "
      f"(mean accepted/step={mean_acc:.2f} of K={K})", flush=True)

match = plain_gen[:min(len(plain_gen), len(eagle_gen))] == eagle_gen[:min(len(plain_gen), len(eagle_gen))]
print(f"\n=== RESULT ===", flush=True)
print(f"  plain  : {plain_tps:.2f} tok/s", flush=True)
print(f"  EAGLE  : {eagle_tps:.2f} tok/s", flush=True)
print(f"  speedup: {eagle_tps/plain_tps:.2f}x   mean_accept={mean_acc:.2f}   "
      f"output_identical={match}", flush=True)
p.terminate()
