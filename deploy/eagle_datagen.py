#!/usr/bin/env python3
"""EAGLE-3 self-distillation data generator for Prithvi's own model.

Pulls diverse instruction prompts, generates PRITHVI's own completions via a
local llama-server (q8 gguf on ijru's 3060), and writes the ShareGPT-format
JSONL that EAGLE-3's traineagle3/main.py expects:

    {"id": <int>, "conversations": [{"from":"human","value":<prompt>},
                                    {"from":"gpt","value":<prithvi-response>}]}

The completions are Prithvi's OWN distribution (his trained voice) — that's the
whole point: EAGLE trains the draft head to mimic *this* model, which a generic
off-the-shelf draft cannot. Training (main.py) recomputes hidden states from the
f16 safetensors over these token sequences; q8-generated text is on-distribution
and fine for the data corpus.

Usage:
  python eagle_datagen.py --n 6000 --endpoint http://127.0.0.1:8080/v1 \
      --out ~/eagle-data/train.jsonl --test-out ~/eagle-data/test.jsonl \
      --test-n 200 --concurrency 4
Resumable: re-running skips ids already present in --out.
"""
import argparse, json, os, sys, threading, queue, time
import urllib.request

def load_prompts(n):
    """Diverse instruction prompts. Alpaca (general/factual) + gsm8k (reasoning)
    + a code slice, so the head sees factual/reasoning/code content classes."""
    from datasets import load_dataset
    prompts = []
    # general / factual / instruction-following
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    for ex in alpaca:
        instr = ex["instruction"].strip()
        if ex.get("input", "").strip():
            instr += "\n\n" + ex["input"].strip()
        if instr:
            prompts.append(instr)
    # reasoning
    try:
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        prompts += [ex["question"].strip() for ex in gsm.select(range(min(2000, len(gsm))))]
    except Exception as e:
        print(f"[warn] gsm8k skipped: {e}", file=sys.stderr)
    # de-dup, cap, shuffle deterministically
    seen, uniq = set(), []
    for p in prompts:
        if p not in seen:
            seen.add(p); uniq.append(p)
    import random; random.Random(42).shuffle(uniq)
    return uniq[:n]

def generate(endpoint, prompt, timeout=180):
    body = json.dumps({
        "model": "prithvi",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7, "top_p": 0.9, "max_tokens": 1024,
    }).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--test-n", type=int, default=200)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--out", default=os.path.expanduser("~/eagle-data/train.jsonl"))
    ap.add_argument("--test-out", default=os.path.expanduser("~/eagle-data/test.jsonl"))
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try: done.add(json.loads(line)["id"])
            except Exception: pass
    print(f"[datagen] target {args.n}+{args.test_n}, already done {len(done)}", flush=True)

    prompts = load_prompts(args.n + args.test_n)
    print(f"[datagen] loaded {len(prompts)} unique prompts", flush=True)

    work = queue.Queue()
    for i, p in enumerate(prompts):
        if i not in done:
            work.put((i, p))
    out_lock = threading.Lock()
    fout = open(args.out, "a"); ftest = open(args.test_out, "a")
    counters = {"ok": 0, "err": 0}

    def worker():
        while True:
            try: i, p = work.get_nowait()
            except queue.Empty: return
            try:
                resp = generate(args.endpoint, p)
                rec = {"id": i, "conversations": [
                    {"from": "human", "value": p}, {"from": "gpt", "value": resp}]}
                with out_lock:
                    f = ftest if i >= args.n else fout
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                    counters["ok"] += 1
                    if counters["ok"] % 50 == 0:
                        print(f"[datagen] {counters['ok']} ok / {counters['err']} err", flush=True)
            except Exception as e:
                counters["err"] += 1
                if counters["err"] <= 5:
                    print(f"[datagen] err id={i}: {e}", file=sys.stderr, flush=True)
            finally:
                work.task_done()

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(args.concurrency)]
    [t.start() for t in ts]; [t.join() for t in ts]
    fout.close(); ftest.close()
    print(f"[datagen] DONE ok={counters['ok']} err={counters['err']} → {args.out} (+test {args.test_out})", flush=True)

if __name__ == "__main__":
    main()
