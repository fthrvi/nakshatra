#!/usr/bin/env python3
"""
bench_placement.py — A/B the placement engine: ROUTE-WHOLE vs SPLIT, measured tok/s.

The virtual-environment engine's core claim is "route, don't split": run the whole model on
one node (0 inter-node hops) instead of splitting it across a WAN link where every decode token
serializes through the hop. This harness MEASURES that — it runs client.py against two chain
configs (same prompt, N reps), parses client.py's
    [chain] generated <N> tokens in <X>s  (<Y> tok/s)
line, and reports each arm's median tok/s + the route-whole speedup. Built for the cross-box
question (hub vs hub+ijru over ~170ms) but works for ANY two chain YAMLs — local or cross-box.

It does NOT summon workers or generate chains — that's deliberate (placement/lifecycle own that).
Point it at two ready chain YAMLs (build them with serve_planner / placement_feed, or hand-write
a controlled pair) whose workers are already summoned.

Usage:
  bench_placement.py --model-path <tokenizer.gguf> \
      --route-config route_whole.yaml --split-config split.yaml \
      --prompt "Explain network latency." --reps 3 --max-tokens 128
  bench_placement.py --selftest          # parser/units check — no infra needed
"""
from __future__ import annotations
import argparse, json, re, statistics, subprocess, sys
from pathlib import Path

# matches client.py:1055 — "[chain] generated 64 tokens in 8.00s  (8.00 tok/s)"
TOKS_RE = re.compile(r"generated\s+(\d+)\s+tokens\s+in\s+([\d.]+)s\s+\(\s*([\d.]+)\s*tok/s\)")
CLIENT = str(Path(__file__).resolve().parent / "client.py")


def parse_tok_s(output: str):
    """Extract {tokens, elapsed, tok_s} from client.py output, or None if the line is absent."""
    m = TOKS_RE.search(output or "")
    if not m:
        return None
    return {"tokens": int(m.group(1)), "elapsed": float(m.group(2)), "tok_s": float(m.group(3))}


def speedup(route_tok_s, split_tok_s):
    """Route-whole speedup vs split (>1 = route-whole faster). None if either missing/zero."""
    if not route_tok_s or not split_tok_s:
        return None
    return round(route_tok_s / split_tok_s, 3)


def run_arm(name, config_path, *, model_path, prompt, max_tokens, timeout):
    """One client.py run → parsed result or {'error': ...}. Never raises."""
    cmd = [sys.executable, CLIENT, "--config", config_path, "--model-path", model_path,
           "--prompt", prompt, "--max-tokens", str(max_tokens)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": f"launch failed: {e!r}"}
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    r = parse_tok_s(out)
    if r is None:
        return {"error": f"no tok/s line (client.py exit {p.returncode})", "tail": out[-400:]}
    return r


def bench(*, route_config, split_config, model_path, prompt, reps, max_tokens, timeout):
    arms = {}
    for name, cfg in (("route_whole", route_config), ("split", split_config)):
        runs = []
        for i in range(reps):
            r = run_arm(name, cfg, model_path=model_path, prompt=prompt,
                        max_tokens=max_tokens, timeout=timeout)
            print(f"  {name} rep{i + 1}/{reps}: {r}", flush=True)
            if "tok_s" in r:
                runs.append(r["tok_s"])
        arms[name] = {"runs": runs,
                      "median_tok_s": round(statistics.median(runs), 3) if runs else None,
                      "ok": len(runs), "attempts": reps}
    rw, sp = arms["route_whole"]["median_tok_s"], arms["split"]["median_tok_s"]
    return {"arms": arms, "route_whole_speedup": speedup(rw, sp),
            "prompt": prompt, "reps": reps, "max_tokens": max_tokens}


def main():
    ap = argparse.ArgumentParser(description="placement A/B: route-whole vs split tok/s")
    ap.add_argument("--route-config", help="chain YAML for route-whole (1 solo worker)")
    ap.add_argument("--split-config", help="chain YAML for the split (>=2 workers)")
    ap.add_argument("--model-path", help="tokenizer GGUF (client.py --model-path)")
    ap.add_argument("--prompt", default="Explain network latency in three sentences.")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--out", help="write the JSON result here too")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        assert parse_tok_s("[chain] generated 64 tokens in 8.00s  (8.00 tok/s)") == \
            {"tokens": 64, "elapsed": 8.0, "tok_s": 8.0}
        assert parse_tok_s("no such line") is None
        assert speedup(40.0, 5.0) == 8.0 and speedup(0, 5) is None
        print("selftest OK")
        return
    if not (a.route_config and a.split_config and a.model_path):
        ap.error("need --route-config, --split-config and --model-path (or --selftest)")

    res = bench(route_config=a.route_config, split_config=a.split_config, model_path=a.model_path,
                prompt=a.prompt, reps=a.reps, max_tokens=a.max_tokens, timeout=a.timeout)
    print(json.dumps(res, indent=2))
    rw = res["arms"]["route_whole"]["median_tok_s"]
    sp = res["arms"]["split"]["median_tok_s"]
    if res["route_whole_speedup"]:
        print(f"\nROUTE-WHOLE {rw} tok/s  vs  SPLIT {sp} tok/s  "
              f"→ route-whole is {res['route_whole_speedup']}x")
    else:
        print("\n(one arm produced no tok/s — see per-rep errors above)")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
