#!/usr/bin/env python3
"""smoke_gateway.py — end-to-end smoke for the Nakshatra Ollama gateway.

Starts ``nakshatra_serve.py`` as a real subprocess and exercises every
endpoint over HTTP exactly as Prithvi's ``_call_ollama`` would, asserting
Ollama-shaped responses. This is the Phase F driver.

Two modes:
  # NOW (no cluster needed): validate the gateway plumbing with a stub
  # backend — proves the Prithvi OLLAMA_HOST cutover works before the chain.
  python3 scripts/smoke_gateway.py --models scripts/serve_models.example.yaml --stub

  # PHASE F (cluster up): real chain — the model's chain_yaml workers must be
  # serving. Confirms nakshatra_serve -> client.py -> chain -> tokens back.
  python3 scripts/smoke_gateway.py --models scripts/serve_models.yaml --model llama-3.3-70b

Exit 0 = all checks passed; nonzero = first failure (printed).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _req(port, path, payload=None, stream=False, timeout=120):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    if stream:
        return r.status, [json.loads(x) for x in body.splitlines() if x.strip()]
    return getattr(r, "status", 200), json.loads(body)


def _wait_health(port, deadline=15.0):
    t0 = time.time()
    while time.time() - t0 < deadline:
        try:
            s, b = _req(port, "/health", timeout=2)
            if s == 200 and b.get("status") == "ok":
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.2)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="serve_models.yaml")
    ap.add_argument("--model", default=None, help="model name to chat (default: first in config)")
    ap.add_argument("--stub", action="store_true",
                    help="run nakshatra_serve with --stub-backend (no chain needed)")
    ap.add_argument("--port", type=int, default=11599)
    args = ap.parse_args()

    cmd = [sys.executable, str(HERE / "nakshatra_serve.py"),
           "--models", args.models, "--bind", "127.0.0.1", "--port", str(args.port)]
    if args.stub:
        cmd.append("--stub-backend")
    srv = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    checks, failed = [], None
    def check(name, ok, detail=""):
        nonlocal failed
        checks.append((name, ok, detail))
        if not ok and failed is None:
            failed = name
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    try:
        if not _wait_health(args.port):
            print("  [FAIL] server did not become healthy"); srv.terminate(); return 1
        s, ver = _req(args.port, "/api/version")
        check("/api/version namespaced", ver.get("version", "").startswith("nakshatra-"), ver.get("version"))
        s, tags = _req(args.port, "/api/tags")
        names = [m["name"] for m in tags.get("models", [])]
        check("/api/tags lists models", bool(names), f"models={names}")
        model = args.model or (names[0] if names else None)
        if not model:
            check("a model to chat", False, "no models in config"); raise SystemExit
        s, show = _req(args.port, "/api/show", {"name": model})
        check("/api/show known model 200", s == 200, model)
        try:
            _req(args.port, "/api/show", {"name": "definitely-not-a-model"})
            check("/api/show unknown -> 404", False, "got 2xx")
        except urllib.error.HTTPError as e:
            check("/api/show unknown -> 404", e.code == 404, f"code={e.code}")

        # non-streaming chat
        s, chat = _req(args.port, "/api/chat", {
            "model": model, "stream": False,
            "messages": [{"role": "user", "content": "Reply with the single word: pong."}]})
        content = chat.get("message", {}).get("content", "")
        check("/api/chat non-stream ok-shape",
              chat.get("done") is True and isinstance(content, str),
              f"content={content[:60]!r}")
        check("/api/chat non-stream produced text", len(content) > 0, f"{len(content)} chars")

        # streaming chat
        s, chunks = _req(args.port, "/api/chat", {
            "model": model, "stream": True,
            "messages": [{"role": "user", "content": "Reply with the single word: pong."}]}, stream=True)
        nonfinal = [c for c in chunks if not c.get("done")]
        final = chunks[-1] if chunks else {}
        streamed = "".join(c.get("message", {}).get("content", "") for c in chunks)
        check("/api/chat stream: chunks + terminal done",
              len(chunks) >= 1 and final.get("done") is True,
              f"{len(nonfinal)} content chunk(s) + done")
        check("/api/chat stream: produced text", len(streamed) > 0, f"{streamed[:60]!r}")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except subprocess.TimeoutExpired:
            srv.kill()

    npass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n{npass}/{len(checks)} checks passed.")
    if failed:
        print(f"FAILED at: {failed}")
        return 1
    print("gateway smoke GREEN" + ("  (stub backend — chain not exercised)" if args.stub else "  (real chain)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
