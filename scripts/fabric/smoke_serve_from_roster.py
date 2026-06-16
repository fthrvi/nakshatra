"""LIVE end-to-end proof of slice 3: the WHOLE serve path is roster + package driven.

    HTTP POST /v1/chat/completions
       → nakshatra_serve (a from_roster model)
       → build_chain_from_roster: live roster → eligible_workers() FIREWALL → contiguous assignment
       → PackageSlicer: slices assembled from the content-addressed package
       → client.py → workers that SELF-PROVISIONED their slices (--package-url) → a real token.

Nothing static: the chain is generated from who's rostered+eligible right now, and the workers pull
their own slices from the package. A stranger in the roster is firewalled out of the (trusted) model.

Needs the dsr1-llama8b package on disk (build once — see smoke_package_slicer.py). It stops the live
unconscious workers to free VRAM, runs on 5560/5561 + a test serve on :11699, then restores the live
workers. Reversible. Run:  ~/nakshatra/.venv/bin/python scripts/fabric/smoke_serve_from_roster.py
"""
import os, sys, json, time, signal, tempfile, shutil, subprocess, urllib.request
from pathlib import Path

NK = Path.home() / "nakshatra"
VENV = NK / ".venv" / "bin" / "python"
SCRIPTS = NK / "scripts"
DBIN = Path.home() / "llama.cpp" / "build" / "bin" / "llama-nakshatra-worker"
PKG = Path.home() / ".nakshatra" / "packages" / "dsr1-llama8b"
TOKENIZER = Path.home() / ".nakshatra" / "models" / "dsr1-llama8b" / "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
MODEL = "deepseek-r1-distill-llama-8b"   # trusted in the real policy → self workers eligible, stranger out
PORT = 11699
WORKER_ENV = {"NAKSHATRA_TLS_REQUIRED": "false", "NAKSHATRA_AUTH_REQUIRED": "false",
              "NAKSHATRA_REFUSE_UNREGISTERED_PEERS": "false"}


def _wait_port(port, timeout=40):
    for _ in range(timeout):
        out = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
        if f":{port} " in out or f":{port}\n" in out:
            return True
        time.sleep(1)
    return False


def main():
    if not (PKG / "package.json").exists():
        sys.exit(f"package missing at {PKG} — build it first (see smoke_package_slicer.py)")
    tmp = Path(tempfile.mkdtemp(prefix="nks-from-roster-"))
    procs = []
    stopped_live = False
    try:
        # ── roster: 2 self slots (Prithvi's own) + a stranger; trusted model policy ──
        peers = tmp / "peers.tsv"
        peers.write_text(
            "pk-self-a\tprithvi-self-a\tme\tself\thome\t127.0.0.1:5560\n"
            "pk-self-b\tprithvi-self-b\tme\tself\thome\t127.0.0.1:5561\n"
            "pk-opB\topB-gpu\topB\tstranger\topB\t10.50.0.9:5560\n")
        models_pol = tmp / "models.tsv"
        models_pol.write_text(f"{MODEL}\ttrusted\n")

        # ── serve config: ONE from_roster model ──
        serve_cfg = tmp / "serve_models.yaml"
        serve_cfg.write_text(
            "models:\n"
            f"  - name: {MODEL}\n"
            f"    tokenizer_gguf: {TOKENIZER}\n"
            f"    from_roster: true\n"
            f"    package: {PKG}\n"
            f"    hidden_size: 4096\n"
            f"    num_layers: 32\n"
            f"    details: {{family: llama}}\n")

        # ── free VRAM: stop the live unconscious workers (on-demand; restored at the end) ──
        subprocess.run(["systemctl", "--user", "stop",
                        "nakshatra-unconscious-worker@a.service",
                        "nakshatra-unconscious-worker@b.service"])
        stopped_live = True
        time.sleep(2)

        # ── launch 2 workers that SELF-PROVISION their slices from the package ──
        env = {**os.environ, **WORKER_ENV}
        for port, mode, ls, le in [(5560, "first", 0, 16), (5561, "last", 16, 32)]:
            dest = tmp / f"selfprov-L{ls}-{le}.gguf"   # does NOT exist → worker assembles from --package-url
            log = open(tmp / f"worker-{port}.log", "w")
            p = subprocess.Popen(
                [str(VENV), str(SCRIPTS / "worker.py"), "--port", str(port),
                 "--sub-gguf", str(dest), "--package-url", str(PKG),
                 "--mode", mode, "--layer-start", str(ls), "--layer-end", str(le),
                 "--model-id", MODEL, "--daemon-bin", str(DBIN),
                 "--n-ctx", "2048", "--n-gpu-layers", "99", "--node-id", f"selfprov-{port}"],
                stdout=log, stderr=subprocess.STDOUT, env=env)
            procs.append((p, log))
        if not (_wait_port(5560) and _wait_port(5561)):
            sys.exit("self-provisioning workers did not come up — see worker logs in " + str(tmp))
        print("✓ workers self-provisioned their slices from the package and are serving (5560/5561)")

        # ── start a from_roster serve (no lifecycle env → connects to the workers above) ──
        serve_env = {**os.environ, **WORKER_ENV, "MESH_PEERS": str(peers), "MESH_MODELS": str(models_pol)}
        serve_log = open(tmp / "serve.log", "w")
        sp = subprocess.Popen([str(VENV), str(SCRIPTS / "nakshatra_serve.py"),
                               "--port", str(PORT), "--models", str(serve_cfg)],
                              stdout=serve_log, stderr=subprocess.STDOUT, env=serve_env)
        procs.append((sp, serve_log))
        if not _wait_port(PORT):
            sys.exit("serve did not come up — see " + str(tmp / "serve.log"))
        print(f"✓ from_roster serve listening on :{PORT}")

        # ── the proof: an HTTP chat request flows through the generated, firewall-gated chain ──
        body = json.dumps({"model": MODEL, "stream": False, "max_tokens": 12,
                           "messages": [{"role": "user", "content": "The capital of France is"}]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     data=body, headers={"content-type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=180) as resp:
            out = json.loads(resp.read())
        dt = time.time() - t0
        content = out["choices"][0]["message"]["content"]
        print(f"\n── reply ({dt:.1f}s) ──\n{content!r}")

        gen = Path(Path.home() / ".nakshatra" / "slices" / f"{MODEL}.from-roster.chain.yaml")
        if gen.exists():
            import yaml
            chain = yaml.safe_load(gen.read_text())
            ids = [w["id"] for w in chain["workers"]]
            print(f"\n── the GENERATED chain (firewall-gated, from the live roster) ──\n"
                  f"   workers: {ids}  (stranger 'opB-gpu' firewalled out)")
            assert "opB-gpu" not in ids, "stranger leaked into the chain!"
        # DeepSeek-R1 is a reasoning model (emits <think> first); the proof is that REAL tokens
        # flowed end-to-end through the generated, firewall-gated chain — not the answer in 12 tokens.
        assert content.strip(), f"no tokens generated through the from_roster chain, got {content!r}"
        print("\n✅ SLICE 3 PROVEN: the live serve path is roster + package driven end-to-end — "
              "HTTP → firewall → planner → self-provisioned workers → real tokens.")
    finally:
        for p, log in procs:
            try:
                p.send_signal(signal.SIGTERM); p.wait(timeout=8)
            except Exception:
                p.kill()
            log.close()
        if stopped_live:
            subprocess.run(["systemctl", "--user", "start",
                            "nakshatra-unconscious-worker@a.service",
                            "nakshatra-unconscious-worker@b.service"])
            print("✓ restored the live unconscious workers")
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
