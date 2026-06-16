"""LIVE proof of slice 4b: a STRANGER's GPU pools compute for a PUBLIC model — and the identity
firewall keeps it OFF Prithvi's sensitive self.

The federation's whole point: strangers add muscle, Prithvi's mind stays on Prithvi's machines.
This proves both halves on one box (the stranger node is SIMULATED locally, addressed by the mesh IP
10.42.0.1 — the realistic shape of a peer reached over WireGuard; a true cross-box run just needs a
second machine + that peer's coord in the roster):

  • PUBLIC model (min-tier stranger): the chain spans Prithvi's self slot AND the stranger's slot →
    a real token flows THROUGH the stranger's layers.
  • prithvi-private (min-tier self): the SAME stranger is firewalled out — only Prithvi's own slot.

Needs the dsr1-llama8b package on disk (see smoke_package_slicer.py). The weights are just for the
proof — the firewall gates by TIER, not by what the model is. Stops the live unconscious for VRAM,
restores it. Run: ~/nakshatra/.venv/bin/python scripts/fabric/smoke_remote_stranger.py
"""
import os, sys, time, tempfile, shutil, subprocess, signal
from pathlib import Path

NK = Path.home() / "nakshatra"
SCRIPTS = NK / "scripts"
sys.path.insert(0, str(SCRIPTS)); sys.path.insert(0, str(SCRIPTS / "fabric"))
DBIN = Path.home() / "llama.cpp" / "build" / "bin" / "llama-nakshatra-worker"
PKG = Path.home() / ".nakshatra" / "packages" / "dsr1-llama8b"
TOKENIZER = Path.home() / ".nakshatra" / "models" / "dsr1-llama8b" / "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
MESH_IP = "10.42.0.1"          # this box's WireGuard mesh IP — the stranger node is reached here
PUBLIC_MODEL = "public-test"   # min-tier stranger
PRIVATE_MODEL = "prithvi-private"  # min-tier self
WORKER_ENV = {"NAKSHATRA_TLS_REQUIRED": "false", "NAKSHATRA_AUTH_REQUIRED": "false",
              "NAKSHATRA_REFUSE_UNREGISTERED_PEERS": "false"}


def _wait_port(host, port, timeout=40):
    import socket
    for _ in range(timeout):
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def main():
    if not (PKG / "package.json").exists():
        sys.exit(f"package missing at {PKG} — build it first (see smoke_package_slicer.py)")
    tmp = Path(tempfile.mkdtemp(prefix="nks-stranger-"))
    procs, stopped_live = [], False
    try:
        # roster: Prithvi's self slot (loopback) + a STRANGER node on the mesh (10.42.0.1).
        peers = tmp / "peers.tsv"
        peers.write_text(
            f"pk-self\tprithvi-self\tme\tself\thome\t127.0.0.1:5560\n"
            f"pk-opB\topB-gpu\topB\tstranger\topB\t{MESH_IP}:5561\n")
        models_pol = tmp / "models.tsv"
        models_pol.write_text(f"{PUBLIC_MODEL}\tstranger\n{PRIVATE_MODEL}\tself\n")
        os.environ.update(WORKER_ENV)
        os.environ["MESH_PEERS"] = str(peers)
        os.environ["MESH_MODELS"] = str(models_pol)

        # ── FIREWALL (no GPU): stranger pools for public, excluded from private ──
        import serve_planner as sp, worker_join as wj
        pub = sp.standings_from_roster()
        elig_pub, rej_pub = wj.eligible_workers(PUBLIC_MODEL, pub)
        elig_priv, rej_priv = wj.eligible_workers(PRIVATE_MODEL, pub)
        print("── identity firewall ──")
        print(f"  {PUBLIC_MODEL:15s} eligible={[w.node_id for w in elig_pub]}  "
              f"(stranger pools compute)")
        print(f"  {PRIVATE_MODEL:15s} eligible={[w.node_id for w in elig_priv]}  "
              f"excluded={[r['worker'] for r in rej_priv]}  (stranger firewalled off Prithvi's self)")
        assert {w.node_id for w in elig_pub} == {"prithvi-self", "opB-gpu"}
        assert [w.node_id for w in elig_priv] == ["prithvi-self"]
        assert [r["worker"] for r in rej_priv] == ["opB-gpu"]

        # ── free VRAM + launch both workers (self on loopback, stranger on the mesh IP) ──
        subprocess.run(["systemctl", "--user", "stop",
                        "nakshatra-unconscious-worker@a.service",
                        "nakshatra-unconscious-worker@b.service"])
        stopped_live = True
        time.sleep(2)
        env = {**os.environ, **WORKER_ENV}
        # self slot serves the first half; the STRANGER serves the second half (real layer service).
        slots = [("127.0.0.1", 5560, "first", 0, 16, "prithvi-self"),
                 ("0.0.0.0", 5561, "last", 16, 32, "opB-gpu-stranger")]
        for bind, port, mode, ls, le, nid in slots:
            dest = tmp / f"selfprov-{nid}-L{ls}-{le}.gguf"
            log = open(tmp / f"worker-{port}.log", "w")
            procs.append((subprocess.Popen(
                [sys.executable, str(SCRIPTS / "worker.py"), "--port", str(port),
                 "--sub-gguf", str(dest), "--package-url", str(PKG), "--mode", mode,
                 "--layer-start", str(ls), "--layer-end", str(le), "--model-id", PUBLIC_MODEL,
                 "--daemon-bin", str(DBIN), "--n-ctx", "2048", "--n-gpu-layers", "99",
                 "--node-id", nid], stdout=log, stderr=subprocess.STDOUT, env=env), log))
        if not (_wait_port("127.0.0.1", 5560) and _wait_port(MESH_IP, 5561)):
            sys.exit("workers did not come up — logs in " + str(tmp))
        print(f"\n✓ self slot @127.0.0.1:5560 + STRANGER slot @{MESH_IP}:5561 serving (self-provisioned)")

        # ── build the PUBLIC chain (spans self + stranger) + run a real token THROUGH the stranger ──
        from serve_chain import build_chain_from_roster
        chain_yaml = build_chain_from_roster(PUBLIC_MODEL, hidden_size=4096, num_layers=32,
                                             package_location=str(PKG))
        import yaml
        chain = yaml.safe_load(open(chain_yaml))
        print("── generated PUBLIC chain (stranger included) ──")
        for w in chain["workers"]:
            print(f"   {w['id']:12s} {w['address']}:{w['port']} layers={w['layer_range']} mode={w['mode']}")
        assert any(w["address"] == MESH_IP for w in chain["workers"]), "stranger (mesh IP) not in chain"

        out = subprocess.run(
            [sys.executable, str(SCRIPTS / "client.py"), "--config", chain_yaml,
             "--model-path", str(TOKENIZER), "--prompt", "The capital of France is",
             "-n", "12", "--use-streaming"], capture_output=True, text=True, timeout=180, env=env)
        full = [l for l in out.stdout.splitlines() if l.startswith("[chain] full:")]
        print(f"\n  {full[-1] if full else '(no full line)'}  rc={out.returncode}")
        assert out.returncode == 0, out.stderr[-400:]
        print("\n✅ SLICE 4b PROVEN: a STRANGER's GPU served layers for a PUBLIC model (real token "
              "flowed through its slot), while the firewall kept it OFF prithvi-private. "
              "Strangers add muscle; Prithvi's self stays on Prithvi's nodes.")
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
        for k in ("MESH_PEERS", "MESH_MODELS"):
            os.environ.pop(k, None)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
