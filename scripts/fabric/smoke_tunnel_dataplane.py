"""DE-RISK for the ijru cross-box (junction-rendezvous) path: does the nakshatra chain data-plane
(the gRPC Forward streaming) survive the meshd MuxTunnel that exposes a NAT'd peer's worker?

meshd reaches a NAT'd peer by building an encrypted tunnel over the rendezvous relay; its CLIENT side
exposes a LOCAL PORT wired straight at the peer's gRPC worker (TunnelHandle.local_port), its SERVER
side forwards each stream to its own local worker. The chain just dials that local port. The open
question: does client.py's gRPC chain traffic survive the MuxTunnel framing/multiplexing?

This proves it WITHOUT ijru/the relay: connect two MuxTunnels with a local socketpair (the byte-pipe
that in production is meshd's encrypted relay channel — transparent to the mux), put worker-B BEHIND
the tunnel (reachable ONLY via the client local port), and serve a 2-worker chain. If a token flows,
the data-plane holds over the exact tunnel meshd uses; the crypto/relay leg is a separate, live
byte-pipe (meshd's secure_channel). Needs the dsr1-llama8b package. Stops the live unconscious for
VRAM, restores it. Run: ~/nakshatra/.venv/bin/python scripts/fabric/smoke_tunnel_dataplane.py
"""
import os, sys, time, json, socket, tempfile, shutil, subprocess, signal, threading
from pathlib import Path

NK = Path.home() / "nakshatra"
SCRIPTS = NK / "scripts"
sys.path.insert(0, str(SCRIPTS))
from transport.mux_tunnel import MuxTunnel
DBIN = Path.home() / "llama.cpp" / "build" / "bin" / "llama-nakshatra-worker"
PKG = Path.home() / ".nakshatra" / "packages" / "dsr1-llama8b"
TOKENIZER = Path.home() / ".nakshatra" / "models" / "dsr1-llama8b" / "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
MID = "tunnel-dataplane-test"
WORKER_ENV = {"NAKSHATRA_TLS_REQUIRED": "false", "NAKSHATRA_AUTH_REQUIRED": "false",
              "NAKSHATRA_REFUSE_UNREGISTERED_PEERS": "false"}


def _wait_port(host, port, timeout=120):
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
    tmp = Path(tempfile.mkdtemp(prefix="nks-tunnel-"))
    procs, stopped_live, muxes = [], False, []
    try:
        subprocess.run(["systemctl", "--user", "stop",
                        "nakshatra-unconscious-worker@a.service",
                        "nakshatra-unconscious-worker@b.service"])
        stopped_live = True
        time.sleep(2)

        # two self-provisioning workers: A=first[0,16) @5560, B=last[16,32) @5561
        env = {**os.environ, **WORKER_ENV}
        for port, mode, ls, le in [(5560, "first", 0, 16), (5561, "last", 16, 32)]:
            dest = tmp / f"sp-L{ls}-{le}.gguf"
            log = open(tmp / f"w{port}.log", "w")
            procs.append((subprocess.Popen(
                [sys.executable, str(SCRIPTS / "worker.py"), "--port", str(port),
                 "--sub-gguf", str(dest), "--package-url", str(PKG), "--mode", mode,
                 "--layer-start", str(ls), "--layer-end", str(le), "--model-id", MID,
                 "--daemon-bin", str(DBIN), "--n-ctx", "2048", "--n-gpu-layers", "99",
                 "--node-id", f"w{port}"], stdout=log, stderr=subprocess.STDOUT, env=env), log))
        if not (_wait_port("127.0.0.1", 5560) and _wait_port("127.0.0.1", 5561)):
            keep_logs = Path(tempfile.mkdtemp(prefix="nks-tunnel-FAILlogs-"))
            for f in tmp.glob("w*.log"):
                shutil.copy(f, keep_logs)
            sys.exit(f"workers did not come up — logs preserved in {keep_logs}")

        # ── put worker-B (5561) BEHIND a MuxTunnel (the meshd NAT'd-peer mechanism) ──
        a, b = socket.socketpair()                       # in prod this is meshd's encrypted relay channel
        server_mux = MuxTunnel(a)                         # SERVER side: forwards each stream → worker-B
        client_mux = MuxTunnel(b)                         # CLIENT side: exposes a local port at worker-B
        muxes = [server_mux, client_mux]
        threading.Thread(target=server_mux.run_server, args=("127.0.0.1", 5561), daemon=True).start()
        tunnel_port = client_mux.run_client("127.0.0.1", 0)
        print(f"✓ worker-B (5561) is now reachable ONLY via the MuxTunnel at 127.0.0.1:{tunnel_port}")
        # sanity: the tunnel port is open
        assert _wait_port("127.0.0.1", tunnel_port, timeout=5), "tunnel local port not listening"

        # ── chain: worker-A direct, worker-B THROUGH THE TUNNEL ──
        chain = {"model": {"id": MID, "hidden_size": 4096, "num_blocks": 32, "wire_dtype": "f32"},
                 "workers": [
                     {"id": "w-A", "address": "127.0.0.1", "port": 5560,
                      "layer_range": [0, 16], "mode": "first", "sub_gguf_path": str(tmp / "sp-L0-16.gguf")},
                     {"id": "w-B-tunneled", "address": "127.0.0.1", "port": tunnel_port,
                      "layer_range": [16, 32], "mode": "last", "sub_gguf_path": str(tmp / "sp-L16-32.gguf")}]}
        import yaml
        chain_yaml = tmp / "tunnel_chain.yaml"
        chain_yaml.write_text(yaml.safe_dump(chain, sort_keys=False))

        print("── serving a chain layer THROUGH the tunnel (worker-B via the mux) ──")
        out = subprocess.run(
            [sys.executable, str(SCRIPTS / "client.py"), "--config", str(chain_yaml),
             "--model-path", str(TOKENIZER), "--prompt", "The capital of France is",
             "-n", "12", "--use-streaming"], capture_output=True, text=True, timeout=180, env=env)
        full = [l for l in out.stdout.splitlines() if l.startswith("[chain] full:")]
        print(f"  {full[-1] if full else '(no full line)'}  rc={out.returncode}")
        if out.returncode != 0:
            sys.stderr.write(out.stderr[-600:])
            sys.exit("FAIL: chain did not serve through the tunnel")
        print("\n✅ DE-RISK PASSED: the nakshatra chain gRPC data-plane SURVIVES the meshd MuxTunnel — "
              "worker-B served its layers reached only via the tunnel's local port. The junction-"
              "rendezvous path for a NAT'd peer (ijru) is data-plane-sound; crypto/relay leg is meshd's "
              "(live, transparent byte-pipe).")
    finally:
        for m in muxes:
            try:
                m.close()
            except Exception:
                pass
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
