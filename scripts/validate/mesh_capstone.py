#!/usr/bin/env python3
"""Always-on mesh capstone — PROVE auto-wiring carries real gRPC (v1.1).

Stands up the whole always-on path on one box, no hand-holding, and shows a real
gRPC `Info` call traverse an auto-formed encrypted tunnel:

  • a rendezvous relay (untrusted byte-forwarder) on :51820;
  • a real gRPC worker (the peer's worker) on :5531;
  • meshd-B (server role, serves the worker) and meshd-A (consumer-only, client
    role) — each loads a persisted identity, PUBLISHES a signed listing to the
    shared FileRelay, DISCOVERS the other (verify+pin+rank, same-drift-class), and
    AUTO-DIALS the rendezvous relay to form an Ed25519-pinned X25519+ChaCha20
    tunnel — all from the daemon loop, no manual tunnel command.

Then we read meshd-A's status file for the local port it exposed and issue a real
gRPC `Info` to it: the response (the worker's layer range) proves the call went
client → encrypted tunnel → relay → server → worker and back.

Usage: mesh_capstone.py WORKER_GGUF DAEMON_BIN
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import grpc  # noqa: E402
import nakshatra_pb2 as pb  # noqa: E402
import nakshatra_pb2_grpc as pb_grpc  # noqa: E402
from mesh.meshd import MeshNode, MeshConfig  # noqa: E402
from transport.relay import serve as relay_serve  # noqa: E402

WORKER_GGUF, DAEMON_BIN = sys.argv[1], sys.argv[2]
TMP = Path("/tmp/nks-mesh-capstone")
RELAY_DIR = TMP / "relay"
RELAY_PORT = 51820
WORKER_PORT = 5531
MESH = "capstone-demo"
DRIFT = "demo-class-A"


def _meshd(name, worker_addr, serving):
    cfg = MeshConfig(
        mesh_id=MESH, serving=serving, relay_dir=str(RELAY_DIR),
        rendezvous_host="127.0.0.1", rendezvous_port=RELAY_PORT,
        worker_addr=worker_addr, drift_class=DRIFT, endpoint_hint="",
        decode_ms_per_layer=2.0 if worker_addr else None, refresh=2.0,
        identity_file=TMP / f"{name}.key", status_file=TMP / f"{name}-status.json",
        once=False)
    return MeshNode(cfg)


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    RELAY_DIR.mkdir(parents=True, exist_ok=True)
    for f in TMP.glob("*-status.json"):
        f.unlink()

    # 1) rendezvous relay
    threading.Thread(target=relay_serve, args=("127.0.0.1", RELAY_PORT),
                     daemon=True).start()
    time.sleep(0.5)
    print("[capstone] rendezvous relay up on :%d" % RELAY_PORT, flush=True)

    # 2) real gRPC worker B (last half, layers 8..16)
    import subprocess
    worker = subprocess.Popen(
        [str(ROOT.parent / ".venv/bin/python"), str(ROOT / "worker.py"),
         "--port", str(WORKER_PORT), "--sub-gguf", WORKER_GGUF,
         "--mode", "last", "--layer-start", "8", "--layer-end", "16",
         "--model-id", "capstone", "--daemon-bin", DAEMON_BIN,
         "--n-ctx", "256", "--n-gpu-layers", "0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # wait for the worker to answer Info
    for _ in range(60):
        try:
            s = pb_grpc.NakshatraStub(grpc.insecure_channel(f"127.0.0.1:{WORKER_PORT}"))
            info = s.Info(pb.InfoRequest(), timeout=2.0)
            print(f"[capstone] worker up: layers [{info.layer_start},{info.layer_end}) "
                  f"token_embd={info.has_token_embd} lm_head={info.has_lm_head}", flush=True)
            break
        except Exception:
            time.sleep(1.0)
    else:
        print("[capstone] FAIL: worker never came up"); worker.kill(); return 2

    # 3) two meshd daemons — B serves the worker, A is consumer-only
    nodeB = _meshd("B", f"127.0.0.1:{WORKER_PORT}", ["capstone"])
    nodeA = _meshd("A", None, [])
    tB = threading.Thread(target=nodeB.run, daemon=True); tB.start()
    tA = threading.Thread(target=nodeA.run, daemon=True); tA.start()
    print(f"[capstone] meshd-A={nodeA.node_id} (consumer)  "
          f"meshd-B={nodeB.node_id} (serves worker)", flush=True)

    # 4) wait for A's status to show an UP client tunnel with a local port
    a_status = TMP / "A-status.json"
    local_port = None
    for _ in range(40):
        time.sleep(1.0)
        if not a_status.exists():
            continue
        st = json.loads(a_status.read_text())
        ups = [t for t in st.get("tunnels", []) if t.get("alive") and t.get("local_port")]
        if ups:
            local_port = ups[0]["local_port"]
            print(f"[capstone] meshd-A auto-formed tunnel → peer {ups[0]['peer']}, "
                  f"peer worker exposed locally at 127.0.0.1:{local_port}", flush=True)
            break
    if not local_port:
        print("[capstone] FAIL: no tunnel formed"); worker.kill(); return 3

    # 5) THE PROOF: real gRPC Info through the auto-formed encrypted tunnel
    print("[capstone] issuing real gRPC Info THROUGH the tunnel "
          "(client→encrypted relay→server→worker)…", flush=True)
    ok = False
    for _ in range(10):
        try:
            stub = pb_grpc.NakshatraStub(grpc.insecure_channel(f"127.0.0.1:{local_port}"))
            info = stub.Info(pb.InfoRequest(), timeout=5.0)
            print(f"[capstone] ✅ gRPC Info via tunnel → layers "
                  f"[{info.layer_start},{info.layer_end}) "
                  f"lm_head={info.has_lm_head}  — REAL gRPC traversed the "
                  f"discovery-formed, Ed25519-pinned, ChaCha20-encrypted tunnel", flush=True)
            ok = info.layer_end == 16
            break
        except Exception as e:
            print(f"[capstone] (retry) {e}", flush=True)
            time.sleep(1.0)

    nodeA.stop(); nodeB.stop(); worker.terminate()
    try:
        worker.wait(timeout=5)
    except Exception:
        worker.kill()
    print("MESH_CAPSTONE_OK" if ok else "MESH_CAPSTONE_FAIL", flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
