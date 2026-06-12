#!/usr/bin/env python3
"""Prove the STANDING services auto-tunnel to a freshly joining worker peer.

Unlike mesh_capstone.py (which runs both nodes in-process), this validates the
*deployed* always-on services: the systemd `nakshatra-relay` + `nakshatra-meshd`
(a consumer-only orchestrator) are already running. We bring up a NEW worker peer
that joins the SAME shared FileRelay + rendezvous relay, then watch the standing
orchestrator's own status file show an auto-formed tunnel — and push real gRPC
through the port it exposed.

Usage: mesh_join_standing.py WORKER_GGUF DAEMON_BIN DRIFT_CLASS
       (DRIFT_CLASS must match the standing node's, or admission rejects it)
"""
from __future__ import annotations

import json
import subprocess
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

WORKER_GGUF, DAEMON_BIN, DRIFT = sys.argv[1], sys.argv[2], sys.argv[3]
NKS = Path.home() / ".nakshatra"
RELAY_DIR = NKS / "relay"
STANDING_STATUS = NKS / "mesh-status.json"
PEER_TMP = Path("/tmp/nks-join-peer")
WORKER_PORT = 5531


def main() -> int:
    PEER_TMP.mkdir(parents=True, exist_ok=True)
    # 1) a real gRPC worker (the peer's worker)
    worker = subprocess.Popen(
        [str(ROOT.parent / ".venv/bin/python"), str(ROOT / "worker.py"),
         "--port", str(WORKER_PORT), "--sub-gguf", WORKER_GGUF,
         "--mode", "last", "--layer-start", "8", "--layer-end", "16",
         "--model-id", "join", "--daemon-bin", DAEMON_BIN,
         "--n-ctx", "256", "--n-gpu-layers", "0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            pb_grpc.NakshatraStub(grpc.insecure_channel(f"127.0.0.1:{WORKER_PORT}")
                                  ).Info(pb.InfoRequest(), timeout=2.0)
            print(f"[join] worker up on :{WORKER_PORT}", flush=True); break
        except Exception:
            time.sleep(1.0)
    else:
        print("[join] FAIL: worker never came up"); worker.kill(); return 2

    # 2) a transient meshd worker-node joining the STANDING relay + rendezvous
    peer = MeshNode(MeshConfig(
        mesh_id="prithvi-q8", serving=["prithvi-q8"], relay_dir=str(RELAY_DIR),
        rendezvous_host="127.0.0.1", rendezvous_port=51820,
        worker_addr=f"127.0.0.1:{WORKER_PORT}", drift_class=DRIFT,
        endpoint_hint="", decode_ms_per_layer=2.0, refresh=2.0,
        identity_file=PEER_TMP / "peer.key",
        status_file=PEER_TMP / "peer-status.json", once=False))
    threading.Thread(target=peer.run, daemon=True).start()
    print(f"[join] worker-node {peer.node_id} joined; "
          f"watching STANDING orchestrator's status for an auto-tunnel…", flush=True)

    # 3) read the STANDING node's status for the port it exposed to this peer
    local_port = None
    for _ in range(40):
        time.sleep(1.0)
        if not STANDING_STATUS.exists():
            continue
        st = json.loads(STANDING_STATUS.read_text())
        ups = [t for t in st.get("tunnels", [])
               if t.get("alive") and t.get("role") == "client" and t.get("local_port")]
        if ups:
            local_port = ups[0]["local_port"]
            print(f"[join] STANDING orchestrator auto-tunnelled to {ups[0]['peer']}; "
                  f"peer worker exposed at 127.0.0.1:{local_port}", flush=True)
            break
    if not local_port:
        print("[join] FAIL: standing node formed no tunnel"); peer.stop(); worker.kill(); return 3

    # 4) real gRPC through the standing node's exposed port
    ok = False
    for _ in range(10):
        try:
            info = pb_grpc.NakshatraStub(grpc.insecure_channel(f"127.0.0.1:{local_port}")
                                         ).Info(pb.InfoRequest(), timeout=5.0)
            print(f"[join] ✅ gRPC via STANDING service tunnel → layers "
                  f"[{info.layer_start},{info.layer_end})  — the deployed always-on "
                  f"orchestrator carried real gRPC to a just-joined peer", flush=True)
            ok = info.layer_end == 16
            break
        except Exception as e:
            print(f"[join] (retry) {e}", flush=True); time.sleep(1.0)

    peer.stop(); worker.terminate()
    try:
        worker.wait(timeout=5)
    except Exception:
        worker.kill()
    print("MESH_JOIN_OK" if ok else "MESH_JOIN_FAIL", flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
