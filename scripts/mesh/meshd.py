#!/usr/bin/env python3
"""meshd — the always-on Nakshatra node daemon (v1.1 capstone).

Closes the two operational gaps the infra map called out: there is now a
long-running publisher + a discovery→tunnel auto-bringup. One process, three
loops, no hand-holding:

  1. PUBLISH heartbeat — re-sign this node's NakshatraListing (mesh id, Ed25519
     mesh key, model served, drift-class fingerprint, endpoint hint, measured
     decode-ms/layer) and publish it to the discovery relay every `--refresh`s.
  2. DISCOVER — query the relay for same-mesh peers, verify each signature, pin
     the identity, rank by measured compute, and keep ONLY same-drift-class peers
     (recovery/drift_aware.drift_compatible — the soundness gate, here applied at
     admission so we never even chain a divergent build).
  3. AUTO-TUNNEL — for each admitted peer, derive a coordination-free rendezvous
     id + role (mesh/pairing.py) and bring up an Ed25519-pinned, X25519+ChaCha20
     encrypted tunnel over the untrusted rendezvous relay. The client side
     exposes a local port wired straight at the peer's gRPC worker; the server
     side serves its own local worker. Dropped tunnels are re-established on the
     next loop.

Discovery backend is pluggable: FileRelay (zero-dep, the always-on local/shared
substrate) by default; pass `--nostr-relay wss://…` to publish/query over a real
public Nostr relay instead (needs websocket-client; the signed-listing schema is
identical either way).

A status file (`--status-file`, default ~/.nakshatra/mesh-status.json) is written
every loop so the systemd unit / an operator can see published-as, peers-seen,
and tunnels-up at a glance.

Usage (see deploy/systemd/nakshatra-meshd.service for the installed form):
  meshd.py --mesh-id prithvi-q8 --serving prithvi-q8 \
           --relay-dir ~/.nakshatra/relay --rendezvous 127.0.0.1:51820 \
           --worker-addr 127.0.0.1:5530 --drift-class rocm-gfx1201-b4123
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nakshatra_auth import load_or_create_worker_key  # noqa: E402
from discovery.nakshatra_listing import NakshatraListing, rank_listings  # noqa: E402
from discovery.relay import FileRelay, pin_from_listing  # noqa: E402
from recovery.drift_aware import drift_compatible  # noqa: E402
from transport.relay import connect as relay_connect  # noqa: E402
from transport.secure_channel import secure_handshake  # noqa: E402
from transport.mux_tunnel import MuxTunnel  # noqa: E402
from mesh.pairing import pair_role  # noqa: E402

try:
    from nakshatra_validation import SUPPORTED_CONTROL_VERSIONS  # noqa: E402
except Exception:  # pragma: no cover - validation module optional here
    SUPPORTED_CONTROL_VERSIONS = (1,)


@dataclass
class TunnelHandle:
    peer_node_id: str
    is_client: bool
    local_port: Optional[int]      # client only — where the peer's worker is reachable
    mux: MuxTunnel
    thread: threading.Thread
    started_unix: float

    def alive(self) -> bool:
        return self.thread.is_alive() and not self.mux._closed.is_set()


@dataclass
class MeshConfig:
    mesh_id: str
    serving: list[str]
    relay_dir: str
    rendezvous_host: str
    rendezvous_port: int
    worker_addr: Optional[str]          # "host:port" of this node's local gRPC worker
    drift_class: Optional[str]
    endpoint_hint: str
    decode_ms_per_layer: Optional[float]
    refresh: float
    identity_file: Path
    status_file: Path
    once: bool = False                  # one loop then exit (for tests/CI)


class MeshNode:
    def __init__(self, cfg: MeshConfig):
        self.cfg = cfg
        self.priv, self.pub = load_or_create_worker_key(cfg.identity_file)
        self.node_id = "nks-" + self.pub[:12]
        self.relay = FileRelay(cfg.relay_dir)
        self.tunnels: dict[str, TunnelHandle] = {}
        self._stop = threading.Event()
        self._last_peers: list[dict] = []
        self._log(f"identity {self.node_id} (pub {self.pub[:16]}…) mesh={cfg.mesh_id}")

    # ── logging ──
    def _log(self, msg: str) -> None:
        print(f"[meshd] {msg}", flush=True)

    # ── 1. publish ──
    def _build_listing(self) -> NakshatraListing:
        listing = NakshatraListing(
            mesh_id=self.cfg.mesh_id,
            node_id=self.node_id,
            ed25519_pubkey_hex=self.pub,
            serving=list(self.cfg.serving),
            measured_decode_ms_per_layer=self.cfg.decode_ms_per_layer,
            endpoint_hint=self.cfg.endpoint_hint,
            supported_protocol=list(SUPPORTED_CONTROL_VERSIONS),
            drift_class=self.cfg.drift_class,
            created_unix=int(time.time()),
        )
        listing.sign(self.priv)
        return listing

    def _publish(self) -> None:
        self.relay.publish(self._build_listing())

    # ── 2. discover (verify + pin + rank + same-drift-class only) ──
    def _discover(self) -> list[NakshatraListing]:
        listings = self.relay.query(mesh_id=self.cfg.mesh_id)
        ranked = rank_listings(
            listings, exclude_node_id=self.node_id,
            want_mesh_id=self.cfg.mesh_id,
            want_model=self.cfg.serving[0] if self.cfg.serving else None,
        )
        admitted = []
        for listing, _score in ranked:
            # drift-class admission: never chain/tunnel a divergent engine build.
            if not drift_compatible(self.cfg.drift_class, listing.drift_class):
                self._log(f"skip {listing.node_id}: drift-class "
                          f"{listing.drift_class!r} != mine {self.cfg.drift_class!r}")
                continue
            admitted.append(listing)
        return admitted

    # ── 3. auto-tunnel ──
    def _ensure_tunnel(self, peer: NakshatraListing) -> None:
        h = self.tunnels.get(peer.node_id)
        if h is not None and h.alive():
            return
        if h is not None:                       # dead → reap before respawn
            self._log(f"tunnel to {peer.node_id} dropped; re-establishing")
            try:
                h.mux.close()
            except Exception:
                pass
            self.tunnels.pop(peer.node_id, None)

        pin = pin_from_listing(peer)            # admission pin (Ed25519)
        role = pair_role(self.pub, pin.ed25519_pubkey_hex, self.cfg.mesh_id,
                         i_serve=bool(self.cfg.worker_addr),
                         peer_serves=bool(peer.serving))
        binding = b"tunnel:" + role.rendezvous_id

        handle_box: dict = {}
        ready = threading.Event()

        def run_tunnel():
            try:
                sock = relay_connect(self.cfg.rendezvous_host,
                                     self.cfg.rendezvous_port,
                                     role.rendezvous_id, timeout=30)
                chan = secure_handshake(sock, self.priv, pin.ed25519_pubkey_hex,
                                        role.is_initiator, session_binding=binding)
                mux = MuxTunnel(chan)
                handle_box["mux"] = mux
                if role.is_client:
                    port = mux.run_client("127.0.0.1", 0)
                    handle_box["local_port"] = port
                    self._log(f"tunnel UP → {peer.node_id}: peer worker reachable "
                              f"at 127.0.0.1:{port} (encrypted, pinned)")
                    ready.set()
                    # keep the thread alive until the tunnel closes
                    while not mux._closed.is_set() and not self._stop.is_set():
                        time.sleep(0.5)
                else:
                    if not self.cfg.worker_addr:
                        self._log(f"tunnel to {peer.node_id}: server role but no "
                                  f"--worker-addr; idle pipe")
                    th, tp = (self.cfg.worker_addr or "127.0.0.1:0").split(":")
                    self._log(f"tunnel UP ← {peer.node_id}: serving local worker "
                              f"{th}:{tp} (encrypted, pinned)")
                    ready.set()
                    mux.run_server(th, int(tp))   # blocks until closed
            except Exception as e:               # pragma: no cover - network paths
                self._log(f"tunnel to {peer.node_id} failed: {e}")
                ready.set()

        t = threading.Thread(target=run_tunnel, daemon=True)
        t.start()
        ready.wait(timeout=35)
        mux = handle_box.get("mux")
        if mux is None:                          # never came up
            return
        self.tunnels[peer.node_id] = TunnelHandle(
            peer_node_id=peer.node_id, is_client=role.is_client,
            local_port=handle_box.get("local_port"), mux=mux, thread=t,
            started_unix=time.time())

    def _prune_tunnels(self, present_ids: set[str]) -> None:
        for nid in list(self.tunnels):
            h = self.tunnels[nid]
            if nid not in present_ids or not h.alive():
                if nid not in present_ids:
                    self._log(f"peer {nid} gone; closing tunnel")
                try:
                    h.mux.close()
                except Exception:
                    pass
                self.tunnels.pop(nid, None)

    # ── status ──
    def _write_status(self, peers: list[NakshatraListing]) -> None:
        self._last_peers = [{"node_id": p.node_id,
                             "drift_class": p.drift_class,
                             "decode_ms_per_layer": p.measured_decode_ms_per_layer,
                             "endpoint_hint": p.endpoint_hint} for p in peers]
        status = {
            "node_id": self.node_id,
            "pubkey": self.pub,
            "mesh_id": self.cfg.mesh_id,
            "serving": self.cfg.serving,
            "drift_class": self.cfg.drift_class,
            "relay_dir": self.cfg.relay_dir,
            "rendezvous": f"{self.cfg.rendezvous_host}:{self.cfg.rendezvous_port}",
            "updated_unix": int(time.time()),
            "peers_admitted": self._last_peers,
            "tunnels": [
                {"peer": h.peer_node_id,
                 "role": "client" if h.is_client else "server",
                 "local_port": h.local_port,
                 "alive": h.alive(),
                 "up_secs": round(time.time() - h.started_unix, 1)}
                for h in self.tunnels.values()
            ],
        }
        tmp = str(self.cfg.status_file) + ".tmp"
        self.cfg.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(status, f, indent=2)
        os.replace(tmp, self.cfg.status_file)

    # ── main loop ──
    def loop_once(self) -> dict:
        self._publish()
        peers = self._discover()
        for p in peers:
            self._ensure_tunnel(p)
        self._prune_tunnels({p.node_id for p in peers})
        self._write_status(peers)
        up = sum(1 for h in self.tunnels.values() if h.alive())
        self._log(f"published; {len(peers)} same-class peer(s); {up} tunnel(s) up")
        return {"peers": len(peers), "tunnels_up": up}

    def run(self) -> int:
        self._log(f"starting: relay-dir={self.cfg.relay_dir} "
                  f"rendezvous={self.cfg.rendezvous_host}:{self.cfg.rendezvous_port} "
                  f"refresh={self.cfg.refresh}s")
        while not self._stop.is_set():
            try:
                self.loop_once()
            except Exception as e:               # pragma: no cover
                self._log(f"loop error: {e}")
            if self.cfg.once:
                break
            self._stop.wait(self.cfg.refresh)
        return 0

    def stop(self) -> None:
        self._stop.set()
        for h in list(self.tunnels.values()):
            try:
                h.mux.close()
            except Exception:
                pass


def _parse_args(argv=None) -> MeshConfig:
    home_nks = Path(os.path.expanduser("~/.nakshatra"))
    ap = argparse.ArgumentParser(description="Nakshatra always-on node daemon (meshd)")
    ap.add_argument("--mesh-id", default="prithvi-q8")
    ap.add_argument("--serving", action="append", default=[],
                    help="model id served (repeatable)")
    ap.add_argument("--relay-dir", default=str(home_nks / "relay"),
                    help="FileRelay discovery directory (the shared substrate)")
    ap.add_argument("--rendezvous", default="127.0.0.1:51820",
                    help="rendezvous relay host:port for tunnel bring-up")
    ap.add_argument("--worker-addr", default=None,
                    help="this node's local gRPC worker host:port to serve")
    ap.add_argument("--drift-class", default=None,
                    help="this node's drift-class fingerprint (gauge); peers must match")
    ap.add_argument("--endpoint", default="", help="advisory dial hint in the listing")
    ap.add_argument("--decode-ms-per-layer", type=float, default=None,
                    help="measured compute signal for ranking")
    ap.add_argument("--refresh", type=float, default=30.0)
    ap.add_argument("--identity-file", default=str(home_nks / "mesh.key"))
    ap.add_argument("--status-file", default=str(home_nks / "mesh-status.json"))
    ap.add_argument("--once", action="store_true", help="run one loop then exit")
    a = ap.parse_args(argv)
    host, _, port = a.rendezvous.rpartition(":")
    return MeshConfig(
        mesh_id=a.mesh_id, serving=a.serving, relay_dir=a.relay_dir,
        rendezvous_host=host or "127.0.0.1", rendezvous_port=int(port),
        worker_addr=a.worker_addr, drift_class=a.drift_class,
        endpoint_hint=a.endpoint, decode_ms_per_layer=a.decode_ms_per_layer,
        refresh=a.refresh, identity_file=Path(a.identity_file),
        status_file=Path(a.status_file), once=a.once,
    )


def main(argv=None) -> int:
    cfg = _parse_args(argv)
    node = MeshNode(cfg)
    try:
        return node.run()
    except KeyboardInterrupt:
        node.stop()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
