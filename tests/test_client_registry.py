"""Tests for the Sthambha-registry chain-discovery path in client.py.

This is the L2↔L3 integration's own falsifiable claim (four-project
architecture, Phase 3): a client points `--registry <pillar>` at Sthambha,
queries the live peer registry, and assembles a contiguous worker chain
WITHOUT reading a static cluster YAML. The pieces shipped long ago
(worker self-registers + heartbeats; pillar serves /peers and /chain;
client.build_chain_from_registry consumes them) but the discovery+assembly
logic was never unit-tested end to end. This pins it.

Pure stdlib: spins a fake pillar (http.server) serving /chain and /peers,
points the real client.build_chain_from_registry at it. No gRPC, no GPUs —
client.py optional-imports grpc, so the discovery helpers import cleanly
here.

Run with `pytest --noconftest tests/test_client_registry.py` — the project
conftest pulls in hivemind/pytest-asyncio fixtures these tests don't need
(mirrors test_client_tls.py).
"""
from __future__ import annotations

import contextlib
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import client as cli  # noqa: E402


# ── Builders ───────────────────────────────────────────────────────────

def _gpu(*, backend: str = "cuda", chain_status: str = "ok",
         offloaded: int = 40, model: str = "TestGPU") -> dict:
    return {
        "backend": backend, "chain_status": chain_status,
        "actual_layers_offloaded": offloaded, "model": model,
    }


def _peer(node_id: str, *, address: str, layers: tuple[int, int],
          model_id: str = "llama-test", online: bool = True,
          rpc_ms: float = 0.0, gpus: list | None = None,
          spki: str = "") -> dict:
    """A /peers projection entry, shaped exactly like server.py emits."""
    return {
        "node_id": node_id,
        "address": address,
        "is_online": online,
        "recent_rpc_ms": rpc_ms,
        "peer_spki_hash": spki,
        "layer_offerings": [
            {"model_id": model_id, "model_sha256": "",
             "layer_start": layers[0], "layer_end": layers[1]},
        ],
        "hardware": {"gpus": gpus if gpus is not None else [_gpu()]},
    }


# ── Fake pillar ────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _fake_pillar(peers: list[dict], *, chain: list | None = None):
    """Serve GET /peers (always) and GET /chain (404 unless `chain` given).

    chain=None models a pre-Phase-I pillar: /chain 404s, so the client
    falls back to its local /peers-based builder — the path we mainly
    want to pin. Pass a chain list to exercise the pillar-served path.
    """
    peers_body = json.dumps({"peers": peers, "count": len(peers)}).encode()
    chain_body = (json.dumps({"chain": chain, "warnings": []}).encode()
                  if chain is not None else None)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def do_GET(self):
            if self.path.startswith("/chain"):
                if chain_body is None:
                    self.send_error(404, "no /chain on this pillar")
                    return
                body = chain_body
            elif self.path.startswith("/peers"):
                body = peers_body
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    port = _free_port()
    httpd = HTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


# ── _peer_chain_score ──────────────────────────────────────────────────

def test_score_verified_gpu_is_best():
    p = {"hardware": {"gpus": [_gpu(backend="cuda", chain_status="ok",
                                    offloaded=40)]}}
    assert cli._peer_chain_score(p) == 0


def test_score_plain_cpu():
    assert cli._peer_chain_score({"hardware": {"gpus": []}}) == 1


def test_score_drifty_gpu_is_last_resort():
    p = {"hardware": {"gpus": [_gpu(chain_status="drifty")]}}
    assert cli._peer_chain_score(p) == 2


def test_score_declared_gpu_no_offload_is_cpu_tier():
    # backend says GPU but it offloaded 0 layers → really running on CPU.
    p = {"hardware": {"gpus": [_gpu(offloaded=0)]}}
    assert cli._peer_chain_score(p) == 1


# ── _sanitize_spki ─────────────────────────────────────────────────────

def test_sanitize_spki_valid():
    h = "ab" * 32
    assert cli._sanitize_spki(h) == h


def test_sanitize_spki_wrong_length():
    assert cli._sanitize_spki("abcd") == ""


def test_sanitize_spki_non_hex():
    assert cli._sanitize_spki("zz" * 32) == ""


# ── build_chain_from_registry: the integration claim ───────────────────

def test_contiguous_two_worker_chain():
    peers = [
        _peer("worker-last", address="10.0.0.6:5531", layers=(14, 28)),
        _peer("worker-first", address="10.0.0.5:5530", layers=(0, 14)),
    ]
    with _fake_pillar(peers) as url:
        chain = cli.build_chain_from_registry(url, "llama-test")

    assert [c["layer_start"] for c in chain] == [0, 14]
    assert [c["layer_end"] for c in chain] == [14, 28]
    assert chain[0]["address"] == "10.0.0.5"
    assert chain[0]["port"] == 5530
    assert chain[1]["address"] == "10.0.0.6"
    assert chain[1]["port"] == 5531


def test_prefers_pillar_served_chain():
    # /chain present → used verbatim; the (deliberately broken) /peers
    # list must be ignored.
    pillar_chain = [
        {"node_id": "p0", "address": "host0:5500",
         "layer_start": 0, "layer_end": 10},
        {"node_id": "p1", "address": "host1:5500",
         "layer_start": 10, "layer_end": 20},
    ]
    poison = [_peer("should-not-appear", address="bad:1", layers=(0, 20))]
    with _fake_pillar(poison, chain=pillar_chain) as url:
        chain = cli.build_chain_from_registry(url, "llama-test")

    assert [c["id"] for c in chain] == ["p0", "p1"]
    assert chain[1]["address"] == "host1"


def test_rpc_latency_tiebreak_within_tier():
    # Both offer layer 0 with equal quality (verified GPU); lower
    # recent_rpc_ms must win the slot.
    peers = [
        _peer("slow", address="10.0.0.9:5530", layers=(0, 14), rpc_ms=80.0),
        _peer("fast", address="10.0.0.8:5530", layers=(0, 14), rpc_ms=5.0),
        _peer("tail", address="10.0.0.7:5531", layers=(14, 28)),
    ]
    with _fake_pillar(peers) as url:
        chain = cli.build_chain_from_registry(url, "llama-test")

    assert chain[0]["id"] == "fast"
    assert [c["layer_start"] for c in chain] == [0, 14]


def test_offline_peers_skipped():
    peers = [
        _peer("offline-first", address="10.0.0.1:5530", layers=(0, 14),
              online=False),
        _peer("online-first", address="10.0.0.2:5530", layers=(0, 14)),
        _peer("tail", address="10.0.0.3:5531", layers=(14, 28)),
    ]
    with _fake_pillar(peers) as url:
        chain = cli.build_chain_from_registry(url, "llama-test")

    assert chain[0]["id"] == "online-first"


def test_drifty_peer_used_only_as_last_resort(capsys):
    # Only a drifty GPU covers [0,14); it must still be included so the
    # chain can form, with a stderr warning.
    peers = [
        _peer("drifty", address="10.0.0.4:5530", layers=(0, 14),
              gpus=[_gpu(chain_status="drifty")]),
        _peer("tail", address="10.0.0.5:5531", layers=(14, 28)),
    ]
    with _fake_pillar(peers) as url:
        chain = cli.build_chain_from_registry(url, "llama-test")

    assert chain[0]["id"] == "drifty"
    assert "WARNING" in capsys.readouterr().err


def test_no_layer_zero_raises():
    peers = [_peer("mid", address="10.0.0.5:5530", layers=(14, 28))]
    with _fake_pillar(peers) as url:
        with pytest.raises(RuntimeError, match="layer 0"):
            cli.build_chain_from_registry(url, "llama-test")


def test_no_peer_advertises_model_raises():
    with _fake_pillar([]) as url:
        with pytest.raises(RuntimeError, match="no online peers"):
            cli.build_chain_from_registry(url, "llama-test")


def test_malformed_peer_address_raises():
    # The guard fires on an empty port (address ends in ":"); a host:port
    # with no port is the malformed shape it's there to catch.
    peers = [_peer("first", address="badhost:", layers=(0, 14))]
    with _fake_pillar(peers) as url:
        with pytest.raises(RuntimeError, match="malformed address"):
            cli.build_chain_from_registry(url, "llama-test")
