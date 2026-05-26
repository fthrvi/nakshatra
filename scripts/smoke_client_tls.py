#!/usr/bin/env python3
"""Local-machine smoke for client.py SPKI pinning (shipped 2026-05-26).

What this exercises:

  1. Spin up a tiny TLS-enabled gRPC server on 127.0.0.1 that implements
     just the ``Info`` RPC. Uses ``nakshatra_tls.generate_self_signed_cert``
     so the SPKI hash matches what ``nakshatra-cli tls fingerprint`` would
     report on a real worker.

  2. Subprocess scripts/client.py against a temporary YAML config carrying
     the correct SPKI hash + --tls-mode=required. The chain partition is
     trivially satisfied (single first+last worker with token_embd and
     lm_head both true). Asserts the client preflight Info() loop returns
     success.

  3. Subprocess scripts/client.py again with a deliberately-wrong SPKI hash
     in the YAML. Asserts the client exits with a spki_mismatch PinError
     surfaced as the structured "[chain] TLS pin failure" message.

  4. Tear everything down.

This is a strict superset of the unit-test coverage in test_client_tls.py
because it goes through a real TLS handshake against a real server cert,
just without any SSH or cluster dependency.

Run:
  source .venv/bin/activate && python scripts/smoke_client_tls.py

Exit 0 = pass; non-zero = fail.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent import futures
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grpc
import yaml
import nakshatra_pb2 as pb
import nakshatra_pb2_grpc as pb_grpc
import nakshatra_tls as nt


# ── tiny Info-only Nakshatra servicer ──────────────────────────────────


class _InfoOnlyServicer(pb_grpc.NakshatraServicer):
    """Single-worker first+last: layer_range [0,1), has_token_embd +
    has_lm_head both true, hidden_size=8. Lets client.py's _setup_chain
    validate without needing real model artifacts. Any RPC other than
    Info raises UNIMPLEMENTED — we never get there in this smoke."""

    def Info(self, request, context):
        return pb.InfoResponse(
            protocol_version="0.1.0",
            backend="smoke-no-daemon",
            model_id="smoke-model",
            model_content_hash=b"\x00" * 32,
            layer_start=0,
            layer_end=1,
            hidden_size=8,
            wire_dtype="f32",
            kv_cache_tokens_free=256,
            has_token_embd=True,
            has_lm_head=True,
            protocol_capabilities=["streaming"],
        )


# ── helpers ─────────────────────────────────────────────────────────────


def _free_port() -> int:
    """Pick an OS-assigned free port. Bind a transient socket, read the
    chosen port, close. Tiny race window but fine for a smoke."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_tls_server(cert_path: Path, key_path: Path, port: int):
    """Bring up a TLS gRPC server on 127.0.0.1:port. Returns the server
    handle; caller is responsible for server.stop(grace=...)."""
    creds = nt.build_grpc_server_credentials(cert_path, key_path)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb_grpc.add_NakshatraServicer_to_server(_InfoOnlyServicer(), server)
    server.add_secure_port(f"127.0.0.1:{port}", creds)
    server.start()
    return server


def _wait_for_port(port: int, timeout: float = 5.0) -> None:
    """Block until ``127.0.0.1:port`` accepts TCP connections, or raise.
    gRPC's start() returns before the listener is fully ready in CI;
    sock-probe is more reliable than time.sleep()."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"127.0.0.1:{port} never accepted connections")


def _write_yaml(tmpdir: Path, port: int, spki_hash: str) -> Path:
    """Single-worker YAML — first+last chained together so client.py
    accepts it. Has the SPKI hash baked in (or a deliberately-wrong one,
    per caller)."""
    cfg = {
        "model": {
            "id": "smoke-model",
            "hidden_size": 8,
            "num_blocks": 1,
            "wire_dtype": "f32",
        },
        "workers": [
            {
                "id": "smoke-only",
                "address": "127.0.0.1",
                "port": port,
                "layer_range": [0, 1],
                "sub_gguf_path": "/dev/null",
                "mode": "first",
                "peer_spki_hash": spki_hash,
            },
        ],
    }
    path = tmpdir / "smoke.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _run_client(yaml_path: Path, tls_mode: str, model_path: Path,
                timeout_s: float = 15.0) -> subprocess.CompletedProcess:
    """Run scripts/client.py as a subprocess. --max-tokens 0 keeps the
    run to the Info() preflight — we don't want it to try inference
    against our stub servicer."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [
            sys.executable, str(HERE / "client.py"),
            "--config", str(yaml_path),
            "--tls-mode", tls_mode,
            "--model-path", str(model_path),
            "--prompt", "smoke",
            "--max-tokens", "0",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


# ── checks ──────────────────────────────────────────────────────────────


def check_pinned_success(tmpdir: Path, port: int, spki_hash: str,
                          model_path: Path) -> bool:
    """Correct SPKI hash + --tls-mode=required → client preflight Info()
    succeeds and chain validation passes.

    We can't check returncode==0 because client.py runs through
    tokenization after the preflight, and tokenize_local imports
    llama_cpp which isn't in the smoke venv. Instead we assert on the
    "chain OK" log marker — by the time it's printed, the Info RPC has
    completed over the pinned TLS channel, which is what this smoke is
    actually validating."""
    yaml_path = _write_yaml(tmpdir, port, spki_hash)
    res = _run_client(yaml_path, "required", model_path)
    combined_out = res.stdout + res.stderr
    if "smoke-only" not in res.stdout:
        print(f"  ✗ stdout missing expected worker id; got: {res.stdout!r}")
        print(f"    stderr: {res.stderr[-500:]}")
        return False
    if "[chain] OK: contiguous coverage" not in res.stdout:
        print(f"  ✗ stdout missing chain-OK marker")
        print(f"    stdout: {res.stdout[-500:]}")
        print(f"    stderr: {res.stderr[-500:]}")
        return False
    # Negative-side guard: a TLS pin failure would surface as the
    # structured error message before _setup_chain ever printed the OK
    # marker, so finding both means the order is right.
    if "TLS pin failure" in combined_out:
        print(f"  ✗ unexpected TLS pin failure in success path")
        return False
    print("  ✓ --tls-mode=required + correct hash → "
          "client preflight Info() succeeded over pinned channel")
    return True


def check_mismatch_refuses(tmpdir: Path, port: int,
                            model_path: Path) -> bool:
    """Deliberately-wrong SPKI hash → client exits with structured
    spki_mismatch message. This is the load-bearing operator signal —
    if it ever silently passes, MITM defense is gone."""
    wrong_hash = "c" * 64
    yaml_path = _write_yaml(tmpdir, port, wrong_hash)
    res = _run_client(yaml_path, "required", model_path)
    if res.returncode == 0:
        print(f"  ✗ expected non-zero rc for spki_mismatch, got 0")
        print(f"    stdout: {res.stdout[-500:]}")
        return False
    combined = res.stdout + res.stderr
    if "TLS pin failure" not in combined or "spki_mismatch" not in combined:
        print(f"  ✗ output missing expected pin-failure markers")
        print(f"    rc={res.returncode}")
        print(f"    stdout: {res.stdout[-500:]}")
        print(f"    stderr: {res.stderr[-500:]}")
        return False
    print("  ✓ --tls-mode=required + wrong hash → "
          "client exits with spki_mismatch")
    return True


def check_unpinned_required_refuses(tmpdir: Path, port: int,
                                      model_path: Path) -> bool:
    """Empty SPKI hash + --tls-mode=required → client exits with
    unpinned_peer. The other branch of the policy gate."""
    yaml_path = _write_yaml(tmpdir, port, "")  # _sanitize_spki → ""
    res = _run_client(yaml_path, "required", model_path)
    if res.returncode == 0:
        print(f"  ✗ expected non-zero rc for unpinned_peer, got 0")
        return False
    combined = res.stdout + res.stderr
    if "TLS pin failure" not in combined or "unpinned_peer" not in combined:
        print(f"  ✗ output missing unpinned_peer markers")
        print(f"    rc={res.returncode}")
        print(f"    stderr: {res.stderr[-500:]}")
        return False
    print("  ✓ --tls-mode=required + no hash → "
          "client exits with unpinned_peer")
    return True


# ── main ────────────────────────────────────────────────────────────────


def main() -> int:
    tmpdir = Path(tempfile.mkdtemp(prefix="nakshatra-smoke-client-tls-"))
    # client.py wants a real GGUF path for the tokenizer; --max-tokens 0
    # short-circuits before tokenize_local() runs, so any extant path
    # works. We use the smoke YAML path itself as a stand-in.
    fake_model = tmpdir / "tokenizer.gguf"
    fake_model.write_bytes(b"GGUF" + b"\x00" * 16)

    print(f"[smoke] working dir: {tmpdir}")
    cert_path, key_path = nt.generate_self_signed_cert(output_dir=tmpdir)
    spki_hash = nt.compute_spki_hash(cert_path)
    print(f"[smoke] generated cert; SPKI={spki_hash[:16]}…")

    port = _free_port()
    server = _start_tls_server(cert_path, key_path, port)
    print(f"[smoke] TLS gRPC server listening on 127.0.0.1:{port}")

    try:
        _wait_for_port(port)
        passed = 0
        failed = 0
        for fn, args in [
            (check_pinned_success, (tmpdir, port, spki_hash, fake_model)),
            (check_mismatch_refuses, (tmpdir, port, fake_model)),
            (check_unpinned_required_refuses, (tmpdir, port, fake_model)),
        ]:
            if fn(*args):
                passed += 1
            else:
                failed += 1
        print(f"\n[smoke] {passed}/{passed + failed} checks passed")
        return 0 if failed == 0 else 1
    finally:
        server.stop(grace=1.0).wait()
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
