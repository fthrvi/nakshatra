"""RemoteSshController: an explicit `stop` command for remote workers whose ssh login shell is not bash.

A Windows/WSL node (blackwell) lands in PowerShell over ssh, where the default reap - `pkill -f '<pattern>' 2>/dev/null;
pkill -f llama-nakshatra-worker ...` - does not exist. `stop` lets the config name the command; the pkill default stays
for every existing config.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import serve_lifecycle as sl  # noqa: E402


class _Recorder(sl.RemoteSshController):
    def __init__(self, workers):
        super().__init__(workers, log=lambda *_: None)
        self.calls = []

    def _ssh(self, uh, remote_cmd, timeout=30.0):
        self.calls.append((uh, remote_cmd))
        return 0


def _w(**kw):
    base = dict(name="w", ssh="box", launch="start-it", probe=("127.0.0.1", 1), stop_match="worker.py --port 5572")
    base.update(kw)
    return sl.RemoteWorker(**base)


def test_the_default_reap_is_still_pkill_by_pattern():
    ctl = _Recorder([_w()])
    ctl.stop()
    assert ctl.calls == [("box", "pkill -f 'worker.py --port 5572' 2>/dev/null; pkill -f llama-nakshatra-worker 2>/dev/null; true")]


def test_an_explicit_stop_command_replaces_the_pkill_default():
    ctl = _Recorder([_w(stop="wsl -e bash /home/prithvi/nks-q3a.sh stop", stop_match="")])
    ctl.stop()
    assert ctl.calls == [("box", "wsl -e bash /home/prithvi/nks-q3a.sh stop")]


def test_json_loader_reads_stop_and_keeps_stop_match_optional_when_stop_is_given(tmp_path):
    cfg = tmp_path / "remote.json"
    cfg.write_text(json.dumps({"remote_workers": [
        {"name": "a", "ssh": "ijru", "launch": "l", "probe": "10.51.0.14:5572", "stop_match": "worker.py --port 5572"},
        {"name": "b", "ssh": "blackwell", "launch": "l2", "probe": "10.42.0.7:5562", "stop": "wsl -e bash x.sh stop"}]}))
    a, b = sl._remote_workers_from_json(str(cfg))
    assert (a.stop, a.stop_match) == ("", "worker.py --port 5572")
    assert (b.stop, b.stop_match, b.probe) == ("wsl -e bash x.sh stop", "", ("10.42.0.7", 5562))


def test_a_worker_with_neither_stop_nor_stop_match_is_refused_at_load_time(tmp_path):
    cfg = tmp_path / "remote.json"
    cfg.write_text(json.dumps({"remote_workers": [{"name": "c", "ssh": "h", "launch": "l", "probe": "h:1"}]}))
    with pytest.raises(ValueError, match="needs 'stop' or 'stop_match'"):
        sl._remote_workers_from_json(str(cfg))


# ── readiness: a gRPC Info() answer, not a TCP accept ─────────────────────────────────────────────────────────────────
import socket
import threading


def _accepting_socket():
    """A listener that accepts TCP and speaks no gRPC - what blackwell's Windows portproxy does before its WSL backend exists."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)

    def _loop():
        while True:
            try:
                conn, _ = srv.accept()
                conn.close()
            except OSError:
                return

    threading.Thread(target=_loop, daemon=True).start()
    return srv, srv.getsockname()[1]


def test_a_tcp_accept_alone_is_ready_by_default_as_before():
    srv, port = _accepting_socket()
    try:
        assert _Recorder([_w(probe=("127.0.0.1", port))]).is_ready() is True
    finally:
        srv.close()


def test_with_probe_grpc_a_port_that_only_accepts_tcp_is_NOT_ready():
    """The 2026-09-21 cold start: the forwarder accepted, the worker behind it was not up, and the chain was declared ready."""
    srv, port = _accepting_socket()
    try:
        assert _Recorder([_w(probe=("127.0.0.1", port), probe_grpc=True)]).is_ready() is False
    finally:
        srv.close()


def test_with_probe_grpc_a_real_grpc_server_answering_info_is_ready():
    grpc = pytest.importorskip("grpc")
    from concurrent import futures

    import nakshatra_pb2 as pb
    import nakshatra_pb2_grpc as pbg

    class _Servicer(pbg.NakshatraServicer):
        def Info(self, request, context):
            return pb.InfoResponse()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pbg.add_NakshatraServicer_to_server(_Servicer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        assert _Recorder([_w(probe=("127.0.0.1", port), probe_grpc=True)]).is_ready() is True
    finally:
        server.stop(0)


def test_the_json_loader_reads_probe_grpc(tmp_path):
    cfg = tmp_path / "remote.json"
    cfg.write_text(json.dumps({"remote_workers": [
        {"name": "a", "ssh": "h", "launch": "l", "probe": "h:1", "stop": "s", "probe_grpc": True},
        {"name": "b", "ssh": "h", "launch": "l", "probe": "h:2", "stop": "s"}]}))
    a, b = sl._remote_workers_from_json(str(cfg))
    assert (a.probe_grpc, b.probe_grpc) == (True, False)


def _tls_h2_server(tmp_path):
    """A self-signed TLS listener that negotiates h2 and then hangs up - a worker serving gRPC over TLS, as registered workers do."""
    import datetime
    import ssl

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "worker")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(1).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=1)).sign(key, hashes.SHA256()))
    (tmp_path / "c.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (tmp_path / "k.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                       serialization.NoEncryption()))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(tmp_path / "c.pem", tmp_path / "k.pem")
    ctx.set_alpn_protocols(["h2"])
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)

    def _loop():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            try:
                ctx.wrap_socket(conn, server_side=True).close()
            except Exception:
                conn.close()

    threading.Thread(target=_loop, daemon=True).start()
    return srv, srv.getsockname()[1]


def test_a_tls_listener_that_only_completes_a_handshake_is_NOT_ready(tmp_path):
    """A handshake is not readiness: a worker binds its port before it can authenticate anyone (2026-09-21 cold start), and the
    old "TLS+h2 handshake still counts" fallback re-created that bug for ANY Info failure. A real TLS worker answering Info is
    covered in test_worker_cold_start_readiness.py."""
    srv, port = _tls_h2_server(tmp_path)
    try:
        assert _Recorder([_w(probe=("127.0.0.1", port), probe_grpc=True)]).is_ready() is False
    finally:
        srv.close()
