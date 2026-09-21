"""A worker binds its gRPC port BEFORE it registers with the pillar and installs its peer-key resolver.

Measured 2026-09-21 on a scale-to-zero cold start: the lifecycle saw a completed TLS handshake, declared the chain ready, and the
first authenticated request died on "auth required but no peer resolver configured" - which Inference's blanket `except Exception`
then rewrote into `INTERNAL: Inference stream error: ` (context.abort() signals by raising a bare Exception), so the log said
"stream aborted:" and nothing else. Three fixes, three groups of tests:
  1. while STARTING the worker answers UNAVAILABLE ("worker starting"), not UNAUTHENTICATED, on Info and on every authed RPC;
  2. an abort() inside Inference keeps its own status instead of being rewritten to INTERNAL;
  3. the lifecycle probe calls a real Info() (over TLS too), so a starting worker is NOT ready.
"""
import datetime
import sys
from concurrent import futures
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

grpc = pytest.importorskip("grpc")
import nakshatra_pb2 as pb  # noqa: E402
import nakshatra_pb2_grpc as pbg  # noqa: E402
import serve_lifecycle as sl  # noqa: E402
import worker  # noqa: E402


class _Aborted(Exception):
    """grpc's abort() raises a bare Exception after recording the status."""


class _Ctx:
    def __init__(self):
        self._code = None
        self._details = None

    def invocation_metadata(self):
        return []

    def peer(self):
        return "ipv4:127.0.0.1:1"

    def set_code(self, code):
        self._code = code

    def set_details(self, details):
        self._details = details

    def code(self):
        return self._code

    def abort(self, code, details):
        self._code, self._details = code, details
        raise _Aborted()


class _Daemon:
    """Just enough DaemonClient for WorkerServicer.__init__; `boom` makes the first real call fail like a dead pipe."""
    boom = None

    def info(self):
        return {"n_embd": 4, "n_layers": 4, "gpu_offload_status": {}}

    def gpu_offload_status(self):
        return {"uses_gpu": False, "n_offloaded": 0, "total_layers": 4, "backend_hints": []}

    def acquire_session(self, *a, **k):
        return True

    def release_session(self, *a, **k):
        pass

    current_owner = None

    def call(self, *a, **k):
        raise self.boom


def _servicer(*, auth_required=True, starting=False, resolver=None):
    s = worker.WorkerServicer(daemon=_Daemon(), mode="last", layer_start=0, layer_end=14, model_id="m",
                              idem_max_entries=8, idem_ttl_seconds=10.0, peer_resolver=resolver,
                              auth_required=auth_required, refuse_unregistered_peers=False, refuse_unpinned_peers=False)
    s.starting = starting
    return s


# -- 1. STARTING is UNAVAILABLE, not UNAUTHENTICATED -------------------------------------------------------------------

def test_a_starting_worker_answers_info_unavailable():
    ctx = _Ctx()
    with pytest.raises(_Aborted):
        _servicer(starting=True).Info(pb.InfoRequest(), ctx)
    assert ctx._code == grpc.StatusCode.UNAVAILABLE and "starting" in ctx._details


def test_a_starting_worker_answers_an_authenticated_call_unavailable_not_unauthenticated():
    ctx = _Ctx()
    with pytest.raises(_Aborted):
        _servicer(starting=True)._check_grpc_auth(ctx, b"", method_path="/nakshatra.Nakshatra/Forward", is_streaming=False)
    assert ctx._code == grpc.StatusCode.UNAVAILABLE


def test_a_worker_that_is_not_starting_still_refuses_with_no_resolver():
    """Control: the gate must not turn a genuinely unconfigured worker into a retryable one."""
    ctx = _Ctx()
    with pytest.raises(_Aborted):
        _servicer(starting=False)._check_grpc_auth(ctx, b"", method_path="/nakshatra.Nakshatra/Forward", is_streaming=False)
    assert ctx._code == grpc.StatusCode.UNAUTHENTICATED


def test_info_is_unchanged_when_not_starting():
    assert _servicer(starting=False).Info(pb.InfoRequest(), _Ctx()).model_id == "m"


def test_a_worker_is_not_starting_by_default():
    """Direct construction (tests, solo, Mode A) must behave exactly as before; only main() sets `starting`."""
    assert worker.WorkerServicer(daemon=_Daemon(), mode="last", layer_start=0, layer_end=1, model_id="m").starting is False


# -- 2. an abort keeps its status ------------------------------------------------------------------------------------

def test_an_auth_abort_inside_inference_is_not_rewritten_to_internal():
    ctx = _Ctx()
    step = pb.InferenceStep(session_id="s", step_id="1")
    with pytest.raises(_Aborted):
        list(_servicer(starting=True).Inference(iter([step]), ctx))
    assert ctx._code == grpc.StatusCode.UNAVAILABLE, ctx._code


def test_a_real_failure_inside_inference_is_still_internal_and_now_names_its_type():
    class _Dead(_Daemon):
        boom = EOFError()  # str() is '' - exactly what made the original log line unreadable

    s = _servicer(auth_required=False)
    s.daemon = _Dead()
    ctx = _Ctx()
    step = pb.InferenceStep(session_id="s", step_id="1")
    step.token_ids.ids.append(1)
    list(s.Inference(iter([step]), ctx))
    assert ctx._code == grpc.StatusCode.INTERNAL
    assert "EOFError" in ctx._details


# -- 3. the probe needs a real Info ----------------------------------------------------------------------------------

def _selfsigned(tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "nakshatra.local")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(1).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("nakshatra.local")]), critical=False)
            .sign(key, hashes.SHA256()))
    return (key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()),
            cert.public_bytes(serialization.Encoding.PEM))


class _Probe(sl.RemoteSshController):
    def __init__(self, port):
        super().__init__([sl.RemoteWorker(name="w", ssh="box", launch="x", probe=("127.0.0.1", port), stop_match="x",
                                          probe_grpc=True)], log=lambda *_: None)


def _tls_worker(tmp_path, servicer):
    pytest.importorskip("cryptography")
    key, cert = _selfsigned(tmp_path)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pbg.add_NakshatraServicer_to_server(servicer, server)
    port = server.add_secure_port("127.0.0.1:0", grpc.ssl_server_credentials([(key, cert)]))
    server.start()
    return server, port


def test_a_tls_worker_that_is_still_starting_is_NOT_ready_and_becomes_ready_when_it_finishes(tmp_path):
    s = _servicer(auth_required=True, starting=True)
    server, port = _tls_worker(tmp_path, s)
    try:
        probe = _Probe(port)
        assert probe.is_ready() is False  # handshake works, Info says "starting"
        s.starting = False
        assert probe.is_ready() is True
    finally:
        server.stop(0)


# -- 4. "ready" means registered AND able to read the key table, not "registration attempted" ---------------------------

class _Resolver:
    """PillarPeerKeyResolver stand-in: refresh_once() fills the cache from `feed` (a list of dicts, one consumed per call)."""

    def __init__(self, feed=(), cache=None):
        self.cache = dict(cache or {})
        self.feed = list(feed)
        self.refreshes = 0

    def resolve(self, keyid):
        return self.cache.get(keyid)

    def refresh_once(self):
        self.refreshes += 1
        step = self.feed.pop(0) if self.feed else None
        if isinstance(step, Exception):
            raise step
        if step:
            self.cache.update(step)


def test_blocker_is_empty_only_when_registered_and_our_own_key_is_readable():
    b = worker._startup_blocker
    assert "registration" in b(False, _Resolver(cache={"me": "k"}), "me")
    assert "peer-key cache" in b(True, _Resolver(), "me")
    assert b(True, _Resolver(cache={"me": "k"}), "me") == ""
    assert b(True, None, "me") == ""  # no resolver: nothing more to prove


def test_blocker_can_demand_a_consumer_key_too():
    r = _Resolver(cache={"me": "k"})
    assert "'hub'" in worker._startup_blocker(True, r, "me", ("hub",))
    r.cache["hub"] = "k2"
    assert worker._startup_blocker(True, r, "me", ("hub",)) == ""


def _watch(servicer, register, resolver, *, registered=False, max_sleeps=50, required=()):
    sleeps = []

    def _sleep(_s):
        sleeps.append(_s)
        if len(sleeps) >= max_sleeps:
            servicer.starting = False  # test escape hatch: a permanently blocked worker would loop forever by design
    ready = worker._await_startup_ready(servicer, register, resolver, "me", required, registered=registered,
                                        poll_s=0.0, sleep=_sleep)
    return ready, len(sleeps)


def test_a_failed_registration_keeps_the_worker_starting_and_is_retried_until_it_succeeds():
    s = _servicer(starting=True)
    calls = []

    def register():
        calls.append(1)
        return len(calls) >= 3  # 401, 401, then accepted

    resolver = _Resolver(feed=[{"me": "k"}], cache={})
    ready, sleeps = _watch(s, register, resolver)
    assert ready is True and s.starting is False and len(calls) == 3
    assert s.starting_reason == ""


def test_a_worker_whose_first_key_refresh_failed_stays_starting_until_a_refresh_works():
    s = _servicer(starting=True)
    resolver = _Resolver(feed=[OSError("pillar down"), None, {"me": "k"}])
    ready, sleeps = _watch(s, lambda: True, resolver, registered=True)
    assert ready is True and resolver.refreshes == 3 and sleeps == 2


def test_a_worker_that_can_never_register_never_reports_ready():
    s = _servicer(starting=True)
    ready, sleeps = _watch(s, lambda: False, _Resolver(), max_sleeps=5)
    assert ready is False and sleeps == 5
    assert "registration" in s.starting_reason  # ...and says why


def test_info_and_auth_abort_details_carry_the_reason():
    s = _servicer(starting=True)
    s.starting_reason = "registration with the pillar has not succeeded"
    ctx = _Ctx()
    with pytest.raises(_Aborted):
        s.Info(pb.InfoRequest(), ctx)
    assert "registration with the pillar has not succeeded" in ctx._details
    ctx = _Ctx()
    with pytest.raises(_Aborted):
        s._check_grpc_auth(ctx, b"", method_path="/nakshatra.Nakshatra/Forward", is_streaming=False)
    assert "registration with the pillar has not succeeded" in ctx._details


def test_healthz_reports_starting_and_why():
    s = _servicer(starting=True)
    s.starting_reason = "peer-key cache does not hold 'me' yet"
    st = s.auth_stats()
    assert st["starting"] is True and "peer-key cache" in st["starting_reason"]
    assert _servicer(starting=False).auth_stats()["starting"] is False
