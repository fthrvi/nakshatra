"""SignParticipation: the worker signs its OWN belief, and never mints a key to do it."""
import collections
import os, sys, threading, types
from pathlib import Path
sys.path.insert(0, "scripts")

import pytest

grpc = pytest.importorskip("grpc", reason="grpc not installed; run under the worker venv")
pb = pytest.importorskip("nakshatra_pb2")
import nakshatra_auth as wauth  # noqa: E402
from identity_binding import pub_of, verify_participation, participation_message


class Ctx:
    def __init__(self): self.code = None; self.details = None
    def set_code(self, c): self.code = c
    def set_details(self, d): self.details = d


def _servicer(monkey_layers=(0, 13)):
    import worker
    s = object.__new__(worker.WorkerServicer)
    s.layer_start, s.layer_end = monkey_layers
    s._check_grpc_auth = lambda *a, **k: None
    # 2026-09-12 participation binding: SignParticipation now consults the evidence a real
    # Inference/Forward call would have populated. A hand-built servicer never runs
    # WorkerServicer.__init__, so these start empty exactly like a freshly-started worker
    # that has served nothing yet — tests that need "this node actually served the run"
    # must call s._record_session_activity(...) (or set s._last_activity_ts) themselves.
    s._participation_lock = threading.Lock()
    s._served_sessions = collections.OrderedDict()
    s._served_ttl = 900.0
    s._served_max_sessions = 2048
    s._last_activity_ts = 0.0
    return s


def test_signs_its_own_span_not_the_requested_one(tmp_path, monkeypatch):
    """⚠️ THE PROPERTY THE WHOLE SCHEME RESTS ON."""
    kp = tmp_path / "worker.ed25519"; priv = os.urandom(32); kp.write_bytes(priv)
    monkeypatch.setattr(wauth, "WORKER_KEY_PATH", kp)
    s = _servicer((0, 13))
    s._record_session_activity("r")   # this worker genuinely served run "r"
    # the coordinator ASKS for [0,32) — the worker served [0,13)
    req = pb.SignParticipationRequest(run_id="r", output_sha256="h", layer_start=0, layer_end=32)
    resp = s.SignParticipation(req, Ctx())
    assert (resp.layer_start, resp.layer_end) == (0, 13), "echoed the request instead of its own span"
    entry = {"node_id": resp.node_id, "pubkey": resp.pubkey,
             "layer_start": resp.layer_start, "layer_end": resp.layer_end, "sig": resp.sig}
    ok, why = verify_participation(entry, run_id="r", output_sha256="h",
                                   pinned={resp.node_id: pub_of(priv.hex())})
    assert ok, why


def test_signature_is_over_the_frozen_canonical(tmp_path, monkeypatch):
    """The wire format must not have moved."""
    kp = tmp_path / "k"; priv = os.urandom(32); kp.write_bytes(priv)
    monkeypatch.setattr(wauth, "WORKER_KEY_PATH", kp)
    s = _servicer((5, 9))
    s._record_session_activity("run-2")   # this worker genuinely served run "run-2"
    resp = s.SignParticipation(pb.SignParticipationRequest(
        run_id="run-2", output_sha256="deadbeef", layer_start=5, layer_end=9), Ctx())
    assert participation_message("run-2", resp.node_id, 5, 9, "deadbeef") == \
        b"run-2|" + resp.node_id.encode() + b"|[5,9)|deadbeef"


def test_a_missing_key_refuses_and_does_not_create_one(tmp_path, monkeypatch):
    """⚠️ load_or_create would MINT an identity here — orphaning every credit earned under
    the old key, silently, inside a signing call."""
    kp = tmp_path / "absent" / "worker.ed25519"
    monkeypatch.setattr(wauth, "WORKER_KEY_PATH", kp)
    ctx = Ctx()
    resp = _servicer().SignParticipation(
        pb.SignParticipationRequest(run_id="r", output_sha256="h"), ctx)
    assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION
    assert not kp.exists(), "it created a key file"
    assert resp.sig == ""


def test_a_malformed_key_refuses_rather_than_signing_with_it(tmp_path, monkeypatch):
    kp = tmp_path / "k"; kp.write_bytes(b"short")
    monkeypatch.setattr(wauth, "WORKER_KEY_PATH", kp)
    ctx = Ctx()
    _servicer().SignParticipation(pb.SignParticipationRequest(run_id="r", output_sha256="h"), ctx)
    assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION


def test_missing_run_id_or_output_hash_is_refused(tmp_path, monkeypatch):
    kp = tmp_path / "k"; kp.write_bytes(os.urandom(32))
    monkeypatch.setattr(wauth, "WORKER_KEY_PATH", kp)
    for req in (pb.SignParticipationRequest(output_sha256="h"),
                pb.SignParticipationRequest(run_id="r")):
        ctx = Ctx()
        _servicer().SignParticipation(req, ctx)
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT


def test_the_private_key_never_appears_in_the_response(tmp_path, monkeypatch):
    kp = tmp_path / "k"; priv = os.urandom(32); kp.write_bytes(priv)
    monkeypatch.setattr(wauth, "WORKER_KEY_PATH", kp)
    s = _servicer()
    s._record_session_activity("r")
    resp = s.SignParticipation(
        pb.SignParticipationRequest(run_id="r", output_sha256="h"), Ctx())
    assert priv.hex() not in str(resp)


# ── 2026-09-12 participation binding: refuse work never seen ─────────────────────────────

def test_an_unserved_run_id_from_a_cold_node_is_refused(tmp_path, monkeypatch):
    """⚠️⚠️ THE FINDING THIS FILE EXISTS TO CLOSE. Before this fix, ANY authenticated peer
    could call SignParticipation(run_id="anything", output_sha256="anything") on a node that
    had done NOTHING for that run — or anything at all — and receive a validly-signed fake
    participation proof. A freshly-started servicer that has never recorded any activity for
    ANY run must be refused, not signed."""
    kp = tmp_path / "k"; kp.write_bytes(os.urandom(32))
    monkeypatch.setattr(wauth, "WORKER_KEY_PATH", kp)
    ctx = Ctx()
    resp = _servicer().SignParticipation(
        pb.SignParticipationRequest(run_id="never-served-this", output_sha256="whatever"), ctx)
    assert ctx.code == grpc.StatusCode.PERMISSION_DENIED
    assert resp.sig == ""


def test_an_unserved_run_id_is_refused_even_with_recent_unrelated_activity(tmp_path, monkeypatch):
    """Recent Forward activity is only a fallback for the SESSIONLESS path — it must not
    launder an arbitrary run_id that this worker's real Inference sessions never produced."""
    kp = tmp_path / "k"; kp.write_bytes(os.urandom(32))
    monkeypatch.setattr(wauth, "WORKER_KEY_PATH", kp)
    s = _servicer()
    s._record_session_activity("run-that-really-happened")   # a DIFFERENT, real run
    ctx = Ctx()
    resp = s.SignParticipation(
        pb.SignParticipationRequest(run_id="run-that-never-happened", output_sha256="h"), ctx)
    assert ctx.code == grpc.StatusCode.PERMISSION_DENIED
    assert resp.sig == ""


def test_a_forged_output_hash_for_a_real_run_is_refused_when_this_worker_produced_the_tokens():
    """This worker WAS mode="last" for run "r" and actually emitted tokens [1, 2, 3] — the
    real output_sha256 is `receipt.output_sha256([1,2,3])`. A caller claiming a DIFFERENT hash
    for the same run_id must be refused: the worker can check this one itself, and does."""
    from receipt import output_sha256 as real_output_sha256
    s = _servicer()
    s._record_session_activity("r", token_id=1)
    s._record_session_activity("r", token_id=2)
    s._record_session_activity("r", token_id=3)
    real_hash = real_output_sha256([1, 2, 3])
    ctx = Ctx()
    resp = s.SignParticipation(
        pb.SignParticipationRequest(run_id="r", output_sha256="not-" + real_hash), ctx)
    assert ctx.code == grpc.StatusCode.PERMISSION_DENIED
    assert resp.sig == ""


def test_the_real_output_hash_for_a_run_this_worker_finished_is_signed(tmp_path, monkeypatch):
    """Regression: the legitimate case (a).  A worker that genuinely produced the final
    tokens, presented with the MATCHING output_sha256, still signs — not because it trusts
    the caller, but because it independently reproduced the same hash."""
    from receipt import output_sha256 as real_output_sha256
    kp = tmp_path / "k"; priv = os.urandom(32); kp.write_bytes(priv)
    monkeypatch.setattr(wauth, "WORKER_KEY_PATH", kp)
    s = _servicer((13, 26))
    s._record_session_activity("r", token_id=42)
    s._record_session_activity("r", token_id=43)
    real_hash = real_output_sha256([42, 43])
    ctx = Ctx()
    resp = s.SignParticipation(
        pb.SignParticipationRequest(run_id="r", output_sha256=real_hash,
                                    layer_start=13, layer_end=26), ctx)
    assert ctx.code is None, ctx.details
    assert resp.sig != ""
    assert (resp.layer_start, resp.layer_end) == (13, 26)
