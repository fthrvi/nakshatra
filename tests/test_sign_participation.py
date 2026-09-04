"""SignParticipation: the worker signs its OWN belief, and never mints a key to do it."""
import os, sys, types
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
    return s


def test_signs_its_own_span_not_the_requested_one(tmp_path, monkeypatch):
    """⚠️ THE PROPERTY THE WHOLE SCHEME RESTS ON."""
    kp = tmp_path / "worker.ed25519"; priv = os.urandom(32); kp.write_bytes(priv)
    monkeypatch.setattr(wauth, "WORKER_KEY_PATH", kp)
    s = _servicer((0, 13))
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
    resp = _servicer().SignParticipation(
        pb.SignParticipationRequest(run_id="r", output_sha256="h"), Ctx())
    assert priv.hex() not in str(resp)
