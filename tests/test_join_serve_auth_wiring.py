"""The seam the auth bug lived in: observe('serve') → servecmd → act.start_daemon.

Unit tests on servecmd alone could not see this — the coordinator sits at
facts['join']['coordinator'] and only the real observe path passes that dict through. So this
test drives observe() itself and inspects what start_daemon and Popen actually receive.
"""
import sys, pathlib
_S = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S)); sys.path.insert(0, str(_S / "join"))
from join import observe as observe_mod, act as act_mod

_FACTS = {"join": {"coordinator": "https://hub.example", "node_id": "n1"},
          "port": 8080, "role": "middle", "layer_start": 0, "layer_end": 13,
          "slice_path": "/s.gguf", "backend": "cuda", "ngl": 99,
          "python": "/py", "worker_script": "/w.py", "daemon_log": "/tmp/x.log"}


def test_serve_phase_hands_start_daemon_a_pillar_url(monkeypatch):
    seen = {}
    def fake_start(argv, log_path):
        seen["argv"] = list(argv); return {"daemon_pid": None, "daemon_start_error": "fake"}
    monkeypatch.setattr(observe_mod.act, "start_daemon", fake_start)
    observe_mod.observe("serve", dict(_FACTS))
    a = seen["argv"]
    assert "--pillar-url" in a, a
    assert a[a.index("--pillar-url") + 1] == "https://hub.example"


def test_start_daemon_env_forces_auth_on(monkeypatch, tmp_path):
    seen = {}
    class FakeProc:
        pid = 4242
    def fake_popen(argv, **kw):
        seen["env"] = kw.get("env"); seen["argv"] = argv; return FakeProc()
    monkeypatch.setattr(act_mod.subprocess, "Popen", fake_popen)
    out = act_mod.start_daemon(["/py", "/w.py"], str(tmp_path / "w.log"))
    assert out["daemon_pid"] == 4242
    env = seen["env"]
    assert env is not None, "Popen must receive an explicit env, not inherit silently"
    assert env["NAKSHATRA_AUTH_REQUIRED"] == "true"
    assert env["NAKSHATRA_REFUSE_UNREGISTERED_PEERS"] == "true"
    # and it is a COPY of the parent env, not a replacement — PATH etc. must survive
    assert "PATH" in env


def test_env_switch_alone_flips_resolve_auth_required():
    """The env var is the second lock: even with pillar_url '' it must read True."""
    from nakshatra_grpc_auth import resolve_auth_required
    assert resolve_auth_required("true", "") is True
    assert resolve_auth_required(None, "") is False          # the Mode A trap, documented
    assert resolve_auth_required(None, "https://hub") is True
