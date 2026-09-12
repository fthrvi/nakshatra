"""The `--code` path — the one a real machine takes — must reach phase 2.

⚠️⚠️ WHY THIS TEST EXISTS. `test_join_end_to_end.py` passed all six phases while the
production `--code` path died at phase 1 with "missing package_url". The e2e fixture preloaded
every phase's facts; `observe('admit')` supplied only `join` and `now`. A test that passes
only with a fixture the real path never sees is a test of the fixture. Found by an external
review, not by the suite — which is the point.

This drives the REAL `observe()` with the acting layer's network call stubbed to a reachable
coordinator, and asserts admit is satisfied by what observe actually gathers.
"""
import json

import pytest

from join import act, observe as obs
from join.phase_admit import admit
from join.phase_prove import prove
from join.phase_serve import serve
from joincode import encode_join


def test_observe_admit_gathers_what_admit_needs(monkeypatch):
    monkeypatch.setattr(act, "fetch_join_info", lambda coordinator, timeout=15: {
        "package_url": "https://cdn.example.com/m.gguf",
        "coordinator_version": "1.1.3",
        "node_id": "n-test",
    })
    code = encode_join("https://hub.example", "tok", expires_at=0)
    facts = {"join_code": code}
    facts.update(obs.observe("admit", facts))
    ok, why, upd = admit(facts)
    assert ok, why
    assert upd.get("admitted") is True


def test_an_unreachable_coordinator_refuses_honestly(monkeypatch):
    """Not a crash, not a silent pass — "not observed", which is the truth."""
    monkeypatch.setattr(act, "fetch_join_info",
                        lambda coordinator, timeout=15: {"join_info_error": "URLError: refused"})
    code = encode_join("https://hub.example", "tok")
    facts = {"join_code": code}
    facts.update(obs.observe("admit", facts))
    ok, why, _ = admit(facts)
    assert not ok
    assert any("package_url" in w for w in why), why


def test_the_join_code_never_reaches_the_facts_the_coordinator_sees(monkeypatch):
    seen = {}
    def fake(coordinator, timeout=15):
        seen["coordinator"] = coordinator
        return {"package_url": "https://cdn.example.com/m.gguf", "coordinator_version": "1.1.0"}
    monkeypatch.setattr(act, "fetch_join_info", fake)
    code = encode_join("https://hub.example", "tok_SECRET")
    obs.observe("admit", {"join_code": code})
    assert seen["coordinator"] == "https://hub.example"
    assert "tok_SECRET" not in seen["coordinator"]


def test_a_coordinator_that_resolves_private_is_refused_before_the_fetch():
    """The coordinator URL itself is attacker-influenced (it came from a join code)."""
    out = act.fetch_join_info("http://127.0.0.1:1")
    assert "join_info_error" in out and "non-public" in out["join_info_error"]


def test_the_real_serve_then_prove_sequence_can_actually_pass(monkeypatch):
    """⚠️⚠️ WHY THIS TEST EXISTS. Found 2026-09-07: phase_serve.serve() required
    `answered_probe_ms`, but the ONLY code that ever produces that fact is observing PROVE,
    which the real orchestrator (join/__init__.py PHASE_ORDER) runs strictly AFTER serve. So
    on every real `--code` run, observe("serve") never set that fact, and serve() refused
    every single time, unconditionally — a node could pass admit/preflight/acquire/identity
    and still never join. The green end-to-end fixture test hid this because it preloads every
    phase's facts at once instead of gathering them in the real order. This drives the REAL
    observe() dispatcher for serve then prove, in that order, with only the acting layer
    (start_daemon/poll_health/probe) stubbed, and asserts both phases actually pass — which
    they could not before this fix.
    """
    monkeypatch.setattr(act, "start_daemon", lambda argv, log_path: {"daemon_pid": 4242, "daemon_log": log_path})
    monkeypatch.setattr(act, "poll_health", lambda url, attempts, delay_fn, pid=None: [{"http_status": 200}])
    monkeypatch.setattr(obs, "serve_args", lambda pid: ["llama", "--n-gpu-layers", "12"])
    monkeypatch.setattr(act, "probe", lambda url, payload, timeout=120: {
        "probe_body": json.dumps({"tokens": [1, 2, 3], "layers": {"start": 0, "end": 12}}),
        "answered_probe_ms": 42.0,
    })

    facts = {
        "python": "python3", "worker_script": "worker.py",
        "port": 5570, "role": "first", "layer_start": 0, "layer_end": 12,
        "slice_path": "/tmp/slice.gguf", "backend": "cuda", "ngl": 12,
        "accel": "cuda", "n_layers": 12,
        "daemon_log": "/tmp/nakshatra-worker-test.log",
    }
    facts.update(obs.observe("serve", facts))
    ok, why, upd = serve(facts)
    assert ok, why  # this line is exactly what could never pass before the fix
    facts.update(upd)

    facts.update(obs.observe("prove", facts))
    ok, why, upd = prove(facts)
    assert ok, why
    facts.update(upd)

    assert facts["joined"] is True


def test_serve_no_longer_depends_on_a_fact_only_prove_can_produce():
    """serve() must be satisfiable using only what observing SERVE itself can gather — it must
    never again reach for a fact (answered_probe_ms) that only observing PROVE produces."""
    facts = {
        "daemon_pid": 1, "running_argv": ["llama", "--n-gpu-layers", "1"], "accel": "cuda",
    }
    ok, why, _ = serve(facts)
    assert ok, why
