"""The `--code` path — the one a real machine takes — must reach phase 2.

⚠️⚠️ WHY THIS TEST EXISTS. `test_join_end_to_end.py` passed all six phases while the
production `--code` path died at phase 1 with "missing package_url". The e2e fixture preloaded
every phase's facts; `observe('admit')` supplied only `join` and `now`. A test that passes
only with a fixture the real path never sees is a test of the fixture. Found by an external
review, not by the suite — which is the point.

This drives the REAL `observe()` with the acting layer's network call stubbed to a reachable
coordinator, and asserts admit is satisfied by what observe actually gathers.
"""
import pytest

from join import act, observe as obs
from join.phase_admit import admit
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
