"""node_ready must fail closed on every unobserved fact."""
from selfcheck import REQUIRED, node_ready

READY = {"daemon_pid": 4242, "accel": "cuda", "serve_ngl": 99,
         "identity_pubkey": "a" * 64, "model_loaded": True, "answered_probe_ms": 12.5}


def test_a_fully_observed_gpu_node_is_ready():
    ok, why = node_ready(dict(READY))
    assert ok, why


def test_the_silent_cpu_bug_is_caught():
    """The failure this module exists for: daemon up, card idle, health checks green."""
    ok, why = node_ready({**READY, "serve_ngl": 0})
    assert not ok and any("GPU node serving no layers" in r for r in why), why


def test_cpu_node_claiming_offload_is_caught():
    ok, why = node_ready({**READY, "accel": "cpu", "serve_ngl": 99})
    assert not ok and any("claiming offload" in r for r in why), why


def test_each_missing_fact_is_its_own_not_observed_reason():
    for k in REQUIRED:
        facts = {x: v for x, v in READY.items() if x != k}
        ok, why = node_ready(facts)
        assert not ok, k
        assert any(r == f"{k} not observed" for r in why), (k, why)


def test_empty_facts_reports_every_required_key():
    ok, why = node_ready({})
    assert not ok
    assert sorted(r.split()[0] for r in why) == sorted(REQUIRED), why


def test_unregistered_node_is_donating_not_ready():
    ok, why = node_ready({**READY, "identity_pubkey": "A" * 64})
    assert not ok and any("never be credited" in r for r in why), why


def test_a_probe_that_never_answered():
    for bad in (0, -1, 90000, "fast", True):
        ok, why = node_ready({**READY, "answered_probe_ms": bad})
        assert not ok, bad


def test_bools_are_not_ints():
    ok, why = node_ready({**READY, "daemon_pid": True})
    assert not ok and any("positive int" in r for r in why), why


def test_non_dict_never_raises():
    for bad in (None, 42, [], "facts"):
        ok, why = node_ready(bad)
        assert not ok and why
