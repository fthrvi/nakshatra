"""A4 — the latency gate. Pins behaviour the 2026-08-01 plan reported as BROKEN.

That plan said: "The selector picks the split chain at 190 ms today, when routing a whole
14B on the hub would be several times faster. `Node.rtt_ms` exists and `measure_live()`
never populates it."

Re-measured 2026-09-04: it does not reproduce — `plan()` routes whole whenever a node
fits, regardless of peer RTT, and REFUSES rather than splitting across a WAN. These tests
exist so that stays true, because the failure mode is silent: a bad split still serves
tokens, just slowly, and nothing raises.

⚠️ What these do NOT cover: `NKS_SMART_PLACEMENT` is default OFF, so the live serve path
does not call this planner at all. Correct logic behind a disabled flag is not a working
system, and that gap is a decision, not a bug.
"""
import placement as p

HUB = p.Node(name="hub", vram_gb=24.0)
IJRU = p.Node(name="ijru", vram_gb=12.0)
WAN = {("hub", "ijru"): 190.0}      # the real link after ijru moved sites
LAN = {("hub", "ijru"): 0.3}
DC = {("hub", "ijru"): 0.0498}      # CARC easley052-053, measured 2026-09-03


def test_routes_whole_even_when_a_peer_exists_far_away():
    """The original complaint. A model that fits one box must never be split."""
    plan = p.plan(model_gb=8.0, total_layers=32, nodes=[HUB, IJRU], rtt_ms=WAN)
    assert plan.whole_host == "hub"
    assert plan.splits in (None, {}), "split a model that fits one box"


def test_routes_whole_on_lan_too():
    plan = p.plan(model_gb=8.0, total_layers=32, nodes=[HUB, IJRU], rtt_ms=LAN)
    assert plan.whole_host == "hub"


def test_refuses_to_split_across_a_wan():
    """A 190 ms hop per decode step costs more than the stage it buys. Refusing is the
    correct answer — better no plan than a plan that serves tokens slowly and silently."""
    try:
        p.plan(model_gb=30.0, total_layers=32, nodes=[HUB, IJRU], rtt_ms=WAN)
    except ValueError:
        return
    raise AssertionError("planned a split across a 190 ms link")


def test_splits_on_lan_when_nothing_fits():
    plan = p.plan(model_gb=30.0, total_layers=32, nodes=[HUB, IJRU], rtt_ms=LAN)
    assert plan.splits and len(plan.splits) == 2
    assert sum(b - a for a, b in plan.splits.values()) == 32, "layers lost or duplicated"


def test_splits_at_datacentre_rtt():
    """The third regime, measured on CARC: 0.0498 ms. Well inside the 5 ms cluster
    threshold, so an H100-pair split is planned rather than refused."""
    plan = p.plan(model_gb=30.0, total_layers=32, nodes=[HUB, IJRU], rtt_ms=DC)
    assert plan.splits and len(plan.splits) == 2


def test_unknown_rtt_does_not_masquerade_as_local():
    """⚠️ The dangerous default. With no RTT data, nodes must NOT be assumed close —
    an empty matrix silently treated as 0 ms would plan WAN splits as if they were LAN."""
    try:
        p.plan(model_gb=30.0, total_layers=32, nodes=[HUB, IJRU], rtt_ms={})
    except ValueError:
        return
    raise AssertionError("clustered nodes of UNKNOWN latency — absence read as proximity")
