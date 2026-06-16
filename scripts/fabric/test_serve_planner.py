"""serve_planner — the last mile, tested without GPU or a live control plane.

Proves: the IDENTITY FIREWALL gates the chain at every tier (a stranger never gets Prithvi's
sensitive layers); eligible workers get a CONTIGUOUS, contract-valid assignment; and the planner
refuses to serve rather than drop to a lower tier (default-deny)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import worker_join as wj
import serve_planner as sp

RANK = {"stranger": 0, "known": 1, "trusted": 2, "self": 3}
# the real policy shape: prithvi-private is self-only; the unconscious reasoner is trusted; public is open.
MODEL_MIN = {"prithvi-private": "self", "deepseek-r1-distill-llama-8b": "trusted",
             "public-llama-70b": "stranger"}
MT = lambda m: MODEL_MIN.get(m, "self")  # unlisted -> strictest (fail-safe), like admission.py


def _w(node, tier, addr="127.0.0.1", port=5540):
    return wj.WorkerStanding(node, node + "-key", True, tier, "op",
                             {"address": addr, "port": port})


# ── the built-in contiguous partition honors the chain contract ─────────────────────────────────────
def test_contiguous_split_covers_and_tags():
    r = sp.contiguous_split(32, 2)
    assert r == [(0, 16, "first"), (16, 32, "last")]
    # full coverage, no gap, no overlap
    assert r[0][0] == 0 and r[-1][1] == 32
    for (s, e, _), (ns, _ne, _) in zip(r, r[1:]):
        assert e == ns


def test_contiguous_split_remainder_and_middle():
    r = sp.contiguous_split(33, 3)          # 33 / 3 -> 11,11,11
    assert r == [(0, 11, "first"), (11, 22, "middle"), (22, 33, "last")]
    r2 = sp.contiguous_split(32, 3)         # remainder to the front: 11,11,10
    assert [e - s for s, e, _ in r2] == [11, 11, 10]
    assert r2[-1][1] == 32 and r2[1][2] == "middle"


def test_solo_worker_tagged():
    assert sp.contiguous_split(32, 1) == [(0, 32, "solo")]


def test_split_rejects_impossible():
    for bad in (lambda: sp.contiguous_split(0, 1), lambda: sp.contiguous_split(8, 0),
                lambda: sp.contiguous_split(2, 3)):
        try:
            bad(); assert False, "expected ValueError"
        except ValueError:
            pass


# ── THE IDENTITY FIREWALL, applied to a real chain plan ─────────────────────────────────────────────
def test_self_only_model_excludes_friend_and_stranger():
    # Prithvi's own sensitive model: ONLY his self node may serve it.
    pool = [_w("prithvi-self", "self", port=5540), _w("friend", "trusted", port=5550),
            _w("stranger", "stranger", port=5560)]
    plan = sp.plan_chain("prithvi-private", pool, num_layers=32, hidden_size=4096,
                         min_tier_fn=MT, rank=RANK)
    assert plan.eligible == ["prithvi-self"]
    assert {r["worker"] for r in plan.rejected} == {"friend", "stranger"}
    # the chain is well-formed for a solo self node
    assert plan.chain["workers"][0]["id"] == "prithvi-self"
    assert plan.chain["workers"][0]["layer_range"] == [0, 32]
    assert plan.chain["model"]["num_blocks"] == 32


def test_trusted_model_excludes_only_stranger():
    # the live unconscious reasoner (trusted): self + trusted friend may pool; stranger excluded.
    pool = [_w("prithvi-self", "self", port=5540), _w("friend", "trusted", port=5541),
            _w("stranger", "stranger", port=5542)]
    plan = sp.plan_chain("deepseek-r1-distill-llama-8b", pool, num_layers=32, hidden_size=4096,
                         min_tier_fn=MT, rank=RANK)
    assert set(plan.eligible) == {"prithvi-self", "friend"}
    assert [r["worker"] for r in plan.rejected] == ["stranger"]
    # two eligible -> contiguous first/last split that covers [0,32)
    ranges = [tuple(w["layer_range"]) + (w["mode"],) for w in plan.chain["workers"]]
    assert ranges == [(0, 16, "first"), (16, 32, "last")]


def test_public_model_lets_stranger_pool():
    pool = [_w("prithvi-self", "self"), _w("stranger", "stranger")]
    plan = sp.plan_chain("public-llama-70b", pool, num_layers=80, hidden_size=8192,
                         min_tier_fn=MT, rank=RANK)
    assert set(plan.eligible) == {"prithvi-self", "stranger"}
    assert plan.rejected == []


def test_default_deny_refuses_to_serve_when_no_one_eligible():
    # only a stranger online, asked for a self-only model -> refuse (don't drop tier).
    pool = [_w("stranger", "stranger")]
    try:
        sp.plan_chain("prithvi-private", pool, num_layers=32, hidden_size=4096,
                      min_tier_fn=MT, rank=RANK)
        assert False, "expected PermissionError (firewall)"
    except PermissionError as e:
        assert "identity firewall" in str(e)


def test_worker_endpoint_flows_into_yaml():
    pool = [_w("prithvi-self", "self", addr="10.42.0.1", port=5540)]
    plan = sp.plan_chain("prithvi-private", pool, num_layers=16, hidden_size=4096,
                         min_tier_fn=MT, rank=RANK)
    w = plan.chain["workers"][0]
    assert w["address"] == "10.42.0.1" and w["port"] == 5540


# ── the roster bridge: peers.tsv shape -> standings ─────────────────────────────────────────────────
def test_standings_from_roster():
    roster = {
        "pk-self": {"pubkey": "pk-self", "name": "prithvi-self", "operator": "me",
                    "tier": "self", "tenant": "home", "coord": "127.0.0.1:5540"},
        "pk-strange": {"pubkey": "pk-strange", "name": "opB-gpu", "operator": "opB",
                       "tier": "stranger", "tenant": "opB", "coord": "10.50.0.9:5540"},
    }
    standings = sp.standings_from_roster(roster_loader=lambda: roster)
    by = {s.node_id: s for s in standings}
    assert by["prithvi-self"].tier == "self"
    assert by["prithvi-self"].capabilities == {"address": "127.0.0.1", "port": 5540}
    assert by["opB-gpu"].capabilities["address"] == "10.50.0.9"
    # firewall still applies end-to-end through the roster bridge
    plan = sp.plan_chain("prithvi-private", standings, num_layers=16, hidden_size=4096,
                         min_tier_fn=MT, rank=RANK)
    assert plan.eligible == ["prithvi-self"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"all serve_planner tests PASS ({len(fns)})")
