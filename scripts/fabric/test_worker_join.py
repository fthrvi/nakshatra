"""Outsider GPU worker self-join: default-deny, tiered admission, and THE IDENTITY FIREWALL —
a stranger's GPU pools compute for general models but is NEVER assigned Prithvi's sensitive self."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import worker_join as wj

# tier ranks mirror admission.py (stranger<known<trusted<self); injected so the test needs no control plane
RANK = {"stranger": 0, "known": 1, "trusted": 2, "self": 3}
# model policy: Prithvi's own brain is SENSITIVE (self-only); a public model is general.
MODEL_MIN = {"prithvi-brain": "self", "prithvi-journal": "self", "public-llama-70b": "stranger"}


def _admit_stub(roster):
    def admit(req):
        pk = req.get("public_key_hex", "")
        e = roster.get(pk)
        if not e:
            return {"admitted": False, "reason": "not on roster (default-deny)"}
        return {"admitted": True, "tier": e["tier"], "tenant": e.get("tenant", "opB"), "reason": "ok"}
    return admit


def test_default_deny_unlisted_box():
    s = wj.join_as_worker({"public_key_hex": "deadbeef"}, {"gpu": "rtx4090"},
                          admit_fn=_admit_stub({}))
    assert s.admitted is False and s.tier == "denied"


def test_outsider_admitted_at_tier_with_capabilities():
    roster = {"opB-key": {"tier": "stranger", "tenant": "opB"}}
    caps = {"gpu": "rtx-4090", "vram_mb": 24000, "kind": "gpu-worker"}
    s = wj.join_as_worker({"public_key_hex": "opB-key"}, caps, admit_fn=_admit_stub(roster))
    assert s.admitted and s.tier == "stranger" and s.capabilities["gpu"] == "rtx-4090"


def test_identity_firewall():
    # the pool: my own box (self), a vetted friend (trusted), a stranger's GPU (stranger)
    mine = wj.WorkerStanding("mine", "k1", True, "self", "me", {"gpu": "9070xt"})
    friend = wj.WorkerStanding("friend", "k2", True, "trusted", "opA", {"gpu": "3090"})
    stranger = wj.WorkerStanding("stranger", "k3", True, "stranger", "opB", {"gpu": "4090"})
    pool = [mine, friend, stranger]
    mt = lambda m: MODEL_MIN.get(m, "stranger")

    # PRITHVI'S BRAIN (sensitive, self-only) → ONLY his own box; friend + stranger EXCLUDED
    elig, rej = wj.eligible_workers("prithvi-brain", pool, min_tier_fn=mt, rank=RANK)
    assert [w.node_id for w in elig] == ["mine"]
    assert {r["worker"] for r in rej} == {"friend", "stranger"}

    # a vetted-only model (trusted) → his box + friend; stranger excluded
    mt2 = lambda m: "trusted"
    elig2, _ = wj.eligible_workers("some-trusted-model", pool, min_tier_fn=mt2, rank=RANK)
    assert {w.node_id for w in elig2} == {"mine", "friend"}

    # a PUBLIC model (general) → everyone admitted may pool, INCLUDING the stranger's GPU
    elig3, rej3 = wj.eligible_workers("public-llama-70b", pool, min_tier_fn=mt, rank=RANK)
    assert {w.node_id for w in elig3} == {"mine", "friend", "stranger"} and rej3 == []


def test_unadmitted_worker_never_eligible():
    denied = wj.WorkerStanding("x", "k", False, "denied", None, {})
    elig, rej = wj.eligible_workers("public-llama-70b", [denied],
                                    min_tier_fn=lambda m: "stranger", rank=RANK)
    assert elig == [] and rej[0]["reason"] == "not admitted"


if __name__ == "__main__":
    test_default_deny_unlisted_box()
    test_outsider_admitted_at_tier_with_capabilities()
    test_identity_firewall()
    test_unadmitted_worker_never_eligible()
    print("all worker-join tests PASS")
