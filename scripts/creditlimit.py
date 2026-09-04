"""
Credit limit policy for the Sybil defence.

Identities are free: a keypair costs nothing. So if any new account may go to −N, a thousand
accounts are a thousand × N of free compute. The credit limit is the only thing standing
between the network and that, and it must depend on how much the network knows about you.

Tiers (in increasing order of trust):
- "stranger": a first-time visitor with no history. Limit chosen so that farming is
  pointless (too small to be worth attacking) but one real first run works.
- "known": an account with some minimal history (e.g., verified email or social login).
- "trusted": a long-standing, well-behaved account (e.g., multi-factor authenticated,
  repeated positive interactions).
- "self": internal services or the node operator's own machines — unlimited.

The limit for "stranger" is the critical parameter: it decides whether this survives
contact with the internet. We choose 100 units — enough for a few real API calls or
a small batch of operations, but too small to be worth attacking at scale.
"""


# ⚠️⚠️ THE TIER ORDERING IS NOT OURS TO INVENT. `trisul/infra/control-plane/admission.py`
# has defined TIER_RANK = {"stranger":0,"known":1,"trusted":2,"self":3} since before this
# file existed, and `fabric/worker_join.py` already gates the planner by it — that is what
# keeps Prithvi's sensitive models off a stranger's GPU. This module re-derived the same four
# names independently, which is how a repo ends up with two answers to one question and only
# discovers the drift when they disagree about who may be paid.
#
# So: READ the control plane's ordering when it is reachable, and fall back to a local copy
# only so nakshatra still runs standalone. `tier_source()` says which is in force — a
# fallback that is indistinguishable from the real thing is how the drift hides.
def _load_tier_rank():
    import os
    from pathlib import Path
    cp = os.environ.get("NAKSHATRA_ADMISSION_DIR",
                        str(Path.home() / "trisul" / "infra" / "control-plane"))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_adm", Path(cp) / "admission.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rank = getattr(mod, "TIER_RANK", None)
        if isinstance(rank, dict) and rank:
            return dict(rank), "control-plane"
    except Exception:
        pass
    # ⚠️ Kept in sync BY TEST, not by hope: test_creditlimit asserts these match whenever the
    # control plane is reachable, so a change there fails here instead of silently diverging.
    return {"stranger": 0, "known": 1, "trusted": 2, "self": 3}, "local-fallback"


TIER_RANK, _TIER_SOURCE = _load_tier_rank()


def tier_source() -> str:
    """Which ordering is in force — 'control-plane' or 'local-fallback'."""
    return _TIER_SOURCE


TIERS: dict[str, int] = {
    # stranger: first-time visitor, minimal trust
    # 100 units: enough for ~10-20 small operations, but not worth attacking at scale
    "stranger": 100,
    # known: verified identity or minimal history
    # 500 units: allows more substantial use, but still finite
    "known": 500,
    # trusted: long-standing, well-behaved account
    # 5000 units: allows heavy legitimate use, but still monitored
    "trusted": 5000,
    # self: internal or operator-controlled — unlimited
    # Represented as None (no limit)
    "self": None,
}


def may_consume(balance: int, cost: int, tier: str) -> tuple[bool, str]:
    """
    Determine whether a spend is allowed given the current balance, cost, and identity tier.

    Returns:
        (allowed, why): 
        - allowed: True if balance - cost >= -limit (or unlimited for "self")
        - why: human-readable explanation including tier, limit, and resulting balance
    """
    # Validate types first — do not coerce
    if not isinstance(balance, int):
        return (False, f"balance must be int, got {type(balance).__name__}")
    if not isinstance(cost, int):
        return (False, f"cost must be int, got {type(cost).__name__}")

    # Non-positive cost is always allowed
    if cost <= 0:
        return (True, f"cost ≤ 0 — no spend; tier={tier}, balance={balance}")

    # Get limit for tier; unknown tiers default to stranger (tightest limit)
    limit = TIERS.get(tier, TIERS["stranger"])
    tier_name = tier if tier in TIERS else "stranger"

    # "self" is unlimited
    if limit is None:
        new_balance = balance - cost
        return (True, f"tier={tier_name} (unlimited); balance={balance} → {new_balance}")

    # Check if spend would exceed limit
    new_balance = balance - cost
    if new_balance >= -limit:
        return (True, f"tier={tier_name}, limit={limit}; balance={balance} → {new_balance}")
    else:
        return (False, f"tier={tier_name}, limit={limit}; balance={balance} → {new_balance} (would exceed limit)")