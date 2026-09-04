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