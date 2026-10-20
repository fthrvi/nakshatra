"""Pure paid-task verdict policy.

The policy is person-signed inside a Nak grant.  This module deliberately has no I/O: callers
provide the rule result, reward, escrow kind, and roster classification, and receive one action.
"""
from __future__ import annotations

import hashlib
import json

DEFAULT_MODE = "threshold"
DEFAULT_AUTO_MAX = 10_000_000
DEFAULT_ASK_STRANGERS = True
MAX_AUTO_MAX = 10**14
MODES = frozenset({"auto", "ask", "threshold"})
ACCEPT, REJECT, ASK = "accept", "reject", "ask"
DECISION_MAX_SKEW_S = 120


def decision_statement(task_hash: str, decision: str, reason: str, ts: int) -> bytes:
    """Canonical person-signature payload for one explicit paid-task decision."""
    body = {
        "task_hash": task_hash,
        "decision": decision,
        "reason_sha256": hashlib.sha256(reason.encode("utf-8")).hexdigest(),
        "ts": ts,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def policy(constraints: object) -> dict:
    """Return the effective, validated policy; malformed signed values fail safely to ASK."""
    raw = constraints if isinstance(constraints, dict) else {}
    mode = raw.get("verdict_mode", DEFAULT_MODE)
    auto_max = raw.get("verdict_auto_max", DEFAULT_AUTO_MAX)
    ask_strangers = raw.get("verdict_ask_strangers", DEFAULT_ASK_STRANGERS)
    if mode not in MODES:
        mode = "ask"
    if (isinstance(auto_max, bool) or not isinstance(auto_max, int)
            or not 0 <= auto_max <= MAX_AUTO_MAX):
        auto_max = 0
    if not isinstance(ask_strangers, bool):
        ask_strangers = True
    return {"mode": mode, "auto_max": auto_max, "ask_strangers": ask_strangers}


def decide(*, mode: str, reward: int, auto_max: int, ask_strangers: bool,
           stranger: bool, rules_passed: bool, escrowed: bool = True) -> str:
    """Return ``accept``, ``reject``, or ``ask`` for one completed task.

    Free and non-Neuron tasks retain the original deterministic rules verdict.
    """
    if not rules_passed:
        return REJECT
    if not escrowed or reward <= 0:
        return ACCEPT
    if mode == "ask":
        return ASK
    if ask_strangers and stranger:
        return ASK
    if mode == "auto":
        return ACCEPT
    if mode == "threshold":
        return ACCEPT if reward <= auto_max else ASK
    return ASK


def decide_policy(constraints: object, *, reward: int, stranger: bool,
                  rules_passed: bool, escrowed: bool = True) -> str:
    effective = policy(constraints)
    return decide(reward=reward, stranger=stranger, rules_passed=rules_passed,
                  escrowed=escrowed, **effective)


def describe(effective: dict) -> str:
    mode = effective["mode"]
    if mode == "threshold":
        base = f"automatic through {effective['auto_max']} credits; ask above"
    elif mode == "auto":
        base = "automatic"
    else:
        base = "always ask"
    return base + ("; ask for strangers" if effective["ask_strangers"] else "; strangers follow the same rule")
