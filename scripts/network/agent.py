"""Strict wire shapes for M1 agent messages and contact cards."""
from __future__ import annotations

import re
import time
from collections.abc import Callable


INTENTS = ("inform", "ask", "answer", "offer", "accept", "decline", "done")
AGENT_MSG_VERSION = 1
CARD_VERSION = 1
MAX_WANT = 500
MAX_HARNESS = 32
MAX_REFS = 8
MAX_REF_BYTES = 10 * 1024 * 1024
MAX_EXPIRY_S = 30 * 24 * 60 * 60
CARD_SUPPORT = "agent.msg/1"

_THREAD_RE = re.compile(r"[0-9a-f]{32}\Z")
_MID_RE = re.compile(r"[0-9a-f]{24}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _bounded_string(value, name: str, maximum: int, *, minimum: int = 0) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        span = f"{minimum}-{maximum}" if minimum else f"at most {maximum}"
        raise ValueError(f"{name} must be {span} characters")
    return value


def validate_agent_message(body: dict, max_text: int, *, outbound: bool = False,
                           now: int | None = None,
                           reply_known: Callable[[str, str], bool] | None = None) -> dict:
    """Validate and normalize one complete ``agent.msg`` body, including its transport mid."""
    if not isinstance(body, dict):
        raise ValueError("agent.msg payload must be an object")
    allowed = {"v", "mid", "thread", "reply_to", "intent", "text", "want", "expires",
               "budget", "refs", "harness"}
    unknown = set(body) - allowed
    if unknown:
        raise ValueError("unknown agent.msg keys: " + ", ".join(sorted(unknown)))
    required = {"v", "mid", "thread", "reply_to", "intent", "text"}
    missing = required - set(body)
    if missing:
        raise ValueError("missing agent.msg keys: " + ", ".join(sorted(missing)))
    out = dict(body)
    if type(out["v"]) is not int or out["v"] != AGENT_MSG_VERSION:
        raise ValueError("agent.msg v must be 1")
    if not isinstance(out["mid"], str) or _MID_RE.fullmatch(out["mid"]) is None:
        raise ValueError("mid must be 24 lowercase hex characters")
    if not isinstance(out["thread"], str) or _THREAD_RE.fullmatch(out["thread"]) is None:
        raise ValueError("thread must be 32 lowercase hex characters")
    reply_to = out["reply_to"]
    if not isinstance(reply_to, str) or (reply_to and _MID_RE.fullmatch(reply_to) is None):
        raise ValueError("reply_to must be empty or a well-formed mid")
    if outbound and reply_to and (reply_known is None or not reply_known(out["thread"], reply_to)):
        raise ValueError("reply_to is not a known message in that thread")
    if out["intent"] not in INTENTS:
        raise ValueError("intent must be one of: " + ", ".join(INTENTS))
    _bounded_string(out["text"], "text", max_text, minimum=1)
    if "want" in out:
        _bounded_string(out["want"], "want", MAX_WANT)
    if "harness" in out:
        _bounded_string(out["harness"], "harness", MAX_HARNESS)
    if "expires" in out:
        expires = out["expires"]
        if not _plain_int(expires) or expires < 0:
            raise ValueError("expires must be a non-negative unix timestamp")
        latest = int(now if now is not None else time.time()) + MAX_EXPIRY_S
        if outbound:
            out["expires"] = min(expires, latest)
        elif expires > latest:
            raise ValueError("expires must be no more than 30 days in the future")
    if "budget" in out:
        budget = out["budget"]
        if not isinstance(budget, dict) or set(budget) != {"amount", "unit"}:
            raise ValueError("budget must contain exactly amount and unit")
        if not _plain_int(budget["amount"]) or budget["unit"] != "credits":
            raise ValueError("budget must be {amount: int, unit: credits}")
        if outbound and budget["amount"] != 0:
            raise ValueError("paid asks arrive in M2")
    if "refs" in out:
        refs = out["refs"]
        if not isinstance(refs, list) or len(refs) > MAX_REFS:
            raise ValueError(f"refs must be a list of at most {MAX_REFS} items")
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {"sha256", "bytes", "name"}:
                raise ValueError("each ref must contain exactly sha256, bytes, and name")
            if not isinstance(ref["sha256"], str) or _SHA256_RE.fullmatch(ref["sha256"]) is None:
                raise ValueError("ref sha256 must be 64 lowercase hex characters")
            if not _plain_int(ref["bytes"]) or not 0 <= ref["bytes"] <= MAX_REF_BYTES:
                raise ValueError(f"ref bytes must be an integer from 0 to {MAX_REF_BYTES}")
            _bounded_string(ref["name"], "ref name", 120)
    return out


def validate_card(body: dict) -> dict:
    """Validate one complete ``contact.card`` body, including version and transport mid."""
    if not isinstance(body, dict):
        raise ValueError("contact.card payload must be an object")
    allowed = {"v", "mid", "display_name", "offers", "intents", "paid", "price_hint",
               "models", "supports", "updated_at"}
    unknown = set(body) - allowed
    if unknown:
        raise ValueError("unknown contact.card keys: " + ", ".join(sorted(unknown)))
    if set(body) != allowed:
        raise ValueError("contact.card is missing required fields")
    if type(body["v"]) is not int or body["v"] != CARD_VERSION:
        raise ValueError("contact.card v must be 1")
    if not isinstance(body["mid"], str) or _MID_RE.fullmatch(body["mid"]) is None:
        raise ValueError("mid must be 24 lowercase hex characters")
    _bounded_string(body["display_name"], "display_name", 64, minimum=1)
    _bounded_string(body["offers"], "offers", 1000, minimum=1)
    intents = body["intents"]
    if not isinstance(intents, list) or len(intents) > len(INTENTS) or any(i not in INTENTS for i in intents):
        raise ValueError("intents must be a list containing only agent.msg intents")
    if len(set(intents)) != len(intents):
        raise ValueError("intents must not contain duplicates")
    if body["paid"] is not False:
        raise ValueError("paid must be false in M1")
    _bounded_string(body["price_hint"], "price_hint", 200)
    models = body["models"]
    if (not isinstance(models, list) or len(models) > 32 or
            any(not isinstance(model, str) or not 1 <= len(model) <= 64 for model in models)):
        raise ValueError("models must contain at most 32 non-empty strings of at most 64 characters")
    supports = body["supports"]
    if (not isinstance(supports, list) or len(supports) > 32 or
            any(not isinstance(item, str) or not 1 <= len(item) <= 64 for item in supports) or
            CARD_SUPPORT not in supports):
        raise ValueError(f"supports must be a string list including {CARD_SUPPORT}")
    if not _plain_int(body["updated_at"]) or body["updated_at"] < 0:
        raise ValueError("updated_at must be a non-negative unix timestamp")
    return dict(body)
