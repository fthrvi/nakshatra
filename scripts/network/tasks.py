"""Nakshatra tasks: the pure part (spec, hash, acceptance). nakd moves them; this decides them.

A task is posted to contacts, claimed by one, assigned by the poster, done, and its result is judged
by the ACCEPTANCE RULES the poster declared at post time. Those rules are part of the signed task
hash, so the worker sees exactly what will pass before claiming, and the poster cannot move the
goalposts after the work is done.

    post ──task.post──▶ worker sees an OFFER (untrusted external request; nothing runs by itself)
         ◀─task.claim── worker commits
    ─────task.assign──▶ first valid claimer gets it (epoch 1)
         ◀─task.result─ output
    ──task.accept|reject▶ judged by the declared rules, deterministically

Rules (all must pass) are deliberately simple and cheap to evaluate on untrusted output — no regex
(a poster-chosen pattern on worker-chosen text is a ReDoS):
    {"max_words": n}  {"max_chars": n}  {"min_words": n}
    {"contains_all": [s, ...]}  {"contains_none": [s, ...]}    (case-insensitive substring)
    {"sha256": "<hex>"}                                         (verifiable compute: exact output)
    {"json_keys": [k, ...]}                                     (output is a JSON object with these keys)
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from typing import Optional

MAX_TITLE = 120
MAX_INSTRUCTIONS = 16000
MAX_OUTPUT = 64000
MAX_RULES = 12
RULE_KINDS = {"max_words", "max_chars", "min_words", "contains_all", "contains_none", "sha256", "json_keys"}
PRIVACY = {"open"}          # sealed / confidential arrive with the sandbox (NAK_SANDBOX), not before


def canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def task_hash(spec: dict) -> str:
    return hashlib.sha256(canonical(spec)).hexdigest()


def _check_rules(rules) -> None:
    if not isinstance(rules, list) or not 1 <= len(rules) <= MAX_RULES:
        raise ValueError(f"acceptance must be a list of 1-{MAX_RULES} rules")
    for r in rules:
        if not isinstance(r, dict) or len(r) != 1 or next(iter(r)) not in RULE_KINDS:
            raise ValueError(f"unknown acceptance rule {r!r}; allowed: {sorted(RULE_KINDS)}")
        k, v = next(iter(r.items()))
        if k in ("max_words", "max_chars", "min_words"):
            if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= MAX_OUTPUT:
                raise ValueError(f"{k} must be an integer 0-{MAX_OUTPUT}")
        elif k in ("contains_all", "contains_none", "json_keys"):
            if not isinstance(v, list) or not 1 <= len(v) <= 32 or \
                    not all(isinstance(s, str) and 0 < len(s) <= 200 for s in v):
                raise ValueError(f"{k} must be a list of 1-32 short strings")
        elif k == "sha256":
            if not (isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)):
                raise ValueError("sha256 must be 64 lowercase hex characters")


def make_spec(poster: str, title: str, instructions: str, acceptance: list, *, reward: int = 0,
              deadline_s: int = 3600, now: Optional[int] = None) -> dict:
    now = int(now if now is not None else time.time())
    spec = {"v": 1, "poster": poster, "title": str(title), "instructions": str(instructions),
            "acceptance": acceptance, "reward": {"amount": int(reward), "unit": "TEST"},
            "deadline": now + int(deadline_s), "privacy": "open", "nonce": secrets.token_hex(8)}
    validate_spec(spec, now=now)
    return spec


def validate_spec(spec: dict, now: Optional[int] = None) -> None:
    """Raise ValueError unless this is a well-formed, unexpired task spec."""
    now = int(now if now is not None else time.time())
    keys = {"v", "poster", "title", "instructions", "acceptance", "reward", "deadline", "privacy", "nonce"}
    if not isinstance(spec, dict) or set(spec) != keys or spec.get("v") != 1:
        raise ValueError("not a v1 task spec")
    if not isinstance(spec["title"], str) or not 0 < len(spec["title"]) <= MAX_TITLE:
        raise ValueError(f"title must be 1-{MAX_TITLE} characters")
    if not isinstance(spec["instructions"], str) or not 0 < len(spec["instructions"]) <= MAX_INSTRUCTIONS:
        raise ValueError(f"instructions must be 1-{MAX_INSTRUCTIONS} characters")
    _check_rules(spec["acceptance"])
    rw = spec["reward"]
    if not isinstance(rw, dict) or set(rw) != {"amount", "unit"} or rw["unit"] != "TEST" or \
            not isinstance(rw["amount"], int) or rw["amount"] < 0:
        raise ValueError("reward must be {amount: int >= 0, unit: TEST} (devnet settlement comes later)")
    if spec["privacy"] not in PRIVACY:
        raise ValueError(f"privacy must be one of {sorted(PRIVACY)} until the sandbox lands")
    if not isinstance(spec["deadline"], int) or spec["deadline"] <= now:
        raise ValueError("task deadline has passed")
    if not isinstance(spec["poster"], str) or len(spec["poster"]) != 64:
        raise ValueError("poster must be a person key")


def evaluate(spec: dict, output: str) -> tuple[bool, list]:
    """Judge an output against the declared rules. Returns (passed, reasons-for-failure)."""
    if not isinstance(output, str) or len(output) > MAX_OUTPUT:
        return False, [f"output must be text of at most {MAX_OUTPUT} characters"]
    fails = []
    low = output.lower()
    words = len(output.split())
    for r in spec["acceptance"]:
        k, v = next(iter(r.items()))
        if k == "max_words" and words > v:
            fails.append(f"{words} words > max {v}")
        elif k == "min_words" and words < v:
            fails.append(f"{words} words < min {v}")
        elif k == "max_chars" and len(output) > v:
            fails.append(f"{len(output)} chars > max {v}")
        elif k == "contains_all":
            missing = [s for s in v if s.lower() not in low]
            if missing:
                fails.append(f"missing {missing}")
        elif k == "contains_none":
            present = [s for s in v if s.lower() in low]
            if present:
                fails.append(f"must not contain {present}")
        elif k == "sha256" and hashlib.sha256(output.encode("utf-8")).hexdigest() != v:
            fails.append("output does not match the expected sha256")
        elif k == "json_keys":
            try:
                obj = json.loads(output)
            except ValueError:
                obj = None
            if not isinstance(obj, dict):
                fails.append("output is not a JSON object")
            else:
                absent = [key for key in v if key not in obj]
                if absent:
                    fails.append(f"JSON missing keys {absent}")
    return (not fails), fails


def describe_rules(rules: list) -> str:
    parts = []
    for r in rules:
        k, v = next(iter(r.items()))
        parts.append({"max_words": f"at most {v} words", "min_words": f"at least {v} words",
                      "max_chars": f"at most {v} characters", "sha256": "exactly the expected output (sha256)",
                      "contains_all": f"must mention {v}", "contains_none": f"must not contain {v}",
                      "json_keys": f"a JSON object with keys {v}"}[k])
    return "; ".join(parts)
