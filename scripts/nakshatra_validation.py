"""Strict-type parse helpers at worker trust boundaries.

Phase D of the worker hardening sprint (2026-05-20). Mirror of
Sthambha's Phase N central helpers — these refuse lenient coercion of
worker-controlled fields so the L4/M1 class of bug (string-truthy
bypass, enum-allowlist bypass) can't surface on the worker's parse
sites either.

Same shape lives on the pillar side at ``sthambha/core.py`` (
``as_strict_bool`` / ``as_safe_int`` / ``as_safe_float`` / etc.). Keep
the behaviour byte-compatible across repos so a malformed value never
silently differs in interpretation between worker and pillar.

Each helper:
  - Accepts any input type (never raises).
  - Returns a SAFE value of the target type, or the given default.
  - Defends at the parse boundary; callers can assume the return is
    of the declared type and within declared bounds.
"""
from __future__ import annotations

import math
import re
from typing import Optional


# ── Strict booleans ──────────────────────────────────────────────────


def as_strict_bool(value, default: bool = False) -> bool:
    """Refuses lenient ``bool(value)`` coercion.

    Only literal ``True`` and ``False`` pass through. Strings, ints,
    lists, dicts all collapse to ``default``. Closes the L4-class
    bypass where ``bool("false")`` is truthy.
    """
    if value is True:
        return True
    if value is False:
        return False
    return default


# ── Bounded integers ─────────────────────────────────────────────────


def as_safe_int(
    value, default: int = 0, *,
    lo: Optional[int] = None, hi: Optional[int] = None,
) -> int:
    """``int(value)`` with try/except + optional bounds clamp.

    Booleans are explicitly rejected (they are technically ``int`` in
    Python but a worker-controlled ``True`` should not silently become
    ``1`` in an integer field). Floats / strings / None all collapse
    to ``default``.
    """
    if isinstance(value, bool):
        return default
    if not isinstance(value, (int, str)):
        return default
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    if lo is not None and n < lo:
        return lo
    if hi is not None and n > hi:
        return hi
    return n


# ── Finite floats ────────────────────────────────────────────────────


def as_safe_float(
    value, default: float = 0.0, *, allow_negative: bool = True,
) -> float:
    """``float(value)`` with try/except + NaN/Inf reject.

    Mirror of Sthambha O5 ``as_safe_float``. NaN and Inf are common
    side-effects of clock skew or bad math upstream; never propagate
    them into the next stage.
    """
    if isinstance(value, bool):
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    if not allow_negative and f < 0:
        return default
    return f


# ── Positive-allowlist enums ─────────────────────────────────────────


def as_str_enum(value, allowed, default: str) -> str:
    """Positive-allowlist for string enums.

    Returns ``value`` only if it's a string that appears in
    ``allowed``; otherwise ``default``. Closes the M1-class enum
    bypass (an unknown enum value passing the negative ``!= "bad"``
    check while being attacker-controlled).
    """
    if isinstance(value, str) and value in allowed:
        return value
    return default


# ── Bounded hex strings ──────────────────────────────────────────────


_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def as_bounded_hex(value, max_chars: int, default: str = "") -> str:
    """String must be hex AND ≤ ``max_chars`` characters.

    Empty string passes through as ``""`` (caller decides if that's
    valid). Non-string / oversized / non-hex collapse to ``default``.
    Returns lowercase for canonicalisation.
    """
    if not isinstance(value, str):
        return default
    if not value:
        return ""
    if len(value) > max_chars:
        return default
    if not _HEX_RE.match(value):
        return default
    return value.lower()


# ── Length-bounded strings ───────────────────────────────────────────


def as_bounded_str(value, max_bytes: int, default: str = "") -> str:
    """String length-capped at ``max_bytes`` (UTF-8 encoded length).

    Non-string / oversized collapse to ``default``.
    """
    if not isinstance(value, str):
        return default
    if len(value.encode("utf-8")) > max_bytes:
        return default
    return value


# ── Lists of strings ─────────────────────────────────────────────────


def as_str_list(
    value, *, max_items: int = 100, max_item_bytes: int = 256,
) -> list[str]:
    """List of strings with per-item byte cap + item-count cap.

    Non-list returns ``[]``; non-string items filtered out; oversized
    items dropped silently. Caller never sees None or junk.
    """
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:max_items]:
        if isinstance(item, str) and len(item.encode("utf-8")) <= max_item_bytes:
            out.append(item)
    return out
