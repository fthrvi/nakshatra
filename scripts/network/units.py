"""Exact conversion between human USDC amounts and Nakshatra base units."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


BASE_UNITS_PER_USDC = 100_000_000
MAX_SPEND_CAP = 10**14
MAX_AMOUNT_INPUT_LEN = 40

_BASE_RE = re.compile(r"^(?P<number>[0-9]+) base(?: units?)?$", re.IGNORECASE | re.ASCII)
_USDC_RE = re.compile(
    r"^(?P<dollars>\$)?(?P<number>[0-9]+(?:\.[0-9]{1,8})?)(?: ?(?P<unit>USDC))?$",
    re.IGNORECASE | re.ASCII,
)


class AmountError(ValueError):
    """A human amount cannot be represented safely in base units."""


def parse_amount(value: object, *, max_base_units: int = MAX_SPEND_CAP,
                 allow_zero: bool = True) -> tuple[int, bool]:
    """Return ``(base_units, was_legacy_bare_integer)`` using exact decimals.

    A decimal point, ``$`` prefix, or ``USDC`` suffix selects human USDC. An
    explicit ``base`` suffix selects base units. For backward compatibility,
    every bare integer also remains base units; callers should show a notice
    when the returned boolean is true.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise AmountError("amount must be a USDC decimal or integer base-unit amount")
    minimum = 0 if allow_zero else 1
    if isinstance(value, int):
        if not minimum <= value <= max_base_units:
            raise AmountError(f"amount must be between {minimum} and {max_base_units} base units")
        return value, True

    # Bound attacker-controlled text before int() or Decimal() sees it.  Apart
    # from avoiding Python's own integer digit-limit exception, this keeps the
    # grammar deliberately ASCII-only and rejects invisible outer whitespace.
    if len(value) > MAX_AMOUNT_INPUT_LEN:
        raise AmountError(f"amount must be at most {MAX_AMOUNT_INPUT_LEN} characters")
    text = value
    if not text:
        raise AmountError("amount is empty")
    if text.startswith("-"):
        raise AmountError(f"amount must be between {minimum} and {max_base_units} base units")

    base_match = _BASE_RE.fullmatch(text)
    if base_match:
        amount = int(base_match.group("number"))
        legacy = False
    elif re.fullmatch(r"[0-9]+", text, re.ASCII):
        amount = int(text)
        legacy = True
    else:
        match = _USDC_RE.fullmatch(text)
        if not match or not (match.group("dollars") or match.group("unit") or "." in text):
            raise AmountError("use a decimal point, $ or USDC for human units, or append 'base'")
        number = match.group("number")
        try:
            decimal = Decimal(number)
        except InvalidOperation as exc:
            raise AmountError("amount is not a valid decimal") from exc
        scaled = decimal * BASE_UNITS_PER_USDC
        if not decimal.is_finite() or scaled != scaled.to_integral_value():
            raise AmountError("amount cannot be represented in base units")
        amount = int(scaled)
        legacy = False

    if not minimum <= amount <= max_base_units:
        raise AmountError(f"amount must be between {minimum} and {max_base_units} base units")
    return amount, legacy


def format_amount(base_units: int) -> str:
    """Format integer base units without floats or insignificant zeroes."""
    if isinstance(base_units, bool) or not isinstance(base_units, int) or base_units < 0:
        raise AmountError("base units must be a non-negative integer")
    whole, remainder = divmod(base_units, BASE_UNITS_PER_USDC)
    if remainder:
        human = f"{whole}.{remainder:08d}".rstrip("0")
    else:
        human = str(whole)
    return f"{human} USDC ({base_units} base)"
