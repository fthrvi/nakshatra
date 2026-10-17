"""M1 agent.msg/contact.card strict wire-shape tests."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from network.agent import (MAX_EXPIRY_S, MAX_REF_BYTES, validate_agent_message,  # noqa: E402
                           validate_card)
from network.nakd import MAX_TEXT  # noqa: E402


def message(**changes):
    value = {"v": 1, "mid": "a" * 24, "thread": "b" * 32, "reply_to": "", "intent": "ask",
             "text": "hello"}
    value.update(changes)
    return value


def card(**changes):
    value = {"v": 1, "mid": "a" * 24, "display_name": "Alice", "offers": "research and summaries",
             "intents": ["ask", "answer"], "paid": False, "price_hint": "", "models": ["qwen3"],
             "supports": ["agent.msg/1"], "updated_at": 1}
    value.update(changes)
    return value


@pytest.mark.parametrize("change", [
    {"extra": True}, {"v": 2}, {"mid": "A" * 24}, {"thread": "B" * 32}, {"reply_to": "x"},
    {"intent": "command"}, {"text": ""}, {"text": "x" * (MAX_TEXT + 1)}, {"want": "x" * 501},
    {"harness": "x" * 33}, {"expires": True}, {"expires": -1},
    {"budget": {"amount": 0}}, {"budget": {"amount": True, "unit": "credits"}},
    {"budget": {"amount": 0, "unit": "usd"}},
    {"refs": [{"sha256": "a" * 64, "bytes": 0, "name": "n"}] * 9},
    {"refs": [{"sha256": "A" * 64, "bytes": 0, "name": "n"}]},
    {"refs": [{"sha256": "a" * 64, "bytes": -1, "name": "n"}]},
    {"refs": [{"sha256": "a" * 64, "bytes": MAX_REF_BYTES + 1, "name": "n"}]},
    {"refs": [{"sha256": "a" * 64, "bytes": 0, "name": "n" * 121}]},
    {"refs": [{"sha256": "a" * 64, "bytes": 0, "name": "n", "extra": 1}]},
])
def test_agent_message_rejects_every_bad_field_and_unknown_key(change):
    value = message()
    value.update(change)
    with pytest.raises(ValueError):
        validate_agent_message(value, MAX_TEXT)


def test_agent_message_accepts_all_optional_fields_and_caps_sender_expiry():
    value = message(want="please answer", expires=9999999999, budget={"amount": 0, "unit": "credits"},
                    refs=[{"sha256": "c" * 64, "bytes": MAX_REF_BYTES, "name": "report.md"}],
                    harness="hermes")
    out = validate_agent_message(value, MAX_TEXT, outbound=True, now=100,
                                 reply_known=lambda _thread, _mid: True)
    assert out["expires"] == 100 + MAX_EXPIRY_S


def test_sender_refuses_paid_budget_with_m2_message_and_unknown_reply():
    with pytest.raises(ValueError, match="paid asks arrive in M2"):
        validate_agent_message(message(budget={"amount": 1, "unit": "credits"}), MAX_TEXT,
                               outbound=True)
    with pytest.raises(ValueError, match="not a known message"):
        validate_agent_message(message(reply_to="c" * 24), MAX_TEXT, outbound=True,
                               reply_known=lambda _thread, _mid: False)
    # Receivers carry paid metadata without acting on it, ready for the M2 settlement layer.
    assert validate_agent_message(message(budget={"amount": 7, "unit": "credits"}), MAX_TEXT)[
        "budget"]["amount"] == 7


@pytest.mark.parametrize("change", [
    {"extra": True}, {"v": 2}, {"mid": "bad"}, {"display_name": ""},
    {"display_name": "x" * 65}, {"offers": ""}, {"offers": "x" * 1001},
    {"intents": ["ask", "command"]}, {"intents": ["ask", "ask"]}, {"paid": True},
    {"price_hint": "x" * 201}, {"models": ["x"] * 33}, {"models": ["x" * 65]},
    {"supports": []}, {"supports": ["agent.msg/1", 3]}, {"updated_at": True}, {"updated_at": -1},
])
def test_card_rejects_every_bad_field_and_unknown_key(change):
    value = copy.deepcopy(card())
    value.update(change)
    with pytest.raises(ValueError):
        validate_card(value)


def test_card_accepts_declared_m1_shape():
    assert validate_card(card())["supports"] == ["agent.msg/1"]
