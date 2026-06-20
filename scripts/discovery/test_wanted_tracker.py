"""Tests for wanted_tracker.py — the demand-signal source. Pure, clock-injected. No hardware."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # scripts/ on path

from discovery.wanted_tracker import WantedTracker  # noqa: E402


class Clock:
    def __init__(self, t=1000.0): self.t = t
    def __call__(self): return self.t


def test_note_then_wanted():
    c = Clock(); w = WantedTracker(ttl=100, clock=c)
    w.note("prithvi-q8"); w.note("deepseek-r1-distill-llama-8b")
    assert w.wanted() == ["deepseek-r1-distill-llama-8b", "prithvi-q8"]   # deduped + sorted


def test_empty_id_is_noop():
    w = WantedTracker()
    w.note(""); w.note(None)  # type: ignore[arg-type]
    assert w.wanted() == []


def test_dedup_refreshes_ttl():
    c = Clock(); w = WantedTracker(ttl=100, clock=c)
    w.note("m1")
    c.t += 80; w.note("m1")          # refresh before expiry
    c.t += 80                        # 160s since first note, but only 80s since refresh
    assert w.wanted() == ["m1"]      # still alive


def test_expiry():
    c = Clock(); w = WantedTracker(ttl=100, clock=c)
    w.note("m1")
    c.t += 101
    assert w.wanted() == []          # aged out


def test_partial_expiry():
    c = Clock(); w = WantedTracker(ttl=100, clock=c)
    w.note("old")
    c.t += 60
    w.note("new")
    c.t += 50                        # old is 110s (dead), new is 50s (alive)
    assert w.wanted() == ["new"]


def test_capacity_cap_drops_oldest():
    c = Clock(); w = WantedTracker(ttl=10_000, max_models=3, clock=c)
    for i, m in enumerate(["a", "b", "c"]):
        c.t += 1; w.note(m)
    c.t += 1; w.note("d")            # over cap → oldest ("a") dropped
    got = w.wanted()
    assert "a" not in got and set(got) == {"b", "c", "d"}


def test_clear():
    w = WantedTracker()
    w.note("m1"); w.clear()
    assert w.wanted() == []
