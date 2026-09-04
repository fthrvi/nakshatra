"""The phase contract. Harness-owned; not yours to edit.

Every join phase is a PURE DECISION: `(facts) -> (ok, problems, updates)`. It performs no
I/O — no sockets, no disk, no subprocesses. It is handed observations and returns a verdict.

⚠️⚠️ WHY THIS IS ENFORCED RATHER THAN REQUESTED. A phase that can observe can also retry,
sleep, and fail differently between runs — and then it is no longer a function of its inputs,
so the whole join sequence stops being testable without a real GPU, a real network and a real
coordinator. A join path that can only be tested by actually joining is one nobody tests, and
this has to work on a stranger's machine on the first try.

These tests are generic: they import your module by the name the task gave and exercise the
CONTRACT, not your logic. Passing them means your phase composes with the orchestrator.
"""
import inspect
import os
import importlib

MODULE = os.environ.get("PHASE_MODULE", "")
FUNC = os.environ.get("PHASE_FUNC", "")
mod = importlib.import_module(MODULE)
phase = getattr(mod, FUNC)


def _call(facts):
    out = phase(facts)
    assert isinstance(out, tuple) and len(out) == 3, f"must return a 3-tuple, got {out!r}"
    ok, problems, updates = out
    assert isinstance(ok, bool), f"first element must be bool, got {type(ok).__name__}"
    assert isinstance(problems, list), f"second must be list, got {type(problems).__name__}"
    assert all(isinstance(p, str) for p in problems), "problems must all be strings"
    assert isinstance(updates, dict), f"third must be dict, got {type(updates).__name__}"
    return ok, problems, updates


def test_returns_the_contract_shape():
    _call({})


def test_ok_false_always_has_at_least_one_problem():
    """A refusal with no reason is an outage with a shrug."""
    ok, problems, _ = _call({})
    if not ok:
        assert problems, "returned ok=False with no problems — say WHY"


def test_ok_true_never_invents_a_problem():
    ok, problems, _ = _call({})
    # not asserting emptiness on {} — a phase may legitimately refuse — but if it PASSES on
    # empty facts it must not also be complaining.
    if ok:
        assert problems == [], f"ok=True but problems={problems}"


def test_does_not_mutate_the_facts_it_is_given():
    """⚠️ The orchestrator merges updates itself. A phase that mutates `facts` in place makes
    the merge order load-bearing and the sequence irreproducible."""
    facts = {"a": 1, "nested": {"b": 2}, "lst": [3]}
    before = {"a": 1, "nested": {"b": 2}, "lst": [3]}
    _call(facts)
    assert facts == before, f"phase mutated its input: {facts!r} != {before!r}"


def test_never_raises_on_hostile_facts():
    """`facts` carries observations from a machine we do not control."""
    for bad in ({}, {"accel": None}, {"accel": 42}, {"port": "x"}, {"free_bytes": -1},
                {"layers": "nope"}, {"nested": {"deep": object()}}):
        _call(bad)


def test_is_pure_enough_to_call_twice():
    """Same input, same verdict. A phase that answers differently on the second call is
    reading something it was not handed."""
    facts = {"accel": "cuda", "serve_ngl": 99, "port": 8080, "free_bytes": 10**10}
    a = _call(dict(facts))
    b = _call(dict(facts))
    assert a == b, f"not deterministic: {a!r} then {b!r}"


def test_performs_no_io():
    """⚠️ Source-level check: a decision function must not open sockets, read files, or spawn
    processes. Crude on purpose — it catches the import, which is where the temptation is."""
    src = inspect.getsource(mod)
    banned = ("subprocess", "socket.", "urllib", "requests", "os.system", "shutil.",
              "open(", "Path(")
    found = [b for b in banned if b in src]
    assert not found, (f"phase module performs I/O ({found}) — it must DECIDE, not ACT; "
                       "observations arrive in `facts`")
