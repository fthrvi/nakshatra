"""Every join phase obeys the same contract, checked here for all of them at once.

⚠️⚠️ THE RULE: a phase DECIDES, the orchestrator ACTS. `(facts) -> (ok, problems, updates)`,
pure, no I/O. That split is what lets the whole join sequence be tested with a dict — on a
laptop, no GPU, no network, no coordinator. A join path testable only by actually joining is
a join path nobody tests, and this one has to work on a stranger's machine on the first try.

⚠️ The purity check bans I/O, NOT PARSING. An earlier version banned the substring "urllib"
outright, which also forbade `urllib.parse.urlparse` — a pure string function the spec
explicitly told the agent to use. Caught between the two, it hand-parsed URLs and got the
IPv6 bracket form `[::1]` wrong, twice, across sixteen iterations. The contract and the spec
disagreed and there was no way for it to know which was authoritative. Ban `urllib.request`
and `urlopen`; leave parsing alone.
"""
import importlib
import inspect

import pytest

PHASES = ["admit", "preflight", "acquire", "identity", "serve", "prove"]


def _load(name):
    try:
        mod = importlib.import_module(f"join.phase_{name}")
    except ModuleNotFoundError:
        pytest.skip(f"phase_{name} not implemented yet")
    return mod, getattr(mod, name)


def _call(fn, facts):
    out = fn(facts)
    assert isinstance(out, tuple) and len(out) == 3, f"must return a 3-tuple, got {out!r}"
    ok, problems, updates = out
    assert isinstance(ok, bool), f"first element must be bool, got {type(ok).__name__}"
    assert isinstance(problems, list) and all(isinstance(p, str) for p in problems)
    assert isinstance(updates, dict), f"third must be dict, got {type(updates).__name__}"
    return ok, problems, updates


@pytest.mark.parametrize("name", PHASES)
def test_returns_the_contract_shape(name):
    _call(_load(name)[1], {})


@pytest.mark.parametrize("name", PHASES)
def test_a_refusal_always_says_why(name):
    """A refusal with no reason is an outage with a shrug."""
    ok, problems, _ = _call(_load(name)[1], {})
    if not ok:
        assert problems, "returned ok=False with no problems"


@pytest.mark.parametrize("name", PHASES)
def test_does_not_mutate_the_facts_it_is_given(name):
    """⚠️ The orchestrator merges updates itself. A phase that mutates `facts` in place makes
    the merge order load-bearing and the whole sequence irreproducible."""
    facts = {"a": 1, "nested": {"b": 2}, "lst": [3]}
    before = {"a": 1, "nested": {"b": 2}, "lst": [3]}
    _call(_load(name)[1], facts)
    assert facts == before, f"{name} mutated its input"


@pytest.mark.parametrize("name", PHASES)
def test_never_raises_on_hostile_facts(name):
    """`facts` carries observations from a machine we do not control."""
    fn = _load(name)[1]
    for bad in ({}, {"accel": None}, {"accel": 42}, {"port": "x"}, {"free_bytes": -1},
                {"layers": "nope"}, {"running_argv": "not-a-list"},
                {"probe_tokens": {}}, {"nested": {"deep": object()}}):
        _call(fn, bad)


@pytest.mark.parametrize("name", PHASES)
def test_same_facts_same_verdict(name):
    """A phase that answers differently on the second call is reading something it was not
    handed — and then it is no longer a function of its inputs."""
    fn = _load(name)[1]
    facts = {"accel": "cuda", "serve_ngl": 99, "port": 8080, "free_bytes": 10 ** 10}
    assert _call(fn, dict(facts)) == _call(fn, dict(facts)), f"{name} is not deterministic"


@pytest.mark.parametrize("name", PHASES)
def test_performs_no_io(name):
    """Source-level, crude on purpose: it catches the import, which is where the temptation
    lives. Parsing is explicitly allowed — see the module docstring."""
    src = inspect.getsource(_load(name)[0])
    banned = ("subprocess", "socket.socket", "urllib.request", "urlopen", "requests.",
              "os.system", "os.popen", "shutil.", "open(", "Path(")
    found = [b for b in banned if b in src]
    assert not found, f"{name} performs I/O ({found}) — it must DECIDE, not ACT"
