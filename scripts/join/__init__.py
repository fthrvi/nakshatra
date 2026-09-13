"""nakshatra join — a phase-contract decision engine with a stubbed observation layer.

⚠️ SCOPING (2026-09-13): the real `--code` path cannot complete any phase today — no
coordinator serves `GET /v1/join-info`, no keygen exists in this package, and no code
produces the box/package facts later phases require. Live onboarding today is
`~/trisul/infra/onboarding/worker.sh` (strangers) and the `nakshatra-registrar` service
(owned nodes). Full analysis: nakshatra memory `reference_join_package_scoping_and_verdict.md`.
What follows describes the decision engine itself, which is real and tested via
`--observations`; it is the world-facing glue (`act.py`'s I/O) that is missing.

⚠️⚠️ THE DESIGN RULE THAT MAKES THIS TESTABLE: **phases DECIDE, the orchestrator ACTS.**

Every phase is a pure function `(facts) -> (ok, problems, updates)`. It performs no I/O. It
does not open a socket, read a disk, or run a build. It is handed OBSERVATIONS in `facts` and
returns a decision plus new facts to merge.

All the I/O lives in `orchestrate()` below, which is deliberately thin: gather observations,
call the phase, merge, repeat. That split is not tidiness — it is what lets the entire join
sequence be tested end to end with a dict, on a laptop, with no GPU, no network and no
coordinator. A join path that can only be tested by actually joining is a join path nobody
tests, and this one has to work on a stranger's machine on the first try.

⚠️ A phase returning `ok=False` STOPS the sequence. `problems` is the whole list, never the
first — an operator fixing a node wants every reason at once, not one round trip each.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

Phase = Callable[[Dict[str, Any]], Tuple[bool, List[str], Dict[str, Any]]]

#: The order a node actually joins in. Each name resolves to a pure decision function.
PHASE_ORDER = ("admit", "preflight", "acquire", "identity", "serve", "prove")


def orchestrate(facts: Dict[str, Any], phases: Dict[str, Phase],
                observe: Callable[[str, Dict[str, Any]], Dict[str, Any]] | None = None,
                order: tuple[str, ...] = PHASE_ORDER) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Run the phases in order, merging their facts. The ONLY place I/O may happen is inside
    `observe`, which the caller supplies — tests pass a dict-returning stub, production passes
    the function that really reads the disk and dials the coordinator.

    ⚠️ `observe` runs BEFORE each phase, never inside it. A phase that could observe could
    also retry, sleep, or fail differently between runs, and then the decision logic is no
    longer a function of its inputs — which is exactly the property the tests depend on.
    """
    facts = dict(facts)
    problems: List[str] = []
    for name in order:
        if observe is not None:
            try:
                facts.update(observe(name, facts) or {})
            except Exception as e:                      # noqa: BLE001
                # An observation that fails is a fact we do not have, not a crash. Record it
                # and let the phase decide whether it can proceed without that fact.
                problems.append(f"{name}: could not observe ({type(e).__name__}: {e})")
        phase = phases.get(name)
        if phase is None:
            problems.append(f"{name}: no phase implementation registered")
            return False, problems, facts
        ok, why, updates = phase(facts)
        facts.update(updates or {})
        problems.extend(f"{name}: {w}" for w in (why or []))
        if not ok:
            return False, problems, facts
    return True, problems, facts
