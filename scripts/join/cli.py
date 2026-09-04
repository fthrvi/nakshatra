#!/usr/bin/env python3
"""nakshatra join — one command turns a machine into a serving node.

    join.py --code <join-code>                 # the real thing
    join.py --observations facts.json          # every decision, no I/O at all
    join.py --observations facts.json --explain

⚠️ `--observations` is not a demo mode. It is how the join path is tested: the six phases are
pure decisions, so feeding them a dict exercises every branch — including every refusal —
on a laptop with no GPU, no network and no coordinator. A join path testable only by actually
joining is a join path nobody tests, and this one has to work on a stranger's machine on the
first attempt.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from join import PHASE_ORDER, orchestrate
from join.observe import load_observations, observe as real_observe
from join.phase_acquire import acquire
from join.phase_admit import admit
from join.phase_identity import identity
from join.phase_preflight import preflight
from join.phase_prove import prove
from join.phase_serve import serve

PHASES = {"admit": admit, "preflight": preflight, "acquire": acquire,
          "identity": identity, "serve": serve, "prove": prove}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nakshatra join")
    ap.add_argument("--code", default="", help="the join code the coordinator gave you")
    ap.add_argument("--observations", default="",
                    help="JSON of pre-taken observations; runs every decision with NO I/O")
    ap.add_argument("--explain", action="store_true",
                    help="print each phase's verdict as it is reached")
    ap.add_argument("--json", action="store_true", help="machine-readable result")
    a = ap.parse_args(argv)

    facts: Dict[str, Any] = {"join_code": a.code}
    if a.observations:
        try:
            facts.update(load_observations(a.observations))
        except (OSError, ValueError) as e:
            print(f"could not read observations: {e}", file=sys.stderr)
            return 2
        observer = None                    # ⚠️ no I/O at all in this mode, by construction
    else:
        if not a.code:
            print("need --code (or --observations for a dry run)", file=sys.stderr)
            return 2
        observer = real_observe

    seen: list[str] = []

    def traced(phase: str, f: Dict[str, Any]) -> Dict[str, Any]:
        seen.append(phase)
        return real_observe(phase, f) if observer else {}

    ok, problems, out = orchestrate(facts, PHASES,
                                    observe=traced if a.explain or observer else None)

    if a.json:
        print(json.dumps({"joined": ok, "problems": problems,
                          "reached": seen or list(PHASE_ORDER),
                          "facts": {k: v for k, v in out.items()
                                    if k not in ("join", "join_code")}}, indent=2, default=str))
        return 0 if ok else 1

    # ⚠️ `join` and `join_code` are excluded above and here: the join code is a bearer
    # credential and this output gets pasted into issues.
    if ok:
        layers = out.get("serving_layers") or [out.get("layer_start"), out.get("layer_end")]
        print(f"joined — serving layers {layers}, accel={out.get('accel')}, "
              f"ngl={out.get('serve_ngl')}")
        print(f"account: {out.get('account', '(none)')}")
    else:
        # ⚠️ Derive the stopping phase from the PROBLEM, not from the observation trace.
        # `seen` is only populated when an observer runs, so in --observations mode it is
        # empty and a fallback of "admit" told the operator the wrong phase failed — pointing
        # them at the join code when the card was idle. The problems are prefixed
        # "<phase>: ..." by the orchestrator, so the last one names where it actually stopped.
        reached = problems[-1].split(":", 1)[0] if problems else (seen[-1] if seen else "?")
        print(f"NOT joined — stopped at phase '{reached}'")
        for p in problems:
            print(f"  · {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
