"""Is the daemon ready, still coming up, or gone?

⚠️⚠️ ONE 200 IS NOT READY. A server that binds its port before loading weights answers the
first probe and then stalls for minutes while it reads 80 GB off disk. Requiring CONSECUTIVE
successes is the only thing that distinguishes "listening" from "serving", and a join that
declares success on the first 200 hands the network a node that will time out every request.

⚠️ And a refused connection while the process is still ALIVE is `waiting`, not `failed`.
A model loading weights refuses connections for minutes; calling that a failure aborts joins
that were seconds from succeeding.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

MAX_FRUITLESS = 5


def daemon_ready(polls: Any, *, min_ok: int = 2) -> Tuple[str, str]:
    if not isinstance(polls, list) or not polls:
        return "waiting", "no polls yet"
    clean: List[Dict[str, Any]] = [p for p in polls if isinstance(p, dict)]
    if not clean:
        return "waiting", "no usable poll results"

    # ⚠️ FAILURE IS CHECKED FIRST AND NEEDS POSITIVE EVIDENCE. A dead process is the only
    # thing that turns "not yet" into "never" — everything else is patience.
    if any(p.get("process_exited") is True for p in clean):
        return "failed", "the daemon process exited"

    tail = clean[-min_ok:]
    if len(tail) >= min_ok and all(p.get("http_status") == 200 for p in tail):
        return "ready", f"{min_ok} consecutive 200s"

    if not any(p.get("http_status") == 200 for p in clean) and len(clean) >= MAX_FRUITLESS:
        # ⚠️ `>=`, not `>`. The spec said "5 polls with no 200"; an off-by-one here means a
        # node that is genuinely dead is waited on forever, which is how a join hangs.
        last = clean[-1].get("error") or f"HTTP {clean[-1].get('http_status')}"
        return "failed", f"{len(clean)} polls, never a 200 (last: {last})"

    got = sum(1 for p in clean if p.get("http_status") == 200)
    return "waiting", f"{got}/{min_ok} consecutive 200s after {len(clean)} polls"
