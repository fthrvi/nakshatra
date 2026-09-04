"""coverage.py — do the signatures actually cover the work the receipt claims?

A receipt carries two independent stories about one run: `chain`, the stages the COORDINATOR
says ran, and `worker_signatures`, the stages WORKERS attested to. Nothing compared them.

⚠️⚠️ THE GAP IS THE POINT. A gap means the coordinator says layers were served that nobody
signed for — precisely where a fabricated stage hides, and today it is invisible: the receipt
verifies (every signature present is genuine), the layer-map is contiguous, and a stage that
simply has no signature reads as "legacy receipt" rather than "unattested work".

⚠️ `unclaimed` is the mirror and is also real: a node signing for layers the chain never
mentions is either replaying a signature from another run or being handed a chain that was
edited after the fact.

Neither is an accusation. This function reports two lists and no verdict; deciding what a gap
MEANS needs context it does not have (a legacy receipt has gaps everywhere and is merely old).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from identity_binding import verify_participation

Span = Tuple[int, int]

_EMPTY: Dict[str, Any] = {"covered": [], "gaps": [], "unclaimed": [], "fully_covered": False}


def _merge(spans: List[Span]) -> List[Span]:
    """Union of half-open [start, end) intervals, sorted. Adjacent spans join: a chain of
    [0,13) and [13,32) is one covered run [0,32), not two — the boundary is an artefact of
    how the work was split, not a hole in it."""
    out: List[Span] = []
    for a, b in sorted(s for s in spans if s[0] < s[1]):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _subtract(base: List[Span], cut: List[Span]) -> List[Span]:
    """base minus cut, both already merged."""
    out: List[Span] = []
    for a, b in base:
        cur = a
        for ca, cb in cut:
            if cb <= cur or ca >= b:
                continue
            if ca > cur:
                out.append((cur, min(ca, b)))
            cur = max(cur, cb)
            if cur >= b:
                break
        if cur < b:
            out.append((cur, b))
    return out


def _spans(items: Any, a_key: str, b_key: str) -> List[Span]:
    out: List[Span] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        a, b = it.get(a_key), it.get(b_key)
        # bool is a subclass of int; True/False as a layer index is not work
        if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool) \
                and not isinstance(b, bool) and a < b:
            out.append((a, b))
    return out


def signature_coverage(receipt: Dict[str, Any], pinned: Dict[str, str]) -> Dict[str, Any]:
    if not isinstance(receipt, dict):
        return dict(_EMPTY)
    run_id = receipt.get("run_id")
    output_sha256 = receipt.get("output_sha256")
    chain = _merge(_spans(receipt.get("chain"), "layer_start", "layer_end"))
    if not isinstance(run_id, str) or not isinstance(output_sha256, str):
        return dict(_EMPTY)

    verified: List[Span] = []
    entries = receipt.get("worker_signatures")
    if isinstance(entries, list):
        for e in entries:
            if not isinstance(e, dict):
                continue
            try:
                ok, _ = verify_participation(e, run_id=run_id,
                                             output_sha256=output_sha256, pinned=pinned)
            except Exception:
                ok = False
            # ⚠️ An entry that does not verify contributes to NOTHING — not `covered`, and
            # not `unclaimed` either. Letting a forgery create an `unclaimed` finding would
            # let anyone manufacture alarms about a run they had no part in.
            if ok:
                verified.extend(_spans([e], "layer_start", "layer_end"))

    covered = _merge(verified)
    return {
        "covered": covered,
        "gaps": _subtract(chain, covered),
        "unclaimed": _subtract(covered, chain),
        # ⚠️ An EMPTY chain is not "fully covered" — it is nothing. Returning True would let
        # a receipt with no stages at all pass a coverage check.
        "fully_covered": bool(chain) and not _subtract(chain, covered),
    }
