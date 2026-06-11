"""Drift-class-constrained recovery (v1.1 hardening) — the soundness fix Petals papers over.

The client already has alternate-worker recovery (v0.5 M0.5.4): each worker has a
primary candidate + optional alternates; on failure the cursor advances to an
alternate and the chain is rebuilt + history replayed. The hole: it will happily
fail over onto a peer that holds the same *layers* but runs a different *engine
build* — and the cross-machine validation (docs/cross-machine-validation.md §2a)
proved that **silently diverges the generation** (same weights, different FP
rounding → a close-call argmax flips, and the resumed stream drifts).

So recovery must be **drift-class-constrained**: an alternate is only eligible if
it is in the **same drift class** (= same engine build) as the primary it
replaces. The drift class travels in the candidate spec / Nostr listing
(`drift_class`, P1 §8.1), so no new wire field is needed. A failed worker with no
same-class alternate is a *clean* recovery failure — never a silent corruption.

This module is the pure decision; client.py uses `next_compatible_cursor` in its
`_advance_one_alternate` so the existing recovery loop becomes sound.
"""
from __future__ import annotations

from typing import Optional, Sequence, Mapping


def drift_compatible(primary_class: Optional[str], candidate_class: Optional[str]) -> bool:
    """True iff an alternate of `candidate_class` may safely replace a primary of
    `primary_class`.

    Policy:
      • primary has NO class (legacy / unclassified mesh) → any alternate is
        allowed (preserve pre-drift-gauge behaviour; nothing to enforce).
      • primary HAS a class → the alternate must declare the SAME class.
        A mismatched OR unknown-class alternate is refused — we can't prove it
        won't diverge, so we don't risk the generation.
    """
    if not primary_class:
        return True
    return candidate_class == primary_class


def next_compatible_cursor(candidates: Sequence[Mapping], current: int) -> Optional[int]:
    """The next cursor index (> current) whose candidate is drift-compatible with
    the PRIMARY (candidates[0]). Returns None if no compatible alternate remains.

    Skips drift-incompatible alternates entirely — they are never used, because
    recovering onto them would silently corrupt the stream."""
    if not candidates:
        return None
    primary_class = candidates[0].get("drift_class")
    for i in range(current + 1, len(candidates)):
        if drift_compatible(primary_class, candidates[i].get("drift_class")):
            return i
    return None


def first_advanceable_worker(workers: Sequence[Mapping]) -> Optional[Mapping]:
    """Pick the first worker that has a remaining DRIFT-COMPATIBLE alternate and
    advance its cursor to it (mutates `worker['cursor']`). Returns that worker, or
    None if no worker has a safe alternate left. Drop-in for the client's
    `_advance_one_alternate`, now sound under heterogeneity."""
    for w in workers:
        nxt = next_compatible_cursor(w["candidates"], w["cursor"])
        if nxt is not None:
            w["cursor"] = nxt
            return w
    return None
