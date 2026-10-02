"""accounting.py — THE accounting front door (unification U5, 2026-10-03).

Before this, accounting grew in four places that did not know about each other: the inference
gateway's credit hook (`ledger_client.py`), the task escrow (`network/settle.py`), the receipts / settle
key / credit limits (`receipt.py`, `settlekey.py`, `creditlimit.py`), and the signer's SpendBook (in
Sthambha). This module does not replace any of them. It is the ONE place that names both halves of
accounting and which backend serves each, so a new backend (Solana devnet escrow, the repaired Neuron
ledger) is added HERE, once, and every caller gets it.

    METERING — pay for inference as it is served (reciprocal compute credits)
        gate(est_cost) -> (allow, reason)        before a run
        settle(receipt_path) -> delta | None     after a run, from the signed run receipt
        backend today: Neuron ledger service via `ledger_client.LedgerHook`
        (DEFAULT-OFF, FAIL-OPEN: it can never block or break a served reply)

    ESCROW — pay for a task posted to another person
        open(...) / release(...) / refund(...) / status(...)   exactly once per task
        backend today: `network.settle.LedgerAdapter` (local ledger, TEST units)
        next: Solana devnet escrow (an adapter with the same four calls)

Choose backends with env, not code: NAKSHATRA_CREDITS (metering on/off, see ledger_client) and
NAKSHATRA_ESCROW=ledger (the only escrow backend today; anything else is refused, not guessed).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

ESCROW_BACKENDS = ("ledger",)


def metering():
    """The metering half, configured from env. Never raises: a broken backend degrades to a no-op
    that allows every run (the gateway must always be able to serve)."""
    try:
        from ledger_client import LedgerHook
        return LedgerHook()
    except Exception:  # noqa: BLE001 — the inference gateway's safety contract
        from types import SimpleNamespace
        return SimpleNamespace(enabled=False, wants_receipt=False,
                               gate=lambda est: (True, ""), settle=lambda p: None)


def escrow(state_dir: Path):
    """The escrow half for a node's state dir. Refuses an unknown backend rather than guessing."""
    name = os.environ.get("NAKSHATRA_ESCROW", "ledger").strip() or "ledger"
    if name not in ESCROW_BACKENDS:
        raise ValueError(f"NAKSHATRA_ESCROW={name!r} is not a known escrow backend {ESCROW_BACKENDS}")
    from network.settle import LedgerAdapter
    return LedgerAdapter(Path(state_dir) / "settle.sqlite")
