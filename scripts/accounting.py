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
        backend: `neuron-ledger`, via Sthambha-Ed25519-authenticated `ledger_client.LedgerHook`
        (DEFAULT-OFF, FAIL-OPEN: it can never block or break a served reply)

    ESCROW — pay for a task posted to another person
        open(...) / release(...) / refund(...) / status(...)   exactly once per task
        backends: `network.settle.LedgerAdapter` (default local TEST units), or explicit
        `network.settle.NeuronEscrowAdapter` (pinned HTTPS devnet credits)

    WALLET — deposit address, balance, payout binding, and withdrawal requests
        backend: `neuron-ledger`; Solana keys remain outside Nakshatra and the ledger server

Choose backends with env, not code: NAKSHATRA_CREDITS (metering on/off, see ledger_client) and
NAKSHATRA_ESCROW=ledger|neuron. The default remains the local TEST ledger; Neuron is explicit,
fail-closed, and uses the gateway's pinned ledger configuration.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

ESCROW_BACKENDS = ("ledger", "neuron")
METERING_BACKEND = "neuron-ledger"
WALLET_BACKEND = "neuron-ledger"


def metering():
    """The metering half, configured from env.

    The historical hub policy remains fail-open. An outsider gateway explicitly selects
    ``NAKSHATRA_LEDGER_MODE=enforce``; in that mode even an import/construction failure must deny.
    """
    try:
        from ledger_client import LedgerHook
        return LedgerHook()
    except Exception:  # noqa: BLE001 — the inference gateway's safety contract
        from types import SimpleNamespace
        if os.environ.get("NAKSHATRA_LEDGER_MODE", "").strip().lower() == "enforce":
            return SimpleNamespace(
                enabled=True, wants_receipt=True, fail_closed=True,
                gate=lambda est: (False, "ledger client unavailable"),
                settle=lambda p: (_ for _ in ()).throw(
                    RuntimeError("ledger client unavailable")),
            )
        return SimpleNamespace(enabled=False, wants_receipt=False,
                               gate=lambda est: (True, ""), settle=lambda p: None)


def _environment_file(path: Path) -> dict[str, str]:
    """Read the scalar subset emitted by ``nak gateway on`` without evaluating shell syntax."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    values = {}
    for line in lines:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def escrow(state_dir: Path, *, private_key: bytes | None = None, keyid: str | None = None):
    """The escrow half for a node's state dir. Refuses an unknown backend rather than guessing."""
    name = os.environ.get("NAKSHATRA_ESCROW", "ledger").strip() or "ledger"
    if name not in ESCROW_BACKENDS:
        raise ValueError(f"NAKSHATRA_ESCROW={name!r} is not a known escrow backend {ESCROW_BACKENDS}")
    from network.settle import LedgerAdapter
    state_dir = Path(state_dir)
    if name == "ledger":
        return LedgerAdapter(state_dir / "settle.sqlite")
    from network.settle import NeuronEscrowAdapter
    configured = _environment_file(state_dir.parent / "gateway.env")
    url = (os.environ.get("NAKSHATRA_LEDGER_URL", "").strip()
           or configured.get("NAKSHATRA_LEDGER_URL", "").strip())
    pin = (os.environ.get("NAKSHATRA_LEDGER_SPKI_SHA256", "").strip()
           or configured.get("NAKSHATRA_LEDGER_SPKI_SHA256", "").strip())
    mode = (os.environ.get("NAKSHATRA_LEDGER_MODE", "").strip()
            or configured.get("NAKSHATRA_LEDGER_MODE", "").strip())
    try:
        timeout = float(os.environ.get("NAKSHATRA_LEDGER_TIMEOUT", "5") or 5)
    except ValueError:
        timeout = 5.0
    return NeuronEscrowAdapter(url, pin, private_key or b"", keyid or "",
                               mode=mode, timeout=timeout)


def wallet(**kwargs):
    """The fail-closed Neuron wallet client used by explicit CLI/MCP reads and actions."""
    from ledger_client import LedgerHook
    return LedgerHook(enabled=True, mode="enforce", **kwargs)
