"""
ledger_client.py — serve-side hook into the neuron credit-ledger service (the integration seam).

Two calls the inference serve makes around a run:
  • gate(est_cost)      — before: GET /can_consume   (may DENY only when enforced)
  • settle(receipt_path)— after:  POST /settle        (credits the workers from the #20 receipt)

SAFETY (this guards Prithvi's live brain):
  • DEFAULT-OFF: with NAKSHATRA_CREDITS unset, every method is a total no-op — the serve is
    byte-for-byte unchanged.
  • FAIL-OPEN: any ledger error (down, slow, 5xx) → the run PROCEEDS. The credit system can
    never block or break a served reply. Denial happens ONLY when NAKSHATRA_CREDITS_ENFORCE=1
    AND the ledger explicitly says the requester is out of credits.

Env:
  NAKSHATRA_CREDITS=1            enable the hook (still fail-open; advisory unless enforced)
  NAKSHATRA_CREDITS_ENFORCE=1   actually reject when the ledger denies (for when outsiders pay)
  NAKSHATRA_LEDGER_URL          default http://127.0.0.1:8093
  NAKSHATRA_LEDGER_REQUESTER    who is consuming (default "self" = Prithvi; an operator id for outsiders)
  NAKSHATRA_LEDGER_TIMEOUT      seconds (default 2)
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


def _truthy(v: str) -> bool:
    return (v or "").strip() in ("1", "true", "True", "yes", "on")


class LedgerHook:
    def __init__(self):
        self.enabled = _truthy(os.environ.get("NAKSHATRA_CREDITS", ""))
        self.enforce = _truthy(os.environ.get("NAKSHATRA_CREDITS_ENFORCE", ""))
        self.url = os.environ.get("NAKSHATRA_LEDGER_URL", "http://127.0.0.1:8093").rstrip("/")
        self.requester = os.environ.get("NAKSHATRA_LEDGER_REQUESTER", "self")
        try:
            self.timeout = float(os.environ.get("NAKSHATRA_LEDGER_TIMEOUT", "2") or 2)
        except ValueError:
            self.timeout = 2.0

    @property
    def wants_receipt(self) -> bool:
        return self.enabled

    def gate(self, est_cost: int):
        """Return (allow: bool, reason: str). Allows when disabled, on any error (fail-open), or
        when the ledger says yes / enforcement is off. Denies ONLY when enabled+enforce+denied."""
        if not self.enabled:
            return True, ""
        try:
            q = urllib.parse.urlencode({"account": self.requester, "cost": int(est_cost)})
            r = self._get(f"/can_consume?{q}")
            ok = bool(r.get("ok"))
        except Exception as e:
            return True, f"ledger unreachable ({e}); fail-open"
        if ok:
            return True, ""
        if not self.enforce:
            return True, "insufficient credits (advisory; enforcement off)"
        return False, f"insufficient credits (balance {r.get('balance')}, cost {int(est_cost)})"

    def settle(self, receipt_path: str):
        """Credit the workers from a #20 run receipt. Fail-open: a settle failure NEVER affects
        the served reply. Returns the delta dict or None."""
        if not self.enabled:
            return None
        try:
            with open(receipt_path) as f:
                receipt = json.load(f)
            body = {"receipt": receipt, "requester": self.requester}
            own = self._own_nodes(receipt)
            if own:
                body["own_nodes"] = own        # local infra → free
            # identity-binding: attach the holder-of-key-PROVEN accounts (each worker Ed25519-signed the
            # stage it served). Advisory — the ledger credits BOUND accounts instead of trusting the
            # coordinator's say-so. Empty for legacy receipts without worker_signatures (non-breaking).
            try:
                from identity_binding import creditable_accounts
                accts, _ = creditable_accounts(receipt)
                if accts:
                    body["creditable_accounts"] = accts
            except Exception:
                pass
            return self._post("/settle", body)
        except Exception:
            return None

    def _own_nodes(self, receipt):
        """Which serving nodes belong to the requester (→ free). NAKSHATRA_LEDGER_OWN_ALL=1 treats
        every worker in the run as own (right for a single-operator box like Prithvi's); otherwise
        NAKSHATRA_LEDGER_OWN_NODES is a comma list. Empty → nothing free (pure pay)."""
        if _truthy(os.environ.get("NAKSHATRA_LEDGER_OWN_ALL", "")):
            return [c.get("node_id") for c in receipt.get("chain", []) if c.get("node_id")]
        nodes = os.environ.get("NAKSHATRA_LEDGER_OWN_NODES", "").strip()
        return [n.strip() for n in nodes.split(",") if n.strip()] if nodes else []

    # ---- transport ----
    def _get(self, path: str):
        with urllib.request.urlopen(self.url + path, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _post(self, path: str, obj: dict):
        req = urllib.request.Request(self.url + path, data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())
