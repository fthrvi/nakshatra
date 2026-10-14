"""
ledger_client.py — serve-side hook into the neuron credit-ledger service (the integration seam).

Two calls the inference serve makes around a run:
  • gate(est_cost)      — before: GET /can_consume   (may DENY only when enforced)
  • settle(receipt_path)— after:  POST /settle        (credits the workers from the #20 receipt)

SAFETY (this guards Prithvi's live brain):
  • DEFAULT-OFF: with NAKSHATRA_CREDITS unset, every method is a total no-op — the serve is
    byte-for-byte unchanged.
  • FAIL-OPEN BY DEFAULT: any ledger error (down, slow, 5xx) → the run PROCEEDS. The hub's
    existing behaviour is unchanged. An explicit v2 402 denies; a legacy 200 denial is enforced
    only when NAKSHATRA_CREDITS_ENFORCE=1.
  • OUTSIDER ENFORCE MODE: NAKSHATRA_LEDGER_MODE=enforce fails closed on a missing key or any
    transport/protocol error. It also omits the legacy requester assertion: the ledger derives the
    operator from the signing key's roster row.

Env:
  NAKSHATRA_CREDITS=1            enable the hook (still fail-open; advisory unless enforced)
  NAKSHATRA_CREDITS_ENFORCE=1   enforce legacy 200-body denials (v2 uses HTTP 402)
  NAKSHATRA_LEDGER_URL          default http://127.0.0.1:8097
  NAKSHATRA_LEDGER_MODE         default fail-open; "enforce" denies on every ledger failure
  NAKSHATRA_LEDGER_SPKI_SHA256  optional HTTPS SPKI pin (required by `nak gateway on`)
  NAKSHATRA_LEDGER_KEYID        optional roster id for the node key (default: public-key hex)
  NAKSHATRA_LEDGER_REQUESTER    legacy fail-open requester (default "self"; ignored in enforce mode)
  NAKSHATRA_LEDGER_TIMEOUT      seconds (default 2)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import nakshatra_auth


_LOG = logging.getLogger(__name__)


class LedgerUnavailable(RuntimeError):
    """Strict accounting could not prove that the run was chargeable/settled."""


def _truthy(v: str) -> bool:
    return (v or "").strip() in ("1", "true", "True", "yes", "on")


class LedgerHook:
    def __init__(self, *, url: str | None = None, enabled: bool | None = None,
                 mode: str | None = None, spki_hash: str | None = None):
        self.enabled = (_truthy(os.environ.get("NAKSHATRA_CREDITS", ""))
                        if enabled is None else bool(enabled))
        self.enforce = _truthy(os.environ.get("NAKSHATRA_CREDITS_ENFORCE", ""))
        self.mode = (mode if mode is not None else
                     os.environ.get("NAKSHATRA_LEDGER_MODE", "fail-open")).strip().lower()
        self.fail_closed = self.mode == "enforce"
        self.url = (url or os.environ.get(
            "NAKSHATRA_LEDGER_URL", "http://127.0.0.1:8097")).rstrip("/")
        self.spki_hash = (spki_hash if spki_hash is not None else os.environ.get(
            "NAKSHATRA_LEDGER_SPKI_SHA256", "")).strip().lower() or None
        self.requester = os.environ.get("NAKSHATRA_LEDGER_REQUESTER", "self")
        self._private_key = None
        self.keyid = None
        self._last_signature_refusal_log = None
        self._rejected_receipts = set()
        try:
            self.timeout = float(os.environ.get("NAKSHATRA_LEDGER_TIMEOUT", "2") or 2)
        except ValueError:
            self.timeout = 2.0
        if self.enabled:
            try:
                self._private_key, public_key_hex = nakshatra_auth.load_worker_key()
                self.keyid = (
                    os.environ.get("NAKSHATRA_LEDGER_KEYID", "").strip()
                    or public_key_hex
                )
            except Exception as exc:  # noqa: BLE001 - gate applies the selected policy
                policy = "fail-closed" if self.fail_closed else "fail-open"
                _LOG.warning("ledger signing key unavailable (%s); %s", exc, policy)

    @property
    def wants_receipt(self) -> bool:
        return self.enabled

    def gate(self, est_cost: int):
        """Return (allow: bool, reason: str). Allows when disabled, on any error (fail-open), or
        when the ledger says yes / legacy enforcement is off. An explicit v2 402 denies."""
        if not self.enabled:
            return True, ""
        if self._private_key is None and self.fail_closed:
            return False, "ledger signing key unavailable"
        if self._private_key is None:
            return True, "ledger signing key unavailable; fail-open"
        try:
            params = ({"cost": int(est_cost)} if self.fail_closed else
                      {"account": self.requester, "cost": int(est_cost)})
            q = urllib.parse.urlencode(params)
            r = self._get(f"/can_consume?{q}")
            ok = bool(r.get("ok"))
        except urllib.error.HTTPError as e:
            if e.code == 401:
                self._log_signature_refusal()
                if self.fail_closed:
                    return False, "ledger refused this node's signature"
                return True, "ledger refused signature; fail-open"
            if e.code == 402:
                detail = self._http_error_json(e)
                return False, (
                    f"insufficient credits (balance {detail.get('balance')}, "
                    f"cost {int(est_cost)})"
                )
            if self.fail_closed:
                return False, f"ledger HTTP {e.code}"
            return True, f"ledger HTTP {e.code}; fail-open"
        except Exception as e:
            if self.fail_closed:
                return False, f"ledger unreachable ({e})"
            return True, f"ledger unreachable ({e}); fail-open"
        if ok:
            return True, ""
        if self.fail_closed:
            return False, f"insufficient credits (balance {r.get('balance')}, cost {int(est_cost)})"
        if not self.enforce:
            return True, "insufficient credits (advisory; enforcement off)"
        return False, f"insufficient credits (balance {r.get('balance')}, cost {int(est_cost)})"

    def settle(self, receipt_path: str):
        """Credit the workers from a #20 run receipt. Fail-open: a settle failure NEVER affects
        the served reply. Returns the delta dict or None."""
        if not self.enabled:
            return None
        if self._private_key is None:
            if self.fail_closed:
                raise LedgerUnavailable("ledger signing key unavailable")
            return None
        try:
            with open(receipt_path) as f:
                receipt = json.load(f)
            receipt_id = self._receipt_id(receipt)
            if receipt_id in self._rejected_receipts:
                if self.fail_closed:
                    raise LedgerUnavailable(
                        f"ledger previously rejected receipt {receipt_id}"
                    )
                return None
            body = {"receipt": receipt}
            if not self.fail_closed:
                body["requester"] = self.requester
            own = [] if self.fail_closed else self._own_nodes(receipt)
            if own:
                body["own_nodes"] = own        # local infra → free
            # identity-binding: attach the holder-of-key-PROVEN accounts (each worker Ed25519-signed the
            # stage it served). Advisory — the ledger credits BOUND accounts instead of trusting the
            # coordinator's say-so. Empty for legacy receipts without worker_signatures (non-breaking).
            try:
                from identity_binding import creditable_accounts
                from roster import load_roster
                # ⚠️ `pinned` is REQUIRED now, and what we pass decides who can be paid.
                # NAKSHATRA_ROSTER unset ⇒ pin an EMPTY roster, which credits nobody. That is
                # the fail-closed reading and it is deliberate: the alternative — accepting any
                # key when no roster is configured — is precisely the optional-pin defect that
                # let an unregistered key certify any node over any span.
                # ⚠️⚠️ This call used to omit `pinned` entirely and sat inside a bare
                # `except Exception: pass`, so when the argument became required the breakage
                # was INVISIBLE: settle() would simply stop attaching accounts and credit
                # nobody, with nothing in any log. The narrower excepts below exist so a
                # roster that fails to load is distinguishable from a receipt with no
                # signatures — silence here is what made the first bug cost weeks.
                roster_path = os.environ.get("NAKSHATRA_ROSTER", "").strip()
                pinned = load_roster(roster_path) if roster_path else {}
                accts, problems = creditable_accounts(receipt, pinned=pinned)
                if accts:
                    body["creditable_accounts"] = accts
                elif problems and roster_path:
                    body["participation_problems"] = problems[:8]
            except (FileNotFoundError, ValueError) as e:
                # a roster that will not load must not silently become "credit nobody"
                body["participation_problems"] = [f"roster unusable: {e}"]
            except ImportError:
                pass                      # identity_binding/roster absent — legacy deployment
            try:
                return self._post("/settle", body)
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    self._log_signature_refusal()
                elif e.code in (409, 422):
                    self._rejected_receipts.add(receipt_id)
                    outcome = "disputed" if e.code == 409 else "forged"
                    _LOG.warning(
                        "ledger rejected %s receipt (%d); will not retry receipt %s",
                        outcome, e.code, receipt_id,
                    )
                if self.fail_closed:
                    raise LedgerUnavailable(f"ledger settle HTTP {e.code}") from e
                return None
        except LedgerUnavailable:
            raise
        except Exception as e:
            if self.fail_closed:
                raise LedgerUnavailable(f"ledger settle failed ({e})") from e
            return None

    def balance(self) -> dict:
        """Return the signing operator's balance/tier. No account selector is ever sent."""
        if not self.enabled:
            raise LedgerUnavailable("ledger accounting is disabled")
        if self._private_key is None:
            raise LedgerUnavailable("ledger signing key unavailable")
        try:
            value = self._get("/balance")
        except urllib.error.HTTPError as e:
            if e.code == 401:
                self._log_signature_refusal()
            raise LedgerUnavailable(f"ledger balance HTTP {e.code}") from e
        except Exception as e:
            raise LedgerUnavailable(f"ledger balance unavailable ({e})") from e
        if not isinstance(value, dict):
            raise LedgerUnavailable("ledger returned an invalid balance response")
        return value

    @staticmethod
    def _receipt_id(receipt: dict) -> str:
        canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _http_error_json(error: urllib.error.HTTPError) -> dict:
        try:
            value = json.loads(error.read())
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _log_signature_refusal(self) -> None:
        now = time.monotonic()
        if (
            self._last_signature_refusal_log is None
            or now - self._last_signature_refusal_log >= 60.0
        ):
            _LOG.warning("ledger refused signature (keyid %s)", self.keyid)
            self._last_signature_refusal_log = now

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
        authorization, _ = nakshatra_auth.build_signed_envelope(
            self._private_key, self.keyid, "GET", path, b""
        )
        req = urllib.request.Request(
            self.url + path,
            headers={"Authorization": authorization},
            method="GET",
        )
        with self._urlopen(req) as r:
            return json.loads(r.read())

    def _post(self, path: str, obj: dict):
        raw = json.dumps(obj).encode()
        authorization, _ = nakshatra_auth.build_signed_envelope(
            self._private_key, self.keyid, "POST", path, raw
        )
        req = urllib.request.Request(
            self.url + path,
            data=raw,
            headers={
                "Authorization": authorization,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with self._urlopen(req) as r:
            return json.loads(r.read())

    def _urlopen(self, request):
        if self.spki_hash:
            return nakshatra_auth.open_pillar_url(
                request, timeout=self.timeout, expected_spki_hash=self.spki_hash)
        return urllib.request.urlopen(request, timeout=self.timeout)
