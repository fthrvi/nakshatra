"""
junction — the admission-gated Nakshatra operator junction (multi-tenant).

The blind byte-forwarder already exists (`relay.py` `RendezvousRelay`: pairs two sockets by
`rendezvous_id`, forwards raw bytes, NEVER IP-routes). That alone makes the junction multi-tenant-safe
at L3 — there is no shared routing table, so the "every operator on 10.42.0.0/24" address-collision
(the audit's 🔴) cannot happen: a peer reaches ONLY the partner it was paired with, end-to-end
encrypted (the relay sees ciphertext). What `RendezvousRelay` lacks is a DOOR POLICY: it pairs anyone.

This adds the missing gate: an **admission preamble** before the rendezvous. A connecting peer presents
a signed admission request; the junction calls the operator's admission policy
(`trisul/infra/control-plane/admission.py`) → **default-deny** unless the peer is on the operator's
roster and proves its Ed25519 identity. Admitted → delegate to the existing pairing (zero edits to
`relay.py`). Denied → close, no channel.

Wire (peer → junction):  [u32 len][admission-req JSON]  then the normal  [rendezvous_id][data…].
The admission policy is INJECTABLE (`admit_fn`) so this is testable without the control-plane present;
the default lazy-loads `admission.py` from `$NAKSHATRA_ADMISSION_DIR` (default ~/trisul/infra/control-plane).

Plan: trisul/plans/2026-06-14-sthambha-admission-gateway.md + the multi-tenant audit.
"""
from __future__ import annotations
import os, json, struct, socket, logging
from pathlib import Path
from typing import Callable, Optional

from relay import RendezvousRelay, _send_id, _recv_id, _recv_exact, DEFAULT_PORT

log = logging.getLogger("junction")
JUNCTION_PORT = int(os.environ.get("NAKSHATRA_JUNCTION_PORT", "9778"))
_MAX_PREAMBLE = 8192


# ── admission preamble framing ────────────────────────────────────────────────────────────────────
def _send_admission(sock: socket.socket, request: dict) -> None:
    body = json.dumps(request, separators=(",", ":")).encode()
    sock.sendall(struct.pack(">I", len(body)) + body)


def _recv_admission(sock: socket.socket) -> Optional[dict]:
    hdr = _recv_exact(sock, 4)
    if not hdr:
        return None
    (n,) = struct.unpack(">I", hdr)
    if n <= 0 or n > _MAX_PREAMBLE:
        return None
    body = _recv_exact(sock, n)
    if not body:
        return None
    try:
        return json.loads(body.decode())
    except Exception:
        return None


# ── default admission policy (lazy cross-repo import — like cap_guard) ──────────────────────────────
def _default_admit() -> Callable[[dict], dict]:
    adm_dir = os.environ.get("NAKSHATRA_ADMISSION_DIR", str(Path.home() / "trisul" / "infra" / "control-plane"))

    def admit(request: dict) -> dict:
        try:
            import sys
            if adm_dir not in sys.path:
                sys.path.insert(0, adm_dir)
            import admission as adm  # noqa
            peers, models = adm.load_peers(), adm.load_models()
            return adm.admit(request, peers=peers, models=models)
        except Exception as e:
            # FAIL-CLOSED: if the policy can't load, deny (never fail-open the door).
            return {"admitted": False, "reason": f"admission policy unavailable: {e}"}

    return admit


# ── the gated junction ──────────────────────────────────────────────────────────────────────────────
class AdmissionGatedJunction(RendezvousRelay):
    """RendezvousRelay + a default-deny admission door. Untrusted forwarder once admitted."""

    def __init__(self, host: str = "::", port: int = JUNCTION_PORT,
                 admit_fn: Optional[Callable[[dict], dict]] = None) -> None:
        super().__init__(host, port)
        self.admit_fn = admit_fn or _default_admit()
        self.admitted = 0
        self.denied = 0

    def _handle(self, conn: socket.socket) -> None:
        # 1) ADMISSION DOOR — before any rendezvous. Default-deny.
        req = _recv_admission(conn)
        decision = None
        try:
            decision = self.admit_fn(req) if req is not None else {"admitted": False, "reason": "no preamble"}
        except Exception as e:
            decision = {"admitted": False, "reason": f"admit error: {e}"}
        if not decision or not decision.get("admitted"):
            self.denied += 1
            log.warning("JUNCTION deny: %s", (decision or {}).get("reason", "?"))
            try:
                conn.close()
            except OSError:
                pass
            return
        self.admitted += 1
        log.info("JUNCTION admit: tier=%s tenant=%s", decision.get("tier"), decision.get("tenant"))
        # 2) admitted → the normal blind rendezvous (reads rendezvous_id + pairs). No edits to relay.py.
        super()._handle(conn)


def connect_admitted(relay_host: str, relay_port: int, rendezvous_id: bytes,
                     admission_request: dict, timeout: float = 30.0) -> socket.socket:
    """Peer side: open → send the admission preamble → send the rendezvous_id → return the socket
    (paired once the partner arrives). Raises on connect failure; the junction closes the socket if denied."""
    s = socket.create_connection((relay_host, relay_port), timeout=timeout)
    _send_admission(s, admission_request)
    _send_id(s, rendezvous_id)
    s.settimeout(None)
    return s


def serve(host: str = "::", port: int = JUNCTION_PORT) -> None:
    j = AdmissionGatedJunction(host, port)
    p = j.start()
    print(f"[junction] admission-gated operator junction on [{host}]:{p} "
          f"(blind forwarder; default-deny door)", flush=True)
    try:
        j._stop.wait()
    except KeyboardInterrupt:
        j.stop()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Nakshatra admission-gated operator junction")
    ap.add_argument("--host", default="::")
    ap.add_argument("--port", type=int, default=JUNCTION_PORT)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    serve(a.host, a.port)
