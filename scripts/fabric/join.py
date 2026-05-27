"""Fabric /join client — the worker-side consumer of sthambha's join
handshake (``~/sthambha/docs/network-fabric.md`` §4 + §6, server-side
implementation at ``~/sthambha/sthambha/server.py:1558``).

What this module does:

  - POSTs ``/join`` to the pillar with the worker's capability
    declaration, signed under the worker's Ed25519 key (OWNER tier per
    ADR 0006 §7 — anyone else would walk away with the keyring).
  - Parses the rich response into typed NamedTuples (forward /
    backward neighbor blocks, optional WireGuard block, optional NAT-
    traversal block inside it, optional sandbox spec, peer identity).
  - Exposes the per-pair AES-128-GCM key as an ``(self, neighbor) →
    16 bytes`` mapping that ``FabricLink`` (Phase B) consumes.
  - Schedules transparent rekey at ``rotate_at`` via a background
    daemon thread that re-POSTs /join, refreshes the keyring under a
    lock, and signals callers that ``FabricLink.send_seq`` should
    reset to 0 per schema §5.

What this module deliberately doesn't do:

  - Enforce the sandbox spec (open question 4 in the sprint plan):
    parsed + logged + audited only; container spawn + cgroup +
    seccomp wiring is the L4 sandbox supervisor's job.
  - Bring up the WireGuard tunnel (open question 5): parsed + logged
    only; tunnel bringup is a Mode B operational sprint.
  - Boot-policy decisions: the 404 (no plan) case raises; ``worker.py``
    decides whether ``--transport=fabric`` should refuse to start
    (open question 2 — recommended "refuse boot" so the flag is the
    upgrade signal).

Per ``[[reference_nakshatra_layout]]``, this lives at
``scripts/fabric/join.py`` so it can ``import nakshatra_auth as auth``
from the same ``scripts/`` sys.path entry every other module uses.
"""
from __future__ import annotations

import json
import ssl
import sys
import threading
import time
from pathlib import Path
from typing import NamedTuple, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

# Same scripts/ sys.path entry the worker uses — keeps the import
# shape uniform across the fabric package and the rest of nakshatra.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nakshatra_auth as auth  # noqa: E402

try:
    import nakshatra_audit as _audit_mod
    _AUDIT_AVAILABLE = True
except ImportError:
    _audit_mod = None
    _AUDIT_AVAILABLE = False


# ── Response types ──────────────────────────────────────────────────


class NeighborBlock(NamedTuple):
    """One forward or backward neighbor from the active chain plan, as
    served by ``/join`` and ``/fabric/neighbors`` (see
    ``sthambha/plan_store.py:neighbors_for_peer``).

    ``key_hex`` is the per-pair AES-128 key, hex-encoded. Empty when
    the keyring lookup failed pillar-side (defensive — should not
    happen in a healthy plan)."""
    peer_id: str
    address: str
    chosen_transport: str
    key_hex: str


class WireGuardPeerEntry(NamedTuple):
    peer_id: str
    public_key: str
    allowed_ips: str
    endpoint: str


class NatTraversalEntry(NamedTuple):
    peer_id: str
    my_endpoint: str
    peer_endpoint: str
    punch_token: str


class WireGuardBlock(NamedTuple):
    """Bare-WireGuard config for cross-site neighbors. Present in the
    /join response only when at least one neighbor is cross-site +
    declares a wireguard_pubkey + public_endpoint. ADR 0005 #5 /
    network-fabric §10 step 5.

    ``nat_traversal`` is server-side embedded in the WG block when any
    cross-site neighbor declared a public_endpoint (server.py:155);
    empty list when no NAT punching is needed."""
    interface_address: str
    peers: list[WireGuardPeerEntry]
    nat_traversal: list[NatTraversalEntry]


class SandboxSpec(NamedTuple):
    """Per-peer sandbox profile (network-fabric.md §11.1, Mode C
    readiness). Pillar emits; worker host's supervisor enforces. This
    module logs + audits the spec but does NOT enforce — enforcement
    is a separate L4 sandbox-supervisor sprint."""
    seccomp_profile: str
    cpu_threads_limit: int
    ram_limit_gb: float
    allowed_egress: list[str]
    layer_cache_readonly_paths: list[str]
    multi_tenant_isolation: str
    mode_c_compatible: bool


class PeerIdentity(NamedTuple):
    node_id: str
    public_key_hex: str


class JoinResponse(NamedTuple):
    """Parsed /join response. Every block except neighbors + identity
    is optional — same-site all-LAN topologies have no WG block, Mode
    A topologies pre-§11.1 have no sandbox spec."""
    plan_id: str
    rotate_at: float
    forward: Optional[NeighborBlock]
    backward: Optional[NeighborBlock]
    wireguard: Optional[WireGuardBlock]
    sandbox: Optional[SandboxSpec]
    peer_identity: PeerIdentity


# ── Errors ──────────────────────────────────────────────────────────


class JoinError(Exception):
    """Base for /join failures so generic callers can catch the
    whole family with one line."""


class NoPlanError(JoinError):
    """Pillar returned 404 — no active chain plan covers this peer
    yet. Caller decides what to do: ``worker.py --transport=fabric``
    refuses boot (open question 2), other callers may retry later.

    Pillar's exact 404 body is ``{"error": "no active chain plan
    covers peer 'X'; join handshake requires a plan to be live
    first"}``; the error message preserves the pillar's text so an
    operator running `journalctl` sees what they'd see from a manual
    curl."""


class AuthDenied(JoinError):
    """Pillar returned 403. Per ADR 0006 §7 + server.py:1571, /join
    is OWNER tier — the request signer must equal the joining peer's
    node_id. A 403 here usually means the worker is signing with the
    wrong key (e.g. key rotation without re-registering)."""


# ── Defensive parse helpers ─────────────────────────────────────────


def _opt_str(v) -> str:
    return str(v).strip() if v else ""


def _opt_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _opt_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _opt_bool(v) -> bool:
    return v is True


def _opt_list_of_str(v) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(x) for x in v if x]


def _parse_neighbor(block: Optional[dict]) -> Optional[NeighborBlock]:
    """A neighbor block of ``None`` is the chain-end case (the first
    worker has no backward, the last has no forward). Pillar emits
    ``None`` for both fields explicitly per server.py:137. Any other
    falsy / non-dict value is treated equivalently to a missing
    neighbor — defensive parse never crashes on a misshapen response."""
    if not isinstance(block, dict):
        return None
    return NeighborBlock(
        peer_id=_opt_str(block.get("peer_id")),
        address=_opt_str(block.get("address")),
        chosen_transport=_opt_str(block.get("chosen_transport")),
        key_hex=_opt_str(block.get("key_hex")),
    )


def _parse_wireguard(block: Optional[dict]) -> Optional[WireGuardBlock]:
    if not isinstance(block, dict):
        return None
    peers_raw = block.get("peers") or []
    peers: list[WireGuardPeerEntry] = []
    if isinstance(peers_raw, list):
        for p in peers_raw:
            if not isinstance(p, dict):
                continue
            peers.append(WireGuardPeerEntry(
                peer_id=_opt_str(p.get("peer_id")),
                public_key=_opt_str(p.get("public_key")),
                allowed_ips=_opt_str(p.get("allowed_ips")),
                endpoint=_opt_str(p.get("endpoint")),
            ))
    nat_raw = block.get("nat_traversal") or []
    nat: list[NatTraversalEntry] = []
    if isinstance(nat_raw, list):
        for n in nat_raw:
            if not isinstance(n, dict):
                continue
            nat.append(NatTraversalEntry(
                peer_id=_opt_str(n.get("peer_id")),
                my_endpoint=_opt_str(n.get("my_endpoint")),
                peer_endpoint=_opt_str(n.get("peer_endpoint")),
                punch_token=_opt_str(n.get("punch_token")),
            ))
    return WireGuardBlock(
        interface_address=_opt_str(block.get("interface_address")),
        peers=peers,
        nat_traversal=nat,
    )


def _parse_sandbox(block: Optional[dict]) -> Optional[SandboxSpec]:
    if not isinstance(block, dict):
        return None
    return SandboxSpec(
        seccomp_profile=_opt_str(block.get("seccomp_profile")),
        cpu_threads_limit=_opt_int(block.get("cpu_threads_limit")),
        ram_limit_gb=_opt_float(block.get("ram_limit_gb")),
        allowed_egress=_opt_list_of_str(block.get("allowed_egress")),
        layer_cache_readonly_paths=_opt_list_of_str(
            block.get("layer_cache_readonly_paths")),
        multi_tenant_isolation=_opt_str(
            block.get("multi_tenant_isolation")) or "exclusive",
        mode_c_compatible=_opt_bool(block.get("mode_c_compatible")),
    )


def _parse_peer_identity(block: Optional[dict]) -> PeerIdentity:
    """Pillar always emits peer_identity on success; fall back to an
    empty PeerIdentity rather than raise if the field is somehow
    absent — defensive parse stance, and the caller can still inspect
    .node_id == \"\" to detect the bug."""
    if not isinstance(block, dict):
        return PeerIdentity(node_id="", public_key_hex="")
    return PeerIdentity(
        node_id=_opt_str(block.get("node_id")),
        public_key_hex=_opt_str(block.get("public_key_hex")),
    )


def parse_join_response(data: dict) -> JoinResponse:
    """Public for tests + for callers that have a pre-fetched body
    (e.g. replay of an audit-captured response). Strictly defensive —
    a malformed top-level value yields default-empty fields rather
    than raising. Real I/O errors raise from :meth:`JoinClient.join`,
    not here."""
    return JoinResponse(
        plan_id=_opt_str(data.get("plan_id")),
        rotate_at=_opt_float(data.get("rotate_at")),
        forward=_parse_neighbor(data.get("forward")),
        backward=_parse_neighbor(data.get("backward")),
        wireguard=_parse_wireguard(data.get("wireguard")),
        sandbox=_parse_sandbox(data.get("sandbox")),
        peer_identity=_parse_peer_identity(data.get("peer_identity")),
    )


# ── Sandbox / WG audit-on-parse ─────────────────────────────────────


def _emit_audit(event: str, **payload) -> None:
    """Fire-and-forget audit emit. Mirror of the ``_audit`` helper
    in worker.py — no-op when the audit module isn't importable."""
    if _AUDIT_AVAILABLE:
        try:
            _audit_mod.audit(event, **payload)
        except Exception:
            pass


def _log_blocks(resp: JoinResponse, *,
                 node_id: str, pillar_url: str) -> None:
    """Print + audit the optional blocks (sandbox + WG) so operators
    see what the pillar issued even when this sprint doesn't enforce
    or bring up. The next-sprint enforcement code reads the same
    audit events to verify it received the right spec.

    Failure-soft: print failures fall through to audit; audit
    failures are swallowed by ``_emit_audit``. Never crashes /join.
    """
    if resp.sandbox is not None:
        sb = resp.sandbox
        print(f"[fabric/join] sandbox spec received (NOT enforced "
              f"this sprint): seccomp={sb.seccomp_profile!r} "
              f"cpu_threads={sb.cpu_threads_limit} "
              f"ram={sb.ram_limit_gb:.1f}GiB "
              f"isolation={sb.multi_tenant_isolation} "
              f"mode_c={sb.mode_c_compatible}",
              flush=True)
        _emit_audit(
            "fabric_sandbox_spec_received",
            node_id=node_id,
            pillar_url=pillar_url,
            plan_id=resp.plan_id,
            seccomp_profile=sb.seccomp_profile,
            cpu_threads_limit=sb.cpu_threads_limit,
            ram_limit_gb=sb.ram_limit_gb,
            multi_tenant_isolation=sb.multi_tenant_isolation,
            mode_c_compatible=sb.mode_c_compatible,
            allowed_egress_count=len(sb.allowed_egress),
        )
    if resp.wireguard is not None:
        wg = resp.wireguard
        print(f"[fabric/join] WireGuard block received (tunnel NOT "
              f"brought up this sprint): interface_addr="
              f"{wg.interface_address} peers={len(wg.peers)} "
              f"nat_traversal_entries={len(wg.nat_traversal)}",
              flush=True)
        _emit_audit(
            "fabric_wireguard_block_received",
            node_id=node_id,
            pillar_url=pillar_url,
            plan_id=resp.plan_id,
            interface_address=wg.interface_address,
            peer_count=len(wg.peers),
            nat_traversal_count=len(wg.nat_traversal),
        )


# ── JoinClient ───────────────────────────────────────────────────────


# How early before ``rotate_at`` to fire the rekey. Schema §5 says
# rotation IS re-plan (no mid-session rekey), so missing the deadline
# means a chain-stall window of "until the next /join succeeds";
# rolling 60s early keeps the cluster ahead of expiry under clock
# skew. Configurable via the env override below.
DEFAULT_REKEY_SLACK_S = 60.0


class JoinClient:
    """One worker's view of the pillar's /join + rekey lifecycle.

    Typical use (from ``worker.py`` boot in Phase D):

        jc = JoinClient(
            pillar_url="http://node-pi:7777",
            node_id="mac3-1234",
            priv_key=worker_priv_bytes,
            capability_declaration={
                "node_id": "mac3-1234",
                "address": "203.0.113.12:5560",
                "public_key_hex": "...",
                "fabric": {...},  # FabricCapabilities dict
            },
        )
        resp = jc.join()                   # raises NoPlanError on 404
        jc.start_rekey_loop()              # daemon thread
        keyring = jc.keyring()             # {(self, neighbor): key_bytes}

    Thread-safety: ``keyring()`` and ``current()`` take the internal
    lock; ``join()`` and ``stop()`` are caller-serialised (one boot
    path, one shutdown path). The background rekey thread holds the
    lock only while swapping references.
    """

    def __init__(
        self,
        pillar_url: str,
        node_id: str,
        priv_key: bytes,
        capability_declaration: dict,
        *,
        pillar_spki_hash: Optional[str] = None,
        rekey_slack_s: float = DEFAULT_REKEY_SLACK_S,
        request_timeout_s: float = 10.0,
        on_rekey: Optional[callable] = None,
    ):
        if len(priv_key) != 32:
            raise ValueError(
                f"worker Ed25519 priv key must be 32 bytes; got {len(priv_key)}"
            )
        self._pillar_url = (pillar_url or "").rstrip("/")
        self._node_id = node_id
        self._priv_key = priv_key
        self._capability = dict(capability_declaration or {})
        # Allow caller to override the body's node_id but default to
        # ours — protects against the operator typoing one of the two
        # places node_id appears.
        self._capability.setdefault("node_id", node_id)
        self._pillar_spki_hash = pillar_spki_hash
        self._rekey_slack_s = rekey_slack_s
        self._request_timeout_s = request_timeout_s
        self._on_rekey = on_rekey

        self._lock = threading.Lock()
        self._current: Optional[JoinResponse] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── join + parse ──────────────────────────────────────────────

    def join(self) -> JoinResponse:
        """Synchronous /join POST. Returns the parsed response; also
        stores it as ``self._current`` so ``keyring()`` works after
        the first call.

        Raises:
          - :class:`NoPlanError` on HTTP 404 (no chain plan covers
            this peer yet)
          - :class:`AuthDenied` on HTTP 403 (caller-signing mismatch)
          - :class:`JoinError` on any other HTTP / network failure
        """
        url = f"{self._pillar_url}/join"
        body = json.dumps(self._capability).encode("utf-8")
        header_val, _ts = auth.build_signed_envelope(
            self._priv_key, self._node_id, "POST", "/join", body,
        )
        context: Optional[ssl.SSLContext] = None
        if url.startswith("https://"):
            context = auth.build_pillar_ssl_context(self._pillar_spki_hash)
        req = urlrequest.Request(
            url, data=body,
            headers={
                "Authorization": header_val,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(
                req, timeout=self._request_timeout_s, context=context,
            ) as http_resp:
                raw = http_resp.read().decode("utf-8")
        except urlerror.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if e.code == 404:
                raise NoPlanError(err_body or f"/join 404: {e}") from e
            if e.code == 403:
                raise AuthDenied(err_body or f"/join 403: {e}") from e
            raise JoinError(
                f"/join failed: HTTP {e.code} {err_body[:200]}"
            ) from e
        except (urlerror.URLError, OSError, TimeoutError) as e:
            raise JoinError(f"/join transport failure: {e}") from e
        try:
            data = json.loads(raw)
        except ValueError as e:
            raise JoinError(f"/join body not JSON: {e}") from e
        if not isinstance(data, dict):
            raise JoinError(
                f"/join body root must be object, got {type(data).__name__}"
            )
        resp = parse_join_response(data)
        with self._lock:
            self._current = resp
        _log_blocks(resp,
                    node_id=self._node_id, pillar_url=self._pillar_url)
        return resp

    # ── Snapshot accessors ────────────────────────────────────────

    def current(self) -> Optional[JoinResponse]:
        """Most-recent successful /join response, or None if no
        :meth:`join` has succeeded yet."""
        with self._lock:
            return self._current

    def keyring(self) -> dict[tuple[str, str], bytes]:
        """Decoded per-pair keys keyed by ``(self.node_id,
        neighbor_id)``. Returns ``{}`` when no /join has succeeded
        yet OR when both neighbor slots are empty (single-worker
        chain, which is rare but legal at the schema layer).

        Malformed hex (e.g. key_hex = \"\") is silently dropped — the
        ``FabricLink`` constructor would reject it anyway and the
        operator would learn at link bringup; surfacing it here would
        force every caller to reach into NeighborBlock to figure out
        what went missing.
        """
        with self._lock:
            resp = self._current
        if resp is None:
            return {}
        out: dict[tuple[str, str], bytes] = {}
        for nb in (resp.forward, resp.backward):
            if nb is None:
                continue
            if not nb.peer_id or not nb.key_hex:
                continue
            try:
                key = bytes.fromhex(nb.key_hex)
            except ValueError:
                continue
            if len(key) != 16:
                continue
            out[(self._node_id, nb.peer_id)] = key
        return out

    # ── Rekey loop ────────────────────────────────────────────────

    def start_rekey_loop(self) -> None:
        """Spawn a daemon thread that re-POSTs /join shortly before
        ``rotate_at`` expiry. Idempotent (re-call is a no-op while a
        thread is already running). The thread holds no resources
        beyond the JoinClient itself; ``stop()`` cleanly joins it."""
        if self._thread is not None:
            return
        if self._current is None:
            raise RuntimeError(
                "start_rekey_loop() requires a successful join() first"
            )
        self._thread = threading.Thread(
            target=self._rekey_loop, daemon=True, name="fabric-join-rekey",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the rekey loop to exit. Doesn't join — daemon thread,
        and the wait may be longer than the caller wants if /join is
        in flight. Idempotent."""
        self._stop.set()

    def _rekey_loop(self) -> None:
        """Sleep until ``rotate_at - slack``, then re-POST /join.
        On success: swap ``self._current`` under the lock and fire
        ``on_rekey`` callback (e.g. to reset FabricLink.send_seq per
        schema §5). On failure: log + back off + retry. Never raises.
        """
        while not self._stop.is_set():
            with self._lock:
                rotate_at = self._current.rotate_at if self._current else 0.0
            sleep_until = rotate_at - self._rekey_slack_s
            wait_s = max(0.0, sleep_until - time.time())
            if self._stop.wait(timeout=wait_s):
                return
            try:
                new = self.join()
            except JoinError as e:
                sys.stderr.write(
                    f"[fabric/join] rekey failed: {e}; retrying in 30s\n"
                )
                _emit_audit(
                    "fabric_rekey_failed",
                    node_id=self._node_id,
                    pillar_url=self._pillar_url,
                    error=str(e),
                )
                if self._stop.wait(timeout=30.0):
                    return
                continue
            _emit_audit(
                "fabric_rekey_succeeded",
                node_id=self._node_id,
                pillar_url=self._pillar_url,
                plan_id=new.plan_id,
                rotate_at=new.rotate_at,
            )
            if self._on_rekey is not None:
                try:
                    self._on_rekey(new)
                except Exception as e:
                    sys.stderr.write(
                        f"[fabric/join] on_rekey callback raised: {e}\n"
                    )
