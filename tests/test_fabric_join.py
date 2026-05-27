"""Phase C tests for ``scripts/fabric/join.py`` — pillar /join client +
response parser + keyring + rekey scheduling.

Covers the falsifiable checks from the sprint plan Phase C:

  * Mock urlopen for happy + partial + 404 + 403 + 5xx
  * Key extraction shape: ``{(self.node_id, neighbor_id): bytes16}``
  * Defensive parse: missing fields, wrong types, malformed key_hex
  * Sandbox + WireGuard blocks parsed AND audited on parse
  * Rekey-loop wiring: requires prior join(); stop is idempotent + non-blocking
  * 32-byte Ed25519 priv-key policy guard
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib import error as urlerror

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fabric import join as fj  # noqa: E402


PRIV = b"\x42" * 32                                  # any 32 bytes
NODE_A = "worker-a"
NODE_B = "worker-b"
KEY_HEX_AB = "aa" * 16                              # valid hex, 16 bytes
KEY_HEX_BA = "bb" * 16


# ── helpers ──────────────────────────────────────────────────────────


def _mock_urlopen(payload_dict: dict, status: int = 200):
    """Build a urlopen-mock context-manager. Mimics ``urlopen``'s
    ``__enter__ → response_obj`` shape so the production code's
    ``with urlopen(...) as r: r.read()`` path works unchanged."""
    body = json.dumps(payload_dict).encode("utf-8")
    cm = MagicMock()
    cm.__enter__.return_value = io.BytesIO(body)
    cm.__exit__.return_value = False
    return cm


def _http_error(code: int, body_text: str = "") -> urlerror.HTTPError:
    """Build an HTTPError whose .read() returns the given body."""
    fp = io.BytesIO(body_text.encode("utf-8"))
    return urlerror.HTTPError(
        url="http://pillar/join", code=code,
        msg=f"HTTP {code}", hdrs=None, fp=fp,
    )


def _client(**overrides) -> fj.JoinClient:
    """A JoinClient with sensible defaults; override per-test."""
    kwargs = dict(
        pillar_url="http://pillar:7777",
        node_id=NODE_A,
        priv_key=PRIV,
        capability_declaration={
            "node_id": NODE_A,
            "address": "203.0.113.12:5560",
            "public_key_hex": "0" * 64,
            "fabric": {"site_id": "lab", "transports_supported": ["raw_udp"]},
        },
    )
    kwargs.update(overrides)
    return fj.JoinClient(**kwargs)


HAPPY_RESPONSE = {
    "plan_id": "plan-2026-05-26-001",
    "rotate_at": 1_999_999_999.0,
    "forward": {
        "peer_id": NODE_B,
        "address": "203.0.113.11:5561",
        "chosen_transport": "raw_udp",
        "key_hex": KEY_HEX_AB,
    },
    "backward": None,
    "peer_identity": {
        "node_id": NODE_A,
        "public_key_hex": "0" * 64,
    },
}


# ── 1. Construction guards ──────────────────────────────────────────


def test_constructor_rejects_short_priv_key():
    """Ed25519 priv keys are 32 bytes. Fail loud — an operator passing
    a 16-byte AES key by mistake (fabric per-pair vs. worker identity)
    would otherwise sign garbage and the pillar would 403."""
    with pytest.raises(ValueError, match="32 bytes"):
        fj.JoinClient(
            pillar_url="http://x", node_id="a", priv_key=b"short",
            capability_declaration={},
        )


def test_constructor_defaults_capability_node_id():
    """The capability dict's node_id is auto-filled from the
    constructor's node_id if absent — protects against the operator
    setting one of the two places and forgetting the other."""
    c = fj.JoinClient(
        pillar_url="http://x", node_id="auto-fill",
        priv_key=PRIV,
        capability_declaration={},  # no node_id
    )
    assert c._capability["node_id"] == "auto-fill"


# ── 2. join() happy path ────────────────────────────────────────────


def test_join_happy_path_parses_response():
    c = _client()
    with patch.object(fj.urlrequest, "urlopen",
                       return_value=_mock_urlopen(HAPPY_RESPONSE)):
        resp = c.join()
    assert resp.plan_id == "plan-2026-05-26-001"
    assert resp.rotate_at == 1_999_999_999.0
    assert resp.forward is not None
    assert resp.forward.peer_id == NODE_B
    assert resp.forward.key_hex == KEY_HEX_AB
    assert resp.backward is None
    assert resp.peer_identity.node_id == NODE_A
    # current() reflects the just-fetched response
    assert c.current() == resp


def test_join_signs_and_posts_body():
    """Verify the POST goes to /join, carries the capability JSON,
    and includes a properly-formed Authorization header."""
    c = _client()
    captured = {}

    def fake_urlopen(req, *a, **kw):
        captured["url"] = req.full_url
        captured["method"] = req.method
        captured["body"] = req.data
        captured["auth"] = req.headers.get("Authorization")
        captured["content_type"] = req.headers.get("Content-type")
        return _mock_urlopen(HAPPY_RESPONSE)

    with patch.object(fj.urlrequest, "urlopen", side_effect=fake_urlopen):
        c.join()

    assert captured["url"] == "http://pillar:7777/join"
    assert captured["method"] == "POST"
    assert captured["content_type"] == "application/json"
    body = json.loads(captured["body"])
    assert body["node_id"] == NODE_A
    assert body["fabric"]["site_id"] == "lab"
    # AUTH_SCHEME is "Sthambha-Ed25519" (defined in nakshatra_auth) —
    # cross-repo contract per ADR 0006. Asserting prefix + keyid
    # shape guards against accidental rename.
    assert captured["auth"].startswith("Sthambha-Ed25519")
    assert 'keyid="worker-a"' in captured["auth"]


# ── 3. join() error paths ───────────────────────────────────────────


def test_join_404_raises_no_plan_error():
    """Pillar returns 404 when no chain plan covers this peer yet.
    Boot-policy decision (open question 2) is the caller's — this
    method just surfaces the typed exception."""
    body = '{"error": "no active chain plan covers peer \'a\'"}'
    with patch.object(fj.urlrequest, "urlopen",
                       side_effect=_http_error(404, body)):
        with pytest.raises(fj.NoPlanError) as ei:
            _client().join()
    # The pillar's exact error body propagates so an operator running
    # `journalctl` sees what they'd see from a manual curl.
    assert "no active chain plan" in str(ei.value)


def test_join_403_raises_auth_denied():
    """Per ADR 0006 §7 — /join is OWNER tier. A 403 means the signer
    doesn't match the joining peer's registered key."""
    with patch.object(fj.urlrequest, "urlopen",
                       side_effect=_http_error(403, "forbidden")):
        with pytest.raises(fj.AuthDenied):
            _client().join()


def test_join_5xx_raises_generic_join_error():
    """A 500 / 502 / 503 isn't structurally different from 4xx for
    our purposes — caller can't fix it; just surface."""
    with patch.object(fj.urlrequest, "urlopen",
                       side_effect=_http_error(500, "internal error")):
        with pytest.raises(fj.JoinError):
            _client().join()


def test_join_transport_failure_raises_join_error():
    """Network unreachable, pillar down, DNS fail — all map to a
    generic JoinError; the message carries the underlying cause."""
    with patch.object(fj.urlrequest, "urlopen",
                       side_effect=urlerror.URLError("connection refused")):
        with pytest.raises(fj.JoinError, match="transport"):
            _client().join()


def test_join_body_not_json_raises_join_error():
    """A pillar that returned 200 but somehow shipped non-JSON
    (proxy misconfiguration?) — fail with a clear message rather than
    crash."""
    cm = MagicMock()
    cm.__enter__.return_value = io.BytesIO(b"<html>not json</html>")
    cm.__exit__.return_value = False
    with patch.object(fj.urlrequest, "urlopen", return_value=cm):
        with pytest.raises(fj.JoinError, match="not JSON"):
            _client().join()


def test_join_body_root_not_dict_raises_join_error():
    """Pillar returned a JSON list instead of an object — should not
    crash the parser; raise cleanly."""
    cm = MagicMock()
    cm.__enter__.return_value = io.BytesIO(b'["not", "an", "object"]')
    cm.__exit__.return_value = False
    with patch.object(fj.urlrequest, "urlopen", return_value=cm):
        with pytest.raises(fj.JoinError, match="object"):
            _client().join()


# ── 4. Defensive parse — parse_join_response in isolation ───────────


def test_parse_response_with_all_optional_blocks_absent():
    """Forward/backward both None (single-worker chain), no wireguard,
    no sandbox. Should still parse — minimal valid response."""
    resp = fj.parse_join_response({
        "plan_id": "p", "rotate_at": 0,
        "forward": None, "backward": None,
        "peer_identity": {"node_id": "a", "public_key_hex": "0" * 64},
    })
    assert resp.forward is None
    assert resp.backward is None
    assert resp.wireguard is None
    assert resp.sandbox is None


def test_parse_response_with_full_wireguard_block():
    """Cross-site neighbors → WG block + nat_traversal embedded."""
    resp = fj.parse_join_response({
        "plan_id": "p", "rotate_at": 0,
        "forward": {"peer_id": "b", "address": "host:5560",
                    "chosen_transport": "raw_udp", "key_hex": KEY_HEX_AB},
        "backward": None,
        "wireguard": {
            "interface_address": "10.42.1.2/32",
            "peers": [
                {"peer_id": "b", "public_key": "pubkey-b",
                 "allowed_ips": "10.42.3.4/32",
                 "endpoint": "203.0.113.5:51820"},
            ],
            "nat_traversal": [
                {"peer_id": "b", "my_endpoint": "10.0.0.1:5561",
                 "peer_endpoint": "203.0.113.5:5561",
                 "punch_token": "deadbeef"},
            ],
        },
        "peer_identity": {"node_id": "a", "public_key_hex": "0" * 64},
    })
    assert resp.wireguard is not None
    assert resp.wireguard.interface_address == "10.42.1.2/32"
    assert len(resp.wireguard.peers) == 1
    assert resp.wireguard.peers[0].endpoint == "203.0.113.5:51820"
    assert len(resp.wireguard.nat_traversal) == 1
    assert resp.wireguard.nat_traversal[0].punch_token == "deadbeef"


def test_parse_response_with_sandbox_block():
    resp = fj.parse_join_response({
        "plan_id": "p", "rotate_at": 0,
        "forward": None, "backward": None,
        "sandbox": {
            "seccomp_profile": "fabric-strict-v1",
            "cpu_threads_limit": 8,
            "ram_limit_gb": 32.0,
            "allowed_egress": ["http://pillar:7777", "203.0.113.11:5561"],
            "layer_cache_readonly_paths": ["/var/cache/nakshatra/p.gguf"],
            "multi_tenant_isolation": "exclusive",
            "mode_c_compatible": True,
        },
        "peer_identity": {"node_id": "a", "public_key_hex": "0" * 64},
    })
    assert resp.sandbox is not None
    assert resp.sandbox.seccomp_profile == "fabric-strict-v1"
    assert resp.sandbox.cpu_threads_limit == 8
    assert resp.sandbox.ram_limit_gb == 32.0
    assert resp.sandbox.mode_c_compatible is True
    assert "http://pillar:7777" in resp.sandbox.allowed_egress


def test_parse_response_defensive_against_wrong_types():
    """A pillar that drifted from contract (or a malicious response)
    shouldn't crash the parser. Wrong types degrade to defaults, the
    operator sees missing fields downstream and investigates."""
    resp = fj.parse_join_response({
        "plan_id": 42,                       # int instead of str
        "rotate_at": "soon",                 # str instead of float
        "forward": "not a dict",
        "backward": ["not", "a", "dict"],
        "wireguard": 123,
        "sandbox": None,
        "peer_identity": "broken",
    })
    assert resp.plan_id == "42"              # str-coerced
    assert resp.rotate_at == 0.0             # str → float failed → 0
    assert resp.forward is None
    assert resp.backward is None
    assert resp.wireguard is None
    assert resp.sandbox is None
    assert resp.peer_identity.node_id == ""


def test_parse_response_sandbox_defaults_isolation_to_exclusive():
    """When the pillar omits multi_tenant_isolation, default to
    'exclusive' (safest setting). Matches the sthambha-side
    SandboxSpec field default."""
    resp = fj.parse_join_response({
        "plan_id": "p", "rotate_at": 0,
        "forward": None, "backward": None,
        "sandbox": {"seccomp_profile": "x"},  # everything else absent
        "peer_identity": {"node_id": "a", "public_key_hex": ""},
    })
    assert resp.sandbox is not None
    assert resp.sandbox.multi_tenant_isolation == "exclusive"


# ── 5. keyring() extraction ─────────────────────────────────────────


def test_keyring_returns_empty_before_first_join():
    c = _client()
    assert c.keyring() == {}


def test_keyring_extracts_forward_neighbor_key():
    c = _client()
    with patch.object(fj.urlrequest, "urlopen",
                       return_value=_mock_urlopen(HAPPY_RESPONSE)):
        c.join()
    assert c.keyring() == {(NODE_A, NODE_B): b"\xAA" * 16}


def test_keyring_extracts_both_forward_and_backward():
    """Mid-chain worker has both neighbors; keyring carries entries
    for each."""
    response = dict(HAPPY_RESPONSE)
    response["backward"] = {
        "peer_id": "worker-z",
        "address": "host-z:5559",
        "chosen_transport": "raw_udp",
        "key_hex": KEY_HEX_BA,
    }
    c = _client()
    with patch.object(fj.urlrequest, "urlopen",
                       return_value=_mock_urlopen(response)):
        c.join()
    kr = c.keyring()
    assert (NODE_A, NODE_B) in kr
    assert ("worker-a", "worker-z") in kr
    assert kr[(NODE_A, NODE_B)] == b"\xAA" * 16
    assert kr[("worker-a", "worker-z")] == b"\xBB" * 16


def test_keyring_drops_malformed_hex_silently():
    """A neighbor whose key_hex is too short / non-hex / wrong length
    is silently dropped from the keyring. FabricLink would have
    refused such a key anyway; surfacing here would force every
    caller to dig into NeighborBlock to figure out what went missing."""
    response = dict(HAPPY_RESPONSE)
    response["forward"] = {
        "peer_id": NODE_B,
        "address": "x:1",
        "chosen_transport": "raw_udp",
        "key_hex": "not-hex-at-all",
    }
    c = _client()
    with patch.object(fj.urlrequest, "urlopen",
                       return_value=_mock_urlopen(response)):
        c.join()
    assert c.keyring() == {}


def test_keyring_drops_wrong_length_key():
    response = dict(HAPPY_RESPONSE)
    response["forward"] = {
        "peer_id": NODE_B, "address": "x:1",
        "chosen_transport": "raw_udp",
        "key_hex": "aa" * 8,                   # 8 bytes — not 16
    }
    c = _client()
    with patch.object(fj.urlrequest, "urlopen",
                       return_value=_mock_urlopen(response)):
        c.join()
    assert c.keyring() == {}


# ── 6. Sandbox + WG audit-on-parse ──────────────────────────────────


def test_sandbox_block_emits_audit_event():
    """Phase C parses + audits the sandbox spec; enforcement is L4
    work. The audit event is the contract the eventual sandbox
    supervisor sprint reads to verify it received the right spec."""
    response = dict(HAPPY_RESPONSE)
    response["sandbox"] = {
        "seccomp_profile": "fabric-strict-v1",
        "cpu_threads_limit": 4,
        "ram_limit_gb": 16.0,
        "allowed_egress": ["http://pillar:7777"],
        "multi_tenant_isolation": "exclusive",
        "mode_c_compatible": True,
    }
    c = _client()
    with patch.object(fj, "_AUDIT_AVAILABLE", True):
        fake_audit = MagicMock()
        fake_audit.audit = MagicMock()
        with patch.object(fj, "_audit_mod", fake_audit):
            with patch.object(fj.urlrequest, "urlopen",
                               return_value=_mock_urlopen(response)):
                c.join()
    # Audit was called with the sandbox event + key fields.
    sandbox_calls = [
        call for call in fake_audit.audit.call_args_list
        if call.args[0] == "fabric_sandbox_spec_received"
    ]
    assert len(sandbox_calls) == 1
    kwargs = sandbox_calls[0].kwargs
    assert kwargs["seccomp_profile"] == "fabric-strict-v1"
    assert kwargs["multi_tenant_isolation"] == "exclusive"


def test_wireguard_block_emits_audit_event():
    response = dict(HAPPY_RESPONSE)
    response["wireguard"] = {
        "interface_address": "10.42.1.2/32",
        "peers": [{"peer_id": "b", "public_key": "x",
                   "allowed_ips": "10.42.3.4/32",
                   "endpoint": "203.0.113.5:51820"}],
    }
    c = _client()
    with patch.object(fj, "_AUDIT_AVAILABLE", True):
        fake_audit = MagicMock()
        fake_audit.audit = MagicMock()
        with patch.object(fj, "_audit_mod", fake_audit):
            with patch.object(fj.urlrequest, "urlopen",
                               return_value=_mock_urlopen(response)):
                c.join()
    wg_calls = [
        call for call in fake_audit.audit.call_args_list
        if call.args[0] == "fabric_wireguard_block_received"
    ]
    assert len(wg_calls) == 1
    assert wg_calls[0].kwargs["interface_address"] == "10.42.1.2/32"


def test_join_succeeds_when_audit_module_missing():
    """Failure-soft audit: a worker running without nakshatra_audit
    installed must still complete /join cleanly."""
    response = dict(HAPPY_RESPONSE)
    response["sandbox"] = {"seccomp_profile": "x", "cpu_threads_limit": 1,
                            "ram_limit_gb": 1.0}
    c = _client()
    with patch.object(fj, "_AUDIT_AVAILABLE", False):
        with patch.object(fj.urlrequest, "urlopen",
                           return_value=_mock_urlopen(response)):
            resp = c.join()
    assert resp.sandbox is not None  # parsed even though audit absent


# ── 7. Rekey loop ───────────────────────────────────────────────────


def test_start_rekey_loop_requires_prior_join():
    """Calling start_rekey_loop without a successful join first is a
    bug — the loop has nothing to base its sleep on."""
    c = _client()
    with pytest.raises(RuntimeError, match="join"):
        c.start_rekey_loop()


def test_stop_is_idempotent_and_non_blocking():
    """stop() must be safe to call multiple times AND must not block
    even if no rekey loop was ever started — the boot teardown path
    calls it unconditionally."""
    c = _client()
    c.stop()
    c.stop()                                # no exception, no hang


def test_rekey_loop_fires_on_rotate_at_with_callback():
    """The loop sleeps until rotate_at - slack, then re-POSTs /join
    and fires the on_rekey callback so FabricLinks can reset seq.

    Driving the timing deterministically: set rotate_at to a small
    value AND set rekey_slack_s = 0 so the loop wakes immediately;
    second urlopen call returns a new response with a far-future
    rotate_at so the loop then sleeps for ages; we stop() before
    that second wakeup.
    """
    fired_with: list[fj.JoinResponse] = []
    c = _client(rekey_slack_s=0.0,
                 on_rekey=lambda r: fired_with.append(r))

    first_response = dict(HAPPY_RESPONSE)
    first_response["rotate_at"] = time.time() + 0.1     # ~100ms
    second_response = dict(HAPPY_RESPONSE)
    second_response["plan_id"] = "rekeyed-plan"
    second_response["rotate_at"] = time.time() + 10_000.0

    cm_first = _mock_urlopen(first_response)
    cm_second = _mock_urlopen(second_response)

    with patch.object(fj.urlrequest, "urlopen",
                       side_effect=[cm_first, cm_second]):
        c.join()
        c.start_rekey_loop()
        # Give the daemon thread time to wake, fetch, and fire.
        time.sleep(0.5)
        c.stop()

    assert len(fired_with) == 1
    assert fired_with[0].plan_id == "rekeyed-plan"
    assert c.current().plan_id == "rekeyed-plan"


def test_start_rekey_loop_is_idempotent():
    """Re-call after the loop is already running is a no-op (not an
    error, not a new thread). Operators occasionally hot-reload code
    without restarting the worker."""
    c = _client(rekey_slack_s=10_000.0)              # loop sleeps forever
    first_response = dict(HAPPY_RESPONSE)
    first_response["rotate_at"] = time.time() + 10_000.0
    with patch.object(fj.urlrequest, "urlopen",
                       return_value=_mock_urlopen(first_response)):
        c.join()
        c.start_rekey_loop()
        thread_first = c._thread
        c.start_rekey_loop()                          # second call
        assert c._thread is thread_first              # same thread
    c.stop()
