"""Tests for the client-side TLS / SPKI pinning shipped 2026-05-26.

Mirrors test_worker_phase_3_spki.py in shape. Covers the client.py
helpers added to make `scripts/client.py` SPKI-aware so it can preflight
Info() against TLS-only workers (which the 2026-05-21 SPKI sprint left
hardened on the worker side but unreachable from the unmodified client).

Run with `pytest --noconftest tests/test_client_tls.py` — the project
conftest pulls in hivemind/pytest-asyncio fixtures the client tests
don't need.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import client as cli  # noqa: E402
import nakshatra_tls as nt  # noqa: E402


# ── _sanitize_spki ─────────────────────────────────────────────────────


def test_sanitize_spki_valid_lowercase():
    h = "a" * 64
    assert cli._sanitize_spki(h) == h


def test_sanitize_spki_uppercase_normalized_to_lower():
    assert cli._sanitize_spki("A" * 64) == "a" * 64


def test_sanitize_spki_strips_whitespace():
    assert cli._sanitize_spki(f"  {'b' * 64}  ") == "b" * 64


def test_sanitize_spki_wrong_length_rejected():
    assert cli._sanitize_spki("a" * 63) == ""
    assert cli._sanitize_spki("a" * 65) == ""


def test_sanitize_spki_non_hex_rejected():
    """64-char string that isn't hex must be rejected. A pillar (or
    operator) that ships malformed values cannot trick the client into
    pinning against garbage; refusing-with-empty downgrades to the
    --tls-mode policy decision instead."""
    assert cli._sanitize_spki("Z" * 64) == ""


def test_sanitize_spki_none_and_empty():
    assert cli._sanitize_spki(None) == ""
    assert cli._sanitize_spki("") == ""


# ── _open_chain_channel ────────────────────────────────────────────────


def test_open_chain_channel_tls_off_returns_insecure():
    """--tls-mode=off must NEVER call open_pinned_channel, even if a hash
    is available — operator escape hatch, no surprise TLS overhead."""
    with patch.object(cli, "grpc") as mock_grpc:
        with patch.object(cli, "_wtls") as mock_tls:
            cli._open_chain_channel("h:1", "a" * 64, "off")
    mock_grpc.insecure_channel.assert_called_once_with("h:1")
    mock_tls.open_pinned_channel.assert_not_called()


def test_open_chain_channel_module_missing_falls_back_to_insecure():
    """If nakshatra_tls couldn't import, every mode degrades to insecure.
    The --tls-mode=required preflight check in main() refuses startup in
    this case, so reaching this branch with mode=required means the
    operator explicitly overrode the gate — still don't crash."""
    with patch.object(cli, "_TLS_AVAILABLE", False):
        with patch.object(cli, "grpc") as mock_grpc:
            cli._open_chain_channel("h:1", "a" * 64, "auto")
    mock_grpc.insecure_channel.assert_called_once_with("h:1")


def test_open_chain_channel_auto_with_hash_pins():
    """tls_mode=auto + hash present → call open_pinned_channel with
    refuse_unpinned=False (auto falls through on missing-hash workers
    but pins when it can)."""
    with patch.object(cli, "_wtls") as mock_tls:
        mock_tls.open_pinned_channel.return_value = MagicMock(name="channel")
        out = cli._open_chain_channel("h:1", "a" * 64, "auto")
    mock_tls.open_pinned_channel.assert_called_once_with(
        "h:1", "a" * 64, refuse_unpinned=False,
    )
    assert out is mock_tls.open_pinned_channel.return_value


def test_open_chain_channel_auto_no_hash_pins_unpinned_falls_through():
    """tls_mode=auto + hash=None — open_pinned_channel(refuse_unpinned=
    False) returns an insecure channel internally. We don't bypass the
    helper because we want a single path for the legacy bringup."""
    with patch.object(cli, "_wtls") as mock_tls:
        ch = MagicMock(name="insecure-channel")
        mock_tls.open_pinned_channel.return_value = ch
        out = cli._open_chain_channel("h:1", None, "auto")
    mock_tls.open_pinned_channel.assert_called_once_with(
        "h:1", None, refuse_unpinned=False,
    )
    assert out is ch


def test_open_chain_channel_required_no_hash_pin_error_exits():
    """tls_mode=required + hash=None → open_pinned_channel raises
    PinError("unpinned_peer"); we translate to sys.exit so the operator
    sees a clean message instead of a Python traceback."""
    with patch.object(cli, "_wtls") as mock_tls:
        mock_tls.PinError = nt.PinError
        mock_tls.open_pinned_channel.side_effect = nt.PinError(
            "unpinned_peer", address="h:1",
        )
        with pytest.raises(SystemExit) as ei:
            cli._open_chain_channel("h:1", None, "required")
    assert "unpinned_peer" in str(ei.value)
    assert "h:1" in str(ei.value)


def test_open_chain_channel_required_passes_refuse_unpinned_true():
    with patch.object(cli, "_wtls") as mock_tls:
        mock_tls.open_pinned_channel.return_value = MagicMock()
        cli._open_chain_channel("h:1", "a" * 64, "required")
    mock_tls.open_pinned_channel.assert_called_once_with(
        "h:1", "a" * 64, refuse_unpinned=True,
    )


def test_open_chain_channel_mismatch_exits_with_both_hashes():
    """A spki_mismatch surfaces both the expected and actual abbreviated
    hashes (so the operator can correlate against the audit log on the
    peer side). PinError.__str__ already does the abbreviation."""
    with patch.object(cli, "_wtls") as mock_tls:
        mock_tls.PinError = nt.PinError
        mock_tls.open_pinned_channel.side_effect = nt.PinError(
            "spki_mismatch", address="h:1",
            expected="a" * 64, actual="b" * 64,
        )
        with pytest.raises(SystemExit) as ei:
            cli._open_chain_channel("h:1", "a" * 64, "required")
    msg = str(ei.value)
    assert "spki_mismatch" in msg
    assert "a" * 8 in msg
    assert "b" * 8 in msg


def test_open_chain_channel_probe_failed_exits():
    with patch.object(cli, "_wtls") as mock_tls:
        mock_tls.PinError = nt.PinError
        mock_tls.open_pinned_channel.side_effect = nt.PinError(
            "probe_failed", address="h:1", error="connection refused",
        )
        with pytest.raises(SystemExit) as ei:
            cli._open_chain_channel("h:1", "a" * 64, "required")
    assert "probe_failed" in str(ei.value)


# ── _fetch_spki_index ──────────────────────────────────────────────────


def _mock_urlopen(payload: dict):
    """Build a context-manager mock that imitates urllib.request.urlopen."""
    cm = MagicMock()
    cm.__enter__.return_value = io.BytesIO(json.dumps(payload).encode())
    cm.__exit__.return_value = False
    return cm


def test_fetch_spki_index_extracts_address_keyed_map():
    payload = {
        "peers": [
            {"address": "h1:1", "peer_spki_hash": "a" * 64},
            {"address": "h2:2", "peer_spki_hash": "b" * 64},
        ],
    }
    with patch.object(cli.urlrequest, "urlopen",
                       return_value=_mock_urlopen(payload)):
        idx = cli._fetch_spki_index("http://pillar")
    assert idx == {"h1:1": "a" * 64, "h2:2": "b" * 64}


def test_fetch_spki_index_drops_missing_address_or_hash():
    payload = {
        "peers": [
            {"address": "", "peer_spki_hash": "a" * 64},   # no address
            {"address": "h2:2", "peer_spki_hash": ""},     # no hash
            {"address": "h3:3", "peer_spki_hash": "c" * 64},
        ],
    }
    with patch.object(cli.urlrequest, "urlopen",
                       return_value=_mock_urlopen(payload)):
        idx = cli._fetch_spki_index("http://pillar")
    assert idx == {"h3:3": "c" * 64}


def test_fetch_spki_index_drops_malformed_hash():
    """Same defensive parse as PillarPeerKeyResolver — a malformed hash
    is dropped silently and the peer falls back to whatever --tls-mode
    policy says about unpinned peers."""
    payload = {
        "peers": [
            {"address": "h1:1", "peer_spki_hash": "ZZ" * 32},
            {"address": "h2:2", "peer_spki_hash": "abc"},  # too short
            {"address": "h3:3", "peer_spki_hash": "c" * 64},
        ],
    }
    with patch.object(cli.urlrequest, "urlopen",
                       return_value=_mock_urlopen(payload)):
        idx = cli._fetch_spki_index("http://pillar")
    assert idx == {"h3:3": "c" * 64}


def test_fetch_spki_index_network_failure_returns_empty():
    """A failed /peers fetch must not crash the client — log warning,
    return {}, and let --tls-mode policy take over from there. Without
    this, a pillar outage would break every chain that uses --registry."""
    with patch.object(cli.urlrequest, "urlopen",
                       side_effect=OSError("connection refused")):
        with patch("sys.stderr", new_callable=io.StringIO) as err:
            idx = cli._fetch_spki_index("http://pillar")
    assert idx == {}
    assert "SPKI index fetch failed" in err.getvalue()


def test_fetch_spki_index_empty_payload_returns_empty():
    with patch.object(cli.urlrequest, "urlopen",
                       return_value=_mock_urlopen({"peers": []})):
        idx = cli._fetch_spki_index("http://pillar")
    assert idx == {}


# ── _try_pillar_chain consumes inline peer_spki_hash (Phase 4) ─────────


def test_try_pillar_chain_reads_inline_spki_hash():
    """2026-05-26 SPKI Phase 4: pillar's /chain projection ships
    peer_spki_hash inline. _try_pillar_chain must pass it through so
    main() can skip the backfill /peers fetch on the happy path."""
    payload = {
        "chain": [
            {"node_id": "p0" * 7, "address": "h1:1",
             "layer_start": 0, "layer_end": 14,
             "peer_spki_hash": "a" * 64},
            {"node_id": "p1" * 7, "address": "h2:2",
             "layer_start": 14, "layer_end": 28,
             "peer_spki_hash": "b" * 64},
        ],
    }
    with patch.object(cli.urlrequest, "urlopen",
                       return_value=_mock_urlopen(payload)):
        workers = cli._try_pillar_chain("http://pillar", "m")
    assert workers is not None
    assert workers[0]["peer_spki_hash"] == "a" * 64
    assert workers[1]["peer_spki_hash"] == "b" * 64


def test_try_pillar_chain_handles_legacy_pillar_without_hash():
    """A pre-Phase-4 pillar omits peer_spki_hash. _try_pillar_chain
    must still return a chain — the field defaults to "" and main()
    falls back to the /peers backfill."""
    payload = {
        "chain": [
            {"node_id": "p0" * 7, "address": "h1:1",
             "layer_start": 0, "layer_end": 14},
            # No peer_spki_hash field at all.
        ],
    }
    with patch.object(cli.urlrequest, "urlopen",
                       return_value=_mock_urlopen(payload)):
        workers = cli._try_pillar_chain("http://pillar", "m")
    assert workers is not None
    assert workers[0]["peer_spki_hash"] == ""


def test_try_pillar_chain_drops_malformed_inline_hash():
    """A pillar that ships a malformed hash (rare — its own parse layer
    rejects them at /peer ingest, but defense-in-depth) still gives us
    a usable chain entry with an empty hash."""
    payload = {
        "chain": [
            {"node_id": "p0" * 7, "address": "h1:1",
             "layer_start": 0, "layer_end": 14,
             "peer_spki_hash": "ZZ" * 32},  # non-hex
        ],
    }
    with patch.object(cli.urlrequest, "urlopen",
                       return_value=_mock_urlopen(payload)):
        workers = cli._try_pillar_chain("http://pillar", "m")
    assert workers is not None
    assert workers[0]["peer_spki_hash"] == ""


# ── _fetch_spki_index (continued) ──────────────────────────────────────


def test_fetch_spki_index_non_dict_peer_skipped():
    """A pillar that ships malformed peer entries (e.g. strings or
    nulls instead of dicts) shouldn't crash the client — skip them."""
    payload = {"peers": [None, "garbage", {"address": "h1:1",
                                            "peer_spki_hash": "a" * 64}]}
    with patch.object(cli.urlrequest, "urlopen",
                       return_value=_mock_urlopen(payload)):
        idx = cli._fetch_spki_index("http://pillar")
    assert idx == {"h1:1": "a" * 64}
