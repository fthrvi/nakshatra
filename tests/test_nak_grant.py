"""Human grant renewal boundary. All identity state lives under pytest's temporary HOME."""
from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from test_nakd import signer_mod  # noqa: F401

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from network import join_setup, nak, nak_mcp  # noqa: E402


class _NotATerminal:
    def isatty(self):
        return False


def _temporary_identity(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NAK_NET_DIR", raising=False)
    monkeypatch.delenv("NAK_SIGNER_DIR", raising=False)
    join_setup.ensure_identity(tmp_path)
    state = tmp_path / ".sthambha" / "signer"
    return state, state / "agents" / "agent.key", state / "agents" / "agent.grant.json"


def _state_snapshot(root: Path):
    return {
        path.relative_to(root).as_posix():
            (path.lstat().st_mode, path.read_bytes() if path.is_file() else b"")
        for path in sorted(root.rglob("*"))
    }


def test_grant_status_prints_public_authority_and_renewal_warning(monkeypatch, capsys):
    status = {"agent": "agent", "caps": ["nak.msg"], "expires_at": 1_800_000_000,
              "not_before": 1, "revoked": False, "grant_id": "g",
              "warning": "grant expires soon; run: nak grant renew"}
    monkeypatch.setattr(nak, "call", lambda _sock, req: {"grant": status})

    assert nak.main(["grant", "status"]) == 0

    out = capsys.readouterr().out
    assert "agent agent" in out and "caps nak.msg" in out and "grant expires" in out
    assert "WARNING:" in out and "nak grant renew" in out


def test_charter_lapse_and_legacy_status_text_is_exact_in_cli_and_mcp(monkeypatch, capsys):
    charter = {
        "agent": "agent", "caps": ["nak.msg"], "expires_at": 2_700_000,
        "status": ("active while used — lapses if unused for 30 days "
                   "(next: 1970-01-30 06:40:00 UTC); charter until 1970-02-01 06:00:00 UTC; "
                   "renew wakes it"),
        "warning": "",
    }
    full_status = {"person": "a" * 64, "node": "b" * 64, "custody": "test", "relay": "local",
                   "contacts": [], "pending_requests": 0, "open_invites": 0, "outbox_refused": 0,
                   "newest_refusal": None, "grant": charter}
    monkeypatch.setattr(nak, "call", lambda _sock, _req: full_status)
    assert nak.main(["status"]) == 0
    assert charter["status"] in capsys.readouterr().out

    class Client:
        def status(self):
            return {"person": "a" * 64, "node": "b" * 64, "custody": "test", "contacts": [],
                    "pending_requests": 0, "outbox_refused": 0, "grant": charter}

    assert charter["status"] in nak_mcp.Server(Client()).call("nak_status", {})

    lapsed = dict(charter, status=("LAPSED: unused since 2026-09-01 12:00:00 UTC. "
                                   "Wake it with: nak grant renew"))
    nak._print_grant_status(lapsed)
    assert lapsed["status"] in capsys.readouterr().out

    legacy = dict(charter, status="grant expires 1970-02-01 06:00:00 UTC")
    nak._print_grant_status(legacy)
    assert legacy["status"] in capsys.readouterr().out


def test_face_id_grant_renew_defaults_to_a_365_day_charter_and_preserves_authority(
        tmp_path, monkeypatch, capsys):
    state, key_path, grant_path = _temporary_identity(tmp_path, monkeypatch)
    before_key = key_path.read_bytes()
    current = json.loads(grant_path.read_text())
    current["constraints"] = {"per_tx_cap": 7, "day_cap": 19}
    person_priv, _ = signer_mod.load_key(tmp_path / ".sthambha" / "person-TEST.key")
    current = signer_mod.issue_grant(
        person_priv, "agent", current["subject_pub"], current["caps"],
        node="*", ttl_s=3600, constraints=current["constraints"])
    signer_mod.save_grant(grant_path, current)
    live_signer = signer_mod.Signer(state)

    assert nak.main(["grant", "renew", "--yes"]) == 0

    renewed = json.loads(grant_path.read_text())
    assert key_path.read_bytes() == before_key
    assert renewed["grant_id"] != current["grant_id"]
    assert renewed["subject_pub"] == current["subject_pub"]
    assert renewed["node"] == current["node"] == "*"
    assert renewed["caps"] == current["caps"]
    assert renewed["constraints"] == current["constraints"]
    assert 364 * 86400 <= renewed["expires_at"] - int(time.time()) <= 365 * 86400
    assert renewed["lease"]["max_s"] == 7 * 86400
    assert renewed["lease"]["idle_s"] == 30 * 86400
    assert renewed["lease"]["issuer_pub"] == signer_mod.load_or_create_lease_key(state)[1]
    assert stat.S_IMODE((state / "lease.key").stat().st_mode) == 0o600
    assert live_signer.grant_status("agent")["grant_id"] == renewed["grant_id"]
    envelope = live_signer.sign_envelope("agent", "msg", "cd" * 32, "ab" * 32)
    assert signer_mod.verify_envelope(envelope, (state / "person.pub").read_text().strip())["grant_id"] == \
        renewed["grant_id"]
    assert "signer reload is automatic" in capsys.readouterr().out


def test_grant_renew_no_lease_keeps_the_legacy_fixed_expiry_path(tmp_path, monkeypatch, capsys):
    _state, _key_path, grant_path = _temporary_identity(tmp_path, monkeypatch)

    assert nak.main(["grant", "renew", "--yes", "--ttl", "30", "--no-lease"]) == 0

    renewed = json.loads(grant_path.read_text())
    assert "lease" not in renewed
    assert 29 * 86400 <= renewed["expires_at"] - int(time.time()) <= 30 * 86400
    assert "fixed-expiry legacy grant" in capsys.readouterr().out


def test_grant_renew_old_signer_fails_cleanly_without_mutation_and_no_lease_still_works(
        tmp_path, monkeypatch, capsys):
    _state, _key_path, grant_path = _temporary_identity(tmp_path, monkeypatch)
    before = _state_snapshot(tmp_path)
    monkeypatch.delattr(signer_mod, "load_or_create_lease_key")

    with pytest.raises(SystemExit) as refused:
        nak.main(["grant", "renew", "--yes"])

    assert str(refused.value) == "signer too old for sliding grants; update the node or use --no-lease"
    assert _state_snapshot(tmp_path) == before
    assert nak.main(["grant", "renew", "--yes", "--no-lease"]) == 0
    assert "lease" not in json.loads(grant_path.read_text())
    assert "fixed-expiry legacy grant" in capsys.readouterr().out


def test_grant_renew_wakes_a_lapsed_charter(tmp_path, monkeypatch):
    state, _key_path, grant_path = _temporary_identity(tmp_path, monkeypatch)
    person_priv, _ = signer_mod.load_key(tmp_path / ".sthambha" / "person-TEST.key")
    current = json.loads(grant_path.read_text())
    old = int(time.time()) - 31 * 86400
    lapsed = signer_mod.issue_grant(
        person_priv, "agent", current["subject_pub"], current["caps"], node=current["node"],
        ttl_s=365 * 86400, now=old, lease=current["lease"])
    signer_mod.save_grant(grant_path, lapsed)
    assert signer_mod.Signer(state).grant_status("agent")["problem"].startswith("lapsed: unused since ")

    assert nak.main(["grant", "renew", "--yes"]) == 0

    status = signer_mod.Signer(state).grant_status("agent")
    assert status["valid"] is True
    assert status["grant_id"] != lapsed["grant_id"]


def test_grant_renew_requires_human_confirmation_and_explicit_caps_for_widening(
        tmp_path, monkeypatch):
    _state, key_path, grant_path = _temporary_identity(tmp_path, monkeypatch)
    before_key = key_path.read_bytes()
    monkeypatch.setattr(nak.sys, "stdin", _NotATerminal())
    with pytest.raises(SystemExit, match="interactive-only"):
        nak.main(["grant", "renew"])
    before = json.loads(grant_path.read_text())
    with pytest.raises(SystemExit, match="between 1 and 365"):
        nak.main(["grant", "renew", "--yes", "--ttl", "366"])

    assert nak.main(["grant", "renew", "--yes", "--caps", "nak.msg,nak.task.claim,nak.task.post"]) == 0
    after = json.loads(grant_path.read_text())
    assert key_path.read_bytes() == before_key
    assert before["caps"] == ["nak.msg", "nak.task.claim"]
    assert after["caps"] == ["nak.msg", "nak.task.claim", "nak.task.post"]


def test_grant_renew_without_person_key_prints_exact_same_key_command(tmp_path, monkeypatch, capsys):
    state, key_path, grant_path = _temporary_identity(tmp_path, monkeypatch)
    before_key, before_grant = key_path.read_bytes(), grant_path.read_bytes()
    (tmp_path / ".sthambha" / "person-TEST.key").unlink()

    assert nak.main(["grant", "renew", "--yes", "--ttl", "30"]) == 1

    err = capsys.readouterr().err
    assert ("python -m sthambha.signer delegate --state " + str(state)) in err
    assert "--agent agent --caps nak.msg,nak.task.claim --ttl 2592000 --reuse-key" in err
    assert key_path.read_bytes() == before_key and grant_path.read_bytes() == before_grant


@pytest.mark.parametrize("bad_name", [
    "agent.name", "agent name", "agent\tname", "agent\x1fname", "agént", "a" * 65,
    "agent\x00name", "../agents/agent",
])
def test_grant_renew_rejects_noncanonical_agent_without_mutation(tmp_path, monkeypatch, bad_name):
    state, _key_path, _grant_path = _temporary_identity(tmp_path, monkeypatch)
    net_dir = tmp_path / ".nakshatra" / "net"
    (net_dir / "agent").write_text(bad_name + "\n")
    before = {
        path.relative_to(tmp_path).as_posix():
            (path.lstat().st_mode, path.read_bytes() if path.is_file() else b"")
        for path in sorted(tmp_path.rglob("*"))
    }

    with pytest.raises(SystemExit, match="bad agent name"):
        nak.main(["grant", "renew", "--yes"])

    after = {
        path.relative_to(tmp_path).as_posix():
            (path.lstat().st_mode, path.read_bytes() if path.is_file() else b"")
        for path in sorted(tmp_path.rglob("*"))
    }
    assert after == before


def test_grant_renew_refuses_key_symlink_swapped_at_validation_open_boundary(
        tmp_path, monkeypatch):
    state, key_path, grant_path = _temporary_identity(tmp_path, monkeypatch)
    outside = tmp_path / "outside-agent.key"
    outside.write_bytes(key_path.read_bytes())
    outside_before = outside.read_bytes()
    grant_before = grant_path.read_bytes()

    def swap_key(_state, name):
        assert name == "agent"
        key_path.unlink()
        key_path.symlink_to(outside)

    monkeypatch.setattr(signer_mod, "_agent_open_hook", swap_key)
    with pytest.raises(SystemExit, match="non-regular agent state file"):
        nak.main(["grant", "renew", "--yes"])

    assert outside.read_bytes() == outside_before
    assert grant_path.read_bytes() == grant_before


def test_grant_renew_has_no_daemon_updater_supervisor_timer_or_mcp_entrypoint():
    production = []
    for root in (_REPO / "scripts", _REPO / "release", _REPO / "deploy"):
        if not root.exists():
            continue
        production.extend(path for path in root.rglob("*") if path.is_file() and
                          path.suffix in {".py", ".sh", ".service", ".timer"})
    callers = []
    for path in production:
        text = path.read_text(errors="replace")
        if "_grant_renew(" in text:
            callers.append(path.relative_to(_REPO).as_posix())
    assert callers == ["scripts/network/nak.py"]
    nak_source = (_REPO / "scripts" / "network" / "nak.py").read_text()
    assert nak_source.count("_grant_renew(") == 2       # definition + explicit CLI dispatch
    mcp_source = (_REPO / "scripts" / "network" / "nak_mcp.py").read_text()
    assert "nak_grant_renew" not in mcp_source


def test_grant_renew_sets_spend_caps_only_when_named(tmp_path, monkeypatch, capsys):
    _state, key_path, grant_path = _temporary_identity(tmp_path, monkeypatch)
    before_key = key_path.read_bytes()
    before = json.loads(grant_path.read_text())
    assert nak.main(["grant", "renew", "--yes"]) == 0
    assert json.loads(grant_path.read_text()).get("constraints") == before.get("constraints")

    assert nak.main(["grant", "renew", "--yes", "--caps", "nak.msg,nak.pay,nak.task.claim",
                     "--per-tx-cap", "500", "--day-cap", "2000"]) == 0
    after = json.loads(grant_path.read_text())
    assert key_path.read_bytes() == before_key
    assert after["constraints"]["per_tx_cap"] == 500 and after["constraints"]["day_cap"] == 2000
    assert "nak.pay" in after["caps"]
    assert "500 per payment, 2000 per day" in capsys.readouterr().out

    assert nak.main(["grant", "renew", "--yes", "--day-cap", "3000"]) == 0
    again = json.loads(grant_path.read_text())
    assert again["constraints"]["per_tx_cap"] == 500 and again["constraints"]["day_cap"] == 3000


@pytest.mark.parametrize("flag,value", [
    ("--per-tx-cap", "-1"), ("--day-cap", str(10**14 + 1)),
    ("--idle-days", "0"), ("--idle-days", "91"),
    ("--lease-days", "0"), ("--lease-days", "8"),
])
def test_grant_renew_refuses_out_of_range_caps_without_change(tmp_path, monkeypatch, flag, value):
    _state, _key_path, grant_path = _temporary_identity(tmp_path, monkeypatch)
    before = grant_path.read_bytes()
    with pytest.raises(SystemExit, match="between"):
        nak.main(["grant", "renew", "--yes", flag, value])
    assert grant_path.read_bytes() == before


def test_grant_renew_cap_change_without_person_key_changes_nothing(tmp_path, monkeypatch, capsys):
    state, _key_path, grant_path = _temporary_identity(tmp_path, monkeypatch)
    before = grant_path.read_bytes()
    (Path.home() / ".sthambha" / "person-TEST.key").unlink()
    assert nak.main(["grant", "renew", "--yes", "--per-tx-cap", "5"]) == 1
    assert grant_path.read_bytes() == before
    assert "Spend caps can only be changed" in capsys.readouterr().err


def test_mcp_tool_list_has_no_grant_wake_or_lease_mutation_surface():
    tools = nak_mcp.Server(object(), allow_connect=True).list_tools()
    for tool in tools:
        name = tool["name"].lower()
        properties = {key.lower() for key in tool["inputSchema"]["properties"]}
        assert not any(word in name for word in ("grant", "renew", "wake", "lease"))
        assert not properties.intersection({"grant", "renew", "wake", "idle_days", "lease_days", "lease_key"})
