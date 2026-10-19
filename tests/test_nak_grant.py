"""Human grant renewal boundary. All identity state lives under pytest's temporary HOME."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from test_nakd import signer_mod  # noqa: F401

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from network import join_setup, nak  # noqa: E402


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


def test_grant_status_prints_public_authority_and_renewal_warning(monkeypatch, capsys):
    status = {"agent": "agent", "caps": ["nak.msg"], "expires_at": 1_800_000_000,
              "not_before": 1, "revoked": False, "grant_id": "g",
              "warning": "grant expires soon; run: nak grant renew"}
    monkeypatch.setattr(nak, "call", lambda _sock, req: {"grant": status})

    assert nak.main(["grant", "status"]) == 0

    out = capsys.readouterr().out
    assert "agent agent" in out and "caps nak.msg" in out and "grant expires" in out
    assert "WARNING:" in out and "nak grant renew" in out


def test_grant_renew_preserves_agent_key_caps_and_constraints_without_restart(tmp_path, monkeypatch, capsys):
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
    assert 89 * 86400 <= renewed["expires_at"] - int(time.time()) <= 90 * 86400
    assert live_signer.grant_status("agent")["grant_id"] == renewed["grant_id"]
    envelope = live_signer.sign_envelope("agent", "msg", "cd" * 32, "ab" * 32)
    assert signer_mod.verify_envelope(envelope, (state / "person.pub").read_text().strip())["grant_id"] == \
        renewed["grant_id"]
    assert "signer reload is automatic" in capsys.readouterr().out


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


@pytest.mark.parametrize("flag,value", [("--per-tx-cap", "-1"), ("--day-cap", str(10**14 + 1))])
def test_grant_renew_refuses_out_of_range_caps_without_change(tmp_path, monkeypatch, flag, value):
    _state, _key_path, grant_path = _temporary_identity(tmp_path, monkeypatch)
    before = grant_path.read_bytes()
    with pytest.raises(SystemExit, match="between 0 and"):
        nak.main(["grant", "renew", "--yes", flag, value])
    assert grant_path.read_bytes() == before


def test_grant_renew_cap_change_without_person_key_changes_nothing(tmp_path, monkeypatch, capsys):
    state, _key_path, grant_path = _temporary_identity(tmp_path, monkeypatch)
    before = grant_path.read_bytes()
    (Path.home() / ".sthambha" / "person-TEST.key").unlink()
    assert nak.main(["grant", "renew", "--yes", "--per-tx-cap", "5"]) == 1
    assert grant_path.read_bytes() == before
    assert "Spend caps can only be changed" in capsys.readouterr().err
