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
