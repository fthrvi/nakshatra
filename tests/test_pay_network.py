"""The one-file payment network switch and its real-money gates."""
from __future__ import annotations

import io
import json
import os
import stat
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from network import join_setup, nak, nakd, pay_chain, pay_network, pay_voucher  # noqa: E402


PROGRAM = pay_network.DEVNET_PROGRAM
MINT = pay_network.DEVNET_MINT
MAINNET_PROGRAM = pay_chain.b58encode(bytes([73]) * 32)
RPC = "http://127.0.0.1:18999"


@pytest.fixture(autouse=True)
def _clean_environment(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in (
        "NAK_PAY_RPC", "NAKSHATRA_PAY_RPC_URL", "NAK_PAY_PROGRAM", "NAK_PAY_MINT",
    ):
        monkeypatch.delenv(name, raising=False)
    return home


def _value(*, active="devnet", devnet_rpc=RPC, mainnet_rpc=RPC,
           mainnet_program=MAINNET_PROGRAM):
    return {
        "active": active,
        "networks": {
            "devnet": {
                "rpc": devnet_rpc,
                "genesis": pay_network.DEVNET_GENESIS_HASH,
                "program": PROGRAM,
                "mint": MINT,
                "explorer_cluster": "devnet",
            },
            "mainnet": {
                "rpc": mainnet_rpc,
                "genesis": pay_network.MAINNET_GENESIS_HASH,
                "program": mainnet_program,
                "mint": pay_network.MAINNET_MINT,
                "explorer_cluster": "",
            },
        },
    }


def _write(path: Path, value, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    path.chmod(mode)
    return path


def _config(home: Path, **kwargs):
    return _write(pay_network.network_path(home), _value(**kwargs))


def _limits(home: Path):
    return _write(pay_network.limits_path(home), {
        "cap_total": 1_000_000, "days": 30,
        "per_tx": 100_000, "per_day": 200_000,
    })


def _arm(home: Path, *, program=MAINNET_PROGRAM):
    return _write(pay_network.arm_path(home), {
        "network": "mainnet",
        "genesis": pay_network.MAINNET_GENESIS_HASH,
        "program": program,
        "mint": pay_network.MAINNET_MINT,
        "armed_at": 1_700_000_000,
    })


class _Response:
    def __init__(self, result):
        self.result = result

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _limit=-1):
        return json.dumps({"jsonrpc": "2.0", "id": 1, "result": self.result}).encode()


def _rpc(genesis):
    return pay_chain.PayRpc(RPC, opener=lambda _request, timeout: _Response(genesis))


def test_absent_file_preserves_devnet_and_devnet_only_mainnet_text():
    assert _rpc(pay_network.DEVNET_GENESIS_HASH).ensure_cluster() == pay_network.DEVNET_GENESIS_HASH
    with pytest.raises(pay_voucher.VoucherError, match="DEVNET-ONLY"):
        _rpc(pay_network.MAINNET_GENESIS_HASH).ensure_cluster()


def test_devnet_file_is_the_only_source_and_disagreement_is_refused(
        _clean_environment, monkeypatch):
    home = _clean_environment
    _config(home)
    assert pay_chain.configured_rpc_url(os.environ) == RPC
    assert pay_chain.configured_pubkey(os.environ, "NAK_PAY_PROGRAM") == PROGRAM
    assert pay_chain.configured_pubkey(os.environ, "NAK_PAY_MINT") == MINT
    monkeypatch.setenv("NAK_PAY_RPC", "http://127.0.0.1:19000")
    with pytest.raises(SystemExit, match="disagrees.*only payment network source"):
        pay_chain.configured_rpc_url(os.environ)


@pytest.mark.parametrize(
    ("active", "returned", "expected"),
    [
        ("devnet", pay_network.MAINNET_GENESIS_HASH, "devnet"),
        ("mainnet", pay_network.DEVNET_GENESIS_HASH, "mainnet"),
    ],
)
def test_live_rpc_genesis_must_match_selected_network(
        _clean_environment, active, returned, expected):
    home = _clean_environment
    _config(home, active=active)
    if active == "mainnet":
        _limits(home)
        _arm(home)
    with pytest.raises(pay_voucher.VoucherError, match=f"active payment network {expected}"):
        _rpc(returned).ensure_cluster()


@pytest.mark.parametrize("kind", ["mode", "symlink", "json", "remote-http"])
def test_bad_network_file_is_refused(_clean_environment, kind):
    home = _clean_environment
    path = pay_network.network_path(home)
    if kind == "mode":
        _write(path, _value(), 0o644)
        match = "mode 0600"
    elif kind == "symlink":
        target = _write(home / "target.json", _value())
        path.parent.mkdir(parents=True)
        path.symlink_to(target)
        match = "regular file, not a symlink"
    elif kind == "json":
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        path.chmod(0o600)
        match = "invalid JSON"
    else:
        _write(path, _value(devnet_rpc="http://198.51.100.8:8899"))
        match = "must use https"
    with pytest.raises(pay_network.NetworkConfigError, match=match):
        pay_network.load_document(home=home, environ={})


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("program", "mainnet program not configured"),
        ("arm-missing", "arm file is missing"),
        ("arm-mismatch", "arm file does not match"),
        ("limits", "limits.json is required"),
    ],
)
def test_each_mainnet_gate_names_its_refusal(_clean_environment, case, match):
    home = _clean_environment
    _config(home, active="mainnet",
            mainnet_program="" if case == "program" else MAINNET_PROGRAM)
    if case not in {"program", "arm-missing"}:
        _arm(home, program=PROGRAM if case == "arm-mismatch" else MAINNET_PROGRAM)
    if case not in {"program", "limits"}:
        _limits(home)
    with pytest.raises(pay_network.NetworkConfigError, match=match):
        pay_network.active_network(home=home, environ={})


def test_fully_gated_mainnet_is_accepted_and_uses_unsuffixed_explorer(_clean_environment):
    home = _clean_environment
    _config(home, active="mainnet")
    _limits(home)
    _arm(home)
    network = pay_network.active_network(home=home, environ={})
    assert network is not None and network.name == "mainnet"
    assert _rpc(pay_network.MAINNET_GENESIS_HASH).ensure_cluster() == pay_network.MAINNET_GENESIS_HASH
    assert _rpc(pay_network.MAINNET_GENESIS_HASH).cluster_label() == ""
    assert pay_chain.explorer_link("signature", "") == \
        "https://explorer.solana.com/tx/signature"


def test_use_changes_only_active_keeps_exact_backup_and_refuses_unarmed_mainnet(
        _clean_environment):
    home = _clean_environment
    path = _config(home)
    before = path.read_bytes()
    with pytest.raises(pay_network.NetworkConfigError, match="arm file is missing"):
        pay_network.use_network("mainnet", home=home, environ={})
    _limits(home)
    _arm(home)
    _path, backup = pay_network.use_network("mainnet", home=home, environ={})
    assert backup.name.endswith(".bak")
    assert backup.read_bytes() == before
    after = json.loads(path.read_text())
    expected = json.loads(before)
    expected["active"] = "mainnet"
    assert after == expected


def test_cli_use_names_only_installed_services(_clean_environment, capsys):
    home = _clean_environment
    _config(home)
    units = home / ".config" / "systemd" / "user"
    units.mkdir(parents=True)
    (units / "nak-net.service").write_text("[Service]\n")
    (units / "nakshatra-gateway.service").write_text("[Service]\n")
    assert nak.main(["pay", "network", "use", "devnet"]) == 0
    shown = capsys.readouterr().out
    assert "nak-net" in shown
    assert "nakshatra-gateway" in shown


def test_init_migrates_environment_to_private_file_and_will_not_overwrite(
        _clean_environment):
    home = _clean_environment
    environ = {"NAK_PAY_RPC": RPC, "NAK_PAY_PROGRAM": PROGRAM, "NAK_PAY_MINT": MINT}
    path = pay_network.initialize_from_environment(home=home, environ=environ)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())["networks"]["devnet"]["rpc"] == RPC
    with pytest.raises(pay_network.NetworkConfigError, match="refusing to overwrite"):
        pay_network.initialize_from_environment(home=home, environ=environ)


def test_status_prints_rpc_host_without_path_query_or_keys(_clean_environment, capsys):
    home = _clean_environment
    secret = "API-KEY-DO-NOT-PRINT"
    _config(home, devnet_rpc=f"https://rpc.example/v2/{secret}?token={secret}")
    assert nak.main(["pay", "network", "status"]) == 0
    shown = capsys.readouterr().out
    assert "rpc host: rpc.example" in shown
    assert secret not in shown


class _TTY(io.StringIO):
    def isatty(self):
        return True


def test_arm_requires_tty_and_exact_phrase_then_writes_private_file(
        _clean_environment, monkeypatch):
    home = _clean_environment
    _config(home)
    monkeypatch.setattr(nak.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(nak.sys, "stdout", io.StringIO())
    with pytest.raises(SystemExit, match="interactive-only"):
        nak.main(["pay", "network", "arm", "mainnet"])

    monkeypatch.setattr(nak.sys, "stdin", _TTY("WRONG\n"))
    monkeypatch.setattr(nak.sys, "stdout", _TTY())
    with pytest.raises(SystemExit, match="did not match"):
        nak.main(["pay", "network", "arm", "mainnet"])
    assert not pay_network.arm_path(home).exists()

    phrase = f"ARM MAINNET {MAINNET_PROGRAM[:8]}\n"
    monkeypatch.setattr(nak.sys, "stdin", _TTY(phrase))
    monkeypatch.setattr(nak.sys, "stdout", _TTY())
    assert nak.main(["pay", "network", "arm", "mainnet"]) == 0
    arm = pay_network.arm_path(home)
    assert stat.S_IMODE(arm.stat().st_mode) == 0o600
    assert json.loads(arm.read_text())["program"] == MAINNET_PROGRAM


def test_devnet_starter_guard_refuses_mainnet(_clean_environment):
    home = _clean_environment
    _config(home, active="mainnet")
    with pytest.raises(SystemExit, match="consumer starter is DEVNET-ONLY"):
        nak._require_devnet_payment_flow("consumer starter")


def test_nakd_leaves_starter_pending_without_touching_a_payment_backend_on_mainnet(
        _clean_environment):
    home = _clean_environment
    _config(home, active="mainnet")
    updates = []

    class Store:
        @staticmethod
        def starter_rows(states):
            assert states == ("pending",)
            return [{"invite_id": "starter-1"}]

        @staticmethod
        def starter_update(invite_id, **values):
            updates.append((invite_id, values))

    node = object.__new__(nakd.Node)
    node.store = Store()
    node._starter_lock = threading.Lock()
    assert node._attempt_starter("starter-1", now=100) is False
    assert updates == [("starter-1", {
        "state": "pending",
        "reason": "consumer starter is DEVNET-ONLY and is refused on mainnet",
    })]


def test_join_setup_refuses_mainnet_starter_before_setup(_clean_environment):
    home = _clean_environment
    _config(home, active="mainnet")
    invite = {"kind": "consumer", "consumer": {"starter": 100}}
    with pytest.raises(SystemExit, match="consumer starter is DEVNET-ONLY"):
        join_setup._require_devnet_starter(home, invite)


@pytest.mark.parametrize(("name", "genesis", "rpc", "match"), [
    ("prod", pay_network.MAINNET_GENESIS_HASH, RPC, "only 'mainnet' may"),
    ("dev2", pay_network.DEVNET_GENESIS_HASH, RPC, "only 'devnet' may"),
    ("local", pay_chain.b58encode(bytes([9]) * 32),
     "https://rpc.example.com", "must use a loopback rpc"),
])
def test_real_clusters_cannot_hide_under_another_name(
        _clean_environment, name, genesis, rpc, match):
    # Renaming a mainnet entry must never skip the arm file and limits gates.
    home = _clean_environment
    value = _value(active=name)
    value["networks"][name] = {
        "rpc": rpc, "genesis": genesis, "program": PROGRAM, "mint": MINT}
    _write(pay_network.network_path(home), value)
    with pytest.raises(pay_network.NetworkConfigError, match=match):
        pay_network.active_network(home=home)


def test_loopback_test_network_is_still_allowed(_clean_environment):
    home = _clean_environment
    value = _value(active="local")
    value["networks"]["local"] = {
        "rpc": RPC, "genesis": pay_chain.b58encode(bytes([9]) * 32),
        "program": PROGRAM, "mint": MINT}
    _write(pay_network.network_path(home), value)
    assert pay_network.active_network(home=home).name == "local"
