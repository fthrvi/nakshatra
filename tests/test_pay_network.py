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


def _arm(home: Path, *, program=MAINNET_PROGRAM, rpc=RPC):
    return _write(pay_network.arm_path(home), {
        "network": "mainnet",
        "rpc": rpc,
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
    assert stat.S_IMODE(path.with_name(".network.lock").stat().st_mode) == 0o600
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


class _ChangingTTY(_TTY):
    def __init__(self, value, change):
        super().__init__(value)
        self.change = change

    def readline(self, *args, **kwargs):
        self.change()
        return super().readline(*args, **kwargs)


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
    armed = json.loads(arm.read_text())
    assert armed["program"] == MAINNET_PROGRAM
    assert armed["rpc"] == RPC


def test_arm_mainnet_owns_tty_prompt_and_refuses_changed_document(_clean_environment):
    home = _clean_environment
    path = _config(home)
    with pytest.raises(pay_network.NetworkConfigError, match="interactive-only"):
        pay_network.arm_mainnet(
            home=home, environ={}, input_stream=io.StringIO(), output_stream=io.StringIO())

    def change_config():
        changed = _value(mainnet_rpc="http://127.0.0.1:19000")
        _write(path, changed)

    shown = _TTY()
    typed = _ChangingTTY(f"ARM MAINNET {MAINNET_PROGRAM[:8]}\n", change_config)
    with pytest.raises(pay_network.NetworkConfigError, match="changed during.*prompt"):
        pay_network.arm_mainnet(
            home=home, environ={}, input_stream=typed, output_stream=shown)
    assert "mainnet rpc host: 127.0.0.1:18999" in shown.getvalue()
    assert f"mainnet program: {MAINNET_PROGRAM}" in shown.getvalue()
    assert f"mainnet mint: {pay_network.MAINNET_MINT}" in shown.getvalue()
    assert not pay_network.arm_path(home).exists()


def test_changing_mainnet_rpc_disarms_existing_record(_clean_environment):
    home = _clean_environment
    path = _config(home, active="mainnet")
    _limits(home)
    _arm(home)
    assert pay_network.active_network(home=home, environ={}).name == "mainnet"
    _write(path, _value(active="mainnet", mainnet_rpc="http://127.0.0.1:19000"))
    with pytest.raises(pay_network.NetworkConfigError, match="arm file does not match"):
        pay_network.active_network(home=home, environ={})


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


def test_cached_mainnet_object_rechecks_disarm_for_reads_and_sends(_clean_environment):
    home = _clean_environment
    _config(home, active="mainnet")
    _limits(home)
    _arm(home)
    calls = []
    rpc = pay_chain.PayRpc(RPC)

    def call(method, _params):
        calls.append(method)
        if method == "getGenesisHash":
            return pay_network.MAINNET_GENESIS_HASH
        if method == "getBlockHeight":
            return 10
        if method == "sendTransaction":
            pytest.fail("a disarmed mainnet transaction was broadcast")
        raise AssertionError(method)

    rpc._rpc = call
    assert rpc.block_height() == 10
    assert calls.count("getGenesisHash") == 1
    assert pay_network.disarm_mainnet(home=home)
    with pytest.raises(pay_voucher.VoucherError, match="arm file is missing"):
        rpc.block_height()
    transaction = pay_chain.SignedTransaction(b"tx", b"message", (), "signature")
    with pytest.raises(pay_voucher.VoucherError, match="arm file is missing"):
        rpc.send_and_confirm(transaction)
    assert "sendTransaction" not in calls
    assert calls.count("getGenesisHash") == 1


def test_send_rechecks_gates_immediately_before_broadcast(_clean_environment):
    home = _clean_environment
    _config(home, active="mainnet")
    _limits(home)
    _arm(home)
    calls = []
    rpc = pay_chain.PayRpc(RPC)

    def call(method, _params):
        calls.append(method)
        if method == "getGenesisHash":
            return pay_network.MAINNET_GENESIS_HASH
        if method == "sendTransaction":
            pytest.fail("sendTransaction ran after the arm was removed")
        raise AssertionError(method)

    rpc._rpc = call
    ensure = rpc._ensure_cluster
    checks = 0

    def disarm_after_first_check():
        nonlocal checks
        ensure()
        checks += 1
        if checks == 1:
            pay_network.disarm_mainnet(home=home)

    rpc._ensure_cluster = disarm_after_first_check
    transaction = pay_chain.SignedTransaction(b"tx", b"message", (), "signature")
    with pytest.raises(pay_voucher.VoucherError, match="arm file is missing"):
        rpc.send_and_confirm(transaction)
    assert checks == 1
    assert calls == ["getGenesisHash"]


@pytest.mark.parametrize("component", [".nakshatra", "pay"])
def test_symlinked_payment_directories_are_refused(_clean_environment, component):
    home = _clean_environment
    target = home / "redirected"
    target.mkdir()
    if component == ".nakshatra":
        (home / ".nakshatra").symlink_to(target, target_is_directory=True)
    else:
        nak = home / ".nakshatra"
        nak.mkdir()
        (nak / "pay").symlink_to(target, target_is_directory=True)
    with pytest.raises(pay_network.NetworkConfigError, match="symlinked payment directory"):
        pay_network.load_document(home=home, environ={})


@pytest.mark.parametrize("filename", ["mainnet.armed", "limits.json"])
def test_mainnet_gate_files_are_opened_without_following_symlinks(
        _clean_environment, filename):
    home = _clean_environment
    _config(home, active="mainnet")
    if filename != "mainnet.armed":
        _arm(home)
    if filename != "limits.json":
        _limits(home)
    target = _write(home / f"real-{filename}", {})
    path = pay_network.network_path(home).with_name(filename)
    path.symlink_to(target)
    with pytest.raises(pay_network.NetworkConfigError, match="regular file, not a symlink"):
        pay_network.active_network(home=home, environ={})


def test_network_file_swap_between_stat_and_open_is_refused(
        _clean_environment, monkeypatch):
    home = _clean_environment
    path = _config(home)
    replacement = _write(home / "replacement.json", _value(active="mainnet"))
    real_open = os.open
    swapped = False

    def swapping_open(name, flags, *args, **kwargs):
        nonlocal swapped
        if name == path.name and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            os.replace(replacement, path)
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(pay_network.os, "open", swapping_open)
    with pytest.raises(pay_network.NetworkConfigError, match="changed while opening"):
        pay_network.load_document(home=home, environ={})


def test_concurrent_use_calls_serialize_backup_and_update(
        _clean_environment, monkeypatch):
    home = _clean_environment
    path = _config(home)
    _limits(home)
    _arm(home)
    first_inside = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()
    original_atomic = pay_network._atomic_write_json_at
    results = {}
    errors = []

    def controlled_atomic(pay_fd, target, value):
        if threading.current_thread().name == "use-mainnet" and target.name == path.name:
            first_inside.set()
            assert release_first.wait(2)
        return original_atomic(pay_fd, target, value)

    monkeypatch.setattr(pay_network, "_atomic_write_json_at", controlled_atomic)

    def switch(label, network):
        try:
            results[label] = pay_network.use_network(network, home=home, environ={})
        except Exception as exc:  # surfaced below with the originating thread
            errors.append(exc)
        finally:
            if label == "second":
                second_done.set()

    first = threading.Thread(
        target=switch, args=("first", "mainnet"), name="use-mainnet")
    second = threading.Thread(
        target=switch, args=("second", "devnet"), name="use-devnet")
    first.start()
    assert first_inside.wait(2)
    second.start()
    assert not second_done.wait(0.1), "the second use call bypassed the mutation lock"
    release_first.set()
    first.join(2)
    second.join(2)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert json.loads(results["first"][1].read_text())["active"] == "devnet"
    assert json.loads(results["second"][1].read_text())["active"] == "mainnet"
    assert json.loads(path.read_text())["active"] == "devnet"


def test_invalid_join_code_keeps_forgejo_main_setup_order(
        _clean_environment, monkeypatch):
    from network import client
    import joincode

    home = _clean_environment
    events = []

    def invalid(_value):
        events.append("decode")
        raise ValueError("expired")

    class Client:
        def __init__(self, _sock):
            pass

        def redeem(self, *_args, **_kwargs):
            events.append("redeem")
            return {"state": "pending", "name": "friend"}

    def identity(_home):
        events.append("identity")
        return {key: "ready" for key in ("node_key", "person_key", "signer", "agent")}

    monkeypatch.setattr(joincode, "decode_invite", invalid)
    monkeypatch.setattr(join_setup, "ensure_identity", identity)
    monkeypatch.setattr(client, "NakClient", Client)
    monkeypatch.setenv("NAK_JOIN_NO_SYSTEMD", "1")
    with pytest.raises(SystemExit, match="join code became invalid.*expired"):
        join_setup.main(["--invite", "invalid", "--home", str(home)])
    assert events == ["decode", "identity", "redeem", "decode"]
