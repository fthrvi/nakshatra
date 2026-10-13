"""Client defaults are derived only from isolated, uid-owned release state."""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import client_defaults as defaults  # noqa: E402


def test_explicit_environment_wins_over_release_defaults(tmp_path):
    registrar = tmp_path / ".nakshatra" / "registrar.env"
    registrar.parent.mkdir()
    registrar.write_text("NKS_NODE_ID=registrar\n")
    env = {
        "NAKSHATRA_P2P_SOCKET": "/explicit/p2p.sock",
        "NKS_REGISTRY_KEYID": "explicit-node",
        "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
    }

    assert defaults.resolve_p2p_socket(env) == "/explicit/p2p.sock"
    assert defaults.resolve_registry_keyid(
        env, home=tmp_path,
        key_loader=lambda _path: (_ for _ in ()).throw(AssertionError("key loaded")),
    ) == ("explicit-node", "NKS_REGISTRY_KEYID environment variable")


def test_socket_default_requires_an_existing_uid_owned_unix_socket(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    socket_dir = runtime / "nakshatra"
    socket_dir.mkdir(parents=True)
    socket_path = socket_dir / "p2p.sock"
    env = {"XDG_RUNTIME_DIR": str(runtime)}

    assert defaults.resolve_p2p_socket(env) == ""
    socket_path.write_text("not a socket")
    assert defaults.resolve_p2p_socket(env) == ""
    socket_path.unlink()
    control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        control.bind(str(socket_path))
        assert defaults.resolve_p2p_socket(env) == str(socket_path)
        real_uid = os.getuid()
        monkeypatch.setattr(defaults.os, "getuid", lambda: real_uid + 1)
        assert defaults.resolve_p2p_socket(env) == ""
    finally:
        control.close()


def test_socket_path_falls_back_to_run_user_uid():
    assert defaults.default_p2p_socket_path({}, uid=1234) == Path(
        "/run/user/1234/nakshatra/p2p.sock"
    )


def test_registry_keyid_comes_from_release_registrar_env(tmp_path):
    registrar = tmp_path / ".nakshatra" / "registrar.env"
    registrar.parent.mkdir()
    registrar.write_text('# release-managed registrar\nNKS_NODE_ID="hub"\n')

    keyid, source = defaults.resolve_registry_keyid(
        {}, home=tmp_path,
        key_loader=lambda _path: (_ for _ in ()).throw(AssertionError("key loaded")),
    )

    assert keyid == "hub"
    assert source == f"NKS_NODE_ID in {registrar}"


def test_registry_keyid_falls_back_to_release_worker_public_key(tmp_path):
    public_key_hex = "ab" * 32
    seen = []

    keyid, source = defaults.resolve_registry_keyid(
        {}, home=tmp_path,
        key_loader=lambda path: (seen.append(path) or b"private", public_key_hex),
    )

    key_path = tmp_path / ".nakshatra" / "keys" / "worker.ed25519"
    assert keyid == public_key_hex
    assert source == f"worker public key at {key_path}"
    assert seen == [key_path]
