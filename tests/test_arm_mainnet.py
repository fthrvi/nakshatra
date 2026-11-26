"""Hub-side release-key mainnet arm requester."""
from __future__ import annotations

import base64
import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "release"))

import arm_mainnet  # noqa: E402
import broker_signer  # noqa: E402
import releasekit  # noqa: E402


PROGRAM = "5w5aSyzkAmgYBcG3qop5EuLgpbMhAe7rnneuFkGjYKYc"
RPC = "https://mainnet.example.test/rpc"


def test_build_statement_is_exact_canonical_shared_format(tmp_path):
    limits = tmp_path / "limits.mainnet.json"
    limits.write_bytes(b'{"cap_total":1}\n')
    statement = arm_mainnet.build_statement(
        node_pubkey="ab" * 32, rpc=RPC, program=PROGRAM, limits=limits,
        issued_at=1_700_000_000)
    assert set(statement) == {
        "schema", "node_pubkey", "network", "genesis", "rpc", "program", "mint",
        "limits_sha256", "issued_at",
    }
    assert statement["schema"] == "nakshatra.mainnet-arm.v1"
    assert statement["limits_sha256"] == hashlib.sha256(limits.read_bytes()).hexdigest()
    assert releasekit.canonical(statement) == json.dumps(
        statement, sort_keys=True, separators=(",", ":")).encode()


def test_broker_signer_uses_one_mainnet_arm_item_and_rejects_substitution(monkeypatch):
    statement = {
        "schema": "nakshatra.mainnet-arm.v1", "node_pubkey": "ab" * 32,
        "network": "mainnet", "genesis": arm_mainnet.GENESIS, "rpc": RPC,
        "program": PROGRAM, "mint": arm_mainnet.MINT,
        "limits_sha256": "cd" * 32, "issued_at": 1_700_000_000,
    }
    key = ed25519.Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes_raw().hex()
    raw = releasekit.canonical(statement)
    seen = {}

    def sign(kind, items, summary, config_dir):
        seen.update(kind=kind, items=items, summary=summary, config_dir=config_dir)
        return [key.sign(raw)], [raw], public

    monkeypatch.setattr(broker_signer, "_sign", sign)
    signature = broker_signer.sign_mainnet_arm(statement, Path("broker"))
    assert seen["kind"] == "mainnet_arm" and len(seen["items"]) == 1
    assert base64.b64decode(seen["items"][0]["b64"]) == raw
    assert releasekit.verify_signature(signature, public, raw)

    monkeypatch.setattr(
        broker_signer, "_sign",
        lambda *_args, **_kwargs: ([key.sign(raw)], [raw + b" "], public))
    with pytest.raises(broker_signer.BrokerSigningError, match="substituted"):
        broker_signer.sign_mainnet_arm(statement)


def test_requester_verifies_and_writes_private_signed_file(tmp_path, monkeypatch):
    limits = tmp_path / "limits.mainnet.json"
    limits.write_bytes(b'{"cap_total":1}\n')
    key = ed25519.Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes_raw().hex()

    def sign(statement, _config):
        raw = releasekit.canonical(statement)
        return {
            "alg": "Ed25519", "keyid": public[:16],
            "value": base64.b64encode(key.sign(raw)).decode(),
        }

    monkeypatch.setattr(broker_signer, "sign_mainnet_arm", sign)
    monkeypatch.setattr(broker_signer, "load_release_public", lambda _config: public)
    output = tmp_path / "node.arm"
    assert arm_mainnet.main([
        "--node-pubkey", "ab" * 32, "--rpc", RPC, "--program", PROGRAM,
        "--limits", str(limits), "--out", str(output),
    ]) == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    value = json.loads(output.read_text())
    raw = base64.b64decode(value["statement_b64"], validate=True)
    assert releasekit.verify_signature(value["signature"], public, raw)
