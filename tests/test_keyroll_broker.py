"""Key-roll coverage for the release approval-broker new-key side."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ed25519


REL = Path(__file__).resolve().parent.parent / "release"
sys.path.insert(0, str(REL))

import keyroll as KR  # noqa: E402
import releasekit as rk  # noqa: E402
from test_release_broker import _broker_config  # noqa: E402


def test_keyroll_new_side_broker_happy_path_is_one_enqueue(tmp_path):
    old = ed25519.Ed25519PrivateKey.generate()
    old_path = tmp_path / "old.key"
    old_path.write_text(old.private_bytes_raw().hex())
    output = tmp_path / "keyroll.json"
    with _broker_config(tmp_path) as (broker, config, private):
        assert KR.main([
            "make", "--old-key", str(old_path), "--new-broker", str(config),
            "--issued-at", "123", "--reason", "move into broker custody",
            "--out", str(output),
        ]) == 0
        chain = rk.validate_keyroll_chain(json.loads(output.read_text()))
        assert chain[0]["new_pubkey"] == private.public_key().public_bytes_raw().hex()
        assert len(broker.enqueues) == 1 and broker.sign_calls == 1
        assert broker.enqueues[0]["exact_content"]["kind"] == "keyroll"
