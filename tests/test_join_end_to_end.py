"""The whole join sequence, every phase, with no I/O — which is the point.

⚠️ Each phase has its own tests. None of them proves the SEQUENCE holds: that what one phase
puts in `facts` is what the next can read, and that a refusal stops where it should. Six
phases that each pass alone can still disagree at every seam.
"""
import json
from pathlib import Path

import pytest

from join import orchestrate
from join.cli import PHASES

GOOD = json.loads((Path(__file__).parent / "fixtures" / "join_good.json").read_text())


def run(**overrides):
    facts = dict(GOOD)
    facts.update(overrides)
    return orchestrate(facts, PHASES)


def test_a_healthy_node_joins():
    ok, problems, out = run()
    assert ok, problems
    assert out["joined"] is True
    assert out["serving_layers"] == [0, 32]
    assert out["account"].startswith("nak:")


def test_the_silent_cpu_node_is_refused_at_serve():
    """⚠️ THE FAILURE THIS PATH EXISTS FOR. Every earlier check passes, the daemon is up, the
    probe answers — and the card is idle. An installer shipped exactly this for weeks."""
    ok, problems, _ = run(running_argv=["llama-server", "--n-gpu-layers", "0"])
    assert not ok
    assert any(p.startswith("serve:") and "idle" in p for p in problems), problems


def test_ssrf_is_refused_before_anything_is_fetched():
    """A coordinator that can aim a node at cloud metadata reads its credentials. This must
    fail at ADMIT — the first phase — not after the download."""
    ok, problems, _ = run(package_url="https://169.254.169.254/latest/meta-data/")
    assert not ok
    assert problems and problems[0].startswith("admit:"), problems


def test_a_rekeyed_node_is_refused_at_identity():
    ok, problems, _ = run(roster_pubkey="c" * 64)
    assert not ok
    assert any(p.startswith("identity:") and "re-key" in p for p in problems), problems


def test_an_expired_join_code_is_refused():
    ok, problems, _ = run(join={**GOOD["join"], "expires_at": 500}, now=1000)
    assert not ok and problems[0].startswith("admit:")


def test_a_truncated_download_is_refused_at_acquire():
    ok, problems, _ = run(downloaded_bytes=999)
    assert not ok
    assert any(p.startswith("acquire:") and "size mismatch" in p for p in problems), problems


def test_a_full_disk_is_refused_at_preflight():
    ok, problems, _ = run(free_bytes=1000)
    assert not ok
    assert any(p.startswith("preflight:") and "disk" in p for p in problems), problems


@pytest.mark.parametrize("phase_break,expected", [
    ({"package_url": "http://x.example/m.gguf"}, "admit"),
    ({"port": 80}, "preflight"),
    ({"download_exit_code": 1, "download_stderr": "404 Not Found"}, "acquire"),
    ({"registered": False}, "identity"),
    ({"daemon_pid": 0}, "serve"),
    ({"probe_tokens": []}, "prove"),
])
def test_each_phase_can_stop_the_sequence(phase_break, expected):
    """⚠️ Every phase must actually be reachable AND able to refuse. A phase that can never
    stop the sequence is decoration, and one that is never reached is worse."""
    ok, problems, _ = run(**phase_break)
    assert not ok, f"{phase_break} should have stopped at {expected}"
    assert any(p.startswith(f"{expected}:") for p in problems), (expected, problems)


def test_no_credential_reaches_the_result():
    """⚠️ The join code is a bearer credential and this output is pasted into issues."""
    _, problems, out = run(join={**GOOD["join"], "token": "tok_SUPERSECRET"})
    blob = json.dumps({"p": problems, "o": {k: str(v) for k, v in out.items()
                                            if k not in ("join", "join_code")}})
    assert "tok_SUPERSECRET" not in blob
