"""deploy/deep-bw/nks-q3a.blackwell.sh - blackwell's stage must YIELD to a model already on the card.

Real bash; nvidia-smi / systemctl / systemd-run are stubs on PATH so nothing touches a GPU or a service.
"""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "deep-bw" / "nks-q3a.blackwell.sh"


def _run(tmp_path, *, free_mb=None, active=False, env=None):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    stubs = {
        "systemctl": f'echo "systemctl $*" >> {log}; [ "$2" = "is-active" ] && exit {0 if active else 3}; exit 0',
        "systemd-run": f'echo "systemd-run $*" >> {log}; exit 0',
        "pkill": "exit 0",
    }
    if free_mb is not None:
        stubs["nvidia-smi"] = f'echo "{free_mb}"'
    for name, body in stubs.items():
        p = bindir / name
        p.write_text("#!/usr/bin/env bash\n" + body + "\n")
        p.chmod(0o755)
    e = {"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path), **(env or {})}
    r = subprocess.run(["bash", str(SCRIPT), "start"], capture_output=True, text=True, env=e, timeout=20)
    return r, (log.read_text() if log.exists() else "")


def test_it_refuses_when_another_model_holds_the_card(tmp_path):
    """Ollama's 10 GB coder loaded -> ~2 GB free: do not start (would OOM, or evict the coder)."""
    r, calls = _run(tmp_path, free_mb=2100)
    assert r.returncode == 75 and "refusing to start" in r.stderr and "2100" in r.stderr
    assert "systemd-run" not in calls


def test_it_starts_when_the_card_is_free(tmp_path):
    r, calls = _run(tmp_path, free_mb=12227)
    assert r.returncode == 0 and "systemd-run" in calls


def test_the_threshold_is_overridable(tmp_path):
    r, _ = _run(tmp_path, free_mb=5000, env={"NKS_MIN_FREE_VRAM_MB": "4000"})
    assert r.returncode == 0


def test_without_nvidia_smi_it_has_no_opinion_and_starts(tmp_path):
    r, calls = _run(tmp_path, free_mb=None)
    assert r.returncode == 0 and "systemd-run" in calls


def test_an_already_running_stage_is_left_alone_even_though_it_is_what_holds_the_vram(tmp_path):
    """The guard must not refuse to 'start' a worker that is already up: its own 6.5 GB is why free VRAM is low."""
    r, calls = _run(tmp_path, free_mb=3000, active=True)
    assert r.returncode == 0 and "systemd-run" not in calls
