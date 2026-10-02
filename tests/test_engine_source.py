"""U6c: the engine's one source of truth reproduces the exact tree, and refuses anything else."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LOCAL_UPSTREAM = Path.home() / "llama.cpp"     # a local clone stands in for GitHub (no network in tests)
pytestmark = pytest.mark.skipif(not (LOCAL_UPSTREAM / ".git").exists(), reason="no local llama.cpp clone")


def _run(dest, engine_dir=ROOT / "engine"):
    return subprocess.run(["bash", str(engine_dir / "source.sh"), str(dest)], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "ENGINE_UPSTREAM": str(LOCAL_UPSTREAM)},
                          timeout=300)


def test_source_reproduces_the_engine_tree(tmp_path):
    r = _run(tmp_path / "src")
    assert r.returncode == 0, r.stdout + r.stderr
    base = dict(l.split("=", 1) for l in (ROOT / "engine" / "BASE").read_text().splitlines() if "=" in l and not l.startswith("#"))
    got = subprocess.check_output(["git", "-C", str(tmp_path / "src"), "rev-parse", "HEAD^{tree}"], text=True).strip()
    assert got == base["TREE"]
    assert "solo" in (tmp_path / "src" / "examples" / "nakshatra-spike" / "worker_daemon.cpp").read_text()


def test_a_tampered_patch_is_refused_and_nothing_is_left(tmp_path):
    eng = tmp_path / "engine"
    shutil.copytree(ROOT / "engine", eng)
    p = sorted((eng / "patches").glob("*.patch"))[-1]
    p.write_text(p.read_text().replace('mode_str == "solo"', 'mode_str == "s0lo"', 1))   # a CODE line
    r = _run(tmp_path / "src", engine_dir=eng)
    assert r.returncode != 0 and not (tmp_path / "src").exists()


def test_the_stale_rebuild_script_refuses():
    r = subprocess.run(["bash", str(ROOT / "deploy" / "rebuild-worker.sh")], capture_output=True, text=True)
    assert r.returncode == 2 and "retired" in r.stderr


def test_a_modified_working_tree_is_rebuilt_not_trusted(tmp_path):
    assert _run(tmp_path / "src").returncode == 0
    f = tmp_path / "src" / "examples" / "nakshatra-spike" / "worker_daemon.cpp"
    f.write_text(f.read_text() + "\n// tampered after checkout\n")
    (tmp_path / "src" / "build").mkdir()
    (tmp_path / "src" / "build" / "CMakeCache.txt").write_text("CMAKE_PROJECT_INCLUDE:FILEPATH=/evil.cmake\n")
    r = _run(tmp_path / "src")
    assert r.returncode == 0, r.stdout + r.stderr                             # rebuilt from scratch, every run
    assert "tampered after checkout" not in f.read_text()
    assert not (tmp_path / "src" / "build").exists()                          # no stale build cache survives


def test_caller_git_hooks_and_templates_cannot_run(tmp_path):
    hooks = tmp_path / "evil-hooks"
    hooks.mkdir()
    marker = tmp_path / "hook-ran"
    for h in ("post-checkout", "applypatch-msg", "post-applypatch", "pre-applypatch"):
        (hooks / h).write_text(f"#!/bin/sh\ntouch {marker}\n")
        (hooks / h).chmod(0o755)
    tmpl = tmp_path / "tmpl"
    (tmpl / "hooks").mkdir(parents=True)
    import shutil as _sh
    for h in hooks.iterdir():
        _sh.copy(h, tmpl / "hooks" / h.name)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "ENGINE_UPSTREAM": str(LOCAL_UPSTREAM),
           "GIT_TEMPLATE_DIR": str(tmpl), "GIT_CONFIG_PARAMETERS": f"'core.hooksPath={hooks}'"}
    r = subprocess.run(["bash", str(ROOT / "engine" / "source.sh"), str(tmp_path / "src")], capture_output=True,
                       text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not marker.exists()
