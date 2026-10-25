import asyncio
import gc
import logging
import os
import subprocess as _subprocess
from contextlib import suppress
from pathlib import Path as _Path

import pytest

# 2026-05-26 drive-by — import-tolerant conftest so the hardening +
# SPKI test files (which need none of the petals-derived fixtures
# below) can be collected on a stripped-down venv without --noconftest.
# When psutil / hivemind are present, the cleanup_children fixture
# does its full work; when absent, the fixture no-ops and tests still
# pass. Keeps the existing tests/ layout (no need to migrate files
# into a tests/hardened/ subdir, which wouldn't actually escape this
# conftest anyway — pytest walks up the directory tree).
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False

try:
    from hivemind.utils.crypto import RSAPrivateKey
    from hivemind.utils.logging import get_logger
    from hivemind.utils.mpfuture import MPFuture
    _HIVEMIND_AVAILABLE = True
    logger = get_logger(__name__)
except ImportError:
    RSAPrivateKey = None
    MPFuture = None
    _HIVEMIND_AVAILABLE = False
    logger = logging.getLogger(__name__)


@pytest.fixture
def event_loop():
    """
    This overrides the ``event_loop`` fixture from pytest-asyncio
    (e.g. to make it compatible with ``asyncio.subprocess``).

    This fixture is identical to the original one but does not call ``loop.close()`` in the end.
    Indeed, at this point, the loop is already stopped (i.e. next tests are free to create new loops).
    However, finalizers of objects created in the current test may reference the current loop and fail if it is closed.
    For example, this happens while using ``asyncio.subprocess`` (the ``asyncio.subprocess.Process`` finalizer
    fails if the loop is closed, but works if the loop is only stopped).
    """

    yield asyncio.get_event_loop()


@pytest.fixture(autouse=True, scope="session")
def cleanup_children():
    yield

    if _HIVEMIND_AVAILABLE:
        with RSAPrivateKey._process_wide_key_lock:
            RSAPrivateKey._process_wide_key = None

    gc.collect()  # Call .__del__() for removed objects

    if _PSUTIL_AVAILABLE:
        children = psutil.Process().children(recursive=True)
        if children:
            logger.info(f"Cleaning up {len(children)} leftover child processes")
            for child in children:
                with suppress(psutil.NoSuchProcess):
                    child.terminate()
            psutil.wait_procs(children, timeout=1)
            for child in children:
                with suppress(psutil.NoSuchProcess):
                    child.kill()

    if _HIVEMIND_AVAILABLE:
        MPFuture.reset_backend()

# ⚠️ scripts/ ON THE PATH, ONCE, HERE — not prepended into each test file.
# The modules under test live in scripts/, not an installed package, so every test that
# imports one needs it on sys.path. Doing that per-file means every new test file either
# repeats the incantation or fails with ModuleNotFoundError that looks like a missing
# dependency rather than a layout quirk. conftest.py is the one place pytest guarantees to
# load before collection, so it is the one place this belongs.
import sys as _sys
_scripts = str(_Path(__file__).resolve().parents[1] / "scripts")
if _scripts not in _sys.path:
    _sys.path.insert(0, _scripts)


_FORBIDDEN_HOST_COMMANDS = frozenset({
    "journalctl", "loginctl", "pkexec", "runuser", "su", "sudo", "systemctl",
})
_ORIGINAL_POPEN = _subprocess.Popen
_allowed_command_root = None


def _guarded_popen(args, *popenargs, **kwargs):
    """Refuse host-control commands unless a test names its tmp fake absolutely."""
    argv0 = args[0] if isinstance(args, (list, tuple)) and args else args
    try:
        command = os.fspath(argv0)
    except TypeError:
        command = ""
    if command and _Path(command).name in _FORBIDDEN_HOST_COMMANDS:
        command_path = _Path(command)
        allowed = False
        if command_path.is_absolute() and _allowed_command_root is not None:
            try:
                command_path.resolve().relative_to(_allowed_command_root.resolve())
                allowed = True
            except (OSError, ValueError):
                pass
        if not allowed:
            raise AssertionError(
                f"TRIPWIRE: test reached real {_Path(command).name}; "
                "use an absolute fake executable under tmp_path"
            )
    return _ORIGINAL_POPEN(args, *popenargs, **kwargs)


@pytest.fixture(autouse=True)
def no_real_system_commands(tmp_path, tmp_path_factory):
    """Catch direct Python launches and commands hidden behind a child process."""
    global _allowed_command_root

    tripwire = tmp_path_factory.mktemp("host-command-tripwire")
    for name in _FORBIDDEN_HOST_COMMANDS:
        script = tripwire / name
        script.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' 'TRIPWIRE: test reached real {name}' >&2\n"
            "exit 97\n"
        )
        script.chmod(0o755)

    old_path = os.environ.get("PATH")
    os.environ["PATH"] = str(tripwire) + (os.pathsep + old_path if old_path else "")
    previous_root = _allowed_command_root
    previous_popen = _subprocess.Popen
    _allowed_command_root = tmp_path
    _subprocess.Popen = _guarded_popen
    try:
        yield tripwire
    finally:
        _subprocess.Popen = previous_popen
        _allowed_command_root = previous_root
        if old_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = old_path


# ── real-machine guard (2026-10-03) ──────────────────────────────────────────────────────────────
# A release test once rewrote the hub's REAL ~/.config/systemd/user/nak-update.service (a
# monkeypatch.undo() also undid the fixture's tmp redirect). Fail the whole run if any test changes the
# real node's units or messaging/signer state, instead of finding out from a broken updater.
import hashlib as _hashlib


def _real_state_digest():
    home = _Path.home()
    systemd = home / ".config" / "systemd" / "user"
    paths = {
        home / ".nakshatra" / "net" / name
        for name in ("agent", "release-url", "name", "direct-port", "p2p-dial")
    }
    paths.update({home / ".sthambha" / "signer" / "person.pub",
                  home / ".sthambha" / "signer" / "node.pub",
                  home / ".nakshatra" / "worker.env",
                  home / ".nakshatra" / "inference.env",
                  home / ".nakshatra" / "worker" / "build.json"})
    for root in (home / ".nakshatra" / "keys", home / ".config" / "nakshatra-sidecar"):
        if root.exists():
            paths.update(p for p in root.rglob("*") if p.is_file() or p.is_symlink())
    for pattern in ("nak-*.service*", "nakshatra-*.service*"):
        for root in systemd.glob(pattern):
            if root.is_dir():
                paths.update(p for p in root.rglob("*") if p.is_file() or p.is_symlink())
            else:
                paths.add(root)
    h = _hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p).encode())
        if p.is_file():
            h.update(p.read_bytes())
        elif p.is_symlink():
            h.update(str(p.readlink()).encode())
    return h.hexdigest()


@pytest.fixture(scope="session", autouse=True)
def _never_touch_the_real_node():
    before = _real_state_digest()
    yield
    assert _real_state_digest() == before, \
        "a test modified the REAL node's systemd units or net/signer state (see tests/conftest.py guard)"
