import subprocess

import pytest

from conftest import _FORBIDDEN_HOST_COMMANDS


def test_forbidden_host_command_guard_and_path_tripwires(
        no_real_system_commands, tmp_path):
    for name in _FORBIDDEN_HOST_COMMANDS:
        with pytest.raises(AssertionError, match=rf"TRIPWIRE: test reached real {name}"):
            subprocess.run([name, "--test-must-not-run"], check=False)

        fake = tmp_path / name
        fake.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' 'TRIPWIRE: test reached real {name}' >&2\n"
            "exit 97\n"
        )
        fake.chmod(0o755)
        result = subprocess.run([str(fake)], capture_output=True, text=True, check=False)
        assert result.returncode == 97
        assert result.stderr.strip() == f"TRIPWIRE: test reached real {name}"

        nested = subprocess.run(
            ["bash", "-c", 'exec "$1"', "tripwire-shell", name],
            capture_output=True,
            text=True,
            check=False,
        )
        assert nested.returncode == 97
        assert nested.stderr.strip() == f"TRIPWIRE: test reached real {name}"


def test_guard_survives_a_test_monkeypatching_subprocess_run(monkeypatch):
    real_run = subprocess.run
    calls = []
    monkeypatch.setattr(
        subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)) or "mocked")

    assert subprocess.run(["systemctl", "--user", "restart", "test.service"]) == "mocked"
    assert calls == [(["systemctl", "--user", "restart", "test.service"], {})]

    monkeypatch.undo()
    with pytest.raises(AssertionError, match="TRIPWIRE: test reached real systemctl"):
        real_run(["systemctl", "--user", "restart", "test.service"])
