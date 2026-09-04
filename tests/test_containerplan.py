import pytest
from containerplan import container_argv


def test_container_argv_exact():
    result = container_argv("myimage:latest", "/opt/install.sh",
                           network="host", memory_gb=16, timeout_s=7200)
    expected = [
        "docker", "run",
        "--rm",
        "--network", "host",
        "--memory", "16g",
        "-v", "/opt/install.sh:/provision.sh:ro",
        "--stop-timeout", "7200",
        "myimage:latest",
        "bash", "/provision.sh"
    ]
    assert result == expected


def test_container_argv_default_values():
    result = container_argv("alpine", "/install.sh")
    expected = [
        "docker", "run",
        "--rm",
        "--network", "bridge",
        "--memory", "8g",
        "-v", "/install.sh:/provision.sh:ro",
        "--stop-timeout", "3600",
        "alpine",
        "bash", "/provision.sh"
    ]
    assert result == expected


def test_container_argv_has_rm_flag():
    result = container_argv("alpine", "/install.sh")
    assert "--rm" in result


def test_container_argv_mount_is_ro():
    result = container_argv("alpine", "/install.sh")
    # The mount is split across two elements: "-v" and the spec
    mount_spec = result[result.index("-v") + 1]
    assert ":ro" in mount_spec


def test_container_argv_custom_network():
    result = container_argv("alpine", "/install.sh", network="overlay")
    assert "--network" in result
    assert "overlay" in result


def test_container_argv_custom_memory():
    result = container_argv("alpine", "/install.sh", memory_gb=4)
    assert "--memory" in result
    assert "4g" in result


def test_empty_image_raises():
    with pytest.raises(ValueError):
        container_argv("", "/install.sh")


def test_image_with_space_raises():
    with pytest.raises(ValueError):
        container_argv("my image", "/install.sh")


def test_empty_script_path_raises():
    with pytest.raises(ValueError):
        container_argv("alpine", "")


def test_relative_script_path_raises():
    with pytest.raises(ValueError):
        container_argv("alpine", "install.sh")


def test_zero_memory_raises():
    with pytest.raises(ValueError):
        container_argv("alpine", "/install.sh", memory_gb=0)


def test_negative_memory_raises():
    with pytest.raises(ValueError):
        container_argv("alpine", "/install.sh", memory_gb=-1)


def test_zero_timeout_raises():
    with pytest.raises(ValueError):
        container_argv("alpine", "/install.sh", timeout_s=0)


def test_negative_timeout_raises():
    with pytest.raises(ValueError):
        container_argv("alpine", "/install.sh", timeout_s=-1)


def test_no_privileged_flag():
    result = container_argv("alpine", "/install.sh")
    assert "--privileged" not in result


def test_no_pid_host_flag():
    result = container_argv("alpine", "/install.sh")
    assert "--pid=host" not in result


def test_no_host_mount():
    result = container_argv("alpine", "/install.sh")
    assert ":/host" not in result
    assert "/:/host" not in result