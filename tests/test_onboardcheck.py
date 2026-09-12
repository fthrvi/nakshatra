import pytest
from onboardcheck import onboard_succeeded


def test_successful_run():
    result = {
        "exit_code": 0,
        "stdout": "all good",
        "daemon_pid": 12345,
        "running_argv": ["--n-gpu-layers", "99"],
        "detected_accel": "cuda",
        "http_status": 200
    }
    ok, problems = onboard_succeeded(result)
    assert ok is True
    assert problems == []


def test_nonzero_exit():
    result = {
        "exit_code": 1,
        "stdout": "some error",
        "daemon_pid": 12345,
        "running_argv": ["--n-gpu-layers", "99"],
        "detected_accel": "cuda",
        "http_status": 200
    }
    ok, problems = onboard_succeeded(result)
    assert ok is False
    assert "exit_code is not 0" in problems


def test_missing_pid():
    result = {
        "exit_code": 0,
        "stdout": "all good",
        "daemon_pid": None,
        "running_argv": ["--n-gpu-layers", "99"],
        "detected_accel": "cuda",
        "http_status": 200
    }
    ok, problems = onboard_succeeded(result)
    assert ok is False
    assert "daemon_pid is missing or not a positive integer" in problems


def test_ngl_0_on_cuda_ignoring_stdout():
    # Even though stdout claims 99, the process table shows 0
    result = {
        "exit_code": 0,
        "stdout": "serve flags: --n-gpu-layers 99",
        "daemon_pid": 12345,
        "running_argv": ["--n-gpu-layers", "0"],
        "detected_accel": "cuda",
        "http_status": 200
    }
    ok, problems = onboard_succeeded(result)
    assert ok is False
    assert "--n-gpu-layers is 0 while detected_accel is cuda" in problems


def test_ngl_0_on_cpu_is_fine():
    result = {
        "exit_code": 0,
        "stdout": "serve flags: --n-gpu-layers 0",
        "daemon_pid": 12345,
        "running_argv": ["--n-gpu-layers", "0"],
        "detected_accel": "cpu",
        "http_status": 200
    }
    ok, problems = onboard_succeeded(result)
    assert ok is True
    assert problems == []


def test_ngl_with_equals_form():
    result = {
        "exit_code": 0,
        "stdout": "all good",
        "daemon_pid": 12345,
        "running_argv": ["--n-gpu-layers=99"],
        "detected_accel": "rocm",
        "http_status": 200
    }
    ok, problems = onboard_succeeded(result)
    assert ok is True
    assert problems == []


def test_ngl_max_not_mistaken():
    result = {
        "exit_code": 0,
        "stdout": "all good",
        "daemon_pid": 12345,
        "running_argv": ["--n-gpu-layers-max", "100"],
        "detected_accel": "cuda",
        "http_status": 200
    }
    ok, problems = onboard_succeeded(result)
    assert ok is False
    assert "--n-gpu-layers not found in running_argv while detected_accel is cuda" in problems


def test_http_503():
    result = {
        "exit_code": 0,
        "stdout": "all good",
        "daemon_pid": 12345,
        "running_argv": ["--n-gpu-layers", "99"],
        "detected_accel": "cuda",
        "http_status": 503
    }
    ok, problems = onboard_succeeded(result)
    assert ok is False
    assert "http_status is not 200" in problems


def test_several_problems_at_once():
    result = {
        "exit_code": 1,
        "stdout": "some error",
        "daemon_pid": 0,
        "running_argv": ["--n-gpu-layers", "0"],
        "detected_accel": "cuda",
        "http_status": 503
    }
    ok, problems = onboard_succeeded(result)
    assert ok is False
    assert len(problems) == 4
    assert "exit_code is not 0" in problems
    assert "daemon_pid is missing or not a positive integer" in problems
    assert "--n-gpu-layers is 0 while detected_accel is cuda" in problems
    assert "http_status is not 200" in problems


def test_malformed_and_missing_keys():
    result = {
        "exit_code": "not_an_int",
        "daemon_pid": "not_an_int",
        "running_argv": "not_a_list",
        "detected_accel": "vulkan",
        "http_status": "not_an_int"
    }
    ok, problems = onboard_succeeded(result)
    assert ok is False
    assert len(problems) >= 4


def test_never_raises_on_any_result():
    # Test with None
    try:
        onboard_succeeded(None)
    except Exception:
        pytest.fail("onboard_succeeded raised on None input")
    
    # Test with empty dict
    try:
        ok, problems = onboard_succeeded({})
        assert isinstance(ok, bool)
        assert isinstance(problems, list)
    except Exception:
        pytest.fail("onboard_succeeded raised on empty dict")
    
    # Test with weird types
    try:
        ok, problems = onboard_succeeded({
            "exit_code": "abc",
            "daemon_pid": -1,
            "running_argv": [1, 2, 3],
            "detected_accel": 123,
            "http_status": None
        })
        assert isinstance(ok, bool)
        assert isinstance(problems, list)
    except Exception:
        pytest.fail("onboard_succeeded raised on weird types")