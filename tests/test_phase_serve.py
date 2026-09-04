import pytest
from join.phase_serve import serve


def test_healthy_gpu_node():
    facts = {
        "daemon_pid": 12345,
        "running_argv": ["llama", "--n-gpu-layers", "99"],
        "accel": "cuda",
        "answered_probe_ms": 150.0,
    }
    success, problems, updates = serve(facts)
    assert success is True
    assert problems == []
    assert updates == {"serving": True, "serve_ngl": 99}


def test_ngl_0_on_cuda_card_idle():
    facts = {
        "daemon_pid": 12345,
        "running_argv": ["llama", "--n-gpu-layers", "0"],
        "accel": "cuda",
        "answered_probe_ms": 150.0,
    }
    success, problems, updates = serve(facts)
    assert success is False
    assert "the daemon is up and the card is idle" in problems


def test_ngl_parsed_from_equals_form():
    facts = {
        "daemon_pid": 12345,
        "running_argv": ["llama", "--n-gpu-layers=42"],
        "accel": "rocm",
        "answered_probe_ms": 100.0,
    }
    success, problems, updates = serve(facts)
    assert success is True
    assert problems == []
    assert updates == {"serving": True, "serve_ngl": 42}


def test_n_gpus_layers_max_not_mistaken():
    facts = {
        "daemon_pid": 12345,
        "running_argv": ["llama", "--n-gpu-layers-max", "100"],
        "accel": "cuda",
        "answered_probe_ms": 150.0,
    }
    success, problems, updates = serve(facts)
    assert success is False
    assert "running_argv has no --n-gpu-layers" in problems


def test_flag_absent():
    facts = {
        "daemon_pid": 12345,
        "running_argv": ["llama"],
        "accel": "cuda",
        "answered_probe_ms": 150.0,
    }
    success, problems, updates = serve(facts)
    assert success is False
    assert "running_argv has no --n-gpu-layers" in problems


def test_ngl_99_on_cpu():
    facts = {
        "daemon_pid": 12345,
        "running_argv": ["llama", "--n-gpu-layers", "99"],
        "accel": "cpu",
        "answered_probe_ms": 150.0,
    }
    success, problems, updates = serve(facts)
    assert success is False
    assert "claiming offload with no GPU" in problems


def test_bad_pid():
    facts = {
        "daemon_pid": -1,
        "running_argv": ["llama", "--n-gpu-layers", "99"],
        "accel": "cuda",
        "answered_probe_ms": 150.0,
    }
    success, problems, updates = serve(facts)
    assert success is False
    assert "daemon_pid is not a positive int" in problems


def test_probe_none():
    facts = {
        "daemon_pid": 12345,
        "running_argv": ["llama", "--n-gpu-layers", "99"],
        "accel": "cuda",
        "answered_probe_ms": None,
    }
    success, problems, updates = serve(facts)
    assert success is False
    assert "probe did not answer in time" in problems


def test_probe_zero():
    facts = {
        "daemon_pid": 12345,
        "running_argv": ["llama", "--n-gpu-layers", "99"],
        "accel": "cuda",
        "answered_probe_ms": 0,
    }
    success, problems, updates = serve(facts)
    assert success is False
    assert "probe did not answer in time" in problems


def test_probe_too_slow():
    facts = {
        "daemon_pid": 12345,
        "running_argv": ["llama", "--n-gpu-layers", "99"],
        "accel": "cuda",
        "answered_probe_ms": 60000,
    }
    success, problems, updates = serve(facts)
    assert success is False
    assert "probe did not answer in time" in problems


def test_missing_observations():
    facts = {
        "daemon_pid": 12345,
        # missing running_argv, accel, answered_probe_ms
    }
    success, problems, updates = serve(facts)
    assert success is False
    assert len(problems) >= 1


def test_hostile_types():
    facts = {
        "daemon_pid": "not an int",
        "running_argv": "not a list",
        "accel": "cuda",
        "answered_probe_ms": 150.0,
    }
    success, problems, updates = serve(facts)
    assert success is False
    assert "daemon_pid is not a positive int" in problems


def test_determinism():
    facts = {
        "daemon_pid": 12345,
        "running_argv": ["llama", "--n-gpu-layers", "99"],
        "accel": "cuda",
        "answered_probe_ms": 150.0,
    }
    results = [serve(facts) for _ in range(5)]
    assert all(r == results[0] for r in results)