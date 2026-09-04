import pytest
from join.phase_prove import prove


def test_successful_probe():
    facts = {
        "probe_tokens": [1, 2, 3, 4],
        "probe_layers": [2, 5],
        "n_layers": 12,
        "serve_ngl": 3,
        "accel": "cuda",
        "answered_probe_ms": 150.5
    }
    success, problems, updates = prove(facts)
    assert success is True
    assert problems == []
    assert updates == {"joined": True, "serving_layers": [2, 5]}


def test_empty_probe_tokens():
    facts = {
        "probe_tokens": [],
        "probe_layers": [2, 5],
        "n_layers": 12,
        "serve_ngl": 3,
        "accel": "cuda",
        "answered_probe_ms": 150.5
    }
    success, problems, updates = prove(facts)
    assert success is False
    assert "probe_tokens is empty or not a list of ints — it produced nothing" in problems


def test_non_int_tokens():
    facts = {
        "probe_tokens": [1, "two", 3],
        "probe_layers": [2, 5],
        "n_layers": 12,
        "serve_ngl": 3,
        "accel": "cuda",
        "answered_probe_ms": 150.5
    }
    success, problems, updates = prove(facts)
    assert success is False
    assert "probe_tokens is empty or not a list of ints — it produced nothing" in problems


def test_malformed_probe_layers():
    facts = {
        "probe_tokens": [1, 2, 3],
        "probe_layers": [2],  # only one element
        "n_layers": 12,
        "serve_ngl": 3,
        "accel": "cuda",
        "answered_probe_ms": 150.5
    }
    success, problems, updates = prove(facts)
    assert success is False
    assert "probe_layers is not a 2-list" in problems


def test_range_exceeding_n_layers():
    facts = {
        "probe_tokens": [1, 2, 3],
        "probe_layers": [2, 15],
        "n_layers": 12,
        "serve_ngl": 3,
        "accel": "cuda",
        "answered_probe_ms": 150.5
    }
    success, problems, updates = prove(facts)
    assert success is False
    assert "probe_layers is not a 2-list with 0 <= start < end <= n_layers" in problems


def test_inverted_range():
    facts = {
        "probe_tokens": [1, 2, 3],
        "probe_layers": [5, 2],
        "n_layers": 12,
        "serve_ngl": 3,
        "accel": "cuda",
        "answered_probe_ms": 150.5
    }
    success, problems, updates = prove(facts)
    assert success is False
    assert "probe_layers is not a 2-list with 0 <= start < end <= n_layers" in problems


def test_ngl_zero_on_cuda_idle_card():
    facts = {
        "probe_tokens": [1, 2, 3],
        "probe_layers": [2, 5],
        "n_layers": 12,
        "serve_ngl": 0,
        "accel": "cuda",
        "answered_probe_ms": 150.5
    }
    success, problems, updates = prove(facts)
    assert success is False
    assert "a node can pass every earlier check, answer the probe, and still be running on CPU with an idle card" in problems


def test_ngl_zero_on_cpu_is_fine():
    facts = {
        "probe_tokens": [1, 2, 3],
        "probe_layers": [2, 5],
        "n_layers": 12,
        "serve_ngl": 0,
        "accel": "cpu",
        "answered_probe_ms": 150.5
    }
    success, problems, updates = prove(facts)
    assert success is True
    assert problems == []
    assert updates == {"joined": True, "serving_layers": [2, 5]}


def test_90_second_probe():
    facts = {
        "probe_tokens": [1, 2, 3],
        "probe_layers": [2, 5],
        "n_layers": 12,
        "serve_ngl": 3,
        "accel": "cuda",
        "answered_probe_ms": 90000  # 90 seconds
    }
    success, problems, updates = prove(facts)
    assert success is False
    assert "it answered, eventually, in a way no requester will wait for" in problems


def test_missing_observations():
    facts = {
        "probe_tokens": [1, 2, 3],
        # missing probe_layers, n_layers, serve_ngl, accel, answered_probe_ms
    }
    success, problems, updates = prove(facts)
    assert success is False
    assert len(problems) >= 1  # At least one problem should be reported


def test_hostile_types():
    facts = {
        "probe_tokens": "not a list",
        "probe_layers": "not a list",
        "n_layers": "twelve",
        "serve_ngl": "three",
        "accel": 123,
        "answered_probe_ms": "fast"
    }
    success, problems, updates = prove(facts)
    assert success is False
    assert len(problems) >= 1


def test_determinism():
    facts = {
        "probe_tokens": [1, 2, 3],
        "probe_layers": [2, 5],
        "n_layers": 12,
        "serve_ngl": 3,
        "accel": "cuda",
        "answered_probe_ms": 150.5
    }
    results = [prove(facts) for _ in range(5)]
    assert all(r == results[0] for r in results)