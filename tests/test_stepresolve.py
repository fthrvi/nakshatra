import pytest
from stepresolve import next_steps


def test_empty_done_returns_only_dependency_free_steps():
    steps = [
        {"id": "a", "needs": []},
        {"id": "b", "needs": ["a"]},
    ]
    result = next_steps(steps, [])
    assert result == [{"id": "a", "needs": []}]


def test_step_appears_when_needs_are_met():
    steps = [
        {"id": "a", "needs": []},
        {"id": "b", "needs": ["a"]},
    ]
    result = next_steps(steps, ["a"])
    assert result == [{"id": "b", "needs": ["a"]}]


def test_done_id_never_reappears():
    steps = [
        {"id": "a", "needs": []},
        {"id": "b", "needs": ["a"]},
    ]
    result1 = next_steps(steps, [])
    result2 = next_steps(steps, ["a"])
    assert result1 == [{"id": "a", "needs": []}]
    assert result2 == [{"id": "b", "needs": ["a"]}]


def test_step_needing_nonexistent_id_never_returned():
    steps = [
        {"id": "a", "needs": ["nonexistent"]},
        {"id": "b", "needs": []},
    ]
    result = next_steps(steps, [])
    assert result == [{"id": "b", "needs": []}]


def test_unknown_id_in_done_ignored():
    steps = [
        {"id": "a", "needs": []},
        {"id": "b", "needs": ["a"]},
    ]
    result = next_steps(steps, ["unknown_id"])
    assert result == [{"id": "a", "needs": []}]


def test_all_done_returns_empty_list():
    steps = [
        {"id": "a", "needs": []},
        {"id": "b", "needs": ["a"]},
    ]
    result = next_steps(steps, ["a", "b"])
    assert result == []


def test_step_needing_two_others():
    steps = [
        {"id": "a", "needs": []},
        {"id": "b", "needs": []},
        {"id": "c", "needs": ["a", "b"]},
    ]
    result = next_steps(steps, ["a"])
    assert result == [{"id": "b", "needs": []}]
    
    result = next_steps(steps, ["a", "b"])
    assert result == [{"id": "c", "needs": ["a", "b"]}]


def test_order_follows_steps_not_alphabetical():
    steps = [
        {"id": "z", "needs": []},
        {"id": "a", "needs": []},
        {"id": "m", "needs": []},
    ]
    result = next_steps(steps, [])
    assert result == [
        {"id": "z", "needs": []},
        {"id": "a", "needs": []},
        {"id": "m", "needs": []},
    ]


def test_malformed_steps_returns_empty_list():
    assert next_steps("not a list", []) == []
    assert next_steps([1, 2, 3], []) == []
    assert next_steps([{"id": "a"}], []) == []
    assert next_steps([{"id": "a", "needs": "not a list"}], []) == []


def test_malformed_done_returns_empty_list():
    assert next_steps([], "not a list") == []
    assert next_steps([], [1, 2, 3]) == []


def test_empty_steps_and_done():
    assert next_steps([], []) == []