"""Task spec, hash binding and deterministic acceptance (network/tasks.py)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from network import tasks as T  # noqa: E402

P = "c" * 64


def spec(rules, **kw):
    return T.make_spec(P, "summarise", "Summarise the attached text.", rules, now=1000, **kw)


def test_rules_all_must_pass():
    s = spec([{"max_words": 5}, {"contains_all": ["Nepal"]}, {"contains_none": ["lorem"]}])
    assert T.evaluate(s, "Nepal is in Asia") == (True, [])
    ok, why = T.evaluate(s, "lorem ipsum dolor sit amet consectetur")
    assert not ok and len(why) == 3


def test_sha256_and_json_keys_rules():
    out = "42"
    s = spec([{"sha256": hashlib.sha256(out.encode()).hexdigest()}])
    assert T.evaluate(s, "42")[0] and not T.evaluate(s, "43")[0]
    s = spec([{"json_keys": ["answer", "source"]}])
    assert T.evaluate(s, json.dumps({"answer": 1, "source": "x"}))[0]
    assert T.evaluate(s, "[1,2]")[1] == ["output is not a JSON object"]


@pytest.mark.parametrize("bad", [[], [{"regex": ".*"}], [{"max_words": -1}], [{"max_words": True}],
                                 [{"contains_all": []}], [{"sha256": "XYZ"}], [{"a": 1, "b": 2}]])
def test_bad_rules_refused(bad):
    with pytest.raises(ValueError):
        spec(bad)


def test_spec_validation():
    s = spec([{"max_words": 5}])
    T.validate_spec(s, now=1000)
    with pytest.raises(ValueError, match="deadline"):
        T.validate_spec(s, now=s["deadline"])
    for k, v in (("privacy", "sealed"), ("reward", {"amount": 5, "unit": "SOL"}), ("title", ""), ("extra", 1)):
        with pytest.raises(ValueError):
            T.validate_spec(dict(s, **{k: v}), now=1000)


def test_hash_binds_the_rules():
    s = spec([{"max_words": 5}])
    assert T.task_hash(s) != T.task_hash(dict(s, acceptance=[{"max_words": 500}]))


def test_oversized_output_fails():
    assert not T.evaluate(spec([{"max_chars": 10}]), "x" * (T.MAX_OUTPUT + 1))[0]
