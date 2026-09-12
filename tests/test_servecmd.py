"""servecmd — the argv the join path hands to worker.py.

⚠️ This file was EMPTY until 2026-09-04. `serve_argv` shipped with no tests, which is how it
could omit `--pillar-url` for months: worker.py's resolve_auth_required() reads (env unset,
pillar_url "") as "Mode A legacy" — TLS on, peer auth OFF, push-address SSRF gate OFF — and
nothing here said otherwise.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import pytest
from servecmd import serve_argv

_BASE = {"port": 8080, "role": "middle", "layer_start": 0, "layer_end": 13,
         "slice_path": "/s.gguf", "backend": "cuda", "ngl": 99}


def test_basic_shape():
    a = serve_argv("/py", "/w.py", dict(_BASE))
    assert a[:2] == ["/py", "/w.py"]
    for flag in ("--port", "--role", "--layer-start", "--layer-end", "--sub-gguf",
                 "--gpu-backend", "--n-gpu-layers"):
        assert flag in a


def test_missing_required_fact_raises():
    with pytest.raises(ValueError):
        serve_argv("/py", "/w.py", {k: v for k, v in _BASE.items() if k != "port"})


def test_bad_role_and_bad_range_raise():
    with pytest.raises(ValueError):
        serve_argv("/py", "/w.py", {**_BASE, "role": "coordinator"})
    with pytest.raises(ValueError):
        serve_argv("/py", "/w.py", {**_BASE, "layer_end": 0})


# ── the auth door: a joined node MUST come up in Mode B/C, never Mode A ──────────────────

def test_join_coordinator_becomes_pillar_url_so_worker_authenticates():
    """The coordinator lives where observe.py puts it: facts['join']['coordinator']."""
    a = serve_argv("/py", "/w.py", {**_BASE, "join": {"coordinator": "https://hub.example"}})
    assert "--pillar-url" in a
    assert a[a.index("--pillar-url") + 1] == "https://hub.example"


def test_flat_coordinator_also_honoured():
    a = serve_argv("/py", "/w.py", {**_BASE, "coordinator": "https://hub.example"})
    assert a[a.index("--pillar-url") + 1] == "https://hub.example"


def test_no_coordinator_means_no_flag_not_an_empty_one():
    # An empty --pillar-url "" still reads as Mode A; omit rather than lie.
    for facts in (dict(_BASE), {**_BASE, "join": {}}, {**_BASE, "join": {"coordinator": ""}},
                  {**_BASE, "join": "not-a-dict"}, {**_BASE, "coordinator": None}):
        assert "--pillar-url" not in serve_argv("/py", "/w.py", facts)
