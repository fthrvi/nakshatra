"""Golden compatibility checks against the live hub parent, cd3ae8b.

The parent serve/accounting/ledger sources are extracted with ``git show``
into an isolated temporary scripts directory and imported from there. Bodies
are compared after removing only top-level protocol-generated volatile fields:
``id``, ``created``, ``created_at``, and ``total_duration``.
"""
from __future__ import annotations

import http.client
import importlib
import importlib.util
import io
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import nakshatra_serve as current_serve  # noqa: E402


PARENT = "cd3ae8b"
VOLATILE_FIELDS = frozenset({"id", "created", "created_at", "total_duration"})


@pytest.fixture(scope="module")
def parent_serve(tmp_path_factory):
    """Load the reviewed parent and its accounting stack from git objects."""
    scripts = tmp_path_factory.mktemp("parent-cd3ae8b")
    for name in ("nakshatra_serve.py", "accounting.py", "ledger_client.py"):
        shown = subprocess.run(
            ["git", "show", f"{PARENT}:scripts/{name}"],
            cwd=ROOT, check=True, capture_output=True)
        (scripts / name).write_bytes(shown.stdout)

    saved = {name: sys.modules.pop(name, None)
             for name in ("accounting", "ledger_client")}
    sys.path.insert(0, str(scripts))
    try:
        # Prove the parent dependency pair really resolves from the extracted
        # directory rather than silently borrowing HEAD's already-imported copy.
        ledger = importlib.import_module("ledger_client")
        accounting = importlib.import_module("accounting")
        assert Path(ledger.__file__).parent == scripts
        assert Path(accounting.__file__).parent == scripts

        spec = importlib.util.spec_from_file_location(
            "nakshatra_serve_cd3ae8b", scripts / "nakshatra_serve.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(scripts))
        sys.modules.pop("accounting", None)
        sys.modules.pop("ledger_client", None)
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module


@contextmanager
def _running(module, backend):
    entry = module.ModelEntry(
        name="golden-model", tokenizer_gguf="/fake/model.gguf",
        chain_yaml="/fake/chain.yaml", num_layers=2)
    server = module.build_server(
        "127.0.0.1", 0, {entry.name: entry}, backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.01)
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@dataclass(frozen=True)
class Snapshot:
    status: int | None
    content_type: str | None
    body: object
    transport_error: str | None = None


def _without_volatile(value):
    """Strip generated envelope fields, preserving every nested semantic key."""
    if not isinstance(value, dict):
        return value
    return {key: item for key, item in value.items()
            if key not in VOLATILE_FIELDS}


def test_normalisation_preserves_nested_ids():
    value = {"id": "volatile", "created": 123,
             "choice": {"id": "semantic", "created": "also-semantic"}}
    assert _without_volatile(value) == {
        "choice": {"id": "semantic", "created": "also-semantic"}}


def _normalise_body(content_type: str, body: bytes):
    text = body.decode("utf-8")
    if "text/event-stream" in content_type:
        events = []
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            events.append(data if data == "[DONE]"
                          else _without_volatile(json.loads(data)))
        return events
    if "application/x-ndjson" in content_type:
        return [_without_volatile(json.loads(line))
                for line in text.splitlines() if line.strip()]
    return _without_volatile(json.loads(text))


def _request(port: int, path: str, payload: dict) -> Snapshot:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=2)
    except urllib.error.HTTPError as error:
        response = error
    except (http.client.RemoteDisconnected, urllib.error.URLError) as error:
        return Snapshot(None, None, None, type(error).__name__)
    with response:
        content_type = response.headers.get("Content-Type", "")
        body = response.read()
        return Snapshot(response.status, content_type,
                        _normalise_body(content_type, body))


def _protocol_snapshot(module, path: str, *, stream: bool) -> Snapshot:
    if stream:
        backend = module.ChainChatBackend()
        backend._ledger_hook_cached = SimpleNamespace(
            enabled=True, wants_receipt=True, fail_closed=False,
            gate=lambda _estimate: (True, ""), settle=lambda _path: None)
    else:
        backend = module.StubChatBackend("golden reply")
    payload = {
        "model": "golden-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
    }
    with _running(module, backend) as port:
        return _request(port, path, payload)


@pytest.mark.parametrize(
    ("name", "path", "stream", "content_type"),
    [
        ("openai_nonstream", "/v1/chat/completions", False, "application/json"),
        ("openai_sse", "/v1/chat/completions", True, "text/event-stream"),
        ("ollama_nonstream", "/api/chat", False, "application/json"),
        ("ollama_ndjson", "/api/chat", True, "application/x-ndjson"),
    ],
)
def test_default_protocol_golden_matches_parent(
        parent_serve, monkeypatch, name, path, stream, content_type):
    monkeypatch.setenv("NAKSHATRA_CREDITS", "1")
    monkeypatch.delenv("NAKSHATRA_LEDGER_MODE", raising=False)
    if stream:
        class FakePopen:
            def __init__(self, *_args, **_kwargs):
                self.stdout = io.StringIO(
                    "[chain] step 1: id=1 'golden'\n"
                    "[chain] step 2: id=2 ' reply'\n")
                self.stderr = io.StringIO("")

            def wait(self):
                return 0

        monkeypatch.setattr(subprocess, "Popen", FakePopen)

    parent = _protocol_snapshot(parent_serve, path, stream=stream)
    current = _protocol_snapshot(current_serve, path, stream=stream)

    assert current == parent, name
    assert current.status == 200
    assert current.content_type.startswith(content_type)


class _FakeLedgerHook:
    """The ledger seam used by both extracted parent and HEAD backends."""

    enabled = True
    fail_closed = False

    def __init__(self, scenario: str):
        self.scenario = scenario
        self.wants_receipt = scenario == "free_settle_charged_0"

    def gate(self, _estimate):
        if self.scenario == "ledger_402":
            return False, "insufficient credits (balance 1, cost 2)"
        if self.scenario == "gate_exception":
            raise RuntimeError("fake gate exploded")
        if self.scenario == "ledger_down":
            return True, "ledger unavailable; fail-open"
        return True, ""

    def settle(self, receipt_path):
        assert Path(receipt_path).is_file()
        if self.scenario == "free_settle_charged_0":
            return {"charged": 0}
        return None


def _ledger_snapshot(module, scenario: str) -> Snapshot:
    backend = module.ChainChatBackend()
    backend._ledger_hook_cached = _FakeLedgerHook(scenario)
    payload = {
        "model": "golden-model",
        "messages": [{"role": "user", "content": "meter this"}],
        "stream": False,
    }
    with _running(module, backend) as port:
        return _request(port, "/v1/chat/completions", payload)


@pytest.mark.parametrize(
    "scenario",
    ["allow", "free_settle_charged_0", "ledger_402", "gate_exception", "ledger_down"],
)
def test_default_ledger_golden_matches_parent(
        parent_serve, monkeypatch, scenario):
    monkeypatch.setenv("NAKSHATRA_CREDITS", "1")
    monkeypatch.delenv("NAKSHATRA_LEDGER_MODE", raising=False)

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0, stderr="", stdout="[chain] gen:  'metered reply'\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    parent = _ledger_snapshot(parent_serve, scenario)
    current = _ledger_snapshot(current_serve, scenario)

    assert current == parent, scenario
    if scenario == "ledger_402":
        assert current.status == 502
        assert current.body == {"error": {
            "message": "chain generation failed", "type": "server_error", "code": None}}
    elif scenario == "gate_exception":
        assert current.status == 500
        assert current.body == {"error": "internal server error"}
    else:
        assert current.status == 200
        assert "nakshatra" not in current.body
