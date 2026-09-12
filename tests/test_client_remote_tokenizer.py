"""Tests for RemoteTokenizer / tokenize_local's daemon path (2026-09-12, perf/resident-tokenizer-
cache) — no real model, no real tokenizer_daemon.py process: a fake Unix-socket server standing in
for the daemon, matching this suite's convention of fakes over real dependencies (see
test_client_stream_spec.py's fake streamers)."""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import client as cli  # noqa: E402


class _FakeTokenizerDaemon:
    """A real Unix socket server (so RemoteTokenizer's actual socket code runs end to end),
    scripted with canned responses per request — no llama_cpp, no gguf file."""

    def __init__(self, responder):
        self.path = tempfile.mktemp(suffix=".sock")
        self._responder = responder
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.path)
        self._server.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        try:
            conn, _ = self._server.accept()
        except OSError:
            return
        with conn:
            buf = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
                while b"\n" in buf:
                    line, _, buf = buf.partition(b"\n")
                    req = json.loads(line.decode("utf-8"))
                    resp = self._responder(req)
                    conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))

    def close(self):
        try:
            self._server.close()
        finally:
            try:
                os.unlink(self.path)
            except OSError:
                pass


def test_remote_tokenizer_tokenize_and_detokenize_round_trip():
    def responder(req):
        if req["op"] == "tokenize":
            assert req["text"] == "hello"
            return {"ids": [1, 2, 3]}
        if req["op"] == "detokenize":
            assert req["ids"] == [1, 2, 3]
            return {"text": "hello"}
        return {"error": "unexpected op"}

    daemon = _FakeTokenizerDaemon(responder)
    try:
        rt = cli.RemoteTokenizer(daemon.path)
        try:
            assert rt.tokenize(b"hello") == [1, 2, 3]
            assert rt.detokenize([1, 2, 3]) == b"hello"
        finally:
            rt.close()
    finally:
        daemon.close()


def test_remote_tokenizer_reuses_one_connection_for_many_calls():
    """A chat turn calls detokenize once per generated token — must NOT reconnect per call,
    or the whole point of a resident daemon is defeated."""
    call_count = {"n": 0}

    def responder(req):
        call_count["n"] += 1
        return {"text": "x"}

    daemon = _FakeTokenizerDaemon(responder)
    try:
        rt = cli.RemoteTokenizer(daemon.path)
        try:
            for _ in range(5):
                rt.detokenize([1])
        finally:
            rt.close()
    finally:
        daemon.close()
    assert call_count["n"] == 5, "5 logical calls, but must be over the SAME connection"


def test_remote_tokenizer_raises_on_daemon_error_response():
    daemon = _FakeTokenizerDaemon(lambda req: {"error": "boom"})
    try:
        rt = cli.RemoteTokenizer(daemon.path)
        try:
            try:
                rt.tokenize(b"x")
                assert False, "expected RuntimeError"
            except RuntimeError as e:
                assert "boom" in str(e)
        finally:
            rt.close()
    finally:
        daemon.close()


def test_tokenize_local_uses_the_daemon_when_socket_given():
    def responder(req):
        if req["op"] == "tokenize":
            return {"ids": [9, 9, 9]}
        return {"error": "unexpected"}

    daemon = _FakeTokenizerDaemon(responder)
    try:
        tokens, llama = cli.tokenize_local("/does/not/matter.gguf", "hi", daemon.path)
        assert tokens == [9, 9, 9]
        assert isinstance(llama, cli.RemoteTokenizer)
        llama.close()
    finally:
        daemon.close()


def test_tokenize_local_falls_back_to_local_load_when_socket_unreachable():
    """The daemon is a latency optimization, never a hard dependency — a stale/missing socket
    path must not fail the request, just fall back (verified here at the connect-failure level;
    tokenize_local's fallback branch then calls the real llama_cpp import, which this test does
    not exercise further — that path is what test_client_registry.py-style tests already cover
    implicitly by every existing single-worker test running without a daemon at all)."""
    import pytest
    nonexistent_socket = "/tmp/definitely-not-a-real-tokenizer-daemon-socket.sock"
    with pytest.raises(ValueError, match="does not exist"):
        # No daemon at that path -> RemoteTokenizer's connect() raises -> tokenize_local catches
        # it and falls through to the real `from llama_cpp import Llama` local-load path, which
        # then fails on the nonexistent GGUF path instead — proving the fallback branch was
        # actually reached (not that the socket error propagated unhandled).
        cli.tokenize_local("/does/not/exist.gguf", "hi", nonexistent_socket)
