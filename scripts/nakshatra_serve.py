#!/usr/bin/env python3
"""Nakshatra serving frontend — Ollama-compatible HTTP gateway.

Phase E of the 2026-05-30 Ollama-gateway sprint. Ships the HTTP scaffold
+ `/api/version` + `/health` (A), the model registry + `/api/tags` +
`/api/show` (B), non-streaming `/api/chat` (C), streaming `/api/chat`
(stream:true → newline-delimited JSON, D), and per-family chat templates
— Llama-3 + Gemma (E). Remaining: the live cluster smoke (F). The plan lives
at `~/trisul/plans/2026-05-30-nakshatra-ollama-gateway-sprint.md`.

**Why this exists:** Prithvi's live gateway
(http://203.0.113.10:8080) speaks OpenAI to users and calls its
backend via Ollama HTTP `/api/chat`. To put the fleet behind Prithvi
without touching Prithvi's code, Nakshatra serves the same wire shape
Ollama does — Prithvi flips one env var (`OLLAMA_HOST`) and routes
its `_call_ollama` path here.

**Scope discipline:** this file owns HTTP framing, endpoint dispatch,
and the model registry. Tokenization, chain orchestration, and chat
templating land in Phases C–E. The registry only needs to *describe*
which models are servable and where their chains live; it does not load
weights or touch the chain. Keep the no-models runtime dependency-free
(PyYAML is imported lazily, only when `--models` is given).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


# ── Wire constants ──────────────────────────────────────────────────


# Default port matches Ollama exactly so Prithvi's `OLLAMA_HOST`
# override is just the hostname (no port change) when the cutover
# happens. Operator can override via --port if a real Ollama is
# already running on the same machine.
DEFAULT_PORT = 11434

# Version string surfaced via /api/version. Ollama's real string
# looks like "0.1.40" — we deliberately namespace ours so consumers
# that branch on it (rare) see "this isn't real Ollama". Most clients
# (including Prithvi's `_call_ollama` per the integration-surface
# memory) ignore the version field entirely.
VERSION_STRING = "nakshatra-0.1.0-ollama-compat"

log = logging.getLogger("nakshatra_serve")


# ── Model registry (Phase B) ────────────────────────────────────────


class ModelConfigError(Exception):
    """serve_models.yaml is missing, unreadable, or invalid. Raised at
    boot so a misconfigured gateway fails fast instead of serving an
    empty/odd model list."""


@dataclass
class ModelEntry:
    """One servable model: a ``name`` Prithvi selects by, the tokenizer
    GGUF the chain was split from, and where its chain lives — a static
    cluster YAML (``chain_yaml``) or a pillar registry URL
    (``registry_url``). Chain orchestration is Phase C+; this phase only
    needs the entry to exist and be advertised via /api/tags."""
    name: str
    tokenizer_gguf: str
    chain_yaml: Optional[str] = None
    registry_url: Optional[str] = None
    details: dict = field(default_factory=dict)


def _load_models(path: str) -> dict[str, ModelEntry]:
    """Parse + validate serve_models.yaml → ``{name: ModelEntry}``.

    Format::

        models:
          - name: llama-3.3-70b
            tokenizer_gguf: /models/llama-3.3-70b/...Q4_K_M.gguf
            chain_yaml: scripts/cluster_5worker.yaml   # or registry_url: http://...
            details: {family: llama, parameter_size: "70B", quantization_level: Q4_K_M}

    Raises ``ModelConfigError`` on a missing file, bad YAML, or any entry
    lacking a name, a tokenizer, or a chain (chain_yaml | registry_url),
    or on a duplicate model name.
    """
    try:
        import yaml  # lazy: keeps the no-models runtime stdlib-only
    except ImportError as e:  # pragma: no cover
        raise ModelConfigError(f"--models needs PyYAML installed ({e})")
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
    except FileNotFoundError:
        raise ModelConfigError(f"models config not found: {path}")
    except yaml.YAMLError as e:
        raise ModelConfigError(f"models config is not valid YAML: {e}")

    if not isinstance(doc, dict) or "models" not in doc:
        raise ModelConfigError(f"{path}: expected a top-level 'models:' list")
    entries = doc["models"]
    if not isinstance(entries, list) or not entries:
        raise ModelConfigError(f"{path}: 'models' must be a non-empty list")

    registry: dict[str, ModelEntry] = {}
    for i, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ModelConfigError(f"{path}: model #{i} is not a mapping")
        name, tok = raw.get("name"), raw.get("tokenizer_gguf")
        chain_yaml, registry_url = raw.get("chain_yaml"), raw.get("registry_url")
        missing = [k for k, v in (("name", name), ("tokenizer_gguf", tok)) if not v]
        if missing:
            raise ModelConfigError(f"{path}: model #{i} missing {', '.join(missing)}")
        if not (chain_yaml or registry_url):
            raise ModelConfigError(
                f"{path}: model {name!r} needs chain_yaml or registry_url")
        if name in registry:
            raise ModelConfigError(f"{path}: duplicate model name {name!r}")
        registry[name] = ModelEntry(
            name=name, tokenizer_gguf=tok, chain_yaml=chain_yaml,
            registry_url=registry_url, details=raw.get("details") or {})
    return registry


def _ollama_tag(entry: ModelEntry) -> dict:
    """Render a ModelEntry as one Ollama /api/tags entry. Only ``name`` is
    load-bearing for Prithvi's model selection; the rest are real-where-
    cheap (``size`` + ``modified_at`` from the tokenizer GGUF if it's on
    THIS host) or stable, deterministic placeholders — never fabricated."""
    import hashlib
    import os
    size, modified_at = 0, "1970-01-01T00:00:00Z"
    try:
        st = os.stat(entry.tokenizer_gguf)
        size = st.st_size
        modified_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))
    except OSError:
        pass  # artifact lives on the chain, not the gateway host — placeholders
    digest = "sha256:" + hashlib.sha256(entry.name.encode()).hexdigest()
    details = {"format": "gguf"}
    details.update(entry.details)
    return {"name": entry.name, "model": entry.name,
            "modified_at": modified_at, "size": size,
            "digest": digest, "details": details}


# ── Chat generation backends (Phase C) ──────────────────────────────


# Default reply length when a request omits options.num_predict.
DEFAULT_NUM_PREDICT = 256


class ChainBackendError(RuntimeError):
    """The chain failed to produce a reply (subprocess error, timeout, or
    unparseable output). Mapped to a 502 at the HTTP layer."""


@dataclass
class GenerationResult:
    """What a backend returns for one /api/chat turn."""
    text: str
    eval_count: int = 0          # tokens generated
    prompt_eval_count: int = 0   # prompt tokens
    done_reason: str = "stop"


class ChatBackend:
    """Turns (model, rendered prompt) into an assistant reply. Injected into
    the server so the HTTP/format layer is testable without a live chain
    (StubChatBackend) while the real chain wiring (ChainChatBackend) stays
    swappable."""

    def generate(self, entry: ModelEntry, prompt: str,
                 max_tokens: int, options: dict) -> GenerationResult:
        raise NotImplementedError

    def generate_stream(self, entry: ModelEntry, prompt: str,
                        max_tokens: int, options: dict):
        """Yield assistant-text deltas as they're produced. Default: emit the
        full non-streaming reply as a single delta, so any backend streams
        (degenerately). Backends that can emit token-by-token override this."""
        yield self.generate(entry, prompt, max_tokens, options).text


class StubChatBackend(ChatBackend):
    """Deterministic backend for tests + local wire-smoke — point Prithvi at
    it to verify the Ollama plumbing end-to-end before the cluster is up.
    Returns a canned reply; never touches the chain."""

    def __init__(self, reply: str = "ok"):
        self._reply = reply

    def generate(self, entry, prompt, max_tokens, options):
        return GenerationResult(
            text=self._reply,
            eval_count=len(self._reply.split()),
            prompt_eval_count=len(prompt.split()),
        )

    def generate_stream(self, entry, prompt, max_tokens, options):
        # Per-word deltas so streaming consumers see N>1 chunks whose
        # concatenation equals the full reply.
        for i, word in enumerate(self._reply.split()):
            yield word if i == 0 else " " + word


def _parse_client_output(stdout: str) -> str:
    """Recover the generated text from scripts/client.py's stdout. client.py
    prints ``[chain] gen:  '<repr>'``; we literal-eval the repr to get the
    exact string back (handles embedded quotes / escaped newlines). Takes the
    LAST such line (a failure-recovery replay can print more than one)."""
    import ast
    captured = None
    for line in stdout.splitlines():
        if line.startswith("[chain] gen:"):
            captured = line[len("[chain] gen:"):].strip()
    if captured is None:
        raise ChainBackendError("client.py produced no '[chain] gen:' line")
    try:
        text = ast.literal_eval(captured)
    except (ValueError, SyntaxError) as e:
        raise ChainBackendError(f"unparseable generated text: {e}")
    if not isinstance(text, str):
        raise ChainBackendError("generated text was not a string")
    return text


class ChainChatBackend(ChatBackend):
    """Real backend: drives the live chain by invoking ``scripts/client.py``
    as a subprocess — deliberately NON-INVASIVE, so client.py's tangled,
    cluster-load-bearing ``main()`` stays untouched. The gateway renders the
    chat template; client.py tokenizes + walks the chain + detokenizes; we
    parse its generated text. Per-request subprocess spawn is fine at v0.1
    single-user latencies (one 70B chain step dwarfs the ~100ms python
    spawn); an in-process refactor is a later optimization.

    End-to-end is box-only (needs the chain up); the gateway-side parser and
    template render are unit-tested without it (Phase F does the live smoke).
    """

    # client.py prints `[chain] step N: id=<id> '<text>'` per generated
    # token (flushed) — the streaming hook below parses the quoted text.
    _STEP_RE = re.compile(r"\[chain\] step \d+: id=\S+ '(.*)'\s*$")

    def __init__(self, scripts_dir: Optional[str] = None,
                 timeout_s: float = 1800.0):
        self._scripts = scripts_dir or os.path.dirname(os.path.abspath(__file__))
        self._timeout = timeout_s

    def _cmd(self, entry, prompt, max_tokens) -> list:
        cmd = [sys.executable, os.path.join(self._scripts, "client.py"),
               "--model-path", entry.tokenizer_gguf,
               "--prompt", prompt, "--max-tokens", str(max_tokens),
               "--use-streaming"]
        if entry.chain_yaml:
            cmd += ["--config", entry.chain_yaml]
        else:
            cmd += ["--registry", entry.registry_url or "", "--model-id", entry.name]
        return cmd

    def generate(self, entry, prompt, max_tokens, options):
        import subprocess
        try:
            out = subprocess.run(self._cmd(entry, prompt, max_tokens),
                                 capture_output=True, text=True,
                                 timeout=self._timeout)
        except subprocess.TimeoutExpired:
            raise ChainBackendError(
                f"chain generation timed out after {self._timeout}s")
        if out.returncode != 0:
            raise ChainBackendError(
                f"client.py exited {out.returncode}: {out.stderr.strip()[-400:]}")
        return GenerationResult(text=_parse_client_output(out.stdout))

    def generate_stream(self, entry, prompt, max_tokens, options):
        """Stream tokens as client.py emits them: spawn it with Popen, parse
        each flushed `[chain] step ...` line, yield the detokenized text.
        Best-effort single-quote parse (a token with a literal quote is rare);
        box-smoked in Phase F."""
        import subprocess
        proc = subprocess.Popen(self._cmd(entry, prompt, max_tokens),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        try:
            for line in proc.stdout:
                m = self._STEP_RE.search(line.rstrip("\n"))
                if m:
                    yield m.group(1)
        finally:
            proc.stdout.close()
            rc = proc.wait()
            err = proc.stderr.read()
            proc.stderr.close()
        if rc != 0:
            raise ChainBackendError(f"client.py exited {rc}: {err.strip()[-400:]}")


def _render_llama3(messages: list) -> str:
    """Llama-3 chat format: ``<|start_header_id|>role<|end_header_id|>`` turns
    terminated by ``<|eot_id|>``, ending with an open assistant header."""
    parts = ["<|begin_of_text|>"]
    for m in messages:
        parts.append(f"<|start_header_id|>{m.get('role', 'user')}<|end_header_id|>"
                     f"\n\n{m.get('content', '')}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _render_gemma(messages: list) -> str:
    """Gemma chat format: ``<start_of_turn>role`` / ``<end_of_turn>`` turns,
    ending with an open ``model`` turn. Gemma has no system role and uses
    ``user``/``model`` (not ``assistant``) — fold any leading system message
    into the first user turn, and map ``assistant`` → ``model``."""
    msgs = list(messages)
    system = ""
    if msgs and msgs[0].get("role") == "system":
        system = (msgs[0].get("content") or "").strip()
        msgs = msgs[1:]
    out = ["<bos>"]
    for m in msgs:
        grole = "model" if m.get("role") == "assistant" else "user"
        content = m.get("content", "")
        if grole == "user" and system:        # prepend folded system once
            content = f"{system}\n\n{content}"
            system = ""
        out.append(f"<start_of_turn>{grole}\n{content}<end_of_turn>\n")
    out.append("<start_of_turn>model\n")
    return "".join(out)


def _render_prompt(messages: list, entry: ModelEntry) -> str:
    """Render Ollama ``messages`` into one prompt string, dispatching by model
    family (Phase E). Llama-3 and Gemma — the two families actually in play
    (the 70B chain and Prithvi's conscious gemma3) — have hand-written
    templates; unknown families default to Llama-3. The fully general path
    (apply the GGUF's embedded ``tokenizer.chat_template`` via llama_cpp) is a
    box-side follow-on; it needs the GGUF + llama_cpp, so it can't be unit-
    tested on the gateway host."""
    family = (entry.details.get("family") or "").lower()
    if "gemma" in family:
        return _render_gemma(messages)
    return _render_llama3(messages)


# ── HTTP handler ────────────────────────────────────────────────────


class NakshatraServeHandler(BaseHTTPRequestHandler):
    """Endpoint dispatch + JSON response helpers.

    Subsequent phases bolt new endpoints into ``do_GET`` / ``do_POST``
    via the same if/elif cascade pattern Sthambha's PillarHandler uses
    — keeps the read-the-handler-top-to-bottom-and-understand-every-
    route property that hand-rolled stdlib HTTP makes free.

    The model registry is read off ``self.server.models`` (set by
    ``build_server``), so handlers stay stateless per-request.
    """

    # Suppress the default per-request line BaseHTTPRequestHandler
    # writes to stderr — we emit our own line below with elapsed ms.
    def log_message(self, format, *args):
        pass

    # ── Response helpers ──────────────────────────────────────────

    def _json(self, status: int, body: dict) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _json_error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _read_json_body(self) -> dict:
        """Read + parse a JSON request body. Returns {} for an empty body;
        raises ValueError on malformed JSON (caller maps to 400)."""
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        return json.loads(raw)

    @property
    def _models(self) -> dict:
        return getattr(self.server, "models", {})

    # ── Per-request timing wrapper ────────────────────────────────

    def _dispatch(self, method: str, handler) -> None:
        """Wrap a per-method dispatcher with timing + error catch.
        Logs ``METHOD path -> status (Nms)`` to stderr regardless of
        outcome."""
        t0 = time.monotonic()
        status = 500
        try:
            status = handler()
        except Exception as e:
            # Don't leak stack traces to clients (could carry path
            # names or config bits); log them + return a generic 500.
            log.exception("handler error on %s %s", method, self.path)
            try:
                self._json_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "internal server error",
                )
            except Exception:
                # Response may already be partially sent (e.g.
                # streaming); nothing we can do.
                pass
            status = 500
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            log.info("%s %s -> %d (%.1fms)",
                     method, self.path, status, elapsed_ms)

    # ── do_GET / do_POST entry points ────────────────────────────

    def do_GET(self):
        self._dispatch("GET", self._route_get)

    def do_POST(self):
        self._dispatch("POST", self._route_post)

    # ── Dispatch tables ──────────────────────────────────────────

    def _route_get(self) -> int:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return 200
        if path == "/api/version":
            self._json(HTTPStatus.OK, {"version": VERSION_STRING})
            return 200
        if path == "/api/tags":
            self._json(HTTPStatus.OK,
                       {"models": [_ollama_tag(e) for e in self._models.values()]})
            return 200
        # Phase C+ endpoints land here as new branches.
        self._json_error(
            HTTPStatus.NOT_FOUND,
            f"unsupported endpoint: GET {path}",
        )
        return 404

    def _route_post(self) -> int:
        path = self.path.split("?", 1)[0]
        if path == "/api/show":
            return self._handle_show()
        if path == "/api/chat":
            return self._handle_chat()
        # Phase D adds streaming on the same /api/chat path (stream:true).
        self._json_error(
            HTTPStatus.NOT_FOUND,
            f"unsupported endpoint: POST {path}",
        )
        return 404

    # ── Endpoint handlers ─────────────────────────────────────────

    def _handle_show(self) -> int:
        """Ollama ``/api/show`` — return a model's details, or 404 if it
        isn't in the registry. Lets a consumer (or a smoke test) confirm a
        model is servable before issuing /api/chat. Body: ``{"name": "..."}``
        (``model`` accepted as an alias)."""
        try:
            req = self._read_json_body()
        except ValueError:
            self._json_error(HTTPStatus.BAD_REQUEST, "invalid JSON body")
            return 400
        name = req.get("name") or req.get("model")
        entry = self._models.get(name) if name else None
        if entry is None:
            self._json_error(HTTPStatus.NOT_FOUND, f"model {name!r} not found")
            return 404
        tag = _ollama_tag(entry)
        self._json(HTTPStatus.OK, {
            "details": tag["details"],
            "model_info": {"name": entry.name, "size": tag["size"],
                           "modified_at": tag["modified_at"]},
            # Phase E fills these from the GGUF; empty placeholders for now.
            "modelfile": "", "parameters": "", "template": "",
        })
        return 200

    def _handle_chat(self) -> int:
        """Ollama ``/api/chat``. Validates the model + messages, renders the
        prompt, then returns either one Ollama-shaped JSON reply (Phase C) or
        a newline-delimited JSON stream (Phase D) depending on ``stream``."""
        try:
            req = self._read_json_body()
        except ValueError:
            self._json_error(HTTPStatus.BAD_REQUEST, "invalid JSON body")
            return 400
        name = req.get("model")
        entry = self._models.get(name) if name else None
        if entry is None:
            self._json_error(HTTPStatus.NOT_FOUND, f"model {name!r} not found")
            return 404
        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            self._json_error(HTTPStatus.BAD_REQUEST,
                             "'messages' must be a non-empty list")
            return 400
        backend = getattr(self.server, "chat_backend", None)
        if backend is None:
            self._json_error(HTTPStatus.SERVICE_UNAVAILABLE,
                             "no chat backend configured")
            return 503
        options = req.get("options") or {}
        max_tokens = int(options.get("num_predict") or DEFAULT_NUM_PREDICT)
        prompt = _render_prompt(messages, entry)

        if req.get("stream"):
            return self._stream_chat(entry, prompt, max_tokens, options, backend)

        t0 = time.monotonic()
        try:
            result = backend.generate(entry, prompt, max_tokens, options)
        except ChainBackendError as e:
            log.error("chain backend error: %s", e)
            self._json_error(HTTPStatus.BAD_GATEWAY, "chain generation failed")
            return 502
        total_ns = int((time.monotonic() - t0) * 1e9)
        self._json(HTTPStatus.OK, {
            "model": entry.name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": result.text},
            "done": True,
            "done_reason": result.done_reason,
            "total_duration": total_ns,
            "prompt_eval_count": result.prompt_eval_count,
            "eval_count": result.eval_count,
        })
        return 200

    def _stream_chat(self, entry, prompt, max_tokens, options, backend) -> int:
        """Ollama streaming /api/chat (Phase D): newline-delimited JSON — N
        ``{done:false}`` content chunks then one ``{done:true}`` with summary
        stats. The body is delimited by connection close (HTTP/1.0 default);
        standard streaming clients (incl. Prithvi's line-reading
        ``_call_ollama``) consume it incrementally. Headers are already sent
        once the first chunk goes out, so a mid-stream failure can only emit a
        terminal error chunk — not change the status code."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t0 = time.monotonic()
        eval_count = 0

        def _write(obj: dict) -> None:
            self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
            self.wfile.flush()

        try:
            for delta in backend.generate_stream(entry, prompt, max_tokens, options):
                _write({"model": entry.name, "created_at": created,
                        "message": {"role": "assistant", "content": delta},
                        "done": False})
                eval_count += 1
        except ChainBackendError as e:
            log.error("stream backend error: %s", e)
            try:
                _write({"model": entry.name,
                        "error": "chain generation failed", "done": True})
            except Exception:
                pass  # client already gone
            return 502
        _write({"model": entry.name, "created_at": created,
                "message": {"role": "assistant", "content": ""},
                "done": True, "done_reason": "stop",
                "total_duration": int((time.monotonic() - t0) * 1e9),
                "eval_count": eval_count})
        return 200


# ── Server lifecycle ────────────────────────────────────────────────


def build_server(bind: str, port: int,
                 models: Optional[dict] = None,
                 chat_backend: "Optional[ChatBackend]" = None
                 ) -> ThreadingHTTPServer:
    """Construct + return a ThreadingHTTPServer bound to (bind, port).

    ``models`` is the registry from :func:`_load_models` (``{}`` if no
    ``--models`` was given); ``chat_backend`` drives ``/api/chat`` (a
    :class:`ChainChatBackend` in production, a :class:`StubChatBackend` in
    tests). Both are attached to the server so per-request handlers read
    them via ``self.server.models`` / ``self.server.chat_backend``.

    Caller is responsible for ``server.serve_forever()`` /
    ``server.shutdown()`` + ``server.server_close()``. Returning the
    server object instead of starting it lets tests spin one up on
    an ephemeral port without process-level setup."""
    server = ThreadingHTTPServer((bind, port), NakshatraServeHandler)
    server.models = models or {}
    server.chat_backend = chat_backend
    return server


# ── CLI entry point ────────────────────────────────────────────────


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Nakshatra Ollama-compat HTTP serving frontend "
                    "(Phase E — registry + /api/tags + /api/show + "
                    "/api/chat streaming & non-streaming + Llama-3/Gemma "
                    "templates).",
    )
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"port to bind (default {DEFAULT_PORT}, "
                         f"matches Ollama)")
    ap.add_argument("--bind", default="0.0.0.0",
                    help="address to bind (default 0.0.0.0 so other "
                         "tailnet hosts can reach)")
    ap.add_argument("--models", default=None,
                    help="path to serve_models.yaml (model registry). "
                         "Omit to run with an empty registry "
                         "(/api/tags returns []).")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)

    models: dict[str, ModelEntry] = {}
    if args.models:
        try:
            models = _load_models(args.models)
        except ModelConfigError as e:
            log.error("model config error: %s", e)
            return 2

    # Real chain backend whenever we have models to serve; the per-request
    # subprocess into client.py needs no construction-time deps, so this is
    # cheap even when the cluster is down (errors surface at request time).
    chat_backend = ChainChatBackend() if models else None
    server = build_server(args.bind, args.port, models, chat_backend)
    log.info("nakshatra_serve listening on %s:%d (version=%s, %d model(s): %s)",
             args.bind, args.port, VERSION_STRING, len(models),
             ", ".join(models) or "none")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutdown signal received")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
