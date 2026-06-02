#!/usr/bin/env python3
"""Nakshatra serving frontend — Ollama-compatible HTTP gateway.

Phase B of the 2026-05-30 Ollama-gateway sprint. Ships the HTTP scaffold
+ `/api/version` + `/health` (Phase A) AND the model registry + `/api/tags`
+ `/api/show` (Phase B). `/api/chat` (the actual chain walk) is Phase C+D.
The plan lives at `~/trisul/plans/2026-05-30-nakshatra-ollama-gateway-sprint.md`.

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
        # Phase C+ (/api/chat) lands here.
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


# ── Server lifecycle ────────────────────────────────────────────────


def build_server(bind: str, port: int,
                 models: Optional[dict] = None) -> ThreadingHTTPServer:
    """Construct + return a ThreadingHTTPServer bound to (bind, port).

    ``models`` is the registry from :func:`_load_models` (``{}`` if no
    ``--models`` was given); it's attached to the server so per-request
    handlers can read it via ``self.server.models``.

    Caller is responsible for ``server.serve_forever()`` /
    ``server.shutdown()`` + ``server.server_close()``. Returning the
    server object instead of starting it lets tests spin one up on
    an ephemeral port without process-level setup."""
    server = ThreadingHTTPServer((bind, port), NakshatraServeHandler)
    server.models = models or {}
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
                    "(Phase B — registry + /api/tags + /api/show).",
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

    server = build_server(args.bind, args.port, models)
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
