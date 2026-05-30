#!/usr/bin/env python3
"""Nakshatra serving frontend — Ollama-compatible HTTP gateway.

Phase A of the 2026-05-30 Ollama-gateway sprint. This phase ships the
HTTP scaffold + `/api/version` + `/health` only; model registry +
/api/tags arrives in Phase B; /api/chat in C+D. The plan lives at
`~/trisul/plans/2026-05-30-nakshatra-ollama-gateway-sprint.md`.

**Why this exists:** Prithvi's live gateway
(http://203.0.113.10:8080) speaks OpenAI to users and calls its
backend via Ollama HTTP `/api/chat`. To put the fleet behind Prithvi
without touching Prithvi's code, Nakshatra serves the same wire shape
Ollama does — Prithvi flips one env var (`OLLAMA_HOST`) and routes
its `_call_ollama` path here.

**Scope discipline:** this file owns HTTP framing + endpoint
dispatch. Model loading, tokenization, chain orchestration, chat
templating all land in later phases. Keep the scaffold dependency-
free (stdlib only) so Phase A is shippable in isolation.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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


# ── HTTP handler ────────────────────────────────────────────────────


class NakshatraServeHandler(BaseHTTPRequestHandler):
    """Endpoint dispatch + JSON response helpers.

    Subsequent phases bolt new endpoints into ``do_GET`` / ``do_POST``
    via the same if/elif cascade pattern Sthambha's PillarHandler uses
    — keeps the read-the-handler-top-to-bottom-and-understand-every-
    route property that hand-rolled stdlib HTTP makes free.
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
        # Phase B+ endpoints land here as new branches.
        self._json_error(
            HTTPStatus.NOT_FOUND,
            f"unsupported endpoint: GET {path}",
        )
        return 404

    def _route_post(self) -> int:
        path = self.path.split("?", 1)[0]
        # Phase C+ endpoints land here.
        self._json_error(
            HTTPStatus.NOT_FOUND,
            f"unsupported endpoint: POST {path}",
        )
        return 404


# ── Server lifecycle ────────────────────────────────────────────────


def build_server(bind: str, port: int) -> ThreadingHTTPServer:
    """Construct + return a ThreadingHTTPServer bound to (bind, port).

    Caller is responsible for ``server.serve_forever()`` /
    ``server.shutdown()`` + ``server.server_close()``. Returning the
    server object instead of starting it lets tests spin one up on
    an ephemeral port without process-level setup."""
    server = ThreadingHTTPServer((bind, port), NakshatraServeHandler)
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
                    "(Phase A — scaffold only).",
    )
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"port to bind (default {DEFAULT_PORT}, "
                         f"matches Ollama)")
    ap.add_argument("--bind", default="0.0.0.0",
                    help="address to bind (default 0.0.0.0 so other "
                         "tailnet hosts can reach)")
    ap.add_argument("--models", default=None,
                    help="Phase B+ — path to serve_models.yaml; "
                         "ignored in Phase A but accepted so the "
                         "argv contract doesn't change between phases")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)

    server = build_server(args.bind, args.port)
    log.info("nakshatra_serve listening on %s:%d (version=%s)",
             args.bind, args.port, VERSION_STRING)
    if args.models:
        log.info("Phase A: --models %s parsed but not yet "
                 "registered (lands in Phase B)", args.models)
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
