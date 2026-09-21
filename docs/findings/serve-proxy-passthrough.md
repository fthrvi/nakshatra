# `/v1/chat/completions` for `engine_url` entries: from text re-render to reverse proxy

Branch `serve/proxy-passthrough` (base `main` @ `5443737`). Found 2026-09-19 in the agent-landscape bake-off
(Stage A); fixed and proven 2026-09-21.

## The problem, measured

Same model file (Qwen3-14B Q4_K_M, thinking off), same engine (llama.cpp v0.4.1 `llama-server --jinja`, one
RTX 3060), same 50-case tool-call fixture (34 single-call, 6 parallel, 6 should-not-call, 4 must-ask; stream +
non-stream; temperature 0):

| path | schema-valid | correct tool | correct args |
|---|---|---|---|
| direct to llama-server | 100% | 95% | 95% |
| through the gateway BEFORE this change | **0%** | 0% | 0% |
| through the gateway AFTER this change | **100%** | 95% | 95% |

`ProxyChatBackend` forwarded only `model/messages/max_tokens/stream/temperature/top_p` and rebuilt every reply from
the engine's text, so a tool-aware model never saw its tools (it answered "as a text-based AI I can't read files")
and clients never received `tool_calls`. The streaming path additionally hardcoded `finish_reason: "stop"` and
emitted only `delta.content`.

## The change (engine_url entries only, `/v1/chat/completions` only)

Byte-faithful reverse proxy: the request goes to the engine with ONLY `model` rewritten (`engine_model`); the answer
comes back unchanged — `tool_calls`, streamed tool-call deltas, `usage` (incl. `prompt_tokens_details.cached_tokens`),
real `finish_reason`, `reasoning_content`, SSE framing, and the engine's own non-200 status + body. The response
`model` is set back to the public entry name (as before). The client's `Authorization` header is NOT forwarded. Engine
unreachable → clean 502. Nakshatra's advertising (`/v1/models`, `/api/tags`), scale-to-zero lifecycle (`_begin_session`)
and entry-proxy routing around it are unchanged.

Per-entry controls in `serve_models.yaml`:
- `passthrough: false` — force the legacy text-only path (documented limitation: tools are dropped).
- `default_max_tokens: N` — opt-in migration lever: injected only when the request has neither `max_tokens` nor
  `max_completion_tokens`. The legacy path always forced 256; passthrough leaves the engine's default alone unless set.
- Entries with `think` (reasoning split) always keep the legacy path — the `<think>` splitting lives there.

## Proof

Fixed gateway vs direct, real engine (ijru RTX 3060), 100 responses (50 cases x stream/non-stream): **100/100 identical
tool calls (name + parsed arguments), 100/100 identical `finish_reason`, `usage.prompt_tokens` present in 100/100**
(the old gateway dropped usage on the streaming path). Growth workload through the gateway: `cached` tokens visible per
turn (1024 -> 3144 -> 4248 -> 5288), TTFT tracks only new tokens. Raw results: `experiments/bakeoff-2026-09/`.

Tests (`tests/test_nakshatra_serve.py`, +10): field forwarding fidelity (tools/tool_choice/stop/seed/response_format/
stream_options/unknown fields), no forced `max_tokens`, streamed tool-call deltas + final usage + `[DONE]`, engine error
status relayed, engine-down 502, auth header not forwarded, both opt-outs, `default_max_tokens` lever. Mutation-checked:
against the ORIGINAL code 6 of the 10 fail (the 5 that encode the bug, plus the new `default_max_tokens` option); the
other 4 are regression guards for behavior that must not change (engine-down 502, no auth leak, the two opt-outs). 155 pass across the four gateway-importing test files under `.venv`. Three unrelated gRPC/TLS tests
(`test_worker_phase_a::test_a1_grpc_message_cap_constant_is_16mib`, two in `test_client_tls`) fail identically on main.

## Deployment notes (NOT deployed)

- Live `nakshatra-unconscious` (:11599) and `nakshatra-deep` (:11600) run the old code until restarted. After a restart,
  `engine_url` entries (e.g. the roster's `qwen3:14b` / `llama3.1:8b` proxies to local Ollama) change behavior: tools and
  usage now flow, and callers that omit `max_tokens` are no longer capped at 256. Set `default_max_tokens: 256` on an
  entry to keep the old cap while callers migrate.
- `/api/chat` (Ollama-shaped) is unchanged and still text-only; only the OpenAI surface is passthrough.

## Still open (not in this change)

Chain (split) entries remain text-only — no tool calls on the split path; no client auth / rate limiting; unbounded
concurrency (KV race fix is on `worker/kv-session-ownership-gate`); Anthropic Messages and OpenAI Responses wire formats
are not served (Claude Code and Codex CLI cannot connect); static model registry.
