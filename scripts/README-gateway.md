# Nakshatra serving gateway (`nakshatra_serve.py`)

An HTTP serving frontend that puts the distributed inference **fleet** behind a
standard wire shape — **Ollama** *and* **OpenAI** — so any client (Prithvi, Open
WebUI, Cursor, LangChain, the `openai` SDK) can chat with the chain by changing
one URL. It owns HTTP framing, a model registry, prompt templating, and response
shaping; the actual generation runs through `scripts/client.py` over the cluster.

## Run it

```bash
# Real chain (workers must be serving the model's chain_yaml):
python3 scripts/nakshatra_serve.py --models scripts/serve_models.yaml

# Wire-smoke with NO cluster (canned replies) — validate a client/cutover first:
python3 scripts/nakshatra_serve.py --models scripts/serve_models.example.yaml --stub-backend
```

Default port is **11434** (Ollama's), so a consumer's only change is the host.
`--bind 0.0.0.0` (default) lets other tailnet hosts reach it.

## Model registry — `serve_models.yaml`

```yaml
models:
  - name: llama-3.3-70b                                  # the name clients request
    tokenizer_gguf: /models/llama-3.3-70b/...Q4_K_M.gguf # tokenizer the chain was split from
    chain_yaml: scripts/cluster_5worker.yaml             # OR  registry_url: http://node-pi:7777
    details: {family: llama, parameter_size: "70B", quantization_level: Q4_K_M}
```

`family` selects the chat template (`llama` / `gemma`; unknown → llama). Copy
`serve_models.example.yaml` and keep the real file local (it carries on-disk paths).

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | liveness |
| `/api/version` | GET | Ollama version hello (namespaced `nakshatra-…`) |
| `/api/tags` | GET | Ollama model list |
| `/api/show` | POST | Ollama model details, or 404 |
| `/api/chat` | POST | Ollama chat — `stream:false` (one JSON) or `stream:true` (ndjson) |
| `/v1/models` | GET | OpenAI model list |
| `/v1/chat/completions` | POST | OpenAI chat — non-stream (`chat.completion`) or `stream:true` (SSE) |

All responses carry permissive CORS (single-tenant tailnet, no auth — matches Prithvi).

## Client examples

```bash
H=localhost:11434
# Ollama, non-streaming
curl -s $H/api/chat -d '{"model":"llama-3.3-70b","stream":false,
  "messages":[{"role":"user","content":"capital of France?"}]}'
# Ollama, streaming (newline-delimited JSON)
curl -sN $H/api/chat -d '{"model":"llama-3.3-70b","stream":true,
  "messages":[{"role":"user","content":"capital of France?"}]}'
# OpenAI, streaming (SSE)
curl -sN $H/v1/chat/completions -d '{"model":"llama-3.3-70b","stream":true,
  "messages":[{"role":"user","content":"capital of France?"}]}'
```

```python
# openai SDK — point base_url at the gateway, any api_key
from openai import OpenAI
c = OpenAI(base_url="http://localhost:11434/v1", api_key="nakshatra")
print(c.chat.completions.create(model="llama-3.3-70b",
      messages=[{"role": "user", "content": "capital of France?"}]).choices[0].message.content)
```

## Cut Prithvi over to the fleet

Prithvi's gateway calls its backend via Ollama `/api/chat`. Point its
`OLLAMA_HOST` at a running `nakshatra_serve` (`<host>:11434`) — **no Prithvi code
change.** Validate the wire first with `--stub-backend`, then swap to the real
chain config.

## Smoke

```bash
# now (no cluster): proves the whole server end-to-end with a stub backend
python3 scripts/smoke_gateway.py --models scripts/serve_models.example.yaml --stub
# Phase F (cluster up): the real chain
python3 scripts/smoke_gateway.py --models scripts/serve_models.yaml --model llama-3.3-70b
```

## Status

Phases A–E shipped + the OpenAI-compat surface + CORS; 40 tests pass. The only
remaining item is **Phase F** — the live cluster smoke — which is hardware-gated
on the worker nodes being up. Bring-up: `trisul/plans/2026-06-02-cluster-bringup-runbook.md`.

**Note:** `ChainChatBackend` spawns `client.py` per request (fine at 70B
latencies; the chain step dwarfs the ~100 ms python spawn). An in-process refactor
is a later throughput optimization.
