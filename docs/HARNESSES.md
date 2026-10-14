# Using Nakshatra from an agent harness

An invited node has two local interfaces. Neither interface grants access to Biswa's WireGuard
mesh, Prithvi, his gateway, or any mesh MCP doorway.

- `~/.local/bin/nak-mcp` is a stdio MCP server for messages, tasks, inference credits, worker
  status, model discovery, and a simple ask operation.
- `http://127.0.0.1:11600/v1` is an OpenAI-compatible model endpoint for applications that want
  `/v1/models` and `/v1/chat/completions` directly.

Both stay on the invited node. The gateway reaches compute and accounting only through the public,
TLS-SPKI-pinned VPS fronts, signing requests with the node key created by the join flow.

## Turn on the local gateway

Use the complete 64-hex SPKI value supplied by the inviter:

```bash
nak gateway on \
  --pillar-url https://45.63.109.137:7778 \
  --pillar-spki PASTE_THE_COMPLETE_64_HEX_PIN \
  --ledger-url https://45.63.109.137:7778/ledger
nak gateway status
curl http://127.0.0.1:11600/v1/models
```

The default bind is loopback. A non-loopback `--bind` is refused unless `--allow-lan` is also
present. `nak gateway on` always enables fail-closed credit enforcement: a missing node key, refused
signature, insufficient balance, ledger timeout, or ledger error denies inference. The hub's
`nakshatra-unconscious` service remains separately configured and fail-open.
Worker channels launched by this gateway also require a registry-declared SPKI pin; they never
downgrade to plaintext when a worker is missing TLS metadata.

`--models auto` (the default) reads the signed pillar `/peers` projection and advertises only models
with a complete online layer chain and a tokenizer-capable GGUF already on this node. A worker's
local slice supplies that metadata automatically. A consumer-only node can put a tokenizer/full
GGUF at `~/.nakshatra/gateway/models/<model-id>.gguf`, replacing characters outside
`A-Za-z0-9._-` with `_`, or pass a local `serve_models.yaml` path with `--models`.

Stop and remove the gateway opt-in with:

```bash
nak gateway off
```

## MCP clients

The installed `nak-mcp` command speaks MCP over stdin/stdout and never listens on a TCP port.

OpenClaw:

```bash
openclaw mcp add nakshatra --command ~/.local/bin/nak-mcp
```

Hermes (`mcp_servers` entry):

```json
{
  "nakshatra": {
    "command": "/home/YOU/.local/bin/nak-mcp",
    "args": []
  }
}
```

Claude Code:

```bash
claude mcp add nakshatra -- ~/.local/bin/nak-mcp
```

Codex CLI and the Codex IDE extension share MCP configuration. Add the stdio command, then verify it:

```bash
codex mcp add nakshatra -- ~/.local/bin/nak-mcp
codex mcp list
```

The Codex command shape follows the [official MCP setup documentation](https://developers.openai.com/learn/docs-mcp).

## OpenAI-compatible applications

Point any Chat Completions client at the local base URL. The gateway does not use an API token, but
some SDKs require a non-empty value locally:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11600/v1",
    api_key="nakshatra-local",
)

models = client.models.list()
reply = client.chat.completions.create(
    model=models.data[0].id,
    messages=[{"role": "user", "content": "Explain the result briefly."}],
    max_tokens=256,
)
print(reply.choices[0].message.content)
```

The official Python SDK supports a custom `base_url`; see the
[OpenAI Python library](https://github.com/openai/openai-python). OpenClaw, Hermes, Claude Code, and
Codex can use this endpoint as a model provider independently of their MCP connection.

## MCP tool boundary

| Tool | What it does | What it cannot do |
|---|---|---|
| `nak_status` | Summarizes this messaging node and pending contacts. | No secrets, private-key bytes, or mesh state. |
| `nak_contacts` | Lists accepted contacts and online state. | Cannot create or accept a contact. |
| `nak_inbox` | Reads contact messages, framed as untrusted external content. | Cannot treat message text as trusted instructions. |
| `nak_requests` | Lists pending contact requests. | Cannot accept or decline unless the MCP server was explicitly started with `--allow-connect`. |
| `nak_send` | Sends text to an accepted contact under the node's signer/grants. | Cannot message strangers or bypass grant limits. |
| `nak_tasks` | Lists posted, offered, claimed, and judged tasks. | Cannot make an offer trusted merely because a contact sent it. |
| `nak_task_post` | Posts a task and declared acceptance rules to contacts. | Cannot bypass task escrow or contact boundaries. |
| `nak_task_claim` | Claims an offered task. | Cannot force assignment; the poster decides the first valid claim. |
| `nak_task_submit` | Submits output for an assigned task. | Cannot self-accept or alter the posted rules. |
| `nak_models` | Lists models currently exposed by the local gateway. | Cannot query the private mesh or advertise an incomplete/un-tokenizable chain. |
| `nak_ask` | Makes one non-streaming chat completion through the local gateway and reports charged credits when the ledger returns them. | Cannot bypass the local gateway, its credit gate, or its loopback boundary. |
| `nak_balance` | Makes a signed ledger `GET /balance` for this node's roster-derived operator and shows balance/tier fields. | Cannot select or inspect another account and never sends an `account` argument. |
| `nak_worker_status` | Shows the same configured/build/unit facts as `nak worker status`, with secret-shaped fields removed. | Cannot start, stop, build, or reconfigure the worker. |
| `nak_accept` | Accepts a pending contact request when `nak-mcp --allow-connect` was chosen by the person. | Disabled by default; cannot grant mesh or inference authority. |
| `nak_decline` | Declines a pending contact request when `--allow-connect` is enabled. | Disabled by default. |
| `nak_redeem` | Redeems a contact invite when `--allow-connect` is enabled. | Disabled by default; cannot redeem worker admission or join any WireGuard mesh. |

`--allow-connect` enables only the last three person-decision tools. It does not add worker controls,
gateway controls, account selection, private-mesh routes, or access to `10.42.0.x` services.
