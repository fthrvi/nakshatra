# Using Nakshatra from an agent harness

An invited node has two local interfaces. Neither one grants access to Biswa's WireGuard mesh,
Prithvi, his gateway, or any mesh MCP doorway.

- `~/.local/bin/nak-mcp` is a stdio MCP server for messages, tasks, inference credits, worker
  status, model discovery, and a simple ask operation.
- `http://127.0.0.1:11600/v1` is an OpenAI-compatible endpoint for `/v1/models` and
  `/v1/chat/completions`.

Both stay on the invited node. The gateway reaches compute and accounting only through public,
TLS-SPKI-pinned VPS fronts and signs requests with the node key made during join.

On 2026-10-23 a clean outsider joined through the public host on release `0.15.22` and completed a
raw stdio `initialize` plus `tools/list` exchange with `nak-mcp`; that release returned server version
`0.3` and its then-current 18-tool default set. Current source has 19. See
[the clean-box finding](findings/2026-10-22-clean-box-join.md).

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
present. `nak gateway on` always enables fail-closed credit enforcement: a missing node key,
refused signature, insufficient balance, ledger timeout, or ledger error denies inference. The
hub's `nakshatra-unconscious` service remains separately configured and fail-open. Worker channels
launched by this gateway require a registry-declared SPKI pin and do not fall back to plaintext.

`--models auto` is the default. It reads the signed pillar projection and advertises only models
with a complete online layer chain and a tokenizer-capable GGUF already on this node. A worker's
local slice supplies that metadata. A consumer can put a tokenizer or full GGUF at
`~/.nakshatra/gateway/models/<model-id>.gguf`, replacing characters outside `A-Za-z0-9._-` with
`_`, or pass a local `serve_models.yaml` path with `--models`.

Stop and remove the gateway opt-in with:

```bash
nak gateway off
```

## MCP clients

The installed `nak-mcp` command speaks MCP over stdin/stdout and never opens a TCP port. Replace
`YOU` with the local account name. Use an absolute path because not every harness expands `~`.

### OpenClaw

Checked against installed OpenClaw `2026.4.5` on 2026-10-22. Its current command is `mcp set`, not
the older `mcp add` form.

```bash
openclaw mcp set nakshatra '{"command":"/home/YOU/.local/bin/nak-mcp","args":[]}'
openclaw mcp show nakshatra
```

### Hermes

Hermes was not installed on this hub, so this block was not re-checked on 2026-10-22. Add this
entry to the harness's `mcp_servers` object:

```json
{
  "nakshatra": {
    "command": "/home/YOU/.local/bin/nak-mcp",
    "args": []
  }
}
```

### Claude Code

Checked against the installed Claude Code CLI on 2026-10-22. `--scope user` makes the server
available outside one project.

```bash
claude mcp add --scope user nakshatra -- /home/YOU/.local/bin/nak-mcp
claude mcp list
```

### Codex

Checked against the installed Codex CLI and the
[official Codex MCP configuration](https://developers.openai.com/codex/mcp) on 2026-10-22. Add
this table to `~/.codex/config.toml`; the CLI and IDE extension share it.

```toml
[mcp_servers.nakshatra]
command = "/home/YOU/.local/bin/nak-mcp"
args = []
```

Then run `codex mcp list` to check it.

### Any other MCP client

Configure a stdio server with no arguments. The client must send and receive JSON-RPC on the
process's standard input and output.

```json
{
  "name": "nakshatra",
  "transport": "stdio",
  "command": "/home/YOU/.local/bin/nak-mcp",
  "args": []
}
```

## Small local models

Small models sometimes claim success before checking the network state. After every task post,
claim, or submit, make the reply's exact `verify` call and do not report completion until
`nak_task_status` shows the expected authoritative state. Treat `done` as confirmation that the
local action completed, not as proof of assignment, acceptance, or payment. Before worker setup,
`nak_worker_status` with `preflight: true` (opt-in, read-only) gives the same package verdicts as
`nak worker preflight --json`; after setup, the rest of the status confirms the build and unit.
These checks confirm network state, not whether arbitrary output is correct.

## OpenAI-compatible applications

Point a Chat Completions client at the local base URL. The gateway does not require an API token,
but some clients insist on a non-empty value. This block verifies both provider routes:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:11600/v1
export OPENAI_API_KEY=nakshatra-local
curl -fsS "$OPENAI_BASE_URL/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
curl -fsS "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama32-3b","messages":[{"role":"user","content":"Say hello briefly."}],"max_tokens":32}'
```

OpenClaw, Hermes, Claude Code, and Codex can use this endpoint as a model provider independently
of their MCP connection.

## MCP tool boundary

There are 19 default tools and 3 connection tools enabled only with `--allow-connect`.

| Tool | What it does | What it cannot do |
|---|---|---|
| `nak_status` | Summarizes this messaging node and pending contacts. | No secrets, private-key bytes, or mesh state. |
| `nak_contacts` | Lists accepted contacts and online state. | Cannot create or accept a contact. |
| `nak_inbox` | Reads contact messages, framed as untrusted external content. | Cannot treat message text as trusted instructions. |
| `nak_requests` | Lists pending contact requests. | Cannot accept or decline unless the MCP server was started with `--allow-connect`. |
| `nak_send` | Sends text to an accepted contact under the node's signer and grant. | Cannot message strangers or bypass grant limits. |
| `nak_agent_send` | Sends a typed, threaded agent message to a card-capable contact. | Does not transfer files; budget fields do not move money. |
| `nak_threads` | Lists typed agent-message threads. | Does not make outside messages trusted. |
| `nak_thread` | Reads one contact-scoped agent thread. | Cannot read a thread without its ID or unique prefix. |
| `nak_card` | Shows this agent's card or a contact's advertised card. | A contact card remains an untrusted external claim. |
| `nak_tasks` | Lists offered, claimed, assigned, submitted, and judged tasks. | Cannot prove that arbitrary work is correct. |
| `nak_task_status` | Reads one task's authoritative local state and its read-only ledger hold state when escrowed. | Cannot change a task, verdict, hold, or payment. |
| `nak_task_post` | Posts a task and declared acceptance rules to contacts. | Cannot bypass task escrow or contact boundaries. |
| `nak_task_claim` | Claims an offered task. | Cannot force assignment; the poster decides the first valid claim. |
| `nak_task_submit` | Submits output for an assigned task and returns its precheck. | Cannot self-accept or alter the posted rules. |
| `nak_models` | Lists models exposed by the local gateway. | Cannot query the private mesh or advertise an incomplete chain. |
| `nak_ask` | Makes one non-streaming completion through the local gateway and reports charged credits when returned. | Cannot bypass the gateway, credit gate, or loopback boundary. |
| `nak_balance` | Reads this node operator's signed ledger balance and tier. | Cannot select or inspect another account. |
| `nak_wallet` | Reads this operator's deposit address, balance, withdrawable amount, payout state, and pending withdrawals. | Cannot bind an address or withdraw; use the human CLI. |
| `nak_worker_status` | Shows worker configuration, verified build state, and unit state with secret-shaped fields removed; `preflight: true` adds the read-only signed-package hardware check. | Cannot start, stop, build, or reconfigure the worker. |
| `nak_accept` | Accepts a pending request when `--allow-connect` was chosen by the person. | Disabled by default; cannot grant mesh or inference authority. |
| `nak_decline` | Declines a pending request when `--allow-connect` is enabled. | Disabled by default. |
| `nak_redeem` | Redeems a contact invite when `--allow-connect` is enabled. | Disabled by default; cannot redeem worker admission or join WireGuard. |

`--allow-connect` enables only the last three tools. It does not add worker controls, gateway
controls, account selection, private-mesh routes, or access to `10.42.0.x` services.
