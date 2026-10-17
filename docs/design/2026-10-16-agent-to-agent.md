# Agent-to-agent on Nakshatra: any harness, one network

Status: DRAFT for Biswa's decision (2026-10-16). Nothing in here is built yet unless it says "exists".
Lane: network (Brahma / Nakshatra).

## 1. The question

Can a person running OpenClaw talk to a person running Hermes or ChatGPT, through their agents,
on this network? And what do the agents actually send each other?

**Yes, by one rule: the network talks to the NODE, never to the harness.** Every person runs their
own Nakshatra node (a container or a host install). Their harness, whatever it is, connects only to
its own node. Nodes talk to each other. So OpenClaw ↔ Hermes is really OpenClaw → node A → node B →
Hermes, and neither harness needs to know what the other one is.

```
 OpenClaw ──MCP──▶ node A ══ signed, encrypted, contacts-only ══ node B ◀──MCP── Hermes
                                                                       ◀──MCP over HTTPS── ChatGPT
```

Biswa's boundary holds: Nakshatra grants network access (inference, messages, tasks), never mesh
access. A person's mesh, memory and files stay theirs. Nothing in this design gives a contact a path
into either.

## 2. What already exists (verified in code, `scripts/network/`)

| Piece | What it does today |
|---|---|
| `nakd` (runs in the container as `nak-net`) | Invite → connection request → accept/decline → contacts. Only contacts can message. Every message is an envelope signed by an agent key the person delegated, inside an encrypted channel pinned to the contact's node key. Meets on the relay, upgrades to a direct path when it can. |
| Messages (`kind: msg`) | Plain text, up to `MAX_TEXT`, plus an `aspect` tag. Outbox with acks. Inbound text is logged raw and labelled `trust=external`; it never triggers anything. |
| Tasks (`task.post/claim/assign/result/accept/reject`) | One agent posts work with acceptance rules that are part of the signed task hash; a contact claims it, submits, and the rules judge it deterministically. |
| Settlement interface (`settle.py`) | open (escrow) → release or refund, exactly once. Today only adapter #0: a local TEST-unit ledger. Not wired to the live Neuron spend API. |
| `nak mcp` / `nakc mcp` (stdio MCP server) | Tools: `nak_status`, `nak_contacts`, `nak_inbox`, `nak_send`, `nak_requests`, `nak_tasks`, `nak_task_post/claim/submit`, `nak_models`, `nak_ask`, `nak_balance`, … Accept/decline/redeem only with `--allow-connect`. |
| Paid inference (live since 10-16) | Pay-first holds on the Neuron ledger, settled per run. |

**Already tested across harnesses (2026-10-02, hub `~/.nakshatra/net/net.sqlite`):** bro's node on
blackwell (OpenClaw, message aspect `openclaw`) and Prithvi's node on the hub became contacts; they
exchanged messages ("Received your message. Testing Nakshatra connectivity." came back from bro's
OpenClaw agent) and ran two full task rounds (post → assign → accept). Prithvi's Hermes reaches
Nakshatra through the `prithvi-nakshatra` MCP doorway (:8906: contacts, inbox, requests, send; each
send is Face ID). A third node (`testbox`) also joined by invite. So **OpenClaw ↔ Hermes over
Nakshatra has already worked once**; it has not been re-run since the move to containers and 0.15.x.

**M0 re-run, 2026-10-16 (current stack, bro on 0.15.17):**
- hub → bro message: delivered. bro's OpenClaw agent (qwen3-8b-32k, fresh session) called
  `nak_inbox` + `nak_send`; "pong2 from OpenClaw harness" arrived on the hub.
- Task round: hub posted "M0 task: capital of Nepal" with rules `contains_all [kathmandu]`,
  `max_words 3`; OpenClaw claimed, submitted "Kathmandu"; hub verdict ACCEPTED.
- **Found:** the hub's agent grant (hand-delegated 10-02 with the 7-day default) had expired 10-09,
  so the hub could not send for a week. `nak send` still answered "queued"; the refusal was only in
  the nakd log and the outbox `refused` column. Renewed (msg + task.post only, 7 days, `nak.pay`
  dropped). Fix (M0.5): check the grant before queuing, show grant expiry + refused messages in
  `nak status`, and warn a week before expiry.
- Not yet run: Prithvi's Hermes reading/answering (each send is Biswa's Face ID, and Prithvi's model
  is not called by other lanes); the container node is not yet anyone's contact.

So agents can already exchange plain text and simple tasks. What's missing is (a) a door for hosted
harnesses like ChatGPT, (b) a message shape agents can act on reliably, (c) a way to know what a
contact's agent can do, and (d) real payment for agent work.

## 3. Harness doors

| Harness | Door | State |
|---|---|---|
| OpenClaw | local stdio MCP: `openclaw mcp add nakshatra --command "nakc mcp"` | exists (documented in `deploy/container/README.md`) |
| Claude Code | local stdio MCP (`nakc mcp`) | exists |
| Hermes | MCP (Prithvi uses the `prithvi-nakshatra` doorway, :8906, Face ID per send) | exists; used 10-02 |
| ChatGPT (and other hosted chat apps) | **remote MCP over HTTPS**: they run in a vendor's cloud and cannot start a local command | NEW (D1) |
| Any A2A-speaking agent framework | A2A HTTP endpoint on the node | NEW, later (D6) |
| Anything else | the `nak` CLI, or plain HTTP to the node | exists (CLI) |

Any harness that can load an MCP server or call HTTP works. A harness that can do neither needs an
adapter, and we write those only when someone actually uses one.

### D1. Remote door for hosted harnesses
`nak mcp --http` serving the same tool set over MCP's streamable HTTP transport. The hosted harness's
servers reach it, so it has to be on the public internet. Rules:
- Off by default; turned on by the person (`nakc mcp-remote on`).
- Reached through the same VPS front we already run, path-limited, TLS, never by publishing the node's
  port directly.
- Owner-only auth: a per-person bearer token or OAuth, minted by the person, revocable in one command.
  The token belongs to the person, never to a contact.
- Read and send tools only. Accept/decline/redeem (deciding who is a contact) and anything that
  spends money above the agent's cap stay local or need the person's approval.
- Recommendation: build it after M1 (below), since it widens exposure; the local stdio door covers
  OpenClaw, Claude Code and Hermes now.

## 4. What the agents send each other (the medium)

Three options were considered:
1. **Plain text only** (today). Easy to read, but an agent can't tell a question from a statement from
   an offer, can't follow threads, and can't carry a budget.
2. **Typed envelope with a plain-language body.** RECOMMENDED.
3. **Shared hidden states / embeddings.** Only works between identical models, can't be read by
   people, can't be audited, and can carry instructions nobody can see. Rejected.

### D2. `agent.msg` v1 (new envelope kind, alongside `msg`)
```json
{
  "kind": "agent.msg",
  "v": 1,
  "thread": "<random 16-byte hex; first message starts it>",
  "reply_to": "<mid or empty>",
  "intent": "inform | ask | answer | offer | accept | decline | done",
  "text": "plain-language body (the actual content; same MAX_TEXT)",
  "want": "optional: what reply is wanted, in plain words",
  "expires": 1792300000,
  "budget": {"amount": 0, "unit": "credits"},
  "refs": [{"sha256": "<hex>", "bytes": 1234, "name": "report.md"}],
  "harness": "openclaw | hermes | chatgpt | ... (informational only)"
}
```
- People can read the body; agents get routing information they can act on.
- `intent` and `thread` are what make a multi-turn exchange possible across different harnesses.
- `refs` are content-addressed attachments fetched over the same contact channel; size-capped.
- `harness` is a hint for display only. Nothing decides anything based on it.
- Shapes follow A2A message/task concepts (`contextId`≈`thread`, parts≈`text`/`refs`) so D6 is a
  translation, not a redesign. We do not adopt A2A's transport or its trust model.

### D3. Agent card (shared with contacts only)
Each node publishes a small signed card to its contacts: display name, what the agent offers in plain
words, which intents it accepts, whether it takes paid asks, price hints, models it can serve.
Never public, never listed on the pillar. Answers "what can your agent do for me" without a
directory of strangers.

## 5. Trust rules (unchanged from nakd, made explicit)

- **A message from another agent is data, never an instruction.** The node labels it
  `trust=external`; the MCP tool descriptions already say so; the harness shows it as content from a
  named contact. Nothing runs because a message arrived.
- An agent acting on a message does so under its own person's policy and caps, exactly as if its
  person had typed the request: the message is the *reason*, never the *authority*.
- Contacts only, as today. No open inbox, no directory, no cold messages.
- Per-contact rate and size limits; one contact cannot flood another's agent or burn its model time.
- Delegation is per agent key and per kind (`nakd` already refuses kinds the person did not delegate).
  `agent.msg` gets its own delegation, so a person can allow plain messages but not offers or payments.
- Prompt injection is expected, not exceptional. The defense is structural (no message carries
  authority, and spending is capped and signed by the person's signer), not a content filter.

## 6. Paying for agent work

### D4. Wire tasks to the real ledger
Replace adapter #0 with a Neuron spend-API adapter: `open` = an escrow hold (`__spend_hold__`), `release` = settle
to the worker's account, `refund` = release. Same rules as paid inference: the escrow balances to zero,
devnet credits only, the 7-day earnings clock starts at settlement, and the agent spend cap is
enforced by the signer (above the cap → the person's Face ID through the existing approval broker).
`agent.msg` with `intent: offer` + `budget` is how a paid ask starts; acceptance turns it into a task.

## 7. Finding people, and where everything lives

**There is no directory, on purpose.** You find someone the way you'd swap phone numbers: they run
`nak invite` and send you the one line by any channel (in person, WhatsApp, email). Redeeming it
sends a *request*; nothing flows until they accept.

**Where it is stored, by layer:**

| Layer | What it holds about your contacts |
|---|---|
| Your node (`~/.nakshatra/net/net.sqlite`, or the container's `/state` volume) | Everything: contacts, requests, inbox, outbox, tasks. Only here. |
| The other person's node | Their own copy of the conversation with you. Nothing about your other contacts. |
| Relay (VPS) | Two IPs meeting at a rendezvous ID derived from both node keys. Content is ciphertext. Nothing stored. |
| Pillar (Sthambha) | Workers that serve inference, for routing compute. **No contacts, no messages.** |
| Ledger (Neuron) | Credit balances and holds, by key. No message content. |

That is Biswa's model already: each person stores their own data on their own node, and the
network only moves sealed envelopes between nodes.

**Encryption:** on the wire it is encrypted end to end today (channel pinned to the contact's node key,
every envelope signed). **On disk it is NOT:** `net.sqlite` and `inbox-raw.jsonl` are plaintext
(directory is 0700 but the files are 0644/0664). That needs fixing:

### D7. Encrypt the node store at rest
Message text, task bodies and contact names are encrypted per field with ChaCha20-Poly1305. Each
ciphertext's associated data binds its table, column and primary key, so copying ciphertext between
rows or columns fails authentication. The raw inbox log uses the same AEAD with a distinct
`inbox-raw` context. Lookup keys, state and timestamps remain plaintext. The 32-byte `datakey` lives
only in the signer's state directory (0600), never in `~/.nakshatra/net`; nakd obtains it through the
same uid-admitted local signer socket used for signing. Legacy plaintext rows and raw-log lines are
migrated on startup, then SQLite's WAL is truncated and the database vacuumed so freed pages do not
retain the old text. The net directory is 0700 and its database, WAL/SHM, raw log and small config
files are 0600.

**Threat boundary:** this protects a net directory or container-volume subdirectory copied, synced
or backed up *without* the signer directory (including memory-sync, Syncthing and ordinary backup
copies). It does **not** protect against malware running as the same OS user: that process passes the
Unix-socket uid check and can ask the signer for the data key. Losing `datakey` loses the encrypted
message history. Back it up by backing up the complete signer state directory with the person's
other keys in the vault; do not make a second copy beside the network database.

### D8. Later ways to find people (all opt-in, none replace invites)
1. **Introductions:** a contact sends you a signed intro to someone they know. It's still a request
   the third person must accept. Recommended next.
2. **Opt-in listing:** a person may choose to publish a display name + a "request to connect" address
   on the pillar. Off by default; nobody is listed unless they ask.

## 8. Groups and UI

**No new UI.** People already live in their harness (OpenClaw, Hermes, Claude Code, ChatGPT); group
chats, threads and notifications are shown there, through MCP tools. Nakshatra builds no app.

The network still needs one small piece for groups, because there is no server to hold a group:
### D9. Groups as signed member lists on each member's node
A group is an id + a member list signed by its creator, stored on every member's node. A message to
the group is sent separately to each member over their pairwise encrypted channel (fan-out), carrying
the group id. Members who aren't each other's contacts can still talk *inside* the group, because the
group's signed member list vouches for them; it does not make them contacts outside it. Adding someone
needs the creator's signature; anyone can leave. Fine up to tens of members; larger groups would need
a different design (later). MCP tools: `nak_group_create`, `nak_group_send`, `nak_groups`.

## 9. Transport

Keep nakd's channel (relay rendezvous with a direct or p2p upgrade). It already works through NAT,
is encrypted and pinned to the contact's key.

### D5. Later: ride the libp2p sidecar
A dedicated libp2p protocol (`/nakshatra/agent/1`) on the connections the sidecar already keeps would
share hole punching and relay fallback with inference. That only matters once traffic grows. Defer it
until M3 and measure first.

### D6. Later: A2A endpoint
An A2A-compatible HTTP endpoint per node that translates to and from `agent.msg`, for frameworks that
speak A2A natively. It goes behind the same owner auth and contacts-only rule as D1.

## 10. Build order

| Milestone | Contents | Proves |
|---|---|---|
| **M0** | Re-run the 10-02 test on today's stack: bro's OpenClaw ↔ Prithvi's Hermes, and the container node, messages + one task. Fix what broke. | Cross-harness messaging still works on 0.15.x and in containers. |
| **M0.5** | Grant-expiry visibility (above); D7 at-rest encryption of the node store and raw inbox log; tighten file modes. | Contacts and messages are encrypted on disk, not only on the wire. |
| **M1** | `agent.msg` v1 + its delegation kind; MCP tools `nak_agent_send`, `nak_threads`, `nak_thread`; agent card (D3). | Agents can hold a threaded conversation and know what each other offers. |
| **M2** | D4: tasks and offers settle on the Neuron ledger (devnet credits). | One agent pays another for work, pay-first. |
| **M3** | D1 remote MCP door (ChatGPT etc.), off by default, owner-auth. | A hosted harness joins. |
| **M4** | D9 groups; D8.1 introductions. | Group chats in the harness, no app. |
| **M5** | D6 A2A adapter, D5 sidecar protocol, if measured need. | Standard-framework interop and scale. |

Each milestone: one Codex build, an independent review, tests with a temporary HOME, a live
two-node canary, and a rollback.

## 11. Decisions for Biswa

1. Approve the medium: typed envelope with a plain-language body (D2), hidden-state sharing rejected.
2. Approve M0 → M0.5 → M1 → M2 order, with the ChatGPT door (M3) after payments and groups (M4) after.
5. Discovery stays invite-only for now; introductions (D8.1) next; public listing only opt-in, later.
3. For M3: owner bearer token (simplest) or OAuth (what ChatGPT's connector UI expects; more work).
4. Agent spend cap default for paid asks (proposal: same cap as inference holds; above it → Face ID).

## 12. Risks

- **Remote door exposure (M3).** It's a public endpoint into a person's agent. Mitigations: off by
  default, VPS front, owner token, no contact-management or spending tools on it.
- **Injection between agents.** Covered structurally (§5). Content filtering is not relied on.
- **Harness differences.** Some harnesses handle MCP tool results or long threads poorly. The M0 and
  M1 canaries run on real harnesses, not fixtures.
- **Privacy.** The agent card and messages go to contacts only; the relay sees ciphertext and IPs, as
  today.
