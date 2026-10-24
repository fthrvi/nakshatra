# Nakshatra: ready for outsiders

Bar: a stranger, using **their own harness** (OpenClaw, Hermes, Claude Code, Codex, any MCP client),
can **join, sell compute, buy inference and handle credits** with no hand-holding from Biswa.
Direction: Biswa 2026-10-20 (trisul `plans/2026-10-17-houses-roads-windows.md` §9): no own app or harness.

Sizes: **S** = under a day · **M** = a few days, one review cycle · **L** = a week or more / needs a design.
**D** marks a decision that is Biswa's, not engineering.

State as of 2026-10-23 (release 0.15.22 on the public canary; clean outsider join verified end to end).

## 1. Join

| Item | Exists | Missing | Size |
|---|---|---|---|
| One-line install from an invite | 2026-10-23: `nak invite` → one line completed from a fresh outsider container on `0.15.22`; sha-pinned installer, signed invite, pinned release key, keys made locally (docs/JOIN.md) | — | — |
| Getting an invite at all | D8.2: a person can opt in with `nak address on` and post a signed `nka1.` one-line installer; a stranger can ask without an invite, and only that person can accept. A rostered known+ member may also create worker or consumer invites at `stranger` tier: 3 outstanding, 5 per rolling 30 days, and at most 0.10 USDC devnet starter credit. The owner may invite at any tier. **Address flow verified end to end from a cut-off box on 0.15.24 (2026-10-23).** | No public listing or directory (v1 is shareable-address only); where an address or invite is shared remains a human choice | — |
| Being accepted | `nak requests` shows the invite kind and tier. Address requests and every stranger-tier worker/consumer request require the inviter's interactive, person-signed `nak accept`; MCP and bare socket acceptance are refused. There is no owner pre-veto; post-hoc suspension is the owner control. | Off-box person-key custody is still needed for the signature to represent a hardware-backed human act | (see custody row) |
| Public release host | 2026-10-23: outsider routing proved (`10.42.0.1` blocked, public `latest.json` 200); the invite used `~/.nakshatra/net/release-url`; signed `0.15.22` installed from `45.63.109.137:8960`; first-install units, acceptance, messaging, MCP, grant renewal, and restart recovery all passed ([finding](findings/2026-10-22-clean-box-join.md)) | — | — |
| Platforms | Linux, Windows via WSL (systemd), any OS with Docker via `nakc` | macOS native (no systemd); a Windows note that WSL stops idle distros (bro's durability problem) | M (mac) · S (note) |
| Release signing + custody | Works, but the release key is `release-TEST.key` and person keys are plaintext TEST files (`custody test`) | A non-TEST release key held off the hub; person key off-box (vault/TPM + Face ID plan of 2026-10-20) before strangers trust our updates | M + **D** |

## 2. Harness over MCP

| Item | Exists | Missing | Size |
|---|---|---|---|
| MCP server | `nak-mcp`, 20 default tools plus 4 opt-in connection/report tools: messages, threads, agent cards, tasks (post/claim/submit/status), message delivery state, balance, wallet (read-only), models, ask, worker status, and optional accept/decline/redeem/report; remote text always framed untrusted | — | — |
| Harness docs | HARNESSES.md has one copy-paste block each for OpenClaw, Hermes, Claude Code, Codex config.toml, generic stdio, and the OpenAI endpoint; installed OpenClaw/Claude/Codex syntax was checked on 2026-10-22; a clean `0.15.22` outsider completed raw `nak-mcp` initialize + tools/list on 2026-10-23 and received that release's 18 tools; current source has 20 default tools | Hermes was not installed, so its harness-specific configuration remains unverified | S |
| Agent permission lifetime | JOIN.md states the join caps, ≤365-day charter, 7-day wire lease, and 30-day idle lapse; on 2026-10-23 a freshly joined `0.15.22` node started with `nak.msg,nak.task.claim` and the documented capped-payment renewal succeeded interactively | — | — |
| Small-model robustness | MCP accepts parameter aliases; write replies carry `done` plus an exact `verify` call; `nak_task_status` reads the local task and escrow state; descriptions and HARNESSES tell agents to verify before reporting | — | — |

## 3. Sell compute

| Item | Exists | Missing | Size |
|---|---|---|---|
| Worker admission | Owner: `nak invite --kind worker` at any tier. Known+ member: `--tier stranger`, under the §1 caps; redeem proves the invitee's node key and stranger acceptance is person-signed | — | — |
| Running a worker | `nak worker preflight` reports NVIDIA, AMD ROCm/sysfs, CPU fallback, RAM, worker-state disk, runtime readiness, and a plain verdict for each signed package before `build`; `--json` and `nak_worker_status` with `preflight=true` (opt-in; status polls never run GPU tools) expose the same shape. JOIN.md also shows build/on/off/status, owner priority, and idle unload/reload. | — | — |
| Models to serve | The signed release catalog records conservative requirements for `llama32-3b` (4 GiB VRAM, or 8 GiB available RAM for CPU fallback; 12 GiB disk). | More signed packages matching common consumer GPUs | M |
| Getting paid | Paid inference live (hold before compute, worker earns); M2 task escrow live (hold at assign, paid on accept) | Earnings cash out only at known+ after a 7-day hold, and withdrawals are not switched on (see §5) | (see §5) |

## 4. Buy inference

| Item | Exists | Missing | Size |
|---|---|---|---|
| Consumer node | `nak invite --kind consumer`, `nak gateway on` (enforce), OpenAI-compatible local endpoint, paid run settled end to end | A stranger has no credits except the overdraft (stranger tier $0.01 ≈ a few requests) | (see §5) |
| Starter credits | `--starter N` is paid once on accept from the person-signed invite, through the inviter's `nak.pay` caps; it stays spend-only and devnet-only | Production funding/policy beyond devnet | D |
| Availability | Workers: hub + bro (bro sleeps when WSL idles) | Enough always-on capacity that a stranger's first request is not "no worker" | **D** / ops |

## 5. Credits (devnet only)

| Item | Exists | Missing | Size |
|---|---|---|---|
| Ledger | Neuron ledger live, zero-sum checked, spend holds + task holds, public front on the VPS | — | — |
| Deposits (USDC → credits) | SPL vault + indexer written and reviewed (S1–S6 landed) | Indexer not installed; S7 devnet drill not run (needs Biswa present for Face ID) | M |
| Withdrawals | S4 payouts code landed | Not switched on; Face ID above 5 USDC / 50 per day | M (after S7) |
| `nak wallet` from MCP | Read-only balance | Deposit address + instructions an agent can show its human | S |
| Mainnet, fees, KYC/legal | — | All undecided | **D** |

## 6. Safety rails before strangers

| Item | Exists | Missing | Size |
|---|---|---|---|
| Revocation | Person-signed cumulative revocations are local immediately, pushed over authenticated contact sessions, and exportable/importable as `nkr1.` blobs | Older contacts ignore the new frame until they update; the ledger still trusts node keys, which this does not revoke | S (upgrade rollout) |
| Task escrow fairness | Pay-first holds; before submission a rejected/expired task refunds normally; after submission the payer can pay or dispute but cannot refund; disputes wait for an operator and have a deadline+14-day backstop; workers confirm every claimed outcome through their own scoped ledger status with bounded polling | Operator dispute handling is still a human operational duty | **D** / ops |
| Sybil / credit-line abuse | Overdraft tiers, 7-day maturity, cash-out only at known+ | Two colluding accounts can convert credit-line value into earnings (inference and tasks alike) | L + **D** |
| Spend caps in human units | Exact-decimal CLI/MCP input accepts USDC or explicit base units; grant status, balances, wallets, starters, and task rewards show human USDC beside unchanged signed/base-unit values. Bare integers remain base units with a notice. | — | — |
| Abuse/report path | Members can file node-signed reports through outbound DLP and format checks; CLI provenance and every trust/untrust/stage/install/recover/abandon/import act require a short-lived single-use person proof; operator output visibly fences untrusted text; signed operator updates are staged before HTTP and retried byte-for-byte, definitive server refusals clear the exact pending record, ambiguous pending state has a person-proved manual escape, and unknown raw node keys require an explicit force flag; bounded trusted-operator records propagate to contacts and fence suspended nodes' messages/tasks at one pre-mutation decision point; accepted inbound task steps keep that decision through their reply, while background finalization logs an omitted reply rather than rolling back final state; trusted contacts require person-proofed untrust before removal, without reopening stale replay | Coordinated server/ledger/relay/junction/pillar rollout and restart window | M (rollout) |

## Shortest path to "a stranger can do it alone"

1. **D:** admission policy for strangers + who may accept (unblocks §1, §3, §4).
2. **M:** S7 devnet drill → deposits on (with Biswa); starter credits are wired for devnet.
3. **M:** non-TEST release key + person key off-box.
4. **D / ops:** staff task-dispute review; roll the shared-revocation version out to every contact.
