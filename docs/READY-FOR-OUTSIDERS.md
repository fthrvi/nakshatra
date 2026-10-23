# Nakshatra: ready for outsiders

Bar: a stranger, using **their own harness** (OpenClaw, Hermes, Claude Code, Codex, any MCP client),
can **join, sell compute, buy inference and handle credits** with no hand-holding from Biswa.
Direction: Biswa 2026-10-20 (trisul `plans/2026-10-17-houses-roads-windows.md` §9): no own app or harness.

Sizes: **S** = under a day · **M** = a few days, one review cycle · **L** = a week or more / needs a design.
**D** marks a decision that is Biswa's, not engineering.

State as of 2026-10-22 (release 0.15.20 live on hub, bro, container; first-join fix on branch only).

## 1. Join

| Item | Exists | Missing | Size |
|---|---|---|---|
| One-line install from an invite | `nak invite` → one line; sha-pinned installer, signed invite, pinned release key, keys made locally (docs/JOIN.md) | — | — |
| Getting an invite at all | Only from an existing member running `nak invite` | A way for a stranger to ask for one without knowing a member (public request page or `nak join --request`, queued for an operator) | M + **D** (who may admit strangers) |
| Being accepted | `nak requests` / `nak accept` by the inviter | Today every outsider is accepted by Biswa by hand. Needs a written admission policy per tier (stranger/known/trusted) and who besides Biswa may accept | S code, **D** policy |
| Public release host | 2026-10-22: outsider routing proved (`10.42.0.1` blocked, public `latest.json` 200); invite used `~/.nakshatra/net/release-url`; signed `0.15.20` installed from `45.63.109.137:8960`; the missing first-install units are fixed on `join/outsider-refresh` for host and `nakc` paths | Publish the next signed release containing the fix, then rerun the clean-box join end to end ([finding](findings/2026-10-22-clean-box-join.md)) | S release + rerun |
| Platforms | Linux, Windows via WSL (systemd), any OS with Docker via `nakc` | macOS native (no systemd); a Windows note that WSL stops idle distros (bro's durability problem) | M (mac) · S (note) |
| Release signing + custody | Works, but the release key is `release-TEST.key` and person keys are plaintext TEST files (`custody test`) | A non-TEST release key held off the hub; person key off-box (vault/TPM + Face ID plan of 2026-10-20) before strangers trust our updates | M + **D** |

## 2. Harness over MCP

| Item | Exists | Missing | Size |
|---|---|---|---|
| MCP server | `nak-mcp`, 22 tools: messages, threads, agent cards, tasks (post/claim/submit), balance, wallet (read-only), models, ask, worker status; remote text always framed untrusted | — | — |
| Harness docs | 2026-10-22: HARNESSES.md has one copy-paste block each for OpenClaw, Hermes, Claude Code, Codex config.toml, generic stdio, and the OpenAI endpoint; installed OpenClaw/Claude/Codex syntax checked; all 21 MCP tools listed | Hermes was not installed; clean-box MCP must be rerun after the fixed signed release is published | S release + rerun |
| Agent permission lifetime | 2026-10-22: JOIN.md states the join caps, ≤365-day charter, 7-day wire lease, 30-day idle lapse, and tested `nak grant renew --caps …` forms for posting and capped payment | Live renewal on a freshly joined node must be rerun after the fixed signed release is published | S release + rerun |
| Small-model robustness | MCP accepts parameter aliases, plain "Not done: needs X" errors | 8B agents still report false success; a `nak_task_status` / verify step the agent is told to call | S |

## 3. Sell compute

| Item | Exists | Missing | Size |
|---|---|---|---|
| Worker admission | `nak invite --kind worker --tier known` → redeem proves node key → roster row | Same "who invites a stranger" gap as §1 | (see §1) |
| Running a worker | 2026-10-22: JOIN.md now shows `nak worker build/on/off/status`, required signed-package/pillar inputs, owner lease (machine owner's use wins), and unload/reload on idle | A GPU preflight that tells the outsider "your card can serve X" before they start | M |
| Models to serve | One signed package (`llama32-3b`) | More signed packages and a published catalog matching common consumer GPUs | M |
| Getting paid | Paid inference live (hold before compute, worker earns); M2 task escrow live (hold at assign, paid on accept) | Earnings cash out only at known+ after a 7-day hold, and withdrawals are not switched on (see §5) | (see §5) |

## 4. Buy inference

| Item | Exists | Missing | Size |
|---|---|---|---|
| Consumer node | `nak invite --kind consumer`, `nak gateway on` (enforce), OpenAI-compatible local endpoint, paid run settled end to end | A stranger has no credits except the overdraft (stranger tier $0.01 ≈ a few requests) | (see §5) |
| Starter credits | `--starter N` records a pending request | The signed hub transfer that pays it is not wired | S |
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
| Revocation | Local: your signer stops signing | Contacts don't learn of revocations; the 7-day lease bounds a copied key only once every contact runs ≥0.15.20 | M |
| Task escrow fairness | Pay-first holds, one open hold per task, auto-refund at deadline+24h | A payer can refund an open hold after the worker submits; needs a rule (e.g. hold locks at submit until verdict or deadline) | M + **D** |
| Sybil / credit-line abuse | Overdraft tiers, 7-day maturity, cash-out only at known+ | Two colluding accounts can convert credit-line value into earnings (inference and tasks alike) | L + **D** |
| Spend caps in human units | Caps enforced by the signer in base units (10^8 per USDC) | CLI/MCP show and accept USDC, not base units | S |
| Abuse/report path | — | A way for a member to report and for an operator to suspend a node | M |

## Shortest path to "a stranger can do it alone"

1. **D:** admission policy for strangers + who may accept (unblocks §1, §3, §4).
2. **S:** publish the signed release containing the first-join unit fix, then rerun the clean-box join from outside the mesh through MCP and live grant renewal.
3. **M:** S7 devnet drill → deposits on (with Biswa) → wire starter credits.
4. **M:** non-TEST release key + person key off-box.
5. **M + D:** refund-after-submit rule; shared revocation.
