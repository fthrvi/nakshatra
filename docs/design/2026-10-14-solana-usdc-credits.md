# Solana USDC deposits and payouts for Nakshatra credits — design (DEVNET ONLY)

**Status:** design, not built. 2026-10-14. Architect: Fable (network lane).
**Scope rule from Biswa:** no own chain; Solana settlement; real money stays DEVNET until he
separately decides mainnet (legal weight: money transmission, taxes, KYC — see §9).

## 0. The shape in one paragraph

The Neuron ledger v2 stays THE meter. USDC touches the system only at two edges: a
**deposit** (USDC arrives on-chain in a per-operator deposit account the hub treasury owns;
an indexer credits the operator) and a **withdrawal** (the ledger holds the amount, a
separate keyholder process pays USDC from the treasury vault, the ledger finalises).
No custom program. The treasury is modelled as one more ledger account, so the 09-04
mutual-credit invariant `sum(all balances) == 0` keeps holding with money in the system,
and the daily proof is one comparison: `-balance(treasury) == on-chain vault USDC × rate`.

## 1. What exists and what is reused (read 2026-10-13)

| Reused as-is | Where | Note |
|---|---|---|
| Ledger v2 store: accounts/receipts/journal(append-only)/nonces, `settle`, `transfer`, `dispute`, `_tx` BEGIN IMMEDIATE | `~/neuron/python/core/ledger_store.py` (journal `kind` is free TEXT, no CHECK → new kinds need no migration; `kind` at :163-175) | live at 10.42.0.1:8097, `~/.neuron/credits/ledger.sqlite` |
| Signed-envelope auth, roster → operator/tier, `verify_receipt` (receipt_id = content+run_id) | `~/neuron/python/ledger_auth.py`, `ledger_server.py:247-332` (`V2LedgerService`) | every new route is one more branch in `do_GET`/`do_POST` |
| Roster TSV `pubkey name operator tier tenant [coord]`; extra columns ignored (`load_roster` reads `fields[:5]`, coord = `fields[5]`) | `ledger_auth.py:161-185`, `~/.nakshatra/admission-peers.tsv` | a 7th column is append-safe |
| Client signer, enforce mode, `balance()` | `~/nakshatra/scripts/ledger_client.py` (+ `~/wt/nak-consumer` branch `network/outsider-consumer`: `NAKSHATRA_LEDGER_MODE=enforce`, `nak gateway on --ledger-url`, MCP `nak_balance`) | add `wallet()`/`withdraw()` beside `balance()` |
| Accounting front door + env-selected backends | `~/nakshatra/scripts/accounting.py` (`ESCROW_BACKENDS`, `NAKSHATRA_ESCROW`) | money edges get named here, nowhere else |
| Task escrow adapter contract open/release/refund/status | `~/nakshatra/scripts/network/settle.py:32-47`, used by `nakd.py:746-755` via signer `approve_spend` | STAYS in-ledger (units are now USDC-backed, so in-ledger escrow IS money escrow) |
| Person-key signer with caps, `approve_spend` → `needs_human`, `NAK_CUSTODY` | `~/sthambha/sthambha/signer.py:297-331` | an outsider's `nak withdraw` goes through THEIR signer caps |
| Human approval path (Touch ID / Face ID via broker :8902) | `~/trisul/infra/pillar-agent/write_gate.py:603` `_fingerprint_presence` | the ONLY approval door; tests inject fake presence, never hit :8902 |
| Vault-held Solana key + solders signing + stdlib JSON-RPC + DRY-RUN/kill-file/audit pattern | `~/trisul/infra/sol-desk/executor.py` (`vault_get` :97, `rpc` :231, `Signer` :279-297, `KILL_FILE` :44, `armed` :115) | copy the pattern into Neuron; the sol-desk service itself is disabled, venv empty |
| Vault | `~/trisul/infra/vault/vault.py` (`vault get/put/list`, audited) | treasury secret lives here, nowhere else |
| Tooling on hub | solana-cli 4.1.2 (config = devnet, `~/.config/solana/id.json` does NOT exist), spl-token, solana-test-validator, anchor 1.2.0 | no `solders`/`base58` in system python |

**Parked, not reused:** `~/neuron-escrow` (worktree on `escrow/devnet-v0`, untracked Anchor
counter template — nothing escrow-specific written); `neuron/python/core/wallet.py` and
`chain_client.py` (Substrate/ss58, PAUSED). Branch `forgejo/incentive/v1-credit-ledger` is the
v1 JSON ledger already migrated; nothing on it touches Solana.

## 2. Architecture decision: plain SPL vault + indexer, no Anchor program

**Decision:** USDC moves by ordinary SPL `TransferChecked` into token accounts owned by one hub
treasury key; a Python indexer reads finalized transactions and writes ledger rows. The Anchor
escrow (hackathon plan days 8-9; council "devnet ESCROW = money truth") is superseded for v1.

Why, against a custom program:
- **Audit surface.** Zero custom on-chain code; the only program involved is SPL Token. No
  upgrade authority to protect, no IDL, no PDA/rent logic, no Rust toolchain in the test path.
- **Custody is identical either way.** A program with a treasury authority is still a hot key
  that can drain it; the program adds nothing until there is multi-party control (Squads
  multisig does that without our code — mainnet DECISION).
- **Cost.** No deploy (~2-3 SOL), no program upkeep; per-operator deposit account rent is
  ~0.002 SOL each and reclaimable.
- **Binding without a memo.** A per-operator token account owned by the treasury binds a
  deposit by *destination*, not by user-written memo text (exchanges and most wallets cannot
  attach memos). That removes the "forged deposit claim" class entirely: nobody claims a
  deposit; the hub discovers it.
- What we give up: on-chain escrow semantics visible to third parties. Not needed while the
  ledger is the meter; revisit only if two hubs must settle with each other.

This is the ONE decision that needs Biswa (it reverses a prior decided item) — see §10.

## 3. Money model inside the ledger

- Reserved account `treasury` (tier `self`). **Deposit** = `transfer(treasury → operator)`;
  treasury goes negative = "the network holds USDC it owes". **Withdrawal** =
  `withdraw_hold(operator)` then `withdraw_finalize` (operator → treasury) or
  `withdraw_release` (hold undone). `sum(balances) == 0` is preserved.
- **Unit and rate (fixed):** keep the ledger unit = 1 token-layer, and fix
  `RATE = 100 credits per µUSDC` (1 USDC = 10^8 credits). No re-denomination of existing
  rows; bro's +280 stays +280 (= $0.0000028). Rate is a constant in Neuron, changed only by a
  git change + restart; every deposit/withdraw journal row records the rate used (`detail.rate`).
  Changing it re-prices every balance, so treat it as permanent; price adjustment happens via
  §4's multiplier, not via RATE. `MAX_UNITS = 10^12` = $10,000 per op/balance — fine on devnet;
  mainnet must raise it (DECISION later, with the int64 bound in `ledger_store.py:22-26`).
- New journal kinds: `deposit`, `withdraw_hold`, `withdraw`, `withdraw_release`, `refund`,
  `starter` (§5). Existing kinds untouched.
- New tables (all `CREATE TABLE IF NOT EXISTS`, idempotent, no row migration):
  - `chain_refs(ref PRIMARY KEY, kind, account, amount, slot, ts)` — `ref = <tx_sig>:<ix_index>`;
    the deposit idempotency key.
  - `deposit_accounts(operator PRIMARY KEY, token_account, created_ts)`.
  - `payout_addresses(operator PRIMARY KEY, address, bound_ts, previous)` — 48 h cooling
    after a change before withdrawals to the new address.
  - `withdrawals(id PRIMARY KEY, operator, amount, address, state CHECK IN
    ('pending','approved','sent','finalized','released','expired'), tx_sig, approval_ref,
    created_ts, updated_ts)` — `tx_sig` is UNIQUE and recorded BEFORE broadcast.
- `withdrawable(operator)` = `min(balance, net_deposits + funded_matured_earnings)` where
  `net_deposits = deposits − withdrawals − holds` and an `earn` row is **funded** iff the
  requester's balance before the charge was ≥ `charged` (recorded in `detail.funded` at settle
  time — one extra JSON field in `ledger_store.settle` :325-330), **matured** iff older than
  `HOLD_S` (7 days, = dispute window) and the receipt is still `settled`. Earnings paid from
  someone's overdraft are never cashable until that someone deposits.

## 4. Pricing (Biswa 2026-10-13: consumers get ~18 tokens today)

Problem: units = tokens × layers (10 tokens on llama32-3b = 280 units) while TIER_LIMITS are
stranger 100 / known 500 / trusted 5000 (`ledger_store.py:30-35`, mirrored
`nakshatra/scripts/creditlimit.py`). Those limits were sized as "harmless overdraft" before any
price existed.

**Decision — price per generated token, weighted by served layers, one parameter:**
`charge = n_generated × total_layers × multiplier(model_class)`, i.e. the existing
token-layer unit with `multiplier = 1` for every class in v1. At RATE 100/µUSDC this prices
llama32-3b (28 layers) at $0.28 per 1M generated tokens and a 70B (80 layers) at $0.80/M —
market-shaped, derived only from facts the workers sign (`chain` layer map, `n_generated`),
so no trust in an asserted `model_id` (receipt.py:52 says weights are NOT verified).
- Each worker is credited its own stage's token-layers (unchanged, `ledger_auth.py:394`).
- **Network fee: 0 in v1** (mutual credit; the hub earns as a worker, not as a toll). A
  withdrawal fee covering SOL tx cost comes with mainnet (DECISION).
- **Who sets prices:** the hub operator, in a signed `prices.tsv` (model_class → multiplier,
  integer ≥ 1) next to the roster; the ledger records the multiplier in `receipts`/journal
  detail. v1 ships the table with `default 1`; the table is the hook, not the work.
- Prompt tokens are unpriced (receipts carry `n_generated` only) — accepted for v1; flagged.
- **Credit limits become a small overdraft only:** stranger 1,000,000 ($0.01), known
  25,000,000 ($0.25), trusted 250,000,000 ($2.50), self unlimited. Update BOTH
  `ledger_store.TIER_LIMITS` and `creditlimit.py` in the same slice (they are mirrored by hand —
  the comment at `creditlimit.py:24-31` already warns about drift). Deposits are THE consumer
  path; the overdraft exists so a first run works before the first deposit indexes.

## 5. Consumer-only nodes (Biswa 2026-10-13)

A slow laptop running only a harness must use the network without serving and without a
worker invite.
- **Invite kind `consumer`:** `nak invite --kind consumer --tier known [--starter N]`.
  `joincode._validate_invite_kind` (`scripts/joincode.py:134-152`) gains `consumer` with a
  `{redeem, code, tier, starter}` block (same shape as `worker`, no port/coord). Redeem reuses
  the node-key-signed Redeem v2 path (`scripts/network/join_setup.py:99` →
  `trisul/infra/onboarding/onboard_server.py`), which appends the roster row
  `pubkey name operator tier tenant - consumer` (7th column `role`; `load_roster` ignores it,
  the planner/registrar must skip `role=consumer` rows and never summon them).
- **Tier mapping:** the invite's tier sets only the overdraft (§4); default `known` because
  the inviter vouches. `self` is never granted by invite (existing rule).
- **Starter allowance:** the inviter PAYS it — on redeem the hub records
  `transfer(inviter_operator → new_operator, starter)` with journal detail
  `{"starter": invite_id}` (kind `starter`), bounded by the inviter's own balance+overdraft.
  Nobody mints. Default 0.
- **How they use it:** `nak gateway on --ledger-url https://45.63.109.137:7778/ledger`
  (nak-consumer branch) — loopback OpenAI gateway in enforce mode; harnesses use `nak-mcp`
  (`nak_ask`, `nak_balance`, new `nak_wallet`). They need no mesh, no WireGuard, no worker
  port.
- **Threat model line (goes into `docs/THREAT_MODEL.md`):** a consumer node opens NO inbound
  port (no gRPC `Info/Forward/Inference`, no file server), never holds a layer, and therefore
  sees no one else's prompts or activations. Its own prompts are visible to the workers that
  serve it, exactly as today for any requester. Its node key can only spend its own balance
  (bounded by deposits + overdraft) — compromise cost = that balance, never USDC (§6).

## 6. Wallets and keys

**Decision: a separate Solana keypair; never derive from the node key.** Same curve, wrong key.
- Blast radius: the node key sits on every box, is loaded by long-running services (worker,
  gateway, ledger client) and signs envelopes constantly; its compromise today costs at most
  one overdraft. Reusing it as the wallet turns that into "steal the USDC". The person key
  (`~/.sthambha/person-TEST.key`) is the correct money ancestor; nodes are delegates.
- Rotation differs: node key = roster edit; wallet = move funds.
- Privacy: a wallet derived from the node key links a public node identity to its whole
  on-chain history for every stranger that sees the roster.
- **Nodes do not need a local Solana key at all** to deposit (send from any wallet to their
  deposit account) or to withdraw (any bound payout address). `nak wallet init` creates one
  only for convenience/devnet faucet; it is stored by the sthambha signer like agent keys
  (`signer.py` keeps keys in its own dir, uid-checked socket), never under `~/.nakshatra/keys`.
- **Hub treasury key:** vault entries `nak-treasury-secret` / `nak-treasury-pubkey`
  (sol-desk pattern). Loaded by ONE process (`nak-treasury`, §7), never by the ledger server.
- **`nak wallet` UX:** `address` (deposit account; created on first call), `bind <addr>`
  (payout address; 48 h cooling on change), `status` (balance, withdrawable, USDC equivalent,
  pending withdrawals), `init` (optional local key), `faucet` (devnet only).
  **MCP `nak_wallet`:** read-only, returns exactly `status` + deposit address; no key, no
  path, no withdraw tool in MCP v1 (money leaving = CLI + the operator's signer caps).

## 7. Deposit and withdrawal flows

**Deposit.** Operator signs `GET /wallet` → ledger returns its deposit token account (queued
for creation by `nak-treasury` if absent; owner = treasury, mint = `NAK_USDC_MINT`). Operator
sends USDC there from anywhere. **Indexer** (`nak-indexer`, read-only RPC, no key): every 15 s
`getSignaturesForAddress` per deposit account at commitment `finalized` (~13 s on devnet; no
reorg handling needed beyond that), `getTransaction(jsonParsed)`, accept only
`transfer`/`transferChecked` of the configured mint INTO a known deposit account, write
`store.deposit(ref, operator, amount_µ × RATE, slot)` — `ref` UNIQUE makes replays and restarts
idempotent. Cross-check: per deposit-account on-chain balance must equal the sum of its indexed
deposits (a mismatch pauses that account and alerts). Unbound deposits can only land on the
main vault ATA (never advertised); they get a `refund` only by a human-approved payout.

**Withdrawal.** Operator signs `POST /withdraw {amount, address}` (address must be the bound
one, past cooling). Ledger: tier ≥ `known` for earnings (any tier for own net deposits),
`amount ≥ MIN_WITHDRAW` (1 USDC), `amount ≤ withdrawable`, per-operator 1/day, then
`withdraw_hold` (journal −amount, row `pending`) — **debit before send**. `nak-treasury` loop:
- `pending` → if `amount ≤ NAK_PAYOUT_AUTO_CAP` and day total ≤ `NAK_PAYOUT_DAY_CAP`: `approved`;
  else `_fingerprint_presence(keyid, "nak.payout", ..., description rendered from parsed
  fields)` → Face ID/Touch ID via :8902; unanswered = stays pending, re-asked once, then
  `released`.
- `approved` → build `TransferChecked` vault ATA → recipient ATA (create recipient ATA
  idempotently; treasury pays rent on devnet — mainnet deducts it), sign, **store `tx_sig` +
  state `sent` BEFORE broadcast**, then `sendTransaction`.
- `sent` → poll to `finalized` → `withdraw_finalize` (treasury += amount). If the blockhash
  expires without landing → `expired` → rebuild under a new blockhash (the old signature can
  never land, so no double pay); after 3 expiries → `released`.
- Batching: one tx per withdrawal in v1 (simplest reconcile); multi-transfer batching later.
- Fees: SOL for tx fees from the treasury; alert when treasury SOL < 0.05.
- Fail-closed defaults: `NAK_FUNDS=off` ⇒ DRY-RUN (everything logged, nothing broadcast);
  kill file `~/.neuron/credits/payout.off` stops payouts immediately; indexer keeps running.

**Hub ledger down:** deposits are on-chain and get indexed on restart (idempotent). No
withdrawal can be requested; `sent` ones still finalize. Consumer gateways in enforce mode
fail closed (already designed); Prithvi's own gateway stays fail-open (unchanged).

## 8. Reconciliation, safety, limits

- **Daily proof (`nak-reconcile`, timer):** `−balance(treasury) == RATE × (Σ on-chain USDC in
  all treasury-owned token accounts) − Σ(sent-not-finalized holds)`; plus `sum(all balances)==0`;
  plus Σ `chain_refs` deposits == Σ journal `deposit`. Writes a signed report to
  `~/.neuron/credits/reconcile/<date>.json`; any mismatch drops the kill file and pings Biswa
  (Buzz, via the existing broker notify path). Ledger already in `backup-sources.tsv`.
- **Rate limits:** per-operator 1 withdrawal/day, `NAK_PAYOUT_DAY_CAP` global
  (devnet default 50 USDC), `NAK_PAYOUT_AUTO_CAP` per tx (devnet default 5 USDC), hub RPC
  budget guard on the indexer.
- **Tiers vs deposits:** `can_consume` stays `balance − cost ≥ −limit`; a deposited account
  simply has a positive balance and the overdraft is noise. Earning is open to all tiers;
  cashing out is gated (§3 funded/matured, §7 known+).
- **Devnet faucet:** own SPL test mint (`NAK_USDC_MINT`, mint authority in vault) so tests
  never depend on Circle's devnet faucet; `nak wallet faucet` mints 10 test-USDC to the
  caller's deposit account on devnet only (refuses on any other cluster). Circle's devnet USDC
  mint is a config value, not a code change.

## 9. Threat model (additions to `docs/THREAT_MODEL.md`)

| Threat | Defence |
|---|---|
| Forged deposit claim | No claim path exists; the indexer reads finalized chain state into a destination-bound account. |
| Deposit replay / double index | `chain_refs.ref = sig:ix` UNIQUE; per-account on-chain balance cross-check. |
| Front-running | Nothing to front-run: deposits are plain transfers to a private-by-destination account; withdrawals are hub-signed. |
| Treasury key theft | Vault-only secret; single keyholder process; auto-cap + day-cap + Face ID above cap; kill file; reconcile alarms; mainnet: multisig + hot/cold split (DECISION). |
| Malicious worker inflating receipts | Requester signs the envelope carrying `n_generated`/`chain`; worker signs only its stage over `output_sha256` (`ledger_auth.py:270-333`). Garbage output → dispute within `HOLD_S` claws back before it is cashable. |
| Sybil workers farming credits for cash | Earnings from overdraft are **unfunded** (never withdrawable); withdrawals of earnings need `known+`; 7-day hold = dispute window; per-operator/day and global caps; starter allowances are paid by the inviter. Self-dealing (deposit, pay own worker, withdraw) nets to zero minus fees. |
| Consumer node compromise | No inbound port, no layers, no prompts of others; node key spends only its balance; USDC key absent or caps-bound in its own signer. |
| Hub operator (us) | Journal append-only triggers; signed daily proof; roster + prices in git. The hub CAN still mint by editing SQLite — only honesty and the published proof bound that; say so in the doc. |
| Payout to wrong address | Bound address only, 48 h cooling on change, description shown to the human is rendered from parsed fields (signer.py finding 3 pattern). |

**Legal weight (mainnet only, flagged now):** holding and paying out customers' USDC is money
transmission (FinCEN MSB + state licences), payouts create tax reporting, KYC thresholds apply,
and memory records that real-money payouts wait for a legal/visa check. Devnet carries none of
this; the switches (`NAK_FUNDS`, `NAK_SOLANA_CLUSTER`) refuse mainnet unless every other switch
is `enforced`.

## 10. Ordered build plan — each slice = one Codex task with tests, devnet only

Impact÷effort ranking is the order. No slice touches `~/.hermes-*`, the live gateway, the
outreach loop, or the hub GPU. Everything is additive behind defaults that do nothing.

| # | Slice (repo, files) | Verify | Rollback |
|---|---|---|---|
| S1 | **Ledger money model** (Neuron): `core/ledger_store.py` +`treasury`, tables `chain_refs`/`deposit_accounts`/`payout_addresses`/`withdrawals`, methods `deposit`, `withdraw_hold/finalize/release`, `withdrawable`, `detail.funded` on earn rows; new `TIER_LIMITS` (§4) + `creditlimit.py` mirror; `RATE` constant. Pure SQLite, no network. | `python -m pytest neuron/python/tests` incl. new: sum==0 with deposits, idempotent `deposit(ref)`, unfunded earnings not withdrawable, hold→release restores, int64 bounds. Existing suite green. | Tables are additive; `git revert` + restart; no data migration to undo. |
| S2 | **Consumer admission** (Nakshatra + trisul onboarding): `joincode.py` kind `consumer` (+`starter`), `nak invite --kind consumer`, Redeem v2 appends `role=consumer` row, `starter` transfer at redeem; planner/registrar skip `role=consumer`. | `tests/test_joincode*.py`, `tests/test_nak_*.py`; a fake onboard server; roster fixture with a consumer row proves `load_roster` and the planner ignore it. | Rows removable; invite kind unknown to old code is rejected (existing validation). |
| S3 | **Chain read side** (Neuron): `python/chain/solana_rpc.py` (stdlib JSON-RPC, from sol-desk `rpc`), `python/chain/indexer.py`, user unit `nak-indexer.service` (default disabled). `solders`+`base58` pinned in `python/requirements-chain.txt`, venv `~/.neuron/venv-chain` (not git). | `solana-test-validator` fixture: create mint, treasury, deposit account, transfer, assert one `deposit` row; replay the same tx → no second row; restart → no change. | Disable unit; rows stay (append-only, correct). |
| S4 | **Keyholder/payout** (Neuron): `python/chain/payout.py` = `nak-treasury` (vault key, deposit-account creation, approve→sent-before-broadcast→finalize, expiry rebuild, caps, kill file, DRY-RUN unless `NAK_FUNDS=on`), approval via `write_gate._fingerprint_presence` with injected presence in tests. | Validator tests: cap → auto; over cap → fake presence; blockhash expiry → new sig, exactly one landed transfer; kill file halts. **Never calls :8902 for real.** | Kill file; disable unit; holds `released` by a one-shot CLI. |
| S5 | **Routes + client + UX**: Neuron `ledger_server.py` `GET /wallet`, `POST /wallet/bind`, `POST /withdraw` (signed envelopes, same `_row_identity`); Nakshatra `ledger_client.wallet()/withdraw()`, `nak wallet …`, MCP `nak_wallet` (read-only), `accounting.py` names the backend; VPS front path-allow `/ledger/wallet*`,`/ledger/withdraw` (nginx site, backup first). | `test_ledger_server.py` route tests (401 unsigned, 403 unbound address, 422 under min); `tests/test_nak_mcp_consumer.py` shows no key material in `nak_wallet`; `nak gateway status` unchanged. | Remove nginx locations from the site file backup; routes 404 when the feature env is unset. |
| S6 | **Reconcile + runbook**: `python/chain/reconcile.py` + timer, Buzz alert, kill on mismatch; `docs/COMPONENTS.md` rows; THREAT_MODEL §5/§9 text; `backup-sources.tsv` already covers the DB. | Validator test: inject a bogus journal row in a temp copy → mismatch → kill file written. | Disable timer. |
| S7 | **Devnet drill** (manual, logged): own test mint; bro's node (`known`) deposits 10 test-USDC, runs inference through the front, hub worker earns, bro withdraws own net deposit; a hub-side withdrawal over `AUTO_CAP` exercises the real Face ID once, with Biswa present. | Reconcile report equals chain; journal shows deposit → spend → earn → withdraw. | `NAK_FUNDS=off`. |

**Test strategy:** unit = SQLite only (S1/S2/S5); integration = `solana-test-validator`
(S3/S4/S6, already installed, no network); devnet = S7 only, by hand, never in CI.
All tests use `HOME=$(mktemp -d)`, fake presence, and a throwaway roster.

**Out of scope (v1):** mainnet; Anchor/on-chain escrow; payment channels; prompt-token
pricing and per-model multipliers ≠ 1; KYC; multisig; batching multiple payouts per tx;
cross-hub settlement; any change to Prithvi's gateway env or the outreach loop.

## 11. DECISIONS for Biswa

**DECIDED 2026-10-13 (Biswa): 1 = plain SPL vault + indexer (Anchor escrow parked); 2 = RATE 100 credits/µUSDC and overdrafts $0.01/$0.25/$2.50 accepted; 3 = devnet payout defaults accepted (AUTO_CAP 5, DAY_CAP 50, 7-day hold, known+ to cash earnings). Mainnet, fees, multisig, legal/KYC remain his, undecided.**

1. **DECISION (blocks S3):** accept *plain SPL vault + indexer* and park the Anchor escrow
   (reverses the 10-02 "devnet ESCROW = money truth" item). Recommendation: accept; the
   scaffold stays untracked in `~/neuron-escrow` for a later cross-hub need.
2. **DECISION (blocks S1):** `RATE = 100 credits/µUSDC` (1 USDC = 10^8 token-layers) and the
   new overdrafts ($0.01 / $0.25 / $2.50). Recommendation: accept; it keeps every existing
   ledger row valid.
3. **DECISION (S4 defaults):** `AUTO_CAP` 5 USDC/tx, `DAY_CAP` 50 USDC, hold 7 days,
   earnings cashable at `known+`. Recommendation: accept for devnet; revisit at mainnet.
4. **Not now, explicitly his:** mainnet, fees, multisig, legal/KYC (§9).
