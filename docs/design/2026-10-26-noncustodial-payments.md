# Nakshatra non-custodial payments on Solana

**Status:** design, not built. 2026-10-26. Devnet now; mainnet is Biswa's call.
**Supersedes:** the money half of `2026-10-14-solana-usdc-credits.md` (custodial vault, deposit
accounts, sweeps, payouts) and the escrow-transfer money moves of
`2026-10-15-payment-bound-admission.md` §1.1/§4.1. The *admission* half of both (claim tokens,
direct-only transport, per-worker cumulative receipts, session isolation) carries over unchanged.

**Owner decisions (Biswa, 2026-10-26):**
- Nakshatra never holds anyone's assets. The treasury only earns a fee on transactions. Users load
  and transact from their own wallets.
- Adding Nakshatra to a harness creates the user's wallet automatically, on their own device, with
  no extra steps: one onboarding script, idempotent.
- **Nakshatra runs nothing in the money path and controls nothing.** The network is open source;
  users interact with the program directly. No fee sponsorship, no Nakshatra-run crank, no pause
  switch. Users hold the SOL for their own transactions.
- **The treasury fee is 3%**, for R&D, fixed in the program.

These owner revisions override any earlier wording below that assumes a sponsor, a crank, a pause
switch, a launch cap or a Nakshatra-held arbiter (see §14).

Unit: USDC (classic SPL token, 6 decimals).

## 0. The shape in one paragraph

One Anchor program, `nakshatra_pay`, holds **per-payer escrows**: a PDA plus a program-owned USDC
vault. Money leaves an escrow only three ways, all enforced by the program:
- (a) the owner withdraws **uncommitted** funds at any time;
- (b) a worker, or anyone acting for it, redeems a **payer-signed cumulative voucher** on a
  **channel** and receives the delta minus a capped fee;
- (c) a **task** pays out or refunds by the payer's acceptance, the arbiter's pay/refund, or a
  timeout.

Agents pay without a human per request through an on-chain **delegate**: a session key with a
lifetime cap and expiry, which the wallet registers once. The Neuron ledger stops being money; it
becomes the admission meter and the channel book, a read model of the chain plus vouchers. Redeems
and timeouts are permissionless: each worker's own node submits them and pays its own fees.
Nakshatra runs no service in the money path. A 3% fee lands in the R&D treasury's token account.

## 1. Decision and the alternatives rejected

**Picked:** one-way payer→worker payment channels with cumulative vouchers, verified on-chain
through the Ed25519 precompile, plus a separate fixed-reward task family. Both draw from one
per-payer escrow.

| Rejected | Why |
|---|---|
| Per-request SPL transfer / Solana Pay / x402 per call | One tx per request: ~0.4–2 s on the inference path, 5k+ lamports per call, the consumer burns SOL continuously, no per-token granularity. Could later sit on top for one-shot stranger tasks; not the rail. |
| Keep Neuron credits as money and mirror to chain | Two sources of truth, and still custodial between deposit and withdrawal, which the owner rejected. |
| State compression / cNFT-style vouchers | Nothing to compress; adds an indexer trust root to the money path. |
| Two-way or multi-hop channels | Payer→worker only; a worker never owes a payer. |
| Time-based streaming (Streamflow-style) | Pays for wall-clock time, not delivered tokens; disconnects overpay. |
| 7-day hold + arbiter on inference redeems (today's `HOLD_S`) | A voucher *is* the payer's consent after seeing the tokens. In this model the worker cannot claim, only present what the payer signed. Holds stay for tasks (§5). |

## 2. On-chain program: accounts, seeds, instructions, events

Toolchain on the hub: anchor-cli 1.2.0 (avm), cargo-build-sbf 4.1.0 / platform-tools v1.54,
solana-cli 4.1.2, `solana-test-validator`. Starting point: the untracked Anchor scaffold at
`/home/prithvi/neuron-escrow/escrow/` (counter template only), moved to
`/home/prithvi/neuron/programs/nakshatra_pay/`. Classic SPL Token only (`spl_token::ID` pinned; no
token-2022, no transfer hooks); the mint is pinned in `Config`.

### 2.1 Accounts (all PDAs, bump stored, Anchor discriminators)

| Account | Seeds | Fields |
|---|---|---|
| `Config` | `["config"]` | `authority`, `pending_authority: Option`, `mint`, `fee_wallet` (token account, mint-checked), `fee_bps: u16` (≤ `FEE_BPS_MAX = 1000`), `arbiter`, `paused: bool` (blocks opens/funds only, never withdraw/close/timeout), `close_grace_s` (default 86 400), `accept_grace_s` (86 400), `dispute_backstop_s` (14 days), `max_escrow_amount: u64` (launch cap per escrow; 0 = none), `bump` |
| `Escrow` | `["escrow", config, owner]` | `owner`, `vault`, `committed: u64` (Σ open channel `cap − redeemed` + Σ open task `amount`), `channel_seq: u64`, `open_channels: u32`, `open_tasks: u32`, `bump`, `vault_bump` |
| vault | `["vault", escrow]` | SPL token account, `authority = escrow` PDA, `mint = config.mint` |
| `Delegate` | `["delegate", escrow, delegate_pubkey]` | `escrow`, `key`, `cap_total: u64` (lifetime), `spent: u64`, `expires_at: i64`, `flags: u8` (OPEN_CHANNEL, OPEN_TASK, ACCEPT_TASK, DISPUTE_TASK), `bump` |
| `Channel` | `["channel", escrow, channel_seq_le]` | `escrow`, `seq`, `worker` (payee wallet), `voucher_signer` (the owner or delegate key that opened it), `delegate: Option<Pubkey>`, `cap`, `redeemed`, `fee_bps_snapshot`, `status` (Open, Closing), `close_at: i64`, `rent_payer`, `bump` |
| `Task` | `["task", escrow, task_hash]` | `escrow`, `worker`, `amount`, `deadline`, `task_hash: [u8;32]`, `result_hash: [u8;32]`, `status` (Open, Submitted, Disputed), `submitted_at`, `disputed_at`, `reason_code: u8`, `delegate: Option`, `fee_bps_snapshot`, `rent_payer`, `bump` |

Seeds never recur: `channel_seq` only increases per escrow, so a closed channel's address is never
re-created and an old voucher can never land on a fresh account (the re-init replay class).

### 2.2 Instructions

**Config:**
- `config_init(fee_bps, fee_wallet, arbiter, grace values)`.
- `config_update(...)` (authority only; `fee_bps ≤ FEE_BPS_MAX`).
- `config_propose_authority(new)` / `config_accept_authority`: a two-step handover, so the authority can never be bricked.

**Escrow:**
- `escrow_init()`: the owner signs; creates the escrow and vault. The rent payer is recorded.
- `escrow_deposit(amount)`: owner's token account → vault. A raw SPL transfer into the vault also works and is counted by `vault.amount`; the wrapper exists only for a uniform event.
- `escrow_withdraw(amount)`: owner only; `amount ≤ vault.amount − committed`. No notice period is needed, because in-flight work is protected per channel and per task, not per escrow.
- `escrow_close()`: owner only; requires `committed == 0`, no open channels and no open tasks. Sweeps the remainder, closes vault and escrow, and returns rent.

**Delegate:**
- `delegate_set(key, cap_total, expires_at, flags)`: owner signs; upsert; `cap_total ≥ spent`.
- `delegate_revoke(key)`: owner only; closes the account.

Revoking a delegate does not invalidate channels it already opened: their cap was committed at
open and `voucher_signer` is stored on the channel. The owner closes those with
`channel_close_request`.

**Channel:**
- `channel_open(worker, cap)`. The signer is the owner, or a delegate with OPEN_CHANNEL, `now < expires_at` and `spent + cap ≤ cap_total`. It sets `committed += cap`, requires `vault.amount ≥ committed` (fully collateralized at open), snapshots `fee_bps`, and increments `channel_seq`.
- `channel_fund(add)`: same rules.
- `channel_redeem(cumulative_total)`: permissionless signer; see §3.
- `channel_close_request()`: by the owner or `voucher_signer`. Sets `status = Closing` and `close_at = now + close_grace_s`; the worker may still redeem until then.
- `channel_close_finalize()`: permissionless after `close_at`, or at any time when signed by `worker` (cooperative close). It sets `committed −= cap − redeemed`. If a delegate is set and alive, it also sets `delegate.spent −= cap − redeemed`, so the delegate cap is an exact lifetime spend bound. It closes the account and returns rent to `rent_payer`.

**Task:**
- `task_open(worker, amount, deadline, task_hash)`: owner, or delegate with OPEN_TASK. `deadline ∈ [now+60 s, now+30 d]`; `committed += amount`.
- `task_submit(result_hash)`: worker; `now ≤ deadline`; Open → Submitted.
- `task_accept()`: owner, or delegate with ACCEPT_TASK; pays.
- `task_dispute(reason_code)`: owner, or delegate with DISPUTE_TASK. Only while Submitted and `now < submitted_at + accept_grace_s`; → Disputed.
- `task_resolve(Pay|Refund)`: `config.arbiter` only, Disputed only. The parties and amount are fixed at open, so it **cannot redirect** funds.
- `task_timeout()`: permissionless. Exactly today's `_advance_task_hold_in`:
  - Open ∧ `now > deadline + accept_grace_s` → refund;
  - Submitted ∧ `now ≥ submitted_at + accept_grace_s` → pay (silence pays);
  - Disputed ∧ `now ≥ disputed_at + dispute_backstop_s` → pay.

"Pay" means `amount·(1−fee)` to the worker's token account and the fee to `fee_wallet`, then
`committed −= amount` and close. "Refund" means `committed −= amount` (the funds simply become
uncommitted in the vault), then close.

**Events.** These use `emit_cpi!` (self-CPI), so they survive log truncation and the indexer
parses them from inner instructions:
`EscrowInit`, `Deposited`, `Withdrawn`, `DelegateSet`, `DelegateRevoked`, `ChannelOpened`,
`ChannelFunded`, `Redeemed{channel, seq, cumulative_total, delta, fee, by}`, `ChannelClosing`,
`ChannelClosed`, `TaskOpened`, `TaskSubmitted`, `TaskDisputed`,
`TaskPaid{how: accept|silence|arbiter|backstop}`, `TaskRefunded{how: expired|arbiter}`,
`ConfigUpdated`.

## 3. Micropayments: cumulative vouchers and the Ed25519 pattern

**Voucher message** (88 bytes, fixed layout, no JSON on-chain):
`b"NAK_PAY_VOUCHER1" (16) || program_id (32) || channel (32) || cumulative_total_micro_usdc u64 LE (8)`.
It has no nonce and no expiry. The channel address is unique forever and `channel.redeemed` only
increases, so replay is a no-op. The only time bound that matters is the close grace.

**Transaction shape for `channel_redeem`:**
- ix[0] is an Ed25519 program instruction with one signature. The pubkey is `channel.voucher_signer` or `escrow.owner`; the message is the 88 bytes.
- ix[1] is `channel_redeem(cumulative_total)`.

The program:
1. Takes `instructions_sysvar` with `address = sysvar::instructions::ID`.
2. Loads the ed25519 instruction with `get_instruction_relative(-1, …)`, a relative index rather than a caller-supplied one. Checks `program_id == ed25519_program::ID` and that it has no accounts.
3. Parses the header: `num_signatures == 1`, then the 14-byte offsets struct.
   **Pitfall 1:** the signature, public-key and message instruction indexes must each equal
   `u16::MAX` ("this instruction"). Otherwise the precompile verified bytes living in another
   instruction, chosen by the attacker.
4. **Pitfall 2:** checks `message_data_size == 88` and that the pubkey, signature and message slices lie inside the ed25519 instruction data. It then compares the pubkey to the accepted signers, and the message to one it **rebuilds itself** from the channel account and the `cumulative_total` argument. It never trusts "an ed25519 instruction exists".
5. **Pitfall 3:** the message carries `program_id` and the channel, so a voucher cannot be replayed across programs, forks or channels. The domain tag separates it from anything else the same key signs, since the node key also signs envelopes and receipts.
6. **Pitfall 4:** requires `get_stack_height() == 1`, rejecting being CPI'd into, so "-1" always means the real neighbouring instruction.
7. Then checks:
   - status is Open or Closing;
   - `redeemed < cumulative_total ≤ cap`.

   Then computes:
   - `delta = cumulative_total − redeemed` (checked math);
   - `fee = delta·fee_bps_snapshot/10 000` (floor, on each delta).

   Then does two `transfer_checked` CPIs signed by the escrow PDA, sets
   `redeemed = cumulative_total` and `committed −= delta`, and emits.

**Worker-side loss bound.** Today the worker checkpoints every 8 generated tokens or 15 s
(`paid_admission.py:709-711`). The exchange becomes:
1. The worker sends its cumulative receipt.
2. The payer's gateway checks it against its own token count.
3. The payer returns a voucher for the new cumulative total.
4. The worker verifies the signature locally, persists it, and continues.

If no voucher arrives within one checkpoint, the worker stops streaming. The worker's loss is at
most one checkpoint of work; the payer's loss is zero by construction.

**Mapping from `spend_usage_receipts`.** Today `cumulative_cost` is per `(hold_id, worker_keyid)`
and the ledger pays the delta from the hold. The receipts stay exactly as they are: they remain the
worker's signed claim and the admission meter.

The channel is per (payer escrow, worker) and spans sessions. So the voucher total is
`channel_base + cumulative_cost(this hold)`, where `channel_base` is the total vouched before this
hold began, kept in the payer's own book (`~/.nakshatra/pay/channels.sqlite`). Receipts stay in
credits (`RATE = 100` credits/µUSDC). The voucher is `floor(cumulative_credits / RATE)` µUSDC:
the floor of the cumulative total, never of deltas, so there is no rounding drift.

**Double-spend:** prevented by per-channel `redeemed` on-chain plus the worker's own latest-voucher
store. Two vouchers for the same total are harmless; a lower total is rejected.

**Close/timeout:** the payer calls `channel_close_request`, a 24 h grace follows (the worker's
outbox retry window), then anyone finalizes. The worker can close cooperatively at once.

## 4. Who signs vouchers: the delegate is the session key

The user's wallet signs **once**: `escrow_init`, `escrow_deposit`, and
`delegate_set(node_key, cap_total, expires_at, flags)`. After that, the consumer node's existing
Ed25519 node key (`~/.nakshatra/keys/worker.ed25519`) is the delegate. It opens channels within
`cap_total`, signs vouchers, and opens and accepts tasks. Its compromise costs at most
`cap_total − spent`, now enforced by the program rather than by the hub.

Relation to existing custody:
- **Two caps.** The sthambha signer's `spend-cap` grant (`per_tx_cap`, `day_cap`, `needs_human`) is the **soft, local** cap; the on-chain `Delegate` is the **hard** cap.
- **Wallet-changing acts.** `delegate_set` and `escrow_withdraw` go through the same `person_custody` approval door. Broker mode means Face ID through :8902, the only approval path.
- **Starter grants.** The inviter registers the newcomer's node key as a delegate on the **inviter's own escrow**, with `cap_total = starter`, `expires_at = invite TTL` and OPEN_CHANNEL only. Nobody mints, the inviter pays, and Sybils can only spend what a real escrow put behind them.

## 5. Paying only for delivered work

The program verifies signatures, amounts, time and state. It cannot verify tokens, output hashes
or model weights.
- **Inference (channels):** the payer sees every token before it signs the next voucher. There is no hold, no arbiter and no refund path on-chain: vouchers are final on redeem. For garbage output, the payer stops signing and uses a roster report or suspension (unchanged).
- **Tasks:** results arrive as a blob after the fact, so the dispute machinery stays, moved on-chain one-to-one from `task_holds`. The flow is submit → 24 h to accept or dispute → silence pays; a dispute goes to the arbiter for Pay or Refund, and a 14-day backstop pays if the arbiter is silent.
- **The arbiter is narrow.** It can only choose between the two pre-named parties, for the pre-fixed amount, and only while the task is Disputed. It cannot touch channels, undisputed tasks or escrows.

## 6. What happens to each existing feature

| Existing | Fate |
|---|---|
| Spend holds (authorize/claim/settle/abandon/release) | **Off-chain, kept** as admission meter and claim token. `authorize_spend` stops moving credits and checks `channel.cap − redeemed − outstanding_vouched ≥ max_cost` instead. |
| `spend_usage_receipts` | Kept, plus a payer-signed `spend_vouchers` table. |
| Task holds | **On-chain** Task family; `settle.py` gains a `ChainTaskAdapter`; ledger task tables become a mirror. |
| Disputes | Tasks: on-chain. Inference: dropped (the voucher is consent). |
| 7-day maturity, funded/unfunded earnings, withdrawable, legacy funding projection | **Dropped**: every redeem is funded by definition. |
| Overdraft / tier credit limits as money | **Dropped** as money; tiers stay for roster and serving policy. |
| Starter grants | **On-chain** as an inviter-funded delegate (§4). |
| Deposit accounts, sweeps, payout addresses, withdrawals + presence | **Dropped**: the user's own wallet plus `escrow_withdraw`. The code pattern is reused by the crank. |
| Payment-bound admission (10-15) | Kept; "collateralized" now means channel capacity on-chain. |
| Sybil rules | A fee on every transaction, inviter-funded starters, and roster tiers. |
| `RATE`, per-token-layer pricing | Kept off-chain; on-chain amounts are plain µUSDC. |

There are two instruction families but **one program**: they share the escrow and the fee logic,
and a second program would double the audit surface and the upgrade-authority exposure.

## 7. Fees, fee wallet, indexer and reconcile

- **Fee:** taken only on `channel_redeem` and task "pay", from the delta, at the `fee_bps` snapshotted at open. A config change never re-prices work already in flight. The program caps it at 10%; the devnet default is **1%**; the mainnet value is Biswa's call.
- **Fee wallet:** a USDC token account owned by a Squads multisig (on devnet, a vault-held key is acceptable). Fees are the only thing the treasury ever holds.
- **Indexer:** the loop stays and the source changes. It reads `getSignaturesForAddress(program_id)` at finalized, parses `emit_cpi` events, and writes `chain_events`. `solana_rpc.py` carries over unchanged.
- **Reconcile** (signed daily report, kill file, Buzz alert) checks new invariants per escrow:
  - `vault.amount ≥ committed`;
  - `committed == Σ open (cap − redeemed) + Σ open task amounts`;
  - the fee wallet's inflow equals the fees in the events;
  - the channel book's outstanding vouchers never exceed on-chain `cap − redeemed`.

  On a mismatch, the kill file stops the **crank**. There is nothing custodial to halt.

## 8. Keys and authorities: what each compromise can do

| Key | Held by | Compromise can… | Cannot… |
|---|---|---|---|
| Program upgrade authority | devnet: vault key; **mainnet: Squads 2-of-3 with a time-lock**, then frozen after audit and a quiet period | replace the program and so drain everything; the only key that can, hence multisig, time-lock and a planned freeze | — |
| Config authority | the same multisig | raise the fee to at most 10%, change the fee wallet or arbiter, pause opens | block withdraws, closes or timeouts (exempt from pause in code) |
| Arbiter | hub vault hot key | mis-resolve a *disputed* task between its two parties | touch channels, escrows, undisputed tasks or amounts |
| Crank / sponsor key | hub vault, SOL only | waste the capped SOL float | move USDC (redeem pays the channel's fixed worker) |
| User wallet | user's device | everything on its own escrow | touch anyone else's escrow |
| Delegate (node key) | user's node | spend up to `cap_total − spent` into channels/tasks it opens | withdraw, revoke, or exceed the cap |
| Worker node key | worker | sign receipts (claims) | redeem more than the payer vouched |

## 9. UX and costs

- **SOL:** a consumer would need about 0.01 SOL once (escrow and vault ≈ 0.0039, delegate ≈ 0.0017, each channel ≈ 0.0021 rent-exempt, refunded on close) plus about 5k lamports per transaction. With sponsorship (§13.4) the user needs none.
- **Workers need no SOL.** Redeem and timeout are permissionless and the hub crank submits them, creating the payee's token account idempotently (~0.002 SOL per new worker, hub-paid).
- **Transaction counts.** A consumer does 3 once (init, deposit, delegate), plus 1 per worker channel (long-lived), plus an occasional fund or withdraw. A worker does 0 during inference: the crank redeems when the unredeemed amount reaches 1 USDC or after 24 h, so about one transaction per channel per day.
- **Finality.** The worker UI uses confirmed. The indexer, channel book and capacity reads use finalized (~13 s), as today.
- **Batching.** v1 does one redeem per channel per transaction; batching several signatures per transaction can come later.

## 10. Security checklist (program) and audit plan

**Checklist** for every instruction:
- Signer checks on owner, delegate, worker, arbiter and authority.
- Every PDA: seeds + bump equal the stored bump.
- Explicit escrow relations on channel, task, delegate and vault.
- Token accounts: mint == config mint, the expected owner, `token_program == spl_token::ID` (no token-2022).
- Checked u64 math everywhere, with the fee computed through a u128 intermediate.
- No `init_if_needed`.
- Accounts close only from a terminal state, and cannot be resurrected in the same transaction.
- `paused` exempts withdraw, close and timeout.
- Deadline inclusive/exclusive semantics exactly as `submit_task_hold`.
- Ed25519 parsing exactly per §3.
- Strictly sized instruction data.
- Duplicate-account attacks: the payee account, fee wallet and vault must all differ.
- No `remaining_accounts`.
- Events are emitted after the state change.
- The IDL is committed, and the build is verifiable (`anchor build --verifiable`).

**Audit plan before mainnet:**
1. An adversarial test matrix in-repo (M2/M3), on `solana-test-validator` and `litesvm` (clock warp).
2. Sec3 X-Ray, `cargo audit` and clippy pedantic in CI.
3. A Codex adversarial review plus an Opus final pass, i.e. two independent reads of the ed25519 parser.
4. One external audit (OtterSec/Neodyme class) on the frozen commit.
5. Launch with `max_escrow_amount` (for example 100 USDC) and a global TVL alarm in reconcile for 90 days.
6. A bug-bounty note in `docs/THREAT_MODEL.md`.
7. The upgrade authority goes to a multisig with a time-lock at deploy; the freeze decision comes after the quiet period.

## 11. Migration and phased build plan

**Migration from the custodial devnet:** there is no mainnet value, so **no migration code**.
1. Freeze the old ledger: `NAK_FUNDS=off`, `NAK_FUNDS_ROUTES=off`, kill file.
2. Publish the final signed reconcile report and a balance snapshot.
3. Let the devnet operators re-deposit into their own escrows on the new program.

The Neuron SQLite stays as the admission meter; its money tables stop being written.

| # | Milestone | Test |
|---|---|---|
| M0 | Move the scaffold to `neuron/programs/nakshatra_pay/`; pin anchor 1.2.0 and platform-tools; `Config` and the config instructions | `anchor test` on a local validator: init; a fee over the cap is rejected; two-step authority |
| M1 | Escrow: init, deposit, withdraw, close; vault PDA | withdraw > uncommitted fails; a raw SPL transfer into the vault is counted; close refunds rent; a token-2022 mint is rejected |
| M2 | Delegate, Channel, `channel_redeem` with the ed25519 parser | adversarial matrix: wrong signer; offsets pointing at another instruction; size ≠ 88; tampered total; replay; total > cap; redeem on Closing after `close_at`; delegate cap exceeded or expired; an existing voucher still redeems after revoke; fee floor; pause blocks open but not close/withdraw; clock warp for grace |
| M3 | Task family, arbiter, `task_timeout` | port the task-hold test cases: silence pays, expired refund, dispute → resolve pay/refund, the 14-day backstop pays, the arbiter cannot act on Open or Submitted |
| M4 | Python: `neuron/python/chain/program_client.py` (instruction builders from the IDL); voucher sign/verify in `paid_admission.py` (consumer signs; worker verifies and stops serving without a voucher); `nak pay` CLI; `ChainTaskAdapter`; capacity check behind `NAK_PAY_MODE=chain` (unset = today's behaviour) | temp HOME, fake presence, fake worker: a paid session against the test validator, with the voucher chain matching the receipts; both repos' tests green |
| M4a | **Onboarding wallet** (§13.1–13.3, 13.5) | see §13.6 |
| M4b | **Fee sponsorship** (§13.4) | see §13.6 |
| M4c | **Automatic delegate and starter** (§13.3–13.4) | see §13.6 |
| M5 | Crank (the payout.py engine reused: sign before broadcast, process lock, kill file, dry-run unless `NAK_CRANK=on`); indexer on program events; reconcile invariants; systemd units default-disabled | end to end on the test validator: deposit → delegate → session → voucher → crank redeem; worker receives delta − fee, the fee wallet receives the fee, reconcile is clean; a bogus voucher is refused and flagged |
| M6 | Devnet drill with Biswa present: a real wallet, a real worker, one task disputed and resolved by the arbiter; then retire the custodial routes | the explorer and the signed reconcile report agree; the old routes return 404 |
| M7 | Mainnet gate (Biswa only): multisig, audit, TVL cap, legal check (fee income only, no transmission of user funds) | — |

**Rollback at every step:**
- The program is additive on devnet.
- Leaving `NAK_PAY_MODE` unset returns the ledger to today's behaviour.
- With the crank off, nothing moves.
- User funds are always withdrawable by the user, whether or not our services are up. That is the non-custodial property, and M1 tests it first.

**Carries over unchanged:**
- `solana_rpc.py`;
- the payout.py transaction/lock/kill-file pattern;
- reconcile's report signing and alerting;
- `paid_admission.py` metering and outbox;
- the spend and task tables, as the meter;
- the `person_custody` approval door;
- the `settle.py` adapter contract;
- consumer invites with `starter`.

## 12. The one decision for Biswa

**Inference vouchers are final: no 7-day hold, no arbiter and no refund on channel redeems.** This
reverses the 2026-10-15 decision on seven-day maturity for paid earnings.

Recommendation: accept. The payer signs only after receiving each checkpoint of tokens, so the hold
no longer protects anyone, and keeping it would force a custodial-style pending state back into the
program. Disputes and the arbiter stay for tasks.

The fee default (1%) and the mainnet multisig, audit and legal items are also his, but none of them
blocks the devnet build.

## 13. Wallet created automatically at onboarding

**Requirement:** adding Nakshatra to a harness creates the Solana wallet automatically, on the
user's own device, with zero extra steps, from one script, idempotently.

### 13.1 Where it hooks

There is exactly one onboarding path, and it already makes two keys. The invite one-liner
(`docs/JOIN.md`) downloads the pinned release and runs `release/install.py`, which runs
`join_setup.main` → `ensure_identity(home)` (`scripts/network/join_setup.py`). That function
already creates the node key (`~/.nakshatra/keys/worker.ed25519`, `O_CREAT|O_EXCL`, 0600) and the
person key idempotently, then the agent grant. **The wallet becomes the next step of that same
function:**
- **Wallet seed.** `~/.nakshatra/pay/wallet.ed25519` holds a raw 32-byte seed. The directory is 0700 and the file is created 0600 with `O_CREAT|O_EXCL`, exactly like the node key. It must not use `signer.save_key`, which truncates and would overwrite.
  - If a wallet file already exists, it is kept.
  - If an existing file is unreadable or invalid, the whole run refuses before writing anything.
  - **Re-running never replaces a wallet.**
- **Address.** `~/.nakshatra/pay/wallet.pub` holds the Solana address (base58 of the ed25519 pubkey).
- **Nodes that already joined.** The established-identity path runs the same step, so every existing node gets its wallet on its next join or update, with no new command.
- **Installer and docs.** `release/install.py` needs no change, since it already calls join. The final printed line gains the pay address. `docs/JOIN.md` step 4 becomes "makes your keys: a node key, a person key, and a pay wallet".
- **Windows.** nak runs inside WSL, so the wallet lives in the WSL home. `friend-join.ps1` installs mesh DNS, CA and agent pointers, not nak, so it needs no wallet step.
- **No network needed.** Creating the key needs no RPC. The on-chain `escrow_init` is sponsored (13.4) and deferred if the hub is unreachable, so joining never blocks on the chain.

### 13.2 Storage and custody per OS

This reuses `person_custody` exactly. A one-word custody file `~/.nakshatra/pay/custody` holds
`file` or `broker`, and `KINDS` gains `wallet-withdraw`, `wallet-delegate` and `wallet-close`.

**The wallet seed never leaves the device, in either mode.** `broker` means "key on the device,
consent on the phone": the broker gates the signature but never holds the key. A broker-held key
would make Nakshatra the custodian. The default custody is `file`, the same tier the person key has
today.

| OS | Default | Why |
|---|---|---|
| Linux (the supported node OS) | raw seed file 0600 in a 0700 directory | identical to the node key; a TPM cannot sign ed25519 and sealing only re-wraps a file (optional later); libsecret needs a desktop session that headless nodes lack |
| Windows | the same file inside the WSL distro | DPAPI/Credential Manager are unreachable from WSL without PowerShell interop, and the WSL filesystem is readable by the same Windows user anyway, so DPAPI adds nothing against same-user malware |
| macOS (consumer harness, when it exists) | a Keychain generic-password item holding the seed | the Secure Enclave cannot hold ed25519 (P-256 only) |
| Hub (Biswa's own node) | vault entry `nak-pay-wallet`, custody `broker` via :8902 | the existing pattern |

### 13.3 Key relationship: a separate wallet key, with the node key as its delegate

One script creates three keys, and the user never sees any of them:

| Key | Role | Custody | Signs |
|---|---|---|---|
| person key | authority: grants, invites, revoke, suspend | file now, broker later | rarely; human decisions |
| node key | hot identity on every service | file, loaded by nakd and the gateway | envelopes, receipts, **vouchers** (as the on-chain delegate) |
| **wallet key** | money root | file, never off the device | escrow init/deposit/withdraw/close, delegate set/revoke |

Why not reuse an existing key:
- The node key is loaded by long-running processes and signs constantly. Its compromise must cost at most the delegate cap, never the escrow.
- The person key must be rotatable, or moved to broker custody, without moving funds.
- A wallet derived from a public roster key would link the node to its whole on-chain history for every stranger.

"Fewer keys" buys nothing, because the three already have different custody paths.

**Automatic delegation.** `delegate_set` needs an escrow, so it runs at **first funding**, not at
join. `nak pay ensure` is idempotent and is called by `nak gateway on` and by every `nak pay`
command. It:
- registers the node key as a delegate with defaults `cap_total = 5 USDC`, `expires_at = +30 days` and all four flags;
- writes the matching local soft cap: 1 USDC per transaction, 5 USDC per day.

In `file` custody this is silent; in `broker` custody it is one phone approval. The limits change
with `nak pay limits --cap 20 --days 90 --per-tx 2 --per-day 10`. `nak pay ensure` renews the
delegate within the saved limits before it expires.

### 13.4 Funding UX: the user never needs SOL, and may never need to deposit

1. **Fee-payer sponsorship (Nakshatra pays SOL and rent).**
   - **Flow.** The consumer builds a transaction with the sponsor as fee payer, signs as the instruction authority, and POSTs the partially signed bytes to `POST /pay/sponsor`. That is a new ledger route with a signed envelope, so the caller must be a roster or invited key.
   - **What the sponsor accepts.** The sponsor holds only a small SOL float and accepts a transaction **only if**:
     - every instruction is in the allowlist `{nakshatra_pay, ATA create_idempotent, ComputeBudget}`;
     - the sponsor appears only as fee payer or rent payer, never as a transfer source or token authority;
     - the per-wallet budget holds (5 tx/day, 0.02 SOL/day; global 0.5 SOL/day on devnet).

     Then it co-signs, broadcasts, and records the transaction.
   - **Rent.** Rent-bearing accounts name the sponsor as `rent_payer`, so rent returns when they close. In steady state, sponsorship costs transaction fees, not rent.
   - **Worst case.** The float is capped (devnet 0.5 SOL; mainnet about 1 SOL, topped up by a timer), so the worst case of a parser bug is losing the float. This addresses the known Solana pitfall that an arbitrary instruction in a sponsored transaction can drain the fee payer.
   - **Withdrawals.** `escrow_withdraw` is sponsorable only to the wallet's own token account.
2. **Inviter-funded starter: spend before owning anything.** `nak invite --kind consumer --starter 1.00` makes the inviter's acceptance step call `delegate_set(newcomer node key, cap_total = starter, expires_at = invite TTL, OPEN_CHANNEL)` on the **inviter's** escrow.
   - The newcomer spends from the inviter's escrow immediately; their own escrow isn't needed until they deposit.
   - It replaces `grant_starter` one-to-one, bounded by `STARTER_MAX`.
   - The inviter can revoke at any time.
3. **One address to deposit to.** Onboarding sponsors `escrow_init`, and `nak pay address` prints the **vault** token account. Raw USDC transfers into it are accepted, so exchanges and wallets can send without the user owning a token account or any SOL. `nak pay status` shows the balance, the committed amount and the delegate cap. A withdrawal creates the wallet's token account idempotently in the same sponsored transaction.
4. **Devnet:** `nak pay faucet` mints test USDC straight into the vault (own test mint; refuses any cluster but devnet).

### 13.5 Backup and recovery

**Default: accept loss for small balances, and show nothing at onboarding.** A 24-word phrase
printed by a non-interactive installer that an agent harness runs ends up in a log or a transcript,
which is worse than no backup. Instead:
- **On-demand backup.** The wallet file joins the existing "back up your person key" guidance.
  - `nak pay backup` prints the seed as a BIP39 24-word phrase on demand.
  - `nak pay export --keypair` prints the 64-byte keypair that Phantom and Solflare import.
- **Exposure is capped twice.** On-chain, `max_escrow_amount` launches at 100 USDC. And `nak pay deposit` / `status` warn whenever more than 20 USDC is in escrow, until `nak pay backup --confirm` records an acknowledgement.
- **Device lost without a backup.** Escrow funds are unrecoverable; that is what non-custodial means, stated plainly. Inviter-funded starters are unaffected: they're on the inviter's escrow, and the inviter revokes the dead delegate. In-flight channels: workers still redeem vouchers they already received, and the rest returns to the lost escrow after the close grace.
- **Not in v1.** An encrypted backup of the seed to the user's own pillar or mesh. If it's ever built, it goes from the user's device to the user's device, never through Nakshatra.

### 13.6 Onboarding milestones

| # | Milestone | Test |
|---|---|---|
| M4a | Wallet step in `ensure_identity` and the established-identity path; final message; `person_custody.KINDS` gains the wallet kinds; `nak pay address/status/backup/export`; `docs/JOIN.md` step 4; the release lock gains `solders` with hashes (the one new dependency) | temp HOME: the first join creates `wallet.ed25519` 0600 and `wallet.pub`; a second join keeps the bytes identical; a corrupt existing file makes the run refuse before any write; `wallet.pub` matches the seed; the test makes no network calls |
| M4b | Sponsor route `POST /pay/sponsor`, allowlist parser, budgets, `sponsored_tx` table, float alarm | test validator: a sponsored `escrow_init` lands while the user holds 0 SOL; a transaction containing a transfer from the sponsor is rejected; the 6th transaction of the day is rejected; closing the escrow returns rent to the sponsor; the parser is fuzzed with mutated instruction lists |
| M4c | `nak pay ensure` (automatic delegate with defaults, the local soft cap, renewal); `nak pay limits`; inviter `--starter` → delegate on the inviter's escrow | after `nak pay faucet`, the delegate exists with 5 USDC / 30 days and the gateway completes a paid session with no human prompt (file custody); broker custody asks exactly once; a starter invite lets an empty escrow run a paid session from the inviter's escrow; revoke stops it |

Rollback for all three: the files are additive (`~/.nakshatra/pay/`), and the sponsor route is off
unless `NAK_PAY_SPONSOR=on`. Nothing here touches the hub GPU, Prithvi's gateway environment or the
outreach loop.

The devnet defaults (5 USDC / 30-day delegate, 1 USDC per transaction, the 20 USDC backup warning,
0.02 SOL/day sponsorship) can each be changed with one config line.

## 14. Owner revisions (Biswa, 2026-10-26) — these override earlier sections

1. **No fee sponsorship.** Drop §13.4 item 1, the `POST /pay/sponsor` route and milestone M4b. Users
   hold the SOL for their own transactions (about 0.01 SOL once for rent, refunded on close, plus
   ~0.000005 SOL per transaction). Onboarding (§13.1) creates the wallet and prints its address; the
   user funds it with USDC and a little SOL. `rent_payer` is always the signer who created the account.
2. **No Nakshatra-run crank.** Redeem and `task_timeout` stay permissionless. The worker's own node
   runs the redeemer (when its unredeemed total passes a threshold or once a day) and pays its own
   fees. The crank code in M5 becomes part of the worker node, not a hub service.
3. **Fee: 3%** (`fee_bps = 300`), paid to the R&D treasury's USDC token account.
4. **No pause switch and no `max_escrow_amount`.** Both were operator-only settings; remove them from
   `Config`. Indexer and reconcile leave the money path: any node may run them as monitoring tools,
   with no kill switch, because nothing custodial exists to stop.
5. **Upgradeable, controlled by a Nakshatra multisig, behind an announced delay.** The upgrade authority
   is a multisig (Squads), never one key. Every upgrade, and every `Config` change (fee, fee wallet), is
   announced on chain and takes effect only after **48 hours**. That delay is longer than the 6-hour
   withdrawal window, so anyone who disagrees can withdraw before it applies. Config changes go
   through `config_propose` → `config_apply` (after `effective_at = now + 48 h`), enforced by the
   program; program upgrades get the same delay through the multisig's time-lock. Be honest in user
   docs: the upgrade key can change any rule, and the delay is the protection.
6. **6-hour withdrawal window.** `escrow_withdraw` becomes `escrow_withdraw_request(amount)` →
   `escrow_withdraw_execute()` after `withdraw_at = now + 6 h`. At execute time the amount must still
   be ≤ `vault.amount − committed`. Workers keep redeeming signed vouchers during the window, so a
   payer cannot walk away mid-job. Channel caps stay committed (fully collateralized), so a worker
   need not watch for withdrawal requests to stay safe. Close-request grace stays 24 h for channels.
7. **Inference payments are final** (accepted). The payer's own node receives every token before it
   signs the next cumulative voucher, and a worker cannot redeem without that signature, so a node
   cannot be paid for work it did not do. Delivered and signed-for work is never refunded.
8. **Tasks: paid on verification, refunded only for non-delivery or failed verification.** This replaces
   the accept/dispute/arbiter flow of §2.2/§5:
   - At `task_open` the payer names a **verifier key**: the payer's own node (deterministic checks:
     tests pass, hash matches) or a **verifier node** for results that need judgment.
   - `task_submit(result_hash)` by the worker; the verifier signs `pass` or `fail` over
     `(program, task, result_hash)`; `task_verify(pass)` pays the worker at once (minus fee);
     `task_verify(fail)` refunds the payer.
   - No submission by the deadline → anyone can refund. Submitted but the verifier silent for
     `accept_grace_s` → silence pays the worker (protects workers from an absent verifier).
   - Fairness rule to settle in M3: a payer who verifies its own task could falsely fail good work.
     So payer-as-verifier is allowed only for checks the worker can reproduce, and the result is
     logged on chain; judgment tasks use a verifier node. **Decentralized verification** (other nodes
     re-run a sample of work and compare; cheating workers lose reputation or a stake) is the path,
     designed in detail at M3. No Nakshatra-held arbiter.
