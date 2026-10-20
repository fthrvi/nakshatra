# M2 node-side report — 2026-10-18

Implementation commit: `505343c` (`net: add Neuron task escrow`)

## What I built

- Added the opt-in `NAKSHATRA_ESCROW=neuron` backend. It signs the fixed-price task-hold contract with the node Ed25519 key, requires ledger enforce mode, requires pinned HTTPS, and reads the same ledger URL/SPKI configuration as the gateway (environment first, `gateway.env` fallback). The existing local `ledger` backend remains the default.
- Kept the signer cap check at post, but deferred the Neuron `/spend/task/authorize` call until assignment so the hold binds the winning contact card's `ledger_keyid`. The task hash is the idempotent `client_nonce`, and the encrypted node store durably records the authorization and returned hold id.
- Implemented worker pay-first verification on assignment and submit. An exact open hold must match task hash, reward amount, and the worker's own ledger key; missing, mismatched, refunded, or unreachable holds remain visibly unfunded and cannot be submitted.
- Wired accept to `/spend/task/pay`, reject/expiry to `/spend/task/refund`, and the existing reconciliation loop to retry pending settlement exactly once. Local state is not marked paid until the ledger reports paid. Task rendering exposes funded/unfunded/paid/refunded (and payment-pending) state.
- Added optional encrypted `ledger_keyid` to contact cards. `nak card set --paid` is permitted only on the Neuron backend and fills the key from the node identity; there is no manual key argument and no MCP mutation tool.
- Enabled proposal-only `agent.msg` budgets for `intent=offer|ask`, unit `credits`, and integer amounts `1..10^12`; they are displayed to receivers and never move funds.
- Added `nak tasks` as a list alias and `nak escrow status <task>`. Existing task CLI/MCP reward paths now render credit units and funding state; MCP refuses to describe an unfunded assignment as workable.
- Added a shared fake ledger with conservation checks plus two-node coverage for pay, refund, expiry, mismatch refusal, unpaid-card refusal, over-cap refusal, crash reconciliation, and ledger-down retry. Added signed-route/SPKI tests for the real Neuron adapter and encryption/budget matrices.
- Did not touch installer/updater code, `INBOX.md`, the real node, a signer socket, systemd, or a real ledger.

## Tests: before vs after

Same command (temporary HOME and required Sthambha PYTHONPATH), before: **135 passed, 1 failed**. After: **146 passed, 1 failed**. The one failure is identical before and after: `test_schema_migration_is_idempotent_on_a_forgejo_main_store` assumes `forgejo/main` has no M1 `agent_messages` table, but the current moving ref already contains it.

Expanded final M2 command, excluding only that known baseline assertion: **157 passed, 1 deselected** in 36.93s. Python compilation checks also passed for every changed runtime module.

## Open questions / remaining validation

- The Neuron server side was being built in parallel, so this branch was tested against the frozen contract and an injected contract-faithful fake ledger, not a live server. A two-node live canary against the completed server remains the integration/cutover check.
- The unrelated moving-`forgejo/main` migration test should eventually pin a historical pre-M1 commit (or build its legacy schema explicitly) so the full focused command is green independently of remote branch movement.

## Fix round 1

Review: rv-m2.md. Suite (`HOME=$(mktemp -d)`, tests/test_accounting, nak_agent, nak_direct, nakd, nak_grant, nak_mcp_consumer, nak_neuron_escrow, nak_settle, nak_store_encryption, nak_tasks, nak_wallet, nak_join, nakshatra_auth): before 288 passed / 10 failed, after 302 passed / 10 failed. The same 10 fail before and after (9 `test_nak_direct` p2p tests and `test_schema_migration_is_idempotent_on_a_forgejo_main_store`, pre-existing on forgejo/main). `test_nakshatra_tls` and `test_nak_worker_service` need `grpc` and are not collectable here.

| Finding | Change | Proven by |
|---|---|---|
| 1 Same-second 401 | `NeuronEscrowAdapter._next_ts` signs with `max(int(time.time()), last_ts + 1)` under a lock (`build_signed_envelope` already took `timestamp_unix`; no other caller changed); `_request` retries once on HTTP 401 with a fresh timestamp. | `ReplayLedgerHTTP` (verifies the Ed25519 signature, rejects a seen sha256(signature) with 401): `test_adapter_never_signs_the_same_second_twice`, `test_two_adapters_same_key_same_second_retry_once_on_401`, `test_adapter_gives_up_after_one_401_retry`, and the two-node `test_verify_at_assign_then_immediate_submit_and_status_against_replay_ledger` (real adapter). Mutation (plain time, no retry) fails 4 of them. |
| 2 Card keyid unbound | Card receipt nacks and stores nothing when `ledger_keyid` != the contact's authenticated `node`; assign re-checks it and refuses with no hold. | `test_forged_ledger_keyid_card_is_refused_and_stores_nothing`, `test_assign_refuses_a_card_keyid_that_is_not_the_contacts_node_and_opens_no_hold`; both fail when their guard is removed. |
| 2b Surviving mutation | New test stores a card with a ledger key but `paid: false`; assign must refuse and open no hold. | `test_assign_requires_the_card_to_say_paid_even_with_a_ledger_key` (fails with the paid guard removed). |
| 4 Error mapping | `InsufficientFunds` (402), `HoldNotFound` (404), `HoldConflict` (409), all `SettlementError` with `.status`. `reconcile_escrow` re-reads `/spend/task/status` after a refused release/refund and adopts the ledger's terminal state (`_adopt_terminal`), also when the ledger is terminal in the other state. | `test_adapter_maps_ledger_statuses_to_distinct_errors`, `test_reconcile_adopts_the_ledgers_terminal_state_on_409`. |
| 5 Hold deadline | `_verify_worker_hold` requires hold deadline >= spec deadline. | `test_worker_refuses_a_hold_that_expires_before_the_task_deadline`. |
| 6 Claim race | On a paid claim the `posted -> assigned` CAS (assignee, epoch 1, funding `pending`) happens BEFORE the hold is authorized, so a loser gets the clean `task.reject` and no hold. If authorize fails the CAS is reverted to `posted` (assignee cleared, funding `unfunded`): the poster is not stuck, the worker's claim is nacked and may claim again, and no money moved. After authorize, `hold_id` + `funded` are recorded in one CAS before `task.assign` is queued; if the task left `assigned` meanwhile (expired) the just-opened hold is refunded at once. A same-assignee claim replayed after a crash (assigned, no hold_id) re-runs the idempotent authorize and gets the same hold. | `test_loser_of_a_claim_race_gets_the_clean_reject_and_no_hold`, `test_failed_authorize_after_the_cas_puts_the_task_back_for_a_retry`. |

Test hygiene: `_assign` now waits for the worker's hold verdict (it previously raced the worker's own verification and was occasionally flaky). Not changed: the poster can still refund an open hold after submit (design limit, needs Biswa), and the sybil/credit-line and cap-unit nits.

## Fix round 2

Review: rv2-m2.md. Suite (same command as round 1): round 1 302 passed / 10 failed; round 2 309 passed / 10 failed (same 10 pre-existing failures).

| Finding | Change | Proving test |
|---|---|---|
| 1 Timestamp drift lockout | `_next_ts` raises local `LedgerBusy` (a `SettlementError`) when `last_ts + 1 > now + 30` (`MAX_TS_LEAD_S`), without sending and without advancing `_last_ts`. A 401 is retried once only if the failed request's ts was <= now (same-second collision), never when already ahead. | `test_signing_backlog_is_refused_locally_and_never_locks_the_ledger_out` (150 calls in one second against the replay fake that now also enforces the +/-60 s window: exactly 30 sent, rest refused locally in < 50 ms, no 401, works after the clock advances), `test_recovery_loop_over_many_stranded_tasks_defers_instead_of_locking_out` (40 tasks x 3 calls defer some, clean after 61 s), `test_a_401_is_not_retried_when_the_timestamp_is_already_ahead_of_the_clock`. Mutations (no cap, unconditional retry) fail them. |
| 2 Orphan hold after ambiguous authorize | `EscrowRefused(definitive)`: definitive = ledger status in 402/403/404/409/422. Only a definitive refusal reverts to `posted`. Unavailable/timeout/5xx/backlog leave the task `assigned`, funding `pending`, same assignee. `reconcile_escrow` (called by every sweep) first runs `_finish_pending_assignments`, re-running the idempotent authorize (nonce = task hash): it returns the same hold, which is recorded (`hold_id` + `funded`) and then `task.assign` is queued; a definitive refusal reverts. Sweep does not expire an assigned paid task that has no recorded hold; after the deadline the replay still returns the existing hold (round-1 fix 7), which is recorded without notifying the worker and refunded by the next sweep. No status-by-nonce route exists in the contract, so the identical authorize is used. | `test_ambiguous_authorize_leaves_the_task_pending_and_reconcile_adopts_the_same_hold`, `test_ambiguous_authorize_then_expiry_refunds_the_unrecorded_hold` (fake that commits then drops the response). Mutation (always revert) fails both. |
| 3 "Worker may claim again" | Made true (small change): a definitive revert nacks with the `task reopened: ` prefix; on receiving a nack with that prefix, the worker node (`_claim_reopened`, using new `Store.outbox_frame`) moves the row `claimed -> offered`, only for a `task.claim` it sent to that poster. No hand reset in the test. | `test_definitive_refusal_puts_the_task_back_and_the_worker_can_claim_again`. |
| 4a Orphan refund mutation | New test: the task is expired while the hold is opening (hook after the ledger commit); the hold must be refunded. | `test_hold_opened_for_a_task_that_expired_meanwhile_is_refunded` (fails with the refund removed). |
| 4b `_adopt_terminal` in refused-release | New test: the ledger sweeps the hold as pay arrives and a stale status read still says open; reconcile's release is refused (409) and must adopt `refunded`. | `test_reconcile_adopts_the_ledgers_state_when_release_is_refused` (fails with the adopt call removed). |

Side effect to note: a pending (unknown-outcome) claim is nacked to the worker as "escrow pending"; the worker row stays `claimed` and receives `task.assign` once reconcile records the hold.

## Fix round 3

Review: rv3-m2.md. Suite (same command): round 2 309 passed / 10 failed; round 3 312 passed / 10 failed (same 10 pre-existing failures).

| Finding | Change | Proving test |
|---|---|---|
| 1 Unrecorded hold refunded/paid by the ledger stays pending forever | `settle.py` raises typed `HoldNotOpen(state, hold)` (a `SettlementError`) when a response carries a terminal state where `open` was expected. `_open_escrow` maps it to a definitive `EscrowRefused` carrying the hold. In `_open_assignment`: `refunded` reverts the task to `posted` (with the `task reopened:` nack on the claim path; the sweep then expires it normally, no hold means no further ledger calls); `paid` is adopted (`hold_id` recorded, funding `paid`) only if the hold's `task_hash` and `worker_keyid` match this task and its assignee's card key, otherwise the task is reverted. | `test_unrecorded_hold_refunded_by_the_ledger_reaches_posted_then_expired` (commit-then-drop, ledger refunds, sweep + reconcile: posted then expired, worker row expires on its own sweep, authorize call count unchanged by further sweeps), `test_ledger_paid_hold_is_adopted_only_for_this_tasks_worker` (adopt, and mismatched worker reverts). Forcing `definitive=False` or dropping the worker match fails them. |
| Sweep-skip mutation | Direct test: with the recovery authorize failing, `sweep()` past the deadline expires nothing and the task stays `assigned`. | `test_sweep_does_not_expire_an_assigned_paid_task_whose_hold_is_unrecorded` (fails with the skip removed). |

Note: on the reconcile/sweep path (no live claim to nack) the worker is not told the task reopened; its row expires on its own sweep. Not changed (nits in rv3): roster change mid-recovery orphan, one-slot-per-second throughput after a burst.

---

# Sliding grant node-side report — 2026-10-20

## What changed

- `nak grant renew` now issues a 365-day charter by default, backed by the signer's private lease key and carrying the contract defaults of a 7-day wire lease and a 30-day idle lapse. `--idle-days` (1–90), `--lease-days` (1–7), and `--no-lease` are human-only CLI controls; invalid values are rejected before signer state changes. Existing agent key, caps, node binding, and constraints remain unchanged unless the existing explicit cap flags say otherwise. Renewal creates a new grant id, so it wakes a lapsed charter.
- The confirmation prompt and completion output explain whether the operator is creating an active-while-used charter or a legacy fixed-expiry grant. The no-person-key recovery command carries the same lease choice.
- Genuine first initialization in `join_setup` now creates the same 365-day/7-day/30-day charter. Every established-authority and partial-residue path remains read-only and refuses to mint or repair, including the full M0.5 residue matrix.
- `nakd` now trusts the signer's `valid`/`problem` result instead of reimplementing expiry precedence. It renders the contract charter/lapse text into daemon status; both `nak status`/`nak grant status` and the MCP status line consume that one rendered value. Legacy grants keep the fixed-expiry text.
- MCP gained no grant, renewal, wake, lease-setting, or lease-key mutation surface. A runtime tool-list regression test checks names and input properties.
- Added two-real-node coverage for leased-envelope delivery, local lapsed-sender refusal, and a forced resend whose daemon signatures do not advance `last_used`.

## Every `nakd` signer call

| Call site | Operation | Classification | `touch` |
|---|---|---|---|
| `Node.__init__` | `whoami` | startup identity read; not a signing operation | n/a |
| `Node.__init__` | `data_key` | startup encrypted-store key retrieval; not an agent signing operation | n/a |
| `_grant_status` | `grant_status` | read-only preflight/status/warning check; signer guarantees no mint or write | n/a |
| `_touch_signing_activity` | `sign_envelope` | one signature at an agent/human initiating boundary (`send`, typed send, card/contact/task queue actions) before durable queueing | `true` |
| `_redeem_loop` via `_sign` | `sign_envelope` | first successful invite-request signing is human initiated; later loop attempts are daemon retries | first `true`, then `false` |
| `_run_session` via `_sign` | `sign_envelope` | background delivery and every resend of an already queued item; the initiating boundary already touched once | `false` |
| `_approve_reward` | `approve_spend` | agent/human task-post payment authorization | `true` |

Daemon-created queued task traffic is also classified at its queue boundary: pending-assignment recovery queues `task.assign` with `touch=false`; escrow reconciliation queues terminal verdicts with `touch=false`; expiry sweep queues `task.reject` with `touch=false`. Every reply forced by an inbound envelope now also uses `touch=false`; only explicit local human/agent entry points retain `touch=true`. Acks are transport frames and do not call the signer.

## Verification

All Python/pytest commands used a fresh `HOME=$(mktemp -d)` and `PYTHONPATH=/home/prithvi/wt/sth-lease:/home/prithvi/.local/lib/python3.12/site-packages`; loopback-only suites ran with sandbox socket permission. Nothing touched the real signer/node homes, systemd, or running sockets.

- Required focused files (`test_nak_grant`, `test_nak_join`, `test_nakd`, `test_nak_neuron_escrow`): **178 passed** in 34.20s.
- Established full network matrix (the 13 dependency-clean files recorded by the prior reports): branch **328 passed / 10 failed** versus exported `forgejo/main` **316 passed / 10 failed**. Net: **+12 passed, no new failure**.
- The identical 10 failures are the known baseline: nine `test_nak_direct` p2p tests whose subprocess stubs return `None`, plus `test_schema_migration_is_idempotent_on_a_forgejo_main_store` against the moving `forgejo/main` schema.
- Literal repository-wide `pytest -q` cannot collect in this checkout and stopped with 39 environment/dependency errors: missing Petals/Hivemind/grpc/pytest-asyncio, protobuf runtime mismatch, the local `scripts/packaging` shadow, and the required `INITIAL_PEERS` variable. This is separate from the dependency-clean project matrix and occurred before executing tests.
- `py_compile` passed for every changed Python module/test; `git diff --check` passed.

The node side was verified against Sthambha signer commit `93ab1b9` (`signer: add sliding charter leases`).

## Fix round 1

Review: `/home/prithvi/wt/rv-lease.md`, findings B1 and N8.

- B1: all envelopes queued as a consequence of inbound processing now use `touch=false`. This includes the automatic first assignment, both already-assigned rejection paths, paid assignment through `_open_assignment`, automatic accept/reject verdicts, and the card published when an inbound `contact.accept` activates a connection. Automatic assignment is a protocol decision, not a human/agent decision. The `_open_assignment` default is therefore also `false`. Local accept, send, card set, post, claim, submit, and agent-message entry points continue to touch.
- N8: charter renewal detects a signer without `load_or_create_lease_key` before opening mutable agent state, and genuine join first-init detects it before creating any identity or authority file. Both exit cleanly with `signer too old for sliding grants; update the node or use --no-lease`. Renewal with `--no-lease` remains compatible and omits the new `lease` argument entirely.

### Complete `nakd` signer-call inventory

Every queued item is signed for delivery in `_run_session` with `touch=false`; a local initiating queue boundary first calls `_touch_signing_activity` with `touch=true`. The table lists every direct signer call and every `_sign`, `_touch_signing_activity`, `_queue`, `_send_card_if_set`, and `_open_assignment` call site that determines that classification.

| Call site | Envelope/action | Classification | `touch` |
|---|---|---|---|
| `Node.__init__` | `whoami` | startup identity read | n/a |
| `Node.__init__` | `data_key` | startup encrypted-store key retrieval | n/a |
| `_grant_status` | `grant_status` | startup/daily/local read-only preflight; signer does not mint or write | n/a |
| `_redeem_loop` → `_sign` | `contact.request` | local redeem on the first successful signing attempt; later daemon retries | first `true`, then `false` |
| `accept` → `_queue` | `contact.accept` | explicit local accept | `true` |
| `accept` → `_send_card_if_set` → `_queue` | `contact.card` | card accompanying explicit local accept | `true` |
| `send` → `_queue` | `msg` | explicit local send | `true` |
| `agent_send` → `_touch_signing_activity` | `agent.msg` | explicit local agent send before durable queueing | `true` |
| `card_set` → `_send_card_if_set` → `_queue` | `contact.card` per active contact | explicit local card update | `true` |
| `_receive_checked(contact.accept)` → `_send_card_if_set` → `_queue` | `contact.card` | reply forced by inbound accept | `false` |
| `post_task` → `_approve_reward` | `approve_spend` | explicit local paid-task post | `true` |
| `post_task` → `_queue` | `task.post` per target | explicit local task post | `true` |
| `_finish_pending_assignments` → `_open_assignment` → `_queue` | `task.assign` | daemon recovery/reconciliation | `false` |
| `reconcile_escrow` → `_queue` | `task.accept` / `task.reject` | daemon settlement reconciliation | `false` |
| `sweep` → `_queue` | `task.reject` | daemon expiry response | `false` |
| `claim_task` → `_queue` | `task.claim` | explicit local claim | `true` |
| `submit_task` → `_queue` | `task.result` | explicit local submit | `true` |
| `_task_receive(task.claim)` → `_queue` | first free `task.assign` | automatic reply to inbound claim | `false` |
| `_task_receive(task.claim)` → `_queue` | free-task `task.reject` for another assignee | automatic reply to inbound claim | `false` |
| `_task_receive(task.claim)` → `_queue` | paid-task `task.reject` for another assignee | automatic reply to inbound claim | `false` |
| `_task_receive(task.claim)` → `_open_assignment` → `_queue` | paid `task.assign` | automatic reply to inbound claim | `false` |
| `_task_receive(task.result)` → `_queue` | `task.accept` / `task.reject` verdict | automatic reply to inbound result | `false` |
| `_queue` → `_touch_signing_activity` | pre-queue signer check for the local rows above | records the initiating local activity once | `true` iff the call-site row is `true` |
| `_run_session` → `_sign` | delivery and every resend of every queued item | daemon background work | `false` |

Acks/nacks, pings, and direct-address frames are transport frames and never call the signer.

### Verification

All pytest runs used a newly created temporary `HOME`, the Sthambha signer branch on `PYTHONPATH`, and only loopback in-process relays. No real node home, signer socket, service, deploy, pull, or push was touched.

- Required focused files (`test_nak_grant`, `test_nak_join`, `test_nakd`, `test_nak_neuron_escrow`): **182 passed** in 38.26s (round-0 branch: 178 passed; +4 regressions).
- Final four new regressions after the last hardening edit: **4 passed** in 3.89s.
- Established 13-file matrix: branch **332 passed / 10 failed** versus clean Git-backed `forgejo/main@29ccaf0` **316 passed / 10 failed**. Net: **+16 passed, no new failure**.
- The same ten baseline failures occur on both revisions: nine `test_nak_direct` subprocess-stub failures (`None.returncode`) and `test_schema_migration_is_idempotent_on_a_forgejo_main_store` against the moving `forgejo/main` schema.
- `py_compile` for all changed Python runtime/test files and `git diff --check` passed.
