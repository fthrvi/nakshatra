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
